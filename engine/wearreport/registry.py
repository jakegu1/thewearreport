"""The TfL JamCam registry: which cameras are available right now.

Availability changes over time, so the registry is fetched fresh on every sweep and
never persisted. Only metadata is read here; frames are fetched elsewhere.

Run `python -m wearreport.registry` to print the current count.

`bounded_get` is the engine's one-attempt HTTP GET: no redirects, a size cap and one
wall-clock deadline per request. It lives here, in a standard-library-only module, so
that every engine module can use it.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import math
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from wearreport.settings import load_settings

JAMCAM_URL = "https://api.tfl.gov.uk/Place/Type/JamCam"
USER_AGENT = "wearreport-engine (+https://github.com/jakegu1/thewearreport)"
TIMEOUT_S = 30
MAX_ATTEMPTS = 3
BACKOFF_S = (1, 2)  # sleep before attempts 2 and 3
# The real response is about 1 MB; a larger body than this is refused, never parsed.
MAX_BODY_BYTES = 16 * 1024 * 1024
RETRYABLE_CLIENT_ERRORS = frozenset({408, 429})
READ_CHUNK = 64 * 1024

logger = logging.getLogger("wearreport.registry")

Fetch = Callable[[str, float], bytes]
Sleep = Callable[[float], None]


class RegistryError(RuntimeError):
    """The camera registry could not be fetched or decoded."""


@dataclass(frozen=True, slots=True)
class Camera:
    id: str
    name: str
    lat: float
    lon: float
    image_url: str


@dataclass(frozen=True, slots=True)
class Registry:
    cameras: list[Camera]
    skipped_unavailable: int
    skipped_malformed: int


def http_fetch(url: str, timeout: float) -> bytes:
    """GET `url` with the standard library and return the body.

    Reads at most MAX_BODY_BYTES + 1 bytes, so an oversized body is detected without
    being held in memory.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body: bytes = resp.read(MAX_BODY_BYTES + 1)
    return body


class BodyTooLarge(Exception):
    """A response body is, or declares itself, larger than the caller's cap."""


def bounded_get(
    url: str, *, timeout_s: float, max_bytes: int, schemes: tuple[str, ...] = ("https",)
) -> bytes:
    """GET `url` in exactly one request and return its body.

    - Only URLs whose scheme is in `schemes` are opened (ValueError otherwise).
    - Redirects are never followed: a 3xx raises urllib.error.HTTPError, as any other
      status outside 2xx does.
    - One wall-clock deadline of `timeout_s` covers connecting, the TLS handshake (and a
      proxy tunnel), the status line, the headers and the body. When it passes, the socket
      is shut down and TimeoutError is raised, however slowly the server trickles bytes.
      Known limit: DNS resolution happens before the deadline can interrupt it.
    - A body larger than `max_bytes`, declared or counted, raises BodyTooLarge; a body
      shorter than its Content-Length raises http.client.IncompleteRead.

    Other failures raise OSError (including urllib.error.URLError) or
    http.client.HTTPException. Exception messages never contain the URL or the body.
    """
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme not in schemes:
        raise ValueError(f"URL scheme {scheme!r} is not allowed")
    with _Deadline(timeout_s) as deadline:
        try:
            body = _get(url, timeout_s, max_bytes, deadline)
        except (OSError, http.client.HTTPException, ValueError):
            if deadline.expired:
                raise TimeoutError("request deadline passed") from None
            raise
        if deadline.expired:  # the socket was shut down under a read that then saw EOF
            raise TimeoutError("request deadline passed")
    return body


def _get(url: str, timeout_s: float, max_bytes: int, deadline: _Deadline) -> bytes:
    opener = urllib.request.OpenerDirector()
    # No HTTPRedirectHandler: a 3xx reaches HTTPDefaultErrorHandler and raises HTTPError.
    for handler in (
        urllib.request.ProxyHandler(),
        _HTTPHandler(deadline),
        _HTTPSHandler(deadline),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with opener.open(req, timeout=timeout_s) as resp:
        declared = resp.headers.get("Content-Length", "")
        length = int(declared) if declared.isdigit() and len(declared) < 19 else None
        if length is not None and length > max_bytes:
            raise BodyTooLarge(f"response declares more than {max_bytes} bytes")
        body = bytearray()
        while chunk := resp.read1(READ_CHUNK):
            body += chunk
            if len(body) > max_bytes:
                raise BodyTooLarge(f"response exceeds {max_bytes} bytes")
        # read1() does not raise IncompleteRead; a cut-off body must not be used.
        if length is not None and len(body) != length:
            raise http.client.IncompleteRead(b"", length - len(body))
        return bytes(body)


class _Deadline:
    """One request's wall-clock deadline.

    Each socket the request opens is registered here through a duplicate descriptor. When
    the deadline passes, a timer thread shuts every registered socket down, which ends any
    blocked connect, TLS handshake or read. The duplicate refers to the same connection
    even after TLS wraps the original socket, and it is closed only on exit, so the timer
    can never touch a descriptor that has been reused elsewhere.
    """

    def __init__(self, seconds: float) -> None:
        self._end = time.monotonic() + seconds
        self._lock = threading.Lock()
        self._handles: list[socket.socket] = []
        self._expired = False
        self._done = False
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True

    def __enter__(self) -> _Deadline:
        self._timer.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._timer.cancel()
        with self._lock:
            self._done = True
            for handle in self._handles:
                handle.close()
            self._handles.clear()

    @property
    def expired(self) -> bool:
        with self._lock:
            return self._expired

    def remaining(self) -> float:
        return self._end - time.monotonic()

    def open_socket(self, address: tuple[str, int], timeout: object) -> socket.socket:
        """socket.create_connection, with each connect attempt bounded by the deadline."""
        per_op = float(timeout) if isinstance(timeout, int | float) else None
        host, port = address
        # DNS resolution is outside the deadline: getaddrinfo cannot be interrupted.
        infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        error: OSError = OSError("no address to connect to")
        for family, kind, proto, _, sockaddr in infos:
            remaining = self.remaining()
            if remaining <= 0:
                raise TimeoutError("request deadline passed while connecting")
            sock = socket.socket(family, kind, proto)
            try:
                sock.settimeout(remaining if per_op is None else min(per_op, remaining))
                sock.connect(sockaddr)
                sock.settimeout(per_op)
                self._watch(sock)
            except OSError as exc:
                sock.close()
                error = exc
                continue
            return sock
        raise error

    def _watch(self, sock: socket.socket) -> None:
        with self._lock:
            if self._done:
                raise TimeoutError("request already finished")
            handle = sock.dup()
            self._handles.append(handle)
            if self._expired:
                _shut_down(handle)

    def _expire(self) -> None:
        with self._lock:
            if self._done:
                return
            self._expired = True
            for handle in self._handles:
                _shut_down(handle)


def _shut_down(handle: socket.socket) -> None:
    with contextlib.suppress(OSError):  # already closed by the peer
        handle.shutdown(socket.SHUT_RDWR)


class _DeadlineConnection(http.client.HTTPConnection):
    """An HTTP(S) connection whose sockets are opened and watched by a _Deadline."""

    def __init__(self, host: str, *, deadline: _Deadline, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        # HTTPConnection.connect() opens its socket through this attribute.
        self._create_connection = self._open_socket
        self._deadline = deadline

    def _open_socket(
        self, address: tuple[str, int], timeout: object, source_address: object = None
    ) -> socket.socket:
        return self._deadline.open_socket(address, timeout)


class _DeadlineHTTPSConnection(_DeadlineConnection, http.client.HTTPSConnection):
    pass


class _HTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, deadline: _Deadline) -> None:
        super().__init__()
        self._deadline = deadline

    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        def connection(host: str, **kwargs: Any) -> http.client.HTTPConnection:
            return _DeadlineConnection(host, deadline=self._deadline, **kwargs)

        return self.do_open(connection, req=req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, deadline: _Deadline) -> None:
        self._tls = ssl.create_default_context()
        super().__init__(context=self._tls)
        self._deadline = deadline

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        def connection(host: str, **kwargs: Any) -> http.client.HTTPConnection:
            return _DeadlineHTTPSConnection(host, deadline=self._deadline, **kwargs)

        return self.do_open(connection, req=req, context=self._tls)


def list_cameras(
    app_key: str | None, *, fetch: Fetch = http_fetch, sleep: Sleep = time.sleep
) -> list[Camera]:
    """Return the cameras TfL currently lists as available."""
    return fetch_registry(app_key, fetch=fetch, sleep=sleep).cameras


def fetch_registry(
    app_key: str | None, *, fetch: Fetch = http_fetch, sleep: Sleep = time.sleep
) -> Registry:
    """Fetch and parse the registry, keeping the skip counts."""
    url = JAMCAM_URL
    if app_key:
        url += "?" + urllib.parse.urlencode({"app_key": app_key})
    places = _fetch_places(url, fetch, sleep)
    result = parse_places(places)
    logger.info(
        "jamcam registry loaded",
        extra={
            "available": len(result.cameras),
            "skipped_unavailable": result.skipped_unavailable,
            "skipped_malformed": result.skipped_malformed,
        },
    )
    return result


def _fetch_places(url: str, fetch: Fetch, sleep: Sleep) -> list[object]:
    # Error messages name the exception type only: `url` may carry the app key.
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _decode_places(fetch(url, TIMEOUT_S))
        except urllib.error.HTTPError as exc:
            # A client error will not change on retry; 408 (timeout) and 429 (rate limited) may.
            if 400 <= exc.code < 500 and exc.code not in RETRYABLE_CLIENT_ERRORS:
                raise RegistryError(
                    f"JamCam registry refused the request (HTTP {exc.code})"
                ) from None
            error = type(exc).__name__
        except (OSError, http.client.HTTPException, ValueError) as exc:
            error = type(exc).__name__
        if attempt == MAX_ATTEMPTS:
            raise RegistryError(
                f"JamCam registry unavailable after {MAX_ATTEMPTS} attempts ({error})"
            ) from None
        logger.warning("jamcam registry attempt failed", extra={"attempt": attempt, "error": error})
        sleep(BACKOFF_S[attempt - 1])
    raise AssertionError("unreachable")


def _decode_places(body: bytes) -> list[object]:
    """Parse the registry body. Raises RegistryError, without retry, for a hostile body."""
    if len(body) > MAX_BODY_BYTES:
        raise RegistryError(f"JamCam registry response exceeds {MAX_BODY_BYTES} bytes")
    try:
        payload = json.loads(body)
    except RecursionError:
        raise RegistryError("JamCam registry response is nested too deeply") from None
    if not isinstance(payload, list):
        raise ValueError("response is not a JSON array")
    return payload


def parse_places(places: list[object]) -> Registry:
    """Keep available cameras; count unavailable and malformed entries."""
    cameras: list[Camera] = []
    unavailable = malformed = 0
    for place in places:
        if not isinstance(place, dict):
            malformed += 1
            continue
        props = _properties(place)
        if props is None:
            malformed += 1
            continue
        if props.get("available") != "true":
            unavailable += 1
            continue
        camera = _camera(place, props)
        if camera is None:
            malformed += 1
        else:
            cameras.append(camera)
    return Registry(cameras, skipped_unavailable=unavailable, skipped_malformed=malformed)


def _properties(place: Mapping[str, object]) -> dict[str, object] | None:
    items = place.get("additionalProperties")
    if not isinstance(items, list):
        return None
    return {
        item["key"]: item.get("value")
        for item in items
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }


def _camera(place: Mapping[str, object], props: Mapping[str, object]) -> Camera | None:
    cam_id, name, image_url = place.get("id"), place.get("commonName"), props.get("imageUrl")
    lat, lon = _coordinate(place.get("lat"), 90), _coordinate(place.get("lon"), 180)
    if not (isinstance(cam_id, str) and cam_id and isinstance(name, str)):
        return None
    if lat is None or lon is None:
        return None
    if not (isinstance(image_url, str) and image_url.startswith("https://")):
        return None
    return Camera(id=cam_id, name=name, lat=lat, lon=lon, image_url=image_url)


def _coordinate(value: object, limit: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    # Compare before converting: float() of a huge JSON integer raises OverflowError.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if abs(value) > limit:
        return None
    return float(value)


def main(*, fetch: Fetch = http_fetch, sleep: Sleep = time.sleep) -> int:
    try:
        result = fetch_registry(load_settings().tfl_app_key, fetch=fetch, sleep=sleep)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"available cameras: {len(result.cameras)}")
    print(f"skipped unavailable: {result.skipped_unavailable}")
    print(f"skipped malformed: {result.skipped_malformed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
