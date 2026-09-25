from __future__ import annotations

import contextlib
import http.client
import json
import logging
import socket
import threading
import time
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from wearreport import registry

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "jamcam_places.json"


def _place(place_id: str = "cam", **overrides: object) -> dict[str, Any]:
    place: dict[str, Any] = {
        "id": place_id,
        "commonName": f"Camera {place_id}",
        "lat": 51.5,
        "lon": -0.1,
        "additionalProperties": [
            {"key": "available", "value": "true"},
            {"key": "imageUrl", "value": f"https://example.test/{place_id}"},
        ],
    }
    for key, value in overrides.items():
        if value is None:
            place.pop(key, None)
        else:
            place[key] = value
    return place


def _never_sleep(seconds: float) -> None:
    raise AssertionError("unexpected sleep")


# Parsing -------------------------------------------------------------------------------


def test_fixture_counts() -> None:
    result = registry.parse_places(json.loads(FIXTURE.read_text()))
    assert len(result.cameras) == 8
    assert result.skipped_unavailable == 1
    assert result.skipped_malformed == 1


def test_valid_place_becomes_camera() -> None:
    (cam,) = registry.parse_places([_place("a", lat=51, lon=0)]).cameras
    assert cam == registry.Camera(
        id="a", name="Camera a", lat=51.0, lon=0.0, image_url="https://example.test/a"
    )
    assert isinstance(cam.lat, float)


@pytest.mark.parametrize(
    "place",
    [
        "not an object",
        None,
        _place(id=None),
        _place(id=""),
        _place(id=42),
        _place(commonName=None),
        _place(lat=None),
        _place(lon="-0.1"),
        _place(lat=True),
        _place(lat=float("nan")),
        _place(lat=91.0),
        _place(lon=-180.5),
        _place(additionalProperties=None),
        _place(additionalProperties={"available": "true"}),
    ],
    ids=repr,
)
def test_malformed_places_are_counted(place: object) -> None:
    result = registry.parse_places([place, _place("ok")])
    assert [c.id for c in result.cameras] == ["ok"]
    assert result.skipped_malformed == 1
    assert result.skipped_unavailable == 0


@pytest.mark.parametrize(
    "image_url", ["http://example.test/a", "HTTPS://example.test/a", "", 7, "//example.test/a"]
)
def test_image_url_must_be_https(image_url: object) -> None:
    props = [{"key": "available", "value": "true"}, {"key": "imageUrl", "value": image_url}]
    result = registry.parse_places([_place(additionalProperties=props)])
    assert result.cameras == []
    assert result.skipped_malformed == 1


@pytest.mark.parametrize("available", ["false", "True", "", None])
def test_anything_but_available_true_is_unavailable_not_malformed(available: object) -> None:
    props: list[object] = [{"key": "imageUrl", "value": "https://example.test/a"}]
    if available is not None:
        props.append({"key": "available", "value": available})
    result = registry.parse_places([_place(additionalProperties=props)])
    assert result.cameras == []
    assert result.skipped_unavailable == 1
    assert result.skipped_malformed == 0


def test_junk_property_items_are_ignored() -> None:
    props = [
        "junk",
        {"value": "no key"},
        {"key": 3, "value": "numeric key"},
        {"key": "available", "value": "true"},
        {"key": "imageUrl", "value": "https://example.test/a"},
    ]
    assert len(registry.parse_places([_place(additionalProperties=props)]).cameras) == 1


def test_empty_registry_is_not_padded() -> None:
    result = registry.parse_places([])
    assert result == registry.Registry(cameras=[], skipped_unavailable=0, skipped_malformed=0)


# Fetching and retries -------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None),  # type: ignore[arg-type]
        urllib.error.URLError("dns failure"),
        http.client.IncompleteRead(b"[{"),
        ConnectionResetError("reset"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_transport_errors_are_retried(error: Exception) -> None:
    attempts: list[int] = []
    body = json.dumps([_place("ok")]).encode()

    def fetch(url: str, timeout: float) -> bytes:
        attempts.append(1)
        if len(attempts) == 1:
            raise error
        return body

    sleeps: list[float] = []
    cams = registry.list_cameras(None, fetch=fetch, sleep=sleeps.append)
    assert [c.id for c in cams] == ["ok"]
    assert sleeps == [1]


def test_unexpected_errors_are_not_swallowed() -> None:
    def fetch(url: str, timeout: float) -> bytes:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        registry.list_cameras(None, fetch=fetch, sleep=_never_sleep)


def test_failed_attempts_are_logged_without_the_app_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="wearreport")

    def fetch(url: str, timeout: float) -> bytes:
        raise OSError(f"cannot reach {url}")

    with pytest.raises(registry.RegistryError) as info:
        registry.list_cameras("s3cr3t", fetch=fetch, sleep=lambda s: None)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [getattr(r, "attempt") for r in warnings] == [1, 2]  # noqa: B009
    assert all(getattr(r, "error") == "OSError" for r in warnings)  # noqa: B009
    assert "s3cr3t" not in caplog.text
    assert "s3cr3t" not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_app_key_is_url_encoded() -> None:
    urls: list[str] = []

    def fetch(url: str, timeout: float) -> bytes:
        urls.append(url)
        return b"[]"

    registry.list_cameras("a b&c", fetch=fetch, sleep=_never_sleep)
    assert urls == ["https://api.tfl.gov.uk/Place/Type/JamCam?app_key=a+b%26c"]


# Command line ---------------------------------------------------------------------------


def test_main_uses_app_key_from_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    urls: list[str] = []

    def fake_bounded_get(url: str, *, timeout_s: float, max_bytes: int) -> bytes:
        urls.append(url)
        return FIXTURE.read_bytes()

    monkeypatch.setattr(registry, "bounded_get", fake_bounded_get)
    monkeypatch.setenv("TFL_APP_KEY", "k123")
    assert registry.main() == 0
    assert urls == ["https://api.tfl.gov.uk/Place/Type/JamCam?app_key=k123"]
    out = capsys.readouterr().out
    assert "available cameras: 8\n" in out
    assert "skipped unavailable: 1\n" in out
    assert "k123" not in out


def test_main_reports_failure_with_exit_code_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_bounded_get(url: str, *, timeout_s: float, max_bytes: int) -> bytes:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(registry, "bounded_get", fake_bounded_get)
    monkeypatch.delenv("TFL_APP_KEY", raising=False)
    sleeps: list[float] = []
    assert registry.main(sleep=sleeps.append) == 1
    assert sleeps == [1, 2]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RegistryError" not in captured.err
    assert "unavailable after 3 attempts (URLError)" in captured.err


# Hostile responses (a sweep survives one bad entry or response) -----------------------


@pytest.mark.parametrize("field", ["lat", "lon"])
@pytest.mark.parametrize("value", [10**400, -(10**400), 10**309])
def test_coordinate_too_large_for_a_float_is_malformed(field: str, value: int) -> None:
    places = json.loads(json.dumps([_place("big", **{field: value}), _place("ok")]))
    result = registry.parse_places(places)
    assert [c.id for c in result.cameras] == ["ok"]
    assert result.skipped_malformed == 1


def test_integer_coordinates_in_range_are_kept() -> None:
    (cam,) = registry.parse_places([_place("a", lat=-90, lon=180)]).cameras
    assert (cam.lat, cam.lon) == (-90.0, 180.0)


def test_deeply_nested_body_is_a_registry_error_and_not_retried() -> None:
    calls: list[int] = []
    body = b'[{"a": ' * 100_000 + b"1" + b"}]" * 100_000

    def fetch(url: str, timeout: float) -> bytes:
        calls.append(1)
        return body

    with pytest.raises(registry.RegistryError, match="nested too deeply"):
        registry.list_cameras(None, fetch=fetch, sleep=_never_sleep)
    assert calls == [1]


def test_body_at_the_cap_is_parsed_and_over_it_is_refused() -> None:
    cap = registry.MAX_BODY_BYTES
    at_cap = b"[" + b" " * (cap - 2) + b"]"
    assert registry.list_cameras(None, fetch=lambda u, t: at_cap, sleep=_never_sleep) == []
    calls: list[int] = []

    def over(url: str, timeout: float) -> bytes:
        calls.append(1)
        return at_cap + b" "

    with pytest.raises(registry.RegistryError, match="exceeds"):
        registry.list_cameras(None, fetch=over, sleep=_never_sleep)
    assert calls == [1]  # a deterministic refusal is not retried


@pytest.mark.parametrize("status", [400, 401, 403, 404, 418, 451])
def test_client_errors_are_not_retried(status: int, caplog: pytest.LogCaptureFixture) -> None:
    calls: list[int] = []

    def fetch(url: str, timeout: float) -> bytes:
        calls.append(1)
        raise urllib.error.HTTPError(url, status, "no", {}, None)  # type: ignore[arg-type]

    with pytest.raises(registry.RegistryError, match=f"HTTP {status}") as info:
        registry.list_cameras("s3cr3t", fetch=fetch, sleep=_never_sleep)
    assert calls == [1]
    assert "s3cr3t" not in str(info.value) + caplog.text
    assert info.value.__cause__ is None and info.value.__suppress_context__


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_timeouts_rate_limit_and_server_errors_are_retried(status: int) -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def fetch(url: str, timeout: float) -> bytes:
        calls.append(1)
        raise urllib.error.HTTPError(url, status, "busy", {}, None)  # type: ignore[arg-type]

    with pytest.raises(registry.RegistryError, match="after 3 attempts"):
        registry.list_cameras(None, fetch=fetch, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [1, 2]


# bounded_get: one request, no redirects, one wall-clock deadline ---------------------


class _Handler(BaseHTTPRequestHandler):
    hits: list[str]

    def log_message(self, format: str, *args: object) -> None:
        pass

    def reply(self, status: int, body: bytes, **headers: str) -> None:
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def _serve(do_get: Any) -> Iterator[tuple[str, list[str]]]:
    hits: list[str] = []

    class Handler(_Handler):
        def do_GET(self) -> None:
            hits.append(self.path)
            do_get(self)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    httpd.block_on_close = False
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/x", hits
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url: str, timeout_s: float = 5, max_bytes: int = 1000) -> bytes:
    return registry.bounded_get(url, timeout_s=timeout_s, max_bytes=max_bytes, schemes=("http",))


def test_bounded_get_returns_the_body_and_sends_the_user_agent() -> None:
    agents: list[str] = []

    def ok(h: _Handler) -> None:
        agents.append(h.headers["User-Agent"])
        h.reply(200, b"[]", **{"Content-Length": "2"})

    with _serve(ok) as (url, hits):
        assert _get(url) == b"[]"
    assert hits == ["/x"]
    assert agents[0].startswith("wearreport")


@pytest.mark.parametrize(
    "url", ["http://example.test/x", "file:///etc/passwd", "ftp://example.test/x", "data:,x"]
)
def test_bounded_get_is_https_only_by_default(url: str) -> None:
    with pytest.raises(ValueError, match="not allowed") as info:
        registry.bounded_get(url, timeout_s=1, max_bytes=10)
    assert "example.test" not in str(info.value)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_bounded_get_never_follows_a_redirect(status: int) -> None:
    def redirect(h: _Handler) -> None:
        if h.path == "/x":
            h.reply(status, b"", Location="/elsewhere", **{"Content-Length": "0"})
        else:
            h.reply(200, b"[]", **{"Content-Length": "2"})

    with _serve(redirect) as (url, hits), pytest.raises(urllib.error.HTTPError) as info:
        _get(url)
    assert info.value.code == status
    info.value.close()
    assert hits == ["/x"]


def test_bounded_get_deadline_covers_trickled_headers() -> None:
    def trickle(h: _Handler) -> None:
        with contextlib.suppress(OSError):
            h.wfile.write(b"HTTP/1.1 200 OK\r\n")
            for _ in range(60):
                h.wfile.write(b"X")
                h.wfile.flush()
                time.sleep(0.9)

    with _serve(trickle) as (url, hits):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            _get(url, timeout_s=1)
        assert time.monotonic() - started < 2
    assert hits == ["/x"]


def test_bounded_get_deadline_covers_a_trickled_body_without_length() -> None:
    def trickle(h: _Handler) -> None:
        h.send_response(200)
        h.end_headers()
        with contextlib.suppress(OSError):
            for _ in range(60):
                h.wfile.write(b" ")
                h.wfile.flush()
                time.sleep(0.1)

    with _serve(trickle) as (url, _):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            _get(url, timeout_s=0.5)
        assert time.monotonic() - started < 1.5


def test_bounded_get_deadline_covers_a_silent_server() -> None:
    """Accepts the connection, then never answers: a TLS handshake that stalls looks alike."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            _get(f"http://127.0.0.1:{port}/x", timeout_s=0.5)
        assert time.monotonic() - started < 1.5


def test_bounded_get_refuses_a_declared_oversized_body() -> None:
    with (
        _serve(lambda h: h.reply(200, b"x" * 2000, **{"Content-Length": "2000"})) as (url, _),
        pytest.raises(registry.BodyTooLarge),
    ):
        _get(url, max_bytes=1000)


def test_bounded_get_stops_reading_past_the_cap() -> None:
    def endless(h: _Handler) -> None:
        h.send_response(200)
        h.end_headers()
        with contextlib.suppress(OSError):
            for _ in range(1000):
                h.wfile.write(b"x" * 1000)

    with _serve(endless) as (url, _), pytest.raises(registry.BodyTooLarge):
        _get(url, max_bytes=1000)


def test_bounded_get_body_at_the_cap_is_returned() -> None:
    with _serve(lambda h: h.reply(200, b"x" * 1000, **{"Content-Length": "1000"})) as (url, _):
        assert len(_get(url, max_bytes=1000)) == 1000


def test_bounded_get_refuses_a_short_body() -> None:
    def short(h: _Handler) -> None:
        h.reply(200, b"x" * 10, **{"Content-Length": "500"})
        h.connection.shutdown(socket.SHUT_RDWR)

    with _serve(short) as (url, _), pytest.raises(http.client.IncompleteRead):
        _get(url)


def test_bounded_get_connection_refused_is_an_os_error() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(OSError) as info:
        _get(f"http://127.0.0.1:{port}/x")
    assert not isinstance(info.value, TimeoutError)


def test_bounded_get_leaves_no_timer_thread_behind() -> None:
    before = threading.active_count()
    with _serve(lambda h: h.reply(200, b"[]", **{"Content-Length": "2"})) as (url, _):
        for _ in range(5):
            _get(url)
    deadline = time.monotonic() + 2
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= before


# http_fetch: the registry's fetch goes through bounded_get ----------------------------


@pytest.fixture
def allow_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let http_fetch reach the plain-HTTP test server; everything else stays real."""
    real = registry.bounded_get

    def bounded_get(url: str, *, timeout_s: float, max_bytes: int) -> bytes:
        return real(url, timeout_s=timeout_s, max_bytes=max_bytes, schemes=("http",))

    monkeypatch.setattr(registry, "bounded_get", bounded_get)


@pytest.mark.parametrize("url", ["http://example.test/x", "file:///etc/passwd", "ftp://x/y"])
def test_http_fetch_is_https_only_and_does_not_retry_a_refused_scheme(url: str) -> None:
    with pytest.raises(registry.RegistryError, match="must use HTTPS") as info:
        registry.http_fetch(url, 30)
    assert "example.test" not in str(info.value)
    calls: list[int] = []

    def fetch(_: str, timeout: float) -> bytes:
        calls.append(1)
        return registry.http_fetch(url, timeout)

    with pytest.raises(registry.RegistryError, match="must use HTTPS"):
        registry.list_cameras(None, fetch=fetch, sleep=_never_sleep)
    assert calls == [1]


@pytest.mark.usefixtures("allow_http")
def test_http_fetch_returns_the_body_and_sends_the_user_agent() -> None:
    agents: list[str] = []

    def ok(h: _Handler) -> None:
        agents.append(h.headers["User-Agent"])
        h.reply(200, b"[]", **{"Content-Length": "2"})

    with _serve(ok) as (url, hits):
        assert registry.http_fetch(url, 5) == b"[]"
    assert hits == ["/x"]
    assert agents[0].startswith("wearreport")


@pytest.mark.usefixtures("allow_http")
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_http_fetch_never_follows_a_redirect(status: int) -> None:
    def redirect(h: _Handler) -> None:
        if h.path == "/x":
            h.reply(status, b"", Location="/elsewhere", **{"Content-Length": "0"})
        else:
            h.reply(200, b"[]", **{"Content-Length": "2"})

    with _serve(redirect) as (url, hits):
        with pytest.raises(urllib.error.HTTPError) as info:
            registry.http_fetch(url, 5)
        assert info.value.code == status
        info.value.close()
        assert hits == ["/x"]
        # Through the retry loop: each attempt is one request, and none reaches the target.
        with pytest.raises(registry.RegistryError, match="after 3 attempts"):
            registry.list_cameras(
                None, fetch=lambda _, t: registry.http_fetch(url, t), sleep=lambda s: None
            )
    assert hits == ["/x"] * 4


@pytest.mark.usefixtures("allow_http")
def test_http_fetch_trickled_headers_hit_the_deadline() -> None:
    def trickle(h: _Handler) -> None:
        with contextlib.suppress(OSError):
            h.wfile.write(b"HTTP/1.1 200 OK\r\n")
            for _ in range(60):
                h.wfile.write(b"X")
                h.wfile.flush()
                time.sleep(0.9)

    with _serve(trickle) as (url, hits):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            registry.http_fetch(url, 1)
        assert time.monotonic() - started < 2
    assert hits == ["/x"]


def _oversized_declared(h: _Handler) -> None:
    h.send_response(200)
    h.send_header("Content-Length", str(registry.MAX_BODY_BYTES + 1))
    h.end_headers()
    with contextlib.suppress(OSError):
        h.wfile.write(b" " * 1000)


def _oversized_streamed(h: _Handler) -> None:
    h.send_response(200)
    h.end_headers()
    chunk = b" " * (1024 * 1024)
    with contextlib.suppress(OSError):
        for _ in range(registry.MAX_BODY_BYTES // len(chunk) + 2):
            h.wfile.write(chunk)


@pytest.mark.usefixtures("allow_http")
@pytest.mark.parametrize("serve", [_oversized_declared, _oversized_streamed])
def test_http_fetch_oversized_body_is_a_registry_error_and_not_retried(serve: Any) -> None:
    with _serve(serve) as (url, hits):
        with pytest.raises(registry.RegistryError, match="exceeds") as info:
            registry.http_fetch(url, 5)
        assert info.value.__cause__ is None and info.value.__suppress_context__
        with pytest.raises(registry.RegistryError, match="exceeds"):
            registry.list_cameras(
                None, fetch=lambda _, t: registry.http_fetch(url, t), sleep=_never_sleep
            )
    assert hits == ["/x", "/x"]
