"""Fetch one frame from every camera of a sweep, in memory.

Frames are downloaded into memory, decoded with `cv2.imdecode` and returned as arrays.
Nothing is written to disk and no image bytes are logged (AGENTS.md INV-1). Each camera
gets exactly one attempt: a failed frame is recorded with its error category and never
retried, so a sweep's cost and duration stay bounded.

Run `python -m wearreport.fetch --live` for one sweep of the real JamCams, or
`python -m wearreport.fetch --dry-run` for one sweep of a local fake server (no network).
"""

from __future__ import annotations

import argparse
import http.client
import logging
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, get_args

import cv2
import numpy as np
import numpy.typing as npt

from wearreport import registry
from wearreport.settings import load_settings

CONCURRENCY = 24
TIMEOUT_S = 20
# JamCam stills are about 30 kB; anything near this is not a camera frame.
MAX_FRAME_BYTES = 8 * 1024 * 1024
READ_CHUNK = 64 * 1024
DRY_RUN_CAMERAS = 50

ErrorKind = Literal["timeout", "http", "decode", "network"]
ERROR_KINDS: tuple[ErrorKind, ...] = get_args(ErrorKind)

logger = logging.getLogger("wearreport.fetch")


@dataclass(frozen=True, slots=True)
class FrameResult:
    camera_id: str
    frame: npt.NDArray[np.uint8] | None
    seconds: float
    error: ErrorKind | None


class _FetchError(Exception):
    def __init__(self, kind: ErrorKind) -> None:
        super().__init__(kind)
        self.kind: ErrorKind = kind


def fetch_sweep(
    cameras: Sequence[registry.Camera],
    *,
    concurrency: int = CONCURRENCY,
    timeout_s: float = TIMEOUT_S,
) -> list[FrameResult]:
    """Fetch and decode one frame per camera; results follow the order of `cameras`.

    `timeout_s` bounds each socket operation and the download of each body as a whole.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if not timeout_s > 0:
        raise ValueError("timeout_s must be positive")
    opener = _opener()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="fetch") as pool:
        results = list(pool.map(lambda cam: fetch_frame(cam, timeout_s, opener), cameras))
    counts = Counter(r.error for r in results)
    logger.info(
        "sweep fetched",
        extra={
            "cameras": len(results),
            "fetched": counts[None],
            **{f"errors_{kind}": counts[kind] for kind in ERROR_KINDS},
            "seconds": round(time.monotonic() - started, 2),
        },
    )
    return results


def fetch_frame(
    camera: registry.Camera, timeout_s: float, opener: urllib.request.OpenerDirector
) -> FrameResult:
    """One attempt at one camera. Failures become a FrameResult, never an exception."""
    started = time.monotonic()
    frame: npt.NDArray[np.uint8] | None = None
    error: ErrorKind | None = None
    try:
        body = _download(opener, camera.image_url, timeout_s, started + timeout_s)
        frame = _decode(body)
        del body
    except _FetchError as exc:
        error = exc.kind
        logger.debug("frame failed", extra={"camera_id": camera.id, "error": error})
    return FrameResult(camera.id, frame, time.monotonic() - started, error)


def _opener() -> urllib.request.OpenerDirector:
    """HTTP(S) only: unlike urlopen, no file:, ftp: or data: handlers, even on redirect."""
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(),
        urllib.request.HTTPRedirectHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


def _download(
    opener: urllib.request.OpenerDirector, url: str, timeout_s: float, deadline: float
) -> bytes:
    if not url.startswith(("https://", "http://")):
        raise _FetchError("network")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": registry.USER_AGENT})  # noqa: S310
        with opener.open(req, timeout=timeout_s) as resp:
            declared = resp.headers.get("Content-Length")
            length = int(declared) if declared is not None and declared.isdigit() else None
            if length is not None and length > MAX_FRAME_BYTES:
                raise _FetchError("http")
            body = bytearray()
            while chunk := resp.read1(READ_CHUNK):
                body += chunk
                if len(body) > MAX_FRAME_BYTES:
                    raise _FetchError("http")
                if time.monotonic() > deadline:
                    raise _FetchError("timeout")
            # read1() does not raise IncompleteRead; a cut-off frame must not be decoded.
            if length is not None and len(body) != length:
                raise _FetchError("network")
            return bytes(body)
    except urllib.error.HTTPError as exc:
        exc.close()
        raise _FetchError("http") from None
    except urllib.error.URLError as exc:
        kind: ErrorKind = "timeout" if isinstance(exc.reason, TimeoutError) else "network"
        raise _FetchError(kind) from None
    except TimeoutError:
        raise _FetchError("timeout") from None
    except (OSError, http.client.HTTPException, ValueError):
        # ValueError: a URL urllib cannot parse.
        raise _FetchError("network") from None


def _decode(body: bytes) -> npt.NDArray[np.uint8]:
    if not body:
        raise _FetchError("decode")
    try:
        frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        raise _FetchError("decode") from None
    if frame is None or frame.ndim != 3 or frame.dtype != np.uint8:
        raise _FetchError("decode")
    return np.asarray(frame, dtype=np.uint8)


# Command line ---------------------------------------------------------------------------


def _print_summary(listed: int, results: Sequence[FrameResult], seconds: float) -> None:
    counts = Counter(r.error for r in results)
    fetched = counts[None]
    print(f"cameras listed: {listed}")
    print(f"frames fetched: {fetched}")
    print(f"fetched percent: {100 * fetched / listed if listed else 0:.1f}")
    for kind in ERROR_KINDS:
        print(f"errors {kind}: {counts[kind]}")
    print(f"seconds: {seconds:.1f}")


def _sweep(cameras: Sequence[registry.Camera], concurrency: int, timeout_s: float) -> int:
    started = time.monotonic()
    results = fetch_sweep(cameras, concurrency=concurrency, timeout_s=timeout_s)
    seconds = time.monotonic() - started
    _print_summary(len(cameras), results, seconds)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wearreport.fetch", description=__doc__)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--live", action="store_true", help="sweep the real TfL JamCams")
    target.add_argument(
        "--dry-run",
        action="store_true",
        help=f"sweep {DRY_RUN_CAMERAS} cameras of a local fake server (no network)",
    )
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per frame")
    args = ap.parse_args(argv)

    if args.dry_run:
        from wearreport.testing.fake_cameras import FakeCameraServer

        with FakeCameraServer() as server:
            return _sweep(server.cameras(DRY_RUN_CAMERAS), args.concurrency, args.timeout)

    started = time.monotonic()
    try:
        cameras = registry.list_cameras(load_settings().tfl_app_key)
    except registry.RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"registry seconds: {time.monotonic() - started:.1f}")
    return _sweep(cameras, args.concurrency, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
