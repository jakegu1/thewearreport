"""Fetch one frame from every camera of a sweep, in memory.

Frames are downloaded into memory, decoded with `cv2.imdecode` and returned as arrays.
Nothing is written to disk and no image bytes are logged (AGENTS.md INV-1). Each camera
gets exactly one attempt, one request with no redirects, under one wall-clock deadline: a
failed frame is recorded with its error category and never retried, so a sweep's cost and
duration stay bounded.

Only complete JPEGs reach the decoder: a body must start with the JPEG start-of-image
marker and end with the end-of-image marker. Other formats are refused before decoding,
because some of OpenCV's decoders go through a temporary file (see `wearreport._cv`). A
body with more than `MAX_SCANS` start-of-scan markers is refused before decoding too,
since each scan costs the decoder a pass over the whole frame and the decode cannot be
interrupted.

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
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from wearreport import registry
from wearreport._cv import cv2
from wearreport.settings import load_settings

CONCURRENCY = 24
TIMEOUT_S = 20
# JamCam stills are about 30 kB; anything near this is not a camera frame.
MAX_FRAME_BYTES = 8 * 1024 * 1024
DRY_RUN_CAMERAS = 50
JPEG_SOI = b"\xff\xd8\xff"  # start-of-image marker and the first byte of the next marker
JPEG_EOI = b"\xff\xd9"
# Encoders may pad after the end-of-image marker; it must appear this close to the end.
EOI_WINDOW = 16
# The most start-of-scan markers (FF DA) a body may hold. The decoder's work grows with
# the number of scans times the declared area, and it cannot be interrupted; a baseline
# frame has one scan and the engine's progressive encoder writes about ten. Counted over
# the whole body, so anything that adds markers (a thumbnail) only makes it stricter.
MAX_SCANS = 32
JPEG_SOS = b"\xff\xda"

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

    `timeout_s` is one wall-clock deadline per frame, covering the connection, the
    headers and the body (see `registry.bounded_get`; DNS resolution is outside it).
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if not timeout_s > 0:
        raise ValueError("timeout_s must be positive")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="fetch") as pool:
        results = list(pool.map(lambda cam: fetch_frame(cam, timeout_s), cameras))
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


def fetch_frame(camera: registry.Camera, timeout_s: float) -> FrameResult:
    """One attempt at one camera. Failures become a FrameResult, never an exception."""
    started = time.monotonic()
    frame: npt.NDArray[np.uint8] | None = None
    error: ErrorKind | None = None
    try:
        body = _download(camera.image_url, timeout_s)
        frame = _decode(body)
        del body
    except _FetchError as exc:
        error = exc.kind
        logger.debug("frame failed", extra={"camera_id": camera.id, "error": error})
    return FrameResult(camera.id, frame, time.monotonic() - started, error)


def _download(url: str, timeout_s: float) -> bytes:
    try:
        return registry.bounded_get(
            url, timeout_s=timeout_s, max_bytes=MAX_FRAME_BYTES, schemes=("https", "http")
        )
    except registry.BodyTooLarge:
        raise _FetchError("http") from None
    except urllib.error.HTTPError as exc:  # any status outside 2xx, redirects included
        exc.close()
        raise _FetchError("http") from None
    except urllib.error.URLError as exc:
        kind: ErrorKind = "timeout" if isinstance(exc.reason, TimeoutError) else "network"
        raise _FetchError(kind) from None
    except TimeoutError:
        raise _FetchError("timeout") from None
    except (OSError, http.client.HTTPException, ValueError):
        # ValueError: a URL that is not http(s), or that urllib cannot parse.
        raise _FetchError("network") from None


def count_scans(body: bytes) -> int:
    """The start-of-scan markers anywhere in `body`, inside other segments included."""
    return body.count(JPEG_SOS)


def _decode(body: bytes) -> npt.NDArray[np.uint8]:
    # Never hand the decoder anything but a complete JPEG (see the module docstring).
    if not body.startswith(JPEG_SOI) or JPEG_EOI not in body[-EOI_WINDOW:]:
        raise _FetchError("decode")
    if count_scans(body) > MAX_SCANS:
        raise _FetchError("decode")  # before decoding: each scan costs a pass over the frame
    try:
        frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:  # includes images over wearreport._cv.MAX_IMAGE_PIXELS
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
