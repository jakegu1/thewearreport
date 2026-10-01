"""A one-off pilot: how many people, and how tall, the detector finds on HD camera stills.

Run `python -m wearreport.tools.pilot_heights --live`. It fetches the City of Austin's
list of traffic cameras that are turned on (one keyless request to its open-data portal),
selects the cameras inside a bounding box (by default downtown Austin), fetches one still
from each, in memory, runs the sweep's detector, and prints one JSON object of aggregate
counts to stdout: frames, resolutions, persons and umbrellas, and person box heights in
bands. Nothing else is stored, logged or published. It is a measurement only: no
workflow runs it, and nothing it prints reaches the data branch.

Privacy (AGENTS.md INV-1): frames exist only in memory and are dropped after detection.
No image bytes, camera ids, URLs, coordinates or boxes are written, logged or printed;
the output holds counts only.

Safety:
- Without `--live` the tool refuses (exit 2) before any network use, and it refuses
  (exit 1) when the sun is below the horizon at central Austin.
- The camera list is read from one pinned URL on one pinned host, in one request with no
  redirect, a body cap and a timeout. A list that is not a JSON array is an error (exit
  1); a record that is malformed is skipped and counted.
- An image is fetched only from the one host the list uses for screenshots, over HTTPS,
  with no userinfo and no explicit port; any other URL, and any redirect, is refused and
  counted. Each camera gets one attempt; one wall-clock deadline (`--timeout`) bounds the
  whole pass.
- Only complete JPEGs reach the decoder (the start- and end-of-image markers are checked
  first, as the sweep does). The engine caps every decoded image at
  `_cv.MAX_IMAGE_PIXELS` (one megapixel), which a 1920x1080 still exceeds, so a frame
  over the cap is decoded by the JPEG decoder at 1/2, 1/4 or 1/8 of its size, the
  smallest reduction that fits under the cap; the cap itself still applies. The
  detector letterboxes every frame to 640x640 anyway (a 1920x1080 frame to 640x360,
  whether it starts at 1920x1080 or 960x540), and box heights are scaled back to pixels
  of the original frame, whose size is read from the JPEG header.
"""

from __future__ import annotations

import argparse
import datetime
import http.client
import json
import math
import sys
import time
import urllib.error
import urllib.parse
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from wearreport import aggregate, detect, fetch, registry
from wearreport._cv import MAX_IMAGE_PIXELS, cv2

SOURCE = "austin"
DATASET_HOST = "data.austintexas.gov"
DATASET_URL = f"https://{DATASET_HOST}/resource/b4k4-adkb.json?camera_status=TURNED_ON&$limit=2000"
# Every screenshot_address in the dataset was https on this host when the tool was written.
SCREENSHOT_HOST = "cctv.austinmobility.io"
# The list was about 0.55 MB for 820 cameras.
MAX_DATASET_BYTES = 5 * 1024 * 1024
DATASET_TIMEOUT_S = 30.0
FRAME_TIMEOUT_S = float(fetch.TIMEOUT_S)  # per request, within the pass's deadline
MAX_URL_LENGTH = 2048
# A JSON integer with more digits than this is treated as out of range, not parsed.
MAX_INT_DIGITS = 30

AUSTIN = (30.2672, -97.7431)  # central Austin: latitude and longitude, degrees
DEFAULT_BBOX = (30.260, -97.755, 30.285, -97.735)  # south, west, north, east
DEFAULT_MAX_CAMERAS = 100
MAX_CAMERAS = 1000
DEFAULT_TIMEOUT_S = 600
MAX_TIMEOUT_S = 3600
MODELS = ("yolox_m", "yolox_s")
DEFAULT_MODEL = "yolox_m"  # the sweep's model
CONCURRENCY = 8

# Person box heights in pixels of the original frame: (name, lowest height in the band).
BANDS: tuple[tuple[str, int], ...] = (
    ("<31", 0),
    ("31-45", 31),
    ("46-79", 46),
    ("80-119", 80),
    ("120-199", 120),
    ("200+", 200),
)
HD = (1920, 1080)

# The JPEG decoder's reduced-size modes, smallest reduction first.
REDUCTIONS: tuple[tuple[int, int], ...] = (
    (1, cv2.IMREAD_COLOR),
    (2, cv2.IMREAD_REDUCED_COLOR_2),
    (4, cv2.IMREAD_REDUCED_COLOR_4),
    (8, cv2.IMREAD_REDUCED_COLOR_8),
)
# Start-of-frame markers (C0-CF, except DHT C4, JPG C8 and DAC CC) carry the frame size.
SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# Markers without a length field: TEM and the restart markers.
STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD8)})

FailureKind = Literal["timeout", "http", "decode", "network", "detector"]
FAILURE_KINDS: tuple[FailureKind, ...] = (*fetch.ERROR_KINDS, "detector")

Frame = npt.NDArray[np.uint8]
BBox = tuple[float, float, float, float]


class PilotError(RuntimeError):
    """The pass cannot run: the camera list could not be fetched or is not a list."""


class FrameDetector(Protocol):
    def detect(self, frame: Frame) -> list[detect.Detection]: ...


@dataclass(frozen=True, slots=True)
class UrlPolicy:
    """The one scheme and network location a URL may have."""

    scheme: str
    netloc: str

    def allows(self, url: str) -> bool:
        """True for a printable-ASCII URL without spaces or backslashes whose scheme and
        network location are exactly the policy's: no userinfo, no other host, and no
        port other than the one in `netloc` (none at all for the pinned hosts)."""
        if not (0 < len(url) <= MAX_URL_LENGTH and url.isascii() and url.isprintable()):
            return False
        if " " in url or "\\" in url:
            return False
        try:
            parts = urllib.parse.urlsplit(url)
        except ValueError:
            return False
        return (
            parts.scheme == self.scheme
            and parts.netloc == self.netloc
            and parts.path.startswith("/")
        )


DATASET_POLICY = UrlPolicy("https", DATASET_HOST)
SCREENSHOT_POLICY = UrlPolicy("https", SCREENSHOT_HOST)


@dataclass(frozen=True, slots=True)
class Selection:
    listed: int
    skipped: int
    refused: int
    urls: list[str]


# Camera list ----------------------------------------------------------------------------


def _parse_int(text: str) -> float | int:
    # json would otherwise raise ValueError for the whole list on one integer over
    # Python's digit limit; a huge number is out of range for one record only.
    return math.inf if len(text.lstrip("-")) > MAX_INT_DIGITS else int(text)


def decode_dataset(body: bytes) -> list[object]:
    """The camera list as a JSON array. Raises PilotError for anything else."""
    if len(body) > MAX_DATASET_BYTES:
        raise PilotError(f"camera list exceeds {MAX_DATASET_BYTES} bytes")
    try:
        payload = json.loads(body, parse_int=_parse_int)
    except RecursionError:
        raise PilotError("camera list is nested too deeply") from None
    except (ValueError, TypeError, OverflowError):  # includes UnicodeDecodeError
        raise PilotError("camera list is not valid JSON") from None
    if not isinstance(payload, list):
        raise PilotError("camera list is not a JSON array")
    return payload


def _coordinate(value: object, limit: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if abs(value) > limit:  # compared before float(): a huge int would overflow
        return None
    return float(value)


def _camera(record: object) -> tuple[float, float, str] | None:
    """(latitude, longitude, screenshot URL) of a well-formed record, else None."""
    if not isinstance(record, dict):
        return None
    url, location = record.get("screenshot_address"), record.get("location")
    if not isinstance(url, str) or not isinstance(location, dict):
        return None
    coordinates = location.get("coordinates")  # GeoJSON: longitude, then latitude
    if location.get("type") != "Point" or not isinstance(coordinates, list):
        return None
    if len(coordinates) != 2:
        return None
    lon, lat = _coordinate(coordinates[0], 180), _coordinate(coordinates[1], 90)
    if lon is None or lat is None:
        return None
    return lat, lon, url


def select_cameras(
    records: Sequence[object],
    bbox: BBox,
    *,
    policy: UrlPolicy = SCREENSHOT_POLICY,
    max_cameras: int = DEFAULT_MAX_CAMERAS,
) -> Selection:
    """The screenshot URLs of the first `max_cameras` cameras inside `bbox` (inclusive)
    whose URL `policy` allows, in list order. Malformed records are counted as skipped,
    and cameras inside the box with any other URL as refused."""
    south, west, north, east = bbox
    urls: list[str] = []
    skipped = refused = 0
    for record in records:
        camera = _camera(record)
        if camera is None:
            skipped += 1
            continue
        lat, lon, url = camera
        if not (south <= lat <= north and west <= lon <= east):
            continue
        if not policy.allows(url):
            refused += 1
        elif len(urls) < max_cameras:
            urls.append(url)
    return Selection(len(records), skipped, refused, urls)


def fetch_dataset(url: str, policy: UrlPolicy, timeout_s: float) -> list[object]:
    """One GET of the camera list. Raises PilotError for any failure; messages name the
    failure's type only."""
    if not policy.allows(url):
        raise PilotError("camera list URL is not on the pinned host")
    try:
        body = registry.bounded_get(
            url, timeout_s=timeout_s, max_bytes=MAX_DATASET_BYTES, schemes=(policy.scheme,)
        )
    except registry.BodyTooLarge:
        raise PilotError(f"camera list exceeds {MAX_DATASET_BYTES} bytes") from None
    except urllib.error.HTTPError as exc:  # any status outside 2xx, redirects included
        exc.close()
        raise PilotError(f"camera list request failed (HTTP {exc.code})") from None
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise PilotError(f"camera list request failed ({type(exc).__name__})") from None
    return decode_dataset(body)


# Frames ---------------------------------------------------------------------------------


class _Refused(Exception):
    """The image host answered with a redirect, which is never followed."""


class _Failed(Exception):
    def __init__(self, kind: FailureKind) -> None:
        super().__init__(kind)
        self.kind: FailureKind = kind


@dataclass(frozen=True, slots=True, eq=False)
class _Decoded:
    frame: Frame
    width: int  # of the original frame
    height: int


def _download(url: str, scheme: str, timeout_s: float) -> bytes:
    """One GET of an image, failures categorised as `fetch` categorises them, except a
    redirect, which raises _Refused."""
    try:
        return registry.bounded_get(
            url, timeout_s=timeout_s, max_bytes=fetch.MAX_FRAME_BYTES, schemes=(scheme,)
        )
    except registry.BodyTooLarge:
        raise _Failed("http") from None
    except urllib.error.HTTPError as exc:
        exc.close()
        if 300 <= exc.code < 400:
            raise _Refused from None
        raise _Failed("http") from None
    except urllib.error.URLError as exc:
        raise _Failed("timeout" if isinstance(exc.reason, TimeoutError) else "network") from None
    except TimeoutError:
        raise _Failed("timeout") from None
    except (OSError, http.client.HTTPException, ValueError):
        raise _Failed("network") from None


def jpeg_size(body: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's start-of-frame header, or None if none is found
    before the image data starts."""
    pos = 2  # after the start-of-image marker
    while pos + 4 <= len(body):
        if body[pos] != 0xFF:
            return None
        marker = body[pos + 1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker in STANDALONE_MARKERS:
            pos += 2
            continue
        if marker in (0xD8, 0xD9, 0xDA):  # another SOI, EOI or start of scan
            return None
        length = int.from_bytes(body[pos + 2 : pos + 4], "big")
        if length < 2:
            return None
        if marker in SOF_MARKERS:
            if length < 7 or pos + 9 > len(body):
                return None
            height = int.from_bytes(body[pos + 5 : pos + 7], "big")
            width = int.from_bytes(body[pos + 7 : pos + 9], "big")
            return (width, height) if width and height else None
        pos += 2 + length
    return None


def decode_frame(body: bytes) -> _Decoded:
    """Decode a complete JPEG in memory, reduced by the smallest factor that brings it
    under the engine's pixel cap. Raises _Failed("decode") for anything else."""
    if not body.startswith(fetch.JPEG_SOI) or fetch.JPEG_EOI not in body[-fetch.EOI_WINDOW :]:
        raise _Failed("decode")
    size = jpeg_size(body)
    if size is None:
        raise _Failed("decode")
    width, height = size
    fits = [
        (flag, -(-width // factor), -(-height // factor))
        for factor, flag in REDUCTIONS
        if -(-width // factor) * -(-height // factor) <= MAX_IMAGE_PIXELS
    ]
    if not fits:
        raise _Failed("decode")
    flag, out_w, out_h = fits[0]
    try:
        frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), flag)
    except cv2.error:
        raise _Failed("decode") from None
    if frame is None or frame.ndim != 3 or frame.dtype != np.uint8:
        raise _Failed("decode")
    if frame.shape[:2] != (out_h, out_w) or frame.shape[2] != 3:
        raise _Failed("decode")  # the header and the decoded image disagree
    return _Decoded(np.asarray(frame, dtype=np.uint8), width, height)


def _fetch_one(url: str, scheme: str, end: float) -> _Decoded:
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise _Failed("timeout")
    body = _download(url, scheme, min(FRAME_TIMEOUT_S, remaining))
    try:
        return decode_frame(body)
    finally:
        del body


# The pass -------------------------------------------------------------------------------


def height_band(height: int) -> str:
    """The band of a person box `height` pixels tall (see BANDS)."""
    name = BANDS[0][0]
    for band, lowest in BANDS:
        if height >= lowest:
            name = band
    return name


@dataclass(slots=True)
class Tally:
    """Counts only: never a URL, a box or a camera."""

    refused: int = 0
    frames_ok: int = 0
    persons: int = 0
    umbrellas: int = 0
    failed: Counter[FailureKind] = field(default_factory=Counter)
    resolutions: Counter[str] = field(default_factory=Counter)
    bands: Counter[str] = field(default_factory=Counter)
    bands_hd: Counter[str] = field(default_factory=Counter)

    def add(self, decoded: _Decoded, found: Sequence[detect.Detection]) -> None:
        scale_x = decoded.width / decoded.frame.shape[1]
        scale_y = decoded.height / decoded.frame.shape[0]
        is_hd = (decoded.width, decoded.height) == HD
        self.frames_ok += 1
        self.resolutions[f"{decoded.width}x{decoded.height}"] += 1
        for d in found:
            if d.label == "umbrella":
                self.umbrellas += 1
                continue
            x1, y1, x2, y2 = d.box
            original = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            band = height_band(aggregate.box_height(original))
            self.persons += 1
            self.bands[band] += 1
            if is_hd:
                self.bands_hd[band] += 1


def _bands(counts: Counter[str]) -> dict[str, int]:
    return {name: counts[name] for name, _ in BANDS}


def run_frames(
    urls: Sequence[str],
    detector: FrameDetector,
    *,
    scheme: str,
    end: float,
    tally: Tally,
    concurrency: int = CONCURRENCY,
) -> None:
    """Fetch every URL once, `concurrency` at a time, and detect in each frame as it
    arrives, until the monotonic deadline `end`. Each frame is dropped after detection;
    only counts reach `tally`."""
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="pilot")
    try:
        pending: set[Future[_Decoded]] = {pool.submit(_fetch_one, u, scheme, end) for u in urls}
        while pending:
            done, pending = wait(
                pending, timeout=max(0.0, end - time.monotonic()) + 1.0, return_when=FIRST_COMPLETED
            )
            for future in done:
                _settle(future, detector, end, tally)
            if not done and time.monotonic() > end + 1.0:
                # Every request is bounded by the deadline; this is a backstop only.
                tally.failed["timeout"] += len(pending)
                for future in pending:
                    future.cancel()
                pending = set()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _settle(future: Future[_Decoded], detector: FrameDetector, end: float, tally: Tally) -> None:
    try:
        decoded = future.result()
    except _Refused:
        tally.refused += 1
        return
    except _Failed as exc:
        tally.failed[exc.kind] += 1
        return
    try:
        if time.monotonic() > end:
            tally.failed["timeout"] += 1
            return
        try:
            found = detector.detect(decoded.frame)
        except detect.DetectorError:
            tally.failed["detector"] += 1
            return
        tally.add(decoded, found)
    finally:
        del decoded


def report(
    *,
    started_at: datetime.datetime,
    sun_elevation: float,
    model: str,
    selection: Selection,
    tally: Tally,
) -> dict[str, object]:
    """The pass's output: exactly the keys below, counts only."""
    return {
        "source": SOURCE,
        "started_at": started_at.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "sun_elevation_deg": round(sun_elevation, 1),
        "model": model,
        "cameras_listed": selection.listed,
        "cameras_selected": len(selection.urls),
        "records_skipped": selection.skipped,
        "refused_url": selection.refused + tally.refused,
        "frames_ok": tally.frames_ok,
        "frames_failed": {kind: tally.failed[kind] for kind in FAILURE_KINDS},
        "resolutions": dict(tally.resolutions),
        "persons_total": tally.persons,
        "umbrellas_total": tally.umbrellas,
        "persons_by_height_band": _bands(tally.bands),
        "persons_by_height_band_1080p": _bands(tally.bands_hd),
    }


# Daylight -------------------------------------------------------------------------------

# The spot-check tool's solar_elevation, copied: the static privacy guard forbids any
# other engine module from importing the one module allowed to write images, and the
# acceptance tests assert that both give the same value.


def solar_elevation(moment: datetime.datetime, latitude: float, longitude: float) -> float:
    """The sun's elevation above the horizon, in degrees, at `moment` (an aware datetime)
    seen from `latitude`, `longitude`: the geometric position of its centre, without
    refraction. NOAA's general solar position formulae (after Meeus, "Astronomical
    Algorithms"), good to a minute of time or so."""
    moment = moment.astimezone(datetime.UTC)
    j2000 = datetime.datetime(2000, 1, 1, 12, tzinfo=datetime.UTC)
    t = (moment - j2000).total_seconds() / 86400 / 36525  # Julian centuries from J2000.0
    mean_long = math.radians((280.46646 + t * (36000.76983 + t * 0.0003032)) % 360)
    mean_anomaly = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    centre = (
        math.sin(mean_anomaly) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * mean_anomaly) * (0.019993 - 0.000101 * t)
        + math.sin(3 * mean_anomaly) * 0.000289
    )
    omega = math.radians(125.04 - 1934.136 * t)
    apparent_long = math.radians(
        math.degrees(mean_long) + centre - 0.00569 - 0.00478 * math.sin(omega)
    )
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    obliquity = math.radians(23 + (26 + seconds / 60) / 60 + 0.00256 * math.cos(omega))
    declination = math.asin(math.sin(obliquity) * math.sin(apparent_long))
    y = math.tan(obliquity / 2) ** 2
    equation_of_time = 4 * math.degrees(  # minutes
        y * math.sin(2 * mean_long)
        - 2 * eccentricity * math.sin(mean_anomaly)
        + 4 * eccentricity * y * math.sin(mean_anomaly) * math.cos(2 * mean_long)
        - 0.5 * y * y * math.sin(4 * mean_long)
        - 1.25 * eccentricity * eccentricity * math.sin(2 * mean_anomaly)
    )
    midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes = (moment - midnight).total_seconds() / 60
    hour_angle = math.radians((minutes + equation_of_time + 4 * longitude) / 4 - 180)
    lat = math.radians(latitude)
    cos_zenith = math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(
        declination
    ) * math.cos(hour_angle)
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))


# Command line ---------------------------------------------------------------------------


def _bbox(text: str) -> BBox:
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected S,W,N,E")
    try:
        south, west, north, east = (float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("expected four numbers S,W,N,E") from None
    values = (south, west, north, east)
    if not all(math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError("coordinates must be finite")
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise argparse.ArgumentTypeError("need -90 <= S < N <= 90 and -180 <= W < E <= 180")
    return values


def _max_cameras(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected an integer") from None
    if not 1 <= value <= MAX_CAMERAS:
        raise argparse.ArgumentTypeError(f"must be 1 to {MAX_CAMERAS}")
    return value


def _timeout(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a number of seconds") from None
    if not 0 < value <= MAX_TIMEOUT_S:  # also rejects NaN and infinity
        raise argparse.ArgumentTypeError(f"must be more than 0 and at most {MAX_TIMEOUT_S}")
    return value


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.tools.pilot_heights",
        description="Measure person box heights on Austin's HD camera stills; print counts.",
    )
    ap.add_argument("--live", action="store_true", help="required: run one pass (network)")
    ap.add_argument(
        "--bbox",
        type=_bbox,
        default=DEFAULT_BBOX,
        metavar="S,W,N,E",
        help="cameras inside this box, degrees (default: downtown Austin)",
    )
    ap.add_argument(
        "--max-cameras",
        type=_max_cameras,
        default=DEFAULT_MAX_CAMERAS,
        metavar="N",
        help=f"at most N cameras, 1 to {MAX_CAMERAS} (default {DEFAULT_MAX_CAMERAS})",
    )
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    ap.add_argument(
        "--timeout",
        type=_timeout,
        default=DEFAULT_TIMEOUT_S,
        metavar="S",
        help=f"wall-clock deadline for the whole pass, seconds (default {DEFAULT_TIMEOUT_S})",
    )
    return ap


def open_detector(model: str) -> FrameDetector:
    """The sweep's detector: the pinned model, default thresholds."""
    return detect.Detector(detect.model_path(f"{model}.onnx"))


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def main(
    argv: Sequence[str] | None = None,
    *,
    now: Callable[[], datetime.datetime] = _utc_now,
    open_detector: Callable[[str], FrameDetector] = open_detector,
    dataset_url: str = DATASET_URL,
    dataset_policy: UrlPolicy = DATASET_POLICY,
    image_policy: UrlPolicy = SCREENSHOT_POLICY,
) -> int:
    ap = _parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:  # usage errors (2) and --help (0)
        return exc.code if isinstance(exc.code, int) else 2
    if not args.live:
        ap.print_usage(sys.stderr)
        print(
            f"{ap.prog}: error: refusing to run without --live (it uses the network)",
            file=sys.stderr,
        )
        return 2

    started = time.monotonic()
    end = started + args.timeout
    started_at = now()
    elevation = solar_elevation(started_at, *AUSTIN)
    if elevation < 0:
        print(
            f"pilot_heights: the sun is below the horizon in Austin ({elevation:.1f} degrees);"
            " run it in daylight",
            file=sys.stderr,
        )
        return 1

    try:
        detector = open_detector(args.model)
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise PilotError("deadline passed before the camera list was fetched")
        records = fetch_dataset(dataset_url, dataset_policy, min(DATASET_TIMEOUT_S, remaining))
    except (PilotError, detect.DetectorError) as exc:
        print(f"pilot_heights: error: {exc}", file=sys.stderr)
        return 1
    selection = select_cameras(
        records, args.bbox, policy=image_policy, max_cameras=args.max_cameras
    )
    del records
    tally = Tally()
    run_frames(selection.urls, detector, scheme=image_policy.scheme, end=end, tally=tally)
    result = report(
        started_at=started_at,
        sun_elevation=elevation,
        model=args.model,
        selection=selection,
        tally=tally,
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
