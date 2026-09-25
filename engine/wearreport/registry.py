"""The TfL JamCam registry: which cameras are available right now.

Availability changes over time, so the registry is fetched fresh on every sweep and
never persisted. Only metadata is read here; frames are fetched elsewhere.

Run `python -m wearreport.registry` to print the current count.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from wearreport.settings import load_settings

JAMCAM_URL = "https://api.tfl.gov.uk/Place/Type/JamCam"
USER_AGENT = "wearreport-engine (+https://github.com/jakegu1/thewearreport)"
TIMEOUT_S = 30
MAX_ATTEMPTS = 3
BACKOFF_S = (1, 2)  # sleep before attempts 2 and 3
# The real response is about 1 MB; a larger body than this is refused, never parsed.
MAX_BODY_BYTES = 16 * 1024 * 1024

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
            # A client error will not change on retry; 429 (rate limited) may.
            if 400 <= exc.code < 500 and exc.code != 429:
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
