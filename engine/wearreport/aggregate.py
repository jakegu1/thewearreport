"""One sweep, end to end, reduced to one aggregate record (schema `sweep.v1`).

`run_sweep` lists the cameras, fetches one frame from each in memory, runs the detector
on every frame, reads the weather and returns the record that `build_record` makes from
the per-camera counts. Privacy (AGENTS.md INV-1): each frame is dropped as soon as the
detector has seen it, and only counts leave the detection loop: no boxes, coordinates,
scores or pixels reach the record or the logs. Honesty (INV-6): every camera the
registry listed appears in exactly one of `frames_ok` and `frames_failed`, and nothing is
estimated or filled in; a sweep whose weather is unavailable records `null`. A camera id
the registry lists more than once counts once. An id that cannot be published as a record
key is never fetched and never appears in `per_camera`, but it still counts in
`cameras_listed` and under `frames_failed.invalid_id`, a category that is present only
when it is at least 1 (a missing category means 0).

`check_record` enforces the schema (data/schema/sweep.v1.json) and the rules between
fields that a JSON Schema cannot express. The publisher runs it before writing a record
and on every record it reads back.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import metadata
from typing import Any, Final

from wearreport import detect, fetch, registry, weather
from wearreport.settings import load_settings

logger = logging.getLogger("wearreport.aggregate")

SCHEMA_VERSION: Final = "sweep.v1"
SOURCE: Final = "tfl-jamcam"
ERROR_KINDS: Final[tuple[str, ...]] = (*fetch.ERROR_KINDS, "detect")
# Listed ids that cannot be published; this category is present only when it is non-zero.
INVALID_ID: Final = "invalid_id"
TFL_ATTRIBUTION: Final = "Powered by TfL Open Data"
WEATHER_ATTRIBUTION: Final[Mapping[str, str]] = {
    "metoffice": weather.METOFFICE_ATTRIBUTION,
    "openmeteo": "Weather data by Open-Meteo.com",
}
# The registry lists about 900 cameras; far more means the response is not what we expect.
MAX_CAMERAS: Final = 2000

SWEEP_ID = re.compile(
    r"[0-9]{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3])[0-5][0-9]Z"
)
UTC_TIME = re.compile(
    r"[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)
CAMERA_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
ENGINE_VERSION = re.compile(r"[0-9A-Za-z.+_-]{1,64}")
MODEL_NAME = re.compile(r"[a-z0-9_]{1,32}")
SHA256 = re.compile(r"[0-9a-f]{64}")

RECORD_KEYS: Final = frozenset(
    {
        "schema",
        "sweep_id",
        "started_at",
        "finished_at",
        "source",
        "cameras_listed",
        "frames_ok",
        "frames_failed",
        "persons_total",
        "umbrellas_total",
        "per_camera",
        "weather",
        "engine_version",
        "model",
        "model_sha256",
        "attribution",
    }
)
WEATHER_KEYS: Final = frozenset({"temp_c", "apparent_c", "precip_mm", "observed_at", "source"})
COUNT_KEYS: Final = frozenset({"persons", "umbrellas"})

Record = dict[str, Any]


class RecordError(ValueError):
    """A record does not satisfy schema sweep.v1 or its rules between fields."""


class SweepError(RuntimeError):
    """The sweep could not produce a record (for example, the registry is unusable)."""


@dataclass(frozen=True, slots=True)
class Observation:
    """What one camera contributed to a sweep: an error category, or two counts."""

    camera_id: str
    error: str | None
    persons: int = 0
    umbrellas: int = 0


# Formatting and parsing ----------------------------------------------------------------


def format_utc(moment: datetime) -> str:
    """YYYY-MM-DDTHH:MM:SSZ for an aware datetime; fractions of a second are dropped."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise RecordError("time must be timezone-aware")
    t = moment.astimezone(UTC)
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}T{t.hour:02d}:{t.minute:02d}:{t.second:02d}Z"


def sweep_id_for(started_at: datetime) -> str:
    """The sweep id: the UTC minute `started_at` falls in, YYYYMMDDTHHMMZ."""
    stamp = format_utc(started_at)
    return f"{stamp[0:4]}{stamp[5:7]}{stamp[8:10]}T{stamp[11:13]}{stamp[14:16]}Z"


def parse_utc(text: object) -> datetime:
    """Parse a YYYY-MM-DDTHH:MM:SSZ string; raise RecordError for anything else."""
    if not isinstance(text, str) or not UTC_TIME.fullmatch(text):
        raise RecordError("time is not YYYY-MM-DDTHH:MM:SSZ")
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:  # e.g. 2026-02-30
        raise RecordError("time is not a calendar date") from None


def engine_version() -> str:
    """The installed engine version, or "unknown" when the package metadata is missing."""
    try:
        version = metadata.version("wearreport")
    except metadata.PackageNotFoundError:
        return "unknown"
    return version if ENGINE_VERSION.fullmatch(version) else "unknown"


# Building and checking records ---------------------------------------------------------


def _weather_fields(conditions: weather.Conditions) -> dict[str, Any]:
    return {
        "temp_c": conditions.temp_c,
        "apparent_c": conditions.apparent_c,
        "precip_mm": conditions.precip_mm,
        "observed_at": format_utc(conditions.observed_at),
        "source": conditions.source,
    }


def build_record(
    observations: Sequence[Observation],
    *,
    started_at: datetime,
    finished_at: datetime,
    weather: weather.Conditions | None,
    engine_version: str,
    model_name: str,
    model_sha256: str,
) -> Record:
    """The sweep.v1 record for one sweep: one observation per camera the registry listed.

    An observation with error INVALID_ID stands for a listed id that cannot be published;
    its camera_id is only used to detect duplicates. Raises RecordError when the result
    would not be a valid record (a duplicate camera id, a malformed id in per_camera, an
    unknown error category, a negative count...).
    """
    failed: Counter[str] = Counter()
    per_camera: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for obs in observations:
        if obs.camera_id in seen:
            raise RecordError("duplicate camera id")
        seen.add(obs.camera_id)
        if obs.error is not None:
            if obs.error not in ERROR_KINDS and obs.error != INVALID_ID:
                raise RecordError("unknown error category")
            failed[obs.error] += 1
        elif obs.persons or obs.umbrellas:
            per_camera[obs.camera_id] = {"persons": obs.persons, "umbrellas": obs.umbrellas}
    frames_failed = {kind: failed[kind] for kind in ERROR_KINDS}
    if failed[INVALID_ID]:
        frames_failed[INVALID_ID] = failed[INVALID_ID]
    attribution = [TFL_ATTRIBUTION]
    if weather is not None:
        attribution.append(WEATHER_ATTRIBUTION[weather.source])
    record: Record = {
        "schema": SCHEMA_VERSION,
        "sweep_id": sweep_id_for(started_at),
        "started_at": format_utc(started_at),
        "finished_at": format_utc(finished_at),
        "source": SOURCE,
        "cameras_listed": len(observations),
        "frames_ok": len(observations) - sum(frames_failed.values()),
        "frames_failed": frames_failed,
        "persons_total": sum(c["persons"] for c in per_camera.values()),
        "umbrellas_total": sum(c["umbrellas"] for c in per_camera.values()),
        "per_camera": per_camera,
        "weather": None if weather is None else _weather_fields(weather),
        "engine_version": engine_version,
        "model": model_name,
        "model_sha256": model_sha256,
        "attribution": attribution,
    }
    check_record(record)
    return record


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecordError(f"{what} is not a non-negative integer")
    return value


def _object(value: object, keys: frozenset[str], what: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RecordError(f"{what} does not have exactly the expected fields")
    return value


def _string(value: object, pattern: re.Pattern[str], what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise RecordError(f"{what} is malformed")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RecordError(f"{what} is not a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        finite = False
    if not finite:
        raise RecordError(f"{what} is not finite")
    return float(value)


def _check_frames_failed(value: object) -> int:
    """The total of a frames_failed object: the five categories, plus invalid_id >= 1."""
    required = frozenset(ERROR_KINDS)
    if isinstance(value, dict) and INVALID_ID in value:
        fields = _object(value, required | {INVALID_ID}, "frames_failed")
        if _count(fields[INVALID_ID], "frames_failed count") < 1:
            raise RecordError("frames_failed.invalid_id is present but not at least 1")
    else:
        fields = _object(value, required, "frames_failed")
    return sum(_count(v, "frames_failed count") for v in fields.values())


def _check_weather(value: object) -> None:
    if value is None:
        return
    fields = _object(value, WEATHER_KEYS, "weather")
    for name in ("temp_c", "apparent_c", "precip_mm"):
        _number(fields[name], f"weather.{name}")
    if _number(fields["precip_mm"], "weather.precip_mm") < 0:
        raise RecordError("weather.precip_mm is negative")
    parse_utc(fields["observed_at"])
    if fields["source"] not in WEATHER_ATTRIBUTION:
        raise RecordError("weather.source is unknown")


def _check_attribution(value: object, conditions: object) -> None:
    expected = [TFL_ATTRIBUTION]
    if isinstance(conditions, dict):
        expected.append(WEATHER_ATTRIBUTION[conditions["source"]])
    if value != expected:
        raise RecordError("attribution does not match the sources used")


def check_record(record: object) -> None:
    """Raise RecordError unless `record` is a valid sweep.v1 record.

    Covers everything data/schema/sweep.v1.json says, plus: sweep_id is the minute of
    started_at, finished_at is not earlier, the counts add up, and the attribution names
    exactly the sources used. frames_failed has the five categories of ERROR_KINDS, and
    also invalid_id when that count is at least 1.
    """
    rec = _object(record, RECORD_KEYS, "record")
    if rec["schema"] != SCHEMA_VERSION or rec["source"] != SOURCE:
        raise RecordError("schema or source is wrong")
    sweep_id = _string(rec["sweep_id"], SWEEP_ID, "sweep_id")
    started, finished = parse_utc(rec["started_at"]), parse_utc(rec["finished_at"])
    if sweep_id_for(started) != sweep_id:
        raise RecordError("sweep_id is not the minute of started_at")
    if finished < started:
        raise RecordError("finished_at is before started_at")
    listed = _count(rec["cameras_listed"], "cameras_listed")
    ok = _count(rec["frames_ok"], "frames_ok")
    if ok + _check_frames_failed(rec["frames_failed"]) != listed:
        raise RecordError("frames_ok and frames_failed do not add up to cameras_listed")
    per_camera = rec["per_camera"]
    if not isinstance(per_camera, dict) or len(per_camera) > min(ok, MAX_CAMERAS):
        raise RecordError("per_camera is not a map, or has more entries than frames_ok")
    persons = umbrellas = 0
    for camera_id, value in per_camera.items():
        _string(camera_id, CAMERA_ID, "camera id")
        counts = _object(value, COUNT_KEYS, "per_camera entry")
        p = _count(counts["persons"], "persons")
        u = _count(counts["umbrellas"], "umbrellas")
        if p == 0 and u == 0:
            raise RecordError("per_camera holds an entry with nothing detected")
        persons, umbrellas = persons + p, umbrellas + u
    if _count(rec["persons_total"], "persons_total") != persons:
        raise RecordError("persons_total is not the sum over per_camera")
    if _count(rec["umbrellas_total"], "umbrellas_total") != umbrellas:
        raise RecordError("umbrellas_total is not the sum over per_camera")
    _check_weather(rec["weather"])
    _string(rec["engine_version"], ENGINE_VERSION, "engine_version")
    _string(rec["model"], MODEL_NAME, "model")
    _string(rec["model_sha256"], SHA256, "model_sha256")
    _check_attribution(rec["attribution"], rec["weather"])


# The pipeline --------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _default_list_cameras() -> Sequence[registry.Camera]:
    return registry.list_cameras(load_settings().tfl_app_key)


def usable_cameras(
    cameras: Sequence[registry.Camera],
) -> tuple[list[registry.Camera], list[str]]:
    """Split the listed cameras into those whose id can be published and the ids that
    cannot. Each id counts once: a duplicate keeps its first listing. The registry is
    external data, so the unusable ids are never logged or published."""
    if len(cameras) > MAX_CAMERAS:
        raise SweepError(f"the registry lists more than {MAX_CAMERAS} cameras")
    kept: dict[str, registry.Camera] = {}
    invalid: dict[str, None] = {}
    for camera in cameras:
        if camera.id in kept or camera.id in invalid:
            continue
        if CAMERA_ID.fullmatch(camera.id):
            kept[camera.id] = camera
        else:
            invalid[camera.id] = None
    if len(kept) != len(cameras):
        logger.warning(
            "cameras not fetched: malformed or duplicate id",
            extra={
                "invalid_ids": len(invalid),
                "duplicates": len(cameras) - len(kept) - len(invalid),
            },
        )
    return list(kept.values()), list(invalid)


def detect_counts(detector: detect.Detector, results: list[fetch.FrameResult]) -> list[Observation]:
    """One observation per fetch result, in order. Consumes `results`: each frame is
    released as soon as the detector has seen it, and only the two counts are kept."""
    results.reverse()
    observations: list[Observation] = []
    while results:
        result = results.pop()
        camera_id, frame, error = result.camera_id, result.frame, result.error
        del result
        if error is not None or frame is None:
            # fetch returns a frame exactly when it has no error; anything else is unusable.
            observations.append(Observation(camera_id, error or "decode"))
            continue
        try:
            found = detector.detect(frame)
        except (ValueError, detect.DetectorError) as exc:
            logger.warning(
                "detection failed", extra={"camera_id": camera_id, "error": type(exc).__name__}
            )
            observations.append(Observation(camera_id, "detect"))
            continue
        finally:
            del frame
        persons = sum(d.label == "person" for d in found)
        umbrellas = sum(d.label == "umbrella" for d in found)
        del found
        observations.append(Observation(camera_id, None, persons, umbrellas))
    return observations


def run_sweep(
    detector: detect.Detector,
    model_sha256: str,
    *,
    model_name: str,
    list_cameras: Callable[[], Sequence[registry.Camera]] | None = None,
    fetch_frames: Callable[[Sequence[registry.Camera]], list[fetch.FrameResult]] | None = None,
    conditions: Callable[[datetime], weather.Conditions | None] | None = None,
    now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> Record:
    """Run one sweep and return its record. Nothing is written anywhere.

    The weather is read first, for the minute the sweep starts, so that a
    WeatherConfigError (the dev-only flag set in production) stops the sweep before any
    camera is contacted. Raises SweepError when the registry cannot be used, and lets
    WeatherConfigError propagate. A detector that fails on a frame (DetectorError or
    ValueError) does not stop the sweep: that frame is counted under "detect", so a model
    that fails at inference yields a record with every usable frame failed as "detect".
    """
    started_at = (now or _utc_now)().astimezone(UTC).replace(microsecond=0)
    clock_start = monotonic()
    read_weather = conditions or weather.current_conditions
    current = read_weather(started_at)
    try:
        listed = (list_cameras or _default_list_cameras)()
    except registry.RegistryError as exc:
        raise SweepError(str(exc)) from None
    cameras, invalid = usable_cameras(listed)
    results = (fetch_frames or fetch.fetch_sweep)(cameras)
    if [r.camera_id for r in results] != [c.id for c in cameras]:
        raise SweepError("fetch results do not match the cameras listed")
    observations = detect_counts(detector, results)
    del results
    observations += [Observation(camera_id, INVALID_ID) for camera_id in invalid]
    finished_at = started_at + timedelta(seconds=max(0.0, monotonic() - clock_start))
    record = build_record(
        observations,
        started_at=started_at,
        finished_at=finished_at,
        weather=current,
        engine_version=engine_version(),
        model_name=model_name,
        model_sha256=model_sha256,
    )
    logger.info(
        "sweep aggregated",
        extra={
            "sweep_id": record["sweep_id"],
            "cameras_listed": record["cameras_listed"],
            "frames_ok": record["frames_ok"],
            **{f"failed_{k}": v for k, v in record["frames_failed"].items()},
            "persons_total": record["persons_total"],
            "umbrellas_total": record["umbrellas_total"],
            "weather": None if current is None else current.source,
        },
    )
    return record
