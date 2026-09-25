"""The record builder, check_record and the sweep pipeline (wearreport.aggregate)."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import jsonschema
import numpy as np
import pytest

from wearreport import aggregate, detect, fetch, registry, weather

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[3] / "data" / "schema" / "sweep.v1.json").read_text()
)
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
T0 = datetime(2026, 7, 15, 12, 30, 5, 250_000, tzinfo=UTC)
DIGEST = "a" * 64
FRAME = np.zeros((288, 352, 3), dtype=np.uint8)


def _obs(cam: str, error: str | None = None, p: int = 0, u: int = 0) -> aggregate.Observation:
    return aggregate.Observation(cam, error, p, u)


def _build(
    observations: Sequence[aggregate.Observation] = (),
    started: datetime = T0,
    conditions: weather.Conditions | None = None,
) -> dict[str, Any]:
    return aggregate.build_record(
        observations,
        started_at=started,
        finished_at=started + timedelta(minutes=3),
        weather=conditions,
        engine_version="0.0.0",
        model_name="yolox_m",
        model_sha256=DIGEST,
    )


def _valid() -> dict[str, Any]:
    conditions = weather.Conditions(
        9.5, 7.0, 1.5, datetime(2026, 7, 15, 13, tzinfo=UTC), "metoffice"
    )
    return _build(
        [_obs("A", p=2), _obs("B", u=1), _obs("C"), _obs("D", "network")], conditions=conditions
    )


# Time formatting -----------------------------------------------------------------------


def test_format_utc_drops_fractions_and_converts_offsets() -> None:
    bst = datetime(2026, 7, 15, 13, 30, 5, 999_999, tzinfo=ZoneInfo("Europe/London"))
    assert aggregate.format_utc(bst) == "2026-07-15T12:30:05Z"
    assert aggregate.format_utc(datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=-5)))) == (
        "2026-01-01T05:00:00Z"
    )


def test_format_utc_refuses_naive_times() -> None:
    with pytest.raises(aggregate.RecordError):
        aggregate.format_utc(datetime(2026, 7, 15, 12, 0))


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (datetime(2026, 7, 15, 23, 59, 59, 999_999, tzinfo=UTC), "20260715T2359Z"),
        (datetime(2026, 7, 16, 0, 0, tzinfo=UTC), "20260716T0000Z"),
        # 00:30 BST on 16 July is still 15 July in UTC: the id and the directory use UTC.
        (datetime(2026, 7, 16, 0, 30, tzinfo=ZoneInfo("Europe/London")), "20260715T2330Z"),
        (datetime(2026, 12, 31, 23, 59, 30, tzinfo=UTC), "20261231T2359Z"),
    ],
)
def test_sweep_id_is_the_utc_minute(moment: datetime, expected: str) -> None:
    assert aggregate.sweep_id_for(moment) == expected


@pytest.mark.parametrize(
    "text",
    [
        "2026-02-30T12:00:00Z",
        "2026-07-15T12:00:00",
        "2026-07-15T12:00:00+00:00",
        "2026-07-15 12:00:00Z",
        "2026-07-15T24:00:00Z",
        "",
        None,
        20260715,
    ],
)
def test_parse_utc_rejects_anything_but_utc_seconds(text: object) -> None:
    with pytest.raises(aggregate.RecordError):
        aggregate.parse_utc(text)


def test_engine_version_comes_from_the_package() -> None:
    assert aggregate.engine_version() == "0.0.0"


# build_record --------------------------------------------------------------------------


def test_build_record_counts_and_attribution() -> None:
    record = _valid()
    VALIDATOR.validate(record)
    assert record["frames_ok"] == 3 and record["cameras_listed"] == 4
    assert record["per_camera"] == {
        "A": {"persons": 2, "umbrellas": 0},
        "B": {"persons": 0, "umbrellas": 1},
    }
    assert record["attribution"] == ["Powered by TfL Open Data", "Powered by Met Office data"]
    assert _build()["attribution"] == ["Powered by TfL Open Data"]


def test_build_record_credits_open_meteo_in_development() -> None:
    dev = weather.Conditions(9.5, 7.0, 0.0, T0, "openmeteo")
    record = _build([_obs("A", p=1)], conditions=dev)
    VALIDATOR.validate(record)
    assert record["attribution"][1] == "Weather data by Open-Meteo.com"


@pytest.mark.parametrize(
    "observations",
    [
        [_obs("A", p=1), _obs("A")],  # duplicate camera
        [_obs("A", "teapot")],  # unknown error category
        [_obs("A", p=-1, u=2)],  # negative count
        [_obs("A/../B", p=1)],  # camera id that is not publishable
        [_obs("", p=1)],
        [
            aggregate.Observation("A", None, True, 0)
        ],  # a bool is not a count  # type: ignore[arg-type]
    ],
)
def test_build_record_refuses_inconsistent_observations(
    observations: list[aggregate.Observation],
) -> None:
    with pytest.raises(aggregate.RecordError):
        _build(observations)


def test_build_record_refuses_non_finite_weather() -> None:
    bad = weather.Conditions(float("nan"), 7.0, 0.0, T0, "metoffice")
    with pytest.raises(aggregate.RecordError):
        _build(conditions=bad)


# check_record against the schema -------------------------------------------------------

Mutation = Callable[[dict[str, Any]], object]

BOTH_REJECT: list[tuple[str, Mutation]] = [
    ("extra root key", lambda r: r.update(boxes=[])),
    ("missing key", lambda r: r.pop("model")),
    ("schema version", lambda r: r.update(schema="sweep.v2")),
    ("sweep_id month 13", lambda r: r.update(sweep_id="20261315T1230Z")),
    ("started_at offset", lambda r: r.update(started_at="2026-07-15T12:30:05+00:00")),
    ("count is a string", lambda r: r.update(cameras_listed="4")),
    ("count is a bool", lambda r: r.update(frames_ok=True)),
    ("count is a fraction", lambda r: r.update(persons_total=2.5)),
    ("frames_failed missing kind", lambda r: r["frames_failed"].pop("detect")),
    ("frames_failed extra kind", lambda r: r["frames_failed"].update(dns=0)),
    ("per_camera list", lambda r: r.update(per_camera=[])),
    (
        "per_camera id with slash",
        lambda r: r["per_camera"].update({"a/b": {"persons": 1, "umbrellas": 0}}),
    ),
    (
        "per_camera id too long",
        lambda r: r["per_camera"].update({"x" * 65: {"persons": 1, "umbrellas": 0}}),
    ),
    ("per_camera coordinates", lambda r: r["per_camera"]["A"].update(x=10, y=20)),
    ("per_camera all zero", lambda r: r["per_camera"].update(Z={"persons": 0, "umbrellas": 0})),
    ("weather extra", lambda r: r["weather"].update(wind=3)),
    ("weather missing field", lambda r: r["weather"].pop("precip_mm")),
    ("weather negative rain", lambda r: r["weather"].update(precip_mm=-0.1)),
    ("weather string temp", lambda r: r["weather"].update(temp_c="9")),
    ("weather source", lambda r: r["weather"].update(source="nws")),
    ("weather time", lambda r: r["weather"].update(observed_at="yesterday")),
    ("engine_version spaces", lambda r: r.update(engine_version="1 2")),
    ("model uppercase", lambda r: r.update(model="YOLOX")),
    ("digest uppercase", lambda r: r.update(model_sha256="A" * 64)),
    ("attribution missing TfL", lambda r: r.update(attribution=["Powered by Met Office data"])),
    ("attribution other text", lambda r: r["attribution"].append("Data by someone")),
]

# Rules between fields: the JSON Schema cannot express them, check_record enforces them.
ONLY_CHECK_REJECTS: list[tuple[str, Mutation]] = [
    ("sweep_id other minute", lambda r: r.update(sweep_id="20260715T1231Z")),
    ("finished before started", lambda r: r.update(finished_at="2026-07-15T12:30:04Z")),
    ("counts do not add up", lambda r: r.update(frames_ok=4)),
    ("persons_total not the sum", lambda r: r.update(persons_total=3)),
    ("umbrellas_total not the sum", lambda r: r.update(umbrellas_total=0)),
    ("more per_camera than frames_ok", lambda r: r.update(frames_ok=1, cameras_listed=2)),
    ("attribution without weather", lambda r: r.update(weather=None)),
    ("started_at not a date", lambda r: r.update(started_at="2026-02-30T12:30:05Z")),
    # JSON Schema counts 2.0 as an integer; the engine never writes one, and refuses it.
    ("count written as 2.0", lambda r: r.update(persons_total=2.0)),
]


@pytest.mark.parametrize("mutate", [m for _, m in BOTH_REJECT], ids=[n for n, _ in BOTH_REJECT])
def test_schema_and_check_record_reject_the_same_records(mutate: Mutation) -> None:
    record = _valid()
    mutate(record)
    with pytest.raises(jsonschema.ValidationError):
        VALIDATOR.validate(record)
    with pytest.raises(aggregate.RecordError):
        aggregate.check_record(record)


@pytest.mark.parametrize(
    "mutate", [m for _, m in ONLY_CHECK_REJECTS], ids=[n for n, _ in ONLY_CHECK_REJECTS]
)
def test_check_record_enforces_rules_between_fields(mutate: Mutation) -> None:
    record = _valid()
    mutate(record)
    with pytest.raises(aggregate.RecordError):
        aggregate.check_record(record)


@pytest.mark.parametrize(
    "value",
    [None, [], "record", 3, {"schema": "sweep.v1"}],
)
def test_check_record_rejects_non_records(value: object) -> None:
    with pytest.raises(aggregate.RecordError):
        aggregate.check_record(value)


@pytest.mark.parametrize("number", [float("inf"), float("nan"), 10**400])
def test_check_record_rejects_numbers_json_cannot_hold(number: float) -> None:
    record = _valid()
    record["weather"]["temp_c"] = number
    with pytest.raises(aggregate.RecordError):
        aggregate.check_record(record)


def test_sample_record_passes_check_record() -> None:
    sample = Path(__file__).resolve().parents[3] / "data" / "schema" / "samples" / "sweep.v1.json"
    aggregate.check_record(json.loads(sample.read_text()))


# The pipeline --------------------------------------------------------------------------


def _camera(cam_id: str) -> registry.Camera:
    return registry.Camera(cam_id, cam_id, 51.5, -0.1, f"https://example.invalid/{cam_id}.jpg")


class _FakeDetector(detect.Detector):
    """Finds `people[i]` persons in the i-th frame it sees; raises for "bad" frames."""

    def __init__(self, people: Sequence[int | Exception]) -> None:
        self._people = list(people)
        self.model_name = "fake"
        self.seen = 0

    def detect(self, frame: np.ndarray) -> list[detect.Detection]:
        outcome = self._people[self.seen]
        self.seen += 1
        if isinstance(outcome, Exception):
            raise outcome
        box = (111.5, 222.5, 333.5, 444.5)
        found = [detect.Detection("person", 0.8765, box) for _ in range(outcome)]
        return [*found, detect.Detection("umbrella", 0.8765, box)] if outcome == 3 else found


def _results(cameras: Sequence[registry.Camera], errors: dict[str, str]) -> list[fetch.FrameResult]:
    return [
        fetch.FrameResult(c.id, None, 0.1, errors[c.id])  # type: ignore[arg-type]
        if c.id in errors
        else fetch.FrameResult(c.id, FRAME.copy(), 0.1, None)
        for c in cameras
    ]


def _sweep(
    cameras: Sequence[registry.Camera],
    detector: detect.Detector,
    *,
    errors: dict[str, str] | None = None,
    conditions: weather.Conditions | None = None,
    now: datetime = T0,
    elapsed: float = 42.0,
) -> dict[str, Any]:
    clock = iter([100.0, 100.0 + elapsed])
    return aggregate.run_sweep(
        detector,
        DIGEST,
        model_name="yolox_m",
        list_cameras=lambda: cameras,
        fetch_frames=lambda cams: _results(cams, errors or {}),
        conditions=lambda at: conditions,
        now=lambda: now,
        monotonic=lambda: next(clock),
    )


def test_run_sweep_counts_each_camera_once() -> None:
    cams = [_camera(f"JamCams_{i}") for i in range(5)]
    detector = _FakeDetector([3, 0, ValueError("bad frame"), 1])
    record = _sweep(cams, detector, errors={"JamCams_2": "timeout"})
    VALIDATOR.validate(record)
    assert record["cameras_listed"] == 5 and record["frames_ok"] == 3
    assert record["frames_failed"] == {
        "timeout": 1,
        "http": 0,
        "decode": 0,
        "network": 0,
        "detect": 1,
    }
    assert record["per_camera"] == {
        "JamCams_0": {"persons": 3, "umbrellas": 1},
        "JamCams_4": {"persons": 1, "umbrellas": 0},
    }
    assert record["started_at"] == "2026-07-15T12:30:05Z"
    assert record["finished_at"] == "2026-07-15T12:30:47Z"


def test_run_sweep_records_no_detection_detail(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    record = _sweep([_camera("A"), _camera("B")], _FakeDetector([3, 2]))
    standard = set(vars(logging.makeLogRecord({})))
    extras = [{k: v for k, v in vars(r).items() if k not in standard} for r in caplog.records]
    text = json.dumps(record) + caplog.text + repr(extras)
    for detail in ("box", "score", "0.8765", "111.5", "444.5", "example.invalid"):
        assert detail not in text


def test_run_sweep_with_a_broken_model_marks_every_frame() -> None:
    cams = [_camera(c) for c in "ABC"]
    broken = detect.DetectorError("inference failed: Fail")
    record = _sweep(cams, _FakeDetector([broken] * 3))
    assert record["frames_ok"] == 0 and record["frames_failed"]["detect"] == 3
    assert record["per_camera"] == {} and record["persons_total"] == 0


def test_run_sweep_with_zero_cameras() -> None:
    record = _sweep([], _FakeDetector([]))
    VALIDATOR.validate(record)
    assert record["cameras_listed"] == 0 and record["frames_ok"] == 0


def test_run_sweep_with_every_frame_failing() -> None:
    cams = [_camera(c) for c in "ABCD"]
    errors = dict(zip("ABCD", ("timeout", "http", "decode", "network"), strict=True))
    record = _sweep(cams, _FakeDetector([]), errors=errors)
    VALIDATOR.validate(record)
    assert record["frames_ok"] == 0
    assert record["frames_failed"] == {
        "timeout": 1,
        "http": 1,
        "decode": 1,
        "network": 1,
        "detect": 0,
    }


def test_run_sweep_skips_duplicate_and_unpublishable_camera_ids(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cams = [_camera("A"), _camera("A"), _camera("bad id"), _camera("B")]
    record = _sweep(cams, _FakeDetector([1, 1]))
    assert record["cameras_listed"] == 2
    assert set(record["per_camera"]) == {"A", "B"}
    assert any(getattr(r, "skipped", None) == 2 for r in caplog.records)


def test_run_sweep_fetches_a_duplicated_camera_once_from_its_first_listing() -> None:
    first = registry.Camera("A", "A", 51.5, -0.1, "https://example.invalid/first.jpg")
    again = registry.Camera("A", "A again", 51.6, -0.2, "https://example.invalid/again.jpg")
    fetched: list[registry.Camera] = []

    def fetch_frames(cams: Sequence[registry.Camera]) -> list[fetch.FrameResult]:
        fetched.extend(cams)
        return _results(cams, {})

    clock = iter([100.0, 101.0])
    record = aggregate.run_sweep(
        _FakeDetector([2, 1]),
        DIGEST,
        model_name="yolox_m",
        list_cameras=lambda: [first, _camera("B"), again],
        fetch_frames=fetch_frames,
        conditions=lambda at: None,
        now=lambda: T0,
        monotonic=lambda: next(clock),
    )
    assert fetched == [first, _camera("B")]
    assert record["cameras_listed"] == 2 and record["frames_ok"] == 2
    assert record["per_camera"] == {
        "A": {"persons": 2, "umbrellas": 0},
        "B": {"persons": 1, "umbrellas": 0},
    }


def test_run_sweep_refuses_an_absurd_registry() -> None:
    cams = [_camera(f"C{i}") for i in range(aggregate.MAX_CAMERAS + 1)]
    with pytest.raises(aggregate.SweepError):
        _sweep(cams, _FakeDetector([]))


def test_run_sweep_turns_a_registry_failure_into_a_sweep_error() -> None:
    def down() -> list[registry.Camera]:
        raise registry.RegistryError("JamCam registry unavailable after 3 attempts (URLError)")

    with pytest.raises(aggregate.SweepError, match="registry unavailable"):
        aggregate.run_sweep(
            _FakeDetector([]), DIGEST, model_name="m", list_cameras=down, conditions=lambda at: None
        )


def test_run_sweep_stops_on_a_weather_config_error_before_the_cameras() -> None:
    listed: list[bool] = []

    def refuse(at: datetime) -> weather.Conditions | None:
        raise weather.WeatherConfigError("Open-Meteo must never be used in production")

    def list_cameras() -> list[registry.Camera]:
        listed.append(True)
        return []

    with pytest.raises(weather.WeatherConfigError):
        aggregate.run_sweep(
            _FakeDetector([]), DIGEST, model_name="m", list_cameras=list_cameras, conditions=refuse
        )
    assert listed == []


def test_run_sweep_refuses_fetch_results_for_other_cameras() -> None:
    with pytest.raises(aggregate.SweepError):
        aggregate.run_sweep(
            _FakeDetector([0]),
            DIGEST,
            model_name="m",
            list_cameras=lambda: [_camera("A")],
            fetch_frames=lambda cams: _results([_camera("B")], {}),
            conditions=lambda at: None,
        )


def test_run_sweep_records_the_weather_for_the_start_minute() -> None:
    seen: list[datetime] = []
    observed = datetime(2026, 7, 15, 13, tzinfo=UTC)

    def conditions(at: datetime) -> weather.Conditions:
        seen.append(at)
        return weather.Conditions(20.0, 19.5, 0.0, observed, "metoffice")

    record = aggregate.run_sweep(
        _FakeDetector([]),
        DIGEST,
        model_name="m",
        list_cameras=lambda: [],
        conditions=conditions,
        now=lambda: T0,
    )
    assert seen == [T0.replace(microsecond=0)]
    assert record["weather"]["observed_at"] == "2026-07-15T13:00:00Z"


@pytest.mark.parametrize(
    ("now", "sweep_id", "finished"),
    [
        # Starts before midnight UTC, finishes after it: filed under the start date.
        (datetime(2026, 7, 15, 23, 59, 58, tzinfo=UTC), "20260715T2359Z", "2026-07-16T00:00:40Z"),
        # The clocks go forward at 01:00 UTC on 29 March 2026: 00:59:30 GMT is 00:59:30 UTC.
        (
            datetime(2026, 3, 29, 0, 59, 30, tzinfo=ZoneInfo("Europe/London")),
            "20260329T0059Z",
            "2026-03-29T01:00:12Z",
        ),
        # They go back at 01:00 UTC on 25 October 2026: 01:30 London happens twice.
        (
            datetime(2026, 10, 25, 1, 30, fold=1, tzinfo=ZoneInfo("Europe/London")),
            "20261025T0130Z",
            "2026-10-25T01:30:42Z",
        ),
        (
            datetime(2026, 10, 25, 1, 30, fold=0, tzinfo=ZoneInfo("Europe/London")),
            "20261025T0030Z",
            "2026-10-25T00:30:42Z",
        ),
    ],
)
def test_run_sweep_times_are_utc_around_midnight_and_clock_changes(
    now: datetime, sweep_id: str, finished: str
) -> None:
    record = _sweep([], _FakeDetector([]), now=now)
    assert record["sweep_id"] == sweep_id
    assert record["finished_at"] == finished
    VALIDATOR.validate(record)


def test_run_sweep_finish_time_never_precedes_the_start() -> None:
    record = _sweep([], _FakeDetector([]), elapsed=-5.0)  # a monotonic clock gone wrong
    assert record["finished_at"] == record["started_at"]


def test_detect_counts_releases_frames_as_it_goes() -> None:
    results = _results([_camera("A"), _camera("B")], {})
    detector = _FakeDetector([1, 0])
    observations = aggregate.detect_counts(detector, results)
    assert results == []  # consumed: no frame is held after it has been seen
    assert observations == [_obs("A", None, 1, 0), _obs("B", None, 0, 0)]


def test_records_are_deterministic() -> None:
    first = _valid()
    assert first == _valid() and first is not _valid()
    assert copy.deepcopy(first) == first
