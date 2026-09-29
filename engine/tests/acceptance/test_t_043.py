"""Acceptance tests for T-043 (person counts by box height in every sweep record). The
task contract: do not edit.

The dry-sweep test needs the model file: it skips only when the file is missing and
WEARREPORT_REQUIRE_MODEL is unset; CI sets it, so there a missing model fails.
"""

from __future__ import annotations

import copy
import json
import os
import re
import socket
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from wearreport import aggregate, cli, detect, fetch, publish, registry, weather
from wearreport.testing.fake_cameras import FakeCameraServer
from wearreport.tools import spotcheck

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "data" / "schema" / "sweep.v1.json"
SAMPLE_PATH = ROOT / "data" / "schema" / "samples" / "sweep.v1.json"
PEOPLE = ROOT / "fixtures" / "detect" / "people_street.jpg"
REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
FIELD = "persons_by_height"
ALLOWED_KEYS = {str(h) for h in range(120)} | {"120+"}
T0 = datetime(2026, 7, 15, 12, 30, 5, tzinfo=UTC)
FRAME = np.zeros((288, 352, 3), dtype=np.uint8)
DIGEST = "c" * 64

# One frame's detections: (label, height in pixels, left edge, width, score).
Found = Sequence[tuple[str, float, float, float, float]]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(_load(SCHEMA_PATH))


class _Detector(detect.Detector):
    """Returns `frames[i]` for the i-th frame it sees."""

    def __init__(self, frames: Sequence[Found]) -> None:
        self._frames = list(frames)
        self.model_name = "fake"
        self.seen = 0

    def detect(self, frame: np.ndarray) -> list[detect.Detection]:
        found = self._frames[self.seen]
        self.seen += 1
        return [
            detect.Detection(label, score, (x, 7.25, x + w, 7.25 + h))  # type: ignore[arg-type]
            for label, h, x, w, score in found
        ]


def _people(*heights: float) -> list[tuple[str, float, float, float, float]]:
    return [("person", h, 3.0, 20.0, 0.9) for h in heights]


def _camera(cam_id: str) -> registry.Camera:
    return registry.Camera(cam_id, cam_id, 51.5, -0.1, f"https://example.invalid/{cam_id}.jpg")


def _sweep(frames: Sequence[Found], errors: dict[int, str] | None = None) -> dict[str, Any]:
    """A sweep over one camera per entry of `frames`; camera i fails with errors[i]."""
    errors = errors or {}
    cameras = [_camera(f"JamCams_{i:02d}") for i in range(len(frames))]
    detector = _Detector([f for i, f in enumerate(frames) if i not in errors])

    def fetch_frames(cams: Sequence[registry.Camera]) -> list[fetch.FrameResult]:
        return [
            fetch.FrameResult(c.id, None, 0.1, errors[i])  # type: ignore[arg-type]
            if i in errors
            else fetch.FrameResult(c.id, FRAME.copy(), 0.1, None)
            for i, c in enumerate(cams)
        ]

    clock = iter([100.0, 142.0])
    return aggregate.run_sweep(
        detector,
        DIGEST,
        model_name="yolox_m",
        list_cameras=lambda: cameras,
        fetch_frames=fetch_frames,
        conditions=lambda _: None,
        now=lambda: T0,
        monotonic=lambda: next(clock),
    )


def _mixed() -> dict[str, Any]:
    return _sweep(
        [
            [*_people(5.4, 5.6, 119.4), ("umbrella", 40.0, 0.0, 30.0, 0.8)],
            [],
            _people(0.2, 119.5, 120.0, 400.0),
            [("umbrella", 12.0, 0.0, 30.0, 0.7)],
            _people(1.0),
        ],
        errors={1: "timeout"},
    )


def _without_field(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != FIELD}


# AC1: height -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("box", "height"),
    [
        ((0.0, 10.0, 5.0, 50.0), 40),
        ((0.0, 10.0, 5.0, 50.4), 40),
        ((0.0, 10.0, 5.0, 50.6), 41),
        ((0.0, 0.0, 5.0, 0.3), 0),
        ((0.0, 100.0, 5.0, 350.0), 250),
    ],
)
def test_ac1_height_is_the_rounded_box_height_in_source_pixels(
    box: tuple[float, float, float, float], height: int
) -> None:
    assert aggregate.box_height(box) == height == round(box[3] - box[1])


def test_ac1_sweep_and_spotcheck_agree_on_the_same_boxes() -> None:
    boxes = [
        (3.0, 7.25, 40.0, 7.25 + h)
        for h in (0.2, 0.5, 1.5, 2.5, 5.49, 5.5, 60.5, 119.4, 119.5, 120.0, 287.9)
    ] + [(1.5, 0.5, 9.0, 3.0), (0.0, 10.5, 1.0, 130.5), (2.0, 33.3, 4.0, 199.9)]
    persons = [detect.Detection("person", 0.9, b) for b in boxes]
    samples = [
        spotcheck.Sample(FRAME, tuple(persons[:5])),
        spotcheck.Sample(FRAME, tuple(persons[5:])),
    ]
    by_number = spotcheck.box_heights(samples)
    assert [by_number[n] for n in sorted(by_number)] == [aggregate.box_height(b) for b in boxes]


# AC2: field ------------------------------------------------------------------------


def test_ac2_sweep_record_holds_the_histogram(validator: jsonschema.Draft202012Validator) -> None:
    record = _mixed()
    validator.validate(record)
    aggregate.check_record(record)
    assert record[FIELD] == {"0": 1, "1": 1, "5": 1, "6": 1, "119": 1, "120+": 3}
    assert record["persons_total"] == 8 and record["umbrellas_total"] == 2
    assert sum(record[FIELD].values()) == record["persons_total"]


def test_ac2_keys_and_values_have_the_published_form() -> None:
    record = _sweep([_people(*[float(h) for h in range(0, 260, 3)]), _people(7.0, 7.0, 7.0)])
    histogram = record[FIELD]
    assert set(histogram) <= ALLOWED_KEYS
    assert all(k == "120+" or str(int(k)) == k for k in histogram)
    assert all(type(v) is int and v >= 1 for v in histogram.values())
    assert histogram["7"] == 3 + 0  # 7 is not a multiple of 3: only the second camera
    assert histogram["120+"] == len(range(120, 260, 3))
    assert sum(histogram.values()) == record["persons_total"]


def test_ac2_bucket_edges() -> None:
    record = _sweep([_people(118.6, 119.49, 119.5, 120.49, 121.0, 10_000.0)])
    assert record[FIELD] == {"119": 2, "120+": 4}


@pytest.mark.parametrize(
    ("frames", "errors"),
    [
        ([[], []], {}),
        ([[("umbrella", 50.0, 0.0, 10.0, 0.9)]], {}),
        ([[], []], {0: "timeout", 1: "decode"}),
        ([], {}),
    ],
    ids=["nobody", "umbrella-only", "all-failed", "no-cameras"],
)
def test_ac2_no_person_gives_an_empty_object(frames: list[Found], errors: dict[int, str]) -> None:
    record = _sweep(frames, errors)
    assert record[FIELD] == {} and record["persons_total"] == 0


def test_ac2_umbrellas_are_not_counted() -> None:
    umbrellas = [("umbrella", float(h), 0.0, 10.0, 0.9) for h in (5, 50, 500)]
    with_umbrellas = _sweep([[*_people(30.0, 90.0), *umbrellas]])
    assert with_umbrellas[FIELD] == {"30": 1, "90": 1}


# AC3: validation -------------------------------------------------------------------


def test_ac3_a_record_with_or_without_the_field_is_accepted(
    validator: jsonschema.Draft202012Validator,
) -> None:
    record = _mixed()
    aggregate.check_record(record)
    old = _without_field(record)
    aggregate.check_record(old)
    validator.validate(old)


def _rejected(record: object) -> None:
    with pytest.raises(aggregate.RecordError):
        aggregate.check_record(record)


@pytest.mark.parametrize(
    "key",
    [
        "007",
        "05",
        "-1",
        "120",
        "121",
        " 5",
        "5 ",
        "",
        "1.0",
        "+5",
        "1e2",
        "\N{ARABIC-INDIC DIGIT FIVE}",
        "120+ ",
        "x",
    ],
)
def test_ac3_a_key_outside_the_allowed_set_is_rejected(
    key: str, validator: jsonschema.Draft202012Validator
) -> None:
    record = _mixed()
    del record[FIELD]["1"]
    record[FIELD][key] = 1
    _rejected(record)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(record)


def test_ac3_a_key_that_is_not_a_string_is_rejected() -> None:
    record = _mixed()
    del record[FIELD]["1"]
    record[FIELD][1] = 1
    _rejected(record)


@pytest.mark.parametrize("value", [0, -1, 1.0, True, "1", None, [1], {"n": 1}])
def test_ac3_a_value_that_is_not_a_positive_integer_is_rejected(
    value: object, validator: jsonschema.Draft202012Validator
) -> None:
    record = _mixed()
    record[FIELD]["1"] = value
    _rejected(record)
    if value != 1.0 or value is True:  # JSON Schema counts 1.0 as an integer; check_record not
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(record)


@pytest.mark.parametrize("value", [[], [["5", 1]], None, "5:1", 8, 8.0, True])
def test_ac3_a_non_object_is_rejected(
    value: object, validator: jsonschema.Draft202012Validator
) -> None:
    record = _mixed()
    record[FIELD] = value
    _rejected(record)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(record)


@pytest.mark.parametrize("delta", [-1, 1])
def test_ac3_a_sum_other_than_persons_total_is_rejected(delta: int) -> None:
    record = _mixed()
    record[FIELD]["120+"] += delta
    if record[FIELD]["120+"] == 0:
        del record[FIELD]["120+"]
    _rejected(record)


def test_ac3_a_histogram_for_a_sweep_without_persons_is_rejected() -> None:
    record = _sweep([[]])
    record[FIELD] = {"5": 1}
    _rejected(record)


def test_ac3_hostile_input_raises_record_error_only() -> None:
    deep: object = 1
    for _ in range(200_000):
        deep = [deep]
    many = {str(i): 1 for i in range(1_000_000)}
    huge = 10**5000
    for value in (deep, {"5": deep}, many, {"5": huge}, {"5": -huge}, {"5": float("inf")}):
        record = _mixed()
        record[FIELD] = value
        _rejected(record)
    record = _mixed()
    record[FIELD]["120+"] += huge
    _rejected(record)


def test_ac3_rejections_do_not_mutate_the_record() -> None:
    record = _mixed()
    record[FIELD]["007"] = 1
    before = copy.deepcopy(record)
    _rejected(record)
    assert record == before


# AC4: schema -----------------------------------------------------------------------


def test_ac4_schema_declares_the_optional_field() -> None:
    schema = _load(SCHEMA_PATH)
    assert schema["properties"]["schema"] == {"const": "sweep.v1"}
    assert FIELD not in schema["required"]
    field = schema["properties"][FIELD]
    assert field["type"] == "object"
    assert field["additionalProperties"] is False
    (pattern,) = field["patternProperties"]
    for key in ALLOWED_KEYS:
        assert re.search(pattern, key), key
    for key in ("007", "-1", "120", "121", " 5", "", "1200+"):
        assert not re.search(pattern, key), key
    assert "check_record" in field["description"]
    assert "persons_total" in field["description"]
    assert FIELD in schema["description"]


def test_ac4_sample_carries_a_consistent_field(validator: jsonschema.Draft202012Validator) -> None:
    sample = _load(SAMPLE_PATH)
    validator.validate(sample)
    assert sample["schema"] == "sweep.v1"
    assert sample[FIELD] and sum(sample[FIELD].values()) == sample["persons_total"]
    aggregate.check_record(sample)
    validator.validate(_without_field(sample))
    aggregate.check_record(_without_field(sample))


# AC5: published data unchanged otherwise -------------------------------------------


def test_ac5_existing_fields_are_unchanged() -> None:
    record = _mixed()
    assert _without_field(record) == {
        "schema": "sweep.v1",
        "sweep_id": "20260715T1230Z",
        "started_at": "2026-07-15T12:30:05Z",
        "finished_at": "2026-07-15T12:30:47Z",
        "source": "tfl-jamcam",
        "cameras_listed": 5,
        "frames_ok": 4,
        "frames_failed": {"timeout": 1, "http": 0, "decode": 0, "network": 0, "detect": 0},
        "persons_total": 8,
        "umbrellas_total": 2,
        "per_camera": {
            "JamCams_00": {"persons": 3, "umbrellas": 1},
            "JamCams_02": {"persons": 4, "umbrellas": 0},
            "JamCams_03": {"persons": 0, "umbrellas": 1},
            "JamCams_04": {"persons": 1, "umbrellas": 0},
        },
        "weather": None,
        "engine_version": aggregate.engine_version(),
        "model": "yolox_m",
        "model_sha256": DIGEST,
        "attribution": ["Powered by TfL Open Data"],
    }


def _write(data_dir: Path, record: dict[str, Any]) -> None:
    path = publish.record_path(data_dir, record["sweep_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(publish.serialize(record))


def _series() -> list[dict[str, Any]]:
    """Sweeps every 20 minutes over a day, some failed, some with no person."""
    records = []
    for i in range(60):
        started = T0 - timedelta(minutes=20 * i)
        frames: list[Found] = [_people(*[float(10 + 13 * j) for j in range(i % 7)]), []]
        errors = {0: "timeout", 1: "http"} if i % 11 == 3 else {}
        record = _sweep(frames, errors)
        stamp = {
            "sweep_id": aggregate.sweep_id_for(started),
            "started_at": aggregate.format_utc(started),
            "finished_at": aggregate.format_utc(started + timedelta(seconds=42)),
        }
        records.append({**record, **stamp})
    return records


def test_ac5_status_is_the_same_with_and_without_the_field(tmp_path: Path) -> None:
    records = _series()
    mixed, old = tmp_path / "mixed", tmp_path / "old"
    for i, record in enumerate(records):
        aggregate.check_record(record)
        _write(mixed, record if i % 2 else _without_field(record))
        _write(old, _without_field(record))
    now = T0 + timedelta(minutes=5)
    status = publish.compute_status(mixed, now=now)
    assert status == publish.compute_status(old, now=now)
    assert status["records_invalid"] == 0 and status["sweeps_24h"] == 60
    assert status["median_persons_daytime_24h"] is not None


def test_ac5_old_records_still_load(tmp_path: Path) -> None:
    for record in _series()[:5]:
        _write(tmp_path, _without_field(record))
    loaded = [publish.load_record(p) for p in publish.record_files(tmp_path)]
    assert len(loaded) == 5 and all(FIELD not in r for r in loaded)


# AC6: privacy ----------------------------------------------------------------------


def test_ac6_histogram_holds_only_counts_per_height() -> None:
    """Moving, resizing, re-scoring or re-ordering boxes, or moving them to another
    camera, changes nothing in the histogram: it carries heights and counts only."""
    first = _sweep(
        [[("person", 40.0, 3.0, 20.0, 0.9), ("person", 150.0, 50.0, 60.0, 0.4)], _people(40.0)]
    )
    second = _sweep(
        [_people(40.0, 40.0), [("person", 150.0, 300.0, 2.0, 0.99)]],
    )
    assert first[FIELD] == second[FIELD] == {"40": 2, "120+": 1}
    text = json.dumps(first[FIELD])
    for detail in ("JamCams", "0.9", "0.4", "3.0", "50", "60", "7.25", "150"):
        assert detail not in text


# AC7: a dry sweep carries the field --------------------------------------------------


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] == "127.0.0.1"):
            raise AssertionError(f"non-local connection attempted: {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    yield


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCameraServer]:
    for var in ("METOFFICE_API_KEY", "WEARREPORT_DEV_WEATHER", "WEARREPORT_ENV"):
        monkeypatch.delenv(var, raising=False)
    with FakeCameraServer() as server:
        cams = server.cameras(6)
        server.serve_body(cams[0].id, PEOPLE.read_bytes())
        server.serve_body(cams[1].id, PEOPLE.read_bytes())
        server.serve_404(cams[2].id)
        server.serve_corrupt(cams[3].id)

        def list_cameras(app_key: str | None, **kwargs: object) -> list[registry.Camera]:
            return cams

        monkeypatch.setattr(registry, "list_cameras", list_cameras)
        yield server


def test_ac7_dry_sweep_record_carries_the_field(
    fake_registry: FakeCameraServer,
    offline: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    validator: jsonschema.Draft202012Validator,
) -> None:
    if not detect.model_path("yolox_s.onnx").is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"yolox_s.onnx is missing and {REQUIRE_MODEL} is set")
        pytest.skip("yolox_s.onnx is missing; run make setup")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["sweep", "--dry-run", "--model", "yolox_s.onnx"]) == 0
    match = re.search(r"^directory: (.+)$", capsys.readouterr().out, re.MULTILINE)
    assert match
    (path,) = Path(match.group(1)).glob("sweeps/*/*/*/*.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    validator.validate(record)
    aggregate.check_record(record)
    assert record["persons_total"] >= 2
    assert sum(record[FIELD].values()) == record["persons_total"]
    assert set(record[FIELD]) <= ALLOWED_KEYS
    assert len(path.read_bytes()) < publish.MAX_RECORD_BYTES // 100


def test_ac7_largest_possible_histogram_stays_small() -> None:
    record = _sweep([_people(*[float(h) for h in range(121)] * 1000)])
    assert set(record[FIELD]) == ALLOWED_KEYS
    extra = len(publish.serialize(record)) - len(publish.serialize(_without_field(record)))
    assert extra < 2048


def test_ac7_weather_is_still_recorded_alongside() -> None:
    conditions = weather.Conditions(9.5, 7.0, 0.0, T0.replace(minute=0, second=0), "metoffice")
    cameras = [_camera("A")]
    clock = iter([0.0, 1.0])
    record = aggregate.run_sweep(
        _Detector([_people(64.0)]),
        DIGEST,
        model_name="yolox_m",
        list_cameras=lambda: cameras,
        fetch_frames=lambda cams: [fetch.FrameResult("A", FRAME.copy(), 0.1, None)],
        conditions=lambda _: conditions,
        now=lambda: T0,
        monotonic=lambda: next(clock),
    )
    assert record[FIELD] == {"64": 1} and record["weather"]["source"] == "metoffice"
