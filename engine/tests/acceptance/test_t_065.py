"""Acceptance tests for T-065 (person boxes with an umbrella box, by height, in every
sweep record). The task contract: do not edit.

Every detection here is synthetic: a fake detector returns hand-written boxes for blank
in-memory frames, and nothing contacts a network host.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from wearreport import aggregate, detect, fetch, registry

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "data" / "schema" / "sweep.v1.json"
SAMPLE_PATH = ROOT / "data" / "schema" / "samples" / "sweep.v1.json"
FIELD = "umbrella_persons_by_height"
HEIGHTS = "persons_by_height"
ALLOWED_KEYS = {str(h) for h in range(120)} | {"120+"}
T0 = datetime(2026, 7, 15, 12, 30, 5, tzinfo=UTC)
FRAME = np.zeros((288, 352, 3), dtype=np.uint8)
DIGEST = "c" * 64

Box = tuple[float, float, float, float]
# One frame's detections, in the detector's output order: (label, box).
Found = Sequence[tuple[str, Box]]

# The reference person: x 10..30, y 100..140, so h = 40 and its top centre is (20, 100).
# An umbrella is a candidate for it when 10 <= ux <= 30 and 60 <= uy <= 120.
PERSON: Box = (10.0, 100.0, 30.0, 140.0)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(_load(SCHEMA_PATH))


def _umbrella(ux: float, uy: float, half: float = 5.0) -> Box:
    """An umbrella box centred on (ux, uy)."""
    return (ux - half, uy - half, ux + half, uy + half)


class _Detector(detect.Detector):
    """Returns `frames[i]` for the i-th frame it sees; raises for the frames in `broken`."""

    def __init__(self, frames: Sequence[Found], broken: Sequence[int] = ()) -> None:
        self._frames = list(frames)
        self._broken = set(broken)
        self.model_name = "fake"
        self.seen = 0

    def detect(self, frame: np.ndarray) -> list[detect.Detection]:
        index = self.seen
        self.seen += 1
        if index in self._broken:
            raise detect.DetectorError("synthetic failure")
        return [
            detect.Detection(label, 0.9, box)  # type: ignore[arg-type]
            for label, box in self._frames[index]
        ]


def _camera(cam_id: str) -> registry.Camera:
    return registry.Camera(cam_id, cam_id, 51.5, -0.1, f"https://example.invalid/{cam_id}.jpg")


def _sweep(
    frames: Sequence[Found],
    errors: dict[int, str] | None = None,
    broken: Sequence[int] = (),
) -> dict[str, Any]:
    """A sweep over one camera per entry of `frames`. Camera i fails to fetch with
    errors[i]; the detector raises on the frames (among those fetched) listed in `broken`."""
    errors = errors or {}
    cameras = [_camera(f"JamCams_{i:02d}") for i in range(len(frames))]
    detector = _Detector([f for i, f in enumerate(frames) if i not in errors], broken)

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


def _one(found: Found) -> dict[str, int]:
    """The new field for a sweep of one camera that detected `found`."""
    record = _sweep([found])
    aggregate.check_record(record)
    result: dict[str, int] = record[FIELD]
    return result


def _mixed() -> dict[str, Any]:
    """A valid record with umbrella persons under two keys."""
    return _sweep(
        [
            [
                ("person", PERSON),
                ("person", (100.0, 50.0, 120.0, 350.0)),
                ("umbrella", _umbrella(20.0, 95.0)),
                ("umbrella", _umbrella(110.0, 40.0)),
                ("umbrella", _umbrella(300.0, 10.0)),
            ],
            [("person", (5.0, 10.0, 25.0, 50.0))],
            [("person", PERSON)],
        ],
        errors={2: "timeout"},
    )


# AC1: field ------------------------------------------------------------------------


def test_ac1_the_constant_names_the_field() -> None:
    assert aggregate.UMBRELLA_HEIGHTS == FIELD
    assert aggregate.HEIGHTS == HEIGHTS


def test_ac1_a_record_with_heights_has_the_field(
    validator: jsonschema.Draft202012Validator,
) -> None:
    record = _mixed()
    validator.validate(record)
    aggregate.check_record(record)
    assert record[FIELD] == {"40": 1, "120+": 1}
    assert list(record[FIELD]) == ["40", "120+"]
    assert record[HEIGHTS] == {"40": 2, "120+": 1}


@pytest.mark.parametrize(
    ("frames", "errors"),
    [
        ([[], []], {}),
        ([[("person", PERSON)]], {}),
        ([[("umbrella", _umbrella(20.0, 95.0))]], {}),
        ([[("person", PERSON), ("umbrella", _umbrella(200.0, 95.0))]], {}),
        ([[], []], {0: "timeout", 1: "decode"}),
        ([], {}),
    ],
    ids=["nobody", "no-umbrella", "umbrella-only", "umbrella-elsewhere", "all-failed", "none"],
)
def test_ac1_no_one_qualifies_gives_an_empty_object(
    frames: list[Found], errors: dict[int, str]
) -> None:
    record = _sweep(frames, errors)
    aggregate.check_record(record)
    assert record[FIELD] == {}


def test_ac1_keys_and_values_have_the_published_form() -> None:
    found: list[tuple[str, Box]] = []
    for i, h in enumerate(range(0, 260, 7)):
        x = 40.0 * i
        found += [("person", (x, 200.0, x + 20.0, 200.0 + h)), ("umbrella", _umbrella(x + 10, 199))]
    record = _sweep([found, found])
    aggregate.check_record(record)
    field = record[FIELD]
    assert set(field) <= ALLOWED_KEYS
    assert all(k == "120+" or str(int(k)) == k for k in field)
    assert all(type(v) is int and v >= 1 for v in field.values())
    order = [str(h) for h in range(120)] + ["120+"]
    assert list(field) == sorted(field, key=order.index)
    assert field["7"] == 2
    assert field["120+"] == 2 * len(range(126, 260, 7))
    assert "0" not in field  # a 0 px person has a window of a single point: y1


def test_ac1_a_record_built_without_heights_has_neither_field() -> None:
    record = aggregate.build_record(
        [aggregate.Observation("JamCams_00", None, 2, 1)],
        started_at=T0,
        finished_at=T0,
        weather=None,
        engine_version="0.0.0",
        model_name="yolox_m",
        model_sha256=DIGEST,
    )
    assert HEIGHTS not in record and FIELD not in record
    aggregate.check_record(record)


# AC2: association rule -------------------------------------------------------------


@pytest.mark.parametrize(
    ("ux", "uy", "candidate"),
    [
        (20.0, 100.0, True),  # on the top centre
        (10.0, 90.0, True),  # ux on x1
        (30.0, 90.0, True),  # ux on x2
        (9.99, 90.0, False),  # just left of x1
        (30.01, 90.0, False),  # just right of x2
        (20.0, 60.0, True),  # uy on y1 - h
        (20.0, 120.0, True),  # uy on y1 + h/2
        (20.0, 59.99, False),  # just above y1 - h
        (20.0, 120.01, False),  # just below y1 + h/2
        (10.0, 60.0, True),  # a corner of the window
        (30.0, 120.0, True),  # the opposite corner
        (9.99, 59.99, False),
    ],
)
def test_ac2_candidate_window_edges(ux: float, uy: float, candidate: bool) -> None:
    expected = {"40": 1} if candidate else {}
    assert _one([("person", PERSON), ("umbrella", _umbrella(ux, uy))]) == expected
    assert _one([("umbrella", _umbrella(ux, uy)), ("person", PERSON)]) == expected


def test_ac2_window_uses_the_rounded_box_height() -> None:
    # The raw height is 40.4, so h = box_height = 40 and the window ends at 100 + 20 = 120,
    # not at 100 + 20.2.
    person: Box = (10.0, 100.0, 30.0, 140.4)
    assert aggregate.box_height(person) == 40
    assert _one([("person", person), ("umbrella", _umbrella(20.0, 120.0))]) == {"40": 1}
    assert _one([("person", person), ("umbrella", _umbrella(20.0, 120.1))]) == {}
    # The raw height is 39.6, so h = 40 and the window starts at 100 - 40 = 60.
    person = (10.0, 100.0, 30.0, 139.6)
    assert _one([("person", person), ("umbrella", _umbrella(20.0, 60.0))]) == {"40": 1}


def test_ac2_the_umbrella_centre_counts_not_its_extent() -> None:
    # A huge umbrella that covers the person but is centred outside the window.
    assert _one([("person", PERSON), ("umbrella", (0.0, 0.0, 300.0, 300.0))]) == {}


def test_ac2_two_people_under_one_umbrella_count_once() -> None:
    first: Box = (10.0, 100.0, 30.0, 140.0)  # top centre (20, 100), h = 40
    second: Box = (12.0, 104.0, 32.0, 154.0)  # top centre (22, 104), h = 50
    umbrella = _umbrella(20.0, 98.0)  # a candidate for both; nearer to the first
    for found in (
        [("person", first), ("person", second), ("umbrella", umbrella)],
        [("person", second), ("person", first), ("umbrella", umbrella)],
        [("umbrella", umbrella), ("person", second), ("person", first)],
    ):
        record = _sweep([found])
        assert record[FIELD] == {"40": 1}
        assert record[HEIGHTS] == {"40": 1, "50": 1}
        assert record["umbrellas_total"] == 1


def test_ac2_two_umbrellas_over_one_person_count_once() -> None:
    record = _sweep(
        [
            [
                ("umbrella", _umbrella(20.0, 95.0)),
                ("person", PERSON),
                ("umbrella", _umbrella(15.0, 90.0)),
            ]
        ]
    )
    assert record[FIELD] == {"40": 1}
    assert record["umbrellas_total"] == 2


def test_ac2_the_spare_umbrella_goes_to_another_person() -> None:
    # Two people side by side and two umbrellas, each a candidate for both.
    left: Box = (10.0, 100.0, 30.0, 140.0)  # h = 40, top centre (20, 100)
    right: Box = (20.0, 100.0, 40.0, 150.0)  # h = 50, top centre (30, 100)
    found = [
        ("person", left),
        ("person", right),
        ("umbrella", _umbrella(25.0, 99.0)),
        ("umbrella", _umbrella(29.0, 99.0)),
    ]
    assert _one(found) == {"40": 1, "50": 1}


def test_ac2_matching_is_greedy_by_distance() -> None:
    # U0 is nearer to P1 than to P0 and is a candidate for both; U1 is a candidate for
    # P1 only. Greedy matching gives U0 to P1, so P0 stays unmatched and U1 is unused,
    # although a maximum matching would pair both people.
    p0: Box = (10.0, 100.0, 30.0, 140.0)  # h = 40, top centre (20, 100)
    p1: Box = (20.0, 100.0, 40.0, 150.0)  # h = 50, top centre (30, 100)
    u0 = _umbrella(29.0, 100.0)  # distance 9 from P0, 1 from P1
    u1 = _umbrella(35.0, 100.0)  # outside P0's window (x2 = 30); distance 5 from P1
    assert _one([("person", p0), ("person", p1), ("umbrella", u0), ("umbrella", u1)]) == {"50": 1}
    assert _one([("umbrella", u1), ("umbrella", u0), ("person", p1), ("person", p0)]) == {"50": 1}


def test_ac2_ties_go_to_the_earlier_person() -> None:
    # Two people whose top centres are equally far from the one umbrella.
    p40: Box = (10.0, 100.0, 30.0, 140.0)  # h = 40, top centre (20, 100)
    p50: Box = (10.0, 104.0, 30.0, 154.0)  # h = 50, top centre (20, 104)
    umbrella = _umbrella(20.0, 102.0)  # distance 2 from both
    assert _one([("person", p40), ("person", p50), ("umbrella", umbrella)]) == {"40": 1}
    assert _one([("person", p50), ("person", p40), ("umbrella", umbrella)]) == {"50": 1}
    assert _one([("umbrella", umbrella), ("person", p50), ("person", p40)]) == {"50": 1}


def test_ac2_ties_between_umbrellas_go_to_the_earlier_umbrella() -> None:
    # One person, two umbrellas equally far from its top centre; the earlier umbrella takes
    # the person, so the later one stays free for a second person who can only use it.
    person: Box = (10.0, 100.0, 30.0, 140.0)  # h = 40, top centre (20, 100)
    other: Box = (22.0, 100.0, 42.0, 160.0)  # h = 60, top centre (32, 100)
    near_left = _umbrella(17.0, 100.0)  # distance 3 from person; outside other's window
    near_right = _umbrella(23.0, 100.0)  # distance 3 from person; distance 9 from other
    assert _one(
        [("person", person), ("person", other), ("umbrella", near_left), ("umbrella", near_right)]
    ) == {"40": 1, "60": 1}
    # With the umbrellas the other way round, the earlier one is near_right: it goes to
    # the person and the other is left with nothing.
    assert _one(
        [("person", person), ("person", other), ("umbrella", near_right), ("umbrella", near_left)]
    ) == {"40": 1}


def test_ac2_a_tall_person_counts_under_120_plus() -> None:
    tall: Box = (100.0, 50.0, 140.0, 350.0)  # h = 300: window y -250..200
    assert _one([("person", tall), ("umbrella", _umbrella(120.0, -100.0))]) == {"120+": 1}
    assert _one([("person", tall), ("umbrella", _umbrella(120.0, 200.0))]) == {"120+": 1}
    assert _one([("person", tall), ("umbrella", _umbrella(120.0, 200.5))]) == {}


def test_ac2_umbrellas_in_other_frames_are_not_associated() -> None:
    record = _sweep([[("person", PERSON)], [("umbrella", _umbrella(20.0, 95.0))]])
    assert record[FIELD] == {}


def test_ac2_failed_frames_are_never_counted(validator: jsonschema.Draft202012Validator) -> None:
    pair: Found = [("person", PERSON), ("umbrella", _umbrella(20.0, 95.0))]
    tall: Found = [("person", (0.0, 50.0, 40.0, 350.0)), ("umbrella", _umbrella(20.0, 40.0))]
    record = _sweep([pair, tall, pair], errors={0: "timeout"}, broken=[0])
    validator.validate(record)
    aggregate.check_record(record)
    assert record["frames_failed"]["timeout"] == 1 and record["frames_failed"]["detect"] == 1
    assert record[FIELD] == {"40": 1}
    assert record[HEIGHTS] == {"40": 1}


def test_ac2_failed_observations_are_never_counted() -> None:
    record = aggregate.build_record(
        [
            aggregate.Observation("JamCams_00", None, 1, 1, (40,), (40,)),
            aggregate.Observation("JamCams_01", "detect"),
            aggregate.Observation("JamCams_02", "timeout", 0, 0, None, (50,)),
        ],
        started_at=T0,
        finished_at=T0,
        weather=None,
        engine_version="0.0.0",
        model_name="yolox_m",
        model_sha256=DIGEST,
    )
    assert record[FIELD] == {"40": 1}


# AC3: consistency ------------------------------------------------------------------


def _rejected(record: object) -> None:
    with pytest.raises(aggregate.RecordError):
        aggregate.check_record(record)


def test_ac3_records_without_the_field_are_valid(
    validator: jsonschema.Draft202012Validator,
) -> None:
    record = _mixed()
    del record[FIELD]
    aggregate.check_record(record)
    validator.validate(record)
    del record[HEIGHTS]
    aggregate.check_record(record)
    validator.validate(record)


def test_ac3_the_field_without_persons_by_height_is_rejected() -> None:
    record = _mixed()
    del record[HEIGHTS]
    _rejected(record)
    record = _sweep([[]])
    del record[HEIGHTS]
    _rejected(record)


@pytest.mark.parametrize(
    "key", ["007", "05", "-1", "120", "121", " 40", "40 ", "", "40.0", "+40", "120+ ", "x"]
)
def test_ac3_a_bad_key_is_rejected(key: str, validator: jsonschema.Draft202012Validator) -> None:
    record = _mixed()
    del record[FIELD]["40"]
    record[FIELD][key] = 1
    _rejected(record)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(record)


def test_ac3_a_key_that_is_not_a_string_is_rejected() -> None:
    record = _mixed()
    del record[FIELD]["40"]
    record[FIELD][40] = 1
    _rejected(record)


@pytest.mark.parametrize("value", [0, -1, 1.0, True, "1", None, [1], {"n": 1}, 2**64])
def test_ac3_a_bad_count_is_rejected(value: object) -> None:
    record = _mixed()
    record[FIELD]["40"] = value
    _rejected(record)


@pytest.mark.parametrize("value", [[], [["40", 1]], None, "40:1", 1, 1.0, True])
def test_ac3_a_non_object_is_rejected(
    value: object, validator: jsonschema.Draft202012Validator
) -> None:
    record = _mixed()
    record[FIELD] = value
    _rejected(record)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(record)


def test_ac3_a_count_above_persons_by_height_is_rejected() -> None:
    record = _mixed()
    assert record[HEIGHTS]["120+"] == 1 and record["umbrellas_total"] == 3
    record[FIELD]["120+"] = 2
    _rejected(record)


def test_ac3_a_key_absent_from_persons_by_height_is_rejected() -> None:
    record = _mixed()
    assert "41" not in record[HEIGHTS]
    record[FIELD] = {"40": 1, "41": 1}
    _rejected(record)


def test_ac3_a_sum_above_umbrellas_total_is_rejected() -> None:
    record = _sweep(
        [
            [
                ("person", PERSON),
                ("person", (100.0, 100.0, 120.0, 140.0)),
                ("person", (200.0, 100.0, 220.0, 140.0)),
                ("umbrella", _umbrella(20.0, 95.0)),
                ("umbrella", _umbrella(110.0, 95.0)),
            ]
        ]
    )
    aggregate.check_record(record)
    assert record[FIELD] == {"40": 2} and record[HEIGHTS] == {"40": 3}
    assert record["umbrellas_total"] == 2
    record[FIELD]["40"] = 3  # within persons_by_height, above umbrellas_total
    _rejected(record)


def test_ac3_counts_up_to_both_limits_are_accepted() -> None:
    record = _mixed()
    record[FIELD] = {"40": 2, "120+": 1}  # persons_by_height {"40": 2, "120+": 1}; 3 umbrellas
    aggregate.check_record(record)


def test_ac3_hostile_input_raises_record_error_only() -> None:
    deep: object = 1
    for _ in range(200_000):
        deep = [deep]
    many = {str(i): 1 for i in range(1_000_000)}
    huge = 10**5000
    for value in (deep, {"40": deep}, many, {"40": huge}, {"40": -huge}, {"40": float("inf")}):
        record = _mixed()
        record[FIELD] = value
        _rejected(record)


def test_ac3_rejections_do_not_mutate_the_record() -> None:
    record = _mixed()
    record[FIELD]["120+"] = 5
    before = copy.deepcopy(record)
    _rejected(record)
    assert record == before


# AC4: schema -----------------------------------------------------------------------


def test_ac4_schema_declares_the_optional_property() -> None:
    schema = _load(SCHEMA_PATH)
    assert schema["properties"]["schema"] == {"const": "sweep.v1"}
    assert FIELD not in schema["required"]
    field = schema["properties"][FIELD]
    heights = schema["properties"][HEIGHTS]
    assert field["type"] == "object"
    assert field["additionalProperties"] is False
    assert field["patternProperties"].keys() == heights["patternProperties"].keys()
    (pattern,) = field["patternProperties"]
    for key in ALLOWED_KEYS:
        assert re.search(pattern, key), key
    for key in ("007", "-1", "120", "121", " 5", "", "1200+"):
        assert not re.search(pattern, key), key
    assert "check_record" in field["description"]
    assert "umbrella" in field["description"] and HEIGHTS in field["description"]
    assert FIELD in schema["description"]


def test_ac4_sample_carries_a_consistent_field(validator: jsonschema.Draft202012Validator) -> None:
    sample = _load(SAMPLE_PATH)
    validator.validate(sample)
    aggregate.check_record(sample)
    assert sample["schema"] == "sweep.v1"
    assert sample[FIELD]
    assert all(sample[FIELD][k] <= sample[HEIGHTS].get(k, 0) for k in sample[FIELD])
    assert sum(sample[FIELD].values()) <= sample["umbrellas_total"]
    without = {k: v for k, v in sample.items() if k != FIELD}
    validator.validate(without)
    aggregate.check_record(without)


def test_ac4_a_run_sweep_record_validates_against_the_schema(
    validator: jsonschema.Draft202012Validator,
) -> None:
    record = _mixed()
    validator.validate(record)
    aggregate.check_record(record)
    assert record["schema"] == "sweep.v1"


# AC5: everything else unchanged ----------------------------------------------------


def test_ac5_other_fields_are_computed_as_before() -> None:
    record = _mixed()
    assert {k: v for k, v in record.items() if k != FIELD} == {
        "schema": "sweep.v1",
        "sweep_id": "20260715T1230Z",
        "started_at": "2026-07-15T12:30:05Z",
        "finished_at": "2026-07-15T12:30:47Z",
        "source": "tfl-jamcam",
        "cameras_listed": 3,
        "frames_ok": 2,
        "frames_failed": {"timeout": 1, "http": 0, "decode": 0, "network": 0, "detect": 0},
        "persons_total": 3,
        "umbrellas_total": 3,
        HEIGHTS: {"40": 2, "120+": 1},
        "per_camera": {
            "JamCams_00": {"persons": 2, "umbrellas": 3},
            "JamCams_01": {"persons": 1, "umbrellas": 0},
        },
        "weather": None,
        "engine_version": aggregate.engine_version(),
        "model": "yolox_m",
        "model_sha256": DIGEST,
        "attribution": ["Powered by TfL Open Data"],
    }


def test_ac5_observations_without_the_new_field_compare_as_before() -> None:
    old = aggregate.Observation("JamCams_00", None, 1, 0, (40,))
    assert old == aggregate.Observation("JamCams_00", None, 1, 0, (40,), ())


def test_ac5_the_field_holds_only_counts_per_height() -> None:
    """Moving the pairs, or splitting them over cameras, changes nothing in the field: it
    carries heights and counts only."""
    first = _sweep(
        [
            [
                ("person", PERSON),
                ("umbrella", _umbrella(20.0, 95.0)),
                ("person", (200.0, 10.0, 230.0, 310.0)),
                ("umbrella", _umbrella(215.0, 0.0)),
            ]
        ]
    )
    second = _sweep(
        [
            [("person", (50.0, 120.0, 90.0, 160.0)), ("umbrella", _umbrella(70.0, 119.0))],
            [("person", (1.0, 2.0, 3.0, 202.0)), ("umbrella", _umbrella(2.0, 2.0))],
        ]
    )
    assert first[FIELD] == second[FIELD] == {"40": 1, "120+": 1}
    text = json.dumps(first[FIELD])
    for detail in ("JamCams", "0.9", "95", "215", "200", "230", "310", "20.0"):
        assert detail not in text
