"""Unit tests for the umbrella-person association and umbrella_persons_by_height."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from wearreport import aggregate

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((ROOT / "data" / "schema" / "sweep.v1.json").read_text(encoding="utf-8"))
SAMPLE = json.loads(
    (ROOT / "data" / "schema" / "samples" / "sweep.v1.json").read_text(encoding="utf-8")
)
T0 = datetime(2026, 7, 15, 12, 30, 5, tzinfo=UTC)
PERSON = (10.0, 100.0, 30.0, 140.0)  # h = 40, top centre (20, 100)


def _umbrella(ux: float, uy: float) -> tuple[float, float, float, float]:
    return (ux - 5, uy - 5, ux + 5, uy + 5)


def _build(*observations: aggregate.Observation) -> dict[str, Any]:
    return aggregate.build_record(
        observations,
        started_at=T0,
        finished_at=T0,
        weather=None,
        engine_version="0.0.0",
        model_name="yolox_m",
        model_sha256="c" * 64,
    )


# match_umbrellas -----------------------------------------------------------------------


def test_no_boxes_no_pairs() -> None:
    assert aggregate.match_umbrellas([], []) == []
    assert aggregate.match_umbrellas([PERSON], []) == []
    assert aggregate.match_umbrellas([], [_umbrella(20, 100)]) == []


def test_pairs_come_in_matching_order() -> None:
    right = (20.0, 100.0, 40.0, 150.0)  # h = 50, top centre (30, 100)
    umbrellas = [_umbrella(25, 99), _umbrella(29, 99)]
    # (U1, right) is the nearest pair; then U0 goes to PERSON (tied with right, earlier).
    assert aggregate.match_umbrellas([PERSON, right], umbrellas) == [(1, 1), (0, 0)]


def test_tie_between_umbrellas_uses_the_earlier_one() -> None:
    umbrellas = [_umbrella(17, 100), _umbrella(23, 100)]
    assert aggregate.match_umbrellas([PERSON], umbrellas) == [(0, 0)]
    assert aggregate.match_umbrellas([PERSON], umbrellas[::-1]) == [(0, 0)]


def test_zero_height_person_window_is_its_top_edge() -> None:
    flat = (10.0, 100.0, 30.0, 100.2)
    assert aggregate.match_umbrellas([flat], [_umbrella(20, 100)]) == [(0, 0)]
    assert aggregate.match_umbrellas([flat], [_umbrella(20, 100.1)]) == []


# build_record ----------------------------------------------------------------------------


def test_build_record_orders_keys_by_height() -> None:
    record = _build(
        aggregate.Observation("A", None, 3, 3, (200, 7, 40), (200, 7, 40)),
    )
    assert list(record[aggregate.UMBRELLA_HEIGHTS]) == ["7", "40", "120+"]


@pytest.mark.parametrize(
    "observation",
    [
        aggregate.Observation("A", None, 1, 1, (40,), (41,)),
        aggregate.Observation("A", None, 1, 2, (40,), (40, 40)),
        aggregate.Observation("A", None, 2, 1, (40, 40), (40, 40)),
        aggregate.Observation("A", None, 1, 1, None, (40,)),
        aggregate.Observation("A", None, 1, 1, (40,), (-1,)),
    ],
    ids=["not-a-person-height", "more-than-persons", "more-than-umbrellas", "no-heights", "neg"],
)
def test_build_record_refuses_inconsistent_observations(
    observation: aggregate.Observation,
) -> None:
    with pytest.raises(aggregate.RecordError):
        _build(observation)


def test_umbrella_heights_need_person_heights_in_every_frame() -> None:
    with pytest.raises(aggregate.RecordError):
        _build(
            aggregate.Observation("A", None, 1, 1, (40,), (40,)),
            aggregate.Observation("B", None, 1, 1, None, ()),
        )


# check_record ----------------------------------------------------------------------------


def test_messages_do_not_echo_keys_or_values() -> None:
    record = dict(SAMPLE)
    record[aggregate.UMBRELLA_HEIGHTS] = {"secret-key": 1}
    with pytest.raises(aggregate.RecordError) as raised:
        aggregate.check_record(record)
    assert "secret-key" not in str(raised.value)
    record[aggregate.UMBRELLA_HEIGHTS] = {"57": 123456789}
    with pytest.raises(aggregate.RecordError) as raised:
        aggregate.check_record(record)
    assert "123456789" not in str(raised.value)


def test_a_record_that_is_not_a_dict_is_refused() -> None:
    with pytest.raises(aggregate.RecordError):
        aggregate.check_record([aggregate.UMBRELLA_HEIGHTS])


# Schema ----------------------------------------------------------------------------------


def test_schema_requires_persons_by_height_alongside_the_field() -> None:
    validator = jsonschema.Draft202012Validator(SCHEMA)
    validator.validate(SAMPLE)
    without = {k: v for k, v in SAMPLE.items() if k != aggregate.HEIGHTS}
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(without)


def test_schema_patterns_match_for_both_histograms() -> None:
    props = SCHEMA["properties"]
    assert (
        props[aggregate.UMBRELLA_HEIGHTS]["patternProperties"]
        == props[aggregate.HEIGHTS]["patternProperties"]
    )
    assert props[aggregate.UMBRELLA_HEIGHTS]["maxProperties"] == len(aggregate.HEIGHT_KEYS)
