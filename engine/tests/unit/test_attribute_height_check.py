"""The attribute summary's min_height_px bounds are spotcheck's --min-height range (T-057)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from wearreport.tools import spotcheck, spotcheck_summary

LOW = spotcheck.NEAR_FIELD_MIN_HEIGHT_PX
HIGH = spotcheck.MAX_ATTRIBUTE_MIN_HEIGHT_PX


def _raw(min_height: int, heights: list[int]) -> bytes:
    record: dict[str, Any] = {
        "date": "2026-10-05",
        "started_at": "2026-10-05T10:00Z",
        "light": "day",
        "frames": 1,
        "detector": {},
        "min_height_px": min_height,
        "judge": None,
        "crops_shown": len(heights),
        "crops_rejected": 0,
        "crops": [[h, "ynu", None] for h in heights],
    }
    return json.dumps(record).encode()


@pytest.mark.parametrize("value", [LOW, HIGH])
def test_the_ends_of_the_min_height_range_are_read(value: int) -> None:
    labelling = spotcheck_summary.parse_labelling(_raw(value, [value]))
    assert labelling.crops == ((value, "ynu", None),)


@pytest.mark.parametrize("value", [LOW - 1, HIGH + 1])
def test_just_outside_the_min_height_range_is_refused(value: int) -> None:
    with pytest.raises(ValueError, match="min_height_px"):
        spotcheck_summary.parse_labelling(_raw(value, [HIGH + 1]))


def test_a_crop_one_pixel_below_min_height_px_is_refused() -> None:
    with pytest.raises(ValueError, match="a crop height is below min_height_px"):
        spotcheck_summary.parse_labelling(_raw(46, [60, 45, 46]))


def test_an_empty_session_needs_no_crop_check() -> None:
    assert spotcheck_summary.parse_labelling(_raw(46, [])).crops == ()
