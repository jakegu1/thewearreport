"""Unit tests for T-053: the attribute record's threshold, the flag's parser and the
summary's height bands. Synthetic data only."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from wearreport import detect
from wearreport.tools import spotcheck, spotcheck_summary

INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
NOON = datetime.datetime(2026, 10, 12, 11, 22, 33, tzinfo=datetime.UTC)


def _item(number: int) -> spotcheck.ReviewItem:
    image = np.zeros((60, 40, 3), dtype=np.uint8)
    return spotcheck.ReviewItem(number, f"crop-{number:04d}.png", (number,), image)


def _record(**kwargs: int) -> dict[str, object]:
    return spotcheck.attribute_record(
        [_item(1)],
        {1: "ynn"},
        {1: 50},
        [],
        frames_reviewed=1,
        info=INFO,
        day=NOON.date(),
        started_at=NOON,
        judge=None,
        **kwargs,
    )


def test_the_record_defaults_to_the_near_field_threshold() -> None:
    assert _record()["min_height_px"] == spotcheck.NEAR_FIELD_MIN_HEIGHT_PX == 31


def test_the_record_carries_the_threshold_given() -> None:
    assert _record(min_height=46)["min_height_px"] == 46
    assert _record(min_height=46) == {**_record(), "min_height_px": 46}


def test_the_flag_parses_whole_numbers_within_the_bounds() -> None:
    parser = spotcheck.build_parser()
    assert parser.parse_args(["--n", "1"]).min_height is None
    for value in (31, 46, 200):
        assert parser.parse_args(["--n", "1", "--min-height", str(value)]).min_height == value
    for bad in ("30", "201", "4.5", "abc", "nan", "1e3"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--n", "1", "--min-height", bad])


def _labelling(light: str, *crops: tuple[int, str]) -> spotcheck_summary.Labelling:
    return spotcheck_summary.Labelling(
        started_at=NOON,
        light=light,
        shown=len(crops),
        rejected=0,
        crops=tuple((height, letters, None) for height, letters in crops),
    )


def _counts(lines: list[str]) -> list[tuple[int, int]]:
    found = []
    for line in lines:
        _, _, tail = line.partition(": ")
        crops, answered = tail.split(", ")[:2]
        found.append((int(crops.split()[0]), int(answered.split()[0])))
    return found


def test_band_edges_fall_in_the_right_band() -> None:
    day = _labelling(
        "day",
        (30, "yyy"),  # below every band: not counted
        (31, "yyy"),
        (35, "uyy"),
        (36, "yyy"),
        (40, "yyy"),
        (41, "yyy"),
        (45, "yyy"),
        (46, "nyy"),
        (50, "yyy"),
        (51, "yyy"),
        (60, "uyy"),
        (61, "yyy"),
        (5000, "nyy"),
    )
    lines = spotcheck_summary.band_lines([day], 0)
    assert [line.split(" px")[0] for line in lines] == [
        "outer_layer height 31-35",
        "outer_layer height 36-40",
        "outer_layer height 41-45",
        "outer_layer height 46-50",
        "outer_layer height 51-60",
        "outer_layer height 61+",
    ]
    assert _counts(lines) == [(2, 1), (2, 2), (2, 2), (2, 2), (2, 1), (2, 2)]
    assert lines[0].endswith("share 0.5000")
    assert lines[1].endswith("share 1.0000")


def test_bands_count_day_and_twilight_and_leave_dark_out() -> None:
    labellings = [
        _labelling("day", (33, "yuu")),
        _labelling("twilight", (33, "unu")),
        _labelling("dark", (33, "nnn"), (70, "yyy")),
    ]
    lines = [spotcheck_summary.band_lines(labellings, index) for index in range(3)]
    assert _counts(lines[0])[0] == (2, 1)
    assert _counts(lines[1])[0] == (2, 1)
    assert _counts(lines[2])[0] == (2, 0)
    assert lines[2][0].endswith("share 0.0000")
    assert all(counts[-1] == (0, 0) for counts in map(_counts, lines))
    assert all(line.split(", share ")[1] == "n/a" for line in (lines[0][-1], lines[1][2]))
    assert all("day and twilight (dark excluded)" in line for group in lines for line in group)
