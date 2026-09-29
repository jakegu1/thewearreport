"""Unit tests for the weekly summary of the spot-check statistics (T-036), and for the one
privacy-guard relaxation it needs: its own module name and the statistics directory."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import pytest

import privacy_guard
from wearreport.tools import spotcheck_summary as summary

SUMMARY = "engine/wearreport/tools/spotcheck_summary.py"
OTHER = "engine/wearreport/tools/other.py"
ANSWERS = ("person", "in_vehicle", "not_person", "unsure")


def _messages(source: str, filename: str) -> list[str]:
    return [f.message for f in privacy_guard.scan_source(source, filename)]


# The privacy guard ----------------------------------------------------------------------


def test_the_summary_may_name_the_statistics_directory_and_itself() -> None:
    source = (
        'DEFAULT_DIR = "spotchecks"\n'
        'PROG = "python -m wearreport.tools.spotcheck_summary"\n'
        'print("Spotchecks", DEFAULT_DIR, PROG)\n'
    )
    assert _messages(source, SUMMARY) == []


@pytest.mark.parametrize(
    "source",
    [
        'x = "spotchecks"\n',
        'x = "wearreport.tools.spotcheck_summary"\n',
    ],
)
def test_another_module_still_may_not(source: str) -> None:
    assert _messages(source, OTHER)


@pytest.mark.parametrize(
    "source",
    [
        "from wearreport.tools import spotcheck\n",
        "import wearreport.tools.spotcheck\n",
        'import importlib\nimportlib.import_module("wearreport.tools.spotcheck")\n',
        'import runpy\nrunpy.run_path("engine/wearreport/tools/spotcheck.py")\n',
        'x = "SpotCheck"\n',
        'x = "spotchecks/../spotcheck"\n',
        'x = b"wearreport.tools.spotcheck"\n',
        "import importlib\nimportlib.import_module(name)\n",
    ],
)
def test_the_summary_still_may_not_use_the_exempt_module(source: str) -> None:
    assert _messages(source, SUMMARY)


def test_the_relaxation_names_one_file() -> None:
    assert list(privacy_guard.STEM_WORDS_ALLOWED) == [SUMMARY]
    assert (Path(privacy_guard.__file__).parents[1] / SUMMARY).is_file()


# The summary ----------------------------------------------------------------------------


def _table(person: int, not_person_says_person: int, not_person: int) -> dict[str, Any]:
    """A confusion where the judge agrees with the reviewer except on `not_person_says_person`
    boxes the reviewer marked not a person."""
    rows = {
        "person": (person, 0, 0, 0),
        "in_vehicle": (0, 0, 0, 0),
        "not_person": (not_person_says_person, 0, not_person, 0),
    }
    return {label: dict(zip(ANSWERS, counts, strict=True)) for label, counts in rows.items()}


def _write(directory: Path, name: str, day: str, model: str | None = None) -> None:
    record: dict[str, Any] = {"date": day, "boxes_shown": 10, "boxes_not_person": 1}
    if model is not None:
        record["judge"] = {"model": model, "confusion": _table(9, 0, 1)}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(record), encoding="utf-8")


def test_weeks_follow_the_iso_calendar_across_a_year_end() -> None:
    records = [
        summary.Record(datetime.date(2027, 1, 4), 10, 1, None, None),
        summary.Record(datetime.date(2026, 12, 31), 10, 2, None, None),
        summary.Record(datetime.date(2027, 1, 3), 10, 3, None, None),  # ISO 2026-W53
    ]
    rows = summary.weeks(records, None)
    assert [row.label for row in rows] == ["2026-W53", "2027-W01"]
    assert rows[0].reviewer == pytest.approx(15 / 20) and rows[0].shown == 20
    assert rows[1].reviewer == pytest.approx(0.9)
    assert all(row.judge is None and row.corrected is None for row in rows)
    assert not summary.condition_holds(rows)


def test_each_judge_model_is_summarised_on_its_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "a.json", "2026-09-15", "di-qwen3-vl-235b")
    _write(tmp_path, "b.json", "2026-09-16", "di-gemma-4-31b")
    _write(tmp_path, "c.json", "2026-09-22", "di-gemma-4-31b")
    assert summary.main(["--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    sections = out.split("Judge ")[1:]
    assert [s.split(",", 1)[0] for s in sections] == ["di-gemma-4-31b", "di-qwen3-vl-235b"]
    gemma, qwen = sections
    # gemma has an earlier week for W39; qwen has no W39 data and no earlier week in W38.
    assert "corrected 0.9000" in gemma.splitlines()[2]
    assert "corrected n/a" in qwen.splitlines()[1] and "judge n/a" in qwen.splitlines()[2]


def test_an_empty_directory_prints_no_week(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert summary.main(["--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "0 statistics file(s)" in out
    assert "two weeks within 3 points: no" in out


def test_a_missing_directory_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert summary.main(["--dir", str(tmp_path / "nowhere")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_a_file_that_is_too_large_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "big.json").write_bytes(b" " * (summary.MAX_FILE_BYTES + 1) + b"{}")
    assert summary.main(["--dir", str(tmp_path)]) == 1
    assert "big.json" in capsys.readouterr().err


def test_the_judge_precision_leaves_unsure_answers_out() -> None:
    table = _table(6, 2, 2)
    table["person"]["unsure"] = 5
    assert summary.judge_precision(table) == pytest.approx(8 / 10)
    assert summary.judge_precision(summary.empty()) is None


def test_a_perfect_judge_needs_no_correction() -> None:
    earlier = _table(90, 0, 10)
    current = _table(40, 0, 10)
    assert summary.corrected_estimate(current, earlier) == pytest.approx(0.8)


# The height summary (T-041) -------------------------------------------------------------

UTC = datetime.UTC


def _box_file(directory: Path, name: str, started_at: str, boxes: list[list[Any]]) -> None:
    record = {
        "date": started_at[:10],
        "started_at": started_at,
        "light": "day",
        "frames": 1,
        "detector": {"model": "stub"},
        "boxes": boxes,
    }
    (directory / "boxes").mkdir(parents=True, exist_ok=True)
    (directory / "boxes" / name).write_text(json.dumps(record), encoding="utf-8")


def _record(data: Path, name: str, started_at: str, precip: float) -> None:
    folder = data / "sweeps" / started_at[:4] / started_at[5:7] / started_at[8:10]
    folder.mkdir(parents=True, exist_ok=True)
    record = {"started_at": started_at, "weather": {"precip_mm": precip}}
    (folder / name).write_text(json.dumps(record), encoding="utf-8")


def test_wilson_stays_within_zero_and_one() -> None:
    assert summary.wilson(0, 5)[0] == 0.0 and 0 < summary.wilson(0, 5)[1] < 1
    assert summary.wilson(5, 5)[1] == 1.0 and 0 < summary.wilson(5, 5)[0] < 1
    lo, hi = summary.wilson(1, 2)
    assert lo == pytest.approx(1 - hi)


def test_the_rain_join_takes_the_earlier_record_on_a_tie(tmp_path: Path) -> None:
    moment = datetime.datetime(2026, 9, 20, 10, 0, tzinfo=UTC)
    _record(tmp_path, "20260920T0950Z.json", "2026-09-20T09:50:00Z", 0.0)
    _record(tmp_path, "20260920T1010Z.json", "2026-09-20T10:10:00Z", 2.0)
    assert summary.rain_condition(tmp_path, moment) == "dry"


def test_the_rain_join_reads_only_records_near_the_session(tmp_path: Path) -> None:
    moment = datetime.datetime(2026, 9, 20, 10, 0, tzinfo=UTC)
    folder = tmp_path / "sweeps" / "2026" / "09" / "20"
    folder.mkdir(parents=True)
    (folder / "20260920T1200Z.json").write_bytes(b"not json")  # far off: never read
    (folder / "20261320T1000Z.json").write_bytes(b"not json")  # not a sweep id
    (folder / "notes.json").write_bytes(b"not json")
    assert summary.rain_condition(tmp_path, moment) == "unknown"


@pytest.mark.parametrize(
    "weather",
    [{"precip_mm": True}, {"precip_mm": -0.1}, {"precip_mm": 1e999}, {}, [], "wet"],
)
def test_a_sweep_record_with_a_bad_precipitation_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], weather: Any
) -> None:
    d, data = tmp_path / "s", tmp_path / "data"
    _box_file(d, "a.json", "2026-09-20T10:00Z", [[40, "person"]])
    folder = data / "sweeps" / "2026" / "09" / "20"
    folder.mkdir(parents=True)
    record = {"started_at": "2026-09-20T10:05:00Z", "weather": weather}
    (folder / "20260920T1005Z.json").write_text(json.dumps(record), encoding="utf-8")
    assert summary.main(["--heights", "--dir", str(d), "--data-dir", str(data)]) == 1
    assert "20260920T1005Z.json" in capsys.readouterr().err


def test_heights_with_a_missing_directory_or_data_dir_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert summary.main(["--heights", "--dir", str(tmp_path / "none")]) == 1
    assert "boxes" in capsys.readouterr().err
    _box_file(tmp_path, "a.json", "2026-09-20T10:00Z", [[40, "person"]])
    args = ["--heights", "--dir", str(tmp_path), "--data-dir", str(tmp_path / "missing")]
    assert summary.main(args) == 1
    assert "missing" in capsys.readouterr().err


def test_a_height_with_only_unsure_boxes_is_a_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _box_file(tmp_path, "a.json", "2026-09-20T10:00Z", [[40, "person"], [90, "unsure"]])
    assert summary.main(["--heights", "--dir", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert ">= 90 px: n=0 precision n/a wilson n/a kept 0.0000" in lines
    assert "coverage: 1 session, 1 date, 1 judged boxes" in lines[-1]
    assert "judged share >= 90 px: 0 of 1 (0.0000)" in lines


def test_the_judged_share_is_undefined_without_boxes() -> None:
    assert summary.tally([]).judged_share is None
    assert summary.tally([(40, "unsure")], 50).judged_share is None
    assert summary.tally([(40, "person"), (60, "unsure")]).judged_share == 0.5


def test_a_tally_without_boxes_or_below_the_share_does_not_qualify() -> None:
    assert not summary.qualifies(summary.Tally(0, 0, 0))
    assert summary.qualifies(summary.Tally(100, 100, 25))
    assert not summary.qualifies(summary.Tally(100, 100, 26))
