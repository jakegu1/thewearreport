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
