"""Additional offline input and timestamp regression checks."""

from __future__ import annotations

import datetime as dt
import sys
from zoneinfo import ZoneInfo

import pytest

from wearreport.tools import local_label_plan as planner


def test_historical_offsets_preserve_the_actual_instant() -> None:
    data = planner.build_plan(
        ZoneInfo("Europe/Amsterdam"), dt.date(1900, 1, 10), 1, 0, 1440, planner.WEEKDAYS
    )
    for name in ("london_opportunities", "calgary_noon_coverage"):
        rows = data[name]
        assert isinstance(rows, list) and rows
        for row in rows:
            source = dt.datetime.fromisoformat(row["source_local"])
            display = dt.datetime.fromisoformat(row["display_local"])
            utc = dt.datetime.fromisoformat(row["utc"])
            assert source == display == utc
            assert display.strftime("%H:%M:%S") == row["display_time"]


def test_unknown_large_option_is_bounded(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        planner.main(["--zone", "UTC", "--" + "x" * 10000])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert len(output.err) < 2000 and "unknown options" in output.err


def test_preflight_rejects_an_unsupported_python_without_reading_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 11))
    with pytest.raises(planner.PlanError, match=r"Python 3\.12"):
        planner.preflight(ZoneInfo("UTC"))
