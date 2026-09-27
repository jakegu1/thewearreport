"""Acceptance tests for T-038 (sweep schedule off the top of the hour: a plain UTC cron,
with the gate job alone keeping runs to London daytime). The task contract: do not edit.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from wearreport import schedule

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "sweep.yml"
README = ROOT / "README.md"
LONDON = ZoneInfo("Europe/London")
CRON = "7-59/20 6-20 * * *"
# A year with both DST changes (28 March and 31 October 2027).
YEAR = 2027


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _header(text: str) -> list[str]:
    lines = text.splitlines()
    end = lines.index("on:")
    return [line for line in lines[:end] if line.startswith("#")]


def _operations(readme: str) -> str:
    match = re.search(r"^## Operations\n(.*?)(?=^## )", readme, re.MULTILINE | re.DOTALL)
    assert match, "README has no § Operations"
    return match.group(1)


def _firings(year: int) -> list[datetime]:
    """Every firing of CRON in `year`, expanded by hand (not by the code under test)."""
    day, out = date(year, 1, 1), []
    while day.year == year:
        for hour in range(6, 21):
            for minute in (7, 27, 47):
                out.append(datetime.combine(day, time(hour, minute), tzinfo=UTC))
        day += timedelta(days=1)
    return out


# AC1: schedule -------------------------------------------------------------------------


def test_ac1_one_utc_cron_entry_and_no_timezone_key(text: str) -> None:
    assert re.findall(r"^\s*- cron: .*$", text, re.MULTILINE) == [f'    - cron: "{CRON}"']
    assert "timezone:" not in text
    assert f'on:\n  schedule:\n    - cron: "{CRON}"\n  workflow_dispatch:\n' in text


def test_ac1_rest_of_the_workflow_is_byte_for_byte_unchanged(text: str) -> None:
    assert text.startswith("name: sweep\n\n")
    # Between the name and `on:` only the header comment and blank lines.
    before = text[: text.index("\non:\n")].splitlines()[1:]
    assert all(line == "" or line.startswith("#") for line in before)


# AC2: coverage -------------------------------------------------------------------------


def test_ac2_every_london_daytime_slot_has_a_firing_the_gate_opens() -> None:
    firings = set(_firings(YEAR))
    day = date(YEAR, 1, 1)
    days = 0
    while day.year == YEAR:
        for hour in range(7, 21):
            for minute in (7, 27, 47):
                local = datetime.combine(day, time(hour, minute), tzinfo=LONDON)
                utc = local.astimezone(UTC)
                assert utc in firings, f"no firing for {local.isoformat()}"
                assert schedule.is_open(utc), f"gate closed at {local.isoformat()}"
        days += 1
        day += timedelta(days=1)
    assert days == 365


def test_ac2_the_gate_opens_no_firing_outside_london_daytime() -> None:
    firings = _firings(YEAR)
    opened = [f for f in firings if schedule.is_open(f)]
    for firing in firings:
        local = firing.astimezone(LONDON).time()
        assert schedule.is_open(firing) == (time(7) <= local < time(21)), firing
    assert len(firings) == 365 * 15 * 3
    assert len(opened) == 365 * 42


def test_ac2_unit_test_reads_the_cron_from_the_workflow() -> None:
    units = sorted((ROOT / "engine" / "tests" / "unit").glob("test_schedule*.py"))
    sources = [p.read_text(encoding="utf-8") for p in units]
    assert any("sweep.yml" in s and "is_open" in s and "cron" in s for s in sources)


# AC3: docs -----------------------------------------------------------------------------


def test_ac3_readme_operations_describes_the_utc_schedule() -> None:
    ops = _operations(README.read_text(encoding="utf-8"))
    assert f"`{CRON}`" in ops
    assert "UTC" in ops
    assert "7, 27" in ops and "47" in ops
    assert "06" in ops and "20" in ops
    assert "07:00" in ops and "21:00" in ops
    assert "42" in ops
    assert "top of the hour" in ops.lower() or "top of every hour" in ops.lower()
    assert "timezone" not in ops


def test_ac3_workflow_header_says_it_in_one_line(text: str) -> None:
    lines = [
        line
        for line in _header(text)
        if "UTC" in line and "7, 27, 47" in line and "07:00-21:00" in line
    ]
    assert len(lines) == 1
    assert "timezone" not in "\n".join(_header(text))
