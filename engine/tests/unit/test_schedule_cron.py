"""The sweep workflow's cron against the daytime gate (wearreport.schedule.is_open): over a
year with both DST changes, every London daytime slot has a firing the gate opens, and
the gate opens no firing outside London daytime. The cron is read from the workflow."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from wearreport import schedule

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "sweep.yml"
LONDON = ZoneInfo("Europe/London")
YEAR = 2027  # BST from 28 March to 31 October
SLOT_MINUTES = (7, 27, 47)  # the London slots: 07:07, 07:27, ..., 20:47


def expand_field(field: str, low: int, high: int) -> list[int]:
    """The values one cron field matches: `*`, `N`, `A-B`, each with an optional `/STEP`,
    comma-separated. Anything else raises ValueError."""
    values: set[int] = set()
    for part in field.split(","):
        match = re.fullmatch(r"(\*|(\d+)(?:-(\d+))?)(?:/(\d+))?", part)
        if match is None:
            raise ValueError(f"unsupported cron field {field!r}")
        whole, first, last, step = match.groups()
        if whole == "*":
            start, end = low, high
        else:
            start = int(first)
            end = int(last) if last is not None else (high if step is not None else start)
        stride = int(step) if step is not None else 1
        if not low <= start <= end <= high or stride < 1:
            raise ValueError(f"cron field {field!r} out of range {low}-{high}")
        values.update(range(start, end + 1, stride))
    return sorted(values)


def workflow_cron() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    crons = [m.group(1) for m in re.finditer(r'^\s*- cron: "([^"]*)"\s*$', text, re.M)]
    assert len(crons) == 1, crons
    return crons[0]


def firings(cron: str, year: int) -> list[datetime]:
    """Every UTC time `cron` fires in `year` (GitHub cron is UTC). Only daily schedules
    (day of month, month and day of week all `*`) are supported."""
    minute, hour, *days = cron.split()
    assert days == ["*", "*", "*"], f"not a daily schedule: {cron!r}"
    minutes, hours = expand_field(minute, 0, 59), expand_field(hour, 0, 23)
    out, day = [], date(year, 1, 1)
    while day.year == year:
        out += [datetime.combine(day, time(h, m), tzinfo=UTC) for h in hours for m in minutes]
        day += timedelta(days=1)
    return out


def test_expand_field() -> None:
    assert expand_field("7-59/20", 0, 59) == [7, 27, 47]
    assert expand_field("6-20", 0, 23) == list(range(6, 21))
    assert expand_field("*/20", 0, 59) == [0, 20, 40]
    assert expand_field("5,7-8", 0, 59) == [5, 7, 8]
    assert expand_field("30/15", 0, 59) == [30, 45]


@pytest.mark.parametrize("field", ["", "a", "7-", "-7", "7-59/0", "60", "20-10", "1,,2", "*/x"])
def test_expand_field_rejects_malformed(field: str) -> None:
    with pytest.raises(ValueError):
        expand_field(field, 0, 59)


def test_every_london_daytime_slot_has_a_firing_the_gate_opens() -> None:
    fired = set(firings(workflow_cron(), YEAR))
    day, slots = date(YEAR, 1, 1), 0
    while day.year == YEAR:
        for hour in range(7, 21):
            for minute in SLOT_MINUTES:
                local = datetime.combine(day, time(hour, minute), tzinfo=LONDON)
                utc = local.astimezone(UTC)
                assert utc in fired, f"no firing for {local.isoformat()}"
                assert schedule.is_open(utc), f"the gate is closed at {local.isoformat()}"
                slots += 1
        day += timedelta(days=1)
    print(f"london slots checked: {slots} over 365 days, each with a firing the gate opens")
    assert slots == 365 * 42


def test_the_gate_opens_no_firing_outside_london_daytime() -> None:
    all_firings = firings(workflow_cron(), YEAR)
    opened = 0
    for firing in all_firings:
        local = firing.astimezone(LONDON).time()
        daytime = time(7) <= local < time(21)
        assert schedule.is_open(firing) == daytime, firing.isoformat()
        opened += daytime
    print(f"cron firings checked: {len(all_firings)}, opened by the gate: {opened}")
    assert opened == 365 * 42


def test_year_covers_both_dst_changes() -> None:
    offsets = {
        datetime(YEAR, m, 1, 12, tzinfo=UTC).astimezone(LONDON).utcoffset() for m in (1, 7, 12)
    }
    assert offsets == {timedelta(0), timedelta(hours=1)}
    assert datetime(YEAR, 12, 1, tzinfo=LONDON).utcoffset() == timedelta(0)
