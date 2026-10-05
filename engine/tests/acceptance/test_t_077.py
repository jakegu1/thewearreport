"""Acceptance tests for T-077: the labelling launcher's helper (`label_window city`)
answers "label London now" only when London is in daylight and its local time is in the
daily window 09:30 <= t < 15:00 (Europe/London: BST or GMT as applicable). Calgary is
unchanged. The task contract: do not edit.

Fixed clocks only; nothing reaches the network and no image is opened.
"""

from __future__ import annotations

import datetime
import json
from zoneinfo import ZoneInfo

import pytest

from wearreport.tools import label_window, pilot_heights, spotcheck

UTC = datetime.UTC
LONDON_TZ = ZoneInfo("Europe/London")
MINUTE = datetime.timedelta(minutes=1)
WHERE = {"calgary": spotcheck.CALGARY, "london": spotcheck.LONDON}


def _at(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text).replace(tzinfo=UTC)


def _local(text: str) -> datetime.datetime:
    """A London wall-clock time, as an aware UTC moment."""
    return datetime.datetime.fromisoformat(text).replace(tzinfo=LONDON_TZ).astimezone(UTC)


def _lit(moment: datetime.datetime, city: str) -> bool:
    """Daylight as spotcheck decides it: anything but its "dark"."""
    return spotcheck.light_at(moment, WHERE[city]) != "dark"


def _london_window(moment: datetime.datetime) -> bool:
    local = moment.astimezone(LONDON_TZ)
    minutes = local.hour * 60 + local.minute
    return 9 * 60 + 30 <= minutes < 15 * 60


def _startable(moment: datetime.datetime, city: str) -> bool:
    if city == "london" and not _london_window(moment):
        return False
    return _lit(moment, city)


def _expected_next(
    now: datetime.datetime, cities: tuple[str, ...]
) -> tuple[datetime.datetime, str]:
    """The first whole UTC minute at or after `now` when one of `cities` may start (Calgary
    first when both): daylight, and for London its window."""
    start = now.replace(second=0, microsecond=0)
    if start < now:
        start += MINUTE
    for i in range(48 * 60 + 1):
        moment = start + i * MINUTE
        for city in cities:
            if _startable(moment, city):
                return moment, city
    raise AssertionError("no window within 48 hours")


def _city(capsys: pytest.CaptureFixture[str], *args: str) -> dict[str, object]:
    assert label_window.main(["city", *args]) == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, dict)
    return data


# AC1: the window itself, on a BST date, a GMT date and both sides of the clock change ----

# (London wall clock, inside the window)
EDGES = [
    ("09:29:00", False),
    ("09:29:59", False),
    ("09:30:00", True),
    ("14:59:00", True),
    ("14:59:59", True),
    ("15:00:00", False),
]


@pytest.mark.parametrize("date", ["2026-10-05", "2026-11-05"])
@pytest.mark.parametrize(("clock", "inside"), EDGES)
def test_ac1_window_edges_in_london_local_time(date: str, clock: str, inside: bool) -> None:
    assert label_window.in_window("london", _local(f"{date}T{clock}")) is inside


@pytest.mark.parametrize(
    ("utc", "inside"),
    [
        # 2026-10-05 is BST (UTC+1): 09:30-15:00 London is 08:30-14:00 UTC.
        ("2026-10-05T08:29", False),
        ("2026-10-05T08:30", True),
        ("2026-10-05T13:59", True),
        ("2026-10-05T14:00", False),
        # 2026-11-05 is GMT (UTC+0): 09:30-15:00 London is 09:30-15:00 UTC.
        ("2026-11-05T09:29", False),
        ("2026-11-05T09:30", True),
        ("2026-11-05T14:59", True),
        ("2026-11-05T15:00", False),
    ],
)
def test_ac1_window_in_utc_on_bst_and_gmt_dates(utc: str, inside: bool) -> None:
    assert label_window.in_window("london", _at(utc)) is inside


@pytest.mark.parametrize(
    ("utc", "inside"),
    [
        # Saturday 2026-10-24, still BST.
        ("2026-10-24T08:29", False),
        ("2026-10-24T08:30", True),
        ("2026-10-24T13:59", True),
        ("2026-10-24T14:00", False),
        ("2026-10-24T14:30", False),  # 15:30 BST
        # Sunday 2026-10-25, GMT from 01:00 UTC.
        ("2026-10-25T08:30", False),  # 08:30 GMT
        ("2026-10-25T09:29", False),
        ("2026-10-25T09:30", True),
        ("2026-10-25T14:30", True),  # 14:30 GMT
        ("2026-10-25T14:59", True),
        ("2026-10-25T15:00", False),
    ],
)
def test_ac1_both_sides_of_the_october_clock_change(utc: str, inside: bool) -> None:
    assert label_window.in_window("london", _at(utc)) is inside


def test_ac1_a_moment_in_another_zone_is_judged_by_london_time() -> None:
    calgary = ZoneInfo("America/Edmonton")
    # 02:30 MDT on 2026-10-05 is 09:30 BST; 07:59 MDT is 14:59 BST; 08:00 MDT is 15:00.
    assert label_window.in_window("london", datetime.datetime(2026, 10, 5, 2, 30, tzinfo=calgary))
    assert label_window.in_window("london", datetime.datetime(2026, 10, 5, 7, 59, tzinfo=calgary))
    assert not label_window.in_window(
        "london", datetime.datetime(2026, 10, 5, 8, 0, tzinfo=calgary)
    )


# AC2: `city --source london` ---------------------------------------------------------------


def test_ac2_after_the_window_in_daylight_gives_tomorrows_opening(
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = _at("2026-10-04T16:47")  # 17:47 BST: still light, past 15:00
    assert _lit(now, "london")
    assert _city(capsys, "--source", "london", "--now", "2026-10-04T16:47Z") == {
        "city": None,
        "next_window": "2026-10-05T08:30Z",
        "next_city": "london",
    }


@pytest.mark.parametrize(
    "now", ["2026-10-05T08:30Z", "2026-10-05T11:00Z", "2026-10-05T13:59Z", "2026-11-05T09:30Z"]
)
def test_ac2_in_the_window_and_daylight_labels_london(
    capsys: pytest.CaptureFixture[str], now: str
) -> None:
    assert _lit(_at(now[:-1]), "london")
    assert _city(capsys, "--source", "london", "--now", now) == {"city": "london"}


@pytest.mark.parametrize("now", ["2026-10-05T08:29Z", "2026-10-05T14:00Z", "2026-11-05T15:00Z"])
def test_ac2_daylight_outside_the_window_is_refused(
    capsys: pytest.CaptureFixture[str], now: str
) -> None:
    moment = _at(now[:-1])
    assert _lit(moment, "london")
    out = _city(capsys, "--source", "london", "--now", now)
    expected, city = _expected_next(moment, ("london",))
    assert out == {
        "city": None,
        "next_window": expected.strftime("%Y-%m-%dT%H:%MZ"),
        "next_city": city,
    }


def test_ac2_the_window_without_daylight_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot_heights, "solar_elevation", lambda *_: -30.0)
    now = _at("2026-10-05T11:00")
    assert label_window.in_window("london", now)
    assert label_window.choose("london", now).city is None


# AC3: --source auto never picks London outside the window; Calgary unchanged ------------


def test_ac3_auto_never_picks_london_outside_its_window() -> None:
    start = _at("2026-10-23T00:00")
    for i in range(0, 4 * 24 * 60, 11):  # every 11 minutes over the clock change
        moment = start + i * MINUTE
        choice = label_window.choose("auto", moment)
        if choice.city == "london":
            assert _london_window(moment) and _lit(moment, "london"), moment
        if choice.city is None:
            assert choice.next_window is not None
            assert choice.next_city != "london" or _london_window(choice.next_window)


def test_ac3_auto_picks_london_in_its_window_when_calgary_is_dark() -> None:
    now = _at("2026-06-21T09:00")  # 03:00 in Calgary, 10:00 in London
    assert not _lit(now, "calgary")
    assert label_window.choose("auto", now).city == "london"


def test_ac3_auto_in_london_daylight_before_the_window_waits() -> None:
    now = _at("2026-06-21T07:00")  # 01:00 in Calgary, 08:00 BST in London: light, too early
    assert _lit(now, "london") and not _lit(now, "calgary")
    choice = label_window.choose("auto", now)
    assert choice.city is None
    assert (choice.next_window, choice.next_city) == _expected_next(now, ("calgary", "london"))
    assert choice.next_window == _at("2026-06-21T08:30")


def test_ac3_calgary_answers_are_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    # The three Calgary clocks of the existing T-067 tests.
    calgary_day = _at("2026-06-21T19:00")  # 13:00 in Calgary, 20:00 in London
    calgary_dark = _at("2026-06-21T09:00")  # 03:00 in Calgary, 10:00 in London
    assert label_window.choose("auto", calgary_day) == label_window.Choice("calgary")
    assert label_window.choose("calgary", calgary_day) == label_window.Choice("calgary")
    choice = label_window.choose("calgary", calgary_dark)
    expected, city = _expected_next(calgary_dark, ("calgary",))
    assert (choice.city, choice.next_window, choice.next_city) == (None, expected, "calgary")
    assert city == "calgary"
    assert _city(capsys, "--now", "2026-06-21T13:00-06:00") == {"city": "calgary"}


# AC4: next_window is the first whole UTC minute when daylight and the window both hold --


# Clocks with no city startable (Calgary dark too, for auto), and the cities asked about.
NOT_NOW = [
    ("2026-12-21T04:00:00", "auto"),  # dark in both: London's dawn comes before its window
    ("2026-12-21T04:00:59", "auto"),
    ("2026-10-24T05:00:30", "auto"),  # 23:00 MDT in Calgary, 06:00 BST in London
    ("2026-10-25T06:00:00", "auto"),  # the morning after the clock change
    ("2026-12-21T04:00:00", "london"),
    ("2026-10-04T16:47:01", "london"),
    ("2026-10-24T14:00:30", "london"),  # BST afternoon before the clock change
    ("2026-10-25T00:30:00", "london"),  # the night of the clock change
]


@pytest.mark.parametrize(("now", "source"), NOT_NOW)
def test_ac4_next_window_is_the_first_minute_both_hold(now: str, source: str) -> None:
    moment = _at(now)
    cities = ("calgary", "london") if source == "auto" else ("london",)
    choice = label_window.choose(source, moment)
    assert choice.city is None
    expected, city = _expected_next(moment, cities)
    assert (choice.next_window, choice.next_city) == (expected, city)
    assert expected >= moment
    assert expected.second == 0 and expected.microsecond == 0
    assert label_window.choose(source, moment) == choice  # deterministic


def test_ac4_dark_both_waits_for_londons_window_not_its_dawn() -> None:
    now = _at("2026-12-21T04:00")
    choice = label_window.choose("london", now)
    assert choice.next_window == _at("2026-12-21T09:30")  # GMT: 09:30 London
    assert _lit(_at("2026-12-21T09:29"), "london")  # light already, but before the window


def test_ac4_a_late_dawn_inside_the_window_is_found_to_the_minute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opens = _at("2026-10-05T10:37")  # 11:37 BST

    def elevation(moment: datetime.datetime, latitude: float, longitude: float) -> float:
        lit = (latitude, longitude) == label_window.CITIES["london"] and moment >= opens
        return 10.0 if lit else -20.0

    monkeypatch.setattr(pilot_heights, "solar_elevation", elevation)
    choice = label_window.choose("auto", _at("2026-10-05T07:00") + datetime.timedelta(seconds=5))
    assert (choice.city, choice.next_window, choice.next_city) == (None, opens, "london")


def test_ac4_daylight_only_after_the_window_opens_next_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opens = _at("2026-10-05T14:37")  # 15:37 BST: after the window closes

    def elevation(moment: datetime.datetime, latitude: float, longitude: float) -> float:
        lit = (latitude, longitude) == label_window.CITIES["london"] and moment >= opens
        return 10.0 if lit else -20.0

    monkeypatch.setattr(pilot_heights, "solar_elevation", elevation)
    choice = label_window.choose("london", _at("2026-10-05T12:00"))
    assert (choice.city, choice.next_window, choice.next_city) == (
        None,
        _at("2026-10-06T08:30"),
        "london",
    )


def test_ac4_no_window_within_the_search_gives_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot_heights, "solar_elevation", lambda *_: -30.0)
    assert label_window.choose("london", _at("2026-10-05T11:00")) == label_window.Choice(None)
