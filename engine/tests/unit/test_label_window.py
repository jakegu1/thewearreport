"""Unit tests of the labelling launcher's helper, with fixed clocks and files written here."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from wearreport.tools import label_window, pilot_heights

UTC = datetime.UTC


def test_choose_prefers_calgary_when_both_are_lit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot_heights, "solar_elevation", lambda *_: 5.0)
    now = datetime.datetime(2026, 6, 21, 12, tzinfo=UTC)
    assert label_window.choose("auto", now) == label_window.Choice("calgary")


def test_the_threshold_itself_is_daylight(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.datetime(2026, 6, 21, 12, tzinfo=UTC)
    monkeypatch.setattr(pilot_heights, "solar_elevation", lambda *_: -6.0)
    assert label_window.in_daylight("london", now)
    monkeypatch.setattr(pilot_heights, "solar_elevation", lambda *_: -6.0001)
    assert not label_window.in_daylight("london", now)


def test_no_window_within_the_search_gives_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot_heights, "solar_elevation", lambda *_: -30.0)
    choice = label_window.choose("auto", datetime.datetime(2026, 6, 21, 12, tzinfo=UTC))
    assert choice == label_window.Choice(None)
    assert label_window.main(["city", "--now", "2026-06-21T12:00Z"]) == 0


def test_the_next_window_is_found_to_the_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    opens = datetime.datetime(2026, 6, 21, 14, 37, tzinfo=UTC)

    def elevation(moment: datetime.datetime, latitude: float, longitude: float) -> float:
        lit = (latitude, longitude) == label_window.CITIES["london"] and moment >= opens
        return 10.0 if lit else -20.0

    monkeypatch.setattr(pilot_heights, "solar_elevation", elevation)
    now = datetime.datetime(2026, 6, 21, 12, 0, 30, tzinfo=UTC)
    choice = label_window.choose("auto", now)
    assert (choice.city, choice.next_window, choice.next_city) == (None, opens, "london")
    assert label_window.choose("calgary", now).next_window is None


def test_now_in_another_zone_is_converted(capsys: pytest.CaptureFixture[str]) -> None:
    assert label_window.main(["city", "--now", "2026-06-21T13:00-06:00"]) == 0
    assert json.loads(capsys.readouterr().out) == {"city": "calgary"}


def test_count_of_a_missing_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert label_window.main(["count", str(tmp_path / "none.json")]) == 1
    assert "cannot read it" in capsys.readouterr().err


def test_count_of_a_file_without_crops_is_zero(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text('{"crops": []}\n', encoding="utf-8")
    counts = label_window.count([path])
    assert (counts.kept, counts.judged) == (0, 0)
