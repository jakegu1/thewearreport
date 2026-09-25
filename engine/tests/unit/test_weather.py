"""Unit tests for wearreport.weather. Responses are synthetic and built in code."""

from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.message import Message
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from wearreport import settings, weather

AT = datetime(2026, 3, 2, 9, 40, tzinfo=UTC)
KEY = "unit-test-placeholder"


def mo_step(t: datetime, temp: float = 8.0, **overrides: Any) -> dict[str, Any]:
    """A DataHub hourly `timeSeries` entry (subset of the documented parameters)."""
    step: dict[str, Any] = {
        "time": t.strftime("%Y-%m-%dT%H:%MZ"),
        "screenTemperature": temp,
        "feelsLikeTemperature": temp - 3.0,
        "totalPrecipAmount": 0.4,
        "precipitationRate": 0.4,
        "probOfPrecipitation": 60,
    }
    step.update(overrides)
    return step


def mo_body(steps: list[dict[str, Any]]) -> bytes:
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-0.13, 51.51, 25]},
        "properties": {
            "requestPointDistance": 90.0,
            "modelRunDate": "2026-03-02T09:00Z",
            "timeSeries": steps,
        },
    }
    return json.dumps(
        {"type": "FeatureCollection", "features": [feature], "parameters": []}
    ).encode()


def om_body(times: list[str], temps: list[Any], offset: int = 0) -> bytes:
    hourly = {
        "time": times,
        "temperature_2m": temps,
        "apparent_temperature": temps,
        "precipitation": [0.0] * len(temps),
    }
    return json.dumps({"utc_offset_seconds": offset, "timezone": "GMT", "hourly": hourly}).encode()


class FakeHttp:
    def __init__(self, *outcomes: bytes | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.urls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> bytes:
        self.urls.append(url)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/", code, "error", Message(), None)


def cfg(**environ: str) -> settings.Settings:
    return settings.load_settings(environ)


def mo(http: FakeHttp, budget: weather.RequestBudget | None = None) -> weather.Conditions | None:
    return weather.current_conditions(
        AT,
        settings=cfg(METOFFICE_API_KEY=KEY),
        http_get=http,
        budget=budget or weather.RequestBudget(10),
    )


# Settings ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"), [("", None), ("  ", None), ("OpenMeteo", "openmeteo")]
)
def test_dev_weather_setting(raw: str, expected: str | None) -> None:
    assert cfg(WEARREPORT_DEV_WEATHER=raw).dev_weather == expected


def test_dev_weather_rejects_unknown_source() -> None:
    with pytest.raises(settings.SettingsError, match="openmeteo"):
        cfg(WEARREPORT_DEV_WEATHER="metoffice")


# Time step selection ----------------------------------------------------------------


def test_tie_between_two_steps_picks_the_earlier() -> None:
    at = datetime(2026, 3, 2, 9, 30, tzinfo=UTC)
    steps = [mo_step(at - timedelta(minutes=30), 1.0), mo_step(at + timedelta(minutes=30), 2.0)]
    result = weather.current_conditions(
        at, settings=cfg(METOFFICE_API_KEY=KEY), http_get=FakeHttp(mo_body(steps))
    )
    assert result is not None and result.temp_c == 1.0


def test_unordered_steps_still_pick_nearest() -> None:
    base = datetime(2026, 3, 2, 8, 0, tzinfo=UTC)
    steps = [mo_step(base + timedelta(hours=h), float(h)) for h in (3, 0, 2, 1)]
    result = mo(FakeHttp(mo_body(steps)))
    assert result is not None
    assert result.observed_at == datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    assert result.temp_c == 2.0


def test_empty_time_series_is_none() -> None:
    assert mo(FakeHttp(mo_body([]))) is None


# Malformed values -------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "8.0", True, float("nan")])
def test_non_numeric_values_are_rejected(value: Any) -> None:
    step = mo_step(datetime(2026, 3, 2, 10, 0, tzinfo=UTC), screenTemperature=value)
    assert mo(FakeHttp(mo_body([step]))) is None


def test_bad_time_string_is_rejected(caplog: pytest.LogCaptureFixture) -> None:
    step = mo_step(datetime(2026, 3, 2, 10, 0, tzinfo=UTC), time="yesterday")
    with caplog.at_level(logging.WARNING, logger="wearreport.weather"):
        assert mo(FakeHttp(mo_body([step]))) is None
    assert caplog.records[-1].__dict__["error_type"] == "WeatherError"


def test_invalid_utf8_is_rejected() -> None:
    assert mo(FakeHttp(b"\xff\xfe")) is None


# Retry policy and request cap -------------------------------------------------------


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retried(code: int, caplog: pytest.LogCaptureFixture) -> None:
    http = FakeHttp(http_error(code))
    with caplog.at_level(logging.WARNING, logger="wearreport.weather"):
        assert mo(http) is None
    assert len(http.urls) == 1
    record = caplog.records[-1]
    assert (record.__dict__["error_type"], record.__dict__["status"]) == ("HTTPError", code)


def test_rate_limit_is_retried() -> None:
    http = FakeHttp(http_error(429), mo_body([mo_step(datetime(2026, 3, 2, 10, 0, tzinfo=UTC))]))
    assert mo(http) is not None
    assert len(http.urls) == 2


def test_malformed_body_is_not_retried() -> None:
    http = FakeHttp(b"{", b"{")
    assert mo(http) is None
    assert len(http.urls) == 1


def test_retry_is_skipped_when_budget_runs_out() -> None:
    budget = weather.RequestBudget(1)
    http = FakeHttp(TimeoutError(), TimeoutError())
    assert mo(http, budget) is None
    assert len(http.urls) == 1
    assert budget.used == 1


def test_budget_never_exceeds_cap() -> None:
    budget = weather.RequestBudget(3)
    assert [budget.try_acquire() for _ in range(5)] == [True, True, True, False, False]
    assert budget.used == 3


# Provider selection -----------------------------------------------------------------


def test_development_without_key_or_flag_is_none(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="wearreport.weather"):
        assert weather.current_conditions(AT, settings=cfg(), http_get=FakeHttp()) is None
    assert "METOFFICE_API_KEY" in caplog.text


def test_dev_flag_wins_over_key_outside_production() -> None:
    provider = weather.select_provider(
        cfg(METOFFICE_API_KEY=KEY, WEARREPORT_DEV_WEATHER="openmeteo", WEARREPORT_ENV="test")
    )
    assert isinstance(provider, weather.OpenMeteoProvider)


def test_production_dev_flag_raises_config_error() -> None:
    with pytest.raises(weather.WeatherConfigError):
        weather.select_provider(
            cfg(WEARREPORT_ENV="production", WEARREPORT_DEV_WEATHER="openmeteo")
        )


def test_settings_default_to_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEARREPORT_ENV", "test")
    monkeypatch.setenv("WEARREPORT_DEV_WEATHER", "openmeteo")
    http = FakeHttp(om_body(["2026-03-02T10:00"], [6.5]))
    result = weather.current_conditions(AT, http_get=http, budget=weather.RequestBudget(10))
    assert result is not None and result.source == "openmeteo"


# Open-Meteo parsing -----------------------------------------------------------------


def om(body: bytes) -> weather.Conditions | None:
    return weather.current_conditions(
        AT,
        settings=cfg(WEARREPORT_DEV_WEATHER="openmeteo"),
        http_get=FakeHttp(body),
        budget=weather.RequestBudget(10),
    )


def test_openmeteo_request_asks_for_utc_hourly_fields() -> None:
    http = FakeHttp(om_body(["2026-03-02T10:00"], [6.5]))
    weather.current_conditions(
        AT,
        settings=cfg(WEARREPORT_DEV_WEATHER="openmeteo"),
        http_get=http,
        budget=weather.RequestBudget(10),
    )
    query = parse_qs(urlsplit(http.urls[0]).query)
    assert query["timezone"] == ["UTC"]
    assert query["hourly"] == ["temperature_2m,apparent_temperature,precipitation"]


def test_openmeteo_null_value_is_rejected() -> None:
    assert om(om_body(["2026-03-02T10:00"], [None])) is None


def test_openmeteo_length_mismatch_is_rejected() -> None:
    assert om(om_body(["2026-03-02T09:00", "2026-03-02T10:00"], [6.5])) is None


def test_openmeteo_non_utc_response_is_rejected() -> None:
    assert om(om_body(["2026-03-02T10:00"], [6.5], offset=3600)) is None


def test_openmeteo_no_step_within_30_minutes_is_none() -> None:
    assert om(om_body(["2026-03-02T11:00"], [6.5])) is None


# Default HTTP function --------------------------------------------------------------


def test_urllib_get_refuses_plain_http() -> None:
    with pytest.raises(ValueError):
        weather.urllib_get("http://example.invalid/", {}, 1.0)


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_urllib_get_sends_headers_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        seen["headers"] = dict(request.header_items())
        seen["timeout"] = timeout
        return _FakeResponse(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert weather.urllib_get("https://example.invalid/", {"apikey": "x"}, 10.0) == b"{}"
    assert seen == {"headers": {"Apikey": "x"}, "timeout": 10.0}


def test_urllib_get_rejects_oversized_body(monkeypatch: pytest.MonkeyPatch) -> None:
    big = b" " * (weather.MAX_BODY_BYTES + 10)
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _FakeResponse(big))
    with pytest.raises(weather.WeatherError):
        weather.urllib_get("https://example.invalid/", {}, 10.0)
