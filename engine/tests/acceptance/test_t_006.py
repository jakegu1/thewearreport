"""Acceptance tests for T-006 (weather adapter). These are the task contract: do not edit.

Every response is built in code by the helpers below, following the documented schemas
(Met Office Weather DataHub site-specific hourly GeoJSON; Open-Meteo `/v1/forecast`
hourly arrays). No network calls: the HTTP function is injected.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tomllib
import urllib.error
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from wearreport import settings, weather

AT = datetime(2026, 6, 15, 12, 10, tzinfo=UTC)
SECRET = "not-a-real-metoffice-key"  # noqa: S105 - synthetic test value


# Synthetic response builders ---------------------------------------------------------


def _mo_time(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def metoffice_step(t: datetime, temp: float, feels: float, precip: float) -> dict[str, Any]:
    """One `timeSeries` entry of the DataHub hourly spot forecast."""
    return {
        "time": _mo_time(t),
        "screenTemperature": temp,
        "maxScreenAirTemp": temp + 0.5,
        "minScreenAirTemp": temp - 0.5,
        "screenDewPointTemperature": temp - 6.0,
        "feelsLikeTemperature": feels,
        "windSpeed10m": 3.1,
        "windDirectionFrom10m": 250,
        "windGustSpeed10m": 6.2,
        "max10mWindGust": 7.0,
        "visibility": 20000,
        "screenRelativeHumidity": 70.0,
        "mslp": 101500,
        "uvIndex": 3,
        "significantWeatherCode": 7,
        "precipitationRate": precip,
        "totalPrecipAmount": precip,
        "totalSnowAmount": 0,
        "probOfPrecipitation": 10,
    }


def metoffice_body(steps: list[dict[str, Any]]) -> bytes:
    """A DataHub `SpotForecastFeatureCollection` for central London."""
    doc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-0.13, 51.51, 25]},
                "properties": {
                    "requestPointDistance": 120.5,
                    "modelRunDate": "2026-06-15T11:00Z",
                    "timeSeries": steps,
                },
            }
        ],
        "parameters": [
            {
                "screenTemperature": {
                    "type": "Parameter",
                    "description": "Screen Air Temperature",
                    "unit": {"label": "degrees Celsius", "symbol": {"type": "Cel"}},
                }
            }
        ],
    }
    return json.dumps(doc).encode()


def hourly_steps(start: datetime, n: int = 4) -> list[dict[str, Any]]:
    return [
        metoffice_step(start + timedelta(hours=i), 15.0 + i, 13.0 + i, 0.1 * i) for i in range(n)
    ]


def openmeteo_body(start: datetime, temps: list[float]) -> bytes:
    """An Open-Meteo `/v1/forecast` response with UTC hourly arrays."""
    times = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(len(temps))]
    doc = {
        "latitude": 51.5,
        "longitude": -0.13,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "apparent_temperature": "°C",
            "precipitation": "mm",
        },
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "apparent_temperature": [t - 2.0 for t in temps],
            "precipitation": [0.0 for _ in temps],
        },
    }
    return json.dumps(doc).encode()


class FakeHttp:
    """Records every request and replays scripted outcomes (bytes or an exception)."""

    def __init__(self, *outcomes: bytes | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> bytes:
        self.calls.append((url, dict(headers), timeout))
        outcome = self.outcomes.pop(0) if self.outcomes else OSError("no scripted outcome")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/", code, "error", Message(), None)


def mo_settings(key: str | None = SECRET, env: str = "development") -> settings.Settings:
    environ = {"WEARREPORT_ENV": env}
    if key is not None:
        environ["METOFFICE_API_KEY"] = key
    return settings.load_settings(environ)


def dev_settings(env: str = "development") -> settings.Settings:
    return settings.load_settings({"WEARREPORT_ENV": env, "WEARREPORT_DEV_WEATHER": "openmeteo"})


@pytest.fixture(autouse=True)
def fresh_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a fresh per-process request counter."""
    monkeypatch.setattr(weather, "_PROCESS_BUDGET", weather.RequestBudget(10))


# AC1: interface ---------------------------------------------------------------------


def test_ac1_naive_datetime_raises_value_error() -> None:
    http = FakeHttp()
    with pytest.raises(ValueError):
        weather.current_conditions(
            datetime(2026, 6, 15, 12, 0), settings=mo_settings(), http_get=http
        )
    assert http.calls == []


def test_ac1_conditions_is_frozen_dataclass_with_fields() -> None:
    names = [f.name for f in dataclasses.fields(weather.Conditions)]
    assert names == ["temp_c", "apparent_c", "precip_mm", "observed_at", "source"]
    c = weather.Conditions(1.0, 0.0, 0.0, AT, "metoffice")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.temp_c = 2.0  # type: ignore[misc]


# AC2: Met Office provider -----------------------------------------------------------


def test_ac2_metoffice_request_and_nearest_step() -> None:
    start = datetime(2026, 6, 15, 11, 0, tzinfo=UTC)
    http = FakeHttp(metoffice_body(hourly_steps(start)))
    result = weather.current_conditions(AT, settings=mo_settings(), http_get=http)
    assert result == weather.Conditions(
        temp_c=16.0,
        apparent_c=14.0,
        precip_mm=0.1,
        observed_at=datetime(2026, 6, 15, 12, 0, tzinfo=UTC),
        source="metoffice",
    )
    assert len(http.calls) == 1
    url, headers, timeout = http.calls[0]
    parts = urlsplit(url)
    assert (parts.scheme, parts.netloc) == ("https", "data.hub.api.metoffice.gov.uk")
    assert parts.path == "/sitespecific/v0/point/hourly"
    query = parse_qs(parts.query)
    assert float(query["latitude"][0]) == 51.51
    assert float(query["longitude"][0]) == -0.13
    assert headers["apikey"] == SECRET
    assert SECRET not in url
    assert timeout == 10


def test_ac2_metoffice_accepts_non_utc_aware_datetime() -> None:
    start = datetime(2026, 6, 15, 11, 0, tzinfo=UTC)
    http = FakeHttp(metoffice_body(hourly_steps(start)))
    bst = AT.astimezone(timezone(timedelta(hours=1)))
    result = weather.current_conditions(bst, settings=mo_settings(), http_get=http)
    assert result is not None
    assert result.observed_at == datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def test_ac2_metoffice_no_step_within_30_minutes_is_none() -> None:
    far = datetime(2026, 6, 15, 14, 0, tzinfo=UTC)  # nearest step is 1h50m away
    http = FakeHttp(metoffice_body(hourly_steps(far)))
    assert weather.current_conditions(AT, settings=mo_settings(), http_get=http) is None


def test_ac2_metoffice_step_exactly_30_minutes_away_is_used() -> None:
    at = datetime(2026, 6, 15, 12, 30, tzinfo=UTC)
    step = metoffice_step(datetime(2026, 6, 15, 13, 0, tzinfo=UTC), 20.0, 19.0, 0.0)
    http = FakeHttp(metoffice_body([step]))
    result = weather.current_conditions(at, settings=mo_settings(), http_get=http)
    assert result is not None and result.temp_c == 20.0


def test_ac2_settings_expose_dev_weather_flag() -> None:
    assert settings.load_settings({}).dev_weather is None
    assert dev_settings().dev_weather == "openmeteo"


# AC3: Open-Meteo dev provider -------------------------------------------------------


def test_ac3_openmeteo_used_only_with_flag() -> None:
    start = datetime(2026, 6, 15, 10, 0, tzinfo=UTC)
    http = FakeHttp(openmeteo_body(start, [10.0, 11.0, 12.0, 13.0]))
    result = weather.current_conditions(AT, settings=dev_settings(), http_get=http)
    assert result == weather.Conditions(
        temp_c=12.0,
        apparent_c=10.0,
        precip_mm=0.0,
        observed_at=datetime(2026, 6, 15, 12, 0, tzinfo=UTC),
        source="openmeteo",
    )
    assert urlsplit(http.calls[0][0]).netloc == "api.open-meteo.com"


def test_ac3_openmeteo_not_used_without_flag() -> None:
    http = FakeHttp(metoffice_body(hourly_steps(datetime(2026, 6, 15, 11, 0, tzinfo=UTC))))
    result = weather.current_conditions(AT, settings=mo_settings(), http_get=http)
    assert result is not None and result.source == "metoffice"
    assert all("open-meteo" not in url for url, _, _ in http.calls)


def test_ac3_openmeteo_provider_raises_in_production() -> None:
    with pytest.raises(Exception):  # noqa: B017 - any exception type is acceptable
        weather.OpenMeteoProvider(settings.Environment.PRODUCTION, FakeHttp())


def test_ac3_production_with_flag_never_calls_openmeteo() -> None:
    http = FakeHttp()
    with pytest.raises(Exception):  # noqa: B017 - misconfiguration must fail loudly
        weather.current_conditions(AT, settings=dev_settings("production"), http_get=http)
    assert http.calls == []


def test_ac3_production_without_key_is_none_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    http = FakeHttp()
    with caplog.at_level(logging.WARNING, logger="wearreport.weather"):
        result = weather.current_conditions(
            AT, settings=mo_settings(key=None, env="production"), http_get=http
        )
    assert result is None
    assert http.calls == []
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# AC4: call cap ----------------------------------------------------------------------


def test_ac4_at_most_one_retry_per_call() -> None:
    http = FakeHttp(TimeoutError(), TimeoutError(), TimeoutError())
    assert weather.current_conditions(AT, settings=mo_settings(), http_get=http) is None
    assert len(http.calls) == 2
    assert all(timeout == 10 for _, _, timeout in http.calls)


def test_ac4_retry_then_success() -> None:
    body = metoffice_body(hourly_steps(datetime(2026, 6, 15, 11, 0, tzinfo=UTC)))
    http = FakeHttp(http_error(503), body)
    result = weather.current_conditions(AT, settings=mo_settings(), http_get=http)
    assert result is not None and result.temp_c == 16.0
    assert len(http.calls) == 2


def test_ac4_process_cap_of_ten_requests(caplog: pytest.LogCaptureFixture) -> None:
    http = FakeHttp(*[TimeoutError() for _ in range(20)])
    for _ in range(5):
        assert weather.current_conditions(AT, settings=mo_settings(), http_get=http) is None
    assert len(http.calls) == 10
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="wearreport.weather"):
        assert weather.current_conditions(AT, settings=mo_settings(), http_get=http) is None
    assert len(http.calls) == 10
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_ac4_no_runtime_dependency_added() -> None:
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["dependencies"] == []
    assert weather.urllib_get.__module__ == "wearreport.weather"


# AC5: errors ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcomes",
    [
        pytest.param((http_error(401),), id="http-401"),
        pytest.param((http_error(500), http_error(500)), id="http-500"),
        pytest.param((TimeoutError(), TimeoutError()), id="timeout"),
        pytest.param((urllib.error.URLError("dns"), urllib.error.URLError("dns")), id="url"),
        pytest.param((b"{not json",), id="json"),
        pytest.param((b'{"type": "FeatureCollection", "features": []}',), id="no-features"),
        pytest.param((b"[]",), id="wrong-shape"),
    ],
)
def test_ac5_provider_errors_return_none_and_log_type(
    outcomes: tuple[bytes | BaseException, ...], caplog: pytest.LogCaptureFixture
) -> None:
    http = FakeHttp(*outcomes)
    with caplog.at_level(logging.DEBUG, logger="wearreport.weather"):
        assert weather.current_conditions(AT, settings=mo_settings(), http_get=http) is None
    assert caplog.records, "the error must be logged"


@pytest.mark.parametrize(
    "field", ["time", "screenTemperature", "feelsLikeTemperature", "totalPrecipAmount"]
)
def test_ac5_missing_field_returns_none(field: str) -> None:
    step = metoffice_step(datetime(2026, 6, 15, 12, 0, tzinfo=UTC), 16.0, 14.0, 0.0)
    del step[field]
    http = FakeHttp(metoffice_body([step]))
    assert weather.current_conditions(AT, settings=mo_settings(), http_get=http) is None


def test_ac5_key_never_in_logs_or_repr(caplog: pytest.LogCaptureFixture) -> None:
    good = metoffice_body(hourly_steps(datetime(2026, 6, 15, 11, 0, tzinfo=UTC)))
    leaky = OSError(f"connection failed for apikey={SECRET}")
    http = FakeHttp(http_error(401), leaky, leaky, b"{bad", good)
    cfg = mo_settings()
    seen: list[object] = [cfg]
    with caplog.at_level(logging.DEBUG):
        for _ in range(4):
            seen.append(weather.current_conditions(AT, settings=cfg, http_get=http))
    provider = weather.MetOfficeProvider(SECRET, http)
    seen.append(provider)
    assert seen[-2] is not None
    for obj in seen:
        assert SECRET not in repr(obj)
        assert SECRET not in str(obj)
    for record in caplog.records:
        assert SECRET not in record.getMessage()
        assert SECRET not in repr(record.__dict__)
    assert SECRET not in caplog.text
