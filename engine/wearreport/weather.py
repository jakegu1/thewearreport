"""Weather at sweep time for central London.

Production uses the Met Office Weather DataHub site-specific hourly spot forecast
(https://datahub.metoffice.gov.uk/docs/f/category/site-specific/type/site-specific/api-documentation).
Open-Meteo is non-commercial, so it is available only in development and tests, and
only when `WEARREPORT_DEV_WEATHER=openmeteo` (AGENTS.md INV-2).

`current_conditions` never raises on a provider failure: it logs the error type and
returns None, so the sweep is still recorded and left out of temperature statistics.
It never logs the API key, which travels only in a request header.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol
from urllib.parse import urlencode

from wearreport.settings import Environment, Settings, load_settings

logger = logging.getLogger(__name__)

Source = Literal["metoffice", "openmeteo"]

# Attribution statement the DataHub terms require on published data (INV-5).
METOFFICE_ATTRIBUTION = "Powered by Met Office data"

LATITUDE = 51.51
LONGITUDE = -0.13
TIMEOUT_S = 10.0
ATTEMPTS_PER_CALL = 2  # one retry
PROCESS_REQUEST_CAP = 10
MAX_STEP_DISTANCE = timedelta(minutes=30)
MAX_BODY_BYTES = 1_000_000

METOFFICE_URL = "https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point/hourly"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# (url, headers, timeout in seconds) -> response body. Raises on any transport error.
HttpGet = Callable[[str, Mapping[str, str], float], bytes]


@dataclass(frozen=True, slots=True)
class Conditions:
    temp_c: float
    apparent_c: float
    precip_mm: float
    observed_at: datetime  # the forecast time step used, in UTC
    source: Source


class WeatherError(Exception):
    """A provider response could not be used. Messages never contain credentials."""


class WeatherConfigError(RuntimeError):
    """The weather configuration is not allowed in this environment."""


class Provider(Protocol):
    @property
    def source(self) -> Source: ...

    def fetch(self, at: datetime, budget: RequestBudget) -> Conditions | None: ...


class RequestBudget:
    """Counts HTTP requests against a fixed cap. One instance per process."""

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        return self._used

    def try_acquire(self) -> bool:
        with self._lock:
            if self._used >= self._cap:
                return False
            self._used += 1
            return True


_PROCESS_BUDGET = RequestBudget(PROCESS_REQUEST_CAP)


def urllib_get(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
    """GET `url` over HTTPS with the standard library; raise on any failure."""
    if not url.startswith("https://"):
        raise ValueError("only https URLs are fetched")
    request = urllib.request.Request(url, headers=dict(headers), method="GET")  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body: bytes = response.read(MAX_BODY_BYTES + 1)
    if len(body) > MAX_BODY_BYTES:
        raise WeatherError("response body too large")
    return body


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, OSError | http.client.HTTPException)


def _log_failure(source: Source, exc: BaseException, attempt: int) -> None:
    # Only the type and status: exception text could echo request details.
    logger.warning(
        "weather request failed",
        extra={
            "source": source,
            "error_type": type(exc).__name__,
            "status": exc.code if isinstance(exc, urllib.error.HTTPError) else None,
            "attempt": attempt,
        },
    )


def _request(
    source: Source,
    url: str,
    headers: Mapping[str, str],
    http_get: HttpGet,
    budget: RequestBudget,
) -> bytes | None:
    """Up to ATTEMPTS_PER_CALL requests, each counted against `budget`."""
    for attempt in range(1, ATTEMPTS_PER_CALL + 1):
        if not budget.try_acquire():
            logger.warning(
                "weather request cap reached",
                extra={"source": source, "cap": PROCESS_REQUEST_CAP},
            )
            return None
        try:
            return http_get(url, headers, TIMEOUT_S)
        except Exception as exc:
            _log_failure(source, exc, attempt)
            if not _retryable(exc):
                return None
    return None


def _number(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WeatherError(f"missing or non-numeric field: {name}")
    if not math.isfinite(value):
        raise WeatherError(f"non-finite field: {name}")
    return float(value)


def _mapping(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise WeatherError(f"expected an object: {what}")
    return value


def _sequence(value: object, what: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise WeatherError(f"expected an array: {what}")
    return value


def _utc_time(raw: object, what: str) -> datetime:
    if not isinstance(raw, str):
        raise WeatherError(f"missing or non-string time: {what}")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        raise WeatherError(f"unparseable time: {what}") from None
    # Open-Meteo returns naive times in the requested zone, which is UTC.
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _nearest(times: Sequence[datetime], at: datetime) -> int | None:
    """Index of the time step nearest `at` (earliest on a tie), if within 30 minutes."""
    best: int | None = None
    for i, t in enumerate(times):
        if best is None or abs(t - at) < abs(times[best] - at):
            best = i
    if best is None or abs(times[best] - at) > MAX_STEP_DISTANCE:
        logger.warning("no weather time step within 30 minutes", extra={"steps": len(times)})
        return None
    return best


def _decode(body: bytes) -> object:
    try:
        return json.loads(body)
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
        raise WeatherError(f"invalid JSON ({type(exc).__name__})") from None


def parse_metoffice(body: bytes, at: datetime) -> Conditions | None:
    """Pick the DataHub hourly time step nearest `at` from a FeatureCollection."""
    doc = _mapping(_decode(body), "response")
    features = _sequence(doc.get("features"), "features")
    if not features:
        raise WeatherError("no features")
    properties = _mapping(_mapping(features[0], "feature").get("properties"), "properties")
    series = [
        _mapping(row, "timeSeries[]")
        for row in _sequence(properties.get("timeSeries"), "timeSeries")
    ]
    times = [_utc_time(row.get("time"), "timeSeries[].time") for row in series]
    index = _nearest(times, at)
    if index is None:
        return None
    row = series[index]
    return Conditions(
        temp_c=_number(row, "screenTemperature"),
        apparent_c=_number(row, "feelsLikeTemperature"),
        precip_mm=_number(row, "totalPrecipAmount"),
        observed_at=times[index],
        source="metoffice",
    )


def parse_openmeteo(body: bytes, at: datetime) -> Conditions | None:
    """Pick the Open-Meteo hourly time step nearest `at` from UTC hourly arrays."""
    doc = _mapping(_decode(body), "response")
    if doc.get("utc_offset_seconds") != 0:
        raise WeatherError("response not in UTC")
    hourly = _mapping(doc.get("hourly"), "hourly")
    raw_times = _sequence(hourly.get("time"), "hourly.time")
    times = [_utc_time(t, "hourly.time[]") for t in raw_times]
    index = _nearest(times, at)
    if index is None:
        return None
    row: dict[str, Any] = {}
    for name in ("temperature_2m", "apparent_temperature", "precipitation"):
        column = _sequence(hourly.get(name), f"hourly.{name}")
        if len(column) != len(times):
            raise WeatherError(f"length mismatch: hourly.{name}")
        row[name] = column[index]
    return Conditions(
        temp_c=_number(row, "temperature_2m"),
        apparent_c=_number(row, "apparent_temperature"),
        precip_mm=_number(row, "precipitation"),
        observed_at=times[index],
        source="openmeteo",
    )


@dataclass(frozen=True, slots=True)
class MetOfficeProvider:
    # Excluded from repr so the key never reaches logs by accident.
    api_key: str = field(repr=False)
    http_get: HttpGet = field(default=urllib_get, repr=False)
    source: Source = field(default="metoffice", init=False)

    def fetch(self, at: datetime, budget: RequestBudget) -> Conditions | None:
        url = f"{METOFFICE_URL}?{urlencode({'latitude': LATITUDE, 'longitude': LONGITUDE})}"
        headers = {"apikey": self.api_key, "accept": "application/json"}
        body = _request(self.source, url, headers, self.http_get, budget)
        return None if body is None else parse_metoffice(body, at)


@dataclass(frozen=True, slots=True)
class OpenMeteoProvider:
    """Development and tests only: Open-Meteo's free API is non-commercial."""

    env: Environment
    http_get: HttpGet = field(default=urllib_get, repr=False)
    source: Source = field(default="openmeteo", init=False)

    def __post_init__(self) -> None:
        if self.env is Environment.PRODUCTION:
            raise WeatherConfigError("Open-Meteo must never be used in production")

    def fetch(self, at: datetime, budget: RequestBudget) -> Conditions | None:
        query = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "temperature_2m,apparent_temperature,precipitation",
            "timezone": "UTC",
            "past_days": 1,
            "forecast_days": 2,
        }
        url = f"{OPENMETEO_URL}?{urlencode(query)}"
        body = _request(self.source, url, {"accept": "application/json"}, self.http_get, budget)
        return None if body is None else parse_openmeteo(body, at)


def select_provider(cfg: Settings, http_get: HttpGet = urllib_get) -> Provider | None:
    """The configured provider, or None (with a warning) when none is usable.

    Raises WeatherConfigError when the dev flag is set in production.
    """
    if cfg.dev_weather == "openmeteo":
        return OpenMeteoProvider(cfg.env, http_get)
    if cfg.metoffice_api_key is None:
        logger.warning(
            "no weather provider: METOFFICE_API_KEY is not set",
            extra={"env": cfg.env.value},
        )
        return None
    return MetOfficeProvider(cfg.metoffice_api_key, http_get)


def current_conditions(
    at: datetime,
    *,
    settings: Settings | None = None,
    http_get: HttpGet = urllib_get,
    budget: RequestBudget | None = None,
) -> Conditions | None:
    """Conditions in central London at the forecast time step nearest `at`.

    Returns None when no provider is configured, when no step is within 30 minutes,
    when the request cap is spent, or on any provider error. Raises ValueError for a
    naive `at`, and WeatherConfigError for the dev flag in production.
    """
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("at must be timezone-aware")
    provider = select_provider(settings if settings is not None else load_settings(), http_get)
    if provider is None:
        return None
    try:
        return provider.fetch(at.astimezone(UTC), budget if budget is not None else _PROCESS_BUDGET)
    except WeatherError as exc:
        logger.warning(
            "weather response unusable",
            extra={"source": provider.source, "error_type": type(exc).__name__, "detail": str(exc)},
        )
        return None
