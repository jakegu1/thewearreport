"""Typed runtime configuration read from environment variables.

Every variable is documented in `.env.example`. An empty value counts as unset, so a
copied `.env.example` behaves like an empty environment.
"""

from __future__ import annotations

import enum
import os
from collections.abc import Mapping
from dataclasses import dataclass, field


class SettingsError(ValueError):
    """Raised when an environment variable holds an invalid value."""


class Environment(enum.Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class Settings:
    env: Environment
    # Credentials are excluded from repr so they never reach logs by accident.
    tfl_app_key: str | None = field(repr=False)
    metoffice_api_key: str | None = field(repr=False)
    nws_user_agent: str | None


def _optional(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name, "").strip()
    return value or None


def _environment(environ: Mapping[str, str]) -> Environment:
    raw = _optional(environ, "WEARREPORT_ENV")
    if raw is None:
        return Environment.DEVELOPMENT
    try:
        return Environment(raw.lower())
    except ValueError:
        allowed = ", ".join(e.value for e in Environment)
        raise SettingsError(f"WEARREPORT_ENV must be one of: {allowed}") from None


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Build settings from `environ` (defaults to the process environment)."""
    env = os.environ if environ is None else environ
    return Settings(
        env=_environment(env),
        tfl_app_key=_optional(env, "TFL_APP_KEY"),
        metoffice_api_key=_optional(env, "METOFFICE_API_KEY"),
        nws_user_agent=_optional(env, "NWS_USER_AGENT"),
    )
