from __future__ import annotations

import dataclasses

import pytest

from wearreport.settings import Environment, Settings, SettingsError, load_settings


@pytest.mark.parametrize("raw", ["production", "PRODUCTION", " production "])
def test_environment_is_case_and_whitespace_insensitive(raw: str) -> None:
    assert load_settings({"WEARREPORT_ENV": raw}).env is Environment.PRODUCTION


def test_whitespace_only_values_count_as_unset() -> None:
    loaded = load_settings({"TFL_APP_KEY": "   ", "WEARREPORT_ENV": ""})
    assert loaded.tfl_app_key is None
    assert loaded.env is Environment.DEVELOPMENT


def test_error_lists_allowed_environments() -> None:
    with pytest.raises(SettingsError, match="development, test, production"):
        load_settings({"WEARREPORT_ENV": "prod"})


def test_settings_are_immutable() -> None:
    loaded = load_settings({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.tfl_app_key = "x"  # type: ignore[misc]


def test_repr_keeps_non_secret_fields() -> None:
    loaded = load_settings({"NWS_USER_AGENT": "wearreport (ops@example.com)"})
    assert isinstance(loaded, Settings)
    assert "wearreport (ops@example.com)" in repr(loaded)
