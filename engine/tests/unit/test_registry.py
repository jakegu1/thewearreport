from __future__ import annotations

import http.client
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from wearreport import registry

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "jamcam_places.json"


def _place(place_id: str = "cam", **overrides: object) -> dict[str, Any]:
    place: dict[str, Any] = {
        "id": place_id,
        "commonName": f"Camera {place_id}",
        "lat": 51.5,
        "lon": -0.1,
        "additionalProperties": [
            {"key": "available", "value": "true"},
            {"key": "imageUrl", "value": f"https://example.test/{place_id}"},
        ],
    }
    for key, value in overrides.items():
        if value is None:
            place.pop(key, None)
        else:
            place[key] = value
    return place


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _never_sleep(seconds: float) -> None:
    raise AssertionError("unexpected sleep")


# Parsing -------------------------------------------------------------------------------


def test_fixture_counts() -> None:
    result = registry.parse_places(json.loads(FIXTURE.read_text()))
    assert len(result.cameras) == 8
    assert result.skipped_unavailable == 1
    assert result.skipped_malformed == 1


def test_valid_place_becomes_camera() -> None:
    (cam,) = registry.parse_places([_place("a", lat=51, lon=0)]).cameras
    assert cam == registry.Camera(
        id="a", name="Camera a", lat=51.0, lon=0.0, image_url="https://example.test/a"
    )
    assert isinstance(cam.lat, float)


@pytest.mark.parametrize(
    "place",
    [
        "not an object",
        None,
        _place(id=None),
        _place(id=""),
        _place(id=42),
        _place(commonName=None),
        _place(lat=None),
        _place(lon="-0.1"),
        _place(lat=True),
        _place(lat=float("nan")),
        _place(lat=91.0),
        _place(lon=-180.5),
        _place(additionalProperties=None),
        _place(additionalProperties={"available": "true"}),
    ],
    ids=repr,
)
def test_malformed_places_are_counted(place: object) -> None:
    result = registry.parse_places([place, _place("ok")])
    assert [c.id for c in result.cameras] == ["ok"]
    assert result.skipped_malformed == 1
    assert result.skipped_unavailable == 0


@pytest.mark.parametrize(
    "image_url", ["http://example.test/a", "HTTPS://example.test/a", "", 7, "//example.test/a"]
)
def test_image_url_must_be_https(image_url: object) -> None:
    props = [{"key": "available", "value": "true"}, {"key": "imageUrl", "value": image_url}]
    result = registry.parse_places([_place(additionalProperties=props)])
    assert result.cameras == []
    assert result.skipped_malformed == 1


@pytest.mark.parametrize("available", ["false", "True", "", None])
def test_anything_but_available_true_is_unavailable_not_malformed(available: object) -> None:
    props: list[object] = [{"key": "imageUrl", "value": "https://example.test/a"}]
    if available is not None:
        props.append({"key": "available", "value": available})
    result = registry.parse_places([_place(additionalProperties=props)])
    assert result.cameras == []
    assert result.skipped_unavailable == 1
    assert result.skipped_malformed == 0


def test_junk_property_items_are_ignored() -> None:
    props = [
        "junk",
        {"value": "no key"},
        {"key": 3, "value": "numeric key"},
        {"key": "available", "value": "true"},
        {"key": "imageUrl", "value": "https://example.test/a"},
    ]
    assert len(registry.parse_places([_place(additionalProperties=props)]).cameras) == 1


def test_empty_registry_is_not_padded() -> None:
    result = registry.parse_places([])
    assert result == registry.Registry(cameras=[], skipped_unavailable=0, skipped_malformed=0)


# Fetching and retries -------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None),  # type: ignore[arg-type]
        urllib.error.URLError("dns failure"),
        http.client.IncompleteRead(b"[{"),
        ConnectionResetError("reset"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_transport_errors_are_retried(error: Exception) -> None:
    attempts: list[int] = []
    body = json.dumps([_place("ok")]).encode()

    def fetch(url: str, timeout: float) -> bytes:
        attempts.append(1)
        if len(attempts) == 1:
            raise error
        return body

    sleeps: list[float] = []
    cams = registry.list_cameras(None, fetch=fetch, sleep=sleeps.append)
    assert [c.id for c in cams] == ["ok"]
    assert sleeps == [1]


def test_unexpected_errors_are_not_swallowed() -> None:
    def fetch(url: str, timeout: float) -> bytes:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        registry.list_cameras(None, fetch=fetch, sleep=_never_sleep)


def test_failed_attempts_are_logged_without_the_app_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="wearreport")

    def fetch(url: str, timeout: float) -> bytes:
        raise OSError(f"cannot reach {url}")

    with pytest.raises(registry.RegistryError) as info:
        registry.list_cameras("s3cr3t", fetch=fetch, sleep=lambda s: None)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [getattr(r, "attempt") for r in warnings] == [1, 2]  # noqa: B009
    assert all(getattr(r, "error") == "OSError" for r in warnings)  # noqa: B009
    assert "s3cr3t" not in caplog.text
    assert "s3cr3t" not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_app_key_is_url_encoded() -> None:
    urls: list[str] = []

    def fetch(url: str, timeout: float) -> bytes:
        urls.append(url)
        return b"[]"

    registry.list_cameras("a b&c", fetch=fetch, sleep=_never_sleep)
    assert urls == ["https://api.tfl.gov.uk/Place/Type/JamCam?app_key=a+b%26c"]


def test_http_fetch_sends_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _Response:
        seen.append(req)
        return _Response(b"[]")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert registry.http_fetch("https://example.test/x", 30) == b"[]"
    assert seen[0].full_url == "https://example.test/x"
    assert seen[0].get_header("User-agent", "").startswith("wearreport")


# Command line ---------------------------------------------------------------------------


def test_main_uses_app_key_from_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    urls: list[str] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _Response:
        urls.append(req.full_url)
        return _Response(FIXTURE.read_bytes())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("TFL_APP_KEY", "k123")
    assert registry.main() == 0
    assert urls == ["https://api.tfl.gov.uk/Place/Type/JamCam?app_key=k123"]
    out = capsys.readouterr().out
    assert "available cameras: 8\n" in out
    assert "skipped unavailable: 1\n" in out
    assert "k123" not in out


def test_main_reports_failure_with_exit_code_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _Response:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("TFL_APP_KEY", raising=False)
    sleeps: list[float] = []
    assert registry.main(sleep=sleeps.append) == 1
    assert sleeps == [1, 2]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RegistryError" not in captured.err
    assert "unavailable after 3 attempts (URLError)" in captured.err
