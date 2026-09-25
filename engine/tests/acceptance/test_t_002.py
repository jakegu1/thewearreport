"""Acceptance tests for T-002 (TfL camera registry). These are the task contract: do not edit."""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import sys
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from wearreport import registry

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "engine" / "tests" / "fixtures" / "jamcam_places.json"
FIXTURE_REL = "engine/tests/fixtures/jamcam_places.json"

AVAILABLE_IDS = {
    "JamCams_00002.00865",
    "JamCams_00001.02146",
    "JamCams_00001.04376",
    "JamCams_00001.08305",
    "JamCams_00001.03813",
    "JamCams_00001.06609",
    "JamCams_00001.04342",
    "JamCams_00002.00341",
}
UNAVAILABLE_ID = "JamCams_00001.03758"
MALFORMED_ID = "JamCams_00001.08961"  # recorded entry with its "lat" removed


def _fixture_bytes() -> bytes:
    return FIXTURE.read_bytes()


def _serve_fixture(url: str, timeout: float) -> bytes:
    return _fixture_bytes()


def _no_sleep(seconds: float) -> None:
    raise AssertionError(f"unexpected sleep({seconds})")


def _place(
    place_id: str,
    *,
    lat: object = 51.5,
    lon: object = -0.1,
    available: str = "true",
    image_url: str | None = "https://example.test/cam",
    with_props: bool = True,
) -> dict[str, Any]:
    place: dict[str, Any] = {"id": place_id, "commonName": f"Camera {place_id}"}
    if lat is not None:
        place["lat"] = lat
    if lon is not None:
        place["lon"] = lon
    if with_props:
        props = [{"key": "available", "value": available}]
        if image_url is not None:
            props.append({"key": "imageUrl", "value": image_url})
        place["additionalProperties"] = props
    return place


def _serve(places: list[dict[str, Any]]) -> registry.Fetch:
    body = json.dumps(places).encode()

    def fetch(url: str, timeout: float) -> bytes:
        return body

    return fetch


def _skipped_malformed(caplog: pytest.LogCaptureFixture) -> list[int]:
    return [
        getattr(rec, "skipped_malformed")  # noqa: B009
        for rec in caplog.records
        if rec.name.startswith("wearreport") and hasattr(rec, "skipped_malformed")
    ]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """AC5: no test in this file may open a real connection."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr("socket.socket.connect", refuse)
    monkeypatch.setattr("socket.create_connection", refuse)
    yield


# AC1: typed function and frozen Camera dataclass -------------------------------------


def test_ac1_list_cameras_signature() -> None:
    hints = typing.get_type_hints(registry.list_cameras)
    assert hints["app_key"] == str | None
    assert hints["return"] == list[registry.Camera]


def test_ac1_camera_is_frozen_dataclass_with_fields() -> None:
    assert dataclasses.is_dataclass(registry.Camera)
    names = [f.name for f in dataclasses.fields(registry.Camera)]
    assert names == ["id", "name", "lat", "lon", "image_url"]
    cam = registry.Camera(
        id="JamCams_1", name="A road", lat=51.5, lon=-0.1, image_url="https://example.test/c"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cam.lat = 0.0  # type: ignore[misc]


def test_ac1_list_cameras_returns_cameras_from_fixture() -> None:
    cams = registry.list_cameras(None, fetch=_serve_fixture, sleep=_no_sleep)
    assert all(isinstance(c, registry.Camera) for c in cams)
    first = next(c for c in cams if c.id == "JamCams_00002.00865")
    assert first.name == "A406 Billet Upass E"
    assert first.lat == pytest.approx(51.60067)
    assert first.lon == pytest.approx(-0.01594)
    assert first.image_url.startswith("https://")


# AC2: filtering ----------------------------------------------------------------------


def test_ac2_only_available_entries_with_image_url_are_returned() -> None:
    cams = registry.list_cameras(None, fetch=_serve_fixture, sleep=_no_sleep)
    ids = [c.id for c in cams]
    assert set(ids) == AVAILABLE_IDS
    assert len(ids) == len(AVAILABLE_IDS)
    assert UNAVAILABLE_ID not in ids
    assert MALFORMED_ID not in ids


def test_ac2_non_https_and_empty_image_urls_are_skipped() -> None:
    places = [
        _place("ok", image_url="https://example.test/ok"),
        _place("plain-http", image_url="http://example.test/cam"),
        _place("empty-url", image_url=""),
        _place("no-url", image_url=None),
        _place("off", available="false"),
    ]
    cams = registry.list_cameras(None, fetch=_serve(places), sleep=_no_sleep)
    assert [c.id for c in cams] == ["ok"]


# AC3: malformed entries are skipped, counted and logged -------------------------------


def test_ac3_fixture_malformed_count_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    registry.list_cameras(None, fetch=_serve_fixture, sleep=_no_sleep)
    assert _skipped_malformed(caplog) == [1]


def test_ac3_each_malformed_kind_is_skipped_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    places = [
        _place("ok"),
        _place("no-lat", lat=None),
        _place("no-lon", lon=None),
        _place("no-props", with_props=False),
        _place("http-url", image_url="http://example.test/cam"),
        _place("off", available="false"),  # unavailable, not malformed
    ]
    cams = registry.list_cameras(None, fetch=_serve(places), sleep=_no_sleep)
    assert [c.id for c in cams] == ["ok"]
    assert _skipped_malformed(caplog) == [4]


def test_ac3_zero_malformed_is_logged_as_zero(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    registry.list_cameras(None, fetch=_serve([_place("ok")]), sleep=_no_sleep)
    assert _skipped_malformed(caplog) == [0]


# AC4: standard-library HTTP, timeout, bounded retries, typed error -------------------


def test_ac4_no_runtime_dependencies_added() -> None:
    # The registry is standard-library only: every top-level module it imports is in the
    # standard library or is the engine package itself.
    source = (ROOT / "engine" / "wearreport" / "registry.py").read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add("wearreport" if node.level else (node.module or "").split(".")[0])
    assert imported, "no imports found"
    assert imported - sys.stdlib_module_names <= {"wearreport"}


def test_ac4_each_attempt_uses_30s_timeout() -> None:
    seen: list[float] = []

    def fetch(url: str, timeout: float) -> bytes:
        seen.append(timeout)
        return _fixture_bytes()

    registry.list_cameras(None, fetch=fetch, sleep=_no_sleep)
    assert seen == [30]


def test_ac4_retries_with_exponential_backoff_then_succeeds() -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def flaky(url: str, timeout: float) -> bytes:
        calls.append(url)
        if len(calls) < 3:
            raise TimeoutError("simulated timeout")
        return _fixture_bytes()

    cams = registry.list_cameras(None, fetch=flaky, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [1, 2]
    assert {c.id for c in cams} == AVAILABLE_IDS


def test_ac4_raises_registry_error_after_three_failed_attempts() -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def down(url: str, timeout: float) -> bytes:
        calls.append(url)
        raise OSError("simulated connection refused")

    with pytest.raises(registry.RegistryError):
        registry.list_cameras(None, fetch=down, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [1, 2]


@pytest.mark.parametrize("body", [b"not json", b'{"not": "a list"}', b""])
def test_ac4_unusable_response_raises_registry_error(body: bytes) -> None:
    sleeps: list[float] = []

    def fetch(url: str, timeout: float) -> bytes:
        return body

    with pytest.raises(registry.RegistryError):
        registry.list_cameras(None, fetch=fetch, sleep=sleeps.append)
    assert len(sleeps) <= 2


def test_ac4_registry_error_does_not_leak_app_key() -> None:
    def down(url: str, timeout: float) -> bytes:
        raise OSError(f"failed to reach {url}")

    with pytest.raises(registry.RegistryError) as info:
        registry.list_cameras("s3cr3t-key", fetch=down, sleep=lambda s: None)
    assert "s3cr3t-key" not in str(info.value)


def test_ac4_app_key_is_sent_only_when_given() -> None:
    urls: list[str] = []

    def fetch(url: str, timeout: float) -> bytes:
        urls.append(url)
        return _fixture_bytes()

    registry.list_cameras(None, fetch=fetch, sleep=_no_sleep)
    registry.list_cameras("abc123", fetch=fetch, sleep=_no_sleep)
    assert urls[0] == "https://api.tfl.gov.uk/Place/Type/JamCam"
    assert "app_key" not in urls[0]
    assert urls[1].startswith("https://api.tfl.gov.uk/Place/Type/JamCam?")
    assert "app_key=abc123" in urls[1]


def test_ac4_default_fetch_uses_bounded_get_with_30s_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, object]] = []

    def fake_bounded_get(url: str, *, timeout_s: float, max_bytes: int, **kwargs: object) -> bytes:
        calls.append((url, timeout_s, max_bytes))
        return _fixture_bytes()

    monkeypatch.setattr(registry, "bounded_get", fake_bounded_get)
    cams = registry.list_cameras(None, sleep=_no_sleep)
    assert calls == [(registry.JAMCAM_URL, 30, registry.MAX_BODY_BYTES)]
    assert {c.id for c in cams} == AVAILABLE_IDS


# AC5: trimmed recorded fixture and command-line entry point ---------------------------


def test_ac5_fixture_is_trimmed_real_response_shape() -> None:
    places = json.loads(FIXTURE.read_text())
    assert isinstance(places, list)
    assert 0 < len(places) <= 20
    ids = {p["id"] for p in places}
    assert UNAVAILABLE_ID in ids
    assert MALFORMED_ID in ids
    assert all(p.get("placeType") == "JamCam" for p in places)


def test_ac5_module_entry_point_prints_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeouts: list[object] = []

    def fake_bounded_get(url: str, *, timeout_s: float, max_bytes: int, **kwargs: object) -> bytes:
        timeouts.append(timeout_s)
        return _fixture_bytes()

    # `python -m wearreport.registry` runs the module's `if __name__ == "__main__"` block,
    # which must hand main()'s return code to SystemExit.
    tree = ast.parse((ROOT / "engine" / "wearreport" / "registry.py").read_text())
    guard = tree.body[-1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "__name__ == '__main__'"
    assert [ast.unparse(stmt) for stmt in guard.body] == ["raise SystemExit(main())"]

    monkeypatch.setattr(registry, "bounded_get", fake_bounded_get)
    monkeypatch.delenv("TFL_APP_KEY", raising=False)
    assert registry.main() == 0
    assert timeouts == [30]
    lines = capsys.readouterr().out.splitlines()
    assert "available cameras: 8" in lines
    assert "skipped malformed: 1" in lines


# AC7: fixture licence row ---------------------------------------------------------------


def test_ac7_fixture_licence_row() -> None:
    text = (ROOT / "fixtures" / "LICENSES.md").read_text()
    rows = [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in text.splitlines()
        if ln.strip().startswith("|")
    ]
    assert [
        "TfL Open Data (metadata only)",
        'TfL open data terms (attribution: "Powered by TfL Open Data")',
        FIXTURE_REL,
    ] in rows[2:]
