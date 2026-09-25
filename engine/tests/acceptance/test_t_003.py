"""Acceptance tests for T-003 (in-memory sweep fetcher). The task contract: do not edit."""

from __future__ import annotations

import dataclasses
import http.client
import inspect
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import tomllib
import typing
from collections.abc import Iterator
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pytest

import license_check
import privacy_guard
from wearreport import fetch, registry
from wearreport.testing.fake_cameras import FakeCameraServer

ROOT = Path(__file__).resolve().parents[3]
ERROR_KINDS = ("timeout", "http", "decode", "network")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


def _refused_camera(camera_id: str = "refused") -> registry.Camera:
    url = f"http://127.0.0.1:{_closed_port()}/cam/{camera_id}"
    return registry.Camera(id=camera_id, name="refused", lat=51.5, lon=-0.1, image_url=url)


def _summary(out: str) -> dict[str, float]:
    """Parse the `key: number` lines a sweep command prints."""
    found: dict[str, float] = {}
    for line in out.splitlines():
        m = re.fullmatch(r"([a-z ]+): ([0-9.]+)", line.strip())
        if m:
            found[m.group(1)] = float(m.group(2))
    return found


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Refuse any connection that is not to 127.0.0.1."""
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] == "127.0.0.1"):
            raise AssertionError(f"non-local connection attempted: {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    yield


# AC1: API ----------------------------------------------------------------------------


def test_ac1_fetch_sweep_signature() -> None:
    sig = inspect.signature(fetch.fetch_sweep)
    params = sig.parameters
    assert next(iter(params)) == "cameras"
    assert params["concurrency"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["concurrency"].default == 24
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].default == 20
    assert typing.get_type_hints(fetch.fetch_sweep)["return"] == list[fetch.FrameResult]


def test_ac1_frame_result_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(fetch.FrameResult)
    names = [f.name for f in dataclasses.fields(fetch.FrameResult)]
    assert names == ["camera_id", "frame", "seconds", "error"]
    result = fetch.FrameResult(camera_id="c", frame=None, seconds=0.1, error="http")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.error = None  # type: ignore[misc]


def test_ac1_results_follow_camera_order_and_have_one_attempt_each(offline: None) -> None:
    with FakeCameraServer() as server:
        cams = server.cameras(6)
        server.serve_404(cams[1].id)
        server.serve_corrupt(cams[3].id)
        results = fetch.fetch_sweep(cams, concurrency=3, timeout_s=5)
        assert [r.camera_id for r in results] == [c.id for c in cams]
        assert [server.requests(c.id) for c in cams] == [1] * 6  # no retries, even on failure
    assert [r.error for r in results] == [None, "http", None, "decode", None, None]
    for r in results:
        assert r.seconds >= 0
        if r.error is None:
            assert isinstance(r.frame, np.ndarray)
            assert r.frame.dtype == np.uint8 and r.frame.ndim == 3 and r.frame.shape[2] == 3
        else:
            assert r.frame is None


# AC2: in memory ----------------------------------------------------------------------


def test_ac2_decoding_uses_imdecode() -> None:
    source = Path(inspect.getfile(fetch)).read_text()
    assert "cv2.imdecode(" in source
    for forbidden in ("tempfile", "imwrite", "imread(", "urlretrieve", "lru_cache", "cache"):
        assert forbidden not in source, forbidden


def test_ac2_static_privacy_guard_passes_on_new_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert privacy_guard.main(["--root", str(ROOT)]) == 0
    assert "clean" in capsys.readouterr().out


def test_ac2_static_privacy_guard_scans_the_new_modules() -> None:
    scanned = {p.relative_to(ROOT).as_posix() for p in privacy_guard.engine_files(ROOT)}
    assert "engine/wearreport/fetch.py" in scanned
    assert "engine/wearreport/testing/fake_cameras.py" in scanned


def test_ac2_frames_and_bytes_are_not_logged(
    offline: None, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    caplog.set_level("DEBUG")
    with FakeCameraServer() as server:
        cams = server.cameras(4)
        server.serve_404(cams[0].id)
        fetch.fetch_sweep(cams, concurrency=2, timeout_s=5)
    captured = capsys.readouterr()
    text = caplog.text + captured.out + captured.err
    assert "�" not in text and "\xff\xd8" not in text  # no raw JPEG bytes
    assert "/9j/" not in text  # no base64 JPEG
    assert "array(" not in text and "[[[" not in text  # no printed arrays
    assert "127.0.0.1" not in text  # no URLs


# AC3: failure categories against the fake server ---------------------------------------


def test_ac3_each_category_and_a_mixed_sweep(offline: None) -> None:
    with FakeCameraServer() as server:
        ok, missing, slow, corrupt = server.cameras(4)
        server.serve_404(missing.id)
        server.serve_delay(slow.id, 5.0)
        server.serve_corrupt(corrupt.id)
        refused = _refused_camera()
        cams = [ok, missing, slow, corrupt, refused]
        started = time.monotonic()
        results = fetch.fetch_sweep(cams, concurrency=5, timeout_s=1)
        elapsed = time.monotonic() - started
    by_id = {r.camera_id: r for r in results}
    assert by_id[ok.id].error is None and by_id[ok.id].frame is not None
    assert by_id[missing.id].error == "http"
    assert by_id[slow.id].error == "timeout"
    assert by_id[corrupt.id].error == "decode"
    assert by_id[refused.id].error == "network"
    assert elapsed < 4  # the slow camera did not hold the sweep for its full delay


def test_ac3_unit_tests_cover_every_case() -> None:
    source = (ROOT / "engine" / "tests" / "unit" / "test_fetch.py").read_text()
    for word in ("404", "timeout", "corrupt", "refused", "mix"):
        assert word in source.lower(), word


# AC4: fake camera server -------------------------------------------------------------


def test_ac4_urls_are_local_and_have_no_extension() -> None:
    with FakeCameraServer() as server:
        for cam in server.cameras(3):
            parts = urlsplit(cam.image_url)
            assert parts.hostname == "127.0.0.1"
            assert Path(parts.path).suffix == ""


def _get(url: str) -> tuple[int, bytes]:
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname or "", parts.port, timeout=5)
    try:
        conn.request("GET", parts.path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def test_ac4_jpegs_are_generated_at_request_time() -> None:
    with FakeCameraServer() as server:
        (cam,) = server.cameras(1)
        status1, body1 = _get(cam.image_url)
        status2, body2 = _get(cam.image_url)
    assert status1 == status2 == 200
    assert body1[:3] == body2[:3] == b"\xff\xd8\xff"
    assert body1 != body2


def test_ac4_404_delay_and_corrupt_can_be_configured() -> None:
    with FakeCameraServer() as server:
        missing, slow, corrupt = server.cameras(3)
        server.serve_404(missing.id)
        server.serve_delay(slow.id, 0.5)
        server.serve_corrupt(corrupt.id)
        assert _get(missing.image_url)[0] == 404
        started = time.monotonic()
        status, body = _get(slow.image_url)
        assert status == 200 and body[:3] == b"\xff\xd8\xff"
        assert time.monotonic() - started >= 0.5
        status, body = _get(corrupt.image_url)
        assert status == 200 and len(body) > 0


def test_ac4_no_image_file_is_committed_under_engine() -> None:
    git = shutil.which("git")
    assert git is not None
    tracked = subprocess.run(
        [git, "ls-files", "engine"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert tracked
    assert [p for p in tracked if Path(p).suffix.lower() in IMAGE_SUFFIXES] == []


# AC5: runtime privacy test -----------------------------------------------------------

RUNTIME_TEST = "engine/tests/unit/test_privacy_runtime.py"


def _run_runtime_test(canary: bool) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "WEARREPORT_PRIVACY_CANARY"}
    if canary:
        env["WEARREPORT_PRIVACY_CANARY"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", RUNTIME_TEST],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_ac5_runtime_privacy_test_passes() -> None:
    proc = _run_runtime_test(canary=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r"\b\d+ passed\b", proc.stdout)
    assert "skipped" not in proc.stdout


def test_ac5_runtime_privacy_test_fails_when_the_canary_hook_is_enabled() -> None:
    proc = _run_runtime_test(canary=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "failed" in proc.stdout


def test_ac5_runtime_test_sets_up_the_required_conditions() -> None:
    source = (ROOT / RUNTIME_TEST).read_text()
    for needle in ("HOME", "TMPDIR", "chdir", "imwrite", "os.open", "VideoWriter", "ftyp"):
        assert needle in source, needle


# AC6: dry sweep -----------------------------------------------------------------------


def test_ac6_dry_sweep_runs_offline(offline: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert fetch.main(["--dry-run"]) == 0
    summary = _summary(capsys.readouterr().out)
    assert summary["cameras listed"] == 50
    assert summary["frames fetched"] == 50
    for kind in ERROR_KINDS:
        assert summary[f"errors {kind}"] == 0
    assert summary["seconds"] >= 0


def test_ac6_make_sweep_dry_exits_0() -> None:
    make = shutil.which("make")
    assert make is not None, "make is part of the command contract"
    proc = subprocess.run(
        [make, "sweep-dry"], cwd=ROOT, capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _summary(proc.stdout)
    assert summary["cameras listed"] == 50
    assert summary["frames fetched"] == 50


# AC7: live run command ---------------------------------------------------------------


def test_ac7_live_run_lists_with_the_registry_and_prints_counts(
    offline: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    keys: list[str | None] = []
    with FakeCameraServer() as server:
        cams = server.cameras(5)
        server.serve_404(cams[4].id)

        def list_cameras(app_key: str | None, **kwargs: object) -> list[registry.Camera]:
            keys.append(app_key)
            return cams

        monkeypatch.setattr(registry, "list_cameras", list_cameras)
        monkeypatch.setenv("TFL_APP_KEY", "live-key")
        assert fetch.main(["--live"]) == 0
    out = capsys.readouterr().out
    assert keys == ["live-key"]
    assert "live-key" not in out
    summary = _summary(out)
    assert summary["cameras listed"] == 5
    assert summary["frames fetched"] == 4
    assert summary["errors http"] == 1
    assert "seconds" in summary


def test_ac7_live_run_reports_registry_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def down(app_key: str | None, **kwargs: object) -> list[registry.Camera]:
        raise registry.RegistryError("JamCam registry unavailable after 3 attempts (URLError)")

    monkeypatch.setattr(registry, "list_cameras", down)
    assert fetch.main(["--live"]) == 1
    assert "registry unavailable" in capsys.readouterr().err


# AC8: dependencies -------------------------------------------------------------------


def test_ac8_runtime_dependencies_declared() -> None:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].lower() for d in deps}
    assert {"numpy", "opencv-python-headless"} <= names
    assert "ultralytics" not in names
    assert "opencv-python" not in names  # the GUI build is not needed


def test_ac8_numpy_licence_expression_is_accepted() -> None:
    meta = Message()
    meta["Name"] = "numpy"
    meta["Version"] = "2.0"
    meta["License-Expression"] = "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0"
    assert license_check.license_problem(meta) is None


def test_ac8_license_check_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert license_check.main([]) == 0
    assert "clean" in capsys.readouterr().out


# AC9: registry hardening -------------------------------------------------------------


def _place(place_id: str, **overrides: object) -> dict[str, Any]:
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
    place.update(overrides)
    return place


def _no_sleep(seconds: float) -> None:
    raise AssertionError(f"unexpected sleep({seconds})")


@pytest.mark.parametrize("field", ["lat", "lon"])
def test_ac9_overflowing_coordinate_is_malformed(field: str) -> None:
    huge = int("9" * 400)
    places = json.loads(json.dumps([_place("ok"), _place("huge", **{field: huge})]))
    assert places[1][field] == huge
    result = registry.parse_places(places)
    assert [c.id for c in result.cameras] == ["ok"]
    assert result.skipped_malformed == 1
    for sign in (1, -1):
        mixed = registry.parse_places([_place("ok"), _place("big", **{field: sign * huge})])
        assert [c.id for c in mixed.cameras] == ["ok"]
        assert mixed.skipped_malformed == 1


def test_ac9_deeply_nested_body_raises_registry_error_without_retry() -> None:
    calls: list[str] = []
    body = b"[" * 200_000 + b"]" * 200_000

    def fetch_body(url: str, timeout: float) -> bytes:
        calls.append(url)
        return body

    with pytest.raises(registry.RegistryError):
        registry.list_cameras(None, fetch=fetch_body, sleep=_no_sleep)
    assert len(calls) == 1


def test_ac9_body_over_16_mb_raises_registry_error() -> None:
    body = b"[" + b" " * (16 * 1024 * 1024) + b"]"

    def fetch_body(url: str, timeout: float) -> bytes:
        return body

    with pytest.raises(registry.RegistryError):
        registry.list_cameras(None, fetch=fetch_body, sleep=lambda s: None)


def test_ac9_http_fetch_stops_reading_past_16_mb(offline: None) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Big(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            chunk = b" " * (1024 * 1024)
            try:
                for _ in range(64):
                    self.wfile.write(chunk)
            except OSError:
                pass

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Big)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        try:
            body = registry.http_fetch(url, 10)
        except registry.RegistryError:
            pass  # refusing the body outright is fine too
        else:
            assert len(body) <= 16 * 1024 * 1024 + 1
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_ac9_client_errors_other_than_429_are_not_retried(status: int) -> None:
    import urllib.error

    calls: list[str] = []

    def rejected(url: str, timeout: float) -> bytes:
        calls.append(url)
        raise urllib.error.HTTPError(url, status, "rejected", Message(), None)

    with pytest.raises(registry.RegistryError) as info:
        registry.list_cameras("secret-key", fetch=rejected, sleep=_no_sleep)
    assert len(calls) == 1
    assert "secret-key" not in str(info.value)


def test_ac9_429_is_still_retried() -> None:
    import urllib.error

    calls: list[str] = []
    sleeps: list[float] = []
    body = json.dumps([_place("ok")]).encode()

    def limited(url: str, timeout: float) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "slow down", Message(), None)
        return body

    cams = registry.list_cameras(None, fetch=limited, sleep=sleeps.append)
    assert [c.id for c in cams] == ["ok"]
    assert sleeps == [1]
