"""Acceptance tests for T-004 (detector module and YOLOX-s/m benchmark). The task
contract: do not edit.

Tests that need a model file skip only when the file is missing and
WEARREPORT_REQUIRE_MODEL is unset; CI sets it, so there a missing model fails.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tomllib
import typing
from collections.abc import Iterator
from email.message import Message
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

import license_check
import privacy_guard
from wearreport import benchmark, detect, fetch, registry
from wearreport._cv import cv2
from wearreport.testing.fake_cameras import FakeCameraServer

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "fixtures" / "detect"
FETCH_MODEL = ROOT / "scripts" / "fetch_model.sh"
REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
PEOPLE = FIXTURES / "people_aldgate.jpg"
UMBRELLA = FIXTURES / "umbrella_rain.jpg"


def _model(name: str = "yolox_s.onnx") -> Path:
    """The model file, or a skip (a failure when WEARREPORT_REQUIRE_MODEL is set)."""
    path = detect.model_path(name)
    if not path.is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run scripts/fetch_model.sh")
    return path


@pytest.fixture(scope="module")
def detector() -> detect.Detector:
    return detect.Detector(_model())


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


def _read(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None, path
    return np.asarray(frame)


# AC1: detector API -------------------------------------------------------------------


def test_ac1_detector_signature_and_default_thresholds() -> None:
    params = inspect.signature(detect.Detector).parameters
    assert next(iter(params)) == "model_path"
    for name, default in (("conf", detect.DEFAULT_CONF), ("nms", detect.DEFAULT_NMS)):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default == default
    hints = typing.get_type_hints(detect.Detector.detect)
    assert hints["return"] == list[detect.Detection]


def test_ac1_default_thresholds_are_defined_once() -> None:
    source = (ROOT / "engine" / "wearreport" / "detect.py").read_text()
    assert len(re.findall(r"^DEFAULT_CONF = ", source, flags=re.MULTILINE)) == 1
    assert len(re.findall(r"^DEFAULT_NMS = ", source, flags=re.MULTILINE)) == 1
    for literal in (repr(detect.DEFAULT_CONF), repr(detect.DEFAULT_NMS)):
        assert source.count(literal) == 1, literal
        assert literal not in (ROOT / "engine" / "wearreport" / "benchmark.py").read_text()


def test_ac1_detection_is_a_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(detect.Detection)
    assert [f.name for f in dataclasses.fields(detect.Detection)] == ["label", "score", "box"]
    hints = typing.get_type_hints(detect.Detection)
    assert hints["label"] == Literal["person", "umbrella"]
    d = detect.Detection(label="person", score=0.9, box=(1.0, 2.0, 3.0, 4.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.score = 0.1  # type: ignore[misc]


@pytest.mark.parametrize("path", [PEOPLE, UMBRELLA], ids=lambda p: p.name)
def test_ac1_boxes_are_in_frame_pixels_and_clipped(detector: detect.Detector, path: Path) -> None:
    frame = _read(path)
    height, width = frame.shape[:2]
    found = detector.detect(frame)
    assert found
    for d in found:
        assert d.label in ("person", "umbrella")
        assert detector.conf <= d.score <= 1
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, d


def test_ac1_boxes_are_clipped_on_a_crop_touching_every_edge(detector: detect.Detector) -> None:
    # Cut the people fixture so that people run off every edge of the frame.
    frame = _read(PEOPLE)
    crop = frame[frame.shape[0] // 4 : 3 * frame.shape[0] // 4, 100:-100]
    found = detector.detect(crop)
    assert found
    for d in found:
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= crop.shape[1] and 0 <= y1 < y2 <= crop.shape[0], d


def test_ac1_thresholds_are_parameters(detector: detect.Detector) -> None:
    frame = _read(PEOPLE)
    strict = detect.Detector(_model(), conf=0.9)
    loose = detect.Detector(_model(), conf=0.1)
    assert len(strict.detect(frame)) < len(detector.detect(frame)) < len(loose.detect(frame))
    assert all(d.score >= 0.9 for d in strict.detect(frame))


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((288, 352), dtype=np.uint8),  # grayscale
        np.zeros((288, 352, 4), dtype=np.uint8),  # BGRA
        np.zeros((288, 352, 1), dtype=np.uint8),
        np.zeros((288, 352, 3), dtype=np.float32),
        np.zeros((288, 352, 3), dtype=np.uint16),
        np.zeros((0, 352, 3), dtype=np.uint8),
        np.zeros((288, 0, 3), dtype=np.uint8),
    ],
    ids=["gray", "bgra", "one-channel", "float32", "uint16", "no-rows", "no-columns"],
)
def test_ac1_invalid_frames_raise_value_error(detector: detect.Detector, frame: Any) -> None:
    with pytest.raises(ValueError):
        detector.detect(frame)


# AC2: model download -----------------------------------------------------------------


def test_ac2_script_pins_both_models_and_uses_safe_curl() -> None:
    script = FETCH_MODEL.read_text()
    assert "curl --proto '=https' --tlsv1.2 --max-time 300 --retry 3" in script
    assert "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/" in script
    assert "--with-m" in script
    pins = set(re.findall(r"\b[0-9a-f]{64}\b", script))
    assert pins == set(detect.MODEL_SHA256.values())
    assert set(detect.MODEL_SHA256) == {"yolox_s.onnx", "yolox_m.onnx"}


def test_ac2_models_are_ignored_and_never_committed() -> None:
    assert ".models/" in (ROOT / ".gitignore").read_text().splitlines()
    git = shutil.which("git")
    assert git is not None
    tracked = subprocess.run(
        [git, "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert [p for p in tracked if p.endswith(".onnx") or p.startswith(".models/")] == []
    ignored = subprocess.run(
        [git, "check-ignore", "-q", ".models/yolox_s.onnx"], cwd=ROOT, check=False
    )
    assert ignored.returncode == 0


def _fake_curl(tmp_path: Path, body: str) -> dict[str, str]:
    """An environment whose `curl` logs its arguments and runs `body` into its -o file.

    `body` is shell code; `$url` holds the last argument.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$CURL_LOG"\n'
        'out=""; prev=""; url=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; url="$a"; done\n'
        f'{body} > "$out"\n'
    )
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["CURL_LOG"] = str(tmp_path / "curl.log")
    return env


def _fetch(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    sh = shutil.which("sh")
    assert sh is not None
    return subprocess.run(
        [sh, str(FETCH_MODEL), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_ac2_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    env = _fake_curl(tmp_path, "printf 'not a model'")
    dest = tmp_path / "models"
    proc = _fetch(env, "--dest", str(dest))
    assert proc.returncode != 0
    assert "yolox_s.onnx" in proc.stderr
    assert not dest.exists() or list(dest.iterdir()) == []  # nothing left, not even a part
    log = (tmp_path / "curl.log").read_text()
    assert "--proto =https --tlsv1.2 --max-time 300 --retry 3" in log
    assert "yolox_s.onnx" in log and "yolox_m.onnx" not in log


def test_ac2_corrupt_existing_model_is_not_kept(tmp_path: Path) -> None:
    env = _fake_curl(tmp_path, "printf 'still not a model'")
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "yolox_s.onnx").write_bytes(b"corrupted")
    proc = _fetch(env, "--dest", str(dest))
    assert proc.returncode != 0
    assert list(dest.iterdir()) == []


def test_ac2_with_m_verifies_yolox_m_too(tmp_path: Path) -> None:
    source = _model()
    body = f"case \"$url\" in *yolox_s.onnx) cat '{source}' ;; *) printf bad ;; esac"
    env = _fake_curl(tmp_path, body)
    dest = tmp_path / "models"
    proc = _fetch(env, "--with-m", "--dest", str(dest))
    assert proc.returncode != 0
    assert "yolox_m.onnx" in proc.stderr
    assert [p.name for p in dest.iterdir()] == ["yolox_s.onnx"]  # verified; m discarded
    log = (tmp_path / "curl.log").read_text()
    assert "yolox_s.onnx" in log and "yolox_m.onnx" in log


def test_ac2_verified_download_is_installed(tmp_path: Path) -> None:
    source = _model()
    env = _fake_curl(tmp_path, f"cat '{source}'")
    dest = tmp_path / "models"
    proc = _fetch(env, "--dest", str(dest))
    assert proc.returncode == 0, proc.stderr
    assert [p.name for p in dest.iterdir()] == ["yolox_s.onnx"]
    assert detect.sha256_of(dest / "yolox_s.onnx") == detect.MODEL_SHA256["yolox_s.onnx"]
    # A second run finds the verified file and downloads nothing.
    (tmp_path / "curl.log").unlink()
    assert _fetch(env, "--dest", str(dest)).returncode == 0
    assert not (tmp_path / "curl.log").exists()


def test_ac2_detector_refuses_a_model_that_is_not_pinned(tmp_path: Path) -> None:
    fake = tmp_path / "yolox_s.onnx"
    fake.write_bytes(b"\x08\x07not really onnx")
    with pytest.raises(detect.DetectorError):
        detect.Detector(fake)
    with pytest.raises(detect.DetectorError):
        detect.Detector(tmp_path / "missing.onnx")


def test_ac2_downloaded_models_match_their_pins() -> None:
    for name, digest in detect.MODEL_SHA256.items():
        path = detect.model_path(name)
        if name == "yolox_s.onnx":
            path = _model(name)
        elif not path.is_file():
            continue  # yolox_m is optional (--with-m)
        assert detect.sha256_of(path) == digest, name


# AC3: fixtures -----------------------------------------------------------------------

ALLOWED_FIXTURE_LICENCE = re.compile(r"(CC0 1\.0|Public domain|CC BY \d\.\d)", re.IGNORECASE)


def _licence_rows() -> dict[str, tuple[str, str]]:
    text = (ROOT / "fixtures" / "LICENSES.md").read_text()
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")][2:]
    found = {}
    for row in rows:
        source, licence, file = (c.strip() for c in row.strip().strip("|").split("|"))
        found[file] = (source, licence)
    return found


def test_ac3_fixture_images_are_small_licensed_and_recorded() -> None:
    images = sorted(p for p in FIXTURES.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    assert len(images) >= 2
    assert PEOPLE in images and UMBRELLA in images
    rows = _licence_rows()
    for image in images:
        assert image.stat().st_size <= 300 * 1024, image.name
        rel = image.relative_to(ROOT).as_posix()
        assert rel in rows, f"{rel} is not in fixtures/LICENSES.md"
        source, licence = rows[rel]
        assert "https://" in source and len(source.split()) >= 2, source  # URL and author
        assert ALLOWED_FIXTURE_LICENCE.fullmatch(licence), licence
        assert not re.search(r"\b(SA|NC|ND)\b", licence.upper()), licence


def test_ac3_people_fixture_has_people(detector: detect.Detector) -> None:
    persons = [d for d in detector.detect(_read(PEOPLE)) if d.label == "person"]
    assert len(persons) >= 2
    assert all(d.score >= detect.DEFAULT_CONF for d in persons)


def test_ac3_umbrella_fixture_has_an_umbrella(detector: detect.Detector) -> None:
    umbrellas = [d for d in detector.detect(_read(UMBRELLA)) if d.label == "umbrella"]
    assert len(umbrellas) >= 1
    assert all(d.score >= detect.DEFAULT_CONF for d in umbrellas)


def test_ac3_model_free_unit_tests_exist() -> None:
    source = (ROOT / "engine" / "tests" / "unit" / "test_detect.py").read_text()
    names = re.findall(r"^def (test_\w+)", source, flags=re.MULTILINE)
    for topic in ("letterbox", "unletterbox", "decode", "nms", "class", "clip"):
        assert any(topic in n for n in names), topic


# AC4: the model runs in CI -----------------------------------------------------------


def test_ac4_missing_model_skips_only_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(detect, "MODEL_DIR", tmp_path)
    monkeypatch.delenv(REQUIRE_MODEL, raising=False)
    with pytest.raises(pytest.skip.Exception):
        _model()
    monkeypatch.setenv(REQUIRE_MODEL, "1")
    with pytest.raises(pytest.fail.Exception):
        _model()


def test_ac4_ci_fetches_the_model_and_requires_it() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert re.search(r'WEARREPORT_REQUIRE_MODEL: "?1"?', ci)
    assert "scripts/fetch_model.sh" in ci or "make model" in ci
    makefile = (ROOT / "Makefile").read_text()
    assert re.search(r"^model:.*\n\tsh scripts/fetch_model\.sh", makefile, flags=re.MULTILINE)
    assert re.search(r"^setup:.*\bmodel\b", makefile, flags=re.MULTILINE) or re.search(
        r"^setup:(.*\n\t.*)*\$\(MAKE\) model", makefile, flags=re.MULTILINE
    )


# AC5: latency ------------------------------------------------------------------------


def test_ac5_synthetic_benchmark_prints_the_median(
    offline: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _model()
    assert benchmark.main(["--synthetic", "3"]) == 0
    out = capsys.readouterr().out
    m = re.search(r"^median ms per frame: ([0-9.]+)$", out, flags=re.MULTILINE)
    assert m, out
    assert float(m.group(1)) > 0
    assert re.search(r"^frames: 3$", out, flags=re.MULTILINE)
    assert "352x288" in out


def test_ac5_ci_runs_the_synthetic_benchmark() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "wearreport.benchmark --synthetic 20" in ci


# AC6: live comparison ----------------------------------------------------------------


def _models_dir(tmp_path: Path) -> Path:
    """yolox_s and yolox_m; yolox_m falls back to a copy of yolox_s when not downloaded."""
    s = _model()
    m = detect.model_path("yolox_m.onnx")
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolox_s.onnx").symlink_to(s)
    (models / "yolox_m.onnx").symlink_to(m if m.is_file() else s)
    return models


def _table(out: str) -> dict[str, dict[str, str]]:
    lines = [ln for ln in out.splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = {}
    for line in lines[2:]:
        cells = dict(zip(header, (c.strip() for c in line.strip("|").split("|")), strict=True))
        rows[cells["model"]] = cells
    return rows


def test_ac6_live_uses_the_registry_and_the_fetcher_on_the_same_frames(
    offline: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    models = _models_dir(tmp_path)
    same_model = (models / "yolox_m.onnx").resolve() == (models / "yolox_s.onnx").resolve()
    sweeps: list[int] = []
    real_sweep = fetch.fetch_sweep
    with FakeCameraServer() as server:
        cams = server.cameras(8)
        for cam in cams[:4]:
            server.serve_body(cam.id, PEOPLE.read_bytes())
        server.serve_404(cams[4].id)

        def list_cameras(app_key: str | None, **kwargs: object) -> list[registry.Camera]:
            return cams

        def fetch_sweep(cameras: Any, **kwargs: Any) -> list[fetch.FrameResult]:
            sweeps.append(len(cameras))
            return real_sweep(cameras, **kwargs)

        monkeypatch.setattr(registry, "list_cameras", list_cameras)
        monkeypatch.setattr(fetch, "fetch_sweep", fetch_sweep)
        code = benchmark.main(["--live", "--cameras", "6", "--model-dir", str(models)])
    assert code == 0
    assert sweeps == [6]  # one sweep, the first 6 cameras, shared by both models
    out = capsys.readouterr().out
    table = _table(out)
    assert set(table) == {"yolox_s", "yolox_m"}
    columns = ("frames", "median ms/frame", "persons", "umbrellas", "total s")
    for row in table.values():
        assert all(row[c] for c in columns), row
        assert row["frames"] == "5"  # 6 cameras, one 404
        assert float(row["median ms/frame"]) > 0 and float(row["total s"]) > 0
    assert int(table["yolox_s"]["persons"]) >= 8  # 4 frames of the people fixture
    if same_model:
        assert table["yolox_s"]["persons"] == table["yolox_m"]["persons"]
    assert "127.0.0.1" not in out


def test_ac6_live_calls_the_registry_and_fetcher_by_module() -> None:
    source = (ROOT / "engine" / "wearreport" / "benchmark.py").read_text()
    assert "registry.list_cameras(" in source
    assert "fetch.fetch_sweep(" in source
    assert "--cameras" in source


def test_ac6_benchmark_reports_registry_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    models = _models_dir(tmp_path)

    def down(app_key: str | None, **kwargs: object) -> list[registry.Camera]:
        raise registry.RegistryError("registry request failed: HTTP 503")

    monkeypatch.setattr(registry, "list_cameras", down)
    assert benchmark.main(["--live", "--model-dir", str(models)]) == 1
    assert "registry" in capsys.readouterr().err


def test_ac6_runtime_privacy_test_covers_the_benchmark() -> None:
    source = (ROOT / "engine" / "tests" / "unit" / "test_privacy_runtime.py").read_text()
    assert "from wearreport import benchmark" in source or "wearreport.benchmark" in source
    names = re.findall(r"^def (test_\w+)", source, flags=re.MULTILINE)
    assert any("benchmark" in n for n in names)


# AC8: dependencies -------------------------------------------------------------------


def test_ac8_onnxruntime_declared_and_no_ultralytics() -> None:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].lower() for d in deps}
    assert {"numpy", "opencv-python-headless", "onnxruntime"} <= names
    for path in ("pyproject.toml", "uv.lock"):
        assert "ultralytics" not in (ROOT / path).read_text().lower(), path


def test_ac8_protobuf_legacy_licence_field_is_accepted() -> None:
    meta = Message()
    meta["Name"] = "protobuf"
    meta["Version"] = "7.0"
    meta["License"] = "3-Clause BSD License"
    assert license_check.license_problem(meta) is None


def test_ac8_license_check_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert license_check.main([]) == 0
    out = capsys.readouterr().out
    assert "clean" in out
    for name in ("onnxruntime", "protobuf", "flatbuffers"):
        assert name in out, name


def test_ac8_privacy_guard_scans_and_passes_the_new_modules(
    capsys: pytest.CaptureFixture[str],
) -> None:
    scanned = {p.relative_to(ROOT).as_posix() for p in privacy_guard.engine_files(ROOT)}
    assert {"engine/wearreport/detect.py", "engine/wearreport/benchmark.py"} <= scanned
    assert privacy_guard.main(["--root", str(ROOT)]) == 0


def test_ac8_public_guard_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "public_guard.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
