"""scripts/fetch_model.sh: part files left by a killed run. The download itself, checksum
failures and --with-m are covered by engine/tests/acceptance/test_t_004.py."""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

from wearreport import detect

REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
FETCH_MODEL = Path(__file__).resolve().parents[3] / "scripts" / "fetch_model.sh"


def _model(name: str = "yolox_s.onnx") -> Path:
    path = detect.model_path(name)
    if not path.is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run scripts/fetch_model.sh")
    return path


def _env(tmp_path: Path, body: str) -> dict[str, str]:
    """An environment whose `curl` counts its calls and runs `body` into its -o file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        'echo called >> "$CURL_LOG"\n'
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        f'{body} > "$out"\n'
    )
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["CURL_LOG"] = str(tmp_path / "curl.log")
    return env


def _fetch(env: dict[str, str], dest: Path) -> subprocess.CompletedProcess[str]:
    sh = shutil.which("sh")
    assert sh is not None
    return subprocess.run(
        [sh, str(FETCH_MODEL), "--dest", str(dest)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _calls(tmp_path: Path) -> int:
    log = tmp_path / "curl.log"
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_part_files_left_by_a_killed_download_are_removed(tmp_path: Path) -> None:
    """A real run killed with SIGKILL mid-download leaves its part file; the next run
    removes it and installs only the verified model."""
    dest = tmp_path / "models"
    started = tmp_path / "started"
    hang = f"touch '{started}'; sleep 60"
    sh = shutil.which("sh")
    assert sh is not None
    proc = subprocess.Popen(
        [sh, str(FETCH_MODEL), "--dest", str(dest)],
        env=_env(tmp_path, hang),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not started.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists(), "the fake download never started"
    finally:
        os.killpg(proc.pid, signal.SIGKILL)  # the script and its curl, no traps run
        proc.wait(timeout=10)
    left = [p.name for p in dest.iterdir()]
    assert len(left) == 1 and left[0].startswith(".yolox_s.onnx."), left

    proc2 = _fetch(_env(tmp_path, f"cat '{_model()}'"), dest)
    assert proc2.returncode == 0, proc2.stderr
    assert [p.name for p in dest.iterdir()] == ["yolox_s.onnx"]
    assert detect.sha256_of(dest / "yolox_s.onnx") == detect.MODEL_SHA256["yolox_s.onnx"]


def test_part_files_are_removed_even_when_the_model_is_already_verified(tmp_path: Path) -> None:
    dest = tmp_path / "models"
    dest.mkdir()
    shutil.copyfile(_model(), dest / "yolox_s.onnx")
    for name in (".yolox_s.onnx.a1B2c3", ".yolox_s.onnx.zzzzzz"):
        (dest / name).write_bytes(b"half a model")
    (dest / ".yolox_s.onnx.link").symlink_to(tmp_path / "missing")
    proc = _fetch(_env(tmp_path, "printf unused"), dest)
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in dest.iterdir()) == ["yolox_s.onnx"]
    assert _calls(tmp_path) == 0


def test_part_file_cleanup_is_narrow(tmp_path: Path) -> None:
    """Only this model's part files: not other models', not directories, not other names."""
    dest = tmp_path / "models"
    dest.mkdir()
    keep = [".yolox_m.onnx.a1B2c3", "yolox_s.onnx.a1B2c3", ".yolox_s.onnx", "notes.txt"]
    for name in keep:
        (dest / name).write_bytes(b"keep")
    (dest / ".yolox_s.onnx.dir").mkdir()
    (dest / ".yolox_s.onnx.a1B2c3").write_bytes(b"half a model")
    proc = _fetch(_env(tmp_path, f"cat '{_model()}'"), dest)
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in dest.iterdir()) == sorted(
        [*keep, ".yolox_s.onnx.dir", "yolox_s.onnx"]
    )
    assert _calls(tmp_path) == 1


def test_a_failed_download_after_cleanup_leaves_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / ".yolox_s.onnx.a1B2c3").write_bytes(b"half a model")
    proc = _fetch(_env(tmp_path, "printf 'not a model'"), dest)
    assert proc.returncode != 0
    assert list(dest.iterdir()) == []
