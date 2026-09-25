"""The detector benchmark's model labels (wearreport.benchmark)."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from wearreport import benchmark, detect

REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
SMALL, MEDIUM = b"small model bytes", b"medium model bytes"


def _model(name: str = "yolox_s.onnx") -> Path:
    path = detect.model_path(name)
    if not path.is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run scripts/fetch_model.sh")
    return path


@pytest.fixture
def pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin SMALL as yolox_s and MEDIUM as yolox_m, and open any model as a blank one."""

    class Blank:
        def run(self, tensor: np.ndarray) -> list[np.ndarray]:
            return [np.zeros(detect.OUTPUT_SHAPE, dtype=np.float32)]

    monkeypatch.setattr(
        detect,
        "MODEL_SHA256",
        {
            "yolox_s.onnx": hashlib.sha256(SMALL).hexdigest(),
            "yolox_m.onnx": hashlib.sha256(MEDIUM).hexdigest(),
        },
    )
    monkeypatch.setattr(detect, "open_session", lambda path: Blank())


def _models(tmp_path: Path, s: bytes, m: bytes) -> list[Path]:
    paths = [tmp_path / "yolox_s.onnx", tmp_path / "yolox_m.onnx"]
    for path, data in zip(paths, (s, m), strict=True):
        path.write_bytes(data)
    return paths


def _table_rows(out: str) -> list[str]:
    return [ln.strip("|").split("|")[0].strip() for ln in out.splitlines() if ln.startswith("| ")][
        1:
    ]


def test_identity_names_each_file_by_its_pin(pins: None, tmp_path: Path) -> None:
    assert benchmark.identity(_models(tmp_path, SMALL, MEDIUM)) == [
        "sha256: yolox_s.onnx is the pinned yolox_s.onnx",
        "sha256: yolox_m.onnx is the pinned yolox_m.onnx",
    ]


def test_identity_flags_a_yolox_m_file_holding_yolox_s(pins: None, tmp_path: Path) -> None:
    lines = benchmark.identity(_models(tmp_path, SMALL, SMALL))
    assert lines == [
        "sha256: yolox_s.onnx is the pinned yolox_s.onnx",
        "sha256: yolox_m.onnx is the pinned yolox_s.onnx, not yolox_m.onnx",
        "warning: yolox_s.onnx and yolox_m.onnx have the same SHA-256; "
        "their rows measure one model",
    ]


def test_identity_flags_swapped_files(pins: None, tmp_path: Path) -> None:
    lines = benchmark.identity(_models(tmp_path, MEDIUM, SMALL))
    assert lines == [
        "sha256: yolox_s.onnx is the pinned yolox_m.onnx, not yolox_s.onnx",
        "sha256: yolox_m.onnx is the pinned yolox_s.onnx, not yolox_m.onnx",
    ]


def test_identity_of_unpinned_or_missing_files(pins: None, tmp_path: Path) -> None:
    (tmp_path / "yolox_s.onnx").write_bytes(b"something else")
    lines = benchmark.identity([tmp_path / "yolox_s.onnx", tmp_path / "yolox_m.onnx"])
    assert lines == [
        "sha256: yolox_s.onnx matches no pinned model",
        "sha256: yolox_m.onnx matches no pinned model",  # missing: no warning either
    ]


def test_comparison_keeps_the_table_and_adds_the_lines_below_it(
    pins: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _models(tmp_path, SMALL, SMALL)
    assert benchmark.main(["--dry-run", "--cameras", "2", "--model-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert _table_rows(out) == ["yolox_s", "yolox_m"]  # file stems, as the table always had
    after_table = out.split("| yolox_m |", 1)[1].split("\n", 1)[1].splitlines()
    assert after_table == benchmark.identity(_models(tmp_path, SMALL, SMALL))


def test_synthetic_run_names_its_model_by_digest(
    pins: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _models(tmp_path, MEDIUM, MEDIUM)
    assert benchmark.main(["--synthetic", "1", "--model-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "model: yolox_s\n" in out
    assert "sha256: yolox_s.onnx is the pinned yolox_m.onnx, not yolox_s.onnx\n" in out


def test_real_yolox_s_bytes_under_the_yolox_m_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copyfile(_model(), tmp_path / "yolox_s.onnx")
    shutil.copyfile(_model(), tmp_path / "yolox_m.onnx")
    assert benchmark.main(["--dry-run", "--cameras", "2", "--model-dir", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "sha256: yolox_s.onnx is the pinned yolox_s.onnx" in lines
    assert "sha256: yolox_m.onnx is the pinned yolox_s.onnx, not yolox_m.onnx" in lines
    assert [ln for ln in lines if ln.startswith("warning: ")] == [
        "warning: yolox_s.onnx and yolox_m.onnx have the same SHA-256; their rows measure one model"
    ]
