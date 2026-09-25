"""Acceptance tests for T-030 (engine hardening, and a people fixture without identifiable
faces). The task contract: do not edit.

Tests that need a model file skip only when the file is missing and
WEARREPORT_REQUIRE_MODEL is unset; CI sets it, so there a missing model fails. No face
detector is used anywhere: the fixture's privacy proxy is the height of the person boxes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import types
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wearreport import aggregate, benchmark, detect, publish
from wearreport._cv import cv2
from wearreport.tools import spotcheck

ROOT = Path(__file__).resolve().parents[3]
UNIT = ROOT / "engine" / "tests" / "unit"
FIXTURES = ROOT / "fixtures" / "detect"
PEOPLE = FIXTURES / "people_street.jpg"
OLD_PEOPLE = "people_" + "aldgate.jpg"  # split so that this file does not name it
FETCH_MODEL = ROOT / "scripts" / "fetch_model.sh"
REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
N_ANCHORS, ROW = detect.OUTPUT_SHAPE[1], detect.OUTPUT_SHAPE[2]
PERSON, CAR = 0, 2
T0 = datetime(2026, 7, 15, 12, 30, 5, tzinfo=UTC)


def _model(name: str = "yolox_s.onnx") -> Path:
    path = detect.model_path(name)
    if not path.is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run scripts/fetch_model.sh")
    return path


def _read(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None, path
    return np.asarray(frame)


def _unit_module(name: str) -> types.ModuleType:
    """A unit-test module loaded by path, to reuse its helpers."""
    spec = importlib.util.spec_from_file_location(f"t030_{name}", UNIT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = sys.modules.get(spec.name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses look their module up here
        spec.loader.exec_module(module)
    return module


class _Stub:
    def __init__(self, raw: np.ndarray) -> None:
        self.raw = raw

    def run(self, tensor: np.ndarray) -> Sequence[object]:
        return [self.raw[None].copy()]


def _raw_with_a_person() -> np.ndarray:
    raw = np.zeros((N_ANCHORS, ROW), dtype=np.float32)
    raw[100, 4] = 1.0  # objectness
    raw[100, 5 + PERSON] = 0.9
    return raw


def _detect(raw: np.ndarray) -> list[detect.Detection]:
    detector = detect.Detector.from_session(_Stub(raw), name="stub")
    return detector.detect(np.zeros((288, 352, 3), dtype=np.uint8))


# AC1: non-finite or out-of-range model output is an error --------------------------------


def _nan_in_a_non_target_class(raw: np.ndarray) -> None:
    raw[5000, 5 + CAR] = np.nan


def _inf_objectness(raw: np.ndarray) -> None:
    raw[5000, 4] = np.inf


def _minus_inf_regression(raw: np.ndarray) -> None:
    raw[5000, 2] = -np.inf


def _score_of_25(raw: np.ndarray) -> None:
    raw[5000, 5 + PERSON] = 25.0


def _negative_objectness(raw: np.ndarray) -> None:
    raw[5000, 4] = -0.01


@pytest.mark.parametrize(
    "corrupt",
    [
        _nan_in_a_non_target_class,
        _inf_objectness,
        _minus_inf_regression,
        _score_of_25,
        _negative_objectness,
    ],
    ids=lambda f: f.__name__.strip("_"),
)
def test_ac1_bad_model_output_raises(corrupt: Any) -> None:
    raw = _raw_with_a_person()
    assert len(_detect(raw)) == 1  # the uncorrupted output is fine
    corrupt(raw)
    with pytest.raises(detect.DetectorError):
        _detect(raw)


def test_ac1_scores_of_exactly_zero_and_one_are_valid() -> None:
    raw = _raw_with_a_person()
    raw[200, 4] = 1.0
    raw[200, 5 + CAR] = 1.0
    raw[300, 4] = 0.0
    assert [d.label for d in _detect(raw)] == ["person"]


def test_ac1_a_box_whose_coordinates_overflow_is_dropped() -> None:
    raw = _raw_with_a_person()
    raw[5000, :2] = 3e38  # finite, but the decoded centre overflows float32
    raw[5000, 4] = 1.0
    raw[5000, 5 + PERSON] = 0.95
    assert [d.score for d in _detect(raw)] == [pytest.approx(0.9)]
    doc = detect.Detector.detect.__doc__ or ""
    assert "overflow" in doc and "DetectorError" in doc


def test_ac1_real_model_gives_the_same_detections_as_postprocess() -> None:
    path = _model()
    session = detect.open_session(path)
    detector = detect.Detector(path)
    for fixture in (PEOPLE, FIXTURES / "umbrella_rain.jpg"):
        frame = _read(fixture)
        tensor, scale = detect.letterbox(frame)
        (out, *_) = session.run(tensor)
        raw = np.asarray(out, dtype=np.float32)[0]
        expected = detect.postprocess(
            raw, scale, frame.shape[1], frame.shape[0], detector.conf, detector.nms
        )
        assert expected
        assert detector.detect(frame) == expected


# AC2: the test-only constructor --------------------------------------------------------


def test_ac2_from_session_says_it_is_for_tests_and_skips_the_pin() -> None:
    doc = detect.Detector.from_session.__doc__ or ""
    assert re.search(r"tests only", doc, re.IGNORECASE), doc
    assert "SHA-256" in doc


def test_ac2_no_engine_module_but_detect_references_from_session() -> None:
    engine = ROOT / "engine" / "wearreport"
    users = [
        p.relative_to(ROOT).as_posix()
        for p in engine.rglob("*.py")
        if p.name != "detect.py" or p.parent != engine
        if "from_session" in p.read_text(encoding="utf-8")
    ]
    assert users == []
    source = (UNIT / "test_detect.py").read_text(encoding="utf-8")
    assert re.search(r"^def test_\w*from_session\w*\(", source, re.MULTILINE)


# AC3: the child-process privacy test runs without the telemetry variable -----------------


def test_ac3_child_environment_has_no_telemetry_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    privacy = _unit_module("test_privacy_runtime")
    seen: list[dict[str, str]] = []

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(args, 0, b"frames fetched: 50\n", b"")

    monkeypatch.setenv(detect.TELEMETRY_ENV, "1")
    monkeypatch.setattr(subprocess, "run", run)
    privacy._run_dry_sweep_process(tmp_path, "")
    assert len(seen) == 1
    assert detect.TELEMETRY_ENV not in seen[0]


def test_ac3_mutant_setting_the_variable_after_the_import_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    privacy = _unit_module("test_privacy_runtime")
    mutant = tmp_path / "mutant"
    shutil.copytree(
        ROOT / "engine" / "wearreport",
        mutant / "wearreport",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    source = (mutant / "wearreport" / "detect.py").read_text(encoding="utf-8")
    setting = 'os.environ[TELEMETRY_ENV] = "1"\n'
    importing = "import onnxruntime as ort"
    assert source.count(setting) == 1 and importing in source
    source = source.replace(setting, "")
    line_end = source.index("\n", source.index(importing)) + 1
    source = source[:line_end] + setting + source[line_end:]
    (mutant / "wearreport" / "detect.py").write_text(source, encoding="utf-8")
    prelude = "from wearreport import detect\nassert detect.__file__.startswith(MUTANT)\n"

    clean = tmp_path / "clean"
    clean.mkdir()
    real = "from wearreport import detect\nassert 'mutant' not in detect.__file__\n"
    privacy._run_dry_sweep_process(clean, real).assert_clean()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(mutant))
    report = privacy._run_dry_sweep_process(run_dir, f"MUTANT = {str(mutant)!r}\n" + prelude)
    with pytest.raises(AssertionError):
        report.assert_clean()


# AC4: the benchmark labels by digest ----------------------------------------------------


class _Blank:
    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return [np.zeros(detect.OUTPUT_SHAPE, dtype=np.float32)]


def _table_models(out: str) -> list[str]:
    rows = [ln for ln in out.splitlines() if ln.startswith("|")][2:]
    return [row.strip("|").split("|")[0].strip() for row in rows]


def _fake_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s_bytes: bytes, m_bytes: bytes
) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolox_s.onnx").write_bytes(s_bytes)
    (models / "yolox_m.onnx").write_bytes(m_bytes)
    pins = {
        "yolox_s.onnx": hashlib.sha256(b"small").hexdigest(),
        "yolox_m.onnx": hashlib.sha256(b"medium").hexdigest(),
    }
    monkeypatch.setattr(detect, "MODEL_SHA256", pins)
    monkeypatch.setattr(detect, "open_session", lambda path: _Blank())
    return models


def test_ac4_a_mislabelled_model_is_named_by_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    models = _fake_models(tmp_path, monkeypatch, b"small", b"small")
    assert benchmark.main(["--dry-run", "--cameras", "2", "--model-dir", str(models)]) == 0
    out = capsys.readouterr().out
    assert _table_models(out) == ["yolox_s", "yolox_m"]  # the file stems, as before
    lines = out.splitlines()
    assert "sha256: yolox_s.onnx is the pinned yolox_s.onnx" in lines
    assert "sha256: yolox_m.onnx is the pinned yolox_s.onnx, not yolox_m.onnx" in lines
    warnings = [ln for ln in lines if ln.startswith("warning: ")]
    assert len(warnings) == 1
    assert "yolox_s.onnx" in warnings[0] and "yolox_m.onnx" in warnings[0]
    assert "same SHA-256" in warnings[0]


def test_ac4_correctly_named_models_carry_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    models = _fake_models(tmp_path, monkeypatch, b"small", b"medium")
    assert benchmark.main(["--dry-run", "--cameras", "2", "--model-dir", str(models)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "sha256: yolox_s.onnx is the pinned yolox_s.onnx" in lines
    assert "sha256: yolox_m.onnx is the pinned yolox_m.onnx" in lines
    assert not [ln for ln in lines if ln.startswith("warning: ")]


def test_ac4_real_yolox_s_bytes_under_the_yolox_m_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    shutil.copyfile(_model(), models / "yolox_s.onnx")
    shutil.copyfile(_model(), models / "yolox_m.onnx")
    assert benchmark.main(["--dry-run", "--cameras", "2", "--model-dir", str(models)]) == 0
    out = capsys.readouterr().out
    assert _table_models(out) == ["yolox_s", "yolox_m"]
    assert "sha256: yolox_m.onnx is the pinned yolox_s.onnx, not yolox_m.onnx" in out
    assert re.search(r"^warning: .*same SHA-256", out, re.MULTILINE)


# AC5: no stale part files ---------------------------------------------------------------


def _fetch(tmp_path: Path, dest: Path, source: Path) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        'echo called >> "$CURL_LOG"\n'
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        f"cat '{source}' > \"$out\"\n"
    )
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["CURL_LOG"] = str(tmp_path / "curl.log")
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


def test_ac5_stale_part_files_are_removed_before_a_download(tmp_path: Path) -> None:
    dest = tmp_path / "models"
    dest.mkdir()
    for name in (".yolox_s.onnx.Ab12Cd", ".yolox_s.onnx.zzzzzz"):
        (dest / name).write_bytes(b"half a model")
    proc = _fetch(tmp_path, dest, _model())
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in dest.iterdir()) == ["yolox_s.onnx"]
    assert (tmp_path / "curl.log").exists()


def test_ac5_stale_part_files_are_removed_when_the_model_is_present(tmp_path: Path) -> None:
    dest = tmp_path / "models"
    dest.mkdir()
    shutil.copyfile(_model(), dest / "yolox_s.onnx")
    (dest / ".yolox_s.onnx.Q1w2E3").write_bytes(b"half a model")
    proc = _fetch(tmp_path, dest, _model())
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in dest.iterdir()) == ["yolox_s.onnx"]
    assert not (tmp_path / "curl.log").exists()  # verified: nothing downloaded


# AC6: quiet default logger ----------------------------------------------------------------


def test_ac6_default_logger_severity_is_set_before_the_first_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []

    class Input:
        name = "images"
        shape = (1, 3, 640, 640)

    class FakeSession:
        def __init__(self, data: bytes, sess_options: Any, providers: list[str]) -> None:
            calls.append(("session", None))

        def get_inputs(self) -> list[Input]:
            return [Input()]

    model = tmp_path / "yolox_s.onnx"
    model.write_bytes(b"m")
    monkeypatch.setitem(detect.MODEL_SHA256, "test", hashlib.sha256(b"m").hexdigest())
    monkeypatch.setattr(
        "wearreport.detect.ort.set_default_logger_severity",
        lambda level: calls.append(("severity", level)),
    )
    monkeypatch.setattr("wearreport.detect.ort.InferenceSession", FakeSession)
    detect.Detector(model)
    assert calls[0] == ("severity", 3)
    assert calls.index(("session", None)) > 0


# AC7: publish checks directories before it cleans --------------------------------------


def _record() -> Any:
    observations = [aggregate.Observation(f"C{i:04d}", None) for i in range(10)]
    return aggregate.build_record(
        observations,
        started_at=T0,
        finished_at=T0,
        weather=None,
        engine_version="0.0.0",
        model_name="yolox_m",
        model_sha256="b" * 64,
    )


@pytest.mark.parametrize("level", [1, 2, 3, 4], ids=["sweeps", "year", "month", "day"])
def test_ac7_no_cleaning_through_a_symlinked_directory(tmp_path: Path, level: int) -> None:
    data, outside = tmp_path / "data", tmp_path / "outside"
    data.mkdir()
    outside.mkdir()
    record = _record()
    day = publish.record_path(data, record["sweep_id"]).parent
    parts = day.relative_to(data).parts  # sweeps, YYYY, MM, DD
    link = data.joinpath(*parts[:level])
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    target = outside.joinpath(*parts[level:])
    target.mkdir(parents=True, exist_ok=True)
    temp = target / f".{record['sweep_id']}.json.0123456789abcdef.tmp"
    temp.write_bytes(b"not ours")
    with pytest.raises(publish.PublishError):
        publish.publish(data, record, now=T0)
    assert temp.read_bytes() == b"not ours"
    assert not (data / "status.json").exists()


# AC8: the people fixture -----------------------------------------------------------------

ALLOWED_LICENCE = re.compile(r"(CC0 1\.0|Public domain|CC BY \d\.\d)", re.IGNORECASE)


def _licence_rows() -> dict[str, tuple[str, str]]:
    text = (ROOT / "fixtures" / "LICENSES.md").read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")][2:]
    found = {}
    for row in rows:
        source, licence, file = (c.strip() for c in row.strip().strip("|").split("|"))
        found[file] = (source, licence)
    return found


def test_ac8_the_old_fixture_is_gone_and_named_nowhere() -> None:
    assert not (FIXTURES / OLD_PEOPLE).exists()
    git = shutil.which("git")
    assert git is not None
    found = subprocess.run(
        [git, "grep", "-l", "-F", OLD_PEOPLE, "--", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.stdout.split() == []


def test_ac8_the_new_fixture_is_small_licensed_and_recorded() -> None:
    assert PEOPLE.is_file()
    assert PEOPLE.stat().st_size <= 300 * 1024
    source, licence = _licence_rows()[PEOPLE.relative_to(ROOT).as_posix()]
    assert re.search(r"https://\S+", source) and len(source.split()) >= 3, source
    assert ALLOWED_LICENCE.fullmatch(licence), licence
    assert not re.search(r"\b(SA|NC|ND)\b", licence.upper()), licence
    frame = _read(PEOPLE)
    assert frame.shape[1] <= 1024


def test_ac8_people_are_small_in_the_frame() -> None:
    detector = detect.Detector(_model())
    frame = _read(PEOPLE)
    persons = [d for d in detector.detect(frame) if d.label == "person"]
    assert len(persons) >= 2
    tallest = max(d.box[3] - d.box[1] for d in persons)
    assert tallest <= 0.2 * frame.shape[0], tallest / frame.shape[0]


def test_ac8_the_fixture_keeps_the_older_detector_tests_meaningful() -> None:
    frame = _read(PEOPLE)
    default = detect.Detector(_model())
    counts = [len(detect.Detector(_model(), conf=c).detect(frame)) for c in (0.9, 0.35, 0.1)]
    assert counts[0] < counts[1] < counts[2], counts
    crop = frame[frame.shape[0] // 4 : 3 * frame.shape[0] // 4, 100:-100]
    assert [d for d in default.detect(crop) if d.label == "person"]


# AC9: the write audit sees rename targets ---------------------------------------------


def test_ac9_moving_a_file_out_of_the_review_directory_is_caught(tmp_path: Path) -> None:
    unit = _unit_module("test_spotcheck")
    assert "args[1]" in unit.CONFINED
    events, stats_file = unit._confined_run(tmp_path, "move_out", "crops")
    moved = [Path(p) for _kind, p in events if Path(p).name == "moved.png"]
    assert moved and not moved[0].is_relative_to(tmp_path / "tmp")
    with pytest.raises(AssertionError, match=r"moved\.png"):
        unit._assert_writes_confined(events, stats_file)


# AC10: the stale-directory sweep cannot be cut short ------------------------------------

SWEEP_CHILD = r"""
import os, shutil, signal, sys, time
import numpy as np
from wearreport.tools import spotcheck as sc

real_rmtree = shutil.rmtree
def rmtree(path, *args, **kwargs):
    if "stale" in os.path.basename(str(path)):
        os.kill(os.getpid(), getattr(signal, sys.argv[1]))
        time.sleep(0.2)  # the handler runs here unless the removal is a critical step
    return real_rmtree(path, *args, **kwargs)
shutil.rmtree = rmtree

class Untouched:
    def detect(self, frame):
        raise AssertionError("the sweep of stale directories should have stopped the run")

pipeline = sc.Pipeline(frames=lambda: [np.zeros((8, 8, 3), np.uint8)], detector=Untouched(),
                       info=sc.DetectorInfo(model="stub", sha256="0" * 64, conf=0.35))
sys.exit(sc.main(sys.argv[2:], pipeline=pipeline, reviewer=None))
"""


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGINT", "SIGHUP"])
def test_ac10_a_signal_during_the_stale_sweep_waits_for_the_removal(
    tmp_path: Path, signame: str
) -> None:
    tmp = tmp_path / "tmp"
    stale = tmp / (spotcheck.TEMP_PREFIX + "stale")
    (stale / "sub").mkdir(parents=True)
    for i in range(20):
        (stale / "sub" / f"crop-{i}.png").write_bytes(b"x")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    env = {k: v for k, v in os.environ.items() if k not in spotcheck.CI_VARIABLES}
    env.update(TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp), PYTHONDONTWRITEBYTECODE="1")
    args = ["--n", "1", "--min-persons", "1", "--reviewer", "tester", "--timeout", "60"]
    args += ["--out-dir", str(tmp_path / "out"), "--judgements", str(tmp_path / "j.json")]
    proc = subprocess.run(
        [sys.executable, "-c", SWEEP_CHILD, signame, *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = (proc.stdout + proc.stderr).decode(errors="replace")
    signum = int(getattr(signal, signame))
    assert proc.returncode == 128 + signum, output
    assert signame in output
    assert not os.path.lexists(stale), output
    assert list(tmp.iterdir()) == []
    assert not (tmp_path / "out").exists()
