"""Acceptance tests for T-008 (live spot-check tool, statistics only). The task contract:
do not edit.

Every frame in these tests is synthetic (uniform colour or random noise) or one of the
licensed fixtures under fixtures/detect/. No test opens a rendered image to look at it:
rendered files are checked through counts, sizes and magic bytes only.

Tests that need a model file skip only when the file is missing and
WEARREPORT_REQUIRE_MODEL is unset; CI sets it, so there a missing model fails.
"""

from __future__ import annotations

import collections.abc
import datetime
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import typing
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import privacy_guard
from wearreport import detect, fetch, registry
from wearreport._cv import cv2
from wearreport.testing.fake_cameras import FakeCameraServer
from wearreport.tools import spotcheck

ROOT = Path(__file__).resolve().parents[3]
TOOL = "engine/wearreport/tools/spotcheck.py"
FIXTURES = ROOT / "fixtures" / "detect"
REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
H, W = 288, 352
DAY = datetime.date(2026, 9, 25)
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
WAIT_S = 60
FIELDS = {
    "date",
    "reviewer",
    "mode",
    "frames_reviewed",
    "boxes_shown",
    "boxes_not_person",
    "boxes_in_vehicle",
    "persons_missed",
    "precision_person",
    "precision_pedestrian",
    "recall_estimate",
    "detector",
}
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")


def _model(name: str = "yolox_s.onnx") -> Path:
    path = detect.model_path(name)
    if not path.is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run scripts/fetch_model.sh")
    return path


def _box(k: int) -> tuple[float, float, float, float]:
    """The k-th stub person box: 20x60 pixels, side by side along the frame."""
    return (10.0 + 30 * k, 50.0, 30.0 + 30 * k, 110.0)


class StubDetector:
    """Finds counts[i] people (and one umbrella) in the frame whose colour is 10*i."""

    conf = detect.DEFAULT_CONF

    def __init__(self, counts: Sequence[int]) -> None:
        self.counts = list(counts)

    def detect(self, frame: npt.NDArray[np.uint8]) -> list[detect.Detection]:
        i = int(frame[0, 0, 0]) // 10
        found = [detect.Detection("person", 0.9, _box(k)) for k in range(self.counts[i])]
        return [*found, detect.Detection("umbrella", 0.8, (300.0, 200.0, 340.0, 240.0))]


def _frames(n: int) -> list[npt.NDArray[np.uint8]]:
    return [np.full((H, W, 3), 10 * i, dtype=np.uint8) for i in range(n)]


def _pipeline(counts: Sequence[int]) -> spotcheck.Pipeline:
    frames = _frames(len(counts))
    return spotcheck.Pipeline(frames=lambda: frames, detector=StubDetector(counts), info=INFO)


Decide = Callable[[spotcheck.ReviewItem, str], spotcheck.Judgement]


def _all_pedestrians(item: spotcheck.ReviewItem, mode: str) -> spotcheck.Judgement:
    return spotcheck.Judgement(frozenset(), frozenset(), 0 if mode == "frames" else None)


class Scripted:
    """A reviewer that answers from a script and records what it was shown."""

    def __init__(self, decide: Decide = _all_pedestrians) -> None:
        self.decide = decide
        self.items: list[spotcheck.ReviewItem] = []
        self.workdir: Path | None = None
        self.listing: list[str] = []
        self.dir_mode = 0
        self.on_judge: Callable[[], None] | None = None

    def judge(
        self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
    ) -> dict[int, spotcheck.Judgement]:
        self.items = list(items)
        dirs = list(Path(tempfile.gettempdir()).glob(spotcheck.TEMP_PREFIX + "*"))
        if dirs:
            (self.workdir,) = dirs
            self.listing = sorted(p.name for p in self.workdir.iterdir())
            self.dir_mode = stat.S_IMODE(self.workdir.stat().st_mode)
        if self.on_judge:
            self.on_judge()
        return {item.number: self.decide(item, mode) for item in items}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Outside CI, in a fresh working directory, HOME and TMPDIR; yields the TMPDIR."""
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    work, home, tmp = tmp_path / "work", tmp_path / "home", tmp_path / "tmp"
    for d in (work, home, tmp):
        d.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    yield tmp


def _run(
    args: Sequence[str],
    out_dir: Path,
    *,
    pipeline: spotcheck.Pipeline | None = None,
    reviewer: Any = None,
) -> int:
    argv = [*args, "--reviewer", "tester", "--out-dir", str(out_dir)]
    return spotcheck.main(argv, pipeline=pipeline, reviewer=reviewer, today=DAY)


def _stats(out_dir: Path, name: str = "2026-09-25.json") -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out_dir / name).read_text(encoding="utf-8"))
    return data


def _leftovers(tmp: Path) -> list[Path]:
    return sorted(tmp.glob(spotcheck.TEMP_PREFIX + "*"))


def _is_image(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(16).startswith(MAGIC)


# AC1: sampling ------------------------------------------------------------------------


def test_ac1_command_line_defaults() -> None:
    args = spotcheck.build_parser().parse_args(["--n", "20"])
    assert args.n == 20
    assert args.mode == "crops"
    assert args.min_persons == 3
    assert args.seed is None
    assert args.model == "yolox_m.onnx" == spotcheck.DEFAULT_MODEL
    assert args.timeout == 30 * 60
    assert Path(args.out_dir) == Path("spotchecks")
    assert isinstance(args.reviewer, str) and args.reviewer
    parsed = spotcheck.build_parser().parse_args(
        ["--n", "5", "--mode", "frames", "--min-persons", "1", "--seed", "4"]
    )
    assert (parsed.mode, parsed.min_persons, parsed.seed) == ("frames", 1, 4)


@pytest.mark.parametrize(
    "bad",
    [["--n", "0"], ["--n", "x"], ["--n", "5", "--mode", "video"], ["--n", "5", "--timeout", "0"]],
)
def test_ac1_rejects_bad_arguments(bad: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        spotcheck.build_parser().parse_args(bad)
    assert exc.value.code != 0


def _marked(i: int) -> npt.NDArray[np.uint8]:
    return np.full((H, W, 3), 10 * i, dtype=np.uint8)


def test_ac1_samples_frames_with_enough_persons_at_random() -> None:
    counts = [0, 1, 2, 3, 4, 5, 3, 3, 0, 6]
    detector = StubDetector(counts)
    frames = [_marked(i) for i in range(len(counts))]
    qualifying = {i for i, c in enumerate(counts) if c >= 3}

    def picked(seed: int, n: int = 3) -> list[int]:
        samples = spotcheck.sample(frames, detector, n=n, min_persons=3, seed=seed)
        for s in samples:
            assert all(d.label == "person" for d in s.persons)
            assert len(s.persons) == counts[int(s.frame[0, 0, 0]) // 10]
        return [int(s.frame[0, 0, 0]) // 10 for s in samples]

    first = picked(11)
    assert len(first) == 3 and len(set(first)) == 3 and set(first) <= qualifying
    assert picked(11) == first  # seeded: reproducible
    assert len({tuple(picked(seed)) for seed in range(20)}) > 1  # random
    assert set(picked(1, n=50)) == qualifying  # up to N
    assert spotcheck.sample(frames, detector, n=5, min_persons=7, seed=0) == []


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] == "127.0.0.1"):
            raise AssertionError(f"non-local connection attempted: {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


def test_ac1_live_path_uses_registry_fetch_and_detector_defaults(
    env: Path, offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _model()
    bodies = [(FIXTURES / name).read_bytes() for name in ("people_street.jpg", "umbrella_rain.jpg")]
    calls: dict[str, list[Any]] = {"list": [], "fetch": [], "detector": []}
    real_fetch_sweep = fetch.fetch_sweep
    real_detector = detect.Detector

    class SpyDetector(real_detector):  # type: ignore[valid-type,misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["detector"].append((args, kwargs))
            super().__init__(*args, **kwargs)

    with FakeCameraServer() as server:
        cameras = server.cameras(6)
        for i, cam in enumerate(cameras[:4]):
            server.serve_body(cam.id, bodies[i % 2])

        def list_cameras(*args: Any, **kwargs: Any) -> list[registry.Camera]:
            calls["list"].append(args)
            return cameras

        def fetch_sweep(cams: Sequence[registry.Camera], **kwargs: Any) -> Any:
            calls["fetch"].append(list(cams))
            return real_fetch_sweep(cams, **kwargs)

        monkeypatch.setattr(registry, "list_cameras", list_cameras)
        monkeypatch.setattr(fetch, "fetch_sweep", fetch_sweep)
        monkeypatch.setattr(detect, "Detector", SpyDetector)
        reviewer = Scripted()
        out = tmp_path / "out"
        args = ["--n", "2", "--min-persons", "1", "--seed", "3", "--model", "yolox_s.onnx"]
        assert _run(args, out, reviewer=reviewer) == 0

    assert len(calls["list"]) == 1
    assert calls["fetch"] == [cameras]
    ((det_args, det_kwargs),) = calls["detector"]
    assert Path(det_args[0]) == detect.model_path("yolox_s.onnx")
    assert "conf" not in det_kwargs and "nms" not in det_kwargs
    stats = _stats(out)
    assert stats["frames_reviewed"] == 2
    assert stats["detector"] == {
        "model": "yolox_s",
        "sha256": detect.MODEL_SHA256["yolox_s.onnx"],
        "conf": detect.DEFAULT_CONF,
    }
    # AC2: rendered file names and the statistics carry no camera id.
    assert reviewer.listing
    text = " ".join(reviewer.listing) + json.dumps(stats)
    for cam in cameras:
        assert cam.id not in text
        assert cam.id.split("_")[-1] not in text


# AC2: rendering -----------------------------------------------------------------------


def test_ac2_workdir_is_a_private_mkdtemp_directory(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prefixes: list[object] = []
    real_mkdtemp = tempfile.mkdtemp

    def mkdtemp(*args: Any, **kwargs: Any) -> str:
        prefixes.append(kwargs.get("prefix"))
        return str(real_mkdtemp(*args, **kwargs))

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    reviewer = Scripted()
    assert (
        _run(
            ["--n", "2", "--min-persons", "1"],
            tmp_path / "o",
            pipeline=_pipeline([2, 3]),
            reviewer=reviewer,
        )
        == 0
    )
    assert prefixes == ["wearreport-spotcheck-"] == [spotcheck.TEMP_PREFIX]
    assert reviewer.workdir is not None
    assert reviewer.workdir.parent == env
    assert reviewer.workdir.name.startswith("wearreport-spotcheck-")
    assert reviewer.dir_mode == 0o700
    assert not reviewer.workdir.exists()


def test_ac2_crops_mode_one_numbered_crop_per_detection(env: Path, tmp_path: Path) -> None:
    reviewer = Scripted()

    def check_files() -> None:
        assert reviewer.workdir is not None
        images = sorted(p.name for p in reviewer.workdir.iterdir() if _is_image(p))
        assert images == sorted(item.file for item in reviewer.items)
        for item in reviewer.items:
            (number,) = item.boxes
            assert number == item.number
            crop = cv2.imread(str(reviewer.workdir / item.file), cv2.IMREAD_COLOR)
            assert crop is not None
            k = [0, 1, 0, 1, 2][number - 1]  # counts [2, 3]: boxes 1-2 in frame 0, 3-5 in 1
            x1, y1, x2, y2 = spotcheck.crop_bounds(_box(k), W, H)
            height, width = crop.shape[:2]
            assert height * (x2 - x1) == width * (y2 - y1)  # the crop's shape, maybe scaled
            assert height >= y2 - y1

    reviewer.on_judge = check_files
    args = ["--n", "2", "--min-persons", "1", "--mode", "crops"]
    assert _run(args, tmp_path / "o", pipeline=_pipeline([2, 3]), reviewer=reviewer) == 0
    assert [item.number for item in reviewer.items] == [1, 2, 3, 4, 5]


def test_ac2_crop_bounds_have_a_50_percent_margin_clipped_to_the_frame() -> None:
    assert spotcheck.crop_bounds((100.0, 100.0, 120.0, 140.0), 352, 288) == (90, 80, 130, 160)
    assert spotcheck.crop_bounds((0.0, 0.0, 10.0, 20.0), 352, 288) == (0, 0, 15, 30)
    assert spotcheck.crop_bounds((340.0, 270.0, 352.0, 288.0), 352, 288) == (334, 261, 352, 288)
    assert spotcheck.crop_bounds((10.5, 20.5, 11.5, 22.5), 352, 288) == (10, 19, 12, 24)


def test_ac2_frames_mode_one_image_per_frame_with_numbered_boxes(env: Path, tmp_path: Path) -> None:
    reviewer = Scripted()

    def check_files() -> None:
        assert reviewer.workdir is not None
        images = sorted(p.name for p in reviewer.workdir.iterdir() if _is_image(p))
        assert images == sorted(item.file for item in reviewer.items)
        for item in reviewer.items:
            frame = cv2.imread(str(reviewer.workdir / item.file), cv2.IMREAD_COLOR)
            assert frame is not None
            height, width = frame.shape[:2]
            assert height * W == width * H and height >= H

    reviewer.on_judge = check_files
    args = ["--n", "3", "--min-persons", "1", "--mode", "frames"]
    assert _run(args, tmp_path / "o", pipeline=_pipeline([2, 3, 1]), reviewer=reviewer) == 0
    assert len(reviewer.items) == 3
    numbers = [n for item in reviewer.items for n in item.boxes]
    assert numbers == list(range(1, 7))
    assert sorted(len(item.boxes) for item in reviewer.items) == [1, 2, 3]


# AC3: cleanup (child processes) --------------------------------------------------------

CHILD = r"""
import json, sys
import numpy as np
from wearreport import detect
from wearreport.tools import spotcheck

H, W = 288, 352
counts = json.loads(sys.argv[1])
scenario = sys.argv[2]

class Stub:
    conf = detect.DEFAULT_CONF
    def detect(self, frame):
        n = counts[int(frame[0, 0, 0]) // 10]
        return [detect.Detection("person", 0.9, (10.0 + 30 * k, 50.0, 30.0 + 30 * k, 110.0))
                for k in range(n)]

frames = [np.full((H, W, 3), 10 * i, dtype=np.uint8) for i in range(len(counts))]
pipeline = spotcheck.Pipeline(
    frames=lambda: frames,
    detector=Stub(),
    info=spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF),
)

class Ok:
    def judge(self, items, mode, deadline):
        return {i.number: spotcheck.Judgement(frozenset(), frozenset(),
                                              0 if mode == "frames" else None) for i in items}

class Boom:
    def judge(self, items, mode, deadline):
        raise RuntimeError("reviewer failed")

reviewer = {"ok": Ok(), "raise": Boom()}.get(scenario)
sys.exit(spotcheck.main(sys.argv[3:], pipeline=pipeline, reviewer=reviewer))
"""


def _child_env(tmp: Path) -> dict[str, str]:
    environ = {k: v for k, v in os.environ.items() if k not in spotcheck.CI_VARIABLES}
    environ.update(TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp), PYTHONDONTWRITEBYTECODE="1")
    return environ


def _spawn(
    tmp_path: Path, scenario: str, args: Sequence[str], counts: Sequence[int] = (2, 3)
) -> tuple[subprocess.Popen[bytes], Path]:
    tmp = tmp_path / "tmp"
    tmp.mkdir(exist_ok=True)
    argv = [*args, "--reviewer", "tester", "--out-dir", str(tmp_path / "out")]
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD, json.dumps(list(counts)), scenario, *argv],
        cwd=tmp_path,
        env=_child_env(tmp),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc, tmp


def _wait_for_review(tmp: Path, proc: subprocess.Popen[bytes]) -> Path:
    """The tool's directory, once it holds the rendered images and the numbering."""
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        for d in _leftovers(tmp):
            if (d / spotcheck.NUMBERING_FILE).is_file():
                time.sleep(0.2)  # the numbering is written after the images
                return d
        assert proc.poll() is None, proc.communicate()
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("the review directory never appeared")


def _finish(proc: subprocess.Popen[bytes]) -> tuple[int, str]:
    out, err = proc.communicate(timeout=WAIT_S)
    return proc.returncode, (out + err).decode(errors="replace")


def test_ac3_normal_run_deletes_the_directory(tmp_path: Path) -> None:
    proc, tmp = _spawn(tmp_path, "ok", ["--n", "2", "--min-persons", "1"])
    code, output = _finish(proc)
    assert code == 0, output
    assert _leftovers(tmp) == []
    assert (tmp_path / "out" / f"{datetime.date.today().isoformat()}.json").is_file()


def test_ac3_exception_deletes_the_directory(tmp_path: Path) -> None:
    proc, tmp = _spawn(tmp_path, "raise", ["--n", "2", "--min-persons", "1"])
    code, output = _finish(proc)
    assert code != 0
    assert "reviewer failed" in output
    assert _leftovers(tmp) == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=lambda s: s.name)
def test_ac3_signal_deletes_the_directory(tmp_path: Path, signum: signal.Signals) -> None:
    judgements = tmp_path / "judgements.json"
    args = ["--n", "2", "--min-persons", "1", "--judgements", str(judgements)]
    proc, tmp = _spawn(tmp_path, "json", args)
    workdir = _wait_for_review(tmp, proc)
    assert sum(_is_image(p) for p in workdir.iterdir()) == 5  # not vacuous
    proc.send_signal(signum)
    code, output = _finish(proc)
    assert code != 0, output
    assert not workdir.exists()
    assert _leftovers(tmp) == []
    assert not (tmp_path / "out").exists()


def test_ac3_review_timeout_deletes_the_directory(tmp_path: Path) -> None:
    judgements = tmp_path / "judgements.json"
    args = ["--n", "2", "--min-persons", "1", "--judgements", str(judgements), "--timeout", "2"]
    started = time.monotonic()
    proc, tmp = _spawn(tmp_path, "json", args)
    workdir = _wait_for_review(tmp, proc)
    code, output = _finish(proc)
    assert code != 0, output
    assert "timeout" in output.lower() or "timed out" in output.lower()
    assert not workdir.exists()
    assert _leftovers(tmp) == []
    assert time.monotonic() - started < WAIT_S


def test_ac3_start_removes_stale_directories_older_than_the_timeout(tmp_path: Path) -> None:
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    stale, fresh = tmp / (spotcheck.TEMP_PREFIX + "stale"), tmp / (spotcheck.TEMP_PREFIX + "fresh")
    other = tmp / "unrelated-old"
    for d in (stale, fresh, other):
        d.mkdir(mode=0o700)
        (d / "leftover.bin").write_bytes(b"\x00" * 8)
    old = time.time() - 2 * 3600
    for d in (stale, other):
        os.utime(d / "leftover.bin", (old, old))
        os.utime(d, (old, old))
    proc, _ = _spawn(tmp_path, "ok", ["--n", "1", "--min-persons", "1", "--timeout", "3600"])
    code, output = _finish(proc)
    assert code == 0, output
    assert not stale.exists()
    assert fresh.is_dir() and other.is_dir()


def test_ac3_readme_documents_sigkill() -> None:
    readme = (ROOT / "spotchecks" / "README.md").read_text(encoding="utf-8")
    assert "SIGKILL" in readme
    assert spotcheck.TEMP_PREFIX in readme


# AC4: CI refusal ----------------------------------------------------------------------


@pytest.mark.parametrize("var", ["CI", "GITHUB_ACTIONS"])
@pytest.mark.parametrize("value", ["true", "1", "false", " "])
def test_ac4_refuses_to_run_in_ci_before_any_network_access(
    env: Path, monkeypatch: pytest.MonkeyPatch, var: str, value: str, tmp_path: Path
) -> None:
    attempts: list[object] = []

    def connect(self: socket.socket, address: Any) -> None:
        attempts.append(address)
        raise AssertionError("network access")

    def list_cameras(*args: Any, **kwargs: Any) -> list[registry.Camera]:
        attempts.append("list_cameras")
        raise AssertionError("registry access")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(registry, "list_cameras", list_cameras)
    monkeypatch.setenv(var, value)
    assert _run(["--n", "3"], tmp_path / "o") != 0
    assert _run(["--n", "3"], tmp_path / "o", pipeline=_pipeline([3]), reviewer=Scripted()) != 0
    assert attempts == []
    assert list(env.iterdir()) == []
    assert not (tmp_path / "o").exists()


def test_ac4_empty_values_are_not_ci(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CI", "")
    monkeypatch.setenv("GITHUB_ACTIONS", "")
    assert (
        _run(
            ["--n", "1", "--min-persons", "1"],
            tmp_path / "o",
            pipeline=_pipeline([3]),
            reviewer=Scripted(),
        )
        == 0
    )


def test_ac4_command_refuses_in_ci(tmp_path: Path) -> None:
    environ = _child_env(tmp_path)
    environ["CI"] = "true"
    proc = subprocess.run(
        [sys.executable, "-m", "wearreport.tools.spotcheck", "--n", "3"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        timeout=WAIT_S,
        check=False,
    )
    assert proc.returncode != 0
    assert b"CI" in proc.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == []


# AC5: judgements ----------------------------------------------------------------------


def test_ac5_prints_the_directory_and_the_numbering_while_it_exists(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reviewer = Scripted()
    seen: list[str] = []
    reviewer.on_judge = lambda: seen.append(capsys.readouterr().out)
    args = ["--n", "2", "--min-persons", "1", "--mode", "frames"]
    assert _run(args, tmp_path / "o", pipeline=_pipeline([2, 3]), reviewer=reviewer) == 0
    (printed,) = seen
    assert reviewer.workdir is not None and str(reviewer.workdir) in printed
    lines = printed.splitlines()
    for item in reviewer.items:
        (line,) = [ln for ln in lines if item.file in ln]
        assert set(re.findall(r"\d+", line.replace(item.file, ""))) >= {str(n) for n in item.boxes}


def _items(mode: spotcheck.Mode) -> list[spotcheck.ReviewItem]:
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    if mode == "frames":
        return [
            spotcheck.ReviewItem(number=1, file="frame-001.png", boxes=(1, 2, 3), image=blank),
            spotcheck.ReviewItem(number=2, file="frame-002.png", boxes=(4,), image=blank),
        ]
    return [
        spotcheck.ReviewItem(number=n, file=f"crop-{n:04d}.png", boxes=(n,), image=blank)
        for n in (1, 2)
    ]


def test_ac5_parses_valid_judgements() -> None:
    frames = {"1": {"not_person": [2], "in_vehicle": [3], "missed": 1}, "2": {"missed": 0}}
    got = spotcheck.parse_judgements(json.dumps(frames).encode(), _items("frames"), "frames")
    assert got == {
        1: spotcheck.Judgement(frozenset({2}), frozenset({3}), 1),
        2: spotcheck.Judgement(frozenset(), frozenset(), 0),
    }
    crops = {"1": {"not_person": [1]}, "2": {"in_vehicle": [2]}}
    got = spotcheck.parse_judgements(json.dumps(crops).encode(), _items("crops"), "crops")
    assert got == {
        1: spotcheck.Judgement(frozenset({1}), frozenset(), None),
        2: spotcheck.Judgement(frozenset(), frozenset({2}), None),
    }


@pytest.mark.parametrize(
    ("mode", "raw"),
    [
        ("frames", b"not json"),
        ("frames", b"[]"),
        ("frames", b'{"1": {"missed": 0}}'),  # image 2 missing
        ("frames", b'{"1": {"missed": 0}, "2": {"missed": 0}, "3": {"missed": 0}}'),
        ("frames", b'{"1": {"not_person": [4], "missed": 0}, "2": {"missed": 0}}'),  # box 4 is in 2
        (
            "frames",
            b'{"1": {"not_person": [2], "in_vehicle": [2], "missed": 0}, "2": {"missed": 0}}',
        ),
        ("frames", b'{"1": {}, "2": {"missed": 0}}'),  # missed is required in frames mode
        ("frames", b'{"1": {"missed": -1}, "2": {"missed": 0}}'),
        ("frames", b'{"1": {"missed": 1.5}, "2": {"missed": 0}}'),
        ("frames", b'{"1": {"missed": true}, "2": {"missed": 0}}'),
        ("frames", b'{"1": {"missed": "2"}, "2": {"missed": 0}}'),
        ("frames", b'{"1": {"missed": NaN}, "2": {"missed": 0}}'),
        ("frames", b'{"1": {"missed": 0, "extra": 1}, "2": {"missed": 0}}'),
        ("frames", b'{"1": {"not_person": "2", "missed": 0}, "2": {"missed": 0}}'),
        ("frames", b'{"1": {"not_person": [true], "missed": 0}, "2": {"missed": 0}}'),
        ("frames", b"\xff\xfe"),
        ("frames", b"[" * 100_000 + b"]" * 100_000),
        ("frames", b'{"1": {"missed": 1' + b"0" * 5000 + b'}, "2": {"missed": 0}}'),
        ("crops", b'{"1": {"missed": 0}, "2": {}}'),  # no misses in crops mode
        ("crops", b'{"1": {"not_person": [2]}, "2": {}}'),
    ],
)
def test_ac5_rejects_invalid_judgements(mode: spotcheck.Mode, raw: bytes) -> None:
    with pytest.raises(spotcheck.JudgementError):
        spotcheck.parse_judgements(raw, _items(mode), mode)


def test_ac5_keyboard_lines() -> None:
    frame, crop = _items("frames")[0], _items("crops")[1]
    assert spotcheck.parse_line("", frame, "frames") == spotcheck.Judgement(
        frozenset(), frozenset(), 0
    )
    assert spotcheck.parse_line("n1 v3 m2", frame, "frames") == spotcheck.Judgement(
        frozenset({1}), frozenset({3}), 2
    )
    assert spotcheck.parse_line("n2", crop, "crops") == spotcheck.Judgement(
        frozenset({2}), frozenset(), None
    )
    for bad in ("n9", "n1 v1", "m-1", "x", "n", "n1 n1"):
        with pytest.raises(spotcheck.JudgementError):
            spotcheck.parse_line(bad, frame, "frames")
    for bad in ("m1", "n1", "v3", "p2"):
        with pytest.raises(spotcheck.JudgementError):
            spotcheck.parse_line(bad, crop, "crops")


def _judge_by_file(
    tmp: Path, path: Path, first: bytes | None, decide: Callable[[dict[str, Any]], dict[str, Any]]
) -> threading.Thread:
    """Write judgements for the tool's numbering once it appears; an invalid file first."""

    def run() -> None:
        deadline = time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            for d in _leftovers(tmp):
                numbering = d / spotcheck.NUMBERING_FILE
                if numbering.is_file():
                    time.sleep(0.2)
                    data = json.loads(numbering.read_text(encoding="utf-8"))
                    if first is not None:
                        path.write_bytes(first)
                        time.sleep(4 * spotcheck.JSON_POLL_S)
                        path.unlink()
                    tmp_file = path.with_suffix(".partial")
                    tmp_file.write_text(json.dumps(decide(data)), encoding="utf-8")
                    tmp_file.replace(path)
                    return
            time.sleep(0.05)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _frames_judgements(numbering: dict[str, Any]) -> dict[str, Any]:
    assert numbering["mode"] == "frames"
    out: dict[str, Any] = {}
    for image in numbering["images"]:
        boxes = image["boxes"]
        out[str(image["image"])] = {"not_person": boxes[:1], "missed": 1}
    return out


def test_ac5_json_file_invalid_then_valid(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    judgements = tmp_path / "reviewer" / "judgements.json"
    judgements.parent.mkdir()
    thread = _judge_by_file(env, judgements, b'{"1": {"missed": "lots"}}', _frames_judgements)
    args = ["--n", "2", "--min-persons", "1", "--mode", "frames", "--judgements", str(judgements)]
    assert _run(args, tmp_path / "o", pipeline=_pipeline([2, 3])) == 0
    thread.join()
    assert "rejected" in capsys.readouterr().out.lower()
    stats = _stats(tmp_path / "o")
    assert (stats["boxes_shown"], stats["boxes_not_person"], stats["persons_missed"]) == (5, 2, 2)


def test_ac5_existing_judgements_file_is_refused(env: Path, tmp_path: Path) -> None:
    judgements = tmp_path / "old.json"
    judgements.write_text("{}", encoding="utf-8")
    args = ["--n", "1", "--min-persons", "1", "--judgements", str(judgements)]
    assert _run(args, tmp_path / "o", pipeline=_pipeline([3])) != 0
    assert _leftovers(env) == []


def test_ac5_keyboard_rejects_then_accepts(tmp_path: Path) -> None:
    proc, tmp = _spawn(
        tmp_path, "keyboard", ["--n", "1", "--min-persons", "1", "--mode", "frames"], counts=(2,)
    )
    out, err = proc.communicate(b"n7\nn1 m2\n", timeout=WAIT_S)
    assert proc.returncode == 0, (out + err).decode(errors="replace")
    assert b"rejected" in out.lower()
    stats = json.loads(next((tmp_path / "out").iterdir()).read_text(encoding="utf-8"))
    assert (stats["boxes_shown"], stats["boxes_not_person"], stats["persons_missed"]) == (2, 1, 2)
    assert _leftovers(tmp) == []


def test_ac5_reviewer_is_a_small_typed_protocol() -> None:
    bases: tuple[object, ...] = spotcheck.Reviewer.__mro__
    assert typing.Protocol in bases
    hints = typing.get_type_hints(spotcheck.Reviewer.judge)
    assert typing.get_origin(hints["return"]) is collections.abc.Mapping
    assert typing.get_args(hints["return"]) == (int, spotcheck.Judgement)
    assert isinstance(spotcheck.KeyboardReviewer(0), spotcheck.Reviewer)
    assert isinstance(spotcheck.JsonFileReviewer(Path("j.json")), spotcheck.Reviewer)
    assert isinstance(Scripted(), spotcheck.Reviewer)


# AC6: privacy-guard exemption ---------------------------------------------------------

SPOTCHECK_WRITE = """\
import os

def write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
"""


def test_ac6_exemption_is_exactly_the_spotcheck_path() -> None:
    assert privacy_guard.IMAGE_WRITE_EXEMPTION == TOOL
    source = (ROOT / "scripts" / "privacy_guard.py").read_text(encoding="utf-8")
    assert source.count("spotcheck") == 1  # the constant, and nothing else names the tool
    assert privacy_guard.scan_source(SPOTCHECK_WRITE, TOOL) == []


@pytest.mark.parametrize(
    "path",
    [
        "engine/wearreport/x.py",
        "engine/wearreport/detect.py",
        "engine/wearreport/tools/__init__.py",
        "engine/wearreport/tools/other.py",
        "engine/wearreport/spotcheck.py",
        "engine/wearreport/tools/spotcheck_extra.py",
        "engine/wearreport/tools/sub/spotcheck.py",
        "./engine/wearreport/tools/spotcheck.py",
        "engine/wearreport/tools/../tools/spotcheck.py",
    ],
)
def test_ac6_the_same_write_elsewhere_is_flagged(path: str) -> None:
    assert privacy_guard.scan_source(SPOTCHECK_WRITE, path)


@pytest.mark.parametrize(
    "source",
    [
        "import cv2\ncv2.imwrite(p, frame)\n",
        "open('a.png', 'wb').write(b)\n",
        "img.save(p)\n",
        "frame.tofile(p)\n",
        "from pathlib import Path\nPath(p).write_bytes(b)\n",
        "import numpy as np\nnp.save(p, frame)\n",
        "import zipfile\nzipfile.ZipFile(p, 'w')\n",
        "import urllib.request\nurllib.request.urlretrieve(u, p)\n",
        "import tempfile\ntempfile.mkstemp(suffix='.jpg')\n",
        "keep('frame.jpg')\n",
        "import cv2\nw = map(cv2.imwrite, ps, fs)\n",
        "from os import *\n",
    ],
)
def test_ac6_every_other_rule_still_applies_to_the_tool(source: str) -> None:
    assert privacy_guard.scan_source(source, TOOL)


def test_ac6_guard_scans_the_tool_and_passes(capsys: pytest.CaptureFixture[str]) -> None:
    scanned = {p.relative_to(ROOT).as_posix() for p in privacy_guard.engine_files(ROOT)}
    assert TOOL in scanned
    assert privacy_guard.main(["--root", str(ROOT)]) == 0
    assert "clean" in capsys.readouterr().out


def test_ac6_cli_flags_the_write_outside_the_tool(tmp_path: Path) -> None:
    for rel in (TOOL, "engine/wearreport/other.py"):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(SPOTCHECK_WRITE, encoding="utf-8")
    assert privacy_guard.main(["--root", str(tmp_path)]) == 1
    (tmp_path / "engine/wearreport/other.py").unlink()
    assert privacy_guard.main(["--root", str(tmp_path)]) == 0


# AC7: statistics ----------------------------------------------------------------------


def test_ac7_statistics_fields_types_and_text_only(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "o"
    assert (
        _run(
            ["--n", "2", "--min-persons", "1", "--mode", "frames"],
            out,
            pipeline=_pipeline([2, 3]),
            reviewer=Scripted(),
        )
        == 0
    )
    (path,) = out.iterdir()
    assert path.name == "2026-09-25.json"
    raw = path.read_bytes()
    raw.decode("ascii")
    assert all(b >= 0x20 or b in b"\n\r\t" for b in raw)
    assert not raw.startswith(MAGIC)
    stats = json.loads(raw)
    assert set(stats) == FIELDS
    assert stats["date"] == "2026-09-25"
    assert stats["reviewer"] == "tester"
    assert stats["mode"] == "frames"
    for key in (
        "frames_reviewed",
        "boxes_shown",
        "boxes_not_person",
        "boxes_in_vehicle",
        "persons_missed",
    ):
        assert type(stats[key]) is int, key
    for key in ("precision_person", "precision_pedestrian", "recall_estimate"):
        assert type(stats[key]) is float, key
    assert set(stats["detector"]) == {"model", "sha256", "conf"}
    assert stats["detector"] == {"model": "stub", "sha256": "0" * 64, "conf": detect.DEFAULT_CONF}


def test_ac7_never_overwrites(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "o"
    out.mkdir()
    (out / "2026-09-25.json").write_text("first\n", encoding="utf-8")
    for expected in ("2026-09-25-2.json", "2026-09-25-3.json"):
        assert (
            _run(
                ["--n", "1", "--min-persons", "1"],
                out,
                pipeline=_pipeline([3]),
                reviewer=Scripted(),
            )
            == 0
        )
        assert _stats(out, expected)["boxes_shown"] == 3
    assert (out / "2026-09-25.json").read_text(encoding="utf-8") == "first\n"


def test_ac7_default_out_dir_is_spotchecks(env: Path) -> None:
    code = spotcheck.main(
        ["--n", "1", "--min-persons", "1", "--reviewer", "t"],
        pipeline=_pipeline([3]),
        reviewer=Scripted(),
        today=DAY,
    )
    assert code == 0
    assert _stats(Path("spotchecks"))["boxes_shown"] == 3


# AC8: arithmetic ----------------------------------------------------------------------


def _decide(script: dict[int, tuple[set[int], set[int], int | None]]) -> Decide:
    def decide(item: spotcheck.ReviewItem, mode: str) -> spotcheck.Judgement:
        not_person, in_vehicle, missed = script.get(item.number, (set(), set(), 0))
        return spotcheck.Judgement(
            frozenset(not_person), frozenset(in_vehicle), missed if mode == "frames" else None
        )

    return decide


@pytest.mark.parametrize(
    ("mode", "counts", "script", "expected"),
    [
        (
            "crops",
            [4, 6],
            {1: ({1}, set(), None), 2: ({2}, set(), None), 3: (set(), {3}, None)},
            (2, 10, 2, 1, None, 0.8, 0.7, None),
        ),
        (
            "frames",
            [4, 6],
            {1: ({1}, {2}, 1), 2: ({5}, set(), 2)},
            (2, 10, 2, 1, 3, 0.8, 0.7, 8 / 11),
        ),
        (
            "frames",
            [0, 0],
            {1: (set(), set(), 2), 2: (set(), set(), 0)},
            (2, 0, 0, 0, 2, None, None, 0.0),
        ),
        ("frames", [0], {1: (set(), set(), 0)}, (1, 0, 0, 0, 0, None, None, None)),
        ("frames", [2], {1: ({1, 2}, set(), 0)}, (1, 2, 2, 0, 0, 0.0, 0.0, None)),
        ("frames", [3], {1: ({1}, {2, 3}, 0)}, (1, 3, 1, 2, 0, 2 / 3, 0.0, 1.0)),
        ("crops", [0], {}, (1, 0, 0, 0, None, None, None, None)),
    ],
    ids=[
        "crops",
        "frames",
        "zero-boxes-missed",
        "zero-boxes",
        "all-wrong",
        "vehicles",
        "crops-empty",
    ],
)
def test_ac8_statistics_arithmetic(
    env: Path,
    tmp_path: Path,
    mode: spotcheck.Mode,
    counts: list[int],
    script: dict[int, tuple[set[int], set[int], int | None]],
    expected: tuple[Any, ...],
) -> None:
    out = tmp_path / "o"
    args = ["--n", str(len(counts)), "--min-persons", "0", "--mode", mode, "--seed", "0"]
    assert _run(args, out, pipeline=_pipeline(counts), reviewer=Scripted(_decide(script))) == 0
    stats = _stats(out)
    keys = (
        "frames_reviewed",
        "boxes_shown",
        "boxes_not_person",
        "boxes_in_vehicle",
        "persons_missed",
        "precision_person",
        "precision_pedestrian",
        "recall_estimate",
    )
    for key, want in zip(keys, expected, strict=True):
        if want is None or isinstance(want, int):
            assert stats[key] == want, key
        else:
            assert stats[key] == pytest.approx(want, abs=1e-4), key


def test_ac8_no_qualifying_frame_writes_nothing(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "o"
    assert (
        _run(
            ["--n", "3", "--min-persons", "5"], out, pipeline=_pipeline([1, 2]), reviewer=Scripted()
        )
        != 0
    )
    assert not out.exists()
    assert _leftovers(env) == []


# AC9: runtime privacy (a full dry run against the fake camera server) -----------------


def _files(roots: Sequence[Path]) -> dict[Path, bytes]:
    found: dict[Path, bytes] = {}
    for root in roots:
        for dirpath, _dirs, names in os.walk(root):
            for name in names:
                path = Path(dirpath, name)
                found[path] = path.read_bytes()
    return found


def test_ac9_full_dry_run_leaves_only_the_statistics_file(
    env: Path, tmp_path: Path, offline: None
) -> None:
    _model()
    work, home = Path.cwd(), Path(os.environ["HOME"])
    watched = [work, home, env]
    before = _files(watched)
    judgements = tmp_path / "reviewer" / "judgements.json"
    judgements.parent.mkdir()
    rendered: list[int] = []

    def decide(numbering: dict[str, Any]) -> dict[str, Any]:
        (workdir,) = _leftovers(env)
        rendered.append(sum(_is_image(p) for p in workdir.iterdir()))
        return {
            str(image["image"]): {"not_person": [], "in_vehicle": []}
            for image in numbering["images"]
        }

    thread = _judge_by_file(env, judgements, None, decide)
    args = [
        "--dry-run",
        "--model",
        "yolox_s.onnx",
        "--n",
        "3",
        "--min-persons",
        "1",
        "--seed",
        "1",
        "--judgements",
        str(judgements),
        "--reviewer",
        "tester",
        "--out-dir",
        str(work / "stats"),
    ]
    assert spotcheck.main(args, today=DAY) == 0
    thread.join()
    assert rendered and rendered[0] > 0  # images really were rendered during the run
    after = _files(watched)
    new = sorted(set(after) - set(before))
    assert new == [work / "stats" / "2026-09-25.json"]
    for path, data in after.items():
        assert not data.startswith(MAGIC), path
        assert b"\xff\xd8\xff" not in data and b"\x89PNG" not in data, path
    assert _leftovers(env) == []
    stats = _stats(work / "stats")
    assert stats["frames_reviewed"] == 3 and stats["boxes_shown"] > 0


def test_ac9_dry_run_refuses_the_default_statistics_directory(env: Path) -> None:
    args = ["--dry-run", "--model", "yolox_s.onnx", "--n", "1", "--reviewer", "t"]
    assert spotcheck.main(args, reviewer=Scripted(), today=DAY) != 0
    assert not Path("spotchecks").exists()
