"""Acceptance tests for T-037 (the spot-check on native Windows, with an in-window
reviewer). The task contract: do not edit.

Every image here is synthetic (uniform colours) or comes from the licensed fixture photos
served by the fake camera server. The window tests read the window's widgets and the
PhotoImage's size and pixels through Tk; nothing is saved or looked at otherwise.

The window tests need Tk and a display. On Windows they always run (and fail if Tk is
missing); elsewhere they skip without one unless WEARREPORT_REQUIRE_TK is set. On Linux,
run them under `xvfb-run`. Tests that need a model file skip only when the file is
missing and WEARREPORT_REQUIRE_MODEL is unset.
"""

from __future__ import annotations

import ast
import datetime
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport.tools import spotcheck

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "engine" / "wearreport" / "tools" / "spotcheck.py"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "windows.yml"
FETCH_SH = ROOT / "scripts" / "fetch_model.sh"
FETCH_PS1 = ROOT / "scripts" / "fetch_model.ps1"
REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
REQUIRE_TK = "WEARREPORT_REQUIRE_TK"
WINDOWS = sys.platform == "win32"
H, W = 288, 352
DAY = datetime.date(2026, 9, 27)
WAIT_S = 60
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".ppm"}
PEDESTRIAN = spotcheck.Judgement(frozenset(), frozenset(), None)

Frame = npt.NDArray[np.uint8]


# Helpers ------------------------------------------------------------------------------


def _need_window() -> None:
    """Skip (or fail, on Windows or with WEARREPORT_REQUIRE_TK) without Tk and a display."""
    try:
        import tkinter

        tkinter.Tk().destroy()
    except Exception as exc:  # ImportError without Tk, TclError without a display
        if WINDOWS or os.environ.get(REQUIRE_TK):
            pytest.fail(f"no Tk window here: {exc}")
        pytest.skip(
            "no Tk display here (on Linux run under xvfb-run); the windows-latest CI job "
            "runs the window tests"
        )


def _model(name: str) -> Path:
    path = detect.model_path(name)
    if not path.is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run scripts/fetch_model.sh or fetch_model.ps1")
    return path


def _box(k: int) -> tuple[float, float, float, float]:
    return (10.0 + 30 * k, 50.0, 30.0 + 30 * k, 110.0)


class Stub:
    """counts[i] people in the frame whose colour is 10*i."""

    def __init__(self, counts: Sequence[int]) -> None:
        self.counts = list(counts)

    def detect(self, frame: Frame) -> list[detect.Detection]:
        n = self.counts[int(frame[0, 0, 0]) // 10]
        return [detect.Detection("person", 0.9, _box(k)) for k in range(n)]


def _pipeline(counts: Sequence[int]) -> spotcheck.Pipeline:
    frames = [np.full((H, W, 3), 10 * i, dtype=np.uint8) for i in range(len(counts))]
    return spotcheck.Pipeline(frames=lambda: frames, detector=Stub(counts), info=INFO)


def _untouchable() -> spotcheck.Pipeline:
    def frames() -> list[Frame]:
        raise AssertionError("the sweep must not start")

    return spotcheck.Pipeline(frames=frames, detector=Stub([]), info=INFO)


class Untouchable:
    def judge(
        self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
    ) -> dict[int, spotcheck.Judgement]:
        raise AssertionError("the review must not start")


class Scripted:
    """Answers "all pedestrians", and records whether a review directory existed."""

    def __init__(self) -> None:
        self.directories: list[Path] = []

    def judge(
        self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
    ) -> dict[int, spotcheck.Judgement]:
        self.directories = _leftovers(Path(tempfile.gettempdir()))
        return {
            item.number: spotcheck.Judgement(
                frozenset(), frozenset(), 0 if mode == "frames" else None
            )
            for item in items
        }


def _crop(number: int, height: int, width: int, bgr: tuple[int, int, int]) -> spotcheck.ReviewItem:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = bgr
    return spotcheck.ReviewItem(number, f"crop-{number:04d}.png", (number,), image)


def _crops(numbers: Sequence[int]) -> list[spotcheck.ReviewItem]:
    return [_crop(n, 60, 40, (10 * i, 20, 30)) for i, n in enumerate(numbers)]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Outside CI, in a fresh working directory, HOME and temporary directory; yields
    the temporary directory."""
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    work, home, tmp = tmp_path / "work", tmp_path / "home", tmp_path / "tmp"
    for d in (work, home, tmp):
        d.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    yield tmp


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")):
            raise AssertionError(f"non-local connection attempted: {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


def _run(args: Sequence[str], out: Path, **kwargs: Any) -> int:
    argv = [*args, "--reviewer", "tester", "--out-dir", str(out)]
    return spotcheck.main(argv, today=DAY, **kwargs)


def _leftovers(tmp: Path) -> list[Path]:
    return sorted(tmp.glob(spotcheck.TEMP_PREFIX + "*"))


def _files(roots: Sequence[Path]) -> dict[Path, bytes]:
    found: dict[Path, bytes] = {}
    for root in roots:
        for dirpath, _dirs, names in os.walk(root):
            for name in names:
                path = Path(dirpath, name)
                found[path] = path.read_bytes()
    return found


def _child_env(tmp: Path) -> dict[str, str]:
    environ = {k: v for k, v in os.environ.items() if k not in spotcheck.CI_VARIABLES}
    environ.update(TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp), PYTHONDONTWRITEBYTECODE="1")
    return environ


# Driving the window -------------------------------------------------------------------

SEQUENCES = {
    "Return": "<Return>",
    "space": "<space>",
    "BackSpace": "<BackSpace>",
}


def _widgets(widget: Any) -> Iterator[Any]:
    yield widget
    for child in widget.winfo_children():
        yield from _widgets(child)


def _snapshot(root: Any) -> dict[str, Any]:
    """What the window shows: its texts, and each image's size, source file and corner
    pixel (RGB)."""
    texts: list[str] = []
    photos: list[dict[str, Any]] = []
    toplevels = 0
    for widget in _widgets(root):
        if widget is not root and widget.winfo_class() == "Toplevel":
            toplevels += 1
        options = widget.keys()
        if "text" in options and str(widget.cget("text")):
            texts.append(str(widget.cget("text")))
        if "image" in options and str(widget.cget("image")):
            name = str(widget.cget("image"))
            pixel = root.tk.splitlist(root.tk.call(name, "get", 0, 0))
            photos.append(
                {
                    "width": int(root.tk.call("image", "width", name)),
                    "height": int(root.tk.call("image", "height", name)),
                    "file": str(root.tk.call(name, "cget", "-file")),
                    "pixel": tuple(int(v) for v in pixel),
                }
            )
    return {"texts": texts, "photos": photos, "toplevels": toplevels}


class Keys:
    """A window driver: sends `keys` one at a time (then `then`, repeatedly, until the
    window closes), recording what the window shows before each key. "CLOSE" closes the
    window as its title-bar button does."""

    def __init__(
        self,
        *keys: str,
        then: str | None = None,
        during: Callable[[], None] | None = None,
    ) -> None:
        self.keys = list(keys)
        self.then = then
        self.during = during
        self.shown: list[dict[str, Any]] = []
        self.root: Any = None

    def __call__(self, root: Any) -> None:
        self.root = root
        root.after(20, self._step)

    def _step(self) -> None:
        root = self.root
        self.shown.append(_snapshot(root))
        if self.during is not None:
            self.during()
        if self.keys:
            key = self.keys.pop(0)
        elif self.then is not None:
            key = self.then
        else:
            return
        if key == "CLOSE":
            root.tk.call(root.wm_protocol("WM_DELETE_WINDOW"))
        else:
            root.focus_force()
            root.event_generate(SEQUENCES.get(key, f"<KeyPress-{key}>"))
        root.after(20, self._step)


def _closed(root: Any) -> bool:
    import tkinter

    try:
        return not root.winfo_exists()
    except tkinter.TclError:
        return True


def _judge_in_window(
    items: Sequence[spotcheck.ReviewItem], keys: Keys, timeout: float = WAIT_S
) -> Mapping[int, spotcheck.Judgement]:
    reviewer = spotcheck.WindowReviewer(driver=keys)
    return reviewer.judge(items, "crops", time.monotonic() + timeout)


# AC1: the in-window reviewer ----------------------------------------------------------


def test_ac1_window_shows_each_crop_enlarged_from_memory_with_number_count_and_legend() -> None:
    _need_window()
    items = [
        _crop(41, 30, 20, (255, 0, 0)),  # BGR blue
        _crop(52, 240, 100, (0, 0, 255)),  # BGR red
        _crop(63, 60, 40, (0, 255, 0)),  # BGR green
    ]
    keys = Keys("Return", "Return", "Return")
    judgements = _judge_in_window(items, keys)
    assert set(judgements) == {41, 52, 63}
    assert len(keys.shown) >= 3
    for k, (item, shown) in enumerate(zip(items, keys.shown, strict=False), start=1):
        assert shown["toplevels"] == 0  # one window
        (photo,) = shown["photos"]
        height, width = item.image.shape[:2]
        assert photo["width"] % width == 0 and photo["height"] % height == 0
        factor = photo["width"] // width
        assert factor == photo["height"] // height >= 1
        assert photo["file"] == ""  # drawn from data in memory, not from a file
        blue, green, red = (int(v) for v in item.image[0, 0])
        assert photo["pixel"] == (red, green, blue)
        texts = shown["texts"]
        assert any(str(item.number) in t and f"{k} of 3" in t for t in texts), texts
        legend = [t for t in texts if all(w in t for w in ("Enter", "Space", "Backspace"))]
        assert len(legend) == 1 and "\n" not in legend[0]
        assert re.search(r"\bn\b", legend[0]) and re.search(r"\bv\b", legend[0])
        assert re.search(r"\bq\b", legend[0])
    small = keys.shown[0]["photos"][0]
    assert small["width"] // 20 >= 2  # a small crop is enlarged


def test_ac1_keys_record_pedestrian_not_person_and_in_vehicle() -> None:
    _need_window()
    items = _crops([1, 2, 3, 4])
    judgements = _judge_in_window(items, Keys("Return", "space", "n", "v"))
    assert dict(judgements) == {
        1: PEDESTRIAN,
        2: PEDESTRIAN,
        3: spotcheck.Judgement(frozenset({3}), frozenset(), None),
        4: spotcheck.Judgement(frozenset(), frozenset({4}), None),
    }


def test_ac1_backspace_goes_back_one_image_and_changes_it() -> None:
    _need_window()
    items = _crops([7, 8, 9])
    keys = Keys("BackSpace", "n", "BackSpace", "v", "n", "Return")
    judgements = _judge_in_window(items, keys)
    assert dict(judgements) == {
        7: spotcheck.Judgement(frozenset(), frozenset({7}), None),
        8: spotcheck.Judgement(frozenset({8}), frozenset(), None),
        9: PEDESTRIAN,
    }
    counters = [next(t for t in s["texts"] if " of 3" in t) for s in keys.shown[:6]]
    assert ["1 of 3" in c for c in counters] == [True, True, False, True, False, False]
    assert "2 of 3" in counters[2] and "2 of 3" in counters[4] and "3 of 3" in counters[5]


@pytest.mark.parametrize("stop", ["q", "CLOSE"])
def test_ac1_q_or_closing_the_window_stops_without_statistics(
    env: Path, tmp_path: Path, stop: str
) -> None:
    _need_window()
    with pytest.raises(spotcheck.ReviewAborted):
        _judge_in_window(_crops([1, 2]), Keys("Return", stop))
    keys = Keys("n", stop)
    reviewer = spotcheck.WindowReviewer(driver=keys)
    out = tmp_path / "out"
    code = _run(
        ["--n", "2", "--min-persons", "1", "--view", "window"],
        out,
        pipeline=_pipeline([2, 1]),
        reviewer=reviewer,
    )
    assert code == 1
    assert not out.exists()
    assert _closed(keys.root)
    assert _leftovers(env) == []


def test_ac1_same_judgements_and_statistics_as_the_keyboard_reviewer(
    env: Path, tmp_path: Path
) -> None:
    _need_window()
    items = _crops([1, 2, 3, 4])
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"\nn\nv\n\n")
        keyboard = spotcheck.KeyboardReviewer(read_fd, out=io.StringIO())
        by_keyboard = keyboard.judge(items, "crops", time.monotonic() + WAIT_S)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    by_window = _judge_in_window(items, Keys("Return", "n", "v", "space"))
    assert dict(by_window) == dict(by_keyboard)
    stats = [
        spotcheck.compute_stats(
            items, j, mode="crops", frames_reviewed=2, reviewer="t", info=INFO, day=DAY
        )
        for j in (by_keyboard, by_window)
    ]
    assert stats[0] == stats[1]

    # The same through the command: identical statistics files.
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"n\n\nv\n\n\n")
        keyboard = spotcheck.KeyboardReviewer(read_fd, out=io.StringIO())
        args = ["--n", "2", "--min-persons", "1", "--view", "files"]
        assert _run(args, tmp_path / "a", pipeline=_pipeline([3, 2]), reviewer=keyboard) == 0
    finally:
        os.close(read_fd)
        os.close(write_fd)
    window = spotcheck.WindowReviewer(driver=Keys("n", "Return", "v", "space", "Return"))
    args = ["--n", "2", "--min-persons", "1", "--view", "window"]
    assert _run(args, tmp_path / "b", pipeline=_pipeline([3, 2]), reviewer=window) == 0
    name = f"{DAY.isoformat()}.json"
    first = json.loads((tmp_path / "a" / name).read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "b" / name).read_text(encoding="utf-8"))
    assert first == second
    assert first["boxes_shown"] == 5 and first["boxes_not_person"] == 1
    assert first["boxes_in_vehicle"] == 1


def test_ac1_window_review_writes_no_file_and_creates_no_review_directory(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _need_window()
    work, home = Path.cwd(), Path(os.environ["HOME"])
    watched = [work, home, env]
    tempfile.gettempdir()  # tempfile's own writability probe, before the snapshot
    before = _files(watched)
    made: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def mkdtemp(*args: Any, **kwargs: Any) -> str:
        made.append("mkdtemp")
        return str(real_mkdtemp(*args, **kwargs))

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    during: list[dict[Path, bytes]] = []
    keys = Keys("n", then="Return", during=lambda: during.append(_files(watched)))
    code = _run(
        ["--n", "3", "--min-persons", "1", "--view", "window"],
        work / "stats",
        pipeline=_pipeline([2, 1, 3]),
        reviewer=spotcheck.WindowReviewer(driver=keys),
    )
    assert code == 0
    assert during, "the driver never ran"
    assert all(snapshot == before for snapshot in during)  # nothing new while reviewing
    assert made == []
    assert _leftovers(env) == []
    after = _files(watched)
    assert sorted(set(after) - set(before)) == [work / "stats" / f"{DAY.isoformat()}.json"]
    stats = json.loads((work / "stats" / f"{DAY.isoformat()}.json").read_text(encoding="utf-8"))
    assert stats["boxes_shown"] == 6 and stats["boxes_not_person"] == 1


WINDOW_CHILD = r"""
import json, os, sys, tempfile
import numpy as np
from wearreport import detect
from wearreport.tools import spotcheck as sc

tempfile.gettempdir()  # tempfile's own writability probe, before the audit

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
EVENTS = {"os.mkdir", "os.rename", "os.replace", "os.link", "os.symlink", "os.truncate",
          "tempfile.mkstemp", "tempfile.mkdtemp", "shutil.copyfile", "shutil.move"}

def report(kind, path):
    text = os.path.abspath(os.fsdecode(path)) if path is not None else "?"
    os.write(2, ("\n@@audit " + kind + " " + text + "\n").encode("utf-8", "backslashreplace"))

def audit(event, args):
    if event == "open":
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & WRITE_FLAGS)
        if writing and not isinstance(path, int):
            report("write", path)
    elif event in EVENTS:
        report(event, args[0])

class Stub:
    def detect(self, frame):
        n = int(frame[0, 0, 0]) // 10 + 1
        return [detect.Detection("person", 0.9, (10.0 + 30 * k, 50.0, 30.0 + 30 * k, 110.0))
                for k in range(n)]

frames = [np.full((288, 352, 3), 10 * i, dtype=np.uint8) for i in range(3)]
pipeline = sc.Pipeline(frames=lambda: frames, detector=Stub(),
                       info=sc.DetectorInfo(model="stub", sha256="0" * 64, conf=0.35))

def drive(root):
    def step():
        root.focus_force()
        root.event_generate("<Return>")
        root.after(20, step)
    root.after(20, step)

sys.addaudithook(audit)
sys.exit(sc.main(sys.argv[1:], pipeline=pipeline, reviewer=sc.WindowReviewer(driver=drive)))
"""
AUDIT = re.compile(r"^@@audit (\S+) (.*)$", re.MULTILINE)


def test_ac1_window_review_process_writes_nothing_but_the_statistics(tmp_path: Path) -> None:
    """An audit hook in a separate process sees every file the run opens for writing and
    every directory it makes: in window mode, only the statistics file."""
    _need_window()
    work, tmp = tmp_path / "work", tmp_path / "tmp"
    work.mkdir()
    tmp.mkdir()
    args = ["--n", "3", "--min-persons", "1", "--view", "window", "--reviewer", "tester"]
    args += ["--out-dir", "stats"]
    result = subprocess.run(
        [sys.executable, "-c", WINDOW_CHILD, *args],
        cwd=work,
        env=_child_env(tmp),
        capture_output=True,
        timeout=WAIT_S * 2,
        check=False,
    )
    stderr = result.stderr.decode(errors="replace")
    assert result.returncode == 0, stderr
    stats_file = work / "stats" / f"{datetime.date.today().isoformat()}.json"
    assert json.loads(stats_file.read_text(encoding="utf-8"))["boxes_shown"] == 6
    events = [(kind, Path(path)) for kind, path in AUDIT.findall(stderr)]
    assert [k for k, _ in events if k == "tempfile.mkdtemp"] == []
    stray = [(k, p) for k, p in events if p not in (stats_file, stats_file.parent)]
    assert stray == []
    assert sorted(_files([work, tmp])) == [stats_file]


def test_ac1_frames_mode_with_the_window_view_is_refused(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--n", "2", "--min-persons", "1", "--view", "window", "--mode", "frames"]
    code = _run(args, tmp_path / "out", pipeline=_untouchable(), reviewer=Untouchable())
    assert code != 0
    err = capsys.readouterr().err
    assert "frames" in err and "window" in err
    assert not (tmp_path / "out").exists()
    assert _leftovers(env) == []
    items = [spotcheck.ReviewItem(1, "frame-0001.png", (1, 2), np.zeros((4, 4, 3), np.uint8))]
    with pytest.raises(spotcheck.SpotcheckError, match="frames"):
        spotcheck.WindowReviewer().judge(items, "frames", time.monotonic() + WAIT_S)


def test_ac1_review_timeout_closes_the_window_and_exits_without_statistics(
    env: Path, tmp_path: Path
) -> None:
    _need_window()
    keys = Keys()  # never answers
    with pytest.raises(spotcheck.ReviewTimeout):
        _judge_in_window(_crops([1]), keys, timeout=0.5)
    assert _closed(keys.root)

    keys = Keys("Return")  # answers the first crop, then nothing
    started = time.monotonic()
    code = _run(
        ["--n", "2", "--min-persons", "1", "--view", "window", "--timeout", "1"],
        tmp_path / "out",
        pipeline=_pipeline([2, 1]),
        reviewer=spotcheck.WindowReviewer(driver=keys),
    )
    assert code == 3
    assert time.monotonic() - started < WAIT_S / 2
    assert _closed(keys.root)
    assert not (tmp_path / "out").exists()
    assert _leftovers(env) == []


# AC2: the default view ----------------------------------------------------------------


def test_ac2_window_is_the_default_view_on_windows_only() -> None:
    assert spotcheck.default_view("win32") == "window"
    for platform in ("linux", "darwin", "freebsd14", "cygwin"):
        assert spotcheck.default_view(platform) == "files"
    assert spotcheck.default_view() == ("window" if WINDOWS else "files")


def test_ac2_view_option() -> None:
    parser = spotcheck.build_parser()
    assert parser.parse_args(["--n", "5", "--view", "window"]).view == "window"
    assert parser.parse_args(["--n", "5", "--view", "files"]).view == "files"
    with pytest.raises(SystemExit):
        parser.parse_args(["--n", "5", "--view", "browser"])


def test_ac2_a_run_without_view_uses_this_platforms_default(env: Path, tmp_path: Path) -> None:
    reviewer = Scripted()
    args = ["--n", "1", "--min-persons", "1"]
    assert _run(args, tmp_path / "out", pipeline=_pipeline([2]), reviewer=reviewer) == 0
    # files: the images are in a review directory while the reviewer judges them
    assert bool(reviewer.directories) == (not WINDOWS)
    assert _leftovers(env) == []


def test_ac2_window_view_is_available_on_every_platform(env: Path, tmp_path: Path) -> None:
    reviewer = Scripted()
    args = ["--n", "1", "--min-persons", "1", "--view", "window"]
    assert _run(args, tmp_path / "out", pipeline=_pipeline([2]), reviewer=reviewer) == 0
    assert reviewer.directories == []
    reviewer = Scripted()
    args = ["--n", "1", "--min-persons", "1", "--view", "files"]
    assert _run(args, tmp_path / "out2", pipeline=_pipeline([2]), reviewer=reviewer) == 0
    assert len(reviewer.directories) == 1
    assert _leftovers(env) == []


# AC3: the rest of the tool on Windows -------------------------------------------------


def test_ac3_a_locked_directory_is_kept_and_deleted_once_unlocked(env: Path) -> None:
    directory = spotcheck.ReviewDirectory()
    path = directory.create()
    assert path.parent == env and path.name.startswith(spotcheck.TEMP_PREFIX)
    later = time.time() + 3600
    try:
        assert spotcheck.remove_stale(env, 60, later) == 0
        assert path.is_dir()
    finally:
        assert directory.remove()
    assert not path.exists()
    stale = Path(tempfile.mkdtemp(prefix=spotcheck.TEMP_PREFIX, dir=env))
    (stale / "sub").mkdir()
    (stale / "sub" / "crop-0001.png").write_bytes(b"x")
    keep = env / "other-dir"
    keep.mkdir()
    assert spotcheck.remove_stale(env, 60, later) == 1
    assert not stale.exists() and keep.is_dir()


LOCK_CHILD = r"""
import sys
from wearreport.tools import spotcheck as sc
directory = sc.ReviewDirectory()
print(directory.create(), flush=True)
sys.stdin.readline()
"""


def test_ac3_another_process_lock_is_respected_until_it_is_killed(tmp_path: Path) -> None:
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-c", LOCK_CHILD],
        env=_child_env(tmp),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdout is not None
        path = Path(proc.stdout.readline().decode().strip())
        assert path.is_dir() and path.parent == tmp
        later = time.time() + 3600
        assert spotcheck.remove_stale(tmp, 60, later) == 0
        assert path.is_dir()
    finally:
        proc.kill()  # SIGKILL, or TerminateProcess on Windows: no clean-up runs
        proc.communicate(timeout=WAIT_S)
    assert path.is_dir()  # what a killed run leaves behind...
    deadline = time.monotonic() + 10
    while spotcheck.remove_stale(tmp, 60, later) == 0 and time.monotonic() < deadline:
        time.sleep(0.1)  # Windows releases a dead process's locks shortly after it ends
    assert not path.exists()  # ...the next run deletes


def test_ac3_a_signal_during_the_review_deletes_the_directory(env: Path, tmp_path: Path) -> None:
    before = signal.getsignal(signal.SIGTERM)

    class Raise:
        def judge(
            self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
        ) -> dict[int, spotcheck.Judgement]:
            assert len(_leftovers(env)) == 1
            signal.raise_signal(signal.SIGTERM)
            time.sleep(WAIT_S)
            return {}

    started = time.monotonic()
    code = _run(
        ["--n", "1", "--min-persons", "1", "--view", "files"],
        tmp_path / "out",
        pipeline=_pipeline([2]),
        reviewer=Raise(),
    )
    assert code == 128 + signal.SIGTERM
    assert time.monotonic() - started < WAIT_S / 2
    assert _leftovers(env) == []
    assert not (tmp_path / "out").exists()
    assert signal.getsignal(signal.SIGTERM) == before


SIGNAL_CHILD = r"""
import sys
import numpy as np
from wearreport import detect
from wearreport.tools import spotcheck as sc

class Stub:
    def detect(self, frame):
        return [detect.Detection("person", 0.9, (10.0, 50.0, 30.0, 110.0))]

frames = [np.full((288, 352, 3), 7, dtype=np.uint8)]
pipeline = sc.Pipeline(frames=lambda: frames, detector=Stub(),
                       info=sc.DetectorInfo(model="stub", sha256="0" * 64, conf=0.35))
sys.exit(sc.main(sys.argv[1:], pipeline=pipeline))
"""


def test_ac3_ctrl_break_or_ctrl_c_in_a_running_tool_deletes_the_directory(
    tmp_path: Path,
) -> None:
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    judgements = tmp_path / "judgements.json"
    args = ["--n", "1", "--min-persons", "1", "--view", "files", "--judgements", str(judgements)]
    args += ["--reviewer", "tester", "--out-dir", str(tmp_path / "out")]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        [sys.executable, "-c", SIGNAL_CHILD, *args],
        env=_child_env(tmp),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=flags,
    )
    deadline = time.monotonic() + WAIT_S
    while not any((d / spotcheck.NUMBERING_FILE).is_file() for d in _leftovers(tmp)):
        assert proc.poll() is None, proc.communicate()
        assert time.monotonic() < deadline, "the review directory never appeared"
        time.sleep(0.05)
    time.sleep(0.3)
    if sys.platform == "win32":
        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        expected = signal.SIGBREAK
    else:
        proc.send_signal(signal.SIGINT)
        expected = signal.SIGINT
    _out, err = proc.communicate(timeout=WAIT_S)
    assert proc.returncode == 128 + expected, err.decode(errors="replace")
    assert b"no statistics written" in err
    assert _leftovers(tmp) == []
    assert not (tmp_path / "out").exists()


def test_ac3_the_review_timeout_stops_a_reviewer_that_ignores_its_deadline(
    env: Path, tmp_path: Path
) -> None:
    before = signal.getsignal(signal.SIGINT)

    class Hang:
        def judge(
            self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
        ) -> dict[int, spotcheck.Judgement]:
            time.sleep(WAIT_S)
            return {}

    for view in ("files", "window"):
        started = time.monotonic()
        code = _run(
            ["--n", "1", "--min-persons", "1", "--view", view, "--timeout", "0.5"],
            tmp_path / "out",
            pipeline=_pipeline([2]),
            reviewer=Hang(),
        )
        assert code == 3
        assert time.monotonic() - started < WAIT_S / 2
        assert _leftovers(env) == []
        assert not (tmp_path / "out").exists()
    assert signal.getsignal(signal.SIGINT) == before
    time.sleep(0.2)
    assert [t for t in threading.enumerate() if isinstance(t, threading.Timer)] == []


def test_ac3_keyboard_reviewer_reads_times_out_and_stops_without_select() -> None:
    items = _crops([1, 2])
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"n\n")
        reviewer = spotcheck.KeyboardReviewer(read_fd, out=io.StringIO())
        started = time.monotonic()
        with pytest.raises(spotcheck.ReviewTimeout):
            reviewer.judge(items, "crops", time.monotonic() + 0.5)
        assert time.monotonic() - started < WAIT_S / 2
    finally:
        os.close(read_fd)
        os.close(write_fd)
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"n\n")
        os.close(write_fd)
        reviewer = spotcheck.KeyboardReviewer(read_fd, out=io.StringIO())
        with pytest.raises(spotcheck.ReviewAborted):
            reviewer.judge(items, "crops", time.monotonic() + WAIT_S)
    finally:
        os.close(read_fd)


@pytest.mark.parametrize("view", ["files", "window"])
def test_ac3_refuses_to_run_in_ci(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, view: str
) -> None:
    monkeypatch.setenv("CI", "true")
    args = ["--n", "1", "--min-persons", "1", "--view", view]
    assert _run(args, tmp_path / "out", pipeline=_untouchable(), reviewer=Untouchable()) == 2
    assert list(env.iterdir()) == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("view", ["files", "window"])
def test_ac3_refuses_a_temporary_directory_inside_a_git_work_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], view: str
) -> None:
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    tree = tmp_path / "checkout"
    (tree / ".git").mkdir(parents=True)
    tmp = tree / "build" / "tmp"
    tmp.mkdir(parents=True)
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    args = ["--n", "1", "--min-persons", "1", "--view", view]
    assert _run(args, tmp_path / "out", pipeline=_untouchable(), reviewer=Untouchable()) == 1
    assert "git work tree" in capsys.readouterr().err
    assert list(tmp.iterdir()) == []


def test_ac3_no_hard_coded_temporary_directory() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "/tmp" not in source  # noqa: S108  (the test is that it is absent)
    assert "\\temp" not in source.lower()


# AC4: installing on Windows -----------------------------------------------------------

BLOCK_LLAMA = r"""
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "llama_cpp" or name == "wearreport.tools.judge":
            raise ImportError(f"{name} is blocked")
        return None

sys.meta_path.insert(0, Block())
from wearreport.tools import spotcheck
spotcheck.build_parser().parse_args(["--n", "1", "--view", "window"])
print(sorted(m for m in sys.modules if "llama" in m or m.endswith(".judge")))
"""


def test_ac4_spotcheck_never_imports_llama_cpp(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", BLOCK_LLAMA],
        env=_child_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imported += [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    ]
    assert not [name for name in imported if "llama" in name or name.endswith(".judge")]


def _pins(text: str) -> set[str]:
    return set(re.findall(r"\b[0-9a-f]{64}\b", text))


def test_ac4_fetch_model_ps1_pins_the_same_models_as_fetch_model_sh() -> None:
    sh = FETCH_SH.read_text(encoding="utf-8")
    ps1 = FETCH_PS1.read_text(encoding="utf-8")
    sha_m = re.search(r'SHA256_YOLOX_M="([0-9a-f]{64})"', sh)
    sha_s = re.search(r'SHA256_YOLOX_S="([0-9a-f]{64})"', sh)
    base = re.search(r'BASE_URL="(https://[^"]+)"', sh)
    assert sha_m and sha_s and base
    assert sha_m.group(1) == detect.MODEL_SHA256["yolox_m.onnx"]
    assert _pins(ps1) == {sha_m.group(1), sha_s.group(1)}
    assert base.group(1) in ps1
    assert "yolox_m.onnx" in ps1
    assert "SHA256" in ps1 and "Get-FileHash" in ps1


def _powershell() -> str:
    found = shutil.which("powershell") or shutil.which("pwsh")
    if found is None:
        if WINDOWS:
            pytest.fail("PowerShell is missing")
        pytest.skip("PowerShell is not installed here; the windows-latest CI job runs this")
    return found


def _dead_proxy_env() -> dict[str, str]:
    """The environment with every download sent to a closed local port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proxy = f"http://127.0.0.1:{port}"
    environ = dict(os.environ)
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"):
        environ[name] = proxy
    for name in ("NO_PROXY", "no_proxy"):
        environ.pop(name, None)
    return environ


def _fetch_ps1(dest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(FETCH_PS1),
            "-Dest",
            str(dest),
        ],
        env=_dead_proxy_env(),
        capture_output=True,
        text=True,
        timeout=WAIT_S * 3,
        check=False,
    )


def test_ac4_fetch_model_ps1_fails_closed(tmp_path: Path) -> None:
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "yolox_m.onnx").write_bytes(b"not the model")
    (dest / "yolox_s.onnx").write_bytes(b"not the model either")
    (dest / ".yolox_m.onnx.a1b2c3").write_bytes(b"a part file from a killed run")
    result = _fetch_ps1(dest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert sorted(p.name for p in dest.iterdir()) == []  # no unverified file left


def test_ac4_fetch_model_ps1_keeps_verified_models_without_downloading(tmp_path: Path) -> None:
    _powershell()
    dest = tmp_path / "models"
    dest.mkdir()
    for name in ("yolox_s.onnx", "yolox_m.onnx"):
        shutil.copyfile(_model(name), dest / name)
    result = _fetch_ps1(dest)  # every download would fail
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("yolox_s.onnx", "yolox_m.onnx"):
        assert detect.sha256_of(dest / name) == detect.MODEL_SHA256[name]
    assert sorted(p.name for p in dest.iterdir()) == ["yolox_m.onnx", "yolox_s.onnx"]


def test_ac4_readme_documents_the_windows_install_on_any_drive() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for needed in (
        "uv sync --locked --no-install-package llama-cpp-python",
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "D:\\uv",
        "fetch_model.ps1",
        "--view window",
    ):
        assert needed in readme, needed
    spotchecks = (ROOT / "spotchecks" / "README.md").read_text(encoding="utf-8")
    assert "--view window" in spotchecks and "Windows" in spotchecks


# AC5: CI ------------------------------------------------------------------------------


def _jobs(text: str) -> dict[str, str]:
    """Each job's text in a workflow file, by job id."""
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\s*$", body, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_ac5_ci_has_a_least_privilege_windows_job() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    jobs = _jobs(text)
    windows = [job for job in jobs.values() if re.search(r"runs-on:\s*windows-latest", job)]
    assert len(windows) == 1
    (job,) = windows
    assert "uv sync --locked --no-install-package llama-cpp-python" in job
    assert "fetch_model.ps1" in job
    assert "test_t_037.py" in job and "test_spotcheck" in job
    assert "permissions" not in job  # the workflow's read-only default applies
    assert re.search(r"^permissions:\s*\n\s+contents: read\s*$", text, re.MULTILINE)
    assert "pull_request_target" not in text
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", u) for u in uses)


def test_ac5_dry_run_with_the_window_reviewer_writes_statistics_and_no_image(
    env: Path, tmp_path: Path, offline: None
) -> None:
    """The dry run the Windows CI job performs: the fixture photos, the default model,
    the window reviewer answering through injected key events."""
    _need_window()
    _model(spotcheck.DEFAULT_MODEL)
    work, home = Path.cwd(), Path(os.environ["HOME"])
    watched = [work, home, env]
    tempfile.gettempdir()
    before = _files(watched)
    keys = Keys("n", then="Return")
    args = ["--dry-run", "--view", "window", "--n", "3", "--min-persons", "1", "--seed", "1"]
    code = _run(args, work / "stats", reviewer=spotcheck.WindowReviewer(driver=keys))
    assert code == 0
    stats_file = work / "stats" / f"{DAY.isoformat()}.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    print("statistics file:", stats_file.name, json.dumps(stats, sort_keys=True))
    assert stats["mode"] == "crops" and stats["frames_reviewed"] == 3
    assert stats["boxes_shown"] >= 3 and stats["boxes_not_person"] == 1
    assert stats["detector"]["model"] == Path(spotcheck.DEFAULT_MODEL).stem
    after = _files(watched)
    assert sorted(set(after) - set(before)) == [stats_file]
    images = [
        p
        for p, data in after.items()
        if p.suffix.lower() in IMAGE_SUFFIXES or data.startswith(MAGIC)
    ]
    print("image files after the run:", len(images))
    assert images == []
    assert _leftovers(env) == []


# AC6: privacy guard -------------------------------------------------------------------


def test_ac6_privacy_guard_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "privacy_guard.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


WRITERS = {
    "open",
    "fdopen",
    "mkdtemp",
    "mkstemp",
    "NamedTemporaryFile",
    "TemporaryFile",
    "SpooledTemporaryFile",
    "TemporaryDirectory",
    "imwrite",
    "write",
    "write_bytes",
    "write_text",
    "tofile",
    "save",
    "dump",
    "write_image",
    "write_numbering",
    "copyfile",
    "urlretrieve",
}


def _window_code(tree: ast.Module) -> list[ast.AST]:
    """The WindowReviewer class and every module-level function or class it reaches by
    name, transitively."""
    defined = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }
    todo, seen = ["WindowReviewer"], set()
    while todo:
        name = todo.pop()
        if name in seen or name not in defined:
            continue
        seen.add(name)
        for node in ast.walk(defined[name]):
            if isinstance(node, ast.Name):
                todo.append(node.id)
            elif isinstance(node, ast.Attribute):
                todo.append(node.attr)
    return [defined[name] for name in sorted(seen)]


def test_ac6_the_window_reviewer_adds_no_file_write() -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    code = _window_code(tree)
    assert any(isinstance(n, ast.ClassDef) and n.name == "WindowReviewer" for n in code)
    found: list[str] = []
    for top in code:
        for node in ast.walk(top):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in WRITERS:
                found.append(f"line {node.lineno}: {name}()")
            if any(k.arg == "file" for k in node.keywords):
                found.append(f"line {node.lineno}: file=")
    assert found == []
    names = {n.name for n in code if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    assert "ReviewDirectory" not in names


# AC7: contract ------------------------------------------------------------------------


def test_ac7_the_window_reviewer_is_a_reviewer() -> None:
    reviewer = spotcheck.WindowReviewer()
    assert isinstance(reviewer, spotcheck.Reviewer)
    assert spotcheck.VIEWS == ("files", "window")
