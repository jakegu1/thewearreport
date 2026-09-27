"""Unit tests for the spot-check's window view and its Windows code paths (T-037).

Every image here is synthetic. The window tests need Tk and a display: they skip without
one, except on Windows or with WEARREPORT_REQUIRE_TK set (on Linux, run them under
`xvfb-run`). The Windows-only tests run on the windows-latest CI job; the timer alarm and
the threaded standard-input reader are also exercised here on any platform, by switching
the tool onto those paths.
"""

from __future__ import annotations

import contextlib
import datetime
import gc
import io
import os
import signal
import sys
import tempfile
import time
import weakref
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport._cv import cv2
from wearreport.tools import spotcheck

H, W = 288, 352
DAY = datetime.date(2026, 9, 27)
WAIT_S = 60
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
REQUIRE_TK = "WEARREPORT_REQUIRE_TK"
PEDESTRIAN = spotcheck.Judgement(frozenset(), frozenset(), None)
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows code path")


def _need_window() -> None:
    try:
        import tkinter

        tkinter.Tk().destroy()
    except Exception as exc:  # ImportError without Tk, TclError without a display
        if sys.platform == "win32" or os.environ.get(REQUIRE_TK):
            pytest.fail(f"no Tk window here: {exc}")
        pytest.skip("no Tk display here (on Linux run under xvfb-run)")


class Stub:
    def __init__(self, counts: Sequence[int]) -> None:
        self.counts = list(counts)

    def detect(self, frame: npt.NDArray[np.uint8]) -> list[detect.Detection]:
        n = self.counts[int(frame[0, 0, 0]) // 10]
        return [
            detect.Detection("person", 0.9, (10.0 + 30 * k, 50.0, 30.0 + 30 * k, 110.0))
            for k in range(n)
        ]


def _pipeline(counts: Sequence[int]) -> spotcheck.Pipeline:
    frames = [np.full((H, W, 3), 10 * i, dtype=np.uint8) for i in range(len(counts))]
    return spotcheck.Pipeline(frames=lambda: frames, detector=Stub(counts), info=INFO)


def _crops(numbers: Sequence[int]) -> list[spotcheck.ReviewItem]:
    image = np.full((60, 40, 3), 90, dtype=np.uint8)
    return [spotcheck.ReviewItem(n, f"crop-{n:04d}.png", (n,), image) for n in numbers]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    work, tmp = tmp_path / "work", tmp_path / "tmp"
    work.mkdir()
    tmp.mkdir()
    monkeypatch.chdir(work)
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    yield tmp


def _run(args: Sequence[str], out: Path, **kwargs: Any) -> int:
    argv = [*args, "--reviewer", "tester", "--out-dir", str(out)]
    return spotcheck.main(argv, today=DAY, **kwargs)


def _send(*keys: str, extra: Any = None) -> Any:
    """A driver that sends `keys`, 20 ms apart, then calls `extra(root)` if given."""

    def drive(root: Any) -> None:
        pending = list(keys)

        def step() -> None:
            if pending:
                root.focus_force()
                root.event_generate(pending.pop(0))
                root.after(20, step)
            elif extra is not None:
                extra(root)

        root.after(20, step)

    return drive


# The view -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("view", "mode", "judgements", "platform", "expected"),
    [
        (None, "crops", None, "win32", "window"),
        (None, "crops", None, "linux", "files"),
        (None, "frames", None, "win32", "files"),
        (None, "crops", "j.json", "win32", "files"),
        ("files", "crops", None, "win32", "files"),
        ("window", "crops", None, "linux", "window"),
        ("files", "frames", "j.json", "linux", "files"),
    ],
)
def test_resolve_view(
    view: spotcheck.View | None,
    mode: spotcheck.Mode,
    judgements: str | None,
    platform: str,
    expected: str,
) -> None:
    assert spotcheck.resolve_view(view, mode, judgements, platform) == expected


@pytest.mark.parametrize(("mode", "judgements"), [("frames", None), ("crops", "j.json")])
def test_resolve_view_refuses_window_with_frames_or_a_judgements_file(
    mode: spotcheck.Mode, judgements: str | None
) -> None:
    with pytest.raises(spotcheck.SpotcheckError):
        spotcheck.resolve_view("window", mode, judgements, "win32")


def test_window_with_a_judgements_file_is_refused_before_the_sweep(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--n", "1", "--view", "window", "--judgements", str(tmp_path / "j.json")]
    assert _run(args, tmp_path / "out", pipeline=_pipeline([])) == 1
    assert "--judgements" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("height", "width", "scale"),
    [(30, 20, 4), (120, 50, 4), (240, 100, 2), (480, 200, 1), (2000, 100, 1), (100, 1000, 1)],
)
def test_window_scale_is_a_bounded_whole_factor(height: int, width: int, scale: int) -> None:
    assert spotcheck.window_scale(height, width) == scale
    assert width * scale <= max(width, spotcheck.WINDOW_MAX_WIDTH)


def test_png_data_is_the_enlarged_image_in_memory() -> None:
    import base64

    image = np.zeros((5, 3, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(5)[:, None] * 40
    image[:, :, 2] = np.arange(3)[None, :] * 80
    data = spotcheck._png_data(image, 3)
    raw = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (15, 9, 3)
    assert np.array_equal(decoded[::3, ::3], image)


def test_missing_tkinter_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tkinter", None)
    with pytest.raises(spotcheck.SpotcheckError, match="tkinter"):
        spotcheck.WindowReviewer().judge(_crops([1]), "crops", time.monotonic() + WAIT_S)


def test_no_display_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform == "win32":
        pytest.skip("Windows always has a display for Tk")
    pytest.importorskip("tkinter")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(spotcheck.SpotcheckError, match="window"):
        spotcheck.WindowReviewer().judge(_crops([1]), "crops", time.monotonic() + WAIT_S)


# The window ---------------------------------------------------------------------------


def test_window_accepts_keypad_enter_and_capitals_and_ignores_keys_after_the_last() -> None:
    _need_window()
    keys = ["<KP_Enter>", "<KeyPress-N>", "<KeyPress-V>", "<KeyPress-n>", "<KeyPress-q>"]
    reviewer = spotcheck.WindowReviewer(driver=_send(*keys))
    judgements = reviewer.judge(_crops([1, 2, 3]), "crops", time.monotonic() + WAIT_S)
    assert dict(judgements) == {
        1: PEDESTRIAN,
        2: spotcheck.Judgement(frozenset({2}), frozenset(), None),
        3: spotcheck.Judgement(frozenset(), frozenset({3}), None),
    }


@pytest.mark.parametrize("last", ["<Return>", "<KeyPress-q>"])
def test_window_leaves_no_tk_object_for_another_thread_to_free(last: str) -> None:
    """Tcl aborts the process when one of its objects is freed in a thread other than
    the one that made it; the garbage collector can run in any thread. So nothing of the
    window may outlive the review in a reference cycle."""
    _need_window()
    roots: list[weakref.ref[Any]] = []
    send = _send("<Return>", last)

    def drive(root: Any) -> None:
        roots.append(weakref.ref(root))
        send(root)

    gc.disable()  # only the review's own clean-up may free it
    try:
        with contextlib.suppress(spotcheck.ReviewAborted):
            spotcheck.WindowReviewer(driver=drive).judge(
                _crops([1, 2]), "crops", time.monotonic() + WAIT_S
            )
        assert roots and roots[0]() is None
    finally:
        gc.enable()


def test_window_with_nothing_to_review_opens_nothing() -> None:
    reviewer = spotcheck.WindowReviewer(driver=lambda root: pytest.fail("no window"))
    assert reviewer.judge([], "crops", time.monotonic() + WAIT_S) == {}


def test_a_signal_while_the_window_waits_stops_the_tool(env: Path, tmp_path: Path) -> None:
    _need_window()
    before = signal.getsignal(signal.SIGTERM)
    raise_term = _send("<Return>", extra=lambda root: signal.raise_signal(signal.SIGTERM))
    code = _run(
        ["--n", "2", "--min-persons", "1", "--view", "window"],
        tmp_path / "out",
        pipeline=_pipeline([2, 1]),
        reviewer=spotcheck.WindowReviewer(driver=raise_term),
    )
    assert code == 128 + signal.SIGTERM
    assert not (tmp_path / "out").exists()
    assert signal.getsignal(signal.SIGTERM) == before


def test_an_error_in_the_driver_propagates_and_the_window_closes() -> None:
    _need_window()
    roots: list[Any] = []

    def drive(root: Any) -> None:
        roots.append(root)
        raise RuntimeError("driver failed")

    with pytest.raises(RuntimeError, match="driver failed"):
        spotcheck.WindowReviewer(driver=drive).judge(
            _crops([1]), "crops", time.monotonic() + WAIT_S
        )
    import tkinter

    with pytest.raises(tkinter.TclError):
        roots[0].winfo_exists()


# The timer alarm and the threaded reader, on any platform ------------------------------


def _deliver(signum: int) -> None:
    signal.raise_signal(signum)
    time.sleep(0.05)


def test_timer_alarm_raises_the_timeout_and_ctrl_c_stays_an_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spotcheck, "ALARM_SIGNALS", ())  # the Windows path
    guard = spotcheck._SignalGuard()
    guard.install()
    try:
        guard.alarm(0.2)
        started = time.monotonic()
        with pytest.raises(spotcheck.ReviewTimeout):
            for _ in range(WAIT_S * 10):  # short sleeps: Linux wakes only the raising thread
                time.sleep(0.1)
        assert time.monotonic() - started < WAIT_S / 2
    finally:
        guard.restore()
    guard = spotcheck._SignalGuard()
    guard.install()
    try:
        guard.alarm(WAIT_S)
        with pytest.raises(spotcheck.Interrupted) as raised:
            _deliver(signal.SIGINT)
        assert raised.value.signum == signal.SIGINT
    finally:
        guard.restore()


def test_timer_alarm_cancelled_by_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotcheck, "ALARM_SIGNALS", ())
    guard = spotcheck._SignalGuard()
    guard.install()
    try:
        guard.alarm(0.2)
        guard.alarm(0)
        time.sleep(0.5)  # nothing raised
        assert guard._timer is None
    finally:
        guard.restore()


def _judge_over_a_pipe(
    data: bytes, close: bool, items: Sequence[spotcheck.ReviewItem]
) -> dict[int, spotcheck.Judgement] | type[BaseException]:
    """Judge `items` from a pipe holding `data`, closed by its writer if `close`: the
    judgements, or the type of what was raised."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, data)
        if close:
            os.close(write_fd)
            write_fd = -1
        reviewer = spotcheck.KeyboardReviewer(read_fd, out=io.StringIO())
        try:
            return dict(reviewer.judge(items, "crops", time.monotonic() + 0.5))
        except (spotcheck.ReviewTimeout, spotcheck.ReviewAborted) as exc:
            return type(exc)
    finally:
        if write_fd != -1:
            os.close(write_fd)  # first, so that a reading thread sees the end and stops
        time.sleep(0.1)
        os.close(read_fd)


def _check_reader() -> None:
    items = _crops([1, 2, 3])
    assert _judge_over_a_pipe(b"n\nx\nv\n\n", True, items) == {
        1: spotcheck.Judgement(frozenset({1}), frozenset(), None),
        2: spotcheck.Judgement(frozenset(), frozenset({2}), None),
        3: PEDESTRIAN,
    }
    started = time.monotonic()
    assert _judge_over_a_pipe(b"n\n", False, items) is spotcheck.ReviewTimeout
    assert time.monotonic() - started < WAIT_S / 2
    assert _judge_over_a_pipe(b"n\n", True, items) is spotcheck.ReviewAborted


@pytest.mark.skipif(sys.platform == "win32", reason="a stand-in for the Windows primitive")
def test_windows_console_reader_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotcheck, "WINDOWS", True)
    monkeypatch.setattr(spotcheck, "_input_kind", lambda fd: "console")
    _check_reader()


@pytest.mark.skipif(sys.platform == "win32", reason="a stand-in for the Windows primitive")
def test_windows_pipe_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl
    import select
    import termios

    def waiting(fd: int) -> int | None:
        """What PeekNamedPipe tells: bytes waiting, or None once the writer is gone."""
        count = int.from_bytes(fcntl.ioctl(fd, termios.FIONREAD, b"\0\0\0\0"), sys.byteorder)
        if count == 0 and select.select([fd], [], [], 0)[0]:
            return None
        return count

    monkeypatch.setattr(spotcheck, "WINDOWS", True)
    monkeypatch.setattr(spotcheck, "_input_kind", lambda fd: "pipe")
    monkeypatch.setattr(spotcheck, "_pipe_waiting", waiting)
    _check_reader()


def test_the_windows_lock_file_lies_next_to_the_directory_outside_the_prefix(
    tmp_path: Path,
) -> None:
    directory = tmp_path / (spotcheck.TEMP_PREFIX + "abc123")
    lock = spotcheck.lock_file(directory)
    assert lock == tmp_path / f".{spotcheck.TEMP_PREFIX}abc123.lock"
    assert not lock.is_relative_to(directory)
    assert not lock.name.startswith(spotcheck.TEMP_PREFIX)  # never taken for a directory


# Windows only -------------------------------------------------------------------------


@windows_only
def test_windows_lock_file_lies_outside_excludes_a_second_lock_and_is_deleted(
    env: Path,
) -> None:
    directory = spotcheck.ReviewDirectory()
    path = directory.create()
    lock = spotcheck.lock_file(path)
    assert lock.parent == path.parent and lock.is_file()
    assert list(path.iterdir()) == []  # nothing held open inside the directory
    assert spotcheck._lock(path) is None
    assert directory.remove()
    assert not path.exists() and not lock.exists()


@windows_only
def test_windows_lock_refuses_a_file_and_a_missing_path(env: Path) -> None:
    plain = env / (spotcheck.TEMP_PREFIX + "file")
    plain.write_text("x", encoding="utf-8")
    assert spotcheck._lock(plain) is None
    assert spotcheck._lock(env / "missing") is None


@windows_only
def test_windows_image_files_are_binary(env: Path) -> None:
    directory = spotcheck.ReviewDirectory()
    path = directory.create()
    try:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:, :, 1] = 10  # PNG data with newline bytes in it
        item = spotcheck.ReviewItem(1, "crop-0001.png", (1,), image)
        directory.write_image(item)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        assert (path / item.file).read_bytes() == encoded.tobytes()
    finally:
        directory.remove()


@windows_only
def test_windows_handles_ctrl_break() -> None:
    assert getattr(signal, "SIGBREAK") in spotcheck.HANDLED_SIGNALS  # noqa: B009  (Windows only)
    assert {signal.SIGINT, signal.SIGTERM} <= set(spotcheck.HANDLED_SIGNALS)
    assert spotcheck.ALARM_SIGNALS == ()
