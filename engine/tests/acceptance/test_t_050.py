"""Acceptance tests for T-050: with --confirm-stop, a single `q` in the review window asks
before it stops. The task contract: do not edit.

Every image here is synthetic (uniform colours); the window tests read the window's label
texts and image names through Tk, nothing else. The window tests need Tk and a display:
they skip without one, except on Windows or with WEARREPORT_REQUIRE_TK set (on Linux, run
them under `xvfb-run`).
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport.tools import spotcheck

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "spotchecks" / "README.md"
WINDOWS = sys.platform == "win32"
REQUIRE_TK = "WEARREPORT_REQUIRE_TK"
H, W = 288, 352
DAY = datetime.date(2026, 10, 12)
NOON = datetime.datetime(2026, 10, 12, 11, 22, 33, tzinfo=datetime.UTC)
WAIT_S = 60
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
ASK = "Stop and discard this session? Press q again to stop, any other key to continue."
OLD_WINDOW_LEGEND = (
    "Enter or Space: pedestrian   n: not a person   v: person in a vehicle   "
    "u: cannot tell   Backspace: back   q: stop"
)
OLD_ATTRIBUTE_LEGEND = (
    "y: yes   n: no   u: cannot tell   x: not a person or nothing can be told   "
    "Backspace: back   q: stop"
)
PEDESTRIAN = spotcheck.Judgement(frozenset(), frozenset(), None)

Frame = npt.NDArray[np.uint8]


# Helpers ------------------------------------------------------------------------------


def _need_window() -> None:
    try:
        import tkinter

        tkinter.Tk().destroy()
    except Exception as exc:  # ImportError without Tk, TclError without a display
        if WINDOWS or os.environ.get(REQUIRE_TK):
            pytest.fail(f"no Tk window here: {exc}")
        pytest.skip("no Tk display here (on Linux run under xvfb-run)")


# Boxes by height in source-frame pixels: 60 (near field) and 30 (not).
B60 = (110.0, 20.0, 140.0, 80.0)
B60B = (160.0, 20.0, 190.0, 80.0)
B30 = (10.0, 50.0, 30.0, 80.0)


class Stub:
    """boxes[i] are the person boxes in the frame whose colour is 10*i."""

    def __init__(self, boxes: Sequence[Sequence[tuple[float, float, float, float]]]) -> None:
        self.boxes = [list(b) for b in boxes]

    def detect(self, frame: Frame) -> list[detect.Detection]:
        return [detect.Detection("person", 0.9, b) for b in self.boxes[int(frame[0, 0, 0]) // 10]]


def _pipeline(boxes: Sequence[Sequence[tuple[float, float, float, float]]]) -> spotcheck.Pipeline:
    frames = []
    for i in range(len(boxes)):
        frame = np.full((H, W, 3), 10 * i, dtype=np.uint8)
        frame[10:130, :, 1] = np.arange(W, dtype=np.uint8)[None, :]
        frame[0, 0, 0] = 10 * i
        frames.append(frame)
    return spotcheck.Pipeline(frames=lambda: frames, detector=Stub(boxes), info=INFO)


def _crop(number: int) -> spotcheck.ReviewItem:
    image = np.full((60, 40, 3), (10 * number) % 250, dtype=np.uint8)
    return spotcheck.ReviewItem(number, f"crop-{number:04d}.png", (number,), image)


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


def _run(args: Sequence[str], out: Path, **kwargs: Any) -> int:
    argv = [*args, "--reviewer", "tester", "--out-dir", str(out)]
    kwargs.setdefault("clock", lambda: NOON)
    return spotcheck.main(argv, today=DAY, **kwargs)


def _leftovers(tmp: Path) -> list[Path]:
    return sorted(tmp.glob(spotcheck.TEMP_PREFIX + "*"))


class Keys:
    """A window driver: before each key it records the window's label texts in packing
    order and the name of the image shown, then sends the key, 20 ms apart. "CLOSE"
    closes the window as its title-bar button does. A class, not a closure that schedules
    itself (see the T-037 window tests)."""

    def __init__(self, *keys: str) -> None:
        self.pending = list(keys)
        self.seen: list[tuple[list[str], str]] = []
        self.root: Any = None

    def __call__(self, root: Any) -> None:
        self.root = root
        root.after(20, self._step)

    def _step(self) -> None:
        root = self.root
        if not self.pending:
            return
        labels = [w for w in root.pack_slaves() if w.winfo_class() == "Label"]
        texts = [str(w.cget("text")) for w in labels]
        image = next((str(w.cget("image")) for w in labels if str(w.cget("image"))), "")
        self.seen.append((texts, image))
        key = self.pending.pop(0)
        if key == "CLOSE":
            root.tk.call(root.wm_protocol("WM_DELETE_WINDOW"))
        else:
            root.focus_force()
            root.event_generate(key if key.startswith("<") else f"<KeyPress-{key}>")
        root.after(20, self._step)


def _judge(keys: Keys, confirm: bool = True, timeout: float = WAIT_S) -> Any:
    reviewer = spotcheck.WindowReviewer(driver=keys, confirm_stop=confirm)
    items = [_crop(1), _crop(2), _crop(3)]
    return reviewer.judge(items, "crops", time.monotonic() + timeout)


def _label(keys: Keys, confirm: bool = True, timeout: float = WAIT_S) -> Any:
    reviewer = spotcheck.WindowReviewer(driver=keys, confirm_stop=confirm)
    items = [_crop(1), _crop(2)]
    return reviewer.attributes(items, time.monotonic() + timeout)


PEDESTRIAN_WINDOW = ["--n", "2", "--min-persons", "1", "--view", "window"]
ATTRIBUTE_WINDOW = ["--attributes", "--n", "2", "--min-persons", "1", "--view", "window"]


# AC1: the flag ------------------------------------------------------------------------


def test_ac1_the_parser_accepts_confirm_stop_and_it_is_off_by_default() -> None:
    parser = spotcheck.build_parser()
    assert parser.parse_args(["--n", "1"]).confirm_stop is False
    assert parser.parse_args(["--n", "1", "--confirm-stop"]).confirm_stop is True
    assert "--confirm-stop" in parser.format_help()


def test_ac1_legends_are_unchanged() -> None:
    assert spotcheck.WINDOW_LEGEND == OLD_WINDOW_LEGEND
    assert spotcheck.ATTRIBUTE_LEGEND == OLD_ATTRIBUTE_LEGEND


def test_ac1_the_window_reviewer_does_not_ask_by_default() -> None:
    assert spotcheck.WindowReviewer().confirm_stop is False


@pytest.mark.parametrize("args", [PEDESTRIAN_WINDOW, ATTRIBUTE_WINDOW])
def test_ac1_the_flag_reaches_a_given_window_reviewer(
    env: Path, tmp_path: Path, args: list[str]
) -> None:
    _need_window()
    keys = Keys("q", "q")
    reviewer = spotcheck.WindowReviewer(driver=keys)
    out = tmp_path / "out"
    code = _run(
        [*args, "--confirm-stop"], out, pipeline=_pipeline([[B60], [B60B]]), reviewer=reviewer
    )
    assert code == 1
    assert not out.exists()
    assert len(keys.seen) == 2
    assert ASK in keys.seen[1][0]  # the first q asked


def test_ac1_the_flag_is_refused_without_the_window_view(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def frames() -> list[Frame]:
        raise AssertionError("the sweep must not start")

    pipeline = spotcheck.Pipeline(frames=frames, detector=Stub([]), info=INFO)
    out = tmp_path / "out"
    args = ["--n", "2", "--min-persons", "1", "--view", "files", "--confirm-stop"]
    reviewer = spotcheck.WindowReviewer()
    assert _run(args, out, pipeline=pipeline, reviewer=reviewer) == 1
    assert "--confirm-stop" in capsys.readouterr().err
    assert not out.exists()


# AC2 and AC3: the first q asks, the second stops -----------------------------------------


def test_ac2_the_first_q_asks_in_the_header_and_keeps_the_crop() -> None:
    _need_window()
    keys = Keys("Return", "q", "q")
    with pytest.raises(spotcheck.ReviewAborted):
        _judge(keys)
    assert len(keys.seen) == 3
    before, asking = keys.seen[1], keys.seen[2]
    header_before, header_asking = before[0][0], asking[0][0]
    assert "2 of 3" in header_before
    assert header_asking == ASK  # the header line, exactly
    assert asking[0][-1] == OLD_WINDOW_LEGEND  # the legend is untouched
    assert ASK not in asking[0][-1]
    assert asking[1] == before[1]  # the same crop


@pytest.mark.parametrize("q", ["Q", "q"])
def test_ac2_capital_q_asks_too(q: str) -> None:
    _need_window()
    keys = Keys(q, "Q" if q == "q" else "q")
    with pytest.raises(spotcheck.ReviewAborted):
        _label(keys)
    assert keys.seen[1][0][0] == ASK


def test_ac2_the_attribute_window_keeps_the_question_while_asking() -> None:
    _need_window()
    keys = Keys("y", "q", "q")
    with pytest.raises(spotcheck.ReviewAborted):
        _label(keys)
    before, asking = keys.seen[1], keys.seen[2]
    assert asking[0][0] == ASK
    assert asking[0][1:] == before[0][1:]  # the question and the legend are unchanged
    assert "Question 2 of 3" in asking[0][1]
    assert asking[0][-1] == OLD_ATTRIBUTE_LEGEND
    assert asking[1] == before[1]


def test_ac3_q_then_q_stops_the_pedestrian_review_with_nothing_written(
    env: Path, tmp_path: Path
) -> None:
    _need_window()
    keys = Keys("n", "q", "q")
    reviewer = spotcheck.WindowReviewer(driver=keys)
    out = tmp_path / "out"
    args = [*PEDESTRIAN_WINDOW, "--confirm-stop"]
    assert _run(args, out, pipeline=_pipeline([[B60], [B60B]]), reviewer=reviewer) == 1
    assert not out.exists()
    assert _leftovers(env) == []


def test_ac3_q_then_q_stops_the_attribute_session_with_nothing_written(
    env: Path, tmp_path: Path
) -> None:
    _need_window()
    keys = Keys("y", "n", "q", "q")
    reviewer = spotcheck.WindowReviewer(driver=keys)
    out = tmp_path / "out"
    args = [*ATTRIBUTE_WINDOW, "--confirm-stop"]
    assert _run(args, out, pipeline=_pipeline([[B60], [B60B]]), reviewer=reviewer) == 1
    assert not out.exists()


# AC4: any other key continues ---------------------------------------------------------


def test_ac4_q_then_y_continues_and_y_is_ignored_in_the_attribute_session(
    env: Path, tmp_path: Path
) -> None:
    _need_window()
    # Crop 1: y, then q y (asked, y ignored), then n u: "ynu". Crop 2: x (rejected).
    keys = Keys("y", "q", "y", "n", "u", "x")
    reviewer = spotcheck.WindowReviewer(driver=keys)
    out = tmp_path / "out"
    args = [*ATTRIBUTE_WINDOW, "--confirm-stop"]
    assert _run(args, out, pipeline=_pipeline([[B60, B30], [B60B]]), reviewer=reviewer) == 0
    record = json.loads((out / "attributes" / f"{DAY.isoformat()}.json").read_text("utf-8"))
    assert record["crops_shown"] == 2 and record["crops_rejected"] == 1
    assert sorted(record["crops"]) == [[60, "ynu", None]]
    # After the ignored y the question is hidden and the same question is back.
    before_q, asking, after = keys.seen[1], keys.seen[2], keys.seen[3]
    assert asking[0][0] == ASK
    assert after[0] == before_q[0]
    assert after[1] == before_q[1]


def test_ac4_q_then_a_bound_key_is_ignored_in_the_pedestrian_review(
    env: Path, tmp_path: Path
) -> None:
    _need_window()
    # q n: asked, n ignored. Then v (crop 1), Return (crop 2).
    keys = Keys("q", "n", "v", "Return")
    reviewer = spotcheck.WindowReviewer(driver=keys)
    out = tmp_path / "out"
    args = [*PEDESTRIAN_WINDOW, "--confirm-stop"]
    assert _run(args, out, pipeline=_pipeline([[B60], [B60B]]), reviewer=reviewer) == 0
    stats = json.loads((out / f"{DAY.isoformat()}.json").read_text(encoding="utf-8"))
    assert stats["boxes_shown"] == 2
    assert stats["boxes_in_vehicle"] == 1 and stats["boxes_not_person"] == 0


@pytest.mark.parametrize("other", ["a", "<BackSpace>", "<Return>", "x", "u"])
def test_ac4_any_other_key_hides_the_question_and_changes_nothing(other: str) -> None:
    _need_window()
    # Crop 1 "yn" so far; q, then the other key (ignored); then u; crop 2: n n n.
    keys = Keys("y", "n", "q", other, "u", "n", "n", "n")
    got = _label(keys)
    assert dict(got) == {1: "ynu", 2: "nnn"}
    before_q, after = keys.seen[2], keys.seen[4]
    assert keys.seen[3][0][0] == ASK
    assert after == before_q
    assert ASK not in " ".join(after[0])


def test_ac4_backspace_while_asking_does_not_go_back() -> None:
    _need_window()
    keys = Keys("n", "q", "<BackSpace>", "v", "<Return>")
    got = _judge(keys)
    assert dict(got) == {
        1: spotcheck.Judgement(frozenset({1}), frozenset(), None),
        2: spotcheck.Judgement(frozenset(), frozenset({2}), None),
        3: PEDESTRIAN,
    }


def test_ac4_q_after_a_dismissed_question_asks_again() -> None:
    _need_window()
    keys = Keys("q", "a", "q", "q")
    with pytest.raises(spotcheck.ReviewAborted):
        _judge(keys)
    assert [seen[0][0] == ASK for seen in keys.seen] == [False, True, False, True]


def test_ac4_a_shift_press_does_not_dismiss_the_question() -> None:
    _need_window()
    # Q typed with Shift: the Shift press is not "another key".
    keys = Keys("q", "<KeyPress-Shift_L>", "Q")
    with pytest.raises(spotcheck.ReviewAborted):
        _judge(keys)
    assert keys.seen[2][0][0] == ASK


# AC5: closing the window and the timeout still stop at once ---------------------------


@pytest.mark.parametrize("first", [[], ["q"]])
def test_ac5_closing_the_window_stops_at_once(first: list[str]) -> None:
    _need_window()
    keys = Keys(*first, "CLOSE")
    with pytest.raises(spotcheck.ReviewAborted):
        _judge(keys)
    with pytest.raises(spotcheck.ReviewAborted):
        _label(Keys(*first, "CLOSE"))


def test_ac5_the_timeout_stops_at_once_even_while_asking() -> None:
    _need_window()
    started = time.monotonic()
    with pytest.raises(spotcheck.ReviewTimeout):
        _judge(Keys("q"), timeout=0.5)
    with pytest.raises(spotcheck.ReviewTimeout):
        _label(Keys("q"), timeout=0.5)
    assert time.monotonic() - started < 10


# Unchanged without the flag -----------------------------------------------------------


def test_without_the_flag_a_single_q_stops(env: Path, tmp_path: Path) -> None:
    _need_window()
    keys = Keys("n", "q")
    with pytest.raises(spotcheck.ReviewAborted):
        _judge(keys, confirm=False)
    assert all(ASK not in " ".join(texts) for texts, _image in keys.seen)
    with pytest.raises(spotcheck.ReviewAborted):
        _label(Keys("y", "q"), confirm=False)
    for args in (PEDESTRIAN_WINDOW, ATTRIBUTE_WINDOW):
        out = tmp_path / "out"
        reviewer = spotcheck.WindowReviewer(driver=Keys("q"))
        assert _run(args, out, pipeline=_pipeline([[B60], [B60B]]), reviewer=reviewer) == 1
        assert not out.exists()


def test_without_the_flag_the_window_shows_the_same_labels() -> None:
    _need_window()
    keys = Keys("Return", "Return", "Return")
    _judge(keys, confirm=False)
    for k, (texts, _image) in enumerate(keys.seen, start=1):
        assert texts[0].startswith(f"Image {k} (") and texts[0].endswith(f": {k} of 3")
        assert texts[-1] == OLD_WINDOW_LEGEND


# AC6: the docs ------------------------------------------------------------------------


def test_ac6_readme_documents_confirm_stop_next_to_both_key_tables() -> None:
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    # The window key tables: their `q` rows, in the pedestrian review and the attribute session.
    q_rows = [
        i
        for i, line in enumerate(lines)
        if line.startswith("| `q`") and "closing the window" in line
    ]
    assert len(q_rows) == 2, "expected a q row in both window key tables"
    for row in q_rows:
        nearby = "\n".join(lines[max(0, row - 12) : row + 12])
        assert "--confirm-stop" in nearby, f"no --confirm-stop near line {row + 1}"
    assert ASK in text
    options = [line for line in lines if line.startswith("| `--confirm-stop`")]
    assert len(options) == 1  # in the options table too
