"""Unit tests for the spot-check tool (T-008), including its runtime privacy test.

Every frame here is synthetic (uniform colour) or one of the licensed fixture photos
served by the fake camera server. Rendered files are only counted, sized and scanned for
magic bytes; no test looks at one.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import re
import signal
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

H, W = 288, 352
DAY = datetime.date(2026, 9, 25)
WAIT_S = 60
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"BM", b"RIFF", b"II*\x00")
SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG", b"/9j/", b"iVBORw0KGgo")
REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"


def _box(k: int) -> tuple[float, float, float, float]:
    return (10.0 + 30 * k, 50.0, 30.0 + 30 * k, 110.0)


class Stub:
    """counts[i] people in the frame whose colour is 10*i."""

    def __init__(self, counts: Sequence[int]) -> None:
        self.counts = list(counts)

    def detect(self, frame: npt.NDArray[np.uint8]) -> list[detect.Detection]:
        n = self.counts[int(frame[0, 0, 0]) // 10]
        return [detect.Detection("person", 0.9, _box(k)) for k in range(n)]


def _pipeline(counts: Sequence[int]) -> spotcheck.Pipeline:
    frames = [np.full((H, W, 3), 10 * i, dtype=np.uint8) for i in range(len(counts))]
    return spotcheck.Pipeline(frames=lambda: frames, detector=Stub(counts), info=INFO)


def _ok(item: spotcheck.ReviewItem, mode: str) -> spotcheck.Judgement:
    return spotcheck.Judgement(frozenset(), frozenset(), 0 if mode == "frames" else None)


class Scripted:
    def __init__(
        self, decide: Callable[[spotcheck.ReviewItem, str], spotcheck.Judgement] = _ok
    ) -> None:
        self.decide = decide
        self.calls = 0

    def judge(
        self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
    ) -> Mapping[int, spotcheck.Judgement]:
        self.calls += 1
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


def _leftovers(tmp: Path) -> list[Path]:
    return sorted(tmp.glob(spotcheck.TEMP_PREFIX + "*"))


def _run(args: Sequence[str], out: Path, **kwargs: Any) -> int:
    argv = [*args, "--reviewer", "tester", "--out-dir", str(out)]
    return spotcheck.main(argv, today=DAY, **kwargs)


def _items(mode: spotcheck.Mode) -> list[spotcheck.ReviewItem]:
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    if mode == "frames":
        return [
            spotcheck.ReviewItem(number=1, file="frame-0001.png", boxes=(1, 2, 3), image=blank),
            spotcheck.ReviewItem(number=2, file="frame-0002.png", boxes=(4,), image=blank),
            spotcheck.ReviewItem(number=3, file="frame-0003.png", boxes=(), image=blank),
        ]
    return [
        spotcheck.ReviewItem(number=n, file=f"crop-{n:04d}.png", boxes=(n,), image=blank)
        for n in (1, 2)
    ]


# Judgements ---------------------------------------------------------------------------

FRAMES_OK = {"1": {"missed": 0}, "2": {"missed": 0}, "3": {"missed": 0}}


def _frames(**changes: Any) -> bytes:
    data: dict[str, Any] = {k: dict(v) for k, v in FRAMES_OK.items()}
    for key, value in changes.items():
        data[key.removeprefix("i")] = value
    return json.dumps(data).encode()


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"null",
        b'"text"',
        b"\xef\xbb\xbf{}",  # UTF-8 byte-order mark
        b'{"1": {"missed": 0}, "2": {"missed": 0}, "3": {"missed": 0},}',
        _frames(i1=None),
        _frames(i1=[]),
        _frames(i1={"missed": 1e999}),
        _frames(i1={"missed": -0.0}),
        _frames(i1={"missed": spotcheck.MAX_MISSED + 1}),
        _frames(i1={"missed": 0, "not_person": [1, 1]}),
        _frames(i1={"missed": 0, "not_person": [1.0]}),
        _frames(i1={"missed": 0, "not_person": [None]}),
        _frames(i1={"missed": 0, "not_person": {"1": 1}}),
        _frames(i1={"missed": 0, "in_vehicle": [0]}),
        _frames(i1={"missed": 0, "in_vehicle": [-1]}),
        _frames(i1={"missed": 0, "in_vehicle": [10**30]}),
        _frames(i3={"missed": 0, "not_person": [1]}),  # image 3 has no boxes
        _frames(**{"i01": {"missed": 0}}),
        _frames(**{"i 1": {"missed": 0}}),
        _frames(**{"i1.0": {"missed": 0}}),
        _frames(**{"i0": {"missed": 0}}),
        b'{"1": {"missed": 0}, "1": {"missed": 0}, "2": {"missed": 0}}',  # 3 missing
        b'{"1": {"missed": 0, "missed": 0}, "2": {"missed": 0}, "3": {"missed": Infinity}}',
        b'{"1": {"missed": 0}, "2": {"missed": 0}, "3": {"missed": -Infinity}}',
        b'{"1": ' + b"[" * 50_000 + b"]" * 50_000 + b"}",
        b"{" + b" " * (spotcheck.MAX_JUDGEMENTS_BYTES + 1) + b"}",
        b"\x00" * 64,
        b"\xff\xd8\xff\xe0",  # a JPEG header is not a judgements file
    ],
)
def test_parse_judgements_rejects_hostile_input(raw: bytes) -> None:
    with pytest.raises(spotcheck.JudgementError):
        spotcheck.parse_judgements(raw, _items("frames"), "frames")


def test_parse_judgements_messages_do_not_echo_long_input() -> None:
    key = "x" * 10_000
    with pytest.raises(spotcheck.JudgementError) as exc:
        spotcheck.parse_judgements(json.dumps({key: {}}).encode(), _items("frames"), "frames")
    assert len(str(exc.value)) < 200


def test_parse_judgements_accepts_empty_lists_and_frames_without_boxes() -> None:
    raw = _frames(i1={"not_person": [], "in_vehicle": [1, 3], "missed": 2})
    got = spotcheck.parse_judgements(raw, _items("frames"), "frames")
    assert got[1] == spotcheck.Judgement(frozenset(), frozenset({1, 3}), 2)
    assert got[3] == spotcheck.Judgement(frozenset(), frozenset(), 0)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", (set(), set(), 0)),
        ("  ", (set(), set(), 0)),
        ("n1,v2", ({1}, {2}, 0)),
        ("N1 V3 M4", ({1}, {3}, 4)),
        ("m0", (set(), set(), 0)),
        ("n1 n2 n3", ({1, 2, 3}, set(), 0)),
    ],
)
def test_parse_line_frames(line: str, expected: tuple[set[int], set[int], int]) -> None:
    item = _items("frames")[0]
    got = spotcheck.parse_line(line, item, "frames")
    assert got == spotcheck.Judgement(frozenset(expected[0]), frozenset(expected[1]), expected[2])


@pytest.mark.parametrize(
    "line",
    ["n4", "n0", "v", "m1 m2", "m", "m1001", "n1x", "1", "n 1", "n1;v2", "q1", "n9999999", "é"],
)
def test_parse_line_frames_rejects(line: str) -> None:
    with pytest.raises(spotcheck.JudgementError):
        spotcheck.parse_line(line, _items("frames")[0], "frames")


def test_parse_line_crops_bare_letters_mean_this_crop() -> None:
    crop = _items("crops")[1]
    assert spotcheck.parse_line("v", crop, "crops").in_vehicle == frozenset({2})
    assert spotcheck.parse_line("n", crop, "crops").not_person == frozenset({2})
    assert spotcheck.parse_line("", crop, "crops") == spotcheck.Judgement(
        frozenset(), frozenset(), None
    )
    for bad in ("n v", "m0", "n1"):
        with pytest.raises(spotcheck.JudgementError):
            spotcheck.parse_line(bad, crop, "crops")


def test_validate_rejects_wrong_types_from_a_reviewer() -> None:
    item = _items("frames")[0]
    bad = [
        spotcheck.Judgement(frozenset({True}), frozenset(), 0),
        spotcheck.Judgement({1}, frozenset(), 0),  # type: ignore[arg-type]
        spotcheck.Judgement(frozenset(), frozenset(), None),
        spotcheck.Judgement(frozenset(), frozenset(), True),
        spotcheck.Judgement(frozenset(), frozenset(), 1.0),  # type: ignore[arg-type]
    ]
    for judgement in bad:
        with pytest.raises(spotcheck.JudgementError):
            spotcheck.validate(judgement, item, "frames")


# Rendering and sampling ---------------------------------------------------------------


@pytest.mark.parametrize(
    "box",
    [
        (0.0, 0.0, 1.0, 1.0),
        (351.0, 287.0, 352.0, 288.0),
        (0.0, 0.0, 352.0, 288.0),
        (5.5, 6.5, 5.75, 6.75),
    ],
)
def test_render_crop_handles_edges_and_tiny_boxes(box: tuple[float, float, float, float]) -> None:
    frame = np.full((H, W, 3), 40, dtype=np.uint8)
    image = spotcheck.render_crop(frame, detect.Detection("person", 0.9, box), 123)
    x1, y1, x2, y2 = spotcheck.crop_bounds(box, W, H)
    height, width = image.shape[:2]
    assert height % (y2 - y1) == 0 and width % (x2 - x1) == 0
    assert height // (y2 - y1) == width // (x2 - x1) <= spotcheck.MAX_CROP_SCALE
    assert image.dtype == np.uint8 and image.shape[2] == 3


@pytest.mark.parametrize(
    "box",
    [
        (float("nan"), 0.0, 1.0, 1.0),
        (5.0, 5.0, 5.0, 9.0),
        (9.0, 5.0, 5.0, 9.0),
        (400.0, 0.0, 410.0, 5.0),
    ],
)
def test_crop_bounds_rejects_bad_boxes(box: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        spotcheck.crop_bounds(box, W, H)


def test_render_frame_keeps_large_frames_at_their_size() -> None:
    frame = np.zeros((823, 1024, 3), dtype=np.uint8)
    person = detect.Detection("person", 0.9, (10.0, 10.0, 50.0, 90.0))
    assert spotcheck.render_frame(frame, [person], [7]).shape == (823, 1024, 3)


def test_render_numbers_boxes_across_the_check() -> None:
    samples = [
        spotcheck.Sample(
            np.zeros((H, W, 3), np.uint8), tuple(Stub([n]).detect(np.zeros((H, W, 3), np.uint8)))
        )
        for n in (2, 0, 3)
    ]
    frames = spotcheck.render(samples, "frames")
    assert [item.boxes for item in frames] == [(1, 2), (), (3, 4, 5)]
    assert [item.file for item in frames] == ["frame-0001.png", "frame-0002.png", "frame-0003.png"]
    crops = spotcheck.render(samples, "crops")
    assert [(item.number, item.boxes, item.file) for item in crops] == [
        (n, (n,), f"crop-{n:04d}.png") for n in range(1, 6)
    ]


def test_sample_can_pick_every_qualifying_frame_and_keeps_order() -> None:
    counts = [3, 0, 3, 3, 1, 3, 3]
    frames = [np.full((H, W, 3), 10 * i, dtype=np.uint8) for i in range(len(counts))]
    picked: set[int] = set()
    for seed in range(60):
        chosen = [
            int(s.frame[0, 0, 0]) // 10
            for s in spotcheck.sample(frames, Stub(counts), n=2, min_persons=3, seed=seed)
        ]
        assert len(chosen) == 2 and chosen == sorted(chosen)
        picked.update(chosen)
    assert picked == {0, 2, 3, 5, 6}


def test_sample_ignores_umbrellas() -> None:
    class Umbrellas:
        def detect(self, frame: npt.NDArray[np.uint8]) -> list[detect.Detection]:
            return [detect.Detection("umbrella", 0.9, _box(k)) for k in range(5)]

    frames = [np.zeros((H, W, 3), dtype=np.uint8)]
    assert spotcheck.sample(frames, Umbrellas(), n=1, min_persons=1, seed=0) == []


# Statistics ---------------------------------------------------------------------------


def test_compute_stats_rounds_and_keeps_counts() -> None:
    items = _items("frames")
    judgements = {
        1: spotcheck.Judgement(frozenset({1}), frozenset(), 0),
        2: spotcheck.Judgement(frozenset(), frozenset(), 0),
        3: spotcheck.Judgement(frozenset(), frozenset(), 0),
    }
    stats = spotcheck.compute_stats(
        items, judgements, mode="frames", frames_reviewed=3, reviewer="r", info=INFO, day=DAY
    )
    assert stats["precision_person"] == 0.75
    assert stats["recall_estimate"] == 1.0
    judgements[1] = spotcheck.Judgement(frozenset({1}), frozenset({2}), 0)
    stats = spotcheck.compute_stats(
        items[:1], judgements, mode="frames", frames_reviewed=1, reviewer="r", info=INFO, day=DAY
    )
    assert stats["precision_person"] == 0.6667 and stats["precision_pedestrian"] == 0.3333


def test_write_stats_failure_keeps_the_numbers_in_the_message(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(spotcheck.SpotcheckError) as exc:
        spotcheck.write_stats({"boxes_shown": 12345}, blocker / "out", DAY)
    assert "12345" in str(exc.value)


def test_stats_failure_after_review_leaves_nothing(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def full(*args: Any, **kwargs: Any) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(spotcheck, "_write_new", full)
    assert (
        _run(
            ["--n", "1", "--min-persons", "1"],
            tmp_path / "o",
            pipeline=_pipeline([3]),
            reviewer=Scripted(),
        )
        == 1
    )
    assert _leftovers(env) == []
    err = capsys.readouterr().err
    assert "No space left" in err and '"boxes_shown": 3' in err


# Refusals before any work -------------------------------------------------------------


class Untouchable:
    def __call__(self) -> list[npt.NDArray[np.uint8]]:
        raise AssertionError("the pipeline must not run")


def _untouchable() -> spotcheck.Pipeline:
    return spotcheck.Pipeline(frames=Untouchable(), detector=Stub([]), info=INFO)


def test_ci_refusal_precedes_argument_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert spotcheck.main(["--no-such-option"], pipeline=_untouchable()) == 2


def test_unwritable_out_dir_is_refused_before_the_sweep(env: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert _run(["--n", "1"], blocker / "out", pipeline=_untouchable(), reviewer=Scripted()) == 1
    assert list(env.iterdir()) == []


@pytest.mark.parametrize("name", ["", "../x", "a" * 65, "x\ny", "-x", "é"])
def test_reviewer_name_is_restricted(name: str) -> None:
    with pytest.raises(SystemExit):
        spotcheck.build_parser().parse_args(["--n", "1", "--reviewer", name])


@pytest.mark.parametrize("value", ["nan", "inf", "-1", str(spotcheck.MAX_TIMEOUT_S + 1)])
def test_timeout_is_bounded(value: str) -> None:
    with pytest.raises(SystemExit):
        spotcheck.build_parser().parse_args(["--n", "1", "--timeout", value])


def test_model_must_be_a_pinned_name() -> None:
    with pytest.raises(SystemExit):
        spotcheck.build_parser().parse_args(["--n", "1", "--model", "../../evil.onnx"])


def test_dry_run_refuses_the_default_out_dir_spelled_differently(env: Path) -> None:
    args = ["--dry-run", "--n", "1", "--reviewer", "t", "--out-dir", "./spotchecks/"]
    assert spotcheck.main(args, reviewer=Scripted()) == 1


# In-process review paths --------------------------------------------------------------


def test_invalid_judgement_from_a_reviewer_is_an_error(env: Path, tmp_path: Path) -> None:
    def bad(item: spotcheck.ReviewItem, mode: str) -> spotcheck.Judgement:
        return spotcheck.Judgement(frozenset({99}), frozenset(), None)

    assert (
        _run(
            ["--n", "1", "--min-persons", "1"],
            tmp_path / "o",
            pipeline=_pipeline([2]),
            reviewer=Scripted(bad),
        )
        == 1
    )
    assert _leftovers(env) == []
    assert not (tmp_path / "o").exists()


def test_missing_judgement_from_a_reviewer_is_an_error(env: Path, tmp_path: Path) -> None:
    class Partial:
        def judge(
            self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
        ) -> dict[int, spotcheck.Judgement]:
            return {}

    assert (
        _run(
            ["--n", "1", "--min-persons", "1"],
            tmp_path / "o",
            pipeline=_pipeline([2]),
            reviewer=Partial(),
        )
        == 1
    )
    assert _leftovers(env) == []


def test_alarm_stops_a_reviewer_that_ignores_its_deadline(env: Path, tmp_path: Path) -> None:
    class Hang:
        def judge(
            self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
        ) -> dict[int, spotcheck.Judgement]:
            time.sleep(WAIT_S)
            return {}

    started = time.monotonic()
    code = _run(
        ["--n", "1", "--min-persons", "1", "--timeout", "0.5"],
        tmp_path / "o",
        pipeline=_pipeline([2]),
        reviewer=Hang(),
    )
    assert code == 3
    assert time.monotonic() - started < WAIT_S / 2
    assert _leftovers(env) == []
    assert signal.getsignal(signal.SIGALRM) in (signal.SIG_DFL, signal.SIG_IGN, None) or callable(
        signal.getsignal(signal.SIGALRM)
    )
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_signal_handlers_are_restored(env: Path, tmp_path: Path) -> None:
    before = {s: signal.getsignal(s) for s in (*spotcheck.HANDLED_SIGNALS, signal.SIGALRM)}
    assert (
        _run(
            ["--n", "1", "--min-persons", "1"],
            tmp_path / "o",
            pipeline=_pipeline([2]),
            reviewer=Scripted(),
        )
        == 0
    )
    assert {s: signal.getsignal(s) for s in before} == before


def _deliver(signum: int) -> None:
    os.kill(os.getpid(), signum)
    time.sleep(0.05)  # the Python-level handler runs at the latest here


def test_signal_guard_raises_only_the_first_signal() -> None:
    guard = spotcheck._SignalGuard()
    guard.install()
    try:
        with pytest.raises(spotcheck.Interrupted) as raised:
            _deliver(signal.SIGINT)
        assert raised.value.signum == signal.SIGINT
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
            _deliver(signum)  # dropped: clean-up is under way
        with guard.critical():
            _deliver(signal.SIGTERM)
        # and nothing is held back to be raised later
    finally:
        guard.restore()


def test_signal_guard_holds_a_signal_inside_critical_then_latches() -> None:
    guard = spotcheck._SignalGuard()
    guard.install()
    try:
        with pytest.raises(spotcheck.Interrupted) as raised, guard.critical():
            _deliver(signal.SIGTERM)
            _deliver(signal.SIGINT)
        assert raised.value.signum == signal.SIGTERM  # the first one, once done
        _deliver(signal.SIGHUP)  # dropped
    finally:
        guard.restore()


def test_signal_guard_stop_drops_every_signal() -> None:
    guard = spotcheck._SignalGuard()
    guard.install()
    try:
        guard.stop()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            _deliver(signum)
    finally:
        guard.restore()


def test_encoding_failure_while_writing_leaves_nothing(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []
    real = spotcheck.ReviewDirectory.write_image

    def write_image(self: spotcheck.ReviewDirectory, item: spotcheck.ReviewItem) -> None:
        if written:
            raise RuntimeError("encoder broke")
        real(self, item)
        written.append(item.file)

    monkeypatch.setattr(spotcheck.ReviewDirectory, "write_image", write_image)
    with pytest.raises(RuntimeError, match="encoder broke"):
        _run(
            ["--n", "1", "--min-persons", "1"],
            tmp_path / "o",
            pipeline=_pipeline([3]),
            reviewer=Scripted(),
        )
    assert written and _leftovers(env) == []


def test_write_image_refuses_names_outside_the_directory(env: Path) -> None:
    directory = spotcheck.ReviewDirectory()
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        directory.create()
        for name in ("../escape.png", "/tmp/x.png", ".hidden.png", "sub/x.png"):  # noqa: S108
            with pytest.raises(ValueError):
                directory.write_image(spotcheck.ReviewItem(1, name, (1,), blank))
        item = spotcheck.ReviewItem(1, "crop-0001.png", (1,), blank)
        directory.write_image(item)
        with pytest.raises(FileExistsError):
            directory.write_image(item)  # O_EXCL: never overwrites, never follows a link
        assert directory.path is not None
        mode = os.stat(directory.path / item.file).st_mode & 0o777
        assert mode == 0o600
    finally:
        assert directory.remove()
    assert _leftovers(env) == []
    assert not (env.parent / "escape.png").exists()


def test_write_image_needs_a_directory() -> None:
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        spotcheck.ReviewDirectory().write_image(spotcheck.ReviewItem(1, "a.png", (1,), blank))


def test_keyboard_reviewer_over_a_pipe() -> None:
    read_fd, write_fd = os.pipe()
    out = io.StringIO()
    try:
        os.write(write_fd, b"x\nn9\n" + b"n" * (spotcheck.MAX_LINE_BYTES + 10) + b"\nn1 m2\nv4\n\n")
        os.close(write_fd)
        got = spotcheck.KeyboardReviewer(read_fd, out).judge(
            _items("frames"), "frames", time.monotonic() + 10
        )
    finally:
        os.close(read_fd)
    assert got == {
        1: spotcheck.Judgement(frozenset({1}), frozenset(), 2),
        2: spotcheck.Judgement(frozenset(), frozenset({4}), 0),
        3: spotcheck.Judgement(frozenset(), frozenset(), 0),
    }
    assert out.getvalue().count("rejected") == 3


def test_keyboard_reviewer_times_out_and_stops_at_end_of_input() -> None:
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(spotcheck.ReviewTimeout):
            spotcheck.KeyboardReviewer(read_fd, io.StringIO()).judge(
                _items("frames"), "frames", time.monotonic() + 0.3
            )
        os.write(write_fd, b"\n")
        os.close(write_fd)
        with pytest.raises(spotcheck.ReviewAborted):
            spotcheck.KeyboardReviewer(read_fd, io.StringIO()).judge(
                _items("frames"), "frames", time.monotonic() + 5
            )
    finally:
        os.close(read_fd)


def test_keyboard_q_stops_the_review() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"q\n")
    os.close(write_fd)
    try:
        with pytest.raises(spotcheck.ReviewAborted):
            spotcheck.KeyboardReviewer(read_fd, io.StringIO()).judge(
                _items("crops"), "crops", time.monotonic() + 5
            )
    finally:
        os.close(read_fd)


def _json_reviewer(path: Path) -> tuple[spotcheck.JsonFileReviewer, io.StringIO]:
    out = io.StringIO()
    return spotcheck.JsonFileReviewer(path, out), out


@pytest.mark.parametrize("kind", ["fifo", "directory", "device", "too_large"])
def test_json_reviewer_rejects_paths_that_are_not_small_files(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "judgements.json"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    elif kind == "device":
        path.symlink_to("/dev/zero")
    else:
        path.write_bytes(b" " * (spotcheck.MAX_JUDGEMENTS_BYTES + 5))
    reviewer, out = _json_reviewer(path)
    started = time.monotonic()
    with pytest.raises(spotcheck.ReviewTimeout):
        reviewer.judge(_items("crops"), "crops", time.monotonic() + 1)
    assert time.monotonic() - started < 10
    assert out.getvalue().count("rejected") == 1  # said once, not on every poll


def test_json_reviewer_retries_after_a_partial_write(tmp_path: Path) -> None:
    path = tmp_path / "judgements.json"
    reviewer, out = _json_reviewer(path)

    def write() -> None:
        path.write_text('{"1": {', encoding="utf-8")
        time.sleep(3 * spotcheck.JSON_POLL_S)
        path.write_text('{"1": {}, "2": {"not_person": [2]}}', encoding="utf-8")

    thread = threading.Thread(target=write)
    thread.start()
    got = reviewer.judge(_items("crops"), "crops", time.monotonic() + 20)
    thread.join()
    assert got[2].not_person == frozenset({2})
    assert "rejected" in out.getvalue()


# Stale directories --------------------------------------------------------------------


def test_remove_stale_is_narrow(tmp_path: Path) -> None:
    now = time.time()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("x", encoding="utf-8")
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    stale = tmp / (spotcheck.TEMP_PREFIX + "stale")
    stale.mkdir()
    (stale / "sub").mkdir()
    (stale / "sub" / "f").write_text("x", encoding="utf-8")
    locked = tmp / (spotcheck.TEMP_PREFIX + "locked")
    locked.mkdir()
    link = tmp / (spotcheck.TEMP_PREFIX + "link")
    link.symlink_to(outside)
    plain = tmp / (spotcheck.TEMP_PREFIX + "file")
    plain.write_text("x", encoding="utf-8")
    other = tmp / "other-dir"
    other.mkdir()
    old = now - 7200
    for p in (stale, locked, plain, other):
        os.utime(p, (old, old))
    os.utime(link, (old, old), follow_symlinks=False)
    fd = spotcheck._lock(locked)
    assert fd is not None
    try:
        assert spotcheck.remove_stale(tmp, 3600, now) == 1
    finally:
        os.close(fd)
    assert not stale.exists()
    assert locked.is_dir() and link.is_symlink() and plain.is_file() and other.is_dir()
    assert (outside / "keep").is_file()
    assert spotcheck.remove_stale(tmp, 3600, now) == 1  # unlocked now
    assert spotcheck.remove_stale(tmp_path / "missing", 1) == 0


# Child processes: every exit path ----------------------------------------------------

CHILD = r"""
import json, os, shutil, signal, sys, tempfile, time
import numpy as np
from wearreport import detect
from wearreport.tools import spotcheck as sc

counts = json.loads(sys.argv[1])
scenario = sys.argv[2]

class Stub:
    def detect(self, frame):
        n = counts[int(frame[0, 0, 0]) // 10]
        return [detect.Detection("person", 0.9, (10.0 + 30 * k, 50.0, 30.0 + 30 * k, 110.0))
                for k in range(n)]

frames = [np.full((288, 352, 3), 10 * i, dtype=np.uint8) for i in range(len(counts))]
pipeline = sc.Pipeline(frames=lambda: frames, detector=Stub(),
                       info=sc.DetectorInfo(model="stub", sha256="0" * 64, conf=0.35))

class Ok:
    def judge(self, items, mode, deadline):
        return {i.number: sc.Judgement(frozenset(), frozenset(), 0 if mode == "frames" else None)
                for i in items}

def kill_self(signum):
    os.kill(os.getpid(), signum)
    time.sleep(0.2)  # the handler runs here, or when the critical step ends

reviewer = Ok()
if scenario in ("json", "keyboard"):
    reviewer = None
elif scenario.startswith("while_writing_"):
    real_write = sc.ReviewDirectory.write_image
    def write_image(self, item):
        real_write(self, item)
        kill_self(getattr(signal, scenario.removeprefix("while_writing_")))
    sc.ReviewDirectory.write_image = write_image
elif scenario == "in_mkdtemp":
    real_mkdtemp = tempfile.mkdtemp
    def mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        kill_self(signal.SIGINT)
        return path
    tempfile.mkdtemp = mkdtemp
elif scenario == "in_rmtree":
    real_rmtree = shutil.rmtree
    def rmtree(path, *args, **kwargs):
        kill_self(signal.SIGTERM)
        return real_rmtree(path, *args, **kwargs)
    shutil.rmtree = rmtree
elif scenario == "parser_breaks":
    reviewer = None
    def broken(*args, **kwargs):
        raise RuntimeError("parser broke")
    sc.parse_judgements = broken
elif scenario == "stats_fail":
    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")
    sc._write_new = full
elif scenario.startswith("at_cleanup:"):
    # at_cleanup:<first>:<burst>. The reviewer is stopped by <first> ("-": it returns),
    # then each burst signal is sent to this process every time critical() is called
    # from then on, so it lands as clean-up starts, before critical() can defer it.
    _, first, burst = scenario.split(":")
    burst = [getattr(signal, name) for name in burst.split(",")]
    reviewing = []
    class Stopped(Ok):
        def judge(self, items, mode, deadline):
            reviewing.append(True)
            if first != "-":
                kill_self(getattr(signal, first))
            return super().judge(items, mode, deadline)
    reviewer = Stopped()
    real_critical = sc._SignalGuard.critical
    def critical(self):
        if reviewing:
            for signum in burst:
                os.kill(os.getpid(), signum)
        return real_critical(self)
    sc._SignalGuard.critical = critical
elif scenario == "signal_in_handler":
    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")
    sc._write_new = full
    class Stderr:
        def __init__(self, real):
            self.real, self.sent = real, False
        def write(self, text):
            if not self.sent:
                self.sent = True
                kill_self(signal.SIGTERM)
            return self.real.write(text)
        def flush(self):
            self.real.flush()
    sys.stderr = Stderr(sys.stderr)
sys.exit(sc.main(sys.argv[3:], pipeline=pipeline, reviewer=reviewer))
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
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        for d in _leftovers(tmp):
            if (d / spotcheck.NUMBERING_FILE).is_file():
                time.sleep(0.2)
                return d
        assert proc.poll() is None, proc.communicate()
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("the review directory never appeared")


def _finish(proc: subprocess.Popen[bytes]) -> tuple[int, str]:
    out, err = proc.communicate(timeout=WAIT_S)
    return proc.returncode, (out + err).decode(errors="replace")


@pytest.mark.parametrize(
    ("scenario", "code", "message"),
    [
        ("while_writing_SIGTERM", 128 + signal.SIGTERM, "SIGTERM"),
        ("while_writing_SIGINT", 128 + signal.SIGINT, "SIGINT"),
        ("while_writing_SIGHUP", 128 + signal.SIGHUP, "SIGHUP"),
        ("in_mkdtemp", 128 + signal.SIGINT, "SIGINT"),
        ("in_rmtree", 128 + signal.SIGTERM, "SIGTERM"),
        ("stats_fail", 1, "No space left"),
    ],
)
def test_child_exit_paths_leave_no_directory(
    tmp_path: Path, scenario: str, code: int, message: str
) -> None:
    proc, tmp = _spawn(tmp_path, scenario, ["--n", "2", "--min-persons", "1"])
    returncode, output = _finish(proc)
    assert returncode == code, output
    assert message in output
    assert _leftovers(tmp) == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGQUIT], ids=lambda s: s.name)
def test_child_other_signals_while_waiting(tmp_path: Path, signum: signal.Signals) -> None:
    args = ["--n", "2", "--min-persons", "1", "--judgements", str(tmp_path / "j.json")]
    proc, tmp = _spawn(tmp_path, "json", args)
    workdir = _wait_for_review(tmp, proc)
    proc.send_signal(signum)
    returncode, output = _finish(proc)
    assert returncode == 128 + signum, output
    assert not workdir.exists() and _leftovers(tmp) == []


@pytest.mark.parametrize(
    ("scenario", "code"),
    [
        # One signal as clean-up starts, after a normal review.
        ("at_cleanup:-:SIGTERM", 128 + signal.SIGTERM),
        # Ctrl-C, then more signals as clean-up starts: the first one decides the exit.
        ("at_cleanup:SIGINT:SIGTERM,SIGINT,SIGHUP", 128 + signal.SIGINT),
        ("at_cleanup:-:SIGHUP,SIGTERM,SIGINT", 128 + signal.SIGHUP),
    ],
)
def test_child_signals_as_cleanup_starts_still_clean_up(
    tmp_path: Path, scenario: str, code: int
) -> None:
    proc, tmp = _spawn(tmp_path, scenario, ["--n", "2", "--min-persons", "1"])
    returncode, output = _finish(proc)
    assert returncode == code, output
    assert "Traceback" not in output
    assert _leftovers(tmp) == []
    assert not (tmp_path / "out").exists()


def test_child_signal_while_reporting_an_error(tmp_path: Path) -> None:
    proc, tmp = _spawn(tmp_path, "signal_in_handler", ["--n", "2", "--min-persons", "1"])
    returncode, output = _finish(proc)
    assert returncode == 1, output
    assert "No space left" in output and "Traceback" not in output
    assert _leftovers(tmp) == []


STRESS_RUNS = 20
BURST = (signal.SIGINT, signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def test_child_repeated_signals_stress(tmp_path: Path) -> None:
    """INT, TERM, INT, HUP back to back while the review is open, in many processes."""
    runs = []
    for k in range(STRESS_RUNS):
        base = tmp_path / f"run{k}"
        base.mkdir()
        args = ["--n", "2", "--min-persons", "1", "--judgements", str(base / "j.json")]
        runs.append((base, *_spawn(base, "json", args)))
    for _base, proc, tmp in runs:
        _wait_for_review(tmp, proc)
        for signum in BURST:
            proc.send_signal(signum)  # a no-op once the process has exited
    for base, proc, tmp in runs:
        returncode, output = _finish(proc)
        # Pending signals are delivered lowest number first, so any of them may win.
        assert returncode in {128 + s for s in BURST}, output
        assert "Traceback" not in output
        assert _leftovers(tmp) == [], base


def test_child_parser_exception_while_polling(tmp_path: Path) -> None:
    judgements = tmp_path / "j.json"
    args = ["--n", "2", "--min-persons", "1", "--judgements", str(judgements)]
    proc, tmp = _spawn(tmp_path, "parser_breaks", args)
    workdir = _wait_for_review(tmp, proc)
    judgements.write_text("{}", encoding="utf-8")
    returncode, output = _finish(proc)
    assert returncode != 0
    assert "parser broke" in output
    assert not workdir.exists() and _leftovers(tmp) == []


def test_child_keyboard_timeout_and_end_of_input(tmp_path: Path) -> None:
    args = ["--n", "1", "--min-persons", "1", "--mode", "frames", "--timeout", "1"]
    proc, tmp = _spawn(tmp_path, "keyboard", args, counts=(2,))
    assert proc.wait(timeout=WAIT_S) == 3  # stdin stays open and silent
    assert proc.stdin is not None
    proc.stdin.close()
    assert _leftovers(tmp) == []
    proc, tmp = _spawn(tmp_path, "keyboard", args[:-2], counts=(2,))
    returncode, output = _finish(proc)  # stdin closed at once
    assert returncode == 1 and "input ended" in output
    assert _leftovers(tmp) == []


def test_second_instance_never_deletes_a_running_instances_directory(tmp_path: Path) -> None:
    args = ["--n", "2", "--min-persons", "1", "--judgements", str(tmp_path / "j.json")]
    first, tmp = _spawn(tmp_path, "json", args)
    workdir = _wait_for_review(tmp, first)
    old = time.time() - 7200
    os.utime(workdir, (old, old))  # looks stale to anyone with a short timeout
    second, _ = _spawn(tmp_path, "ok", ["--n", "1", "--min-persons", "1", "--timeout", "1"])
    returncode, output = _finish(second)
    assert returncode == 0, output
    assert workdir.is_dir()
    assert sum(1 for _ in workdir.iterdir()) == 6  # 5 crops and the numbering
    first.send_signal(signal.SIGTERM)
    assert _finish(first)[0] == 128 + signal.SIGTERM
    assert _leftovers(tmp) == []


def test_sigkill_leftover_is_removed_by_the_next_run(tmp_path: Path) -> None:
    args = ["--n", "2", "--min-persons", "1", "--judgements", str(tmp_path / "j.json")]
    proc, tmp = _spawn(tmp_path, "json", args)
    workdir = _wait_for_review(tmp, proc)
    proc.kill()
    proc.communicate(timeout=WAIT_S)
    assert workdir.is_dir()  # SIGKILL cannot be handled
    returncode, output = _finish(_spawn(tmp_path, "ok", ["--n", "1", "--min-persons", "1"])[0])
    assert returncode == 0 and workdir.is_dir()  # younger than the timeout: kept
    old = time.time() - 7200
    os.utime(workdir, (old, old))
    returncode, output = _finish(_spawn(tmp_path, "ok", ["--n", "1", "--min-persons", "1"])[0])
    assert returncode == 0, output
    assert "Deleted 1 review directories" in output
    assert _leftovers(tmp) == []


# Runtime privacy (AC9): the real command, from interpreter start to exit --------------

AUDITED = r"""
import os, sys

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
FILE_EVENTS = {"os.mkdir", "os.rename", "os.replace", "os.link", "os.symlink",
               "os.truncate", "tempfile.mkstemp", "tempfile.mkdtemp"}
PROCESS_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                  "os.spawn", "os.fork", "os.forkpty"}

def report(kind, text):
    os.write(2, ("\n@@audit " + kind + " " + text.replace("\n", " ") + "\n").encode(
        "utf-8", "backslashreplace"))

def audit(event, args):
    if event == "open":
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & WRITE_FLAGS)
        if writing and not isinstance(path, int):
            report("write", os.fsdecode(path))
    elif event in FILE_EVENTS:
        report("write", os.fsdecode(args[0]) if args[0] is not None else "?")
    elif event in PROCESS_EVENTS:
        report("escape", f"{event} {args[0]!s}")
    elif event in ("socket.connect", "socket.sendto"):
        address = args[1]
        if not (isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")):
            report("escape", f"{event} {address!r}")

sys.addaudithook(audit)
import runpy
sys.argv = ["wearreport.tools.spotcheck", *sys.argv[1:]]
runpy.run_module("wearreport.tools.spotcheck", run_name="__main__", alter_sys=True)
"""
MARKER = re.compile(r"^@@audit (\w+) (.*)$", re.MULTILINE)


def _model() -> None:
    if not detect.model_path("yolox_s.onnx").is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"yolox_s.onnx is missing and {REQUIRE_MODEL} is set")
        pytest.skip("yolox_s.onnx is missing; run scripts/fetch_model.sh")


def _all_files(roots: Sequence[Path]) -> dict[Path, bytes]:
    found: dict[Path, bytes] = {}
    for root in roots:
        for dirpath, _dirs, names in os.walk(root):
            for name in names:
                found[Path(dirpath, name)] = Path(dirpath, name).read_bytes()
    return found


@pytest.mark.parametrize("mode", ["crops", "frames"])
def test_dry_run_process_writes_images_only_into_its_directory(tmp_path: Path, mode: str) -> None:
    _model()
    work, home, tmp = tmp_path / "work", tmp_path / "home", tmp_path / "tmp"
    for d in (work, home, tmp):
        d.mkdir()
    judgements = tmp_path / "reviewer" / "judgements.json"
    judgements.parent.mkdir()
    environ = _child_env(tmp)
    environ["HOME"] = str(home)
    environ = {k: v for k, v in environ.items() if not k.startswith("XDG_")}
    args = [
        "--dry-run",
        "--model",
        "yolox_s.onnx",
        "--n",
        "4",
        "--min-persons",
        "1",
        "--mode",
        mode,
    ]
    args += ["--judgements", str(judgements), "--reviewer", "tester", "--out-dir", "stats"]
    proc = subprocess.Popen(
        [sys.executable, "-c", AUDITED, *args],
        cwd=work,
        env=environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    workdir = _wait_for_review(tmp, proc)
    images = [p for p in workdir.iterdir() if p.read_bytes().startswith(IMAGE_MAGIC)]
    assert images
    numbering = json.loads((workdir / spotcheck.NUMBERING_FILE).read_text(encoding="utf-8"))
    judgements.write_text(json.dumps(numbering["template"]), encoding="utf-8")
    out, err = proc.communicate(timeout=WAIT_S)
    stderr = err.decode(errors="replace")
    assert proc.returncode == 0, stderr
    output = out.decode(errors="replace") + MARKER.sub("", stderr)

    stats_file = work / "stats" / f"{datetime.date.today().isoformat()}.json"
    events = MARKER.findall(stderr)
    assert [text for kind, text in events if kind == "escape"] == []
    written = {work / text for kind, text in events if kind == "write"}  # cwd-relative
    allowed = {stats_file, work / "stats"}
    for path in written - allowed:
        assert path.is_relative_to(tmp), path
        top = path.relative_to(tmp).parts[0]
        # tempfile.gettempdir() checks TMPDIR by writing b"blat" to a random 8-character
        # name and deleting it; the file scan below proves nothing of it is left.
        probe = len(path.relative_to(tmp).parts) == 1 and re.fullmatch(r"[a-z0-9_]{8}", top)
        assert top.startswith(spotcheck.TEMP_PREFIX) or probe, path
    assert {p.name for p in written if p.suffix == spotcheck.IMAGE_SUFFIX} == {
        p.name for p in images
    }

    files = _all_files([work, home, tmp])
    assert list(files) == [stats_file]
    for data in files.values():
        assert not any(sig in data for sig in SIGNATURES)
    assert not any(sig.decode("latin-1") in output for sig in SIGNATURES)
    assert "Fake_" not in output and "Fake_" not in stats_file.read_text(encoding="utf-8")
    assert "/cam/" not in output and "http" not in output
    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    assert stats["frames_reviewed"] == 4 and stats["mode"] == mode
    assert _leftovers(tmp) == []
