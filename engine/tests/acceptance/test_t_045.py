"""Acceptance tests for T-045: attribute labelling of near-field crops, with a one-off
paired model test.

Every frame here is synthetic (uniform colours with a gradient), every file is written by
the test or by the tool into a temporary directory, and the hosted model is a fake server
on 127.0.0.1. Nothing reaches the network. The window tests need Tk and a display: they
skip without one, except on Windows or with WEARREPORT_REQUIRE_TK set (on Linux, run them
under `xvfb-run`).
"""

from __future__ import annotations

import ast
import base64
import datetime
import http.server
import io
import json
import math
import os
import re
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
from wearreport._cv import cv2
from wearreport.tools import judge_hosted, spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "spotchecks" / "README.md"
TOOL = ROOT / "engine" / "wearreport" / "tools" / "spotcheck.py"
WINDOWS = sys.platform == "win32"
REQUIRE_TK = "WEARREPORT_REQUIRE_TK"
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
KEY_ENV = "DEEPINFRA_API_KEY"
MODEL = "di-qwen3-vl-235b"
H, W = 288, 352
DAY = datetime.date(2026, 10, 5)
NOON = datetime.datetime(2026, 10, 5, 11, 22, 33, tzinfo=datetime.UTC)
WAIT_S = 60
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".ppm"}
FIELDS = {
    "date",
    "started_at",
    "light",
    "frames",
    "detector",
    "min_height_px",
    "judge",
    "crops_shown",
    "crops_rejected",
    "crops",
}
QUESTIONS = (
    ("outer_layer", "Outer layer (coat or jacket)?"),
    ("bare_legs", "Bare legs (shorts or short skirt)?"),
    ("umbrella", "Holding an open umbrella?"),
)
OLD_WINDOW_LEGEND = (
    "Enter or Space: pedestrian   n: not a person   v: person in a vehicle   "
    "u: cannot tell   Backspace: back   q: stop"
)
ANSWER = re.compile(r"[ynu]{3}")
Z = 1.959963984540054  # the 97.5th percentile of the standard normal distribution

Frame = npt.NDArray[np.uint8]


# Helpers ------------------------------------------------------------------------------

# Boxes by height in source-frame pixels: 30, 31, 60, 30.4 (rounds to 30) and 45.
B30 = (10.0, 50.0, 30.0, 80.0)
B31 = (60.0, 50.0, 80.0, 81.0)
B60 = (110.0, 20.0, 140.0, 80.0)
B30_4 = (160.0, 50.0, 180.0, 80.4)
B45 = (210.0, 40.0, 230.0, 85.0)


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
        frame[10:130, :, 1] = np.arange(W, dtype=np.uint8)[None, :]  # crops differ
        frame[10:130, :, 2] = (np.arange(120, dtype=np.uint8) * 2)[:, None]
        frame[0, 0, 0] = 10 * i
        frames.append(frame)
    return spotcheck.Pipeline(frames=lambda: frames, detector=Stub(boxes), info=INFO)


def _never() -> spotcheck.Pipeline:
    def frames() -> list[Frame]:
        raise AssertionError("the sweep must not start")

    return spotcheck.Pipeline(frames=frames, detector=Stub([]), info=INFO)


class Answers:
    """An attribute reviewer: answers[k] for crop k ("ynu"-form, or None for `x`),
    `default` otherwise. Records the crops it was shown, and calls `during` (if given)
    when the labelling opens."""

    def __init__(
        self,
        answers: Mapping[int, str | None] | None = None,
        default: str | None = "ynn",
        during: Callable[[], None] | None = None,
    ) -> None:
        self.answers = dict(answers or {})
        self.default = default
        self.during = during
        self.items: list[spotcheck.ReviewItem] = []

    def attributes(
        self, items: Sequence[spotcheck.ReviewItem], deadline: float
    ) -> dict[int, str | None]:
        if self.during is not None:
            self.during()
        self.items = list(items)
        return {item.number: self.answers.get(item.number, self.default) for item in items}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Outside CI, in a fresh working directory, HOME and temporary directory; yields
    the temporary directory."""
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    for name in (*PROXY_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
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
    argv = ["--attributes", *args, "--reviewer", "tester", "--out-dir", str(out)]
    kwargs.setdefault("clock", lambda: NOON)
    return spotcheck.main(argv, today=DAY, **kwargs)


def _files(roots: Sequence[Path]) -> dict[Path, bytes]:
    found: dict[Path, bytes] = {}
    for root in roots:
        for dirpath, _dirs, names in os.walk(root):
            for name in names:
                path = Path(dirpath, name)
                found[path] = path.read_bytes()
    return found


def _json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _record(out: Path, name: str | None = None) -> dict[str, Any]:
    return _json(out / "attributes" / (name or f"{DAY.isoformat()}.json"))


def _need_window() -> None:
    try:
        import tkinter

        tkinter.Tk().destroy()
    except Exception as exc:  # ImportError without Tk, TclError without a display
        if WINDOWS or os.environ.get(REQUIRE_TK):
            pytest.fail(f"no Tk window here: {exc}")
        pytest.skip("no Tk display here (on Linux run under xvfb-run)")


def _crop(number: int) -> spotcheck.ReviewItem:
    image = np.full((60, 40, 3), (10 * number) % 250, dtype=np.uint8)
    return spotcheck.ReviewItem(number, f"crop-{number:04d}.png", (number,), image)


# The fake hosted model ----------------------------------------------------------------


def _chat(text: str | None) -> bytes:
    body = {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 300, "completion_tokens": 12},
    }
    return json.dumps(body).encode()


class _Handler(http.server.BaseHTTPRequestHandler):
    server: Any

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        fake = self.server.fake
        with fake.lock:
            fake.bodies.append(body)
            status, reply = fake.replies.pop(0) if fake.replies else fake.default
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class FakeModel:
    def __init__(self, replies: Sequence[tuple[int, bytes]], default: tuple[int, bytes]) -> None:
        self.replies = list(replies)
        self.default = default
        self.bodies: list[bytes] = []
        self.lock = threading.Lock()
        self.url = ""

    def requests(self) -> list[dict[str, Any]]:
        with self.lock:
            return [json.loads(body) for body in self.bodies]

    def texts(self) -> list[str]:
        found = []
        for request in self.requests():
            [message] = request["messages"]
            found += [p["text"] for p in message["content"] if p["type"] == "text"]
        return found

    def images(self) -> list[Frame]:
        images = []
        for request in self.requests():
            [message] = request["messages"]
            parts = message["content"]
            [url] = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
            png = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8)
            images.append(np.asarray(cv2.imdecode(png, cv2.IMREAD_COLOR), dtype=np.uint8))
        return images


@pytest.fixture
def model_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., FakeModel]]:
    for name in (*PROXY_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    started: list[http.server.ThreadingHTTPServer] = []

    def start(
        *replies: tuple[int, bytes],
        default: tuple[int, bytes] = (200, _chat("outer=yes legs=no umbrella=unsure")),
    ) -> FakeModel:
        fake = FakeModel(replies, default)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        server.fake = fake  # type: ignore[attr-defined]
        fake.url = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return fake

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def _judged(
    args: Sequence[str], out: Path, fake: FakeModel, max_requests: int = 50, **kwargs: Any
) -> int:
    judge = ["--judge", MODEL, "--judge-max-requests", str(max_requests)]
    return _run(
        [*args, *judge],
        out,
        judge_endpoint=fake.url,
        judge_sleep=lambda seconds: None,
        **kwargs,
    )


WINDOW = ["--n", "5", "--min-persons", "1", "--view", "window"]


# AC1: near-field only -----------------------------------------------------------------


def test_ac1_the_constant() -> None:
    assert spotcheck.NEAR_FIELD_MIN_HEIGHT_PX == 31
    defined = []
    for path in sorted((ROOT / "engine" / "wearreport").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else []
            if isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "NEAR_FIELD_MIN_HEIGHT_PX":
                    defined.append(path.relative_to(ROOT).as_posix())
    assert defined == ["engine/wearreport/tools/spotcheck.py"]


def test_ac1_only_boxes_of_at_least_31_px_are_shown(env: Path, tmp_path: Path) -> None:
    reviewer = Answers()
    out = tmp_path / "out"
    assert _run(WINDOW, out, pipeline=_pipeline([[B30, B31, B30_4]]), reviewer=reviewer) == 0
    assert len(reviewer.items) == 1  # the 31 px box only
    [item] = reviewer.items
    assert item.boxes == (1,)  # numbered over the near-field crops
    record = _record(out)
    assert record["min_height_px"] == 31
    assert record["crops_shown"] == 1 and record["crops_rejected"] == 0
    assert record["crops"] == [[31, "ynn", None]]


def test_ac1_the_crop_is_the_one_the_tool_renders_for_that_box(env: Path, tmp_path: Path) -> None:
    reviewer = Answers()
    pipeline = _pipeline([[B30, B31]])
    assert _run(WINDOW, tmp_path / "out", pipeline=pipeline, reviewer=reviewer) == 0
    [frame] = list(pipeline.frames())
    expected = spotcheck.render_crop(frame, detect.Detection("person", 0.9, B31), 1)
    [item] = reviewer.items
    assert np.array_equal(item.image, expected)


def test_ac1_min_persons_counts_near_field_boxes(env: Path, tmp_path: Path) -> None:
    # Frame 0: one near-field box (31) and three small ones. Frame 1: two near-field
    # boxes (60, 45). Frame 2: two near-field boxes (31, 60) and one small one.
    boxes = [[B31, B30, B30_4, B30], [B60, B45], [B31, B60, B30]]
    reviewer = Answers()
    out = tmp_path / "out"
    args = ["--n", "5", "--min-persons", "2", "--view", "window"]
    assert _run(args, out, pipeline=_pipeline(boxes), reviewer=reviewer) == 0
    record = _record(out)
    assert record["frames"] == 2
    assert record["crops_shown"] == 4 == len(reviewer.items)
    assert sorted(h for h, _r, _m in record["crops"]) == [31, 45, 60, 60]


def test_ac1_no_frame_with_enough_near_field_boxes_writes_nothing(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--n", "5", "--min-persons", "1", "--view", "window"]
    assert _run(args, tmp_path / "out", pipeline=_pipeline([[B30, B30_4]]), reviewer=Answers()) == 1
    assert "near-field" in capsys.readouterr().err
    assert not (tmp_path / "out" / "attributes").exists()


# AC2: questions and keys --------------------------------------------------------------


def test_ac2_questions_and_legend() -> None:
    assert tuple(spotcheck.ATTRIBUTE_QUESTIONS) == QUESTIONS
    legend = spotcheck.ATTRIBUTE_LEGEND
    for key in ("y", "n", "u", "x", "Backspace", "q"):
        assert re.search(rf"(^|\s){key}:", legend), key
    assert "cannot tell" in legend
    assert spotcheck.WINDOW_LEGEND == OLD_WINDOW_LEGEND


class _Keys:
    """A window driver: before each key it records the texts of the window's labels and
    whether one of them shows an image, then sends the key, 20 ms apart. A class, not a
    closure that schedules itself (see the T-037 window tests)."""

    def __init__(self, keys: Sequence[str]) -> None:
        self.pending = list(keys)
        self.seen: list[tuple[list[str], bool]] = []
        self.root: Any = None

    def __call__(self, root: Any) -> None:
        self.root = root
        root.after(20, self._step)

    def _step(self) -> None:
        root = self.root
        if not self.pending:
            return
        labels = [w for w in root.winfo_children() if w.winfo_class() == "Label"]
        texts = [str(w.cget("text")) for w in labels]
        self.seen.append((texts, any(str(w.cget("image")) for w in labels)))
        root.focus_force()
        root.event_generate(self.pending.pop(0))
        root.after(20, self._step)


def _press(*keys: str) -> _Keys:
    return _Keys([k if k.startswith("<") else f"<KeyPress-{k}>" for k in keys])


def test_ac2_each_key_in_the_window_one_question_at_a_time() -> None:
    _need_window()
    # Crop 1: y n u, then BackSpace from crop 2 back into crop 1, and n: "ynn".
    # Crop 2: x (rejected), BackSpace undoes the x, then Y y u: "yyu".
    # Crop 3: n, then x at the second question: rejected.
    # Crop 4: u n y: "uny".
    keys = ["y", "n", "u", "<BackSpace>", "n", "x", "<BackSpace>", "Y", "y", "u"]
    keys += ["n", "x", "u", "n", "y"]
    driver = _press(*keys)
    reviewer = spotcheck.WindowReviewer(driver=driver)
    got = reviewer.attributes([_crop(k) for k in (1, 2, 3, 4)], time.monotonic() + WAIT_S)
    assert dict(got) == {1: "ynn", 2: "yyu", 3: None, 4: "uny"}
    # The question on screen before each key, and a crop always shown.
    expected = [0, 1, 2, 0, 2, 0, 0, 0, 1, 2, 0, 1, 0, 1, 2]
    assert len(driver.seen) == len(expected)
    for (texts, shows_image), question in zip(driver.seen, expected, strict=True):
        assert shows_image
        on_screen = " ".join(texts)
        assert QUESTIONS[question][1] in on_screen
        assert [q for _k, q in QUESTIONS if q in on_screen] == [QUESTIONS[question][1]]
    assert any(spotcheck.ATTRIBUTE_LEGEND in " ".join(texts) for texts, _ in driver.seen)


def test_ac2_q_in_the_window_stops() -> None:
    _need_window()
    reviewer = spotcheck.WindowReviewer(driver=_press("y", "n", "q"))
    with pytest.raises(spotcheck.ReviewAborted):
        reviewer.attributes([_crop(1), _crop(2)], time.monotonic() + WAIT_S)


def test_ac2_q_in_the_window_writes_nothing(env: Path, tmp_path: Path) -> None:
    _need_window()
    reviewer = spotcheck.WindowReviewer(driver=_press("y", "n", "y", "q"))
    out = tmp_path / "out"
    assert _run(WINDOW, out, pipeline=_pipeline([[B31, B60]]), reviewer=reviewer) == 1
    assert not out.exists()


def test_ac2_the_window_session_writes_the_answers(env: Path, tmp_path: Path) -> None:
    _need_window()
    out = tmp_path / "out"
    reviewer = spotcheck.WindowReviewer(driver=_press("y", "y", "n", "x", "n", "u", "y"))
    # Boxes in frame order: 31 px (crop 1), 60 px (crop 2), 45 px (crop 3).
    assert _run(WINDOW, out, pipeline=_pipeline([[B31, B30, B60, B45]]), reviewer=reviewer) == 0
    record = _record(out)
    assert record["crops_shown"] == 3 and record["crops_rejected"] == 1
    assert sorted(record["crops"]) == [[31, "yyn", None], [45, "nuy", None]]


def _entry(a: object, b: object, c: object) -> dict[str, object]:
    return {"outer_layer": a, "bare_legs": b, "umbrella": c}


def test_ac2_the_json_file_accepts_the_same_answers() -> None:
    items = [_crop(1), _crop(2), _crop(3)]
    raw = json.dumps({"1": _entry("y", "n", "u"), "2": "x", "3": _entry("u", "u", "y")})
    got = spotcheck.parse_attribute_judgements(raw.encode(), items)
    assert got == {1: "ynu", 2: None, 3: "uuy"}


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        pytest.param("X", "capital-x", id="capital-x"),
        pytest.param("y", "a-bare-answer", id="bare-answer"),
        pytest.param("ynn", "a-string-of-answers", id="string-of-answers"),
        pytest.param(None, "null", id="null"),
        pytest.param(1, "number", id="number"),
        pytest.param(["y", "n", "u"], "list", id="list"),
        pytest.param({"outer_layer": "y", "bare_legs": "n"}, "missing-key", id="missing-key"),
        pytest.param({**_entry("y", "n", "u"), "note": "y"}, "extra-key", id="extra-key"),
        pytest.param(_entry("yes", "n", "u"), "word", id="word"),
        pytest.param(_entry("Y", "n", "u"), "capital", id="capital"),
        pytest.param(_entry("y", "x", "u"), "x-inside", id="x-inside"),
        pytest.param(_entry("y", "n", None), "null-answer", id="null-answer"),
        pytest.param(_entry("y", "n", True), "bool-answer", id="bool-answer"),
    ],
)
def test_ac2_a_malformed_entry_is_an_error_naming_it(entry: object, why: str) -> None:
    items = [_crop(1), _crop(2), _crop(3)]
    raw = json.dumps({"1": "x", "2": entry, "3": _entry("n", "n", "n")}).encode()
    with pytest.raises(spotcheck.JudgementError, match=r"\b2\b"):
        spotcheck.parse_attribute_judgements(raw, items)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b'{"1": "x", "2": "x"}', id="a-crop-missing"),
        pytest.param(b'{"1": "x", "2": "x", "3": "x", "4": "x"}', id="no-such-crop"),
        pytest.param(b'{"1": "x", "1": "x", "2": "x", "3": "x"}', id="duplicate-key"),
        pytest.param(b'["x", "x", "x"]', id="not-an-object"),
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"\xff\xfe", id="not-utf8"),
        pytest.param(b"[" * 100000, id="deep-nesting"),
        pytest.param(b" " * (spotcheck.MAX_JUDGEMENTS_BYTES + 1), id="too-large"),
    ],
)
def test_ac2_a_malformed_file_is_an_error(raw: bytes) -> None:
    with pytest.raises(spotcheck.JudgementError):
        spotcheck.parse_attribute_judgements(raw, [_crop(1), _crop(2), _crop(3)])


def test_ac2_the_json_file_reviewer_reads_attribute_answers(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"1": "x", "2": _entry("n", "y", "n")}), encoding="utf-8")
    reviewer = spotcheck.JsonFileReviewer(path, out=io.StringIO())
    got = reviewer.attributes([_crop(1), _crop(2)], time.monotonic() + WAIT_S)
    assert dict(got) == {1: None, 2: "nyn"}


def _answer_when_ready(tmp: Path, path: Path, content: object) -> threading.Thread:
    """Write `content` to `path` once the review directory's numbering is there."""

    def write() -> None:
        deadline = time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            if list(tmp.glob(f"{spotcheck.TEMP_PREFIX}*/{spotcheck.NUMBERING_FILE}")):
                path.write_text(json.dumps(content), encoding="utf-8")
                return
            time.sleep(0.05)

    thread = threading.Thread(target=write, daemon=True)
    thread.start()
    return thread


def test_ac2_the_json_file_session_end_to_end(env: Path, tmp_path: Path) -> None:
    answers = tmp_path / "answers" / "a.json"
    answers.parent.mkdir()
    thread = _answer_when_ready(env, answers, {"1": _entry("y", "n", "n"), "2": "x"})
    out = tmp_path / "out"
    args = ["--n", "5", "--min-persons", "1", "--judgements", str(answers)]
    assert _run(args, out, pipeline=_pipeline([[B31, B60]])) == 0
    thread.join()
    record = _record(out)
    assert record["crops"] == [[31, "ynn", None]]
    assert record["crops_shown"] == 2 and record["crops_rejected"] == 1
    assert not list(env.glob(f"{spotcheck.TEMP_PREFIX}*"))  # the review directory is gone


def test_ac2_a_malformed_json_file_writes_nothing(env: Path, tmp_path: Path) -> None:
    answers = tmp_path / "answers" / "a.json"
    answers.parent.mkdir()
    thread = _answer_when_ready(env, answers, {"1": _entry("y", "n", "n"), "2": "maybe"})
    out = tmp_path / "out"
    args = ["--n", "5", "--min-persons", "1", "--judgements", str(answers), "--timeout", "2"]
    assert _run(args, out, pipeline=_pipeline([[B31, B60]])) == 3  # rejected, then timed out
    thread.join()
    assert not out.exists()
    assert not list(env.glob(f"{spotcheck.TEMP_PREFIX}*"))


# AC3: allowed combinations ------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--mode", "frames", "--view", "files", "--judgements", "J"], id="frames"),
        pytest.param(["--mode", "frames", "--view", "window"], id="frames-window"),
        pytest.param(["--view", "files"], id="keyboard"),
        pytest.param(["--view", "window", "--record-boxes"], id="record-boxes-window"),
        pytest.param(["--judgements", "J", "--record-boxes"], id="record-boxes-json"),
        pytest.param(["--view", "window", "--judge", MODEL], id="judge-without-max"),
        pytest.param(["--view", "window", "--judge-max-requests", "5"], id="max-without-judge"),
        pytest.param(
            ["--view", "window", "--dry-run", "--judge", MODEL, "--judge-max-requests", "5"],
            id="dry-run-judge-not-local",
        ),
    ],
)
def test_ac3_refused_before_any_request(
    env: Path,
    offline: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
) -> None:
    def no_pipeline(model: str) -> spotcheck.Pipeline:
        raise AssertionError("no pipeline may be opened")

    monkeypatch.setattr(spotcheck, "live_pipeline", no_pipeline)
    monkeypatch.setattr(spotcheck, "dry_run_pipeline", no_pipeline)
    args = [str(tmp_path / "j.json") if a == "J" else a for a in args]
    out = tmp_path / "out"
    code = _run(["--n", "2", *args], out, pipeline=None, reviewer=Answers())
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("spotcheck: ")
    assert not out.exists() and not (tmp_path / "j.json").exists()


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--mode", "frames", "--view", "files", "--judgements", "J"], id="frames"),
        pytest.param(["--view", "files"], id="keyboard"),
        pytest.param(["--view", "window", "--record-boxes"], id="record-boxes"),
    ],
)
def test_ac3_the_refusal_names_attributes(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    args = [str(tmp_path / "j.json") if a == "J" else a for a in args]
    assert _run(["--n", "2", *args], tmp_path / "out", pipeline=_never()) == 1
    assert "--attributes" in capsys.readouterr().err


def test_ac3_a_dry_run_with_a_local_judge_is_allowed(
    env: Path, tmp_path: Path, model_server: Callable[..., FakeModel]
) -> None:
    fake = model_server()
    out = tmp_path / "out"
    args = [*WINDOW, "--dry-run"]
    code = _judged(args, out, fake, pipeline=_pipeline([[B31]]), reviewer=Answers())
    assert code == 0
    assert _record(out)["crops"] == [[31, "ynn", "ynu"]]


# AC4: the attribute file --------------------------------------------------------------


def test_ac4_exactly_its_fields(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    reviewer = Answers({1: "yyn", 2: None, 3: "uun"})
    boxes = [[B31, B30], [B60, B45]]
    assert _run(WINDOW, out, pipeline=_pipeline(boxes), reviewer=reviewer) == 0
    record = _record(out)
    assert set(record) == FIELDS
    assert record["date"] == DAY.isoformat()
    assert record["started_at"] == "2026-10-05T11:22Z"  # the clock, to the minute
    assert record["light"] == "day"
    assert record["frames"] == 2
    assert record["detector"] == {"model": "stub", "sha256": "0" * 64, "conf": INFO.conf}
    assert record["min_height_px"] == 31
    assert record["judge"] is None
    assert record["crops_shown"] == 3 and record["crops_rejected"] == 1
    assert sorted(record["crops"]) == [[31, "yyn", None], [45, "uun", None]]
    for entry in record["crops"]:
        assert isinstance(entry, list) and len(entry) == 3
        height, answers, model = entry
        assert type(height) is int and ANSWER.fullmatch(answers) and model is None
    text = (out / "attributes" / f"{DAY.isoformat()}.json").read_text(encoding="utf-8")
    for word in ("x1", "y1", "width", "camera", "frame_index", "image", "png", "base64"):
        assert word not in text


def test_ac4_the_light_and_started_at(env: Path, tmp_path: Path) -> None:
    dusk = datetime.datetime(2026, 9, 27, 18, 5, 59, tzinfo=datetime.UTC)  # civil twilight
    out = tmp_path / "out"
    reviewer = Answers()
    assert (
        _run(WINDOW, out, pipeline=_pipeline([[B31]]), reviewer=reviewer, clock=lambda: dusk) == 0
    )
    record = _record(out)
    assert record["started_at"] == "2026-09-27T18:05Z" and record["light"] == "twilight"


def test_ac4_nothing_else_is_written(
    env: Path, tmp_path: Path, model_server: Callable[..., FakeModel]
) -> None:
    fake = model_server()
    work, home = Path.cwd(), Path(os.environ["HOME"])
    watched = [work, home, env]
    tempfile.gettempdir()  # tempfile's own writability probe, before the snapshot
    before = _files(watched)
    reviewer = Answers({2: None})
    boxes = [[B31, B60], [B45]]
    assert _judged(WINDOW, work / "stats", fake, pipeline=_pipeline(boxes), reviewer=reviewer) == 0
    after = _files(watched)
    name = f"{DAY.isoformat()}.json"
    assert sorted(set(after) - set(before)) == [work / "stats" / "attributes" / name]
    assert sorted(p.name for p in (work / "stats").iterdir()) == ["attributes"]
    assert not [
        p
        for p, data in after.items()
        if p.suffix.lower() in IMAGE_SUFFIXES or data.startswith(MAGIC)
    ]
    text = after[work / "stats" / "attributes" / name].decode("utf-8")
    assert "base64" not in text and "image" not in text


def test_ac4_nothing_else_is_written_with_the_json_file(env: Path, tmp_path: Path) -> None:
    work, home = Path.cwd(), Path(os.environ["HOME"])
    watched = [work, home, env]
    tempfile.gettempdir()
    before = _files(watched)
    answers = tmp_path / "answers" / "a.json"
    answers.parent.mkdir()
    thread = _answer_when_ready(env, answers, {"1": "x", "2": _entry("n", "n", "y")})
    args = ["--n", "5", "--min-persons", "1", "--judgements", str(answers)]
    assert _run(args, work / "stats", pipeline=_pipeline([[B31, B60]])) == 0
    thread.join()
    after = _files(watched)
    name = f"{DAY.isoformat()}.json"
    assert sorted(set(after) - set(before)) == [work / "stats" / "attributes" / name]
    assert not [p for p, data in after.items() if data.startswith(MAGIC)]


def test_ac4_no_overwrite(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "attributes").mkdir(parents=True)
    name = f"{DAY.isoformat()}.json"
    (out / "attributes" / name).write_text("kept", encoding="utf-8")
    assert _run(WINDOW, out, pipeline=_pipeline([[B31]]), reviewer=Answers()) == 0
    assert (out / "attributes" / name).read_text(encoding="utf-8") == "kept"
    assert _record(out, f"{DAY.isoformat()}-2.json")["crops_shown"] == 1
    assert _run(WINDOW, out, pipeline=_pipeline([[B31]]), reviewer=Answers()) == 0
    assert (out / "attributes" / f"{DAY.isoformat()}-3.json").is_file()
    assert sorted(p.name for p in out.iterdir()) == ["attributes"]  # no statistics file


def test_ac4_privacy_guard_still_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "privacy_guard.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# AC5: the hosted attribute judge ------------------------------------------------------


def test_ac5_the_judge_class_and_prompt() -> None:
    judge_class = judge_hosted.LiveAttributeJudge
    assert judge_class is not judge_hosted.LiveCropJudge
    assert not issubclass(judge_class, judge_hosted.LiveCropJudge)
    prompt = judge_hosted.ATTRIBUTE_PROMPT
    assert prompt != judge_hosted.PROMPT
    assert "outer=" in prompt and "legs=" in prompt and "umbrella=" in prompt


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        pytest.param("outer=yes legs=no umbrella=unsure", "ynu", id="well-formed"),
        pytest.param("outer=no legs=yes umbrella=yes", "nyy", id="well-formed-2"),
        pytest.param("outer=unsure legs=unsure umbrella=no\n", "uun", id="trailing-newline"),
        pytest.param("outer=yes legs=no umbrella=no. It is raining.", "uuu", id="extra-text"),
        pytest.param("Sure! outer=yes legs=no umbrella=no", "uuu", id="text-before"),
        pytest.param("legs=no outer=yes umbrella=no", "uuu", id="wrong-order"),
        pytest.param("outer=yes umbrella=no legs=no", "uuu", id="wrong-order-2"),
        pytest.param("outer=yes legs=no", "uuu", id="missing-part"),
        pytest.param("outer=maybe legs=no umbrella=no", "uuu", id="bad-value"),
        pytest.param("outer=yes\nlegs=no\numbrella=no", "uuu", id="three-lines"),
        pytest.param(
            "outer=yes legs=no umbrella=no" + " " * judge_hosted.MAX_ATTRIBUTE_REPLY_CHARS,
            "uuu",
            id="oversized",
        ),
        pytest.param("x" * 100_000, "uuu", id="huge"),
        pytest.param("", "uuu", id="empty"),
        pytest.param("   ", "uuu", id="blank"),
    ],
)
def test_ac5_the_strict_parser(reply: str, expected: str) -> None:
    assert judge_hosted.parse_attributes(reply) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("outer=yes legs=no umbrella=unsure", "ynu", id="well-formed"),
        pytest.param("outer=yes legs=no umbrella=no, I think", "uuu", id="extra-text"),
        pytest.param("legs=no outer=yes umbrella=no", "uuu", id="wrong-order"),
        pytest.param("outer=yes legs=no umbrella=no" + "!" * 200, "uuu", id="oversized"),
        pytest.param("", "uuu", id="empty"),
        pytest.param(None, "uuu", id="no-content"),
    ],
)
def test_ac5_the_judge_over_http(
    model_server: Callable[..., FakeModel], content: str | None, expected: str
) -> None:
    fake = model_server((200, _chat(content)))
    judge = judge_hosted.LiveAttributeJudge(
        MODEL, budget=judge_hosted.RequestBudget(5), endpoint=fake.url, sleep=lambda s: None
    )
    image = np.full((90, 60, 3), 77, dtype=np.uint8)
    try:
        assert judge.classify_attributes(image) == expected
    finally:
        judge.close()
    [request] = fake.requests()
    assert request["model"] == judge_hosted.DEEPINFRA[MODEL].model_id
    assert fake.texts() == [judge_hosted.ATTRIBUTE_PROMPT]
    [sent] = fake.images()
    assert np.array_equal(sent, image)
    assert judge.usage.requests == 1


def test_ac5_the_endpoint_is_pinned_and_a_whole_frame_refused(
    model_server: Callable[..., FakeModel],
) -> None:
    budget = judge_hosted.RequestBudget(5)
    for bad in ("https://example.com", "http://api.deepinfra.com", "https://api.deepinfra.com/x"):
        with pytest.raises(judge_hosted.JudgeError):
            judge_hosted.LiveAttributeJudge(MODEL, budget=budget, endpoint=bad)
    with pytest.raises(judge_hosted.JudgeError):
        judge_hosted.LiveAttributeJudge("nova-2-lite", budget=budget)
    live = judge_hosted.LiveAttributeJudge(MODEL, budget=budget)
    assert live.url == judge_hosted.DEEPINFRA_ORIGIN + judge_hosted.DEEPINFRA_PATH
    live.close()
    fake = model_server()
    judge = judge_hosted.LiveAttributeJudge(MODEL, budget=budget, endpoint=fake.url)
    side = judge_hosted.MAX_IMAGE_SIDE + 1
    for bad_image in (np.zeros((side, 10, 3), np.uint8), np.zeros((10, 10), np.uint8)):
        with pytest.raises(ValueError):
            judge.classify_attributes(bad_image)
    judge.close()
    assert fake.requests() == []


def test_ac5_retries_budget_and_a_redacted_error(
    model_server: Callable[..., FakeModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
    monkeypatch.setenv(KEY_ENV, key)
    fake = model_server(
        (503, b"{}"), (500, b"{}"), default=(400, json.dumps({"error": key}).encode())
    )
    budget = judge_hosted.RequestBudget(10)
    judge = judge_hosted.LiveAttributeJudge(
        MODEL, budget=budget, endpoint=fake.url, sleep=lambda s: None
    )
    with pytest.raises(judge_hosted.JudgeError) as failed:
        judge.classify_attributes(np.zeros((40, 30, 3), np.uint8))
    assert key not in str(failed.value) and "abcdefgh" not in str(failed.value)
    assert budget.used == 3  # two retried server errors, then the 400
    judge.close()
    one = judge_hosted.RequestBudget(1)
    fake2 = model_server()
    judge = judge_hosted.LiveAttributeJudge(MODEL, budget=one, endpoint=fake2.url)
    assert judge.classify_attributes(np.zeros((40, 30, 3), np.uint8)) == "ynu"
    with pytest.raises(judge_hosted.RequestLimitReached):
        judge.classify_attributes(np.zeros((40, 30, 3), np.uint8))
    judge.close()


def test_ac5_the_model_gets_pixels_only_after_the_review_and_no_rejected_crop(
    env: Path, tmp_path: Path, model_server: Callable[..., FakeModel]
) -> None:
    fake = model_server()
    requests_during_review: list[int] = []
    reviewer = Answers(
        {1: "yyy", 2: None, 3: "nnu", 4: None, 5: "uuu"},
        during=lambda: requests_during_review.append(len(fake.requests())),
    )
    out = tmp_path / "out"
    boxes = [[B31, B60], [B45, B31, B60]]
    assert _judged(WINDOW, out, fake, pipeline=_pipeline(boxes), reviewer=reviewer) == 0
    assert requests_during_review == [0]
    kept = [item for item in reviewer.items if item.number in (1, 3, 5)]
    sent = fake.images()
    assert len(sent) == len(kept) == 3
    for image, item in zip(sent, kept, strict=True):
        assert np.array_equal(image, item.image)
    # Every request carries the fixed prompt and nothing else: no reviewer answer.
    assert fake.texts() == [judge_hosted.ATTRIBUTE_PROMPT] * 3
    record = _record(out)
    assert record["judge"] == MODEL
    assert record["crops_shown"] == 5 and record["crops_rejected"] == 2
    assert sorted(record["crops"]) == [[31, "yyy", "ynu"], [45, "nnu", "ynu"], [60, "uuu", "ynu"]]


def test_ac5_the_call_site_passes_the_pixels_only() -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "classify_attributes"
    ]
    assert len(calls) == 1
    [call] = calls
    assert len(call.args) == 1 and not call.keywords


@pytest.mark.parametrize(
    ("replies", "max_requests", "answered"),
    [
        pytest.param([], 2, 2, id="request-cap"),
        pytest.param([(200, _chat("outer=no legs=no umbrella=no"))], 50, 1, id="then-http-404"),
        pytest.param([(200, _chat("outer=no legs=no umbrella=no")), (200, b"{")], 50, 1, id="bad"),
    ],
)
def test_ac5_a_judge_failure_keeps_the_reviewer_labels(
    env: Path,
    tmp_path: Path,
    model_server: Callable[..., FakeModel],
    capsys: pytest.CaptureFixture[str],
    replies: list[tuple[int, bytes]],
    max_requests: int,
    answered: int,
) -> None:
    default = (200, _chat("outer=yes legs=no umbrella=unsure"))
    if replies:
        default = (404, b'{"error": "no such model"}')
    fake = model_server(*replies, default=default)
    reviewer = Answers({3: None})
    out = tmp_path / "out"
    boxes = [[B31, B60], [B45, B31, B60]]  # five crops, one rejected: four for the model
    code = _judged(WINDOW, out, fake, max_requests, pipeline=_pipeline(boxes), reviewer=reviewer)
    assert code == 0
    assert "judge stopped" in capsys.readouterr().err
    record = _record(out)
    assert record["crops_shown"] == 5 and record["crops_rejected"] == 1
    assert len(record["crops"]) == 4
    assert all(r == "ynn" for _h, r, _m in record["crops"])
    models = [m for _h, _r, m in record["crops"]]
    assert sum(m is not None for m in models) == answered
    assert models.count(None) == 4 - answered


# AC6: the summary ---------------------------------------------------------------------

DETECTOR = {"model": "yolox_m", "sha256": "0" * 64, "conf": 0.3}


def _attr_file(
    directory: Path,
    name: str,
    crops: Sequence[tuple[int, str, str | None]],
    started_at: str = "2026-10-05T10:00Z",
    light: str = "day",
    judge: str | None = MODEL,
    rejected: int = 0,
) -> None:
    record = {
        "date": started_at[:10],
        "started_at": started_at,
        "light": light,
        "frames": 4,
        "detector": DETECTOR,
        "min_height_px": 31,
        "judge": judge,
        "crops_shown": len(crops) + rejected,
        "crops_rejected": rejected,
        "crops": [list(c) for c in crops],
    }
    (directory / "attributes").mkdir(parents=True, exist_ok=True)
    (directory / "attributes" / name).write_text(json.dumps(record), encoding="utf-8")


def _summary(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, list[str], str]:
    code = spotcheck_summary.main(["--attributes", *argv])
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.err


def _wilson(k: int, n: int) -> str:
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z / d * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return f"wilson [{max(0.0, centre - half):.4f}, {min(1.0, centre + half):.4f}]"


def _counts(yes: int, no: int, unsure: int) -> str:
    share = f"{yes / (yes + no):.4f} {_wilson(yes, yes + no)}" if yes + no else "n/a wilson n/a"
    return f"yes {yes}, no {no}, cannot tell {unsure}; yes share {share} (n={yes + no})"


def _measure(k: int, n: int) -> str:
    return f"{k / n:.4f} (n={n}) {_wilson(k, n)}" if n else "n/a (n=0) wilson n/a"


def _block(lines: list[str], attribute: str) -> list[str]:
    start = lines.index(f"{attribute}:")
    block = []
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        block.append(line)
    return block


def _example(d: Path) -> None:
    # (height, reviewer, model) per crop.
    day = [
        (40, "ynn", "ynn"),
        (50, "ynn", "nnn"),
        (60, "yyn", "uyn"),
        (45, "nnn", "ynn"),
        (35, "nnn", "nnn"),
        (70, "unn", "ynn"),
        (33, "ynn", None),
    ]
    _attr_file(d, "2026-10-05.json", day, rejected=2)
    twilight = [(80, "nny", None), (31, "yny", None)]
    _attr_file(d, "2026-10-06.json", twilight, "2026-10-06T17:30Z", "twilight", judge=None)


def test_ac6_named_constants() -> None:
    assert spotcheck_summary.MIN_POSITIVES == 30
    assert spotcheck_summary.MIN_NEGATIVES == 30
    assert spotcheck_summary.MIN_LABELLED == 200
    assert spotcheck_summary.TARGET_ATTRIBUTE_ACCURACY == 0.85


def test_ac6_hand_computed_example(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # outer_layer, the first letter. Reviewer: yes on crops 1, 2, 3, 7 and 9 (5), no on
    # 4, 5 and 8 (3), cannot tell on 6 (1). Yes share 5/8. Day: yes 4, no 2, cannot tell
    # 1; twilight: yes 1, no 1; dark: none.
    # Paired (reviewer yes or no, model asked): crops 1 to 5 (crop 6 is a reviewer u,
    # crop 7 was not asked). Model yes on crops 1 and 4: precision 1/2. Reviewer yes on
    # 1, 2, 3: the model says yes on 1, no on 2 and u on 3, and the u counts as wrong:
    # recall 1/3. Reviewer no on 4, 5: the model says no on 5: specificity 1/2. Model u: 1.
    # bare_legs, the second letter. Reviewer: yes on crop 3, no on the other 8. Paired:
    # crops 1 to 6 (6 is a reviewer n here). Model yes on crop 3 only: precision 1/1,
    # recall 1/1, specificity 5/5.
    # umbrella, the third letter. Reviewer: yes on 8 and 9 (neither paired), no on the
    # other 7. Paired: crops 1 to 6, all no and all model no: precision n/a (the model
    # never says yes), recall n/a (no paired reviewer yes), specificity 6/6.
    # Every attribute is short of every count: insufficient.
    d = tmp_path / "spotchecks"
    _example(d)
    code, lines, err = _summary(["--dir", str(d)], capsys)
    assert code == 0, err
    assert lines[0] == f"2 attribute file(s) in {d / 'attributes'}"
    assert "crops: 11 shown, 2 rejected, 9 labelled, 6 with a model answer" in lines
    assert _block(lines, "outer_layer") == [
        f"  reviewer: {_counts(5, 3, 1)}",
        f"  light day: {_counts(4, 2, 1)}",
        f"  light twilight: {_counts(1, 1, 0)}",
        f"  light dark: {_counts(0, 0, 0)}",
        "  model: 5 paired, model cannot tell 1",
        f"  model precision {_measure(1, 2)}",
        f"  model recall {_measure(1, 3)}",
        f"  model specificity {_measure(1, 2)}",
        "  verdict: insufficient (reviewer yes 3 < 30; reviewer no 2 < 30; "
        "reviewer yes+no 5 < 200)",
    ]
    assert _block(lines, "bare_legs") == [
        f"  reviewer: {_counts(1, 8, 0)}",
        f"  light day: {_counts(1, 6, 0)}",
        f"  light twilight: {_counts(0, 2, 0)}",
        f"  light dark: {_counts(0, 0, 0)}",
        "  model: 6 paired, model cannot tell 0",
        f"  model precision {_measure(1, 1)}",
        f"  model recall {_measure(1, 1)}",
        f"  model specificity {_measure(5, 5)}",
        "  verdict: insufficient (reviewer yes 1 < 30; reviewer no 5 < 30; "
        "reviewer yes+no 6 < 200)",
    ]
    assert _block(lines, "umbrella") == [
        f"  reviewer: {_counts(2, 7, 0)}",
        f"  light day: {_counts(0, 7, 0)}",
        f"  light twilight: {_counts(2, 0, 0)}",
        f"  light dark: {_counts(0, 0, 0)}",
        "  model: 6 paired, model cannot tell 0",
        f"  model precision {_measure(0, 0)}",
        f"  model recall {_measure(0, 0)}",
        f"  model specificity {_measure(6, 6)}",
        "  verdict: insufficient (reviewer yes 0 < 30; reviewer no 6 < 30; "
        "reviewer yes+no 6 < 200)",
    ]
    assert "precision 0.5000 (n=2) wilson [0.0945, 0.9055]" in lines  # checked by hand


def _verdict(tmp_path: Path, crops: list[tuple[int, str, str | None]], capsys: Any) -> str:
    d = tmp_path / "spotchecks"
    _attr_file(d, "a.json", crops)
    code, lines, err = _summary(["--dir", str(d)], capsys)
    assert code == 0, err
    [line] = [x for x in _block(lines, "outer_layer") if x.startswith("  verdict: ")]
    return line.removeprefix("  verdict: ")


def _paired(
    yy: int = 34, yn: int = 0, yu: int = 6, ny: int = 6, nn: int = 136, nu: int = 18
) -> list[tuple[int, str, str | None]]:
    """Crops by (reviewer, model) on outer_layer: yy reviewer yes and model yes, ... The
    default is exactly at the bar: reviewer yes 40, no 160, labelled 200; precision
    34/40, recall 34/40, specificity 136/160, each exactly 0.85."""
    cells = [("y", "y", yy), ("y", "n", yn), ("y", "u", yu), ("n", "y", ny), ("n", "n", nn)]
    cells.append(("n", "u", nu))
    return [(40, f"{r}nn", f"{m}nn") for r, m, count in cells for _ in range(count)]


def test_ac6_verdict_pass_at_the_bar(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _verdict(tmp_path, _paired(), capsys) == "pass"


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        # Recall 33/40 = 0.825 (one model u more); one false yes fewer keeps the
        # precision at 33/38 = 0.868.
        pytest.param({"yy": 33, "yu": 7, "ny": 5, "nu": 19}, "fail (recall)", id="recall"),
        # Precision 34/41 = 0.829 (one more false yes, one fewer model u on a no).
        pytest.param({"ny": 7, "nu": 17}, "fail (precision)", id="precision"),
        # Specificity 135/160 = 0.844 (one model u more on a no).
        pytest.param({"nn": 135, "nu": 19}, "fail (specificity)", id="specificity"),
        # A model that always says "cannot tell": recall and specificity 0, precision n/a.
        pytest.param(
            {"yy": 0, "yu": 40, "ny": 0, "nn": 0, "nu": 160},
            "fail (precision, recall, specificity)",
            id="always-unsure",
        ),
        pytest.param(
            {"yy": 29, "yu": 0, "nn": 165}, "insufficient (reviewer yes 29 < 30)", id="yes"
        ),
        pytest.param({"nn": 135}, "insufficient (reviewer yes+no 199 < 200)", id="labelled"),
    ],
)
def test_ac6_each_verdict_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cells: dict[str, int], expected: str
) -> None:
    assert _verdict(tmp_path, _paired(**cells), capsys) == expected


def test_ac6_verdict_reviewer_no_short(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Reviewer yes 180 (all model yes), no 29 (all model no): labelled 209.
    crops = _paired(yy=180, yu=0, ny=0, nn=29, nu=0)
    assert _verdict(tmp_path, crops, capsys) == "insufficient (reviewer no 29 < 30)"


def test_ac6_verdict_no_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "spotchecks"
    _attr_file(d, "a.json", [(40, "ynn", None)] * 50 + [(40, "nnn", None)] * 200, judge=None)
    code, lines, err = _summary(["--dir", str(d)], capsys)
    assert code == 0, err
    block = _block(lines, "outer_layer")
    assert "  model: none" in block
    assert block[-1] == "  verdict: no model"


def test_ac6_a_model_u_on_a_reviewer_u_is_not_paired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    crops = [*_paired(), *[(40, "unn", "unn")] * 50, *[(40, "unn", "ynn")] * 50]
    assert _verdict(tmp_path, crops, capsys) == "pass"


def _sweep(data: Path, started_at: str, precip: float | None) -> None:
    moment = datetime.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    sweep_id = moment.strftime("%Y%m%dT%H%MZ")
    weather = None
    if precip is not None:
        weather = {
            "temp_c": 12.0,
            "apparent_c": 11.0,
            "precip_mm": precip,
            "observed_at": started_at,
            "source": "metoffice",
        }
    record = {
        "schema": "sweep.v1",
        "sweep_id": sweep_id,
        "started_at": started_at,
        "finished_at": started_at,
        "persons_total": 0,
        "weather": weather,
    }
    path = data / "sweeps" / moment.strftime("%Y/%m/%d") / f"{sweep_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_ac6_the_rain_join(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d, data = tmp_path / "spotchecks", tmp_path / "data"
    _attr_file(d, "a.json", [(40, "nny", "nny"), (50, "nnn", None)], "2026-10-05T10:00Z")
    _attr_file(d, "b.json", [(40, "ynn", None)], "2026-10-05T14:00Z")
    _attr_file(d, "c.json", [(40, "unu", None)], "2026-10-05T18:00Z", "twilight")
    _sweep(data, "2026-10-05T10:12:00Z", 0.6)  # rain, 12 minutes away
    _sweep(data, "2026-10-05T09:40:00Z", 0.0)  # dry, but further away
    _sweep(data, "2026-10-05T14:25:00Z", 0.0)  # dry
    _sweep(data, "2026-10-05T18:31:00Z", 2.0)  # 31 minutes away: unknown
    code, lines, err = _summary(["--dir", str(d), "--data-dir", str(data)], capsys)
    assert code == 0, err
    umbrella = _block(lines, "umbrella")
    assert f"  rain rain: {_counts(1, 1, 0)}" in umbrella
    assert f"  rain dry: {_counts(0, 1, 0)}" in umbrella
    assert f"  rain unknown: {_counts(0, 0, 1)}" in umbrella
    outer = _block(lines, "outer_layer")
    assert f"  rain dry: {_counts(1, 0, 0)}" in outer
    assert f"  rain unknown: {_counts(0, 0, 1)}" in outer
    code, lines, err = _summary(["--dir", str(d)], capsys)
    assert code == 0, err
    assert not [line for line in lines if line.startswith("  rain ")]  # no --data-dir


def test_ac6_a_malformed_sweep_record_is_an_error_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d, data = tmp_path / "spotchecks", tmp_path / "data"
    _attr_file(d, "a.json", [(40, "nny", None)], "2026-10-05T10:00Z")
    path = data / "sweeps" / "2026" / "10" / "05" / "20261005T1005Z.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"started_at": "2026-10-05T10:05:00Z", "weather": {"precip_mm": "x"}}')
    code, _lines, err = _summary(["--dir", str(d), "--data-dir", str(data)], capsys)
    assert code == 1 and "20261005T1005Z.json" in err


VALID: dict[str, Any] = {
    "date": "2026-10-05",
    "started_at": "2026-10-05T10:00Z",
    "light": "day",
    "frames": 3,
    "detector": DETECTOR,
    "min_height_px": 31,
    "judge": MODEL,
    "crops_shown": 3,
    "crops_rejected": 1,
    "crops": [[40, "ynn", "ynu"], [33, "uuu", None]],
}


def _mutated(**changes: Any) -> bytes:
    record = {**VALID, **changes}
    return json.dumps({k: v for k, v in record.items() if v is not ...}).encode()


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"[]", id="not-an-object"),
        pytest.param(_mutated(crops=...), id="missing-field"),
        pytest.param(_mutated(camera="JamCams_00001.01234"), id="extra-field"),
        pytest.param(_mutated(date="2026-02-30"), id="bad-date"),
        pytest.param(_mutated(started_at="2026-10-05 10:00"), id="bad-started-at"),
        pytest.param(_mutated(light="dusk"), id="bad-light"),
        pytest.param(_mutated(frames=-1), id="negative-frames"),
        pytest.param(_mutated(min_height_px=31.5), id="float-min-height"),
        pytest.param(_mutated(judge=7), id="judge-not-a-name"),
        pytest.param(_mutated(judge="../../etc"), id="judge-bad-name"),
        pytest.param(_mutated(crops_rejected=4), id="more-rejected-than-shown"),
        pytest.param(_mutated(crops_shown=5), id="counts-disagree"),
        pytest.param(_mutated(crops=[[40, "ynn", "ynu"], [33, "yn", None]]), id="short-answer"),
        pytest.param(_mutated(crops=[[40, "YNN", "ynu"], [33, "uuu", None]]), id="capitals"),
        pytest.param(_mutated(crops=[[40, "ynx", "ynu"], [33, "uuu", None]]), id="x-answer"),
        pytest.param(_mutated(crops=[[40, "ynn", "yes"], [33, "uuu", None]]), id="bad-model"),
        pytest.param(_mutated(crops=[[40, "ynn"], [33, "uuu", None]]), id="two-items"),
        pytest.param(_mutated(crops=[[True, "ynn", None], [33, "uuu", None]]), id="bool-height"),
        pytest.param(_mutated(crops=[[-1, "ynn", None], [33, "uuu", None]]), id="neg-height"),
        pytest.param(_mutated(crops=[[40.5, "ynn", None], [33, "uuu", None]]), id="float-h"),
        pytest.param(_mutated(crops={"a": 1}), id="crops-not-a-list"),
        pytest.param(
            _mutated(judge=None, crops=[[40, "ynn", "ynu"], [33, "uuu", None]]),
            id="model-answer-without-judge",
        ),
        pytest.param(b'{"frames": 1e999}', id="huge-number"),
        pytest.param(json.dumps({**VALID, "frames": 10**400}).encode(), id="huge-int"),
        pytest.param(b"[" * 100000, id="deep-nesting"),
        pytest.param(b"\xff\xfe", id="not-utf8"),
        pytest.param(b" " * (spotcheck_summary.MAX_FILE_BYTES + 1), id="too-large"),
    ],
)
def test_ac6_a_malformed_attribute_file_is_an_error_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes
) -> None:
    d = tmp_path / "spotchecks"
    (d / "attributes").mkdir(parents=True)
    (d / "attributes" / "2026-10-05.json").write_text(json.dumps(VALID), encoding="utf-8")
    (d / "attributes" / "2026-10-06-2.json").write_bytes(content)
    code, _lines, err = _summary(["--dir", str(d)], capsys)
    assert code == 1
    assert "2026-10-06-2.json" in err


def test_ac6_a_valid_file_is_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "spotchecks"
    (d / "attributes").mkdir(parents=True)
    (d / "attributes" / "2026-10-05.json").write_text(json.dumps(VALID), encoding="utf-8")
    code, lines, err = _summary(["--dir", str(d)], capsys)
    assert code == 0, err
    assert "crops: 3 shown, 1 rejected, 2 labelled, 1 with a model answer" in lines


def test_ac6_no_attribute_directory_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _lines, err = _summary(["--dir", str(tmp_path / "nowhere")], capsys)
    assert code == 1 and err.startswith("summary: ")


def _stats_file(directory: Path, day: str, shown: int, not_person: int) -> None:
    record = {
        "date": day,
        "reviewer": "tester",
        "mode": "crops",
        "frames_reviewed": 5,
        "boxes_shown": shown,
        "boxes_not_person": not_person,
        "boxes_in_vehicle": 0,
        "persons_missed": None,
        "precision_person": round(1 - not_person / shown, 4),
        "precision_pedestrian": round(1 - not_person / shown, 4),
        "recall_estimate": None,
        "detector": DETECTOR,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{day}.json").write_text(json.dumps(record), encoding="utf-8")


def _box_file(directory: Path) -> None:
    record = {
        "date": "2026-09-20",
        "started_at": "2026-09-20T10:00Z",
        "light": "day",
        "frames": 3,
        "detector": DETECTOR,
        "boxes": [[40, "person"], [22, "unsure"]],
    }
    (directory / "boxes").mkdir(parents=True, exist_ok=True)
    (directory / "boxes" / "2026-09-20.json").write_text(json.dumps(record), encoding="utf-8")


def test_ac6_default_and_heights_output_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain, both = tmp_path / "plain", tmp_path / "both"
    for d in (plain, both):
        _stats_file(d, "2026-09-15", 100, 10)
        _stats_file(d, "2026-09-22", 50, 4)
        _box_file(d)
    _example(both)
    (both / "attributes" / "broken.json").write_bytes(b"not json")  # never read by these
    for argv in ([], ["--heights"]):
        assert spotcheck_summary.main([*argv, "--dir", str(plain)]) == 0
        expected = capsys.readouterr()
        assert spotcheck_summary.main([*argv, "--dir", str(both)]) == 0
        got = capsys.readouterr()
        assert got.out == expected.out.replace(str(plain), str(both)) and got.err == expected.err
        assert "attribute" not in got.out and "verdict" not in got.out
    # A malformed statistics file is still exit 1, and --data-dir alone still exit 2.
    (both / "2026-09-29.json").write_bytes(b"not json")
    assert spotcheck_summary.main(["--dir", str(both)]) == 1
    with pytest.raises(SystemExit) as stopped:
        spotcheck_summary.main(["--dir", str(both), "--data-dir", str(tmp_path)])
    assert stopped.value.code == 2


# AC8: docs ----------------------------------------------------------------------------


def test_ac8_readme_documents_the_attribute_session() -> None:
    text = README.read_text(encoding="utf-8")
    assert "--attributes" in text and "attributes/" in text
    for key in ("y", "n", "u", "x"):
        assert re.search(rf"^\|\s*`{key}`\s*\|", text, re.MULTILINE), key
    for _key, question in QUESTIONS:
        assert question in text
    for field in FIELDS:
        assert f"`{field}`" in text
    for word in (
        "MIN_POSITIVES",
        "MIN_NEGATIVES",
        "MIN_LABELLED",
        "TARGET_ATTRIBUTE_ACCURACY",
        "NEAR_FIELD_MIN_HEIGHT_PX",
        "ATTRIBUTE_PROMPT",
    ):
        assert word in text, word
    section = text.split("## Attribute", 1)[1]
    assert re.search(r"counts and labels only", section, re.IGNORECASE)
    assert re.search(r"cannot tell[^.]*counts as wrong", section, re.IGNORECASE)
