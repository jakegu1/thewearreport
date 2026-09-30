"""Acceptance tests for T-053: attribute sessions show only crops tall enough to judge
(--min-height), the attribute summary shows the judgeable share by height, and the hosted
judge retries a dropped connection. The task contract: do not edit.

Every frame here is synthetic (uniform colours with a gradient), the reviewers are
scripted, every file is written by the test or by the tool into a temporary directory, and
the hosted model is a fake server on 127.0.0.1. Nothing reaches the network.
"""

from __future__ import annotations

import datetime
import json
import re
import socket
import struct
import tempfile
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport.tools import judge_hosted, spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "spotchecks" / "README.md"
H, W = 288, 352
DAY = datetime.date(2026, 10, 12)
NOON = datetime.datetime(2026, 10, 12, 11, 22, 33, tzinfo=datetime.UTC)
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
KEY_ENV = "DEEPINFRA_API_KEY"
MODEL = "di-qwen3-vl-235b"
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
WINDOW = ["--n", "5", "--min-persons", "1", "--view", "window"]
FLAG_REFUSAL = (
    "spotcheck: --min-height applies to attribute sessions only; add --attributes or drop it"
)

Frame = npt.NDArray[np.uint8]
Box = tuple[float, float, float, float]


def _box(x: float, height: float) -> Box:
    return (x, 60.0, x + 20.0, 60.0 + height)


# Boxes by height in source-frame pixels.
B30 = _box(10.0, 30.0)
B31 = _box(40.0, 31.0)
B45 = _box(70.0, 45.0)
B46 = _box(100.0, 46.0)
B60 = _box(130.0, 60.0)
BOXES = [[B31, B45], [B46, B60, B30]]


# Helpers ------------------------------------------------------------------------------


class Stub:
    """boxes[i] are the person boxes in the frame whose colour is 10*i."""

    def __init__(self, boxes: Sequence[Sequence[Box]]) -> None:
        self.boxes = [list(b) for b in boxes]

    def detect(self, frame: Frame) -> list[detect.Detection]:
        return [detect.Detection("person", 0.9, b) for b in self.boxes[int(frame[0, 0, 0]) // 10]]


def _pipeline(boxes: Sequence[Sequence[Box]] = BOXES) -> spotcheck.Pipeline:
    frames = []
    for i in range(len(boxes)):
        frame = np.full((H, W, 3), 10 * i, dtype=np.uint8)
        frame[10:200, :, 1] = np.arange(W, dtype=np.uint8)[None, :]  # crops differ
        frame[0, 0, 0] = 10 * i
        frames.append(frame)
    return spotcheck.Pipeline(frames=lambda: frames, detector=Stub(boxes), info=INFO)


def _never() -> spotcheck.Pipeline:
    def frames() -> list[Frame]:
        raise AssertionError("the sweep must not start")

    return spotcheck.Pipeline(frames=frames, detector=Stub([]), info=INFO)


class Answers:
    """An attribute reviewer answering `default` for every crop; records what it saw."""

    def __init__(self, default: str | None = "ynn") -> None:
        self.default = default
        self.items: list[spotcheck.ReviewItem] = []
        self.calls = 0

    def attributes(
        self, items: Sequence[spotcheck.ReviewItem], deadline: float
    ) -> dict[int, str | None]:
        self.calls += 1
        self.items = list(items)
        return {item.number: self.default for item in items}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Outside CI, offline, in a fresh working directory, HOME and temporary directory;
    yields the temporary directory. The live and dry-run pipelines must not be opened."""
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

    def no_network(self: socket.socket, address: Any) -> None:
        raise AssertionError(f"connection attempted: {address!r}")

    def no_pipeline(*args: Any, **kwargs: Any) -> spotcheck.Pipeline:
        raise AssertionError("the pipeline must not be opened")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(spotcheck, "live_pipeline", no_pipeline)
    monkeypatch.setattr(spotcheck, "dry_run_pipeline", no_pipeline)
    yield tmp


def _run(args: Sequence[str], out: Path, **kwargs: Any) -> int:
    argv = [*args, "--reviewer", "tester", "--out-dir", str(out)]
    return spotcheck.main(argv, today=DAY, clock=lambda: NOON, **kwargs)


def _record_bytes(out: Path) -> bytes:
    return (out / "attributes" / f"{DAY.isoformat()}.json").read_bytes()


def _record(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_record_bytes(out).decode("utf-8"))
    return data


def _files(root: Path) -> list[Path]:
    return sorted(root.rglob("*")) if root.exists() else []


# AC1: the flag ------------------------------------------------------------------------


def test_the_boxes_are_the_heights_the_tests_say() -> None:
    heights = [spotcheck.box_height(b) for b in (B30, B31, B45, B46, B60)]
    assert heights == [30, 31, 45, 46, 60]


def test_ac1_min_height_46_shows_only_46_and_60_and_records_46(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    reviewer = Answers()
    args = ["--attributes", *WINDOW, "--min-height", "46"]
    assert _run(args, out, pipeline=_pipeline(), reviewer=reviewer) == 0
    assert len(reviewer.items) == 2
    assert [item.boxes for item in reviewer.items] == [(1,), (2,)]
    record = _record(out)
    assert record["min_height_px"] == 46
    assert type(record["min_height_px"]) is int
    assert record["crops_shown"] == 2 and record["crops_rejected"] == 0
    assert record["crops"] == [[46, "ynn", None], [60, "ynn", None]]


def test_ac1_the_crops_shown_are_those_of_the_tall_boxes(env: Path, tmp_path: Path) -> None:
    reviewer = Answers()
    pipeline = _pipeline()
    args = ["--attributes", *WINDOW, "--min-height", "46"]
    assert _run(args, tmp_path / "out", pipeline=pipeline, reviewer=reviewer) == 0
    frames = list(pipeline.frames())
    expected = [
        spotcheck.render_crop(frames[1], detect.Detection("person", 0.9, B46), 1),
        spotcheck.render_crop(frames[1], detect.Detection("person", 0.9, B60), 2),
    ]
    assert len(reviewer.items) == 2
    for item, image in zip(reviewer.items, expected, strict=True):
        assert np.array_equal(item.image, image)


def test_ac1_min_persons_counts_the_tall_boxes_only(env: Path, tmp_path: Path) -> None:
    # Frame 0 has two near-field boxes (31, 45) but none of at least 46 px.
    out = tmp_path / "out"
    reviewer = Answers()
    args = ["--attributes", "--n", "5", "--min-persons", "2", "--view", "window"]
    args += ["--min-height", "46"]
    assert _run(args, out, pipeline=_pipeline(), reviewer=reviewer) == 0
    assert [item.boxes for item in reviewer.items] == [(1,), (2,)]
    assert _record(out)["frames"] == 1


def test_ac1_the_bounds_are_accepted(env: Path, tmp_path: Path) -> None:
    for n, shown in (("31", [31, 45, 46, 60]), ("200", [])):
        out = tmp_path / f"out-{n}"
        reviewer = Answers()
        boxes = [[B31, B45], [B46, B60, B30], [_box(150.0, 200.0)]]
        args = ["--attributes", *WINDOW, "--min-height", n]
        assert _run(args, out, pipeline=_pipeline(boxes), reviewer=reviewer) == 0
        record = _record(out)
        assert record["min_height_px"] == int(n)
        assert [crop[0] for crop in record["crops"]] == [*shown, 200]


def test_ac1_without_the_flag_nothing_changes(env: Path, tmp_path: Path) -> None:
    plain, explicit = tmp_path / "plain", tmp_path / "explicit"
    assert _run(["--attributes", *WINDOW], plain, pipeline=_pipeline(), reviewer=Answers()) == 0
    args = ["--attributes", *WINDOW, "--min-height", "31"]
    assert _run(args, explicit, pipeline=_pipeline(), reviewer=Answers()) == 0
    record = _record(plain)
    assert record["min_height_px"] == 31 == spotcheck.NEAR_FIELD_MIN_HEIGHT_PX
    assert [crop[0] for crop in record["crops"]] == [31, 45, 46, 60]
    # --min-height 31 is the default made explicit: the same bytes.
    assert _record_bytes(plain) == _record_bytes(explicit)


def test_ac1_refused_without_attributes(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    args = ["--n", "5", "--min-persons", "1", "--judgements", str(tmp_path / "j.json")]
    for extra in (["--min-height", "46"], ["--min-height", "31"]):
        assert _run([*args, *extra], out, pipeline=_never()) == 1
        captured = capsys.readouterr()
        assert captured.err == FLAG_REFUSAL + "\n"
        assert captured.out == ""
    assert not out.exists()
    assert _files(env) == []


def test_ac1_refused_without_attributes_before_the_pipeline_is_opened(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert _run(["--n", "5", "--min-height", "46"], out) == 1
    assert capsys.readouterr().err == FLAG_REFUSAL + "\n"
    assert not out.exists()


@pytest.mark.parametrize("value", ["30", "201", "4.5", "abc", "", "-46", "0", "1e2", "46.0"])
def test_ac1_values_out_of_range_or_not_integers_are_refused(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    out = tmp_path / "out"
    reviewer = Answers()
    with pytest.raises(SystemExit) as refused:
        _run(["--attributes", *WINDOW, "--min-height", value], out, reviewer=reviewer)
    assert refused.value.code not in (0, None)
    err = capsys.readouterr().err
    assert "--min-height" in err
    assert reviewer.calls == 0
    assert not out.exists()
    assert _files(env) == []


# AC2: the line that names the threshold -----------------------------------------------


def test_ac2_the_threshold_line_names_n(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    args = ["--attributes", *WINDOW, "--min-height", "61"]
    assert _run(args, out, pipeline=_pipeline(), reviewer=Answers()) == 1
    err = capsys.readouterr().err
    assert "(boxes at least 61 px tall)" in err
    assert "31 px" not in err
    assert not out.exists()


def test_ac2_without_the_flag_the_line_names_31(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert (
        _run(["--attributes", *WINDOW], out, pipeline=_pipeline([[B30]]), reviewer=Answers()) == 1
    )
    err = capsys.readouterr().err
    assert err.endswith(
        "spotcheck: no frame had at least 1 near-field person detections "
        "(boxes at least 31 px tall)\n"
    )


# AC3: judgeable share by height -------------------------------------------------------


def _attribute_file(
    folder: Path, name: str, started_at: str, light: str, crops: list[list[Any]]
) -> None:
    record = {
        "date": started_at[:10],
        "started_at": started_at,
        "light": light,
        "frames": 3,
        "detector": {"model": "stub", "sha256": "0" * 64, "conf": 0.35},
        "min_height_px": 31,
        "judge": None,
        "crops_shown": len(crops) + 1,
        "crops_rejected": 1,
        "crops": crops,
    }
    (folder / name).write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def labelled(tmp_path: Path) -> Path:
    """A day file, a twilight file and a dark file (which the bands leave out)."""
    folder = tmp_path / "spot" / "attributes"
    folder.mkdir(parents=True)
    day = [[31, "ynu", None], [35, "uuu", None], [40, "nyu", None], [46, "yyy", None]]
    day.append([70, "unn", None])
    _attribute_file(folder, "2026-10-12.json", "2026-10-12T11:00Z", "day", day)
    twilight = [[36, "uyu", None], [50, "nnn", None], [60, "uuy", None]]
    _attribute_file(folder, "2026-10-13.json", "2026-10-13T17:30Z", "twilight", twilight)
    dark = [[31, "yyy", None], [44, "nnn", None], [65, "nnn", None]]
    _attribute_file(folder, "2026-10-14.json", "2026-10-14T21:00Z", "dark", dark)
    return tmp_path / "spot"


def _band(name: str, band: str, crops: int, answered: int) -> str:
    # Not indented: an attribute's indented block (T-045) ends at its verdict line.
    share = f"{answered / crops:.4f}" if crops else "n/a"
    return (
        f"{name} height {band} px, day and twilight (dark excluded): {crops} crop(s), "
        f"{answered} answered yes or no, share {share}"
    )


BAND_LINE = re.compile(r"\w+ height ")


BANDS = ("31-35", "36-40", "41-45", "46-50", "51-60", "61+")
EXPECTED_BANDS = {
    "outer_layer": [(2, 1), (2, 1), (0, 0), (2, 2), (1, 0), (1, 0)],
    "bare_legs": [(2, 1), (2, 2), (0, 0), (2, 2), (1, 0), (1, 1)],
    "umbrella": [(2, 0), (2, 0), (0, 0), (2, 2), (1, 1), (1, 1)],
}


def _summary(directory: Path, capsys: pytest.CaptureFixture[str]) -> list[str]:
    assert spotcheck_summary.main(["--attributes", "--dir", str(directory)]) == 0
    return capsys.readouterr().out.splitlines()


def test_ac3_one_line_per_band_after_each_attribute(
    labelled: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _summary(labelled, capsys)
    for name, counts in EXPECTED_BANDS.items():
        start = lines.index(f"{name}:")
        verdict = next(i for i in range(start, len(lines)) if lines[i].startswith("  verdict: "))
        expected = [_band(name, band, *c) for band, c in zip(BANDS, counts, strict=True)]
        assert lines[verdict + 1 : verdict + 1 + len(BANDS)] == expected
        after = verdict + 1 + len(BANDS)
        assert after == len(lines) or not BAND_LINE.match(lines[after])


def test_ac3_existing_lines_are_unchanged(
    labelled: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _summary(labelled, capsys)
    kept = [line for line in lines if not BAND_LINE.match(line)]
    labellings = spotcheck_summary.load_labellings(labelled)
    everything = spotcheck_summary.attributes_report(labellings, labelled, rain=False)
    assert kept == [line for line in everything if not BAND_LINE.match(line)]
    assert len(everything) == len(kept) + 3 * len(BANDS)
    # The lines before the bands, as they were: counts over every file, dark included.
    assert lines[0] == f"3 attribute file(s) in {labelled / 'attributes'}"
    assert lines[1] == "crops: 14 shown, 3 rejected, 11 labelled, 0 with a model answer"
    start = lines.index("outer_layer:")
    assert lines[start + 1].startswith("  reviewer: yes 3, no 4, cannot tell 4; ")
    assert lines[start + 4].startswith("  light dark: yes 1, no 2, cannot tell 0; ")
    assert lines[start + 5] == "  model: none"
    assert lines[start + 6] == "  verdict: no model"


def test_ac3_the_dark_file_changes_no_band(
    labelled: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with_dark = [line for line in _summary(labelled, capsys) if BAND_LINE.match(line)]
    (labelled / "attributes" / "2026-10-14.json").unlink()
    without = [line for line in _summary(labelled, capsys) if BAND_LINE.match(line)]
    assert with_dark == without
    assert len(with_dark) == 3 * len(BANDS)


def test_ac3_no_files_still_prints_every_band(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "attributes").mkdir()
    lines = _summary(tmp_path, capsys)
    for name in EXPECTED_BANDS:
        for band in BANDS:
            assert lines.count(_band(name, band, 0, 0)) == 1


# AC4: the judge retries a dropped connection -------------------------------------------


def _chat(text: str) -> bytes:
    body = {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 300, "completion_tokens": 12},
    }
    return json.dumps(body).encode()


def _read_request(conn: socket.socket) -> bytes:
    """One HTTP request, headers and body (by Content-Length)."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(65536)
        if not chunk:
            return data
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    found = re.search(rb"(?im)^content-length:\s*(\d+)", head)
    length = int(found.group(1)) if found else 0
    while len(body) < length:
        chunk = conn.recv(65536)
        if not chunk:
            break
        body += chunk
    return head + b"\r\n\r\n" + body


class ResettingServer:
    """A server on 127.0.0.1 that reads each request, then resets the connection (a TCP
    RST, before any HTTP response) for the first `resets` requests, and answers the rest
    with `reply`."""

    def __init__(self, resets: int, reply: bytes) -> None:
        self.resets = resets
        self.reply = reply
        self.requests = 0
        self.head_first = False  # send a response head before the reset
        self.lock = threading.Lock()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                _read_request(conn)
                with self.lock:
                    self.requests += 1
                    reset = self.requests <= self.resets
                if reset:
                    if self.head_first:
                        head = "HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n{"
                        conn.sendall(head.encode())
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    continue
                head = (
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(self.reply)}\r\nConnection: close\r\n\r\n"
                )
                conn.sendall(head.encode() + self.reply)

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def resetting(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[int, bytes], ResettingServer]]:
    for name in (*PROXY_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    started: list[ResettingServer] = []

    def start(resets: int, reply: bytes) -> ResettingServer:
        server = ResettingServer(resets, reply)
        started.append(server)
        return server

    yield start
    for server in started:
        server.close()


CROP = np.full((90, 60, 3), 77, dtype=np.uint8)
# The messages a dropped connection gave before T-053, which stay as they were: a reset
# while waiting for the reply (raised by http.client, outside urllib's URLError), and a
# connection that could not be made (a URLError).
RESET_MESSAGE = "the request failed (ConnectionResetError)"
REFUSED_MESSAGE = "cannot reach DeepInfra (ConnectionRefusedError)"


def _attribute_judge(
    url: str, budget: judge_hosted.RequestBudget, sleeps: list[float]
) -> judge_hosted.LiveAttributeJudge:
    return judge_hosted.LiveAttributeJudge(
        MODEL, budget=budget, endpoint=url, timeout=10.0, sleep=sleeps.append
    )


def test_ac4_a_reset_connection_is_retried_then_succeeds(
    resetting: Callable[[int, bytes], ResettingServer],
) -> None:
    server = resetting(1, _chat("outer=yes legs=no umbrella=unsure"))
    budget = judge_hosted.RequestBudget(10)
    sleeps: list[float] = []
    judge = _attribute_judge(server.url, budget, sleeps)
    try:
        assert judge.classify_attributes(CROP) == "ynu"
        assert judge.usage.requests == 2
    finally:
        judge.close()
    assert server.requests == 2
    assert budget.used == 2
    assert sleeps == [judge_hosted.BACKOFF_SECONDS]


def test_ac4_the_crop_judge_retries_a_reset_too(
    resetting: Callable[[int, bytes], ResettingServer],
) -> None:
    server = resetting(2, _chat("person"))
    budget = judge_hosted.RequestBudget(10)
    sleeps: list[float] = []
    judge = judge_hosted.LiveCropJudge(
        MODEL, budget=budget, endpoint=server.url, timeout=10.0, sleep=sleeps.append
    )
    try:
        assert judge.classify_live_crop(CROP) == "person"
        assert judge.usage.requests == 3
    finally:
        judge.close()
    assert server.requests == 3
    assert budget.used == 3
    assert sleeps == [judge_hosted.BACKOFF_SECONDS, 2 * judge_hosted.BACKOFF_SECONDS]


def test_ac4_gives_up_after_max_retries_with_the_same_message(
    resetting: Callable[[int, bytes], ResettingServer],
) -> None:
    server = resetting(1000, _chat("outer=yes legs=no umbrella=unsure"))
    budget = judge_hosted.RequestBudget(100)
    sleeps: list[float] = []
    judge = _attribute_judge(server.url, budget, sleeps)
    try:
        with pytest.raises(judge_hosted.JudgeError) as failed:
            judge.classify_attributes(CROP)
        assert judge.usage.requests == judge_hosted.MAX_RETRIES + 1
    finally:
        judge.close()
    assert judge_hosted.MAX_RETRIES == 3
    assert str(failed.value) == RESET_MESSAGE
    assert not isinstance(failed.value, judge_hosted.RequestLimitReached)
    assert server.requests == judge_hosted.MAX_RETRIES + 1
    assert budget.used == judge_hosted.MAX_RETRIES + 1
    backoff = judge_hosted.BACKOFF_SECONDS
    assert sleeps == [backoff * 2**k for k in range(judge_hosted.MAX_RETRIES)]


def test_ac4_a_refused_connection_is_retried_and_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (*PROXY_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    closed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()  # nothing listens there: every connection is refused
    budget = judge_hosted.RequestBudget(100)
    sleeps: list[float] = []
    judge = _attribute_judge(f"http://127.0.0.1:{port}", budget, sleeps)
    try:
        with pytest.raises(judge_hosted.JudgeError) as failed:
            judge.classify_attributes(CROP)
        assert judge.usage.requests == judge_hosted.MAX_RETRIES + 1
    finally:
        judge.close()
    assert str(failed.value) == REFUSED_MESSAGE
    assert budget.used == judge_hosted.MAX_RETRIES + 1
    assert len(sleeps) == judge_hosted.MAX_RETRIES


def test_ac4_a_reset_after_the_response_began_is_not_retried(
    resetting: Callable[[int, bytes], ResettingServer],
) -> None:
    server = resetting(1000, _chat("outer=yes legs=no umbrella=unsure"))
    server.head_first = True
    budget = judge_hosted.RequestBudget(100)
    sleeps: list[float] = []
    judge = _attribute_judge(server.url, budget, sleeps)
    try:
        with pytest.raises(judge_hosted.JudgeError):
            judge.classify_attributes(CROP)
        assert judge.usage.requests == 1
    finally:
        judge.close()
    assert server.requests == 1
    assert sleeps == []


def test_ac4_each_retry_takes_from_the_budget(
    resetting: Callable[[int, bytes], ResettingServer],
) -> None:
    server = resetting(1000, _chat("outer=yes legs=no umbrella=unsure"))
    budget = judge_hosted.RequestBudget(2)
    sleeps: list[float] = []
    judge = _attribute_judge(server.url, budget, sleeps)
    try:
        with pytest.raises(judge_hosted.RequestLimitReached):
            judge.classify_attributes(CROP)
        assert judge.usage.requests == 2
    finally:
        judge.close()
    assert server.requests == 2
    assert budget.used == 2


# AC5: the docs ------------------------------------------------------------------------


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start : end if end != -1 else len(text)]


def test_ac5_the_readme_documents_the_flag_and_the_bands() -> None:
    text = README.read_text(encoding="utf-8")
    assert re.search(r"(?m)^\| `--min-height N` \|", text)
    section = _section(text, "## Attribute session (`--attributes`)")
    assert "`--min-height N`" in section
    for bound in ("31", "200"):
        assert bound in section
    assert "height band" in section
    for band in ("31-35", "36-40", "41-45", "46-50", "51-60", "61+"):
        assert band in section
    assert "dark" in section
    assert re.search(r"attribute threshold[^.]*chosen", section, re.S)
