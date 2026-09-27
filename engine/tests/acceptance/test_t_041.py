"""Acceptance tests for T-041: spot-check baseline data and a near-field threshold.

Every frame here is synthetic (uniform colours or the fake camera server's noise), or, in
the dry run, one of the licensed fixture photos. Every file is written by the test or by
the tool into a temporary directory. The judge, where used, is a fake server on
127.0.0.1. Nothing reaches the network. The published sunrise, sunset and civil twilight
times are the US Naval Observatory's (aa.usno.navy.mil, "Sun and Moon Data for One Day"),
in UTC, for 51.5074 N, 0.1278 W.
"""

from __future__ import annotations

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
from wearreport.testing.fake_cameras import FakeCameraServer
from wearreport.tools import spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "spotchecks" / "README.md"
WINDOWS = sys.platform == "win32"
REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
REQUIRE_TK = "WEARREPORT_REQUIRE_TK"
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
MODEL = "di-qwen3-vl-235b"
H, W = 288, 352
DAY = datetime.date(2026, 9, 27)
NOON = datetime.datetime(2026, 9, 27, 12, 34, 56, tzinfo=datetime.UTC)
WAIT_S = 60
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".ppm"}
OLD_FIELDS = {
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
BOX_FIELDS = {"date", "started_at", "light", "frames", "detector", "boxes"}
LABELS = {"person", "in_vehicle", "not_person", "unsure"}
Z = 1.959963984540054  # the 97.5th percentile of the standard normal distribution

Frame = npt.NDArray[np.uint8]


# Helpers ------------------------------------------------------------------------------

# Box heights in source-frame pixels: 60, 45.4, 80.6 and 33.6, which round to 60, 45, 81
# and 34.
BOXES = (
    (10.0, 50.0, 30.0, 110.0),
    (60.0, 50.0, 80.0, 95.4),
    (110.0, 20.0, 150.0, 100.6),
    (180.0, 70.0, 200.0, 103.6),
)
HEIGHTS = (60, 45, 81, 34)


class Stub:
    """counts[i] people in the frame whose colour is 10*i, with the boxes of BOXES."""

    def __init__(self, counts: Sequence[int]) -> None:
        self.counts = list(counts)

    def detect(self, frame: Frame) -> list[detect.Detection]:
        n = self.counts[int(frame[0, 0, 0]) // 10]
        return [detect.Detection("person", 0.9, BOXES[k % len(BOXES)]) for k in range(n)]


class Everyone:
    """One person in every frame."""

    def detect(self, frame: Frame) -> list[detect.Detection]:
        return [detect.Detection("person", 0.9, BOXES[0])]


def _pipeline(counts: Sequence[int]) -> spotcheck.Pipeline:
    frames = []
    for i in range(len(counts)):
        frame = np.full((H, W, 3), 10 * i, dtype=np.uint8)
        frame[20:120, :, 1] = np.arange(W, dtype=np.uint8)[None, :]  # crops differ
        frame[0, 0, 0] = 10 * i
        frames.append(frame)
    return spotcheck.Pipeline(frames=lambda: frames, detector=Stub(counts), info=INFO)


class Labeller:
    """Labels box k as labels[k] ("person", "in_vehicle", "not_person" or "unsure";
    default person). Calls `during` (if given) when the review opens."""

    def __init__(
        self, labels: Mapping[int, str] | None = None, during: Callable[[], None] | None = None
    ) -> None:
        self.labels = dict(labels or {})
        self.during = during
        self.items: list[spotcheck.ReviewItem] = []

    def judge(
        self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
    ) -> dict[int, spotcheck.Judgement]:
        if self.during is not None:
            self.during()
        self.items = list(items)
        judgements: dict[int, spotcheck.Judgement] = {}
        for item in items:

            def boxes(label: str, item: spotcheck.ReviewItem = item) -> frozenset[int]:
                return frozenset(b for b in item.boxes if self.labels.get(b, "person") == label)

            judgements[item.number] = spotcheck.Judgement(
                boxes("not_person"),
                boxes("in_vehicle"),
                0 if mode == "frames" else None,
                unsure=boxes("unsure"),
            )
        return judgements


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


def _model(name: str) -> None:
    if not detect.model_path(name).is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run scripts/fetch_model.sh or fetch_model.ps1")


def _need_window() -> None:
    try:
        import tkinter

        tkinter.Tk().destroy()
    except Exception as exc:  # ImportError without Tk, TclError without a display
        if WINDOWS or os.environ.get(REQUIRE_TK):
            pytest.fail(f"no Tk window here: {exc}")
        pytest.skip("no Tk display here; the unit window tests run in the Windows job")


def _crop(number: int) -> spotcheck.ReviewItem:
    image = np.full((60, 40, 3), (10 * number) % 250, dtype=np.uint8)
    return spotcheck.ReviewItem(number, f"crop-{number:04d}.png", (number,), image)


def _frame_item() -> spotcheck.ReviewItem:
    return spotcheck.ReviewItem(1, "frame-0001.png", (1, 2, 3), np.zeros((H, W, 3), np.uint8))


# AC1: progress ------------------------------------------------------------------------

PROGRESS = re.compile(r"^spotcheck: (fetched|detected) (\d+) of (\d+)$", re.MULTILINE)
REVIEW_LINE = "spotcheck: opening the review"


def test_ac1_progress_lines_on_a_dry_run(
    env: Path, offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _model(spotcheck.DEFAULT_MODEL)
    seen: list[str] = []
    reviewer = Labeller(during=lambda: seen.append(capsys.readouterr().err))
    args = ["--dry-run", "--view", "files", "--n", "3", "--min-persons", "1", "--seed", "1"]
    assert _run(args, tmp_path / "out", reviewer=reviewer) == 0
    [before_review] = seen
    err = before_review + capsys.readouterr().err
    listed = f"spotcheck: {spotcheck.DRY_RUN_CAMERAS} cameras listed"
    assert listed in before_review.splitlines()
    lines = PROGRESS.findall(before_review)
    fetched = [(int(k), int(n)) for kind, k, n in lines if kind == "fetched"]
    detected = [(int(k), int(n)) for kind, k, n in lines if kind == "detected"]
    assert fetched[-1] == (spotcheck.DRY_RUN_CAMERAS, spotcheck.DRY_RUN_CAMERAS)
    assert detected and detected[-1][0] == detected[-1][1] <= spotcheck.DRY_RUN_CAMERAS
    order = [before_review.index(listed), before_review.index("fetched")]
    order += [before_review.index("detected"), before_review.index(REVIEW_LINE)]
    assert order == sorted(order)  # the review line comes last, before the review opens
    assert "http" not in err and "/cam/" not in err and "Fake_" not in err


def test_ac1_progress_at_least_every_100_frames_and_only_counts(
    env: Path, offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeCameraServer() as server:
        cameras = server.cameras(250)
        pipeline = spotcheck.Pipeline(
            frames=lambda: spotcheck.sweep_frames(cameras), detector=Everyone(), info=INFO
        )
        seen: list[str] = []
        reviewer = Labeller(during=lambda: seen.append(capsys.readouterr().err))
        args = ["--view", "files", "--n", "2", "--min-persons", "1", "--seed", "3"]
        assert _run(args, tmp_path / "out", pipeline=pipeline, reviewer=reviewer) == 0
    [err] = seen
    assert "spotcheck: 250 cameras listed" in err.splitlines()
    for kind in ("fetched", "detected"):
        counts = [int(k) for found, k, n in PROGRESS.findall(err) if found == kind and n == "250"]
        assert counts == sorted(counts) and counts[-1] == 250
        gaps = [b - a for a, b in zip([0, *counts], counts, strict=False)]
        assert max(gaps) <= 100, (kind, counts)
    assert err.rstrip().splitlines()[-1].startswith(REVIEW_LINE)
    for camera in cameras:
        assert camera.id not in err
    assert "http" not in err and "/cam/" not in err


# AC2: cannot tell ---------------------------------------------------------------------


def test_ac2_u_on_the_keyboard_line() -> None:
    assert spotcheck.parse_line("u", _crop(7), "crops").unsure == frozenset({7})
    assert spotcheck.parse_line("U7", _crop(7), "crops").unsure == frozenset({7})
    judgement = spotcheck.parse_line("u2 n1 m1", _frame_item(), "frames")
    assert judgement.unsure == frozenset({2}) and judgement.not_person == frozenset({1})
    assert judgement.missed == 1
    assert spotcheck.parse_line("", _crop(7), "crops").unsure == frozenset()
    for bad in ("u2 n2", "u9", "u1 u1"):
        with pytest.raises(spotcheck.JudgementError):
            spotcheck.parse_line(bad, _frame_item(), "frames")


def test_ac2_u_through_the_keyboard_reviewer() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"u\nu6\n\nv\n")
    os.close(write_fd)
    items = [_crop(5), _crop(6), _crop(7), _crop(8)]
    try:
        got = spotcheck.KeyboardReviewer(read_fd, io.StringIO()).judge(
            items, "crops", time.monotonic() + WAIT_S
        )
    finally:
        os.close(read_fd)
    assert got[5].unsure == frozenset({5}) and got[6].unsure == frozenset({6})
    assert got[7].unsure == frozenset() and got[8].in_vehicle == frozenset({8})


def test_ac2_unsure_in_the_json_judgements_file() -> None:
    items = [_crop(1), _crop(2)]
    raw = b'{"1": {"unsure": [1]}, "2": {"not_person": [2], "unsure": []}}'
    got = spotcheck.parse_judgements(raw, items, "crops")
    assert got[1].unsure == frozenset({1}) and got[2].unsure == frozenset()
    frames = spotcheck.parse_judgements(
        b'{"1": {"unsure": [3], "in_vehicle": [2], "missed": 0}}', [_frame_item()], "frames"
    )
    assert frames[1].unsure == frozenset({3})
    for bad in (
        b'{"1": {"unsure": [1], "not_person": [1]}, "2": {}}',
        b'{"1": {"unsure": [1], "in_vehicle": [1]}, "2": {}}',
        b'{"1": {"unsure": 1}, "2": {}}',
        b'{"1": {"unsure": [9]}, "2": {}}',
        b'{"1": {"unsure": [1, 1]}, "2": {}}',
        b'{"1": {"unsure": [true]}, "2": {}}',
    ):
        with pytest.raises(spotcheck.JudgementError):
            spotcheck.parse_judgements(bad, items, "crops")


def test_ac2_u_in_the_window_legend_and_the_window() -> None:
    assert "u: cannot tell" in spotcheck.WINDOW_LEGEND
    _need_window()
    items = [_crop(1), _crop(2), _crop(3)]
    keys = ["u", "Return", "n"]

    def driver(root: Any) -> None:
        def step() -> None:
            if keys:
                root.focus_force()
                key = keys.pop(0)
                root.event_generate("<Return>" if key == "Return" else f"<KeyPress-{key}>")
                root.after(30, step)

        root.after(30, step)

    got = spotcheck.WindowReviewer(driver=driver).judge(items, "crops", time.monotonic() + WAIT_S)
    assert got[1].unsure == frozenset({1})
    assert got[2] == spotcheck.Judgement(frozenset(), frozenset(), None)
    assert got[3].not_person == frozenset({3})


def test_ac2_unsure_left_out_of_the_statistics_and_counted_in_the_box_file(
    env: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    reviewer = Labeller({1: "unsure", 2: "not_person", 3: "in_vehicle"})
    args = ["--n", "1", "--min-persons", "1", "--view", "files", "--record-boxes"]
    assert _run(args, out, pipeline=_pipeline([4]), reviewer=reviewer) == 0
    stats = _json(out / f"{DAY.isoformat()}.json")
    assert set(stats) == OLD_FIELDS
    assert stats["boxes_shown"] == 3  # box 1 left out
    assert stats["boxes_not_person"] == 1 and stats["boxes_in_vehicle"] == 1
    assert stats["precision_person"] == round(2 / 3, 4)
    assert stats["precision_pedestrian"] == round(1 / 3, 4)
    assert stats["frames_reviewed"] == 1
    boxes = _json(out / "boxes" / f"{DAY.isoformat()}.json")["boxes"]
    assert sorted(map(tuple, boxes)) == sorted(
        [(60, "unsure"), (45, "not_person"), (81, "in_vehicle"), (34, "person")]
    )


def test_ac2_readme_lists_the_key() -> None:
    text = README.read_text(encoding="utf-8")
    assert re.search(r"^\|\s*`u`\s*\|.*cannot tell", text, re.MULTILINE | re.IGNORECASE)
    assert "`unsure`" in text


# The paired judge sees only the crops the reviewer did not mark unsure.


def _chat(text: str = "person") -> bytes:
    body = {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 300, "completion_tokens": 2},
    }
    return json.dumps(body).encode()


class _Judge(http.server.BaseHTTPRequestHandler):
    server: Any

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.bodies.append(body)
        reply = _chat()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _sent_images(bodies: Sequence[bytes]) -> list[Frame]:
    images = []
    for body in bodies:
        [message] = json.loads(body)["messages"]
        [url] = [p["image_url"]["url"] for p in message["content"] if p["type"] == "image_url"]
        png = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8)
        images.append(np.asarray(cv2.imdecode(png, cv2.IMREAD_COLOR), dtype=np.uint8))
    return images


def test_ac2_the_judge_gets_only_the_crops_not_marked_unsure(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (*PROXY_ENV, "DEEPINFRA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Judge)
    server.daemon_threads = True
    server.bodies = []  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "out"
        reviewer = Labeller({1: "unsure", 3: "unsure", 4: "not_person"})
        args = ["--n", "1", "--min-persons", "1", "--view", "files"]
        args += ["--judge", MODEL, "--judge-max-requests", "10"]
        code = _run(
            args,
            out,
            pipeline=_pipeline([4]),
            reviewer=reviewer,
            judge_endpoint=f"http://127.0.0.1:{server.server_address[1]}",
            judge_sleep=lambda seconds: None,
        )
    finally:
        server.shutdown()
        server.server_close()
    assert code == 0
    sent = _sent_images(server.bodies)  # type: ignore[attr-defined]
    judged = [item for item in reviewer.items if item.number in (2, 4)]
    assert len(sent) == len(judged) == 2
    for image, item in zip(sent, judged, strict=True):
        assert np.array_equal(image, item.image)
    block = _json(out / f"{DAY.isoformat()}.json")["judge"]
    assert block["status"] == "complete" and block["requests"] == 2
    assert block["confusion"]["person"]["person"] == 1
    assert block["confusion"]["not_person"]["person"] == 1
    assert sum(sum(row.values()) for row in block["confusion"].values()) == 2


# AC3: the per-box file ----------------------------------------------------------------


def test_ac3_the_box_file_has_exactly_its_fields(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    reviewer = Labeller({2: "not_person"})
    args = ["--n", "2", "--min-persons", "1", "--view", "files", "--record-boxes"]
    assert _run(args, out, pipeline=_pipeline([4, 2]), reviewer=reviewer) == 0
    stats = _json(out / f"{DAY.isoformat()}.json")
    record = _json(out / "boxes" / f"{DAY.isoformat()}.json")
    assert set(record) == BOX_FIELDS
    assert record["date"] == DAY.isoformat()
    assert record["started_at"] == "2026-09-27T12:34Z"  # the clock, to the minute
    assert record["light"] == "day"
    assert record["frames"] == 2 == stats["frames_reviewed"]
    assert record["detector"] == stats["detector"]
    for entry in record["boxes"]:
        assert isinstance(entry, list) and len(entry) == 2
        assert type(entry[0]) is int and entry[1] in LABELS
    expected = [(h, "person") for h in (*HEIGHTS, *HEIGHTS[:2])]
    expected[1] = (45, "not_person")  # box 2
    assert sorted(map(tuple, record["boxes"])) == sorted(expected)
    text = (out / "boxes" / f"{DAY.isoformat()}.json").read_text(encoding="utf-8")
    for word in ("x1", "y1", "x2", "width", "camera", "frame_index", "image", "png", "base64"):
        assert word not in text


def test_ac3_started_at_is_taken_when_the_sweep_begins(env: Path, tmp_path: Path) -> None:
    times = iter(
        [
            datetime.datetime(2026, 9, 27, 18, 5, 59, tzinfo=datetime.UTC),
            datetime.datetime(2026, 9, 27, 22, 0, tzinfo=datetime.UTC),
        ]
    )
    calls: list[str] = []

    def frames() -> list[Frame]:
        calls.append("sweep")
        return list(_pipeline([2]).frames())

    def clock() -> datetime.datetime:
        calls.append("clock")
        return next(times)

    pipeline = spotcheck.Pipeline(frames=frames, detector=Stub([2]), info=INFO)
    args = ["--n", "1", "--min-persons", "1", "--view", "files", "--record-boxes"]
    assert _run(args, tmp_path / "out", pipeline=pipeline, reviewer=Labeller(), clock=clock) == 0
    assert calls[:2] == ["clock", "sweep"]
    record = _json(tmp_path / "out" / "boxes" / f"{DAY.isoformat()}.json")
    assert record["started_at"] == "2026-09-27T18:05Z"
    assert record["light"] == "twilight"  # sunset 17:47, end of civil twilight 18:21


def test_ac3_nothing_else_is_written(env: Path, tmp_path: Path) -> None:
    work, home = Path.cwd(), Path(os.environ["HOME"])
    watched = [work, home, env]
    tempfile.gettempdir()  # tempfile's own writability probe, before the snapshot
    before = _files(watched)
    args = ["--n", "2", "--min-persons", "1", "--view", "files", "--record-boxes"]
    assert _run(args, work / "stats", pipeline=_pipeline([3, 4]), reviewer=Labeller()) == 0
    after = _files(watched)
    name = f"{DAY.isoformat()}.json"
    assert sorted(set(after) - set(before)) == sorted(
        [work / "stats" / name, work / "stats" / "boxes" / name]
    )
    assert not [
        p
        for p, data in after.items()
        if p.suffix.lower() in IMAGE_SUFFIXES or data.startswith(MAGIC)
    ]


def test_ac3_no_overwrite(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "boxes").mkdir(parents=True)
    name = f"{DAY.isoformat()}.json"
    (out / "boxes" / name).write_text("kept", encoding="utf-8")
    args = ["--n", "1", "--min-persons", "1", "--view", "files", "--record-boxes"]
    assert _run(args, out, pipeline=_pipeline([2]), reviewer=Labeller()) == 0
    assert (out / "boxes" / name).read_text(encoding="utf-8") == "kept"
    second = f"{DAY.isoformat()}-2.json"
    assert _json(out / second)["boxes_shown"] == 2
    assert len(_json(out / "boxes" / second)["boxes"]) == 2
    assert not (out / name).exists()  # the statistics file keeps its box file's name
    assert _run(args, out, pipeline=_pipeline([2]), reviewer=Labeller()) == 0
    third = f"{DAY.isoformat()}-3.json"
    assert (out / third).is_file() and (out / "boxes" / third).is_file()


def test_ac3_without_record_boxes_no_box_file_or_directory(env: Path, tmp_path: Path) -> None:
    work, home = Path.cwd(), Path(os.environ["HOME"])
    watched = [work, home, env]
    tempfile.gettempdir()  # tempfile's own writability probe, before the snapshot
    before = _files(watched)
    reviewer = Labeller({1: "unsure", 2: "not_person"})
    args = ["--n", "2", "--min-persons", "1", "--view", "files"]
    assert _run(args, work / "stats", pipeline=_pipeline([3, 4]), reviewer=reviewer) == 0
    after = _files(watched)
    assert sorted(set(after) - set(before)) == [work / "stats" / f"{DAY.isoformat()}.json"]
    assert not os.path.lexists(work / "stats" / "boxes")
    assert set(_json(work / "stats" / f"{DAY.isoformat()}.json")) == OLD_FIELDS


def test_ac3_privacy_guard_still_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "privacy_guard.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# AC3: the light at started_at ---------------------------------------------------------

# USNO, UTC, 51.5074 N 0.1278 W: (date, sunrise, sunset, end of civil twilight).
PUBLISHED = [
    ("2026-03-20", "06:03", "18:13", "18:47"),
    ("2026-06-21", "03:43", "20:22", "21:09"),
    ("2026-09-27", "05:55", "17:47", "18:21"),
    ("2026-12-21", "08:04", "15:53", "16:34"),
]
SUNRISE_DEG = -0.833  # the published times: the sun's upper limb on the horizon
CIVIL_DEG = -6.0


def _at(day: str, hhmm: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(f"{day}T{hhmm}:00+00:00")


def _crossing(published: datetime.datetime, level: float) -> datetime.datetime:
    """When the elevation passes `level` within 30 minutes of `published` (bisection)."""
    lo = published - datetime.timedelta(minutes=30)
    hi = published + datetime.timedelta(minutes=30)
    f_lo = spotcheck.solar_elevation(lo) - level
    assert f_lo * (spotcheck.solar_elevation(hi) - level) < 0
    for _ in range(40):
        mid = lo + (hi - lo) / 2
        f_mid = spotcheck.solar_elevation(mid) - level
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return lo


@pytest.mark.parametrize(("day", "rise", "sunset", "dusk"), PUBLISHED)
def test_ac3_solar_elevation_matches_published_times(
    day: str, rise: str, sunset: str, dusk: str
) -> None:
    for hhmm, level in ((rise, SUNRISE_DEG), (sunset, SUNRISE_DEG), (dusk, CIVIL_DEG)):
        published = _at(day, hhmm)
        error = abs((_crossing(published, level) - published).total_seconds())
        assert error <= 120, (day, hhmm, error)


@pytest.mark.parametrize(("day", "rise", "sunset", "dusk"), PUBLISHED)
def test_ac3_light_categories_at_known_times(day: str, rise: str, sunset: str, dusk: str) -> None:
    minutes = datetime.timedelta(minutes=1)
    assert spotcheck.light_at(_at(day, rise) - 12 * minutes) == "twilight"
    assert spotcheck.light_at(_at(day, rise) + 12 * minutes) == "day"
    assert spotcheck.light_at(_at(day, "12:00")) == "day"
    # The published sunset is at -0.833 degrees: the sun's centre is below 0 before it.
    assert spotcheck.light_at(_at(day, sunset) - 12 * minutes) == "day"
    assert spotcheck.light_at(_at(day, sunset) + 3 * minutes) == "twilight"
    assert spotcheck.light_at(_at(day, dusk) - 3 * minutes) == "twilight"
    assert spotcheck.light_at(_at(day, dusk) + 3 * minutes) == "dark"
    assert spotcheck.light_at(_at(day, "00:00")) == "dark"


def test_ac3_light_thresholds() -> None:
    assert spotcheck.light_category(0.0) == "day"
    assert spotcheck.light_category(35.0) == "day"
    assert spotcheck.light_category(-0.001) == "twilight"
    assert spotcheck.light_category(-6.0) == "twilight"
    assert spotcheck.light_category(-6.001) == "dark"
    assert spotcheck.LONDON == (51.5074, -0.1278)


# AC4: the height summary --------------------------------------------------------------

DETECTOR = {"model": "yolox_m", "sha256": "0" * 64, "conf": 0.3}


def _session(
    directory: Path,
    name: str,
    started_at: str,
    light: str,
    boxes: Sequence[tuple[int, str, int]],
    date: str | None = None,
) -> None:
    """A per-box file; `boxes` holds (height, label, how many)."""
    record = {
        "date": date or started_at[:10],
        "started_at": started_at,
        "light": light,
        "frames": 5,
        "detector": DETECTOR,
        "boxes": [[h, label] for h, label, count in boxes for _ in range(count)],
    }
    (directory / "boxes").mkdir(parents=True, exist_ok=True)
    (directory / "boxes" / name).write_text(json.dumps(record), encoding="utf-8")


def _sweep(data: Path, started_at: str, precip: float | None) -> None:
    """A published sweep record (the fields the summary reads, and a few others)."""
    moment = datetime.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    sweep_id = moment.strftime("%Y%m%dT%H%MZ")
    weather = None
    if precip is not None:
        weather = {
            "temp_c": 15.0,
            "apparent_c": 14.0,
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


def _wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z / d * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return centre - half, centre + half


def _stat(k: int, n: int) -> str:
    lo, hi = _wilson(k, n)
    return f"n={n} precision {k / n:.4f} wilson [{lo:.4f}, {hi:.4f}]"


def _heights(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = spotcheck_summary.main(["--heights", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_ac4_named_constants() -> None:
    assert spotcheck_summary.TARGET_PRECISION == 0.90
    assert spotcheck_summary.MIN_LOWER_BOUND == 0.85
    assert spotcheck_summary.MIN_BOXES_ABOVE == 100
    assert spotcheck_summary.BASELINE_BOXES == 300
    assert spotcheck_summary.MIN_CONDITION_BOXES == 50
    assert spotcheck_summary.RAIN_JOIN_MINUTES == 30


def _threshold_example(d: Path) -> None:
    # Judged boxes by height: 20 px 5 people and 45 not; 40 px 99 people (90 on foot, 9
    # in a vehicle) and 1 not; 60 px 30 people and 10 not; 80 px 60 people. And 6 unsure
    # boxes (4 at 40 px, 2 at 80 px), left out.
    boxes = [(20, "person", 5), (20, "not_person", 45), (40, "person", 60)]
    boxes += [(40, "not_person", 1), (40, "unsure", 4)]
    _session(d, "a.json", "2026-09-20T10:00Z", "day", boxes)
    boxes = [(40, "person", 30), (40, "in_vehicle", 9), (60, "person", 30)]
    boxes += [(60, "not_person", 10)]
    _session(d, "b.json", "2026-09-21T18:30Z", "twilight", boxes)
    _session(d, "c.json", "2026-09-21T20:30Z", "dark", [(80, "person", 60), (80, "unsure", 2)])


def test_ac4_hand_computed_threshold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Boxes >= H, judged: 250 (194 people), 200 (189), 100 (90) and 60 (60) for
    # H = 20, 40, 60, 80. Precision: 0.776, 0.945, 0.900, 1.000.
    # Wilson 95%, z = 1.96: with p = 189/200 = 0.945 and n = 200, z^2/n = 0.019207,
    #   centre = (0.945 + 0.009604) / 1.019207 = 0.936614,
    #   half = 1.96 / 1.019207 * sqrt(0.945 * 0.055 / 200 + 3.841459 / 160000)
    #        = 1.923028 * sqrt(0.000259875 + 0.000024009) = 1.923028 * 0.016849 = 0.032401,
    #   lower = 0.9042 >= 0.85. With p = 0.9 and n = 100, lower = 0.8256 < 0.85, and 60
    #   boxes are fewer than 100. So 20 px fails the precision, 60 px the lower bound, 80
    #   px the count: the threshold is 40 px, which keeps 189 of 194 people (0.9742).
    d = tmp_path / "spotchecks"
    _threshold_example(d)
    code, out, err = _heights(["--dir", str(d)], capsys)
    assert code == 0, err
    lines = out.splitlines()
    rows = [line for line in lines if line.startswith(">= ")]
    assert rows == [
        f">= 20 px: {_stat(194, 250)} kept 1.0000",
        f">= 40 px: {_stat(189, 200)} kept {189 / 194:.4f}",
        f">= 60 px: {_stat(90, 100)} kept {90 / 194:.4f}",
        f">= 80 px: {_stat(60, 60)} kept {60 / 194:.4f}",
    ]
    assert "wilson [0.9042," in rows[1] and "wilson [0.8256," in rows[2]
    assert "kept 0.9742" in rows[1]
    assert "threshold: 40 px" in lines
    assert "unsure boxes (left out): 6" in lines
    assert "at 40 px:" in lines
    # At 40 px: day 60 of 61, twilight 69 of 79, dark 60 of 60.
    assert f"  light day: {_stat(60, 61)}" in lines
    assert f"  light twilight: {_stat(69, 79)}" in lines
    assert f"  light dark: {_stat(60, 60)}" in lines
    assert not [line for line in lines if line.startswith("  rain ")]  # no --data-dir
    [coverage] = [line for line in lines if line.startswith("coverage:")]
    assert coverage == (
        "coverage: 3 sessions, 2 dates, 250 judged boxes "
        "(day 111, twilight 79, dark 60; rain 0, dry 0, unknown 250); "
        "baseline complete: no (judged boxes 250 < 300; rain 0 < 50)"
    )


def test_ac4_none_when_no_height_qualifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "spotchecks"
    # 99 boxes at 100% precision: too few. Adding 20 px boxes brings the count up but
    # the precision down: 120 boxes, 99 people, 0.825.
    _session(d, "a.json", "2026-09-20T10:00Z", "day", [(50, "person", 99), (20, "not_person", 21)])
    code, out, err = _heights(["--dir", str(d)], capsys)
    assert code == 0, err
    lines = out.splitlines()
    assert "threshold: none" in lines
    assert "over all boxes:" in lines
    assert f"  light day: {_stat(99, 120)}" in lines
    assert "  light twilight: n=0 precision n/a wilson n/a" in lines


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param([("2026-09-20T09:45:10Z", 0.4), ("2026-09-20T10:20:00Z", 0.0)], "rain"),
        pytest.param([("2026-09-20T09:35:00Z", 0.4), ("2026-09-20T10:10:00Z", 0.0)], "dry"),
        pytest.param([("2026-09-20T10:30:00Z", 0.2)], "rain", id="exactly-30-minutes"),
        pytest.param([("2026-09-20T10:30:01Z", 0.2), ("2026-09-20T09:29:00Z", 0.0)], "unknown"),
        pytest.param([("2026-09-20T09:55:00Z", None), ("2026-09-20T10:20:00Z", 1.0)], "unknown"),
        pytest.param([], "unknown", id="no-record"),
    ],
)
def test_ac4_rain_join(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    records: Sequence[tuple[str, float | None]],
    expected: str,
) -> None:
    d, data = tmp_path / "spotchecks", tmp_path / "data"
    (data / "sweeps").mkdir(parents=True)
    _session(d, "a.json", "2026-09-20T10:00Z", "day", [(50, "person", 3), (50, "not_person", 1)])
    for started_at, precip in records:
        _sweep(data, started_at, precip)
    code, out, err = _heights(["--dir", str(d), "--data-dir", str(data)], capsys)
    assert code == 0, err
    lines = out.splitlines()
    for condition in ("rain", "dry", "unknown"):
        want = _stat(3, 4) if condition == expected else "n=0 precision n/a wilson n/a"
        assert f"  rain {condition}: {want}" in lines


def test_ac4_rain_join_across_midnight(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d, data = tmp_path / "spotchecks", tmp_path / "data"
    _session(d, "a.json", "2026-09-21T00:10Z", "dark", [(50, "person", 2)])
    _sweep(data, "2026-09-20T23:50:00Z", 0.3)
    code, out, err = _heights(["--dir", str(d), "--data-dir", str(data)], capsys)
    assert code == 0, err
    assert f"  rain rain: {_stat(2, 2)}" in out.splitlines()


def test_ac4_a_malformed_sweep_record_is_an_error_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d, data = tmp_path / "spotchecks", tmp_path / "data"
    _session(d, "a.json", "2026-09-20T10:00Z", "day", [(50, "person", 2)])
    path = data / "sweeps" / "2026" / "09" / "20" / "20260920T1005Z.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"started_at": "2026-09-20T10:05:00Z", "weather": {"precip_mm": "x"}}')
    code, _out, err = _heights(["--dir", str(d), "--data-dir", str(data)], capsys)
    assert code == 1
    assert "20260920T1005Z.json" in err


# Coverage: (started_at, light, judged boxes, rain condition). The base passes: 300
# boxes, 3 sessions on 2 dates, day 200, twilight 100, rain 100.
BASE = [
    ("2026-09-20T10:00Z", "day", 100, "rain"),
    ("2026-09-21T18:30Z", "twilight", 100, "dry"),
    ("2026-09-21T11:00Z", "day", 100, "unknown"),
]
FAILING = {
    "judged boxes 299 < 300": [BASE[0], BASE[1], ("2026-09-21T11:00Z", "day", 99, "unknown")],
    "sessions 2 < 3": [
        ("2026-09-20T10:00Z", "day", 150, "rain"),
        ("2026-09-21T18:30Z", "twilight", 150, "dry"),
    ],
    "dates 1 < 2": [("2026-09-21T09:00Z", "day", 100, "rain"), BASE[1], BASE[2]],
    "day 49 < 50": [
        ("2026-09-20T10:00Z", "dark", 151, "rain"),
        BASE[1],
        ("2026-09-21T11:00Z", "day", 49, "unknown"),
    ],
    "twilight 49 < 50": [
        ("2026-09-20T10:00Z", "day", 151, "rain"),
        ("2026-09-21T18:30Z", "twilight", 49, "dry"),
        BASE[2],
    ],
    "rain 49 < 50": [
        ("2026-09-20T10:00Z", "day", 49, "rain"),
        BASE[1],
        ("2026-09-21T11:00Z", "day", 151, "unknown"),
    ],
}


def _coverage(
    tmp_path: Path, sessions: Sequence[tuple[str, str, int, str]], capsys: Any
) -> tuple[int, str]:
    d, data = tmp_path / "spotchecks", tmp_path / "data"
    (data / "sweeps").mkdir(parents=True)
    for i, (started_at, light, count, rain) in enumerate(sessions):
        _session(d, f"s{i}.json", started_at, light, [(50, "person", count), (50, "unsure", 2)])
        if rain != "unknown":
            _sweep(data, started_at[:-1] + ":00Z", 1.5 if rain == "rain" else 0.0)
    code, out, err = _heights(["--dir", str(d), "--data-dir", str(data)], capsys)
    assert code == 0, err
    [line] = [line for line in out.splitlines() if line.startswith("coverage:")]
    return len(sessions), line


def test_ac4_coverage_complete(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _n, line = _coverage(tmp_path, BASE, capsys)
    assert line == (
        "coverage: 3 sessions, 2 dates, 300 judged boxes "
        "(day 200, twilight 100, dark 0; rain 100, dry 100, unknown 100); "
        "baseline complete: yes"
    )


@pytest.mark.parametrize("reason", sorted(FAILING))
def test_ac4_each_coverage_condition_fails_on_its_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], reason: str
) -> None:
    _n, line = _coverage(tmp_path, FAILING[reason], capsys)
    assert line.endswith(f"; baseline complete: no ({reason})")


VALID = {
    "date": "2026-09-20",
    "started_at": "2026-09-20T10:00Z",
    "light": "day",
    "frames": 3,
    "detector": DETECTOR,
    "boxes": [[40, "person"], [22, "unsure"]],
}


def _mutated(**changes: Any) -> bytes:
    record = {**VALID, **changes}
    return json.dumps({k: v for k, v in record.items() if v is not ...}).encode()


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"[]", id="not-an-object"),
        pytest.param(_mutated(boxes=...), id="missing-field"),
        pytest.param(_mutated(camera="JamCams_00001.01234"), id="extra-field"),
        pytest.param(_mutated(date="2026-02-30"), id="bad-date"),
        pytest.param(_mutated(started_at="2026-09-20 10:00"), id="bad-started-at"),
        pytest.param(_mutated(started_at="2026-09-20T25:00Z"), id="bad-hour"),
        pytest.param(_mutated(light="dusk"), id="bad-light"),
        pytest.param(_mutated(frames=-1), id="negative-frames"),
        pytest.param(_mutated(boxes=[[40, "maybe"]]), id="bad-label"),
        pytest.param(_mutated(boxes=[[-1, "person"]]), id="negative-height"),
        pytest.param(_mutated(boxes=[[True, "person"]]), id="bool-height"),
        pytest.param(_mutated(boxes=[[40.5, "person"]]), id="float-height"),
        pytest.param(_mutated(boxes=[[40, "person", 3]]), id="three-items"),
        pytest.param(_mutated(boxes=[{"h": 40}]), id="object-box"),
        pytest.param(_mutated(boxes="40"), id="boxes-not-a-list"),
        pytest.param(b'{"frames": 1e999}', id="huge-number"),
        pytest.param(json.dumps({**VALID, "frames": 10**400}).encode(), id="huge-int"),
        pytest.param(b"[" * 100000, id="deep-nesting"),
        pytest.param(b"\xff\xfe", id="not-utf8"),
        pytest.param(b" " * (spotcheck_summary.MAX_FILE_BYTES + 1), id="too-large"),
    ],
)
def test_ac4_a_malformed_box_file_is_an_error_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes
) -> None:
    d = tmp_path / "spotchecks"
    (d / "boxes").mkdir(parents=True)
    (d / "boxes" / "2026-09-20.json").write_text(json.dumps(VALID), encoding="utf-8")
    (d / "boxes" / "2026-09-21-2.json").write_bytes(content)
    code, _out, err = _heights(["--dir", str(d)], capsys)
    assert code == 1
    assert "2026-09-21-2.json" in err


def test_ac4_a_valid_box_file_is_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "spotchecks"
    (d / "boxes").mkdir(parents=True)
    (d / "boxes" / "2026-09-20.json").write_text(json.dumps(VALID), encoding="utf-8")
    code, out, err = _heights(["--dir", str(d)], capsys)
    assert code == 0, err
    assert "unsure boxes (left out): 1" in out.splitlines()
    assert "threshold: none" in out.splitlines()


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


def test_ac4_default_summary_unchanged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plain, both = tmp_path / "plain", tmp_path / "both"
    for d in (plain, both):
        _stats_file(d, "2026-09-15", 100, 10)
        _stats_file(d, "2026-09-22", 50, 4)
    _threshold_example(both)
    assert spotcheck_summary.main(["--dir", str(plain)]) == 0
    expected = capsys.readouterr()
    assert spotcheck_summary.main(["--dir", str(both)]) == 0
    got = capsys.readouterr()
    assert got.out == expected.out.replace(str(plain), str(both)) and got.err == expected.err
    assert "threshold" not in got.out and "coverage" not in got.out
    # A malformed statistics file is still exit 1, and an unknown option still exit 2.
    (both / "2026-09-29.json").write_bytes(b"not json")
    assert spotcheck_summary.main(["--dir", str(both)]) == 1
    with pytest.raises(SystemExit) as stopped:
        spotcheck_summary.main(["--dir", str(both), "--data-dir", str(tmp_path)])
    assert stopped.value.code == 2


# AC6: docs ----------------------------------------------------------------------------


def test_ac6_readme_documents_the_box_file_and_heights() -> None:
    text = README.read_text(encoding="utf-8")
    assert "boxes/" in text and "--heights" in text and "--data-dir" in text
    for field in BOX_FIELDS:
        assert f"`{field}`" in text
    for word in ("twilight", "dark", "TARGET_PRECISION", "BASELINE_BOXES"):
        assert word in text
    stats_section = text.split("## Statistics file", 1)[1].split("\n## ", 1)[0]
    assert re.search(r"unsure[^.]*left out|left out[^.]*unsure", stats_section, re.IGNORECASE)
