"""Acceptance tests for T-036: a paired spot-check with a hosted judge.

Every judge request goes to a fake DeepInfra server on the loopback interface, and every
image is synthetic (or, in the dry run, rendered from the licensed fixture photos): no
test reaches the network or sends a real crop anywhere. These tests run on Linux and in
the Windows job.
"""

from __future__ import annotations

import ast
import base64
import datetime
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport._cv import cv2
from wearreport.tools import judge_hosted, spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "engine" / "wearreport" / "tools"
TOOL = TOOLS / "spotcheck.py"
HOSTED = TOOLS / "judge_hosted.py"
WINDOWS_WORKFLOW = ROOT / ".github" / "workflows" / "windows.yml"
WINDOWS = sys.platform == "win32"
REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
MODEL = "di-qwen3-vl-235b"
KEY_ENV = "DEEPINFRA_API_KEY"
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
H, W = 288, 352
DAY = datetime.date(2026, 9, 27)
WAIT_S = 60
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".ppm"}
REVIEWER_ROWS = ("person", "in_vehicle", "not_person")
JUDGE_COLUMNS = ("person", "in_vehicle", "not_person", "unsure")
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
JUDGE_FIELDS = {
    "model",
    "provider",
    "status",
    "requests",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "confusion",
    "judge_precision",
}

Frame = npt.NDArray[np.uint8]


# A fake DeepInfra server --------------------------------------------------------------


@dataclass
class Reply:
    status: int = 200
    body: bytes = b""
    delay: float = 0.0


def chat(
    text: str | None = "person", prompt_tokens: int = 300, completion_tokens: int = 2
) -> Reply:
    body = {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return Reply(body=json.dumps(body).encode())


def error(status: int, message: str = "failed") -> Reply:
    return Reply(status=status, body=json.dumps({"error": {"message": message}}).encode())


@dataclass
class FakeJudge:
    """A DeepInfra chat completions endpoint on 127.0.0.1: it answers `replies` in order,
    then `default`, and records every request."""

    replies: list[Reply]
    default: Reply = field(default_factory=chat)
    requests: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    server: http.server.ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def images(self) -> list[Frame]:
        """The image of each request, decoded from its PNG data URL."""
        found: list[Frame] = []
        for request in self.requests:
            [message] = json.loads(request["body"])["messages"]
            urls = [p["image_url"]["url"] for p in message["content"] if p["type"] == "image_url"]
            assert len(urls) == 1  # one crop per request
            assert urls[0].startswith("data:image/png;base64,")
            png = np.frombuffer(base64.b64decode(urls[0].split(",", 1)[1]), np.uint8)
            decoded = cv2.imdecode(png, cv2.IMREAD_COLOR)
            assert decoded is not None
            found.append(np.asarray(decoded, dtype=np.uint8))
        return found


class _Handler(http.server.BaseHTTPRequestHandler):
    server: Any

    def do_POST(self) -> None:
        fake: FakeJudge = self.server.fake
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        with fake.lock:
            fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
            reply = fake.replies.pop(0) if fake.replies else fake.default
        if reply.delay:
            time.sleep(reply.delay)
        try:
            self.send_response(reply.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)
        except OSError:
            pass  # the client gave up (a timeout test)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def judge_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., FakeJudge]]:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(KEY_ENV, raising=False)
    started: list[FakeJudge] = []

    def start(*replies: Reply) -> FakeJudge:
        fake = FakeJudge(list(replies))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        server.fake = fake  # type: ignore[attr-defined]
        fake.server = server
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(fake)
        return fake

    yield start
    for fake in started:
        assert fake.server is not None
        fake.server.shutdown()
        fake.server.server_close()


# The spot-check side ------------------------------------------------------------------


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
    frames = []
    for i in range(len(counts)):
        frame = np.full((H, W, 3), 10 * i, dtype=np.uint8)
        frame[40:120, :, 1] = np.arange(W, dtype=np.uint8)[None, :]  # crops differ
        frame[0, 0, 0] = 10 * i
        frames.append(frame)
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


class Labeller:
    """A reviewer that labels crop k as labels[k] ("person", "in_vehicle" or
    "not_person"; default person), and records what it saw while it judged."""

    def __init__(
        self,
        labels: Mapping[int, str] | None = None,
        fake: FakeJudge | None = None,
        tamper: bool = False,
        invalid: bool = False,
    ) -> None:
        self.labels = dict(labels or {})
        self.fake = fake
        self.tamper = tamper
        self.invalid = invalid
        self.items: list[spotcheck.ReviewItem] = []
        self.requests_while_judging: int | None = None
        self.directories: list[Path] = []
        self.finished_at: float | None = None

    def judge(
        self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
    ) -> dict[int, spotcheck.Judgement]:
        self.items = list(items)
        if self.fake is not None:
            self.requests_while_judging = len(self.fake.requests)
        self.directories = _leftovers(Path(tempfile.gettempdir()))
        if self.tamper:
            # Overwrite every rendered file with other (synthetic) pixels: a judge that read
            # the review directory would see these, not the crops in memory.
            for directory in self.directories:
                for path in directory.glob("crop-*"):
                    blank = np.zeros((20, 10, 3), dtype=np.uint8)
                    ok, encoded = cv2.imencode(".png", blank)
                    assert ok
                    path.write_bytes(encoded.tobytes())
        judgements: dict[int, spotcheck.Judgement] = {}
        for item in items:
            label = self.labels.get(item.number, "person")
            judgements[item.number] = spotcheck.Judgement(
                frozenset(item.boxes) if label == "not_person" else frozenset(),
                frozenset(item.boxes) if label == "in_vehicle" else frozenset(),
                None,
            )
        if self.invalid:
            judgements[items[0].number] = spotcheck.Judgement(frozenset({999}), frozenset(), None)
        self.finished_at = time.monotonic()
        return judgements


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Outside CI, in a fresh working directory, HOME and temporary directory; yields
    the temporary directory."""
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(KEY_ENV, raising=False)
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


def _paired(
    fake: FakeJudge | None,
    out: Path,
    *extra: str,
    view: str = "files",
    max_requests: int | None = 100,
    **kwargs: Any,
) -> int:
    argv = ["--n", "5", "--min-persons", "1", "--view", view, "--reviewer", "tester"]
    argv += ["--out-dir", str(out), *extra]
    if fake is not None:
        argv += ["--judge", MODEL]
        kwargs.setdefault("judge_endpoint", fake.url)
    if max_requests is not None and fake is not None:
        argv += ["--judge-max-requests", str(max_requests)]
    kwargs.setdefault("judge_sleep", lambda seconds: None)
    kwargs.setdefault("pipeline", _pipeline([4]))
    return spotcheck.main(argv, today=DAY, **kwargs)


def _stats(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / f"{DAY.isoformat()}.json").read_text("utf-8"))
    return data


def _model_path(name: str) -> Path:
    path = detect.model_path(name)
    if not path.is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run scripts/fetch_model.sh or fetch_model.ps1")
    return path


# AC1: options -------------------------------------------------------------------------


def test_ac1_judge_accepts_only_a_deepinfra_name() -> None:
    parser = spotcheck.build_parser()
    for name in judge_hosted.DEEPINFRA:
        args = parser.parse_args(["--n", "1", "--judge", name, "--judge-max-requests", "5"])
        assert args.judge == name and args.judge_max_requests == 5
    for bad in ("nova-2-lite", "qwen3.5-4b", "chosen", "https://example.org/x", "../x", ""):
        with pytest.raises(SystemExit):
            parser.parse_args(["--n", "1", "--judge", bad, "--judge-max-requests", "5"])
    for bad in ("0", "-1", "x", "1.5"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--n", "1", "--judge", MODEL, "--judge-max-requests", bad])
    if not WINDOWS:  # judge.py needs the POSIX-only resource module
        from wearreport.tools import judge

        assert judge.DEEPINFRA is judge_hosted.DEEPINFRA


def test_ac1_judge_is_refused_in_frames_mode(
    env: Path, judge_server: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = judge_server()
    code = _paired(
        fake, tmp_path / "out", "--mode", "frames", pipeline=_untouchable(), reviewer=Untouchable()
    )
    assert code != 0
    assert "frames" in capsys.readouterr().err
    assert fake.requests == [] and not (tmp_path / "out").exists()
    assert _leftovers(env) == []


def test_ac1_judge_needs_judge_max_requests(
    env: Path, judge_server: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = judge_server()
    code = _paired(
        fake, tmp_path / "out", max_requests=None, pipeline=_untouchable(), reviewer=Untouchable()
    )
    assert code != 0
    assert "--judge-max-requests" in capsys.readouterr().err
    assert fake.requests == [] and not (tmp_path / "out").exists()


def test_ac1_judge_max_requests_without_judge_is_refused(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--n", "1", "--judge-max-requests", "3", "--out-dir", str(tmp_path / "out")]
    code = spotcheck.main(argv, pipeline=_untouchable(), reviewer=Untouchable(), today=DAY)
    assert code != 0
    assert "--judge" in capsys.readouterr().err


def test_ac1_dry_run_with_a_real_endpoint_is_refused(
    env: Path, offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--n", "1", "--dry-run", "--judge", MODEL, "--judge-max-requests", "3"]
    argv += ["--out-dir", str(tmp_path / "out")]
    code = spotcheck.main(argv, pipeline=_untouchable(), reviewer=Untouchable(), today=DAY)
    assert code != 0
    assert "--dry-run" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_ac1_at_most_n_requests_retries_included(
    env: Path, judge_server: Any, tmp_path: Path
) -> None:
    # Crop 1 is answered on its second attempt; the budget of 2 ends the run there.
    fake = judge_server(error(500), chat("person"))
    code = _paired(fake, tmp_path / "out", max_requests=2, reviewer=Labeller())
    assert code == 0
    assert len(fake.requests) == 2
    block = _stats(tmp_path / "out")["judge"]
    assert block["status"] == "incomplete" and block["requests"] == 2
    assert sum(sum(row.values()) for row in block["confusion"].values()) == 1


def test_ac1_without_judge_the_statistics_are_unchanged(
    env: Path, judge_server: Any, tmp_path: Path
) -> None:
    fake = judge_server()
    reviewer = Labeller({2: "not_person", 3: "in_vehicle"})
    argv = ["--n", "5", "--min-persons", "1", "--view", "files", "--reviewer", "tester"]
    argv += ["--out-dir", str(tmp_path / "out")]
    assert spotcheck.main(argv, pipeline=_pipeline([4]), reviewer=reviewer, today=DAY) == 0
    text = (tmp_path / "out" / f"{DAY.isoformat()}.json").read_text("utf-8")
    items = list(reviewer.items)
    assert len(items) == 4
    judgements = dict(Labeller({2: "not_person", 3: "in_vehicle"}).judge(items, "crops", 0.0))
    expected = spotcheck.compute_stats(
        items,
        judgements,
        mode="crops",
        frames_reviewed=1,
        reviewer="tester",
        info=INFO,
        day=DAY,
    )
    assert text == json.dumps(expected, indent=2, ensure_ascii=True) + "\n"
    assert set(json.loads(text)) == OLD_FIELDS
    assert fake.requests == []


# AC2: order and independence ----------------------------------------------------------


@pytest.mark.parametrize("view", ["files", "window"])
def test_ac2_the_judge_is_called_only_after_the_review(
    env: Path, judge_server: Any, tmp_path: Path, view: str
) -> None:
    fake = judge_server()
    reviewer = Labeller(fake=fake)
    assert _paired(fake, tmp_path / "out", view=view, reviewer=reviewer) == 0
    assert reviewer.requests_while_judging == 0
    assert len(fake.requests) == 4
    assert _stats(tmp_path / "out")["judge"]["status"] == "complete"


def test_ac2_invalid_judgements_mean_no_judge_request(
    env: Path, judge_server: Any, tmp_path: Path
) -> None:
    fake = judge_server()
    code = _paired(fake, tmp_path / "out", reviewer=Labeller(invalid=True))
    assert code != 0
    assert fake.requests == []
    assert _leftovers(env) == []


@pytest.mark.parametrize("view", ["files", "window"])
def test_ac2_the_judge_sees_each_crop_as_rendered_from_memory(
    env: Path, judge_server: Any, tmp_path: Path, view: str
) -> None:
    fake = judge_server()
    reviewer = Labeller(fake=fake, tamper=view == "files")
    assert _paired(fake, tmp_path / "out", view=view, reviewer=reviewer) == 0
    if view == "files":
        assert len(reviewer.directories) == 1  # the files were there, and were tampered with
    sent = fake.images()
    assert len(sent) == len(reviewer.items) == 4
    for image, item in zip(sent, reviewer.items, strict=True):
        assert np.array_equal(image, item.image)
    assert _leftovers(env) == []


def test_ac2_the_judge_never_reads_the_answers() -> None:
    source = TOOL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "classify_live_crop"
    ]
    assert len(calls) == 1
    [call] = calls
    names = {n.id for n in ast.walk(call) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(call) if isinstance(n, ast.Attribute)}
    assert not {"judgements", "judgement", "not_person", "in_vehicle"} & names


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(lambda: [error(404, "no such model")], id="http-error"),
        pytest.param(lambda: [error(503)] * 10, id="server-errors"),
        pytest.param(lambda: [chat("person"), Reply(delay=3.0, body=b"{}")], id="timeout"),
        pytest.param(lambda: [Reply(body=b"not json")], id="malformed"),
    ],
)
@pytest.mark.parametrize("view", ["files", "window"])
def test_ac2_a_judge_failure_keeps_the_reviewer_statistics_and_deletes_the_directory(
    env: Path,
    judge_server: Any,
    tmp_path: Path,
    view: str,
    failure: Callable[[], list[Reply]],
) -> None:
    fake = judge_server(*failure())
    reviewer = Labeller({2: "not_person"})
    code = _paired(fake, tmp_path / "out", view=view, reviewer=reviewer, judge_timeout=0.5)
    assert code == 0
    stats = _stats(tmp_path / "out")
    assert stats["boxes_shown"] == 4 and stats["boxes_not_person"] == 1
    assert stats["precision_person"] == 0.75
    block = stats["judge"]
    assert block["status"] == "incomplete"
    assert block["requests"] == len(fake.requests) >= 1
    assert _leftovers(env) == []
    if view == "files":
        assert len(reviewer.directories) == 1
    else:
        assert reviewer.directories == []


def test_ac2_window_view_creates_no_image_and_no_directory(
    env: Path, judge_server: Any, tmp_path: Path
) -> None:
    work, home = Path.cwd(), Path(os.environ["HOME"])
    watched = [work, home, env]
    before = _files(watched)
    fake = judge_server()
    reviewer = Labeller(fake=fake)
    assert _paired(fake, work / "stats", view="window", reviewer=reviewer) == 0
    assert reviewer.directories == []
    stats_file = work / "stats" / f"{DAY.isoformat()}.json"
    after = _files(watched)
    assert sorted(set(after) - set(before)) == [stats_file]
    assert not [
        p
        for p, data in after.items()
        if p.suffix.lower() in IMAGE_SUFFIXES or data.startswith(MAGIC)
    ]


# AC3: privacy -------------------------------------------------------------------------


def test_ac3_only_crops_one_per_request(env: Path, judge_server: Any, tmp_path: Path) -> None:
    fake = judge_server()
    reviewer = Labeller()
    assert _paired(fake, tmp_path / "out", pipeline=_pipeline([2, 3]), reviewer=reviewer) == 0
    assert len(fake.requests) == 5
    for image, item in zip(fake.images(), reviewer.items, strict=True):
        assert image.shape == item.image.shape
        assert image.shape[:2] != (H, W) and image.shape[:2] != (2 * H, 2 * W)
    for request in fake.requests:
        assert request["path"] == judge_hosted.DEEPINFRA_PATH
        body = json.loads(request["body"])
        assert body["model"] == judge_hosted.DEEPINFRA[MODEL].model_id
        [message] = body["messages"]
        texts = [p["text"] for p in message["content"] if p["type"] == "text"]
        assert texts == [judge_hosted.PROMPT]


def test_ac3_the_endpoint_is_pinned() -> None:
    assert judge_hosted.DEEPINFRA_ORIGIN == "https://api.deepinfra.com"
    budget = judge_hosted.RequestBudget(1)
    live = judge_hosted.LiveCropJudge(MODEL, budget=budget)
    assert live.url == "https://api.deepinfra.com" + judge_hosted.DEEPINFRA_PATH
    live.close()
    for bad in (
        "https://example.org",
        "https://api.deepinfra.com.example.org",
        "http://api.deepinfra.com",
        "http://10.0.0.1:8080",
        "https://api.deepinfra.com/v1",
    ):
        with pytest.raises(judge_hosted.JudgeError):
            judge_hosted.LiveCropJudge(MODEL, budget=budget, endpoint=bad)
    with pytest.raises(judge_hosted.JudgeError):
        judge_hosted.LiveCropJudge("nova-2-lite", budget=budget)


def test_ac3_another_endpoint_is_refused_before_the_sweep(
    env: Path, tmp_path: Path, offline: None
) -> None:
    code = _paired(
        None,
        tmp_path / "out",
        "--judge",
        MODEL,
        "--judge-max-requests",
        "5",
        judge_endpoint="https://example.org",
        pipeline=_untouchable(),
        reviewer=Untouchable(),
    )
    assert code != 0
    assert not (tmp_path / "out").exists()


def test_ac3_nothing_about_a_crop_is_printed_or_written(
    env: Path, judge_server: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work, home = Path.cwd(), Path(os.environ["HOME"])
    watched = [work, home, env]
    before = _files(watched)
    fake = judge_server(chat("person"), chat("other"), error(500), chat("vehicle"))
    reviewer = Labeller()
    assert _paired(fake, work / "stats", reviewer=reviewer) == 0
    out = capsys.readouterr()
    printed = out.out + out.err
    for request in fake.requests:
        data_url = json.loads(request["body"])["messages"][0]["content"][0]["image_url"]["url"]
        assert data_url[22:80] not in printed
    assert "base64" not in printed and "data:image" not in printed
    after = _files(watched)
    assert sorted(set(after) - set(before)) == [work / "stats" / f"{DAY.isoformat()}.json"]
    text = after[work / "stats" / f"{DAY.isoformat()}.json"].decode("utf-8")
    assert "base64" not in text and "image" not in text


def test_ac3_only_the_spotcheck_module_uses_the_live_crop_entry_point() -> None:
    users = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "engine" / "wearreport").rglob("*.py")
        if "classify_live_crop" in path.read_text(encoding="utf-8")
        or "LiveCropJudge" in path.read_text(encoding="utf-8")
    )
    assert users == [
        "engine/wearreport/tools/judge_hosted.py",
        "engine/wearreport/tools/spotcheck.py",
    ]


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


@pytest.mark.parametrize("var", spotcheck.CI_VARIABLES)
def test_ac3_still_refuses_to_run_in_ci(
    env: Path, judge_server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, var: str
) -> None:
    fake = judge_server()
    monkeypatch.setenv(var, "true")
    code = _paired(fake, tmp_path / "out", pipeline=_untouchable(), reviewer=Untouchable())
    assert code == 2
    assert fake.requests == []


# AC4: the credential ------------------------------------------------------------------


def test_ac4_the_key_is_sent_as_a_bearer_token_when_set(
    env: Path, judge_server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = judge_server()
    key = "di-" + "k" * 40
    monkeypatch.setenv(KEY_ENV, key)
    assert _paired(fake, tmp_path / "out", reviewer=Labeller()) == 0
    assert len(fake.requests) == 4
    for request in fake.requests:
        assert request["headers"]["Authorization"] == f"Bearer {key}"


def test_ac4_no_header_without_the_variable(
    env: Path, judge_server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = judge_server()
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-" + "t" * 40)  # never sent
    assert _paired(fake, tmp_path / "out", reviewer=Labeller()) == 0
    assert len(fake.requests) == 4
    for request in fake.requests:
        assert "authorization" not in {k.lower() for k in request["headers"]}


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
def test_ac4_a_reply_echoing_the_key_does_not_leak_it(
    env: Path,
    judge_server: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    key = "sk" + "Q7" * 12 + "zz9"
    monkeypatch.setenv(KEY_ENV, key)
    echo = f"invalid key {key} (Bearer {key}) {key[:12]} {key[4:]}"
    fake = judge_server(*[error(status, echo)] * 5)
    assert _paired(fake, tmp_path / "out", reviewer=Labeller()) == 0
    out = capsys.readouterr()
    text = out.out + out.err + (tmp_path / "out" / f"{DAY.isoformat()}.json").read_text("utf-8")
    assert key not in text and key[:12] not in text and key[4:] not in text
    assert _stats(tmp_path / "out")["judge"]["status"] == "incomplete"


def test_ac4_a_malformed_key_is_refused_before_the_sweep(
    env: Path,
    judge_server: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = judge_server()
    key = "bad key\nwith spaces"
    monkeypatch.setenv(KEY_ENV, key)
    code = _paired(fake, tmp_path / "out", pipeline=_untouchable(), reviewer=Untouchable())
    assert code != 0
    err = capsys.readouterr().err
    assert KEY_ENV in err and "with spaces" not in err
    assert fake.requests == []


def test_ac4_env_example_documents_the_variable_empty() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert f"{KEY_ENV}=" in lines


# AC5: statistics ----------------------------------------------------------------------


def test_ac5_the_judge_block(env: Path, judge_server: Any, tmp_path: Path) -> None:
    # Reviewer: 1 person, 2 not a person, 3 in a vehicle, 4 person.
    # Judge:    person,   person,         other,          unsure.
    fake = judge_server(chat("person"), chat("person"), chat("other"), chat("unsure"))
    reviewer = Labeller({2: "not_person", 3: "in_vehicle"})
    assert _paired(fake, tmp_path / "out", reviewer=reviewer) == 0
    stats = _stats(tmp_path / "out")
    assert set(stats) == OLD_FIELDS | {"judge"}
    assert stats["precision_person"] == 0.75
    block = stats["judge"]
    assert set(block) == JUDGE_FIELDS
    assert block["model"] == MODEL
    assert block["provider"] == "DeepInfra"
    assert block["status"] == "complete"
    assert block["requests"] == 4
    assert block["input_tokens"] == 1200 and block["output_tokens"] == 8
    assert block["cost_usd"] == pytest.approx((1200 * 0.20 + 8 * 0.88) / 1e6)
    assert set(block["confusion"]) == set(REVIEWER_ROWS)
    for row in block["confusion"].values():
        assert set(row) == set(JUDGE_COLUMNS)
    assert block["confusion"] == {
        "person": {"person": 1, "in_vehicle": 0, "not_person": 0, "unsure": 1},
        "in_vehicle": {"person": 0, "in_vehicle": 0, "not_person": 1, "unsure": 0},
        "not_person": {"person": 1, "in_vehicle": 0, "not_person": 0, "unsure": 0},
    }
    # Confident answers: 3, of which 2 say a person (on foot or in a vehicle).
    assert block["judge_precision"] == round(2 / 3, 4)


def test_ac5_readme_documents_the_block_and_the_summary() -> None:
    readme = (ROOT / "spotchecks" / "README.md").read_text(encoding="utf-8")
    for needed in (
        "--judge",
        "--judge-max-requests",
        KEY_ENV,
        "`judge`",
        "`confusion`",
        "`judge_precision`",
        "`status`",
        "`incomplete`",
        "`cost_usd`",
        "wearreport.tools.spotcheck_summary",
        "Rogan",
    ):
        assert needed in readme, needed


# AC6: the summary ---------------------------------------------------------------------


def _confusion(
    person: Sequence[int], in_vehicle: Sequence[int], not_person: Sequence[int]
) -> dict[str, dict[str, int]]:
    rows = {"person": person, "in_vehicle": in_vehicle, "not_person": not_person}
    return {r: dict(zip(JUDGE_COLUMNS, counts, strict=True)) for r, counts in rows.items()}


def _file(
    directory: Path,
    day: str,
    shown: int,
    not_person: int,
    confusion: dict[str, dict[str, int]] | None,
    name: str | None = None,
) -> None:
    record: dict[str, Any] = {
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
        "detector": {"model": "yolox_m", "sha256": "0" * 64, "conf": 0.3},
    }
    if confusion is not None:
        record["judge"] = {
            "model": MODEL,
            "provider": "DeepInfra",
            "status": "complete",
            "requests": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "cost_usd": 0.0,
            "confusion": confusion,
            "judge_precision": None,
        }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name or f"{day}.json")).write_text(json.dumps(record), encoding="utf-8")


W38 = _confusion((80, 0, 5, 0), (0, 5, 0, 0), (2, 0, 8, 0))  # Se 85/90, Sp 8/10
W39 = _confusion((43, 0, 2, 0), (0, 0, 0, 0), (1, 0, 4, 0))
W40 = W39


def _rg(apparent: Fraction, se: Fraction, sp: Fraction) -> float:
    value = (apparent + sp - 1) / (se + sp - 1)
    return float(min(Fraction(1), max(Fraction(0), value)))


def test_ac6_rogan_gladen_on_hand_computed_examples() -> None:
    # Se 0.9, Sp 0.8, apparent 0.7: (0.7 + 0.8 - 1) / (0.9 + 0.8 - 1) = 0.5 / 0.7.
    earlier = _confusion((90, 0, 10, 0), (0, 0, 0, 0), (2, 0, 8, 0))
    current = _confusion((60, 0, 20, 5), (10, 0, 0, 0), (0, 0, 10, 0))
    assert spotcheck_summary.corrected_estimate(current, earlier) == pytest.approx(5 / 7)
    # in_vehicle counts as a person on both sides, and unsure answers are left out.
    earlier_v = _confusion((40, 45, 10, 7), (0, 5, 0, 0), (1, 1, 8, 3))
    assert spotcheck_summary.corrected_estimate(current, earlier_v) == pytest.approx(5 / 7)
    # Clipped to [0, 1].
    high = _confusion((95, 0, 5, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    assert spotcheck_summary.corrected_estimate(high, earlier) == 1.0
    low = _confusion((10, 0, 90, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    assert spotcheck_summary.corrected_estimate(low, earlier) == 0.0


def test_ac6_no_earlier_data_or_a_singular_confusion_is_not_available() -> None:
    current = _confusion((60, 0, 20, 0), (0, 0, 0, 0), (0, 0, 20, 0))
    empty = _confusion((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    assert spotcheck_summary.corrected_estimate(current, empty) is None
    # Se 0.5 and Sp 0.5: the judge's answer says nothing about the truth.
    coin = _confusion((5, 0, 5, 0), (0, 0, 0, 0), (5, 0, 5, 0))
    assert spotcheck_summary.corrected_estimate(current, coin) is None
    # No negatives earlier: specificity unknown.
    no_negatives = _confusion((9, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    assert spotcheck_summary.corrected_estimate(current, no_negatives) is None
    # Only unsure answers this week.
    unsure = _confusion((0, 0, 0, 9), (0, 0, 0, 0), (0, 0, 0, 1))
    assert spotcheck_summary.corrected_estimate(unsure, W38) is None


def _week_line(output: str, week: str) -> str:
    [line] = [line for line in output.splitlines() if line.startswith(week)]
    return line


def test_ac6_summary_per_iso_week(
    tmp_path: Path, offline: None, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _file(d, "2026-09-15", 100, 10, W38)  # ISO week 38
    _file(d, "2026-09-22", 30, 3, W39)  # week 39, two files
    _file(d, "2026-09-24", 20, 2, None)  # an old file without a judge block
    _file(d, "2026-09-29", 50, 5, W40)  # week 40
    (d / "README.md").write_text("not a statistics file", encoding="utf-8")
    assert spotcheck_summary.main(["--dir", str(d)]) == 0
    output = capsys.readouterr().out

    first = _week_line(output, "2026-W38")
    assert "reviewer 0.9000 (n=100)" in first
    assert "judge 0.8700" in first
    assert "corrected n/a" in first

    second = _week_line(output, "2026-W39")
    assert "reviewer 0.9000 (n=50)" in second
    assert "judge 0.8800" in second
    expected = _rg(Fraction(44, 50), Fraction(85, 90), Fraction(8, 10))
    assert f"corrected {expected:.4f}" in second
    assert f"diff {100 * (expected - 0.9):+.2f} pts" in second
    assert "within 3 pts: yes" in second

    third = _week_line(output, "2026-W40")
    expected = _rg(Fraction(44, 50), Fraction(128, 135), Fraction(12, 15))
    assert f"corrected {expected:.4f}" in third
    assert "within 3 pts: yes" in third
    assert "two weeks within 3 points: yes" in output


def test_ac6_the_condition_fails_when_a_week_is_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _file(d, "2026-09-15", 100, 10, W38)
    _file(d, "2026-09-22", 50, 5, W39)
    _file(d, "2026-09-29", 50, 20, W40)  # the reviewer finds 60%: the judge is far off
    assert spotcheck_summary.main(["--dir", str(d)]) == 0
    output = capsys.readouterr().out
    assert "within 3 pts: no" in _week_line(output, "2026-W40")
    assert "two weeks within 3 points: no" in output


def test_ac6_one_week_is_not_enough(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "stats"
    _file(d, "2026-09-15", 100, 10, W38)
    assert spotcheck_summary.main(["--dir", str(d)]) == 0
    output = capsys.readouterr().out
    assert "corrected n/a" in _week_line(output, "2026-W38")
    assert "two weeks within 3 points: no" in output


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b"[]",
        b'{"date": "2026-02-30", "boxes_shown": 1, "boxes_not_person": 0}',
        b'{"date": "2026-09-15", "boxes_shown": -1, "boxes_not_person": 0}',
        b'{"date": "2026-09-15", "boxes_shown": 1, "boxes_not_person": 2}',
        b'{"date": "2026-09-15", "boxes_shown": 1e999, "boxes_not_person": 0}',
        b'{"date": "2026-09-15", "boxes_shown": 1, "boxes_not_person": 0, "judge": 3}',
        b'{"date": "2026-09-15", "boxes_shown": 1, "boxes_not_person": 0,'
        b' "judge": {"model": "x", "confusion": {"person": {"person": "1"}}}}',
        pytest.param(b"[" * 100000, id="deep-nesting"),
        b"\xff\xfe",
    ],
)
def test_ac6_a_malformed_file_is_an_error_naming_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes
) -> None:
    d = tmp_path / "stats"
    d.mkdir()
    (d / "2026-09-15.json").write_bytes(content)
    assert spotcheck_summary.main(["--dir", str(d)]) != 0
    assert "2026-09-15.json" in capsys.readouterr().err


def test_ac6_summary_reads_counts_only_and_makes_no_network_call() -> None:
    source = (TOOLS / "spotcheck_summary.py").read_text(encoding="utf-8")
    imports = set(re.findall(r"^\s*(?:import|from)\s+([\w.]+)", source, re.MULTILINE))
    for forbidden in ("urllib", "http", "socket", "ssl", "requests", "wearreport.fetch"):
        assert not any(i == forbidden or i.startswith(forbidden + ".") for i in imports)
    assert "judge_hosted" not in source


# AC9: Windows and the T-037 contract ---------------------------------------------------

BLOCK = r"""
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in ("llama_cpp", "resource") or name == "wearreport.tools.judge":
            raise ImportError(f"{name} is blocked")
        return None

sys.meta_path.insert(0, Block())
from wearreport.tools import judge_hosted, spotcheck, spotcheck_summary
args = spotcheck.build_parser().parse_args(
    ["--n", "1", "--judge", "di-qwen3-vl-235b", "--judge-max-requests", "3"]
)
assert args.judge == "di-qwen3-vl-235b"
print(sorted(m for m in sys.modules if "llama" in m or m.endswith(".judge") or m == "resource"))
"""


def test_ac9_judge_hosted_imports_without_llama_cpp_resource_or_judge(tmp_path: Path) -> None:
    environ = {k: v for k, v in os.environ.items() if k not in spotcheck.CI_VARIABLES}
    result = subprocess.run(
        [sys.executable, "-c", BLOCK],
        env=environ,
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
    tree = ast.parse(HOSTED.read_text(encoding="utf-8"))
    imported = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    imported += [
        f"{n.module}.{a.name}"
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module
        for a in n.names
    ]
    assert imported
    bad = [
        i for i in imported if "llama" in i or i.split(".")[0] == "resource" or i.endswith(".judge")
    ]
    assert bad == []


@pytest.mark.skipif(WINDOWS, reason="judge.py needs the POSIX-only resource module")
def test_ac9_judge_still_exports_the_moved_names() -> None:
    from wearreport.tools import judge

    for name in (
        "Answer",
        "ANSWERS",
        "PROMPT",
        "ANSWER_WORDS",
        "parse_answer",
        "validate_image",
        "JudgeError",
        "HostedCandidate",
        "DEEPINFRA",
        "DEEPINFRA_REGION",
        "cost_usd",
        "mark_licensed",
        "is_licensed",
        "RequestBudget",
        "RequestLimitReached",
        "Usage",
        "DeepInfraClassifier",
        "REQUEST_TIMEOUT",
        "MAX_RETRIES",
        "MAX_RESPONSE_BYTES",
        "MAX_ERROR_CHARS",
        "AUTH_STATUSES",
        "DEEPINFRA_PATH",
    ):
        assert getattr(judge, name) is getattr(judge_hosted, name), name
    assert judge.BedrockClassifier.__mro__[1].__module__ == "wearreport.tools.judge_hosted"


def test_ac9_the_bake_off_paths_still_refuse_an_unmarked_crop(judge_server: Any) -> None:
    fake = judge_server()
    rng = np.random.default_rng(5)
    image: Frame = rng.integers(0, 256, (120, 60, 3), dtype=np.uint8)
    bakeoff = judge_hosted.DeepInfraClassifier(
        judge_hosted.DEEPINFRA[MODEL],
        budget=judge_hosted.RequestBudget(5),
        endpoint=fake.url,
        sleep=lambda seconds: None,
    )
    with pytest.raises(judge_hosted.JudgeError, match="gold-set"):
        bakeoff.classify(image)
    assert fake.requests == []
    assert bakeoff.classify(judge_hosted.mark_licensed(image)) == "person"
    assert len(fake.requests) == 1
    # The live entry point takes the unmarked crop, and only through its own method.
    live = judge_hosted.LiveCropJudge(
        MODEL, budget=judge_hosted.RequestBudget(5), endpoint=fake.url, sleep=lambda s: None
    )
    assert not hasattr(live, "classify")
    assert live.classify_live_crop(image) == "person"
    assert len(fake.requests) == 2
    live.close()
    with pytest.raises(judge_hosted.JudgeError):
        live.classify_live_crop(image)


def test_ac9_the_windows_job_runs_these_tests() -> None:
    text = WINDOWS_WORKFLOW.read_text(encoding="utf-8")
    step = text.split("- name: Spot-check unit and acceptance tests", 1)[1].split("- name:", 1)[0]
    assert "engine/tests/acceptance/test_t_036.py" in step
    assert "engine/tests/acceptance/test_t_037.py" in step


# The paired dry run -------------------------------------------------------------------


def test_dry_run_paired_with_the_fake_judge(
    env: Path, judge_server: Any, tmp_path: Path, offline: None
) -> None:
    """The dry run over the fixture photos, the default model, and the fake judge; prints
    the statistics file (the PR's evidence)."""
    _model_path(spotcheck.DEFAULT_MODEL)
    fake = judge_server(chat("other"), chat("person"))
    work = Path.cwd()
    argv = ["--dry-run", "--view", "files", "--n", "3", "--min-persons", "1", "--seed", "1"]
    argv += ["--reviewer", "tester", "--out-dir", str(work / "stats")]
    argv += ["--judge", MODEL, "--judge-max-requests", "500"]
    reviewer = Labeller({1: "not_person"})
    code = spotcheck.main(
        argv, reviewer=reviewer, today=DAY, judge_endpoint=fake.url, judge_sleep=lambda s: None
    )
    assert code == 0
    stats_file = work / "stats" / f"{DAY.isoformat()}.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    print("statistics file:", stats_file.name, json.dumps(stats, indent=2, sort_keys=True))
    block = stats["judge"]
    assert block["status"] == "complete"
    assert block["requests"] == stats["boxes_shown"] == len(fake.requests)
    assert block["confusion"]["not_person"]["not_person"] == 1
    assert _leftovers(env) == []
