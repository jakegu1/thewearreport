"""Unit tests for the paired spot-check's judge path (T-036): what a hostile or failing
DeepInfra reply may and may not make the tool do.

Every judge request goes to a fake server on the loopback interface, and every image is
synthetic: no test reaches the network or sends a real crop anywhere.
"""

from __future__ import annotations

import datetime
import http.server
import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport.tools import spotcheck

KEY_ENV = "DEEPINFRA_API_KEY"
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
MODEL = "di-qwen3-vl-235b"
DAY = datetime.date(2026, 9, 27)
H, W = 288, 352
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)

Frame = npt.NDArray[np.uint8]


# A fake DeepInfra server ----------------------------------------------------------------


@dataclass
class Reply:
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


def chat(text: str = "person") -> Reply:
    body = {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 300, "completion_tokens": 2},
    }
    return Reply(body=json.dumps(body).encode())


def error(status: int, message: str) -> Reply:
    return Reply(status=status, body=json.dumps({"error": {"message": message}}).encode())


@dataclass
class Fake:
    """A server on 127.0.0.1 that answers `replies` in order, then `default`, to any GET or
    POST, and records every request."""

    replies: list[Reply]
    default: Reply = field(default_factory=chat)
    requests: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    server: http.server.ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_address[1]}"


class _Handler(http.server.BaseHTTPRequestHandler):
    server: Any

    def _answer(self) -> None:
        fake: Fake = self.server.fake
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        with fake.lock:
            fake.requests.append(
                {"method": self.command, "path": self.path, "headers": dict(self.headers)}
            )
            reply = fake.replies.pop(0) if fake.replies else fake.default
        del body
        try:
            self.send_response(reply.status)
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)
        except OSError:
            pass

    do_GET = _answer
    do_POST = _answer

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Fake]]:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    started: list[Fake] = []

    def start(*replies: Reply, default: Reply | None = None) -> Fake:
        fake = Fake(list(replies)) if default is None else Fake(list(replies), default)
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


# The spot-check side --------------------------------------------------------------------


class Stub:
    """Each frame's detections: `boxes[i]` in frame i, whose first pixel is i."""

    def __init__(self, boxes: Sequence[Sequence[tuple[float, float, float, float]]]) -> None:
        self.boxes = boxes

    def detect(self, frame: Frame) -> list[detect.Detection]:
        return [detect.Detection("person", 0.9, b) for b in self.boxes[int(frame[0, 0, 0])]]


def _pipeline(
    boxes: Sequence[Sequence[tuple[float, float, float, float]]] | None = None,
    size: tuple[int, int] = (H, W),
) -> spotcheck.Pipeline:
    if boxes is None:
        boxes = [[(10.0 + 30 * k, 50.0, 30.0 + 30 * k, 110.0) for k in range(4)]]
    frames = []
    for i in range(len(boxes)):
        frame = np.full((*size, 3), 40, dtype=np.uint8)
        frame[40:120, :, 1] = (np.arange(size[1]) % 256).astype(np.uint8)[None, :]
        frame[0, 0, 0] = i
        frames.append(frame)
    return spotcheck.Pipeline(frames=lambda: frames, detector=Stub(boxes), info=INFO)


class Everyone:
    """A reviewer that says every box is a person."""

    def judge(
        self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
    ) -> Mapping[int, spotcheck.Judgement]:
        return {i.number: spotcheck.Judgement(frozenset(), frozenset(), None) for i in items}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Outside CI, in a fresh working directory and temporary directory."""
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(KEY_ENV, raising=False)
    work, tmp = tmp_path / "work", tmp_path / "tmp"
    work.mkdir()
    tmp.mkdir()
    monkeypatch.chdir(work)
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr("tempfile.tempdir", None)
    return tmp_path / "out"


def _run(judge: Fake, out: Path, pipeline: spotcheck.Pipeline | None = None) -> int:
    argv = ["--n", "5", "--min-persons", "1", "--view", "files", "--reviewer", "tester"]
    argv += ["--out-dir", str(out), "--judge", MODEL, "--judge-max-requests", "20"]
    return spotcheck.main(
        argv,
        pipeline=pipeline or _pipeline(),
        reviewer=Everyone(),
        today=DAY,
        judge_endpoint=judge.url,
        judge_timeout=5.0,
        judge_sleep=lambda seconds: None,
    )


def _stats_text(out: Path) -> str:
    return (out / f"{DAY.isoformat()}.json").read_text("utf-8")


# Redirects are never followed -----------------------------------------------------------


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_not_followed_and_the_key_goes_nowhere_else(
    env: Path,
    serve: Callable[..., Fake],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    key = "rd" + "K9" * 14
    monkeypatch.setenv(KEY_ENV, key)
    target = serve()
    moved = Reply(status=status, headers={"Location": f"{target.url}/steal"})
    judge = serve(default=moved)
    assert _run(judge, env) == 0
    assert target.requests == []
    assert len(judge.requests) == 1  # a redirect is not retried either
    stats = json.loads(_stats_text(env))
    assert stats["judge"]["status"] == "incomplete"
    assert stats["boxes_shown"] == 4  # the reviewer's statistics are kept
    out = capsys.readouterr()
    for text in (out.out, out.err, _stats_text(env)):
        assert key not in text
