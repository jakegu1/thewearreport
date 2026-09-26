"""Unit tests for the hosted judge backends: DeepInfra (`DeepInfraClassifier`,
`deepinfra_main`) and Bedrock's injected-credential path.

Every request goes to a local fake server on the loopback interface: no test reaches the
network. Every image is synthetic and marked as a gold-set crop by the test itself, except
where a test checks that an unmarked image is refused.
"""

from __future__ import annotations

import base64
import http.server
import json
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport._cv import cv2
from wearreport.tools import goldset, judge

ROOT = Path(__file__).resolve().parents[3]
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")


@dataclass
class Reply:
    status: int = 200
    body: bytes = b""
    delay: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)


def chat(
    text: str | None = "person", finish: str = "stop", usage: dict[str, Any] | None = None
) -> Reply:
    body: dict[str, Any] = {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        ],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 300, "completion_tokens": 2, "total_tokens": 302},
    }
    return Reply(body=json.dumps(body).encode())


def converse(text: str = "person") -> Reply:
    body = {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 120, "outputTokens": 2},
    }
    return Reply(body=json.dumps(body).encode())


def error(status: int, message: str = "error") -> Reply:
    return Reply(status=status, body=json.dumps({"error": {"message": message}}).encode())


class FakeServer:
    """Answers each POST with the next scripted Reply (the last one repeats) and records
    what it received."""

    def __init__(self, replies: list[Reply]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                fake.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                reply = fake.replies.pop(0) if len(fake.replies) > 1 else fake.replies[0]
                if reply.delay:
                    time.sleep(reply.delay)
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", "application/json")
                    for key, value in reply.headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(reply.body)))
                    self.end_headers()
                    self.wfile.write(reply.body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., FakeServer]]:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    started: list[FakeServer] = []

    def start(*replies: Reply) -> FakeServer:
        fake = FakeServer(list(replies))
        started.append(fake)
        return fake

    yield start
    for fake in started:
        fake.close()


def crop(seed: int = 0) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    image: npt.NDArray[np.uint8] = rng.integers(0, 256, (120, 60, 3), dtype=np.uint8)
    return judge.mark_licensed(image)


def candidate(reasoning_off: bool = False) -> judge.HostedCandidate:
    return next(c for c in judge.DEEPINFRA.values() if c.reasoning_off is reasoning_off)


def deepinfra(fake: FakeServer, limit: int = 100, **kwargs: Any) -> judge.DeepInfraClassifier:
    return judge.DeepInfraClassifier(
        kwargs.pop("candidate", None) or candidate(),
        budget=judge.RequestBudget(limit),
        endpoint=fake.url,
        sleep=lambda seconds: None,
        **kwargs,
    )


def _crops(n: int = 4) -> judge.Crops:
    labels: list[goldset.Label] = ["person", "in_vehicle", "not_person", "person"]

    def crops() -> Iterator[judge.Crop]:
        for i in range(n):
            yield labels[i % 4], crop(i), 20 + 15 * (i % 4), i % 2 == 0

    return crops


# The DeepInfra backend --------------------------------------------------------------------


def test_deepinfra_sends_one_crop_as_a_data_url_and_no_credential(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(judge.TOKEN_ENV, "bedrock-" + "t" * 40)  # never sent to DeepInfra
    fake = server(chat("other"))
    image = crop()
    clf = deepinfra(fake)
    assert clf.classify(image) == "not_person"
    [request] = fake.requests
    assert request["path"] == "/v1/openai/chat/completions"
    assert "Authorization" not in request["headers"]
    assert "authorization" not in {k.lower() for k in request["headers"]}
    sent = json.loads(request["body"])
    assert sent["model"] == candidate().model_id
    assert sent["max_tokens"] == judge.REMOTE_MAX_TOKENS and sent["temperature"] == 0
    assert "reasoning_effort" not in sent
    [message] = sent["messages"]
    parts = {part["type"]: part for part in message["content"]}
    assert parts["text"]["text"] == judge.PROMPT
    url = parts["image_url"]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    png = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8)
    decoded = cv2.imdecode(png, cv2.IMREAD_COLOR)
    assert decoded is not None and np.array_equal(decoded, image)
    assert clf.usage == judge.Usage(requests=1, input_tokens=300, output_tokens=2)


def test_deepinfra_asks_reasoning_models_not_to_reason(server: Any) -> None:
    fake = server(chat("person"))
    assert deepinfra(fake, candidate=candidate(reasoning_off=True)).classify(crop()) == "person"
    assert json.loads(fake.requests[0]["body"])["reasoning_effort"] == "none"


@pytest.mark.parametrize(
    "reply",
    [
        chat("I cannot help with that."),
        chat(""),
        chat(None),
        chat("person", finish="content_filter"),
        chat("person or vehicle"),
        chat("x" * 500, finish="length"),
    ],
)
def test_deepinfra_refused_empty_filtered_or_unparseable_replies_are_unsure(
    server: Any, reply: Reply
) -> None:
    fake = server(reply)
    assert deepinfra(fake).classify(crop()) == "unsure"
    assert len(fake.requests) == 1


def test_deepinfra_throttling_is_retried_then_succeeds(server: Any) -> None:
    waits: list[float] = []
    fake = server(error(429), error(429), chat("vehicle"))
    clf = judge.DeepInfraClassifier(
        candidate(), budget=judge.RequestBudget(10), endpoint=fake.url, sleep=waits.append
    )
    assert clf.classify(crop()) == "in_vehicle"
    assert len(fake.requests) == 3 and waits == [2.0, 4.0]
    assert clf.usage.requests == 3


def test_deepinfra_server_errors_are_retried_at_most_three_times(server: Any) -> None:
    fake = server(error(503, "overloaded"))
    with pytest.raises(judge.JudgeError, match="503"):
        deepinfra(fake).classify(crop())
    assert len(fake.requests) == 1 + judge.MAX_RETRIES


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_deepinfra_client_errors_are_not_retried(server: Any, status: int) -> None:
    fake = server(error(status, "bad request"))
    with pytest.raises(judge.JudgeError, match=f"DeepInfra answered HTTP {status}: bad request"):
        deepinfra(fake).classify(crop())
    assert len(fake.requests) == 1


def test_deepinfra_errors_redact_an_echoed_bearer_credential(server: Any) -> None:
    secret = "di-" + "s" * 40
    fake = server(error(401, f"invalid key Bearer {secret}"))
    with pytest.raises(judge.JudgeError) as caught:
        deepinfra(fake).classify(crop())
    assert secret not in str(caught.value) and "Bearer [redacted]" in str(caught.value)


@pytest.mark.parametrize(
    "body",
    [
        {"detail": "Model is not available"},
        {"error": "no such model"},
        {"message": "plain message"},
    ],
)
def test_deepinfra_error_messages_of_every_shape_are_reported(
    server: Any, body: dict[str, str]
) -> None:
    fake = server(Reply(status=404, body=json.dumps(body).encode()))
    with pytest.raises(judge.JudgeError) as caught:
        deepinfra(fake).classify(crop())
    assert next(iter(body.values())) in str(caught.value)


def test_deepinfra_every_request_has_a_timeout(server: Any) -> None:
    fake = server(Reply(delay=1.5, body=chat().body))
    clf = deepinfra(fake, timeout=0.3)
    with pytest.raises(judge.JudgeError, match=r"(?i)time"):
        clf.classify(crop())
    assert len(fake.requests) == 1  # a timeout is not retried


def test_deepinfra_redirects_are_not_followed(server: Any) -> None:
    elsewhere = server(chat())
    fake = server(Reply(status=307, headers={"Location": elsewhere.url + "/v1/x"}))
    with pytest.raises(judge.JudgeError, match="307"):
        deepinfra(fake).classify(crop())
    assert elsewhere.requests == []


@pytest.mark.parametrize(
    "reply",
    [
        Reply(body=b"not json"),
        Reply(body=b"\xff\xfe"),
        Reply(body=b"[" * 100_000 + b"]" * 100_000),
        Reply(body=b'{"choices": [{"message": {"content": "person"}}], "x": NaN}'),
        Reply(body=b'{"n": 1e999999}'),
        Reply(body=b"[]"),
        Reply(body=b'{"choices": []}'),
        Reply(body=b'{"choices": "person"}'),
        Reply(body=b'{"choices": ["person"]}'),
        Reply(body=b'{"choices": [{"message": "person"}]}'),
        Reply(body=b'{"choices": [{"message": {"content": ["person"]}}]}'),
        Reply(body=b" " * (judge.MAX_RESPONSE_BYTES + 1)),
    ],
)
def test_deepinfra_hostile_replies_raise_judge_error(server: Any, reply: Reply) -> None:
    fake = server(reply)
    with pytest.raises(judge.JudgeError):
        deepinfra(fake).classify(crop())


@pytest.mark.parametrize(
    "usage",
    [{}, {"prompt_tokens": -1, "completion_tokens": 2}, {"prompt_tokens": "300"}, None],
)
def test_deepinfra_missing_usage_is_counted_not_invented(server: Any, usage: Any) -> None:
    body = json.loads(chat().body)
    if usage is None:
        del body["usage"]
    else:
        body["usage"] = usage
    fake = server(Reply(body=json.dumps(body).encode()))
    clf = deepinfra(fake)
    assert clf.classify(crop()) == "person"
    assert clf.usage == judge.Usage(requests=1, input_tokens=0, output_tokens=0, missing=1)


def test_deepinfra_refuses_images_that_are_not_gold_or_control_crops(server: Any) -> None:
    fake = server(chat())
    clf = deepinfra(fake)
    with pytest.raises(judge.JudgeError, match="not a gold-set or control crop"):
        clf.classify(np.zeros((120, 60, 3), np.uint8))
    with pytest.raises(judge.JudgeError):
        clf.classify(crop().copy())
    assert fake.requests == []


def test_deepinfra_stops_at_its_request_budget(server: Any) -> None:
    fake = server(chat())
    clf = deepinfra(fake, limit=2)
    clf.classify(crop(1))
    clf.classify(crop(2))
    with pytest.raises(judge.RequestLimitReached):
        clf.classify(crop(3))
    assert len(fake.requests) == 2


def test_a_closed_deepinfra_classifier_refuses_to_answer(server: Any) -> None:
    fake = server(chat())
    clf = deepinfra(fake)
    clf.close()
    with pytest.raises(judge.JudgeError, match="closed"):
        clf.classify(crop())
    assert fake.requests == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://api.example.org",
        "https://api.example.org/v1",
        "https://user@api.example.org",
        "https://api.example.org?x=1",
        "ftp://127.0.0.1",
    ],
)
def test_the_deepinfra_endpoint_must_be_an_https_origin(endpoint: str) -> None:
    with pytest.raises(judge.JudgeError):
        judge.DeepInfraClassifier(candidate(), budget=judge.RequestBudget(1), endpoint=endpoint)


def test_the_default_deepinfra_endpoint_is_the_chat_completions_api() -> None:
    clf = judge.DeepInfraClassifier(candidate(), budget=judge.RequestBudget(1))
    assert clf._url == "https://api.deepinfra.com/v1/openai/chat/completions"


@pytest.mark.parametrize("model_id", ["no-slash", "a/b/c", "a/b c", "/x", "a/" + "b" * 200])
def test_a_malformed_deepinfra_model_id_is_refused(model_id: str) -> None:
    bad = judge.HostedCandidate(
        name="x",
        family="x",
        model_id=model_id,
        region=judge.DEEPINFRA_REGION,
        input_usd_per_mtok=1,
        output_usd_per_mtok=1,
        licence="x",
        card="https://deepinfra.com/x",
    )
    with pytest.raises(judge.JudgeError, match="model ID"):
        judge.DeepInfraClassifier(bad, budget=judge.RequestBudget(1))


def test_the_deepinfra_registry_is_complete_and_apart() -> None:
    assert judge.DEEPINFRA
    ids = [c.model_id for c in judge.DEEPINFRA.values()]
    assert len(ids) == len(set(ids))
    assert not set(judge.DEEPINFRA) & (set(judge.HOSTED) | set(judge.CANDIDATES))
    for name, c in judge.DEEPINFRA.items():
        assert name == c.name and name.startswith("di-")
        assert re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", c.model_id)
        assert c.card == "https://deepinfra.com/" + c.model_id
        assert c.input_usd_per_mtok > 0 and c.output_usd_per_mtok > 0
        assert c.licence.strip() and c.region == judge.DEEPINFRA_REGION
        assert judge.priced(name) is c
    assert judge.priced(next(iter(judge.CANDIDATES))) is None


# The DeepInfra command line ---------------------------------------------------------------


def _argv(*extra: str, models: str | None = None) -> list[str]:
    return ["--bakeoff", "--models", models or candidate().name, *extra]


def test_the_deepinfra_command_line_has_no_path_to_another_image() -> None:
    dests = {a.dest for a in judge.build_deepinfra_parser()._actions}
    assert dests == {"help", "bakeoff", "models", "subset", "limit", "max_requests"}
    subset = next(a for a in judge.build_deepinfra_parser()._actions if a.dest == "subset")
    assert set(subset.choices or ()) == {"all", "screen"}


def test_a_deepinfra_run_refuses_to_start_without_max_requests(
    server: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server(chat())
    for extra in ([], ["--max-requests", "0"], ["--max-requests", "-3"]):
        assert judge.deepinfra_main(_argv(*extra), crops=_crops(), endpoint=fake.url) == 2
    assert fake.requests == []
    assert "--max-requests" in capsys.readouterr().err


def test_a_deepinfra_run_stops_at_max_requests_and_prints_usage(
    server: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server(chat("person"))
    code = judge.deepinfra_main(_argv("--max-requests", "3"), crops=_crops(10), endpoint=fake.url)
    out = capsys.readouterr().out
    assert code == 0 and len(fake.requests) == 3
    assert "request limit" in out
    assert "requests 3, input tokens 900, output tokens 6" in out
    assert f"region {judge.DEEPINFRA_REGION}" in out


def test_a_deepinfra_run_reports_cost_heights_agreement_and_writes_nothing(
    server: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = server(chat("person"))
    monkeypatch.chdir(tmp_path)
    a, b = list(judge.DEEPINFRA)[:2]
    code = judge.deepinfra_main(
        _argv("--max-requests", "20", models=f"{a},{b}"), crops=_crops(4), endpoint=fake.url
    )
    assert code == 0 and len(fake.requests) == 8
    assert list(tmp_path.iterdir()) == []
    out = capsys.readouterr()
    for name in (a, b):
        c = judge.DEEPINFRA[name]
        assert c.model_id in out.out and c.licence in out.out
    assert out.out.count("requests 4, input tokens 1200, output tokens 8") == 2
    assert out.out.count("cost per 300 crops $") >= 2
    assert "by person height" in out.out and f"agreement of {a} and {b}" in out.out
    sent = json.loads(fake.requests[0]["body"])
    url = sent["messages"][0]["content"][0]["image_url"]["url"]
    for text in (out.out, out.err):
        assert url[22:62] not in text
        assert "http" not in text and "messages" not in text


def test_deepinfra_refuses_names_of_other_backends(capsys: pytest.CaptureFixture[str]) -> None:
    for name in (next(iter(judge.HOSTED)), next(iter(judge.CANDIDATES)), "chosen"):
        argv = ["--bakeoff", "--max-requests", "5", "--models", name]
        assert judge.deepinfra_main(argv, crops=_crops()) == 2


def test_the_cheapest_passing_deepinfra_model_is_the_verdict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cheap, dear = sorted(list(judge.DEEPINFRA.values())[:2], key=lambda c: c.input_usd_per_mtok)
    pairs: list[tuple[goldset.Label, judge.Answer]] = []
    pairs += [("person", "person")] * 85
    pairs += [("in_vehicle", "in_vehicle")] * 15
    pairs += [("not_person", "not_person")] * 100

    def run(name: str) -> judge.Run:
        r = judge.Run(name, answers=list(pairs), seconds=10.0)
        r.held_out = [i % 2 == 0 for i in range(len(pairs))]
        r.usage = judge.Usage(requests=len(pairs), input_tokens=300 * len(pairs), output_tokens=0)
        return r

    judge.bakeoff([dear.name, cheap.name], run, print)
    out = capsys.readouterr().out
    assert f"cheapest model that passes: {cheap.name}" in out
    per_300 = judge.cost_usd(cheap, 300 * 300, 0)
    assert f"(${per_300:.4f} per 300 crops)" in out


def test_the_deepinfra_module_runs_from_the_command_line() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "wearreport.tools.judge_deepinfra"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 2 and "--max-requests" in proc.stdout


# Bedrock: the credential the environment injects -------------------------------------------


def test_bedrock_without_the_variable_sends_no_authorization_when_injected(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(judge.TOKEN_ENV, raising=False)
    fake = server(converse("person"))
    clf = judge.BedrockClassifier(
        next(iter(judge.HOSTED.values())),
        budget=judge.RequestBudget(5),
        endpoint=fake.url,
        injected_credential=True,
    )
    assert clf.classify(crop()) == "person"
    assert "authorization" not in {k.lower() for k in fake.requests[0]["headers"]}


def test_bedrock_with_the_variable_sends_it_even_when_injection_is_allowed(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = "fake-" + "k" * 40
    monkeypatch.setenv(judge.TOKEN_ENV, token)
    fake = server(converse("person"))
    clf = judge.BedrockClassifier(
        next(iter(judge.HOSTED.values())),
        budget=judge.RequestBudget(5),
        endpoint=fake.url,
        injected_credential=True,
    )
    clf.classify(crop())
    assert fake.requests[0]["headers"]["Authorization"] == f"Bearer {token}"


def test_a_bedrock_run_without_the_variable_relies_on_the_injected_credential(
    server: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(judge.TOKEN_ENV, raising=False)
    fake = server(converse("person"))
    argv = ["--bakeoff", "--backend", "bedrock", "--max-requests", "4"]
    argv += ["--models", next(iter(judge.HOSTED))]
    assert judge.main(argv, crops=_crops(2), endpoint=fake.url) == 0
    assert len(fake.requests) == 2
    for request in fake.requests:
        assert "authorization" not in {k.lower() for k in request["headers"]}
    assert "requests 2, input tokens 240, output tokens 4" in capsys.readouterr().out
