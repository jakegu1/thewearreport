"""Acceptance tests for T-033 (judge bake-off on hosted vision models served by Amazon
Bedrock). The task contract: do not edit.

Every request here goes to a local fake Bedrock server on the loopback interface: no test
reaches the network, and no remote run is part of `make check`. Every image is synthetic
and marked as a gold-set crop by the test itself, except where a test checks that an
unmarked image is refused. The bearer token is made up at run time.
"""

from __future__ import annotations

import base64
import http.server
import json
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport.tools import goldset, judge

ROOT = Path(__file__).resolve().parents[3]
FETCH_JUDGE = ROOT / "scripts" / "fetch_judge_model.sh"
FETCH_GOLDSET = ROOT / "scripts" / "fetch_goldset.sh"
JUDGE_WORKFLOW = ROOT / ".github" / "workflows" / "judge.yml"
JUDGE_SOURCE = ROOT / "engine" / "wearreport" / "tools" / "judge.py"
TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"  # noqa: S105  (a variable name)
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")


# A fake Bedrock Converse endpoint ------------------------------------------------------


@dataclass
class Reply:
    status: int = 200
    body: bytes = b""
    delay: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)


def converse(text: str | None = "person", stop: str = "end_turn") -> Reply:
    content = [] if text is None else [{"text": text}]
    body: dict[str, Any] = {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop,
        "usage": {"inputTokens": 120, "outputTokens": 2, "totalTokens": 122},
    }
    return Reply(body=json.dumps(body).encode())


def no_usage(text: str = "person") -> Reply:
    body = {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
    }
    return Reply(body=json.dumps(body).encode())


def error(status: int, kind: str, message: str = "") -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"message": message or kind}).encode(),
        headers={"x-amzn-ErrorType": f"{kind}:http://internal.amazon.com/coral/"},
    )


class FakeBedrock:
    """Answers each request with the next scripted Reply (the last one repeats) and
    records what it received."""

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
def token(monkeypatch: pytest.MonkeyPatch) -> str:
    value = "fake-bearer-" + secrets.token_hex(24)
    monkeypatch.setenv(TOKEN_ENV, value)
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    return value


@pytest.fixture
def bedrock() -> Iterator[Any]:
    servers: list[FakeBedrock] = []

    def start(*replies: Reply) -> FakeBedrock:
        server = FakeBedrock(list(replies))
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def hosted(index: int = 0) -> judge.HostedCandidate:
    return list(judge.HOSTED.values())[index]


def crop(seed: int = 0, height: int = 120, width: int = 60) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    image: npt.NDArray[np.uint8] = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return judge.mark_licensed(image)


def classifier(server: FakeBedrock, limit: int = 100, **kwargs: Any) -> judge.BedrockClassifier:
    return judge.BedrockClassifier(
        hosted(),
        budget=judge.RequestBudget(limit),
        endpoint=server.url,
        sleep=lambda seconds: None,
        **kwargs,
    )


# AC1: the backend -----------------------------------------------------------------------


def test_ac1_bedrock_classifier_sends_one_crop_with_the_fixed_prompt(
    bedrock: Any, token: str
) -> None:
    server = bedrock(converse("Person"))
    clf = classifier(server)
    assert isinstance(clf, judge.Classifier)
    assert clf.classify(crop()) == "person"
    clf.close()
    (request,) = server.requests
    candidate = hosted()
    assert request["path"] == f"/model/{candidate.model_id}/converse"
    assert request["headers"]["Authorization"] == f"Bearer {token}"
    body = json.loads(request["body"])
    (message,) = body["messages"]
    assert message["role"] == "user"
    images = [c["image"] for c in message["content"] if "image" in c]
    texts = [c["text"] for c in message["content"] if "text" in c]
    assert len(images) == 1 and texts == [judge.PROMPT]
    assert images[0]["format"] in ("png", "jpeg")
    raw = base64.b64decode(images[0]["source"]["bytes"])
    assert raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff")
    config = body["inferenceConfig"]
    assert config["temperature"] == 0
    assert 1 <= config["maxTokens"] <= 16


@pytest.mark.parametrize(
    "reply",
    [
        converse("banana"),
        converse("I'm sorry, but I can't help with identifying people in images."),
        converse(""),
        converse(None),
        converse("person", stop="content_filtered"),
        converse("person", stop="guardrail_intervened"),
    ],
    ids=["unparseable", "refusal", "empty-text", "no-content", "content-filter", "guardrail"],
)
def test_ac1_unparseable_refused_empty_or_filtered_replies_are_unsure(
    bedrock: Any, token: str, reply: Reply
) -> None:
    server = bedrock(reply)
    clf = classifier(server)
    assert clf.classify(crop()) == "unsure"
    assert len(server.requests) == 1


def test_ac1_the_token_comes_from_the_environment_only(
    bedrock: Any, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = bedrock(converse())
    monkeypatch.delenv(TOKEN_ENV)
    with pytest.raises(judge.JudgeError, match=TOKEN_ENV):
        classifier(server).classify(crop())
    assert server.requests == []
    source = JUDGE_SOURCE.read_text()
    assert TOKEN_ENV in source
    assert "AWS_SECRET_ACCESS_KEY" not in source and "AWS_ACCESS_KEY_ID" not in source


def test_ac1_an_http_error_does_not_leak_the_token(
    bedrock: Any, token: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # A hostile error body that echoes the request's Authorization header.
    server = bedrock(error(403, "AccessDeniedException", f"denied for Bearer {token}"))
    clf = classifier(server)
    with pytest.raises(judge.JudgeError) as caught:
        clf.classify(crop())
    exc = caught.value
    text = str(exc) + repr(exc) + "".join(traceback.format_exception(exc))
    assert token not in text and token[-16:] not in text
    assert "403" in str(exc) and "AccessDeniedException" in str(exc)
    out = capsys.readouterr()
    assert token not in out.out and token not in out.err
    assert len(server.requests) == 1  # a 4xx is not retried


def test_ac1_standard_library_only() -> None:
    source = JUDGE_SOURCE.read_text()
    imports = set(re.findall(r"^\s*(?:import|from)\s+([\w.]+)", source, re.MULTILINE))
    for forbidden in ("boto3", "botocore", "requests", "httpx", "aiohttp", "urllib3"):
        assert not any(i.split(".")[0] == forbidden for i in imports), forbidden
    assert "urllib.request" in imports or "urllib" in imports
    lock = (ROOT / "uv.lock").read_text()
    for forbidden in ("boto3", "botocore", "requests", "httpx"):
        assert not re.search(rf'^name = "{forbidden}"$', lock, re.MULTILINE), forbidden


def test_ac1_throttling_is_retried_then_succeeds(bedrock: Any, token: str) -> None:
    server = bedrock(
        error(429, "ThrottlingException"), error(429, "ThrottlingException"), converse("other")
    )
    waits: list[float] = []
    clf = judge.BedrockClassifier(
        hosted(), budget=judge.RequestBudget(10), endpoint=server.url, sleep=waits.append
    )
    assert clf.classify(crop()) == "not_person"
    assert len(server.requests) == 3
    assert len(waits) == 2 and waits[1] > waits[0] > 0  # backoff
    assert clf.usage.requests == 3


def test_ac1_server_errors_are_retried_at_most_three_times(bedrock: Any, token: str) -> None:
    server = bedrock(error(503, "ServiceUnavailableException"))
    clf = classifier(server)
    with pytest.raises(judge.JudgeError, match="503"):
        clf.classify(crop())
    assert len(server.requests) == 1 + 3
    assert judge.MAX_RETRIES == 3


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_ac1_client_errors_are_not_retried(bedrock: Any, token: str, status: int) -> None:
    server = bedrock(error(status, "ValidationException"))
    with pytest.raises(judge.JudgeError, match=str(status)):
        classifier(server).classify(crop())
    assert len(server.requests) == 1


def test_ac1_every_request_has_a_timeout(bedrock: Any, token: str) -> None:
    server = bedrock(Reply(body=converse().body, delay=3.0))
    clf = classifier(server, timeout=0.3)
    start = time.monotonic()
    with pytest.raises(judge.JudgeError, match=r"(?i)time"):
        clf.classify(crop())
    assert time.monotonic() - start < 2.5
    assert 0 < judge.REQUEST_TIMEOUT <= 120


def test_ac1_redirects_are_not_followed(bedrock: Any, token: str) -> None:
    other = bedrock(converse())
    server = bedrock(Reply(status=307, headers={"Location": other.url + "/steal"}))
    with pytest.raises(judge.JudgeError):
        classifier(server).classify(crop())
    assert other.requests == []  # the token never goes anywhere else


# AC7: hostile responses -------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        Reply(body=b"{not json"),
        Reply(body=b"\xff\xfe\x00"),
        Reply(body=b"[" * 100_000 + b"]" * 100_000),
        Reply(body=b'{"output": ' * 50_000 + b"1" + b"}" * 50_000),
        Reply(body=b'{"a": "' + b"x" * (judge.MAX_RESPONSE_BYTES + 10) + b'"}'),
        Reply(body=b'{"output": 1e999999}'),
        Reply(body=b'{"output": {"message": {"content": "person"}}, "stopReason": 7}'),
        Reply(body=b"[]"),
    ],
    ids=["malformed", "not-utf8", "deep-list", "deep-object", "huge", "overflow", "types", "list"],
)
def test_ac7_hostile_responses_raise_judge_error(bedrock: Any, token: str, reply: Reply) -> None:
    server = bedrock(reply)
    with pytest.raises(judge.JudgeError):
        classifier(server).classify(crop())


def test_ac7_a_missing_usage_field_is_counted_not_invented(bedrock: Any, token: str) -> None:
    server = bedrock(no_usage("person"), converse("other"))
    clf = classifier(server)
    assert clf.classify(crop(1)) == "person"
    assert clf.classify(crop(2)) == "not_person"
    assert clf.usage.requests == 2
    assert clf.usage.missing == 1
    assert clf.usage.input_tokens == 120 and clf.usage.output_tokens == 2


# AC2: safety limits -----------------------------------------------------------------------


def _bakeoff_argv(*extra: str, models: str | None = None) -> list[str]:
    return [
        "--bakeoff",
        "--backend",
        "bedrock",
        "--models",
        models or hosted().name,
        *extra,
    ]


def _crops(n: int = 4) -> judge.Crops:
    labels: list[goldset.Label] = ["person", "in_vehicle", "not_person", "person"]

    def crops() -> Iterator[judge.Crop]:
        for i in range(n):
            yield labels[i % 4], crop(i), 20 + 15 * (i % 4), i % 2 == 0

    return crops


def test_ac2_a_remote_run_refuses_to_start_without_max_requests(
    bedrock: Any, token: str, capsys: pytest.CaptureFixture[str]
) -> None:
    server = bedrock(converse())
    for extra in ([], ["--max-requests", "0"], ["--max-requests", "-3"]):
        code = judge.main(_bakeoff_argv(*extra), crops=_crops(), endpoint=server.url)
        assert code == 2, extra
    assert server.requests == []
    assert "--max-requests" in capsys.readouterr().err


def test_ac2_a_run_stops_at_max_requests_and_prints_usage(
    bedrock: Any, token: str, capsys: pytest.CaptureFixture[str]
) -> None:
    server = bedrock(converse("person"))
    code = judge.main(_bakeoff_argv("--max-requests", "3"), crops=_crops(10), endpoint=server.url)
    out = capsys.readouterr().out
    assert code == 0
    assert len(server.requests) == 3
    assert "request limit" in out
    assert re.search(r"requests 3\b", out)
    assert re.search(r"input tokens 360\b", out) and re.search(r"output tokens 6\b", out)


def test_ac2_the_backend_refuses_images_that_are_not_gold_or_control_crops(
    bedrock: Any, token: str, capsys: pytest.CaptureFixture[str]
) -> None:
    server = bedrock(converse())
    clf = classifier(server)
    unmarked = np.zeros((120, 60, 3), np.uint8)
    with pytest.raises(judge.JudgeError):
        clf.classify(unmarked)
    marked = crop(3)
    copy = marked.copy()  # a copy is not a gold crop
    with pytest.raises(judge.JudgeError):
        clf.classify(copy)
    with pytest.raises(ValueError):
        marked[0, 0, 0] = 1  # a marked crop cannot be changed into another image

    def other_images() -> Iterator[judge.Crop]:
        yield "person", np.zeros((120, 60, 3), np.uint8)

    code = judge.main(_bakeoff_argv("--max-requests", "5"), crops=other_images, endpoint=server.url)
    assert code == 0
    assert server.requests == []
    assert "not a gold-set or control crop" in capsys.readouterr().out


def test_ac2_no_command_line_path_sends_another_image() -> None:
    parser = judge.build_parser()
    dests = {a.dest for a in parser._actions}
    assert dests <= {
        "help",
        "bakeoff",
        "backend",
        "models",
        "subset",
        "limit",
        "threads",
        "max_requests",
    }, dests
    subset = next(a for a in parser._actions if a.dest == "subset")
    assert set(subset.choices or ()) == {"all", "screen"}
    backend = next(a for a in parser._actions if a.dest == "backend")
    assert set(backend.choices or ()) == {"local", "bedrock"}


def test_ac2_a_remote_run_writes_and_logs_nothing(
    bedrock: Any,
    token: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = bedrock(converse("person"))
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    code = judge.main(_bakeoff_argv("--max-requests", "10"), crops=_crops(4), endpoint=server.url)
    assert code == 0
    assert sorted(tmp_path.rglob("*")) == before
    out = capsys.readouterr()
    sent = json.loads(server.requests[0]["body"])
    image_b64 = next(c for c in sent["messages"][0]["content"] if "image" in c)["image"]
    for text in (out.out, out.err):
        assert token not in text
        assert image_b64["source"]["bytes"][:40] not in text
        assert "http" not in text and "messages" not in text


# AC3: the report --------------------------------------------------------------------------


def test_ac3_hosted_registry_entries_are_complete() -> None:
    assert judge.HOSTED
    for c in judge.HOSTED.values():
        assert c.name not in judge.CANDIDATES
        assert re.fullmatch(r"[a-z0-9.-]+", c.name)
        assert re.fullmatch(r"[a-z]{2}(-[a-z]+)+-\d", c.region), c.region
        assert re.fullmatch(r"(eu\.|us\.|global\.)?[a-z0-9-]+\.[\w.:-]+", c.model_id), c.model_id
        assert not re.search(r"anthropic|openai|claude|gpt", c.model_id, re.IGNORECASE)
        assert c.input_usd_per_mtok > 0 and c.output_usd_per_mtok > 0
        assert c.licence.strip()
        assert c.card.startswith("https://docs.aws.amazon.com/bedrock/")


def test_ac3_cost_comes_from_the_published_prices() -> None:
    c = hosted()
    expected = (2_000_000 * c.input_usd_per_mtok + 500_000 * c.output_usd_per_mtok) / 1e6
    assert judge.cost_usd(c, 2_000_000, 500_000) == pytest.approx(expected)
    assert judge.cost_usd(c, 0, 0) == 0


def test_ac3_report_has_region_usage_cost_heights_and_agreement(
    bedrock: Any, token: str, capsys: pytest.CaptureFixture[str]
) -> None:
    if len(judge.HOSTED) < 2:
        pytest.skip("one hosted candidate")
    server = bedrock(converse("person"))
    models = f"{hosted(0).name},{hosted(1).name}"
    code = judge.main(
        _bakeoff_argv("--max-requests", "20", models=models),
        crops=_crops(4),
        endpoint=server.url,
    )
    out = capsys.readouterr().out
    assert code == 0 and len(server.requests) == 8
    for c in (hosted(0), hosted(1)):
        assert c.name in out and c.region in out and c.model_id in out
    assert out.count("requests 4, input tokens 480, output tokens 8") == 2
    assert out.count("cost per 300 crops $") >= 2
    assert "by person height" in out
    assert f"agreement of {hosted(0).name} and {hosted(1).name}" in out
    assert re.search(r"requests made 8\b", out)


# AC4: held-out check ----------------------------------------------------------------------


def test_ac4_held_out_items_are_the_gold_set_outside_the_screen_subset() -> None:
    manifest = goldset.load_manifest()
    held = judge.held_out(manifest)
    screen = {i.id for i in goldset.screening_subset(manifest)}
    assert len(screen) == goldset.SCREEN_SIZE
    assert {i.id for i in held}.isdisjoint(screen)
    assert len(held) == len(manifest.items) - len(screen)


def _run(name: str, answers: list[tuple[Any, Any]], held: list[bool]) -> judge.Run:
    run = judge.Run(name)
    run.answers = answers
    run.held_out = held
    run.heights = [40] * len(answers)
    run.seconds = 0.5 * len(answers)
    return run


def test_ac4_a_model_passes_only_if_the_full_set_and_the_held_out_items_pass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    screen: list[tuple[Any, Any]] = [("person", "person")] * 60 + [
        ("in_vehicle", "in_vehicle")
    ] * 10
    screen += [("not_person", "not_person")] * 30
    held: list[tuple[Any, Any]] = [("person", "person")] * 228 + [("person", "not_person")] * 12
    held += [("in_vehicle", "in_vehicle")] * 40
    held += [("not_person", "not_person")] * 108 + [("not_person", "person")] * 12
    answers = screen + held
    flags = [False] * len(screen) + [True] * len(held)
    assert judge.passes(judge.score(answers), 0.5)  # the full set alone would pass
    assert not judge.passes(judge.score(held), 0.5)
    name = hosted().name
    judge.bakeoff([name], lambda n: _run(n, answers, flags), print)
    out = capsys.readouterr().out
    assert "held out" in out
    assert "no single model passes" in out

    perfect = [(t, t) for t, _ in answers]
    judge.bakeoff([name], lambda n: _run(n, perfect, flags), print)
    out = capsys.readouterr().out
    assert f"cheapest model that passes: {name}" in out


def test_ac4_without_held_out_items_there_is_no_pass(capsys: pytest.CaptureFixture[str]) -> None:
    perfect = [("person", "person")] * 60 + [("not_person", "not_person")] * 40
    name = hosted().name
    judge.bakeoff([name], lambda n: _run(n, perfect, [False] * 100), print)
    assert "no single model passes" in capsys.readouterr().out


# AC5: the quality bar, unchanged ----------------------------------------------------------


def test_ac5_the_quality_bar_is_unchanged() -> None:
    assert judge.MIN_ACCURACY == 0.95
    assert judge.MAX_UNSURE == 0.10
    assert judge.MAX_PRECISION_ERROR == 3.0
    assert judge.MIXES == (0.80, 0.90, 0.95)
    assert judge.BAR_CROPS == 300 and judge.BAR_SECONDS == 3600
    assert goldset.CROP_MARGIN == 0.5 and goldset.SCREEN_SIZE == 100


# AC6: the choice ---------------------------------------------------------------------------


def test_ac6_a_choice_is_a_pinned_hosted_model_or_none() -> None:
    if judge.CHOSEN is None:
        return
    assert judge.CHOSEN in judge.HOSTED
    chosen = judge.HOSTED[judge.CHOSEN]
    assert chosen.model_id and chosen.region
    assert chosen.model_id in JUDGE_SOURCE.read_text()


def test_ac6_the_judge_workflow_gains_no_key_or_remote_call() -> None:
    wf = JUDGE_WORKFLOW.read_text()
    assert "secrets." not in wf
    assert not re.search(r"bedrock|AWS_", wf, re.IGNORECASE)


def _sh(script: Path, *args: str, env: dict[str, str] | None = None) -> Any:
    sh = shutil.which("sh")
    assert sh is not None
    return subprocess.run(
        [sh, str(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )


# Also fix: the T-029 review minors --------------------------------------------------------


@pytest.mark.parametrize("model", [" ", "", "  \t "])
def test_fix_fetch_judge_model_with_no_model_exits_non_zero(tmp_path: Path, model: str) -> None:
    dest = tmp_path / "judge"
    proc = _sh(FETCH_JUDGE, "--model", model, "--dest", str(dest))
    assert proc.returncode != 0
    assert not dest.exists() or list(dest.iterdir()) == []


def test_fix_the_closed_judge_test_fails_or_skips_like_the_other_weight_tests() -> None:
    source = (ROOT / "engine" / "tests" / "unit" / "test_judge.py").read_text()
    body = source.split("def test_a_closed_judge_refuses_to_answer", 1)[1]
    body = body.split("\ndef ", 1)[0]
    assert "WEARREPORT_REQUIRE_JUDGE" in body or "probe" in body.split(")", 1)[0]


@pytest.mark.parametrize("char", ["\x00", "\x07", "\x1b", "\x7f", "\x85", "\u202e"])
def test_fix_manifest_urls_with_control_characters_are_refused(tmp_path: Path, char: str) -> None:
    data = json.loads(goldset.MANIFEST_PATH.read_text())
    data["sources"][0]["url"] = "https://example.org/a" + char + "b.jpg"
    raw = json.dumps(data).encode()
    with pytest.raises(goldset.GoldsetError):
        goldset.parse_manifest(raw)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(raw)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "curl.log"
    curl = bin_dir / "curl"
    curl.write_text(f'#!/bin/sh\necho "$*" >> "{log}"\nexit 1\n')
    curl.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}
    proc = _sh(FETCH_GOLDSET, "--manifest", str(manifest), "--dest", str(tmp_path / "g"), env=env)
    assert proc.returncode != 0
    assert not log.exists()


def test_the_fake_server_is_on_the_loopback_interface(bedrock: Any) -> None:
    server = bedrock(converse())
    host = server.url.split("//", 1)[1].split(":", 1)[0]
    assert socket.inet_aton(host) == socket.inet_aton("127.0.0.1")
