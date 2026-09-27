"""Acceptance tests for T-035 (judge context experiment: crop margin and render size).
The task contract: do not edit.

Every request here goes to a local fake DeepInfra server on the loopback interface: no
test reaches the network, and no remote run is part of `make check`. Every image is
synthetic and marked as a gold-set crop by the test itself.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import re
import socket
import threading
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport.tools import goldset, judge, judge_context, spotcheck

ROOT = Path(__file__).resolve().parents[3]
CONTEXT_SOURCE = ROOT / "engine" / "wearreport" / "tools" / "judge_context.py"
CONTROLS = ROOT / "fixtures" / "goldset" / "controls"
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")

# Measured on the code before this task: the defaults must reproduce it byte for byte.
DEFAULTS_DIGEST = "d64e6ad4e751e13bbc9ba2fbcbee0de004eb29e796582f408f15c2052cfcc43c"
CONTROLS_DIGEST = "cc16db94cc1b4709d1ecf0d6a3371f6b00338de6d849e699dd5bd71dcc95952a"
EARLIER_CONTRACTS = {
    "test_t_029.py": "c431e949526912a406dc6f947c1906f1c73f0fe933dcb0f579ce915851761e9d",
    "test_t_033.py": "abce7db97f2c409341681b0a8960f0125f4b926b1a0bbc2d38cf45cd36420d99",
}
SPEC_VARIANTS = {
    "m0.5": (0.5, 240),
    "m1.0": (1.0, 240),
    "m2.0": (2.0, 240),
    "m1.0-r480": (1.0, 480),
}


# A fake DeepInfra chat completions endpoint ----------------------------------------------


def chat_body(text: str = "person") -> bytes:
    body = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 300, "completion_tokens": 2, "total_tokens": 302},
    }
    return json.dumps(body).encode()


class FakeDeepInfra:
    """Answers every POST with (status, body) and records what it received."""

    def __init__(self, status: int = 200, body: bytes | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        fake = self
        reply = body if body is not None else chat_body()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                fake.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": self.rfile.read(length),
                    }
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., FakeDeepInfra]]:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    started: list[FakeDeepInfra] = []

    def start(status: int = 200, body: bytes | None = None) -> FakeDeepInfra:
        fake = FakeDeepInfra(status, body)
        started.append(fake)
        return fake

    yield start
    for fake in started:
        fake.close()


def crop(seed: int = 0) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    image: npt.NDArray[np.uint8] = rng.integers(0, 256, (120, 60, 3), dtype=np.uint8)
    return judge.mark_licensed(image)


def synthetic(seen: list[str] | None = None) -> Callable[[judge_context.Variant], judge.Crops]:
    """Four synthetic crops per variant, recording which variants were asked for."""
    labels: list[goldset.Label] = ["person", "in_vehicle", "not_person", "person"]

    def for_variant(variant: judge_context.Variant) -> judge.Crops:
        if seen is not None:
            seen.append(variant.name)

        def crops() -> Iterator[judge.Crop]:
            for i in range(4):
                yield labels[i], crop(i), 20 + 15 * i, i % 2 == 0

        return crops

    return for_variant


def model() -> str:
    return "di-qwen3-vl-235b"


def argv(*extra: str, variants: str = "m0.5,m1.0", models: str | None = None) -> list[str]:
    return ["--variants", variants, "--models", models or model(), *extra]


# AC1: a variant parameter, not an edit ---------------------------------------------------


def _defaults_digest(**variant: Any) -> str:
    h = hashlib.sha256()
    margin = variant.get("margin")
    target = variant.get("target_height")
    for seed in range(12):
        rng = np.random.default_rng(1000 + seed)
        height, width = int(rng.integers(40, 400)), int(rng.integers(40, 500))
        photo: npt.NDArray[np.uint8] = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        x1, y1 = float(rng.uniform(0, width - 20)), float(rng.uniform(0, height - 20))
        box = (x1, y1, float(rng.uniform(x1 + 10, width)), float(rng.uniform(y1 + 10, height)))
        m: dict[str, Any] = {} if margin is None else {"margin": margin}
        t: dict[str, Any] = {} if target is None else {"target_height": target}
        h.update(repr(goldset.crop_bounds(box, width, height, **m)).encode())
        d = goldset.degrade(photo, box, 15 + seed * 5, 35 + seed * 2, **m)
        h.update(d.small.tobytes())
        h.update(d.jpeg)
        h.update(d.frame.tobytes())
        h.update(repr(d.box).encode())
        r = goldset.render_crop(d.frame, d.box, seed + 1, **m, **t)
        h.update(repr(r.shape).encode())
        h.update(r.tobytes())
    return h.hexdigest()


def test_ac1_defaults_are_byte_identical_to_the_code_before_this_task() -> None:
    assert _defaults_digest() == DEFAULTS_DIGEST
    explicit = _defaults_digest(
        margin=goldset.CROP_MARGIN, target_height=goldset.CROP_TARGET_HEIGHT
    )
    assert explicit == DEFAULTS_DIGEST


def test_ac1_the_constants_and_the_spotcheck_tool_are_unchanged() -> None:
    assert goldset.CROP_MARGIN == spotcheck.CROP_MARGIN == 0.5
    assert goldset.CROP_TARGET_HEIGHT == 240
    assert judge.CHOSEN is None
    assert judge.MIN_ACCURACY == 0.95 and judge.MAX_UNSURE == 0.10
    assert judge.MAX_PRECISION_ERROR == 3.0 and judge.MIXES == (0.80, 0.90, 0.95)


def test_ac1_the_committed_control_crops_are_unchanged() -> None:
    files = sorted(CONTROLS.glob("*.jpg"))
    assert hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest() == CONTROLS_DIGEST


def test_ac1_control_crops_with_explicit_defaults_match_when_sources_are_present() -> None:
    manifest = goldset.load_manifest()
    items = [item for item in manifest.items if item.control is not None][:5]
    if not all((goldset.SOURCE_DIR / item.source).exists() for item in items):
        pytest.skip("the gold-set source images are not downloaded")
    for item in items:
        default = goldset.degrade_item(manifest, item)
        explicit = goldset.degrade_item(manifest, item, margin=goldset.CROP_MARGIN)
        assert default.jpeg == explicit.jpeg, item.id


def test_ac1_earlier_acceptance_tests_are_unchanged() -> None:
    for name, digest in EARLIER_CONTRACTS.items():
        data = (ROOT / "engine" / "tests" / "acceptance" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


# AC5: a variant changes the crop as specified --------------------------------------------


def test_ac5_the_variants_are_the_ones_specified() -> None:
    got = {name: (v.margin, v.target_height) for name, v in judge_context.VARIANTS.items()}
    assert got == SPEC_VARIANTS
    assert judge_context.BASELINE == "m0.5"
    base = judge_context.VARIANTS["m0.5"]
    assert (base.margin, base.target_height) == (goldset.CROP_MARGIN, goldset.CROP_TARGET_HEIGHT)


@pytest.mark.parametrize(
    ("margin", "expected"),
    [(0.5, (90, 80, 130, 160)), (1.0, (80, 60, 140, 180)), (2.0, (60, 20, 160, 220))],
)
def test_ac5_the_margin_grows_the_box_on_every_side(
    margin: float, expected: tuple[int, int, int, int]
) -> None:
    assert goldset.crop_bounds((100.0, 100.0, 120.0, 140.0), 1000, 1000, margin=margin) == expected


def test_ac5_a_wider_margin_is_clipped_to_the_photo() -> None:
    box = (5.0, 10.0, 25.0, 50.0)  # 20 x 40, near the top-left corner of a 60 x 90 photo
    assert goldset.crop_bounds(box, 60, 90, margin=2.0) == (0, 0, 60, 90)
    assert goldset.crop_bounds(box, 60, 90, margin=1.0) == (0, 0, 45, 90)
    for margin in (0.5, 1.0, 2.0):
        left, top, right, bottom = goldset.crop_bounds(box, 60, 90, margin=margin)
        assert 0 <= left < right <= 60 and 0 <= top < bottom <= 90


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf"), 10.5])
def test_ac5_a_nonsensical_margin_is_refused(margin: float) -> None:
    with pytest.raises(ValueError):
        goldset.crop_bounds((100.0, 100.0, 120.0, 140.0), 1000, 1000, margin=margin)


def test_ac5_a_wider_margin_never_gives_more_pixels_on_the_person() -> None:
    rng = np.random.default_rng(7)
    photo: npt.NDArray[np.uint8] = rng.integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
    box = (700.0, 400.0, 760.0, 560.0)  # 60 x 160: every margin fits inside the photo
    boxes = {}
    for margin in (0.5, 1.0, 2.0):
        d = goldset.degrade(photo, box, 40, 50, margin=margin)
        x1, y1, x2, y2 = d.box
        assert abs((y2 - y1) - 40) <= 1.0 and abs((x2 - x1) - 15) <= 1.0, margin
        height, width = d.frame.shape[:2]
        assert abs(height - 40 * (1 + 2 * margin)) <= 2, margin
        assert abs(width - 15 * (1 + 2 * margin)) <= 2, margin
        boxes[margin] = (x2 - x1, y2 - y1)
    assert all(abs(w - 15) <= 1 and abs(h - 40) <= 1 for w, h in boxes.values())


def test_ac5_the_render_target_changes_only_the_enlargement() -> None:
    rng = np.random.default_rng(3)
    frame: npt.NDArray[np.uint8] = rng.integers(0, 256, (60, 30, 3), dtype=np.uint8)
    box = (10.0, 20.0, 20.0, 40.0)  # margin 1.0 covers the whole frame
    small = goldset.render_crop(frame, box, 1, margin=1.0, target_height=240)
    large = goldset.render_crop(frame, box, 1, margin=1.0, target_height=480)
    assert small.shape == (240, 120, 3)  # 240 // 60 = 4
    assert large.shape == (480, 240, 3)  # 480 // 60 = 8
    # the enlargement is capped as before
    tiny = np.ascontiguousarray(frame[:20, :10])
    capped = goldset.render_crop(tiny, (3.0, 5.0, 7.0, 15.0), 1, margin=1.0, target_height=480)
    assert capped.shape[0] == 20 * goldset.MAX_CROP_SCALE


def test_ac5_a_render_target_out_of_range_is_refused() -> None:
    frame = np.zeros((60, 30, 3), np.uint8)
    for target in (0, -240, 100_000):
        with pytest.raises(ValueError):
            goldset.render_crop(frame, (10.0, 20.0, 20.0, 40.0), 1, target_height=target)


# AC2 and AC5: the experiment command -----------------------------------------------------


def test_ac2_the_parser_has_no_path_url_or_image_input() -> None:
    parser = judge_context.build_parser()
    dests = {a.dest for a in parser._actions}
    assert dests == {"help", "variants", "models", "subset", "max_requests"}, dests
    subset = next(a for a in parser._actions if a.dest == "subset")
    assert set(subset.choices or ()) == {"all", "screen"}
    assert subset.default == "screen"


def test_ac2_max_requests_is_required(server: Any, capsys: pytest.CaptureFixture[str]) -> None:
    fake = server()
    with pytest.raises(SystemExit) as caught:
        judge_context.main(argv(), endpoint=fake.url, crops=synthetic())
    assert caught.value.code == 2
    assert "--max-requests" in capsys.readouterr().err
    for n in ("0", "-3"):
        assert (
            judge_context.main(argv("--max-requests", n), endpoint=fake.url, crops=synthetic()) == 2
        )
    assert fake.requests == []


@pytest.mark.parametrize(
    "variants",
    ["m0.7", "m1.0,m3.0", "../m1.0", "/var/crop.png", "m1.0,fixtures/goldset", "https://x/y", ""],
)
def test_ac5_unknown_or_path_like_variants_are_refused(
    server: Any, capsys: pytest.CaptureFixture[str], variants: str
) -> None:
    fake = server()
    seen: list[str] = []
    code = judge_context.main(
        argv("--max-requests", "50", variants=variants), endpoint=fake.url, crops=synthetic(seen)
    )
    assert code == 2
    assert fake.requests == [] and seen == []
    assert "variant" in capsys.readouterr().err


@pytest.mark.parametrize(
    "models",
    ["nova-2-lite", "qwen3.5-4b", "chosen", "../weights.gguf", "/etc/passwd", "Qwen/Qwen3-VL"],
)
def test_ac5_unknown_or_path_like_models_are_refused(server: Any, models: str) -> None:
    fake = server()
    code = judge_context.main(
        argv("--max-requests", "50", models=models), endpoint=fake.url, crops=synthetic()
    )
    assert code == 2 and fake.requests == []


def test_ac2_chosen_is_never_read_or_set(server: Any, capsys: pytest.CaptureFixture[str]) -> None:
    source = CONTEXT_SOURCE.read_text()
    assert "CHOSEN" not in source
    fake = server()
    code = judge_context.main(argv("--max-requests", "20"), endpoint=fake.url, crops=synthetic())
    assert code == 0
    assert judge.CHOSEN is None
    out = capsys.readouterr().out
    assert "diagnostic only: no model is chosen" in out
    assert "cheapest model that passes" not in out and "most accurate model" not in out


def test_ac2_it_reuses_the_deepinfra_classifier_budget_and_guard(
    server: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server()
    code = judge_context.main(argv("--max-requests", "6"), endpoint=fake.url, crops=synthetic())
    out = capsys.readouterr().out
    assert code == 0
    assert len(fake.requests) == 6  # the budget is shared by the whole run
    assert "request limit of 6 reached" in out
    request = fake.requests[0]
    assert request["path"] == judge.DEEPINFRA_PATH
    assert "Authorization" not in request["headers"]
    sent = json.loads(request["body"])
    assert sent["model"] == judge.DEEPINFRA[model()].model_id
    assert sent["messages"][0]["content"][1]["text"] == judge.PROMPT

    def unmarked(variant: judge_context.Variant) -> judge.Crops:
        def crops() -> Iterator[judge.Crop]:
            yield "person", np.zeros((120, 60, 3), np.uint8)

        return crops

    other = server()
    judge_context.main(argv("--max-requests", "6"), endpoint=other.url, crops=unmarked)
    assert other.requests == []
    assert "not a gold-set or control crop" in capsys.readouterr().out


# AC3: the report -------------------------------------------------------------------------


def test_ac3_report_per_variant_and_model_with_the_change_against_the_baseline(
    server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server()
    monkeypatch.chdir(tmp_path)
    seen: list[str] = []
    models = "di-qwen3-vl-235b,di-gemma-4-31b"
    code = judge_context.main(
        argv("--max-requests", "100", variants="m0.5,m1.0,m2.0,m1.0-r480", models=models),
        endpoint=fake.url,
        crops=synthetic(seen),
    )
    out = capsys.readouterr().out
    assert code == 0 and len(fake.requests) == 32
    assert set(seen) == set(SPEC_VARIANTS)
    assert list(tmp_path.iterdir()) == []
    for variant in SPEC_VARIANTS:
        for name in models.split(","):
            assert f"{variant} / {name}" in out, (variant, name)
    assert out.count("requests 4, input tokens 1200, output tokens 8") == 8
    assert out.count("cost $") >= 8
    assert out.count("accuracy on confident answers") >= 8
    assert out.count("precision error (points) at true precision 80%") >= 8
    assert out.count("by person height") == 8
    assert out.count("confusion (rows: truth; columns: answer)") >= 8
    for variant in ("m1.0", "m2.0", "m1.0-r480"):
        for name in models.split(","):
            assert f"{variant} / {name} against m0.5" in out, (variant, name)
    assert re.search(r"requests made 32\b", out)
    assert "http" not in out and "messages" not in out


def test_ac3_without_the_baseline_the_change_is_not_invented(
    server: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server()
    code = judge_context.main(
        argv("--max-requests", "20", variants="m1.0"), endpoint=fake.url, crops=synthetic()
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "no m0.5 run" in out


def test_ac3_the_full_set_run_reports_the_source_split_held_out_items_and_the_bar(
    server: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server()
    code = judge_context.main(
        argv("--max-requests", "20", "--subset", "all", variants="m1.0"),
        endpoint=fake.url,
        crops=synthetic(),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "held out (by source photo)" in out
    assert re.search(r"full set and held-out items: (PASS|FAIL)", out)
    assert "full set and held-out items: PASS" not in out  # "person" to everything fails


def test_ac3_a_screen_run_names_what_may_go_to_the_full_set(
    server: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server()
    judge_context.main(argv("--max-requests", "20"), endpoint=fake.url, crops=synthetic())
    out = capsys.readouterr().out
    assert "screen only: no pass decision" in out
    assert "eligible for a full-set run" in out


def test_ac3_the_screen_eligibility_rule(capsys: pytest.CaptureFixture[str]) -> None:
    def scores(correct: int, wrong: int, unsure: int) -> judge.Scores:
        pairs: list[tuple[goldset.Label, judge.Answer]] = []
        pairs += [("person", "person")] * correct
        pairs += [("not_person", "person")] * wrong
        pairs += [("person", "unsure")] * unsure
        return judge.score(pairs)

    assert judge_context.eligible(scores(90, 10, 0))  # 90.0%: within 5 points of 95%
    assert not judge_context.eligible(scores(89, 11, 0))
    assert judge_context.eligible(scores(85, 0, 15))  # 15% unsure
    assert not judge_context.eligible(scores(84, 0, 16))


# Also fix: held-out items split by source photo ------------------------------------------


def test_fix_held_out_by_source_shares_no_photo_with_the_screen_subset() -> None:
    manifest = goldset.load_manifest()
    screen = goldset.screening_subset(manifest)
    held = judge_context.held_out_by_source(manifest)
    screen_sources = {item.source for item in screen}
    assert held
    assert all(item.source not in screen_sources for item in held)
    assert {i.id for i in held} <= {i.id for i in judge.held_out(manifest)}
    # every item from a photo without a screen item is held out
    expected = [item.id for item in manifest.items if item.source not in screen_sources]
    assert [i.id for i in held] == expected
    # the pinned functions are unchanged
    assert len(judge.held_out(manifest)) == len(manifest.items) - goldset.SCREEN_SIZE
    assert len(screen) == goldset.SCREEN_SIZE


def test_fix_the_run_reports_how_many_items_move(
    server: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = goldset.load_manifest()
    moved = len(judge.held_out(manifest)) - len(judge_context.held_out_by_source(manifest))
    fake = server()
    judge_context.main(argv("--max-requests", "20"), endpoint=fake.url, crops=synthetic())
    out = capsys.readouterr().out
    assert f"{moved} items move out of the held-out set" in out


# Also fix: redaction ---------------------------------------------------------------------


def test_fix_long_runs_are_redacted_from_deepinfra_errors_without_a_known_token(
    server: Any,
) -> None:
    echoed = "sk_live-" + "Ab3" * 6  # 26 characters of [A-Za-z0-9_-]
    body = json.dumps({"error": {"message": f"key {echoed} has no access"}}).encode()
    fake = server(400, body)
    clf = judge.DeepInfraClassifier(
        judge.DEEPINFRA[model()], budget=judge.RequestBudget(2), endpoint=fake.url
    )
    with pytest.raises(judge.JudgeError) as caught:
        clf.classify(crop())
    exc = caught.value
    text = str(exc) + repr(exc) + "".join(traceback.format_exception(exc))
    assert echoed[:12] not in text and echoed[-12:] not in text
    assert "400" in str(exc) and "has no access" in str(exc)


def test_fix_short_words_in_deepinfra_errors_survive(server: Any) -> None:
    body = json.dumps({"error": {"message": "model is busy, try later"}}).encode()
    fake = server(400, body)
    clf = judge.DeepInfraClassifier(
        judge.DEEPINFRA[model()], budget=judge.RequestBudget(2), endpoint=fake.url
    )
    with pytest.raises(judge.JudgeError, match="model is busy, try later"):
        clf.classify(crop())


def test_fix_the_bedrock_error_kind_goes_through_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    token = "QwErTyUiOpAsDfGhJkLzXcVbNm"  # noqa: S105  (made up; letters only)
    monkeypatch.setenv(judge.TOKEN_ENV, token)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = b'{"message": "no"}'
            self.send_response(400)
            self.send_header("x-amzn-ErrorType", f"{token}:http://internal/")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        clf = judge.BedrockClassifier(
            next(iter(judge.HOSTED.values())),
            budget=judge.RequestBudget(2),
            endpoint=f"http://127.0.0.1:{httpd.server_address[1]}",
        )
        with pytest.raises(judge.JudgeError) as caught:
            clf.classify(crop())
    finally:
        httpd.shutdown()
        httpd.server_close()
    exc = caught.value
    text = str(exc) + repr(exc) + "".join(traceback.format_exception(exc))
    assert token not in text and token[:12] not in text
    assert "400" in str(exc)


def test_the_fake_server_is_on_the_loopback_interface(server: Any) -> None:
    fake = server()
    host = fake.url.split("//", 1)[1].split(":", 1)[0]
    assert socket.inet_aton(host) == socket.inet_aton("127.0.0.1")
