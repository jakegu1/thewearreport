"""Unit tests for wearreport.tools.judge. Every image here is synthetic.

Tests that need the judge weights skip only when they are missing and
WEARREPORT_REQUIRE_JUDGE is unset (the judge workflow sets it).
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport.tools import goldset, judge

ROOT = Path(__file__).resolve().parents[3]
FETCH_JUDGE = ROOT / "scripts" / "fetch_judge_model.sh"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("person - a person on foot", "person"),
        ("Person", "person"),
        ("  other.", "not_person"),
        ("Vehicle", "in_vehicle"),
        ("vehicle - a person inside a car", "unsure"),  # names two answers
        ("persons", "unsure"),
        ("the person", "unsure"),
        ("unsure", "unsure"),
        ("x" * (judge.MAX_ANSWER_CHARS + 1), "unsure"),
    ],
)
def test_parse_answer(text: str, expected: str) -> None:
    assert judge.parse_answer(text) == expected


def test_parse_answer_refuses_non_text() -> None:
    assert judge.parse_answer(None) == "unsure"  # type: ignore[arg-type]
    assert judge.parse_answer(b"person") == "unsure"  # type: ignore[arg-type]


def test_llama_cpp_environment_switches_are_removed_before_it_loads() -> None:
    env = dict(os.environ, MTMD_DEBUG_EMBEDDINGS="1", GGML_BACKEND_PATH="/nonexistent/evil.so")
    env["LLAMA_TRACE"] = "1"
    code = (
        "import os\n"
        "from wearreport.tools import judge\n"
        "judge.load_runtime()\n"
        "print(sorted(k for k in os.environ if k.startswith(('GGML_', 'LLAMA_', 'MTMD_'))))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


def test_every_candidate_has_its_own_files_and_a_template() -> None:
    names = [n for c in judge.CANDIDATES.values() for n in (c.model_file, c.mmproj_file)]
    assert len(names) == len(set(names))
    for c in judge.CANDIDATES.values():
        assert c.template in judge.TEMPLATES
        assert "{media}" in judge.TEMPLATES[c.template]
        assert c.card.startswith("https://huggingface.co/")
        assert all(re.fullmatch(r"[\w.-]+\.gguf", n) for n in (c.model_file, c.mmproj_file))


def test_fetch_script_pins_every_candidate_and_names_the_chosen_one() -> None:
    script = FETCH_JUDGE.read_text()
    assert f'CHOSEN="{judge.CHOSEN}"' in script
    for c in judge.CANDIDATES.values():
        for value in (c.repo, c.revision, c.model_file, c.model_sha256, c.mmproj_file):
            assert value in script, (c.name, value)
        assert c.mmproj_sha256 in script


def _counts(**cells: int) -> list[tuple[Any, Any]]:
    pairs: list[tuple[Any, Any]] = []
    for key, n in cells.items():
        truth, answer = key.split("__")
        pairs += [(truth, answer)] * n
    return pairs


def test_score_with_no_items_fails_the_bar() -> None:
    scores = judge.score([])
    assert scores.n == 0 and scores.unsure_rate == 1.0 and scores.accuracy == 0.0
    assert not judge.passes(scores, 1.0)


def test_score_without_negatives_cannot_estimate_precision() -> None:
    scores = judge.score(_counts(person__person=10))
    assert all(math.isnan(e) for e in scores.precision_error.values())
    assert scores.accuracy == 1.0
    assert not judge.passes(scores, 1.0)


def test_in_vehicle_answers_count_as_people_for_precision() -> None:
    # The judge confuses the two person classes, which costs accuracy but not precision.
    scores = judge.score(
        _counts(person__in_vehicle=50, in_vehicle__person=10, not_person__not_person=40)
    )
    assert scores.accuracy == pytest.approx(0.4)
    assert all(abs(e) < 1e-9 for e in scores.precision_error.values())


def test_unsure_answers_are_left_out_of_the_estimate() -> None:
    scores = judge.score(
        _counts(
            person__person=40,
            person__unsure=10,
            in_vehicle__in_vehicle=10,
            not_person__not_person=40,
        )
    )
    assert scores.unsure_rate == pytest.approx(0.1)
    # Unsure on people only: the estimate leans towards the negatives.
    assert all(e < 0 for e in scores.precision_error.values())


def test_a_judge_that_misses_negatives_overestimates_precision() -> None:
    scores = judge.score(
        _counts(
            person__person=50,
            in_vehicle__in_vehicle=10,
            not_person__not_person=30,
            not_person__person=10,
        )
    )
    errors = scores.precision_error
    # 25% of negatives are called people: est = (p + 0.25(1-p)) / 1
    for mix in judge.MIXES:
        assert errors[mix] == pytest.approx(100 * 0.25 * (1 - mix))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10), np.uint8),
        np.zeros((10, 10, 4), np.uint8),
        np.zeros((10, 10, 3), np.float32),
        np.zeros((0, 10, 3), np.uint8),
        np.zeros((judge.MAX_IMAGE_SIDE + 1, 10, 3), np.uint8),
        [[0, 0, 0]],
    ],
)
def test_validate_image_refuses_bad_images(image: object) -> None:
    with pytest.raises(ValueError):
        judge.validate_image(image)


class _Fake:
    def __init__(self, answers: list[judge.Answer]) -> None:
        self.answers = answers
        self.closed = False

    def classify(self, image: npt.NDArray[np.uint8]) -> judge.Answer:
        return self.answers.pop(0)

    def close(self) -> None:
        self.closed = True


def _crops() -> Iterator[tuple[goldset.Label, npt.NDArray[np.uint8]]]:
    for label in ("person", "person", "in_vehicle", "not_person"):
        yield label, np.zeros((32, 16, 3), np.uint8)


def test_bakeoff_reports_a_model_that_cannot_open_and_agreement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fakes = {
        "qwen3.5-2b": ["person", "person", "in_vehicle", "not_person"],
        "qwen3-vl-2b": ["person", "not_person", "in_vehicle", "not_person"],
    }
    opened: list[_Fake] = []

    def open_judge(candidate: judge.Candidate) -> judge.Classifier:
        if candidate.name not in fakes:
            raise judge.JudgeError("weights missing")
        fake = _Fake(list(fakes[candidate.name]))  # type: ignore[arg-type]
        opened.append(fake)
        return fake

    code = judge.main(["--bakeoff"], open_judge=open_judge, crops=_crops)
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("not run: weights missing") == len(judge.CANDIDATES) - 2
    assert "agreement of qwen3.5-2b and qwen3-vl-2b" in out
    assert all(fake.closed for fake in opened)
    assert "accuracy on confident answers 100.0% (4/4)" in out
    assert "unsure 25.0% (1/4)" in out  # the agreement: one disagreement


@pytest.mark.parametrize(
    "argv", [[], ["--bakeoff", "--models", "nope"], ["--bakeoff", "--limit", "0"]]
)
def test_main_refuses_bad_arguments(argv: list[str]) -> None:
    assert judge.main(argv, open_judge=lambda c: _Fake([]), crops=_crops) == 2


def test_chosen_is_an_alias_on_the_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    judge.main(
        ["--bakeoff", "--models", "chosen"],
        open_judge=lambda c: _Fake(["person", "person", "in_vehicle", "not_person"]),
        crops=_crops,
    )
    out = capsys.readouterr().out
    assert f"== {judge.CHOSEN} (" in out and out.count("== ") == 1


# With the chosen model's weights -----------------------------------------------------------


@pytest.fixture(scope="module")
def chosen() -> Iterator[judge.Judge]:
    c = judge.CANDIDATES[judge.CHOSEN]
    if not all((judge.MODEL_DIR / n).is_file() for n in (c.model_file, c.mmproj_file)):
        if os.environ.get("WEARREPORT_REQUIRE_JUDGE"):
            pytest.fail("the judge weights are missing and WEARREPORT_REQUIRE_JUDGE is set")
        pytest.skip("the judge weights are missing")
    jd = judge.Judge(c)
    try:
        yield jd
    finally:
        jd.close()


def test_replies_are_deterministic(chosen: judge.Judge) -> None:
    rng = np.random.default_rng(5)
    image: npt.NDArray[np.uint8] = rng.integers(0, 256, (200, 100, 3), dtype=np.uint8)
    first = chosen.reply(image)
    assert chosen.reply(image) == first
    assert len(first) <= 200


def test_a_closed_judge_refuses_to_answer() -> None:
    c = judge.CANDIDATES[judge.CHOSEN]
    if not all((judge.MODEL_DIR / n).is_file() for n in (c.model_file, c.mmproj_file)):
        pytest.skip("the judge weights are missing")
    jd = judge.Judge(c)
    jd.close()
    jd.close()  # twice is fine
    with pytest.raises(judge.JudgeError):
        jd.classify(np.zeros((50, 30, 3), np.uint8))
