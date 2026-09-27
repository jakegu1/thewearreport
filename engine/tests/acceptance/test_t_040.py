"""Acceptance tests for T-040: the spot-check summary by session.

Every statistics file here is synthetic (counts only): no image, no network.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from wearreport.tools import spotcheck_summary

QWEN = "di-qwen3-vl-235b"
GEMMA = "di-gemma-4-31b"
JUDGE_COLUMNS = ("person", "in_vehicle", "not_person", "unsure")
Table = dict[str, dict[str, int]]


def _confusion(person: tuple[int, ...], not_person: tuple[int, ...]) -> Table:
    rows = {"person": person, "in_vehicle": (0, 0, 0, 0), "not_person": not_person}
    return {r: dict(zip(JUDGE_COLUMNS, counts, strict=True)) for r, counts in rows.items()}


# Columns: the judge says person, in_vehicle, not_person, unsure.
# A: reviewer 90 persons, 10 not; judge Se 85/90, Sp 8/10, apparent 87/100.
A = _confusion((85, 0, 5, 0), (2, 0, 8, 0))
# B: reviewer 90 persons, 10 not; judge Se 80/90, Sp 10/10, apparent 80/100.
B = _confusion((80, 0, 10, 0), (0, 0, 10, 0))
# A2: A doubled, for sessions of 200 boxes.
A2 = _confusion((170, 0, 10, 0), (4, 0, 16, 0))
# Only unsure answers: no apparent precision, so no corrected estimate.
UNSURE = _confusion((0, 0, 0, 90), (0, 0, 0, 10))
# Wild answers: would move any calibration they entered.
WILD = _confusion((0, 0, 90, 0), (10, 0, 0, 0))


def _file(
    directory: Path,
    name: str,
    day: str,
    shown: int,
    not_person: int,
    confusion: Table | None,
    model: str = QWEN,
    status: str = "complete",
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
            "model": model,
            "provider": "DeepInfra",
            "status": status,
            "requests": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "cost_usd": 0.0,
            "confusion": confusion,
            "judge_precision": None,
        }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(record), encoding="utf-8")


def _run(directory: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, str, str]:
    code = spotcheck_summary.main(["--by-session", "--dir", str(directory), *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _row(output: str, name: str) -> str:
    [line] = [line for line in output.splitlines() if line.startswith(name)]
    return line


def _skipped(output: str, name: str) -> str:
    [line] = [line for line in output.splitlines() if line.startswith(f"skipped {name}")]
    return line


def _last(output: str) -> str:
    return output.rstrip("\n").splitlines()[-1]


def _rg(apparent: Fraction, se: Fraction, sp: Fraction) -> Fraction:
    value = (apparent + sp - 1) / (se + sp - 1)
    return min(Fraction(1), max(Fraction(0), value))


# AC1: the mode and the default ---------------------------------------------------------

DEFAULT_OUTPUT = (
    "5 statistics file(s) in {d}\n"
    "Judge di-gemma-4-31b, corrected with the confusion of earlier weeks (Rogan-Gladen):\n"
    "2026-W38  reviewer 0.9000 (n=200)  judge n/a (n=0)  corrected n/a  diff n/a"
    "  within 3 pts: n/a\n"
    "2026-W39  reviewer 0.9000 (n=160)  judge 0.8700 (n=100)  corrected n/a  diff n/a"
    "  within 3 pts: n/a\n"
    "two weeks within 3 points: no\n"
    "Judge di-qwen3-vl-235b, corrected with the confusion of earlier weeks (Rogan-Gladen):\n"
    "2026-W38  reviewer 0.9000 (n=200)  judge 0.8700 (n=200)  corrected n/a  diff n/a"
    "  within 3 pts: n/a\n"
    "2026-W39  reviewer 0.9000 (n=160)  judge 0.8000 (n=100)  corrected 0.8060  diff -9.40 pts"
    "  within 3 pts: no\n"
    "two weeks within 3 points: no\n"
)


def _mixed(d: Path) -> None:
    _file(d, "2026-09-15.json", "2026-09-15", 100, 10, A)
    _file(d, "2026-09-16.json", "2026-09-16", 100, 10, A)
    _file(d, "2026-09-22.json", "2026-09-22", 100, 10, B)
    _file(d, "2026-09-23.json", "2026-09-23", 40, 4, A, model=GEMMA, status="incomplete")
    _file(d, "2026-09-24.json", "2026-09-24", 20, 2, None)


def test_ac1_without_by_session_the_output_is_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _mixed(d)
    # Captured from the tool before T-040, on the same files.
    assert spotcheck_summary.main(["--dir", str(d)]) == 0
    captured = capsys.readouterr()
    assert captured.out == DEFAULT_OUTPUT.format(d=d)
    assert captured.err == ""


def test_ac1_the_only_model_in_qualifying_sessions_is_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _mixed(d)  # gemma appears only in an incomplete, small session
    code, out, _ = _run(d, capsys)
    assert code == 0
    assert QWEN in out
    assert _row(out, "2026-09-15.json")


def test_ac1_several_models_are_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _file(d, "a.json", "2026-09-15", 100, 10, A)
    _file(d, "b.json", "2026-09-16", 100, 10, A, model=GEMMA)
    code, _, err = _run(d, capsys)
    assert code == 2
    assert QWEN in err and GEMMA in err


def test_ac1_model_selects_one_of_several(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _file(d, "a.json", "2026-09-15", 100, 10, A)
    _file(d, "b.json", "2026-09-16", 100, 10, A, model=GEMMA)
    code, out, _ = _run(d, capsys, "--model", GEMMA)
    assert code == 0
    assert _row(out, "b.json")
    assert QWEN in _skipped(out, "a.json")


# AC2 and AC3: qualifying sessions, leave-one-session-out -------------------------------


def test_ac3_a_hand_computed_leave_one_out_example(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _file(d, "s1.json", "2026-09-15", 100, 10, A)
    _file(d, "s2.json", "2026-09-16", 100, 10, A)
    _file(d, "s3.json", "2026-09-16", 100, 10, B)
    # None of these may enter any calibration: WILD would move every estimate.
    _file(d, "x-incomplete.json", "2026-09-16", 100, 10, WILD, status="incomplete")
    _file(d, "x-small.json", "2026-09-16", 99, 10, WILD)
    _file(d, "x-other.json", "2026-09-16", 100, 10, WILD, model=GEMMA)
    _file(d, "x-none.json", "2026-09-16", 100, 10, None)
    code, out, _ = _run(d, capsys, "--model", QWEN)
    assert code == 0

    # s1: the others are s2 (A) and s3 (B), pooled.
    #   Reviewer says a person: judge person 85 + 80 = 165, judge not 5 + 10 = 15:
    #     Se = 165 / 180 = 11/12.
    #   Reviewer says not a person: judge person 2 + 0 = 2, judge not 8 + 10 = 18:
    #     Sp = 18 / 20 = 9/10.
    #   s1's judge answers: 87 person out of 100 confident: q = 87/100.
    #   (q + Sp - 1) / (Se + Sp - 1) = (0.87 + 0.9 - 1) / (11/12 + 0.9 - 1)
    #                                = 0.77 / (49/60) = 33/35 = 0.942857...
    #   Reviewer: 90/100. Difference +4.29 points: not within 3.
    s1 = _rg(Fraction(87, 100), Fraction(11, 12), Fraction(9, 10))
    assert s1 == Fraction(33, 35)
    row = _row(out, "s1.json")
    assert "2026-09-15" in row
    assert "reviewer 0.9000 (n=100)" in row
    assert "judge 0.8700 (n=100)" in row
    assert f"corrected {float(s1):.4f}" in row
    assert "corrected 0.9429" in row
    assert "diff +4.29 pts" in row
    assert "within 3 pts: no" in row

    # s2 is the same as s1, with the same others (s1 is A, s3 is B): the same figures.
    assert "corrected 0.9429" in _row(out, "s2.json")

    # s3: the others are s1 and s2 (both A).
    #   Se = 170 / 180 = 17/18, Sp = 16 / 20 = 4/5, q = 80/100.
    #   (0.8 + 0.8 - 1) / (17/18 + 0.8 - 1) = 0.6 / (67/90) = 54/67 = 0.805970...
    #   Reviewer 0.9: -9.40 points.
    #   (With s3's own answers in the calibration it would be 90/107 = 0.8411: not this.)
    s3 = _rg(Fraction(80, 100), Fraction(17, 18), Fraction(4, 5))
    assert s3 == Fraction(54, 67)
    row = _row(out, "s3.json")
    assert "2026-09-16" in row
    assert "judge 0.8000 (n=100)" in row
    assert f"corrected {float(s3):.4f}" in row
    assert "diff -9.40 pts" in row
    assert "within 3 pts: no" in row
    assert _last(out).startswith("ready: no (")


def test_ac2_skipped_files_are_listed_with_their_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _file(d, "s1.json", "2026-09-15", 100, 10, A)
    _file(d, "x-incomplete.json", "2026-09-16", 100, 10, A, status="incomplete")
    _file(d, "x-small.json", "2026-09-16", 99, 10, A)
    _file(d, "x-other.json", "2026-09-16", 100, 10, A, model=GEMMA)
    _file(d, "x-none.json", "2026-09-16", 100, 10, None)
    code, out, _ = _run(d, capsys, "--model", QWEN)
    assert code == 0
    assert "incomplete" in _skipped(out, "x-incomplete.json")
    assert "fewer than 100" in _skipped(out, "x-small.json")
    assert GEMMA in _skipped(out, "x-other.json")
    assert "no judge block" in _skipped(out, "x-none.json")
    for name in ("x-incomplete.json", "x-small.json", "x-other.json", "x-none.json"):
        assert not [line for line in out.splitlines() if line.startswith(name)]
    # s1 has no other qualifying session: no calibration, so n/a.
    assert "corrected n/a" in _row(out, "s1.json")


def test_ac2_one_hundred_boxes_qualify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert spotcheck_summary.MIN_SESSION_BOXES == 100
    d = tmp_path / "stats"
    _file(d, "s1.json", "2026-09-15", 100, 10, A)
    code, out, _ = _run(d, capsys)
    assert code == 0
    assert _row(out, "s1.json")
    assert "skipped" not in out


# AC4: the readiness verdict ------------------------------------------------------------


def test_ac4_the_thresholds_are_named_constants() -> None:
    assert spotcheck_summary.MIN_SESSIONS == 3
    assert spotcheck_summary.MIN_DAYS == 2
    assert spotcheck_summary.MIN_POOLED_BOXES == 300
    assert spotcheck_summary.MIN_SESSION_BOXES == 100
    assert spotcheck_summary.WITHIN_POINTS == 3


def _ready_three(d: Path) -> None:
    # Three A sessions: each one's others pool to Se 170/180 = 17/18, Sp 16/20 = 4/5, and
    # q = 87/100: (0.87 + 0.8 - 1) / (17/18 - 0.2) = 0.67 / (67/90) = 0.9, the reviewer's.
    _file(d, "s1.json", "2026-09-15", 100, 10, A)
    _file(d, "s2.json", "2026-09-16", 100, 10, A)
    _file(d, "s3.json", "2026-09-16", 100, 10, A)


def test_ac4_ready_when_every_condition_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _ready_three(d)
    code, out, _ = _run(d, capsys)
    assert code == 0
    for name in ("s1.json", "s2.json", "s3.json"):
        row = _row(out, name)
        assert "corrected 0.9000" in row and "diff +0.00 pts" in row
        assert "within 3 pts: yes" in row
    assert _last(out) == "ready: yes"


def test_ac4_too_few_sessions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "stats"
    # Two sessions of 200 boxes on two dates, each within (A2 is A doubled: 0.9 again).
    _file(d, "s1.json", "2026-09-15", 200, 20, A2)
    _file(d, "s2.json", "2026-09-16", 200, 20, A2)
    code, out, _ = _run(d, capsys)
    assert code == 0
    assert "within 3 pts: yes" in _row(out, "s1.json")
    assert "within 3 pts: yes" in _row(out, "s2.json")
    last = _last(out)
    assert last.startswith("ready: no (") and "session" in last


def test_ac4_too_few_dates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "stats"
    for name in ("s1.json", "s2.json", "s3.json"):
        _file(d, name, "2026-09-16", 100, 10, A)
    code, out, _ = _run(d, capsys)
    assert code == 0
    assert all("within 3 pts: yes" in _row(out, n) for n in ("s1.json", "s2.json", "s3.json"))
    last = _last(out)
    assert last.startswith("ready: no (") and "date" in last


def test_ac4_a_session_not_within(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "stats"
    _ready_three(d)
    _file(d, "s4.json", "2026-09-17", 100, 40, A)  # the reviewer finds 60%
    code, out, _ = _run(d, capsys)
    assert code == 0
    assert "within 3 pts: no" in _row(out, "s4.json")
    last = _last(out)
    assert last.startswith("ready: no (") and "s4.json" in last


def test_ac4_a_session_without_an_estimate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "stats"
    _ready_three(d)
    # Only unsure answers: its confident counts are 0, so it moves no other calibration.
    _file(d, "s4.json", "2026-09-17", 100, 10, UNSURE)
    code, out, _ = _run(d, capsys)
    assert code == 0
    assert all("within 3 pts: yes" in _row(out, n) for n in ("s1.json", "s2.json", "s3.json"))
    assert "corrected n/a" in _row(out, "s4.json")
    last = _last(out)
    assert last.startswith("ready: no (") and "s4.json" in last


def test_ac4_too_few_pooled_boxes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # With the shipped constants, 3 sessions of at least 100 boxes pool to at least 300,
    # so this condition can only fail on its own with a higher threshold.
    monkeypatch.setattr(spotcheck_summary, "MIN_POOLED_BOXES", 301)
    d = tmp_path / "stats"
    _ready_three(d)  # 300 boxes pooled
    code, out, _ = _run(d, capsys)
    assert code == 0
    last = _last(out)
    assert last.startswith("ready: no (") and "boxes" in last


# The hostile files of T-036 AC6, now in this mode --------------------------------------


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
def test_hostile_files_are_rejected_with_the_file_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes
) -> None:
    d = tmp_path / "stats"
    _ready_three(d)
    (d / "2026-09-15-2.json").write_bytes(content)
    code, _, err = _run(d, capsys)
    assert code == 1
    assert "2026-09-15-2.json" in err


# AC6: the docs -------------------------------------------------------------------------


def test_ac6_docs() -> None:
    root = Path(__file__).resolve().parents[3]
    readme = (root / "spotchecks" / "README.md").read_text(encoding="utf-8")
    for needed in ("--by-session", "ready: yes", "ready: no", "100", "300"):
        assert needed in readme, needed
    doc = spotcheck_summary.__doc__ or ""
    assert "--by-session" in doc and "ISO week" in doc
