"""Acceptance tests for T-042: a near-field threshold that also requires most boxes to be
judgeable.

Every per-box file here is synthetic, written by the test into a temporary directory.
Nothing reaches the network and no image is involved.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from wearreport.tools import spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "spotchecks" / "README.md"
DETECTOR = {"model": "yolox_m", "sha256": "0" * 64, "conf": 0.3}
Z = 1.959963984540054  # the 97.5th percentile of the standard normal distribution
RULE = (
    "threshold rule: precision >= 0.90, Wilson lower bound >= 0.85, "
    "at least 100 judged boxes, judged share >= 0.80"
)


def _session(
    directory: Path,
    name: str,
    started_at: str,
    light: str,
    boxes: Sequence[tuple[int, str, int]],
) -> None:
    """A per-box file; `boxes` holds (height, label, how many)."""
    record = {
        "date": started_at[:10],
        "started_at": started_at,
        "light": light,
        "frames": 5,
        "detector": DETECTOR,
        "boxes": [[h, label] for h, label, count in boxes for _ in range(count)],
    }
    (directory / "boxes").mkdir(parents=True, exist_ok=True)
    (directory / "boxes" / name).write_text(json.dumps(record), encoding="utf-8")


def _stat(k: int, n: int) -> str:
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z / d * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return f"n={n} precision {p:.4f} wilson [{lo:.4f}, {hi:.4f}]"


def _heights(d: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, list[str], str]:
    code = spotcheck_summary.main(["--heights", "--dir", str(d)])
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.err


def _example(d: Path) -> None:
    # 20 px: 10 people and 60 unsure. 40 px: 120 people, 5 not a person and 10 unsure.
    _session(d, "a.json", "2026-09-20T10:00Z", "day", [(20, "person", 10), (20, "unsure", 60)])
    boxes = [(40, "person", 120), (40, "not_person", 5), (40, "unsure", 10)]
    _session(d, "b.json", "2026-09-21T18:30Z", "twilight", boxes)


# AC1, AC2, AC3 ------------------------------------------------------------------------


def test_hand_computed_combined_rule_picks_a_larger_height(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # H = 20: judged 10 + 120 + 5 = 135, people 130, precision 130/135 = 0.9630.
    #   Wilson 95%, z = 1.96, p = 0.962963, n = 135: z^2/n = 0.028455,
    #   centre = (0.962963 + 0.014228) / 1.028455 = 0.950154,
    #   half = 1.96 / 1.028455 * sqrt(0.962963 * 0.037037 / 135 + 3.841459 / 72900)
    #        = 1.905736 * sqrt(0.000264193 + 0.000052695) = 1.905736 * 0.017801 = 0.033925,
    #   lower = 0.9162 >= 0.85, and 135 >= 100 judged boxes: the precision conditions hold,
    #   so they alone would choose 20 px.
    #   Judged share: 135 of 135 + 70 unsure = 205, 135/205 = 0.6585 < 0.80: it fails.
    # H = 40: judged 125, people 120, precision 0.9600, Wilson lower 0.9098 >= 0.85,
    #   125 >= 100; judged share 125 of 135, 0.9259 >= 0.80: it qualifies.
    # The threshold is 40 px.
    d = tmp_path / "spotchecks"
    _example(d)
    code, lines, err = _heights(d, capsys)
    assert code == 0, err
    rows = [line for line in lines if line.startswith(">= ")]
    assert rows == [
        f">= 20 px: {_stat(130, 135)} kept 1.0000",
        f">= 40 px: {_stat(120, 125)} kept {120 / 130:.4f}",
    ]
    assert "wilson [0.9162," in rows[0] and "precision 0.9630" in rows[0]
    assert "judged share >= 20 px: 135 of 205 (0.6585)" in lines
    assert "judged share >= 40 px: 125 of 135 (0.9259)" in lines
    assert "threshold: 40 px" in lines
    assert "at 40 px:" in lines


def test_the_exact_report_lines_and_their_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "spotchecks"
    _example(d)
    code, lines, err = _heights(d, capsys)
    assert code == 0, err
    assert lines == [
        f"2 per-box file(s) in {d / 'boxes'}",
        "unsure boxes (left out): 70",
        f">= 20 px: {_stat(130, 135)} kept 1.0000",
        f">= 40 px: {_stat(120, 125)} kept {120 / 130:.4f}",
        "judged share >= 20 px: 135 of 205 (0.6585)",
        "judged share >= 40 px: 125 of 135 (0.9259)",
        RULE,
        "threshold: 40 px",
        "at 40 px:",
        "  light day: n=0 precision n/a wilson n/a",
        f"  light twilight: {_stat(120, 125)}",
        "  light dark: n=0 precision n/a wilson n/a",
        "coverage: 2 sessions, 2 dates, 135 judged boxes "
        "(day 10, twilight 125, dark 0; rain 0, dry 0, unknown 135); "
        "baseline complete: no (judged boxes 135 < 300; sessions 2 < 3; day 10 < 50; "
        "rain 0 < 50)",
    ]
    assert len([line for line in lines if line.startswith(">= ")]) == 2  # the rows only


def test_a_share_of_exactly_080_qualifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 100 people and 25 unsure at 50 px: 100/125 = 0.80 exactly; precision 1.0, Wilson
    # lower 0.9630, 100 judged boxes.
    d = tmp_path / "spotchecks"
    _session(d, "a.json", "2026-09-20T10:00Z", "day", [(50, "person", 100), (50, "unsure", 25)])
    code, lines, err = _heights(d, capsys)
    assert code == 0, err
    assert "judged share >= 50 px: 100 of 125 (0.8000)" in lines
    assert "threshold: 50 px" in lines


def test_one_unsure_box_more_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 100/126 = 0.7937 < 0.80.
    d = tmp_path / "spotchecks"
    _session(d, "a.json", "2026-09-20T10:00Z", "day", [(50, "person", 100), (50, "unsure", 26)])
    code, lines, err = _heights(d, capsys)
    assert code == 0, err
    assert "judged share >= 50 px: 100 of 126 (0.7937)" in lines
    assert "threshold: none" in lines


def test_none_when_every_precise_height_is_mostly_unsure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 30 px: 100 people, 40 unsure; 60 px: 110 people, 30 unsure. Both heights meet the
    # precision conditions (precision 1.0 over 210 and 110 judged boxes), but the judged
    # share is 210/280 = 0.75 and 110/140 = 0.7857.
    d = tmp_path / "spotchecks"
    boxes = [(30, "person", 100), (30, "unsure", 40), (60, "person", 110), (60, "unsure", 30)]
    _session(d, "a.json", "2026-09-20T10:00Z", "day", boxes)
    code, lines, err = _heights(d, capsys)
    assert code == 0, err
    assert f">= 30 px: {_stat(210, 210)} kept 1.0000" in lines
    assert f">= 60 px: {_stat(110, 110)} kept {110 / 210:.4f}" in lines
    assert "judged share >= 30 px: 210 of 280 (0.7500)" in lines
    assert "judged share >= 60 px: 110 of 140 (0.7857)" in lines
    assert "threshold: none" in lines
    assert "over all boxes:" in lines


def test_only_unsure_boxes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "spotchecks"
    _session(d, "a.json", "2026-09-20T10:00Z", "day", [(30, "unsure", 3), (45, "unsure", 1)])
    code, lines, err = _heights(d, capsys)
    assert code == 0, err
    assert "judged share >= 30 px: 0 of 4 (0.0000)" in lines
    assert "judged share >= 45 px: 0 of 1 (0.0000)" in lines
    assert "threshold: none" in lines
    assert RULE in lines


def test_named_constants() -> None:
    assert spotcheck_summary.MIN_JUDGED_SHARE == 0.80
    assert spotcheck_summary.TARGET_PRECISION == 0.90
    assert spotcheck_summary.MIN_LOWER_BOUND == 0.85
    assert spotcheck_summary.MIN_BOXES_ABOVE == 100
    assert spotcheck_summary.BASELINE_BOXES == 300
    assert spotcheck_summary.MIN_CONDITION_BOXES == 50
    assert spotcheck_summary.RAIN_JOIN_MINUTES == 30


def test_the_default_summary_does_not_mention_the_judged_share(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "spotchecks"
    _example(d)
    stats = {"date": "2026-09-20", "boxes_shown": 10, "boxes_not_person": 1}
    (d / "2026-09-20.json").write_text(json.dumps(stats), encoding="utf-8")
    assert spotcheck_summary.main(["--dir", str(d)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"1 statistics file(s) in {d}",
        "No judge block: reviewer only.",
        "2026-W38  reviewer 0.9000 (n=10)  judge n/a (n=0)  corrected n/a  diff n/a  "
        "within 3 pts: n/a",
        "two weeks within 3 points: no",
    ]


# AC5 ----------------------------------------------------------------------------------


def test_readme_documents_the_judged_share() -> None:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Near-field threshold", 1)[1].split("\n## ", 1)[0]
    assert "judged share" in section
    assert "MIN_JUDGED_SHARE" in section and "0.80" in section
    assert "verif" in section
