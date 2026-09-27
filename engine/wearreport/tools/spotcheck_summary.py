"""Summarise the spot-check statistics files, reviewer against hosted judge, per ISO week
(the default) or per session (`--by-session`).

  python -m wearreport.tools.spotcheck_summary [--dir spotchecks]
  python -m wearreport.tools.spotcheck_summary --by-session [--dir spotchecks] [--model NAME]

Per ISO week

Reads every `*.json` statistics file in the directory (counts only: no image, no network)
and prints, for each ISO week and each judge model found in a `judge` block:

- the reviewer's precision (person boxes over boxes shown, pooled over the week's files)
  and n, the boxes shown;
- the judge's precision, computed as in the statistics file (person or in_vehicle answers
  over its confident answers, pooled) and n, its confident answers;
- a corrected judge estimate: the judge's answers this week, corrected by inverting the
  reviewer x judge confusion pooled over all *earlier* weeks (Rogan-Gladen). With the
  earlier weeks' sensitivity Se = P(judge says person | reviewer says person) and
  specificity Sp = P(judge says not a person | reviewer says not a person), both on
  confident answers, and this week's apparent precision q, the estimate is
  (q + Sp - 1) / (Se + Sp - 1), clipped to [0, 1]. It is "n/a" without earlier data, when
  Se or Sp is undefined, or when Se + Sp - 1 is 0 (the judge's answer says nothing);
- the difference from the reviewer's precision, in points, and whether it is within
  WITHIN_POINTS.

Last, whether the condition holds: the two latest weeks with a corrected estimate are both
within WITHIN_POINTS of the reviewer. A file without a `judge` block counts for the
reviewer only.

Per session (--by-session)

A session is one statistics file. For one judge model (`--model`, or the only model found
in qualifying sessions; several without `--model` is an error, exit 2), a session
qualifies when its `judge` block has that model, `status` `complete` and at least
MIN_SESSION_BOXES boxes shown. The others are listed as skipped, with the reason, and
enter no calibration. Each qualifying session gets one row: the same figures as a week,
except that its corrected estimate inverts the confusion pooled over every *other*
qualifying session (leave one session out). The last line is `ready: yes` when there are
at least MIN_SESSIONS qualifying sessions on at least MIN_DAYS distinct dates, every one
has a corrected estimate within WITHIN_POINTS, and they pool at least MIN_POOLED_BOXES
boxes shown; otherwise `ready: no (<the first condition not met>)`.

In both modes a malformed file is an error that names it (exit 1).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = "spotchecks"
MAX_FILE_BYTES = 1 << 20
MAX_COUNT = 10**9
REVIEWER_LABELS = ("person", "in_vehicle", "not_person")
OTHER_REVIEWER_LABELS = ("unsure",)  # allowed, counted for neither side
JUDGE_ANSWERS = ("person", "in_vehicle", "not_person", "unsure")
POSITIVE = ("person", "in_vehicle")
WITHIN_POINTS = 3.0
WEEKS_NEEDED = 2
MIN_SESSION_BOXES = 100
MIN_SESSIONS = 3
MIN_DAYS = 2
MIN_POOLED_BOXES = 300
COMPLETE = "complete"
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MODEL = re.compile(r"[A-Za-z0-9][\w.-]{0,63}")

Confusion = dict[str, dict[str, int]]


class SummaryError(ValueError):
    """A statistics file cannot be read; the message names it."""


@dataclass(frozen=True, slots=True)
class Record:
    """What the summary reads from one statistics file."""

    day: datetime.date
    shown: int
    not_person: int
    model: str | None
    confusion: Confusion | None
    status: str | None = None  # the judge block's status, when it is a string
    name: str = ""  # the file name


def _count(value: object, what: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_COUNT:
        raise ValueError(f"{what} is not a count")
    return value


def _confusion(value: object) -> Confusion:
    if not isinstance(value, dict):
        raise ValueError("confusion is not an object")
    unknown = set(value) - set(REVIEWER_LABELS) - set(OTHER_REVIEWER_LABELS)
    if unknown or not set(REVIEWER_LABELS) <= set(value):
        raise ValueError("confusion does not have the reviewer's labels")
    table: Confusion = {}
    for label, row in value.items():
        if not isinstance(row, dict) or set(row) != set(JUDGE_ANSWERS):
            raise ValueError("a confusion row does not have the judge's answers")
        table[label] = {answer: _count(row[answer], "a confusion cell") for answer in row}
    return table


def parse_record(raw: bytes) -> Record:
    """One statistics file; raises ValueError (or a subclass) when it is malformed."""
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"larger than {MAX_FILE_BYTES} bytes")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("not an object")
    day = data.get("date")
    if not isinstance(day, str) or not _DATE.fullmatch(day):
        raise ValueError("date is not YYYY-MM-DD")
    shown = _count(data.get("boxes_shown"), "boxes_shown")
    not_person = _count(data.get("boxes_not_person"), "boxes_not_person")
    if not_person > shown:
        raise ValueError("more boxes not a person than boxes shown")
    model: str | None = None
    confusion: Confusion | None = None
    if "judge" in data:
        block = data["judge"]
        if not isinstance(block, dict):
            raise ValueError("judge is not an object")
        name = block.get("model")
        if not isinstance(name, str) or not _MODEL.fullmatch(name):
            raise ValueError("judge.model is not a model name")
        model, confusion = name, _confusion(block.get("confusion"))
        status = block.get("status")
        judge_status = status if isinstance(status, str) else None
    else:
        judge_status = None
    return Record(
        datetime.date.fromisoformat(day), shown, not_person, model, confusion, judge_status
    )


def load(directory: Path) -> list[Record]:
    """Every statistics file in `directory`, in name order."""
    try:
        paths = sorted(p for p in directory.iterdir() if p.suffix == ".json" and p.is_file())
    except OSError as exc:
        raise SummaryError(f"cannot read {directory}: {exc.strerror}") from None
    records = []
    for path in paths:
        try:
            with open(path, "rb") as fh:
                raw = fh.read(MAX_FILE_BYTES + 1)
            records.append(dataclasses.replace(parse_record(raw), name=path.name))
        except OSError as exc:
            raise SummaryError(f"cannot read {path.name}: {exc.strerror}") from None
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
            reason = str(exc) if type(exc) is ValueError else type(exc).__name__
            raise SummaryError(f"{path.name} is not a statistics file ({reason})") from None
    return records


# The maths -----------------------------------------------------------------------------


def empty() -> Confusion:
    return {label: dict.fromkeys(JUDGE_ANSWERS, 0) for label in REVIEWER_LABELS}


def pooled(tables: Sequence[Confusion]) -> Confusion:
    total = empty()
    for table in tables:
        for label, row in table.items():
            if label in total:
                for answer, n in row.items():
                    total[label][answer] += n
    return total


def _answers(table: Confusion, labels: Sequence[str]) -> tuple[int, int]:
    """The judge's confident answers on the rows `labels`: (says a person, says not)."""
    says_person = sum(table[label][a] for label in labels for a in POSITIVE)
    says_not = sum(table[label]["not_person"] for label in labels)
    return says_person, says_not


def judge_precision(table: Confusion) -> float | None:
    """The share of the judge's confident answers that say a person, or None."""
    says_person, says_not = _answers(table, REVIEWER_LABELS)
    confident = says_person + says_not
    return says_person / confident if confident else None


def corrected_estimate(current: Confusion, earlier: Confusion) -> float | None:
    """This week's precision from the judge's answers alone (`current`, whose reviewer
    labels are not used), corrected with the pooled confusion of earlier weeks (Rogan-
    Gladen), clipped to [0, 1]; None when it cannot be computed."""
    apparent = judge_precision(current)
    tp, fn = _answers(earlier, POSITIVE)  # reviewer: a person
    fp, tn = _answers(earlier, ("not_person",))  # reviewer: not a person
    if apparent is None or not (tp + fn) or not (fp + tn):
        return None
    sensitivity, specificity = tp / (tp + fn), tn / (fp + tn)
    determinant = sensitivity + specificity - 1
    if abs(determinant) < 1e-12:
        return None
    return min(1.0, max(0.0, (apparent + specificity - 1) / determinant))


@dataclass(frozen=True, slots=True)
class Week:
    week: tuple[int, int]  # ISO year, ISO week
    reviewer: float | None
    shown: int
    judge: float | None
    confident: int
    corrected: float | None

    @property
    def label(self) -> str:
        return f"{self.week[0]}-W{self.week[1]:02d}"

    @property
    def diff_points(self) -> float | None:
        return _points(self.corrected, self.reviewer)

    @property
    def within(self) -> bool | None:
        return _within(self.diff_points)


def _points(corrected: float | None, reviewer: float | None) -> float | None:
    if corrected is None or reviewer is None:
        return None
    return 100 * (corrected - reviewer)


def _within(diff: float | None) -> bool | None:
    return None if diff is None else abs(diff) <= WITHIN_POINTS + 1e-9


def _iso_week(day: datetime.date) -> tuple[int, int]:
    year, week, _ = day.isocalendar()
    return year, week


def weeks(records: Sequence[Record], model: str | None) -> list[Week]:
    """One row per ISO week, with the judge's figures for `model` (None: no judge)."""
    by_week: dict[tuple[int, int], list[Record]] = {}
    for record in records:
        by_week.setdefault(_iso_week(record.day), []).append(record)
    rows: list[Week] = []
    earlier: list[Confusion] = []
    for key in sorted(by_week):
        group = by_week[key]
        shown = sum(r.shown for r in group)
        positive = sum(r.shown - r.not_person for r in group)
        tables = [r.confusion for r in group if r.model == model and r.confusion is not None]
        current = pooled(tables) if tables else None
        judge: float | None = None
        corrected: float | None = None
        says_person = says_not = 0
        if current is not None:
            says_person, says_not = _answers(current, REVIEWER_LABELS)
            judge = judge_precision(current)
            corrected = corrected_estimate(current, pooled(earlier)) if earlier else None
            earlier.append(current)
        rows.append(
            Week(
                week=key,
                reviewer=positive / shown if shown else None,
                shown=shown,
                judge=judge,
                confident=says_person + says_not,
                corrected=corrected,
            )
        )
    return rows


def condition_holds(rows: Sequence[Week]) -> bool:
    """The two latest weeks with a corrected estimate are both within WITHIN_POINTS."""
    judged = [row for row in rows if row.within is not None]
    return len(judged) >= WEEKS_NEEDED and all(row.within for row in judged[-WEEKS_NEEDED:])


# Per session ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Session:
    name: str
    day: datetime.date
    reviewer: float | None
    shown: int
    judge: float | None
    confident: int
    corrected: float | None

    @property
    def diff_points(self) -> float | None:
        return _points(self.corrected, self.reviewer)

    @property
    def within(self) -> bool | None:
        return _within(self.diff_points)


def skip_reason(record: Record, model: str | None) -> str | None:
    """Why `record` is not a qualifying session for `model` (None: any model), or None."""
    if record.model is None or record.confusion is None:
        return "no judge block"
    if model is not None and record.model != model:
        return f"judge model {record.model}, not {model}"
    if record.status != COMPLETE:
        return (
            "judge status incomplete"
            if record.status == "incomplete"
            else "judge status not complete"
        )
    if record.shown < MIN_SESSION_BOXES:
        return f"{record.shown} boxes shown, fewer than {MIN_SESSION_BOXES}"
    return None


def qualifying_models(records: Sequence[Record]) -> list[str]:
    """The judge models of the sessions that would qualify for their own model."""
    return sorted(
        {r.model for r in records if r.model is not None and skip_reason(r, None) is None}
    )


def sessions(records: Sequence[Record], model: str) -> list[Session]:
    """One row per qualifying session, each corrected with every other one's confusion."""
    chosen = [
        (r, r.confusion)
        for r in records
        if r.confusion is not None and skip_reason(r, model) is None
    ]
    rows: list[Session] = []
    for i, (record, table) in enumerate(chosen):
        others = [other for j, (_, other) in enumerate(chosen) if j != i]
        says_person, says_not = _answers(table, REVIEWER_LABELS)
        shown = record.shown
        rows.append(
            Session(
                name=record.name,
                day=record.day,
                reviewer=(shown - record.not_person) / shown if shown else None,
                shown=shown,
                judge=judge_precision(table),
                confident=says_person + says_not,
                corrected=corrected_estimate(table, pooled(others)) if others else None,
            )
        )
    return rows


def readiness(rows: Sequence[Session]) -> str | None:
    """The first readiness condition `rows` do not meet, or None when they meet them all."""
    if len(rows) < MIN_SESSIONS:
        return f"{len(rows)} qualifying session(s), fewer than {MIN_SESSIONS}"
    days = len({row.day for row in rows})
    if days < MIN_DAYS:
        return f"{days} distinct date(s), fewer than {MIN_DAYS}"
    for row in rows:
        if row.corrected is None:
            return f"{row.name} has no corrected estimate"
        if not row.within:
            return f"{row.name} is not within {WITHIN_POINTS:g} points"
    total = sum(row.shown for row in rows)
    if total < MIN_POOLED_BOXES:
        return f"{total} boxes shown pooled, fewer than {MIN_POOLED_BOXES}"
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_week(row: Week) -> str:
    diff = row.diff_points
    within = row.within
    return (
        f"{row.label}  reviewer {_fmt(row.reviewer)} (n={row.shown})  "
        f"judge {_fmt(row.judge)} (n={row.confident})  corrected {_fmt(row.corrected)}  "
        f"diff {'n/a' if diff is None else f'{diff:+.2f} pts'}  "
        f"within {WITHIN_POINTS:g} pts: {'n/a' if within is None else 'yes' if within else 'no'}"
    )


def format_session(row: Session) -> str:
    diff = row.diff_points
    within = row.within
    return (
        f"{row.name}  {row.day.isoformat()}  reviewer {_fmt(row.reviewer)} (n={row.shown})  "
        f"judge {_fmt(row.judge)} (n={row.confident})  corrected {_fmt(row.corrected)}  "
        # round(...) + 0.0 turns a rounding residue's -0.00 into +0.00
        f"diff {'n/a' if diff is None else f'{round(diff, 2) + 0.0:+.2f} pts'}  "
        f"within {WITHIN_POINTS:g} pts: {'n/a' if within is None else 'yes' if within else 'no'}"
    )


def _model_name(value: str) -> str:
    if not _MODEL.fullmatch(value):
        raise argparse.ArgumentTypeError("not a model name")
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.tools.spotcheck_summary",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", default=DEFAULT_DIR, help="where the statistics files are")
    ap.add_argument(
        "--by-session",
        action="store_true",
        help="one row per session, each calibrated on all the other sessions",
    )
    ap.add_argument(
        "--model",
        type=_model_name,
        help="the judge model (with --by-session; default: the only one found)",
    )
    return ap


def by_session(records: Sequence[Record], model: str | None) -> int:
    """Print the per-session summary; 2 when the model is ambiguous, else 0."""
    if model is None:
        found = qualifying_models(records)
        if len(found) > 1:
            print(
                f"summary: several judge models in qualifying sessions: {', '.join(found)};"
                " choose one with --model",
                file=sys.stderr,
            )
            return 2
        model = found[0] if found else None
    if model is None:
        print("No qualifying session with a judge model.")
        rows: list[Session] = []
    else:
        print(f"Judge {model}, each session corrected with the confusion of all other sessions:")
        rows = sessions(records, model)
    for row in rows:
        print(format_session(row))
    for record in records:
        reason = skip_reason(record, model)
        if reason is not None:
            print(f"skipped {record.name}: {reason}")
    unmet = readiness(rows)
    print("ready: yes" if unmet is None else f"ready: no ({unmet})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.model is not None and not args.by_session:
        ap.error("--model needs --by-session")
    directory = Path(args.dir)
    try:
        records = load(directory)
    except SummaryError as exc:
        print(f"summary: {exc}", file=sys.stderr)
        return 1
    print(f"{len(records)} statistics file(s) in {directory}")
    if args.by_session:
        return by_session(records, args.model)
    found = sorted({r.model for r in records if r.model is not None})
    models: list[str | None] = list(found) if found else [None]
    for model in models:
        if model is None:
            print("No judge block: reviewer only.")
        else:
            print(f"Judge {model}, corrected with the confusion of earlier weeks (Rogan-Gladen):")
        rows = weeks(records, model)
        for row in rows:
            print(format_week(row))
        verdict = "yes" if condition_holds(rows) else "no"
        print(f"two weeks within {WITHIN_POINTS:g} points: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
