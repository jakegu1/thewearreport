"""Summarise the spot-check statistics files per ISO week, reviewer against hosted judge.

  python -m wearreport.tools.spotcheck_summary [--dir spotchecks]

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
reviewer only. A malformed file is an error that names it.

  python -m wearreport.tools.spotcheck_summary --heights [--dir spotchecks] [--data-dir PATH]

reads instead every per-box file, `<dir>/boxes/*.json` (box heights and labels only), and
chooses a near-field threshold: the smallest box height H whose boxes (those at least H
pixels tall) reach TARGET_PRECISION, with a Wilson 95% lower bound of at least
MIN_LOWER_BOUND, over at least MIN_BOXES_ABOVE judged boxes, and whose judged share is at
least MIN_JUDGED_SHARE. Precision is the boxes labelled person or in_vehicle over the
judged ones (person, in_vehicle, not_person); unsure boxes are left out of it and counted.
The judged share is the judged boxes over the judged and unsure ones: a box nobody can
verify is not counted. It prints every candidate height and its judged share, the threshold
(or none), the precision at it per light and, with `--data-dir` (a checkout of the data
branch), per rain condition, and whether the baseline is complete (see `coverage`).

A session is "rain" when the published sweep record nearest its `started_at`, within
RAIN_JOIN_MINUTES, has `weather.precip_mm` > 0, "dry" when it is 0, and "unknown"
otherwise (no such record, or no weather in it). Only the records within that window are
read; a malformed one is an error that names it.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

DEFAULT_DIR = "spotchecks"
MAX_FILE_BYTES = 1 << 20
MAX_COUNT = 10**9
REVIEWER_LABELS = ("person", "in_vehicle", "not_person")
OTHER_REVIEWER_LABELS = ("unsure",)  # allowed, counted for neither side
JUDGE_ANSWERS = ("person", "in_vehicle", "not_person", "unsure")
POSITIVE = ("person", "in_vehicle")
WITHIN_POINTS = 3.0
WEEKS_NEEDED = 2
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MODEL = re.compile(r"[A-Za-z0-9][\w.-]{0,63}")

# The height summary (--heights).
BOXES_DIR = "boxes"
BOX_FIELDS = frozenset({"date", "started_at", "light", "frames", "detector", "boxes"})
BOX_LABELS = ("person", "in_vehicle", "not_person", "unsure")
JUDGED_LABELS = ("person", "in_vehicle", "not_person")
LIGHTS = ("day", "twilight", "dark")
RAIN_CONDITIONS = ("rain", "dry", "unknown")
MAX_HEIGHT = 100_000  # pixels; no camera frame is this tall
TARGET_PRECISION = 0.90
MIN_LOWER_BOUND = 0.85
MIN_BOXES_ABOVE = 100
MIN_JUDGED_SHARE = 0.80
WILSON_Z = NormalDist().inv_cdf(0.975)  # a two-sided 95% interval
# The baseline is complete with BASELINE_BOXES judged boxes from BASELINE_SESSIONS
# sessions on BASELINE_DATES dates, and MIN_CONDITION_BOXES judged boxes in each of
# BASELINE_CONDITIONS (dark is reported, not required).
BASELINE_BOXES = 300
BASELINE_SESSIONS = 3
BASELINE_DATES = 2
MIN_CONDITION_BOXES = 50
BASELINE_CONDITIONS = ("day", "twilight", "rain")
RAIN_JOIN_MINUTES = 30
MAX_RECORD_BYTES = 8 << 20
SWEEPS_DIR = "sweeps"
EPSILON = 1e-12  # a ratio this close to a bar meets it
_STARTED_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z")
_UTC_SECONDS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_SWEEP_FILE = re.compile(r"(\d{8}T\d{4}Z)\.json")

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
    return Record(datetime.date.fromisoformat(day), shown, not_person, model, confusion)


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
            records.append(parse_record(raw))
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
        if self.corrected is None or self.reviewer is None:
            return None
        return 100 * (self.corrected - self.reviewer)

    @property
    def within(self) -> bool | None:
        diff = self.diff_points
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


# The height summary --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Session:
    """What the height summary reads from one per-box file, and its rain condition."""

    day: datetime.date
    started_at: datetime.datetime
    light: str
    boxes: tuple[tuple[int, str], ...]
    rain: str = "unknown"


def _utc(text: str, pattern: re.Pattern[str], layout: str) -> datetime.datetime:
    if not pattern.fullmatch(text):
        raise ValueError("not a UTC time")
    return datetime.datetime.strptime(text, layout).replace(tzinfo=datetime.UTC)


def _box(value: object) -> tuple[int, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("a box is not [height, label]")
    height, label = value
    if type(height) is not int or not 0 <= height <= MAX_HEIGHT:
        raise ValueError("a box height is not a whole number of pixels")
    if label not in BOX_LABELS:
        raise ValueError("a box label is not one of " + ", ".join(BOX_LABELS))
    return height, label


def parse_session(raw: bytes) -> Session:
    """One per-box file; raises ValueError (or a subclass) when it is malformed."""
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"larger than {MAX_FILE_BYTES} bytes")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("not an object")
    if set(data) != BOX_FIELDS:
        raise ValueError("its fields are not " + ", ".join(sorted(BOX_FIELDS)))
    day = data["date"]
    if not isinstance(day, str) or not _DATE.fullmatch(day):
        raise ValueError("date is not YYYY-MM-DD")
    started = data["started_at"]
    if not isinstance(started, str):
        raise ValueError("started_at is not a UTC time")
    light = data["light"]
    if light not in LIGHTS:
        raise ValueError("light is not one of " + ", ".join(LIGHTS))
    _count(data["frames"], "frames")
    if not isinstance(data["detector"], dict):
        raise ValueError("detector is not an object")
    boxes = data["boxes"]
    if not isinstance(boxes, list):
        raise ValueError("boxes is not a list")
    return Session(
        day=datetime.date.fromisoformat(day),
        started_at=_utc(started, _STARTED_AT, "%Y-%m-%dT%H:%MZ"),
        light=light,
        boxes=tuple(_box(box) for box in boxes),
    )


def _read(path: Path, limit: int) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(limit + 1)


def load_sessions(directory: Path) -> list[Session]:
    """Every per-box file in `directory`/boxes, in name order."""
    boxes = directory / BOXES_DIR
    try:
        paths = sorted(p for p in boxes.iterdir() if p.suffix == ".json" and p.is_file())
    except OSError as exc:
        raise SummaryError(f"cannot read {boxes}: {exc.strerror}") from None
    sessions = []
    for path in paths:
        try:
            sessions.append(parse_session(_read(path, MAX_FILE_BYTES)))
        except OSError as exc:
            raise SummaryError(f"cannot read {path.name}: {exc.strerror}") from None
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
            reason = str(exc) if type(exc) is ValueError else type(exc).__name__
            raise SummaryError(f"{path.name} is not a per-box file ({reason})") from None
    return sessions


def parse_sweep(raw: bytes) -> tuple[datetime.datetime, float | None]:
    """A published sweep record's start and precipitation (None without weather);
    raises ValueError (or a subclass) when it is malformed."""
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError(f"larger than {MAX_RECORD_BYTES} bytes")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("not an object")
    started = data.get("started_at")
    if not isinstance(started, str):
        raise ValueError("started_at is not a UTC time")
    moment = _utc(started, _UTC_SECONDS, "%Y-%m-%dT%H:%M:%SZ")
    if "weather" not in data:
        raise ValueError("no weather field")
    weather = data["weather"]
    if weather is None:
        return moment, None
    if not isinstance(weather, dict):
        raise ValueError("weather is not an object")
    precip = weather.get("precip_mm")
    if isinstance(precip, bool) or not isinstance(precip, int | float):
        raise ValueError("weather.precip_mm is not a number")
    if not math.isfinite(precip) or precip < 0:
        raise ValueError("weather.precip_mm is not a precipitation")
    return moment, float(precip)


def _sweeps_near(
    data_dir: Path, moment: datetime.datetime
) -> Iterable[tuple[Path, datetime.datetime]]:
    """The record files whose sweep id (the start, to the minute) lies within
    RAIN_JOIN_MINUTES of `moment` (itself to the minute)."""
    window = datetime.timedelta(minutes=RAIN_JOIN_MINUTES)
    first, last = moment - window, moment + window
    for day in sorted({first.date(), last.date()}):
        folder = data_dir / SWEEPS_DIR / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
        try:
            names = sorted(os.listdir(folder))
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            raise SummaryError(f"cannot read {folder}: {exc.strerror}") from None
        for name in names:
            match = _SWEEP_FILE.fullmatch(name)
            if match is None:
                continue
            try:
                sweep_id = datetime.datetime.strptime(match.group(1), "%Y%m%dT%H%MZ")
            except ValueError:
                continue  # not a sweep id
            sweep_id = sweep_id.replace(tzinfo=datetime.UTC)
            if first <= sweep_id <= last:
                yield folder / name, sweep_id


def rain_condition(data_dir: Path, started_at: datetime.datetime) -> str:
    """rain, dry or unknown: from the published record nearest `started_at`, within
    RAIN_JOIN_MINUTES (the earlier on a tie)."""
    window = datetime.timedelta(minutes=RAIN_JOIN_MINUTES)
    nearest: tuple[datetime.timedelta, datetime.datetime, float | None] | None = None
    for path, _sweep_id in _sweeps_near(data_dir, started_at):
        try:
            moment, precip = parse_sweep(_read(path, MAX_RECORD_BYTES))
        except OSError as exc:
            raise SummaryError(f"cannot read {path.name}: {exc.strerror}") from None
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
            reason = str(exc) if type(exc) is ValueError else type(exc).__name__
            raise SummaryError(f"{path.name} is not a sweep record ({reason})") from None
        distance = abs(moment - started_at)
        if distance <= window and (nearest is None or (distance, moment) < nearest[:2]):
            nearest = (distance, moment, precip)
    if nearest is None or nearest[2] is None:
        return "unknown"
    return "rain" if nearest[2] > 0 else "dry"


def with_rain(sessions: Sequence[Session], data_dir: Path) -> list[Session]:
    if not data_dir.is_dir():
        raise SummaryError(f"cannot read {data_dir}: not a directory")
    return [dataclasses.replace(s, rain=rain_condition(data_dir, s.started_at)) for s in sessions]


@dataclass(frozen=True, slots=True)
class Tally:
    """Judged boxes, those of them labelled a person (on foot or in a vehicle), and the
    unsure boxes."""

    judged: int
    positive: int
    unsure: int = 0

    @property
    def precision(self) -> float | None:
        return self.positive / self.judged if self.judged else None

    @property
    def judged_share(self) -> float | None:
        total = self.judged + self.unsure
        return self.judged / total if total else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson(self.positive, self.judged) if self.judged else None


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """The Wilson score interval for k successes in n trials (n > 0)."""
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def tally(boxes: Iterable[tuple[int, str]], min_height: int = 0) -> Tally:
    labels = [label for height, label in boxes if height >= min_height]
    judged = sum(1 for label in labels if label in JUDGED_LABELS)
    positive = sum(1 for label in labels if label in ("person", "in_vehicle"))
    unsure = sum(1 for label in labels if label == "unsure")
    return Tally(judged, positive, unsure)


def qualifies(t: Tally) -> bool:
    """At least MIN_BOXES_ABOVE judged boxes, TARGET_PRECISION, MIN_LOWER_BOUND and
    MIN_JUDGED_SHARE."""
    precision, interval, share = t.precision, t.interval, t.judged_share
    if precision is None or interval is None or share is None or t.judged < MIN_BOXES_ABOVE:
        return False
    return (
        precision >= TARGET_PRECISION - EPSILON
        and interval[0] >= MIN_LOWER_BOUND - EPSILON
        and share >= MIN_JUDGED_SHARE - EPSILON
    )


def _all_boxes(sessions: Sequence[Session]) -> list[tuple[int, str]]:
    return [box for s in sessions for box in s.boxes]


def candidate_heights(sessions: Sequence[Session]) -> list[int]:
    return sorted({height for height, _label in _all_boxes(sessions)})


def threshold(sessions: Sequence[Session]) -> int | None:
    """The smallest candidate height whose boxes qualify, or None."""
    boxes = _all_boxes(sessions)
    for height in candidate_heights(sessions):
        if qualifies(tally(boxes, height)):
            return height
    return None


def _stat(t: Tally) -> str:
    interval = t.interval
    if interval is None:
        return f"n={t.judged} precision n/a wilson n/a"
    return (
        f"n={t.judged} precision {_fmt(t.precision)} wilson [{interval[0]:.4f}, {interval[1]:.4f}]"
    )


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def coverage(sessions: Sequence[Session]) -> tuple[str, list[str]]:
    """The coverage line, and what the baseline still lacks (nothing: complete)."""
    judged = tally(_all_boxes(sessions)).judged
    by_light = {
        light: tally(box for s in sessions if s.light == light for box in s.boxes).judged
        for light in LIGHTS
    }
    by_rain = {
        rain: tally(box for s in sessions if s.rain == rain for box in s.boxes).judged
        for rain in RAIN_CONDITIONS
    }
    dates = len({s.day for s in sessions})
    missing = []
    if judged < BASELINE_BOXES:
        missing.append(f"judged boxes {judged} < {BASELINE_BOXES}")
    if len(sessions) < BASELINE_SESSIONS:
        missing.append(f"sessions {len(sessions)} < {BASELINE_SESSIONS}")
    if dates < BASELINE_DATES:
        missing.append(f"dates {dates} < {BASELINE_DATES}")
    counts = by_light | by_rain
    for condition in BASELINE_CONDITIONS:
        if counts[condition] < MIN_CONDITION_BOXES:
            missing.append(f"{condition} {counts[condition]} < {MIN_CONDITION_BOXES}")
    lights = ", ".join(f"{light} {n}" for light, n in by_light.items())
    rains = ", ".join(f"{rain} {n}" for rain, n in by_rain.items())
    line = (
        f"coverage: {_plural(len(sessions), 'session')}, {_plural(dates, 'date')}, "
        f"{judged} judged boxes ({lights}; {rains}); "
        f"baseline complete: {'no (' + '; '.join(missing) + ')' if missing else 'yes'}"
    )
    return line, missing


def heights_report(sessions: Sequence[Session], directory: Path, rain: bool) -> list[str]:
    """The lines `--heights` prints. `rain`: whether the rain conditions are known."""
    boxes = _all_boxes(sessions)
    positives = tally(boxes).positive
    unsure = sum(1 for _height, label in boxes if label == "unsure")
    lines = [
        f"{len(sessions)} per-box file(s) in {directory / BOXES_DIR}",
        f"unsure boxes (left out): {unsure}",
    ]
    heights = candidate_heights(sessions)
    for height in heights:
        above = tally(boxes, height)
        kept = _fmt(above.positive / positives if positives else None)
        lines.append(f">= {height} px: {_stat(above)} kept {kept}")
    for height in heights:
        above = tally(boxes, height)
        lines.append(
            f"judged share >= {height} px: {above.judged} of {above.judged + above.unsure} "
            f"({_fmt(above.judged_share)})"
        )
    chosen = threshold(sessions)
    lines.append(
        f"threshold rule: precision >= {TARGET_PRECISION:.2f}, Wilson lower bound >= "
        f"{MIN_LOWER_BOUND:.2f}, at least {MIN_BOXES_ABOVE} judged boxes, "
        f"judged share >= {MIN_JUDGED_SHARE:.2f}"
    )
    lines.append(f"threshold: {'none' if chosen is None else f'{chosen} px'}")
    lines.append("over all boxes:" if chosen is None else f"at {chosen} px:")
    floor = chosen or 0
    for light in LIGHTS:
        t = tally((box for s in sessions if s.light == light for box in s.boxes), floor)
        lines.append(f"  light {light}: {_stat(t)}")
    if rain:
        for condition in RAIN_CONDITIONS:
            t = tally((box for s in sessions if s.rain == condition for box in s.boxes), floor)
            lines.append(f"  rain {condition}: {_stat(t)}")
    lines.append(coverage(sessions)[0])
    return lines


def _heights_main(directory: Path, data_dir: Path | None) -> int:
    try:
        sessions = load_sessions(directory)
        if data_dir is not None:
            sessions = with_rain(sessions, data_dir)
    except SummaryError as exc:
        print(f"summary: {exc}", file=sys.stderr)
        return 1
    for line in heights_report(sessions, directory, data_dir is not None):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.tools.spotcheck_summary",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", default=DEFAULT_DIR, help="where the statistics files are")
    ap.add_argument(
        "--heights",
        action="store_true",
        help="summarise the per-box files in DIR/boxes and choose a near-field threshold",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="with --heights: a checkout of the data branch, for the rain condition",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    directory = Path(args.dir)
    if args.heights:
        return _heights_main(directory, None if args.data_dir is None else Path(args.data_dir))
    if args.data_dir is not None:
        parser.error("--data-dir needs --heights")
    try:
        records = load(directory)
    except SummaryError as exc:
        print(f"summary: {exc}", file=sys.stderr)
        return 1
    print(f"{len(records)} statistics file(s) in {directory}")
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
