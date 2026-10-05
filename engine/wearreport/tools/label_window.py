"""The labelling launcher's helper (`label.cmd`, `scripts/label.ps1`): which city to label
now, when the next daylight window opens, and how many crops a session's attribute files
hold.

  python -m wearreport.tools.label_window city [--source auto|calgary|london] [--now T]
  python -m wearreport.tools.label_window count FILE [FILE ...]

`city` prints one line of JSON. With `--source auto` (the default) it is
`{"city": "calgary"}` when Calgary is in daylight now, else `{"city": "london"}` when London
is in daylight and its local time (Europe/London, BST or GMT) is in the daily window
[09:30, 15:00), else `{"city": null, "next_window": "YYYY-MM-DDTHH:MMZ", "next_city": NAME}`:
the first whole UTC minute, at or after now and within MAX_SEARCH, when either city may
start by those rules (Calgary first when both). `--source calgary` or `london` asks about that city only.
Daylight is what the spot-check tool's attribute sessions need: the sun not below
DARK_BELOW_DEG (civil twilight counts), from the engine's own solar position code
(`pilot_heights.solar_elevation`). `--now` (an ISO 8601 time with a zone, for tests) fixes
the clock.

`count` reads attribute files (their format is in the spot-check README) and prints one
line of JSON: `{"files": [{"path": P, "kept": K, "judged": J}, ...], "kept": K,
"judged": J}`, where kept counts the crops the reviewer did not reject and judged those the
model answered. A file that is missing, over MAX_FILE_BYTES or malformed is an error
(exit 1).

Exit codes: 0 done, 1 an unreadable or malformed file, 2 a usage error. Standard library
and the engine only; it never reads an image.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import zoneinfo
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from wearreport.tools import pilot_heights

# Latitude and longitude, degrees: the spot-check tool's centres of each city (the
# acceptance tests assert they are the same).
CITIES: dict[str, tuple[float, float]] = {
    "calgary": pilot_heights.CALGARY,
    "london": (51.5074, -0.1278),
}
PREFERENCE = ("calgary", "london")  # the order --source auto tries them in
SOURCES = ("auto", *PREFERENCE)
# The spot-check tool's "dark": an attribute session in the window is refused when the
# sun is below this (its LIGHT_TWILIGHT_DEG; the acceptance tests assert they are equal).
DARK_BELOW_DEG = -6.0
# London sessions must start in this daily window, London local time (BST or GMT):
# LONDON_OPENS <= time < LONDON_CLOSES.
LONDON_ZONE = zoneinfo.ZoneInfo("Europe/London")
LONDON_OPENS = datetime.time(9, 30)
LONDON_CLOSES = datetime.time(15, 0)
MAX_SEARCH = datetime.timedelta(hours=48)
MINUTE = datetime.timedelta(minutes=1)
MAX_FILE_BYTES = 1024 * 1024
WINDOW_FORMAT = "%Y-%m-%dT%H:%MZ"


class LabelWindowError(Exception):
    """An attribute file that cannot be read or is malformed."""


@dataclass(frozen=True)
class Choice:
    city: str | None  # the city to label now, or None when none is in daylight
    next_window: datetime.datetime | None = None  # when city is None
    next_city: str | None = None


@dataclass(frozen=True)
class FileCount:
    path: Path
    kept: int
    judged: int


@dataclass(frozen=True)
class Counts:
    files: tuple[FileCount, ...]

    @property
    def kept(self) -> int:
        return sum(f.kept for f in self.files)

    @property
    def judged(self) -> int:
        return sum(f.judged for f in self.files)


def in_daylight(city: str, moment: datetime.datetime) -> bool:
    """True if the sun is not below DARK_BELOW_DEG over `city` at `moment` (aware)."""
    return pilot_heights.solar_elevation(moment, *CITIES[city]) >= DARK_BELOW_DEG


def in_window(city: str, moment: datetime.datetime) -> bool:
    """True if a session in `city` may start at `moment` (aware) by the clock: for London,
    LONDON_OPENS <= London local time < LONDON_CLOSES; Calgary has no such window."""
    if city != "london":
        return True
    local = moment.astimezone(LONDON_ZONE).time()
    return LONDON_OPENS <= local < LONDON_CLOSES


def can_start(city: str, moment: datetime.datetime) -> bool:
    """True if `city` may be labelled at `moment`: in daylight and in its window."""
    return in_window(city, moment) and in_daylight(city, moment)


def next_window(
    cities: Sequence[str], now: datetime.datetime
) -> tuple[datetime.datetime, str] | None:
    """The first whole UTC minute at or after `now`, within MAX_SEARCH, with daylight in
    one of `cities` and, for London, its window (the earlier in the sequence when several),
    and that city."""
    start = now.astimezone(datetime.UTC).replace(second=0, microsecond=0)
    if start < now:
        start += MINUTE
    for i in range(int(MAX_SEARCH / MINUTE) + 1):
        moment = start + i * MINUTE
        for city in cities:
            if can_start(city, moment):
                return moment, city
    return None


def choose(source: str, now: datetime.datetime) -> Choice:
    """The city to label at `now`: with `source` "auto", Calgary if it is in daylight,
    else London if it is in daylight and its window; otherwise the forced city if it is.
    Otherwise, when the next window opens."""
    cities = PREFERENCE if source == "auto" else (source,)
    for city in cities:
        if can_start(city, now):
            return Choice(city)
    found = next_window(cities, now)
    if found is None:
        return Choice(None)
    return Choice(None, found[0], found[1])


def _crops(raw: bytes) -> list[object]:
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("not an object")
    crops = data.get("crops")
    if not isinstance(crops, list):
        raise ValueError("no crops list")
    return crops


def count_file(path: Path) -> FileCount:
    """The kept and judged crops of one attribute file. Raises LabelWindowError."""
    try:
        with path.open("rb") as fh:
            raw = fh.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise LabelWindowError(f"{path}: cannot read it ({exc.strerror})") from None
    if len(raw) > MAX_FILE_BYTES:
        raise LabelWindowError(f"{path}: larger than {MAX_FILE_BYTES} bytes")
    try:
        crops = _crops(raw)
    except (ValueError, RecursionError, UnicodeDecodeError, OverflowError) as exc:
        raise LabelWindowError(f"{path}: not an attribute file ({type(exc).__name__})") from None
    judged = 0
    for crop in crops:
        if not (isinstance(crop, list) and len(crop) == 3):
            raise LabelWindowError(f"{path}: a crop is not [height, reviewer, model]")
        model = crop[2]
        if model is not None and not isinstance(model, str):
            raise LabelWindowError(f"{path}: a crop's model answer is not text or null")
        judged += model is not None
    return FileCount(path, len(crops), judged)


def count(paths: Iterable[Path]) -> Counts:
    return Counts(tuple(count_file(path) for path in paths))


def _moment(text: str) -> datetime.datetime:
    try:
        moment = datetime.datetime.fromisoformat(text)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {text[:40]!r}") from None
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError("give a time zone, e.g. 2026-06-21T19:00Z")
    if moment.year > datetime.MAXYEAR - 1:
        raise argparse.ArgumentTypeError("too far ahead")
    return moment.astimezone(datetime.UTC)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.tools.label_window",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    city = sub.add_parser("city", help="the city to label now, or the next window")
    city.add_argument("--source", choices=SOURCES, default="auto")
    city.add_argument("--now", type=_moment, default=None, help="UTC clock (tests)")
    files = sub.add_parser("count", help="kept and judged crops of attribute files")
    files.add_argument("files", nargs="+", type=Path)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "city":
        now = args.now or datetime.datetime.now(datetime.UTC)
        choice = choose(args.source, now)
        out: dict[str, object] = {"city": choice.city}
        if choice.city is None:
            window = choice.next_window
            out["next_window"] = None if window is None else window.strftime(WINDOW_FORMAT)
            out["next_city"] = choice.next_city
        print(json.dumps(out))
        return 0
    try:
        counts = count(args.files)
    except LabelWindowError as exc:
        print(f"label_window: {exc}", file=sys.stderr)
        return 1
    files = [{"path": str(f.path), "kept": f.kept, "judged": f.judged} for f in counts.files]
    print(json.dumps({"files": files, "kept": counts.kept, "judged": counts.judged}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
