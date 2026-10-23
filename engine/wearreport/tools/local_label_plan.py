"""Offline prospective London opportunities and Calgary noon coverage.

These calendar calculations are not captures, observations, reservations, daylight
checks, or a scheduler. Nothing in this module starts the labelling launcher.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import tomllib
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
LONDON_HOURS = (10, 12, 14)


class PlanError(Exception):
    """An actionable offline input or local-environment problem."""


def _zone(value: str) -> ZoneInfo:
    if len(value) > 100 or not re.fullmatch(r"[A-Za-z0-9_+-]+(?:/[A-Za-z0-9_+-]+)*", value):
        raise argparse.ArgumentTypeError("use an explicit IANA zone, such as Europe/London")
    try:
        return ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError):
        raise argparse.ArgumentTypeError(
            "unknown IANA zone; check its spelling and the local tzdata installation"
        ) from None


def _date(value: str) -> dt.date:
    try:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError
        result = dt.date.fromisoformat(value)
        if not 1900 <= result.year <= 9998:
            raise ValueError
        return result
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError("use a valid YYYY-MM-DD date in 1900..9998") from None


def _days(value: str) -> int:
    if value not in {str(day) for day in range(1, 15)}:
        raise argparse.ArgumentTypeError("--days must be a whole number from 1 to 14")
    return int(value)


def _minutes(value: str, *, endpoint: bool = False) -> int:
    if endpoint and value == "24:00":
        return 1440
    if not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value):
        suffix = " (24:00 is also allowed)" if endpoint else ""
        raise argparse.ArgumentTypeError(f"use HH:MM in 00:00..23:59{suffix}")
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def _end(value: str) -> int:
    return _minutes(value, endpoint=True)


def _weekdays(value: str) -> tuple[str, ...]:
    names = value.lower().split(",")
    if not names or any(name not in WEEKDAYS for name in names):
        raise argparse.ArgumentTypeError(
            "--weekdays needs comma-separated mon,tue,wed,thu,fri,sat,sun"
        )
    return tuple(name for name in WEEKDAYS if name in names)


def _clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _stamp(moment: dt.datetime) -> str:
    # Historical IANA offsets can include seconds; keep the actual instant intact.
    return moment.isoformat(timespec="seconds" if moment.second else "minutes")


def _slot(
    source: dt.datetime, display: ZoneInfo, start: int, end: int, weekdays: tuple[str, ...]
) -> dict[str, object]:
    local = source.astimezone(display)
    minute = local.hour * 60 + local.minute
    available = WEEKDAYS[local.weekday()] in weekdays
    covered = available and start <= minute < end
    reason = "covered" if covered else "outside_online_window"
    if not available:
        reason = "unavailable_weekday"
    return {
        "source_local": _stamp(source),
        "display_local": _stamp(local),
        "display_date": local.date().isoformat(),
        "display_time": local.strftime("%H:%M:%S" if local.second else "%H:%M"),
        "utc": _stamp(source.astimezone(dt.UTC)),
        "covered": covered,
        "reason": reason,
    }


def build_plan(
    zone: ZoneInfo,
    start_date: dt.date,
    days: int,
    online_start: int,
    online_end: int,
    weekdays: tuple[str, ...],
) -> dict[str, object]:
    """Return opportunities within display-calendar dates; read no files or services."""
    if not 1 <= days <= 14 or not 0 <= online_start < online_end <= 1440:
        raise PlanError("use 1..14 days and an online start earlier than the exclusive end")
    if not weekdays or any(day not in WEEKDAYS for day in weekdays):
        raise PlanError("choose at least one available weekday from mon..sun")
    london: list[dict[str, object]] = []
    calgary: list[dict[str, object]] = []
    try:
        end_date = start_date + dt.timedelta(days=days)
        # Include neighbouring source dates: a source's noon can fall on a different
        # display date. Filtering always uses the converted display date and weekday.
        source_start = start_date - dt.timedelta(days=2)
        for offset in range(days + 4):
            day = source_start + dt.timedelta(days=offset)
            for source_zone, hours, target in (
                ("Europe/London", LONDON_HOURS, london),
                ("America/Edmonton", (12,), calgary),
            ):
                for hour in hours:
                    moment = dt.datetime.combine(day, dt.time(hour), ZoneInfo(source_zone))
                    display_day = moment.astimezone(zone).date()
                    if start_date <= display_day < end_date:
                        row = _slot(moment, zone, online_start, online_end, weekdays)
                        if target is calgary or row["covered"]:
                            target.append(row)
    except (ValueError, OverflowError, ZoneInfoNotFoundError):
        raise PlanError(
            "date range cannot be converted; use 1900..9998 and installed tzdata"
        ) from None
    return {
        "kind": "prospective_opportunities",
        "display_zone": zone.key,
        "start_date": start_date.isoformat(),
        "days": days,
        "online_start": _clock(online_start),
        "online_end_exclusive": _clock(online_end),
        "available_weekdays": list(weekdays),
        "london_source_zone": "Europe/London",
        "london_opportunities": london,
        "calgary_source_zone": "America/Edmonton",
        "calgary_noon_coverage": calgary,
        "notice": "Prospective opportunities only; a human must explicitly start any session.",
    }


def preflight(zone: ZoneInfo) -> dict[str, object]:
    """Inspect local version metadata and the lock; never import models or providers."""
    root = Path(__file__).resolve().parents[3]
    if sys.version_info[:2] != (3, 12):
        raise PlanError(
            "Python 3.12 is required; select the prepared repository Python with -Python"
        )
    try:
        lock = (root / "uv.lock").read_bytes()
        if len(lock) > 512 * 1024:
            raise ValueError
        packages = tomllib.loads(lock.decode("utf-8"))["package"]
        locked = {p["name"]: p["version"] for p in packages if "version" in p}
        versions = {}
        # The Windows locked setup omits llama-cpp-python; human-only entry uses no judge.
        for package in ("wearreport", "numpy", "onnxruntime", "opencv-python-headless", "tzdata"):
            version = metadata.version(package)
            if version != locked.get(package):
                raise PlanError(
                    f"{package} differs from uv.lock; repeat the documented local setup"
                )
            versions[package] = version
    except metadata.PackageNotFoundError:
        raise PlanError(
            "local dependencies are missing; complete the documented one-time setup"
        ) from None
    except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise PlanError(
            "cannot inspect uv.lock; restore the repository lock and prepared environment"
        ) from None
    return {
        "kind": "offline_preflight",
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "module": str(Path(__file__).resolve()),
        "lock": str(root / "uv.lock"),
        "display_zone": zone.key,
        "versions": versions,
        "notice": "Local files and versions only; no daylight, provider, or capture check was run.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--zone", type=_zone, required=True, help="explicit IANA display timezone")
    parser.add_argument(
        "--start-date", type=_date, help="first display date; default: today in --zone"
    )
    parser.add_argument(
        "--days", type=_days, default=7, help="1..14 display-calendar days (default: 7)"
    )
    parser.add_argument(
        "--online-start", type=_minutes, default=0, help="inclusive HH:MM (default: 00:00)"
    )
    parser.add_argument(
        "--online-end", type=_end, default=1440, help="exclusive HH:MM, including 24:00"
    )
    parser.add_argument(
        "--weekdays", type=_weekdays, default=WEEKDAYS, help="comma-separated mon..sun"
    )
    parser.add_argument(
        "--preflight", action="store_true", help="inspect local version metadata only"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error("unknown options; use --help for the supported offline arguments")
    try:
        if args.online_start >= args.online_end:
            raise PlanError("--online-start must be earlier than --online-end; end is exclusive")
        if args.preflight:
            result = preflight(args.zone)
        else:
            day = args.start_date or dt.datetime.now(args.zone).date()
            result = build_plan(
                args.zone, day, args.days, args.online_start, args.online_end, args.weekdays
            )
    except PlanError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
