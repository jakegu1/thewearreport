"""Publish sweep records to a checkout of the `data` branch, append-only, and status.json.

Layout of the data directory:

    sweeps/YYYY/MM/DD/<sweep_id>.json   one record per sweep, one JSON line, immutable
    status.json                         health summary, rewritten after every publish

A record file is created once and never rewritten or deleted: it is written to a
temporary file in the same directory and then hard-linked into place, which fails if the
name already exists, so a record is either absent or complete, and two publishers can
never both create it. Publishing the same record again changes nothing; a different
record with the same sweep_id raises ConflictError and writes nothing. status.json is
replaced atomically (write, then rename). A publisher killed between writing a temporary
file and moving it into place leaves that file behind; the next publish removes such
files from the data directory and from the record's day directory before it writes. The
whole publish runs under an exclusive `flock` on the data directory itself, so concurrent
publishers take turns and no lock file is left behind.

status.json is computed from the record files alone (see `compute_status`). Record files
are external data when read back: each is size-capped, parsed and checked with
`aggregate.check_record`; one that fails is counted in `records_invalid` and left out.
Nothing here commits or pushes; T-007's workflow does that.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import logging
import os
import re
import secrets
import stat
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

from wearreport import aggregate, weather
from wearreport.aggregate import Record, RecordError

logger = logging.getLogger("wearreport.publish")

SWEEPS_DIR: Final = "sweeps"
STATUS_FILE: Final = "status.json"
STATUS_SCHEMA: Final = "status.v1"
# A record with 2000 cameras is about 110 kB; anything larger is not one of ours.
MAX_RECORD_BYTES: Final = 1024 * 1024
WINDOW: Final = timedelta(hours=24)
# A sweep succeeds when at least 90% of the cameras listed gave a usable frame.
SUCCESS_NUMERATOR, SUCCESS_DENOMINATOR = 9, 10
SUCCESS_RULE: Final = "frames_ok >= 90% of cameras_listed, and cameras_listed > 0"
MEDIAN_RULE: Final = "median persons_total over successful sweeps started in London daytime"
LONDON_TZ: Final = "Europe/London"
DAY_START, DAY_END = time(7, 0), time(21, 0)  # London local time, end excluded
STATUS_ATTRIBUTION: Final = (aggregate.TFL_ATTRIBUTION, weather.METOFFICE_ATTRIBUTION)

YEAR, MONTH, DAY = re.compile(r"[0-9]{4}"), re.compile(r"[0-9]{2}"), re.compile(r"[0-9]{2}")
RECORD_FILE = re.compile(rf"({aggregate.SWEEP_ID.pattern})\.json")
# The name `_temporary` gives: .<target name>.<16 hex digits>.tmp
TEMPORARY_FILE = re.compile(r"\..+\.[0-9a-f]{16}\.tmp")


class PublishError(RuntimeError):
    """The data directory cannot be written as the append-only layout requires."""


class ConflictError(PublishError):
    """A different record with the same sweep_id is already published."""


@dataclass(frozen=True, slots=True)
class Published:
    record_path: Path
    status_path: Path
    created: bool  # False when the identical record was already there


def serialize(record: Record) -> bytes:
    """The canonical bytes of a record: sorted keys, no spaces, ASCII, one line."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return text.encode("ascii") + b"\n"


def record_path(data_dir: Path, sweep_id: str) -> Path:
    """sweeps/YYYY/MM/DD/<sweep_id>.json under `data_dir` (the UTC date of the sweep)."""
    if not aggregate.SWEEP_ID.fullmatch(sweep_id):
        raise RecordError("sweep_id is malformed")
    return (
        data_dir / SWEEPS_DIR / sweep_id[0:4] / sweep_id[4:6] / sweep_id[6:8] / f"{sweep_id}.json"
    )


def is_success(record: Record) -> bool:
    """At least 90% of the cameras listed gave a usable frame (and at least one was listed)."""
    listed, ok = record["cameras_listed"], record["frames_ok"]
    return bool(listed > 0 and ok * SUCCESS_DENOMINATOR >= listed * SUCCESS_NUMERATOR)


# Publishing ----------------------------------------------------------------------------


def publish(data_dir: Path, record: Record, *, now: datetime) -> Published:
    """Add `record` to the data directory and rewrite status.json as of `now`.

    Raises RecordError for an invalid record and PublishError (ConflictError for another
    record under the same sweep_id) when it cannot be published; in both cases nothing is
    written.
    """
    aggregate.check_record(record)
    data = serialize(record)
    if len(data) > MAX_RECORD_BYTES:
        raise PublishError(f"record is larger than {MAX_RECORD_BYTES} bytes")
    root = _data_root(data_dir)
    path = record_path(root, record["sweep_id"])
    with _locked(root):
        _remove_stale_temporaries(root)
        _remove_stale_temporaries(path.parent)
        existing = _read_existing(path)
        if existing is None:
            _make_dirs(root, path.parent)
            created = _create_once(path, data)
        elif existing == data:
            created = False
        else:
            raise ConflictError(f"a different record {record['sweep_id']} is already published")
        status = compute_status(root, now=now)
        status_path = root / STATUS_FILE
        _replace(status_path, serialize_status(status))
    logger.info(
        "sweep published",
        extra={"sweep_id": record["sweep_id"], "record_created": created, "status": status},
    )
    return Published(path, status_path, created)


def serialize_status(status: dict[str, Any]) -> bytes:
    text = json.dumps(status, indent=2, allow_nan=False)
    return text.encode("ascii") + b"\n"


def _data_root(data_dir: Path) -> Path:
    try:
        st = os.stat(data_dir)
    except OSError as exc:
        raise PublishError(f"data directory is not usable: {exc.strerror}") from None
    if not stat.S_ISDIR(st.st_mode):
        raise PublishError("data directory is not a directory")
    return data_dir


@contextlib.contextmanager
def _locked(root: Path) -> Iterator[None]:
    """An exclusive flock on the data directory; released when the block exits."""
    try:
        fd = os.open(root, os.O_RDONLY)
    except OSError as exc:
        raise PublishError(f"cannot open the data directory: {exc.strerror}") from None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise PublishError(f"cannot lock the data directory: {exc.strerror}") from None
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def _is_real_dir(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(st.st_mode)


def _make_dirs(root: Path, directory: Path) -> None:
    """Create `directory` under `root`, refusing any component that is a symlink or a file:
    a checkout could otherwise redirect writes outside the data directory."""
    current = root
    for part in directory.relative_to(root).parts:
        current = current / part
        try:
            os.mkdir(current)
        except FileExistsError:
            pass
        except OSError as exc:
            raise PublishError(f"cannot create {part}: {exc.strerror}") from None
        if not _is_real_dir(current):
            raise PublishError(f"{current.relative_to(root)} is not a plain directory")


def _read_existing(path: Path) -> bytes | None:
    """The bytes of an already published record, or None if there is none."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PublishError(f"cannot inspect an existing record: {exc.strerror}") from None
    if not stat.S_ISREG(st.st_mode):
        raise PublishError("an existing record path is not a regular file")
    if not all(_is_real_dir(d) for d in path.parents[:4]):  # sweeps/YYYY/MM/DD
        raise PublishError("an existing record is reached through a symlink")
    try:
        with open(path, "rb") as fh:
            return fh.read(MAX_RECORD_BYTES + 1)
    except OSError as exc:
        raise PublishError(f"cannot read an existing record: {exc.strerror}") from None


def _temporary(directory: Path, name: str) -> Path:
    return directory / f".{name}.{secrets.token_hex(8)}.tmp"


def _remove_stale_temporaries(directory: Path) -> None:
    """Delete temporary files a killed publisher left in `directory`. Called under the lock,
    so no other publisher on this machine has one in flight. Only plain files named as
    `_temporary` names them are removed; a missing directory has none."""
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PublishError(f"cannot list {directory.name}: {exc.strerror}") from None
    for name in names:
        if not TEMPORARY_FILE.fullmatch(name):
            continue
        stale = directory / name
        try:
            if stat.S_ISREG(os.lstat(stale).st_mode):
                os.unlink(stale)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PublishError(f"cannot remove a stale temporary file: {exc.strerror}") from None
        logger.warning("stale temporary file removed", extra={"file": name})


def _write_new(path: Path, data: bytes) -> None:
    with open(path, "x", encoding="ascii", newline="") as fh:
        fh.write(data.decode("ascii"))
        fh.flush()
        os.fsync(fh.fileno())


def _create_once(path: Path, data: bytes) -> bool:
    """Create `path` holding `data` unless it exists; True if this call created it."""
    tmp = _temporary(path.parent, path.name)
    try:
        _write_new(tmp, data)
        try:
            os.link(tmp, path)
        except FileExistsError:  # another publisher won the race
            if _read_existing(path) != data:
                raise ConflictError("a different record is already published") from None
            return False
        return True
    except OSError as exc:
        raise PublishError(f"cannot write the record: {exc.strerror}") from None
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def _replace(path: Path, data: bytes) -> None:
    tmp = _temporary(path.parent, path.name)
    try:
        _write_new(tmp, data)
        os.replace(tmp, path)
    except OSError as exc:
        raise PublishError(f"cannot write {path.name}: {exc.strerror}") from None
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


# Reading records back ------------------------------------------------------------------


def load_record(path: Path) -> Record:
    """Read and check one record file. Raises RecordError for anything but a valid record
    stored under its own sweep_id."""
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode):
            raise RecordError("record path is not a regular file")
        with open(path, "rb") as fh:
            data = fh.read(MAX_RECORD_BYTES + 1)
    except OSError:
        raise RecordError("record file cannot be read") from None
    if len(data) > MAX_RECORD_BYTES:
        raise RecordError("record file is too large")
    try:
        record = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):  # includes UnicodeDecodeError
        raise RecordError("record file is not JSON") from None
    aggregate.check_record(record)
    sweep_id: str = record["sweep_id"]
    day = (sweep_id[0:4], sweep_id[4:6], sweep_id[6:8])
    if path.name != f"{sweep_id}.json" or path.parent.parts[-3:] != day:
        raise RecordError("record file is not stored under its sweep_id")
    checked: Record = record
    return checked


def _subdirs(directory: Path, pattern: re.Pattern[str]) -> list[Path]:
    """Plain subdirectories whose names match `pattern`, newest (largest) first."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return [
        directory / name
        for name in sorted(names, reverse=True)
        if pattern.fullmatch(name) and _is_real_dir(directory / name)
    ]


def record_files(data_dir: Path) -> Iterator[Path]:
    """Every file named like a record under sweeps/, newest sweep_id first. Symlinked
    directories are not followed."""
    sweeps = data_dir / SWEEPS_DIR
    if not _is_real_dir(sweeps):
        return
    for year in _subdirs(sweeps, YEAR):
        for month in _subdirs(year, MONTH):
            for day in _subdirs(month, DAY):
                with contextlib.suppress(OSError):
                    names = sorted(os.listdir(day), reverse=True)
                    yield from (day / n for n in names if RECORD_FILE.fullmatch(n))


# status.json ---------------------------------------------------------------------------


@functools.cache
def _london() -> ZoneInfo:
    return ZoneInfo(LONDON_TZ)


def is_london_daytime(moment: datetime) -> bool:
    """07:00 <= London local time < 21:00, BST or GMT as in force at `moment`."""
    try:
        local = moment.astimezone(_london()).time()
    except OverflowError:  # a moment at the very edge of the datetime range
        return False
    return DAY_START <= local < DAY_END


def compute_status(data_dir: Path, *, now: datetime) -> dict[str, Any]:
    """The health summary as of `now`, from the record files alone.

    - last sweep: the newest record that started at or before `now`;
    - the 24 h window is (now - 24 h, now], by started_at;
    - success rate: successful sweeps / sweeps in the window (null when there are none);
    - daytime sweeps: the window's sweeps that started in London daytime, 07:00 to 21:00
      local time, successful or not;
    - median persons_total over the daytime sweeps that succeeded (null when none did):
      a failed sweep saw too few cameras for its count to stand for the street;
    - consecutive failures: failed sweeps since the newest success, however far back.

    Records that start after `now` are ignored. Reading stops once it is past the window
    and has found a success, so the cost does not grow with the history.
    """
    now = now.astimezone(UTC)
    window_start = now - WINDOW
    last: Record | None = None
    in_window: list[Record] = []
    consecutive = invalid = 0
    streak_open = True
    for path in record_files(data_dir):
        try:
            record = load_record(path)
        except RecordError as exc:
            invalid += 1
            logger.warning(
                "invalid record file left out of status",
                extra={"file": path.name, "error": str(exc)},
            )
            continue
        started = aggregate.parse_utc(record["started_at"])
        if started > now:
            continue
        if last is None:
            last = record
        if streak_open:
            if is_success(record):
                streak_open = False
            else:
                consecutive += 1
        if started > window_start:
            in_window.append(record)
        elif not streak_open:
            break
    successes = sum(is_success(r) for r in in_window)
    daytime = [r for r in in_window if is_london_daytime(aggregate.parse_utc(r["started_at"]))]
    # A failed sweep's persons_total counts what little was seen, not the street: leave
    # it out of the median rather than report an outage as an empty London (INV-6).
    persons = [r["persons_total"] for r in daytime if is_success(r)]
    return {
        "schema": STATUS_SCHEMA,
        "generated_at": aggregate.format_utc(now),
        "last_sweep_id": None if last is None else last["sweep_id"],
        "last_sweep_at": None if last is None else last["started_at"],
        "sweeps_24h": len(in_window),
        "successful_sweeps_24h": successes,
        "success_rate_24h": successes / len(in_window) if in_window else None,
        "success_rule": SUCCESS_RULE,
        "daytime_sweeps_24h": len(daytime),
        "median_persons_daytime_24h": statistics.median(persons) if persons else None,
        "median_rule": MEDIAN_RULE,
        "daytime": f"{DAY_START:%H:%M}-{DAY_END:%H:%M} {LONDON_TZ}",
        "consecutive_failures": consecutive,
        "records_invalid": invalid,
        "attribution": list(STATUS_ATTRIBUTION),
    }
