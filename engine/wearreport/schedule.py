"""The scheduled sweep's helpers: daytime gate, sweep outcome, staged-path check, the
artifact check, the streak search on the data branch, alert.

Run by `.github/workflows/sweep.yml` as `python3 -m wearreport.schedule COMMAND`. The gate,
publish and alert jobs install no dependencies, so importing this module needs the
standard library only. `streak-closed` and `streak-days` check records exactly as the
publisher does, with `wearreport.publish`, which they import when they run: the sweep
job runs them with the engine's virtual environment.

  gate [--now TIME] [--event E --ref R --default-branch B]
                              print `open=true` in London daytime (07:00-21:00), else
                              `open=false` (a GITHUB_OUTPUT line); exit 0. With the
                              event, `open=false` too for a manual run from a ref other
                              than the default branch
  outcome --status PATH       print `outcome=success|failure` and `consecutive_failures=N`
                              from status.json; exit 1 unless the newest sweep succeeded
  check-staged                read `git diff --cached --name-status` on stdin; exit 1
                              unless it only adds records and adds or updates status.json
  check-artifact --dir A      exit 0 and print its paths when the downloaded artifact in A
                              holds only status.json and records a check-staged would
                              accept, each a regular file, JSON, at most MAX_RECORD_BYTES
                              and stored under its own sweep_id; else exit 1
  copy-artifact --artifact A --data-dir D
                              check A as check-artifact does, then copy it into the data
                              branch checkout D (an existing record must be identical)
  streak-closed --data-dir D [--now TIME]
                              exit 0 when the checked-out data branch in D shows where the
                              failure streak ends (a record the publisher counts as a
                              success, or a status.json whose newest sweep succeeded),
                              else exit 1
  streak-days --data-dir D --before sweeps/YYYY/MM/DD [--now TIME] [--max-days N]
                              print the sparse-checkout patterns of the older days that
                              the checkout in D must add to show where the streak ends,
                              newest first; each day's records are read once, from git
  alert --gate-result R --sweep-result R [--publish-result R] [--stage S]
        [--publish-stage S] [--record-failures N]
                              open, comment on or close the `ops-alert` issue

The alert counts failed sweeps: this run's (from the gate, sweep and publish job
results) plus the unbroken run of failures before it, read from the workflow's run
history in the order the runs were created. Runs whose gate closed, or that were
cancelled before they started, neither count nor break the run; nor do runs whose sweep
was skipped as too soon after the previous one (the sweep job succeeded and the publish
job was skipped; this run's stage is `skipped`), and a skip never closes the issue. The
count is at least status.json's `consecutive_failures` (failed records), since a sweep
that exits 1 writes no record and one that publishes a failed record exits 0, and the
workflow fails a job for both. At ALERT_THRESHOLD
failures in a row an issue labelled ALERT_LABEL is opened, or the open one gets a
comment; the next success comments and closes it. While the issue is open, a failure
adds a comment only when its failed stage differs from the newest notice's, or when
there has been no notice for QUIET_PERIOD; each notice carries a hidden marker naming
its stage.

The GitHub API is external data: responses are size-capped and every malformed one
raises GitHubError. Each request has a timeout and a bounded number of retries.
"""

from __future__ import annotations

import argparse
import enum
import http.client
import json
import logging
import os
import re
import secrets
import stat
import subprocess
import sys
import time as time_module
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode, urlsplit
from zoneinfo import ZoneInfo

logger = logging.getLogger("wearreport.schedule")

LONDON_TZ: Final = "Europe/London"
DAY_START, DAY_END = time(7, 0), time(21, 0)  # London local time, end excluded

ALERT_LABEL: Final = "ops-alert"
ALERT_THRESHOLD: Final = 3
WORKFLOW_FILE: Final = "sweep.yml"
GATE_JOB, SWEEP_JOB, PUBLISH_JOB = "gate", "sweep", "publish"
STAGES: Final = frozenset(
    {
        "gate",
        "forced",
        "setup",
        "checkout",
        "sweep",
        "skipped",
        "publish",
        "record",
        "none",
        "unknown",
    }
)
# The stage of a run whose sweep was skipped as too soon after the last one.
SKIPPED_STAGE: Final = "skipped"
# sweep.yml's step names: the Sweep step (sweep job) and the record step (publish job).
# In runs from before the publish job existed (T-007), both were in the sweep job and a
# skip was a successful Sweep step and a skipped record step.
SWEEP_STEP, RECORD_STEP = "Sweep", "Did the published sweep succeed?"
# Previous runs read when counting failures; more than the threshold needs.
LOOKBACK_RUNS: Final = 20
# While the issue is open, a failure in the same stage as the newest notice is noted at
# most this often (a stage change is always noted).
QUIET_PERIOD: Final = timedelta(hours=1)
NOTICE_MARKER = re.compile(r"<!-- ops-alert stage=([a-z]+) -->")
NOTICE_AUTHOR: Final = "github-actions[bot]"

API_URL: Final = "https://api.github.com"
TIMEOUT_SECONDS: Final = 10.0
RETRIES: Final = 2
MAX_RESPONSE_BYTES: Final = 2 * 1024 * 1024
MAX_STATUS_BYTES: Final = 64 * 1024

# The same sweep_id as wearreport.aggregate.SWEEP_ID (a test keeps them equal).
SWEEP_ID = re.compile(
    r"[0-9]{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3])[0-5][0-9]Z"
)
RECORD_PATH = re.compile(
    rf"sweeps/([0-9]{{4}})/([0-9]{{2}})/([0-9]{{2}})/({SWEEP_ID.pattern})\.json"
)
STATUS_PATH: Final = "status.json"
# The same time format as wearreport.aggregate.UTC_TIME (a test keeps them equal).
UTC_TIME = re.compile(
    r"[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)
# wearreport.publish's success rule and record size cap (a test keeps them equal).
SUCCESS_NUMERATOR, SUCCESS_DENOMINATOR = 9, 10
MAX_RECORD_BYTES: Final = 1024 * 1024
# A sweep uploads one record and status.json; far more is not one of ours.
MAX_ARTIFACT_ENTRIES: Final = 64
# The streak search adds at most this many days older than the checked-out window;
# a streak older than that counts as unbounded (at least that many days of failures).
MAX_STREAK_DAYS: Final = 400
DAY_DIR = re.compile(r"sweeps/[0-9]{4}/[0-9]{2}/[0-9]{2}")
GIT_TIMEOUT_SECONDS: Final = 120.0


class ScheduleError(RuntimeError):
    """The schedule helpers cannot do what was asked (bad input, unexpected state)."""


class GitHubError(ScheduleError):
    """The GitHub API failed or answered with something unexpected."""


# Daytime gate --------------------------------------------------------------------------


def is_open(moment: datetime) -> bool:
    """07:00 <= London local time < 21:00 at `moment`, BST or GMT as in force then.

    Raises ValueError for a naive datetime: the gate never guesses a time zone.
    """
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("the gate needs a timezone-aware time")
    local = moment.astimezone(ZoneInfo(LONDON_TZ)).time()
    return DAY_START <= local < DAY_END


def ref_allowed(event: str, ref: str, default_branch: str) -> bool:
    """Whether a run of `event` on `ref` may sweep: scheduled runs always (GitHub starts
    them from the default branch only); any other run only from refs/heads/<default
    branch>, so a dispatch on a task branch never runs its unreviewed code with the
    secrets and the write token."""
    if event == "schedule":
        return True
    return bool(default_branch) and ref == f"refs/heads/{default_branch}"


def _parse_now(text: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        raise ScheduleError("--now must be YYYY-MM-DDTHH:MM:SSZ") from None


# Outcomes ------------------------------------------------------------------------------


class Outcome(enum.StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    # no sweep ran: gate closed, skipped as too soon, or cancelled before starting
    NONE = "none"


class Action(enum.StrEnum):
    OPEN = "open"
    COMMENT = "comment"
    QUIET = "quiet"  # the issue is open and this failure adds nothing new yet
    CLOSE = "close"
    NOTHING = "nothing"


def status_failures(path: Path) -> int:
    """`consecutive_failures` from a status.json. Raises ScheduleError when the file is
    missing, too large or not a status document."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_STATUS_BYTES + 1)
    except OSError:
        raise ScheduleError("status.json cannot be read") from None
    if len(data) > MAX_STATUS_BYTES:
        raise ScheduleError("status.json is too large")
    try:
        status = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):  # includes UnicodeDecodeError
        raise ScheduleError("status.json is not JSON") from None
    value = status.get("consecutive_failures") if isinstance(status, dict) else None
    if type(value) is not int or value < 0:
        raise ScheduleError("status.json has no valid consecutive_failures")
    return value


def current_outcome(
    gate_result: str, sweep_result: str, stage: str = "", publish_result: str = ""
) -> Outcome:
    """This run's outcome from the `needs.<job>.result` of the gate, sweep and publish
    jobs, and the sweep job's stage (`skipped`: the sweep was too soon after the last
    one, so the publish job did not run). An empty `publish_result` means no publish job
    (the sweep job's result alone decides)."""
    if gate_result == "failure":
        return Outcome.FAILURE
    if stage == SKIPPED_STAGE:
        return Outcome.NONE
    if sweep_result in ("failure", "cancelled"):  # cancelled: the job timed out
        return Outcome.FAILURE
    if sweep_result != "success":
        return Outcome.NONE
    if publish_result in ("", "success"):
        return Outcome.SUCCESS
    # failed, timed out, or not run although the sweep ran: nothing was published
    return Outcome.FAILURE


def alert_stage(gate_result: str, sweep_stage: str, publish_result: str, publish_stage: str) -> str:
    """The failed stage the alert reports: the sweep job's, unless the sweep job got
    through (`none`) and the publish job then names one, or failed without naming one."""
    if gate_result == "failure":
        return "gate"
    if sweep_stage not in ("", "none"):
        return sweep_stage
    if publish_stage:
        return publish_stage
    if publish_result in ("failure", "cancelled"):
        return "publish"
    return sweep_stage or "unknown"


def run_outcome(jobs: Sequence[Mapping[str, Any]]) -> Outcome:
    """A finished run's outcome from its jobs (as the jobs API lists them): a failed
    gate, sweep or publish job is a failure; a successful sweep job is a success once
    its publish job succeeded, and no sweep at all when the publish job was skipped (the
    sweep was too soon after the last one). A run from before the publish job existed
    is read as T-007 wrote it. Raises GitHubError when a job's name is not a string."""
    if any(type(job.get("name")) is not str for job in jobs):
        raise GitHubError("run jobs: unexpected response")
    by_name = {job["name"]: job for job in jobs}
    gate, sweep = by_name.get(GATE_JOB), by_name.get(SWEEP_JOB)
    publish_job = by_name.get(PUBLISH_JOB)
    if gate is not None and gate.get("conclusion") == "failure":
        return Outcome.FAILURE
    if sweep is None:
        return Outcome.NONE
    outcome = _job_outcome(sweep)
    if outcome is not Outcome.SUCCESS:
        return outcome
    if publish_job is None:  # the T-007 layout: publishing was a step of the sweep job
        return Outcome.NONE if _skipped(sweep.get("steps")) else Outcome.SUCCESS
    if publish_job.get("conclusion") == "skipped":
        return Outcome.NONE
    return _job_outcome(publish_job)


def _job_outcome(job: Mapping[str, Any]) -> Outcome:
    conclusion = job.get("conclusion")
    if conclusion == "success":
        return Outcome.SUCCESS
    if conclusion in ("failure", "timed_out"):
        return Outcome.FAILURE
    if conclusion == "cancelled" and job.get("steps"):  # it had started: a timeout
        return Outcome.FAILURE
    return Outcome.NONE


def _skipped(steps: Any) -> bool:
    """Whether a T-007 sweep job's steps (as the jobs API lists them) show a skipped
    sweep: the Sweep step succeeded and the record step was skipped. Anything malformed,
    such as a step whose name is not a string, is not."""
    if not isinstance(steps, list):
        return False
    conclusions = {
        step["name"]: step.get("conclusion")
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    }
    return conclusions.get(SWEEP_STEP) == "success" and conclusions.get(RECORD_STEP) == "skipped"


def consecutive_failures(outcomes: Iterable[Outcome]) -> int:
    """Failures, newest first, until the first success; NONE is passed over."""
    count = 0
    for outcome in outcomes:
        if outcome is Outcome.SUCCESS:
            break
        if outcome is Outcome.FAILURE:
            count += 1
    return count


def decide(current: Outcome, failures: int, issue_open: bool) -> Action:
    """What to do with the alert issue after this run."""
    if current is Outcome.SUCCESS:
        return Action.CLOSE if issue_open else Action.NOTHING
    if current is Outcome.FAILURE and failures >= ALERT_THRESHOLD:
        return Action.COMMENT if issue_open else Action.OPEN
    return Action.NOTHING


# Sparse data branch --------------------------------------------------------------------


def _successful_record(data: bytes, relative: str, now: datetime) -> bool:
    """Whether `data`, the record file at `relative` (sweeps/YYYY/MM/DD/<id>.json) in the
    data branch, is a sweep the publisher counts as a success as of `now`: the checks of
    `publish.load_record` (size, JSON, `aggregate.check_record`, stored under its own
    sweep_id), a start not after `now` (`compute_status` ignores later ones), and
    `publish.is_success`. Anything else is not a success, as the publisher leaves it out
    of status.json."""
    try:  # the engine's dependencies (numpy, onnxruntime) come with these
        from wearreport import aggregate, publish
    except ImportError:
        raise ScheduleError(
            "checking records needs the engine's dependencies: run this command with the "
            "engine's virtual environment"
        ) from None
    m = RECORD_PATH.fullmatch(relative)
    if not m or len(data) > publish.MAX_RECORD_BYTES:
        return False
    try:
        record = json.loads(data.decode("utf-8"))
        aggregate.check_record(record)
        started = aggregate.parse_utc(record["started_at"])
    # RecordError is a ValueError; the rest cannot come from a valid record, and hostile
    # data never gets past this function as anything but "not a success".
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        return False
    sweep_id = record["sweep_id"]
    if sweep_id != m.group(4) or sweep_id[0:8] != "".join(m.group(1, 2, 3)):
        return False
    return started <= now and publish.is_success(record)


def _read_regular(path: Path, limit: int) -> bytes | None:
    """The first `limit` + 1 bytes of a regular file (not a symlink); None otherwise."""
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        with open(path, "rb") as fh:
            return fh.read(limit + 1)
    except OSError:
        return None


def streak_closed(data_dir: Path, now: datetime | None = None) -> bool:
    """Whether the checked-out part of the data branch shows where the current failure
    streak ends, so that status.json's `consecutive_failures` can be counted from it.

    True when status.json says the newest published sweep succeeded (older records
    cannot change the count), or when a checked-out record is a success as the publisher
    counts one as of `now` (see `_successful_record`). Records must be regular files
    under their own date, sweeps/YYYY/MM/DD/, with no symlinked directory on the way.
    """
    now = datetime.now(UTC) if now is None else now
    try:
        if status_failures(data_dir / STATUS_PATH) == 0:
            return True
    except ScheduleError:
        pass  # no usable status.json: look at the records
    for path in sorted((data_dir / "sweeps").glob("*/*/*/*.json"), reverse=True):
        relative = path.relative_to(data_dir)
        if not RECORD_PATH.fullmatch(relative.as_posix()):
            continue
        if any((data_dir / parent).is_symlink() for parent in list(relative.parents)[:-1]):
            continue
        data = _read_regular(path, MAX_RECORD_BYTES)
        if data is not None and _successful_record(data, relative.as_posix(), now):
            return True
    return False


@dataclass(frozen=True, slots=True)
class Widening:
    """The older days a sparse data checkout must add, as sparse-checkout patterns
    (/sweeps/YYYY/MM/DD/), newest first. `capped`: the search stopped after max_days
    with no success, so the streak counts as unbounded."""

    days: tuple[str, ...]
    capped: bool


def _git(data_dir: Path, *args: str, stdin: bytes | None = None) -> bytes:
    try:
        done = subprocess.run(  # noqa: S603
            ["git", "-C", str(data_dir), *args],  # noqa: S607
            input=stdin,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScheduleError(f"git {args[0]} failed: {type(exc).__name__}") from None
    if done.returncode != 0:
        raise ScheduleError(f"git {args[0]} failed (exit {done.returncode})")
    return done.stdout


def _older_records(data_dir: Path, before: str) -> dict[str, list[tuple[str, str]]]:
    """(path, blob id) of each record file in HEAD's tree under a day older than `before`,
    by day. Only regular files named and placed as records count, as for streak_closed."""
    days: dict[str, list[tuple[str, str]]] = {}
    listing = _git(data_dir, "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", "sweeps")
    for entry in listing.split(b"\0"):
        meta, tab, raw_path = entry.partition(b"\t")
        fields = meta.split()
        if not tab or len(fields) != 3 or fields[0] != b"100644" or fields[1] != b"blob":
            continue
        try:
            path, oid = raw_path.decode("ascii"), fields[2].decode("ascii")
        except UnicodeDecodeError:
            continue
        m = RECORD_PATH.fullmatch(path)
        if not m or m.group(4)[0:8] != "".join(m.group(1, 2, 3)):
            continue
        day = path.rsplit("/", 1)[0]
        if day < before:
            days.setdefault(day, []).append((path, oid))
    return days


def _blobs(data_dir: Path, oids: Sequence[str]) -> list[bytes]:
    """The contents of `oids`, in order, from one `git cat-file --batch`."""
    out = _git(data_dir, "cat-file", "--batch", stdin="".join(f"{o}\n" for o in oids).encode())
    blobs, pos = [], 0
    for oid in oids:
        end = out.find(b"\n", pos)
        header = out[pos:end].split() if end >= 0 else []
        if (
            len(header) != 3
            or header[0].decode("ascii", "replace") != oid
            or not header[2].isdigit()
        ):
            raise ScheduleError(f"git cat-file: unexpected answer for {oid}")
        size = int(header[2])
        blobs.append(out[end + 1 : end + 1 + size])
        pos = end + 1 + size + 1  # the contents are followed by a newline
    return blobs


def streak_days(
    data_dir: Path, *, before: str, now: datetime | None = None, max_days: int = MAX_STREAK_DAYS
) -> Widening:
    """The older days the shallow, sparse data checkout in `data_dir` must add so that it
    shows where the failure streak ends. `before` is the oldest day already checked out
    (sweeps/YYYY/MM/DD). Nothing to add when the checkout already shows it
    (`streak_closed`). Otherwise the days older than `before` are taken newest first,
    each once: its records are read from git's object store (the shallow fetch holds the
    whole tree) and checked as `streak_closed` checks them, and the search stops at the
    first day with a success, at the end of the history, or after `max_days` days
    (capped: the streak counts as unbounded). Linear in the number of records.
    """
    if not DAY_DIR.fullmatch(before):
        raise ScheduleError("--before must be sweeps/YYYY/MM/DD")
    now = datetime.now(UTC) if now is None else now
    if streak_closed(data_dir, now):
        return Widening((), capped=False)
    older = _older_records(data_dir, before)
    added: list[str] = []
    for day in sorted(older, reverse=True):
        if len(added) >= max_days:
            return Widening(tuple(added), capped=True)
        added.append(f"/{day}/")
        records = sorted(older[day], reverse=True)
        blobs = _blobs(data_dir, [oid for _, oid in records])
        if any(_successful_record(b, p, now) for (p, _), b in zip(records, blobs, strict=True)):
            break
    return Widening(tuple(added), capped=False)


# The artifact from the sweep job -------------------------------------------------------


def check_artifact(root: Path) -> list[str]:
    """Check the artifact the sweep job uploaded, as downloaded to `root`: only regular
    files (no symlink, device, pipe or socket; directories are walked, never followed),
    status.json (a status document) and records `check_staged` would accept as new,
    each at most MAX_RECORD_BYTES and JSON whose sweep_id is its file name. Returns the
    relative paths, sorted; raises ScheduleError naming the first problem."""
    found: list[str] = []
    pending = [""]
    entries = 0
    while pending:
        directory = pending.pop()
        try:
            names = sorted(os.listdir(root / directory))
        except OSError as exc:
            raise ScheduleError(f"artifact: cannot list {directory!r}: {exc.strerror}") from None
        for name in names:
            relative = f"{directory}/{name}" if directory else name
            entries += 1
            if entries > MAX_ARTIFACT_ENTRIES:
                raise ScheduleError("artifact: too many files")
            try:
                mode = os.lstat(root / relative).st_mode
            except OSError as exc:
                raise ScheduleError(
                    f"artifact: cannot inspect {relative!r}: {exc.strerror}"
                ) from None
            if stat.S_ISDIR(mode):
                pending.append(relative)
            elif not stat.S_ISREG(mode):
                raise ScheduleError(f"artifact: {relative!r} is not a regular file")
            elif relative == STATUS_PATH:
                status_failures(root / relative)
                found.append(relative)
            else:
                _check_artifact_record(root, relative)
                found.append(relative)
    if STATUS_PATH not in found:
        raise ScheduleError("artifact: no status.json")
    return sorted(found)


def _check_artifact_record(root: Path, relative: str) -> None:
    if not _is_new_record("A", relative):
        raise ScheduleError(f"artifact: refusing {relative!r}")
    data = _read_regular(root / relative, MAX_RECORD_BYTES)
    if data is None:
        raise ScheduleError(f"artifact: cannot read {relative!r}")
    if len(data) > MAX_RECORD_BYTES:
        raise ScheduleError(f"artifact: {relative!r} is larger than {MAX_RECORD_BYTES} bytes")
    try:
        record = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):  # includes UnicodeDecodeError
        raise ScheduleError(f"artifact: {relative!r} is not JSON") from None
    sweep_id = record.get("sweep_id") if isinstance(record, dict) else None
    if not isinstance(sweep_id, str) or relative.rsplit("/", 1)[1] != f"{sweep_id}.json":
        raise ScheduleError(f"artifact: {relative!r} is not stored under its sweep_id")


def copy_artifact(artifact: Path, data_dir: Path) -> list[str]:
    """Check the artifact in `artifact` (`check_artifact`), then copy it into the data
    branch checkout `data_dir`: a record is created, or left alone when the identical
    record is already there (a different one raises ScheduleError, as records are
    immutable); status.json is replaced. No existing directory on the way may be a
    symlink or anything but a plain directory. Returns the paths copied.

    An artifact status.json older than the checked-out one (an earlier `last_sweep_at`,
    as when an old run's publish job is re-run after later runs published) raises
    ScheduleError before anything is written: it would roll the published status back."""
    paths = check_artifact(artifact)
    _refuse_a_status_rollback(artifact / STATUS_PATH, data_dir / STATUS_PATH)
    for relative in paths:
        data = _read_regular(artifact / relative, MAX_RECORD_BYTES)
        if data is None:
            raise ScheduleError(f"artifact: cannot read {relative!r}")
        target = data_dir / relative
        _plain_dirs(data_dir, target.parent)
        if relative == STATUS_PATH:
            _replace_file(target, data)
        elif _read_regular(target, MAX_RECORD_BYTES) != data:
            _create_file(target, data)
    return paths


def _refuse_a_status_rollback(new: Path, published: Path) -> None:
    """Raise ScheduleError when the status.json at `new` has an earlier `last_sweep_at`
    than the one at `published`. A missing or unusable published status (the first
    publish) or one without a sweep yet allows any new status."""
    data = _read_regular(published, MAX_STATUS_BYTES)
    if data is None or len(data) > MAX_STATUS_BYTES:
        return
    try:
        published_at = _last_sweep_at(data)
    except ScheduleError:
        return
    if published_at is None:
        return
    data = _read_regular(new, MAX_STATUS_BYTES)
    if data is None or len(data) > MAX_STATUS_BYTES:
        raise ScheduleError("artifact: cannot read status.json")
    new_at = _last_sweep_at(data)
    if new_at is None or new_at < published_at:
        raise ScheduleError(
            "artifact: status.json is older than the published one (last_sweep_at): "
            "refusing to roll it back"
        )


def _last_sweep_at(data: bytes) -> datetime | None:
    """A status document's `last_sweep_at` (None when no sweep is published yet), in
    wearreport.aggregate's YYYY-MM-DDTHH:MM:SSZ. Raises ScheduleError for anything else."""
    try:
        status = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):  # includes UnicodeDecodeError
        raise ScheduleError("status.json is not JSON") from None
    value = status.get("last_sweep_at") if isinstance(status, dict) else None
    if value is None:
        return None
    if not isinstance(value, str) or not UTC_TIME.fullmatch(value):
        raise ScheduleError("status.json has no valid last_sweep_at")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:  # e.g. 2026-02-30
        raise ScheduleError("status.json has no valid last_sweep_at") from None


def _plain_dirs(root: Path, directory: Path) -> None:
    current = root
    for part in directory.relative_to(root).parts:
        current = current / part
        try:
            os.mkdir(current)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ScheduleError(f"cannot create {part!r}: {exc.strerror}") from None
        try:
            plain = stat.S_ISDIR(os.lstat(current).st_mode)
        except OSError:
            plain = False
        if not plain:
            raise ScheduleError(
                f"{current.relative_to(root).as_posix()!r} is not a plain directory"
            )


def _create_file(path: Path, data: bytes) -> None:
    """Create `path` holding `data`, as text: the publisher writes records and status.json
    as ASCII, and the engine writes no binary files (INV-1). Exclusive creation never
    follows a symlink."""
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        raise ScheduleError(f"{path.name} is not ASCII, as the publisher writes it") from None
    try:
        with open(path, "x", encoding="ascii", newline="") as fh:
            fh.write(text)
    except FileExistsError:
        raise ScheduleError(f"a different {path.name} is already published") from None
    except OSError as exc:
        raise ScheduleError(f"cannot write {path.name}: {exc.strerror}") from None


def _replace_file(path: Path, data: bytes) -> None:
    tmp = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        _create_file(tmp, data)
        os.replace(tmp, path)
    except OSError as exc:
        raise ScheduleError(f"cannot write {path.name}: {exc.strerror}") from None
    finally:
        tmp.unlink(missing_ok=True)


# Staged paths --------------------------------------------------------------------------


def check_staged(lines: Iterable[str]) -> list[str]:
    """Check `git diff --cached --name-status` output: only added record files, stored
    under their own date, and an added or modified status.json. Returns the paths;
    raises ScheduleError naming the first line that is anything else."""
    paths = []
    for line in lines:
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 2:
            raise ScheduleError(f"unexpected staged change: {line.strip()!r}")
        change, path = fields
        if not (_is_new_record(change, path) or (path == STATUS_PATH and change in ("A", "M"))):
            raise ScheduleError(f"refusing to publish {change} {path!r}")
        paths.append(path)
    return paths


def _is_new_record(change: str, path: str) -> bool:
    """An added sweeps/YYYY/MM/DD/<sweep_id>.json whose directories are its own date."""
    m = RECORD_PATH.fullmatch(path)
    return bool(m and change == "A" and m.group(4)[0:8] == "".join(m.group(1, 2, 3)))


# GitHub API ----------------------------------------------------------------------------

Transport = Callable[[str, str, bytes | None, Mapping[str, str], float], tuple[int, bytes]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: the request carries the token."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())


def _urllib_transport(
    method: str, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)  # noqa: S310
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, exc.read(MAX_RESPONSE_BYTES + 1)


class GitHub:
    """A small GitHub REST client for one repository, authenticated with a token."""

    def __init__(
        self,
        repo: str,
        token: str,
        *,
        api_url: str = API_URL,
        transport: Transport = _urllib_transport,
        timeout: float = TIMEOUT_SECONDS,
        retries: int = RETRIES,
        sleep: Callable[[float], None] = time_module.sleep,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ScheduleError("repository must be OWNER/NAME")
        if not token:
            raise ScheduleError("a GitHub token is required")
        if urlsplit(api_url).scheme != "https":
            raise ScheduleError("the GitHub API URL must be https")
        self.repo = repo
        self._token = token
        self._api_url = api_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout
        self._retries = retries
        self._sleep = sleep

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
        allow: Sequence[int] = (),
    ) -> tuple[int, Any]:
        """Send one request under /repos/OWNER/NAME and return (status, parsed JSON).

        2xx responses, and statuses in `allow`, are returned; anything else raises
        GitHubError. For GET and PATCH (idempotent), network errors, malformed HTTP
        responses, 429 and 5xx are retried `retries` times with a growing pause; a POST
        is sent once, so a lost response cannot open a second issue. Redirects are not
        followed. The token never appears in an error.
        """
        url = f"{self._api_url}/repos/{self.repo}{path}"
        if query:
            url += "?" + urlencode(query)
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "wearreport-sweep",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        what = f"{method} {path}"
        last = what
        attempts = self._retries + 1 if method in ("GET", "PATCH") else 1
        for attempt in range(attempts):
            if attempt:
                self._sleep(2.0**attempt)
            try:
                status, raw = self._transport(method, url, data, headers, self._timeout)
            # URLError, timeouts, malformed responses (http.client: BadStatusLine,
            # IncompleteRead, LineTooLong and the rest of HTTPException)
            except (OSError, ValueError, http.client.HTTPException) as exc:
                logger.warning(
                    "GitHub request failed", extra={"request": what, "error": type(exc).__name__}
                )
                last = f"{what}: {type(exc).__name__}"
                continue
            if status == 429 or status >= 500:
                logger.warning("GitHub request failed", extra={"request": what, "status": status})
                last = f"{what}: HTTP {status}"
                continue
            if not (200 <= status < 300 or status in allow):
                raise GitHubError(f"{what}: HTTP {status}")
            return status, _parse(raw, what)
        raise GitHubError(f"{last} after {attempts} attempt{'s' if attempts > 1 else ''}")


def _parse(raw: bytes, what: str) -> Any:
    if len(raw) > MAX_RESPONSE_BYTES:
        raise GitHubError(f"{what}: response too large")
    if not raw.strip():
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):  # includes UnicodeDecodeError
        raise GitHubError(f"{what}: response is not JSON") from None


def _items(value: Any, key: str | None, what: str) -> list[Mapping[str, Any]]:
    """A list of objects: `value[key]`, or `value` itself when key is None."""
    items = value
    if key is not None:
        items = value.get(key) if isinstance(value, dict) else None
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise GitHubError(f"{what}: unexpected response")
    return items


def _positive_int(value: Any, what: str) -> int:
    if type(value) is not int or value <= 0:
        raise GitHubError(f"{what}: unexpected response")
    return value


def previous_outcomes(gh: GitHub, *, run_id: int, branch: str) -> Iterator[tuple[int, Outcome]]:
    """(run id, outcome) of this workflow's finished runs on `branch`, newest first by
    creation time (not by the listing's order, which a re-run can change), excluding
    `run_id`; at most LOOKBACK_RUNS runs. A run without a valid `created_at` raises
    GitHubError."""
    what = "workflow runs"
    _, listing = gh.request(
        "GET",
        f"/actions/workflows/{WORKFLOW_FILE}/runs",
        query={
            "branch": branch,
            "status": "completed",
            "exclude_pull_requests": "true",
            "per_page": str(LOOKBACK_RUNS),
        },
    )
    runs = [
        (_run_created(run.get("created_at"), what), _positive_int(run.get("id"), what), run)
        for run in _items(listing, "workflow_runs", what)[:LOOKBACK_RUNS]
    ]
    for _, other, run in sorted(runs, key=lambda item: item[0:2], reverse=True):
        if other == run_id or run.get("status") != "completed":
            continue
        _, jobs = gh.request("GET", f"/actions/runs/{other}/jobs", query={"filter": "latest"})
        yield other, run_outcome(_items(jobs, "jobs", "run jobs"))


def _run_created(value: Any, what: str) -> datetime:
    created = _timestamp(value, what)
    if created is None:
        raise GitHubError(f"{what}: unexpected response")
    return created


def _oldest_open_alert_issue(gh: GitHub) -> tuple[int, Mapping[str, Any]] | None:
    _, issues = gh.request(
        "GET",
        "/issues",
        query={"labels": ALERT_LABEL, "state": "open", "per_page": "100"},
    )
    found = [
        (_positive_int(issue.get("number"), "issues"), issue)
        for issue in _items(issues, None, "issues")
        if "pull_request" not in issue and issue.get("state") == "open"
    ]
    return min(found, key=lambda item: item[0], default=None)


def open_alert_issue(gh: GitHub) -> int | None:
    """The number of the oldest open issue labelled ALERT_LABEL, or None."""
    found = _oldest_open_alert_issue(gh)
    return None if found is None else found[0]


def _timestamp(value: Any, what: str) -> datetime | None:
    """A GitHub `YYYY-MM-DDTHH:MM:SSZ` time; None when absent."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise GitHubError(f"{what}: unexpected response")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        raise GitHubError(f"{what}: unexpected response") from None


def _notice_stage(body: Any) -> str | None:
    """The stage named by the last notice marker in `body`, or None."""
    if not isinstance(body, str):
        return None
    stages = [stage for stage in NOTICE_MARKER.findall(body) if stage in STAGES]
    return stages[-1] if stages else None


def last_notice(
    gh: GitHub, number: int, issue: Mapping[str, Any], now: datetime
) -> tuple[datetime, str] | None:
    """(time, stage) of the newest failure notice on the open alert issue within the
    last QUIET_PERIOD, or None. A notice is the issue itself, or a comment by
    NOTICE_AUTHOR, carrying a NOTICE_MARKER. An issue that is gone (404) has none."""
    since = now - QUIET_PERIOD
    notices: list[tuple[datetime, str]] = []
    created, stage = _timestamp(issue.get("created_at"), "issues"), _notice_stage(issue.get("body"))
    if created is not None and stage is not None and created > since:
        notices.append((created, stage))
    status, comments = gh.request(
        "GET",
        f"/issues/{number}/comments",
        query={"since": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "per_page": "100"},
        allow=(404,),
    )
    if status != 404:
        for comment in _items(comments, None, "issue comments"):
            user = comment.get("user")
            if not isinstance(user, dict) or user.get("login") != NOTICE_AUTHOR:
                continue
            created = _timestamp(comment.get("created_at"), "issue comments")
            stage = _notice_stage(comment.get("body"))
            if created is not None and stage is not None and created > since:
                notices.append((created, stage))
    return max(notices, default=None)


def worth_a_comment(stage: str, notice: tuple[datetime, str] | None, now: datetime) -> bool:
    """Whether a failure in `stage` should comment on the open alert issue, given its
    newest notice: when there is none within QUIET_PERIOD, or it named another stage."""
    return notice is None or notice[1] != stage or now - notice[0] >= QUIET_PERIOD


def _ensure_label(gh: GitHub) -> None:
    status, _ = gh.request("GET", f"/labels/{ALERT_LABEL}", allow=(404,))
    if status == 404:
        gh.request(
            "POST",
            "/labels",
            body={
                "name": ALERT_LABEL,
                "color": "d73a4a",
                "description": "Scheduled sweeps are failing",
            },
            allow=(422,),  # created meanwhile
        )


@dataclass(frozen=True, slots=True)
class _Context:
    run_url: str
    stage: str
    record_failures: int | None


def _failure_text(failures: int, ctx: _Context, failed_runs: Sequence[str]) -> str:
    records = (
        "no status.json from this run" if ctx.record_failures is None else str(ctx.record_failures)
    )
    lines = [
        f"Sweeps have failed {failures} times in a row (alert threshold {ALERT_THRESHOLD}).",
        "",
        f"Latest failure: {ctx.run_url}",
        f"- failed stage: `{ctx.stage}`",
        f"- consecutive failed records in status.json: {records}",
    ]
    if failed_runs:
        lines += ["", "Earlier failed runs, newest first:"]
        lines += [f"- {url}" for url in failed_runs]
    lines += [
        "",
        "This issue is closed automatically after the next successful sweep. Further",
        "failures are noted when the failed stage changes, or hourly otherwise.",
        f"<!-- ops-alert stage={ctx.stage} -->",
    ]
    return "\n".join(lines) + "\n"


def alert(
    gh: GitHub,
    *,
    run_id: int,
    branch: str,
    current: Outcome,
    stage: str,
    record_failures: int | None,
    server_url: str,
    now: datetime | None = None,
) -> Action:
    """Count consecutive failures including this run and act on the alert issue."""
    now = datetime.now(UTC) if now is None else now
    if current is Outcome.NONE:
        _log(Action.NOTHING, 0, None)
        return Action.NOTHING
    base = f"{server_url.rstrip('/')}/{gh.repo}/actions/runs"
    ctx = _Context(
        run_url=f"{base}/{run_id}",
        stage=stage if stage in STAGES else "unknown",
        record_failures=record_failures,
    )
    found = _oldest_open_alert_issue(gh)
    issue = None if found is None else found[0]
    if current is Outcome.SUCCESS:
        action = decide(current, 0, issue is not None)
        if action is Action.CLOSE and issue is not None:
            gh.request(
                "POST",
                f"/issues/{issue}/comments",
                body={"body": f"Sweep succeeded in {ctx.run_url}. Closing.\n"},
            )
            gh.request(
                "PATCH", f"/issues/{issue}", body={"state": "closed", "state_reason": "completed"}
            )
        _log(action, 0, issue)
        return action

    failed_runs: list[str] = []
    history: list[Outcome] = []
    for other, outcome in previous_outcomes(gh, run_id=run_id, branch=branch):
        history.append(outcome)
        if outcome is Outcome.SUCCESS:
            break
        if outcome is Outcome.FAILURE:
            failed_runs.append(f"{base}/{other}")
    failures = max(1 + consecutive_failures(history), record_failures or 0)
    action = decide(current, failures, issue is not None)
    if action is Action.COMMENT and found is not None:
        notice = last_notice(gh, found[0], found[1], now)
        if not worth_a_comment(ctx.stage, notice, now):
            action = Action.QUIET
    text = _failure_text(failures, ctx, failed_runs)
    if action is Action.OPEN:
        _ensure_label(gh)
        _, created = gh.request(
            "POST",
            "/issues",
            body={
                "title": f"Sweeps failing: {failures} consecutive failures",
                "body": text,
                "labels": [ALERT_LABEL],
            },
        )
        issue = _positive_int(created.get("number") if isinstance(created, dict) else None, "issue")
    elif action is Action.COMMENT:
        gh.request("POST", f"/issues/{issue}/comments", body={"body": text})
    _log(action, failures, issue)
    return action


def _log(action: Action, failures: int, issue: int | None) -> None:
    logger.info(
        "alert checked", extra={"action": action.value, "failures": failures, "issue": issue}
    )
    print(f"alert: {action.value} (consecutive failures {failures}, issue {issue})")


# Command line --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python3 -m wearreport.schedule",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = ap.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate", help="is it London daytime?")
    gate.add_argument("--now", help="YYYY-MM-DDTHH:MM:SSZ (default: the current time)")
    gate.add_argument("--event", help="github.event_name; also needs --ref, --default-branch")
    gate.add_argument("--ref", help="github.ref")
    gate.add_argument("--default-branch", help="the repository's default branch")
    outcome = commands.add_parser("outcome", help="did the newest published sweep succeed?")
    outcome.add_argument("--status", type=Path, required=True)
    commands.add_parser("check-staged", help="check `git diff --cached --name-status` on stdin")
    check = commands.add_parser("check-artifact", help="check the downloaded sweep artifact")
    check.add_argument("--dir", type=Path, required=True)
    copy = commands.add_parser("copy-artifact", help="check the artifact, copy it into the data")
    copy.add_argument("--artifact", type=Path, required=True)
    copy.add_argument("--data-dir", type=Path, required=True)
    streak = commands.add_parser(
        "streak-closed", help="does the checked-out data branch show where the streak ends?"
    )
    streak.add_argument("--data-dir", type=Path, required=True)
    streak.add_argument("--now", help="YYYY-MM-DDTHH:MM:SSZ (default: the current time)")
    days = commands.add_parser(
        "streak-days", help="which older days must the data checkout add to show it?"
    )
    days.add_argument("--data-dir", type=Path, required=True)
    days.add_argument("--before", required=True, help="the oldest checked-out day, sweeps/Y/M/D")
    days.add_argument("--now", help="YYYY-MM-DDTHH:MM:SSZ (default: the current time)")
    days.add_argument("--max-days", type=int, default=MAX_STREAK_DAYS)
    al = commands.add_parser("alert", help="open, comment on or close the alert issue")
    al.add_argument("--gate-result", required=True)
    al.add_argument("--sweep-result", required=True)
    al.add_argument("--publish-result", default="")
    al.add_argument("--stage", default="")
    al.add_argument("--publish-stage", default="")
    al.add_argument("--record-failures", default="")
    return ap


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, like `wearreport.cli` (which this module cannot import)."""

    _standard = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in self._standard})
        return json.dumps(payload, default=str)


def _env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise ScheduleError(f"{name} is not set")
    return value


def _alert_command(args: argparse.Namespace, environ: Mapping[str, str]) -> int:
    current = current_outcome(args.gate_result, args.sweep_result, args.stage, args.publish_result)
    record_failures = int(args.record_failures) if args.record_failures.isdigit() else None
    run_id = _env(environ, "GITHUB_RUN_ID")
    if not run_id.isdigit():
        raise ScheduleError("GITHUB_RUN_ID is not a number")
    gh = GitHub(
        _env(environ, "GITHUB_REPOSITORY"),
        _env(environ, "GITHUB_TOKEN"),
        api_url=environ.get("GITHUB_API_URL") or API_URL,
    )
    alert(
        gh,
        run_id=int(run_id),
        branch=_env(environ, "GITHUB_REF_NAME"),
        current=current,
        stage=alert_stage(args.gate_result, args.stage, args.publish_result, args.publish_stage),
        record_failures=record_failures,
        server_url=environ.get("GITHUB_SERVER_URL") or "https://github.com",
    )
    return 0


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    args = _parser().parse_args(argv)
    env = os.environ if environ is None else environ
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    try:
        if args.command == "gate":
            now = datetime.now(UTC) if args.now is None else _parse_now(args.now)
            given = [x is not None for x in (args.event, args.ref, args.default_branch)]
            if any(given) and not all(given):
                raise ScheduleError("--event, --ref and --default-branch go together")
            if args.event is not None and not ref_allowed(
                args.event, args.ref, args.default_branch
            ):
                print("open=false")
                print(
                    f"::error::{args.event} runs start from the default branch only; "
                    f"refusing {args.ref!r}",
                    file=sys.stderr,
                )
                return 0
            print(f"open={'true' if is_open(now) else 'false'}")
            return 0
        if args.command == "outcome":
            try:
                failures = status_failures(args.status)
            except ScheduleError as exc:
                print("outcome=failure")
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"outcome={'success' if failures == 0 else 'failure'}")
            print(f"consecutive_failures={failures}")
            return 0 if failures == 0 else 1
        if args.command == "streak-closed":
            now = datetime.now(UTC) if args.now is None else _parse_now(args.now)
            closed = streak_closed(args.data_dir, now)
            print(f"streak-closed: {'yes' if closed else 'no'}")
            return 0 if closed else 1
        if args.command == "streak-days":
            now = datetime.now(UTC) if args.now is None else _parse_now(args.now)
            if args.max_days < 1:
                raise ScheduleError("--max-days must be at least 1")
            widening = streak_days(
                args.data_dir, before=args.before, now=now, max_days=args.max_days
            )
            for day in widening.days:
                print(day)
            if widening.capped:
                print(
                    f"streak-days: no success in the {args.max_days} older days searched; "
                    "the failure streak counts as unbounded",
                    file=sys.stderr,
                )
            return 0
        if args.command == "check-staged":
            for path in check_staged(sys.stdin):
                print(f"staged: {path}")
            return 0
        if args.command == "check-artifact":
            for path in check_artifact(args.dir):
                print(path)
            return 0
        if args.command == "copy-artifact":
            for path in copy_artifact(args.artifact, args.data_dir):
                print(f"copied: {path}")
            return 0
        return _alert_command(args, env)
    except ScheduleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
