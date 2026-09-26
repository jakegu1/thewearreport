"""The scheduled sweep's helpers: daytime gate, sweep outcome, staged-path check, alert.

Run by `.github/workflows/sweep.yml` as `python3 -m wearreport.schedule COMMAND`, with no
dependencies installed, so this module uses the standard library only.

  gate [--now TIME]           print `open=true` in London daytime (07:00-21:00), else
                              `open=false` (a GITHUB_OUTPUT line); exit 0
  outcome --status PATH       print `outcome=success|failure` and `consecutive_failures=N`
                              from status.json; exit 1 unless the newest sweep succeeded
  check-staged                read `git diff --cached --name-status` on stdin; exit 1
                              unless it only adds records and adds or updates status.json
  alert --gate-result R --sweep-result R [--stage S] [--record-failures N]
                              open, comment on or close the `ops-alert` issue

The alert counts failed sweeps: this run's (from the gate and sweep job results) plus
the unbroken run of failures before it, read from the workflow's run history. Runs whose
gate closed, or that were cancelled before they started, neither count nor break the
run. The count is at least status.json's `consecutive_failures` (failed records), since
a sweep that exits 1 writes no record and one that publishes a failed record exits 0,
and the workflow fails its sweep job for both. At ALERT_THRESHOLD failures in a row an
issue labelled ALERT_LABEL is opened, or the open one gets a comment; the next success
comments and closes it.

The GitHub API is external data: responses are size-capped and every malformed one
raises GitHubError. Each request has a timeout and a bounded number of retries.
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import os
import re
import sys
import time as time_module
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
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
GATE_JOB, SWEEP_JOB = "gate", "sweep"
STAGES: Final = frozenset(
    {"gate", "forced", "setup", "checkout", "sweep", "publish", "record", "none", "unknown"}
)
# Previous runs read when counting failures; more than the threshold needs.
LOOKBACK_RUNS: Final = 20

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


def _parse_now(text: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        raise ScheduleError("--now must be YYYY-MM-DDTHH:MM:SSZ") from None


# Outcomes ------------------------------------------------------------------------------


class Outcome(enum.StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    NONE = "none"  # no sweep ran: gate closed, or cancelled before starting


class Action(enum.StrEnum):
    OPEN = "open"
    COMMENT = "comment"
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


def current_outcome(gate_result: str, sweep_result: str) -> Outcome:
    """This run's outcome from the `needs.<job>.result` of the gate and sweep jobs."""
    if gate_result == "failure":
        return Outcome.FAILURE
    if sweep_result == "success":
        return Outcome.SUCCESS
    if sweep_result in ("failure", "cancelled"):  # cancelled: the job timed out
        return Outcome.FAILURE
    return Outcome.NONE


def run_outcome(jobs: Sequence[Mapping[str, Any]]) -> Outcome:
    """A finished run's outcome from its jobs (as the jobs API lists them)."""
    by_name = {job.get("name"): job for job in jobs}
    gate, sweep = by_name.get(GATE_JOB), by_name.get(SWEEP_JOB)
    if gate is not None and gate.get("conclusion") == "failure":
        return Outcome.FAILURE
    if sweep is None:
        return Outcome.NONE
    conclusion = sweep.get("conclusion")
    if conclusion == "success":
        return Outcome.SUCCESS
    if conclusion in ("failure", "timed_out"):
        return Outcome.FAILURE
    if conclusion == "cancelled" and sweep.get("steps"):  # it had started: a timeout
        return Outcome.FAILURE
    return Outcome.NONE


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
        GitHubError. For GET and PATCH (idempotent), network errors, 429 and 5xx are
        retried `retries` times with a growing pause; a POST is sent once, so a lost
        response cannot open a second issue. Redirects are not followed. The token never
        appears in an error.
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
            except (OSError, ValueError) as exc:  # URLError, timeouts, bad responses
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
    """(run id, outcome) of this workflow's finished runs on `branch`, newest first,
    excluding `run_id`; at most LOOKBACK_RUNS runs."""
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
    for run in _items(listing, "workflow_runs", what)[:LOOKBACK_RUNS]:
        other = _positive_int(run.get("id"), what)
        if other == run_id or run.get("status") != "completed":
            continue
        _, jobs = gh.request("GET", f"/actions/runs/{other}/jobs", query={"filter": "latest"})
        yield other, run_outcome(_items(jobs, "jobs", "run jobs"))


def open_alert_issue(gh: GitHub) -> int | None:
    """The number of the oldest open issue labelled ALERT_LABEL, or None."""
    _, issues = gh.request(
        "GET",
        "/issues",
        query={"labels": ALERT_LABEL, "state": "open", "per_page": "100"},
    )
    numbers = [
        _positive_int(issue.get("number"), "issues")
        for issue in _items(issues, None, "issues")
        if "pull_request" not in issue and issue.get("state") == "open"
    ]
    return min(numbers, default=None)


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
    lines += ["", "This issue is closed automatically after the next successful sweep."]
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
) -> Action:
    """Count consecutive failures including this run and act on the alert issue."""
    if current is Outcome.NONE:
        _log(Action.NOTHING, 0, None)
        return Action.NOTHING
    base = f"{server_url.rstrip('/')}/{gh.repo}/actions/runs"
    ctx = _Context(
        run_url=f"{base}/{run_id}",
        stage=stage if stage in STAGES else "unknown",
        record_failures=record_failures,
    )
    issue = open_alert_issue(gh)
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
    outcome = commands.add_parser("outcome", help="did the newest published sweep succeed?")
    outcome.add_argument("--status", type=Path, required=True)
    commands.add_parser("check-staged", help="check `git diff --cached --name-status` on stdin")
    al = commands.add_parser("alert", help="open, comment on or close the alert issue")
    al.add_argument("--gate-result", required=True)
    al.add_argument("--sweep-result", required=True)
    al.add_argument("--stage", default="")
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
    current = current_outcome(args.gate_result, args.sweep_result)
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
        stage=args.stage or ("gate" if args.gate_result == "failure" else "unknown"),
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
        if args.command == "check-staged":
            for path in check_staged(sys.stdin):
                print(f"staged: {path}")
            return 0
        return _alert_command(args, env)
    except ScheduleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
