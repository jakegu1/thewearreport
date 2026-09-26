"""The scheduled sweep's helpers (wearreport.schedule): gate, outcomes, staged paths,
the GitHub client and the alert."""

from __future__ import annotations

import http.server
import io
import json
import threading
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from wearreport import aggregate, publish, schedule

REPO = "owner/repo"
Reply = tuple[int, bytes] | Exception


class Script:
    """A transport that answers from a list of replies and records each request."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[tuple[str, str, Any]] = []

    def __call__(
        self, method: str, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> tuple[int, bytes]:
        assert timeout > 0
        self.requests.append((method, url, json.loads(body) if body else None))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _gh(script: Script, **kw: Any) -> schedule.GitHub:
    return schedule.GitHub(REPO, "s3cret", transport=script, sleep=lambda _: None, **kw)


def _ok(value: Any) -> tuple[int, bytes]:
    return 200, json.dumps(value).encode()


# Shared definitions --------------------------------------------------------------------


def test_sweep_id_and_daytime_match_the_engine() -> None:
    assert schedule.SWEEP_ID.pattern == aggregate.SWEEP_ID.pattern
    assert (schedule.DAY_START, schedule.DAY_END) == (publish.DAY_START, publish.DAY_END)
    assert schedule.LONDON_TZ == publish.LONDON_TZ


def test_gate_accepts_any_aware_time() -> None:
    # 07:30 at UTC+2 is 05:30 UTC, 06:30 BST: closed; an hour later it is open.
    plus_two = timezone(timedelta(hours=2))
    assert schedule.is_open(datetime(2026, 7, 15, 7, 30, tzinfo=plus_two)) is False
    assert schedule.is_open(datetime(2026, 7, 15, 8, 30, tzinfo=plus_two)) is True


def test_gate_command_rejects_a_malformed_time(capsys: pytest.CaptureFixture[str]) -> None:
    assert schedule.main(["gate", "--now", "2026-07-15 12:00"]) == 1
    assert "--now" in capsys.readouterr().err


def test_gate_command_defaults_to_now(capsys: pytest.CaptureFixture[str]) -> None:
    assert schedule.main(["gate"]) == 0
    assert capsys.readouterr().out.strip() in {"open=true", "open=false"}


# status.json ---------------------------------------------------------------------------


def _status(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "status.json"
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not json",
        b"\xff\xfe",
        b"[" * 100_000 + b"]" * 100_000,
        b"[]",
        b'{"consecutive_failures": -1}',
        b'{"consecutive_failures": true}',
        b'{"consecutive_failures": 1.0}',
        b'{"consecutive_failures": "2"}',
        b"{}",
        b" " * (schedule.MAX_STATUS_BYTES + 1),
    ],
)
def test_status_failures_rejects_malformed_files(tmp_path: Path, data: bytes) -> None:
    with pytest.raises(schedule.ScheduleError):
        schedule.status_failures(_status(tmp_path, data))


def test_status_failures_reads_the_count(tmp_path: Path) -> None:
    assert schedule.status_failures(_status(tmp_path, b'{"consecutive_failures": 4}')) == 4


def test_status_failures_missing_file(tmp_path: Path) -> None:
    with pytest.raises(schedule.ScheduleError):
        schedule.status_failures(tmp_path / "absent.json")


def test_outcome_command_prints_the_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _status(tmp_path, b'{"consecutive_failures": 2}')
    assert schedule.main(["outcome", "--status", str(path)]) == 1
    assert capsys.readouterr().out.splitlines() == ["outcome=failure", "consecutive_failures=2"]


# Outcomes and decisions ----------------------------------------------------------------


def _job(name: str, conclusion: str | None, started: bool = True) -> dict[str, Any]:
    return {"name": name, "conclusion": conclusion, "steps": [{"name": "s"}] if started else []}


@pytest.mark.parametrize(
    ("jobs", "expected"),
    [
        ([], schedule.Outcome.NONE),
        ([_job("gate", "success")], schedule.Outcome.NONE),
        ([_job("gate", "failure")], schedule.Outcome.FAILURE),
        ([_job("gate", "success"), _job("sweep", "success")], schedule.Outcome.SUCCESS),
        ([_job("gate", "success"), _job("sweep", "failure")], schedule.Outcome.FAILURE),
        ([_job("gate", "success"), _job("sweep", "timed_out")], schedule.Outcome.FAILURE),
        ([_job("gate", "success"), _job("sweep", "cancelled")], schedule.Outcome.FAILURE),
        ([_job("gate", "success"), _job("sweep", "cancelled", False)], schedule.Outcome.NONE),
        ([_job("gate", "success"), _job("sweep", "skipped", False)], schedule.Outcome.NONE),
        ([_job("gate", "success"), _job("sweep", None)], schedule.Outcome.NONE),
        ([{"name": "sweep", "conclusion": 7}], schedule.Outcome.NONE),
    ],
)
def test_run_outcome(jobs: list[dict[str, Any]], expected: schedule.Outcome) -> None:
    assert schedule.run_outcome(jobs) is expected


def test_consecutive_failures_skips_runs_without_a_sweep() -> None:
    F, S, N = schedule.Outcome.FAILURE, schedule.Outcome.SUCCESS, schedule.Outcome.NONE
    assert schedule.consecutive_failures([]) == 0
    assert schedule.consecutive_failures([F, N, F, S, F, F]) == 2
    assert schedule.consecutive_failures([S, F]) == 0
    assert schedule.consecutive_failures([N, N]) == 0


@pytest.mark.parametrize(
    ("current", "failures", "issue_open", "expected"),
    [
        (schedule.Outcome.FAILURE, 2, False, schedule.Action.NOTHING),
        (schedule.Outcome.FAILURE, 3, False, schedule.Action.OPEN),
        (schedule.Outcome.FAILURE, 3, True, schedule.Action.COMMENT),
        (schedule.Outcome.FAILURE, 1, True, schedule.Action.NOTHING),
        (schedule.Outcome.SUCCESS, 0, True, schedule.Action.CLOSE),
        (schedule.Outcome.SUCCESS, 0, False, schedule.Action.NOTHING),
        (schedule.Outcome.NONE, 9, True, schedule.Action.NOTHING),
    ],
)
def test_decide(
    current: schedule.Outcome, failures: int, issue_open: bool, expected: schedule.Action
) -> None:
    assert schedule.decide(current, failures, issue_open) is expected


# Staged paths --------------------------------------------------------------------------


def test_check_staged_command_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("A\tsweeps/2026/07/15/20260715T1200Z.json\n"))
    assert schedule.main(["check-staged"]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("A\t.github/workflows/sweep.yml\n"))
    assert schedule.main(["check-staged"]) == 1
    assert "refusing" in capsys.readouterr().err


@pytest.mark.parametrize(
    "line",
    [
        "A\tsweeps/2026/13/15/20261315T1200Z.json",  # no month 13
        "A\tsweeps/2026/07/15/sub/20260715T1200Z.json",
        "A\t/status.json",
        "A\tstatus.json\textra",
        "T\tstatus.json",
    ],
)
def test_check_staged_rejects_near_misses(line: str) -> None:
    with pytest.raises(schedule.ScheduleError):
        schedule.check_staged([line])


# GitHub client -------------------------------------------------------------------------


def test_client_sends_the_token_and_parses_json() -> None:
    seen: list[Mapping[str, str]] = []

    def transport(
        method: str, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> tuple[int, bytes]:
        seen.append(headers)
        assert url == "https://api.github.com/repos/owner/repo/labels/x?a=b"
        return 200, b'{"name": "x"}'

    gh = schedule.GitHub(REPO, "s3cret", transport=transport)
    assert gh.request("GET", "/labels/x", query={"a": "b"}) == (200, {"name": "x"})
    assert seen[0]["Authorization"] == "Bearer s3cret"


def test_client_retries_server_errors_a_bounded_number_of_times() -> None:
    pauses: list[float] = []
    script = Script((502, b""), (429, b""), _ok({"ok": 1}))
    gh = schedule.GitHub(REPO, "s3cret", transport=script, sleep=pauses.append)
    assert gh.request("GET", "/x") == (200, {"ok": 1})
    assert pauses == [2.0, 4.0]

    script = Script((500, b""), OSError("reset"), TimeoutError(), _ok({}))
    with pytest.raises(schedule.GitHubError, match="3 attempts") as info:
        _gh(script).request("GET", "/x")
    assert "s3cret" not in str(info.value)
    assert len(script.replies) == 1  # the fourth reply was never asked for


def test_client_sends_a_post_once() -> None:
    script = Script((502, b""), (201, b'{"number": 2}'))
    with pytest.raises(schedule.GitHubError, match="1 attempt"):
        _gh(script).request("POST", "/issues", body={"title": "t"})
    assert len(script.requests) == 1


def test_client_does_not_retry_client_errors() -> None:
    script = Script((403, b'{"message": "no"}'), _ok({}))
    with pytest.raises(schedule.GitHubError, match="HTTP 403"):
        _gh(script).request("POST", "/issues", body={"title": "t"})
    assert len(script.requests) == 1


def test_client_allows_listed_statuses() -> None:
    assert _gh(Script((404, b'{"message": "Not Found"}'))).request(
        "GET", "/labels/x", allow=(404,)
    ) == (404, {"message": "Not Found"})


@pytest.mark.parametrize(
    "raw",
    [
        b"<html>",
        b"\xff",
        b"[" * 100_000,
        b" " * (schedule.MAX_RESPONSE_BYTES + 1),
    ],
)
def test_client_rejects_malformed_responses(raw: bytes) -> None:
    with pytest.raises(schedule.GitHubError):
        _gh(Script((200, raw))).request("GET", "/x")


def test_client_treats_an_empty_body_as_none() -> None:
    assert _gh(Script((204, b""))).request("DELETE", "/x") == (204, None)


@pytest.mark.parametrize(
    ("repo", "token", "api_url"),
    [
        ("owner", "t", schedule.API_URL),
        ("owner/repo/extra", "t", schedule.API_URL),
        (REPO, "", schedule.API_URL),
        (REPO, "t", "http://api.github.com"),
    ],
)
def test_client_rejects_bad_configuration(repo: str, token: str, api_url: str) -> None:
    with pytest.raises(schedule.ScheduleError):
        schedule.GitHub(repo, token, api_url=api_url)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/moved":
            self.send_response(301)
            self.send_header("Location", "/ok")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/slow":
            threading.Event().wait(2)
        status = 200 if self.path == "/ok" else 404
        body = b'{"path": "%s"}' % self.path.encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_urllib_transport_returns_status_and_body(server: str) -> None:
    status, body = schedule._urllib_transport("GET", f"{server}/ok", None, {}, 5.0)
    assert (status, json.loads(body)) == (200, {"path": "/ok"})
    status, body = schedule._urllib_transport("GET", f"{server}/missing", None, {}, 5.0)
    assert status == 404


def test_urllib_transport_does_not_follow_redirects(server: str) -> None:
    status, _ = schedule._urllib_transport("GET", f"{server}/moved", None, {}, 5.0)
    assert status == 301


def test_urllib_transport_times_out(server: str) -> None:
    with pytest.raises(OSError):
        schedule._urllib_transport("GET", f"{server}/slow", None, {}, 0.2)


# Reading run history and issues --------------------------------------------------------


@pytest.mark.parametrize(
    "listing",
    [
        [],
        {"workflow_runs": {}},
        {"workflow_runs": [1]},
        {"workflow_runs": [{"id": "7", "status": "completed"}]},
        {"workflow_runs": [{"id": True, "status": "completed"}]},
        {"workflow_runs": [{"id": -1, "status": "completed"}]},
    ],
)
def test_previous_outcomes_rejects_malformed_listings(listing: Any) -> None:
    with pytest.raises(schedule.GitHubError):
        list(schedule.previous_outcomes(_gh(Script(_ok(listing))), run_id=1, branch="main"))


def test_previous_outcomes_skips_this_run_and_unfinished_runs() -> None:
    listing = {
        "workflow_runs": [
            {"id": 9, "status": "completed"},
            {"id": 8, "status": "in_progress"},
            {"id": 7, "status": "completed"},
        ]
    }
    script = Script(_ok(listing), _ok({"jobs": [_job("sweep", "failure")]}))
    found = list(schedule.previous_outcomes(_gh(script), run_id=9, branch="main"))
    assert found == [(7, schedule.Outcome.FAILURE)]
    assert script.requests[1][1].endswith("/actions/runs/7/jobs?filter=latest")


def test_previous_outcomes_reads_at_most_the_lookback() -> None:
    runs = [{"id": i, "status": "completed"} for i in range(100, 0, -1)]
    jobs = [_ok({"jobs": [_job("sweep", "failure")]})] * schedule.LOOKBACK_RUNS
    script = Script(_ok({"workflow_runs": runs}), *jobs)
    found = list(schedule.previous_outcomes(_gh(script), run_id=1000, branch="main"))
    assert len(found) == schedule.LOOKBACK_RUNS


def test_open_alert_issue_ignores_pull_requests_and_picks_the_oldest() -> None:
    issues = [
        {"number": 12, "state": "open"},
        {"number": 3, "state": "open", "pull_request": {}},
        {"number": 5, "state": "open"},
    ]
    assert schedule.open_alert_issue(_gh(Script(_ok(issues)))) == 5
    assert schedule.open_alert_issue(_gh(Script(_ok([])))) is None
    with pytest.raises(schedule.GitHubError):
        schedule.open_alert_issue(_gh(Script(_ok([{"number": "5", "state": "open"}]))))


# The alert -----------------------------------------------------------------------------


def _alert(script: Script, current: schedule.Outcome, **kw: Any) -> schedule.Action:
    return schedule.alert(
        _gh(script),
        run_id=50,
        branch="main",
        current=current,
        stage=kw.get("stage", "sweep"),
        record_failures=kw.get("record_failures"),
        server_url="https://github.com",
    )


def test_alert_makes_no_request_when_no_sweep_ran() -> None:
    script = Script()
    assert _alert(script, schedule.Outcome.NONE) is schedule.Action.NOTHING
    assert script.requests == []


def test_alert_opens_an_issue_creating_the_label_and_tolerating_a_race() -> None:
    history = {"workflow_runs": [{"id": 49, "status": "completed"}]}
    script = Script(
        _ok([]),  # no open issue
        _ok(history),
        _ok({"jobs": [_job("sweep", "failure")]}),
        (404, b"{}"),  # no label yet
        (422, b"{}"),  # created by someone else meanwhile
        (201, b'{"number": 8}'),
    )
    assert _alert(script, schedule.Outcome.FAILURE, stage="weird<script>", record_failures=3) is (
        schedule.Action.OPEN
    )
    method, url, body = script.requests[-1]
    assert (method, url) == ("POST", "https://api.github.com/repos/owner/repo/issues")
    assert body["labels"] == [schedule.ALERT_LABEL]
    assert "failed stage: `unknown`" in body["body"]  # unknown stage names are not echoed
    assert "https://github.com/owner/repo/actions/runs/49" in body["body"]


def test_alert_rejects_a_malformed_created_issue() -> None:
    script = Script(_ok([]), _ok({"workflow_runs": []}), _ok({}), (201, b"[]"))
    with pytest.raises(schedule.GitHubError):
        _alert(script, schedule.Outcome.FAILURE, record_failures=5)


def test_alert_command_needs_its_environment(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["alert", "--gate-result", "success", "--sweep-result", "failure"]
    env = {"GITHUB_REPOSITORY": REPO, "GITHUB_RUN_ID": "5", "GITHUB_REF_NAME": "main"}
    assert schedule.main(argv, env) == 1
    assert "GITHUB_TOKEN is not set" in capsys.readouterr().err
    assert schedule.main(argv, {**env, "GITHUB_TOKEN": "t", "GITHUB_RUN_ID": "x"}) == 1
    assert "GITHUB_RUN_ID" in capsys.readouterr().err


def test_alert_command_skips_runs_without_a_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["alert", "--gate-result", "success", "--sweep-result", "skipped"]
    env = {
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_RUN_ID": "5",
        "GITHUB_REF_NAME": "main",
        "GITHUB_TOKEN": "t",
        # a closed port: any request would fail, so none may be made
        "GITHUB_API_URL": "https://127.0.0.1:9",
    }
    assert schedule.main(argv, env) == 0
    assert "alert: nothing" in capsys.readouterr().out
