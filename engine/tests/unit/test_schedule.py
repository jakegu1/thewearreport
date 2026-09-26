"""The scheduled sweep's helpers (wearreport.schedule): gate, outcomes, staged paths,
the GitHub client and the alert."""

from __future__ import annotations

import http.client
import http.server
import io
import json
import socketserver
import threading
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

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


@pytest.mark.parametrize(
    ("event", "ref", "default_branch", "allowed"),
    [
        ("schedule", "refs/heads/main", "main", True),
        ("schedule", "", "", True),  # GitHub starts scheduled runs on the default branch
        ("workflow_dispatch", "refs/heads/main", "main", True),
        ("workflow_dispatch", "refs/heads/task/t-999-x", "main", False),
        ("workflow_dispatch", "refs/tags/main", "main", False),
        ("workflow_dispatch", "refs/heads/main", "", False),  # unknown default: refuse
        ("workflow_dispatch", "refs/heads/", "", False),
        ("workflow_dispatch", "refs/heads/mainx", "main", False),
        ("push", "refs/heads/task/x", "main", False),
    ],
)
def test_ref_allowed(event: str, ref: str, default_branch: str, allowed: bool) -> None:
    assert schedule.ref_allowed(event, ref, default_branch) is allowed


def test_gate_command_refuses_a_manual_run_from_another_branch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    noon = ["--now", "2026-07-15T12:00:00Z"]
    args = ["--event", "workflow_dispatch", "--default-branch", "main"]
    assert schedule.main(["gate", *noon, *args, "--ref", "refs/heads/task/x"]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == ["open=false"]
    assert "refs/heads/task/x" in err
    assert schedule.main(["gate", *noon, *args, "--ref", "refs/heads/main"]) == 0
    assert capsys.readouterr().out.splitlines() == ["open=true"]


def test_gate_command_needs_all_three_ref_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert schedule.main(["gate", "--event", "workflow_dispatch", "--ref", "refs/heads/x"]) == 1
    assert "go together" in capsys.readouterr().err


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


# The sparse data branch ----------------------------------------------------------------


def test_success_rule_and_record_cap_match_the_publisher() -> None:
    assert (schedule.SUCCESS_NUMERATOR, schedule.SUCCESS_DENOMINATOR) == (
        publish.SUCCESS_NUMERATOR,
        publish.SUCCESS_DENOMINATOR,
    )
    assert schedule.MAX_RECORD_BYTES == publish.MAX_RECORD_BYTES


def _sweep(started: datetime, usable: int) -> Any:
    observations = [
        aggregate.Observation(f"C{i:04d}", None if i < usable else "timeout") for i in range(10)
    ]
    return aggregate.build_record(
        observations,
        started_at=started,
        finished_at=started + timedelta(minutes=4),
        weather=None,
        engine_version="0.0.0",
        model_name="yolox_m",
        model_sha256="b" * 64,
    )


@pytest.mark.parametrize("usable", [0, 5, 8, 9, 10])
def test_record_succeeded_agrees_with_the_publisher(tmp_path: Path, usable: int) -> None:
    record = _sweep(datetime(2026, 7, 15, 12, 0, tzinfo=UTC), usable)
    path = tmp_path / "r.json"
    path.write_bytes(publish.serialize(record))
    assert schedule.record_succeeded(path) is publish.is_success(record)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not json",
        b"\xff\xfe",
        b"[" * 100_000 + b"]" * 100_000,
        b"[]",
        b'{"cameras_listed": 0, "frames_ok": 0}',
        b'{"cameras_listed": 10, "frames_ok": true}',
        b'{"cameras_listed": 10.0, "frames_ok": 10}',
        b'{"cameras_listed": "10", "frames_ok": 10}',
        b'{"frames_ok": 10}',
        b'{"cameras_listed": 10, "frames_ok": 10}' + b" " * schedule.MAX_RECORD_BYTES,
    ],
)
def test_record_succeeded_is_false_for_malformed_files(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "r.json"
    path.write_bytes(data)
    assert schedule.record_succeeded(path) is False


def test_record_succeeded_is_false_for_a_missing_file_or_a_directory(tmp_path: Path) -> None:
    assert schedule.record_succeeded(tmp_path / "absent.json") is False
    assert schedule.record_succeeded(tmp_path) is False


def _put(data_dir: Path, record: Any) -> Path:
    path = publish.record_path(data_dir, record["sweep_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(publish.serialize(record))
    return path


def test_streak_closed_by_a_successful_record(tmp_path: Path) -> None:
    started = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    _put(tmp_path, _sweep(started, 5))
    assert schedule.streak_closed(tmp_path) is False
    _put(tmp_path, _sweep(started - timedelta(days=1), 10))
    assert schedule.streak_closed(tmp_path) is True


def test_streak_closed_by_a_status_whose_newest_sweep_succeeded(tmp_path: Path) -> None:
    assert schedule.streak_closed(tmp_path) is False  # nothing at all: keep looking
    _status(tmp_path, b'{"consecutive_failures": 2}')
    assert schedule.streak_closed(tmp_path) is False
    _status(tmp_path, b'{"consecutive_failures": 0}')
    assert schedule.streak_closed(tmp_path) is True


def test_streak_closed_ignores_misplaced_and_symlinked_records(tmp_path: Path) -> None:
    record = _sweep(datetime(2026, 7, 15, 12, 0, tzinfo=UTC), 10)
    real = _put(tmp_path / "elsewhere", record)
    # under the wrong date
    wrong_date = tmp_path / "wrong-date"
    moved = wrong_date / "sweeps" / "2026" / "07" / "16" / real.name
    moved.parent.mkdir(parents=True)
    moved.write_bytes(real.read_bytes())
    assert schedule.streak_closed(wrong_date) is False
    # a symlinked record file
    link_file = tmp_path / "link-file"
    linked = publish.record_path(link_file, record["sweep_id"])
    linked.parent.mkdir(parents=True)
    linked.symlink_to(real)
    assert schedule.streak_closed(link_file) is False
    # a record reached through a symlinked directory
    link_dir = tmp_path / "link-dir"
    (link_dir / "sweeps").mkdir(parents=True)
    (link_dir / "sweeps" / "2026").symlink_to(tmp_path / "elsewhere" / "sweeps" / "2026")
    assert publish.record_path(link_dir, record["sweep_id"]).is_file()
    assert schedule.streak_closed(link_dir) is False
    # and the real one does close the streak
    assert schedule.streak_closed(tmp_path / "elsewhere") is True


def test_streak_closed_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert schedule.main(["streak-closed", "--data-dir", str(tmp_path)]) == 1
    _status(tmp_path, b'{"consecutive_failures": 0}')
    assert schedule.main(["streak-closed", "--data-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == ["streak-closed: no", "streak-closed: yes"]


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


# Raw HTTP replies a well-behaved server never sends; http.client raises HTTPException
# subclasses (not OSError or ValueError) for each.
BROKEN_REPLIES = {
    "/garbage": b"garbage\r\n\r\n",  # BadStatusLine
    "/chunked": (  # IncompleteRead: a chunk shorter than announced
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n10\r\nshort"
    ),
    "/longheader": b"HTTP/1.1 200 OK\r\nX-Long: " + b"a" * 70_000 + b"\r\n\r\n",  # LineTooLong
}


class _BrokenHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        request_line = self.rfile.readline().decode("latin-1")
        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
            pass
        path = request_line.split(" ")[1] if " " in request_line else "/"
        self.wfile.write(BROKEN_REPLIES.get(path, b"garbage\r\n\r\n"))


@pytest.fixture
def broken_server() -> Iterator[str]:
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _BrokenHandler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("path", sorted(BROKEN_REPLIES))
def test_urllib_transport_raises_http_exceptions_on_broken_replies(
    broken_server: str, path: str
) -> None:
    with pytest.raises(http.client.HTTPException):
        schedule._urllib_transport("GET", f"{broken_server}{path}", None, {}, 5.0)


@pytest.mark.parametrize("path", sorted(BROKEN_REPLIES))
def test_client_turns_broken_replies_into_github_errors_after_retrying(
    broken_server: str, path: str
) -> None:
    calls: list[str] = []

    def transport(
        method: str, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> tuple[int, bytes]:
        calls.append(method)
        return schedule._urllib_transport(method, f"{broken_server}{path}", body, headers, timeout)

    with pytest.raises(schedule.GitHubError, match="3 attempts") as info:
        schedule.GitHub(REPO, "s3cret", transport=transport, sleep=lambda _: None).request(
            "GET", "/x"
        )
    assert "s3cret" not in str(info.value)
    assert calls == ["GET", "GET", "GET"]  # within the existing bound


@pytest.mark.parametrize(
    "exc",
    [
        http.client.BadStatusLine("garbage"),
        http.client.IncompleteRead(b"short", 16),
        http.client.LineTooLong("header line"),
        http.client.RemoteDisconnected("closed"),
    ],
)
def test_client_retries_a_get_after_an_http_exception(exc: Exception) -> None:
    script = Script(exc, _ok({"ok": 1}))
    assert _gh(script).request("GET", "/x") == (200, {"ok": 1})
    post = Script(exc, _ok({"ok": 1}))
    with pytest.raises(schedule.GitHubError, match=r"1 attempt$"):
        _gh(post).request("POST", "/x", body={})  # a POST is still sent once
    assert len(post.replies) == 1


# Reading run history and issues --------------------------------------------------------


@pytest.mark.parametrize("name", [["x"], {"a": 1}, None, 7, True, 1.5])
def test_run_outcome_rejects_a_job_name_that_is_not_a_string(name: Any) -> None:
    with pytest.raises(schedule.GitHubError):
        schedule.run_outcome([_job("gate", "success"), {"name": name, "conclusion": "success"}])


@pytest.mark.parametrize("name", [["x"], {"a": 1}, None, 7])
def test_previous_outcomes_rejects_hostile_job_names(name: Any) -> None:
    script = Script(
        _ok({"workflow_runs": [{"id": 7, "status": "completed"}]}),
        _ok({"jobs": [{"name": name, "conclusion": "failure", "steps": []}]}),
    )
    with pytest.raises(schedule.GitHubError):
        list(schedule.previous_outcomes(_gh(script), run_id=8, branch="main"))


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


# Alert noise: while the issue is open ---------------------------------------------------


T0 = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class Tracker:
    """An in-memory GitHub that stamps issues and comments with a settable clock and an
    author, and lists comments with `since` the way the REST API does. Every earlier run
    failed, so the streak is always past the threshold."""

    def __init__(self) -> None:
        self.now = T0
        self.issue: dict[str, Any] | None = None
        self.comments: list[dict[str, Any]] = []
        self.comment_listing: tuple[int, bytes] | None = None  # override the reply

    def __call__(
        self, method: str, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> tuple[int, bytes]:
        path, _, query = url.removeprefix(f"https://api.github.com/repos/{REPO}").partition("?")
        payload = json.loads(body) if body else None
        if method == "GET" and path == "/issues":
            return _ok([] if self.issue is None else [self.issue])
        if method == "GET" and path.startswith("/actions/workflows/"):
            return _ok({"workflow_runs": [{"id": 1, "status": "completed"}]})
        if method == "GET" and path == "/actions/runs/1/jobs":
            return _ok({"jobs": [_job("gate", "success"), _job("sweep", "failure")]})
        if method == "GET" and path.startswith("/labels/"):
            return _ok({"name": schedule.ALERT_LABEL})
        if method == "POST" and path == "/issues":
            assert payload is not None
            self.issue = {"number": 3, "state": "open", "created_at": _iso(self.now)}
            self.issue["body"] = payload["body"]
            return 201, b'{"number": 3}'
        if method == "GET" and path == "/issues/3/comments":
            if self.comment_listing is not None:
                return self.comment_listing
            since = _utc(parse_qs(query)["since"][0])
            return _ok([c for c in self.comments if _utc(c["created_at"]) >= since])
        if method == "POST" and path == "/issues/3/comments":
            assert payload is not None
            self.comment(payload["body"])
            return 201, b"{}"
        raise AssertionError(f"unexpected request {method} {path}")

    def comment(self, body: str, login: str = schedule.NOTICE_AUTHOR) -> None:
        self.comments.append({"body": body, "created_at": _iso(self.now), "user": {"login": login}})

    def fail(self, minutes: int, stage: str) -> schedule.Action:
        self.now = T0 + timedelta(minutes=minutes)
        return schedule.alert(
            schedule.GitHub(REPO, "s3cret", transport=self, sleep=lambda _: None),
            run_id=100 + minutes,
            branch="main",
            current=schedule.Outcome.FAILURE,
            stage=stage,
            record_failures=5,
            server_url="https://github.com",
            now=self.now,
        )


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def test_an_open_issue_is_noted_on_a_stage_change_or_hourly() -> None:
    gh = Tracker()
    quiet, comment = schedule.Action.QUIET, schedule.Action.COMMENT
    assert gh.fail(0, "sweep") is schedule.Action.OPEN
    assert "<!-- ops-alert stage=sweep -->" in gh.issue["body"]  # type: ignore[index]
    assert gh.fail(20, "sweep") is quiet  # same stage, 20 minutes after the issue
    assert gh.fail(40, "publish") is comment  # the stage changed
    assert gh.fail(60, "publish") is quiet
    assert gh.fail(80, "publish") is quiet
    assert gh.fail(100, "publish") is comment  # an hour since the last notice
    assert gh.fail(120, "publish") is quiet
    assert gh.fail(140, "sweep") is comment
    assert [schedule._notice_stage(c["body"]) for c in gh.comments] == [
        "publish",
        "publish",
        "sweep",
    ]
    # at most one comment an hour per stage: 3 comments in 140 minutes, not 7
    assert len(gh.comments) == 3


def test_notices_from_anyone_else_do_not_quiet_the_alert() -> None:
    gh = Tracker()
    gh.fail(0, "sweep")
    gh.issue["created_at"] = _iso(T0 - timedelta(hours=2))  # type: ignore[index]
    gh.now = T0 + timedelta(minutes=10)
    gh.comment("quoting <!-- ops-alert stage=sweep -->", login="someone")
    gh.comment("no marker here")
    assert gh.fail(20, "sweep") is schedule.Action.COMMENT


def test_a_gone_issue_has_no_notices() -> None:
    gh = Tracker()
    gh.fail(0, "sweep")
    gh.comment_listing = (404, b'{"message": "Not Found"}')
    gh.issue["created_at"] = None  # type: ignore[index]
    assert gh.fail(20, "sweep") is schedule.Action.COMMENT


@pytest.mark.parametrize(
    "listing",
    [
        _ok({"comments": []}),
        _ok([1]),
        _ok(
            [
                {
                    "body": "<!-- ops-alert stage=sweep -->",
                    "created_at": 5,
                    "user": {"login": schedule.NOTICE_AUTHOR},
                }
            ]
        ),
        _ok(
            [
                {
                    "body": "<!-- ops-alert stage=sweep -->",
                    "created_at": "yesterday",
                    "user": {"login": schedule.NOTICE_AUTHOR},
                }
            ]
        ),
        _ok(
            [
                {
                    "body": "x",
                    "created_at": "2026-13-40T99:00:00Z",
                    "user": {"login": schedule.NOTICE_AUTHOR},
                }
            ]
        ),
        (200, b"[" * 100_000),
    ],
)
def test_last_notice_rejects_malformed_comment_listings(listing: tuple[int, bytes]) -> None:
    gh = Tracker()
    gh.fail(0, "sweep")
    gh.comment_listing = listing
    with pytest.raises(schedule.GitHubError):
        gh.fail(20, "sweep")


@pytest.mark.parametrize("created_at", [7, "2026-07-15 09:00", "2026-02-30T09:00:00Z"])
def test_last_notice_rejects_a_malformed_issue_time(created_at: Any) -> None:
    gh = Tracker()
    gh.fail(0, "sweep")
    gh.issue["created_at"] = created_at  # type: ignore[index]
    with pytest.raises(schedule.GitHubError):
        gh.fail(20, "sweep")


@pytest.mark.parametrize(
    ("stage", "notice", "minutes_later", "expected"),
    [
        ("sweep", None, 0, True),
        ("sweep", "sweep", 20, False),
        ("sweep", "sweep", 59, False),
        ("sweep", "sweep", 60, True),
        ("sweep", "publish", 1, True),
    ],
)
def test_worth_a_comment(
    stage: str, notice: str | None, minutes_later: int, expected: bool
) -> None:
    last = None if notice is None else (T0, notice)
    now = T0 + timedelta(minutes=minutes_later)
    assert schedule.worth_a_comment(stage, last, now) is expected


@pytest.mark.parametrize(
    ("body", "stage"),
    [
        ("<!-- ops-alert stage=sweep -->", "sweep"),
        ("<!-- ops-alert stage=sweep -->\n<!-- ops-alert stage=publish -->", "publish"),
        ("<!-- ops-alert stage=<script> -->", None),
        ("<!-- ops-alert stage=bogus -->", None),
        (None, None),
        (["<!-- ops-alert stage=sweep -->"], None),
    ],
)
def test_notice_stage(body: Any, stage: str | None) -> None:
    assert schedule._notice_stage(body) == stage
