"""Acceptance tests for T-007 (scheduled sweep workflow and failure alerting). The task
contract: do not edit.

The workflow is checked statically (schedule, permissions, concurrency, timeout, pinned
actions, staged paths), and its data-branch steps are run for real against a local bare
repository. The alert logic runs against an in-memory GitHub. Scheduled runs and the
forced-failure test on GitHub are verified after merge (AC7).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from wearreport import aggregate, publish, schedule

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "sweep.yml"
REPO = "owner/repo"
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


# A reader for the subset of YAML the workflow uses ------------------------------------


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _children(lines: list[str], indent: int) -> dict[str, list[str]]:
    """Mapping keys at exactly `indent`, each with its own line and the lines below it."""
    out: dict[str, list[str]] = {}
    key: str | None = None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(rf"^ {{{indent}}}([\w-]+):", line)
        if m:
            key = m.group(1)
            out[key] = [line]
        elif key is not None and _indent(line) > indent:
            out[key].append(line)
        else:
            key = None
    return out


def _scalar(block: list[str]) -> str:
    value = block[0].split(":", 1)[1].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _steps(job: list[str]) -> list[dict[str, str]]:
    """Each step of a job: its keys (name, id, if, uses, run) as text; `run` dedented."""
    body = _children(job, 4)["steps"][1:]
    items: list[list[str]] = []
    for line in body:
        if re.match(r"^ {6}- ", line):
            items.append([" " * 8 + line[8:]])
        elif items:
            items[-1].append(line)
    steps = []
    for item in items:
        step: dict[str, str] = {}
        for key, block in _children(item, 8).items():
            if key == "run" and block[0].rstrip().endswith("|"):
                step[key] = "\n".join(line[10:] for line in block[1:]) + "\n"
            elif key in ("with", "env"):
                step[key] = "\n".join(block[1:])
            else:
                step[key] = _scalar(block)
        steps.append(step)
    return steps


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def top(text: str) -> dict[str, list[str]]:
    return _children(text.splitlines(), 0)


@pytest.fixture(scope="module")
def jobs(top: dict[str, list[str]]) -> dict[str, list[str]]:
    return _children(top["jobs"][1:], 2)


def _permissions(job: list[str]) -> dict[str, str]:
    block = _children(job, 4)["permissions"]
    return {k: _scalar(v) for k, v in _children(block[1:], 6).items()}


def _step(jobs: dict[str, list[str]], job: str, name_part: str) -> dict[str, str]:
    matches = [s for s in _steps(jobs[job]) if name_part in s.get("name", "")]
    assert len(matches) == 1, f"expected one step named like {name_part!r} in {job}"
    return matches[0]


# AC1: schedule and daytime gate --------------------------------------------------------


def test_ac1_cron_every_20_minutes_in_london_time(top: dict[str, list[str]]) -> None:
    on = "\n".join(top["on"])
    assert re.search(r'- cron: "7-59/20 6-20 \* \* \*"\n', on)
    assert "timezone:" not in on
    assert on.count("cron:") == 1
    # 7-20 hours x 3 runs an hour = 42 runs a day
    assert len(range(7, 21)) * len(range(0, 60, 20)) == 42


def test_ac1_gate_runs_first_and_guards_the_sweep(jobs: dict[str, list[str]]) -> None:
    gate = _step(jobs, "gate", "Gate")
    assert "python3 -m wearreport.schedule gate" in gate["run"]
    sweep = _children(jobs["sweep"], 4)
    assert _scalar(sweep["needs"]) == "gate"
    assert _scalar(sweep["if"]) == "needs.gate.outputs.open == 'true'"


# (UTC moment, open?) around both edges of the day, in GMT, BST and on both DST days.
GATE_CASES = [
    # GMT: London = UTC
    ("2026-01-15T06:59:59Z", False),
    ("2026-01-15T07:00:00Z", True),
    ("2026-01-15T20:59:59Z", True),
    ("2026-01-15T21:00:00Z", False),
    # BST: London = UTC + 1
    ("2026-07-15T05:59:59Z", False),
    ("2026-07-15T06:00:00Z", True),
    ("2026-07-15T19:59:59Z", True),
    ("2026-07-15T20:00:00Z", False),
    # Spring forward, Sunday 29 March 2026 (01:00 GMT becomes 02:00 BST)
    ("2026-03-29T01:30:00Z", False),
    ("2026-03-29T05:59:59Z", False),
    ("2026-03-29T06:00:00Z", True),
    ("2026-03-29T19:59:59Z", True),
    ("2026-03-29T20:00:00Z", False),
    # Fall back, Sunday 25 October 2026 (02:00 BST becomes 01:00 GMT)
    ("2026-10-25T01:30:00Z", False),
    ("2026-10-25T06:59:59Z", False),
    ("2026-10-25T07:00:00Z", True),
    ("2026-10-25T20:59:59Z", True),
    ("2026-10-25T21:00:00Z", False),
    # Midnight and noon
    ("2026-07-15T23:00:00Z", False),
    ("2026-01-15T12:00:00Z", True),
]


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


@pytest.mark.parametrize(("moment", "expected"), GATE_CASES)
def test_ac1_gate_for_fixed_dates(moment: str, expected: bool) -> None:
    assert schedule.is_open(_utc(moment)) is expected


def test_ac1_gate_agrees_with_the_publishers_daytime() -> None:
    for moment, _ in GATE_CASES:
        assert schedule.is_open(_utc(moment)) is publish.is_london_daytime(_utc(moment))


def test_ac1_gate_refuses_a_naive_time() -> None:
    with pytest.raises(ValueError):
        schedule.is_open(datetime(2026, 7, 15, 12, 0))


@pytest.mark.parametrize(
    ("moment", "line"),
    [("2026-07-15T06:00:00Z", "open=true"), ("2026-07-15T20:00:00Z", "open=false")],
)
def test_ac1_gate_command_writes_a_github_output_line(moment: str, line: str) -> None:
    result = _schedule_cli("gate", "--now", moment)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [line]


def test_ac1_gate_command_uses_the_standard_library_only() -> None:
    # The gate job has no dependencies installed: `python3 -m wearreport.schedule` must
    # not import numpy, onnxruntime or the rest of the engine.
    code = (
        "import sys, wearreport.schedule as s; "
        "bad = {'numpy', 'onnxruntime', 'cv2', 'wearreport.aggregate', 'wearreport.publish'}; "
        "print(sorted(bad & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=_env()
    )
    assert out.stdout.strip() == "[]"


# AC2: concurrency ----------------------------------------------------------------------


def test_ac2_one_concurrency_group_that_never_cancels_a_running_sweep(
    top: dict[str, list[str]], jobs: dict[str, list[str]]
) -> None:
    concurrency = _children(top["concurrency"][1:], 2)
    assert _scalar(concurrency["group"]) == "sweep"
    assert _scalar(concurrency["cancel-in-progress"]) == "false"
    for job in jobs.values():
        assert "concurrency" not in _children(job, 4)


# AC3: model cache and checksum ---------------------------------------------------------


def test_ac3_models_are_cached_and_verified_on_every_run(jobs: dict[str, list[str]]) -> None:
    steps = _steps(jobs["sweep"])
    names = [s.get("name", s.get("uses", "")) for s in steps]
    restore = next(i for i, s in enumerate(steps) if "actions/cache/restore@" in s.get("uses", ""))
    verify = next(i for i, s in enumerate(steps) if "fetch_model.sh" in s.get("run", ""))
    save = next(i for i, s in enumerate(steps) if "actions/cache/save@" in s.get("uses", ""))
    sweep = names.index("Sweep")
    assert restore < verify < save < sweep
    assert "if" not in steps[verify]  # the checksum is verified whether or not the cache hit
    assert steps[verify]["run"].strip() == "sh scripts/fetch_model.sh --with-m"
    for i in (restore, save):
        assert "path: .models" in steps[i]["with"]
        assert "key: yolox-${{ hashFiles('scripts/fetch_model.sh') }}" in steps[i]["with"]
    assert "--data-dir" in steps[sweep]["run"]  # the default model, yolox_m, is pinned


# AC4: timeout --------------------------------------------------------------------------


def test_ac4_sweep_job_times_out_after_15_minutes(jobs: dict[str, list[str]]) -> None:
    assert _scalar(_children(jobs["sweep"], 4)["timeout-minutes"]) == "15"
    for job in jobs.values():
        assert int(_scalar(_children(job, 4)["timeout-minutes"])) <= 15


# AC5: failure counter and alert issue --------------------------------------------------


@dataclass
class FakeGitHub:
    """An in-memory GitHub REST API: workflow runs with their jobs, labels, issues."""

    runs: list[dict[str, Any]] = field(default_factory=list)  # newest first
    labels: set[str] = field(default_factory=set)
    issues: dict[int, dict[str, Any]] = field(default_factory=dict)
    token_seen: set[str] = field(default_factory=set)

    def add_run(self, run_id: int, sweep: str | None, *, gate: str | None = "success") -> None:
        jobs = []
        if gate is not None:
            jobs.append({"name": "gate", "conclusion": gate, "steps": [{"name": "x"}]})
        if sweep is not None:
            jobs.append({"name": "sweep", "conclusion": sweep, "steps": [{"name": "x"}]})
        self.runs.insert(0, {"id": run_id, "status": "completed", "jobs": jobs})

    def open_issues(self) -> list[dict[str, Any]]:
        return [i for i in self.issues.values() if i["state"] == "open"]

    def __call__(
        self, method: str, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> tuple[int, bytes]:
        self.token_seen.add(headers.get("Authorization", ""))
        parts = urlsplit(url)
        path, query = parts.path, parse_qs(parts.query)
        payload = json.loads(body) if body else None
        base = f"/repos/{REPO}"
        if method == "GET" and path == f"{base}/actions/workflows/sweep.yml/runs":
            assert query.get("branch") == ["main"]
            runs = [{"id": r["id"], "status": r["status"]} for r in self.runs]
            return 200, json.dumps({"total_count": len(runs), "workflow_runs": runs}).encode()
        m = re.fullmatch(rf"{base}/actions/runs/(\d+)/jobs", path)
        if method == "GET" and m:
            run = next(r for r in self.runs if r["id"] == int(m.group(1)))
            return 200, json.dumps({"total_count": len(run["jobs"]), "jobs": run["jobs"]}).encode()
        if method == "GET" and path == f"{base}/issues":
            assert query.get("labels") == [schedule.ALERT_LABEL]
            assert query.get("state") == ["open"]
            found = [
                {"number": n, "state": i["state"], "labels": [{"name": x} for x in i["labels"]]}
                for n, i in sorted(self.issues.items(), reverse=True)
                if i["state"] == "open" and schedule.ALERT_LABEL in i["labels"]
            ]
            return 200, json.dumps(found).encode()
        m = re.fullmatch(rf"{base}/labels/([\w-]+)", path)
        if method == "GET" and m:
            if m.group(1) in self.labels:
                return 200, json.dumps({"name": m.group(1)}).encode()
            return 404, b'{"message": "Not Found"}'
        if method == "POST" and path == f"{base}/labels":
            assert payload is not None
            self.labels.add(payload["name"])
            return 201, json.dumps({"name": payload["name"]}).encode()
        if method == "POST" and path == f"{base}/issues":
            assert payload is not None
            if not set(payload.get("labels", [])) <= self.labels:
                return 422, b'{"message": "Validation Failed"}'
            number = len(self.issues) + 1
            self.issues[number] = {
                "state": "open",
                "title": payload["title"],
                "body": payload["body"],
                "labels": list(payload.get("labels", [])),
                "comments": [],
            }
            return 201, json.dumps({"number": number}).encode()
        m = re.fullmatch(rf"{base}/issues/(\d+)/comments", path)
        if method == "POST" and m:
            assert payload is not None
            self.issues[int(m.group(1))]["comments"].append(payload["body"])
            return 201, json.dumps({"id": 1}).encode()
        m = re.fullmatch(rf"{base}/issues/(\d+)", path)
        if method == "PATCH" and m:
            assert payload is not None
            self.issues[int(m.group(1))]["state"] = payload["state"]
            return 200, json.dumps({"number": int(m.group(1))}).encode()
        return 404, b'{"message": "Not Found"}'


def _alert(fake: FakeGitHub, run_id: int, current: schedule.Outcome, **kw: Any) -> schedule.Action:
    fake.add_run(run_id, None)  # the current run is still in progress
    fake.runs[0]["status"] = "in_progress"
    gh = schedule.GitHub(REPO, "t0ken", transport=fake, sleep=lambda _: None)
    action = schedule.alert(
        gh,
        run_id=run_id,
        branch="main",
        current=current,
        stage=kw.get("stage", "sweep"),
        record_failures=kw.get("record_failures"),
        server_url="https://github.com",
    )
    # afterwards the run is completed with the outcome it reported
    fake.runs[0]["status"] = "completed"
    sweep = {"success": "success", "failure": "failure", "none": None}[current.value]
    if sweep is not None:
        fake.runs[0]["jobs"].append({"name": "sweep", "conclusion": sweep, "steps": [{}]})
    return action


def test_ac5_three_consecutive_failures_open_one_labelled_issue() -> None:
    fake = FakeGitHub()
    fake.add_run(1, "success")
    assert _alert(fake, 2, schedule.Outcome.FAILURE) is schedule.Action.NOTHING
    assert _alert(fake, 3, schedule.Outcome.FAILURE) is schedule.Action.NOTHING
    assert fake.issues == {}
    assert _alert(fake, 4, schedule.Outcome.FAILURE, stage="publish") is schedule.Action.OPEN
    [issue] = fake.open_issues()
    assert issue["labels"] == [schedule.ALERT_LABEL]
    assert "3" in issue["title"]
    assert "https://github.com/owner/repo/actions/runs/4" in issue["body"]
    assert "publish" in issue["body"]
    assert "t0ken" not in issue["body"]
    assert fake.token_seen == {"Bearer t0ken"}


def test_ac5_further_failures_comment_on_the_open_issue() -> None:
    fake = FakeGitHub()
    for run_id in (1, 2, 3):
        _alert(fake, run_id, schedule.Outcome.FAILURE)
    assert _alert(fake, 4, schedule.Outcome.FAILURE) is schedule.Action.COMMENT
    [issue] = fake.open_issues()
    assert len(fake.issues) == 1
    assert len(issue["comments"]) == 1
    assert "4" in issue["comments"][0]


def test_ac5_a_later_success_closes_the_issue() -> None:
    fake = FakeGitHub()
    for run_id in (1, 2, 3):
        _alert(fake, run_id, schedule.Outcome.FAILURE)
    assert len(fake.open_issues()) == 1
    assert _alert(fake, 4, schedule.Outcome.SUCCESS) is schedule.Action.CLOSE
    assert fake.open_issues() == []
    assert fake.issues[1]["comments"]  # says why it closed
    # and the counter starts again from zero
    assert _alert(fake, 5, schedule.Outcome.FAILURE) is schedule.Action.NOTHING
    assert _alert(fake, 6, schedule.Outcome.FAILURE) is schedule.Action.NOTHING
    assert _alert(fake, 7, schedule.Outcome.FAILURE) is schedule.Action.OPEN
    assert len(fake.open_issues()) == 1


def test_ac5_gated_runs_neither_count_nor_break_a_streak() -> None:
    fake = FakeGitHub()
    fake.add_run(1, "failure")
    fake.add_run(2, "skipped")  # outside the day: the gate closed
    fake.add_run(3, None, gate=None)  # a queued run replaced by a newer one never started
    assert _alert(fake, 4, schedule.Outcome.FAILURE) is schedule.Action.NOTHING
    assert _alert(fake, 5, schedule.Outcome.FAILURE) is schedule.Action.OPEN


def test_ac5_a_failed_record_counts_even_without_run_history() -> None:
    # status.json already counts three failed sweeps: alert on the first failing run here.
    fake = FakeGitHub()
    action = _alert(fake, 1, schedule.Outcome.FAILURE, stage="record", record_failures=3)
    assert action is schedule.Action.OPEN


def test_ac5_success_without_an_open_issue_does_nothing() -> None:
    fake = FakeGitHub()
    assert _alert(fake, 1, schedule.Outcome.SUCCESS) is schedule.Action.NOTHING
    assert fake.issues == {}


@pytest.mark.parametrize(
    ("gate", "sweep", "expected"),
    [
        ("success", "success", schedule.Outcome.SUCCESS),
        ("success", "failure", schedule.Outcome.FAILURE),  # e.g. registry failure, exit 1
        ("success", "cancelled", schedule.Outcome.FAILURE),  # the 15 minute timeout
        ("success", "skipped", schedule.Outcome.NONE),  # outside the day
        ("failure", "skipped", schedule.Outcome.FAILURE),  # a broken gate
    ],
)
def test_ac5_current_outcome_from_job_results(
    gate: str, sweep: str, expected: schedule.Outcome
) -> None:
    assert schedule.current_outcome(gate, sweep) is expected


def _record(ok: int) -> Any:
    started = datetime(2026, 7, 15, 12, 0, 5, tzinfo=UTC)
    observations = [
        aggregate.Observation(f"C{i:04d}", None if i < ok else "timeout") for i in range(10)
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


@pytest.mark.parametrize(
    ("ok", "code", "line"), [(10, 0, "outcome=success"), (5, 1, "outcome=failure")]
)
def test_ac5_outcome_command_reads_consecutive_failures(
    tmp_path: Path, ok: int, code: int, line: str
) -> None:
    record = _record(ok)
    publish.publish(tmp_path, record, now=aggregate.parse_utc(record["finished_at"]))
    result = _schedule_cli("outcome", "--status", str(tmp_path / "status.json"))
    assert result.returncode == code
    assert line in result.stdout.splitlines()


def test_ac5_outcome_command_fails_without_a_status_file(tmp_path: Path) -> None:
    result = _schedule_cli("outcome", "--status", str(tmp_path / "status.json"))
    assert result.returncode == 1
    assert "outcome=failure" in result.stdout.splitlines()


# AC6: manual dispatch ------------------------------------------------------------------


def test_ac6_workflow_dispatch_with_a_force_fail_input(
    top: dict[str, list[str]], jobs: dict[str, list[str]]
) -> None:
    on = _children(top["on"][1:], 2)
    assert set(on) == {"schedule", "workflow_dispatch"}
    dispatch = "\n".join(on["workflow_dispatch"])
    assert re.search(r"force_fail:\n(\s+.*\n)*?\s+type: boolean", dispatch + "\n")
    assert re.search(r"default: false", dispatch)
    first = _steps(jobs["sweep"])[0]
    assert first["if"] == "inputs.force_fail"
    assert "exit 1" in first["run"]


# AC7: static workflow check ------------------------------------------------------------


def test_ac7_least_privilege_permissions(
    top: dict[str, list[str]], jobs: dict[str, list[str]]
) -> None:
    assert _scalar(top["permissions"]) == "{}"
    assert set(jobs) == {"gate", "sweep", "alert"}
    assert _permissions(jobs["gate"]) == {"contents": "read"}
    assert _permissions(jobs["sweep"]) == {"contents": "write"}
    assert _permissions(jobs["alert"]) == {"actions": "read", "contents": "read", "issues": "write"}


def test_ac7_every_action_is_pinned_and_every_job_on_ubuntu_24_04(
    text: str, jobs: dict[str, list[str]]
) -> None:
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses
    for ref in uses:
        assert PINNED.match(ref), ref
    for job in jobs.values():
        assert _scalar(_children(job, 4)["runs-on"]) == "ubuntu-24.04"
    for job in jobs.values():
        for step in _steps(job):
            if "actions/checkout@" in step.get("uses", ""):
                assert "persist-credentials: false" in step["with"]


def test_ac7_no_untrusted_triggers_and_no_expressions_in_scripts(
    text: str, jobs: dict[str, list[str]]
) -> None:
    assert "pull_request" not in text
    for job in jobs.values():
        for step in _steps(job):
            assert "${{" not in step.get("run", "")  # values reach scripts through env


def test_ac7_secrets_only_reach_the_sweep_step(jobs: dict[str, list[str]], text: str) -> None:
    assert set(re.findall(r"secrets\.(\w+)", text)) == {"METOFFICE_API_KEY", "TFL_APP_KEY"}
    env = _step(jobs, "sweep", "Sweep")["env"]
    assert "secrets.METOFFICE_API_KEY" in env and "secrets.TFL_APP_KEY" in env


def test_ac7_only_explicit_paths_are_staged(jobs: dict[str, list[str]], text: str) -> None:
    assert not re.search(r"git add\s+(-A|--all|\.(\s|$)|-u)", text)
    run = _step(jobs, "sweep", "Commit and push")["run"]
    adds = re.findall(r"git add(.*)", run)
    assert adds == [" --", " -- status.json"]
    assert "':(glob)sweeps/**/*.json'" in run
    assert "python3 -m wearreport.schedule check-staged" in run


@pytest.mark.parametrize(
    "lines",
    [
        ["A\tsweeps/2026/07/15/20260715T1200Z.json", "A\tstatus.json"],
        ["A\tsweeps/2026/07/15/20260715T1200Z.json", "M\tstatus.json"],
        [],
    ],
)
def test_ac7_check_staged_accepts_new_records_and_status(lines: list[str]) -> None:
    schedule.check_staged(lines)


@pytest.mark.parametrize(
    "lines",
    [
        ["M\tsweeps/2026/07/15/20260715T1200Z.json"],  # records are immutable
        ["D\tsweeps/2026/07/15/20260715T1200Z.json"],
        ["D\tstatus.json"],
        ["A\tsweeps/2026/07/15/.20260715T1200Z.json.0123456789abcdef.tmp"],
        ["A\tsweeps/2026/07/16/20260715T1200Z.json"],  # not under its own date
        ["A\tREADME.md"],
        ["A\tframe.jpg"],
        ["R100\tstatus.json\tstatus2.json"],
        ["garbage"],
    ],
)
def test_ac7_check_staged_rejects_anything_else(lines: list[str]) -> None:
    with pytest.raises(schedule.ScheduleError):
        schedule.check_staged(lines)


# AC8: data branch: orphan on first publish, shallow and sparse, write permissions -------


def _env(**extra: str) -> dict[str, str]:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", os.devnull),
        "PYTHONPATH": str(ROOT / "engine"),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "LANG": "C.UTF-8",
    }
    env.update(extra)
    return env


def _schedule_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wearreport.schedule", *args],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=60,
    )


def _tool(name: str) -> str:
    path = shutil.which(name)
    assert path is not None, f"{name} is required"
    return path


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        [_tool("git"), *args], cwd=cwd, capture_output=True, text=True, check=True, env=_env()
    )
    return out.stdout


def _run_step(script: str, tmp: Path, remote: Path, data_dir: Path) -> None:
    shim = tmp / "bin"
    shim.mkdir(exist_ok=True)
    python3 = shim / "python3"
    if not python3.exists():
        python3.symlink_to(sys.executable)
    env = _env(
        PATH=f"{shim}{os.pathsep}{os.environ['PATH']}",
        DATA_DIR=str(data_dir),
        DATA_REMOTE=str(remote),
        GITHUB_WORKSPACE=str(ROOT),
        GITHUB_RUN_ID="42",
        GITHUB_TOKEN="dummy",  # noqa: S106
        GITHUB_OUTPUT=str(tmp / "output"),
    )
    result = subprocess.run(
        [_tool("bash"), "-e", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def _record_at(started: datetime) -> Any:
    observations = [aggregate.Observation(f"C{i:04d}", None) for i in range(10)]
    return aggregate.build_record(
        observations,
        started_at=started,
        finished_at=started + timedelta(minutes=4),
        weather=None,
        engine_version="0.0.0",
        model_name="yolox_m",
        model_sha256="b" * 64,
    )


def test_ac8_data_branch_is_created_as_an_orphan_then_checked_out_shallow_and_sparse(
    tmp_path: Path, jobs: dict[str, list[str]]
) -> None:
    checkout = _step(jobs, "sweep", "Check out the data branch")["run"]
    commit = _step(jobs, "sweep", "Commit and push")["run"]
    remote = tmp_path / "remote.git"
    subprocess.run([_tool("git"), "init", "-q", "--bare", str(remote)], check=True, env=_env())
    now = datetime.now(UTC).replace(microsecond=0)

    # First publish: no data branch yet, so it starts as an orphan.
    first = tmp_path / "run1"
    _run_step(checkout, tmp_path, remote, first)
    old = _record_at(now - timedelta(days=5))
    publish.publish(first, old, now=now - timedelta(days=5))
    _run_step(commit, tmp_path, remote, first)
    root = _git(remote, "rev-list", "--max-parents=0", "data").split()
    assert root == [_git(remote, "rev-parse", "data").strip()]  # one commit, no parent
    assert _git(remote, "branch", "--list").split() == ["data"]  # nothing else touched

    # Second publish: shallow and sparse, and only the new record and status.json change.
    second = tmp_path / "run2"
    _run_step(checkout, tmp_path, remote, second)
    assert _git(second, "rev-parse", "--is-shallow-repository").strip() == "true"
    assert not publish.record_path(second, old["sweep_id"]).exists()  # outside the sparse set
    new = _record_at(now - timedelta(minutes=5))
    publish.publish(second, new, now=now)
    stale = publish.record_path(second, new["sweep_id"]).parent / ".x.json.0123456789abcdef.tmp"
    stale.write_text("partial")
    (second / "notes.txt").write_text("not data")
    _run_step(commit, tmp_path, remote, second)
    changed = _git(remote, "diff", "--name-status", "data~1", "data").splitlines()
    new_path = publish.record_path(Path(), new["sweep_id"]).as_posix()
    assert sorted(changed) == sorted([f"A\t{new_path}", "M\tstatus.json"])
    tree = _git(remote, "ls-tree", "-r", "--name-only", "data").split()
    assert publish.record_path(Path(), old["sweep_id"]).as_posix() in tree  # history kept

    # Publishing the identical record again commits nothing.
    third = tmp_path / "run3"
    _run_step(checkout, tmp_path, remote, third)
    head = _git(remote, "rev-parse", "data")
    publish.publish(third, new, now=now)
    status = json.loads((third / "status.json").read_text())
    assert status["last_sweep_id"] == new["sweep_id"]
    _run_step(commit, tmp_path, remote, third)
    assert _git(remote, "rev-parse", "data") == head


def test_ac8_only_the_publishing_job_writes_contents_and_only_alert_writes_issues(
    jobs: dict[str, list[str]],
) -> None:
    writers = {name for name, job in jobs.items() if _permissions(job).get("contents") == "write"}
    issues = {name for name, job in jobs.items() if _permissions(job).get("issues") == "write"}
    assert writers == {"sweep"}
    assert issues == {"alert"}
    run = _step(jobs, "sweep", "Commit and push")["run"]
    assert re.findall(r"\bpush\b[^\n]*", run) == ["push -q origin HEAD:refs/heads/data"]
