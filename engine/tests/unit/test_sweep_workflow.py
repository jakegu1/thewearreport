"""sweep.yml's own scripts, run for real: the gate refuses manual runs from other
branches, the data-branch checkout widens its sparse window until the failure
streak's start is in it, so the published consecutive_failures traces back to the
records (INV-6), and a sweep skipped as too soon after the last one (T-039) publishes
nothing and is ignored by the alert. The sweep job checks out and sweeps; the publish
job commits and pushes (T-034).

Uses the acceptance tests' reader for the workflow and their step runner, so these tests
run exactly the scripts the acceptance tests run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.acceptance.test_t_007 import (
    REPO,
    ROOT,
    WORKFLOW,
    FakeGitHub,
    _children,
    _env,
    _git,
    _run_step,
    _steps,
    _tool,
)
from wearreport import aggregate, cli, publish, schedule


@pytest.fixture(scope="module")
def jobs() -> dict[str, list[str]]:
    top = _children(WORKFLOW.read_text(encoding="utf-8").splitlines(), 0)
    return _children(top["jobs"][1:], 2)


def _script(jobs: dict[str, list[str]], job: str, name_part: str) -> str:
    [step] = [s for s in _steps(jobs[job]) if name_part in s.get("name", "")]
    return step["run"]


def _gate(jobs: dict[str, list[str]], tmp_path: Path, event: str, ref: str) -> list[str]:
    """Run the gate step's script at noon London time as `event` on `ref`, with the
    values its env maps from the github context. Returns its GITHUB_OUTPUT lines."""
    [step] = [s for s in _steps(jobs["gate"]) if s.get("id") == "gate"]
    env_map = dict(re.findall(r"^\s*(\w+): \$\{\{ ([\w.]+) \}\}$", step["env"], re.MULTILINE))
    assert env_map == {
        "EVENT": "github.event_name",
        "REF": "github.ref",
        "DEFAULT_BRANCH": "github.event.repository.default_branch",
    }
    context = {
        "github.event_name": event,
        "github.ref": ref,
        "github.event.repository.default_branch": "main",
    }
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    (shim / "python3").unlink(missing_ok=True)
    # a fixed clock: `gate` reads --now only from its arguments, so wrap python3
    (shim / "python3").write_text(
        f'#!/bin/sh\nif [ "$3" = gate ]; then set -- "$@" --now 2026-07-15T11:00:00Z; fi\n'
        f'exec {sys.executable} "$@"\n'
    )
    (shim / "python3").chmod(0o755)
    output = tmp_path / "output"
    output.write_text("")
    env = _env(
        PATH=f"{shim}{os.pathsep}{os.environ['PATH']}",
        GITHUB_OUTPUT=str(output),
        **{name: context[value] for name, value in env_map.items()},
    )
    del env["PYTHONPATH"]  # the step sets its own
    result = subprocess.run(
        [_tool("bash"), "-e", "-c", step["run"]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return output.read_text().splitlines()


@pytest.mark.parametrize(
    ("event", "ref", "expected"),
    [
        ("schedule", "refs/heads/main", "open=true"),
        ("workflow_dispatch", "refs/heads/main", "open=true"),
        ("workflow_dispatch", "refs/heads/task/t-999-x", "open=false"),
        ("workflow_dispatch", "refs/tags/v1", "open=false"),
    ],
)
def test_the_gate_step_refuses_manual_runs_from_other_refs(
    tmp_path: Path, jobs: dict[str, list[str]], event: str, ref: str, expected: str
) -> None:
    assert _gate(jobs, tmp_path, event, ref) == [expected]


def _record(started: datetime, *, ok: bool) -> Any:
    # 10 cameras; a failed sweep has 5 usable frames (below the 90% rule).
    observations = [
        aggregate.Observation(f"C{i:04d}", None if ok or i < 5 else "timeout") for i in range(10)
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


def _history(tmp_path: Path, jobs: dict[str, list[str]], records: list[Any]) -> Path:
    """A bare remote whose `data` branch holds `records` (oldest first) and the
    status.json the publisher wrote after the last of them."""
    remote = tmp_path / "remote.git"
    subprocess.run([_tool("git"), "init", "-q", "--bare", str(remote)], check=True, env=_env())
    seed = tmp_path / "seed"
    _run_step(_script(jobs, "sweep", "Check out the data branch"), tmp_path, remote, seed)
    for record in records:
        publish.publish(seed, record, now=aggregate.parse_utc(record["finished_at"]))
    _run_step(_script(jobs, "publish", "Commit and push"), tmp_path, remote, seed)
    return remote


def _publish_now(
    tmp_path: Path, jobs: dict[str, list[str]], remote: Path, record: Any, now: datetime
) -> tuple[Path, dict[str, Any]]:
    """Run the workflow's checkout, publish `record` as the Sweep step would, then run
    its commit step. Returns the data directory and the published status.json."""
    data_dir = tmp_path / "run"
    _run_step(_script(jobs, "sweep", "Check out the data branch"), tmp_path, remote, data_dir)
    publish.publish(data_dir, record, now=now)
    _run_step(_script(jobs, "publish", "Commit and push"), tmp_path, remote, data_dir)
    status = json.loads(_git(remote, "show", "data:status.json"))
    assert isinstance(status, dict)
    return data_dir, status


def test_a_streak_older_than_the_sparse_window_is_counted_in_full(
    tmp_path: Path, jobs: dict[str, list[str]]
) -> None:
    # The adversarial review's reproduction: a success 6 days ago, failures 5 and 4 days
    # ago, then one more failed sweep now. Three failures in a row, not one.
    now = datetime.now(UTC).replace(microsecond=0)
    older = _record(now - timedelta(days=8), ok=True)
    success = _record(now - timedelta(days=6), ok=True)
    failures = [_record(now - timedelta(days=d), ok=False) for d in (5, 4)]
    remote = _history(tmp_path, jobs, [older, success, *failures])

    data_dir, status = _publish_now(
        tmp_path, jobs, remote, _record(now - timedelta(minutes=5), ok=False), now
    )
    assert status["consecutive_failures"] == 3
    # widened exactly to the success: the older record is still outside the tree
    assert publish.record_path(data_dir, success["sweep_id"]).exists()
    assert not publish.record_path(data_dir, older["sweep_id"]).exists()
    assert _git(data_dir, "rev-parse", "--is-shallow-repository").strip() == "true"


def test_a_streak_with_no_success_counts_the_whole_history(
    tmp_path: Path, jobs: dict[str, list[str]]
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    failures = [_record(now - timedelta(days=d), ok=False) for d in (9, 7, 5)]
    remote = _history(tmp_path, jobs, failures)
    _, status = _publish_now(
        tmp_path, jobs, remote, _record(now - timedelta(minutes=5), ok=False), now
    )
    assert status["consecutive_failures"] == 4


def test_no_widening_after_a_published_success(tmp_path: Path, jobs: dict[str, list[str]]) -> None:
    # status.json says the newest sweep succeeded: older records cannot change the count.
    now = datetime.now(UTC).replace(microsecond=0)
    old_failure = _record(now - timedelta(days=6), ok=False)
    success = _record(now - timedelta(days=5), ok=True)
    remote = _history(tmp_path, jobs, [old_failure, success])
    data_dir, status = _publish_now(
        tmp_path, jobs, remote, _record(now - timedelta(minutes=5), ok=False), now
    )
    assert status["consecutive_failures"] == 1
    assert not publish.record_path(data_dir, success["sweep_id"]).exists()
    assert not publish.record_path(data_dir, old_failure["sweep_id"]).exists()


# A sweep skipped as too soon after the last one (T-039) --------------------------------


def _step_named(jobs: dict[str, list[str]], name: str, job: str = "sweep") -> dict[str, str]:
    [step] = [s for s in _steps(jobs[job]) if s.get("name") == name]
    return step


def _sweep_step(tmp_path: Path, jobs: dict[str, list[str]], uv: str) -> tuple[int, list[str]]:
    """Run the Sweep step's script with `uv` standing in for uv (a shell script body).
    Returns its exit status and its GITHUB_OUTPUT lines."""
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    (shim / "uv").write_text(f"#!/bin/sh\n{uv}")
    (shim / "uv").chmod(0o755)
    output = tmp_path / "output"
    output.write_text("")
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir(exist_ok=True)
    env = _env(
        PATH=f"{shim}{os.pathsep}{os.environ['PATH']}",
        DATA_DIR=str(tmp_path / "data"),
        RUNNER_TEMP=str(runner_temp),
        GITHUB_OUTPUT=str(output),
    )
    result = subprocess.run(
        [_tool("bash"), "-e", "-c", _step_named(jobs, "Sweep")["run"]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return result.returncode, output.read_text().splitlines()


def test_the_sweep_step_reports_a_real_skip(tmp_path: Path, jobs: dict[str, list[str]]) -> None:
    # The real command: status.json says the last sweep started a minute ago.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    last = datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=1)
    publish.publish(data_dir, _record(last, ok=True), now=last + timedelta(minutes=4))
    before = sorted(p.relative_to(data_dir) for p in data_dir.rglob("*"))
    # uv run --locked --no-dev wearreport ARGS...: run the engine's CLI with ARGS
    uv = f'shift 4\nexec {sys.executable} -m wearreport.cli "$@"\n'
    status, output = _sweep_step(tmp_path, jobs, uv)
    assert status == 0
    assert output == ["skipped=true"]
    assert sorted(p.relative_to(data_dir) for p in data_dir.rglob("*")) == before


def test_the_sweep_step_reports_no_skip_for_a_sweep_that_ran(
    tmp_path: Path, jobs: dict[str, list[str]]
) -> None:
    status, output = _sweep_step(tmp_path, jobs, "echo 'sweep_id: 20260715T1227Z'\n")
    assert (status, output) == (0, [])


def test_the_sweep_step_fails_when_the_sweep_fails(
    tmp_path: Path, jobs: dict[str, list[str]]
) -> None:
    # A failure after the skip line would still fail the step (pipefail through tee).
    status, _ = _sweep_step(tmp_path, jobs, f"echo '{cli.SKIPPED_PREFIX} x'\nexit 1\n")
    assert status != 0


def test_the_sweep_step_matches_the_cli_skip_line(jobs: dict[str, list[str]]) -> None:
    assert f"grep -q '^{cli.SKIPPED_PREFIX}'" in _step_named(jobs, "Sweep")["run"]


def test_nothing_is_collected_or_published_on_a_skip(jobs: dict[str, list[str]]) -> None:
    names = [s.get("name") for s in _steps(jobs["sweep"])]
    # run_outcome's step names: the Sweep step in the sweep job, the record step in publish
    assert schedule.SWEEP_STEP in names
    assert schedule.RECORD_STEP in [s.get("name") for s in _steps(jobs["publish"])]
    for step in _steps(jobs["sweep"]):
        if step.get("id") in ("collect", "upload"):
            assert step["if"] == "steps.sweep.outputs.skipped != 'true'"
    publish_job = _children(jobs["publish"], 4)
    assert "needs.sweep.outputs.skipped != 'true'" in publish_job["if"][0]


def _result(jobs: dict[str, list[str]], tmp_path: Path, job: str = "sweep", **outcomes: str) -> str:
    """Run a job's Result step script with the given step outcomes; return its stage."""
    step = _step_named(jobs, "Result", job)
    names = re.findall(r"^\s*(\w+): \$\{\{ steps\.", step["env"], re.MULTILINE)
    values = {name: "success" for name in names}
    if job == "sweep":
        assert "SKIPPED" in names
        values |= {"FORCED": "skipped", "SKIPPED": ""}
    values |= outcomes
    output = tmp_path / "result"
    output.write_text("")
    result = subprocess.run(
        [_tool("bash"), "-e", "-c", step["run"]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(GITHUB_OUTPUT=str(output), **values),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    [line] = output.read_text().splitlines()
    return line.removeprefix("stage=")


def test_the_result_stage_of_a_skip(tmp_path: Path, jobs: dict[str, list[str]]) -> None:
    skip = {"SKIPPED": "true", "COLLECT": "skipped", "UPLOAD": "skipped"}
    assert _result(jobs, tmp_path, **skip) == "skipped"
    assert _result(jobs, tmp_path) == "none"
    assert _result(jobs, tmp_path, SWEEP="failure") == "sweep"
    assert _result(jobs, tmp_path, UPLOAD="failure") == "publish"
    assert _result(jobs, tmp_path, COLLECT="skipped", UPLOAD="skipped") == "publish"
    for stage in ("skipped", "none", "sweep", "publish"):
        assert stage in schedule.STAGES


@pytest.mark.parametrize(
    ("outcomes", "stage"),
    [
        ({}, "none"),
        ({"DOWNLOAD": "failure", "CHECK": "skipped"}, "publish"),
        ({"CHECK": "failure"}, "publish"),
        ({"COPY": "failure", "PUBLISH": "skipped", "RECORD": "skipped"}, "publish"),
        ({"PUBLISH": "failure", "RECORD": "skipped"}, "publish"),
        ({"RECORD": "failure"}, "record"),
    ],
)
def test_the_result_stage_of_the_publish_job(
    tmp_path: Path, jobs: dict[str, list[str]], outcomes: dict[str, str], stage: str
) -> None:
    assert _result(jobs, tmp_path, "publish", **outcomes) == stage
    assert stage in schedule.STAGES


def _alert_run(fake: FakeGitHub, run_id: int, sweep: str) -> schedule.Action:
    """One run as the T-007 workflow reported it, before the publish job existed (such
    runs stay in the history the alert reads): `sweep` is success, failure or skip. The
    alert job runs while the run is in progress; the run then completes with its jobs.
    The four-job layout is covered by test_t_034."""
    fake.add_run(run_id, None)
    fake.runs[0]["status"] = "in_progress"
    gh = schedule.GitHub(REPO, "t0ken", transport=fake, sleep=lambda _: None)
    job_result = "failure" if sweep == "failure" else "success"
    stage = {"success": "none", "failure": "sweep", "skip": "skipped"}[sweep]
    action = schedule.alert(
        gh,
        run_id=run_id,
        branch="main",
        current=schedule.current_outcome("success", job_result, stage),
        stage=stage,
        record_failures=None,
        server_url="https://github.com",
    )
    fake.runs[0]["status"] = "completed"
    record = {"success": "success", "failure": "skipped", "skip": "skipped"}[sweep]
    steps = [
        {"name": schedule.SWEEP_STEP, "conclusion": job_result},
        {"name": schedule.RECORD_STEP, "conclusion": record},
    ]
    fake.runs[0]["jobs"].append({"name": "sweep", "conclusion": job_result, "steps": steps})
    return action


def test_failure_skip_failure_counts_two(jobs: dict[str, list[str]]) -> None:
    fake = FakeGitHub()
    fake.add_run(1, "success")
    assert _alert_run(fake, 2, "failure") is schedule.Action.NOTHING
    assert _alert_run(fake, 3, "skip") is schedule.Action.NOTHING
    assert _alert_run(fake, 4, "failure") is schedule.Action.NOTHING  # 2, not 3
    assert fake.issues == {}
    assert _alert_run(fake, 5, "skip") is schedule.Action.NOTHING
    assert _alert_run(fake, 6, "failure") is schedule.Action.OPEN  # 3: the streak held
    [issue] = fake.open_issues()
    assert "3 consecutive failures" in issue["title"]
    assert "runs/3" not in issue["body"] and "runs/5" not in issue["body"]


def test_a_skip_leaves_an_open_alert_issue_open(jobs: dict[str, list[str]]) -> None:
    fake = FakeGitHub()
    for run_id in (1, 2, 3):
        _alert_run(fake, run_id, "failure")
    [issue] = fake.open_issues()
    comments = len(issue["comments"])
    assert _alert_run(fake, 4, "skip") is schedule.Action.NOTHING
    assert fake.open_issues() == [issue]
    assert len(issue["comments"]) == comments  # not even a comment
    # the next failure still counts on from 3
    assert _alert_run(fake, 5, "failure") is schedule.Action.COMMENT
    assert "4" in issue["comments"][-1]
    assert _alert_run(fake, 6, "success") is schedule.Action.CLOSE
