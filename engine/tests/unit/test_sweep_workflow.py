"""sweep.yml's own scripts, run for real: the gate refuses manual runs from other
branches, and the data-branch checkout widens its sparse window until the failure
streak's start is in it, so the published consecutive_failures traces back to the
records (INV-6).

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
    ROOT,
    WORKFLOW,
    _children,
    _env,
    _git,
    _run_step,
    _steps,
    _tool,
)
from wearreport import aggregate, publish


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
    _run_step(_script(jobs, "sweep", "Commit and push"), tmp_path, remote, seed)
    return remote


def _publish_now(
    tmp_path: Path, jobs: dict[str, list[str]], remote: Path, record: Any, now: datetime
) -> tuple[Path, dict[str, Any]]:
    """Run the workflow's checkout, publish `record` as the Sweep step would, then run
    its commit step. Returns the data directory and the published status.json."""
    data_dir = tmp_path / "run"
    _run_step(_script(jobs, "sweep", "Check out the data branch"), tmp_path, remote, data_dir)
    publish.publish(data_dir, record, now=now)
    _run_step(_script(jobs, "sweep", "Commit and push"), tmp_path, remote, data_dir)
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
