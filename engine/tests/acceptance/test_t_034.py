"""Acceptance tests for T-034 (sweep workflow hardening). The task contract: do not edit.

The sweep is split in two jobs: `sweep` (contents: read) runs the engine and uploads the
new records and status.json as an artifact; `publish` (contents: write) checks that
artifact and pushes it to the `data` branch, running no third-party code. The streak
search on the shallow, sparse data checkout is linear and capped, `streak-closed` counts
a record as a success only when the publisher would, and the alert orders the run
history by time. The workflow's scripts run for real against a local bare repository;
the alert runs against the in-memory GitHub of the T-007 tests.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.acceptance.test_t_007 import (
    PINNED,
    REPO,
    ROOT,
    WORKFLOW,
    FakeGitHub,
    _children,
    _env,
    _git,
    _permissions,
    _run_step,
    _scalar,
    _steps,
    _tool,
)
from wearreport import aggregate, publish, schedule

NOW = datetime(2026, 7, 15, 12, 0, 5, tzinfo=UTC)
# First-party actions the publishing job may use; everything else is third-party code.
FIRST_PARTY = ("actions/checkout@", "actions/download-artifact@")
THIRD_PARTY_COMMAND = re.compile(r"(?<![\w./-])(make|uv|uvx|pip|pip3|curl|wget)(?![\w.-])")


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def jobs(text: str) -> dict[str, list[str]]:
    return _jobs(text)


def _jobs(text: str) -> dict[str, list[str]]:
    top = _children(text.splitlines(), 0)
    return _children(top["jobs"][1:], 2)


def _job_keys(job: list[str]) -> dict[str, list[str]]:
    return _children(job, 4)


def _needs(job: list[str]) -> set[str]:
    value = _scalar(_job_keys(job)["needs"])
    return {n.strip() for n in value.strip("[]").split(",")}


def _step(jobs: dict[str, list[str]], job: str, name_part: str) -> dict[str, str]:
    matches = [s for s in _steps(jobs[job]) if name_part in s.get("name", "")]
    assert len(matches) == 1, f"expected one step named like {name_part!r} in {job}"
    return matches[0]


def _uses(jobs: dict[str, list[str]], job: str, action: str) -> dict[str, str]:
    matches = [s for s in _steps(jobs[job]) if s.get("uses", "").startswith(f"{action}@")]
    assert len(matches) == 1, f"expected one {action} step in {job}"
    return matches[0]


def _action(step: dict[str, str]) -> str:
    """A step's `uses`, without the trailing version comment."""
    return step.get("uses", "").split(" #", 1)[0].strip()


def _with(step: dict[str, str]) -> dict[str, str]:
    """A step's `with:` inputs; a block scalar (`|`) is returned as its lines."""
    out: dict[str, str] = {}
    lines = step.get("with", "").splitlines()
    for key, block in _children(lines, 10).items():
        if block[0].rstrip().endswith("|"):
            out[key] = "\n".join(line.strip() for line in block[1:])
        else:
            out[key] = _scalar(block)
    return out


def third_party_code(jobs: dict[str, list[str]]) -> list[str]:
    """Every way a job holding `contents: write` could run third-party code: an action
    other than the pinned first-party checkout and download-artifact, a step running
    make, uv, pip or curl, or a step running a file from the downloaded artifact."""
    found = []
    for name, job in jobs.items():
        if _permissions(job).get("contents") != "write":
            continue
        artifacts = [
            _with(s).get("path", "")
            for s in _steps(job)
            if s.get("uses", "").startswith("actions/download-artifact@")
        ]
        for step in _steps(job):
            uses = _action(step)
            if uses and not (uses.startswith(FIRST_PARTY) and PINNED.match(uses)):
                found.append(f"{name}: uses {uses}")
            run = step.get("run", "")
            code = "\n".join(line for line in run.splitlines() if not line.lstrip().startswith("#"))
            for m in THIRD_PARTY_COMMAND.finditer(code):
                found.append(f"{name}: runs {m.group(1)}")
            found += [f"{name}: {problem}" for problem in _runs_the_artifact(run, artifacts)]
    return found


def _runs_the_artifact(run: str, artifact_paths: list[str]) -> list[str]:
    """Commands in `run` that could execute a downloaded file: any interpreter or shell
    other than `python3 -m wearreport.schedule` from the checked-out engine, and any
    mention of the artifact that is not an argument to that module."""
    problems = []
    code = "\n".join(line for line in run.splitlines() if not line.lstrip().startswith("#"))
    commands = re.sub(r"\\\n\s*", " ", code).splitlines()
    for command in commands:
        for m in re.finditer(
            r"(?<![\w./-])(bash|sh|source|eval|exec|chmod|node|perl)(?!\w)", command
        ):
            problems.append(f"runs {m.group(1)}")
        for m in re.finditer(r"python3?(?:\.\d+)?((?:\s+-\w+)*)\s+(\S+)", command):
            if not (m.group(1).strip() == "-m" and m.group(2) == "wearreport.schedule"):
                problems.append(f"runs python {m.group(2)}")
        for m in re.finditer(r"\bPYTHONPATH=(\S+)", command):
            if m.group(1).strip("\"'") not in ("engine", "${GITHUB_WORKSPACE}/engine"):
                problems.append(f"imports from {m.group(1)}")
        mentions_artifact = re.search("artifact", command, re.IGNORECASE) or any(
            p and p in command for p in artifact_paths
        )
        if mentions_artifact and "python3 -m wearreport.schedule" not in command:
            problems.append(f"touches the artifact outside wearreport.schedule: {command.strip()}")
    return problems


# AC1: four jobs, and the write job runs no third-party code -----------------------------


def test_ac1_four_jobs_in_order(jobs: dict[str, list[str]]) -> None:
    assert list(jobs) == ["gate", "sweep", "publish", "alert"]
    assert _needs(jobs["sweep"]) == {"gate"}
    assert _needs(jobs["publish"]) == {"sweep"}
    assert _needs(jobs["alert"]) == {"gate", "sweep", "publish"}
    alert_if = _scalar(_job_keys(jobs["alert"])["if"])
    # a status function, so the alert runs after a failed sweep or publish too
    assert "always()" in alert_if or "!cancelled()" in alert_if
    env = _step(jobs, "alert", "ops-alert")["env"]
    assert "needs.sweep.result" in env and "needs.publish.result" in env


def test_ac1_sweep_reads_only_and_uploads_only_records_and_status(
    jobs: dict[str, list[str]],
) -> None:
    assert _permissions(jobs["sweep"]) == {"contents": "read"}
    steps = _steps(jobs["sweep"])
    runs = "\n".join(s.get("run", "") for s in steps)
    assert "uv sync --locked --no-dev" in runs
    assert "sh scripts/fetch_model.sh --with-m" in runs
    assert "uv run --locked --no-dev wearreport sweep" in _step(jobs, "sweep", "Sweep")["run"]
    upload = _uses(jobs, "sweep", "actions/upload-artifact")
    assert PINNED.match(_action(upload))
    inputs = _with(upload)
    paths = inputs["path"].splitlines()
    assert len(paths) == 2
    assert paths[0].endswith("/sweeps/**/*.json")
    assert paths[1].endswith("/status.json")
    assert paths[0].removesuffix("sweeps/**/*.json") == paths[1].removesuffix("status.json")
    assert inputs["retention-days"] == "1"
    assert inputs.get("include-hidden-files", "false") == "false"
    uploads = [s for s in steps if "upload-artifact" in s.get("uses", "")]
    assert len(uploads) == 1


def test_ac1_publish_writes_only_and_uses_first_party_actions(
    jobs: dict[str, list[str]],
) -> None:
    assert _permissions(jobs["publish"]) == {"contents": "write"}
    uses = [_action(s) for s in _steps(jobs["publish"]) if "uses" in s]
    assert uses and all(u.startswith(FIRST_PARTY) and PINNED.match(u) for u in uses)
    download = _uses(jobs, "publish", "actions/download-artifact")
    upload = _uses(jobs, "sweep", "actions/upload-artifact")
    assert _with(download)["name"] == _with(upload)["name"]
    runs = "\n".join(s.get("run", "") for s in _steps(jobs["publish"]))
    assert "python3 -m wearreport.schedule check-artifact" in runs


def test_ac1_no_job_with_contents_write_runs_third_party_code(
    jobs: dict[str, list[str]],
) -> None:
    writers = {n for n, j in jobs.items() if _permissions(j).get("contents") == "write"}
    assert writers == {"publish"}
    assert third_party_code(jobs) == []


@pytest.mark.parametrize(
    "mutant",
    [
        "      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7 # v10.2.0\n",
        "      - run: make check\n",
        "      - run: uv sync --locked\n",
        "      - run: pip install requests\n",
        "      - run: curl -sSf https://example.org\n",
        '      - run: sh "${ARTIFACT_DIR}/status.json"\n',
        '      - run: python3 "${ARTIFACT_DIR}/sweeps/x.json"\n',
        '      - run: PYTHONPATH="${ARTIFACT_DIR}" python3 -m wearreport.schedule gate\n',
        "      - uses: actions/checkout@v7\n",
    ],
)
def test_ac1_the_third_party_check_catches_a_mutant(text: str, mutant: str) -> None:
    # Insert the mutant as the first step of the publish job: the check must flag it.
    start = text.index("\n  publish:\n")
    steps = text.index("    steps:\n", start) + len("    steps:\n")
    mutated = _jobs(text[:steps] + mutant + text[steps:])
    assert third_party_code(mutated) != []


def _record(started: datetime, *, ok: bool) -> Any:
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


def _artifact(root: Path, *records: Any, status: bytes | None = None) -> Path:
    """An artifact as the sweep job uploads it: records under sweeps/ and status.json."""
    for record in records:
        path = publish.record_path(root, record["sweep_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(publish.serialize(record))
    status_bytes = b'{"consecutive_failures": 0}\n' if status is None else status
    (root / "status.json").write_bytes(status_bytes)
    return root


def test_ac1_check_artifact_accepts_records_and_status(tmp_path: Path) -> None:
    record = _record(NOW, ok=True)
    root = _artifact(tmp_path / "a", record)
    paths = schedule.check_artifact(root)
    assert sorted(paths) == sorted(
        [publish.record_path(Path(), record["sweep_id"]).as_posix(), "status.json"]
    )
    out = subprocess.run(
        [sys.executable, "-m", "wearreport.schedule", "check-artifact", "--dir", str(root)],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert sorted(out.stdout.splitlines()) == sorted(paths)


def _bad_artifacts() -> list[tuple[str, Callable[[Path], None]]]:
    record = _record(NOW, ok=True)
    relative = publish.record_path(Path(), record["sweep_id"])

    def symlinked_record(root: Path) -> None:
        target = root.parent / "outside.json"
        target.write_bytes(publish.serialize(record))
        (root / relative).unlink()
        (root / relative).symlink_to(target)

    def symlinked_dir(root: Path) -> None:
        elsewhere = root.parent / "elsewhere"
        shutil.move(root / "sweeps", elsewhere)
        (root / "sweeps").symlink_to(elsewhere)

    def symlinked_status(root: Path) -> None:
        (root / "status.json").unlink()
        (root / "status.json").symlink_to(root / relative)

    def fifo(root: Path) -> None:
        os.mkfifo(root / relative.parent / "20260715T1300Z.json")

    def too_large(root: Path) -> None:
        (root / relative).write_bytes(b"{}" + b" " * publish.MAX_RECORD_BYTES)

    def not_json(root: Path) -> None:
        (root / relative).write_bytes(b"not json\n")

    def not_utf8(root: Path) -> None:
        (root / relative).write_bytes(b"\xff\xfe{}")

    def deep(root: Path) -> None:
        (root / relative).write_bytes(b"[" * 100_000 + b"]" * 100_000)

    def other_id(root: Path) -> None:
        moved = relative.parent / "20260715T1300Z.json"
        (root / relative).rename(root / moved)

    def wrong_date(root: Path) -> None:
        target = root / "sweeps" / "2026" / "07" / "16" / relative.name
        target.parent.mkdir(parents=True)
        (root / relative).rename(target)

    def extra_file(root: Path) -> None:
        (root / "README.md").write_text("hello")

    def script(root: Path) -> None:
        (root / "sweeps" / "run.sh").write_text("#!/bin/sh\n")

    def hidden(root: Path) -> None:
        (root / relative.parent / ".x.json").write_bytes(publish.serialize(record))

    def no_status(root: Path) -> None:
        (root / "status.json").unlink()

    return [
        ("symlinked record", symlinked_record),
        ("symlinked directory", symlinked_dir),
        ("symlinked status", symlinked_status),
        ("fifo", fifo),
        ("record over 1 MiB", too_large),
        ("record not JSON", not_json),
        ("record not UTF-8", not_utf8),
        ("record nested too deep", deep),
        ("path is another sweep_id", other_id),
        ("path under another date", wrong_date),
        ("another file", extra_file),
        ("a script", script),
        ("a hidden file", hidden),
        ("no status.json", no_status),
    ]


@pytest.mark.parametrize(("case", "spoil"), _bad_artifacts(), ids=[c for c, _ in _bad_artifacts()])
def test_ac1_check_artifact_rejects_anything_else(
    tmp_path: Path, case: str, spoil: Callable[[Path], None]
) -> None:
    root = _artifact(tmp_path / "a", _record(NOW, ok=True))
    spoil(root)
    with pytest.raises(schedule.ScheduleError):
        schedule.check_artifact(root)
    out = subprocess.run(
        [sys.executable, "-m", "wearreport.schedule", "check-artifact", "--dir", str(root)],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=60,
    )
    assert out.returncode == 1
    assert "Traceback" not in out.stderr


def test_ac1_check_artifact_rejects_a_record_that_is_not_an_object(tmp_path: Path) -> None:
    record = _record(NOW, ok=True)
    root = _artifact(tmp_path / "a", record)
    path = publish.record_path(root, record["sweep_id"])
    for data in (b"[]", b"42", b'"x"', b"null", b'{"sweep_id": 7}', b"{}"):
        path.write_bytes(data)
        with pytest.raises(schedule.ScheduleError):
            schedule.check_artifact(root)


# The workflow's jobs, run for real -----------------------------------------------------


def _expand(value: str, context: Mapping[str, str]) -> str:
    def one(m: re.Match[str]) -> str:
        return context[m.group(1)]

    return re.sub(r"\$\{\{\s*([\w.]+)\s*\}\}", one, value)


def _run_job(
    tmp: Path,
    jobs: dict[str, list[str]],
    job: str,
    *,
    remote: Path,
    names: list[str],
    downloads: Mapping[str, Path] | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Run the steps of `job` whose names contain one of `names`, in the workflow's
    order, as the runner would: GITHUB_ENV carries over, step env maps the github
    context, and a download-artifact step copies `downloads[name]` to its path. Returns
    the first failed step's result, or None when every step succeeded."""
    runner_temp = tmp / f"runner-{job}"
    runner_temp.mkdir(exist_ok=True)
    github_env = runner_temp / "github_env"
    if not github_env.exists():  # carried over between calls for the same job
        github_env.write_text("")
    context = {
        "runner.temp": str(runner_temp),
        "github.token": "dummy",
        "github.repository": REPO,
        "github.run_id": "42",
    }
    shim = tmp / "bin"
    shim.mkdir(exist_ok=True)
    if not (shim / "python3").exists():
        (shim / "python3").symlink_to(sys.executable)
    for step in _steps(jobs[job]):
        uses = step.get("uses", "")
        if uses.startswith("actions/download-artifact@"):
            inputs = _with(step)
            source = (downloads or {})[inputs["name"]]
            shutil.copytree(source, _expand(inputs["path"], context))
            continue
        if "run" not in step or not any(n in step.get("name", "") for n in names):
            continue
        env = _env(
            PATH=f"{shim}{os.pathsep}{os.environ['PATH']}",
            RUNNER_TEMP=str(runner_temp),
            GITHUB_ENV=str(github_env),
            GITHUB_OUTPUT=str(runner_temp / "output"),
            GITHUB_WORKSPACE=str(ROOT),
            GITHUB_RUN_ID="42",
            DATA_REMOTE=str(remote),
        )
        del env["PYTHONPATH"]  # every step sets its own
        for line in github_env.read_text().splitlines():
            key, _, value = line.partition("=")
            env[key] = value
        for key, value in re.findall(r"^\s*(\w+): (.*)$", step.get("env", ""), re.MULTILINE):
            env[key] = _expand(value, context)
        result = subprocess.run(
            [_tool("bash"), "-e", "-c", step["run"]],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        if result.returncode != 0:
            return result
    return None


SWEEP_STEPS = ["Set DATA_DIR", "Check out the data branch"]
PUBLISH_STEPS = [
    "Set DATA_DIR",
    "Check the",
    "Check out the data branch",
    "Copy",
    "Commit and push",
    "Did the published sweep succeed?",
]


def _one_run(
    tmp: Path, jobs: dict[str, list[str]], remote: Path, record: Any, now: datetime
) -> subprocess.CompletedProcess[str] | None:
    """One scheduled run: the sweep job checks out the data branch and the engine
    publishes `record` into it, the sweep job collects the artifact, the publish job
    downloads, checks and pushes it. Returns the publish job's first failed step."""
    run = tmp / f"run-{record['sweep_id']}"
    run.mkdir()
    assert _run_job(run, jobs, "sweep", remote=remote, names=SWEEP_STEPS) is None
    data_dir = run / "runner-sweep" / "data"
    assert data_dir.is_dir()
    publish.publish(data_dir, record, now=now)
    assert _run_job(run, jobs, "sweep", remote=remote, names=["Collect"]) is None
    upload = _uses(jobs, "sweep", "actions/upload-artifact")
    paths = _with(upload)["path"].splitlines()
    outbox = Path(_expand(paths[1], {"runner.temp": str(run / "runner-sweep")})).parent
    return _run_job(
        run,
        jobs,
        "publish",
        remote=remote,
        names=PUBLISH_STEPS,
        downloads={_with(upload)["name"]: outbox},
    )


def _bare(tmp: Path) -> Path:
    remote = tmp / "remote.git"
    subprocess.run([_tool("git"), "init", "-q", "--bare", str(remote)], check=True, env=_env())
    return remote


def test_ac1_publish_pushes_the_artifact_to_the_data_branch(
    tmp_path: Path, jobs: dict[str, list[str]]
) -> None:
    remote = _bare(tmp_path)
    now = datetime.now(UTC).replace(microsecond=0)
    first = _record(now - timedelta(days=1), ok=True)
    assert _one_run(tmp_path, jobs, remote, first, now - timedelta(days=1)) is None
    assert _git(remote, "rev-list", "--count", "data").strip() == "1"  # an orphan start
    second = _record(now - timedelta(minutes=5), ok=True)
    assert _one_run(tmp_path, jobs, remote, second, now) is None
    changed = _git(remote, "diff", "--name-status", "data~1", "data").splitlines()
    new_path = publish.record_path(Path(), second["sweep_id"]).as_posix()
    assert sorted(changed) == sorted([f"A\t{new_path}", "M\tstatus.json"])
    assert _git(remote, "show", f"data:{new_path}").encode() == publish.serialize(second)
    status = json.loads(_git(remote, "show", "data:status.json"))
    assert status["last_sweep_id"] == second["sweep_id"]
    assert _git(remote, "branch", "--list").split() == ["data"]
    push = _step(jobs, "publish", "Commit and push")["run"]
    assert re.findall(r"\bpush\b[^\n]*", push) == ["push -q origin HEAD:refs/heads/data"]


def test_ac1_publish_pushes_nothing_from_a_spoilt_artifact(
    tmp_path: Path, jobs: dict[str, list[str]]
) -> None:
    remote = _bare(tmp_path)
    now = datetime.now(UTC).replace(microsecond=0)
    assert _one_run(tmp_path, jobs, remote, _record(now - timedelta(days=1), ok=True), now) is None
    head = _git(remote, "rev-parse", "data")
    upload = _with(_uses(jobs, "sweep", "actions/upload-artifact"))
    artifact = _artifact(tmp_path / "spoilt", _record(now - timedelta(minutes=5), ok=True))
    (artifact / "README.md").write_text("not data")
    (tmp_path / "spoilt-run").mkdir()
    failed = _run_job(
        tmp_path / "spoilt-run",
        jobs,
        "publish",
        remote=remote,
        names=PUBLISH_STEPS,
        downloads={upload["name"]: artifact},
    )
    assert failed is not None
    assert _git(remote, "rev-parse", "data") == head


# AC2: a linear, capped search for the start of the failure streak -----------------------


def _history(tmp: Path, records: list[bytes], paths: list[str]) -> Path:
    """A bare remote whose `data` branch holds the given record files in one commit."""
    remote = _bare(tmp)
    work = tmp / "seed"
    subprocess.run([_tool("git"), "init", "-q", "-b", "data", str(work)], check=True, env=_env())
    for data, relative in zip(records, paths, strict=True):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _git(work, "add", "--", "sweeps")
    _git(work, "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q", "-m", "seed")
    _git(work, "push", "-q", str(remote), "data:refs/heads/data")
    return remote


def _days(now: datetime, days: range) -> list[Any]:
    """42 failed sweeps a day (06:07 to 19:47 UTC every 20 minutes) on each of `days`
    days ago, oldest first."""
    out = []
    for back in sorted(days, reverse=True):
        day = (now - timedelta(days=back)).replace(hour=6, minute=7, second=5, microsecond=0)
        out += [_record(day + timedelta(minutes=20 * i), ok=False) for i in range(42)]
    return out


def _paths(records: list[Any]) -> list[str]:
    return [publish.record_path(Path(), r["sweep_id"]).as_posix() for r in records]


def test_ac2_300_days_of_failures_are_searched_in_under_10_seconds(
    tmp_path: Path, jobs: dict[str, list[str]]
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    records = _days(now, range(1, 301))
    assert len(records) == 300 * 42
    remote = _history(tmp_path, [publish.serialize(r) for r in records], _paths(records))
    data_dir = tmp_path / "data"
    checkout = _step(jobs, "sweep", "Check out the data branch")["run"]
    start = time.monotonic()
    _run_step(checkout, tmp_path, remote, data_dir)
    elapsed = time.monotonic() - start
    warnings.warn(f"T-034 AC2: 300 days x 42 records searched in {elapsed:.2f} s", stacklevel=1)
    assert elapsed < 10
    # every day is in the tree, so the published count traces back to all the records
    checked_out = {p.relative_to(data_dir).as_posix() for p in data_dir.glob("sweeps/*/*/*/*")}
    assert checked_out == set(_paths(records))
    new = _record(now - timedelta(minutes=5), ok=False)
    publish.publish(data_dir, new, now=now)
    status = json.loads((data_dir / "status.json").read_text())
    assert status["consecutive_failures"] == 300 * 42 + 1


def _shallow_checkout(tmp: Path, remote: Path, now: datetime) -> tuple[Path, str]:
    """The sweep job's checkout before any widening: status.json and three UTC days."""
    data_dir = tmp / "shallow"
    subprocess.run([_tool("git"), "init", "-q", str(data_dir)], check=True, env=_env())
    window = [f"/sweeps/{(now - timedelta(days=d)):%Y/%m/%d}/" for d in (2, 1, 0)]
    _git(data_dir, "sparse-checkout", "set", "--no-cone", "/status.json", *window)
    _git(data_dir, "fetch", "-q", "--depth=1", "--no-tags", str(remote), "refs/heads/data")
    _git(data_dir, "checkout", "-q", "-b", "data", "FETCH_HEAD")
    return data_dir, f"sweeps/{(now - timedelta(days=2)):%Y/%m/%d}"


def test_ac2_each_older_record_is_checked_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    records = _days(now, range(0, 20))
    records = [r for r in records if aggregate.parse_utc(r["started_at"]) <= now]
    remote = _history(tmp_path, [publish.serialize(r) for r in records], _paths(records))
    data_dir, before = _shallow_checkout(tmp_path, remote, now)
    seen: Counter[str] = Counter()
    original = schedule._successful_record

    def counting(data: bytes, relative: str, moment: datetime) -> bool:
        seen[relative] += 1
        return original(data, relative, moment)

    monkeypatch.setattr(schedule, "_successful_record", counting)
    # One search, as the workflow runs it: the checked-out window, then each older day.
    widening = schedule.streak_days(data_dir, before=before, now=now)
    older = sorted(
        {f"/{p.rsplit('/', 1)[0]}/" for p in _paths(records) if p < before}, reverse=True
    )
    assert list(widening.days) == older  # newest first, each day once
    assert widening.capped is False
    assert set(seen) == set(_paths(records))
    assert max(seen.values()) == 1  # no record is read twice


def test_ac2_the_search_stops_at_the_newest_success(tmp_path: Path) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    failures = _days(now, range(3, 12))
    success = _record((now - timedelta(days=7)).replace(hour=20, minute=0), ok=True)
    records = sorted([*failures, success], key=lambda r: r["sweep_id"])
    remote = _history(tmp_path, [publish.serialize(r) for r in records], _paths(records))
    data_dir, before = _shallow_checkout(tmp_path, remote, now)
    widening = schedule.streak_days(data_dir, before=before, now=now)
    assert widening.days[-1] == f"/sweeps/{(now - timedelta(days=7)):%Y/%m/%d}/"
    assert len(widening.days) == 5  # days 3 to 7 ago
    assert widening.capped is False


def test_ac2_an_explicit_cap_after_which_the_streak_is_unbounded(tmp_path: Path) -> None:
    assert isinstance(schedule.MAX_STREAK_DAYS, int)
    assert 300 < schedule.MAX_STREAK_DAYS <= 1000
    now = datetime.now(UTC).replace(microsecond=0)
    records = _days(now, range(3, 11))
    remote = _history(tmp_path, [publish.serialize(r) for r in records], _paths(records))
    data_dir, before = _shallow_checkout(tmp_path, remote, now)
    widening = schedule.streak_days(data_dir, before=before, now=now, max_days=5)
    assert len(widening.days) == 5
    assert widening.capped is True
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "wearreport.schedule",
            "streak-days",
            "--data-dir",
            str(data_dir),
            "--before",
            before,
            "--max-days",
            "5",
        ],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == list(widening.days)
    assert "unbounded" in out.stderr
    # the alert still works: the counted failures are far above the threshold
    _git(data_dir, "sparse-checkout", "add", *widening.days)
    publish.publish(data_dir, _record(now - timedelta(minutes=5), ok=False), now=now)
    failures = json.loads((data_dir / "status.json").read_text())["consecutive_failures"]
    assert failures == 5 * 42 + 1
    fake = FakeGitHub()
    gh = schedule.GitHub(REPO, "t0ken", transport=fake, sleep=lambda _: None)
    action = schedule.alert(
        gh,
        run_id=1,
        branch="main",
        current=schedule.Outcome.FAILURE,
        stage="record",
        record_failures=failures,
        server_url="https://github.com",
    )
    assert action is schedule.Action.OPEN


# AC3: streak-closed agrees with the publisher -------------------------------------------


def _put(data_dir: Path, data: bytes, sweep_id: str) -> None:
    path = publish.record_path(data_dir, sweep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _lookalikes() -> list[tuple[str, bytes, str]]:
    """(case, bytes, sweep_id of the path) for records whose counts say success but
    that the publisher does not count as one."""
    good = _record(NOW - timedelta(hours=1), ok=True)
    assert publish.is_success(good)

    def spoilt(**changes: Any) -> bytes:
        return json.dumps({**good, **changes}).encode()

    later = _record(NOW + timedelta(hours=1), ok=True)
    other_minute = aggregate.sweep_id_for(NOW - timedelta(hours=2))
    return [
        ("persons_total does not add up", spoilt(persons_total=3), good["sweep_id"]),
        ("an extra field", spoilt(note="x"), good["sweep_id"]),
        ("wrong attribution", spoilt(attribution=["someone"]), good["sweep_id"]),
        ("wrong schema", spoilt(schema="sweep.v0"), good["sweep_id"]),
        ("frames do not add up", spoilt(cameras_listed=9), good["sweep_id"]),
        ("stored under another sweep_id", publish.serialize(good), other_minute),
        ("starts after now", publish.serialize(later), later["sweep_id"]),
    ]


@pytest.mark.parametrize(
    ("case", "data", "sweep_id"), _lookalikes(), ids=[c for c, _, _ in _lookalikes()]
)
def test_ac3_a_record_the_publisher_rejects_does_not_close_the_streak(
    tmp_path: Path, case: str, data: bytes, sweep_id: str
) -> None:
    listed = json.loads(data)
    assert listed["frames_ok"] * 10 >= listed["cameras_listed"] * 9  # looks like a success
    _put(tmp_path, data, sweep_id)
    assert schedule.streak_closed(tmp_path, now=NOW) is False
    # the publisher agrees: no success, so nothing ends a streak
    status = publish.compute_status(tmp_path, now=NOW)
    assert status["successful_sweeps_24h"] == 0
    assert schedule.main(["streak-closed", "--data-dir", str(tmp_path), "--now", _stamp(NOW)]) == 1


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_ac3_a_valid_success_closes_the_streak(tmp_path: Path) -> None:
    good = _record(NOW - timedelta(hours=1), ok=True)
    _put(tmp_path, publish.serialize(good), good["sweep_id"])
    assert schedule.streak_closed(tmp_path, now=NOW) is True
    assert publish.compute_status(tmp_path, now=NOW)["successful_sweeps_24h"] == 1
    assert schedule.main(["streak-closed", "--data-dir", str(tmp_path), "--now", _stamp(NOW)]) == 0


def test_ac3_a_failed_valid_record_does_not_close_the_streak(tmp_path: Path) -> None:
    failed = _record(NOW - timedelta(hours=1), ok=False)
    _put(tmp_path, publish.serialize(failed), failed["sweep_id"])
    assert schedule.streak_closed(tmp_path, now=NOW) is False


def _older_lookalikes() -> list[tuple[str, bytes, str]]:
    # A record in a day older than the checked-out window cannot start after now.
    return [c for c in _lookalikes() if c[0] != "starts after now"]


@pytest.mark.parametrize(
    ("case", "data", "sweep_id"), _older_lookalikes(), ids=[c for c, _, _ in _older_lookalikes()]
)
def test_ac3_the_widening_passes_over_a_lookalike_too(
    tmp_path: Path, case: str, data: bytes, sweep_id: str
) -> None:
    # Failures now and in the window; the lookalike 5 days back; a real success 8 days
    # back. The search must go past the lookalike to the real success.
    now = NOW + timedelta(days=5)
    failures = _days(now, range(0, 7))
    failures = [r for r in failures if aggregate.parse_utc(r["started_at"]) <= now]
    success = _record(NOW - timedelta(days=3), ok=True)
    records = [publish.serialize(r) for r in [success, *failures]]
    paths = _paths([success, *failures])
    records.append(data)
    paths.append(publish.record_path(Path(), sweep_id).as_posix())
    remote = _history(tmp_path, records, paths)
    data_dir, before = _shallow_checkout(tmp_path, remote, now)
    widening = schedule.streak_days(data_dir, before=before, now=now)
    day = publish.record_path(Path(), success["sweep_id"]).parent.as_posix()
    assert widening.days[-1] == f"/{day}/"


# AC4: the run history is ordered by time ------------------------------------------------


def test_ac4_the_fake_gives_each_run_timestamps() -> None:
    fake = FakeGitHub()
    fake.add_run(1, "success")
    fake.add_run(2, "failure")
    for run in fake.runs:
        for key in ("created_at", "updated_at"):
            assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", run[key])
    assert fake.runs[0]["created_at"] > fake.runs[1]["created_at"]
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/sweep.yml/runs?branch=main"
    _, raw = fake("GET", url, None, {}, 1.0)
    listed = json.loads(raw)["workflow_runs"]
    assert all("created_at" in r and "updated_at" in r for r in listed)


def _failure_now(fake: FakeGitHub, run_id: int) -> schedule.Action:
    fake.add_run(run_id, None)
    fake.runs[0]["status"] = "in_progress"
    gh = schedule.GitHub(REPO, "t0ken", transport=fake, sleep=lambda _: None)
    return schedule.alert(
        gh,
        run_id=run_id,
        branch="main",
        current=schedule.Outcome.FAILURE,
        stage="sweep",
        record_failures=None,
        server_url="https://github.com",
    )


def test_ac4_an_older_run_listed_first_does_not_change_the_count() -> None:
    fake = FakeGitHub()
    fake.add_run(1, "success")
    fake.add_run(2, "failure")
    fake.add_run(3, "failure")
    # Run 1 was re-run and updated last, so a listing may put it first; it still
    # started before runs 2 and 3.
    oldest = fake.runs.pop()
    oldest["updated_at"] = "2099-01-01T00:00:00Z"
    fake.runs.insert(0, oldest)
    assert _failure_now(fake, 4) is schedule.Action.OPEN  # 3 failures: 4, 3 and 2
    [issue] = fake.open_issues()
    assert "3 consecutive failures" in issue["title"]


def test_ac4_the_listing_order_alone_does_not_decide() -> None:
    # The same history listed newest first gives the same count.
    fake = FakeGitHub()
    fake.add_run(1, "success")
    fake.add_run(2, "failure")
    fake.add_run(3, "failure")
    assert _failure_now(fake, 4) is schedule.Action.OPEN


class _Listing:
    """A transport that answers the runs listing with `runs` and every jobs listing
    with one failed sweep."""

    def __init__(self, runs: list[dict[str, Any]]) -> None:
        self.runs = runs

    def __call__(
        self, method: str, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> tuple[int, bytes]:
        if url.split("?")[0].endswith("/runs"):
            return 200, json.dumps({"workflow_runs": self.runs}).encode()
        jobs = [
            {"name": "gate", "conclusion": "success"},
            {"name": "sweep", "conclusion": "failure"},
        ]
        return 200, json.dumps({"jobs": jobs}).encode()


@pytest.mark.parametrize("created_at", [None, 7, [], {}, "", "yesterday", "2026-13-01T00:00:00Z"])
def test_ac4_a_run_without_a_valid_time_is_a_typed_error(created_at: Any) -> None:
    runs = [{"id": 7, "status": "completed", "updated_at": "2026-07-15T12:00:00Z"}]
    if created_at is not None:
        runs[0]["created_at"] = created_at
    gh = schedule.GitHub(REPO, "t0ken", transport=_Listing(runs), sleep=lambda _: None)
    with pytest.raises(schedule.GitHubError):
        list(schedule.previous_outcomes(gh, run_id=1, branch="main"))


# AC6: still true -----------------------------------------------------------------------


def test_ac6_concurrency_timeouts_dispatch_and_pins(text: str, jobs: dict[str, list[str]]) -> None:
    top = _children(text.splitlines(), 0)
    concurrency = _children(top["concurrency"][1:], 2)
    assert _scalar(concurrency["group"]) == "sweep"
    assert _scalar(concurrency["cancel-in-progress"]) == "false"
    for job in jobs.values():
        assert "concurrency" not in _job_keys(job)
    assert _scalar(_job_keys(jobs["sweep"])["timeout-minutes"]) == "15"
    assert int(_scalar(_job_keys(jobs["publish"])["timeout-minutes"])) <= 5
    gate = _step(jobs, "gate", "Gate")
    assert '--event "${EVENT}" --ref "${REF}" --default-branch "${DEFAULT_BRANCH}"' in gate["run"]
    first = _steps(jobs["sweep"])[0]
    assert first["if"] == "inputs.force_fail" and "exit 1" in first["run"]
    for ref in re.findall(r"uses:\s*(\S+)", text):
        assert PINNED.match(ref), ref
    assert "pull_request" not in text
    for job in jobs.values():
        assert _scalar(_job_keys(job)["runs-on"]) == "ubuntu-24.04"


def test_ac6_actionlint_passes() -> None:
    actionlint = ROOT / ".tools" / "bin" / "actionlint"
    if not actionlint.is_file():
        if os.environ.get("WEARREPORT_REQUIRE_ACTIONLINT"):
            pytest.fail("actionlint is missing")
        pytest.skip("actionlint is not installed (run `make setup`)")
    out = subprocess.run(
        [str(actionlint), "-shellcheck=", "-pyflakes=", str(WORKFLOW)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr


# AC8: the skip survives the split -------------------------------------------------------


def test_ac8_publish_does_not_run_on_a_skip(jobs: dict[str, list[str]]) -> None:
    sweep = _job_keys(jobs["sweep"])
    outputs = "\n".join(sweep["outputs"])
    assert "skipped: ${{ steps.sweep.outputs.skipped }}" in outputs
    publish_if = _scalar(_job_keys(jobs["publish"])["if"])
    assert "needs.sweep.outputs.skipped != 'true'" in publish_if
    for step in _steps(jobs["sweep"]):
        if "upload-artifact" in step.get("uses", "") or "Collect" in step.get("name", ""):
            assert step["if"] == "steps.sweep.outputs.skipped != 'true'"


def test_ac8_the_stage_that_reaches_the_alert_is_skipped() -> None:
    assert schedule.alert_stage("success", "skipped", "skipped", "") == "skipped"
    current = schedule.current_outcome("success", "success", "skipped", "skipped")
    assert current is schedule.Outcome.NONE
    # and the others: a publish failure is named by the publish job
    assert schedule.alert_stage("success", "none", "failure", "publish") == "publish"
    assert schedule.alert_stage("success", "none", "failure", "record") == "record"
    assert schedule.alert_stage("success", "none", "failure", "") == "publish"
    assert schedule.alert_stage("success", "sweep", "skipped", "") == "sweep"
    assert schedule.alert_stage("failure", "", "skipped", "") == "gate"


@pytest.mark.parametrize(
    ("sweep", "publish_result", "expected"),
    [
        ("success", "success", schedule.Outcome.SUCCESS),
        ("success", "failure", schedule.Outcome.FAILURE),
        ("success", "cancelled", schedule.Outcome.FAILURE),
        ("failure", "skipped", schedule.Outcome.FAILURE),
        ("cancelled", "skipped", schedule.Outcome.FAILURE),
        ("skipped", "skipped", schedule.Outcome.NONE),
    ],
)
def test_ac8_current_outcome_reads_both_jobs(
    sweep: str, publish_result: str, expected: schedule.Outcome
) -> None:
    assert schedule.current_outcome("success", sweep, "none", publish_result) is expected


def _jobs_api(sweep: str | None, publish_job: str | None, *, skip: bool = False) -> list[Any]:
    jobs: list[dict[str, Any]] = [{"name": "gate", "conclusion": "success", "steps": [{}]}]
    if sweep is not None:
        steps = [
            {"name": schedule.SWEEP_STEP, "conclusion": "success" if sweep == "success" else sweep}
        ]
        jobs.append({"name": "sweep", "conclusion": sweep, "steps": steps})
    if publish_job is not None:
        steps = (
            []
            if publish_job == "skipped"
            else [{"name": schedule.RECORD_STEP, "conclusion": publish_job}]
        )
        jobs.append({"name": "publish", "conclusion": publish_job, "steps": steps})
    return jobs


@pytest.mark.parametrize(
    ("sweep", "publish_job", "expected"),
    [
        ("success", "skipped", schedule.Outcome.NONE),  # a skip: nothing to publish
        ("success", "failure", schedule.Outcome.FAILURE),
        ("success", "timed_out", schedule.Outcome.FAILURE),
        ("success", "success", schedule.Outcome.SUCCESS),
        ("failure", "skipped", schedule.Outcome.FAILURE),
        ("skipped", "skipped", schedule.Outcome.NONE),  # the gate closed
    ],
)
def test_ac8_run_outcome_under_the_four_job_layout(
    sweep: str, publish_job: str, expected: schedule.Outcome
) -> None:
    assert schedule.run_outcome(_jobs_api(sweep, publish_job)) is expected


def test_ac8_the_names_run_outcome_matches_are_the_workflows(jobs: dict[str, list[str]]) -> None:
    assert (schedule.GATE_JOB, schedule.SWEEP_JOB, schedule.PUBLISH_JOB) == (
        "gate",
        "sweep",
        "publish",
    )
    assert {schedule.GATE_JOB, schedule.SWEEP_JOB, schedule.PUBLISH_JOB} <= set(jobs)
    assert schedule.SWEEP_STEP in [s.get("name") for s in _steps(jobs[schedule.SWEEP_JOB])]
    assert schedule.RECORD_STEP in [s.get("name") for s in _steps(jobs[schedule.PUBLISH_JOB])]


def _run(fake: FakeGitHub, run_id: int, kind: str) -> schedule.Action:
    """One run under the four-job layout: `kind` is success, failure (the sweep job
    failed), publish-failure or skip. The alert runs while the run is in progress."""
    fake.add_run(run_id, None)
    fake.runs[0]["status"] = "in_progress"
    gh = schedule.GitHub(REPO, "t0ken", transport=fake, sleep=lambda _: None)
    sweep = "failure" if kind == "failure" else "success"
    publish_result = {
        "success": "success",
        "failure": "skipped",
        "publish-failure": "failure",
        "skip": "skipped",
    }[kind]
    sweep_stage = {"failure": "sweep", "skip": "skipped"}.get(kind, "none")
    publish_stage = {"success": "none", "publish-failure": "publish"}.get(kind, "")
    stage = schedule.alert_stage("success", sweep_stage, publish_result, publish_stage)
    action = schedule.alert(
        gh,
        run_id=run_id,
        branch="main",
        current=schedule.current_outcome("success", sweep, sweep_stage, publish_result),
        stage=stage,
        record_failures=None,
        server_url="https://github.com",
    )
    fake.runs[0]["status"] = "completed"
    fake.runs[0]["jobs"] += _jobs_api(sweep, publish_result)[1:]
    return action


def test_ac8_failure_skip_failure_counts_two() -> None:
    fake = FakeGitHub()
    fake.add_run(1, "success")
    assert _run(fake, 2, "failure") is schedule.Action.NOTHING
    assert _run(fake, 3, "skip") is schedule.Action.NOTHING
    assert _run(fake, 4, "publish-failure") is schedule.Action.NOTHING  # 2, not 3
    assert fake.issues == {}
    assert _run(fake, 5, "skip") is schedule.Action.NOTHING
    assert _run(fake, 6, "failure") is schedule.Action.OPEN  # 3: the streak held
    [issue] = fake.open_issues()
    assert "3 consecutive failures" in issue["title"]
    assert "runs/3" not in issue["body"] and "runs/5" not in issue["body"]


def test_ac8_a_skip_leaves_an_open_alert_issue_open_without_a_comment() -> None:
    fake = FakeGitHub()
    for run_id in (1, 2, 3):
        _run(fake, run_id, "publish-failure")
    [issue] = fake.open_issues()
    comments = len(issue["comments"])
    assert _run(fake, 4, "skip") is schedule.Action.NOTHING
    assert fake.open_issues() == [issue]
    assert len(issue["comments"]) == comments
    assert _run(fake, 5, "failure") is schedule.Action.COMMENT
    assert _run(fake, 6, "success") is schedule.Action.CLOSE


HOSTILE = [["sweep"], {"n": 1}, 7, 1.5, None, True]


@pytest.mark.parametrize("name", HOSTILE, ids=[type(n).__name__ for n in HOSTILE])
def test_ac8_a_hostile_job_name_is_a_typed_error(name: Any) -> None:
    jobs = [*_jobs_api("success", "success"), {"name": name, "conclusion": "success"}]
    with pytest.raises(schedule.GitHubError):
        schedule.run_outcome(jobs)


@pytest.mark.parametrize("name", HOSTILE, ids=[type(n).__name__ for n in HOSTILE])
@pytest.mark.parametrize("layout", ["four jobs", "T-007"])
def test_ac8_a_hostile_step_name_never_raises_an_untyped_error(name: Any, layout: str) -> None:
    jobs = _jobs_api("success", "skipped" if layout == "four jobs" else None)
    for job in jobs:
        job["steps"] = [{"name": name, "conclusion": "success"}, *job.get("steps", [])]
    try:
        outcome = schedule.run_outcome(jobs)
    except schedule.GitHubError:
        return
    assert isinstance(outcome, schedule.Outcome)
