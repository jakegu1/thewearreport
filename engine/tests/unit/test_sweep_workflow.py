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
    _scalar,
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


# The job that writes contents runs only what is allowed (T-034) -------------------------

# Everything a job holding `contents: write` may contain. Anything else (container,
# services, uses, strategy, defaults, ...) could run code that is not ours.
WRITE_JOB_KEYS = {"needs", "if", "runs-on", "timeout-minutes", "permissions", "outputs"}
WRITE_JOB_KEYS |= {"env", "steps"}
WRITE_STEP_KEYS = {"name", "id", "if", "uses", "with", "env", "run"}  # no shell:
# The actions such a job may use (pinned to a full SHA), with these inputs: the download
# exactly {name, path}, so it reads this run's artifact with this run's token.
WRITE_ACTIONS = {
    "actions/checkout": {"persist-credentials", "sparse-checkout"},
    "actions/download-artifact": {"name", "path"},
}
# Environment variables that change what a program loads or runs.
LOADER_ENV = re.compile(r"PYTHON\w*|LD_\w*|BASH_ENV|ENV|PATH|GIT_\w*")
# The commands a `run:` script may run, and the shell words around them.
WRITE_COMMANDS = {"git", "python3", "sed", "sort", "cat", "echo", "printf", "base64"}
WRITE_COMMANDS |= {"mapfile", "xargs", "set", "cd", "test", "[", "exit"}
SHELL_KEYWORDS = {"if", "then", "elif", "else", "do", "!"}
SHELL_ENDS = {"fi", "done"}
GIT_SUBCOMMANDS = {"init", "remote", "ls-remote", "sparse-checkout", "fetch", "checkout"}
GIT_SUBCOMMANDS |= {"ls-files", "add", "diff", "commit", "push"}
GIT_CONFIG = re.compile(r"(user\.name|user\.email|http\.extraheader)=.*", re.DOTALL)
# git options that name a program to run on the other side of a fetch or push. git also
# accepts any unambiguous prefix of a long option, so a prefix of these is refused too.
GIT_PROGRAM_OPTIONS = ("--upload-pack", "--receive-pack", "--exec")
# A sed script may only be one s command with these flags: never e (run the pattern
# space), w (write a file), or any other command.
SED_FLAGS = re.compile(r"[gpI0-9]*")
# The only option words sed, mapfile and xargs may take in a write job, each a word of
# its own. Anything else (a bundle such as -ne, a long-option prefix such as --exp, -C,
# -I, ...) is a problem: deny by default rather than model each command's parser.
SED_OPTIONS = {"-n", "-E"}  # and -e <script>, --expression=<script>
MAPFILE_OPTIONS = {"-t"}  # and -d <delimiter>
XARGS_OPTIONS = {"-0", "-r"}
PLAIN_NAME = re.compile(r"[A-Za-z_]\w*")
# The only line a write job may hold at its key indent: a plain, unquoted key.
JOB_KEY_LINE = re.compile(r"^    [\w-]+:( |$)")
# python3 runs only the engine's standard-library helpers from the checked-out engine,
# never with the working directory on sys.path (-P, or PYTHONSAFEPATH=1).
ENGINE_PATHS = {"engine", "${GITHUB_WORKSPACE}/engine"}
# Where a script may write with a redirection.
WRITE_TARGETS = re.compile(
    r"/dev/null|\$\{GITHUB_OUTPUT\}|\$\{GITHUB_ENV\}|\$\{RUNNER_TEMP\}/[\w.-]+"
)
STATUS_FUNCTION = re.compile(r"\b(always|success|failure|cancelled)\s*\(")
ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)=(.*)", re.DOTALL)
REDIRECTION = re.compile(r"[0-9]*(<<<|<<|<|>>|>|&>)(&[0-9-]+)?")
SUBSTITUTION = "$()"  # stands for a command or process substitution inside a word
# The only expansions a write job's script may hold, inside double quotes: ${NAME} and
# ${NAME[@]} (no operator), and $( opening a command substitution. Unquoted, only $? as
# the whole value of NAME=$?. Any other $ or an unquoted { could expand into a word this
# reader never sees (an option such as -ne, --upload-pack=...), so each is a problem.
QUOTED_EXPANSION = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(\[@\])?\}")
STATUS_TARGET = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
WORD_ENDS = " \t\n;&|"


def write_job_problems(text: str) -> list[str]:
    """Everything in the workflow `text` that lets a job holding `contents: write` run
    code other than git, a few shell utilities and the engine's own helpers: a job or
    step key outside the allowlists, an action or action input outside WRITE_ACTIONS, a
    loader variable in any env (workflow, job, step, inline or GITHUB_ENV), a command
    outside WRITE_COMMANDS, and a status function in the job's `if:` (it must not run
    after the sweep failed)."""
    top = _children(text.splitlines(), 0)
    found = [f"workflow: {key}" for key in top if key in ("env", "defaults")]
    for name, job in _children(top["jobs"][1:], 2).items():
        keys = _children(job, 4)
        permissions = _children(keys.get("permissions", [""])[1:], 6)
        if "write" not in "".join(permissions.get("contents", [])):
            continue
        # _children reads only plain keys: a quoted key or one with a space before the
        # colon would be dropped silently, so any other line at this indent is a problem.
        found += [
            f"{name}: unreadable line {line.strip()!r}"
            for line in job[1:]
            if line.strip()
            and not line.lstrip().startswith("#")
            and len(line) - len(line.lstrip(" ")) == 4
            and not JOB_KEY_LINE.match(line)
        ]
        found += [f"{name}: job key {k}" for k in keys if k not in WRITE_JOB_KEYS]
        env = _children(keys.get("env", [""])[1:], 6)
        found += [f"{name}: env {k}" for k in env if LOADER_ENV.fullmatch(k)]
        if "if" in keys and STATUS_FUNCTION.search(_scalar(keys["if"])):
            found.append(f"{name}: a status function in if:")
        for step in _steps(job):
            found += [f"{name}: {problem}" for problem in _step_problems(step)]
    return found


def _step_problems(step: dict[str, str]) -> list[str]:
    found = [f"step key {k}" for k in step if k not in WRITE_STEP_KEYS]
    if "uses" not in step and "run" not in step:
        # e.g. a flow-style step, `- {run: curl x}`, which _steps reads as no keys
        found.append(f"a step with neither uses nor run: {sorted(step)}")
    env = _children(step.get("env", "").splitlines(), 10)
    found += [f"step env {k}" for k in env if LOADER_ENV.fullmatch(k)]
    if "uses" in step:
        action, _, ref = step["uses"].split(" #", 1)[0].strip().partition("@")
        inputs = set(_children(step.get("with", "").splitlines(), 10))
        if action not in WRITE_ACTIONS or not re.fullmatch(r"[0-9a-f]{40}", ref):
            found.append(f"uses {step['uses']}")
        elif not inputs <= WRITE_ACTIONS[action] or (
            action == "actions/download-artifact" and inputs != WRITE_ACTIONS[action]
        ):
            found.append(f"{action} with {sorted(inputs)}")
        elif action == "actions/checkout" and "persist-credentials: false" not in step.get(
            "with", ""
        ):
            found.append("checkout keeps the token")
        if "run" in step:
            found.append("uses and run in one step")
    return found + run_problems(step.get("run", ""))


def run_problems(script: str) -> list[str]:
    """Commands in the shell `script` outside the allowlist, including every command in
    a command or process substitution and after a shell keyword; xargs may run only git.
    A script this cannot read (backquotes, unbalanced quotes) is a problem, never a pass,
    and so is any expansion outside QUOTED_EXPANSION, $( and NAME=$?."""
    try:
        commands, found = read_script(script)
    except ValueError as exc:
        return [f"unreadable script: {exc}"]
    return found + [problem for words in commands for problem in _command_problems(words)]


def shell_commands(script: str) -> list[list[str]]:
    """The simple commands in `script`, each as its words with the quotes removed. A
    command or process substitution is its own command and stands as SUBSTITUTION in
    the word that holds it. Raises ValueError for what this reader does not handle."""
    return read_script(script)[0]


def read_script(script: str) -> tuple[list[list[str]], list[str]]:
    """shell_commands(script), and the expansions in it that are problems: this reader
    removes quotes but never expands, so what an expansion produces is never checked."""
    done: list[list[str]] = []
    found: list[str] = []
    # one level per substitution: [words, current word or None, inside double quotes]
    levels: list[list[Any]] = [[[], None, False]]

    def end_word() -> None:
        if levels[-1][1] is not None:
            levels[-1][0].append(levels[-1][1])
            levels[-1][1] = None

    def end_command() -> None:
        end_word()
        if levels[-1][0]:
            done.append(levels[-1][0])
        levels[-1][0] = []

    def add(text: str) -> None:
        levels[-1][1] = (levels[-1][1] or "") + text

    i = 0
    while i < len(script):
        level, c = levels[-1], script[i]
        if c == "`":
            raise ValueError("backquotes")
        if script.startswith(("$(", "<(", ">("), i) and not script.startswith("$((", i):
            if c == "$" and not level[2]:
                found.append(f"shell expansion {script[i : i + 12]!r}")
            levels.append([[], None, False])
            i += 2
        elif level[2]:  # inside double quotes
            if c == '"':
                level[2] = False
            elif m := QUOTED_EXPANSION.match(script, i):
                add(m.group(0))
                i = m.end() - 1
            elif c == "$":
                found.append(f"shell expansion {script[i : i + 12]!r}")
                add(c)
            elif c == "\\":
                add(script[i : i + 2])
                i += 1
            else:
                add(c)
            i += 1
        elif c == ")" and len(levels) > 1:
            end_command()
            levels.pop()
            add(SUBSTITUTION)
            i += 1
        elif c == "'":
            end = script.find("'", i + 1)
            if end < 0:
                raise ValueError("unbalanced single quote")
            add(script[i + 1 : end])
            i = end + 1
        elif c == '"':
            add("")
            level[2] = True
            i += 1
        elif c == "\\":
            if script[i + 1 : i + 2] != "\n":
                add(script[i + 1 : i + 2])
            i += 2
        elif c == "#" and level[1] is None:
            i = script.find("\n", i) % (len(script) + 1)
        elif (m := REDIRECTION.match(script, i)) and (level[1] is None or c in "<>&"):
            end_word()
            level[0].append(m.group(0).lstrip("0123456789"))
            i = m.end()
        elif c in " \t":
            end_word()
            i += 1
        elif c in "\n;&|":
            end_command()
            i += 1
        elif c == "$":
            status = script.startswith("$?", i) and script[i + 2 : i + 3] in ("", *WORD_ENDS)
            if not (status and STATUS_TARGET.fullmatch(level[1] or "")):
                found.append(f"shell expansion {script[i : i + 12]!r}")
            add(c)
            i += 1
        elif c == "{":
            if level[1] is not None or script[i + 1 : i + 2] not in ("", " ", "\t", "\n"):
                found.append(f"brace expansion {(level[1] or '') + script[i : i + 12]!r}")
            add(c)
            i += 1
        else:
            add(c)
            i += 1
    if len(levels) > 1 or levels[0][2]:
        raise ValueError("unbalanced quotes or parentheses")
    end_command()
    return done, found


def _command_problems(words: list[str]) -> list[str]:
    found: list[str] = []
    plain: list[str] = []
    skip = False
    for word, target in zip(words, [*words[1:], ""], strict=True):
        if skip:  # the target of the previous redirection
            skip = False
            continue
        m = REDIRECTION.fullmatch(word) if word and word[0] in "<>&" else None
        if not m:
            plain.append(word)  # an empty word ('') stays: it is an argument
            continue
        if ">" in m.group(1) and not m.group(2):
            if target == "${GITHUB_ENV}":
                found += _github_env_problems(words)
            elif not WRITE_TARGETS.fullmatch(target):
                found.append(f"writes to {target}")
        skip = not m.group(2)
    words = plain
    while words and words[0] in SHELL_KEYWORDS:
        words = words[1:]
    assigned = {}
    while words and (m := ASSIGNMENT.fullmatch(words[0])):
        assigned[m.group(1)] = m.group(2)
        words = words[1:]
    if not words or words[0] in SHELL_ENDS:
        # an assignment alone lasts for the rest of the script
        return found + [f"sets {k}" for k in assigned if LOADER_ENV.fullmatch(k)]
    if words[0] == "for" and words[2:3] == ["in"]:
        return found  # the loop's words are data; its body is checked command by command
    command, args = words[0], words[1:]
    if command not in WRITE_COMMANDS:
        return [*found, f"runs {command}"]
    if command == "python3":
        return found + _python_problems(assigned, args)
    found += [f"sets {k} for {command}" for k in assigned if LOADER_ENV.fullmatch(k)]
    if command == "xargs":
        found += _xargs_problems(args)
    elif command == "git":
        found += _git_problems(args)
    elif command == "sed":
        found += _sed_problems(args)
    elif command == "printf" and args[:1] and args[0].startswith("-v"):
        target = args[0][2:] or "".join(args[1:2])
        found += _loader_target_problems("printf -v", target)
    elif command == "mapfile":
        found += _mapfile_problems(args)
    return found


def _loader_target_problems(command: str, target: str) -> list[str]:
    name = target.split("[", 1)[0]  # an element of a scalar is the scalar
    return [f"{command} sets {name}"] if LOADER_ENV.fullmatch(name) else []


def _xargs_problems(args: list[str]) -> list[str]:
    """xargs takes only the options in XARGS_OPTIONS, and runs git: everything from its
    first non-option word on is checked, options included, as a command of its own."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 1
    found = [f"xargs option {a!r}" for a in args[:i] if a not in XARGS_OPTIONS]
    if args[i : i + 1] != ["git"]:
        found.append("xargs runs something other than git")
    return found + _command_problems(args[i:])


def _mapfile_problems(args: list[str]) -> list[str]:
    """mapfile takes only -t and -d <delimiter>, and exactly one plain target name that
    is not a loader variable."""
    found, i = [], 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] == "-d":
            i += 1  # the delimiter, even an empty word
        elif args[i] not in MAPFILE_OPTIONS:
            found.append(f"mapfile option {args[i]!r}")
        i += 1
    targets = args[i:]
    if len(targets) != 1 or not PLAIN_NAME.fullmatch(targets[0]):
        return [*found, f"mapfile targets {targets}"]
    return found + _loader_target_problems("mapfile", targets[0])


def _sed_problems(args: list[str]) -> list[str]:
    """sed takes only -n, -E, -e <script> and --expression=<script>, anywhere in its
    arguments (GNU sed reads options after operands too); each script is one s command
    with flags from SED_FLAGS (never e or w)."""
    found, scripts, operands = [], [], []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-e":
            scripts.append("".join(args[i + 1 : i + 2]))
            i += 1
        elif arg.startswith("--expression="):
            scripts.append(arg.partition("=")[2])
        elif not arg.startswith("-"):
            operands.append(arg)
        elif arg not in SED_OPTIONS:
            found.append(f"sed option {arg!r}")
        i += 1
    if not scripts:
        scripts = operands[:1]
    return found + [f"sed script {s!r}" for s in scripts if not _plain_substitution(s)]


def _plain_substitution(script: str) -> bool:
    """Whether `script` is exactly `s<d>regex<d>replacement<d>flags`, flags in SED_FLAGS."""
    if len(script) < 2 or script[0] != "s" or script[1] in "\\\n":
        return False
    delimiter, i, parts = script[1], 2, 0
    while i < len(script) and parts < 2:
        if script[i] == "\\":
            i += 1
        elif script[i] == delimiter:
            parts += 1
        i += 1
    return parts == 2 and SED_FLAGS.fullmatch(script[i:]) is not None


def _github_env_problems(words: list[str]) -> list[str]:
    """A write to GITHUB_ENV sets a variable for every later step: only `echo NAME=...`
    of a name that is not a loader variable."""
    m = ASSIGNMENT.fullmatch(words[1]) if len(words) == 4 and words[0] == "echo" else None
    if m is None:
        return [f"writes GITHUB_ENV: {' '.join(words)}"]
    return [f"sets {m.group(1)} in GITHUB_ENV"] if LOADER_ENV.fullmatch(m.group(1)) else []


def _python_problems(assigned: dict[str, str], args: list[str]) -> list[str]:
    found = []
    for name, value in assigned.items():
        allowed = (name == "PYTHONPATH" and value in ENGINE_PATHS) or (
            name == "PYTHONSAFEPATH" and value == "1"
        )
        if not allowed:
            found.append(f"sets {name}={value} for python3")
    safe = assigned.get("PYTHONSAFEPATH") == "1"
    if args[:1] == ["-P"]:
        safe, args = True, args[1:]
    if args[:2] != ["-m", "wearreport.schedule"]:
        found.append(f"python3 runs {' '.join(args[:2])}")
    if not safe:
        found.append("python3 without -P: the working directory is on sys.path")
    return found


def _git_problems(args: list[str]) -> list[str]:
    found = []
    while args[:1] == ["-c"]:
        if not GIT_CONFIG.fullmatch("".join(args[1:2])):
            found.append(f"git -c {''.join(args[1:2])}")
        args = args[2:]
    if args[:1] == [] or args[0] not in GIT_SUBCOMMANDS:
        found.append(f"git {''.join(args[:1])}")
    for arg in args[1 : (args.index("--") if "--" in args else len(args))]:
        option = arg.partition("=")[0]
        if len(option) > 2 and any(p.startswith(option) for p in GIT_PROGRAM_OPTIONS):
            found.append(f"git {args[0]} {option}")
    return found


def test_the_write_job_runs_only_allowed_code() -> None:
    assert write_job_problems(WORKFLOW.read_text(encoding="utf-8")) == []


def test_the_shell_reader_finds_every_command() -> None:
    script = (
        "set -euo pipefail\n"
        "# a comment; rm -rf /\n"
        'auth="$(printf \'x:%s\' "${T}" | base64 -w0)"\n'
        "mapfile -t days < <(sed -n 's#\\(a\\)#\\1#p' \"f\" | sort -u)\n"
        'if [ "${rc}" -eq 0 ]; then echo "a (b); c" >&2; fi\n'
        "git ls-files -z -- x \\\n  | xargs -0 -r git add -- 2> /dev/null\n"
    )
    assert shell_commands(script) == [
        ["set", "-euo", "pipefail"],
        ["printf", "x:%s", "${T}"],
        ["base64", "-w0"],
        ["auth=$()"],
        ["sed", "-n", "s#\\(a\\)#\\1#p", "f"],
        ["sort", "-u"],
        ["mapfile", "-t", "days", "<", "$()"],
        ["if", "[", "${rc}", "-eq", "0", "]"],
        ["then", "echo", "a (b); c", ">&2"],
        ["fi"],
        ["git", "ls-files", "-z", "--", "x"],
        ["xargs", "-0", "-r", "git", "add", "--", ">", "/dev/null"],
    ]
    for bad in ("echo `id`", "echo 'x", 'echo "x', "echo $(id"):
        with pytest.raises(ValueError):
            shell_commands(bad)


def _into_publish(text: str, where: str, mutant: str) -> str:
    start = text.index("\n  publish:\n")
    at = text.index(where, start) + len(where)
    return text[:at] + mutant + text[at:]


@pytest.mark.parametrize(
    ("where", "mutant"),
    [
        ("\n  publish:\n", "    container: node:22\n"),
        ("\n  publish:\n", "    services:\n      cache:\n        image: redis:7\n"),
        ("\n  publish:\n", "    strategy:\n      matrix:\n        x: [1]\n"),
        ("\n  publish:\n", "    defaults:\n      run:\n        shell: python {0}\n"),
        ("\n  publish:\n", "    uses: ./.github/workflows/other.yml\n"),
        ("\n    env:\n", "      PYTHONPATH: /tmp/x\n"),
        ("\n    env:\n", "      GIT_CONFIG_GLOBAL: /tmp/x\n"),
    ],
)
def test_the_write_job_check_catches_a_job_key(where: str, mutant: str) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert write_job_problems(_into_publish(text, where, mutant)) != []


@pytest.mark.parametrize(
    "mutant",
    [
        "      - run: docker run --rm evil/img\n",
        "      - run: npx some-tool\n",
        "      - run: node x.js\n",
        '      - run: bash "${RUNNER_TEMP}/artifact/x"\n',
        "      - run: . ./x\n",
        "      - run: eval x\n",
        "      - run: echo x | sh\n",
        "      - run: echo $(curl -s https://example.org)\n",
        '      - run: echo "$(curl -s https://example.org)"\n',
        "      - run: git status && make\n",
        "      - run: xargs sh -c x < list\n",
        "      - run: git -c core.fsmonitor=x status\n",
        "      - run: git config core.hooksPath x\n",
        "      - run: sed -i s/a/b/ x\n",
        "      - run: echo x > .git/config\n",
        "      - run: python3 -m wearreport.schedule gate\n",
        "      - run: PYTHONSAFEPATH=1 python3 x.py\n",
        "      - run: PYTHONSAFEPATH=1 PYTHONPATH=/tmp python3 -m wearreport.schedule gate\n",
        "      - run: PATH=/tmp/x\n",
        '      - run: echo "PYTHONPATH=/tmp" >> "${GITHUB_ENV}"\n',
        '      - run: cat x >> "${GITHUB_ENV}"\n',
        '      - run: echo /tmp >> "${GITHUB_PATH}"\n',
        "      - name: x\n        shell: python {0}\n        run: print(1)\n",
        "      - name: x\n        env:\n          PYTHONPATH: /tmp/artifact\n        run: echo\n",
        "      - name: x\n        env:\n          LD_PRELOAD: /tmp/x.so\n        run: echo\n",
        "      - name: x\n        env:\n          BASH_ENV: /tmp/x\n        run: echo\n",
        "      - uses: actions/checkout@v7\n",
        "      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7\n",
        "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1\n"
        "        with:\n          ref: other\n          persist-credentials: false\n",
        "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1\n",
    ],
)
def test_the_write_job_check_catches_a_step(mutant: str) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert write_job_problems(_into_publish(text, "    steps:\n", mutant)) != []


@pytest.mark.parametrize(
    ("where", "mutant"),
    [
        # YAML spellings the line-based reader does not read as keys
        ("    steps:\n", "      - {run: curl x}\n"),
        ("    steps:\n", "      - {uses: evil/img@v1}\n"),
        ("\n  publish:\n", '    "container": node:22\n'),
        ("\n  publish:\n", "    'container': node:22\n"),
        ("\n  publish:\n", "    container : node:22\n"),
        ("\n  publish:\n", "    ? container\n    : node:22\n"),
        ("\n  publish:\n", "    {container: node:22}\n"),
    ],
)
def test_the_write_job_check_catches_an_unread_spelling(where: str, mutant: str) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert write_job_problems(_into_publish(text, where, mutant)) != []


@pytest.mark.parametrize(
    "mutant",
    [
        # allowed commands that can still run code or set a loader variable
        "      - run: sed -n 's/x/id/e' f\n",
        "      - run: sed 'e id' f\n",
        "      - run: sed 's/x/y/w /tmp/x' f\n",
        "      - run: sed 'w /tmp/x' f\n",
        "      - run: sed -e 's/a/b/' -e 'e id' f\n",
        "      - run: sed --expression='s/a/b/;e id' f\n",
        "      - run: sed -f /tmp/script f\n",
        "      - run: git fetch --upload-pack=/tmp/x origin\n",
        "      - run: git fetch --upload-p=/tmp/x origin\n",
        "      - run: git ls-remote --upload-pack /tmp/x origin\n",
        "      - run: git push --receive-pack=/tmp/x origin\n",
        "      - run: git push --exec=/tmp/x origin\n",
        "      - run: printf -v PATH '%s' /tmp\n",
        "      - run: printf -vLD_PRELOAD '%s' /tmp/x.so\n",
        "      - run: printf -v 'PYTHONPATH[0]' '%s' /tmp\n",
        "      - run: mapfile -t PATH < f\n",
        "      - run: mapfile -d x -n 1 GIT_DIR < f\n",
        "      - run: mapfile -C id -c 1 x < f\n",
    ],
)
def test_the_write_job_check_catches_an_allowed_command_running_code(mutant: str) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert write_job_problems(_into_publish(text, "    steps:\n", mutant)) != []


@pytest.mark.parametrize(
    "mutant",
    [
        # option spellings that sed, mapfile or xargs read but a flag parser can miss
        "      - run: sed -e 's/a/b/' -se 'e echo PWNED' f\n",
        "      - run: sed -e 's/a/b/' -ne 'e echo PWNED' f\n",
        "      - run: sed -n -e 's/a/b/' -Ee 'e echo PWNED' f\n",
        "      - run: sed --exp 's/a/b/' --exp 'e echo PWNED' f\n",
        "      - run: sed --e 's/a/b/' --e 'e echo PWNED' f\n",
        "      - run: sed -e 's/a/b/' --expr='e echo PWNED' f\n",
        "      - run: mapfile -tu 0 PATH < f\n",
        "      - run: mapfile -tC id x < f\n",
        "      - run: mapfile -d '' PATH < f\n",
        "      - run: xargs git fetch --upload-pack=/x < f\n",
        "      - run: xargs git ls-remote --upload-pack /x < f\n",
        "      - run: xargs git push --receive-pack=/x < f\n",
    ],
)
def test_the_write_job_check_catches_an_option_spelling(mutant: str) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert write_job_problems(_into_publish(text, "    steps:\n", mutant)) != []


@pytest.mark.parametrize(
    "option",
    [
        *("-s", "-ne", "-Ee", "-i", "-f", "-z", "--posix", "--", "-"),
        *("--e", "--exp", "--expr=s/a/b/", "--expression"),
    ],
)
def test_sed_takes_no_other_option(option: str) -> None:
    assert f"sed option {option!r}" in run_problems(f"sed -n {option} 's/a/b/p' f")
    assert f"sed option {option!r}" in run_problems(f"sed -n 's/a/b/p' f {option}")


@pytest.mark.parametrize(
    "option", ["-tu", "-tC", "-u", "-C", "-c", "-n", "-O", "-s", "-td", "--t", "--"]
)
def test_mapfile_takes_no_other_option(option: str) -> None:
    assert f"mapfile option {option!r}" in run_problems(f"mapfile -t {option} days < f")


@pytest.mark.parametrize(
    "option", ["-0r", "-i", "-I{}", "-a", "-n", "-P", "-e", "-x", "--nu", "--null", "--"]
)
def test_xargs_takes_no_other_option(option: str) -> None:
    assert f"xargs option {option!r}" in run_problems(f"xargs -0 {option} git add -- < f")


@pytest.mark.parametrize(
    ("mutant", "kind"),
    [
        # an expansion the reader does not expand, producing an option word or program
        ("      - run: sed -e 's/a/b/' {-n,-e} 'e echo PWNED' f\n", "brace expansion"),
        ("      - run: sed -e 's/a/b/' $'-ne' 'e echo PWNED' f\n", "shell expansion"),
        ("      - run: sed -e 's/a/b/' $\"-ne\" 'e echo PWNED' f\n", "shell expansion"),
        ("      - run: o=-ne; sed -e 's/a/b/' $o 'e echo PWNED' f\n", "shell expansion"),
        ("      - run: xargs -0 -r git fetch {--upload-pack=/x,} < f\n", "brace expansion"),
        ("      - run: git fetch {--upload-pack=/x,} origin\n", "brace expansion"),
    ],
)
def test_the_write_job_check_catches_an_expansion(mutant: str, kind: str) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    problems = write_job_problems(_into_publish(text, "    steps:\n", mutant))
    assert any(p.startswith(f"publish: {kind}") for p in problems), problems


@pytest.mark.parametrize(
    "word",
    [
        *("$HOME", "${HOME}", "$'-ne'", '$"-ne"', '"$((1+1))"', '"$@"', '"$*"', '"$1"'),
        *('"${!x}"', '"${x:-y}"', '"${x#y}"', '"${x%y}"', '"${x/a/b}"', '"${x^}"'),
        *('"$HOME"', '"$?"', '"${#x}"', '"${x[0]}"', "$(id)", "$?"),
    ],
)
def test_a_write_job_script_takes_no_other_shell_expansion(word: str) -> None:
    problems = run_problems(f"echo {word} x")
    assert any(p.startswith("shell expansion") for p in problems), problems


@pytest.mark.parametrize("word", ["{a,b}", "x{1..3}", "{-n,-e}", '"a"{b,c}', "{,}"])
def test_a_write_job_script_takes_no_brace_expansion(word: str) -> None:
    problems = run_problems(f"echo {word} x")
    assert any(p.startswith("brace expansion") for p in problems), problems


def test_mapfile_takes_exactly_one_plain_target() -> None:
    for script in ("mapfile -t < f", "mapfile -t a b < f", "mapfile -t 'a[0]' < f"):
        assert any(p.startswith("mapfile targets") for p in run_problems(script)), script
    # -d consumes its delimiter even when it is an empty word, never the target after it
    assert run_problems("mapfile -d '' days < f") == []
    assert run_problems("mapfile -d '' PATH < f") == ["mapfile sets PATH"]


def test_the_write_job_check_allows_plain_uses_of_those_commands() -> None:
    for script in (
        "sed -n 's#^\\(a/[0-9]\\{4\\}\\)/[^/]*$#/\\1/#p' f",
        "sed -e 's/a/b/g' -e 's|c|d|2' f",
        "git fetch -q --depth=1 --no-tags origin refs/heads/data",
        "git ls-remote --exit-code --heads origin refs/heads/data",
        "git add -- --exec",
        "printf -v auth '%s' x",
        "printf '%s' x",
        "mapfile -t days < f",
        "mapfile -t -d x days < f",
        'rc=0; a="$(printf \'%s\' "${GITHUB_WORKSPACE}/x" "${days[@]}")" || rc=$?',
    ):
        assert run_problems(script) == [], script


def test_the_write_job_check_catches_a_changed_download_or_if() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    download = "          name: sweep-records\n"
    for extra in ("          run-id: 1\n", "          github-token: x\n"):
        assert write_job_problems(_into_publish(text, download, extra)) != []
    old = "    if: needs.sweep.outputs.skipped != 'true'\n"
    assert old in text
    for cond in ("always() && ", "success() && ", "!cancelled() && "):
        assert write_job_problems(text.replace(old, old.replace("if: ", f"if: {cond}"))) != []


def test_the_write_job_check_catches_workflow_env_and_defaults() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for block in ("env:\n  PYTHONPATH: /tmp\n", "defaults:\n  run:\n    shell: sh\n"):
        assert write_job_problems(text.replace("\njobs:\n", f"\n{block}\njobs:\n")) != []


def test_the_commit_step_never_imports_from_the_data_checkout(
    tmp_path: Path, jobs: dict[str, list[str]]
) -> None:
    # The commit step runs check-staged from the data checkout. A `wearreport` package
    # there must not shadow the engine's: PYTHONSAFEPATH=1 (python3 -P) keeps the
    # working directory off sys.path.
    script = _script(jobs, "publish", "Commit and push")
    assert "PYTHONSAFEPATH=1 \\\n" in script
    assert "python3 -m wearreport.schedule check-staged" in script
    shadow = tmp_path / "wearreport"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(f"open({str(tmp_path / 'ran')!r}, 'w').close()\n")
    env = {**_env(), "PYTHONPATH": str(ROOT / "engine")}
    for safe, ran in (({"PYTHONSAFEPATH": "1"}, False), ({}, True)):
        out = subprocess.run(
            [sys.executable, "-m", "wearreport.schedule", "check-staged"],
            input="A\tstatus.json\n",
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**env, **safe},
            timeout=60,
        )
        assert (tmp_path / "ran").exists() is ran, out.stderr
        (tmp_path / "ran").unlink(missing_ok=True)
