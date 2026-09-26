"""Static checks over .github/workflows/: actionlint reports nothing, and no workflow- or
job-level `env` uses a context GitHub does not allow there (a file GitHub rejects at load
time never runs, so nothing would alert on it)."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
ACTIONLINT = ROOT / ".tools" / "bin" / "actionlint"
ACTIONLINT_ARGS = ("-shellcheck=", "-pyflakes=")  # the same flags as `make workflows`
REQUIRE_ENV = "WEARREPORT_REQUIRE_ACTIONLINT"
# Contexts allowed in workflow-level and jobs.<job_id>.env: no runner, job, steps or env.
ALLOWED_AT_JOB_LEVEL = frozenset(
    {"github", "inputs", "matrix", "needs", "secrets", "strategy", "vars"}
)


def _actionlint() -> Path:
    if not ACTIONLINT.is_file():
        if os.environ.get(REQUIRE_ENV):
            pytest.fail(f"actionlint is missing and {REQUIRE_ENV} is set")
        pytest.skip("actionlint is not installed (run `make setup`)")
    return ACTIONLINT


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_actionlint()), *ACTIONLINT_ARGS, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _workflow_files() -> list[Path]:
    files = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    assert files
    return files


def test_actionlint_reports_nothing_on_any_workflow() -> None:
    result = _run(*(str(p) for p in _workflow_files()), cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_actionlint_rejects_the_runner_context_in_job_level_env(tmp_path: Path) -> None:
    # The error that made sweep.yml fail to load: proves the check above is not a no-op.
    bad = tmp_path / "bad.yml"
    bad.write_text(
        "on: workflow_dispatch\n"
        "jobs:\n"
        "  job:\n"
        "    runs-on: ubuntu-24.04\n"
        "    env:\n"
        "      DATA_DIR: ${{ runner.temp }}/data\n"
        "    steps:\n"
        '      - run: echo "${DATA_DIR}"\n',
        encoding="utf-8",
    )
    result = _run(str(bad), cwd=tmp_path)
    assert result.returncode != 0
    assert 'context "runner" is not allowed here' in result.stdout


def _level_env_contexts(text: str) -> list[tuple[int, str]]:
    """(line number, context) for each expression context used in the workflow-level
    `env` (indent 0) or a job-level `env` (indent 4, under `jobs:`)."""
    found: list[tuple[int, str]] = []
    env_indent: int | None = None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if env_indent is not None and indent <= env_indent:
            env_indent = None
        if re.fullmatch(r"(|    )env:\s*", line):
            env_indent = indent
            continue
        if env_indent is not None:
            for expression in re.findall(r"\$\{\{(.*?)\}\}", line):
                found += [(number, name) for name in re.findall(r"\b([a-z_]+)\.", expression)]
    return found


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_workflow_and_job_level_env_use_only_allowed_contexts(path: Path) -> None:
    used = _level_env_contexts(path.read_text(encoding="utf-8"))
    assert [(n, c) for n, c in used if c not in ALLOWED_AT_JOB_LEVEL] == []


def test_the_env_context_reader_finds_a_runner_context_at_job_level() -> None:
    text = (
        "env:\n  A: ${{ github.sha }}\n"
        "jobs:\n  j:\n    env:\n      B: ${{ runner.temp }}/x\n"
        "    steps:\n      - env:\n          C: ${{ runner.temp }}\n"
    )
    assert _level_env_contexts(text) == [(2, "github"), (6, "runner")]
