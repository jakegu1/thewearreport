"""Acceptance tests for T-080: the labelling launcher (`scripts/label.ps1`) writes its
attribute files outside the git work tree by default, runs the version it pulled, prints
the commit it runs and shows why an update command failed. The task contract: do not edit.

On Linux these tests read the launcher, the guide and the Windows workflow as text. On
Windows (the windows-latest CI job) the launcher runs end to end with its test switches and
the same fakes as the T-067 tests: spotcheck's dry-run pipeline (the licensed fixture
photos, served on this machine), the `--judgements` file reviewer answered by a test
script, fake `git` and `uv` on PATH and a file in place of the clipboard. The restart and
version tests copy the launcher into a scratch repository layout whose fake `git` rewrites
the copy on `pull`. Nothing reaches the network, and no image is opened or looked at: the
test reviewer reads only the review's numbering file.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from wearreport.tools import spotcheck

ROOT = Path(__file__).resolve().parents[3]
LABEL_PS1 = ROOT / "scripts" / "label.ps1"
LABEL_CMD = ROOT / "label.cmd"
LABEL_LONDON_CMD = ROOT / "label-london.cmd"
SPOTCHECKS = ROOT / "spotchecks"
SPOTCHECKS_README = SPOTCHECKS / "README.md"
WINDOWS_WORKFLOW = ROOT / ".github" / "workflows" / "windows.yml"
WINDOWS = sys.platform == "win32"
WAIT_S = 900
LABELS = "wearreport-labels"
RESTART_LINE = "Update: the launcher changed; restarting it."
VERSION = re.compile(r"^Version: (\S+)$", re.MULTILINE)

windows_only = pytest.mark.skipif(
    not WINDOWS, reason="the launcher is a Windows script; the windows-latest CI job runs it"
)


# The launcher's files, as text ----------------------------------------------------------


def test_ac1_default_out_dir_is_outside_the_repository() -> None:
    text = LABEL_PS1.read_text(encoding="utf-8")
    assert "D:\\" + LABELS in text or "'D:\\'" in text
    assert LABELS in text and "USERPROFILE" in text
    assert "Join-Path $Repo 'spotchecks'" not in text


def test_ac3_ac4_launcher_text_names_the_restart_and_version_lines() -> None:
    text = LABEL_PS1.read_text(encoding="utf-8")
    assert RESTART_LINE in text
    assert "Version: " in text and "unknown" in text
    assert "packed-refs" in text


def test_launcher_files_are_still_ascii() -> None:
    for path in (LABEL_PS1, LABEL_CMD, LABEL_LONDON_CMD):
        path.read_bytes().decode("ascii")


def test_ac1_guide_says_where_the_launcher_writes_label_files() -> None:
    text = SPOTCHECKS_README.read_text(encoding="utf-8")
    assert LABELS in text


def test_ac6_windows_job_runs_this_file_and_stays_read_only() -> None:
    text = WINDOWS_WORKFLOW.read_text(encoding="utf-8")
    assert "engine/tests/acceptance/test_t_080.py" in text
    assert "runs-on: windows-latest" in text
    assert re.search(r"permissions:\s*\n\s*contents: read", text)
    assert "secrets." not in text and "pull_request_target" not in text


def test_ac6_public_guard_is_clean() -> None:
    for extra in ([], ["--history"]):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "public_guard.py"), *extra],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


# The launcher end to end, on Windows ----------------------------------------------------

# The test reviewer, as in the T-067 tests: it answers the `--judgements` file from the
# review's numbering file (never an image) and runs spotcheck's main. It logs its argument
# list, never a key.
SHIM = r"""
import json, os, sys, tempfile, threading, time
from pathlib import Path
from wearreport.tools import spotcheck

argv = sys.argv[1:]
log = Path(os.environ["LABEL_TEST_LOG"])
with log.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"argv": argv, "cwd": os.getcwd()}) + "\n")

def answer(path):
    root = Path(tempfile.gettempdir())
    while True:
        for d in root.glob(spotcheck.TEMP_PREFIX + "*"):
            try:
                data = json.loads((d / spotcheck.NUMBERING_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            keys = sorted(data["template"], key=int)
            answers = {k: {"outer_layer": "y", "bare_legs": "n", "umbrella": "n"} for k in keys}
            path.write_text(json.dumps(answers), encoding="utf-8")
            return
        time.sleep(0.2)

if "--judgements" in argv:
    target = Path(argv[argv.index("--judgements") + 1])
    threading.Thread(target=answer, args=(target,), daemon=True).start()
raise SystemExit(spotcheck.main(argv))
"""

FAKE_CMD = (
    '@echo off\r\necho %~n0 %*>>"%LABEL_TEST_TOOLS_LOG%"\r\nexit /b %LABEL_TEST_TOOL_EXIT%\r\n'
)

# A fake git whose pull fails with a known message on stderr, after other lines.
PULL_ERROR = "fatal: Not possible to fast-forward, aborting."
FAILING_GIT_CMD = (
    "@echo off\r\n"
    'echo %~n0 %*>>"%LABEL_TEST_TOOLS_LOG%"\r\n'
    "echo hint: first line 1>&2\r\n"
    "echo hint: second line 1>&2\r\n"
    "(echo.) 1>&2\r\n"
    "echo hint: third line 1>&2\r\n"
    "echo hint: Diverging branches can't be fast-forwarded 1>&2\r\n"
    f"echo {PULL_ERROR} 1>&2\r\n"
    "exit /b 1\r\n"
)

# The fake git of the restart test: on `pull` it rewrites the scratch launcher so that the
# new version prints a marker and ends with exit code 7.
MARKER = "T080-NEW-LAUNCHER-RAN"
REWRITE = r"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="ascii")
anchor = "Set-StrictMode -Version 3.0"
assert anchor in text and text.rstrip().endswith("exit 0")
text = text.replace(anchor, anchor + "\r\nWrite-Host '" + sys.argv[2] + "'", 1)
text = text.rstrip()[: -len("exit 0")] + "exit 7\r\n"
path.write_text(text, encoding="ascii")
"""


@dataclass
class Session:
    code: int
    out: str
    calls: list[dict[str, object]]
    tools: list[str]


@dataclass
class Launcher:
    tmp: Path
    key: str = field(default_factory=lambda: "fake-test-key-" + secrets.token_hex(12))

    @property
    def temp_dir(self) -> Path:
        return self.tmp / "session-tmp"

    @property
    def clipboard_file(self) -> Path:
        return self.tmp / "clipboard.txt"

    @property
    def fakes(self) -> Path:
        return self.tmp / "fakebin"

    def write_fakes(self, git: str = FAKE_CMD) -> None:
        self.fakes.mkdir(exist_ok=True)
        (self.fakes / "git.cmd").write_text(git, encoding="ascii")
        (self.fakes / "uv.cmd").write_text(FAKE_CMD, encoding="ascii")

    def run(
        self,
        *args: str,
        script: Path = LABEL_PS1,
        stdin: str = "\n\n\n\n",
        tool_exit: int = 0,
        env_extra: dict[str, str] | None = None,
    ) -> Session:
        if not self.fakes.is_dir():
            self.write_fakes()
        shim = self.tmp / "shim.py"
        shim.write_text(SHIM, encoding="utf-8")
        log = self.tmp / "calls.jsonl"
        tools_log = self.tmp / "tools.log"
        for path in (log, tools_log):
            path.unlink(missing_ok=True)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in spotcheck.CI_VARIABLES and k not in ("DEEPINFRA_API_KEY", "TMPDIR")
        }
        env["PATH"] = str(self.fakes) + os.pathsep + env.get("PATH", "")
        env["LABEL_TEST_LOG"] = str(log)
        env["LABEL_TEST_TOOLS_LOG"] = str(tools_log)
        env["LABEL_TEST_TOOL_EXIT"] = str(tool_exit)
        env["DEEPINFRA_API_KEY"] = self.key
        env.update(env_extra or {})
        options = [
            "-TempDir",
            str(self.temp_dir),
            "-Python",
            sys.executable,
            "-Spotcheck",
            str(shim),
            "-ClipboardFile",
            str(self.clipboard_file),
            "-Now",
            "2026-06-21T19:00Z",
            "-NoJudge",
            "-DryRun",
            *args,
        ]
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *options,
        ]
        result = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=self.tmp,
            timeout=WAIT_S,
            check=False,
        )
        out = result.stdout + result.stderr
        print(out)
        calls = (
            [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            if log.exists()
            else []
        )
        tools = tools_log.read_text(encoding="utf-8").splitlines() if tools_log.exists() else []
        assert self.key not in out
        return Session(result.returncode, out, calls, tools)


@pytest.fixture
def launcher(tmp_path: Path) -> Launcher:
    return Launcher(tmp_path)


def _value(argv: object, flag: str) -> str:
    assert isinstance(argv, list)
    return str(argv[argv.index(flag) + 1])


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    """Every file under `root`: its size and modification time."""
    found: dict[str, tuple[int, int]] = {}
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            path = Path(dirpath, name)
            stat = path.stat()
            found[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return found


def _attribute_files(out_dir: Path) -> list[Path]:
    attributes = out_dir / spotcheck.ATTRIBUTES_DIR
    return sorted(attributes.glob("*.json")) if attributes.is_dir() else []


# AC1: the default folder ------------------------------------------------------------------


@windows_only
def test_ac1_default_folder_on_the_labels_drive(launcher: Launcher, tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    drive.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    before = _tree(SPOTCHECKS)
    session = launcher.run(
        "-Target", "1", "-LabelsDrive", str(drive), env_extra={"USERPROFILE": str(home)}
    )
    assert session.code == 0
    assert _tree(SPOTCHECKS) == before
    expected = drive / LABELS
    assert len(session.calls) == 1
    assert Path(_value(session.calls[0]["argv"], "--out-dir")) == expected
    files = _attribute_files(expected)
    assert len(files) == 1
    assert files[0].name in session.out
    assert not (home / LABELS).exists()
    assert launcher.clipboard_file.read_text(encoding="utf-8").strip() == (
        files[0].read_text(encoding="utf-8").strip()
    )


@windows_only
def test_ac1_default_folder_in_the_profile_without_the_drive(
    launcher: Launcher, tmp_path: Path
) -> None:
    drive = tmp_path / "no-such-drive"
    home = tmp_path / "home"
    home.mkdir()
    before = _tree(SPOTCHECKS)
    session = launcher.run(
        "-Target", "1", "-LabelsDrive", str(drive), env_extra={"USERPROFILE": str(home)}
    )
    assert session.code == 0
    assert _tree(SPOTCHECKS) == before
    expected = home / LABELS
    assert expected.is_dir()
    assert Path(_value(session.calls[0]["argv"], "--out-dir")) == expected
    assert len(_attribute_files(expected)) == 1
    assert not drive.exists()


# AC2: -OutDir wins ------------------------------------------------------------------------


@windows_only
def test_ac2_out_dir_still_wins(launcher: Launcher, tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    drive.mkdir()
    out = tmp_path / "out"
    before = _tree(SPOTCHECKS)
    session = launcher.run("-Target", "1", "-OutDir", str(out), "-LabelsDrive", str(drive))
    assert session.code == 0
    assert _tree(SPOTCHECKS) == before
    assert Path(_value(session.calls[0]["argv"], "--out-dir")) == out
    files = _attribute_files(out)
    assert len(files) == 1 and files[0].parent == out / "attributes"
    assert not (drive / LABELS).exists()


# AC3 and AC4: a scratch repository ---------------------------------------------------------


def _scratch_repo(tmp_path: Path, git_dir: str = "loose") -> tuple[Path, str]:
    """A repository layout with a copy of the launcher and a `.git` whose HEAD is a random
    commit, stored as `git_dir` says: a loose ref, packed-refs, a detached HEAD, a missing
    ref or no `.git` at all."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copyfile(LABEL_PS1, repo / "scripts" / "label.ps1")
    sha = secrets.token_hex(20)
    git = repo / ".git"
    if git_dir != "none":
        (git / "refs" / "heads").mkdir(parents=True)
    if git_dir == "loose":
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
        (git / "refs" / "heads" / "main").write_text(sha + "\n", encoding="ascii")
    elif git_dir == "packed":
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
        (git / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted\n"
            f"{secrets.token_hex(20)} refs/heads/other\n"
            f"{sha} refs/heads/main\n",
            encoding="ascii",
        )
    elif git_dir == "detached":
        (git / "HEAD").write_text(sha + "\n", encoding="ascii")
    elif git_dir == "broken":
        (git / "HEAD").write_text("ref: refs/heads/missing\n", encoding="ascii")
    return repo, sha


def _restarting_git(launcher: Launcher, repo: Path) -> None:
    rewrite = launcher.tmp / "rewrite.py"
    rewrite.write_text(REWRITE, encoding="utf-8")
    target = repo / "scripts" / "label.ps1"
    git = (
        "@echo off\r\n"
        'echo %~n0 %*>>"%LABEL_TEST_TOOLS_LOG%"\r\n'
        f'if /i "%1"=="pull" "{sys.executable}" "{rewrite}" "{target}" {MARKER}\r\n'
        "exit /b %LABEL_TEST_TOOL_EXIT%\r\n"
    )
    launcher.write_fakes(git=git)


@windows_only
def test_ac3_a_changed_launcher_restarts_once_and_runs_the_new_version(
    launcher: Launcher, tmp_path: Path
) -> None:
    repo, _sha = _scratch_repo(tmp_path)
    _restarting_git(launcher, repo)
    out = tmp_path / "out"
    session = launcher.run(
        "-Target", "1000", "-MaxPasses", "2", "-OutDir", str(out),
        script=repo / "scripts" / "label.ps1",
    )  # fmt: skip
    assert MARKER in (repo / "scripts" / "label.ps1").read_text(encoding="ascii")
    lines = session.out.splitlines()
    assert lines.count(RESTART_LINE) == 1
    assert len([line for line in lines if MARKER in line]) == 1
    assert [t for t in session.tools if t.startswith("git")] == ["git pull --ff-only"]
    assert [t for t in session.tools if t.startswith("uv")] == [
        "uv sync --locked --no-install-package llama-cpp-python"
    ]
    assert len(session.calls) == 2
    assert len(_attribute_files(out)) == 2
    for call in session.calls:
        assert Path(_value(call["argv"], "--out-dir")) == out
        assert Path(str(call["cwd"])).resolve() == repo.resolve()
    assert len(VERSION.findall(session.out)) == 1
    assert session.code == 7


@windows_only
def test_ac3_an_unchanged_launcher_does_not_restart(launcher: Launcher, tmp_path: Path) -> None:
    repo, _sha = _scratch_repo(tmp_path)
    out = tmp_path / "out"
    session = launcher.run(
        "-Target", "1000", "-MaxPasses", "2", "-OutDir", str(out),
        script=repo / "scripts" / "label.ps1",
    )  # fmt: skip
    assert RESTART_LINE not in session.out
    assert "restarting" not in session.out
    assert MARKER not in session.out
    assert session.tools == [
        "git pull --ff-only",
        "uv sync --locked --no-install-package llama-cpp-python",
    ]
    assert len(session.calls) == 2
    assert session.code == 0


@windows_only
@pytest.mark.parametrize("git_dir", ["loose", "packed", "detached"])
def test_ac4_version_is_the_head_commit(launcher: Launcher, tmp_path: Path, git_dir: str) -> None:
    repo, sha = _scratch_repo(tmp_path, git_dir)
    session = launcher.run(
        "-Target", "1", "-OutDir", str(tmp_path / "out"), script=repo / "scripts" / "label.ps1"
    )
    assert session.code == 0
    assert VERSION.findall(session.out) == [sha[:7]]
    assert len(session.calls) == 1


@windows_only
@pytest.mark.parametrize("git_dir", ["broken", "none"])
def test_ac4_an_unreadable_git_prints_unknown_and_runs(
    launcher: Launcher, tmp_path: Path, git_dir: str
) -> None:
    repo, _sha = _scratch_repo(tmp_path, git_dir)
    out = tmp_path / "out"
    session = launcher.run(
        "-Target", "1", "-OutDir", str(out), script=repo / "scripts" / "label.ps1"
    )
    assert session.code == 0
    assert VERSION.findall(session.out) == ["unknown"]
    assert len(session.calls) == 1 and len(_attribute_files(out)) == 1


@windows_only
def test_ac4_version_appears_once_in_a_normal_session(launcher: Launcher, tmp_path: Path) -> None:
    session = launcher.run("-Target", "1", "-OutDir", str(tmp_path / "out"))
    assert session.code == 0
    versions = VERSION.findall(session.out)
    assert len(versions) == 1
    assert versions[0] == "unknown" or re.fullmatch(r"[0-9a-f]{7}", versions[0])


# AC5: why an update failed ---------------------------------------------------------------


@windows_only
def test_ac5_a_failed_pull_shows_its_last_lines(launcher: Launcher, tmp_path: Path) -> None:
    launcher.write_fakes(git=FAILING_GIT_CMD)
    session = launcher.run("-Target", "1", "-OutDir", str(tmp_path / "out"))
    assert session.code == 0
    lines = session.out.splitlines()
    failed = [i for i, line in enumerate(lines) if "git pull failed" in line]
    assert len(failed) == 1
    assert "continuing with the current version." in lines[failed[0]]
    following = []
    for line in lines[failed[0] + 1 :]:
        if not line.startswith("  "):
            break
        following.append(line)
    assert 1 <= len(following) <= 3
    assert following[-1] == "  " + PULL_ERROR
    assert "  hint: first line" not in following
    assert not any("uv sync failed" in line for line in following)
    assert len(session.calls) == 1
    assert [t for t in session.tools if t.startswith("git")] == ["git pull --ff-only"]


@windows_only
def test_ac5_a_silent_failure_stays_one_line(launcher: Launcher, tmp_path: Path) -> None:
    session = launcher.run("-Target", "1", "-OutDir", str(tmp_path / "out"), tool_exit=1)
    assert session.code == 0
    lines = session.out.splitlines()
    for what in ("git pull failed", "uv sync failed"):
        index = [i for i, line in enumerate(lines) if what in line]
        assert len(index) == 1
        assert not lines[index[0] + 1].startswith("  ")
