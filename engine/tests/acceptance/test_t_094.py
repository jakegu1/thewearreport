"""T-094: offline prospective planning and a human-only native Windows entry.

Committed before implementation; do not change this acceptance contract. Native
tests execute copied entries with harmless child stubs, never the real launcher.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = "wearreport.tools.local_label_plan"
PLANNER = ROOT / "engine/wearreport/tools/local_label_plan.py"
ENTRY = ROOT / "label-local.cmd"
SCRIPT = ROOT / "scripts/label-local.ps1"
WORKFLOW = ROOT / ".github/workflows/local-label.yml"
NATIVE = pytest.mark.skipif(
    sys.platform != "win32", reason="requires actual cmd.exe and Windows PowerShell 5.1"
)
LEGACY = {
    "label.cmd": "0fa34d0c1eb33c8951a5a96cfb304fd9c7424dc8167661f0cf02df05c474d748",
    "label-london.cmd": "bb9dad5c0194aa49d10627cfb096515ef191bf536ee3a7b1a9c41a8d46a9ad24",
    "scripts/label.ps1": "a1ce92a1fa9a2e58c9e18f4773bf28cffbd4a85f5ccb893d1b28e55318c5c144",
    "engine/wearreport/tools/label_window.py": (
        "c99079c6f49ddf1680277d67e3b463a936ff20ca0700bd861373291ec6b63c33"
    ),
    ".github/workflows/windows.yml": (
        "51d3c279bf6c72f43f3f2f5eccdd00ffe6bbac9fd965fba028d9e6fa17c7e27f"
    ),
}


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", MODULE, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def plan(*args: str) -> dict[str, object]:
    result = cli(*args)
    assert result.returncode == 0, result.stdout + result.stderr
    data: dict[str, object] = json.loads(result.stdout)
    assert data["kind"] == "prospective_opportunities"
    return data


def rows(data: dict[str, object], name: str) -> list[dict[str, object]]:
    value = data[name]
    assert isinstance(value, list)
    assert all(isinstance(row, dict) for row in value)
    return value


@pytest.mark.parametrize(
    ("day", "london_utc", "calgary_utc"),
    [
        ("2026-07-15", "09:00+00:00", "18:00+00:00"),
        ("2026-12-15", "10:00+00:00", "18:00+00:00"),
        ("2026-10-24", "09:00+00:00", "18:00+00:00"),
        ("2026-10-25", "10:00+00:00", "18:00+00:00"),
        ("2026-10-31", "10:00+00:00", "18:00+00:00"),
        ("2026-11-01", "10:00+00:00", "18:00+00:00"),
        ("2026-03-07", "10:00+00:00", "19:00+00:00"),
        ("2026-03-08", "10:00+00:00", "18:00+00:00"),
    ],
)
def test_dst_source_and_display_offsets(day: str, london_utc: str, calgary_utc: str) -> None:
    data = plan("--zone", "UTC", "--start-date", day, "--days", "1")
    london = rows(data, "london_opportunities")
    calgary = rows(data, "calgary_noon_coverage")
    assert len(london) == 3 and len(calgary) == 1
    assert london[0]["utc"] == f"{day}T{london_utc}"
    assert calgary[0]["utc"] == f"{day}T{calgary_utc}"
    assert str(london[0]["source_local"]).startswith(f"{day}T10:00")
    assert str(calgary[0]["source_local"]).startswith(f"{day}T12:00")
    for row in london + calgary:
        assert row["display_local"] == row["utc"]
        assert row["display_date"] == day
        assert row["display_time"] == str(row["display_local"])[11:16]


def test_window_filters_london_and_reports_both_calgary_coverage_states() -> None:
    data = plan(
        "--zone",
        "UTC",
        "--start-date",
        "2026-11-01",
        "--days",
        "2",
        "--online-start",
        "12:00",
        "--online-end",
        "18:00",
        "--weekdays",
        "sun",
    )
    london = rows(data, "london_opportunities")
    assert [row["display_time"] for row in london] == ["12:00", "14:00"]
    coverage = rows(data, "calgary_noon_coverage")
    assert [row["covered"] for row in coverage] == [False, False]
    assert coverage[0]["reason"] == "outside_online_window"
    assert coverage[1]["reason"] == "unavailable_weekday"
    covered = plan("--zone", "UTC", "--start-date", "2026-11-01", "--days", "1")
    assert rows(covered, "calgary_noon_coverage")[0]["covered"] is True


def test_midnight_endpoint_display_weekday_and_real_date_shift() -> None:
    data = plan(
        "--zone",
        "Asia/Tokyo",
        "--start-date",
        "2026-11-02",
        "--days",
        "1",
        "--online-start",
        "00:00",
        "--online-end",
        "24:00",
        "--weekdays",
        "mon",
    )
    london = rows(data, "london_opportunities")
    assert [row["display_time"] for row in london] == ["19:00", "21:00", "23:00"]
    noon = rows(data, "calgary_noon_coverage")[0]
    assert noon["source_local"] == "2026-11-01T12:00-06:00"
    assert noon["display_date"] == "2026-11-02" and noon["display_time"] == "03:00"
    assert noon["covered"] is True
    excluded = plan(
        "--zone",
        "Asia/Tokyo",
        "--start-date",
        "2026-11-02",
        "--days",
        "1",
        "--online-end",
        "23:00",
        "--weekdays",
        "sun",
    )
    assert rows(excluded, "london_opportunities") == []
    assert rows(excluded, "calgary_noon_coverage")[0]["covered"] is False
    endpoint = plan(
        "--zone",
        "UTC",
        "--start-date",
        "2026-12-15",
        "--days",
        "1",
        "--online-start",
        "10:00",
        "--online-end",
        "14:00",
    )
    assert [r["display_time"] for r in rows(endpoint, "london_opportunities")] == ["10:00", "12:00"]


def test_offline_dates_deterministic_bounded_and_today_uses_selected_zone() -> None:
    args = ("--zone", "Pacific/Kiritimati", "--start-date", "2020-01-01", "--days", "14")
    first, second = cli(*args), cli(*args)
    assert first.returncode == 0 and first.stdout == second.stdout
    assert len(rows(json.loads(first.stdout), "london_opportunities")) == 42
    from zoneinfo import ZoneInfo

    before = dt.datetime.now(ZoneInfo("Pacific/Kiritimati")).date()
    today = plan("--zone", "Pacific/Kiritimati", "--days", "1")["start_date"]
    after = dt.datetime.now(ZoneInfo("Pacific/Kiritimati")).date()
    assert today in {before.isoformat(), after.isoformat()}


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--zone", "not/a-zone"),
        ("--zone", "/etc/passwd"),
        ("--zone", "UTC", "--days", "0"),
        ("--zone", "UTC", "--days", "15"),
        ("--zone", "UTC", "--days", "9" * 5000),
        ("--zone", "UTC", "--start-date", "2026-02-30"),
        ("--zone", "UTC", "--start-date", "9999-12-31"),
        ("--zone", "UTC", "--online-start", "24:00"),
        ("--zone", "UTC", "--online-end", "24:01"),
        ("--zone", "UTC", "--online-start", "15:00", "--online-end", "10:00"),
        ("--zone", "UTC", "--weekdays", "holiday"),
        ("--zone", "UTC", "--weekdays", ""),
        ("--zone", "x" * 5000),
    ],
)
def test_invalid_input_is_actionable_bounded_without_traceback(args: tuple[str, ...]) -> None:
    result = cli(*args)
    assert result.returncode == 2
    assert "error:" in result.stderr.lower() and "Traceback" not in result.stderr
    assert len(result.stdout + result.stderr) < 2000


def test_offline_module_has_no_provider_model_image_or_process_imports() -> None:
    tree = ast.parse(PLANNER.read_text(encoding="utf-8"))
    permitted = {
        "__future__",
        "argparse",
        "collections",
        "datetime",
        "importlib",
        "json",
        "pathlib",
        "re",
        "sys",
        "tomllib",
        "zoneinfo",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split(".")[0] in permitted for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and (node.module or "").split(".")[0] in permitted


def test_ascii_entries_legacy_bytes_and_read_only_native_workflow() -> None:
    batch = ENTRY.read_bytes().decode("ascii")
    script = SCRIPT.read_bytes().decode("ascii")
    assert re.search(r'(?i)-File\s+"%~dp0scripts\\label-local\.ps1"\s+%\*', batch)
    assert "%ERRORLEVEL%" in batch and "exit /b" in batch.lower()
    assert "$PSScriptRoot" in script and "-NoJudge" in script
    assert not re.search(
        r"(?i)Set-ExecutionPolicy|Register-ScheduledTask|schtasks|Invoke-WebRequest|"
        r"Invoke-RestMethod|Start-Process|Get-Credential|DEEPINFRA|API_KEY|\bgit\s|\buv\s",
        script,
    )
    git = shutil.which("git")
    assert git is not None
    for name, digest in LEGACY.items():
        canonical = subprocess.run(
            [git, "show", f"HEAD:{name}"],
            cwd=ROOT,
            capture_output=True,
            timeout=30,
            check=True,
        ).stdout
        assert hashlib.sha256(canonical).hexdigest() == digest, name
        assert (ROOT / name).read_bytes() in (canonical, canonical.replace(b"\n", b"\r\n"))
    workflow = WORKFLOW.read_text(encoding="ascii")
    assert "runs-on: windows-latest" in workflow and "contents: read" in workflow
    assert "engine/tests/acceptance/test_t_094.py" in workflow
    assert "uv sync --locked --no-install-package llama-cpp-python" in workflow
    assert not re.search(r"secrets\.|pull_request_target|fetch_model|schedule:", workflow)
    actions = re.findall(r"uses:\s*([^\s]+)", workflow)
    assert len(actions) == 2 and all(re.search(r"@[0-9a-f]{40}$", x) for x in actions)


@dataclass
class NativeEntry:
    repo: Path
    cwd: Path
    log: Path
    python: Path

    def run(self, *args: str, batch: bool = False) -> subprocess.CompletedProcess[str]:
        if batch:
            cmd = shutil.which("cmd.exe") or "cmd.exe"
            command = subprocess.list2cmdline([str(self.repo / ENTRY.name), *args])
            argv: str | list[str] = f'"{cmd}" /d /s /c "{command}"'
        else:
            argv = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.repo / "scripts" / SCRIPT.name),
                *args,
            ]
        env = dict(os.environ, T094_LOG=str(self.log), T094_CHILD_EXIT="37")
        result = subprocess.run(
            argv,
            cwd=self.cwd,
            env=env,
            input="\n",
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
        print(result.stdout + result.stderr)
        return result

    def calls(self) -> list[dict[str, object]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8-sig").splitlines()]


@pytest.fixture
def native(tmp_path: Path) -> NativeEntry:
    repo = tmp_path / "repo with spaces \u6807\u6ce8"
    cwd = tmp_path / "caller with spaces \u684c\u9762"
    cwd.mkdir()
    (repo / "scripts").mkdir(parents=True)
    (repo / ".venv/Scripts").mkdir(parents=True)
    (repo / "engine/wearreport/tools").mkdir(parents=True)
    (repo / ".models").mkdir()
    shutil.copyfile(ENTRY, repo / ENTRY.name)
    shutil.copyfile(SCRIPT, repo / "scripts" / SCRIPT.name)
    shutil.copyfile(PLANNER, repo / "engine/wearreport/tools" / PLANNER.name)
    for name in (
        "pyproject.toml",
        "uv.lock",
        ".venv/pyvenv.cfg",
        "scripts/label.ps1",
        ".models/yolox_m.onnx",
    ):
        (repo / name).write_text("test-only placeholder", encoding="ascii")
    stub = (
        "param([string]$Kind)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$row = @{kind=$Kind; argv=@($args | ForEach-Object {[string]$_}); "
        "cwd=(Get-Location).Path} | ConvertTo-Json -Compress\n"
        "$row | Add-Content -LiteralPath $env:T094_LOG -Encoding UTF8\n"
        "Write-Host ('T094 child output ' + $Kind)\n"
        "if ($Kind -eq 'label') { exit ([int]$env:T094_CHILD_EXIT) }; exit 0\n"
    )
    (repo / "record.ps1").write_text(stub, encoding="ascii")
    for name, kind in (("label.cmd", "label"), ("python-stub.cmd", "python")):
        (repo / name).write_text(
            "@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass "
            f'-File "%~dp0record.ps1" {kind} %*\nexit /b %ERRORLEVEL%\n',
            encoding="ascii",
        )
    return NativeEntry(repo, cwd, tmp_path / "calls.jsonl", repo / "python-stub.cmd")


@NATIVE
@pytest.mark.parametrize("batch", [False, True])
def test_native_default_help_never_invokes_child(native: NativeEntry, batch: bool) -> None:
    result = native.run(batch=batch)
    assert result.returncode == 0
    assert "offline" in result.stdout.lower() and "StartLondon" in result.stdout
    assert native.calls() == []


@NATIVE
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("out_dir", [False, True])
def test_native_human_start_exact_arguments_cwd_output_and_child_exit(
    native: NativeEntry,
    batch: bool,
    out_dir: bool,
) -> None:
    args = ["-StartLondon"]
    target = native.cwd / "output labels \u6807\u6ce8"
    if out_dir:
        args += ["-OutDir", str(target)]
    result = native.run(*args, batch=batch)
    assert result.returncode == 37
    assert "T094 child output label" in result.stdout
    calls = native.calls()
    assert len(calls) == 1 and calls[0]["kind"] == "label"
    expected = ["-Source", "london", "-MaxPasses", "1", "-NoJudge"]
    if out_dir:
        expected += ["-OutDir", str(target)]
    assert calls[0]["argv"] == expected and Path(str(calls[0]["cwd"])) == native.repo
    assert not target.exists()


@NATIVE
@pytest.mark.parametrize("extra", [["-Now", "2026-01-01"], ["-DryRun"], ["-Zone", "UTC"]])
def test_native_real_mode_refuses_clock_bypass_and_offline_parameters(
    native: NativeEntry,
    extra: list[str],
) -> None:
    assert native.run("-StartLondon", *extra).returncode != 0
    assert native.calls() == []


@NATIVE
def test_native_plan_uses_explicit_python_and_forwards_only_offline_args(
    native: NativeEntry,
) -> None:
    result = native.run(
        "-Zone",
        "UTC",
        "-StartDate",
        "2026-10-25",
        "-Days",
        "2",
        "-OnlineStart",
        "09:00",
        "-OnlineEnd",
        "24:00",
        "-Weekdays",
        "sat,sun",
        "-Python",
        str(native.python),
        batch=True,
    )
    assert result.returncode == 0
    assert native.calls() == [
        {
            "kind": "python",
            "cwd": str(native.repo),
            "argv": [
                "-m",
                MODULE,
                "--zone",
                "UTC",
                "--days",
                "2",
                "--online-start",
                "09:00",
                "--online-end",
                "24:00",
                "--weekdays",
                "sat,sun",
                "--start-date",
                "2026-10-25",
            ],
        }
    ]


@NATIVE
def test_native_actual_offline_plan_from_unicode_path(native: NativeEntry) -> None:
    result = native.run(
        "-Zone",
        "UTC",
        "-StartDate",
        "2026-10-25",
        "-Days",
        "1",
        "-Python",
        sys.executable,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["kind"] == "prospective_opportunities"
    assert native.calls() == []


@NATIVE
def test_native_preflight_checks_local_files_without_install_or_capture(
    native: NativeEntry,
) -> None:
    result = native.run("-Preflight", "-Zone", "UTC", "-Python", str(native.python))
    assert result.returncode == 0
    assert "uv.lock" in result.stdout and str(native.repo) in result.stdout
    assert len(native.calls()) == 1 and native.calls()[0]["kind"] == "python"
    argv = native.calls()[0]["argv"]
    assert isinstance(argv, list) and "--preflight" in argv


@NATIVE
@pytest.mark.parametrize(
    "missing",
    ["label.cmd", "uv.lock", "scripts/label.ps1", "engine/wearreport/tools/local_label_plan.py"],
)
def test_native_preflight_missing_file_is_actionable_before_python(
    native: NativeEntry,
    missing: str,
) -> None:
    (native.repo / missing).unlink()
    result = native.run("-Preflight", "-Zone", "UTC", "-Python", str(native.python))
    assert result.returncode != 0 and Path(missing).name in result.stdout + result.stderr
    assert native.calls() == []


@NATIVE
def test_native_no_implicit_python_install_and_zone_required(native: NativeEntry) -> None:
    result = native.run("-Zone", "UTC")
    assert result.returncode != 0 and "Python" in result.stdout + result.stderr
    assert native.run("-Days", "1", "-Python", str(native.python)).returncode != 0
    assert native.calls() == []


def test_documentation_describes_human_start_prospective_limits_and_removal() -> None:
    guide = (ROOT / "docs/local-label.md").read_text(encoding="utf-8")
    for phrase in (
        "git pull --ff-only",
        "-Preflight",
        "-StartLondon",
        "--zone",
        "prospective",
        "Calgary",
        "human",
        "single pass",
        "numeric",
        "T-093",
        "delete",
        "outputs",
        "daily",
        "AI",
    ):
        assert phrase.lower() in guide.lower()
