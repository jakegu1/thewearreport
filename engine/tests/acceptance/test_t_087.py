"""T-087: desktop shortcuts, with real COM round trips and harmless native launches.

Native cases run only on Windows. Every installation uses an explicit temporary
destination and a copied installer with a harmless label.cmd, never the desktop or
the real labelling launcher. This acceptance contract is committed before the product.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "setup-label-shortcuts.cmd"
INSTALLER = ROOT / "scripts" / "setup-label-shortcuts.ps1"
WORKFLOW = ROOT / ".github" / "workflows" / "windows.yml"
CITIES = ("london", "calgary")
POWERSHELL = shutil.which("powershell.exe") or "powershell.exe"
CMD = shutil.which("cmd.exe") or "cmd.exe"
WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32", reason="real WScript.Shell COM and ShellExecute require native Windows"
)
LEGACY_HASHES = {
    "label.cmd": "0fa34d0c1eb33c8951a5a96cfb304fd9c7424dc8167661f0cf02df05c474d748",
    "label-london.cmd": "bb9dad5c0194aa49d10627cfb096515ef191bf536ee3a7b1a9c41a8d46a9ad24",
    "scripts/label.ps1": "a1ce92a1fa9a2e58c9e18f4773bf28cffbd4a85f5ccb893d1b28e55318c5c144",
    "engine/wearreport/tools/label_window.py": (
        "c99079c6f49ddf1680277d67e3b463a936ff20ca0700bd861373291ec6b63c33"
    ),
}


def test_ascii_entry_uses_process_policy_quotes_path_and_preserves_result() -> None:
    text = ENTRY.read_bytes().decode("ascii")
    INSTALLER.read_bytes().decode("ascii")
    assert re.search(
        r"powershell\.exe\s+-NoProfile\s+-ExecutionPolicy\s+Bypass\s+-File\s+"
        r'"%~dp0scripts\\setup-label-shortcuts\.ps1"\s+%\*',
        text,
        re.IGNORECASE,
    )
    saved = re.search(r'set "([A-Z_]+)=%ERRORLEVEL%"', text, re.IGNORECASE)
    assert saved is not None
    assert text.index(saved.group(0)) < text.lower().index("pause")
    assert re.search(r"exit /b %" + saved.group(1) + "%", text, re.IGNORECASE)
    assert "RunAs" not in text and "Set-ExecutionPolicy" not in text


def test_installer_uses_native_desktop_com_and_no_labelling_or_credentials() -> None:
    text = INSTALLER.read_text(encoding="ascii")
    assert "DesktopDirectory" in text and "GetFolderPath" in text
    assert "$PSScriptRoot" in text and "ShortcutDir" in text
    assert "WScript.Shell" in text and "CreateShortcut" in text
    assert not re.search(
        r"(?i)Set-ExecutionPolicy|Start-Process|Invoke-Expression|Invoke-WebRequest|"
        r"Invoke-RestMethod|\b(?:git|uv)\s|DEEPINFRA|API_KEY|Get-Credential|"
        r"-NoJudge|-Now|-DryRun|-Upload|label\.ps1",
        text,
    )


def test_existing_launchers_and_city_windows_are_byte_identical() -> None:
    for name, digest in LEGACY_HASHES.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def test_windows_workflow_only_adds_the_new_launcher_test_path() -> None:
    text = WORKFLOW.read_bytes().decode("ascii").replace("\r\n", "\n")
    added = "          engine/tests/acceptance/test_t_087.py\n"
    assert text.count(added) == 1
    before = text.replace(added, "")
    assert hashlib.sha256(before.encode("ascii")).hexdigest() == (
        "6a5450dec6019e909c0833f474401097febe77a2718ae6aa614440c5c0a55f64"
    )
    launcher = text.split("- name: Labelling launcher tests (T-067)", 1)[1]
    launcher = launcher.split("- name:", 1)[0]
    assert "engine/tests/acceptance/test_t_087.py" in launcher


def test_documentation_explains_install_use_retained_settings_and_removal() -> None:
    text = (ROOT / "spotchecks" / "README.md").read_text(encoding="utf-8")
    assert "setup-label-shortcuts.cmd" in text and "git pull --ff-only" in text
    for city in CITIES:
        assert f"The Wear Report - {city.title()}" in text
    for phrase in ("single pass", "AI", "time gates", "clipboard", "moving", "delete"):
        assert phrase.lower() in text.lower()
    assert "valid key" in text.lower()


def _quote(text: str | Path) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _powershell(code: str) -> subprocess.CompletedProcess[str]:
    encoded = base64.b64encode(code.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )


def _com(link: Path, *, target: Path | None = None) -> dict[str, str]:
    code = (
        "$ErrorActionPreference = 'Stop'; "
        "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; "
        "$shell = New-Object -ComObject WScript.Shell; "
        f"$link = $shell.CreateShortcut({_quote(link)}); "
    )
    if target is not None:
        code += (
            f"$link.TargetPath = {_quote(target)}; "
            "$link.Arguments = 'old arguments'; $link.WorkingDirectory = 'C:\\'; $link.Save(); "
        )
    code += (
        "@{target=$link.TargetPath; arguments=$link.Arguments; cwd=$link.WorkingDirectory} "
        "| ConvertTo-Json -Compress"
    )
    result = _powershell(code)
    assert result.returncode == 0, result.stdout + result.stderr
    properties: dict[str, str] = json.loads(result.stdout.lstrip("\ufeff"))
    return properties


def _files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


@dataclass
class Installation:
    repo: Path
    destination: Path
    marker: Path

    def link(self, city: str) -> Path:
        return self.destination / f"The Wear Report - {city.title()}.lnk"

    def run(self, destination: Path | None = None) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                POWERSHELL,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.repo / "scripts" / "setup-label-shortcuts.ps1"),
                "-ShortcutDir",
                str(self.destination if destination is None else destination),
            ],
            cwd=self.destination,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        print(result.stdout + result.stderr)
        return result


@pytest.fixture
def installation(tmp_path: Path) -> Installation:
    repo = tmp_path / "repo with spaces \u6807\u6ce8"
    destination = tmp_path / "shortcut folder \u684c\u9762 with spaces"
    (repo / "scripts").mkdir(parents=True)
    destination.mkdir()
    shutil.copyfile(INSTALLER, repo / "scripts" / INSTALLER.name)
    shutil.copyfile(ENTRY, repo / ENTRY.name)
    (repo / "label.cmd").write_text(
        "@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass "
        '-File "%~dp0record.ps1" %*\nexit /b %ERRORLEVEL%\n',
        encoding="ascii",
    )
    (repo / "record.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "@{argv=@($args | ForEach-Object { [string]$_ }); cwd=(Get-Location).Path} "
        "| ConvertTo-Json -Compress | Set-Content -LiteralPath $env:T087_MARKER -Encoding UTF8\n",
        encoding="ascii",
    )
    (destination / "unrelated.txt").write_text("keep this file", encoding="ascii")
    return Installation(repo, destination, tmp_path / "launch.json")


def _assert_properties(installation: Installation) -> None:
    for city in CITIES:
        properties = _com(installation.link(city))
        assert Path(properties["target"]) == installation.repo / "label.cmd"
        assert Path(properties["cwd"]) == installation.repo
        assert properties["arguments"] == f"-Source {city} -MaxPasses 1"


@WINDOWS_ONLY
def test_native_com_round_trip_only_writes_two_named_links(installation: Installation) -> None:
    before = _files(installation.repo)
    result = installation.run()
    assert result.returncode == 0
    assert _files(installation.repo) == before and not installation.marker.exists()
    assert set(_files(installation.destination)) == {
        "unrelated.txt",
        "The Wear Report - London.lnk",
        "The Wear Report - Calgary.lnk",
    }
    assert (installation.destination / "unrelated.txt").read_text() == "keep this file"
    _assert_properties(installation)


@WINDOWS_ONLY
def test_native_reinstall_updates_only_links_owned_by_this_repository(
    installation: Installation,
) -> None:
    assert installation.run().returncode == 0
    for city in CITIES:
        _com(installation.link(city), target=installation.repo / "label.cmd")
    assert installation.run().returncode == 0
    _assert_properties(installation)
    assert (installation.destination / "unrelated.txt").read_text() == "keep this file"


@WINDOWS_ONLY
@pytest.mark.parametrize("city", CITIES)
def test_native_open_each_real_link_passes_exact_arguments_and_working_directory(
    installation: Installation, monkeypatch: pytest.MonkeyPatch, city: str
) -> None:
    assert installation.run().returncode == 0
    monkeypatch.setenv("T087_MARKER", str(installation.marker))
    getattr(os, "startfile")(str(installation.link(city)), "open")  # noqa: B009 -- Windows-only API
    deadline = time.monotonic() + 30
    data = None
    while time.monotonic() < deadline:
        try:
            data = json.loads(installation.marker.read_text(encoding="utf-8-sig"))
            break
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    assert data is not None, "the real shortcut did not launch the harmless stub"
    assert data["argv"] == ["-Source", city, "-MaxPasses", "1"]
    assert Path(data["cwd"]) == installation.repo
    _assert_properties(installation)


@WINDOWS_ONLY
def test_native_missing_target_refuses_before_writing(installation: Installation) -> None:
    (installation.repo / "label.cmd").unlink()
    before = _files(installation.destination)
    result = installation.run()
    assert result.returncode != 0
    assert "label.cmd" in result.stdout + result.stderr
    assert _files(installation.destination) == before


@WINDOWS_ONLY
@pytest.mark.parametrize("kind", ["missing", "file"])
def test_native_invalid_destination_never_falls_back(installation: Installation, kind: str) -> None:
    target = installation.repo / "invalid shortcut destination"
    if kind == "file":
        target.write_text("keep", encoding="ascii")
    before = _files(installation.destination)
    result = installation.run(target)
    assert result.returncode != 0
    assert "directory" in (result.stdout + result.stderr).lower()
    assert _files(installation.destination) == before
    assert target.is_file() if kind == "file" else not target.exists()


@WINDOWS_ONLY
@pytest.mark.parametrize("city", CITIES)
def test_native_foreign_target_conflict_preflights_both_links(
    installation: Installation, city: str
) -> None:
    assert installation.run().returncode == 0
    _com(installation.link(city), target=installation.repo / "foreign.cmd")
    before = _files(installation.destination)
    result = installation.run()
    assert result.returncode != 0
    assert "target" in (result.stdout + result.stderr).lower()
    assert city in (result.stdout + result.stderr).lower()
    assert _files(installation.destination) == before


@WINDOWS_ONLY
def test_native_same_named_directory_refuses_before_either_link_is_written(
    installation: Installation,
) -> None:
    installation.link("calgary").mkdir()
    before = _files(installation.destination)
    result = installation.run()
    assert result.returncode != 0
    assert "calgary" in (result.stdout + result.stderr).lower()
    assert _files(installation.destination) == before
    assert installation.link("calgary").is_dir()


@WINDOWS_ONLY
def test_native_save_error_is_nonzero_and_actionable(installation: Installation) -> None:
    assert installation.run().returncode == 0
    kernel = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009 -- Windows-only API
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Allow reads for COM preflight, but deny the write handle that Save needs.
    handle = kernel.CreateFileW(str(installation.link("calgary")), 0x80000000, 1, None, 3, 0, None)
    last_error = getattr(ctypes, "get_last_error")()  # noqa: B009 -- Windows-only API
    assert handle not in (None, wintypes.HANDLE(-1).value), last_error
    try:
        result = installation.run()
    finally:
        assert kernel.CloseHandle(handle)
    assert result.returncode != 0
    output = (result.stdout + result.stderr).lower()
    assert "calgary" in output and ("save" in output or "write" in output)
    assert "close" in output or "permission" in output or "writable" in output
    assert (installation.destination / "unrelated.txt").read_text() == "keep this file"


@WINDOWS_ONLY
def test_native_batch_entry_keeps_the_installers_failure_exit_code(
    installation: Installation,
) -> None:
    (installation.repo / "scripts" / INSTALLER.name).write_text(
        "param([string]$ShortcutDir)\nWrite-Host 'T087 installer failure'\nexit 7\n",
        encoding="ascii",
    )
    command = f'"{installation.repo / ENTRY.name}" -ShortcutDir "{installation.destination}"'
    result = subprocess.run(
        f'"{CMD}" /d /s /c "{command}"',
        input="\n",
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )
    assert result.returncode == 7
    assert "T087 installer failure" in result.stdout + result.stderr
    assert set(_files(installation.destination)) == {"unrelated.txt"}
