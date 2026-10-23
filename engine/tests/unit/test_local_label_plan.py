"""Additional offline input, timestamp, and native boundary regression checks."""

from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys
from importlib import resources
from zoneinfo import ZoneInfo

import pytest

from tests.acceptance import test_t_094 as acceptance
from wearreport.tools import local_label_plan as planner

pytest_plugins = ["tests.acceptance.test_t_094"]


def test_historical_offsets_preserve_the_actual_instant() -> None:
    data = planner.build_plan(
        ZoneInfo("Africa/Monrovia"), dt.date(1900, 1, 10), 1, 0, 1440, planner.WEEKDAYS
    )
    for name in ("london_opportunities", "calgary_noon_coverage"):
        rows = data[name]
        assert isinstance(rows, list) and rows
        for row in rows:
            source = dt.datetime.fromisoformat(row["source_local"])
            display = dt.datetime.fromisoformat(row["display_local"])
            utc = dt.datetime.fromisoformat(row["utc"])
            assert source == display == utc
            assert display.strftime("%H:%M:%S") == row["display_time"]


def test_unknown_large_option_is_bounded(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        planner.main(["--zone", "UTC", "--" + "x" * 10000])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert len(output.err) < 2000 and "unknown options" in output.err


def test_preflight_rejects_an_unsupported_python_without_reading_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 11))
    with pytest.raises(planner.PlanError, match=r"Python 3\.12"):
        planner.preflight(ZoneInfo("UTC"))


def test_source_and_display_use_packaged_rules_even_with_a_different_host_zone() -> None:
    # A host-derived object with this key need not have the installed package's rules.
    with resources.files("tzdata.zoneinfo").joinpath("UTC").open("rb") as stream:
        different_host_zone = ZoneInfo.from_file(stream, "America/Edmonton")
    data = planner.build_plan(
        different_host_zone, dt.date(2026, 11, 1), 1, 0, 1440, planner.WEEKDAYS
    )
    london = data["london_opportunities"]
    calgary = data["calgary_noon_coverage"]
    assert isinstance(london, list) and isinstance(calgary, list)
    assert [row["display_time"] for row in london] == ["04:00", "06:00", "08:00"]
    assert calgary[0]["utc"] == "2026-11-01T18:00+00:00"
    assert calgary[0]["display_time"] == "12:00"
    assert data["timezone_source"] == "tzdata package" and data["timezone_version"] == "2026.5"


@pytest.mark.parametrize("key", ["localtime", "posixrules", "../UTC", "UTC/../../localtime"])
def test_host_only_and_path_keys_are_rejected(key: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        planner.main(["--zone", key])
    assert exc.value.code == 2
    assert "explicit IANA" in capsys.readouterr().err


def test_unsupported_cli_python_fails_before_loading_any_zone(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 10))
    assert planner.main(["--zone", "UTC", "--preflight"]) == 2
    output = capsys.readouterr()
    assert "Python 3.12" in output.err and "Traceback" not in output.err


@acceptance.NATIVE
@pytest.mark.parametrize("batch", [False, True])
def test_native_output_path_metacharacter_round_trips_without_an_extra_command(
    native: acceptance.NativeEntry, batch: bool
) -> None:
    target = r"C:\labels&ver"
    if batch:
        # Quote the user's argument at the outer cmd boundary. This test probes the
        # product's later PowerShell-to-label.cmd boundary, not an unquoted shell input.
        cmd = shutil.which("cmd.exe") or "cmd.exe"
        command = (
            f'"{cmd}" /d /s /c ""{native.repo / acceptance.ENTRY.name}" '
            f'-StartLondon -OutDir "{target}""'
        )
        result = subprocess.run(
            command,
            cwd=native.cwd,
            env=dict(os.environ, T094_LOG=str(native.log), T094_CHILD_EXIT="37"),
            input="\n",
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
        print(result.stdout + result.stderr)
    else:
        result = native.run("-StartLondon", "-OutDir", target)
    assert result.returncode == 37
    assert "Microsoft Windows" not in result.stdout
    assert native.calls() == [
        {
            "kind": "label",
            "cwd": str(native.repo),
            "argv": ["-Source", "london", "-MaxPasses", "1", "-NoJudge", "-OutDir", target],
        }
    ]
