"""Acceptance tests for T-039 (a sweep that starts too soon after the last one is skipped).
The task contract: do not edit.

AC1 (minimum spacing) and AC5 (README) are tested here. AC2 (alerting) and AC3 (the
workflow file) are evidenced in the pull request. The clock is `aggregate._utc_now`, the
one the sweep already reads its start time from.
"""

from __future__ import annotations

import json
import logging
import re
import socket
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from wearreport import aggregate, cli, publish, registry, weather
from wearreport.testing.fake_cameras import FakeCameraServer

ROOT = Path(__file__).resolve().parents[3]
T0 = datetime(2026, 7, 15, 12, 27, 5, tzinfo=UTC)
UTC_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
MARKER = "t039-private-marker"


class _Proceeded(Exception):
    """Raised by the stand-in detector loader: the sweep went past the spacing check."""


def _record(started: datetime) -> Any:
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


def _published(data_dir: Path, started: datetime) -> Path:
    """A data directory whose last published sweep started at `started`."""
    publish.publish(data_dir, _record(started), now=started + timedelta(minutes=4))
    return data_dir


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _at(monkeypatch: pytest.MonkeyPatch, now: datetime) -> None:
    monkeypatch.setattr(aggregate, "_utc_now", lambda: now)


@pytest.fixture
def proceeds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for the detector loader: record the call and stop the sweep there."""
    calls: list[str] = []

    def load(name: str) -> Any:
        calls.append(name)
        raise _Proceeded

    monkeypatch.setattr(cli, "load_detector", load)
    return calls


@pytest.fixture
def untouchable(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCameraServer]:
    """A fake camera server whose every request fails the test, reachable only through a
    registry that fails the test too; any other connection, detector load or weather read
    fails the test as well."""
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] == "127.0.0.1"):
            raise AssertionError(f"non-local connection attempted: {address!r}")
        real_connect(self, address)

    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError("a skipped sweep must not get this far")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(cli, "load_detector", refuse)
    monkeypatch.setattr(registry, "list_cameras", refuse)
    monkeypatch.setattr(weather, "current_conditions", refuse)
    monkeypatch.setattr(aggregate, "run_sweep", refuse)
    with FakeCameraServer() as server:
        cams = server.cameras(3)
        yield server
        assert all(server.requests(c.id) == 0 for c in cams)


# AC1: minimum spacing -------------------------------------------------------------------


def test_ac1_the_spacing_is_one_named_constant_of_12_minutes() -> None:
    assert timedelta(minutes=12) == publish.MIN_SWEEP_SPACING


def test_ac1_a_start_11_59_after_the_last_is_skipped_and_nothing_is_touched(
    untouchable: FakeCameraServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = _published(tmp_path, T0)
    before = _snapshot(data_dir)
    now = T0 + timedelta(minutes=11, seconds=59)
    _at(monkeypatch, now)
    assert cli.main(["sweep", "--data-dir", str(data_dir)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1, lines
    assert "skipped" in lines[0] and "too soon" in lines[0]
    assert UTC_TIME.findall(lines[0]) == [aggregate.format_utc(now), aggregate.format_utc(T0)]
    assert _snapshot(data_dir) == before


@pytest.mark.parametrize(
    "gap", [timedelta(minutes=12), timedelta(minutes=12, seconds=1), timedelta(hours=3)]
)
def test_ac1_a_start_12_minutes_or_more_after_the_last_runs(
    proceeds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gap: timedelta
) -> None:
    data_dir = _published(tmp_path, T0)
    _at(monkeypatch, T0 + gap)
    with pytest.raises(_Proceeded):
        cli.main(["sweep", "--data-dir", str(data_dir)])
    assert proceeds == [cli.DEFAULT_MODEL]


def test_ac1_with_no_previous_record_the_sweep_runs(
    proceeds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _at(monkeypatch, T0)
    with pytest.raises(_Proceeded):
        cli.main(["sweep", "--data-dir", str(tmp_path)])
    assert proceeds == [cli.DEFAULT_MODEL]


def test_ac1_a_previous_start_in_the_future_does_not_block(
    proceeds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _published(tmp_path, T0 + timedelta(minutes=5))
    _at(monkeypatch, T0)
    with pytest.raises(_Proceeded):
        cli.main(["sweep", "--data-dir", str(data_dir)])
    assert proceeds == [cli.DEFAULT_MODEL]


def test_ac1_a_missing_status_json_lets_the_sweep_run(
    proceeds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _published(tmp_path, T0)
    (data_dir / publish.STATUS_FILE).unlink()
    _at(monkeypatch, T0 + timedelta(minutes=1))
    with pytest.raises(_Proceeded):
        cli.main(["sweep", "--data-dir", str(data_dir)])
    assert proceeds == [cli.DEFAULT_MODEL]


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\xff\xfe" + MARKER.encode(),
        b"{" * 100_000,
        json.dumps({"last_sweep_at": MARKER}).encode(),
        json.dumps({"last_sweep_at": "2026-02-30T12:00:00Z", "x": MARKER}).encode(),
        json.dumps([MARKER]).encode(),
        b'{"last_sweep_at": "' + b"x" * (2 * 1024 * 1024) + b'"}',
    ],
    ids=["empty", "not-utf8", "deep", "bad-time", "bad-date", "not-object", "huge"],
)
def test_ac1_a_corrupt_status_json_lets_the_sweep_run_and_logs_no_data(
    proceeds: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    content: bytes,
) -> None:
    data_dir = _published(tmp_path, T0)
    (data_dir / publish.STATUS_FILE).write_bytes(content)
    _at(monkeypatch, T0 + timedelta(minutes=1))
    with caplog.at_level(logging.INFO), pytest.raises(_Proceeded):
        cli.main(["sweep", "--data-dir", str(data_dir)])
    assert proceeds == [cli.DEFAULT_MODEL]
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "the unusable status.json is logged"
    for r in caplog.records:
        text = r.getMessage() + json.dumps(vars(r), default=str)
        assert MARKER not in text and "xxxxxxxx" not in text


def test_ac1_dry_run_is_unaffected(
    proceeds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _published(tmp_path, T0)
    monkeypatch.chdir(tmp_path)
    _at(monkeypatch, T0 + timedelta(minutes=1))
    with pytest.raises(_Proceeded):
        cli.main(["sweep", "--dry-run"])
    assert proceeds == [cli.DEFAULT_MODEL]


# AC5: README -----------------------------------------------------------------------------


def test_ac5_readme_operations_documents_the_skip() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("## Operations", 1)[1].split("\n## ", 1)[0]
    assert "12 minutes" in section
    assert re.search(r"skip", section, re.IGNORECASE)
    assert re.search(r"trigger", section, re.IGNORECASE)
    assert re.search(r"twice|double|once", section, re.IGNORECASE)
