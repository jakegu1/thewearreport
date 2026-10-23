"""Additional synthetic numeric intake failure and boundary checks."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

from wearreport.tools import label_intake, spotcheck_summary


def _data(minute: int = 0) -> dict[str, Any]:
    return {
        "date": "2026-10-10",
        "started_at": f"2026-10-10T12:{minute:02d}Z",
        "light": "twilight",
        "frames": 1,
        "detector": {"model": "yolox_m", "sha256": "b" * 64, "conf": 1},
        "min_height_px": 31,
        "judge": "synthetic-model",
        "crops_shown": 1,
        "crops_rejected": 0,
        "crops": [[31, "uuu", "ynu"]],
    }


def _incoming(tmp_path: Path) -> Path:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "historical-2.json").write_text(json.dumps(_data()))
    return incoming


@pytest.mark.parametrize("confidence", [0, 0.0, 1, 1.0])
def test_confidence_endpoints_and_model_answers(tmp_path: Path, confidence: float) -> None:
    incoming = _incoming(tmp_path)
    data = _data()
    data["detector"]["conf"] = confidence
    (incoming / "historical-2.json").write_text(json.dumps(data))
    report = label_intake.intake(incoming, tmp_path / "output")
    assert report["sources"]["london"]["human"]["outer_layer"] == {"y": 0, "n": 0, "u": 1}


@pytest.mark.parametrize("missing", ["frames", "detector", "judge", "crops", "date"])
def test_missing_current_fields_refused(tmp_path: Path, missing: str) -> None:
    incoming = _incoming(tmp_path)
    data = _data()
    del data[missing]
    (incoming / "historical-2.json").write_text(json.dumps(data))
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, tmp_path / "output")


def test_parser_duplicate_keys_and_extreme_number_do_not_escape(tmp_path: Path) -> None:
    incoming = _incoming(tmp_path)
    path = incoming / "historical-2.json"
    for raw in (
        json.dumps(_data()).replace('"frames": 1', '"frames": 1, "frames": 1'),
        json.dumps(_data()).replace('"conf": 1', '"conf": 1e99999'),
        json.dumps(_data()).replace('"conf": 1', '"conf": ' + "9" * 10000),
    ):
        path.write_text(raw)
        with pytest.raises(label_intake.IntakeError):
            label_intake.intake(incoming, tmp_path / "output")


def test_read_growth_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    incoming = _incoming(tmp_path)
    path = incoming / "historical-2.json"
    real_open = Path.open
    read_sizes: list[int] = []

    class GrowingFile:
        def __enter__(self) -> GrowingFile:
            self.handle = cast(BinaryIO, real_open(path, "rb"))
            return self

        def __exit__(self, *_args: object) -> None:
            self.handle.close()

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return b" " * size

    monkeypatch.setattr(Path, "open", lambda *_a, **_kw: GrowingFile())
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, tmp_path / "output")
    assert read_sizes == [(1 << 20) + 1]
    assert not (tmp_path / "output").exists()


def test_replaced_inode_is_refused_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming = _incoming(tmp_path)
    real_fstat = os.fstat

    def replaced(fd: int) -> os.stat_result:
        values = list(real_fstat(fd))
        values[stat.ST_INO] += 1
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", replaced)
    with pytest.raises(label_intake.IntakeError, match="changed while opening"):
        label_intake.intake(incoming, tmp_path / "output")


def test_existing_fifo_is_refused_without_opening(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO inputs are a POSIX filesystem feature")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    os.mkfifo(incoming / "fifo.json")
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, tmp_path / "output")


def test_direct_input_directory_symlink_is_refused(tmp_path: Path) -> None:
    incoming = _incoming(tmp_path)
    original = (incoming / "historical-2.json").read_bytes()
    linked = tmp_path / "linked-input"
    linked.symlink_to(incoming, target_is_directory=True)
    with pytest.raises(label_intake.IntakeError, match="input directory"):
        label_intake.intake(linked, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert (incoming / "historical-2.json").read_bytes() == original


def test_input_ancestor_symlink_loop_has_bounded_safe_api_and_cli_errors(tmp_path: Path) -> None:
    marker = "synthetic-path-marker"
    loop = tmp_path / marker
    loop.symlink_to(loop.name, target_is_directory=True)
    incoming, output = loop / "incoming", tmp_path / "output"
    with pytest.raises(label_intake.IntakeError) as error:
        label_intake.intake(incoming, output)
    assert marker not in str(error.value)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wearreport.tools.label_intake",
            "--input",
            str(incoming),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert len(result.stderr) < 200
    assert marker not in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "bad\nname.json",
        "bad\x7fname.json",
        "bad\x85name.json",
        "bad\u202ename.json",
        "CON.json",
        "CON .json",
        "bad:name.json",
        "bad\\name.json",
    ],
)
def test_unsafe_names_refused(tmp_path: Path, name: str) -> None:
    if os.name == "nt":
        pytest.skip("host filesystem already refuses these synthetic names")
    incoming = _incoming(tmp_path)
    (incoming / "historical-2.json").rename(incoming / name)
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, tmp_path / "output")


def test_case_colliding_names_cannot_choose_distinct_session_winner(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("case-insensitive filesystem cannot construct both names")
    incoming = _incoming(tmp_path)
    (incoming / "HISTORICAL-2.json").write_text(json.dumps(_data(1)))
    with pytest.raises(label_intake.IntakeError, match="filename collision"):
        label_intake.intake(incoming, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("part", ["output", "attributes", "original", "manifest", "parent"])
def test_output_symlinks_refused_without_touching_target(tmp_path: Path, part: str) -> None:
    incoming, output = _incoming(tmp_path), tmp_path / "output"
    label_intake.intake(incoming, output)
    target = tmp_path / "target"
    if part == "parent":
        parent = tmp_path / "linked-parent"
        parent.symlink_to(tmp_path, target_is_directory=True)
        candidate = parent / "output"
    else:
        paths = {
            "output": output,
            "attributes": output / "attributes",
            "original": output / "attributes" / "historical-2.json",
            "manifest": output / "manifest.json",
        }
        candidate = output
        path = paths[part]
        path.rename(target)
        path.symlink_to(target, target_is_directory=target.is_dir())
    before = {
        str(p.relative_to(tmp_path)): p.lstat().st_mtime_ns
        for p in tmp_path.rglob("*")
        if not p.is_symlink()
    }
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, candidate)
    assert before == {
        str(p.relative_to(tmp_path)): p.lstat().st_mtime_ns
        for p in tmp_path.rglob("*")
        if not p.is_symlink()
    }


@pytest.mark.parametrize("failure", ["corrupt-stage", "commit", "concurrent-output"])
def test_failed_verification_or_commit_leaves_no_partial_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    incoming, output = _incoming(tmp_path), tmp_path / "output"
    original = (incoming / "historical-2.json").read_bytes()
    real_write, real_commit = label_intake._write, label_intake._commit

    def corrupt(path: Path, text: str) -> None:
        real_write(path, "{}" if path.name == "manifest.json" else text)

    def commit(staging: Path, destination: Path) -> None:
        if failure == "concurrent-output":
            destination.mkdir()
            real_commit(staging, destination)
        else:
            raise OSError("synthetic-sensitive-marker")

    if failure == "corrupt-stage":
        monkeypatch.setattr(label_intake, "_write", corrupt)
    else:
        monkeypatch.setattr(label_intake, "_commit", commit)
    with pytest.raises(label_intake.IntakeError) as error:
        label_intake.intake(incoming, output)
    assert "synthetic-sensitive-marker" not in str(error.value)
    assert (incoming / "historical-2.json").read_bytes() == original
    assert not list(tmp_path.glob(".label-intake-*"))
    if failure == "concurrent-output":
        assert output.is_dir() and not list(output.iterdir())
    else:
        assert not output.exists()


def test_every_record_uses_existing_parser_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming = _incoming(tmp_path)
    (incoming / "other.json").write_text(json.dumps(_data(1)))
    parsed: list[bytes] = []
    real_parse = spotcheck_summary.parse_labelling
    real_write = label_intake._write

    def parse(raw: bytes) -> spotcheck_summary.Labelling:
        parsed.append(raw)
        return real_parse(raw)

    def write(path: Path, text: str) -> None:
        assert len(parsed) >= 2
        real_write(path, text)

    monkeypatch.setattr(spotcheck_summary, "parse_labelling", parse)
    monkeypatch.setattr(label_intake, "_write", write)
    label_intake.intake(incoming, tmp_path / "output")
    assert len(parsed) == 4  # Inputs, followed by reopened JSON Lines observations.


def test_cli_errors_do_not_echo_argument_payloads(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        label_intake.main(["--synthetic-sensitive-marker"])
    assert error.value.code == 2
    assert "synthetic-sensitive-marker" not in capsys.readouterr().err
