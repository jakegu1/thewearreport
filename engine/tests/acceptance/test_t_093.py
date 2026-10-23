"""T-093: offline, immutable intake of synthetic numeric attribute observations."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wearreport.tools import label_intake
from wearreport.tools.spotcheck_summary import parse_labelling


def observation(source: str = "london", minute: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "date": "2026-10-10",
        "started_at": f"2026-10-10T12:{minute:02d}Z",
        "light": "day",
        "frames": 2,
        "detector": {"model": "yolox_m", "sha256": "a" * 64, "conf": 0.5},
        "min_height_px": 31,
        "judge": None,
        "crops_shown": 3,
        "crops_rejected": 1,
        "crops": [[40, "ynu", None], [50, "nyn", None]],
        "camera_yield": {"synthetic-camera": [3, 1, 2]},
    }
    if source != "london":
        result["source"] = source
    return result


def put(folder: Path, name: str = "2026-10-10T1200Z.json", **kwargs: Any) -> bytes:
    folder.mkdir(exist_ok=True)
    raw = json.dumps(observation(**kwargs), indent=2).encode() + b"\r\n"
    (folder / name).write_bytes(raw)
    return raw


def tree(folder: Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_cli_preserves_bytes_and_reports_separate_sources(tmp_path: Path) -> None:
    incoming, output = tmp_path / "incoming", tmp_path / "result"
    original = put(incoming)
    put(incoming, "2026-10-10-calgary-2.json", source="calgary")
    put(incoming, "2026-10-10-austin.json", source="austin")
    (incoming / "unrelated.txt").write_text("untouched")
    (incoming / "nested").mkdir()
    (incoming / "nested" / "bad.json").write_text("invalid")
    before = tree(incoming)
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
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert set(report["sources"]) == {"london", "calgary", "austin"}
    for source in report["sources"].values():
        assert source["sessions"] == 1
        assert source["crops"] == {"shown": 3, "rejected": 1, "labelled": 2}
        assert source["human"]["outer_layer"] == {"y": 1, "n": 1, "u": 0}
        assert source["human"]["umbrella"] == {"y": 0, "n": 1, "u": 1}
        assert source["camera_yield"]["synthetic-camera"] == {
            "shown": 3,
            "rejected": 1,
            "outer_judgeable": 2,
            "sessions": 1,
        }
        assert source["missing_camera_yield_sessions"] == 0
    assert (output / "attributes" / "2026-10-10T1200Z.json").read_bytes() == original
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["report"] == report
    assert len(manifest["sessions"]) == 3
    for entry in manifest["sessions"]:
        raw = (output / "attributes" / entry["name"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["source"] in report["sources"]
        assert entry["started_at"] == "2026-10-10T12:00Z"
    lines = (output / "sessions.jsonl").read_bytes().splitlines()
    assert len(lines) == 3
    for line in lines:
        parse_labelling(line)
    assert tree(incoming) == before


def test_duplicate_is_semantic_and_keeps_first_original(tmp_path: Path) -> None:
    incoming, output = tmp_path / "incoming", tmp_path / "result"
    raw = put(incoming, "a-historical.json")
    (incoming / "z.json").write_text(json.dumps(observation(), sort_keys=True))
    report = label_intake.intake(incoming, output)
    assert report["duplicates"] == 1
    assert report["sources"]["london"]["sessions"] == 1
    assert sorted(p.name for p in (output / "attributes").iterdir()) == ["a-historical.json"]
    assert (output / "attributes" / "a-historical.json").read_bytes() == raw


@pytest.mark.parametrize("field", ["frames", "detector", "camera_yield", "crops"])
def test_conflicting_identity_refuses_entire_batch(tmp_path: Path, field: str) -> None:
    incoming, output = tmp_path / "incoming", tmp_path / "result"
    put(incoming, "a.json")
    changed = observation()
    replacements = {
        "frames": 3,
        "detector": {"model": "other", "sha256": "a" * 64, "conf": 0.5},
        "camera_yield": {"another-camera": [3, 1, 2]},
        "crops": [[41, "ynu", None], [50, "nyn", None]],
    }
    changed[field] = replacements[field]
    (incoming / "b.json").write_text(json.dumps(changed))
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, output)
    assert not output.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["incoming"]


@pytest.mark.parametrize(
    "detector",
    [
        {},
        {"model": "x", "sha256": "a" * 64},
        {"model": "bad/name", "sha256": "a" * 64, "conf": 0.5},
        {"model": "x", "sha256": "A" * 64, "conf": 0.5},
        {"model": "x", "sha256": "a" * 63, "conf": 0.5},
        {"model": "x", "sha256": "g" * 64, "conf": 0.5},
        *[
            {"model": "x", "sha256": "a" * 64, "conf": v}
            for v in [-0.1, 1.1, True, None, "0.5", float("inf"), float("nan")]
        ],
        {"model": "x", "sha256": "a" * 64, "conf": 0.5, "extra": 1},
    ],
)
def test_detector_validation(tmp_path: Path, detector: object) -> None:
    incoming = tmp_path / "incoming"
    put(incoming)
    data = observation()
    data["detector"] = detector
    (incoming / "2026-10-10T1200Z.json").write_text(json.dumps(data))
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not-json-synthetic-sensitive-marker",
        b"\xff",
        b"[]",
        b"null",
        b"1",
        b'{"synthetic-sensitive-marker":1,"synthetic-sensitive-marker":2}',
        b"[" * 2000 + b"0" + b"]" * 2000,
        b'{"detector":NaN}',
    ],
)
def test_hostile_input_has_safe_typed_errors(tmp_path: Path, raw: bytes) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "bad.json").write_bytes(raw)
    with pytest.raises(label_intake.IntakeError) as error:
        label_intake.intake(incoming, tmp_path / "result")
    assert "synthetic-sensitive-marker" not in str(error.value)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("crops_shown", True),
        ("crops", {}),
        ("light", []),
        ("frames", 10**100),
        ("source", "london"),
        ("judge", "bad/name"),
        ("min_height_px", 30),
        ("started_at", "2026-99-99T12:00Z"),
        ("date", "2026-99-99"),
        ("crops", [[40, "ynu", "ynn"], [50, "nyn", None]]),
        ("camera_yield", {"synthetic-sensitive-marker": [4, 1, 2]}),
    ],
)
def test_existing_parser_contract_is_enforced(tmp_path: Path, field: str, value: object) -> None:
    incoming = tmp_path / "incoming"
    put(incoming)
    data = observation()
    data[field] = value
    (incoming / "2026-10-10T1200Z.json").write_text(json.dumps(data))
    with pytest.raises(label_intake.IntakeError) as error:
        label_intake.intake(incoming, tmp_path / "result")
    assert "synthetic-sensitive-marker" not in str(error.value)


def test_optional_camera_yield_and_human_only_are_counted(tmp_path: Path) -> None:
    incoming, output = tmp_path / "incoming", tmp_path / "result"
    put(incoming, "a.json")
    data = observation(minute=1)
    data.pop("camera_yield")
    (incoming / "b.json").write_text(json.dumps(data))
    report = label_intake.intake(incoming, output)["sources"]["london"]
    assert report["sessions"] == 2
    assert report["missing_camera_yield_sessions"] == 1
    assert report["camera_yield"]["synthetic-camera"]["sessions"] == 1
    assert report["camera_yield"]["synthetic-camera"]["shown"] == 3
    assert report["human"]["outer_layer"] == {"y": 2, "n": 2, "u": 0}


@pytest.mark.parametrize(
    "mode",
    ["missing", "empty", "file", "symlink", "overlap", "ancestor", "git-directory", "git-file"],
)
def test_path_safety(tmp_path: Path, mode: str) -> None:
    incoming, output = tmp_path / "incoming", tmp_path / "result"
    if mode == "missing":
        pass
    elif mode == "file":
        incoming.write_text("not a directory")
    else:
        incoming.mkdir()
        if mode != "empty":
            put(incoming)
        if mode == "symlink":
            (incoming / "linked.json").symlink_to(incoming / "2026-10-10T1200Z.json")
        elif mode == "overlap":
            output = incoming / "result"
        elif mode == "ancestor":
            output = tmp_path
        elif mode.startswith("git-"):
            repository = tmp_path / "repository"
            repository.mkdir()
            marker = repository / ".git"
            if mode == "git-directory":
                marker.mkdir()
            else:
                marker.write_text("gitdir: synthetic")
            output = repository / "nested" / "result"
    before = tree(tmp_path)
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, output)
    assert tree(tmp_path) == before


@pytest.mark.parametrize("bound", ["files", "file-bytes", "total-bytes"])
def test_resource_limits(tmp_path: Path, bound: str) -> None:
    incoming, output = tmp_path / "incoming", tmp_path / "result"
    incoming.mkdir()
    if bound == "files":
        for index in range(257):
            (incoming / f"{index}.json").write_text("{}")
    elif bound == "file-bytes":
        (incoming / "large.json").write_bytes(b" " * ((1 << 20) + 1))
    else:
        base = json.dumps(observation()).encode()
        padded = base + b" " * ((1 << 20) - len(base))
        for index in range(33):
            (incoming / f"{index}.json").write_bytes(padded)
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, output)
    assert not output.exists()


def test_idempotence_does_not_touch_output_and_conflicting_output_is_preserved(
    tmp_path: Path,
) -> None:
    incoming, output = tmp_path / "incoming", tmp_path / "result"
    put(incoming)
    first = label_intake.intake(incoming, output)
    before = tree(output)
    stamps = {str(p): p.stat().st_mtime_ns for p in output.rglob("*")}
    assert label_intake.intake(incoming, output) == first
    assert tree(output) == before
    assert stamps == {str(p): p.stat().st_mtime_ns for p in output.rglob("*")}
    data = copy.deepcopy(observation(minute=1))
    (incoming / "2026-10-10T1200Z.json").write_text(json.dumps(data))
    with pytest.raises(label_intake.IntakeError):
        label_intake.intake(incoming, output)
    assert tree(output) == before


def test_atomic_write_failure_cleans_staging_and_preserves_unrelated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming, output = tmp_path / "incoming", tmp_path / "result"
    put(incoming)
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "keep.txt").write_text("keep")
    before = tree(tmp_path)

    def fail_fsync(_fd: int) -> None:
        raise OSError("synthetic-sensitive-marker")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(label_intake.IntakeError) as error:
        label_intake.intake(incoming, output)
    assert "synthetic-sensitive-marker" not in str(error.value)
    assert not output.exists()
    assert tree(tmp_path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["incoming", "unrelated"]
