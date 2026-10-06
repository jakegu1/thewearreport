"""Acceptance tests for T-078: attribute sessions can record per-camera yield counts.

`--record-camera-yield` makes an attribute session add one top-level object to its label
file, `camera_yield`: `{camera id: [shown, rejected, judgeable]}` per camera that gave at
least one shown crop. Counts only; the crop rows stay exactly as before.

The sessions here sweep spotcheck's own dry-run fake camera servers (London's, and
Calgary's) on 127.0.0.1, which serve the licensed fixture photos and synthetic noise.
Most use a stub detector, so that the boxes, and so the counts, are known; one uses the
real model. The reviewer answers by crop number and never sees an image. Nothing reaches
the network.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport.tools import spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
REAL_DIR = ROOT / "spotchecks"
README = REAL_DIR / "README.md"
LABEL_PS1 = ROOT / "scripts" / "label.ps1"
FLAG = "--record-camera-yield"
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
KEY_ENV = "DEEPINFRA_API_KEY"
DAY = datetime.date(2026, 10, 5)
LONDON_NOON = datetime.datetime(2026, 10, 5, 11, 22, 33, tzinfo=datetime.UTC)
CALGARY_NOON = datetime.datetime(2026, 10, 5, 19, 0, tzinfo=datetime.UTC)  # 13:00 MDT
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
# The fields an attribute file has today (T-045), London's without `source`.
FIELDS = {
    "date",
    "started_at",
    "light",
    "frames",
    "detector",
    "min_height_px",
    "judge",
    "crops_shown",
    "crops_rejected",
    "crops",
}
# The dry-run London server's cameras, in camera order (all of them decode: the fixture
# photos, and every third one synthetic noise).
LONDON_IDS = [f"Fake_{i:05d}" for i in range(1, spotcheck.DRY_RUN_CAMERAS + 1)]
# The dry-run Calgary cameras that serve an 840x630 still (every third one serves a
# 320x176 placeholder, which is not shown).
CALGARY_SHOWN = {f"calgary-{i}" for i in range(spotcheck.DRY_RUN_AUSTIN_CAMERAS) if i % 3 != 2}

Frame = npt.NDArray[np.uint8]


# Helpers ------------------------------------------------------------------------------


def answer(number: int) -> str | None:
    """The reviewer's answer for crop `number`: rejected, judgeable on the outer layer
    (y or n first), or not (u first)."""
    return (None, "ynn", "nyu", "uyy", "yuu")[number % 5]


class Answers:
    """An attribute reviewer that answers by crop number (never looks at an image)."""

    def __init__(self) -> None:
        self.numbers: list[int] = []

    def attributes(
        self, items: Sequence[spotcheck.ReviewItem], deadline: float
    ) -> dict[int, str | None]:
        self.numbers = [item.number for item in items]
        return {item.number: answer(item.number) for item in items}


def boxes_for(call: int) -> int:
    """How many person boxes the stub finds on the `call`-th frame it sees (from 0)."""
    return call % 3 + 1


class Stub:
    """Finds boxes_for(k) person boxes, all at least 31 px tall, on the k-th frame."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: Frame) -> list[detect.Detection]:
        count = boxes_for(self.calls)
        self.calls += 1
        return [
            detect.Detection("person", 0.9, (10.0 + 40 * j, 10.0, 40.0 + 40 * j, 50.0 + 7 * j))
            for j in range(count)
        ]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Outside CI, offline but for 127.0.0.1, in a fresh working directory, HOME and
    temporary directory; yields the temporary directory."""
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    for name in (*PROXY_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    work, home, tmp = tmp_path / "work", tmp_path / "home", tmp_path / "tmp"
    for d in (work, home, tmp):
        d.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")):
            raise AssertionError(f"non-local connection attempted: {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    yield tmp


@pytest.fixture
def stub_london(monkeypatch: pytest.MonkeyPatch) -> Callable[[], Stub]:
    """The dry-run London pipeline opens a fresh Stub instead of the model; returns a
    function giving the last one opened."""
    opened: list[Stub] = []

    def open_detector(model: str) -> tuple[Stub, spotcheck.DetectorInfo]:
        opened.append(Stub())
        return opened[-1], INFO

    monkeypatch.setattr(spotcheck, "_open_detector", open_detector)
    return lambda: opened[-1]


def _session(
    out: Path,
    *extra: str,
    reviewer: Answers | None = None,
    clock: datetime.datetime = LONDON_NOON,
    **kwargs: Any,
) -> int:
    argv = [
        "--attributes",
        "--dry-run",
        "--view",
        "window",
        "--n",
        "500",
        "--min-persons",
        "1",
        "--seed",
        "7",
        "--reviewer",
        "tester",
        "--out-dir",
        str(out),
        *extra,
    ]
    return spotcheck.main(
        argv, reviewer=reviewer or Answers(), today=DAY, clock=lambda: clock, **kwargs
    )


def _only_file(out: Path) -> Path:
    files = sorted((out / spotcheck.ATTRIBUTES_DIR).glob("*.json"))
    assert len(files) == 1, files
    return files[0]


def _record(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_only_file(out).read_text(encoding="utf-8"))
    return data


def _judgeable_rows(record: Mapping[str, Any]) -> int:
    return sum(1 for row in record["crops"] if row[1][0] in "yn")


def _check_yield(record: Mapping[str, Any]) -> dict[str, list[int]]:
    """AC2's invariants; returns the camera_yield object."""
    cy = record["camera_yield"]
    assert isinstance(cy, dict) and cy
    for camera, counts in cy.items():
        assert isinstance(camera, str) and camera
        assert isinstance(counts, list) and len(counts) == 3
        assert all(type(c) is int for c in counts)
        shown, rejected, judgeable = counts
        assert shown >= 1, camera  # only cameras with a shown crop
        assert 0 <= rejected <= shown, camera
        assert 0 <= judgeable <= shown - rejected, camera
    assert sum(c[0] for c in cy.values()) == record["crops_shown"]
    assert sum(c[1] for c in cy.values()) == record["crops_rejected"]
    assert sum(c[2] for c in cy.values()) == _judgeable_rows(record)
    return cy


def _expected_london_yield() -> dict[str, list[int]]:
    """What the stub and the reviewer make of the dry-run London sweep: frames in camera
    order, crops numbered 1, 2, ... across them."""
    expected: dict[str, list[int]] = {}
    number = 0
    for call, camera in enumerate(LONDON_IDS):
        for _ in range(boxes_for(call)):
            number += 1
            counts = expected.setdefault(camera, [0, 0, 0])
            given = answer(number)
            counts[0] += 1
            if given is None:
                counts[1] += 1
            elif given[0] in "yn":
                counts[2] += 1
    return expected


def _check_rows(record: Mapping[str, Any]) -> None:
    """AC3: every crop row is [int, str, str|null], today's three fields only."""
    for row in record["crops"]:
        assert isinstance(row, list) and len(row) == 3, row
        height, reviewer, model = row
        assert type(height) is int
        assert isinstance(reviewer, str) and re.fullmatch(r"[ynu]{3}", reviewer)
        assert model is None or isinstance(model, str)


# AC1: without the flag, the file is what the tool writes today --------------------------


def test_ac1_without_the_flag_no_camera_yield(env: Path, stub_london: Any, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert _session(out) == 0
    record = _record(out)
    assert set(record) == FIELDS
    assert "camera_yield" not in _only_file(out).read_text(encoding="utf-8")


def test_ac1_without_the_flag_with_the_real_model(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert _session(out) == 0
    assert set(_record(out)) == FIELDS


def test_ac1_the_flag_changes_nothing_but_adds_camera_yield(
    env: Path, stub_london: Any, tmp_path: Path
) -> None:
    plain, flagged = tmp_path / "plain", tmp_path / "flagged"
    assert _session(plain) == 0
    assert _session(flagged, FLAG) == 0
    without, with_yield = _record(plain), _record(flagged)
    assert set(with_yield) == FIELDS | {"camera_yield"}
    del with_yield["camera_yield"]
    assert with_yield == without  # same crops, same order, same counts


def test_ac1_the_flag_is_refused_outside_an_attribute_session(
    env: Path, stub_london: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--dry-run", "--n", "5", "--view", "files", FLAG, "--out-dir", str(tmp_path / "o")]
    assert spotcheck.main(argv, today=DAY, clock=lambda: LONDON_NOON) == 1
    assert FLAG in capsys.readouterr().err
    assert not (tmp_path / "o").exists() or not any((tmp_path / "o").rglob("*.json"))


# AC2: with the flag, per-camera counts that add up -------------------------------------


def test_ac2_london_dry_run_counts_per_camera(env: Path, stub_london: Any, tmp_path: Path) -> None:
    out = tmp_path / "out"
    reviewer = Answers()
    assert _session(out, FLAG, reviewer=reviewer) == 0
    record = _record(out)
    cy = _check_yield(record)
    assert cy == _expected_london_yield()
    assert record["crops_shown"] == len(reviewer.numbers) == sum(map(boxes_for, range(12)))
    # every count kind occurs, so the sums are not trivially zero
    assert all(sum(c[k] for c in cy.values()) > 0 for k in range(3))
    assert any(c[2] < c[0] - c[1] for c in cy.values())  # some kept crops not judgeable


def test_ac2_london_dry_run_with_the_real_model(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert _session(out, FLAG) == 0
    record = _record(out)
    assert record["crops_shown"] > 0
    cy = _check_yield(record)
    assert set(cy) <= set(LONDON_IDS)


def test_ac2_calgary_dry_run_counts_per_camera(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    stub = Stub()
    code = _session(
        out,
        FLAG,
        "--source",
        "calgary",
        clock=CALGARY_NOON,
        open_detector=lambda model: (stub, INFO),
    )
    assert code == 0
    record = _record(out)
    assert record["source"] == "calgary"
    assert list(record)[-1] == "source"  # still last
    cy = _check_yield(record)
    assert {camera.rsplit("/", 1)[-1] for camera in cy} == CALGARY_SHOWN
    assert len(cy) == len(CALGARY_SHOWN)
    assert record["crops_shown"] == sum(map(boxes_for, range(len(CALGARY_SHOWN))))


def test_ac2_only_cameras_with_a_shown_crop(env: Path, tmp_path: Path, monkeypatch: Any) -> None:
    """A stub that finds people on the first frame only: one camera in camera_yield."""

    class FirstOnly(Stub):
        def detect(self, frame: Frame) -> list[detect.Detection]:
            found = super().detect(frame)
            return found if self.calls == 1 else []

    monkeypatch.setattr(spotcheck, "_open_detector", lambda model: (FirstOnly(), INFO))
    out = tmp_path / "out"
    assert _session(out, FLAG) == 0
    record = _record(out)
    assert _check_yield(record) == {LONDON_IDS[0]: [1, 0, 1]}


# AC3: no camera id on any crop row, nor anywhere else ----------------------------------


def test_ac3_rows_hold_todays_three_fields_and_ids_only_in_camera_yield(
    env: Path, stub_london: Any, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert _session(out, FLAG) == 0
    text = _only_file(out).read_text(encoding="utf-8")
    record = json.loads(text)
    _check_rows(record)
    cameras = list(record["camera_yield"])
    rest = {k: v for k, v in record.items() if k != "camera_yield"}
    rest_text = json.dumps(rest)
    yield_values = json.dumps(list(record["camera_yield"].values()))
    for camera in cameras:
        assert camera not in rest_text
        assert camera not in yield_values
        assert text.count(camera) == 1  # its key, once
    assert "Fake_" not in rest_text and "/cam" not in rest_text and "127.0.0.1" not in rest_text
    # nor anywhere else: no other file is left behind
    assert [p for p in out.rglob("*") if p.is_file()] == [_only_file(out)]
    leftovers = [p for p in env.rglob("*") if p.is_file()]
    assert leftovers == [], leftovers


def test_ac3_calgary_rows_and_ids(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    stub = Stub()
    code = _session(
        out,
        FLAG,
        "--source",
        "calgary",
        clock=CALGARY_NOON,
        open_detector=lambda model: (stub, INFO),
    )
    assert code == 0
    text = _only_file(out).read_text(encoding="utf-8")
    record = json.loads(text)
    _check_rows(record)
    rest_text = json.dumps({k: v for k, v in record.items() if k != "camera_yield"})
    for camera in record["camera_yield"]:
        assert camera not in rest_text
        assert text.count(json.dumps(camera)) == 1
    assert "calgary-" not in rest_text and "127.0.0.1" not in rest_text


def test_ac3_stdout_and_stderr_name_no_camera(
    env: Path, stub_london: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _session(tmp_path / "out", FLAG) == 0
    captured = capsys.readouterr()
    for camera in LONDON_IDS:
        assert camera not in captured.out and camera not in captured.err


# AC4: the summary accepts files with and without camera_yield --------------------------


SUMMARY = ("-m", "wearreport.tools.spotcheck_summary", "--attributes", "--dir", "spotchecks")


def _summary(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *SUMMARY],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def _with_yield(record: dict[str, Any]) -> dict[str, Any]:
    """`record` with a camera_yield that adds up, split over two made-up cameras."""
    shown, rejected = record["crops_shown"], record["crops_rejected"]
    judgeable = _judgeable_rows(record)
    first = [shown - rejected, 0, judgeable] if shown > rejected else None
    cy: dict[str, list[int]] = {}
    if first is not None:
        cy["A_00001"] = first
    if rejected:
        cy["B_00002"] = [rejected, rejected, 0]
    return {**record, "camera_yield": cy}


def _copy_with_yield(tmp_path: Path, every: int) -> Path:
    """A copy of spotchecks/attributes in tmp_path/spotchecks, every `every`-th file
    given a camera_yield; returns tmp_path."""
    target = tmp_path / "spotchecks" / "attributes"
    target.mkdir(parents=True)
    files = sorted((REAL_DIR / "attributes").glob("*.json"))
    assert files
    for k, path in enumerate(files):
        if k % every:
            shutil.copyfile(path, target / path.name)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(_with_yield(record), ensure_ascii=True) + "\n"
        (target / path.name).write_text(text, encoding="utf-8")
    return tmp_path


def test_ac4_the_summary_prints_the_same_report_with_camera_yield(tmp_path: Path) -> None:
    today = _summary(ROOT)
    assert today.returncode == 0, today.stderr
    assert "attribute file(s) in spotchecks" in today.stdout
    for every, name in ((1, "all"), (2, "half")):
        cwd = _copy_with_yield(tmp_path / name, every)
        result = _summary(cwd)
        assert result.returncode == 0, result.stderr
        assert result.stdout == today.stdout


def test_ac4_the_summary_reads_a_flagged_session(
    env: Path, stub_london: Any, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert _session(out, FLAG) == 0
    labelling = spotcheck_summary.parse_labelling(_only_file(out).read_bytes())
    record = _record(out)
    assert labelling.shown == record["crops_shown"]
    assert labelling.rejected == record["crops_rejected"]
    assert len(labelling.crops) == len(record["crops"])


def _flagged_record() -> dict[str, Any]:
    return {
        "date": "2026-10-05",
        "started_at": "2026-10-05T11:22Z",
        "light": "day",
        "frames": 2,
        "detector": {"model": "stub", "sha256": "0" * 64, "conf": 0.3},
        "min_height_px": 31,
        "judge": None,
        "crops_shown": 3,
        "crops_rejected": 1,
        "crops": [[40, "ynn", None], [50, "unn", None]],
        "camera_yield": {"Fake_00001": [2, 1, 1], "Fake_00002": [1, 0, 0]},
    }


def test_ac4_a_consistent_camera_yield_parses() -> None:
    raw = json.dumps(_flagged_record()).encode()
    assert spotcheck_summary.parse_labelling(raw).shown == 3


@pytest.mark.parametrize(
    "camera_yield",
    [
        [],
        None,
        "x",
        {"Fake_00001": [2, 1]},
        {"Fake_00001": [2, 1, 1, 0]},
        {"Fake_00001": "2,1,1"},
        {"Fake_00001": [2, 1, 1], "Fake_00002": [1, 0, True]},
        {"Fake_00001": [2.0, 1, 1], "Fake_00002": [1, 0, 0]},
        {"Fake_00001": [2, -1, 1], "Fake_00002": [1, 2, 0]},
        {"Fake_00001": [2, 3, 0], "Fake_00002": [1, -2, 0]},  # rejected > shown
        {"Fake_00001": [2, 1, 2], "Fake_00002": [1, 0, -1]},  # judgeable > kept
        {"Fake_00001": [0, 0, 0], "Fake_00002": [3, 1, 1]},  # a camera with nothing shown
        {"Fake_00001": [2, 1, 1]},  # shown does not add up
        {"Fake_00001": [2, 0, 1], "Fake_00002": [1, 1, 0]},  # judgeable does not add up
        {"Fake_00001": [2, 1, 1], "Fake_00002": [1, 0, 0], "": [0, 0, 0]},
        {"Fake_00001": [10**400, 1, 1], "Fake_00002": [1, 0, 0]},
        {"Fake_00001": [[[[[2]]]], 1, 1], "Fake_00002": [1, 0, 0]},
    ],
)
def test_ac4_a_malformed_camera_yield_is_refused(camera_yield: object) -> None:
    raw = json.dumps({**_flagged_record(), "camera_yield": camera_yield}).encode()
    with pytest.raises((ValueError, TypeError, KeyError, RecursionError, OverflowError)):
        spotcheck_summary.parse_labelling(raw)


def test_ac4_a_malformed_camera_yield_names_the_file(tmp_path: Path) -> None:
    folder = tmp_path / "spotchecks" / "attributes"
    folder.mkdir(parents=True)
    bad = {**_flagged_record(), "camera_yield": {"Fake_00001": [9, 0, 0]}}
    (folder / "2026-10-05.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    result = _summary(tmp_path)
    assert result.returncode == 1
    assert "2026-10-05.json" in result.stderr
    assert "Fake_00001" not in result.stderr  # the reason names no camera


# AC5: the launcher passes the flag for London and Calgary ------------------------------


def test_ac5_label_ps1_passes_the_flag_for_every_city() -> None:
    text = LABEL_PS1.read_text(encoding="utf-8")
    block = re.search(r"\$PassArguments = @\((.*?)\n\)", text, re.DOTALL)
    assert block is not None
    assert f"'{FLAG}'" in block.group(1)
    # every pass, for either city, starts from $PassArguments
    body = re.search(r"function Get-PassCommand \{(.*?)\n\}", text, re.DOTALL)
    assert body is not None
    lines = [line.strip() for line in body.group(1).splitlines()]
    assert "$command += $PassArguments" in lines
    assert text.count("$PassArguments") == 2  # defined once, added once, unconditionally
    cities = re.search(r"\$CityNames = @\{(.*?)\}", text)
    assert cities is not None and "london" in cities.group(1) and "calgary" in cities.group(1)


def test_ac5_the_guide_documents_the_field_and_the_flag() -> None:
    text = README.read_text(encoding="utf-8")
    assert re.search(rf"^\|\s*`{FLAG}`\s*\|", text, re.MULTILINE)
    start = text.index("### Attribute file")
    end = text.index("\n### ", start + 1)
    section = text[start:end]
    assert re.search(r"^\|\s*`camera_yield`\s*\|", section, re.MULTILINE)
    assert FLAG in section
    assert re.search(r"counts only", section, re.IGNORECASE)
    assert re.search(r"no camera id[^.]*crop", section, re.IGNORECASE)


# AC6: the public guard -----------------------------------------------------------------


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
