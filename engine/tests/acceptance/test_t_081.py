"""T-081: local prequential scores and fail-closed, accepted-last source runs.

Only generated numeric frames reach the labeller. The encoder oracle uses plain numpy;
the real-model case uses a generated crop. Native launcher and process-lock cases run
on Windows CI as well as portable process-lock cases on Linux. This file is frozen
before the implementation; existing acceptance files retain their own contracts.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport._cv import cv2
from wearreport.tools import probe, spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
DAY = datetime.date(2026, 10, 10)
NOON = datetime.datetime(2026, 10, 10, 12, 0, tzinfo=datetime.UTC)
DIM = 768
GRID = [1, 10, 100, 1000]
SHA = "0dd31785a2713f1113ef2272472165c69d580473dae38d7b47568ac587795e70"
ENCODER = "siglip2-base-patch16-224-int8@" + SHA[:12]
PIN = "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
VALUES = [10, 20, 30, 40, 50, 60, 70]
HEIGHTS = [66, 40, 66, 45, 60, 50, 55]
SAFE = ["ynn", "ynu", "nyn", "nnu", "uun", "uuu", None]
INFO = spotcheck.DetectorInfo("synthetic", "0" * 64, detect.DEFAULT_CONF)
Frame = npt.NDArray[np.uint8]
Vector = npt.NDArray[np.float64]
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="native Windows launcher")


class NumericEncoder:
    def __init__(self) -> None:
        self.seen: list[Vector] = []

    def embed(self, image: Frame) -> Vector:
        value = float(image[0, 0, 0])
        result = np.zeros(DIM, dtype=np.float64)
        result[:4] = [value / 30, (value / 30) ** 2, np.sin(value), value % 3]
        self.seen.append(result.copy())
        return result


class NumericDetector:
    def detect(self, image: Frame) -> list[detect.Detection]:
        index = VALUES.index(int(image[0, 0, 0]))
        return [detect.Detection("person", 0.95, (20.0, 20.0, 35.0, 20.0 + HEIGHTS[index]))]


class Answers:
    def __init__(
        self, answers: Sequence[str | None] = SAFE, during: Callable[[], None] | None = None
    ) -> None:
        self.answers = list(answers)
        self.during = during

    def attributes(
        self, items: Sequence[spotcheck.ReviewItem], deadline: float
    ) -> dict[int, str | None]:
        if self.during is not None:
            self.during()
        return {item.number: self.answers[item.number - 1] for item in items}


def _pipeline() -> spotcheck.Pipeline:
    frames = [np.full((120, 100, 3), v, dtype=np.uint8) for v in VALUES]
    return spotcheck.Pipeline(lambda: frames, NumericDetector(), INFO)


def _invoke(
    out: Path,
    directory: Path | None,
    *,
    source: str = "london",
    answers: Sequence[str | None] = SAFE,
    during: Callable[[], None] | None = None,
) -> int:
    argv = [
        "--attributes",
        "--n",
        "10",
        "--min-persons",
        "1",
        "--seed",
        "7",
        "--view",
        "window",
        "--allow-dark",
        "--out-dir",
        str(out),
        "--source",
        source,
    ]
    if directory is not None:
        argv += ["--probe", str(directory)]
    return spotcheck.main(
        argv, pipeline=_pipeline(), reviewer=Answers(answers, during), today=DAY, clock=lambda: NOON
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[NumericEncoder]:
    for name in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    temporary = tmp_path / "temp"
    temporary.mkdir()
    monkeypatch.chdir(tmp_path)
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", None)
    encoder = NumericEncoder()
    monkeypatch.setattr(probe, "open_encoder", lambda: encoder)
    yield encoder


def _labels(out: Path) -> list[Path]:
    return sorted((out / "attributes").glob("*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name))


def _json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _last(out: Path) -> dict[str, Any]:
    return _json(_labels(out)[-1])


def _control(out: Path, source: str = "london") -> Path:
    return out / "probe-runs" / f"{source}.json"


def _state(directory: Path, source: str = "london") -> Path:
    return directory / f"{source}.json"


def _lock(directory: Path, source: str = "london") -> Path:
    canonical = directory.resolve()
    return canonical.parent / (canonical.name + ".locks") / f"{source}.json"


def _all_files(directory: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def _probe_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return [line for line in capsys.readouterr().err.splitlines() if "probe" in line.lower()]


def _oracle(previous: Vector, answers: Sequence[str], current: Vector) -> list[list[list[float]]]:
    """Independent exact-input centering, followed by the unchanged primal ridge fit."""
    from decimal import Decimal, localcontext
    from fractions import Fraction

    output: list[list[list[float]]] = [[] for _ in current]
    judgeable = np.array([answer[0] in "yn" for answer in answers], dtype=np.float64)
    for penalty in GRID:
        heads = []
        for select, target in (
            (np.ones(len(previous), dtype=np.bool_), judgeable),
            (
                judgeable.astype(np.bool_),
                np.array([answer[0] == "y" for answer in answers], dtype=float),
            ),
        ):
            train, target = previous[select], target[select]
            columns, queries = [], []
            with localcontext() as context:
                context.prec = 96
                for index, column in enumerate(train.T):
                    # Exact binary64 inputs; constants have zero mathematical variance.
                    if np.all(column == column[0]):
                        continue
                    values = [Fraction.from_float(float(value)) for value in column]
                    mean = sum(values, Fraction(0)) / len(values)
                    centered = [value - mean for value in values]
                    variance = sum((value * value for value in centered), Fraction(0)) / len(values)
                    scale = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
                    if float(scale) <= 1e-10:
                        continue
                    columns.append(
                        [
                            float((Decimal(value.numerator) / Decimal(value.denominator)) / scale)
                            for value in centered
                        ]
                    )
                    query = [
                        Fraction.from_float(float(value)) - mean for value in current[:, index]
                    ]
                    queries.append(
                        [
                            float((Decimal(value.numerator) / Decimal(value.denominator)) / scale)
                            for value in query
                        ]
                    )
            standardized = (
                np.array(columns, dtype=np.float64).T if columns else np.empty((len(train), 0))
            )
            query = (
                np.array(queries, dtype=np.float64).T if queries else np.empty((len(current), 0))
            )
            weight = np.linalg.solve(
                standardized.T @ standardized + penalty * np.eye(standardized.shape[1]),
                standardized.T @ (target - target.mean()),
            )
            heads.append(np.round(np.clip(query @ weight + target.mean(), 0, 1), 3))
        for index, (judge, coat) in enumerate(zip(*heads, strict=True)):
            output[index].append([float(judge), float(coat)])
    return output


def _assert_metadata(record: Mapping[str, Any], session: int, run_id: str) -> None:
    assert record["probe_run"] == {
        "version": 2,
        "run_id": run_id,
        "session": session,
        "state": "accepted",
        "reason": None,
    }
    block = record["probe"]
    assert set(block) == {"encoder", "lambdas", "trained_sessions", "scores"}
    assert block["encoder"] == ENCODER and block["lambdas"] == GRID
    assert block["trained_sessions"] == session - 1
    assert len(block["scores"]) == len(record["crops"])
    for row in record["crops"]:
        assert isinstance(row, list) and len(row) == 3
    for scores in block["scores"]:
        if session == 1:
            assert scores is None
        else:
            assert len(scores) == 4
            for pair in scores:
                assert len(pair) == 2
                assert all(type(v) is float and 0 <= v <= 1 and round(v, 3) == v for v in pair)


def test_ac1_unflagged_records_and_file_counts_are_identical(
    env: NumericEncoder, tmp_path: Path
) -> None:
    plain, flagged = tmp_path / "plain", tmp_path / "flagged"
    assert _invoke(plain, None) == 0
    raw = _labels(plain)[0].read_bytes()
    assert set(_all_files(plain)) == {_labels(plain)[0]}
    assert env.seen == []
    assert "probe" not in _last(plain) and "probe_run" not in _last(plain)
    assert _invoke(flagged, flagged / "probe") == 0
    record = _last(flagged)
    del record["probe"], record["probe_run"]
    assert (json.dumps(record, ensure_ascii=True) + os.linesep).encode() == raw


def test_ac1_flag_is_attribute_only(env: NumericEncoder, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as failure:
        spotcheck.main(["--n", "1", "--probe", str(tmp_path / "probe")])
    assert failure.value.code == 2
    assert env.seen == []


def test_ac2_ac3_prior_only_numpy_oracle_every_crop_and_sorted_scores(
    env: NumericEncoder, tmp_path: Path
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    previous: list[Vector] = []
    labels: list[str] = []
    for session, answers in enumerate(
        (SAFE, SAFE, ["nnu", "nyn", "ynu", "ynn", "uuu", "uun", None]), 1
    ):
        start = len(env.seen)
        assert _invoke(out, directory, answers=answers) == 0
        record = _last(out)
        run_id = _json(_control(out))["run_id"]
        assert str(uuid.UUID(run_id)) == run_id
        _assert_metadata(record, session, run_id)
        candidates = np.array(env.seen[start:])
        assert candidates.shape == (7, DIM)  # scores precede answers; x is discarded afterward
        current = candidates[[i for i, answer in enumerate(answers) if answer is not None]]
        kept = [(h, a, None) for h, a in zip(HEIGHTS, answers, strict=True) if a is not None]
        order = sorted(range(len(kept)), key=lambda k: (kept[k][0], kept[k][1], ""))
        assert record["crops"] == [list(kept[k]) for k in order]
        if previous:
            expected = _oracle(np.array(previous), labels, current)
            assert record["probe"]["scores"] == [expected[k] for k in order]
        previous.extend(current)
        labels.extend(a for a in answers if a is not None)
        stored = _json(_state(directory))
        assert stored["version"] == 2 and stored["accepted_sessions"] == session
        for cohort in "ynu":
            group = stored["cohorts"][cohort]
            selected = np.array(
                [x for x, a in zip(previous, labels, strict=True) if a[0] == cohort]
            )
            assert group["count"] == len(selected)
            mean = np.array(group["mean"])
            assert np.allclose(mean[:, 0] + mean[:, 1], selected.mean(axis=0))
            centered = selected - selected.mean(axis=0)
            assert np.allclose(group["scatter"], centered.T @ centered)


def test_ac2_source_histories_are_separate(env: NumericEncoder, tmp_path: Path) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory) == 0
    london = _state(directory).read_bytes()
    assert _invoke(out, directory, source="calgary") == 0
    calgary = _json(_labels(out)[-1])
    _assert_metadata(calgary, 1, _json(_control(out, "calgary"))["run_id"])
    assert calgary["source"] == "calgary" and _state(directory).read_bytes() == london
    assert _json(_control(out))["run_id"] != _json(_control(out, "calgary"))["run_id"]


def test_ac3_current_answers_cannot_change_current_predictions(
    env: NumericEncoder, tmp_path: Path
) -> None:
    scores = []
    for name, current in (("a", SAFE), ("b", ["nyn", "nnu", "ynn", "ynu", "uuu", "uun", None])):
        out = tmp_path / name
        directory = out / "probe"
        assert _invoke(out, directory) == 0
        assert _invoke(out, directory, answers=current) == 0
        record = _last(out)
        scores.append(
            sorted(
                (r[0], json.dumps(s))
                for r, s in zip(record["crops"], record["probe"]["scores"], strict=True)
            )
        )
    assert scores[0] == scores[1]


def test_ac4_only_cumulative_stats_labels_control_and_stable_lock_persist(
    env: NumericEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    snapshots: list[dict[Path, bytes]] = []
    original = probe.atomic_json

    def observe(path: Path, value: Mapping[str, object]) -> None:
        original(path, value)
        snapshots.append(_all_files(tmp_path))

    monkeypatch.setattr(probe, "atomic_json", observe)
    assert _invoke(out, directory) == 0
    inode, lock_bytes = _lock(directory).stat().st_ino, _lock(directory).read_bytes()
    assert _invoke(out, directory) == 0
    assert _lock(directory).stat().st_ino == inode and _lock(directory).read_bytes() == lock_bytes
    assert len(lock_bytes) <= 128
    lock = json.loads(lock_bytes)
    assert set(lock) == {"version", "kind"} and lock == {"version": 1, "kind": "probe_source_lock"}
    allowed = {_state(directory), _control(out), _lock(directory), *_labels(out)}
    assert set(_all_files(tmp_path)) == allowed
    for snapshot in snapshots:
        assert set(snapshot) <= allowed
        for path, raw in snapshot.items():
            value = json.loads(raw)
            if path == _state(directory):
                assert set(value) == {
                    "version",
                    "run_id",
                    "source",
                    "encoder",
                    "accepted_sessions",
                    "cohorts",
                }
                assert value["version"] == 2
                assert set(value["cohorts"]) == set("ynu")
                for group in value["cohorts"].values():
                    assert set(group) == {"count", "mean", "scatter"}
                    assert np.array(group["mean"]).shape == (DIM, 2)
                    assert np.array(group["scatter"]).shape == (DIM, DIM)
            elif path == _control(out):
                assert value["version"] == 2
                assert set(value) == {
                    "version",
                    "run_id",
                    "source",
                    "encoder",
                    "state",
                    "accepted_sessions",
                    "reason",
                }
            assert not any(
                word in raw.lower()
                for word in (b"features", b"checkpoint", b"base64", b"image", b"crop_id", b"delta")
            )


@pytest.mark.parametrize(
    "bad",
    [
        ["ynn", "ynn", "nyn", "nyn", "uuu", None, None],
        ["ynn", "nyn", "nyn", "nyn", "uuu", "uuu", None],
        ["ynn", "ynn", "ynn", "nyn", "uuu", "uuu", None],
        ["uuu", "uuu", None, None, None, None, None],
        ["ynn", None, None, None, None, None, None],
    ],
)
@pytest.mark.parametrize("prior", [False, True])
def test_ac4_sparse_increment_never_serializes_even_with_safe_prior_and_candidate(
    env: NumericEncoder,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bad: list[str | None],
    prior: bool,
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    if prior:
        assert _invoke(out, directory) == 0
    old = _state(directory).read_bytes() if prior else None
    capsys.readouterr()
    assert _invoke(out, directory, answers=bad) == 0
    record = _last(out)
    assert "probe" not in record and record["probe_run"]["state"] == "incomplete"
    assert record["crops"]
    assert len(_probe_lines(capsys)) == 1
    assert (_state(directory).read_bytes() if _state(directory).exists() else None) == old
    seen = len(env.seen)
    assert _invoke(out, directory) == 0
    assert len(env.seen) == seen and "probe" not in _last(out)


def test_ac4_zero_cohorts_are_safe_and_have_zero_statistics(
    env: NumericEncoder, tmp_path: Path
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory, answers=["ynn"] * 6 + [None]) == 0
    stored = _json(_state(directory))
    assert stored["cohorts"]["y"]["count"] == 6
    for name in "nu":
        group = stored["cohorts"][name]
        assert group["count"] == 0 and not np.any(group["mean"]) and not np.any(group["scatter"])
    assert _invoke(out, directory, answers=["nnu"] * 6 + [None]) == 0
    assert all(pair == [1.0, 1.0] for row in _last(out)["probe"]["scores"] for pair in row)


def test_ac4_pending_precedes_work_human_label_precedes_stats_and_accepted_is_last(
    env: NumericEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    events: list[str] = []
    atomic = probe.atomic_json
    score_features = probe.score_features
    session = 0

    def observe(path: Path, value: Mapping[str, object]) -> None:
        if path == _control(out):
            events.append(str(value["state"]))
        elif path == _state(directory):
            events.append("aggregate")
            human = _last(out)
            assert "probe" not in human and human["probe_run"]["state"] == "incomplete"
            assert human["probe_run"]["reason"] == "unfinished_session"
        elif path.parent == out / "attributes":
            events.append("accepted")
            assert _json(_control(out))["state"] == "active"
        atomic(path, value)

    def during() -> None:
        control = _json(_control(out))
        assert control["state"] == "pending" and control["accepted_sessions"] == session
        assert len(env.seen) == (session + 1) * 7
        assert events == ["pending", *(["feature"] * 7), "scores"]
        events.append("answers")

    embed = env.embed

    def feature(image: Frame) -> Vector:
        assert len(_labels(out)) == session
        assert _json(_control(out))["state"] == "pending"
        events.append("feature")
        return embed(image)

    def predictions(*args: Any, **kwargs: Any) -> Any:
        assert len(_labels(out)) == session
        assert "answers" not in events
        events.append("scores")
        return score_features(*args, **kwargs)

    monkeypatch.setattr(probe, "atomic_json", observe)
    monkeypatch.setattr(env, "embed", feature)
    monkeypatch.setattr(probe, "score_features", predictions)
    for session in range(2):
        events.clear()
        assert _invoke(out, directory, during=during) == 0
        assert _json(_control(out))["accepted_sessions"] == session + 1
        assert events == [
            "pending",
            *(["feature"] * 7),
            "scores",
            "answers",
            "aggregate",
            "active",
            "accepted",
        ]


@pytest.mark.parametrize("stage", ["arm", "human_sync", "aggregate", "terminal", "accepted"])
def test_ac4_every_commit_interruption_preserves_human_labels_and_stops_continuation(
    env: NumericEncoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage: str,
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory) == 0
    old = _state(directory).read_bytes()
    atomic, sync = probe.atomic_json, probe.sync_label

    def fail(path: Path, value: Mapping[str, object]) -> None:
        hit = (
            (stage == "arm" and path == _control(out) and value.get("state") == "pending")
            or (stage == "aggregate" and path == _state(directory))
            or (stage == "terminal" and path == _control(out) and value.get("state") == "active")
            or (stage == "accepted" and path.parent == out / "attributes")
        )
        if hit:
            raise OSError("sensitive path and feature values must never be printed")
        atomic(path, value)

    def fail_sync(path: Path) -> None:
        if stage == "human_sync":
            raise OSError("sensitive sync failure")
        sync(path)

    monkeypatch.setattr(probe, "atomic_json", fail)
    monkeypatch.setattr(probe, "sync_label", fail_sync)
    capsys.readouterr()
    assert _invoke(out, directory) == 0
    failed = _last(out)
    assert "probe" not in failed and failed["probe_run"]["state"] == "incomplete"
    assert failed["crops"] == _json(_labels(out)[0])["crops"]
    lines = _probe_lines(capsys)
    assert len(lines) == 1 and "sensitive" not in lines[0] and "rollback" not in lines[0]
    if stage in ("arm", "human_sync", "aggregate"):
        assert _state(directory).read_bytes() == old
    else:
        assert _json(_state(directory))["accepted_sessions"] == 2
    if stage == "accepted":
        assert _json(_control(out))["state"] == "active"  # human label still overrides it
    monkeypatch.setattr(probe, "atomic_json", atomic)
    monkeypatch.setattr(probe, "sync_label", sync)
    before = len(env.seen)
    assert _invoke(out, directory) == 0
    assert len(env.seen) == before and _last(out)["probe_run"]["state"] == "stopped"


def test_ac4_ordinary_label_write_failure_never_persists_feature_work(
    env: NumericEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"

    def fail(*args: object, **kwargs: object) -> Path:
        raise spotcheck.SpotcheckError("simulated ordinary label failure")

    monkeypatch.setattr(spotcheck, "write_attributes", fail)
    assert _invoke(out, directory) == 1
    assert len(env.seen) == 7 and not _state(directory).exists()
    assert _json(_control(out))["state"] == "pending"


@pytest.mark.parametrize(
    "damage",
    [
        "missing_control",
        "pending",
        "incomplete",
        "malformed_control",
        "wrong_run",
        "wrong_count",
        "wrong_encoder",
        "missing_label",
        "duplicate_label",
        "bad_label",
        "incomplete_label",
        "unknown_stop",
        "foreign_run",
        "gap",
    ],
)
def test_ac4_ac6_complete_history_is_required_no_selected_subset_or_identity_reset(
    env: NumericEncoder, tmp_path: Path, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory) == 0
    control = _json(_control(out))
    accepted = _labels(out)[0]
    record = _json(accepted)
    if damage == "missing_control":
        _control(out).unlink()
    elif damage == "malformed_control":
        _control(out).write_text("{", encoding="utf-8")
    elif damage in ("pending", "incomplete", "wrong_run", "wrong_count", "wrong_encoder"):
        key, value = {
            "pending": ("state", "pending"),
            "incomplete": ("state", "incomplete"),
            "wrong_run": ("run_id", str(uuid.uuid4())),
            "wrong_count": ("accepted_sessions", 2),
            "wrong_encoder": ("encoder", "other@000000000000"),
        }[damage]
        control[key] = value
        _control(out).write_text(json.dumps(control), encoding="utf-8")
    elif damage == "missing_label":
        accepted.unlink()
    elif damage == "duplicate_label":
        (accepted.parent / "duplicate.json").write_text(json.dumps(record), encoding="utf-8")
    elif damage == "bad_label":
        accepted.write_text("{", encoding="utf-8")
    elif damage == "gap":
        record["probe_run"]["session"] = 2
        accepted.write_text(json.dumps(record), encoding="utf-8")
    elif damage == "foreign_run":
        record["probe_run"]["run_id"] = str(uuid.uuid4())
        accepted.write_text(json.dumps(record), encoding="utf-8")
    else:
        failed = copy.deepcopy(record)
        del failed["probe"]
        failed["probe_run"].update(state="incomplete", reason="unfinished_session")
        if damage == "unknown_stop":
            failed["probe_run"].update(
                run_id=None, session=None, state="stopped", reason="unknown_identity"
            )
        (accepted.parent / "failed.json").write_text(json.dumps(failed), encoding="utf-8")
    old = _state(directory).read_bytes()
    control_bytes = _control(out).read_bytes() if _control(out).exists() else None
    before = len(env.seen)
    capsys.readouterr()
    assert _invoke(out, directory) == 0
    assert env.seen == env.seen[:before] and len(env.seen) == before
    assert _state(directory).read_bytes() == old
    assert (_control(out).read_bytes() if _control(out).exists() else None) == control_bytes
    failed = _last(out)
    assert "probe" not in failed and failed["probe_run"]["state"] == "stopped"
    assert failed["probe_run"]["session"] is None and len(_probe_lines(capsys)) == 1


def test_ac6_unknown_identity_stop_prevents_fresh_history(
    env: NumericEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    _control(out).parent.mkdir(parents=True)
    _control(out).write_text("{}", encoding="utf-8")
    assert _invoke(out, directory) == 0
    record = _last(out)
    assert record["probe_run"]["run_id"] is None and record["probe_run"]["session"] is None
    _control(out).unlink()  # removing the malformed input cannot erase the terminal label
    assert _invoke(out, directory) == 0
    assert env.seen == [] and not _state(directory).exists() and not _control(out).exists()


def test_ac6_seal_twelve_accepts_no_more_work_and_verifies_complete_history(
    env: NumericEncoder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    for _ in range(12):
        assert _invoke(out, directory) == 0
    control = _json(_control(out))
    assert control["state"] == "sealed" and control["accepted_sessions"] == 12
    sessions = sorted(_json(p)["probe_run"]["session"] for p in _labels(out))
    assert sessions == list(range(1, 13))
    old, old_control, before = (
        _state(directory).read_bytes(),
        _control(out).read_bytes(),
        len(env.seen),
    )
    for _ in range(2):
        assert _invoke(out, directory) == 0
        assert _last(out)["probe_run"] == {
            "version": 2,
            "run_id": control["run_id"],
            "session": None,
            "state": "stopped",
            "reason": "collection_complete",
        }
        assert "probe" not in _last(out)
    assert _state(directory).read_bytes() == old and _control(out).read_bytes() == old_control
    assert len(env.seen) == before
    _labels(out)[0].unlink()
    capsys.readouterr()
    assert _invoke(out, directory) == 0
    assert _last(out)["probe_run"]["reason"] != "collection_complete"
    assert len(env.seen) == before and len(_probe_lines(capsys)) == 1


@pytest.mark.parametrize("fault", ["missing", "checksum", "shape", "failed"])
def test_ac6_encoder_failures_are_one_sanitized_line_per_pass_and_keep_prior_bytes(
    env: NumericEncoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fault: str,
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory) == 0
    old = _state(directory).read_bytes()

    def fail() -> NumericEncoder:
        if fault in ("missing", "checksum"):
            raise probe.ProbeError("model_" + fault)
        if fault == "failed":
            raise RuntimeError("sensitive runtime path and feature")
        broken = NumericEncoder()
        broken.embed = lambda image: np.zeros(767, dtype=np.float64)  # type: ignore[method-assign]
        return broken

    monkeypatch.setattr(probe, "open_encoder", fail)
    for _ in range(2):
        capsys.readouterr()
        assert _invoke(out, directory) == 0
        failed = _last(out)
        assert "probe" not in failed and failed["crops"]
        lines = _probe_lines(capsys)
        assert len(lines) == 1 and "sensitive" not in lines[0]
        assert _state(directory).read_bytes() == old


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"[]",
        b"null",
        b"[" * 2000 + b"]" * 2000,
        b'{"version":NaN}',
        b'{"version":Infinity}',
        b'{"version":1e999}',
        b'{"version":true}',
        b'{"version":1,"version":1}',
        b"\xff",
        b" " * (64 * 1024 * 1024 + 1),
    ],
    ids=[
        "malformed",
        "array",
        "null",
        "deep-nesting",
        "nan",
        "infinity",
        "overflow",
        "boolean-version",
        "duplicate-key",
        "invalid-utf8",
        "over-64mib",
    ],
)
def test_ac6_hostile_state_input_is_always_typed_and_sanitized(raw: bytes) -> None:
    with pytest.raises(probe.ProbeError) as failure:
        probe.parse_state(raw)
    assert re.fullmatch(r"[a-z_]+", str(failure.value))


@pytest.mark.parametrize(
    "fault", ["shape", "type", "nan", "inf", "singleton", "zero", "count", "cross"]
)
def test_ac6_unsafe_state_is_untouched_and_disables_feature_work(
    env: NumericEncoder,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fault: str,
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory) == 0
    value = _json(_state(directory))
    group = value["cohorts"]["u"]
    if fault == "shape":
        group["mean"] = [0.0]
    elif fault == "type":
        group["mean"][0][0] = "sensitive"
    elif fault in ("nan", "inf"):
        group["mean"][0][0] = float(fault)
    elif fault == "singleton":
        group["count"] = 1
    elif fault == "zero":
        group["count"] = 0
    elif fault == "count":
        value["cohorts"]["y"]["count"] += 2
    else:
        group["scatter"][0] = [0.0]
    _state(directory).write_text(json.dumps(value), encoding="utf-8")
    old, control, before = _state(directory).read_bytes(), _control(out).read_bytes(), len(env.seen)
    capsys.readouterr()
    assert _invoke(out, directory) == 0
    assert _state(directory).read_bytes() == old and _control(out).read_bytes() == control
    assert len(env.seen) == before and "probe" not in _last(out) and len(_probe_lines(capsys)) == 1


def test_ac6_unwritable_aggregate_path_keeps_labelling(
    env: NumericEncoder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, directory = tmp_path / "out", tmp_path / "not-a-directory"
    directory.write_text("leave this untouched", encoding="utf-8")
    assert _invoke(out, directory) == 0
    assert _last(out)["crops"] and "probe" not in _last(out)
    assert directory.read_text(encoding="utf-8") == "leave this untouched"
    assert env.seen == [] and len(_probe_lines(capsys)) == 1


def test_ac6_unreadable_prior_is_untouched_and_no_features_are_used(
    env: NumericEncoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory) == 0
    old, control, before = _state(directory).read_bytes(), _control(out).read_bytes(), len(env.seen)
    original = Path.open

    def refuse(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == _state(directory):
            raise PermissionError("sensitive path")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", refuse)
        capsys.readouterr()
        assert _invoke(out, directory) == 0
        assert len(env.seen) == before and "probe" not in _last(out)
        lines = _probe_lines(capsys)
        assert len(lines) == 1 and "sensitive" not in lines[0]
    assert _state(directory).read_bytes() == old and _control(out).read_bytes() == control


def test_ac6_failed_twelfth_label_overrides_sealed_control(
    env: NumericEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    for _ in range(11):
        assert _invoke(out, directory) == 0
    atomic = probe.atomic_json

    def refuse(path: Path, value: Mapping[str, object]) -> None:
        if path.parent == out / "attributes":
            raise OSError("simulated final-label failure")
        atomic(path, value)

    monkeypatch.setattr(probe, "atomic_json", refuse)
    assert _invoke(out, directory) == 0
    assert _json(_control(out))["state"] == "sealed"
    assert _last(out)["probe_run"]["state"] == "incomplete"
    monkeypatch.setattr(probe, "atomic_json", atomic)
    before = len(env.seen)
    assert _invoke(out, directory) == 0
    assert len(env.seen) == before
    assert _last(out)["probe_run"]["reason"] != "collection_complete"


def test_ac4_both_stop_sinks_fail_before_any_feature_work(
    env: NumericEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"

    def no_control(path: Path, value: Mapping[str, object]) -> None:
        raise PermissionError("simulated unavailable control")

    def no_label(*args: object, **kwargs: object) -> Path:
        raise spotcheck.SpotcheckError("simulated ordinary label failure")

    monkeypatch.setattr(probe, "atomic_json", no_control)
    monkeypatch.setattr(spotcheck, "write_attributes", no_label)
    assert _invoke(out, directory) == 1
    assert env.seen == [] and not _control(out).exists() and not _state(directory).exists()


def test_ac5_preprocessing_exact_square_bilinear_rgb_float32() -> None:
    image = np.zeros((40, 20, 3), dtype=np.uint8)
    image[:, :, 0] = 30
    image[:, :, 1] = np.arange(40, dtype=np.uint8)[:, None]
    image[:, :, 2] = 200
    square = np.zeros((40, 40, 3), dtype=np.uint8)
    square[:, 10:30] = image
    expected = cv2.resize(square, (224, 224), interpolation=cv2.INTER_LINEAR)
    expected = (expected.astype(np.float32) / np.float32(255) - np.float32(0.5)) / np.float32(0.5)
    expected = np.ascontiguousarray(expected.transpose(2, 0, 1)[None])
    actual = probe.preprocess(image)
    assert actual.dtype == np.float32 and actual.shape == (1, 3, 224, 224)
    assert np.array_equal(actual, expected)


def test_ac5_real_pinned_int8_cpu_encoder_on_synthetic_crop() -> None:
    model = ROOT / ".models" / "siglip2-base-patch16-224-int8.onnx"
    if not model.is_file():
        if os.environ.get("WEARREPORT_REQUIRE_PROBE_MODEL"):
            pytest.fail("the required real int8 probe model is absent")
        pytest.skip("local real int8 probe model absent; CI requires it")
    assert model.stat().st_size == 94_553_333
    assert hashlib.sha256(model.read_bytes()).hexdigest() == SHA
    encoder = probe.SiglipEncoder(model)
    assert encoder.input_name == "pixel_values" and encoder.output_name == "pooler_output"
    image = np.zeros((80, 51, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(51, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(80, dtype=np.uint8)[:, None]
    a, b = encoder.embed(image), encoder.embed(image)
    assert a.shape == (DIM,) and np.isfinite(a).all() and np.array_equal(a, b)
    assert encoder.providers == ["CPUExecutionProvider"]
    print(
        f"REAL SigLIP 2 int8 CPU: sha256={SHA}; pooled={a.shape}; "
        f"deterministic=true; platform={sys.platform}"
    )


def test_ac5_checksum_and_shape_errors_are_typed(tmp_path: Path) -> None:
    with pytest.raises(probe.ProbeError, match="model_missing"):
        probe.SiglipEncoder(tmp_path / "missing.onnx")
    wrong = tmp_path / "wrong.onnx"
    wrong.write_text("not the pinned encoder", encoding="ascii")
    with pytest.raises(probe.ProbeError, match="model_checksum"):
        probe.SiglipEncoder(wrong)


@pytest.mark.parametrize("split_out", [False, True])
def test_ac4_kernel_lock_blocks_stale_prior_interleaving_in_separate_processes(
    env: NumericEncoder,
    tmp_path: Path,
    split_out: bool,
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory) == 0
    old = _state(directory).read_bytes()
    second_out = tmp_path / "second" if split_out else out
    if split_out:
        shutil.copytree(out, second_out)  # only human labels and metadata; no aggregate backup
    ready, release = tmp_path / "ready", tmp_path / "release"
    script = """
import os, runpy, sys, time
from pathlib import Path
ns = runpy.run_path(sys.argv[1])
probe, spotcheck = ns['probe'], ns['spotcheck']
for name in spotcheck.CI_VARIABLES: os.environ.pop(name, None)
encoder = ns['NumericEncoder']()
original = encoder.embed
def embed(image):
    Path(sys.argv[4]).write_text('entered', encoding='ascii')
    deadline = time.monotonic() + 40
    while not Path(sys.argv[5]).exists():
        if time.monotonic() > deadline: raise RuntimeError('test barrier timeout')
        time.sleep(.01)
    return original(image)
encoder.embed = embed
probe.open_encoder = lambda: encoder
raise SystemExit(ns['_invoke'](Path(sys.argv[2]), Path(sys.argv[3])))
"""
    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            __file__,
            str(out),
            str(directory),
            str(ready),
            str(release),
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 40
        while not ready.exists():
            assert first.poll() is None, first.communicate()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert _json(_control(out))["state"] == "pending"
        inode, lock_bytes = _lock(directory).stat().st_ino, _lock(directory).read_bytes()
        before = len(env.seen)
        # Alternate spelling canonicalizes to the very same source lock; with
        # split_out the lock cannot be keyed by the human-label destination.
        assert _invoke(second_out, directory / ".." / directory.name) == 0
        stopped = _last(second_out)
        assert stopped["probe_run"]["state"] == "stopped"
        assert stopped["probe_run"]["reason"] == "source_busy"
        assert stopped["probe_run"]["session"] is None and "probe" not in stopped
        assert len(env.seen) == before and _state(directory).read_bytes() == old
        assert (
            _lock(directory).stat().st_ino == inode and _lock(directory).read_bytes() == lock_bytes
        )
        release.write_text("go", encoding="ascii")
        stdout, stderr = first.communicate(timeout=60)
        assert first.returncode == 0, stdout + stderr
        count = _json(_state(directory))["accepted_sessions"]
        assert count <= 2  # two stale-prior feature candidates can never be committed
        if not split_out:
            assert count == 1  # the contention label is a terminal source stop
        print(
            f"KERNEL LOCK stale-prior interleaving: platform={sys.platform}; "
            f"split_out={split_out}; feature_candidates=1; accepted={count}"
        )
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=20)


def test_ac7_legacy_report_is_identical_and_valid_probe_metadata_is_ignored(
    env: NumericEncoder, tmp_path: Path
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, None) == 0
    legacy = _labels(out)[0].read_bytes()
    parsed = spotcheck_summary.parse_labelling(legacy)
    assert _invoke(out, directory) == 0
    assert spotcheck_summary.parse_labelling(_labels(out)[-1].read_bytes()) == parsed
    failure = _last(out)
    del failure["probe"]
    failure["probe_run"].update(state="incomplete", reason="unfinished_session")
    assert spotcheck_summary.parse_labelling(json.dumps(failure).encode()) == parsed
    stopped = copy.deepcopy(failure)
    stopped["probe_run"].update(
        run_id=None, session=None, state="stopped", reason="unknown_identity"
    )
    assert spotcheck_summary.parse_labelling(json.dumps(stopped).encode()) == parsed


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "probe_only",
        "bad_encoder",
        "bad_grid",
        "bad_trained",
        "length",
        "range",
        "nan",
        "bad_run",
        "bad_session",
        "bad_reason",
        "accepted_without_probe",
        "failure_with_probe",
    ],
)
def test_ac7_malformed_objects_name_the_file_and_unrelated_keys_are_rejected(
    env: NumericEncoder,
    tmp_path: Path,
    damage: str,
) -> None:
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory) == 0
    record = _last(out)
    if damage == "extra":
        record["unrelated"] = 1
    elif damage == "probe_only":
        del record["probe_run"]
    elif damage == "bad_encoder":
        record["probe"]["encoder"] = "other"
    elif damage == "bad_grid":
        record["probe"]["lambdas"] = [1, 10, 100]
    elif damage == "bad_trained":
        record["probe"]["trained_sessions"] = 1
    elif damage == "length":
        record["probe"]["scores"] = []
    elif damage in ("range", "nan"):
        record["probe_run"]["session"] = 2
        record["probe"]["trained_sessions"] = 1
        record["probe"]["scores"] = [[[float("nan") if damage == "nan" else 1.01, 0.0]] * 4] * 6
    elif damage == "bad_run":
        record["probe_run"]["run_id"] = "invalid"
    elif damage == "bad_session":
        record["probe_run"]["session"] = True
    elif damage == "bad_reason":
        record["probe_run"]["reason"] = "exception text"
    elif damage == "accepted_without_probe":
        del record["probe"]
    else:
        record["probe_run"].update(state="incomplete", reason="unfinished_session")
    path = _labels(out)[0]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(spotcheck_summary.SummaryError, match=re.escape(path.name)):
        spotcheck_summary.load_labellings(out)


def test_ac8_ac9_pins_fetchers_native_ci_and_public_privacy_guards() -> None:
    for name in ("fetch_probe_model.sh", "fetch_probe_model.ps1"):
        text = (ROOT / "scripts" / name).read_text(encoding="ascii")
        for value in (
            SHA,
            PIN,
            "huggingface.co",
            "hf-mirror.com",
            "modelscope.cn",
            "vision_model_int8.onnx",
            "siglip2-base-patch16-224-int8.onnx",
        ):
            assert value in text
        assert "timeout" in text.lower() or "max-time" in text
        assert "retry" in text.lower()
        assert "manual" in text.lower() and "94553333" in text.replace("_", "")
    assert "probe-model:" in (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "windows.yml").read_text(encoding="ascii")
    for added in (
        '      WEARREPORT_REQUIRE_PROBE_MODEL: "1"\n',
        "      - name: Fetch the SigLIP 2 probe encoder (SHA-256 verified)\n"
        "        run: powershell -NoProfile -ExecutionPolicy Bypass "
        "-File scripts/fetch_probe_model.ps1\n",
        "          engine/tests/acceptance/test_t_081.py\n",
    ):
        assert workflow.count(added) == 1
    assert "test_t_087.py\n          engine/tests/acceptance/test_t_081.py" in workflow
    for arguments in (
        ["scripts/privacy_guard.py"],
        ["tools/public_guard.py"],
        ["tools/public_guard.py", "--history"],
    ):
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@WINDOWS_ONLY
@pytest.mark.parametrize("source", ["london", "calgary"])
@pytest.mark.parametrize("failed_fetch", [False, True])
def test_ac8_native_launcher_both_cities_fetches_once_and_all_passes_continue(
    tmp_path: Path,
    source: str,
    failed_fetch: bool,
) -> None:
    repo, out = tmp_path / "repo with spaces", tmp_path / "out"
    (repo / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts" / "label.ps1", repo / "scripts" / "label.ps1")
    (repo / "scripts" / "fetch_probe_model.ps1").write_text(
        "[IO.File]::AppendAllText($env:T081_FETCH_LOG, 'fetch' + [Environment]::NewLine)\n"
        "Write-Output 'simulated fetch failure'\nexit 1\n",
        encoding="ascii",
    )
    if not failed_fetch:
        (repo / ".models").mkdir()
        # The launcher checks presence only; the injected numeric encoder does no
        # model substitution. The separate real-model test pins CPU/int8 inference.
        (repo / ".models" / "siglip2-base-patch16-224-int8.onnx").write_text(
            "present", encoding="ascii"
        )
    shim = tmp_path / "shim.py"
    shim.write_text(
        "import json, os, runpy, sys\nfrom pathlib import Path\n"
        f"ns = runpy.run_path({str(Path(__file__).resolve())!r})\n"
        "spotcheck, probe = ns['spotcheck'], ns['probe']\n"
        "for name in spotcheck.CI_VARIABLES: os.environ.pop(name, None)\n"
        "argv = sys.argv[1:]\n"
        "with open(os.environ['T081_CALLS'], 'a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps(argv)+'\\n')\n"
        "if '--judgements' in argv:\n"
        "    argv = [a for a in argv]\n"
        "    i = argv.index('--judgements'); del argv[i:i+2]\n"
        "    argv += ['--view', 'window', '--allow-dark']\n"
        "encoder = ns['NumericEncoder']()\n"
        "def fail(): raise probe.ProbeError('model_missing')\n"
        "probe.open_encoder = fail if os.environ['T081_FAILED'] == '1' else lambda: encoder\n"
        "base = ns['_pipeline']()\n"
        "pipeline = spotcheck.Pipeline(base.frames, base.detector, base.info, "
        "camera_frames=lambda:[('synthetic-'+str(i), frame) "
        "for i, frame in enumerate(base.frames())])\n"
        "raise SystemExit(spotcheck.main(argv, pipeline=pipeline, "
        "reviewer=ns['Answers'](['ynn'] * 7), today=ns['DAY'], clock=lambda:ns['NOON']))\n",
        encoding="utf-8",
    )
    fakes = tmp_path / "fakebin"
    fakes.mkdir()
    for name in ("git", "uv"):
        (fakes / f"{name}.cmd").write_text("@echo off\nexit /b 0\n", encoding="ascii")
    fetch_log, calls = tmp_path / "fetch.log", tmp_path / "calls.jsonl"
    process_env = dict(os.environ)
    process_env.update(
        PATH=str(fakes) + os.pathsep + process_env.get("PATH", ""),
        T081_FETCH_LOG=str(fetch_log),
        T081_CALLS=str(calls),
        T081_FAILED="1" if failed_fetch else "0",
    )
    result = subprocess.run(
        [
            shutil.which("powershell.exe") or "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo / "scripts" / "label.ps1"),
            "-Source",
            source,
            "-NoJudge",
            "-DryRun",
            "-Now",
            "2026-06-21T12:00Z" if source == "london" else "2026-06-21T19:00Z",
            "-Python",
            sys.executable,
            "-Spotcheck",
            str(shim),
            "-TempDir",
            str(tmp_path / "temporary"),
            "-OutDir",
            str(out),
            "-ClipboardFile",
            str(tmp_path / "clipboard.txt"),
            "-Target",
            "1000",
            "-MaxPasses",
            "2",
            "-Interval",
            "0",
        ],
        env=process_env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    combined = result.stdout + result.stderr
    print(combined)
    assert result.returncode == 0, combined
    argv_rows = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    assert len(argv_rows) == 2 and len(_labels(out)) == 2
    for argv in argv_rows:
        assert argv[argv.index("--source") + 1] == source
        assert Path(argv[argv.index("--probe") + 1]) == out / "probe"
        print(f"NATIVE launcher city={source}; --probe={out / 'probe'}; argv={argv}")
    if failed_fetch:
        assert fetch_log.read_text(encoding="utf-8").splitlines() == ["fetch"]
        assert combined.count("Probe encoder: fetch failed; labelling continues.") == 1
        lines = [line for line in combined.splitlines() if line.startswith("spotcheck: probe")]
        assert len(lines) == 2 and all("probe" not in _json(p) for p in _labels(out))
    else:
        assert not fetch_log.exists() and all("probe" in _json(p) for p in _labels(out))
