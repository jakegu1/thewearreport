"""Ridge regressions using generated numeric vectors and an independent primal fit."""

from __future__ import annotations

import datetime
import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport.tools import probe, spotcheck

DIM = 768
GRID = (1, 10, 100, 1000)
Array = npt.NDArray[np.float64]
Frame = npt.NDArray[np.uint8]


def _oracle(previous: Array, outer: Sequence[str], current: Array) -> list[list[list[float]]]:
    """Fit centered, standardized rows directly, without production moments/helpers."""
    result: list[list[list[float]]] = [[] for _ in current]
    judgeable = np.array([answer in "yn" for answer in outer], dtype=np.float64)
    for penalty in GRID:
        predictions = []
        for select, target in (
            (np.ones(len(previous), dtype=np.bool_), judgeable),
            (
                judgeable.astype(np.bool_),
                np.array([answer == "y" for answer in outer], dtype=float),
            ),
        ):
            train, target = previous[select], target[select]
            mean, scale = train.mean(axis=0), train.std(axis=0)
            active = scale > 1e-10
            standardized = (train[:, active] - mean[active]) / scale[active]
            query = (current[:, active] - mean[active]) / scale[active]
            coefficients = np.linalg.solve(
                standardized.T @ standardized + penalty * np.eye(standardized.shape[1]),
                standardized.T @ (target - target.mean()),
            )
            predictions.append(np.round(np.clip(query @ coefficients + target.mean(), 0, 1), 3))
        for row, judge, coat in zip(result, *predictions, strict=True):
            row.append([float(judge), float(coat)])
    return result


def _outer(counts: tuple[int, int, int]) -> list[str]:
    return [name for name, count in zip("ynu", counts, strict=True) for _ in range(count)]


def _state(previous: Array, outer: Sequence[str]) -> probe.State:
    cohorts = {
        name: probe.Cohort.from_rows(previous[np.array([answer == name for answer in outer])])
        for name in "ynu"
    }
    state = probe.State(str(uuid.uuid4()), "london", probe.ENCODER, 1, cohorts)
    state.validate()
    return state


@pytest.mark.parametrize(
    ("constant", "counts"),
    [
        pytest.param(0.31061848998069763, (25, 65, 83), id="judge-173"),
        pytest.param(3.8566718101501465, (122, 81, 62), id="coat-265"),
        pytest.param(1.5706465244293213, (59, 168, 232), id="coat-459"),
        pytest.param(-3.590432643890381, (133, 122, 134), id="negative-389"),
        pytest.param(2.206136465072632, (28, 83, 115), id="coat-226"),
    ],
)
def test_float32_constant_features_have_only_unpenalized_intercepts(
    constant: float, counts: tuple[int, int, int]
) -> None:
    previous32 = np.zeros((sum(counts), DIM), dtype=np.float32)
    previous32[:, 0] = np.float32(constant)
    previous = previous32.astype(np.float64)
    current = np.zeros((3, DIM))
    current[:, 0] = [0, -constant, 2 * constant]
    outer = _outer(counts)
    expected = _oracle(previous, outer, current)
    intercepts = [round(sum(counts[:2]) / sum(counts), 3), round(counts[0] / sum(counts[:2]), 3)]
    assert expected == [[intercepts] * len(GRID)] * len(current)
    assert probe.score_features(_state(previous, outer), current) == expected


@pytest.mark.parametrize("magnitude", [1.0, 1e-7, 1e-9, 1e-11])
def test_mixed_constant_and_small_nonconstant_columns_match_primal_oracle(magnitude: float) -> None:
    counts = (25, 65, 83)
    outer = _outer(counts)
    rng = np.random.default_rng(7081)
    previous = np.zeros((sum(counts), DIM))
    current = np.zeros((6, DIM))
    constants = np.array([0.31061848998069763, -3.590432643890381], dtype=np.float32)
    previous[:, :2] = constants.astype(np.float64)
    current[:, :2] = np.array([[0, 0], [1, -1], [-1, 1], [2, 2], [-2, -2], [3, -3]])
    previous[:, 2:6] = rng.normal(size=(len(previous), 4)) * magnitude
    current[:, 2:6] = rng.normal(size=(len(current), 4)) * magnitude
    previous[:, 2] += np.array([answer != "u" for answer in outer]) * magnitude
    previous[:, 3] += np.array([answer == "y" for answer in outer]) * magnitude
    expected = _oracle(previous, outer, current)
    intercepts = [round(sum(counts[:2]) / sum(counts), 3), round(counts[0] / sum(counts[:2]), 3)]
    if magnitude == 1e-11:
        # Preserve the original std > 1e-10 behavior even for genuinely varying rows.
        assert expected == [[intercepts] * len(GRID)] * len(current)
    else:
        assert any(pair != intercepts for scores in expected for pair in scores)
    assert probe.score_features(_state(previous, outer), current) == expected


class _Float32Encoder:
    def __init__(self, value: float) -> None:
        self.value = np.float32(value)
        self.calls = 0

    def embed(self, image: Frame) -> Array:
        self.calls += 1
        vector = np.zeros(DIM, dtype=np.float32)
        vector[0] = self.value
        return vector.astype(np.float64)


class _Detector:
    def detect(self, image: Frame) -> list[detect.Detection]:
        return [detect.Detection("person", 0.95, (20.0, 20.0, 35.0, 86.0))]


class _Answers:
    def __init__(self, outer: Sequence[str]) -> None:
        self.outer = outer

    def attributes(
        self, items: Sequence[spotcheck.ReviewItem], deadline: float
    ) -> dict[int, str | None]:
        return {item.number: self.outer[item.number - 1] + "nn" for item in items}


def _invoke(out: Path, directory: Path, outer: Sequence[str]) -> int:
    frames = [np.full((120, 100, 3), 10, dtype=np.uint8) for _ in outer]
    info = spotcheck.DetectorInfo("synthetic", "0" * 64, detect.DEFAULT_CONF)
    pipeline = spotcheck.Pipeline(lambda: frames, _Detector(), info)
    return spotcheck.main(
        [
            "--attributes",
            "--n",
            "500",
            "--min-persons",
            "1",
            "--seed",
            "7",
            "--view",
            "window",
            "--allow-dark",
            "--out-dir",
            str(out),
            "--probe",
            str(directory),
        ],
        pipeline=pipeline,
        reviewer=_Answers(outer),
        today=datetime.date(2026, 10, 10),
        clock=lambda: datetime.datetime(2026, 10, 10, 12, tzinfo=datetime.UTC),
    )


def _labels(out: Path) -> list[dict[str, Any]]:
    records = [json.loads(path.read_text()) for path in (out / "attributes").glob("*.json")]
    return sorted(records, key=lambda record: record["probe_run"]["session"])


@pytest.mark.parametrize(
    ("constant", "counts"),
    [
        pytest.param(0.31061848998069763, (25, 65, 83), id="judge-173"),
        pytest.param(3.8566718101501465, (122, 81, 62), id="coat-265"),
    ],
)
def test_accepted_float32_constant_sessions_store_prior_primal_scores(
    constant: float,
    counts: tuple[int, int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    encoder = _Float32Encoder(constant)
    monkeypatch.setattr(probe, "open_encoder", lambda: encoder)
    out, directory = tmp_path / "out", tmp_path / "stats"
    outer = _outer(counts)
    assert _invoke(out, directory, outer) == 0
    [first] = _labels(out)
    assert first["probe_run"]["state"] == "accepted"
    assert first["probe"]["trained_sessions"] == 0
    assert first["probe"]["scores"] == [None] * sum(counts)
    prior = probe.parse_state((directory / "london.json").read_bytes())
    assert prior.counts() == (sum(counts), sum(counts[:2]), counts[0])

    encoder.value = np.float32(0)
    current_outer = [{"y": "n", "n": "y", "u": "u"}[answer] for answer in outer]
    assert _invoke(out, directory, current_outer) == 0
    first_again, second = _labels(out)
    assert first_again == first
    assert second["probe_run"] == {**first["probe_run"], "session": 2}
    assert second["probe"]["lambdas"] == list(GRID)
    assert second["probe"]["trained_sessions"] == 1
    assert len(second["crops"]) == sum(counts)
    assert encoder.calls == 2 * sum(counts)
    previous = np.zeros((sum(counts), DIM), dtype=np.float32)
    previous[:, 0] = np.float32(constant)
    current = np.zeros((sum(counts), DIM))
    assert second["probe"]["scores"] == _oracle(previous.astype(np.float64), outer, current)
