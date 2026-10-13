"""Exact-input primal regressions for cumulative v2 scores and history provenance.

All vectors and frames are generated. The reference centers direct rows with exact
fractions and solves its own Decimal primal system; it uses no scorer helpers.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import uuid
from collections.abc import Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport.tools import probe, spotcheck, spotcheck_summary

DIM = 768
GRID = (1, 10, 100, 1000)
Array = npt.NDArray[np.float64]
Frame = npt.NDArray[np.uint8]


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _solve(matrix: list[list[Decimal]], rhs: list[Decimal]) -> list[Decimal]:
    """Independent partial-pivot elimination of the small direct-design system."""
    augmented = [[*row, value] for row, value in zip(matrix, rhs, strict=True)]
    dimension = len(rhs)
    for column in range(dimension):
        pivot = max(range(column, dimension), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        assert divisor != 0
        for index in range(column, dimension + 1):
            augmented[column][index] /= divisor
        for row in range(column + 1, dimension):
            factor = augmented[row][column]
            for index in range(column, dimension + 1):
                augmented[row][index] -= factor * augmented[column][index]
    result = [Decimal(0)] * dimension
    for row in range(dimension - 1, -1, -1):
        result[row] = augmented[row][-1] - sum(
            (augmented[row][column] * result[column] for column in range(row + 1, dimension)),
            Decimal(0),
        )
    return result


def _direct_head(
    train: Array, target: Sequence[int], query: Array, precision: int
) -> list[list[float]]:
    with localcontext() as context:
        context.prec = precision
        count = len(train)
        columns: list[list[Decimal]] = []
        queries: list[list[Decimal]] = []
        for index, column in enumerate(train.T):
            if np.all(column == column[0]):
                continue
            values = [Fraction.from_float(float(value)) for value in column]
            mean = sum(values, Fraction(0)) / count
            centered = [value - mean for value in values]
            variance = sum((value * value for value in centered), Fraction(0)) / count
            scale = _decimal(variance).sqrt()
            # The unchanged rule compares a binary64 scalar, including rounded ties.
            if float(scale) <= 1e-10:
                continue
            columns.append([_decimal(value) / scale for value in centered])
            queries.append(
                [
                    _decimal(Fraction.from_float(float(value)) - mean) / scale
                    for value in query[:, index]
                ]
            )
        design = list(zip(*columns, strict=True)) if columns else [()] * count
        query_rows = list(zip(*queries, strict=True)) if queries else [()] * len(query)
        dimension = len(columns)
        intercept = Decimal(sum(target)) / count
        gram = [
            [sum((row[i] * row[j] for row in design), Decimal(0)) for j in range(dimension)]
            for i in range(dimension)
        ]
        covariance = [
            sum(
                (
                    row[i] * (Decimal(value) - intercept)
                    for row, value in zip(design, target, strict=True)
                ),
                Decimal(0),
            )
            for i in range(dimension)
        ]
        output = []
        for penalty in GRID:
            penalized = [row.copy() for row in gram]
            for index in range(dimension):
                penalized[index][index] += penalty
            weights = _solve(penalized, covariance)
            predictions = [
                min(
                    Decimal(1),
                    max(
                        Decimal(0),
                        sum(
                            (value * weight for value, weight in zip(row, weights, strict=True)),
                            Decimal(0),
                        )
                        + intercept,
                    ),
                )
                for row in query_rows
            ]
            # Keep the original binary64 clipping/three-decimal rounding convention.
            output.append([round(float(value), 3) for value in predictions])
        return output


def _primal(
    previous: Array, labels: Sequence[str], current: Array, precision: int = 96
) -> list[list[list[float]]]:
    judgeable = [int(label in "yn") for label in labels]
    selected = np.array(judgeable, dtype=np.bool_)
    judges = _direct_head(previous, judgeable, current, precision)
    coats = _direct_head(
        previous[selected],
        [int(label == "y") for label in labels if label in "yn"],
        current,
        precision,
    )
    return [
        [[judges[penalty][row], coats[penalty][row]] for penalty in range(len(GRID))]
        for row in range(len(current))
    ]


def _state(previous: Array, labels: Sequence[str], sessions: int = 1) -> probe.State:
    groups = {
        name: probe.Cohort.from_rows(previous[np.array([label == name for label in labels])])
        for name in "ynu"
    }
    state = probe.State(str(uuid.uuid4()), "london", probe.ENCODER, sessions, groups)
    state.validate()
    return state


def _roundtrip(state: probe.State) -> probe.State:
    raw = json.dumps(state.document(), allow_nan=False, separators=(",", ":")).encode()
    assert len(raw) <= probe.MAX_STATE_BYTES
    restored = probe.parse_state(raw)
    assert restored.document() == state.document()
    return restored


def _collision(count: int, base: float, step: float) -> tuple[Array, Array, list[str], Array]:
    left = np.zeros((count * 3, DIM))
    left[:, 0] = base
    left[:8, 0] = base + step
    left[count : count + 8, 0] = base - step
    right = left.copy()
    right[8:10, 0] = [base + step, base - step]
    labels = [name for name in "ynu" for _ in range(count)]
    current = np.zeros((2, DIM))
    current[:, 0] = [base + step, base - step]
    return left, right, labels, current


@pytest.mark.parametrize(
    ("count", "passes", "base", "step"),
    [
        (
            128,
            1,
            float(np.float32(0.31061848998069763)),
            float(np.spacing(np.float32(0.31061848998069763))),
        ),
        (
            2048,
            4,
            float(np.float32(0.31061848998069763)),
            float(np.spacing(np.float32(0.31061848998069763))),
        ),
        (128, 1, 999999.0, float(8 * np.spacing(999999.0))),
    ],
    ids=["384-safe-crops", "four-safe-batches", "large-offset"],
)
def test_colliding_raw_moments_retain_distinct_correct_predictions(
    count: int, passes: int, base: float, step: float
) -> None:
    left, right, labels, current = _collision(count, base, step)
    for name in "ynu":
        selected = np.array([label == name for label in labels])
        a, b = left[selected], right[selected]
        assert np.array_equal(a.sum(axis=0), b.sum(axis=0))
        assert np.array_equal(a.T @ a, b.T @ b)
    expected_left, expected_right = (_primal(rows, labels, current) for rows in (left, right))
    assert expected_left != expected_right
    for rows, expected in ((left, expected_left), (right, expected_right)):
        assert _primal(rows, labels, current, 144) == expected
        groups = {name: probe.Cohort.empty() for name in "ynu"}
        state = _state(rows, labels)
        width = count // passes
        for session in range(1, passes + 1):
            for name in "ynu":
                selected_rows = rows[np.array([label == name for label in labels])]
                batch = selected_rows[(session - 1) * width : session * width]
                groups[name] = groups[name].plus(probe.Cohort.from_rows(batch))
            state = _roundtrip(probe.State(state.run_id, "london", probe.ENCODER, session, groups))
            groups = dict(state.cohorts)
        assert probe.score_features(state, current) == expected


@pytest.mark.parametrize("positive_changes", [7, 40])
def test_asymmetric_large_centroids_match_independent_stable_primal(positive_changes: int) -> None:
    previous = np.zeros((384, DIM))
    base, step = 999999.0, float(8 * np.spacing(999999.0))
    previous[:, 0] = base
    previous[:positive_changes, 0] += step
    previous[128:131, 0] -= step
    labels = [name for name in "ynu" for _ in range(128)]
    current = np.zeros((2, DIM))
    current[:, 0] = [base + step, base - step]
    expected = _primal(previous, labels, current)
    assert _primal(previous, labels, current, 144) == expected
    assert expected[0][0][1] != 0.5
    assert probe.score_features(_roundtrip(_state(previous, labels)), current) == expected


def test_valid_mixed_scale_rows_with_underflowed_tiny_diagonal_still_score() -> None:
    previous = np.zeros((6, DIM))
    previous[:, :2] = [
        [1e-170, 1],
        [-1e-170, 2],
        [1e-170, -1],
        [-1e-170, -2],
        [1e-170, 3],
        [-1e-170, -3],
    ]
    labels = ["y", "y", "n", "n", "u", "u"]
    current = previous[[0, 2, 4]].copy()
    expected = _primal(previous, labels, current)
    assert expected == _primal(previous, labels, current, 144)
    assert any(pair[1] != 0.5 for scores in expected for pair in scores)
    state = _roundtrip(_state(previous, labels))
    assert probe.score_features(state, current) == expected
    ordinary = previous.copy()
    ordinary[:, 0] = 0
    assert probe.score_features(_roundtrip(_state(ordinary, labels)), current) == expected


@pytest.mark.parametrize("head", ["judgeable", "coat"])
@pytest.mark.parametrize("boundary", ["below", "equal", "above", "wider-rounds-equal"])
def test_strict_binary64_standard_deviation_boundary(head: str, boundary: str) -> None:
    threshold = 1e-10
    amplitude = {
        "below": math.nextafter(threshold, 0),
        "equal": threshold,
        "above": math.nextafter(threshold, math.inf),
        "wider-rounds-equal": threshold,
    }[boundary]
    labels = ["y"] * 2 + ["n"] * 2 + ["u"] * 4
    previous = np.zeros((8, DIM))
    if head == "judgeable":
        previous[:, 0] = [amplitude] * 4 + [-amplitude] * 4
        if boundary == "wider-rounds-equal":
            previous[0, 0] = math.nextafter(amplitude, math.inf)
            previous[4, 0] = -math.nextafter(amplitude, math.inf)
    else:
        labels = ["y"] * 4 + ["n"] * 4 + ["u"] * 2
        previous = np.zeros((10, DIM))
        previous[:8, 0] = [amplitude] * 4 + [-amplitude] * 4
        if boundary == "wider-rounds-equal":
            previous[0, 0] = math.nextafter(amplitude, math.inf)
            previous[4, 0] = -math.nextafter(amplitude, math.inf)
    train = previous if head == "judgeable" else previous[:8]
    exact = [Fraction.from_float(float(value)) for value in train[:, 0]]
    exact_mean = sum(exact, Fraction(0)) / len(exact)
    variance = sum(((value - exact_mean) ** 2 for value in exact), Fraction(0)) / len(exact)
    with localcontext() as context:
        context.prec = 144
        wider_scale = _decimal(variance).sqrt()
        if boundary == "wider-rounds-equal":
            assert wider_scale > Decimal.from_float(threshold)
            assert float(wider_scale) == threshold
        else:
            assert float(wider_scale) == amplitude
    current = np.zeros((2, DIM))
    current[:, 0] = [threshold, -threshold]
    expected = _primal(previous, labels, current)
    assert _primal(previous, labels, current, 144) == expected
    index = 0 if head == "judgeable" else 1
    assert (expected[0][0][index] != 0.5) == (boundary == "above")
    assert probe.score_features(_roundtrip(_state(previous, labels)), current) == expected


class _Encoder:
    def __init__(self, rows: Array) -> None:
        self.rows = rows
        self.calls = 0

    def embed(self, image: Frame) -> Array:
        row: Array = self.rows[self.calls % len(self.rows)].copy()
        self.calls += 1
        return row


class _Detector:
    def detect(self, image: Frame) -> list[detect.Detection]:
        return [detect.Detection("person", 0.95, (20.0, 20.0, 35.0, 86.0))]


def _invoke(out: Path, directory: Path | None, labels: Sequence[str]) -> int:
    class Answers:
        def attributes(
            self, items: Sequence[spotcheck.ReviewItem], deadline: float
        ) -> dict[int, str]:
            return {item.number: labels[item.number - 1] + "nn" for item in items}

    frames = [np.full((120, 100, 3), 10, dtype=np.uint8) for _ in labels]
    info = spotcheck.DetectorInfo("synthetic", "0" * 64, detect.DEFAULT_CONF)
    argv = [
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
    ]
    if directory is not None:
        argv += ["--probe", str(directory)]
    return spotcheck.main(
        argv,
        pipeline=spotcheck.Pipeline(lambda: frames, _Detector(), info),
        reviewer=Answers(),
        today=datetime.date(2026, 10, 10),
        clock=lambda: datetime.datetime(2026, 10, 10, 12, tzinfo=datetime.UTC),
    )


def _records(out: Path) -> list[dict[str, Any]]:
    paths = sorted((out / "attributes").glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    return [json.loads(path.read_bytes()) for path in paths]


@pytest.fixture
def human_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_twelve_accepted_roundtrips_keep_all_prior_rows_constants_and_tiny_columns(
    human_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = [name for name in "ynu" for _ in range(4)]
    rng = np.random.default_rng(8102)
    batch = np.zeros((12, DIM))
    batch[:, 0] = float(np.float32(0.31061848998069763))
    batch[:, 1] = 999999.0
    batch[:, 2] = 999999.0 + np.arange(12) * (8 * np.spacing(999999.0))
    batch[:, 3:6] = rng.normal(size=(12, 3)) * [1e-7, 1e-9, 1e-11]
    batch[:, 3] += np.array([name != "u" for name in labels]) * 1e-7
    encoder = _Encoder(batch)
    monkeypatch.setattr(probe, "open_encoder", lambda: encoder)
    out, directory = tmp_path / "out", tmp_path / "stats"
    for session in range(1, 13):
        assert _invoke(out, directory, labels) == 0
        record = _records(out)[-1]
        assert record["probe_run"]["version"] == 2
        assert record["probe_run"]["state"] == "accepted"
        assert record["probe_run"]["session"] == session
        if session == 1:
            expected: list[probe.Score] = [None] * len(batch)
        else:
            prior = np.tile(batch, (session - 1, 1))
            expected = list(_primal(prior, labels * (session - 1), batch))
            assert expected == _primal(prior, labels * (session - 1), batch, 144)
        order = sorted(range(len(labels)), key=lambda index: labels[index])
        assert record["probe"]["scores"] == [expected[index] for index in order]
        state = probe.parse_state((directory / "london.json").read_bytes())
        assert state.counts() == (session * 12, session * 8, session * 4)
        for cohort in state.cohorts.values():
            group = cohort.document()
            assert set(group) == {"count", "mean", "scatter"}
            assert np.array(group["scatter"])[0, 0] == np.array(group["scatter"])[1, 1] == 0
    frozen = {
        path: path.read_bytes()
        for root in (directory, out / "probe-runs")
        for path in root.glob("*.json")
    }
    assert _invoke(out, directory, labels) == 0
    assert encoder.calls == 144
    assert _records(out)[-1]["probe_run"]["reason"] == "collection_complete"
    assert all(path.read_bytes() == raw for path, raw in frozen.items())


def _v2_document() -> dict[str, Any]:
    rows = np.zeros((6, DIM))
    rows[:, :2] = [[0.3, 0.6], [0.4, 0.8], [0.5, 1.0], [0.6, 1.2], [0.7, 1.4], [0.8, 1.6]]
    with localcontext() as context:
        context.prec = 96
        groups = {}
        for index, name in enumerate("ynu"):
            selected = rows[index * 2 : index * 2 + 2]
            mean = [
                (Decimal.from_float(float(a)) + Decimal.from_float(float(b))) / 2
                for a, b in zip(*selected, strict=True)
            ]
            limbs = [
                [float(value), float(value - Decimal.from_float(float(value)))] for value in mean
            ]
            centered = np.array(
                [
                    [
                        float(Decimal.from_float(float(value)) - center)
                        for value, center in zip(row, mean, strict=True)
                    ]
                    for row in selected
                ]
            )
            groups[name] = {"count": 2, "mean": limbs, "scatter": (centered.T @ centered).tolist()}
    return {
        "version": 2,
        "run_id": str(uuid.uuid4()),
        "source": "london",
        "encoder": probe.ENCODER,
        "accepted_sessions": 1,
        "cohorts": groups,
    }


@pytest.mark.parametrize(
    "fault",
    [
        "boolean-limb",
        "overlap",
        "tie-high",
        "shape",
        "nonfinite",
        "reconstructed-bound",
        "boolean-scatter",
        "negative-diagonal",
        "asymmetric",
        "cauchy",
        "indefinite",
        "tiny-asymmetric",
        "tiny-cauchy",
        "tiny-indefinite",
        "zero-cohort",
        "singleton",
        "unknown-version",
    ],
)
def test_v2_precision_and_centered_scatter_validation(fault: str) -> None:
    value = _v2_document()
    # Admission must succeed first; rejecting every v2 document is not a parser pass.
    assert probe.parse_state(json.dumps(value).encode()).document() == value
    bad = copy.deepcopy(value)
    group = bad["cohorts"]["y"]
    if fault == "boolean-limb":
        group["mean"][0][1] = True
    elif fault == "overlap":
        group["mean"][0][1] = math.ulp(group["mean"][0][0])
    elif fault == "tie-high":
        high = math.nextafter(1.0, math.inf)
        group["mean"][0] = [high, math.ulp(high) / 2]
    elif fault == "shape":
        group["mean"][0] = [0.3]
    elif fault == "nonfinite":
        group["mean"][0][1] = math.inf
    elif fault == "reconstructed-bound":
        group["mean"][0] = [1e6, math.ulp(1e6) / 4]
    elif fault == "boolean-scatter":
        group["scatter"][0][0] = True
    elif fault == "negative-diagonal":
        group["scatter"][0][0] = -1e-30
    elif fault == "asymmetric":
        group["scatter"][0][1] = 0.1
    elif fault == "cauchy":
        group["scatter"][0][1] = group["scatter"][1][0] = 1.0
    elif fault == "indefinite":
        for i in range(3):
            for j in range(3):
                group["scatter"][i][j] = 1.0 if i == j else -0.75
    elif fault.startswith("tiny-"):
        group["scatter"] = np.zeros((DIM, DIM)).tolist()
        for i in range(3):
            group["scatter"][i][i] = 1e-18
        if fault == "tiny-asymmetric":
            group["scatter"][0][1] = 5e-19
        elif fault == "tiny-cauchy":
            group["scatter"][0][1] = group["scatter"][1][0] = 1.1e-18
        else:
            for i in range(3):
                for j in range(i):
                    group["scatter"][i][j] = group["scatter"][j][i] = -7.5e-19
    elif fault == "zero-cohort":
        group["count"] = 0
    elif fault == "singleton":
        group["count"] = 1
    else:
        bad["version"] = 3
    raw = json.dumps(bad).encode()
    with pytest.raises(probe.ProbeError) as raised:
        probe.parse_state(raw)
    assert str(raised.value) in ("invalid_state", "unsafe_increment")
    assert raw == json.dumps(bad).encode()


@pytest.mark.parametrize("sign", [-1, 1])
def test_normalized_even_high_half_ulp_ties_are_accepted(sign: int) -> None:
    value = _v2_document()
    high = sign * 1.0
    value["cohorts"]["y"]["mean"][0] = [high, sign * math.ulp(high) / 2]
    assert probe.parse_state(json.dumps(value).encode()).document() == value


@pytest.mark.parametrize("version", [1, 2])
def test_generic_human_summary_accepts_both_versions_and_ignores_scores(
    version: int, human_env: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = ["y", "y", "n", "n", "u", "u"]
    out = tmp_path / "out"
    assert _invoke(out, None, labels) == 0
    path = next((out / "attributes").glob("*.json"))
    original = json.loads(path.read_bytes())
    capsys.readouterr()
    assert spotcheck_summary.main(["--attributes", "--dir", str(out)]) == 0
    plain = capsys.readouterr().out
    for scores in ([[[0.0, 1.0]] * 4] * 6, [[[1.0, 0.0]] * 4] * 6):
        record = {
            **original,
            "probe_run": {
                "version": version,
                "run_id": str(uuid.uuid4()),
                "session": 2,
                "state": "accepted",
                "reason": None,
            },
            "probe": {
                "encoder": probe.ENCODER,
                "lambdas": list(GRID),
                "trained_sessions": 1,
                "scores": scores,
            },
        }
        path.write_text(json.dumps(record), encoding="utf-8")
        assert probe.validate_label_metadata(record) == record["probe_run"]
        assert spotcheck_summary.main(["--attributes", "--dir", str(out)]) == 0
        assert capsys.readouterr().out == plain


@pytest.mark.parametrize(
    "fault",
    [
        "legacy-all",
        "legacy-control",
        "legacy-aggregate",
        "legacy-history",
        "mixed-history",
        "unknown-history",
    ],
)
def test_legacy_or_mixed_provenance_stops_before_features_and_preserves_bytes(
    fault: str,
    human_env: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels = ["y", "y", "n", "n", "u", "u"]
    rows = np.zeros((6, DIM))
    rows[:, 0] = np.arange(6)
    encoder = _Encoder(rows)
    monkeypatch.setattr(probe, "open_encoder", lambda: encoder)
    out, directory = tmp_path / "out", tmp_path / "stats"
    for _ in range(2):
        assert _invoke(out, directory, labels) == 0
    aggregate, control = directory / "london.json", out / "probe-runs" / "london.json"
    history = sorted((out / "attributes").glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    selected = {
        "legacy-all": [aggregate, control, *history],
        "legacy-control": [control],
        "legacy-aggregate": [aggregate],
        "legacy-history": history,
        "mixed-history": history[:1],
        "unknown-history": history[:1],
    }[fault]
    for path in selected:
        value = json.loads(path.read_bytes())
        if path in history:
            value["probe_run"]["version"] = 3 if fault == "unknown-history" else 1
        else:
            value["version"] = 1
            if path == aggregate and "mean" in value["cohorts"]["y"]:
                # Real legacy shape, not simply a v2 shape bearing a v1 marker.
                for group in value["cohorts"].values():
                    mean = np.array(group.pop("mean"))
                    center = mean[:, 0] + mean[:, 1]
                    scatter = np.array(group.pop("scatter"))
                    group["sum"] = (group["count"] * center).tolist()
                    group["cross"] = (scatter + group["count"] * np.outer(center, center)).tolist()
        path.write_text(json.dumps(value), encoding="utf-8")
    frozen = {path: path.read_bytes() for path in (aggregate, control, *history)}
    before = encoder.calls
    capsys.readouterr()
    assert _invoke(out, directory, labels) == 0
    assert encoder.calls == before
    assert all(path.read_bytes() == raw for path, raw in frozen.items())
    failed = _records(out)[-1]
    assert failed["crops"] == _records(out)[0]["crops"]
    assert "probe" not in failed
    assert failed["probe_run"]["version"] == 2
    assert failed["probe_run"]["state"] == "stopped"
    assert failed["probe_run"]["session"] is None
    lines = [line for line in capsys.readouterr().err.splitlines() if "probe" in line.lower()]
    assert len(lines) == 1
    assert len(lines[0]) < 180
    assert _invoke(out, directory, labels) == 0
    assert encoder.calls == before
    assert all(path.read_bytes() == raw for path, raw in frozen.items())
