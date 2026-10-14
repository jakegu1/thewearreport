"""Synthetic direct-row references for exact v3 moments and dyadic geometry.

The score oracle fits rational primal equations from rows, with its own Decimal
square root. No scorer arithmetic, moment, selection, or rounding helper is used.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
from collections.abc import Callable, Sequence
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
L = 1 << 1074
T = 1e-10
U = math.nextafter(T, math.inf)
IDENTITY = "11111111-1111-1111-1111-111111111111"
Array = npt.NDArray[np.float64]
Frame = npt.NDArray[np.uint8]


def _power(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _pair(center: Fraction) -> list[float]:
    high = float(center)
    low = float(center - Fraction(high))
    encoded = Fraction(high) + Fraction(low)
    high = float(encoded)
    low = float(encoded - Fraction(high))
    return [0.0 if high == 0 else high, 0.0 if low == 0 else low]


def _root(variance: Fraction) -> float:
    with localcontext() as context:
        context.prec = 200
        return float((Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt())


def _exponent(variance: Fraction) -> int:
    if not variance:
        return 0
    power = variance.numerator.bit_length() - variance.denominator.bit_length()
    if variance < _power(power):
        power -= 1
    return power // 2


def _solve(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    augmented = [[*row, value] for row, value in zip(matrix, rhs, strict=True)]
    for column in range(len(rhs)):
        pivot = next(row for row in range(column, len(rhs)) if augmented[row][column])
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(len(rhs)):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [
                    a - factor * b for a, b in zip(augmented[row], augmented[column], strict=True)
                ]
    return [row[-1] for row in augmented]


def _direct_head(rows: Array, targets: Sequence[int], current: Array) -> list[list[float]]:
    columns: list[list[Fraction]] = []
    queries: list[list[Fraction]] = []
    for index, column in enumerate(rows.T):
        if np.all(column == column[0]):
            continue
        values = [Fraction(float(value)) for value in column]
        center = sum(values, Fraction()) / len(values)
        centered = [value - center for value in values]
        variance = sum((value * value for value in centered), Fraction()) / len(values)
        scale = _root(variance)
        if scale > T:
            columns.append([value / Fraction(scale) for value in centered])
            queries.append(
                [
                    Fraction(float(value)) / Fraction(scale) - center / Fraction(scale)
                    for value in current[:, index]
                ]
            )
    intercept = Fraction(sum(targets), len(targets))
    gram = [
        [sum((a * b for a, b in zip(x, y, strict=True)), Fraction()) for y in columns]
        for x in columns
    ]
    covariance = [
        sum((x * (y - intercept) for x, y in zip(column, targets, strict=True)), Fraction())
        for column in columns
    ]
    output: list[list[float]] = [[] for _ in current]
    for penalty in GRID:
        penalized = [row.copy() for row in gram]
        for index in range(len(columns)):
            penalized[index][index] += penalty
        weights = _solve(penalized, covariance)
        for index in range(len(current)):
            prediction = intercept + sum(
                (column[index] * weight for column, weight in zip(queries, weights, strict=True)),
                Fraction(),
            )
            output[index].append(round(float(min(Fraction(1), max(Fraction(), prediction))), 3))
    return output


def _scores(rows: Array, labels: Sequence[str], current: Array) -> list[list[list[float]]]:
    judgeable = [int(label in "yn") for label in labels]
    judges = _direct_head(rows, judgeable, current)
    coats = _direct_head(
        rows[np.array(judgeable, dtype=np.bool_)],
        [int(label == "y") for label in labels if label in "yn"],
        current,
    )
    return [
        [[judge, coat] for judge, coat in zip(j, c, strict=True)]
        for j, c in zip(judges, coats, strict=True)
    ]


def _rows(values: Sequence[float]) -> Array:
    result = np.zeros((len(values), DIM))
    result[:, 0] = values
    return result


def _state(rows: Array, labels: Sequence[str], sessions: int = 1) -> probe.State:
    return probe.State(
        IDENTITY,
        "london",
        probe.ENCODER,
        sessions,
        {
            name: probe.Cohort.from_rows(rows[np.array([label == name for label in labels])])
            for name in "ynu"
        },
    )


def _raw(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _roundtrip(state: probe.State) -> probe.State:
    document = state.document()
    restored = probe.parse_state(_raw(document))
    assert restored.document() == document
    return restored


def _group(rows: Array) -> dict[str, Any]:
    """Independent exact row geometry, also usable as a valid parser fixture."""
    count = len(rows)
    mean = np.zeros((DIM, 2)).tolist()
    scatter = np.zeros((DIM, DIM)).tolist()
    moments = [["0", "0"] for _ in range(DIM)]
    centered: dict[int, list[Fraction]] = {}
    for index, column in enumerate(rows.T):
        if not count or not np.any(column):
            continue
        values = [Fraction(float(value)) for value in column]
        center = sum(values, Fraction()) / count
        mean[index] = _pair(center)
        integers = [int(value * L) for value in values]
        moments[index] = [str(sum(integers)), str(sum(value * value for value in integers))]
        deviations = [value - center for value in values]
        variance = sum((value * value for value in deviations), Fraction()) / count
        if variance:
            unit = _power(-_exponent(variance))
            centered[index] = [value * unit for value in deviations]
    for i, x in centered.items():
        for j, y in centered.items():
            scatter[i][j] = float(sum((a * b for a, b in zip(x, y, strict=True)), Fraction()))
    return {"count": count, "mean": mean, "scatter": scatter, "diagonal": moments}


def _document(rows: Array, labels: Sequence[str]) -> dict[str, Any]:
    return {
        "version": 3,
        "run_id": IDENTITY,
        "source": "london",
        "encoder": probe.ENCODER,
        "accepted_sessions": 1,
        "cohorts": {
            name: _group(rows[np.array([label == name for label in labels])]) for name in "ynu"
        },
    }


@pytest.mark.parametrize("head", ["judgeable", "coat"])
def test_original_apis_exclude_both_known_equal_threshold_score_errors(head: str) -> None:
    width = 12 if head == "judgeable" else 20
    positive = [T] * (3 * width // 4) + [-T] * (width // 4)
    negative = positive if head == "judgeable" else [T] * (width // 4) + [-T] * (3 * width // 4)
    unknown = [T] * 6 + [-T] * 18 if head == "judgeable" else [0.0, 0.0]
    labels = ["y"] * width + ["n"] * width + ["u"] * len(unknown)
    rows, current = _rows(positive + negative + unknown), _rows([T, -T])
    expected = _scores(rows, labels, current)
    index = 0 if head == "judgeable" else 1
    assert all(pair[index] == 0.5 for prediction in expected for pair in prediction)
    state = _state(rows, labels)
    # Score equality fails behaviorally on the old product, before any new field access.
    assert probe.score_features(state, current) == expected
    assert probe.score_features(_roundtrip(state), current) == expected


@pytest.mark.parametrize("head", ["judgeable", "coat"])
@pytest.mark.parametrize("boundary", ["below", "equal", "successor", "wider-equal", "midpoint"])
def test_exact_rounding_boundary_both_heads_all_penalties(head: str, boundary: str) -> None:
    amplitude = {"below": math.nextafter(T, 0), "successor": U}.get(boundary, T)
    values = [amplitude] * 4 + [-amplitude] * 4
    if boundary == "wider-equal":
        values[0], values[4] = U, -U
    if boundary == "midpoint":
        values[4:] = [-U] * 4
    if head == "judgeable":
        labels = ["y"] * 2 + ["n"] * 2 + ["u"] * 4
    else:
        labels = ["y"] * 4 + ["n"] * 4 + ["u"] * 2
        values += [0.0, 0.0]
    rows, current = _rows(values), _rows([T, -T])
    selected = rows[:, 0] if head == "judgeable" else rows[:8, 0]
    exact = [Fraction(float(value)) for value in selected]
    center = sum(exact, Fraction()) / len(exact)
    variance = sum(((value - center) ** 2 for value in exact), Fraction()) / len(exact)
    assert (_root(variance) > T) == (boundary in ("successor", "midpoint"))
    if boundary == "midpoint":
        assert variance == ((Fraction(T) + Fraction(U)) / 2) ** 2
        assert _root(variance) == U
    expected = _scores(rows, labels, current)
    index = 0 if head == "judgeable" else 1
    assert (expected[0][0][index] != 0.5) == (boundary in ("successor", "midpoint"))
    assert probe.score_features(_roundtrip(_state(rows, labels)), current) == expected


def _legacy(rows: Array, labels: Sequence[str]) -> bytes:
    """Reproduce complete historical v2 bytes independently of the current product."""
    groups: dict[str, Any] = {}
    for name in "ynu":
        selected = rows[np.array([label == name for label in labels])]
        mean = np.zeros((DIM, 2))
        with localcontext() as context:
            context.prec = 96
            center = sum((Decimal.from_float(float(x)) for x in selected[:, 0]), Decimal()) / len(
                selected
            )
            mean[0] = _pair(Fraction(center))
            shift = float(center - Decimal.from_float(float(selected[0, 0])))
        centered = selected - selected[0]
        centered[:, 0] -= shift
        groups[name] = {
            "count": len(selected),
            "mean": mean.tolist(),
            "scatter": (centered.T @ centered).tolist(),
        }
    return _raw(
        {
            "version": 2,
            "run_id": IDENTITY,
            "source": "london",
            "encoder": probe.ENCODER,
            "accepted_sessions": 1,
            "cohorts": groups,
        }
    )


def test_complete_legacy_collision_separates_v3_decisions() -> None:
    yn = [U] * 3 + [T] * 6 + [-U] * 3
    below = [
        -T,
        -T,
        -T,
        -U,
        -U,
        -T,
        U,
        U,
        -T,
        -T,
        -U,
        -T,
        -T,
        T,
        -T,
        -T,
        -U,
        -T,
        -U,
        -T,
        U,
        -T,
        U,
        U,
    ]
    above = [
        -T,
        -T,
        -U,
        -U,
        -T,
        -T,
        U,
        U,
        -T,
        -T,
        -T,
        -U,
        -T,
        -T,
        -U,
        U,
        -T,
        -U,
        -U,
        -T,
        U,
        -T,
        U,
        U,
    ]
    labels = ["y"] * 12 + ["n"] * 12 + ["u"] * 24
    histories = [_rows(yn + yn + unknown) for unknown in (below, above)]
    legacy = [_legacy(rows, labels) for rows in histories]
    assert legacy[0] == legacy[1]
    assert len(legacy[0]) == 7_105_881
    assert (
        hashlib.sha256(legacy[0]).hexdigest()
        == "fc03c7df28ac513bdab784769ea30742905d46f00bccd478d071731041b8d18a"
    )
    current = _rows([T, -T])
    expected = [_scores(rows, labels, current) for rows in histories]
    assert expected[0] != expected[1]
    assert all(pair[0] == 0.5 for prediction in expected[0] for pair in prediction)
    states = [_roundtrip(_state(rows, labels)) for rows in histories]
    assert states[0].document() != states[1].document()
    assert (
        states[0].cohorts["u"].document()["diagonal"]
        != states[1].cohorts["u"].document()["diagonal"]
    )
    for state, oracle in zip(states, expected, strict=True):
        assert probe.score_features(state, current) == oracle


@pytest.mark.parametrize(
    "kind", ["tiny-correlated", "min-subnormal", "offset", "upper-rn", "residual"]
)
def test_exact_centroids_and_positive_dyadic_geometry(kind: str) -> None:
    a, b = float.fromhex("0x1.b9ecad3d97ff8p-538"), float.fromhex("0x1.1da4c923914c3p-537")
    rows = np.zeros((4, DIM))
    if kind == "tiny-correlated":
        rows[:, :3] = np.array([[a, b, 1], [-a, -b, 1], [0, 0, 2], [0, 0, 0]], dtype=np.float64)
    elif kind == "min-subnormal":
        tiny = math.ulp(0.0)
        rows[:, :3] = np.array(
            [[tiny, -tiny, -0.0], [-tiny, tiny, 0.0], [0, 0, 0], [0, 0, -0.0]], dtype=np.float64
        )
    elif kind == "offset":
        base = 999999.0
        rows[:, :3] = [
            [base, base, 0],
            [base + math.ulp(base), base, 1],
            [base, base + math.ulp(base), -1],
            [base, base, 0],
        ]
    elif kind == "upper-rn":
        rows[:, 0] = [2, 2, -2, -math.nextafter(2, 0)]
    else:
        rows[:, 0] = [2, 1, 1 + 3 * 2**-51, -math.ulp(0.0)]
    expected = _group(rows)
    group = probe.Cohort.from_rows(rows)
    group.validate()
    actual = group.document()
    assert set(actual) == {"count", "mean", "scatter", "diagonal"}
    assert actual["mean"] == expected["mean"]
    assert actual["diagonal"] == expected["diagonal"]
    assert np.allclose(group.scatter, expected["scatter"], rtol=2e-15, atol=0)
    assert all(math.copysign(1, limb) == 1 for pair in group.mean for limb in pair if limb == 0)
    if kind in ("tiny-correlated", "min-subnormal"):
        assert group.scatter[0, 0] > 0 and group.scatter[0, 1] != 0
        assert not np.any(group.scatter[3:])
    if kind == "upper-rn":
        assert group.scatter[0, 0] == 4 * len(rows)
    if kind == "residual":
        center = sum((Fraction(float(x)) for x in rows[:, 0]), Fraction()) / 4
        initial = float(center)
        assert initial.hex() == "0x1.0000000000001p+0"
        assert float(group.mean[0, 0]).hex() == "0x1.0000000000002p+0"


def test_correlated_underflow_and_ordinary_heads_match_direct_rows() -> None:
    a, b = float.fromhex("0x1.b9ecad3d97ff8p-538"), float.fromhex("0x1.1da4c923914c3p-537")
    rows = np.zeros((12, DIM))
    labels = [name for name in "ynu" for _ in range(4)]
    for index, offset in enumerate((1, -1, 0)):
        rows[index * 4 : index * 4 + 4, :3] = np.array(
            [
                [a, b, offset],
                [-a, -b, offset],
                [0, 0, offset + 1],
                [0, 0, offset - 1],
            ],
            dtype=np.float64,
        )
    state = _roundtrip(_state(rows, labels))
    expected = _scores(rows, labels, rows)
    assert any(pair[1] != 0.5 for prediction in expected for pair in prediction)
    assert probe.score_features(state, rows) == expected
    assert all(
        group.scatter[0, 0] > 0 and group.scatter[1, 1] > 0 for group in state.cohorts.values()
    )


def test_safe_partitions_orders_and_dyadic_merge_transitions() -> None:
    rng = np.random.default_rng(81003)
    rows = np.zeros((96, DIM))
    rows[:, :4] = rng.normal(size=(96, 4)) * [1, 1e-7, 1e-9, 1e-11]
    rows[:, 4] = 999999 + np.arange(96) * math.ulp(999999.0)
    rows[:32, 5] = math.ulp(0.0)
    rows[32:64, 5] = -math.ulp(0.0)
    rows[64:, 5] = 1e-100
    expected = _group(rows)
    for indices in (np.arange(96), np.arange(95, -1, -1), rng.permutation(96)):
        group = probe.Cohort.empty()
        for batch in np.array_split(indices, 12):
            group = group.plus(probe.Cohort.from_rows(rows[batch]))
            group.validate()
        actual = group.document()
        assert actual["diagonal"] == expected["diagonal"]
        assert actual["mean"] == expected["mean"]
        assert np.allclose(group.scatter, expected["scatter"], rtol=8e-14, atol=1e-13)


@pytest.fixture
def valid_document() -> dict[str, Any]:
    rows = np.zeros((12, DIM))
    rows[:, :3] = np.tile([[1, 2, -1], [-1, 1, 2], [2, -1, 1], [-2, -2, -2]], (3, 1))
    return _document(rows, [name for name in "ynu" for _ in range(4)])


@pytest.mark.parametrize(
    "fault",
    [
        "s-bool",
        "q-int",
        "shape",
        "pair-shape",
        "s-leading",
        "q-leading",
        "s-plus",
        "q-minus",
        "s-minus-zero",
        "q-minus-zero",
        "unicode",
        "whitespace",
        "exponent",
        "decimal",
        "s-length",
        "q-length",
        "s-bound",
        "q-bound",
        "negative-d",
        "centroid",
        "diag",
        "negative-zero-high",
        "negative-zero-low",
        "zero-count",
        "singleton",
        "count-bool",
        "cohort-count-cap",
        "total-cap",
        "nonzero-empty",
        "empty-moment",
        "zero-row",
        "infinite",
        "value-cap",
    ],
)
def test_parser_rejects_malformed_exact_entries_and_inconsistent_geometry(
    valid_document: dict[str, Any], fault: str
) -> None:
    assert probe.parse_state(_raw(valid_document)).document() == valid_document
    bad = copy.deepcopy(valid_document)
    group = bad["cohorts"]["y"]
    pair = group["diagonal"][0]
    changes: dict[str, tuple[int, object]] = {
        "s-bool": (0, True),
        "q-int": (1, 0),
        "s-leading": (0, "01"),
        "q-leading": (1, "01"),
        "s-plus": (0, "+1"),
        "q-minus": (1, "-1"),
        "s-minus-zero": (0, "-0"),
        "q-minus-zero": (1, "-0"),
        "unicode": (0, "\u0661"),
        "whitespace": (0, "0\n"),
        "exponent": (0, "1e2"),
        "decimal": (0, "1.0"),
        "s-length": (0, "9" * 341),
        "q-length": (1, "9" * 669),
        "s-bound": (0, str(4 * 1_000_000 * L + 1)),
        "q-bound": (1, str(4 * (1_000_000 * L) ** 2 + 1)),
    }
    if fault in changes:
        index, value = changes[fault]
        pair[index] = value
    elif fault == "shape":
        group["diagonal"].pop()
    elif fault == "pair-shape":
        pair.append("0")
    elif fault == "negative-d":
        pair[:] = ["1", "0"]
    elif fault == "centroid":
        group["mean"][0] = [0.25, 0.0]
    elif fault == "diag":
        group["scatter"][0][0] = math.nextafter(group["scatter"][0][0], math.inf)
    elif fault.startswith("negative-zero-"):
        group["mean"][3][0 if fault.endswith("high") else 1] = -0.0
    elif fault in ("zero-count", "singleton", "count-bool", "cohort-count-cap"):
        group["count"] = {
            "zero-count": 0,
            "singleton": 1,
            "count-bool": True,
            "cohort-count-cap": 1_000_000_001,
        }[fault]
    elif fault == "total-cap":
        zero = _group(np.zeros((2, DIM)))
        zero["count"] = 500_000_001
        bad["cohorts"]["y"] = zero
        bad["cohorts"]["n"] = copy.deepcopy(zero)
    elif fault in ("nonzero-empty", "empty-moment"):
        group.update(_group(np.zeros((0, DIM))))
        if fault == "nonzero-empty":
            group["mean"][0][0] = 1.0
        else:
            group["diagonal"][0][1] = "1"
    elif fault == "zero-row":
        group["scatter"][3][0] = group["scatter"][0][3] = 1e-300
    elif fault == "infinite":
        group["mean"][0][0] = math.inf
    else:
        group["scatter"][0][1] = group["scatter"][1][0] = 1e22
    raw = json.dumps(bad, separators=(",", ":")).encode()
    with pytest.raises(probe.ProbeError) as failure:
        probe.parse_state(raw)
    assert str(failure.value) in ("invalid_state", "unsafe_increment")
    assert raw == json.dumps(bad, separators=(",", ":")).encode()


@pytest.mark.parametrize("magnitude", [1.0, 1e-18])
@pytest.mark.parametrize("fault", ["asymmetric", "cauchy", "indefinite"])
def test_original_hostile_geometry_rejected_in_independent_dyadic_units(
    magnitude: float, fault: str
) -> None:
    amplitude = math.sqrt(magnitude) / 2
    rows = np.zeros((4, DIM))
    rows[:, :3] = [[amplitude] * 3, [amplitude] * 3, [-amplitude] * 3, [-amplitude] * 3]
    value = _document(rows, ["y"] * 4)
    assert probe.parse_state(_raw(value)).document() == value
    physical = np.eye(3) * magnitude
    if fault == "asymmetric":
        physical[0, 1] = magnitude / 2
    elif fault == "cauchy":
        physical[0, 1] = physical[1, 0] = magnitude * 1.1
    else:
        physical[:] = -0.75 * magnitude
        np.fill_diagonal(physical, magnitude)
    variance = Fraction(amplitude) ** 2
    unit = _power(-_exponent(variance))
    # Keep exact moment diagonals; mutate only the hostile off-diagonal entries.
    for i in range(3):
        for j in range(3):
            if i != j:
                value["cohorts"]["y"]["scatter"][i][j] = float(
                    Fraction(float(physical[i, j])) * unit**2
                )
    with pytest.raises(probe.ProbeError, match="invalid_state"):
        probe.parse_state(_raw(value))


def test_minimum_d_exponent_and_maximum_count_are_bounded() -> None:
    count, summed, squares = 31622**2 + 1, 31622, 1
    assert count <= 10**9 and count * squares - summed * summed == 1
    variance = Fraction(1, count * count * L * L)
    assert _exponent(variance) == -1104
    group = _group(np.zeros((2, DIM)))
    group["count"] = count
    group["diagonal"][0] = [str(summed), str(squares)]
    group["mean"][0] = _pair(Fraction(summed, count * L))
    group["scatter"][0][0] = float(count * variance * _power(2208))
    value = _document(np.zeros((2, DIM)), ["y", "y"])
    value["cohorts"]["y"] = group
    parsed = probe.parse_state(_raw(value))
    assert parsed.document() == value
    assert parsed.cohorts["y"].scatter[0, 0] > 0
    cap = _group(np.zeros((2, DIM)))
    cap["count"] = 10**9
    value["cohorts"]["y"] = cap
    assert probe.parse_state(_raw(value)).counts() == (10**9, 10**9, 10**9)
    with pytest.raises(probe.ProbeError):
        probe.parse_state(_raw(value)).cohorts["y"].plus(probe.Cohort.from_rows(np.zeros((2, DIM))))


def test_finite_feature_limit_and_exact_addition() -> None:
    rows = _rows([1e6, -1e6, math.ulp(0.0), -math.ulp(0.0)])
    group = probe.Cohort.from_rows(rows)
    group.validate()
    first, second = (probe.Cohort.from_rows(part) for part in np.array_split(rows, 2))
    assert first.plus(second).document()["diagonal"] == _group(rows)["diagonal"]
    assert probe.Cohort.empty().plus(group).document() == group.document()
    assert group.plus(probe.Cohort.empty()).document() == group.document()
    rows[0, 0] = math.nextafter(1e6, math.inf)
    with pytest.raises(probe.ProbeError, match="encoder_failed"):
        probe.Cohort.from_rows(rows)


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


def _invoke(
    out: Path,
    directory: Path,
    labels: Sequence[str | None],
    during: Callable[[], None] | None = None,
) -> int:
    class Answers:
        def attributes(
            self, items: Sequence[spotcheck.ReviewItem], deadline: float
        ) -> dict[int, str | None]:
            if during is not None:
                during()
            return {
                item.number: None
                if labels[item.number - 1] is None
                else str(labels[item.number - 1]) + "nn"
                for item in items
            }

    frames = [np.full((120, 100, 3), 10, dtype=np.uint8) for _ in labels]
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
        pipeline=spotcheck.Pipeline(
            lambda: frames,
            _Detector(),
            spotcheck.DetectorInfo("synthetic", "0" * 64, detect.DEFAULT_CONF),
        ),
        reviewer=Answers(),
        today=datetime.date(2026, 10, 10),
        clock=lambda: datetime.datetime(2026, 10, 10, 12, tzinfo=datetime.UTC),
    )


@pytest.fixture
def human_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def _records(out: Path) -> list[dict[str, Any]]:
    paths = sorted(
        (out / "attributes").glob("*.json"), key=lambda path: (path.stat().st_mtime_ns, path.name)
    )
    return [json.loads(path.read_bytes()) for path in paths]


def test_twelve_real_accepted_sessions_exact_roundtrips_and_prior_only_scores(
    human_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels: list[str | None] = ["y"] * 4 + ["n"] * 4 + ["u"] * 4 + [None]
    kept = [str(label) for label in labels if label is not None]
    rng = np.random.default_rng(81019)
    batch = np.zeros((13, DIM))
    batch[:, 0] = 999999
    batch[:, 1] = 999999 + np.arange(13) * math.ulp(999999.0)
    batch[:, 2:5] = rng.normal(size=(13, 3)) * [1e-7, 1e-9, 1e-11]
    batch[:, 5] = np.resize([math.ulp(0.0), -math.ulp(0.0), 0, 0], 13)
    encoder = _Encoder(batch)
    monkeypatch.setattr(probe, "open_encoder", lambda: encoder)
    original = probe.score_features
    prepared: list[list[probe.Score]] = []

    def observe(state: probe.State, features: Array) -> list[probe.Score]:
        result = original(state, features)
        prepared.append(result)
        return result

    monkeypatch.setattr(probe, "score_features", observe)
    out, directory = tmp_path / "out", tmp_path / "stats"
    for session in range(1, 13):

        def before_answers(expected_session: int = session) -> None:
            assert encoder.calls == expected_session * 13
            assert len(prepared) == expected_session

        assert _invoke(out, directory, labels, before_answers) == 0
        record = _records(out)[-1]
        assert record["probe_run"]["version"] == 3
        assert record["probe_run"]["state"] == "accepted"
        assert record["probe_run"]["session"] == session
        assert record["probe"]["trained_sessions"] == session - 1
        expected: list[probe.Score] = [None] * 12
        if session > 1:
            expected = list(
                _scores(np.tile(batch[:12], (session - 1, 1)), kept * (session - 1), batch[:12])
            )
        order = sorted(range(12), key=lambda index: kept[index])
        assert record["probe"]["scores"] == [expected[index] for index in order]
        state = probe.parse_state((directory / "london.json").read_bytes())
        assert state.counts() == (session * 12, session * 8, session * 4)
        for name in "ynu":
            selected = batch[:12][np.array([label == name for label in kept])]
            reference = _group(np.tile(selected, (session, 1)))
            group = state.cohorts[name].document()
            assert group["diagonal"] == reference["diagonal"]
            assert group["mean"] == reference["mean"]
            assert state.cohorts[name].scatter[5, 5] > 0
            assert state.cohorts[name].scatter[0, 0] == 0
    paths = [directory / "london.json", out / "probe-runs" / "london.json"]
    frozen = {path: path.read_bytes() for path in paths}
    assert _invoke(out, directory, labels) == 0
    assert encoder.calls == 156
    assert _records(out)[-1]["probe_run"]["reason"] == "collection_complete"
    assert all(path.read_bytes() == raw for path, raw in frozen.items())


@pytest.mark.parametrize("version", [1, 2, 4])
@pytest.mark.parametrize("member", ["aggregate", "control", "history", "all"])
def test_legacy_unknown_and_mixed_histories_preserve_bytes_before_model_work(
    version: int, member: str, human_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = ["y", "y", "n", "n", "u", "u"]
    encoder = _Encoder(_rows([1, 2, -1, -2, 0, 1]))
    monkeypatch.setattr(probe, "open_encoder", lambda: encoder)
    out, directory = tmp_path / "out", tmp_path / "stats"
    for _ in range(2):
        assert _invoke(out, directory, labels) == 0
    aggregate, control = directory / "london.json", out / "probe-runs" / "london.json"
    history = sorted((out / "attributes").glob("*.json"))
    selected = {
        "aggregate": [aggregate],
        "control": [control],
        "history": history[:1],
        "all": [aggregate, control, *history],
    }[member]
    for path in selected:
        value = json.loads(path.read_bytes())
        if path in history:
            value["probe_run"]["version"] = version
        else:
            value["version"] = version
            if path == aggregate and version in (1, 2):
                prior = np.tile(encoder.rows, (2, 1))
                for name, group in value["cohorts"].items():
                    rows = prior[np.array([label == name for label in labels * 2])]
                    exact = [Fraction(float(x)) for x in rows[:, 0]]
                    center = sum(exact, Fraction()) / len(rows)
                    physical = np.zeros((DIM, DIM))
                    physical[0, 0] = float(sum(((x - center) ** 2 for x in exact), Fraction()))
                    group["scatter"] = physical.tolist()
                    del group["diagonal"]
                    if version == 1:
                        mean = np.array(group.pop("mean"))
                        point = mean[:, 0] + mean[:, 1]
                        group["sum"] = (group["count"] * point).tolist()
                        group["cross"] = (
                            physical + group["count"] * np.outer(point, point)
                        ).tolist()
                        del group["scatter"]
        path.write_bytes(_raw(value))
    frozen = {path: path.read_bytes() for path in (aggregate, control, *history)}
    calls = encoder.calls
    for _ in range(2):
        assert _invoke(out, directory, labels) == 0
        assert encoder.calls == calls
        record = _records(out)[-1]
        assert "probe" not in record
        assert record["probe_run"]["version"] == 3
        assert record["probe_run"]["state"] == "stopped"
        assert record["probe_run"]["session"] is None
        assert all(path.read_bytes() == raw for path, raw in frozen.items())


@pytest.mark.parametrize("version", [1, 2, 3])
def test_generic_summary_ignores_valid_optional_versions_and_names_bad_file(
    version: int,
    human_env: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    encoder = _Encoder(_rows([1, 2, -1, -2, 0, 1]))
    monkeypatch.setattr(probe, "open_encoder", lambda: encoder)
    out, directory = tmp_path / "out", tmp_path / "stats"
    assert _invoke(out, directory, ["y", "y", "n", "n", "u", "u"]) == 0
    path = next((out / "attributes").glob("*.json"))
    value = json.loads(path.read_bytes())
    plain = {key: item for key, item in value.items() if key not in ("probe", "probe_run")}
    path.write_bytes(_raw(plain))
    capsys.readouterr()
    assert spotcheck_summary.main(["--attributes", "--dir", str(out)]) == 0
    expected = capsys.readouterr().out
    value["probe_run"]["version"] = version
    path.write_bytes(_raw(value))
    assert spotcheck_summary.main(["--attributes", "--dir", str(out)]) == 0
    assert capsys.readouterr().out == expected
    value["probe_run"]["version"] = 4
    path.write_bytes(_raw(value))
    assert spotcheck_summary.main(["--attributes", "--dir", str(out)]) == 1
    assert path.name in capsys.readouterr().err
