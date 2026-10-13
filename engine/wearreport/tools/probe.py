"""Local prequential ridge scores, with cumulative moments and accepted-last commits.

The encoder, candidate vectors, and fitted heads stay in memory for one pass. Only
validated cumulative moments, human labels, and bounded run/lock metadata persist.
The caller supplies crops before asking questions and publishes human labels before
calling finish. An interrupted source never automatically starts another history.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO, cast

import numpy as np
import numpy.typing as npt

from wearreport import detect
from wearreport._cv import MAX_IMAGE_PIXELS, cv2

if sys.platform == "win32":
    import ctypes
    import msvcrt

    _move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
    _move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    _move_file.restype = ctypes.c_int
else:
    import fcntl

DIM = 768
LAMBDAS = (1, 10, 100, 1000)
MODEL_NAME = "siglip2-base-patch16-224-int8.onnx"
MODEL_SIZE = 94_553_333
MODEL_SHA256 = "0dd31785a2713f1113ef2272472165c69d580473dae38d7b47568ac587795e70"
ENCODER = "siglip2-base-patch16-224-int8@" + MODEL_SHA256[:12]
MODEL_PATH = detect.MODEL_DIR / MODEL_NAME
MAX_STATE_BYTES = 64 << 20
MAX_LABEL_BYTES = 1 << 20
MAX_CONTROL_BYTES = 1024
MAX_COUNT = 10**9
MAX_VALUE = 1e21
MAX_FEATURE = 1e6
SOURCES = frozenset({"london", "calgary", "austin"})
REASONS = frozenset(
    {
        "invalid_state",
        "invalid_history",
        "source_incomplete",
        "unknown_identity",
        "source_busy",
        "encoder_mismatch",
        "model_missing",
        "model_checksum",
        "encoder_failed",
        "unsafe_increment",
        "persistence_failed",
        "unfinished_session",
        "collection_complete",
    }
)
LOCK_TEXT = '{"version":1,"kind":"probe_source_lock"}\n'
Array = npt.NDArray[np.float64]
Frame = npt.NDArray[np.uint8]
Score = list[list[float]] | None


class ProbeError(ValueError):
    """A fixed public reason code, never an external exception or input value."""

    def __init__(self, reason: str) -> None:
        self.reason = reason if reason in REASONS else "invalid_state"
        super().__init__(self.reason)


class Encoder(Protocol):
    def embed(self, image: Frame) -> Array: ...


def preprocess(image: Frame) -> npt.NDArray[np.float32]:
    """Pad an RGB crop with black to square, then the pinned 224px processor."""
    try:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ProbeError("encoder_failed")
        height, width = image.shape[:2]
        side = max(height, width)
        if min(height, width) < 1 or side * side > MAX_IMAGE_PIXELS * 16:
            raise ProbeError("encoder_failed")
        square = np.zeros((side, side, 3), dtype=np.uint8)
        top, left = (side - height) // 2, (side - width) // 2
        square[top : top + height, left : left + width] = image
        resized = cv2.resize(square, (224, 224), interpolation=cv2.INTER_LINEAR)
        values = np.asarray(resized, dtype=np.float32) / np.float32(255)
        values = (values - np.float32(0.5)) / np.float32(0.5)
        return np.ascontiguousarray(values.transpose(2, 0, 1)[None])
    except ProbeError:
        raise
    except Exception:
        raise ProbeError("encoder_failed") from None


class SiglipEncoder:
    """The pinned int8 conversion, loaded from the same bytes that are verified."""

    input_name = "pixel_values"
    output_name = "pooler_output"

    def __init__(self, path: Path = MODEL_PATH) -> None:
        # detect has already disabled telemetry before any runtime import.
        import onnxruntime as ort  # type: ignore[import-untyped]

        try:
            with open(path, "rb") as stream:
                data = stream.read(MODEL_SIZE + 1)
        except OSError:
            raise ProbeError("model_missing") from None
        if len(data) != MODEL_SIZE or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise ProbeError("model_checksum")
        try:
            # detect sets ORT_DISABLE_TELEMETRY before importing the runtime. Its
            # options disable profiling and optimized-model files as well.
            ort.set_default_logger_severity(3)
            self._session = ort.InferenceSession(
                data, sess_options=detect.session_options(), providers=["CPUExecutionProvider"]
            )
            self.providers: list[str] = list(self._session.get_providers())
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if (
                self.providers != ["CPUExecutionProvider"]
                or len(inputs) != 1
                or inputs[0].name != self.input_name
                or inputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 4
                or any(
                    isinstance(actual, int) and actual != expected
                    for actual, expected in zip(inputs[0].shape, [1, 3, 224, 224], strict=True)
                )
                or not any(o.name == self.output_name for o in outputs)
            ):
                raise ProbeError("encoder_failed")
        except ProbeError:
            raise
        except Exception:
            raise ProbeError("encoder_failed") from None

    def embed(self, image: Frame) -> Array:
        try:
            # The engine's crops are BGR. Preprocessing itself takes RGB.
            values = preprocess(image[:, :, ::-1])
            [output] = self._session.run([self.output_name], {self.input_name: values})
            result = np.asarray(output, dtype=np.float64)
            if result.shape != (1, DIM) or not np.isfinite(result).all():
                raise ProbeError("encoder_failed")
            return _feature(result[0])
        except ProbeError:
            raise
        except Exception:
            raise ProbeError("encoder_failed") from None


def open_encoder() -> Encoder:
    return SiglipEncoder()


def _feature(value: object) -> Array:
    if not isinstance(value, np.ndarray) or value.shape != (DIM,):
        raise ProbeError("encoder_failed")
    if value.dtype.kind not in "fiu":
        raise ProbeError("encoder_failed")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(np.abs(result) > MAX_FEATURE):
        raise ProbeError("encoder_failed")
    return result.copy()


def _object(value: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ProbeError("invalid_state")
    return cast(dict[str, object], value)


def _integer(value: object, maximum: int = MAX_COUNT) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ProbeError("invalid_state")
    return value


def _uuid(value: object) -> str:
    if not isinstance(value, str) or len(value) != 36:
        raise ProbeError("unknown_identity")
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError
    except ValueError:
        raise ProbeError("unknown_identity") from None
    return value


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProbeError("invalid_state")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ProbeError("invalid_state")


def _decode(raw: bytes, limit: int) -> object:
    try:
        if not isinstance(raw, bytes) or len(raw) > limit:
            raise ProbeError("invalid_state")
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except ProbeError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise ProbeError("invalid_state") from None


def _read(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        return stream.read(limit + 1)


def _array(value: object, matrix: bool = False) -> Array:
    if not isinstance(value, list) or len(value) != DIM:
        raise ProbeError("invalid_state")
    rows = value if matrix else [value]
    for row in rows:
        if not isinstance(row, list) or len(row) != DIM:
            raise ProbeError("invalid_state")
        if any(type(v) not in (int, float) for v in row):
            raise ProbeError("invalid_state")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(np.abs(result) > MAX_VALUE):
        raise ProbeError("invalid_state")
    return result


@dataclass(frozen=True)
class Cohort:
    count: int
    total: Array
    cross: Array

    @classmethod
    def empty(cls) -> Cohort:
        return cls(0, np.zeros(DIM), np.zeros((DIM, DIM)))

    @classmethod
    def from_rows(cls, rows: Array) -> Cohort:
        return cls(len(rows), rows.sum(axis=0), rows.T @ rows)

    def validate(self) -> None:
        _integer(self.count)
        if (
            self.total.shape != (DIM,)
            or self.cross.shape != (DIM, DIM)
            or not np.isfinite(self.total).all()
            or not np.isfinite(self.cross).all()
            or np.any(np.abs(self.total) > MAX_VALUE)
            or np.any(np.abs(self.cross) > MAX_VALUE)
        ):
            raise ProbeError("invalid_state")
        if self.count == 0:
            if np.any(self.total) or np.any(self.cross):
                raise ProbeError("invalid_state")
            return
        if self.count < 2:
            raise ProbeError("unsafe_increment")
        diagonal = np.diag(self.cross)
        tolerance = 1e-8 * max(1.0, float(np.max(np.abs(self.cross))))
        if (
            np.any(diagonal < 0)
            or not np.allclose(self.cross, self.cross.T, atol=tolerance)
            or np.any(self.total * self.total > self.count * diagonal + tolerance)
        ):
            raise ProbeError("invalid_state")
        bounds = np.sqrt(diagonal[:, None] * diagonal[None, :])
        if np.any(np.abs(self.cross) > bounds + tolerance):
            raise ProbeError("invalid_state")
        active = diagonal > 0
        centered = self.cross[np.ix_(active, active)]
        centered = centered - np.outer(self.total[active], self.total[active]) / self.count
        if len(centered) and np.linalg.eigvalsh(centered).min() < -tolerance:
            raise ProbeError("invalid_state")

    def plus(self, other: Cohort) -> Cohort:
        return Cohort(self.count + other.count, self.total + other.total, self.cross + other.cross)

    def document(self) -> dict[str, object]:
        return {"count": self.count, "sum": self.total.tolist(), "cross": self.cross.tolist()}


@dataclass(frozen=True)
class State:
    run_id: str
    source: str
    encoder: str
    accepted_sessions: int
    cohorts: Mapping[str, Cohort]

    def validate(self) -> None:
        _uuid(self.run_id)
        if self.source not in SOURCES or self.encoder != ENCODER or set(self.cohorts) != set("ynu"):
            raise ProbeError("invalid_state")
        _integer(self.accepted_sessions, 12)
        for group in self.cohorts.values():
            group.validate()
        total, judgeable = self.counts()[0], self.counts()[1]
        _integer(total)
        if self.accepted_sessions == 0 and total == 0:
            return
        if total < 2 or judgeable < 2 or self.accepted_sessions == 0:
            raise ProbeError("unsafe_increment")

    def counts(self) -> tuple[int, int, int]:
        y, n, u = (self.cohorts[name].count for name in "ynu")
        return y + n + u, y + n, y

    def document(self) -> dict[str, object]:
        self.validate()
        return {
            "version": 1,
            "run_id": self.run_id,
            "source": self.source,
            "encoder": self.encoder,
            "accepted_sessions": self.accepted_sessions,
            "cohorts": {key: self.cohorts[key].document() for key in "ynu"},
        }


def parse_state(raw: bytes) -> State:
    """Bounded hostile-input parser. All malformed input has a typed fixed reason."""
    try:
        value = _object(
            _decode(raw, MAX_STATE_BYTES),
            {"version", "run_id", "source", "encoder", "accepted_sessions", "cohorts"},
        )
        if _integer(value["version"], 1) != 1:
            raise ProbeError("invalid_state")
        source, encoder = value["source"], value["encoder"]
        if not isinstance(source, str) or not isinstance(encoder, str):
            raise ProbeError("invalid_state")
        groups = _object(value["cohorts"], set("ynu"))
        cohorts = {}
        for name in "ynu":
            group = _object(groups[name], {"count", "sum", "cross"})
            cohorts[name] = Cohort(
                _integer(group["count"]), _array(group["sum"]), _array(group["cross"], True)
            )
        state = State(
            _uuid(value["run_id"]),
            source,
            encoder,
            _integer(value["accepted_sessions"], 12),
            cohorts,
        )
        state.validate()
        return state
    except ProbeError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise ProbeError("invalid_state") from None


def _head(group: Cohort, target_count: int, target_total: Array, current: Array) -> list[Array]:
    mean_y = target_count / group.count
    if target_count in (0, group.count):
        return [np.full(len(current), mean_y) for _ in LAMBDAS]
    mean = group.total / group.count
    variance = np.maximum(np.diag(group.cross) / group.count - mean * mean, 0)
    active = variance > 1e-20
    scale = np.sqrt(variance[active])
    centered = group.cross[np.ix_(active, active)] - group.count * np.outer(
        mean[active], mean[active]
    )
    standardized = centered / scale[:, None] / scale[None, :]
    xy = (target_total[active] - group.total[active] * mean_y) / scale
    query = (current[:, active] - mean[active]) / scale
    identity = np.eye(len(scale))
    return [
        np.round(
            np.clip(query @ np.linalg.solve(standardized + penalty * identity, xy) + mean_y, 0, 1),
            3,
        )
        for penalty in LAMBDAS
    ]


def score_features(state: State, features: Array) -> list[Score]:
    """Score all candidates using only the prior state, before any current answers."""
    if features.ndim != 2 or features.shape[1] != DIM or not np.isfinite(features).all():
        raise ProbeError("encoder_failed")
    if state.accepted_sessions == 0:
        return [None for _ in features]
    y, n, u = (state.cohorts[name] for name in "ynu")
    judgeable = y.plus(n)
    all_kept = judgeable.plus(u)
    judges = _head(all_kept, judgeable.count, judgeable.total, features)
    coats = _head(judgeable, y.count, y.total, features)
    return [
        [[float(j[i]), float(c[i])] for j, c in zip(judges, coats, strict=True)]
        for i in range(len(features))
    ]


def validate_label_metadata(record: Mapping[str, object]) -> dict[str, object] | None:
    """Validate optional run/score objects; legacy human labels need neither."""
    if "probe_run" not in record:
        if "probe" in record:
            raise ProbeError("invalid_history")
        return None
    try:
        run = _object(record["probe_run"], {"version", "run_id", "session", "state", "reason"})
        if _integer(run["version"], 1) != 1:
            raise ProbeError("invalid_history")
        identity = run["run_id"]
        if identity is not None:
            _uuid(identity)
        session, status, reason = run["session"], run["state"], run["reason"]
        if session is not None and _integer(session, 12) < 1:
            raise ProbeError("invalid_history")
        if identity is None and session is not None:
            raise ProbeError("invalid_history")
        if status not in ("accepted", "incomplete", "stopped"):
            raise ProbeError("invalid_history")
        if status == "accepted":
            if identity is None or session is None or reason is not None or "probe" not in record:
                raise ProbeError("invalid_history")
            block = _object(record["probe"], {"encoder", "lambdas", "trained_sessions", "scores"})
            grid = block["lambdas"]
            if (
                block["encoder"] != ENCODER
                or not isinstance(grid, list)
                or any(type(v) is not int for v in grid)
                or grid != list(LAMBDAS)
                or _integer(block["trained_sessions"], 11) != _integer(session, 12) - 1
            ):
                raise ProbeError("invalid_history")
            scores, crops = block["scores"], record["crops"]
            if (
                not isinstance(scores, list)
                or not isinstance(crops, list)
                or len(scores) != len(crops)
            ):
                raise ProbeError("invalid_history")
            for score in scores:
                if session == 1:
                    if score is not None:
                        raise ProbeError("invalid_history")
                    continue
                if not isinstance(score, list) or len(score) != len(LAMBDAS):
                    raise ProbeError("invalid_history")
                for pair in score:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ProbeError("invalid_history")
                    if any(
                        type(v) not in (float, int)
                        or not math.isfinite(v)
                        or not 0 <= v <= 1
                        or round(v, 3) != v
                        for v in pair
                    ):
                        raise ProbeError("invalid_history")
        else:
            if "probe" in record or not isinstance(reason, str) or reason not in REASONS:
                raise ProbeError("invalid_history")
            if status == "stopped" and session is not None:
                raise ProbeError("invalid_history")
            if reason == "collection_complete" and (identity is None or status != "stopped"):
                raise ProbeError("invalid_history")
        return run
    except ProbeError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise ProbeError("invalid_history") from None


def _control(raw: bytes) -> dict[str, object]:
    value = _object(
        _decode(raw, MAX_CONTROL_BYTES),
        {"version", "run_id", "source", "encoder", "state", "accepted_sessions", "reason"},
    )
    if _integer(value["version"], 1) != 1:
        raise ProbeError("invalid_state")
    _uuid(value["run_id"])
    _integer(value["accepted_sessions"], 12)
    if (
        not isinstance(value["source"], str)
        or value["source"] not in SOURCES
        or not isinstance(value["encoder"], str)
        or len(value["encoder"]) > 128
        or value["state"] not in ("active", "pending", "incomplete", "sealed")
    ):
        raise ProbeError("invalid_state")
    if value["reason"] is not None and (
        not isinstance(value["reason"], str) or value["reason"] not in REASONS
    ):
        raise ProbeError("invalid_state")
    if value["state"] in ("active", "sealed") and value["reason"] is not None:
        raise ProbeError("invalid_state")
    return value


def _sync_directory(path: Path) -> None:
    if sys.platform == "win32":
        # File fsync is native FlushFileBuffers on Windows; opening a directory
        # through os.open is unsupported there.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_label(path: Path) -> None:
    with open(path, "r+", encoding="utf-8") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)
    _sync_directory(path.parent.parent)


def _make_directory(path: Path) -> None:
    """Persist new directory entries as well as the later files on POSIX."""
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _sync_directory(directory.parent)


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    """Durable text staging, atomic replacement, and no surviving normal-path parts."""
    text = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"
    limit = (
        MAX_STATE_BYTES
        if "cohorts" in value
        else MAX_LABEL_BYTES
        if "crops" in value
        else MAX_CONTROL_BYTES
    )
    if len(text.encode("utf-8")) > limit:
        raise ProbeError("persistence_failed")
    _make_directory(path.parent)
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if sys.platform == "win32":
            # REPLACE_EXISTING | WRITE_THROUGH. FlushFileBuffers above and this
            # native move give the Windows commit the same durable ordering.
            if not _move_file(str(temporary), str(path), 0x1 | 0x8):
                raise ProbeError("persistence_failed")
        else:
            os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


class SourceLock:
    """A stable metadata file with a portable, nonblocking exclusive kernel lock."""

    def __init__(self, directory: Path, source: str) -> None:
        self.path = directory.parent / (directory.name + ".locks") / (source + ".json")
        self.stream: TextIO | None = None

    def acquire(self) -> None:
        _make_directory(self.path.parent)
        if self.path.is_symlink():
            raise ProbeError("invalid_state")
        try:
            stream = open(self.path, "x+", encoding="utf-8")  # noqa: SIM115
        except FileExistsError:
            stream = open(self.path, "r+", encoding="utf-8")  # noqa: SIM115
        else:
            stream.write(LOCK_TEXT)
            stream.flush()
            os.fsync(stream.fileno())
            _sync_directory(self.path.parent)
        self.stream = stream
        try:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.close()
            raise ProbeError("source_busy") from None
        stream.seek(0)
        if stream.read(129) != LOCK_TEXT:
            self.close()
            raise ProbeError("invalid_state")

    def close(self) -> None:
        if self.stream is not None:
            # Closing the descriptor releases the kernel lock even on failure.
            self.stream.close()
            self.stream = None


class ProbePass:
    """One locked, registered source pass. Failed input is never replaced or reset."""

    def __init__(self, directory: Path, out_dir: Path, source: str) -> None:
        self._path_failed = False
        try:
            self.directory = Path(os.path.normcase(str(directory.resolve())))
        except (OSError, RuntimeError):
            self.directory = directory.absolute()
            self._path_failed = True
        self.out_dir = out_dir.resolve()
        self.source = source
        self.state_path = self.directory / (source + ".json")
        self.control_path = self.out_dir / "probe-runs" / (source + ".json")
        self.lock = SourceLock(self.directory, source)
        self.state: State | None = None
        self.run_id: str | None = None
        self.session: int | None = None
        self.reason: str | None = None
        self.enabled = False
        self.armed = False
        self.arm_failed = False
        self._notified = False
        self._features: Array | None = None
        self._scores: list[Score] = []

    def __enter__(self) -> ProbePass:
        try:
            if (
                self._path_failed
                or self.source not in SOURCES
                or self.control_path.is_relative_to(self.directory)
                or (self.directory.exists() and not self.directory.is_dir())
            ):
                raise ProbeError("invalid_state")
            self.lock.acquire()
            self._admit()
        except ProbeError as exc:
            self._stop(exc.reason)
        except Exception:
            self._stop("persistence_failed")
        return self

    def __exit__(self, *args: object) -> None:
        self._discard()
        self.lock.close()

    def _recover_identity(self) -> str | None:
        """Read-only identity recovery, never admission or a count reconstruction."""
        identities: set[str] = set()
        try:
            control = _control(_read(self.control_path, MAX_CONTROL_BYTES))
            if control["source"] == self.source:
                identities.add(_uuid(control["run_id"]))
        except (OSError, ProbeError):
            pass
        try:
            for _path, record in self._records():
                run = validate_label_metadata(record)
                if run is not None and run["run_id"] is not None:
                    identities.add(_uuid(run["run_id"]))
        except (OSError, ProbeError):
            pass
        return next(iter(identities)) if len(identities) == 1 else None

    def _stop(self, reason: str) -> None:
        self.reason = reason
        self.enabled = False
        self._discard()
        if not self.armed and self.source in SOURCES:
            self.run_id = self._recover_identity()
            if self.run_id is None:
                self.session = None

    def take_reason(self) -> str | None:
        if self.reason is None or self._notified:
            return None
        self._notified = True
        return self.reason

    def metadata(self) -> dict[str, object]:
        incomplete = self.armed or self.arm_failed
        return {
            "version": 1,
            "run_id": self.run_id,
            "session": self.session if incomplete else None,
            "state": "incomplete" if incomplete else "stopped",
            "reason": "unfinished_session" if self.armed else self.reason,
        }

    def _control_document(self, state: str, count: int) -> dict[str, object]:
        return {
            "version": 1,
            "run_id": self.run_id,
            "source": self.source,
            "encoder": ENCODER,
            "state": state,
            "accepted_sessions": count,
            "reason": "unfinished_session" if state == "pending" else None,
        }

    def _records(self) -> list[tuple[Path, dict[str, object]]]:
        folder = self.out_dir / "attributes"
        if not folder.exists():
            return []
        result = []
        paths = list(folder.glob("*.json"))
        if len(paths) > 100_000:
            raise ProbeError("invalid_history")
        for path in paths:
            value = _decode(_read(path, MAX_LABEL_BYTES), MAX_LABEL_BYTES)
            if not isinstance(value, dict):
                raise ProbeError("invalid_history")
            source = value.get("source", "london")
            if not isinstance(source, str) or source not in SOURCES:
                raise ProbeError("invalid_history")
            if source == self.source:
                result.append((path, cast(dict[str, object], value)))
        return result

    def _history(self, state: State | None, exclude: Path | None = None) -> None:
        accepted: dict[int, tuple[int, int, int]] = {}
        complete_stops = False
        for path, record in self._records():
            if exclude is not None and path == exclude:
                continue  # only this pass's already durable human label
            run = validate_label_metadata(record)
            if run is None:
                continue
            if run["run_id"] is None or state is None or run["run_id"] != state.run_id:
                raise ProbeError("source_incomplete")
            if run["state"] != "accepted":
                if run["reason"] == "collection_complete" and state.accepted_sessions == 12:
                    complete_stops = True
                    continue
                raise ProbeError("source_incomplete")
            session = _integer(run["session"], 12)
            if session in accepted:
                raise ProbeError("invalid_history")
            crops = record.get("crops")
            if not isinstance(crops, list):
                raise ProbeError("invalid_history")
            counts = {name: 0 for name in "ynu"}
            for row in crops:
                if (
                    not isinstance(row, list)
                    or len(row) != 3
                    or type(row[0]) is not int
                    or not isinstance(row[1], str)
                    or re.fullmatch(r"[ynu]{3}", row[1]) is None
                    or (
                        row[2] is not None
                        and (
                            not isinstance(row[2], str) or re.fullmatch(r"[ynu]{3}", row[2]) is None
                        )
                    )
                ):
                    raise ProbeError("invalid_history")
                counts[row[1][0]] += 1
            if (
                len(crops) < 2
                or counts["y"] + counts["n"] < 2
                or any(count == 1 for count in counts.values())
            ):
                raise ProbeError("invalid_history")
            if record.get("crops_shown") != len(crops) + cast(
                int, record.get("crops_rejected", -1)
            ):
                raise ProbeError("invalid_history")
            accepted[session] = len(crops), counts["y"] + counts["n"], counts["y"]
        expected = 0 if state is None else state.accepted_sessions
        if set(accepted) != set(range(1, expected + 1)) or (complete_stops and expected != 12):
            raise ProbeError("invalid_history")
        if state is not None:
            totals = tuple(sum(row[k] for row in accepted.values()) for k in range(3))
            if totals != state.counts():
                raise ProbeError("invalid_history")

    def _admit(self) -> None:
        if self.control_path.is_symlink() or self.state_path.is_symlink():
            raise ProbeError("invalid_state")
        if self.control_path.exists():
            control = _control(_read(self.control_path, MAX_CONTROL_BYTES))
            self.run_id = _uuid(control["run_id"])
            if control["source"] != self.source or control["encoder"] != ENCODER:
                raise ProbeError("encoder_mismatch")
            if control["state"] in ("pending", "incomplete"):
                raise ProbeError("source_incomplete")
            count = _integer(control["accepted_sessions"], 12)
            if count == 0 and not self.state_path.exists():
                state = State(
                    self.run_id, self.source, ENCODER, 0, {n: Cohort.empty() for n in "ynu"}
                )
            else:
                state = parse_state(_read(self.state_path, MAX_STATE_BYTES))
            if (
                state.run_id != self.run_id
                or state.source != self.source
                or state.encoder != ENCODER
                or state.accepted_sessions != count
            ):
                raise ProbeError("invalid_state")
            self._history(state)
            if control["state"] == "sealed" or count == 12:
                if control["state"] != "sealed" or count != 12:
                    raise ProbeError("invalid_state")
                raise ProbeError("collection_complete")
        else:
            if self.state_path.exists():
                raise ProbeError("unknown_identity")
            self._history(None)
            identity = str(uuid.uuid4())
            state = State(identity, self.source, ENCODER, 0, {n: Cohort.empty() for n in "ynu"})
        self.state = state
        # A fresh identity becomes registered only if arming is durable. If arming
        # fails, recover an existing unambiguous identity; never mint a fallback.
        previous_id = self.run_id
        self.run_id = state.run_id
        try:
            atomic_json(
                self.control_path, self._control_document("pending", state.accepted_sessions)
            )
        except Exception:
            self.run_id = previous_id
            self.arm_failed = True
            self.session = state.accepted_sessions + 1 if previous_id is not None else None
            raise ProbeError("persistence_failed") from None
        self.session = state.accepted_sessions + 1
        self.armed = True
        self.enabled = True

    def prepare(self, images: Sequence[Frame]) -> None:
        """Compute all candidates' predictions before the caller asks human questions."""
        if not self.enabled or self.state is None:
            return
        try:
            encoder = open_encoder()
            self._features = np.stack([_feature(encoder.embed(image)) for image in images])
            self._scores = score_features(self.state, self._features)
        except ProbeError as exc:
            self._stop(exc.reason)
        except Exception:
            self._stop("encoder_failed")

    def _discard(self) -> None:
        self._features = None
        self._scores = []

    def select(self, answers: Sequence[str | None]) -> None:
        """Drop rejected vectors and predictions immediately after human answers."""
        if not self.enabled or self._features is None:
            return
        if len(answers) != len(self._features):
            self._stop("encoder_failed")
            return
        kept = [i for i, answer in enumerate(answers) if answer is not None]
        self._features = self._features[kept]
        self._scores = [self._scores[i] for i in kept]

    def finish(
        self,
        path: Path,
        record: dict[str, object],
        answers: Sequence[str | None],
        order: Sequence[int],
    ) -> None:
        """Human-label durability, admissible increment/candidate, then acceptance last."""
        path = path.resolve()
        try:
            sync_label(path)
            if not self.enabled or self.state is None or self._features is None:
                return
            if len(answers) != len(self._features):
                raise ProbeError("encoder_failed")
            kept = [i for i, answer in enumerate(answers) if answer is not None]
            features = self._features[kept]
            scores = [self._scores[i] for i in kept]
            self._discard()  # rejected candidates, too; only kept local rows remain
            labels = [cast(str, answers[i])[0] for i in kept]
            if set(order) != set(range(len(kept))) or len(order) != len(kept):
                raise ProbeError("invalid_history")
            groups = {
                name: Cohort.from_rows(
                    features[np.array([a == name for a in labels], dtype=np.bool_)]
                )
                for name in "ynu"
            }
            increment = State(self.state.run_id, self.source, ENCODER, 1, groups)
            self.state.validate()
            increment.validate()
            candidate = State(
                self.state.run_id,
                self.source,
                ENCODER,
                self.state.accepted_sessions + 1,
                {name: self.state.cohorts[name].plus(groups[name]) for name in "ynu"},
            )
            candidate.validate()
            self._history(self.state, exclude=path)
            # Verify the current ordinary label is still precisely the one that
            # was made durable; no accepted-only history reconciliation occurs.
            if _decode(_read(path, MAX_LABEL_BYTES), MAX_LABEL_BYTES) != record:
                raise ProbeError("invalid_history")
            final: dict[str, object] = {
                **record,
                "probe_run": {
                    "version": 1,
                    "run_id": self.run_id,
                    "session": self.session,
                    "state": "accepted",
                    "reason": None,
                },
                "probe": {
                    "encoder": ENCODER,
                    "lambdas": list(LAMBDAS),
                    "trained_sessions": self.state.accepted_sessions,
                    "scores": [scores[i] for i in order],
                },
            }
            validate_label_metadata(final)
            if len(json.dumps(final, ensure_ascii=True).encode("utf-8")) > MAX_LABEL_BYTES:
                raise ProbeError("persistence_failed")
            # No per-crop serialization precedes all three privacy validations.
            atomic_json(self.state_path, candidate.document())
            terminal = "sealed" if candidate.accepted_sessions == 12 else "active"
            atomic_json(
                self.control_path, self._control_document(terminal, candidate.accepted_sessions)
            )
            atomic_json(path, final)  # acceptance authority, LAST; close only unlocks
            record.update(final)
        except ProbeError as exc:
            self._stop(exc.reason)
        except Exception:
            self._stop("persistence_failed")
        finally:
            self._discard()
