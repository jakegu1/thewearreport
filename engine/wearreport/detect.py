"""Detect people and umbrellas in one BGR frame with YOLOX (ONNX, CPU).

A port of the reference spike (`spike/sweep_detect.py`): letterbox the frame into the
model's 640x640 input, run the network, decode its grid outputs into boxes, keep the
`person` and `umbrella` classes above a confidence threshold, suppress overlapping boxes
per class and map the survivors back to frame pixels, clipped to the frame.

Privacy (AGENTS.md INV-1): the frame and the model input exist only in memory. The
detector keeps no reference to either after `detect` returns, and the onnxruntime session
is opened with profiling off, no optimised-model output and errors-only logging. Error
messages name shapes and dtypes, never pixel values.

onnxruntime's Linux wheels ship Microsoft telemetry: as soon as the library loads it writes
a device ID and an event database under ~/.cache/Microsoft/, a session file and a log in
TMPDIR, and queues uploads to a Microsoft endpoint. The only switch that stops all of it is
ORT_DISABLE_TELEMETRY=1 in the environment before the library loads, so this module sets it
before importing onnxruntime and refuses to import if onnxruntime was loaded without it.

The model files come from `scripts/fetch_model.sh`; a model is opened only if its SHA-256
is one of the pinned digests in `MODEL_SHA256`.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

TELEMETRY_ENV = "ORT_DISABLE_TELEMETRY"

if "onnxruntime" in sys.modules and os.environ.get(TELEMETRY_ENV) != "1":
    raise ImportError(
        "onnxruntime was imported before wearreport.detect, so its telemetry is not disabled"
    )
os.environ[TELEMETRY_ENV] = "1"

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import onnxruntime as ort  # type: ignore[import-untyped]  # noqa: E402  (after the setting)

from wearreport._cv import MAX_IMAGE_PIXELS, cv2  # noqa: E402

ort.disable_telemetry_events()  # the environment variable is what counts; this is a backstop

Label = Literal["person", "umbrella"]
Box = tuple[float, float, float, float]
FloatArray = npt.NDArray[np.float32]

DEFAULT_CONF = 0.35
DEFAULT_NMS = 0.45
INPUT_SIZE = 640  # square model input, in pixels
PAD_VALUE = 114  # letterbox padding, as in YOLOX training
STRIDES = (8, 16, 32)
NUM_CLASSES = 80  # COCO
CLASS_IDS: dict[Label, int] = {"person": 0, "umbrella": 25}
# One row per anchor: cx, cy, w, h, objectness, then one score per class.
OUTPUT_SHAPE = (1, sum((INPUT_SIZE // s) ** 2 for s in STRIDES), 5 + NUM_CLASSES)
# exp() of anything above this overflows float32; clamp the log-sizes first.
MAX_LOG_SIZE = 20.0

# The official YOLOX 0.1.1rc0 release assets; `scripts/fetch_model.sh` pins the same digests.
MODEL_DIR = Path(__file__).resolve().parents[2] / ".models"
MAX_MODEL_BYTES = 256 * 1024 * 1024  # yolox_m is 101 MB
MODEL_SHA256 = {
    "yolox_s.onnx": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    "yolox_m.onnx": "21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f",
}


class DetectorError(RuntimeError):
    """The model cannot be opened, or produced output of the wrong shape or with values
    no working model gives (non-finite numbers, scores outside [0, 1])."""


@dataclass(frozen=True, slots=True)
class Detection:
    label: Label
    score: float
    box: Box  # (x1, y1, x2, y2) in frame pixels, 0 <= x1 < x2 <= width, same for y


class Session(Protocol):
    """What the detector needs from a model: one NCHW float32 input in, outputs out."""

    def run(self, tensor: FloatArray) -> Sequence[object]: ...


def _check_threshold(name: str, value: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if not 0.0 < value <= 1.0:  # also rejects NaN
        raise ValueError(f"{name} must be in (0, 1]")
    return float(value)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_options() -> ort.SessionOptions:
    """Options under which the runtime writes no file: no profile, no optimised model."""
    options = ort.SessionOptions()
    options.enable_profiling = False
    options.optimized_model_filepath = ""
    options.log_severity_level = 3  # errors only
    options.inter_op_num_threads = 1
    return options


def _read_model(model_path: Path) -> bytes:
    try:
        with open(model_path, "rb") as fh:
            data = fh.read(MAX_MODEL_BYTES + 1)
    except OSError as exc:
        raise DetectorError(f"cannot read model {model_path.name}: {exc.strerror}") from None
    if len(data) > MAX_MODEL_BYTES:
        raise DetectorError(f"model {model_path.name} is larger than {MAX_MODEL_BYTES} bytes")
    return data


def open_session(model_path: Path) -> Session:
    """Open a pinned model on the CPU. Raises DetectorError for anything else.

    The bytes that are hashed are the bytes that are loaded, so the file cannot be swapped
    between the check and the load. onnxruntime's default logger, which writes warnings
    to stderr whatever the session options say, is set to errors only before the session
    is created.
    """
    data = _read_model(model_path)
    if hashlib.sha256(data).hexdigest() not in MODEL_SHA256.values():
        raise DetectorError(
            f"model {model_path.name} does not match a pinned SHA-256; run scripts/fetch_model.sh"
        )
    ort.set_default_logger_severity(3)  # errors only
    try:
        session = ort.InferenceSession(
            data, sess_options=session_options(), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # onnxruntime raises its own exception types
        raise DetectorError(
            f"onnxruntime cannot load {model_path.name}: {type(exc).__name__}"
        ) from None
    del data
    inputs = session.get_inputs()
    if len(inputs) != 1 or list(inputs[0].shape) != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        raise DetectorError(f"model {model_path.name} does not take one 1x3x640x640 input")
    return _BoundSession(session, inputs[0].name)


class _BoundSession:
    """An onnxruntime session that feeds its one input by name."""

    def __init__(self, session: ort.InferenceSession, input_name: str) -> None:
        self._session = session
        self._input_name = input_name

    def run(self, tensor: FloatArray) -> Sequence[object]:
        return self._session.run(None, {self._input_name: tensor})  # type: ignore[no-any-return]


# Pre-processing -------------------------------------------------------------------------


def validate_frame(frame: object) -> npt.NDArray[np.uint8]:
    """Return `frame` as a C-contiguous HxWx3 uint8 array, or raise ValueError."""
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame must be a numpy array, not {type(frame).__name__}")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame dtype must be uint8, not {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3 (BGR), not shape {frame.shape}")
    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError(f"frame is empty: shape {frame.shape}")
    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError(f"frame has more than {MAX_IMAGE_PIXELS} pixels: shape {frame.shape}")
    return np.ascontiguousarray(frame)


def letterbox(frame: npt.NDArray[np.uint8]) -> tuple[FloatArray, float]:
    """Resize `frame` to fit INPUT_SIZE, pad bottom and right; return NCHW float32 and scale.

    A frame pixel (x, y) lands at (x * scale, y * scale) in the model input.
    """
    height, width = frame.shape[:2]
    scale = min(INPUT_SIZE / height, INPUT_SIZE / width)
    new_w = min(INPUT_SIZE, max(1, int(width * scale)))
    new_h = min(INPUT_SIZE, max(1, int(height * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor, scale


def unletterbox(boxes: FloatArray, scale: float, width: int, height: int) -> FloatArray:
    """Map xyxy boxes from model-input pixels back to frame pixels, clipped to the frame."""
    out = boxes / np.float32(scale)
    np.clip(out[:, 0::2], 0, width, out=out[:, 0::2])
    np.clip(out[:, 1::2], 0, height, out=out[:, 1::2])
    return out


# Post-processing ------------------------------------------------------------------------


def _grids() -> tuple[FloatArray, FloatArray]:
    grids, strides = [], []
    for stride in STRIDES:
        side = INPUT_SIZE // stride
        xv, yv = np.meshgrid(np.arange(side), np.arange(side))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((side * side, 1), stride))
    return (
        np.concatenate(grids).astype(np.float32),
        np.concatenate(strides).astype(np.float32),
    )


GRIDS, GRID_STRIDES = _grids()


def decode(raw: FloatArray) -> FloatArray:
    """Turn raw grid outputs (N x 85) into xyxy boxes in model-input pixels (N x 4)."""
    centres = (raw[:, :2] + GRIDS) * GRID_STRIDES
    log_sizes = np.clip(raw[:, 2:4], -MAX_LOG_SIZE, MAX_LOG_SIZE)
    half = np.exp(log_sizes) * GRID_STRIDES / 2
    return np.concatenate((centres - half, centres + half), axis=1).astype(np.float32)


def nms(boxes: FloatArray, scores: FloatArray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression over xyxy boxes; indices of kept boxes, best first.

    A box is dropped when its IoU with a kept, higher-scoring box exceeds `threshold`.
    """
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    keep: list[int] = []
    while order.size:
        best, rest = int(order[0]), order[1:]
        keep.append(best)
        w = np.maximum(np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]), 0)
        h = np.maximum(np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]), 0)
        inter = w * h
        union = areas[best] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= threshold]
    return keep


def postprocess(
    raw: FloatArray,
    scale: float,
    width: int,
    height: int,
    conf: float,
    nms_threshold: float,
    *,
    refuse_all_dropped: bool = False,
) -> list[Detection]:
    """Filter, suppress and map raw outputs (N x 85) to detections in a width x height frame.

    A row whose box or score is not finite, including a box whose decoded coordinates
    overflow float32, is dropped, and so is a box with no area once clipped to the frame.
    `Detector.detect` has already refused output that holds NaN, infinity or scores outside
    [0, 1], so only the overflow case reaches here from it.

    With `refuse_all_dropped`, raise DetectorError when at least one person or umbrella row
    reaches `conf` and every such row is dropped: a model whose confident boxes all
    overflow or fall outside the frame is broken, and must not read as an empty frame.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        class_scores = raw[:, 4:5] * raw[:, 5:]
        best_class = class_scores.argmax(axis=1)
        best_score = class_scores.max(axis=1)
        boxes = unletterbox(decode(raw), scale, width, height)
    finite = np.isfinite(boxes).all(axis=1) & np.isfinite(best_score)
    has_area = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    # A box counts for the class with its highest score, as in the spike.
    confident = np.isin(best_class, list(CLASS_IDS.values())) & (best_score >= conf)
    if refuse_all_dropped and confident.any() and not (confident & finite & has_area).any():
        raise DetectorError(
            f"model output has {int(confident.sum())} confident boxes and none inside the frame"
        )
    detections: list[Detection] = []
    for label, class_id in CLASS_IDS.items():
        mask = finite & has_area & confident & (best_class == class_id)
        if not mask.any():
            continue
        class_boxes, class_conf = boxes[mask], best_score[mask]
        for i in nms(class_boxes, class_conf, nms_threshold):
            x1, y1, x2, y2 = (float(v) for v in class_boxes[i])
            detections.append(Detection(label, float(class_conf[i]), (x1, y1, x2, y2)))
    detections.sort(key=lambda d: -d.score)
    return detections


# Detector -------------------------------------------------------------------------------


class Detector:
    """A YOLOX ONNX model that finds people and umbrellas in BGR frames."""

    def __init__(
        self, model_path: str | Path, *, conf: float = DEFAULT_CONF, nms: float = DEFAULT_NMS
    ) -> None:
        self.conf = _check_threshold("conf", conf)
        self.nms = _check_threshold("nms", nms)
        self.model_name = Path(model_path).stem  # e.g. "yolox_s"
        self._session = open_session(Path(model_path))

    @classmethod
    def from_session(
        cls,
        session: Session,
        *,
        name: str,
        conf: float = DEFAULT_CONF,
        nms: float = DEFAULT_NMS,
    ) -> Detector:
        """For tests only: a detector around an already open session, such as a stand-in
        model. It skips `open_session`, so no SHA-256 pin is checked; the engine always
        opens models through `Detector(model_path)`."""
        detector = cls.__new__(cls)
        detector.conf = _check_threshold("conf", conf)
        detector.nms = _check_threshold("nms", nms)
        detector.model_name = name
        detector._session = session
        return detector

    def detect(self, frame: npt.NDArray[np.uint8]) -> list[Detection]:
        """Detections in `frame` (HxWx3 uint8, BGR), highest score first.

        Raises ValueError for a frame of the wrong type, dtype, shape or size, and
        DetectorError if the model returns output of an unexpected shape, any NaN or
        infinity, or an objectness or class score outside [0, 1]: such output is a broken
        model, not a frame without people. A box whose decoded coordinates overflow (a
        finite but huge regression output), or that has no area inside the frame, is
        dropped; but if every person or umbrella box that reaches `conf` is dropped that
        way, that is a broken model too, and raises DetectorError.
        """
        pixels = validate_frame(frame)
        height, width = pixels.shape[:2]
        tensor, scale = letterbox(pixels)
        del pixels
        raw = self._infer(tensor)
        del tensor
        return postprocess(raw, scale, width, height, self.conf, self.nms, refuse_all_dropped=True)

    def _infer(self, tensor: FloatArray) -> FloatArray:
        try:
            outputs = self._session.run(tensor)
        except Exception as exc:  # onnxruntime raises its own exception types
            raise DetectorError(f"inference failed: {type(exc).__name__}") from None
        if not outputs:
            raise DetectorError("model returned no output")
        out = outputs[0]
        if not isinstance(out, np.ndarray) or out.shape != OUTPUT_SHAPE:
            shape = getattr(out, "shape", None)
            raise DetectorError(f"model output has shape {shape}, expected {OUTPUT_SHAPE}")
        raw = np.asarray(out[0], dtype=np.float32)
        if not np.isfinite(raw).all():
            raise DetectorError("model output has NaN or infinite values")
        scores = raw[:, 4:]  # objectness, then one score per class
        if (scores < 0).any() or (scores > 1).any():
            raise DetectorError("model output has scores outside [0, 1]")
        return raw


def model_path(name: str) -> Path:
    """Where `scripts/fetch_model.sh` puts the model file `name` (e.g. "yolox_s.onnx")."""
    return MODEL_DIR / name
