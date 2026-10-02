from __future__ import annotations

import gc
import hashlib
import math
import os
import subprocess
import sys
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wearreport import detect
from wearreport._cv import MAX_IMAGE_PIXELS, cv2

REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "detect"
N_ANCHORS = detect.OUTPUT_SHAPE[1]
ROW = detect.OUTPUT_SHAPE[2]
PERSON, UMBRELLA, CAR = 0, 25, 2


def _model(name: str = "yolox_s.onnx") -> Path:
    path = detect.model_path(name)
    if not path.is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run scripts/fetch_model.sh")
    return path


def anchor(stride: int, gx: int, gy: int) -> int:
    """Index of the anchor at grid cell (gx, gy) of the given stride."""
    offset = 0
    for s in detect.STRIDES:
        side = detect.INPUT_SIZE // s
        if s == stride:
            return offset + gy * side + gx
        offset += side * side
    raise ValueError(stride)


def raw_output(*rows: tuple[int, int, float, tuple[float, float, float, float]]) -> np.ndarray:
    """A raw N x 85 output; each row is (anchor, class, score, (dx, dy, log_w, log_h))."""
    raw = np.zeros((N_ANCHORS, ROW), dtype=np.float32)
    for index, cls, score, regression in rows:
        raw[index, :4] = regression
        raw[index, 4] = 1.0  # objectness
        raw[index, 5 + cls] = score
    return raw


class StubSession:
    """A stand-in model that returns a fixed raw output and records its inputs' shapes."""

    def __init__(self, raw: np.ndarray | None = None, outputs: Sequence[object] | None = None):
        self.raw = raw if raw is not None else np.zeros((N_ANCHORS, ROW), dtype=np.float32)
        self.outputs = outputs
        self.shapes: list[tuple[int, ...]] = []

    def run(self, tensor: np.ndarray) -> Sequence[object]:
        self.shapes.append(tensor.shape)
        assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
        return self.outputs if self.outputs is not None else [self.raw[None].copy()]


def stub_detector(raw: np.ndarray | None = None, **kwargs: Any) -> detect.Detector:
    return detect.Detector.from_session(StubSession(raw), name="stub", **kwargs)


def big_box_everywhere() -> np.ndarray:
    """A confident person at every stride's corner cells, with boxes larger than the input."""
    rows = []
    for stride in detect.STRIDES:
        side = detect.INPUT_SIZE // stride
        for gx, gy in ((0, 0), (side - 1, 0), (0, side - 1), (side - 1, side - 1)):
            rows.append((anchor(stride, gx, gy), PERSON, 0.9, (0.5, 0.5, 8.0, 8.0)))
    return raw_output(*rows)


def assert_inside(found: list[detect.Detection], width: int, height: int) -> None:
    for d in found:
        x1, y1, x2, y2 = d.box
        assert all(math.isfinite(v) for v in d.box), d
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, (d, width, height)


# Letterbox ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width", "scale", "resized"),
    [
        (288, 352, 640 / 352, (523, 640)),
        (640, 640, 1.0, (640, 640)),
        (1280, 640, 0.5, (640, 320)),
        (1, 1, 640.0, (640, 640)),
        (1, 4000, 0.16, (1, 640)),  # a zero-row resize would crash OpenCV
        (4000, 1, 0.16, (640, 1)),
    ],
)
def test_letterbox_scales_and_pads_bottom_right(
    height: int, width: int, scale: float, resized: tuple[int, int]
) -> None:
    frame = np.full((height, width, 3), 7, dtype=np.uint8)
    tensor, got_scale = detect.letterbox(frame)
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert got_scale == pytest.approx(scale)
    rows, cols = resized
    assert (tensor[0, :, :rows, :cols] == 7).all()
    assert (tensor[0, :, rows:, :] == detect.PAD_VALUE).all()
    assert (tensor[0, :, :, cols:] == detect.PAD_VALUE).all()


def test_letterbox_keeps_channel_order() -> None:
    frame = np.zeros((288, 352, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 10, 20, 30  # B, G, R
    tensor, _ = detect.letterbox(frame)
    assert [tensor[0, c, 0, 0] for c in range(3)] == [10, 20, 30]


def test_unletterbox_inverts_the_letterbox_scale() -> None:
    frame_w, frame_h = 352, 288
    _, scale = detect.letterbox(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))
    original = np.array([[10.0, 20.0, 100.0, 200.0], [0.0, 0.0, 352.0, 288.0]], np.float32)
    in_model = original * np.float32(scale)
    back = detect.unletterbox(in_model, scale, frame_w, frame_h)
    np.testing.assert_allclose(back, original, rtol=1e-5)


def test_unletterbox_clips_to_the_frame() -> None:
    boxes = np.array([[-50.0, -5.0, 900.0, 700.0], [700.0, 600.0, 800.0, 640.0]], np.float32)
    out = detect.unletterbox(boxes, 1.0, 352, 288)
    np.testing.assert_array_equal(out[0], [0, 0, 352, 288])
    np.testing.assert_array_equal(out[1], [352, 288, 352, 288])  # entirely outside


# Grid decoding ---------------------------------------------------------------------------


def test_grids_cover_every_anchor_once() -> None:
    assert detect.GRIDS.shape == (N_ANCHORS, 2)
    assert detect.GRID_STRIDES.shape == (N_ANCHORS, 1)
    assert sorted(set(detect.GRID_STRIDES[:, 0].tolist())) == [8, 16, 32]
    assert (detect.GRID_STRIDES[:, 0] == 8).sum() == 80 * 80


@pytest.mark.parametrize(
    ("stride", "gx", "gy"), [(8, 0, 0), (8, 79, 0), (8, 3, 5), (16, 10, 39), (32, 19, 19)]
)
def test_decode_grid_cell_to_box(stride: int, gx: int, gy: int) -> None:
    raw = raw_output((anchor(stride, gx, gy), PERSON, 1.0, (0.5, 0.25, math.log(2), 0.0)))
    box = detect.decode(raw)[anchor(stride, gx, gy)]
    cx, cy = (gx + 0.5) * stride, (gy + 0.25) * stride
    w, h = 2 * stride, stride
    np.testing.assert_allclose(box, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], rtol=1e-5)


def test_decode_keeps_huge_log_sizes_finite() -> None:
    raw = raw_output((0, PERSON, 1.0, (0.0, 0.0, 1e30, np.inf)))
    assert np.isfinite(detect.decode(raw)).all()


# NMS -------------------------------------------------------------------------------------


def test_nms_suppresses_overlaps_and_keeps_the_best() -> None:
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 10, 10.5]], np.float32
    )
    scores = np.array([0.6, 0.9, 0.5, 0.7], np.float32)
    assert detect.nms(boxes, scores, 0.45) == [1, 2]


def test_nms_threshold_is_inclusive_of_equal_iou() -> None:
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], np.float32)  # IoU = 1/3
    scores = np.array([0.9, 0.8], np.float32)
    assert detect.nms(boxes, scores, 1 / 3 + 1e-6) == [0, 1]
    assert detect.nms(boxes, scores, 0.3) == [0]


def test_nms_handles_empty_and_degenerate_boxes() -> None:
    assert detect.nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.45) == []
    boxes = np.array([[5, 5, 5, 5], [5, 5, 5, 5]], np.float32)  # no area: IoU 0
    assert detect.nms(boxes, np.array([0.5, 0.4], np.float32), 0.45) == [0, 1]


def test_nms_matches_opencv_on_random_boxes() -> None:
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 300, size=(200, 2))
    wh = rng.uniform(5, 80, size=(200, 2))
    boxes = np.concatenate((xy, xy + wh), axis=1).astype(np.float32)
    scores = rng.uniform(0.3, 1.0, size=200).astype(np.float32)
    ours = detect.nms(boxes, scores, 0.45)
    theirs = cv2.dnn.NMSBoxes(np.concatenate((xy, wh), axis=1).tolist(), scores.tolist(), 0.0, 0.45)
    assert sorted(ours) == sorted(np.array(theirs).flatten().tolist())


# Class filtering and thresholds ----------------------------------------------------------


def test_postprocess_class_filter_keeps_only_person_and_umbrella() -> None:
    raw = raw_output(
        (anchor(32, 2, 2), PERSON, 0.9, (0, 0, 0, 0)),
        (anchor(32, 8, 8), UMBRELLA, 0.8, (0, 0, 0, 0)),
        (anchor(32, 14, 14), CAR, 0.99, (0, 0, 0, 0)),
    )
    found = detect.postprocess(raw, 1.0, 640, 640, 0.35, 0.45)
    assert [(d.label, round(d.score, 2)) for d in found] == [("person", 0.9), ("umbrella", 0.8)]


def test_postprocess_class_uses_the_best_class_only() -> None:
    raw = raw_output((anchor(32, 2, 2), PERSON, 0.5, (0, 0, 0, 0)))
    raw[anchor(32, 2, 2), 5 + CAR] = 0.6  # a car beats the person at this anchor
    assert detect.postprocess(raw, 1.0, 640, 640, 0.35, 0.45) == []


def test_postprocess_class_score_is_objectness_times_class() -> None:
    raw = raw_output((anchor(32, 2, 2), PERSON, 0.8, (0, 0, 0, 0)))
    raw[anchor(32, 2, 2), 4] = 0.5
    (found,) = detect.postprocess(raw, 1.0, 640, 640, 0.35, 0.45)
    assert found.score == pytest.approx(0.4)
    assert detect.postprocess(raw, 1.0, 640, 640, 0.41, 0.45) == []


def test_postprocess_suppresses_per_class_not_across_classes() -> None:
    same = (0.0, 0.0, 1.0, 1.0)
    raw = raw_output(
        (anchor(32, 5, 5), PERSON, 0.9, same),
        (anchor(32, 5, 5) + 1, PERSON, 0.8, (-1.0, 0.0, 1.0, 1.0)),  # the same box
        (anchor(16, 10, 10), UMBRELLA, 0.7, (0.0, 0.0, 2.0, 2.0)),  # overlaps the person
    )
    found = detect.postprocess(raw, 1.0, 640, 640, 0.35, 0.45)
    assert [d.label for d in found] == ["person", "umbrella"]


def test_postprocess_clip_maps_to_frame_and_drops_padding_boxes() -> None:
    # 352x288 frame: scale 640/352, content fills rows 0..523 of the input.
    scale = 640 / 352
    raw = raw_output(
        (anchor(32, 0, 0), PERSON, 0.9, (0.0, 0.0, 3.0, 3.0)),  # spills off the top left
        (anchor(32, 10, 18), PERSON, 0.8, (0.5, 0.5, 0.0, 0.0)),  # in the padding only
    )
    found = detect.postprocess(raw, scale, 352, 288, 0.35, 0.45)
    assert len(found) == 1
    assert found[0].box[0] == 0 and found[0].box[1] == 0
    assert_inside(found, 352, 288)


def test_postprocess_ignores_non_finite_rows() -> None:
    raw = raw_output((anchor(32, 3, 3), PERSON, 0.9, (0, 0, 0, 0)))
    raw[anchor(32, 6, 6)] = np.nan
    raw[anchor(32, 9, 9)] = np.inf
    raw[anchor(32, 12, 12), :4] = (np.nan, 0, 0, 0)
    raw[anchor(32, 12, 12), 4:6] = 1.0
    found = detect.postprocess(raw, 1.0, 640, 640, 0.35, 0.45)
    assert len(found) == 1
    assert_inside(found, 640, 640)


# Detector with a stand-in model -----------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width"),
    [(1, 1), (1, 4000), (4000, 1), (288, 352), (2, 3), (1000, 1000)],
)
def test_detector_boxes_stay_inside_hostile_frame_shapes_clip(height: int, width: int) -> None:
    frame = np.full((height, width, 3), 200, dtype=np.uint8)
    found = stub_detector(big_box_everywhere()).detect(frame)
    assert found
    assert_inside(found, width, height)


def test_detector_accepts_non_contiguous_frames() -> None:
    base = np.zeros((576, 704, 3), dtype=np.uint8)
    for frame in (base[::2, ::2], base[:, :, ::-1], np.asfortranarray(base[:288, :352])):
        assert not frame.flags.c_contiguous
        found = stub_detector(big_box_everywhere()).detect(frame)
        assert_inside(found, frame.shape[1], frame.shape[0])


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((288, 352), dtype=np.uint8),
        np.zeros((288, 352, 4), dtype=np.uint8),
        np.zeros((288, 352, 1), dtype=np.uint8),
        np.zeros((288, 352, 3, 1), dtype=np.uint8),
        np.zeros((288, 352, 3), dtype=np.float32),
        np.zeros((288, 352, 3), dtype=np.float64),
        np.zeros((288, 352, 3), dtype=np.int8),
        np.zeros((288, 352, 3), dtype=bool),
        np.zeros((0, 352, 3), dtype=np.uint8),
        np.zeros((288, 0, 3), dtype=np.uint8),
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros(0, dtype=np.uint8),
        np.zeros((), dtype=np.uint8),
        np.zeros((1081, 1920, 3), dtype=np.uint8),  # over MAX_IMAGE_PIXELS
        [[[0, 0, 0]]],
        None,
        b"\xff\xd8\xff",
    ],
)
def test_detector_rejects_invalid_frames(frame: Any) -> None:
    session = StubSession()
    detector = detect.Detector.from_session(session, name="stub")
    with pytest.raises(ValueError):
        detector.detect(frame)
    assert session.shapes == []  # never reached the model


def test_pixel_cap_matches_the_decoder_cap() -> None:
    stub_detector().detect(np.zeros((1000, 1000, 3), dtype=np.uint8))
    assert MAX_IMAGE_PIXELS == 1920 * 1080


def test_invalid_frame_errors_carry_no_pixel_values() -> None:
    frame = np.full((5, 5, 4), 213, dtype=np.uint8)
    with pytest.raises(ValueError) as info:
        stub_detector().detect(frame)
    assert "213" not in str(info.value)


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        [np.zeros((1, 100, ROW), np.float32)],
        [np.zeros((N_ANCHORS, ROW), np.float32)],
        [np.zeros((1, N_ANCHORS, 7), np.float32)],
        ["not an array"],
    ],
    ids=["none", "few-anchors", "no-batch", "few-classes", "string"],
)
def test_detector_rejects_malformed_model_output(outputs: list[object]) -> None:
    detector = detect.Detector.from_session(StubSession(outputs=outputs), name="stub")
    with pytest.raises(detect.DetectorError):
        detector.detect(np.zeros((288, 352, 3), dtype=np.uint8))


@pytest.mark.parametrize("dtype", [np.int32, np.uint8, np.bool_, np.complex64, object])
def test_detector_rejects_non_float_model_output(dtype: Any) -> None:
    out = np.zeros((1, N_ANCHORS, ROW), dtype=dtype)
    detector = detect.Detector.from_session(StubSession(outputs=[out]), name="stub")
    with pytest.raises(detect.DetectorError, match="dtype"):
        detector.detect(np.zeros((288, 352, 3), dtype=np.uint8))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_detector_accepts_float_model_output(dtype: Any) -> None:
    out = _one_person()[None].astype(dtype)
    detector = detect.Detector.from_session(StubSession(outputs=[out]), name="stub")
    assert len(detector.detect(np.zeros((288, 352, 3), dtype=np.uint8))) == 1


def _one_person() -> np.ndarray:
    return raw_output((anchor(32, 3, 3), PERSON, 0.9, (0, 0, 0, 0)))


@pytest.mark.parametrize(
    ("row", "column", "value"),
    [
        (anchor(8, 40, 40), 5 + CAR, np.nan),  # a class the detector does not report
        (anchor(8, 40, 40), 4, np.inf),  # objectness
        (anchor(16, 5, 5), 5 + PERSON, -np.inf),
        (anchor(16, 5, 5), 0, np.nan),  # a regression output
        (anchor(16, 5, 5), 3, np.inf),  # a log-size: decode would clamp it, but it is broken
        (anchor(32, 9, 9), 5 + UMBRELLA, 25.0),  # a score of 25, with objectness 0
        (anchor(32, 9, 9), 4, 1.0001),
        (anchor(32, 9, 9), 5 + CAR, -0.0001),
    ],
    ids=[
        "nan-other-class",
        "inf-objectness",
        "minus-inf-person",
        "nan-regression",
        "inf-log-size",
        "score-25",
        "objectness-above-one",
        "negative-score",
    ],
)
def test_detector_refuses_non_finite_or_out_of_range_output(
    row: int, column: int, value: float
) -> None:
    raw = _one_person()
    assert len(stub_detector(raw).detect(np.zeros((288, 352, 3), dtype=np.uint8))) == 1
    raw[row, column] = value
    with pytest.raises(detect.DetectorError):
        stub_detector(raw).detect(np.zeros((288, 352, 3), dtype=np.uint8))


def test_detector_accepts_scores_of_exactly_zero_and_one() -> None:
    raw = _one_person()
    raw[anchor(32, 9, 9), 4] = 1.0
    raw[anchor(32, 9, 9), 5 + CAR] = 1.0
    raw[anchor(32, 12, 12), 4:] = 0.0
    found = stub_detector(raw).detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert [d.label for d in found] == ["person"]


def test_detector_drops_a_box_whose_coordinates_overflow() -> None:
    raw = _one_person()
    overflow = anchor(8, 20, 20)
    raw[overflow] = 0.0
    raw[overflow, :2] = (3e38, 3e38)  # finite, but (dx + gx) * stride overflows float32
    raw[overflow, 4] = 1.0
    raw[overflow, 5 + PERSON] = 0.95
    with np.errstate(over="raise"):  # the overflow stays inside postprocess
        found = stub_detector(raw).detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert [round(d.score, 2) for d in found] == [0.9]
    assert "overflow" in (detect.Detector.detect.__doc__ or "")


def _confident_everywhere() -> np.ndarray:
    """Every anchor a certain person."""
    raw = np.zeros((N_ANCHORS, ROW), dtype=np.float32)
    raw[:, 4] = 1.0
    raw[:, 5 + PERSON] = 1.0
    return raw


@pytest.mark.parametrize("offset", [1e6, -1e30])
def test_detector_refuses_output_whose_confident_boxes_all_leave_the_frame(
    offset: float,
) -> None:
    """Finite centre offsets so far off that every confident box is clipped to no area: a
    broken model, not a frame with 0 people."""
    raw = _confident_everywhere()
    raw[:, 0:2] = offset
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    assert detect.postprocess(raw, 1.0, 640, 640, 0.35, 0.45) == []  # all dropped
    with pytest.raises(detect.DetectorError, match="none inside the frame"):
        stub_detector(raw).detect(frame)


def test_detector_refuses_output_whose_confident_boxes_all_overflow() -> None:
    raw = _confident_everywhere()
    raw[:, 0:2] = 3e38
    with pytest.raises(detect.DetectorError, match="none inside the frame"):
        stub_detector(raw).detect(np.zeros((640, 640, 3), dtype=np.uint8))


def test_detector_keeps_the_boxes_left_when_some_confident_ones_are_dropped() -> None:
    raw = _one_person()
    raw[anchor(32, 10, 18)] = raw[anchor(32, 3, 3)]
    raw[anchor(32, 10, 18), :2] = 1e6  # clipped to no area
    found = stub_detector(raw).detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert [round(d.score, 2) for d in found] == [0.9]


def test_detector_ignores_dropped_rows_below_the_threshold_or_of_other_classes() -> None:
    raw = np.zeros((N_ANCHORS, ROW), dtype=np.float32)
    raw[:, 0:2] = 1e6
    raw[:, 4] = 1.0
    raw[:10, 5 + PERSON] = 0.3  # under conf 0.35
    raw[10:, 5 + CAR] = 1.0  # not a class the detector reports
    assert stub_detector(raw).detect(np.zeros((640, 640, 3), dtype=np.uint8)) == []


def test_detector_wraps_runtime_errors() -> None:
    class Failing:
        def run(self, tensor: np.ndarray) -> Sequence[object]:
            raise RuntimeError("[ONNXRuntimeError] something with /secret/path")

    detector = detect.Detector.from_session(Failing(), name="stub")
    with pytest.raises(detect.DetectorError) as info:
        detector.detect(np.zeros((288, 352, 3), dtype=np.uint8))
    assert "secret" not in str(info.value)


@pytest.mark.parametrize("value", [0, 0.0, -0.1, 1.01, math.nan, math.inf])
def test_thresholds_out_of_range_raise_value_error(value: float) -> None:
    with pytest.raises(ValueError):
        stub_detector(conf=value)
    with pytest.raises(ValueError):
        stub_detector(nms=value)


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_thresholds_of_the_wrong_type_raise_type_error(value: Any) -> None:
    with pytest.raises(TypeError):
        stub_detector(conf=value)


def test_detector_keeps_no_reference_to_the_frame() -> None:
    detector = stub_detector(big_box_everywhere())
    frame = np.zeros((288, 352, 3), dtype=np.uint8)
    ref = weakref.ref(frame)
    detector.detect(frame)
    del frame
    gc.collect()
    assert ref() is None


# Opening a model ---------------------------------------------------------------------------


def test_session_options_write_nothing_to_disk() -> None:
    options = detect.session_options()
    assert options.enable_profiling is False
    assert options.optimized_model_filepath == ""
    assert options.log_severity_level >= 3


def test_open_session_uses_safe_options_and_cpu_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class Input:
        name = "images"
        shape = (1, 3, 640, 640)

    class FakeSession:
        def __init__(self, path: bytes, sess_options: Any, providers: list[str]) -> None:
            seen.update(path=path, options=sess_options, providers=providers)

        def get_inputs(self) -> list[Input]:
            return [Input()]

    model = tmp_path / "yolox_s.onnx"
    model.write_bytes(b"x")
    monkeypatch.setitem(detect.MODEL_SHA256, "test", hashlib.sha256(b"x").hexdigest())
    monkeypatch.setattr("wearreport.detect.ort.InferenceSession", FakeSession)
    detect.Detector(model)
    assert seen["path"] == b"x"  # the verified bytes, not a path read again later
    assert seen["providers"] == ["CPUExecutionProvider"]
    assert seen["options"].enable_profiling is False
    assert seen["options"].optimized_model_filepath == ""


def test_open_session_refuses_unpinned_or_unreadable_models(tmp_path: Path) -> None:
    bad = tmp_path / "yolox_s.onnx"
    bad.write_bytes(b"tampered")
    with pytest.raises(detect.DetectorError, match="pinned SHA-256"):
        detect.Detector(bad)
    with pytest.raises(detect.DetectorError, match="cannot read"):
        detect.Detector(tmp_path / "absent.onnx")
    with pytest.raises(detect.DetectorError, match="cannot read"):
        detect.Detector(tmp_path)  # a directory


def test_open_session_wraps_onnxruntime_load_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "yolox_s.onnx"
    model.write_bytes(b"not onnx")
    monkeypatch.setitem(detect.MODEL_SHA256, "test", hashlib.sha256(b"not onnx").hexdigest())
    with pytest.raises(detect.DetectorError, match="onnxruntime cannot load"):
        detect.Detector(model)


def test_open_session_refuses_oversized_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "yolox_s.onnx"
    model.write_bytes(bytes(65))
    monkeypatch.setattr(detect, "MAX_MODEL_BYTES", 64)
    with pytest.raises(detect.DetectorError, match="larger than"):
        detect.Detector(model)


def test_default_logger_is_quiet_before_the_first_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class Input:
        name = "images"
        shape = (1, 3, 640, 640)

    class FakeSession:
        def __init__(self, path: bytes, sess_options: Any, providers: list[str]) -> None:
            calls.append("session")

        def get_inputs(self) -> list[Input]:
            return [Input()]

    def severity(level: int) -> None:
        calls.append(f"severity {level}")

    model = tmp_path / "yolox_s.onnx"
    model.write_bytes(b"x")
    monkeypatch.setitem(detect.MODEL_SHA256, "test", hashlib.sha256(b"x").hexdigest())
    monkeypatch.setattr("wearreport.detect.ort.set_default_logger_severity", severity)
    monkeypatch.setattr("wearreport.detect.ort.InferenceSession", FakeSession)
    detect.Detector(model)
    detect.Detector(model)
    assert calls == ["severity 3", "session", "severity 3", "session"]


def test_from_session_is_documented_as_test_only_and_used_only_by_detect() -> None:
    doc = detect.Detector.from_session.__doc__ or ""
    assert "tests only" in doc.lower() and "SHA-256" in doc
    package = Path(detect.__file__).resolve().parent
    detect_py = package / "detect.py"
    modules = sorted(package.rglob("*.py"))
    assert detect_py in modules and len(modules) > 5
    users = [p for p in modules if p != detect_py and "from_session" in p.read_text("utf-8")]
    assert users == []


def test_pins_match_the_fetch_script() -> None:
    script = (Path(__file__).resolve().parents[3] / "scripts" / "fetch_model.sh").read_text()
    for digest in detect.MODEL_SHA256.values():
        assert digest in script


# onnxruntime telemetry -------------------------------------------------------------------


def _python(code: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], list[Path]]:
    """Run `code` with an empty HOME and TMPDIR and without the telemetry variable; return
    the result and every file the run left in those two directories."""
    home, tmp = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    tmp.mkdir()
    env = {k: v for k, v in os.environ.items() if k != detect.TELEMETRY_ENV}
    env.update(HOME=str(home), TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE="1")
    env = {k: v for k, v in env.items() if not k.startswith("XDG_")}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return proc, sorted(p for d in (home, tmp) for p in d.rglob("*"))


def test_telemetry_is_disabled_in_the_environment() -> None:
    assert os.environ[detect.TELEMETRY_ENV] == "1"


def test_importing_the_detector_leaves_no_telemetry_files(tmp_path: Path) -> None:
    code = (
        "import numpy as np\n"
        "from wearreport import detect\n"
        "session = detect.ort.InferenceSession\n"  # the library is loaded
        "print('ok')\n"
    )
    proc, left = _python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert left == []


def test_the_telemetry_check_itself_sees_the_files(tmp_path: Path) -> None:
    # Canary: a file written where onnxruntime puts its device ID is found. (Whether the
    # library writes it without the variable depends on the host, so it is planted here.)
    code = (
        "import pathlib\n"
        "d = pathlib.Path.home() / '.cache' / 'Microsoft' / 'DeveloperTools' / '.onnxruntime'\n"
        "d.mkdir(parents=True)\n"
        "(d / 'deviceid').write_text('x')\n"
        "from wearreport import detect\n"
    )
    proc, left = _python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert any(p.name == "deviceid" for p in left), left


def test_import_refuses_when_onnxruntime_was_loaded_first(tmp_path: Path) -> None:
    proc, _ = _python("import onnxruntime\nimport wearreport.detect\n", tmp_path)
    assert proc.returncode != 0
    assert "telemetry" in proc.stderr and "ImportError" in proc.stderr


# With the real model -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def yolox_s() -> detect.Detector:
    return detect.Detector(_model())


@pytest.mark.parametrize(
    ("height", "width"), [(1, 1), (1, 4000), (4000, 1), (2, 2), (288, 352), (1000, 1000)]
)
def test_model_boxes_clip_to_hostile_frame_shapes(
    yolox_s: detect.Detector, height: int, width: int
) -> None:
    rng = np.random.default_rng(height * 7919 + width)
    frame = np.asarray(rng.integers(0, 256, size=(height, width, 3)), dtype=np.uint8)
    assert_inside(yolox_s.detect(frame), width, height)


def test_model_clip_on_fixture_crops(yolox_s: detect.Detector) -> None:
    frame = np.asarray(cv2.imread(str(FIXTURES / "people_street.jpg")), dtype=np.uint8)
    assert frame.ndim == 3
    for crop in (frame[:200], frame[:, :300], frame[-150:, -400:], frame[::3, ::2]):
        assert_inside(yolox_s.detect(crop), crop.shape[1], crop.shape[0])


@pytest.mark.parametrize("name", ["people_street.jpg", "umbrella_rain.jpg"])
def test_model_output_passes_the_checks_unchanged(yolox_s: detect.Detector, name: str) -> None:
    """The output checks refuse only broken output: real frames give exactly what
    postprocess makes of the raw output, as before the checks existed."""
    frame = np.asarray(cv2.imread(str(FIXTURES / name)), dtype=np.uint8)
    tensor, scale = detect.letterbox(frame)
    (out, *_) = detect.open_session(_model()).run(tensor)
    raw = np.asarray(out, dtype=np.float32)[0]
    assert np.isfinite(raw).all() and raw[:, 4:].min() >= 0 and raw[:, 4:].max() <= 1
    expected = detect.postprocess(raw, scale, frame.shape[1], frame.shape[0], 0.35, 0.45)
    assert expected and yolox_s.detect(frame) == expected


def test_model_output_is_deterministic(yolox_s: detect.Detector) -> None:
    frame = np.asarray(cv2.imread(str(FIXTURES / "umbrella_rain.jpg")), dtype=np.uint8)
    assert frame.ndim == 3
    assert yolox_s.detect(frame) == yolox_s.detect(frame.copy())


def test_model_rejects_the_same_invalid_frames(yolox_s: detect.Detector) -> None:
    frames: list[Any] = [
        np.zeros((288, 352), np.uint8),
        np.zeros((288, 352, 4), np.uint8),
        np.zeros((288, 352, 3), np.float32),
        np.zeros((0, 352, 3), np.uint8),
    ]
    for frame in frames:
        with pytest.raises(ValueError):
            yolox_s.detect(frame)
