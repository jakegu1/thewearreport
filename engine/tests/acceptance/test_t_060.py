"""Acceptance tests for T-060: the sweep decoder refuses JPEG bodies with too many scans,
with one shared definition of the limit and of the counting rule. The task contract: do
not edit.

Every image here is synthetic (zeros or random noise, encoded in memory) and is served by
the local fake camera server on 127.0.0.1. Nothing reaches the network and nothing is
written to disk.
"""

from __future__ import annotations

import ast
import inspect
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import aggregate, detect, fetch
from wearreport._cv import cv2
from wearreport.testing.fake_cameras import FakeCameraServer
from wearreport.tools import pilot_heights as ph

SOS = b"\xff\xda"
EOI = b"\xff\xd9"


@pytest.fixture
def server() -> Iterator[FakeCameraServer]:
    with FakeCameraServer() as srv:
        yield srv


@pytest.fixture
def imdecode_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every call to cv2.imdecode (the object both decoders call), by buffer length."""
    calls: list[int] = []
    real = cv2.imdecode

    def spy(buf: Any, flags: int) -> Any:
        calls.append(len(buf))
        return real(buf, flags)

    monkeypatch.setattr(cv2, "imdecode", spy)
    return calls


def _encode(image: npt.NDArray[np.uint8], progressive: bool) -> bytes:
    params = [cv2.IMWRITE_JPEG_PROGRESSIVE, 1] if progressive else []
    ok, buf = cv2.imencode(".jpg", image, params)
    assert ok
    return bytes(buf.tobytes())


def _noise(width: int, height: int, seed: int = 0) -> npt.NDArray[np.uint8]:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _scans(body: bytes) -> list[bytes]:
    """Each scan of a JPEG: its start-of-scan segment and the entropy-coded data after it,
    up to the next marker (FF 00 is a stuffed zero, FF D0-D7 a restart marker)."""
    scans = []
    start = body.find(SOS)
    while start >= 0:
        end = start + 2 + int.from_bytes(body[start + 2 : start + 4], "big")
        while not (body[end] == 0xFF and body[end + 1] not in {0x00, *range(0xD0, 0xD8)}):
            end += 1
        scans.append(body[start:end])
        start = body.find(SOS, end)
    return scans


def _progressive_with_scans(image: npt.NDArray[np.uint8], total: int) -> bytes:
    """A progressive JPEG of `image` holding `total` scans: its smallest scan repeated as
    the last bytes before the end-of-image marker (the decoder only warns about these)."""
    body = _encode(image, progressive=True)
    scans = _scans(body)
    assert len(scans) <= total and body.endswith(EOI)
    smallest = min(scans, key=len)
    body = body[:-2] + smallest * (total - len(scans)) + EOI
    assert body.count(SOS) == total
    return body


def _with_segment(body: bytes, marker: int, payload: bytes) -> bytes:
    """`body` with an APPn or COM segment holding `payload` right after start-of-image."""
    assert body.startswith(b"\xff\xd8")
    segment = bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload
    return body[:2] + segment + body[2:]


def _sweep_one(server: FakeCameraServer, body: bytes) -> fetch.FrameResult:
    (cam,) = server.cameras(1)
    server.serve_body(cam.id, body)
    (result,) = fetch.fetch_sweep([cam])
    return result


def _refused_by_fetch(body: bytes) -> bool:
    try:
        fetch._decode(body)
    except fetch._FetchError as exc:
        assert exc.kind == "decode"
        return True
    return False


def _refused_by_pilot(body: bytes) -> bool:
    try:
        ph.decode_frame(body)
    except ph._Failed as exc:
        assert exc.kind == "decode"
        return True
    return False


# AC1: one definition --------------------------------------------------------------------


def test_ac1_the_limit_and_the_marker_live_in_fetch() -> None:
    assert fetch.MAX_SCANS == 32
    assert fetch.JPEG_SOS == SOS
    assert ph.MAX_SCANS == fetch.MAX_SCANS and ph.JPEG_SOS == fetch.JPEG_SOS


@pytest.mark.parametrize(
    "body",
    [b"", SOS, b"\xff\xd8" + SOS * 3 + EOI, b"\xff" + SOS + b"\xda\xff\xda", b"\xff\xff\xda"],
    ids=["empty", "one", "three", "overlapping-ff", "fill-byte"],
)
def test_ac1_one_helper_counts_markers_over_the_whole_body(body: bytes) -> None:
    assert fetch.count_scans(body) == body.count(SOS)


def test_ac1_pilot_heights_defines_neither_the_limit_nor_the_marker() -> None:
    tree = ast.parse(inspect.getsource(ph))
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert not {"MAX_SCANS", "JPEG_SOS"} & assigned
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "count_scans" not in defined


def test_ac1_both_decoders_count_with_the_shared_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _encode(_noise(64, 48), progressive=False)
    seen: list[bytes] = []
    real = fetch.count_scans

    def spy(data: bytes) -> int:
        seen.append(data)
        return real(data)

    monkeypatch.setattr(fetch, "count_scans", spy)
    fetch._decode(body)
    assert seen == [body]
    ph.decode_frame(body)
    assert seen == [body, body]


def test_ac1_both_decoders_obey_the_shared_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _encode(_noise(64, 48), progressive=False)
    assert not _refused_by_fetch(body) and not _refused_by_pilot(body)
    monkeypatch.setattr(fetch, "MAX_SCANS", 0)
    assert _refused_by_fetch(body) and _refused_by_pilot(body)


# AC2: the sweep decoder -----------------------------------------------------------------


def test_ac2_more_than_max_scans_is_refused_before_imdecode(imdecode_calls: list[int]) -> None:
    body = _progressive_with_scans(_noise(352, 288), fetch.MAX_SCANS + 1)
    with pytest.raises(fetch._FetchError) as exc:
        fetch._decode(body)
    assert exc.value.kind == "decode"
    assert imdecode_calls == []


def test_ac2_exactly_max_scans_decodes_as_today(imdecode_calls: list[int]) -> None:
    body = _progressive_with_scans(_noise(352, 288), fetch.MAX_SCANS)
    frame = fetch._decode(body)
    assert imdecode_calls == [len(body)]
    expected = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert expected is not None and frame.shape == (288, 352, 3)
    assert np.array_equal(frame, expected)


# AC3: bounded through the fake camera server -------------------------------------------


def test_ac3_eight_thousand_extra_scans_are_a_decode_error_within_2_s(
    server: FakeCameraServer, imdecode_calls: list[int]
) -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    base = len(_scans(_encode(image, progressive=True)))
    body = _progressive_with_scans(image, base + 8_000)
    assert len(body) < fetch.MAX_FRAME_BYTES
    started = time.monotonic()
    result = _sweep_one(server, body)
    assert time.monotonic() - started < 2
    assert (result.error, result.frame) == ("decode", None)
    assert imdecode_calls == []


# AC4: unchanged --------------------------------------------------------------------------


@pytest.mark.parametrize("progressive", [False, True], ids=["baseline", "progressive"])
@pytest.mark.parametrize("size", [(352, 288), (1920, 1080)], ids=["352x288", "1920x1080"])
def test_ac4_real_frames_decode_exactly_as_before(
    server: FakeCameraServer, size: tuple[int, int], progressive: bool
) -> None:
    body = _encode(_noise(*size, seed=size[0]), progressive)
    assert fetch.count_scans(body) <= fetch.MAX_SCANS
    result = _sweep_one(server, body)
    assert result.error is None and result.frame is not None
    expected = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert expected is not None
    assert result.frame.dtype == np.uint8 and np.array_equal(result.frame, expected)


def test_ac4_frame_results_keep_their_fields() -> None:
    assert fetch.FrameResult.__slots__ == ("camera_id", "frame", "seconds", "error")
    assert fetch.ERROR_KINDS == ("timeout", "http", "decode", "network")


class _Blank:
    """A stand-in for YOLOX that finds nothing."""

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return [np.zeros(detect.OUTPUT_SHAPE, dtype=np.float32)]


def test_ac4_sweep_records_keep_their_keys_and_count_the_refusal(
    server: FakeCameraServer,
) -> None:
    cams = server.cameras(3)
    server.serve_body(cams[1].id, _progressive_with_scans(_noise(352, 288), fetch.MAX_SCANS + 1))
    server.serve_body(cams[2].id, _progressive_with_scans(_noise(352, 288), fetch.MAX_SCANS))

    def no_weather(_: datetime) -> None:
        return None

    record = aggregate.run_sweep(
        detect.Detector.from_session(_Blank(), name="blank"),
        "0" * 64,
        model_name="blank",
        list_cameras=lambda: cams,
        conditions=no_weather,
    )
    aggregate.check_record(record)
    assert set(record) - {aggregate.HEIGHTS, aggregate.UMBRELLA_HEIGHTS} == aggregate.RECORD_KEYS
    assert record["cameras_listed"] == 3
    assert record["frames_ok"] == 2
    assert record["frames_failed"] == {
        kind: int(kind == "decode") for kind in aggregate.ERROR_KINDS
    }


# Counted over the whole body ------------------------------------------------------------


@pytest.mark.parametrize("marker", [0xE1, 0xFE], ids=["APP1", "COM"])
def test_markers_inside_an_app_or_com_segment_count(marker: int) -> None:
    """A baseline body has one scan; SOS bytes inside a skipped segment still count."""
    body = _encode(_noise(352, 288), progressive=False)
    assert fetch.count_scans(body) == 1
    over = _with_segment(body, marker, SOS * fetch.MAX_SCANS)
    at = _with_segment(body, marker, SOS * (fetch.MAX_SCANS - 1))
    assert fetch.count_scans(over) == fetch.MAX_SCANS + 1
    assert cv2.imdecode(np.frombuffer(over, np.uint8), cv2.IMREAD_COLOR) is not None
    assert _refused_by_fetch(over) and _refused_by_pilot(over)
    assert not _refused_by_fetch(at) and not _refused_by_pilot(at)


def test_extra_scans_just_before_the_end_marker_count() -> None:
    image = _noise(352, 288)
    over = _progressive_with_scans(image, fetch.MAX_SCANS + 1)
    at = _progressive_with_scans(image, fetch.MAX_SCANS)
    assert over[-2:] == EOI and over[:-2].endswith(_scans(over)[-1])
    assert fetch.count_scans(over) == fetch.MAX_SCANS + 1
    assert _refused_by_fetch(over) and _refused_by_pilot(over)
    assert not _refused_by_fetch(at) and not _refused_by_pilot(at)
