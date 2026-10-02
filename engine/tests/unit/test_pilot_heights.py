"""wearreport.tools.pilot_heights: the JPEG header reader, reduced decoding, the camera
list parser, the URL policy and the tally. Every image is synthetic and encoded in
memory; nothing reaches the network."""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable

import numpy as np
import pytest

from wearreport import detect, fetch
from wearreport._cv import MAX_IMAGE_PIXELS, cv2, encode_jpeg
from wearreport.testing.fake_cameras import jpeg_declaring
from wearreport.tools import pilot_heights as ph

INSIDE = {"type": "Point", "coordinates": [-97.745, 30.27]}
URL = "https://cctv.austinmobility.io/image/1.jpg"


def _jpeg(width: int, height: int, value: int = 100) -> bytes:
    return encode_jpeg(np.full((height, width, 3), value, dtype=np.uint8))


# JPEG header --------------------------------------------------------------------------


@pytest.mark.parametrize(("width", "height"), [(352, 288), (1920, 1080), (1, 1), (17, 9)])
def test_jpeg_size_reads_the_frame_header(width: int, height: int) -> None:
    assert ph.jpeg_size(_jpeg(width, height)) == (width, height)


def test_jpeg_size_reads_a_rewritten_header() -> None:
    assert ph.jpeg_size(jpeg_declaring(30000, 20000)) == (30000, 20000)


def test_jpeg_size_skips_fill_bytes_and_tem() -> None:
    body = _jpeg(64, 48)
    padded = body[:2] + b"\xff\xff\xff\x01" + body[2:]  # fill bytes, then TEM
    assert ph.jpeg_size(padded) == (64, 48)


# The markers the JPEG decoder accepts before the frame header: APP0-APP15, COM, DQT,
# DHT, DAC and DRI (each with a length), and TEM (without one).
_SEGMENTS_BEFORE_SOF = {*range(0xE0, 0xF0), 0xFE, 0xDB, 0xC4, 0xCC, 0xDD}
_SOF_8X16 = b"\x00\x0b\x08\x00\x10\x00\x08\x01\x01\x11\x00"  # 8 wide, 16 high


@pytest.mark.parametrize("marker", range(0x100))
def test_jpeg_size_accepts_only_the_decoders_markers_before_the_frame(marker: int) -> None:
    body = _jpeg(64, 48)
    if marker in ph.SOF_MARKERS:
        inserted, expected = _SOF_8X16, (8, 16)  # the first frame header is the one read
    elif marker == 0xFF:
        inserted, expected = b"", (64, 48)  # a fill byte before the next marker
    elif marker == 0x01:
        inserted, expected = b"", (64, 48)  # TEM
    elif marker in _SEGMENTS_BEFORE_SOF:
        inserted, expected = b"\x00\x04\x00\x00", (64, 48)
    else:  # FF 00, RSTn, SOI, EOI, SOS, DNL, DHP, EXP, JPGn and anything else
        inserted, expected = b"\x00\x04\x00\x00", None
    assert ph.jpeg_size(body[:2] + bytes([0xFF, marker]) + inserted + body[2:]) == expected


def test_decoy_frame_header_behind_a_stuffed_zero_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The decoder reads FF 00 as a stuffed zero and decodes the 3000x3000 frame inside
    # what a length-skipping reader would pass over, to reach the small decoy header.
    hidden = _progressive(3000, 3000)[2:]
    decoy = b"\xff\xc2\x00\x11\x08\x00\x3d\xff\xff\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    assert 65535 * 61 <= ph.MAX_HEADER_PIXELS < 3000 * 3000
    body = b"\xff\xd8\xff\x00" + (len(hidden) + 2).to_bytes(2, "big") + hidden + decoy
    body += b"\xff\xd9"
    assert ph.jpeg_size(body) is None
    calls: list[object] = []
    monkeypatch.setattr(cv2, "imdecode", lambda *args: calls.append(args))
    with pytest.raises(ph._Failed) as exc:
        ph.decode_frame(body)
    assert exc.value.kind == "decode"
    assert calls == []


def _with_segments(body: bytes) -> bytes:
    exif = b"\xff\xe1\x00\x10Exif\x00\x00II*\x00\x08\x00\x00\x00"
    return body[:2] + exif + b"\xff\xfe\x00\x07hello" + b"\xff\xe2\x00\x04\x00\x00" + body[2:]


@pytest.mark.parametrize("progressive", [False, True])
@pytest.mark.parametrize("segments", [False, True])
def test_real_1080p_frames_still_decode(progressive: bool, segments: bool) -> None:
    image = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    params = [cv2.IMWRITE_JPEG_PROGRESSIVE, 1] if progressive else []
    ok, buf = cv2.imencode(".jpg", image, params)
    assert ok
    body = _with_segments(buf.tobytes()) if segments else buf.tobytes()
    assert ph.jpeg_size(body) == (1920, 1080)
    decoded = ph.decode_frame(body)
    assert decoded.frame.shape == (1080, 1920, 3)
    assert (decoded.width, decoded.height) == (1920, 1080)


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"\xff\xd8",
        b"\xff\xd8\xff",
        b"\xff\xd8\x00\x00\x00\x00",  # not a marker
        b"\xff\xd8\xff\xe0\x00\x01\x00\x00",  # a segment length below 2
        b"\xff\xd8\xff\xe0\x00\x10",  # a segment that runs past the end
        b"\xff\xd8\xff\xda\x00\x08" + b"\0" * 16,  # image data before any frame header
        b"\xff\xd8\xff\xd9",  # end of image
        b"\xff\xd8\xff\xc0\x00\x11\x08\x00",  # a frame header cut short
        b"\xff\xd8\xff\xc0\x00\x05\x08\x00\x10\x00\x10",  # a frame header too short
        b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x00\x00\x10" + b"\0" * 12,  # zero height
        b"\xff\xd8" + b"\xff\xfe\x00\x02" * 10_000,  # many empty comments, no frame
    ],
)
def test_jpeg_size_refuses_malformed_headers(body: bytes) -> None:
    assert ph.jpeg_size(body) is None


# Decoding -----------------------------------------------------------------------------


def test_small_frames_decode_at_full_size() -> None:
    decoded = ph.decode_frame(_jpeg(352, 288))
    assert decoded.frame.shape == (288, 352, 3)
    assert (decoded.width, decoded.height) == (352, 288)


@pytest.mark.parametrize(
    ("width", "height", "shape"),
    [
        (1920, 1080, (1080, 1920, 3)),
        (1000, 1000, (1000, 1000, 3)),  # exactly the cap
        (1001, 1000, (1000, 1001, 3)),
        (2560, 1440, (720, 1280, 3)),
        (2000, 2000, (1000, 1000, 3)),  # exactly the header bound
        (1921, 1081, (541, 961, 3)),  # odd sizes round up
    ],
)
def test_large_frames_decode_reduced_under_the_cap(
    width: int, height: int, shape: tuple[int, int, int]
) -> None:
    decoded = ph.decode_frame(_jpeg(width, height))
    assert decoded.frame.shape == shape
    assert decoded.frame.shape[0] * decoded.frame.shape[1] <= MAX_IMAGE_PIXELS
    assert (decoded.width, decoded.height) == (width, height)
    assert decoded.frame.dtype == np.uint8 and decoded.frame.flags.c_contiguous


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"GIF89a" + b"\0" * 64,
        b"\x89PNG\r\n\x1a\n" + b"\0" * 64,
        b"\xff\xd8\xff" + b"\0" * 64,  # no end-of-image marker
        _jpeg(64, 48)[:-100],
        b"\xff\xd8\xff\xe0\x00\x10" + b"\0" * 32 + b"\xff\xd9",  # no frame header
        jpeg_declaring(30000, 30000),  # over the cap even at 1/8
        _jpeg(3840, 2160),  # over the header bound
    ],
)
def test_decode_refuses_anything_but_a_complete_jpeg(body: bytes) -> None:
    with pytest.raises(ph._Failed) as exc:
        ph.decode_frame(body)
    assert exc.value.kind == "decode"


def _progressive(width: int, height: int) -> bytes:
    image = np.full((height, width, 3), 100, dtype=np.uint8)
    ok, buf = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 50, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
    )
    assert ok
    body = buf.tobytes()
    assert b"\xff\xc2" in body and b"\xff\xc0" not in body
    return body


def _with_sof(body: bytes, marker: int) -> bytes:
    sof = body.find(b"\xff\xc0")
    assert sof >= 0
    return body[: sof + 1] + bytes([marker]) + body[sof + 2 :]


def test_header_bound_is_four_times_the_cap() -> None:
    assert ph.MAX_HEADER_PIXELS == 4_000_000
    assert ph.MAX_HEADER_PIXELS >= 2560 * 1440


@pytest.mark.parametrize("make", [_jpeg, _progressive])
def test_decode_accepts_a_header_exactly_at_the_bound(make: Callable[[int, int], bytes]) -> None:
    decoded = ph.decode_frame(make(2000, 2000))
    assert ph.MAX_HEADER_PIXELS == 2000 * 2000
    assert decoded.frame.shape == (1000, 1000, 3)
    assert (decoded.width, decoded.height) == (2000, 2000)


@pytest.mark.parametrize("make", [_jpeg, _progressive])
@pytest.mark.parametrize(("width", "height"), [(2001, 2000), (2000, 2001), (4001, 1000)])
def test_decode_refuses_a_header_above_the_bound_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
    make: Callable[[int, int], bytes],
    width: int,
    height: int,
) -> None:
    body = make(width, height)
    assert ph.jpeg_size(body) == (width, height)
    calls: list[object] = []
    monkeypatch.setattr(cv2, "imdecode", lambda *args: calls.append(args))
    with pytest.raises(ph._Failed) as exc:
        ph.decode_frame(body)
    assert exc.value.kind == "decode"
    assert calls == []


@pytest.mark.parametrize("marker", sorted(ph.SOF_MARKERS))
def test_header_bound_holds_for_every_frame_type(
    monkeypatch: pytest.MonkeyPatch, marker: int
) -> None:
    body = _with_sof(jpeg_declaring(2001, 2000), marker)
    assert ph.jpeg_size(body) == (2001, 2000)
    calls: list[object] = []
    monkeypatch.setattr(cv2, "imdecode", lambda *args: calls.append(args))
    with pytest.raises(ph._Failed) as exc:
        ph.decode_frame(body)
    assert exc.value.kind == "decode"
    assert calls == []


def test_decode_reuses_the_sweeps_markers() -> None:
    assert fetch.JPEG_SOI == b"\xff\xd8\xff" and fetch.JPEG_EOI == b"\xff\xd9"
    body = _jpeg(64, 48) + b"\0" * (fetch.EOI_WINDOW + 1)  # end marker too far from the end
    with pytest.raises(ph._Failed):
        ph.decode_frame(body)


# Camera list --------------------------------------------------------------------------


def test_decode_dataset_accepts_a_list() -> None:
    assert ph.decode_dataset(b'[{"a": 1}, 2]') == [{"a": 1}, 2]


@pytest.mark.parametrize(
    "body",
    [b"{}", b"1", b"", b"[", b"\xff\xff", b"[" * 50_000, b"[NaN"],
)
def test_decode_dataset_raises_typed_errors(body: bytes) -> None:
    with pytest.raises(ph.PilotError):
        ph.decode_dataset(body)


def test_decode_dataset_keeps_one_huge_integer_out_of_range() -> None:
    records = ph.decode_dataset(b"[" + b"9" * 10_000 + b", -" + b"9" * 40 + b", 12]")
    assert records == [float("inf"), float("inf"), 12]


def test_decode_dataset_refuses_oversized_bodies() -> None:
    with pytest.raises(ph.PilotError):
        ph.decode_dataset(b"[" + b" " * ph.MAX_DATASET_BYTES + b"]")


def _rec(url: object = URL, coordinates: object = None) -> dict[str, object]:
    location = (
        dict(INSIDE) if coordinates is None else {"type": "Point", "coordinates": coordinates}
    )
    return {"screenshot_address": url, "location": location}


def test_selection_keeps_list_order_and_caps_the_count() -> None:
    urls = [f"https://cctv.austinmobility.io/image/{i}.jpg" for i in range(5)]
    records: list[object] = [_rec(u) for u in urls]
    sel = ph.select_cameras(records, ph.DEFAULT_BBOX, max_cameras=3)
    assert sel.urls == urls[:3]
    assert (sel.listed, sel.skipped, sel.refused) == (5, 0, 0)


def test_selection_counts_refusals_inside_the_box_only() -> None:
    records: list[object] = [
        _rec("http://cctv.austinmobility.io/image/1.jpg"),
        _rec("https://evil.example/1.jpg", [-90.0, 30.0]),  # outside: neither
        _rec(URL),
        {"location": INSIDE},
    ]
    sel = ph.select_cameras(records, ph.DEFAULT_BBOX)
    assert sel.urls == [URL]
    assert (sel.listed, sel.skipped, sel.refused) == (4, 1, 1)


@pytest.mark.parametrize(
    "coordinates",
    [
        [-97.745],
        [-97.745, 30.27, 0.0],
        ["-97.745", 30.27],
        [-97.745, False],
        [float("nan"), 30.27],
        [-97.745, float("inf")],
        [-97.745, 10**400],
        [-181, 30.27],
        [-97.745, -91],
        {"lon": -97.745, "lat": 30.27},
        None,
    ],
)
def test_selection_skips_malformed_coordinates(coordinates: object) -> None:
    record = _rec(URL, coordinates)
    if coordinates is None:
        record = {"screenshot_address": URL, "location": {"type": "Point"}}
    sel = ph.select_cameras([record], ph.DEFAULT_BBOX)
    assert (sel.skipped, sel.urls) == (1, [])


def test_selection_skips_a_location_that_is_not_a_point() -> None:
    record = {"screenshot_address": URL, "location": {"type": "Polygon", "coordinates": [1, 2]}}
    assert ph.select_cameras([record], ph.DEFAULT_BBOX).skipped == 1


# URL policy ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://cctv.austinmobility.io:443/image/1.jpg",  # an explicit port, even 443
        "https://CCTV.austinmobility.io/image/1.jpg",
        "https://cctv.austinmobility.io",  # no path
        "https://cctv.austinmobility.io?x=1",
        "https://cctv.austinmobility.io/image/é.jpg",
        "https://cctv.austinmobility.io/image/1.jpg\t",
        "https://[::1]/image/1.jpg",
        "https://cctv.austinmobility.io/" + "a" * ph.MAX_URL_LENGTH,
    ],
)
def test_policy_refuses_near_misses(url: str) -> None:
    assert not ph.SCREENSHOT_POLICY.allows(url)


def test_policy_allows_a_query_on_the_pinned_host() -> None:
    assert ph.SCREENSHOT_POLICY.allows("https://cctv.austinmobility.io/image/1.jpg?t=2")


# Tally and report ---------------------------------------------------------------------


def test_heights_scale_back_to_the_original_frame() -> None:
    tally = ph.Tally()
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    decoded = ph._Decoded(frame, 1920, 1080)
    found = [
        detect.Detection("person", 0.9, (0.0, 0.0, 10.0, 22.6)),  # 45.2 px -> 45
        detect.Detection("person", 0.9, (0.0, 0.0, 10.0, 22.8)),  # 45.6 px -> 46
        detect.Detection("umbrella", 0.9, (0.0, 0.0, 10.0, 200.0)),
    ]
    tally.add(decoded, found)
    assert tally.bands == {"31-45": 1, "46-79": 1}
    assert tally.bands_hd == tally.bands
    assert (tally.persons, tally.umbrellas, tally.frames_ok) == (2, 1, 1)
    assert tally.resolutions == {"1920x1080": 1}


def test_a_reduced_decode_scales_heights_back_to_the_header_size() -> None:
    assert MAX_IMAGE_PIXELS < 2560 * 1440 <= ph.MAX_HEADER_PIXELS
    decoded = ph.decode_frame(_jpeg(2560, 1440))
    assert decoded.frame.shape == (720, 1280, 3)
    assert (decoded.width, decoded.height) == (2560, 1440)
    tally = ph.Tally()
    found = [
        detect.Detection("person", 0.9, (0.0, 0.0, 10.0, 22.6)),  # 45.2 px -> 45
        detect.Detection("person", 0.9, (0.0, 0.0, 10.0, 22.8)),  # 45.6 px -> 46
    ]
    tally.add(decoded, found)
    assert tally.bands == {"31-45": 1, "46-79": 1}
    assert tally.bands_hd == {}
    assert tally.resolutions == {"2560x1440": 1}


def test_report_rounds_and_orders() -> None:
    tally = ph.Tally(refused=2)
    tally.failed["network"] += 1
    sel = ph.Selection(listed=10, skipped=3, refused=1, urls=["u"] * 4)
    started = datetime.datetime(
        2026, 10, 1, 13, 30, 59, tzinfo=datetime.timezone(datetime.timedelta(hours=-5))
    )
    out = ph.report(started_at=started, sun_elevation=12.349, model="m", selection=sel, tally=tally)
    assert out["started_at"] == "2026-10-01T18:30Z"
    assert out["sun_elevation_deg"] == 12.3
    assert out["refused_url"] == 3
    assert out["frames_failed"] == dict.fromkeys(ph.FAILURE_KINDS, 0) | {"network": 1}
    assert json.loads(json.dumps(out)) == out


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert ph.main(["--help"]) == 0
    assert "--live" in capsys.readouterr().out
