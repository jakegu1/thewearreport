"""wearreport.tools.pilot_heights: the JPEG header reader, reduced decoding, the camera
list parser, the URL policy and the tally. Every image is synthetic and encoded in
memory; nothing reaches the network."""

from __future__ import annotations

import datetime
import json

import numpy as np
import pytest

from wearreport import detect, fetch
from wearreport._cv import MAX_IMAGE_PIXELS, encode_jpeg
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


def test_jpeg_size_skips_fill_and_standalone_markers() -> None:
    body = _jpeg(64, 48)
    padded = body[:2] + b"\xff\xff\xff\xd0" + body[2:]  # fill bytes, then RST0
    assert ph.jpeg_size(padded) == (64, 48)


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
        (1920, 1080, (540, 960, 3)),
        (1000, 1000, (1000, 1000, 3)),  # exactly the cap
        (1001, 1000, (500, 501, 3)),
        (2560, 1440, (720, 1280, 3)),
        (3840, 2160, (540, 960, 3)),
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
    ],
)
def test_decode_refuses_anything_but_a_complete_jpeg(body: bytes) -> None:
    with pytest.raises(ph._Failed) as exc:
        ph.decode_frame(body)
    assert exc.value.kind == "decode"


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
