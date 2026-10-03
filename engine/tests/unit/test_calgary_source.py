"""Unit tests for Calgary as a camera source: the upgrading URL policy, the record fields,
the tally and report at Calgary's frame size, and the spot-check record and file name.
Synthetic data only; nothing reaches the network."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from wearreport import detect
from wearreport._cv import cv2, encode_jpeg
from wearreport.tools import pilot_heights as ph
from wearreport.tools import spotcheck, spotcheck_summary

UPGRADING = ph.UrlPolicy("https", "img.example", upgrade="http")


@pytest.mark.parametrize(
    ("url", "resolved"),
    [
        ("https://img.example/a.jpg", "https://img.example/a.jpg"),
        ("http://img.example/a.jpg", "https://img.example/a.jpg"),
        ("HTTP://img.example/a.jpg", "https://img.example/a.jpg"),
        ("http://img.example/a.jpg?x=1", "https://img.example/a.jpg?x=1"),
        ("http://img.example:443/a.jpg", None),
        ("http://img.example:80/a.jpg", None),
        ("http://u@img.example/a.jpg", None),
        ("http://img.example", None),  # no path
        ("httpx://img.example/a.jpg", None),
        ("ftp://img.example/a.jpg", None),
        ("http:img.example/a.jpg", None),
        ("http", None),
        ("", None),
    ],
)
def test_resolve_upgrades_only_to_an_allowed_url(url: str, resolved: str | None) -> None:
    assert UPGRADING.resolve(url) == resolved


def test_without_upgrade_resolve_is_allows() -> None:
    policy = ph.UrlPolicy("https", "img.example")
    assert policy.upgrade is None
    assert policy.resolve("https://img.example/a.jpg") == "https://img.example/a.jpg"
    assert policy.resolve("http://img.example/a.jpg") is None


def test_the_cities() -> None:
    assert ph.CITIES == {"austin": ph.AUSTIN_CITY, "calgary": ph.CALGARY_CITY}
    austin, calgary = ph.AUSTIN_CITY, ph.CALGARY_CITY
    assert (austin.url_field, austin.point_field) == ("screenshot_address", "location")
    assert (calgary.url_field, calgary.point_field) == ("camera_url", "point")
    assert austin.bands_field == "persons_by_height_band_1080p" and austin.frame_size == ph.HD
    assert calgary.bands_field == "persons_by_height_band_840x630"
    assert calgary.image_policy.upgrade == "http" and austin.image_policy.upgrade is None
    assert calgary.dataset_policy.upgrade is None


def _calgary_record(url: object, point: object) -> dict[str, object]:
    return {"camera_url": url, "camera_location": "x", "quadrant": "NE", "point": point}


def test_selection_reads_the_named_fields_and_resolves() -> None:
    point = {"type": "Point", "coordinates": [-114.07, 51.045]}
    records: list[object] = [
        _calgary_record("http://trafficcam.calgary.ca/loc1.jpg", point),
        _calgary_record("http://trafficcam.calgary.ca:8080/loc2.jpg", point),
        {"screenshot_address": "https://trafficcam.calgary.ca/loc3.jpg", "location": point},
        _calgary_record({"url": "https://trafficcam.calgary.ca/loc4.jpg"}, point),
    ]
    calgary = ph.CALGARY_CITY
    selection = ph.select_cameras(
        records,
        calgary.bbox,
        policy=calgary.image_policy,
        fields=(calgary.url_field, calgary.point_field),
    )
    assert selection.urls == [
        "https://trafficcam.calgary.ca/loc1.jpg",
        "https://trafficcam.calgary.ca/loc4.jpg",
    ]
    assert (selection.listed, selection.skipped, selection.refused) == (4, 1, 1)
    # Austin's fields stay the default: Calgary's records are malformed there.
    assert ph.select_cameras(records, calgary.bbox, policy=calgary.image_policy).skipped == 3


def _decoded(width: int, height: int) -> ph._Decoded:
    return ph._Decoded(np.zeros((height, width, 3), dtype=np.uint8), width, height)


def test_the_tally_counts_bands_at_its_frame_size() -> None:
    person = [detect.Detection("person", 0.9, (0.0, 0.0, 10.0, 90.0))]
    tally = ph.Tally(frame_size=ph.CALGARY_FRAME_SIZE)
    tally.add(_decoded(840, 630), person)
    tally.add(_decoded(1920, 1080), person)
    assert tally.bands == {"80-119": 2} and tally.bands_hd == {"80-119": 1}
    hd = ph.Tally()
    hd.add(_decoded(840, 630), person)
    assert hd.bands_hd == {}


def test_the_report_names_the_city_and_its_bands() -> None:
    selection = ph.Selection(listed=0, skipped=0, refused=0, urls=[])
    started = datetime.datetime(2026, 10, 1, 18, 0, tzinfo=datetime.UTC)
    calgary = ph.report(
        started_at=started,
        sun_elevation=1.0,
        model="m",
        selection=selection,
        tally=ph.Tally(),
        city=ph.CALGARY_CITY,
    )
    austin = ph.report(
        started_at=started, sun_elevation=1.0, model="m", selection=selection, tally=ph.Tally()
    )
    assert calgary["source"] == "calgary" and austin["source"] == "austin"
    assert list(calgary)[:-1] == list(austin)[:-1]
    assert list(calgary)[-1] == "persons_by_height_band_840x630"
    assert list(austin)[-1] == "persons_by_height_band_1080p"


def test_the_parser_defaults_to_austin() -> None:
    args = ph._parser().parse_args(["--live"])
    assert args.source == "austin" and args.bbox is None
    args = ph._parser().parse_args(["--live", "--source", "calgary"])
    assert args.source == "calgary"


def test_a_calgary_record_has_calgarys_light_and_source() -> None:
    record: dict[str, object] = {"date": "2026-12-21", "light": "day", "crops": []}
    dark = datetime.datetime(2026, 12, 21, 14, 0, 30, tzinfo=datetime.UTC)
    sourced = spotcheck.sourced_record(record, "calgary", dark)
    assert sourced == {"date": "2026-12-21", "light": "dark", "crops": [], "source": "calgary"}
    assert spotcheck.sourced_record(record, "austin", dark)["light"] == "day"


def test_calgary_file_names(tmp_path: Path) -> None:
    day = datetime.date(2026, 10, 1)
    names = [spotcheck.write_attributes({"a": 1}, tmp_path, day, "calgary").name for _ in range(2)]
    names.append(spotcheck.write_attributes({"a": 1}, tmp_path, day, "austin").name)
    names.append(spotcheck.write_attributes({"a": 1}, tmp_path, day).name)
    assert names == [
        "2026-10-01-calgary.json",
        "2026-10-01-calgary-2.json",
        "2026-10-01-austin.json",
        "2026-10-01.json",
    ]


def test_the_dry_run_still_has_the_frame_size() -> None:
    photo = encode_jpeg(np.full((400, 600, 3), 90, dtype=np.uint8))
    for size in (ph.CALGARY_FRAME_SIZE, ph.HD):
        still = spotcheck._hd_still(photo, size)
        assert ph.jpeg_size(still) == size
    image = cv2.imdecode(np.frombuffer(spotcheck._hd_still(photo), np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape == (1080, 1920, 3)


def test_the_summary_names_both_sources_in_its_refusal() -> None:
    raw = json.dumps({"source": "paris"}).encode()
    with pytest.raises(ValueError, match='"austin" or "calgary"'):
        spotcheck_summary.parse_labelling(raw)
    assert spotcheck_summary.SOURCES == ("london", "austin", "calgary")
