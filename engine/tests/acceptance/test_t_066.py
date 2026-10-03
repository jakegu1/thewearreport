"""Acceptance tests for T-066: the camera list's still URL given as an object
(`{"url": ..., "description": ...}`), as the City of Calgary's live list gives it. The
task contract: do not edit.

Every camera list here is written by the test. Where the spot-check tool's Calgary path
is under test, a fake transport replaces `registry.bounded_get` in this process and any
socket connection fails the test. Nothing reaches the network.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
from typing import Any

import pytest

from wearreport import registry
from wearreport.tools import pilot_heights as ph
from wearreport.tools import spotcheck

CALGARY = ph.CALGARY_CITY
AUSTIN = ph.AUSTIN_CITY
INSIDE = [-114.07, 51.045]  # lon, lat: inside the default downtown Calgary box
OUTSIDE = [-114.2, 51.1]
AUSTIN_INSIDE = [-97.745, 30.270]
DESCRIPTION = "Camera 87 SECRET-DESCRIPTION"
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")


def _point(coordinates: list[float]) -> dict[str, object]:
    return {"type": "Point", "coordinates": coordinates}


def _live(url: object, coordinates: list[float]) -> dict[str, object]:
    """A record in the live list's shape."""
    return {
        "camera_url": url,
        "camera_location": "Centre Street / 9 Avenue S",
        "quadrant": "SE",
        "point": _point(coordinates),
        ":@computed_region_4a3i_ccfj": "4",
        ":@computed_region_p8tp_5dkv": "7",
        ":@computed_region_kxmf_bzkv": "192",
    }


def _object(url: object, **extra: object) -> dict[str, object]:
    return {"url": url, "description": DESCRIPTION, **extra}


def _select_calgary(records: list[object]) -> ph.Selection:
    return ph.select_cameras(
        records,
        CALGARY.bbox,
        policy=CALGARY.image_policy,
        fields=(CALGARY.url_field, CALGARY.point_field),
    )


def _select_austin(records: list[object]) -> ph.Selection:
    return ph.select_cameras(
        records,
        AUSTIN.bbox,
        policy=AUSTIN.image_policy,
        fields=(AUSTIN.url_field, AUSTIN.point_field),
    )


# AC1: the object form ----------------------------------------------------------------


def test_calgary_object_url_is_read_and_resolved() -> None:
    records: list[object] = [
        _live(_object("http://trafficcam.calgary.ca/loc86.jpg"), INSIDE),
        _live(_object("https://trafficcam.calgary.ca/loc87.jpg"), INSIDE),
    ]
    selection = _select_calgary(records)
    assert selection.urls == [
        "https://trafficcam.calgary.ca/loc86.jpg",
        "https://trafficcam.calgary.ca/loc87.jpg",
    ]
    assert (selection.listed, selection.skipped, selection.refused) == (2, 0, 0)


def test_austin_object_url_is_read_too() -> None:
    record = {
        "screenshot_address": _object("https://cctv.austinmobility.io/image/1.jpg"),
        "location": _point(AUSTIN_INSIDE),
    }
    selection = _select_austin([record])
    assert selection.urls == ["https://cctv.austinmobility.io/image/1.jpg"]
    assert (selection.listed, selection.skipped, selection.refused) == (1, 0, 0)


def test_other_members_are_ignored_whatever_they_hold() -> None:
    odd: dict[str, object] = {
        "url": "http://trafficcam.calgary.ca/loc1.jpg",
        "description": None,
        "extra": [1, {"nested": True}],
        "url2": 7,
    }
    without_description = {"url": "http://trafficcam.calgary.ca/loc2.jpg"}
    selection = _select_calgary([_live(odd, INSIDE), _live(without_description, INSIDE)])
    assert selection.urls == [
        "https://trafficcam.calgary.ca/loc1.jpg",
        "https://trafficcam.calgary.ca/loc2.jpg",
    ]
    assert selection.skipped == 0


def test_the_description_never_reaches_the_selection() -> None:
    selection = _select_calgary([_live(_object("http://trafficcam.calgary.ca/loc1.jpg"), INSIDE)])
    assert DESCRIPTION not in repr(selection)


# AC2: the string form ----------------------------------------------------------------


def test_string_urls_are_read_as_before() -> None:
    records: list[object] = [
        _live("http://trafficcam.calgary.ca/loc1.jpg", INSIDE),
        _live("https://trafficcam.calgary.ca/loc2.jpg", INSIDE),
        _live("http://trafficcam.calgary.ca/loc3.jpg", OUTSIDE),
        _live("http://example.com/loc4.jpg", INSIDE),
    ]
    selection = _select_calgary(records)
    assert selection.urls == [
        "https://trafficcam.calgary.ca/loc1.jpg",
        "https://trafficcam.calgary.ca/loc2.jpg",
    ]
    assert (selection.listed, selection.skipped, selection.refused) == (4, 0, 1)
    austin = {
        "screenshot_address": "https://cctv.austinmobility.io/image/2.jpg",
        "location": _point(AUSTIN_INSIDE),
    }
    assert _select_austin([austin]).urls == ["https://cctv.austinmobility.io/image/2.jpg"]


# AC3: malformed URL fields are skipped -----------------------------------------------


def _deep(depth: int) -> object:
    value: object = "http://trafficcam.calgary.ca/loc1.jpg"
    for _ in range(depth):
        value = {"url": value}
    return value


def _deep_list(depth: int) -> object:
    value: object = "http://trafficcam.calgary.ca/loc1.jpg"
    for _ in range(depth):
        value = [value]
    return value


MALFORMED: list[object] = [
    {},
    {"description": DESCRIPTION},
    {"URL": "http://trafficcam.calgary.ca/loc1.jpg"},
    {"url": None},
    {"url": 86},
    {"url": 8.6},
    {"url": True},
    {"url": ["http://trafficcam.calgary.ca/loc1.jpg"]},
    {"url": {"url": "http://trafficcam.calgary.ca/loc1.jpg"}},
    ["http://trafficcam.calgary.ca/loc1.jpg"],
    [],
    86,
    8.6,
    float("inf"),
    True,
    None,
    _deep(2),
    _deep(100_000),
    _deep_list(100_000),
]


@pytest.mark.parametrize("url", MALFORMED, ids=range(len(MALFORMED)))
def test_a_malformed_url_field_is_skipped(url: object) -> None:
    for record in (_live(url, INSIDE), _live(url, OUTSIDE)):
        selection = _select_calgary([record])
        assert (selection.listed, selection.skipped, selection.refused) == (1, 1, 0)
        assert selection.urls == []


def test_a_missing_url_field_is_skipped() -> None:
    record = _live("x", INSIDE)
    del record["camera_url"]
    selection = _select_calgary([record])
    assert (selection.skipped, selection.urls) == (1, [])


def test_malformed_records_parsed_from_json_are_skipped() -> None:
    body = json.dumps(
        [
            _live({"url": None, "description": "d"}, INSIDE),
            _live({"url": 1}, INSIDE),
            _live([], INSIDE),
            _live(None, INSIDE),
            _live(_deep(200), INSIDE),
            _live(_object("http://trafficcam.calgary.ca/loc1.jpg"), INSIDE),
        ]
    ).encode()
    selection = _select_calgary(ph.decode_dataset(body))
    assert selection.urls == ["https://trafficcam.calgary.ca/loc1.jpg"]
    assert (selection.listed, selection.skipped, selection.refused) == (6, 5, 0)


# AC4: the URL policy is unchanged ----------------------------------------------------


REFUSED: list[str] = [
    "http://example.com/loc1.jpg",
    "https://example.com/loc1.jpg",
    "http://trafficcam.calgary.ca:8080/loc1.jpg",
    "http://user@trafficcam.calgary.ca/loc1.jpg",
    "ftp://trafficcam.calgary.ca/loc1.jpg",
    "//trafficcam.calgary.ca/loc1.jpg",
    "http://trafficcam.calgary.ca.example.com/loc1.jpg",
    "http://trafficcam.calgary.ca/loc 1.jpg",
    "http://trafficcam.calgary.ca\\loc1.jpg",
    "",
    "http://trafficcam.calgary.ca/" + "a" * 3000,
]


@pytest.mark.parametrize("url", REFUSED, ids=range(len(REFUSED)))
def test_an_object_url_off_policy_is_refused_and_counted(url: str) -> None:
    inside = _select_calgary([_live(_object(url), INSIDE)])
    assert (inside.listed, inside.skipped, inside.refused, inside.urls) == (1, 0, 1, [])
    as_string = _select_calgary([_live(url, INSIDE)])
    assert (as_string.skipped, as_string.refused, as_string.urls) == (0, 1, [])
    outside = _select_calgary([_live(_object(url), OUTSIDE)])
    assert (outside.skipped, outside.refused, outside.urls) == (0, 0, [])


def test_object_and_string_urls_resolve_identically() -> None:
    urls = [
        "http://trafficcam.calgary.ca/loc1.jpg",
        "HTTP://trafficcam.calgary.ca/loc2.jpg",
        "https://trafficcam.calgary.ca/loc3.jpg",
        *REFUSED,
    ]
    for url in urls:
        assert _select_calgary([_live(_object(url), INSIDE)]) == _select_calgary(
            [_live(url, INSIDE)]
        )


def test_austin_object_url_keeps_austins_policy() -> None:
    record = {
        "screenshot_address": _object("http://cctv.austinmobility.io/image/1.jpg"),
        "location": _point(AUSTIN_INSIDE),
    }
    selection = _select_austin([record])  # Austin's policy has no http upgrade
    assert (selection.skipped, selection.refused, selection.urls) == (0, 1, [])


# AC5: the live shape, through select_cameras and the spot-check's Calgary path --------


def _live_list() -> list[object]:
    return [
        _live(_object("http://trafficcam.calgary.ca/loc86.jpg"), INSIDE),
        _live(_object("http://trafficcam.calgary.ca/loc1.jpg"), OUTSIDE),
        _live(_object("http://trafficcam.calgary.ca/loc87.jpg"), [-114.06, 51.05]),
        _live(_object("http://trafficcam.calgary.ca/loc2.jpg"), [-113.9, 51.0]),
        _live(_object("http://trafficcam.calgary.ca/loc88.jpg"), [-114.08, 51.041]),
    ]


INSIDE_URLS = [
    "https://trafficcam.calgary.ca/loc86.jpg",
    "https://trafficcam.calgary.ca/loc87.jpg",
    "https://trafficcam.calgary.ca/loc88.jpg",
]


def test_the_live_shape_selects_the_inside_cameras() -> None:
    selection = _select_calgary(_live_list())
    assert selection.urls == INSIDE_URLS
    assert (selection.listed, selection.skipped, selection.refused) == (5, 0, 0)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No socket may connect anywhere, and no proxy is configured."""
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)

    def connect(self: socket.socket, address: Any) -> None:
        raise AssertionError("connection attempted")

    monkeypatch.setattr(socket.socket, "connect", connect)


class Transport:
    """Replaces registry.bounded_get: answers from `bodies` (URL -> body, or an HTTP
    status for an error) and records every URL asked for."""

    def __init__(self, bodies: dict[str, bytes | int]) -> None:
        self.bodies = bodies
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def __call__(
        self, url: str, *, timeout_s: float, max_bytes: int, schemes: tuple[str, ...] = ("https",)
    ) -> bytes:
        with self.lock:
            self.calls.append(url)
        assert timeout_s > 0 and max_bytes > 0 and schemes == ("https",)
        answer = self.bodies.get(url, 404)
        if isinstance(answer, int):
            raise urllib.error.HTTPError(url, answer, "fake", None, None)  # type: ignore[arg-type]
        return answer


def test_the_live_shape_through_the_spotcheck_calgary_path(
    monkeypatch: pytest.MonkeyPatch, offline: None, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = Transport({ph.CALGARY_DATASET_URL: json.dumps(_live_list()).encode()})
    monkeypatch.setattr(registry, "bounded_get", transport)
    frames = list(
        spotcheck.calgary_frames(spotcheck.CalgaryEndpoints(), CALGARY.bbox, timeout_s=10.0)
    )
    assert frames == []  # every still answered 404: failed, never fetched elsewhere
    assert transport.calls[0] == ph.CALGARY_DATASET_URL
    assert sorted(transport.calls[1:]) == INSIDE_URLS
    err = capsys.readouterr().err
    assert (
        "spotcheck: 5 Calgary cameras listed, 3 selected "
        "(0 malformed record(s) skipped, 0 URL(s) refused)"
    ) in err
    for secret in (DESCRIPTION, "trafficcam", "loc86", "51.045", "Centre Street"):
        assert secret not in err


def test_an_object_url_on_another_host_is_refused_not_fetched(
    monkeypatch: pytest.MonkeyPatch, offline: None, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [
        _live(_object("http://trafficcam.calgary.ca/loc86.jpg"), INSIDE),
        _live(_object("http://example.com/loc87.jpg"), INSIDE),
    ]
    transport = Transport({ph.CALGARY_DATASET_URL: json.dumps(records).encode()})
    monkeypatch.setattr(registry, "bounded_get", transport)
    list(spotcheck.calgary_frames(spotcheck.CalgaryEndpoints(), CALGARY.bbox, timeout_s=10.0))
    assert transport.calls == [ph.CALGARY_DATASET_URL, "https://trafficcam.calgary.ca/loc86.jpg"]
    err = capsys.readouterr().err
    assert (
        "2 Calgary cameras listed, 1 selected (0 malformed record(s) skipped, 1 URL(s) refused)"
        in err
    )
    assert "example.com" not in err and DESCRIPTION not in err


# AC6: nothing printed beyond counts --------------------------------------------------


def test_selection_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    _select_calgary([*_live_list(), *[_live(url, INSIDE) for url in MALFORMED]])
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
