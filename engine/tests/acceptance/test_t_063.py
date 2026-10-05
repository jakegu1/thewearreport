"""Acceptance tests for T-063: Calgary as a test camera source for the height pilot
(`pilot_heights --live --source calgary`), the attribute session
(`spotcheck --attributes --source calgary`) and the attribute summary. The task contract:
do not edit.

Every frame here is synthetic (uniform colours with a gradient, encoded in memory), every
camera list is written by the test, and both are served by a local HTTP server on
127.0.0.1, or, where the pinned Calgary hosts themselves are under test, handed back by a
fake transport that replaces `registry.bounded_get` in this process. The detector and the
reviewers are scripted. Nothing reaches the network.
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import aggregate, detect, registry
from wearreport._cv import cv2, encode_jpeg
from wearreport.testing.fake_cameras import jpeg_declaring
from wearreport.tools import pilot_heights, spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
REAL_DIR = ROOT / "spotchecks"
README = REAL_DIR / "README.md"
REAL_FILES = ("2026-09-30.json", "2026-09-30-2.json", "2026-10-01.json", "2026-10-01-2.json")
DAY = datetime.date(2026, 10, 1)
# Sun elevations (Calgary, Austin, London), degrees, by the tools' own formula; checked
# in test_the_clocks_are_what_the_tests_say.
ALL_DAY = datetime.datetime(2026, 10, 1, 16, 0, 12, tzinfo=datetime.UTC)  # 20, 43, 14
CALGARY_DARK = datetime.datetime(2026, 12, 21, 14, 0, tzinfo=datetime.UTC)  # -14, 6, 10
CALGARY_ONLY_DAY = datetime.datetime(2026, 6, 21, 3, 0, tzinfo=datetime.UTC)  # 6, -16, -5.5
LONDON_DARK = datetime.datetime(2026, 10, 1, 21, 0, 40, tzinfo=datetime.UTC)  # 32, 39, -30
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")

CALGARY_DATASET = "https://data.calgary.ca/resource/k7p9-kppz.json"
CALGARY_IMAGES = "trafficcam.calgary.ca"
INSIDE = (-114.07, 51.045)  # lon, lat: inside the default downtown Calgary box
OUTSIDE = (-114.2, 51.1)
AUSTIN_INSIDE = (-97.745, 30.270)
FIELDS = {
    "date",
    "started_at",
    "light",
    "frames",
    "detector",
    "min_height_px",
    "judge",
    "crops_shown",
    "crops_rejected",
    "crops",
}
PILOT_KEYS = [
    "source",
    "started_at",
    "sun_elevation_deg",
    "model",
    "cameras_listed",
    "cameras_selected",
    "records_skipped",
    "refused_url",
    "frames_ok",
    "frames_failed",
    "resolutions",
    "persons_total",
    "umbrellas_total",
    "persons_by_height_band",
    "persons_by_height_band_840x630",
]
BANDS = ["<31", "31-45", "46-79", "80-119", "120-199", "200+"]
DARK_CALGARY = (
    "spotcheck: it is dark in Calgary now (sun below -6°); attribute sessions need daylight. "
    "Use --allow-dark to run anyway."
)

Frame = npt.NDArray[np.uint8]
Box = tuple[float, float, float, float]


# A local Calgary site: the camera list and the stills --------------------------------


@dataclass
class Route:
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


class Site:
    """Serves /dataset (the camera list) and /image/<name> on 127.0.0.1."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.requests: list[str] = []
        self.lock = threading.Lock()
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                with site.lock:
                    site.requests.append(self.path)
                    route = site.routes.get(self.path, Route(404))
                self.send_response(route.status)
                for name, value in route.headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(route.body)))
                self.end_headers()
                self.wfile.write(route.body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def netloc(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"{host!s}:{port}"

    def url(self, path: str) -> str:
        return f"http://{self.netloc}{path}"

    def policy(self) -> pilot_heights.UrlPolicy:
        return pilot_heights.UrlPolicy("http", self.netloc)

    def calgary(self) -> spotcheck.CalgaryEndpoints:
        return spotcheck.CalgaryEndpoints(self.url("/dataset"), self.policy(), self.policy())

    def austin(self) -> spotcheck.AustinEndpoints:
        return spotcheck.AustinEndpoints(self.url("/dataset"), self.policy(), self.policy())

    def image(self, name: str, body: bytes, **kwargs: Any) -> str:
        self.routes[f"/image/{name}"] = Route(body=body, **kwargs)
        return self.url(f"/image/{name}")

    def dataset(self, records: object) -> None:
        self.routes["/dataset"] = Route(body=json.dumps(records).encode())

    def count(self, path: str) -> int:
        with self.lock:
            return self.requests.count(path)

    def images(self) -> list[str]:
        with self.lock:
            return [p for p in self.requests if p.startswith("/image/")]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def site() -> Iterator[Site]:
    s = Site()
    try:
        yield s
    finally:
        s.close()


def _pixels(width: int, height: int, key: int) -> Frame:
    """A frame whose blue channel encodes `key` (read back by Stub), with a gradient in
    green so that crops differ."""
    frame = np.full((height, width, 3), 10 * key + 5, dtype=np.uint8)
    frame[:, :, 1] = (np.arange(width) // 8 % 200).astype(np.uint8)[None, :]
    return frame


def _jpeg(width: int, height: int, key: int) -> bytes:
    return encode_jpeg(_pixels(width, height, key))


def _decoded(body: bytes) -> Frame:
    return np.asarray(cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR), np.uint8)


def _cal(url: object, lon: object = INSIDE[0], lat: object = INSIDE[1]) -> dict[str, Any]:
    """A record of Calgary's camera list."""
    return {
        "camera_url": url,
        "camera_location": "Test Street & 1 Avenue",
        "quadrant": "SW",
        "point": {"type": "Point", "coordinates": [lon, lat]},
    }


def _aus(url: object) -> dict[str, Any]:
    """A record of Austin's camera list."""
    return {
        "camera_id": "1",
        "screenshot_address": url,
        "location": {"type": "Point", "coordinates": list(AUSTIN_INSIDE)},
    }


class Stub:
    """boxes[key] are the detections in a frame whose blue channel encodes `key`; shapes
    lists the shape of every frame that reached the detector."""

    def __init__(self, boxes: dict[int, Sequence[tuple[detect.Label, Box]]] | None = None) -> None:
        self.boxes = boxes or {}
        self.shapes: list[tuple[int, ...]] = []
        self.lock = threading.Lock()

    def detect(self, frame: Frame) -> list[detect.Detection]:
        with self.lock:
            self.shapes.append(frame.shape)
        key = int(frame[frame.shape[0] // 2, 4, 0]) // 10
        return [detect.Detection(label, 0.9, box) for label, box in self.boxes.get(key, [])]


def _persons(*boxes: Box) -> list[tuple[detect.Label, Box]]:
    return [("person", box) for box in boxes]


# Person boxes in full-resolution pixels of an 840x630 still, by height.
B20 = (700.0, 100.0, 710.0, 120.0)
B90 = (400.0, 300.0, 440.0, 390.0)
B200 = (100.0, 200.0, 160.0, 400.0)
B500 = (600.0, 50.0, 760.0, 550.0)


def _scan_bomb(body: bytes) -> bytes:
    """`body` with a comment segment holding 33 start-of-scan markers after its
    start-of-image marker: more than the 32-scan limit, the header still readable."""
    comment = pilot_heights.JPEG_SOS * 33
    segment = b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment
    return body[:2] + segment + body[2:]


def _unlisted_marker(body: bytes) -> bytes:
    """`body` with a segment the header reader does not allow before the frame header."""
    return body[:2] + b"\xff\xf0\x00\x04ab" + body[2:]


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only 127.0.0.1 may be connected to, with no proxy in between."""
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")):
            raise AssertionError(f"non-local connection attempted: {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


class Transport:
    """Replaces registry.bounded_get: answers from `bodies` (URL -> body, or an HTTP
    status for an error) and records every URL asked for and the schemes allowed."""

    def __init__(self, bodies: dict[str, bytes | int]) -> None:
        self.bodies = bodies
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.lock = threading.Lock()

    def __call__(
        self, url: str, *, timeout_s: float, max_bytes: int, schemes: tuple[str, ...] = ("https",)
    ) -> bytes:
        with self.lock:
            self.calls.append((url, tuple(schemes)))
        assert timeout_s > 0 and max_bytes > 0
        answer = self.bodies.get(url, 404)
        if isinstance(answer, int):
            raise urllib.error.HTTPError(url, answer, "fake", None, None)  # type: ignore[arg-type]
        return answer

    def urls(self) -> list[str]:
        with self.lock:
            return [url for url, _schemes in self.calls]


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch, offline: None) -> Transport:
    fake = Transport({})
    monkeypatch.setattr(registry, "bounded_get", fake)
    return fake


# The clocks -------------------------------------------------------------------------


def test_the_clocks_are_what_the_tests_say() -> None:
    def sun(moment: datetime.datetime, where: tuple[float, float]) -> float:
        return pilot_heights.solar_elevation(moment, *where)

    calgary, austin, london = pilot_heights.CALGARY, pilot_heights.AUSTIN, spotcheck.LONDON
    assert min(sun(ALL_DAY, calgary), sun(ALL_DAY, austin), sun(ALL_DAY, london)) > 0
    assert sun(CALGARY_DARK, calgary) < -6 < 0 < sun(CALGARY_DARK, austin)
    assert sun(CALGARY_DARK, london) > 0
    assert sun(CALGARY_ONLY_DAY, austin) < -6 < sun(CALGARY_ONLY_DAY, london) < 0
    assert sun(CALGARY_ONLY_DAY, calgary) > 0
    assert sun(LONDON_DARK, london) < -6 < 0 < sun(LONDON_DARK, calgary)


# AC1: the pilot -----------------------------------------------------------------------


def _pilot(
    site: Site,
    argv: Sequence[str] = ("--live", "--source", "calgary"),
    *,
    stub: Stub | None = None,
    now: datetime.datetime = ALL_DAY,
    opened: list[str] | None = None,
) -> int:
    detector = stub or Stub()

    def open_detector(model: str) -> Stub:
        if opened is not None:
            opened.append(model)
        return detector

    return pilot_heights.main(
        list(argv),
        now=lambda: now,
        open_detector=open_detector,
        dataset_url=site.url("/dataset"),
        dataset_policy=site.policy(),
        image_policy=site.policy(),
    )


def _output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1, out
    result = json.loads(lines[0])
    assert isinstance(result, dict)
    return result


def test_ac1_calgary_is_pinned() -> None:
    assert pilot_heights.CALGARY_DATASET_URL == CALGARY_DATASET
    assert (
        pilot_heights.UrlPolicy("https", "data.calgary.ca") == pilot_heights.CALGARY_DATASET_POLICY
    )
    assert pilot_heights.CALGARY_DATASET_POLICY.allows(CALGARY_DATASET)
    assert pilot_heights.CALGARY_IMAGE_POLICY.scheme == "https"
    assert pilot_heights.CALGARY_IMAGE_POLICY.netloc == CALGARY_IMAGES
    assert pilot_heights.CALGARY_BBOX == (51.040, -114.095, 51.056, -114.045)
    assert pilot_heights.CALGARY == (51.0447, -114.0719)
    assert pilot_heights.CALGARY_FRAME_SIZE == (840, 630)


def test_ac1_the_sources_and_the_default() -> None:
    assert pilot_heights.SOURCES == ("austin", "calgary")
    assert pilot_heights.SOURCE == "austin"


@pytest.mark.parametrize("value", ["paris", "Calgary", "london", ""])
def test_ac1_an_unknown_source_exits_2_before_any_request(
    site: Site, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    opened: list[str] = []
    assert _pilot(site, ["--live", "--source", value], opened=opened) == 2
    assert capsys.readouterr().out == ""
    assert site.requests == [] and opened == []


def test_ac1_calgary_still_needs_live(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    assert _pilot(site, ["--source", "calgary"]) == 2
    captured = capsys.readouterr()
    assert "--live" in captured.err and captured.out == ""
    assert site.requests == []


@pytest.mark.parametrize(
    "url",
    [
        "http://trafficcam.calgary.ca/loc142.jpg",
        "https://trafficcam.calgary.ca/loc142.jpg",
    ],
)
def test_ac1_the_image_host_is_fetched_over_https_only(url: str) -> None:
    policy = pilot_heights.CALGARY_IMAGE_POLICY
    assert policy.resolve(url) == "https://trafficcam.calgary.ca/loc142.jpg"


@pytest.mark.parametrize(
    "url",
    [
        "http://trafficcam.calgary.ca:80/loc142.jpg",
        "https://trafficcam.calgary.ca:443/loc142.jpg",
        "http://trafficcam.calgary.ca:8080/loc142.jpg",
        "http://user@trafficcam.calgary.ca/loc142.jpg",
        "https://user:pw@trafficcam.calgary.ca/loc142.jpg",
        "http://trafficcam.calgary.ca.evil.example/loc142.jpg",
        "https://evil.example/trafficcam.calgary.ca/loc142.jpg",
        "http://sub.trafficcam.calgary.ca/loc142.jpg",
        "http://data.calgary.ca/loc142.jpg",
        "ftp://trafficcam.calgary.ca/loc142.jpg",
        "file:///etc/passwd",
        "//trafficcam.calgary.ca/loc142.jpg",
        "trafficcam.calgary.ca/loc142.jpg",
        "http://trafficcam.calgary.ca\\@evil.example/loc142.jpg",
        "http://trafficcam.calgary.ca/loc 142.jpg",
        "http://trafficcam.calgary.ca/loc\n142.jpg",
        "http://cctv.austinmobility.io/image/1.jpg",
        "",
    ],
)
def test_ac1_any_other_image_url_is_refused(url: str) -> None:
    assert pilot_heights.CALGARY_IMAGE_POLICY.resolve(url) is None
    assert not pilot_heights.CALGARY_IMAGE_POLICY.allows(url)


def test_ac1_austins_policies_do_not_upgrade() -> None:
    policy = pilot_heights.SCREENSHOT_POLICY
    assert policy.resolve("http://cctv.austinmobility.io/image/1.jpg") is None
    assert policy.resolve("https://cctv.austinmobility.io/image/1.jpg") == (
        "https://cctv.austinmobility.io/image/1.jpg"
    )


def _calgary_list(urls: Sequence[str]) -> bytes:
    return json.dumps([_cal(url) for url in urls]).encode()


def test_ac1_the_pinned_hosts_and_the_https_upgrade(
    transport: Transport, capsys: pytest.CaptureFixture[str]
) -> None:
    listed = [
        "http://trafficcam.calgary.ca/loc1.jpg",  # fetched as https
        "https://trafficcam.calgary.ca/loc2.jpg",
        "http://trafficcam.calgary.ca:8080/loc3.jpg",  # refused: a port
        "http://user@trafficcam.calgary.ca/loc4.jpg",  # refused: userinfo
        "https://cctv.example/loc5.jpg",  # refused: another host
        "ftp://trafficcam.calgary.ca/loc6.jpg",  # refused: another scheme
        "https://trafficcam.calgary.ca/loc7.jpg",  # a redirect: refused, not followed
    ]
    transport.bodies.update(
        {
            CALGARY_DATASET: _calgary_list(listed),
            "https://trafficcam.calgary.ca/loc1.jpg": _jpeg(840, 630, 1),
            "https://trafficcam.calgary.ca/loc2.jpg": _jpeg(840, 630, 1),
            "https://trafficcam.calgary.ca/loc7.jpg": 302,
        }
    )
    code = pilot_heights.main(
        ["--live", "--source", "calgary"],
        now=lambda: ALL_DAY,
        open_detector=lambda model: Stub({1: _persons(B90)}),
    )
    assert code == 0
    result = _output(capsys)
    urls = transport.urls()
    assert urls[0] == CALGARY_DATASET
    assert sorted(urls[1:]) == [
        "https://trafficcam.calgary.ca/loc1.jpg",
        "https://trafficcam.calgary.ca/loc2.jpg",
        "https://trafficcam.calgary.ca/loc7.jpg",
    ]
    assert all(schemes == ("https",) for _url, schemes in transport.calls)
    assert result["source"] == "calgary"
    assert result["cameras_listed"] == 7 and result["cameras_selected"] == 3
    assert result["refused_url"] == 5  # four by the policy, one redirect
    assert result["frames_ok"] == 2 and result["persons_total"] == 2


def test_ac1_the_default_box_is_downtown_calgary(
    transport: Transport, capsys: pytest.CaptureFixture[str]
) -> None:
    s, w, n, e = pilot_heights.CALGARY_BBOX
    points = {
        "centre": INSIDE,
        "sw": (w, s),
        "ne": (e, n),
        "west": (w - 0.0001, 51.045),
        "north": (-114.07, n + 0.0001),
        "far": OUTSIDE,
        "austin": AUSTIN_INSIDE,
    }
    records = [
        _cal(f"https://trafficcam.calgary.ca/{name}.jpg", lon, lat)
        for name, (lon, lat) in points.items()
    ]
    transport.bodies[CALGARY_DATASET] = json.dumps(records).encode()
    code = pilot_heights.main(
        ["--live", "--source", "calgary"], now=lambda: ALL_DAY, open_detector=lambda m: Stub()
    )
    assert code == 0
    result = _output(capsys)
    assert result["cameras_listed"] == 7 and result["cameras_selected"] == 3
    assert sorted(transport.urls()[1:]) == [
        f"https://trafficcam.calgary.ca/{name}.jpg" for name in ("centre", "ne", "sw")
    ]


def test_ac1_bbox_is_still_accepted(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    far = site.image("far", _jpeg(840, 630, 1))
    near = site.image("near", _jpeg(840, 630, 1))
    site.dataset([_cal(far, *OUTSIDE), _cal(near)])
    assert (
        _pilot(site, ["--live", "--source", "calgary", "--bbox", "51.09,-114.21,51.11,-114.19"])
        == 0
    )
    result = _output(capsys)
    assert result["cameras_selected"] == 1
    assert site.images() == ["/image/far"]


def _malformed(url: str) -> list[object]:
    deep: object = 1.0
    for _ in range(200):
        deep = [deep]
    return [
        None,
        7,
        "camera",
        [url],
        {"camera_url": url},  # no point
        {"point": {"type": "Point", "coordinates": list(INSIDE)}},  # no URL
        _aus(url),  # Austin's field names
        _cal(42),
        _cal(None),
        _cal(["x"]),
        {"camera_url": url, "point": "51.045,-114.07"},
        {"camera_url": url, "point": {"type": "Point"}},
        {"camera_url": url, "point": {"type": "Polygon", "coordinates": list(INSIDE)}},
        {"camera_url": url, "point": {"type": "Point", "coordinates": [INSIDE[0]]}},
        {"camera_url": url, "point": {"type": "Point", "coordinates": [*INSIDE, 1, 2]}},
        {"camera_url": url, "point": {"type": "Point", "coordinates": "x"}},
        {"camera_url": url, "point": {"type": "Point", "coordinates": deep}},
        {"camera_url": url, "point": deep},
        _cal(url, str(INSIDE[0]), str(INSIDE[1])),
        _cal(url, True, INSIDE[1]),
        _cal(url, INSIDE[0], None),
        _cal(url, INSIDE[0], 91.0),
        _cal(url, -181.0, INSIDE[1]),
        _cal(url, INSIDE[0], {"lat": 51.045}),
    ]


def test_ac1_malformed_records_are_skipped_and_counted(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    good = site.image("good", _jpeg(840, 630, 1))
    bad = _malformed(site.image("bad", _jpeg(840, 630, 1)))
    body = json.dumps([*bad, _cal(good)])
    extra = [
        f'{{"camera_url": "{good}", "point": {{"type": "Point", "coordinates": [{lon}, {lat}]}}}}'
        for lon, lat in [
            ("NaN", "51.045"),
            ("-114.07", "Infinity"),
            ("-Infinity", "51.045"),
            ("1e999", "51.045"),
            ("-114.07", "1" + "0" * 5000),
        ]
    ]
    site.routes["/dataset"] = Route(body=(body[:-1] + ", " + ", ".join(extra) + "]").encode())
    assert _pilot(site) == 0
    result = _output(capsys)
    assert result["cameras_listed"] == len(bad) + 1 + len(extra)
    assert result["records_skipped"] == len(bad) + len(extra)
    assert result["cameras_selected"] == 1 and result["frames_ok"] == 1
    assert site.images() == ["/image/good"]


@pytest.mark.parametrize(
    "body",
    [b"{}", b'"cameras"', b"null", b"[1, 2", b"\xff\xfe\x00[", b"[" * 100_000 + b"]" * 100_000],
)
def test_ac1_a_list_that_is_not_an_array_is_a_typed_error(
    site: Site, capsys: pytest.CaptureFixture[str], body: bytes
) -> None:
    site.routes["/dataset"] = Route(body=body)
    assert _pilot(site) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "Traceback" not in captured.err and captured.err.strip()
    assert site.images() == []


def test_ac1_the_body_cap_holds(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    site.routes["/dataset"] = Route(body=b"[" + b" " * pilot_heights.MAX_DATASET_BYTES + b"]")
    assert _pilot(site) == 1
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("status", [301, 302, 307, 404, 500])
def test_ac1_list_redirects_and_errors_are_typed_errors(
    site: Site, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    site.routes["/dataset"] = Route(status, b"[]", {"Location": site.url("/other")})
    site.routes["/other"] = Route(body=b"[]")
    assert _pilot(site) == 1
    assert capsys.readouterr().out == ""
    assert site.requests == ["/dataset"]  # one request, the redirect not followed


def test_ac1_a_list_on_another_host_is_refused(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    site.dataset([])
    code = pilot_heights.main(
        ["--live", "--source", "calgary"],
        now=lambda: ALL_DAY,
        open_detector=lambda model: Stub(),
        dataset_url=site.url("/dataset"),
        dataset_policy=pilot_heights.CALGARY_DATASET_POLICY,
        image_policy=site.policy(),
    )
    assert code == 1
    assert site.requests == []
    assert capsys.readouterr().out == ""


def test_ac1_refused_image_urls_are_counted_and_never_requested(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = site.image("ok", _jpeg(840, 630, 1))
    host, port = site.netloc.split(":")
    refused = [
        f"http://localhost:{port}/image/ok",
        f"http://user:pw@{site.netloc}/image/ok",
        f"http://user@{site.netloc}/image/ok",
        f"http://{host}:{int(port) + 1}/image/ok",
        f"http://{host}/image/ok",
        f"ftp://{site.netloc}/image/ok",
    ]
    moved = [
        site.image(f"moved{code}", b"", status=code, headers={"Location": ok})
        for code in (301, 302, 303, 307, 308)
    ]
    site.dataset([_cal(ok), *[_cal(u) for u in refused], *[_cal(u) for u in moved]])
    assert _pilot(site) == 0
    result = _output(capsys)
    assert result["refused_url"] == len(refused) + len(moved)
    assert result["cameras_selected"] == 1 + len(moved) and result["frames_ok"] == 1
    assert sum(result["frames_failed"].values()) == 0
    assert site.count("/image/ok") == 1
    for code in (301, 302, 303, 307, 308):
        assert site.count(f"/image/moved{code}") == 1


def test_ac1_the_header_rules_apply(
    site: Site, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    good = _jpeg(840, 630, 1)
    bodies = {
        "good": good,
        "declared": jpeg_declaring(30000, 30000),  # over MAX_HEADER_PIXELS
        "scans": _scan_bomb(good),  # more than 32 scans
        "unlisted": _unlisted_marker(good),  # a marker outside the allowlist
    }
    decoded: list[int] = []
    real = cv2.imdecode

    def imdecode(buf: Any, flags: int) -> Any:
        decoded.append(len(buf))
        return real(buf, flags)

    monkeypatch.setattr(cv2, "imdecode", imdecode)
    site.dataset([_cal(site.image(name, body)) for name, body in bodies.items()])
    assert _pilot(site) == 0
    result = _output(capsys)
    assert result["frames_ok"] == 1
    assert result["frames_failed"]["decode"] == 3
    assert decoded == [len(good)]  # nothing else reached the decoder
    for name in bodies:
        assert site.count(f"/image/{name}") == 1


def test_ac1_the_output_keys_and_their_order(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    site.dataset([])
    assert _pilot(site) == 0
    result = _output(capsys)
    assert list(result) == PILOT_KEYS
    assert result["source"] == "calgary"
    assert list(result["persons_by_height_band"]) == BANDS
    assert list(result["persons_by_height_band_840x630"]) == BANDS
    assert "persons_by_height_band_1080p" not in result


def test_ac1_bands_at_840x630_count_exact_840x630_frames_only(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = Stub(
        {
            1: [*_persons(B20, B90, B200), ("umbrella", B90)],
            2: _persons(B90),
            3: _persons(B200),
            4: _persons(B500),
        }
    )
    urls = [site.image(f"cal{i}", _jpeg(840, 630, 1)) for i in range(2)]
    urls.append(site.image("hd", _jpeg(1920, 1080, 2)))
    urls.append(site.image("wide", _jpeg(841, 630, 3)))
    urls.append(site.image("tall", _jpeg(840, 631, 4)))
    site.dataset([_cal(u) for u in urls])
    assert _pilot(site, stub=stub) == 0
    result = _output(capsys)
    assert result["frames_ok"] == 5
    assert set(stub.shapes) >= {(630, 840, 3)}
    assert result["umbrellas_total"] == 2
    assert result["persons_total"] == 2 * 3 + 1 + 1 + 1
    exact = dict.fromkeys(BANDS, 0)
    for box in (B20, B90, B200):
        exact[pilot_heights.height_band(aggregate.box_height(box))] += 2
    assert result["persons_by_height_band_840x630"] == exact
    assert sum(result["persons_by_height_band"].values()) == result["persons_total"]
    assert result["resolutions"]["840x630"] == 2


def test_ac1_failures_are_counted_by_kind(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    urls = [
        site.image("ok", _jpeg(840, 630, 1)),
        site.image("missing", b"", status=404),
        site.image("broken", b"", status=500),
        site.image("text", b"<html>no image</html>"),
        site.image("truncated", _jpeg(840, 630, 1)[:-200]),
    ]
    site.dataset([_cal(u) for u in urls])
    assert _pilot(site) == 0
    result = _output(capsys)
    assert result["frames_ok"] == 1
    assert result["frames_failed"]["http"] == 2 and result["frames_failed"]["decode"] == 2
    for name in ("ok", "missing", "broken", "text", "truncated"):
        assert site.count(f"/image/{name}") == 1


def test_ac1_refuses_in_the_calgary_night_though_austin_is_in_daylight(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    site.dataset([_cal(site.image("c", _jpeg(840, 630, 1)))])
    opened: list[str] = []
    assert _pilot(site, now=CALGARY_DARK, opened=opened) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Calgary" in captured.err and "Austin" not in captured.err
    assert "sun" in captured.err.lower()
    assert site.requests == [] and opened == []


def test_ac1_runs_in_the_calgary_day_though_austin_is_dark(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    site.dataset([])
    assert _pilot(site, now=CALGARY_ONLY_DAY) == 0
    result = _output(capsys)
    expected = pilot_heights.solar_elevation(CALGARY_ONLY_DAY, 51.0447, -114.0719)
    assert result["sun_elevation_deg"] == round(expected, 1)
    assert result["started_at"] == "2026-06-21T03:00Z"


def test_ac1_counts_only_and_nothing_on_disk(
    site: Site,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def tree(root: Path) -> set[str]:
        return {str(p.relative_to(root)) for p in root.rglob("*")}

    temp, work = tmp_path / "tmp", tmp_path / "work"
    temp.mkdir()
    work.mkdir()
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(temp))
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    monkeypatch.chdir(work)
    caplog.set_level("DEBUG")
    urls = [site.image(f"secretcam{i}", _jpeg(840, 630, 1)) for i in range(3)]
    site.dataset([_cal(u) for u in urls])
    before = (tree(temp), tree(work), tree(ROOT / "engine"))
    assert _pilot(site, stub=Stub({1: [("person", (12.25, 33.5, 47.75, 101.125))]})) == 0
    assert (tree(temp), tree(work), tree(ROOT / "engine")) == before
    captured = capsys.readouterr()
    everything = captured.out + captured.err + caplog.text
    for needle in ("secretcam", "/image/", site.netloc, "12.25", "101.125", "-114.07", "51.045"):
        assert needle not in everything
    assert json.loads(captured.out)["frames_ok"] == 3


# Austin's output stays byte for byte as it was: one fixed pass, printed by the code on
# main before this task.
AUSTIN_PILOT_OUTPUT = (
    json.dumps(
        {
            "source": "austin",
            "started_at": "2026-10-01T16:00Z",
            "sun_elevation_deg": 42.5,
            "model": "yolox_m",
            "cameras_listed": 5,
            "cameras_selected": 3,
            "records_skipped": 1,
            "refused_url": 1,
            "frames_ok": 2,
            "frames_failed": {"timeout": 0, "http": 1, "decode": 0, "network": 0, "detector": 0},
            "resolutions": {"1920x1080": 2},
            "persons_total": 6,
            "umbrellas_total": 2,
            "persons_by_height_band": dict(zip(BANDS, [2, 0, 0, 2, 0, 2], strict=True)),
            "persons_by_height_band_1080p": dict(zip(BANDS, [2, 0, 0, 2, 0, 2], strict=True)),
        }
    )
    + "\n"
)


def _austin_pass(site: Site, argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> str:
    urls = [site.image(f"a{i}", _jpeg(1920, 1080, 1)) for i in range(2)]
    urls.append(site.image("missing", b"", status=404))
    site.dataset([*[_aus(u) for u in urls], _aus("https://elsewhere.example/x.jpg"), {"x": 1}])
    stub = Stub({1: [*_persons(B20, B90, (5.0, 10.0, 25.0, 300.0)), ("umbrella", B90)]})
    assert _pilot(site, argv, stub=stub) == 0
    return capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--live"], ["--live", "--source", "austin"]])
def test_ac1_austins_output_is_byte_identical(
    site: Site, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    assert _austin_pass(site, argv, capsys) == AUSTIN_PILOT_OUTPUT


def test_ac1_austin_still_refuses_in_the_austin_night(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    site.dataset([])
    assert _pilot(site, ["--live"], now=CALGARY_ONLY_DAY) == 1
    assert capsys.readouterr().err.startswith(
        "pilot_heights: the sun is below the horizon in Austin ("
    )


# AC2: the attribute session ---------------------------------------------------------


class Answers:
    """An attribute reviewer answering `default` for every crop; records what it saw."""

    def __init__(self, default: str | None = "ynn") -> None:
        self.default = default
        self.items: list[spotcheck.ReviewItem] = []

    def attributes(
        self, items: Sequence[spotcheck.ReviewItem], deadline: float
    ) -> dict[int, str | None]:
        self.items = list(items)
        return {item.number: self.default for item in items}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: None) -> Iterator[Path]:
    """Outside CI, offline but for 127.0.0.1, in a fresh working directory, HOME and
    temporary directory; yields the temporary directory. The London pipelines must not be
    opened."""
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    work, home, tmp = tmp_path / "work", tmp_path / "home", tmp_path / "tmp"
    for d in (work, home, tmp):
        d.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)

    def no_pipeline(*args: Any, **kwargs: Any) -> spotcheck.Pipeline:
        raise AssertionError("a London pipeline must not be opened")

    monkeypatch.setattr(spotcheck, "live_pipeline", no_pipeline)
    monkeypatch.setattr(spotcheck, "dry_run_pipeline", no_pipeline)
    yield tmp


WINDOW = ["--n", "20", "--min-persons", "1", "--view", "window"]


def _opener(stub: Stub, opened: list[str] | None = None) -> Callable[[str], Any]:
    def open_detector(model: str) -> tuple[Stub, spotcheck.DetectorInfo]:
        if opened is not None:
            opened.append(model)
        return stub, INFO

    return open_detector


def _calgary(
    site: Site | None,
    out: Path,
    args: Sequence[str] = (),
    *,
    stub: Stub | None = None,
    reviewer: Any = None,
    clock: datetime.datetime = ALL_DAY,
    opened: list[str] | None = None,
    window: Sequence[str] = WINDOW,
) -> int:
    argv = ["--attributes", "--source", "calgary", *window, *args]
    argv += ["--reviewer", "tester", "--out-dir", str(out)]
    extra: dict[str, Any] = {} if site is None else {"calgary": site.calgary()}
    return spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: clock,
        reviewer=Answers() if reviewer is None else reviewer,
        open_detector=_opener(stub or Stub(), opened),
        **extra,
    )


def _calgary_file(out: Path, name: str = f"{DAY.isoformat()}-calgary.json") -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "attributes" / name).read_text("utf-8"))
    return data


def _names(out: Path) -> list[str]:
    folder = out / "attributes"
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def _two_cameras(site: Site) -> tuple[bytes, bytes]:
    first, second = _jpeg(840, 630, 1), _jpeg(840, 630, 2)
    site.dataset([_cal(site.image("a", first)), _cal(site.image("b", second))])
    return first, second


def test_ac2_the_option(env: Path) -> None:
    assert spotcheck.SOURCES == ("london", "austin", "calgary")
    args = spotcheck.parse_args(["--n", "1", "--attributes", "--source", "calgary"])
    assert args.source == "calgary"
    args = spotcheck.parse_args(
        ["--n", "1", "--attributes", "--source", "calgary", "--bbox", "51.0,-114.1,51.1,-114.0"]
    )
    assert args.bbox == (51.0, -114.1, 51.1, -114.0)


@pytest.mark.parametrize(
    "argv",
    [
        ["--n", "5", "--source", "calgary"],
        ["--n", "5", "--source", "calgary", "--mode", "frames"],
        ["--n", "5", "--source", "calgary", "--dry-run"],
        ["--n", "5", "--source", "Calgary", "--attributes"],
        ["--n", "5", "--source", "calgary", "--attributes", "--bbox", "51.1,-114.1,51.0,-114.0"],
    ],
)
def test_ac2_usage_errors_exit_2(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as refused:
        spotcheck.main([*argv, "--out-dir", str(tmp_path / "out")], today=DAY)
    assert refused.value.code == 2
    assert capsys.readouterr().err.startswith("usage:")
    assert not (tmp_path / "out").exists()


def test_ac2_the_usage_message_names_calgary(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        spotcheck.main(["--n", "5", "--source", "calgary"], today=DAY)
    err = capsys.readouterr().err
    assert "--source calgary" in err and "--attributes" in err


def test_ac2_the_default_endpoints_are_the_pilots() -> None:
    endpoints = spotcheck.CalgaryEndpoints()
    assert endpoints.dataset_url == pilot_heights.CALGARY_DATASET_URL
    assert endpoints.dataset_policy == pilot_heights.CALGARY_DATASET_POLICY
    assert endpoints.image_policy == pilot_heights.CALGARY_IMAGE_POLICY
    assert spotcheck.CALGARY == pilot_heights.CALGARY


def test_ac2_a_session_through_the_pinned_hosts(
    env: Path, transport: Transport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    listed = [
        "http://trafficcam.calgary.ca/loc1.jpg",
        "https://trafficcam.calgary.ca/loc2.jpg",
        "http://user@trafficcam.calgary.ca/loc3.jpg",
        "https://other.example/loc4.jpg",
    ]
    records = [_cal(url) for url in listed]
    records.append(_cal("https://trafficcam.calgary.ca/far.jpg", *OUTSIDE))
    transport.bodies.update(
        {
            CALGARY_DATASET: json.dumps(records).encode(),
            "https://trafficcam.calgary.ca/loc1.jpg": _jpeg(840, 630, 1),
            "https://trafficcam.calgary.ca/loc2.jpg": _jpeg(840, 630, 2),
        }
    )
    out = tmp_path / "out"
    assert _calgary(None, out, stub=Stub({1: _persons(B90), 2: _persons(B200)})) == 0
    urls = transport.urls()
    assert urls[0] == CALGARY_DATASET
    assert sorted(urls[1:]) == [
        "https://trafficcam.calgary.ca/loc1.jpg",
        "https://trafficcam.calgary.ca/loc2.jpg",
    ]
    assert all(schemes == ("https",) for _url, schemes in transport.calls)
    assert _calgary_file(out)["crops"] == [[90, "ynn", None], [200, "ynn", None]]
    err = capsys.readouterr().err
    assert (
        "spotcheck: 5 Calgary cameras listed, 2 selected (0 malformed record(s) skipped, "
        "2 URL(s) refused)\n"
    ) in err


def test_ac2_frames_come_through_the_list_policy_and_box(
    env: Path, site: Site, tmp_path: Path
) -> None:
    good = site.image("good", _jpeg(840, 630, 1))
    outside = site.image("outside", _jpeg(840, 630, 1))
    redirect = site.image("redirect", b"", status=302, headers={"Location": good})
    site.dataset(
        [
            _cal(good),
            _cal(outside, *OUTSIDE),
            _cal(redirect),
            _cal("http://203.0.113.9/image/x"),
            _aus(good),
            {"camera_id": "broken"},
        ]
    )
    stub = Stub({1: _persons(B90)})
    assert _calgary(site, tmp_path / "out", stub=stub) == 0
    assert site.count("/dataset") == 1
    assert site.count("/image/good") == 1
    assert site.count("/image/outside") == 0
    assert site.count("/image/redirect") == 1
    assert len(stub.shapes) == 1


def test_ac2_bbox_selects(env: Path, site: Site, tmp_path: Path) -> None:
    outside = site.image("outside", _jpeg(840, 630, 1))
    site.dataset([_cal(outside, *OUTSIDE)])
    bbox = "51.09,-114.21,51.11,-114.19"
    assert _calgary(site, tmp_path / "out", ["--bbox", bbox], stub=Stub({1: _persons(B90)})) == 0
    assert site.count("/image/outside") == 1


def test_ac2_other_sizes_are_skipped_counted_and_never_shown(
    env: Path, site: Site, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bodies = {
        "ok": _jpeg(840, 630, 1),
        "hd": _jpeg(1920, 1080, 2),
        "placeholder": _jpeg(320, 176, 2),
        "wide": _jpeg(841, 630, 2),
        "tall": _jpeg(840, 631, 2),
        "half": _jpeg(420, 315, 2),
    }
    site.dataset([_cal(site.image(name, body)) for name, body in bodies.items()])
    stub = Stub({1: _persons(B90), 2: _persons(B200)})
    reviewer = Answers()
    assert _calgary(site, tmp_path / "out", stub=stub, reviewer=reviewer) == 0
    assert stub.shapes == [(630, 840, 3)]
    assert len(reviewer.items) == 1
    record = _calgary_file(tmp_path / "out")
    assert record["frames"] == 1 and record["crops"] == [[90, "ynn", None]]
    err = capsys.readouterr().err
    assert (
        "spotcheck: fetched 1 840x630 frame(s) of 6: 5 not 840x630 (skipped), 0 failed, 0 refused\n"
    ) in err


def test_ac2_the_progress_lines_break_failures_down_by_kind(
    env: Path, site: Site, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = site.image("ok", _jpeg(840, 630, 1))
    urls = [
        ok,
        site.image("err", b"", status=500),
        site.image("cut", _jpeg(840, 630, 1)[:-200]),
        site.image("moved", b"", status=302, headers={"Location": ok}),
        site.image("small", _jpeg(320, 176, 1)),
    ]
    site.dataset([*[_cal(u) for u in urls], {"broken": True}, _cal("http://203.0.113.9/x")])
    assert _calgary(site, tmp_path / "out", stub=Stub({1: _persons(B90)})) == 0
    err = capsys.readouterr().err
    assert (
        "spotcheck: 7 Calgary cameras listed, 5 selected (1 malformed record(s) skipped, "
        "1 URL(s) refused)\n"
    ) in err
    assert (
        "spotcheck: fetched 1 840x630 frame(s) of 5: 1 not 840x630 (skipped), "
        "2 failed (http 1, decode 1), 2 refused\n"
    ) in err
    port = site.netloc.rsplit(":", 1)[1]
    for leak in ("127.0.0.1", port, "203.0.113.9", "/image/", "http://", "Error"):
        assert leak not in err


def test_ac2_the_header_rules_apply(
    env: Path, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = _jpeg(840, 630, 1)
    bodies = {
        "good": good,
        "declared": jpeg_declaring(30000, 30000),
        "scans": _scan_bomb(good),
        "unlisted": _unlisted_marker(good),
    }
    decoded: list[int] = []
    real = cv2.imdecode

    def imdecode(buf: Any, flags: int) -> Any:
        decoded.append(len(buf))
        return real(buf, flags)

    monkeypatch.setattr(cv2, "imdecode", imdecode)
    site.dataset([_cal(site.image(name, body)) for name, body in bodies.items()])
    stub = Stub({1: _persons(B90)})
    assert _calgary(site, tmp_path / "out", stub=stub) == 0
    assert stub.shapes == [(630, 840, 3)]
    assert decoded == [len(good)]  # nothing else reached the decoder
    for name in bodies:
        assert site.count(f"/image/{name}") == 1


def test_ac2_crops_are_cut_from_the_full_resolution_frame(
    env: Path, site: Site, tmp_path: Path
) -> None:
    first, second = _two_cameras(site)
    reviewer = Answers()
    stub = Stub({1: _persons(B90), 2: _persons(B500)})
    assert _calgary(site, tmp_path / "out", stub=stub, reviewer=reviewer) == 0
    frames = {1: _decoded(first), 2: _decoded(second)}
    assert len(reviewer.items) == 2
    for item in reviewer.items:
        key = int(item.image[item.image.shape[0] // 2, 2, 0]) // 10
        box = B90 if key == 1 else B500
        expected = spotcheck.render_crop(
            frames[key], detect.Detection("person", 0.9, box), item.number
        )
        assert np.array_equal(item.image, expected)
    assert sorted(item.image.shape[0] for item in reviewer.items) == [180, 630]


def test_ac2_a_calgary_record_says_its_source(env: Path, site: Site, tmp_path: Path) -> None:
    _two_cameras(site)
    out = tmp_path / "out"
    stub = Stub({1: _persons(B90, B20), 2: _persons(B200)})
    assert _calgary(site, out, stub=stub, reviewer=Answers("nyu")) == 0
    assert _names(out) == [f"{DAY.isoformat()}-calgary.json"]
    record = _calgary_file(out)
    assert list(record) == [*_record_fields(), "source"]
    assert record["source"] == "calgary"
    assert record["started_at"] == "2026-10-01T16:00Z"
    assert record["frames"] == 2 and record["min_height_px"] == 31
    assert record["crops"] == [[90, "nyu", None], [200, "nyu", None]]
    raw = (out / "attributes" / f"{DAY.isoformat()}-calgary.json").read_bytes()
    assert raw.endswith(b', "source": "calgary"}\n')


def _record_fields() -> list[str]:
    """The fields of an attribute record, in the order the tool writes them."""
    return [
        "date",
        "started_at",
        "light",
        "frames",
        "detector",
        "min_height_px",
        "judge",
        "crops_shown",
        "crops_rejected",
        "crops",
    ]


def test_ac2_file_names_never_overwrite_or_take_another_sources_name(
    env: Path, site: Site, tmp_path: Path
) -> None:
    _two_cameras(site)
    out = tmp_path / "out"
    (out / "attributes").mkdir(parents=True)
    others = {
        f"{DAY.isoformat()}.json": b"london\n",
        f"{DAY.isoformat()}-austin.json": b"austin\n",
    }
    for name, body in others.items():
        (out / "attributes" / name).write_bytes(body)
    for _ in range(3):
        assert _calgary(site, out, stub=Stub({1: _persons(B90)})) == 0
    assert _names(out) == sorted(
        [
            *others,
            f"{DAY.isoformat()}-calgary.json",
            f"{DAY.isoformat()}-calgary-2.json",
            f"{DAY.isoformat()}-calgary-3.json",
        ]
    )
    for name, body in others.items():
        assert (out / "attributes" / name).read_bytes() == body


def test_ac2_light_is_calgarys(env: Path, site: Site, tmp_path: Path) -> None:
    _two_cameras(site)
    assert spotcheck.light_at(LONDON_DARK) == "dark"
    assert _calgary(site, tmp_path / "a", stub=Stub({1: _persons(B90)}), clock=LONDON_DARK) == 0
    assert _calgary_file(tmp_path / "a")["light"] == "day"
    assert spotcheck.light_at(CALGARY_ONLY_DAY, spotcheck.AUSTIN) == "dark"
    code = _calgary(site, tmp_path / "b", stub=Stub({1: _persons(B90)}), clock=CALGARY_ONLY_DAY)
    assert code == 0
    assert _calgary_file(tmp_path / "b")["light"] == "day"


def test_ac2_light_in_the_calgary_dark(env: Path, site: Site, tmp_path: Path) -> None:
    _two_cameras(site)
    code = _calgary(
        site, tmp_path / "out", ["--allow-dark"], stub=Stub({1: _persons(B90)}), clock=CALGARY_DARK
    )
    assert code == 0
    assert _calgary_file(tmp_path / "out")["light"] == "dark"


def test_ac2_refuses_in_the_calgary_dark(
    env: Path, site: Site, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _two_cameras(site)
    opened: list[str] = []
    assert _calgary(site, tmp_path / "out", clock=CALGARY_DARK, opened=opened) == 1
    assert capsys.readouterr().err == DARK_CALGARY + "\n"
    assert site.requests == [] and opened == []
    assert not (tmp_path / "out").exists()
    assert DARK_CALGARY.removeprefix("spotcheck: ") == spotcheck.CALGARY_DARK_REFUSAL
    assert spotcheck.DARK_REFUSAL.replace("London", "Calgary") == spotcheck.CALGARY_DARK_REFUSAL


def test_ac2_a_judgements_file_runs_in_the_calgary_dark(
    env: Path, site: Site, tmp_path: Path
) -> None:
    _two_cameras(site)
    window = ["--n", "5", "--min-persons", "1", "--judgements", str(tmp_path / "j.json")]
    code = _calgary(
        site, tmp_path / "out", stub=Stub({1: _persons(B90)}), clock=CALGARY_DARK, window=window
    )
    assert code == 0


def test_ac2_the_window_fits_the_screen(
    env: Path, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[dict[str, Any]] = []

    class Recording:
        def __init__(self, **kwargs: Any) -> None:
            made.append(kwargs)

        def attributes(
            self, items: Sequence[spotcheck.ReviewItem], deadline: float
        ) -> dict[int, str | None]:
            return {item.number: "ynn" for item in items}

    monkeypatch.setattr(spotcheck, "WindowReviewer", Recording)
    _two_cameras(site)
    argv = ["--attributes", "--source", "calgary", *WINDOW]
    argv += ["--reviewer", "tester", "--out-dir", str(tmp_path / "out")]
    code = spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: ALL_DAY,
        calgary=site.calgary(),
        open_detector=_opener(Stub({1: _persons(B90)})),
    )
    assert code == 0
    assert len(made) == 1 and made[0].get("fit_screen") is True


def test_ac2_min_height_filters_full_resolution_heights(
    env: Path, site: Site, tmp_path: Path
) -> None:
    _two_cameras(site)
    stub = Stub({1: _persons(B90, B20), 2: _persons(B200)})
    assert _calgary(site, tmp_path / "out", ["--min-height", "100"], stub=stub) == 0
    record = _calgary_file(tmp_path / "out")
    assert record["min_height_px"] == 100 and record["crops"] == [[200, "ynn", None]]


def test_ac2_nothing_but_the_record_is_written(env: Path, site: Site, tmp_path: Path) -> None:
    def tree(root: Path) -> set[str]:
        return {str(p.relative_to(root)) for p in root.rglob("*")}

    _two_cameras(site)
    work = Path.cwd()
    before = (tree(env), tree(work), tree(ROOT / "engine"))
    out = tmp_path / "out"
    assert _calgary(site, out, stub=Stub({1: _persons(B90), 2: _persons(B200)})) == 0
    assert (tree(env), tree(work), tree(ROOT / "engine")) == before
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*")) == [
        "attributes",
        f"attributes/{DAY.isoformat()}-calgary.json",
    ]


def test_ac2_the_dry_run_uses_a_local_fake_calgary(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = Stub()
    window = [*WINDOW, "--dry-run", "--min-persons", "0"]
    assert _calgary(None, tmp_path / "out", stub=stub, window=window) == 0
    assert stub.shapes and set(stub.shapes) == {(630, 840, 3)}
    err = capsys.readouterr().err
    assert "Calgary cameras listed" in err
    skipped = re.search(r"(\d+) not 840x630", err)
    assert skipped is not None and int(skipped.group(1)) >= 1
    assert _calgary_file(tmp_path / "out")["source"] == "calgary"


def test_ac2_the_dry_run_refuses_the_real_spotchecks_directory(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--attributes", "--source", "calgary", "--dry-run", *WINDOW, "--out-dir", "spotchecks"]
    opened: list[str] = []
    code = spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: ALL_DAY,
        reviewer=Answers(),
        open_detector=_opener(Stub(), opened),
    )
    assert code == 1
    assert "--out-dir" in capsys.readouterr().err
    assert opened == []


# Austin's session stays byte for byte as it was: the record and the progress lines of
# one fixed session, produced by the code on main before this task.
AUSTIN_RECORD = (
    b'{"date": "2026-10-01", "started_at": "2026-10-01T16:00Z", "light": "day", "frames": 1, '
    b'"detector": {"model": "stub", "sha256": '
    b'"0000000000000000000000000000000000000000000000000000000000000000", "conf": 0.35}, '
    b'"min_height_px": 31, "judge": null, "crops_shown": 1, "crops_rejected": 0, '
    b'"crops": [[90, "nyu", null]], "source": "austin"}\n'
)
AUSTIN_ERR = (
    "spotcheck: 5 Austin cameras listed, 3 selected "
    "(1 malformed record(s) skipped, 1 URL(s) refused)\n"
    "spotcheck: fetched 1 1920x1080 frame(s) of 3: 1 not 1920x1080 (skipped), "
    "1 failed (http 1), 1 refused\n"
    "spotcheck: opening the review: 1 crop(s)\n"
)


def test_ac4_an_austin_session_is_byte_identical(
    env: Path, site: Site, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hd = _jpeg(1920, 1080, 1)
    urls = [
        site.image("a", hd),
        site.image("small", _jpeg(320, 176, 1)),
        site.image("err", b"", status=500),
    ]
    site.dataset([*[_aus(u) for u in urls], {"broken": 1}, _aus("http://203.0.113.9/x")])
    out = tmp_path / "out"
    argv = ["--attributes", "--source", "austin", *WINDOW]
    argv += ["--reviewer", "tester", "--out-dir", str(out)]
    code = spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: ALL_DAY,
        reviewer=Answers("nyu"),
        austin=site.austin(),
        open_detector=_opener(Stub({1: _persons((900.0, 500.0, 940.0, 590.0))})),
    )
    assert code == 0
    assert _names(out) == [f"{DAY.isoformat()}-austin.json"]
    path = out / "attributes" / f"{DAY.isoformat()}-austin.json"
    assert path.read_bytes() == AUSTIN_RECORD
    captured = capsys.readouterr()
    assert captured.err == AUSTIN_ERR
    assert captured.out == (
        "Review window open: 1 crop(s). y: yes   n: no   u: cannot tell   "
        "x: not a person or nothing can be told   Backspace: back   q: stop\n"
        f"Attribute labels written to {path}: 1 crop(s) shown, 0 rejected\n"
    )


# AC3: the summary ---------------------------------------------------------------------

VALID: dict[str, Any] = {
    "date": "2026-10-05",
    "started_at": "2026-10-05T16:00Z",
    "light": "day",
    "frames": 3,
    "detector": {"model": "yolox_m", "sha256": "0" * 64, "conf": 0.35},
    "min_height_px": 31,
    "judge": "di-qwen3-vl-235b",
    "crops_shown": 3,
    "crops_rejected": 1,
    "crops": [[90, "ynn", "ynu"], [200, "uuu", None]],
}


def _raw(**changes: Any) -> bytes:
    return json.dumps({**VALID, **changes}).encode()


def test_ac3_a_calgary_record_is_accepted() -> None:
    labelling = spotcheck_summary.parse_labelling(_raw(source="calgary"))
    assert labelling.source == "calgary"
    assert labelling.crops == ((90, "ynn", "ynu"), (200, "uuu", None))
    assert spotcheck_summary.parse_labelling(_raw(source="austin")).source == "austin"
    assert spotcheck_summary.parse_labelling(_raw()).source == "london"


@pytest.mark.parametrize(
    "value", ["london", "Calgary", "CALGARY", "calgary ", "", "paris", None, 1, ["calgary"]]
)
def test_ac3_any_other_source_is_refused(value: Any) -> None:
    with pytest.raises(ValueError):
        spotcheck_summary.parse_labelling(_raw(source=value))


@pytest.mark.parametrize(
    "changes",
    [{"light": "dusk"}, {"min_height_px": 30}, {"crops_shown": 9}, {"extra": 1}],
)
def test_ac3_a_calgary_record_is_checked_like_the_others(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        spotcheck_summary.parse_labelling(_raw(source="calgary", **changes))


def _real_copy(d: Path) -> None:
    (d / "attributes").mkdir(parents=True)
    for name in REAL_FILES:
        shutil.copyfile(REAL_DIR / "attributes" / name, d / "attributes" / name)


def _summary(d: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> list[str]:
    assert spotcheck_summary.main(["--attributes", "--dir", str(d), *extra]) == 0
    return capsys.readouterr().out.replace(str(d), "{DIR}").splitlines()


def _sections(lines: list[str]) -> list[list[str]]:
    starts = [k for k, line in enumerate(lines) if "attribute file(s) in " in line]
    return [lines[a:b] for a, b in zip(starts, [*starts[1:], len(lines)], strict=True)]


def test_ac3_a_calgary_file_adds_a_section_after_london_and_austin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "real"
    _real_copy(d)
    london = _summary(d, capsys)
    (d / "attributes" / "2026-10-05-austin.json").write_bytes(_raw(source="austin"))
    both = _summary(d, capsys)
    (d / "attributes" / "2026-10-05-calgary.json").write_bytes(
        _raw(source="calgary", crops=[[120, "nyn", "nyn"]], crops_shown=2)
    )
    three = _summary(d, capsys)
    assert three[: len(both)] == both and both[: len(london)] == london
    sections = _sections(three)
    assert len(sections) == 3
    assert sections[1][0].startswith("1 Austin attribute file(s)")
    calgary = sections[2]
    assert calgary[0] == "1 Calgary attribute file(s) in {DIR}/attributes"
    assert three[len(both)] == ""  # an empty line before it
    assert "crops: 2 shown, 1 rejected, 1 labelled, 1 with a model answer" in calgary
    assert sum(1 for line in calgary if line.strip().startswith("verdict: ")) == 3


def test_ac3_sources_are_never_pooled(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "attributes").mkdir()
    london = [[60, "ynn", "ynn"]] * 40 + [[60, "nnn", "nnn"]] * 170
    austin = [[90, "nnn", "nnn"]] * 3
    calgary = [[90, "ynn", "nnn"]] * 5
    files = {
        "2026-10-05.json": _raw(crops=london, crops_shown=210, crops_rejected=0),
        "2026-10-05-austin.json": _raw(
            source="austin", crops=austin, crops_shown=3, crops_rejected=0
        ),
        "2026-10-05-calgary.json": _raw(
            source="calgary", crops=calgary, crops_shown=5, crops_rejected=0
        ),
    }
    for name, body in files.items():
        (tmp_path / "attributes" / name).write_bytes(body)
    london_lines, austin_lines, calgary_lines = _sections(_summary(tmp_path, capsys))
    assert london_lines[0] == "1 attribute file(s) in {DIR}/attributes"
    assert austin_lines[0] == "1 Austin attribute file(s) in {DIR}/attributes"
    assert calgary_lines[0] == "1 Calgary attribute file(s) in {DIR}/attributes"

    def reviewer(block: list[str]) -> str:
        return block[block.index("outer_layer:") + 1]

    assert reviewer(london_lines).startswith("  reviewer: yes 40, no 170, cannot tell 0")
    assert reviewer(austin_lines).startswith("  reviewer: yes 0, no 3, cannot tell 0")
    assert reviewer(calgary_lines).startswith("  reviewer: yes 5, no 0, cannot tell 0")

    def verdicts(block: list[str]) -> list[str]:
        return [line.strip() for line in block if line.strip().startswith("verdict: ")]

    assert verdicts(london_lines)[0] == "verdict: pass"
    assert verdicts(calgary_lines)[0].startswith("verdict: insufficient")
    assert calgary_lines[calgary_lines.index("outer_layer:") + 5] == (
        "  model: 5 paired, model cannot tell 0"
    )


def test_ac3_no_calgary_section_without_a_calgary_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "attributes").mkdir()
    (tmp_path / "attributes" / "2026-10-05.json").write_bytes(_raw())
    (tmp_path / "attributes" / "2026-10-05-austin.json").write_bytes(_raw(source="austin"))
    lines = _summary(tmp_path, capsys)
    assert not any("Calgary" in line for line in lines)
    assert len(_sections(lines)) == 2


def test_ac3_with_only_a_calgary_file_london_still_comes_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "attributes").mkdir()
    (tmp_path / "attributes" / "2026-10-05-calgary.json").write_bytes(_raw(source="calgary"))
    sections = _sections(_summary(tmp_path, capsys))
    assert [s[0] for s in sections] == [
        "0 attribute file(s) in {DIR}/attributes",
        "1 Calgary attribute file(s) in {DIR}/attributes",
    ]


def test_ac3_the_calgary_section_has_no_rain_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "attributes").mkdir()
    (tmp_path / "attributes" / "2026-10-05.json").write_bytes(_raw())
    (tmp_path / "attributes" / "2026-10-05-calgary.json").write_bytes(_raw(source="calgary"))
    data = tmp_path / "data"
    data.mkdir()
    london, calgary = _sections(_summary(tmp_path, capsys, "--data-dir", str(data)))
    assert any(line.startswith("  rain ") for line in london)
    assert not any(line.startswith("  rain ") for line in calgary)


# AC6: quality ---------------------------------------------------------------------------


def test_ac6_privacy_guard_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "privacy_guard.py")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ac6_no_new_dependency() -> None:
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    lines = [line.strip().strip('",') for line in deps.splitlines() if line.strip()]
    names = sorted(line.split("=")[0].split(">")[0] for line in lines)
    # T-077 adds tzdata (zoneinfo on Windows); the list is otherwise unchanged.
    assert names == ["llama-cpp-python", "numpy", "onnxruntime", "opencv-python-headless", "tzdata"]


def test_ac6_the_guide_documents_the_source() -> None:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Attribute session (`--attributes`)")
    end = text.index("\n## ", start + 1)
    section = text[start:end]
    assert "--source calgary" in section
    assert "840x630" in section
