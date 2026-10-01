"""Acceptance tests for T-058: a one-off pilot that measures person box heights on HD
camera stills and prints aggregate counts only. The task contract: do not edit.

Every frame here is synthetic (a uniform colour, encoded in memory), the camera list is
written by the test, both are served by a local HTTP server on 127.0.0.1, and the
detector is a scripted stand-in. Nothing reaches the network.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import aggregate, detect, fetch
from wearreport._cv import encode_jpeg
from wearreport.testing.fake_cameras import jpeg_declaring
from wearreport.tools import pilot_heights, spotcheck

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "engine" / "wearreport" / "tools" / "pilot_heights.py"
DAY = datetime.datetime(2026, 10, 1, 18, 30, 42, tzinfo=datetime.UTC)  # 13:30 in Austin
NIGHT = datetime.datetime(2026, 10, 1, 8, 0, tzinfo=datetime.UTC)  # 03:00 in Austin
INSIDE = (-97.745, 30.270)  # lon, lat: inside the default downtown box
OUTSIDE = (-97.818, 30.233)
KEYS = {
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
    "persons_by_height_band_1080p",
}
BANDS = ["<31", "31-45", "46-79", "80-119", "120-199", "200+"]

Frame = npt.NDArray[np.uint8]
Box = tuple[float, float, float, float]
Found = Sequence[tuple[detect.Label, Box]]


# A local site: the camera list and the images ---------------------------------------


@dataclass
class Route:
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0


class Site:
    """Serves /dataset (the camera list) and /image/<name> on 127.0.0.1."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.requests: list[str] = []
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                with site.lock:
                    site.requests.append(self.path)
                    route = site.routes.get(self.path, Route(404))
                if route.delay_s and site.stopping.wait(route.delay_s):
                    return
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
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def netloc(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"{host!s}:{port}"

    def url(self, path: str) -> str:
        return f"http://{self.netloc}{path}"

    def policy(self) -> pilot_heights.UrlPolicy:
        return pilot_heights.UrlPolicy("http", self.netloc)

    def image(self, name: str, body: bytes, **kwargs: Any) -> str:
        self.routes[f"/image/{name}"] = Route(body=body, **kwargs)
        return self.url(f"/image/{name}")

    def dataset(self, records: object) -> None:
        self.routes["/dataset"] = Route(body=json.dumps(records).encode())

    def close(self) -> None:
        self.stopping.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def site() -> Iterator[Site]:
    s = Site()
    try:
        yield s
    finally:
        s.close()


def _jpeg(width: int, height: int, key: int) -> bytes:
    """A uniform frame whose colour encodes `key` (read back by Stub)."""
    return encode_jpeg(np.full((height, width, 3), 10 * key + 5, dtype=np.uint8))


def _record(url: object, lon: object = INSIDE[0], lat: object = INSIDE[1]) -> dict[str, Any]:
    return {
        "camera_id": "1",
        "screenshot_address": url,
        "location": {"type": "Point", "coordinates": [lon, lat]},
    }


class Stub:
    """boxes[key] are the detections in a frame whose colour encodes `key`; shapes[key]
    are the shapes of the frames with that key that reached the detector."""

    def __init__(self, boxes: dict[int, Found] | None = None) -> None:
        self.boxes = boxes or {}
        self.shapes: dict[int, list[tuple[int, ...]]] = {}

    def detect(self, frame: Frame) -> list[detect.Detection]:
        key = int(frame[frame.shape[0] // 2, frame.shape[1] // 2, 0]) // 10
        self.shapes.setdefault(key, []).append(frame.shape)
        return [detect.Detection(label, 0.9, box) for label, box in self.boxes.get(key, [])]


def _box(height: float, y1: float = 10.0) -> Box:
    return (5.0, y1, 25.0, y1 + height)


def _run(
    site: Site,
    argv: Sequence[str] = ("--live",),
    *,
    stub: Stub | None = None,
    now: datetime.datetime = DAY,
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


# AC1: command and options -------------------------------------------------------------


def test_refuses_without_live_and_makes_no_request(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    assert _run(site, [], opened=opened) == 2
    captured = capsys.readouterr()
    assert "--live" in captured.err
    assert captured.out == ""
    assert site.requests == [] and opened == []


def test_module_refuses_without_live_from_the_command_line() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "wearreport.tools.pilot_heights"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 2
    assert "--live" in proc.stderr
    assert proc.stdout == ""


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-cameras", "0"],
        ["--max-cameras", "1001"],
        ["--max-cameras", "ten"],
        ["--model", "yolox_l"],
        ["--model", "yolox_x"],
        ["--bbox", "30.26,-97.755,30.285"],
        ["--bbox", "30.285,-97.755,30.260,-97.735"],  # south above north
        ["--bbox", "30.260,-97.735,30.285,-97.755"],  # west east of east
        ["--bbox", "30.260,-97.755,nan,-97.735"],
        ["--bbox", "30.260,-97.755,91,-97.735"],
        ["--bbox", "a,b,c,d"],
        ["--timeout", "0"],
        ["--timeout", "-5"],
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--unknown"],
    ],
)
def test_invalid_options_exit_2(
    site: Site, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    assert _run(site, ["--live", *extra]) == 2
    assert capsys.readouterr().out == ""
    assert site.requests == []


def test_defaults_and_options_reach_the_pass(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    urls = [site.image(f"c{i}", _jpeg(352, 288, 0)) for i in range(3)]
    site.dataset([_record(u) for u in urls])
    opened: list[str] = []
    assert _run(site, opened=opened) == 0
    result = _output(capsys)
    assert opened == ["yolox_m"] and result["model"] == "yolox_m"
    assert result["cameras_selected"] == 3

    opened.clear()
    assert _run(site, ["--live", "--model", "yolox_s", "--max-cameras", "2"], opened=opened) == 0
    result = _output(capsys)
    assert opened == ["yolox_s"] and result["model"] == "yolox_s"
    assert result["cameras_selected"] == 2 and result["frames_ok"] == 2


def test_default_bbox_is_downtown_austin() -> None:
    assert pilot_heights.DEFAULT_BBOX == (30.260, -97.755, 30.285, -97.735)
    assert pilot_heights.DEFAULT_MAX_CAMERAS == 100
    assert pilot_heights.DEFAULT_TIMEOUT_S == 600
    assert pilot_heights.DEFAULT_MODEL == "yolox_m"


# AC2: camera list ---------------------------------------------------------------------


def test_dataset_url_and_host_are_pinned() -> None:
    assert pilot_heights.DATASET_URL == (
        "https://data.austintexas.gov/resource/b4k4-adkb.json?camera_status=TURNED_ON&$limit=2000"
    )
    assert pilot_heights.UrlPolicy("https", "data.austintexas.gov") == pilot_heights.DATASET_POLICY
    assert pilot_heights.DATASET_POLICY.allows(pilot_heights.DATASET_URL)
    assert pilot_heights.MAX_DATASET_BYTES <= 5 * 1024 * 1024


def test_bbox_selection_is_inclusive(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    s, w, n, e = 30.0, -98.0, 31.0, -97.0
    points = {
        "centre": (-97.5, 30.5),
        "sw": (w, s),
        "ne": (e, n),
        "west_of": (-98.0001, 30.5),
        "north_of": (-97.5, 31.0001),
        "far": (OUTSIDE[0] + 10, OUTSIDE[1]),
    }
    records = [
        _record(site.image(k, _jpeg(352, 288, 0)), lon, lat) for k, (lon, lat) in points.items()
    ]
    site.dataset(records)
    assert _run(site, ["--live", "--bbox", f"{s},{w},{n},{e}"]) == 0
    result = _output(capsys)
    assert result["cameras_listed"] == 6
    assert result["cameras_selected"] == 3
    assert result["records_skipped"] == 0 and result["refused_url"] == 0
    fetched = {p for p in site.requests if p.startswith("/image/")}
    assert fetched == {"/image/centre", "/image/sw", "/image/ne"}


def _malformed(url: str) -> list[object]:
    deep: object = 1.0
    for _ in range(200):
        deep = [deep]
    return [
        None,
        7,
        "camera",
        [url],
        {"screenshot_address": url},  # no location
        {"location": {"type": "Point", "coordinates": list(INSIDE)}},  # no URL
        _record(42),
        _record(None),
        _record(["x"]),
        {"screenshot_address": url, "location": "30.27,-97.745"},
        {"screenshot_address": url, "location": {"type": "Point"}},
        {"screenshot_address": url, "location": {"type": "Point", "coordinates": [INSIDE[0]]}},
        {"screenshot_address": url, "location": {"type": "Point", "coordinates": [*INSIDE, 1, 2]}},
        {"screenshot_address": url, "location": {"type": "Point", "coordinates": "x"}},
        {"screenshot_address": url, "location": {"type": "Point", "coordinates": deep}},
        {"screenshot_address": url, "location": deep},
        _record(url, str(INSIDE[0]), str(INSIDE[1])),
        _record(url, True, INSIDE[1]),
        _record(url, INSIDE[0], None),
        _record(url, INSIDE[0], 91.0),
        _record(url, -181.0, INSIDE[1]),
        _record(url, INSIDE[0], {"lat": 30.27}),
    ]


def test_malformed_records_are_skipped_and_counted(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    good = site.image("good", _jpeg(352, 288, 0))
    bad = _malformed(site.image("bad", _jpeg(352, 288, 0)))
    body = json.dumps([*bad, _record(good)])
    # Non-finite and huge numbers, which json.dumps will not write.
    extra = [
        f'{{"screenshot_address": "{good}", "location": {{"type": "Point", '
        f'"coordinates": [{lon}, {lat}]}}}}'
        for lon, lat in [
            ("NaN", "30.27"),
            ("-97.745", "Infinity"),
            ("-Infinity", "30.27"),
            ("1e999", "30.27"),
            ("-97.745", "1" + "0" * 5000),
            ("-97.745", "1" + "0" * 400 + ".5"),
        ]
    ]
    site.routes["/dataset"] = Route(body=(body[:-1] + ", " + ", ".join(extra) + "]").encode())
    assert _run(site) == 0
    result = _output(capsys)
    assert result["cameras_listed"] == len(bad) + 1 + len(extra)
    assert result["records_skipped"] == len(bad) + len(extra)
    assert result["cameras_selected"] == 1 and result["frames_ok"] == 1
    assert "/image/bad" not in site.requests


@pytest.mark.parametrize(
    "body",
    [
        b"{}",
        b'{"records": []}',
        b'"cameras"',
        b"null",
        b"[1, 2",
        b"not json",
        b"\xff\xfe\x00[",
        b"[" * 100_000 + b"]" * 100_000,
        b"",
    ],
)
def test_malformed_top_level_is_a_typed_error(
    site: Site, capsys: pytest.CaptureFixture[str], body: bytes
) -> None:
    site.routes["/dataset"] = Route(body=body)
    assert _run(site) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err and captured.err.strip()
    assert [p for p in site.requests if p.startswith("/image/")] == []


def test_oversized_camera_list_is_a_typed_error(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    site.routes["/dataset"] = Route(body=b"[" + b" " * pilot_heights.MAX_DATASET_BYTES + b"]")
    assert _run(site) == 1
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("status", [301, 302, 404, 500])
def test_camera_list_errors_and_redirects_are_typed_errors(
    site: Site, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    site.routes["/dataset"] = Route(status, b"[]", {"Location": site.url("/other")})
    site.routes["/other"] = Route(body=b"[]")
    assert _run(site) == 1
    assert capsys.readouterr().out == ""
    assert "/other" not in site.requests
    assert site.requests.count("/dataset") == 1  # one request, no retry


def test_camera_list_from_another_host_is_refused(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    site.dataset([])
    code = pilot_heights.main(
        ["--live"],
        now=lambda: DAY,
        open_detector=lambda model: Stub(),
        dataset_url=site.url("/dataset"),
        dataset_policy=pilot_heights.DATASET_POLICY,
        image_policy=site.policy(),
    )
    assert code == 1
    assert site.requests == []
    assert capsys.readouterr().out == ""


# AC3: host allowlist ------------------------------------------------------------------


def test_screenshot_host_is_pinned() -> None:
    policy = pilot_heights.SCREENSHOT_POLICY
    assert policy == pilot_heights.UrlPolicy("https", "cctv.austinmobility.io")
    assert policy.allows("https://cctv.austinmobility.io/image/123.jpg")
    for url in [
        "http://cctv.austinmobility.io/image/123.jpg",
        "https://cctv.austinmobility.io:8443/image/123.jpg",
        "https://user:pw@cctv.austinmobility.io/image/123.jpg",
        "https://user@cctv.austinmobility.io/image/123.jpg",
        "https://cctv.austinmobility.io.evil.example/image/123.jpg",
        "https://evil.example/cctv.austinmobility.io/image/123.jpg",
        "https://sub.cctv.austinmobility.io/image/123.jpg",
        "ftp://cctv.austinmobility.io/image/123.jpg",
        "file:///etc/passwd",
        "//cctv.austinmobility.io/image/123.jpg",
        "https://cctv.austinmobility.io\\@evil.example/image/123.jpg",
        "https://cctv.austinmobility.io/image/1 23.jpg",
        "https://cctv.austinmobility.io/ima\nge/123.jpg",
        "",
    ]:
        assert not policy.allows(url), url


def test_refused_urls_are_counted_and_never_requested(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = site.image("ok", _jpeg(352, 288, 0))
    host, port = site.netloc.split(":")
    refused = [
        f"http://localhost:{port}/image/ok",  # another host
        f"http://user:pw@{site.netloc}/image/ok",  # userinfo
        f"http://user@{site.netloc}/image/ok",
        f"http://{host}:{int(port) + 1}/image/ok",  # another port
        f"http://{host}/image/ok",  # the default port, not the pinned one
        f"https://{site.netloc}/image/ok",  # another scheme
        f"ftp://{site.netloc}/image/ok",
    ]
    site.dataset([_record(ok), *[_record(u) for u in refused]])
    assert _run(site) == 0
    result = _output(capsys)
    assert result["refused_url"] == len(refused)
    assert result["cameras_selected"] == 1 and result["frames_ok"] == 1
    assert site.requests.count("/image/ok") == 1


def test_redirects_are_refused_and_not_followed(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    target = site.image("target", _jpeg(352, 288, 0))
    moved = [
        site.image(f"moved{code}", b"", status=code, headers={"Location": target})
        for code in (301, 302, 303, 307, 308)
    ]
    site.dataset([_record(u) for u in moved])
    assert _run(site) == 0
    result = _output(capsys)
    assert result["refused_url"] == len(moved)
    assert result["frames_ok"] == 0
    assert "/image/target" not in site.requests
    assert sum(result["frames_failed"].values()) == 0


# AC4: frames in memory ----------------------------------------------------------------


def test_failures_are_counted_by_category(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    urls = [
        site.image("ok", _jpeg(352, 288, 0)),
        site.image("missing", b"", status=404),
        site.image("broken", b"", status=500),
        site.image("text", b"<html>no image</html>"),
        site.image("png", b"\x89PNG\r\n\x1a\n" + b"\0" * 64),
        site.image("truncated", _jpeg(352, 288, 0)[:-200]),
        site.image("huge", jpeg_declaring(30000, 30000)),
        site.image("toobig", b"\xff\xd8\xff" + b"\0" * fetch.MAX_FRAME_BYTES + b"\xff\xd9"),
        site.image("slow", _jpeg(352, 288, 0), delay_s=30.0),
    ]
    site.dataset([_record(u) for u in urls])
    started = time.monotonic()
    assert _run(site, ["--live", "--timeout", "3"]) == 0
    assert time.monotonic() - started < 20
    result = _output(capsys)
    assert result["cameras_selected"] == len(urls)
    assert result["frames_ok"] == 1
    failed = result["frames_failed"]
    assert set(fetch.ERROR_KINDS) <= set(failed)
    assert failed["http"] == 3  # 404, 500 and the body over the cap
    assert failed["decode"] == 4  # text, png, truncated, a declared 30000x30000
    assert failed["timeout"] == 1
    assert sum(failed.values()) == len(urls) - 1
    for name in ("ok", "missing", "toobig", "slow"):
        assert site.requests.count(f"/image/{name}") == 1  # one attempt per camera


def test_shared_deadline_bounds_the_whole_pass(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    urls = [site.image(f"slow{i}", _jpeg(352, 288, 0), delay_s=30.0) for i in range(40)]
    site.dataset([_record(u) for u in urls])
    started = time.monotonic()
    assert _run(site, ["--live", "--timeout", "2"]) == 0
    assert time.monotonic() - started < 15
    result = _output(capsys)
    assert result["frames_ok"] == 0
    assert result["frames_failed"]["timeout"] == 40


def test_output_names_no_camera_url_or_box(
    site: Site, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    urls = [site.image(f"secretcam{i}", _jpeg(352, 288, 1)) for i in range(3)]
    urls.append(site.image("secretcam404", b"", status=404))
    site.dataset([_record(u) for u in urls])
    stub = Stub({1: [("person", (12.25, 33.5, 47.75, 101.125))]})
    assert _run(site, stub=stub) == 0
    captured = capsys.readouterr()
    everything = captured.out + captured.err + caplog.text
    for needle in ("secretcam", "/image/", site.netloc, "127.0.0.1", "12.25", "33.5", "101.125"):
        assert needle not in everything
    assert "-97.745" not in everything and "30.27" not in everything


# AC5: daylight only -------------------------------------------------------------------


def test_refuses_at_night_without_network(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    site.dataset([_record(site.image("c", _jpeg(352, 288, 0)))])
    opened: list[str] = []
    assert _run(site, now=NIGHT, opened=opened) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "sun" in captured.err.lower() or "daylight" in captured.err.lower()
    assert site.requests == [] and opened == []


def test_daylight_uses_the_spot_check_formula_at_central_austin() -> None:
    assert pilot_heights.AUSTIN == (30.2672, -97.7431)
    moment = DAY
    for _ in range(200):
        expected = spotcheck.solar_elevation(moment, 30.2672, -97.7431)
        assert pilot_heights.solar_elevation(moment, 30.2672, -97.7431) == expected
        moment += datetime.timedelta(minutes=37)


def test_sun_elevation_is_reported(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    site.dataset([])
    assert _run(site) == 0
    result = _output(capsys)
    expected = spotcheck.solar_elevation(DAY, 30.2672, -97.7431)
    assert expected > 0
    assert result["sun_elevation_deg"] == round(expected, 1)
    assert result["started_at"] == "2026-10-01T18:30Z"


# AC6 and AC7: output and heights ------------------------------------------------------


def test_output_key_set_is_exact(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    site.dataset([])
    assert _run(site) == 0
    result = _output(capsys)
    assert set(result) == KEYS
    assert result["source"] == "austin"
    assert list(result["persons_by_height_band"]) == BANDS
    assert list(result["persons_by_height_band_1080p"]) == BANDS
    assert result["cameras_listed"] == 0 and result["frames_ok"] == 0
    assert result["persons_total"] == 0 and result["umbrellas_total"] == 0


@pytest.mark.parametrize(
    ("height", "band"),
    [
        (0, "<31"),
        (30, "<31"),
        (31, "31-45"),
        (45, "31-45"),
        (46, "46-79"),
        (79, "46-79"),
        (80, "80-119"),
        (119, "80-119"),
        (120, "120-199"),
        (199, "120-199"),
        (200, "200+"),
        (1080, "200+"),
    ],
)
def test_band_edges(height: int, band: str) -> None:
    assert pilot_heights.height_band(height) == band


def test_bands_over_frames(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    heights = [30, 31, 45, 46, 79, 80, 119, 120, 199, 200]
    small: list[tuple[detect.Label, Box]] = [("person", _box(h)) for h in heights]
    small.append(("person", (0.0, 10.4, 5.0, 41.0)))  # 30.6 px: box_height rounds to 31
    small.append(("umbrella", _box(50)))
    # Boxes are in the pixels of the frame the detector is given; the bands count
    # heights in pixels of the original frame.
    hd: list[tuple[detect.Label, Box]] = [
        ("person", _box(15.5)),
        ("person", _box(40.0)),
        ("umbrella", _box(10)),
    ]
    wide: list[tuple[detect.Label, Box]] = [("person", _box(100))]
    stub = Stub({0: small, 1: hd, 2: wide})
    urls = [site.image("small", _jpeg(352, 288, 0))]
    urls += [site.image(f"hd{i}", _jpeg(1920, 1080, 1)) for i in range(2)]
    urls.append(site.image("wide", _jpeg(1280, 720, 2)))
    site.dataset([_record(u) for u in urls])
    assert _run(site, stub=stub) == 0
    result = _output(capsys)
    assert result["frames_ok"] == 4
    assert result["resolutions"] == {"352x288": 1, "1920x1080": 2, "1280x720": 1}
    assert result["persons_total"] == len(heights) + 1 + 2 * 2 + 1
    assert result["umbrellas_total"] == 1 + 2

    # The whole 1080p frame reaches the detector, at full size or uniformly reduced.
    (hd_shape,) = set(stub.shapes[1])
    assert hd_shape[2] == 3 and hd_shape[1] * 1080 == hd_shape[0] * 1920
    factor = 1080 / hd_shape[0]
    expected_hd = dict.fromkeys(BANDS, 0)
    for _label, (x1, y1, x2, y2) in hd[:2]:
        height = aggregate.box_height((x1 * factor, y1 * factor, x2 * factor, y2 * factor))
        expected_hd[pilot_heights.height_band(height)] += 2
    assert result["persons_by_height_band_1080p"] == expected_hd

    (small_shape,) = set(stub.shapes[0])
    assert small_shape == (288, 352, 3)
    expected = dict(expected_hd)
    for h in [*heights, 31]:
        expected[pilot_heights.height_band(h)] += 1
    (wide_shape,) = set(stub.shapes[2])
    wide_factor = 720 / wide_shape[0]
    expected[pilot_heights.height_band(round(100 * wide_factor))] += 1
    assert result["persons_by_height_band"] == expected
    assert sum(expected.values()) == result["persons_total"]


def test_heights_use_the_aggregate_rule(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    stub = Stub({0: [("person", (0.0, 10.4, 5.0, 41.0)), ("person", (0.0, 10.0, 5.0, 40.4))]})
    site.dataset([_record(site.image("c", _jpeg(352, 288, 0)))])
    assert _run(site, stub=stub) == 0
    bands = _output(capsys)["persons_by_height_band"]
    assert aggregate.box_height((0.0, 10.4, 5.0, 41.0)) == 31
    assert aggregate.box_height((0.0, 10.0, 5.0, 40.4)) == 30
    assert bands["31-45"] == 1 and bands["<31"] == 1


def test_detector_errors_are_counted_not_fatal(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    class Broken(Stub):
        def detect(self, frame: Frame) -> list[detect.Detection]:
            if int(frame[frame.shape[0] // 2, frame.shape[1] // 2, 0]) // 10 == 1:
                raise detect.DetectorError("broken output")
            return super().detect(frame)

    site.dataset(
        [
            _record(site.image("a", _jpeg(352, 288, 0))),
            _record(site.image("b", _jpeg(352, 288, 1))),
        ]
    )
    assert _run(site, stub=Broken()) == 0
    result = _output(capsys)
    assert result["frames_ok"] == 1
    assert sum(result["frames_failed"].values()) == 1


def test_default_detector_is_the_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[tuple[Path, dict[str, object]]] = []

    class Recorder:
        def __init__(self, path: Path, **kwargs: object) -> None:
            made.append((Path(path), kwargs))

    monkeypatch.setattr(detect, "Detector", Recorder)
    pilot_heights.open_detector("yolox_m")
    pilot_heights.open_detector("yolox_s")
    assert made == [
        (detect.model_path("yolox_m.onnx"), {}),
        (detect.model_path("yolox_s.onnx"), {}),
    ]


# AC8: nothing on disk -----------------------------------------------------------------


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def test_nothing_is_written_to_disk(
    site: Site,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp = tmp_path / "tmp"
    work = tmp_path / "work"
    temp.mkdir()
    work.mkdir()
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(temp))
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    monkeypatch.chdir(work)
    system_temp = Path(tempfile.gettempdir())
    assert system_temp == temp
    urls = [site.image(f"c{i}", _jpeg(1920, 1080, 1)) for i in range(4)]
    urls.append(site.image("bad", b"\xff\xd8\xff" + os.urandom(4096)))
    site.dataset([_record(u) for u in urls])
    before = (_tree(temp), _tree(work), _tree(ROOT / "engine"))
    assert _run(site, stub=Stub({1: [("person", _box(60))]})) == 0
    assert (_tree(temp), _tree(work), _tree(ROOT / "engine")) == before
    assert _output(capsys)["frames_ok"] == 4


def test_privacy_guard_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "privacy_guard.py")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert MODULE.is_file()


def test_no_new_dependency() -> None:
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    lines = [line.strip().strip('",') for line in deps.splitlines() if line.strip()]
    names = sorted(line.split("=")[0].split(">")[0] for line in lines)
    assert names == ["llama-cpp-python", "numpy", "onnxruntime", "opencv-python-headless"]
