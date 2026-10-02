"""Unit tests for the spot-check tool's Austin sessions (T-059): the frame pass, the
display size, the record and the light. Frames are synthetic and served on 127.0.0.1."""

from __future__ import annotations

import argparse
import datetime
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from wearreport import detect
from wearreport._cv import encode_jpeg
from wearreport.tools import pilot_heights, spotcheck

INSIDE = [-97.745, 30.270]  # lon, lat: inside the default box
DAY = datetime.date(2026, 10, 1)
AUSTIN_NIGHT = datetime.datetime(2026, 10, 1, 8, 0, tzinfo=datetime.UTC)
LONDON_NIGHT = datetime.datetime(2026, 10, 1, 21, 0, tzinfo=datetime.UTC)


class Host:
    """Serves `routes` (path -> (status, body)) on 127.0.0.1, after `delay_s`."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, bytes]] = {}
        self.delay_s = 0.0
        self.hits: list[str] = []
        self.lock = threading.Lock()
        host = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                with host.lock:
                    host.hits.append(self.path)
                status, body = host.routes.get(self.path, (404, b""))
                if self.path != "/dataset":
                    time.sleep(host.delay_s)
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass  # the client gave up at its deadline

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def endpoints(self, dataset: list[str]) -> spotcheck.AustinEndpoints:
        host, port = self.httpd.server_address[:2]
        netloc = f"{host!s}:{port}"
        records = [
            {
                "screenshot_address": f"http://{netloc}{path}",
                "location": {"type": "Point", "coordinates": INSIDE},
            }
            for path in dataset
        ]
        self.routes["/dataset"] = (200, json.dumps(records).encode())
        policy = pilot_heights.UrlPolicy("http", netloc)
        return spotcheck.AustinEndpoints(f"http://{netloc}/dataset", policy, policy)


@pytest.fixture
def host() -> Iterator[Host]:
    h = Host()
    try:
        yield h
    finally:
        h.httpd.shutdown()
        h.httpd.server_close()


def _jpeg(width: int, height: int) -> bytes:
    return encode_jpeg(np.full((height, width, 3), 90, dtype=np.uint8))


def test_frames_are_full_resolution_and_counted(
    host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    host.routes |= {
        "/a": (200, _jpeg(1920, 1080)),
        "/b": (200, _jpeg(320, 176)),
        "/c": (500, b""),
        "/d": (200, b"\xff\xd8\xff" + b"\0" * 64),
    }
    endpoints = host.endpoints(["/a", "/b", "/c", "/d"])
    frames = list(spotcheck.austin_frames(endpoints, pilot_heights.DEFAULT_BBOX))
    assert [f.shape for f in frames] == [(1080, 1920, 3)]
    err = capsys.readouterr().err
    assert "4 Austin cameras listed, 4 selected" in err
    assert (
        "fetched 1 1920x1080 frame(s) of 4: 1 not 1920x1080 (skipped), 2 failed, 0 refused" in err
    )
    assert "127.0.0.1" not in err


def test_a_camera_list_failure_is_a_spotcheck_error(host: Host) -> None:
    endpoints = host.endpoints([])
    host.routes["/dataset"] = (200, b"{}")
    with pytest.raises(spotcheck.SpotcheckError, match="Austin cameras"):
        list(spotcheck.austin_frames(endpoints, pilot_heights.DEFAULT_BBOX))


def test_the_deadline_bounds_the_pass(host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    host.routes["/a"] = (200, _jpeg(1920, 1080))
    endpoints = host.endpoints(["/a"] * 3)
    host.delay_s = 0.3
    started = time.monotonic()
    frames = list(spotcheck.austin_frames(endpoints, pilot_heights.DEFAULT_BBOX, timeout_s=0.2))
    assert frames == []
    assert time.monotonic() - started < 5
    assert "3 failed" in capsys.readouterr().err


def test_closing_early_stops_the_pass(host: Host) -> None:
    host.routes["/a"] = (200, _jpeg(1920, 1080))
    endpoints = host.endpoints(["/a"] * 20)
    frames = spotcheck.austin_frames(endpoints, pilot_heights.DEFAULT_BBOX, concurrency=2)
    first = next(frames)
    assert first.shape == (1080, 1920, 3)
    frames.close()
    time.sleep(0.2)
    with host.lock:
        assert host.hits.count("/a") <= 4  # at most the window in flight, never all 20


@pytest.mark.parametrize(
    "size", [(0, 10, 10, 10), (10, 0, 10, 10), (10, 10, 0, 10), (10, 10, 10, 0)]
)
def test_display_size_refuses_empty_sizes(size: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        spotcheck.display_size(*size)


def test_display_size_never_exceeds_the_limit() -> None:
    for height in (1, 7, 31, 99, 240, 481, 1080):
        for width in (1, 13, 60, 400, 1201, 1920):
            for limit in ((50, 50), (700, 1200), (1080, 1920)):
                h, w = spotcheck.display_size(height, width, *limit)
                assert 1 <= h <= max(limit[0], 1) and 1 <= w <= max(limit[1], 1)
                assert h <= height * spotcheck.window_scale(height, width)


def test_sourced_record_keeps_london_and_marks_austin() -> None:
    record: dict[str, object] = {"date": "2026-10-01", "light": "dark", "crops": []}
    assert spotcheck.sourced_record(record, "london", LONDON_NIGHT) is record
    austin = spotcheck.sourced_record(record, "austin", LONDON_NIGHT)
    assert list(austin) == ["date", "light", "crops", "source"]
    assert austin["light"] == "day" and austin["source"] == "austin"


def test_write_attributes_names_austin_files(tmp_path: Path) -> None:
    first = spotcheck.write_attributes({"a": 1}, tmp_path, DAY, "austin")
    second = spotcheck.write_attributes({"a": 2}, tmp_path, DAY, "austin")
    london = spotcheck.write_attributes({"a": 3}, tmp_path, DAY)
    assert [p.name for p in (first, second, london)] == [
        "2026-10-01-austin.json",
        "2026-10-01-austin-2.json",
        "2026-10-01.json",
    ]


def _namespace(**changes: object) -> argparse.Namespace:
    values: dict[str, object] = {"judgements": None, "allow_dark": False, "source": "austin"}
    return argparse.Namespace(**(values | changes))


def test_check_light_uses_the_sources_sun() -> None:
    with pytest.raises(spotcheck.SpotcheckError, match="Austin"):
        spotcheck.check_light(_namespace(), lambda: AUSTIN_NIGHT)
    spotcheck.check_light(_namespace(), lambda: LONDON_NIGHT)
    with pytest.raises(spotcheck.SpotcheckError, match="London"):
        spotcheck.check_light(_namespace(source="london"), lambda: LONDON_NIGHT)
    spotcheck.check_light(_namespace(allow_dark=True), lambda: AUSTIN_NIGHT)
    spotcheck.check_light(_namespace(judgements="j.json"), lambda: AUSTIN_NIGHT)


def test_a_namespace_without_a_source_is_london() -> None:
    args = argparse.Namespace(judgements=None, allow_dark=False)
    with pytest.raises(spotcheck.SpotcheckError, match="London"):
        spotcheck.check_light(args, lambda: LONDON_NIGHT)


def test_the_dry_run_stills_are_1080p() -> None:
    body = (spotcheck.FIXTURE_DIR / spotcheck.DRY_RUN_FIXTURES[0]).read_bytes()
    still = spotcheck._hd_still(body)
    assert pilot_heights.jpeg_size(still) == (1920, 1080)
    assert pilot_heights.decode_frame(still).frame.shape == (1080, 1920, 3)


def test_the_detector_runs_on_a_whole_still() -> None:
    assert detect.MAX_IMAGE_PIXELS == 1920 * 1080  # type: ignore[attr-defined]
