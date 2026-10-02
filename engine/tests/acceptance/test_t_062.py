"""Acceptance tests for T-062: an Austin session's progress line says how the failed
frames failed, by the existing failure kinds. The task contract: do not edit.

Every still here is synthetic (a uniform colour, encoded in memory), the camera list is
written by the test, and both are served by a local HTTP server on 127.0.0.1; the
network failure is a closed port on 127.0.0.1. Nothing reaches the network.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from wearreport import fetch
from wearreport._cv import encode_jpeg
from wearreport.tools import pilot_heights, spotcheck

INSIDE = [-97.745, 30.270]  # lon, lat: inside the default box


class Host:
    """Serves `routes` (path -> (status, body, delay_s)) on 127.0.0.1. The status 0 drops
    the connection without an answer."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, bytes, float]] = {}
        host = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                status, body, delay_s = host.routes.get(self.path, (404, b"", 0.0))
                time.sleep(delay_s)
                try:
                    if status == 0:
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                        return
                    self.send_response(status)
                    if 300 <= status < 400:
                        self.send_header("Location", "/elsewhere")
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
        self.netloc = "{}:{}".format(*self.httpd.server_address[:2])

    def endpoints(
        self, paths: list[str], image_netloc: str | None = None
    ) -> spotcheck.AustinEndpoints:
        netloc = image_netloc or self.netloc
        records = [
            {
                "screenshot_address": f"http://{netloc}{path}",
                "location": {"type": "Point", "coordinates": INSIDE},
            }
            for path in paths
        ]
        self.routes["/dataset"] = (200, json.dumps(records).encode(), 0.0)
        return spotcheck.AustinEndpoints(
            f"http://{self.netloc}/dataset",
            pilot_heights.UrlPolicy("http", self.netloc),
            pilot_heights.UrlPolicy("http", netloc),
        )


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


def _truncated_hd() -> bytes:
    body = _jpeg(1920, 1080)
    return body[: len(body) // 2]  # the 1920x1080 header intact, the end missing


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return port


def _fetched_line(
    endpoints: spotcheck.AustinEndpoints, capsys: pytest.CaptureFixture[str], timeout_s: float
) -> tuple[str, str]:
    """The `fetched ...` progress line of one pass, and everything it printed."""
    capsys.readouterr()
    list(spotcheck.austin_frames(endpoints, pilot_heights.DEFAULT_BBOX, timeout_s=timeout_s))
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.startswith("spotcheck: fetched ")]
    assert len(lines) == 1, err
    return lines[0], err


# AC1 and AC4: one case per kind ---------------------------------------------------------


def test_ac1_timeout(host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    host.routes["/slow"] = (200, _jpeg(1920, 1080), 3.0)
    line, _ = _fetched_line(host.endpoints(["/slow"]), capsys, timeout_s=0.5)
    assert line == (
        "spotcheck: fetched 0 1920x1080 frame(s) of 1: 0 not 1920x1080 (skipped), "
        "1 failed (timeout 1), 0 refused"
    )


def test_ac1_network(host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    closed = f"127.0.0.1:{_closed_port()}"
    line, _ = _fetched_line(host.endpoints(["/a", "/b"], image_netloc=closed), capsys, 10.0)
    assert line == (
        "spotcheck: fetched 0 1920x1080 frame(s) of 2: 0 not 1920x1080 (skipped), "
        "2 failed (network 2), 0 refused"
    )


def test_ac1_http(host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    host.routes["/err"] = (500, b"", 0.0)
    line, _ = _fetched_line(host.endpoints(["/err"] * 3), capsys, 10.0)
    assert line == (
        "spotcheck: fetched 0 1920x1080 frame(s) of 3: 0 not 1920x1080 (skipped), "
        "3 failed (http 3), 0 refused"
    )


def test_ac1_decode(host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    host.routes["/cut"] = (200, _truncated_hd(), 0.0)
    line, _ = _fetched_line(host.endpoints(["/cut"]), capsys, 10.0)
    assert line == (
        "spotcheck: fetched 0 1920x1080 frame(s) of 1: 0 not 1920x1080 (skipped), "
        "1 failed (decode 1), 0 refused"
    )


def test_ac1_no_failure_keeps_todays_line(host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    host.routes |= {
        "/hd": (200, _jpeg(1920, 1080), 0.0),
        "/small": (200, _jpeg(320, 176), 0.0),
        "/moved": (302, b"", 0.0),
    }
    line, _ = _fetched_line(host.endpoints(["/hd", "/small", "/moved"]), capsys, 10.0)
    assert line == (
        "spotcheck: fetched 1 1920x1080 frame(s) of 3: 1 not 1920x1080 (skipped), "
        "0 failed, 1 refused"
    )


def test_ac1_mixed_kinds_in_error_kinds_order(
    host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fetch.ERROR_KINDS == ("timeout", "http", "decode", "network")
    host.routes |= {
        "/drop": (0, b"", 0.0),
        "/cut": (200, _truncated_hd(), 0.0),
        "/err": (500, b"", 0.0),
        "/slow": (200, _jpeg(1920, 1080), 4.0),
        "/hd": (200, _jpeg(1920, 1080), 0.0),
        "/small": (200, _jpeg(320, 176), 0.0),
        "/moved": (302, b"", 0.0),
    }
    paths = ["/drop", "/cut", "/err", "/err", "/slow", "/hd", "/small", "/moved"]
    line, err = _fetched_line(host.endpoints(paths), capsys, timeout_s=1.5)
    assert line == (
        "spotcheck: fetched 1 1920x1080 frame(s) of 8: 1 not 1920x1080 (skipped), "
        "5 failed (timeout 1, http 2, decode 1, network 1), 1 refused"
    )
    # AC2: counts only.
    port = host.netloc.rsplit(":", 1)[1]
    for leak in ("127.0.0.1", port, "/drop", "/cut", "/err", "/slow", "http://", "Error"):
        assert leak not in err


def test_ac1_the_deadline_backstop_counts_as_timeout(
    host: Host, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def stuck(url: str, scheme: str, end: float) -> object:
        release.wait(10.0)  # outlives the deadline and its backstop
        raise pilot_heights.FrameFailed("network")

    monkeypatch.setattr(spotcheck, "_austin_still", stuck)
    try:
        line, _ = _fetched_line(host.endpoints(["/a"] * 3), capsys, timeout_s=0.2)
    finally:
        release.set()
    assert line == (
        "spotcheck: fetched 0 1920x1080 frame(s) of 3: 0 not 1920x1080 (skipped), "
        "3 failed (timeout 3), 0 refused"
    )


# AC2: counts only -----------------------------------------------------------------------


def test_ac2_no_url_host_or_exception_text(host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    port = _closed_port()
    closed = f"127.0.0.1:{port}"
    _, err = _fetched_line(host.endpoints(["/camera-7"], image_netloc=closed), capsys, 10.0)
    for leak in ("127.0.0.1", str(port), "camera-7", "http://", "Errno", "Error"):
        assert leak not in err
