from __future__ import annotations

import base64
import contextlib
import logging
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from wearreport import fetch, registry
from wearreport._cv import cv2, encode_jpeg
from wearreport.testing.fake_cameras import FakeCameraServer, jpeg_declaring, radiance_hdr


@pytest.fixture
def server() -> Iterator[FakeCameraServer]:
    with FakeCameraServer() as srv:
        yield srv


def _camera(url: str, camera_id: str = "cam") -> registry.Camera:
    return registry.Camera(id=camera_id, name=camera_id, lat=51.5, lon=-0.1, image_url=url)


def _refused_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/cam/refused"


def _logged(caplog: pytest.LogCaptureFixture) -> str:
    """Everything the captured records hold, `extra=` fields included.

    caplog.text only renders the message; a structured (JSON) formatter would also emit
    every attribute that `extra=` put on the record.
    """
    return caplog.text + "".join(repr(vars(record)) for record in caplog.records)


def _jpeg() -> bytes:
    return encode_jpeg(np.random.default_rng().integers(0, 256, (288, 352, 3), dtype=np.uint8))


@pytest.fixture
def imdecode_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record every call to cv2.imdecode (the object fetch.py decodes with)."""
    calls: list[int] = []
    real = cv2.imdecode

    def spy(buf: Any, flags: int) -> Any:
        calls.append(len(buf))
        return real(buf, flags)

    monkeypatch.setattr(cv2, "imdecode", spy)
    return calls


class _Quiet(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextlib.contextmanager
def _serve(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    """A one-route server for responses the fake camera server does not offer."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    httpd.block_on_close = False
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/x"
    finally:
        httpd.shutdown()
        httpd.server_close()


# Single outcomes -------------------------------------------------------------------------


def test_success_returns_decoded_bgr_frame(server: FakeCameraServer) -> None:
    (cam,) = server.cameras(1)
    (result,) = fetch.fetch_sweep([cam])
    assert result.camera_id == cam.id
    assert result.error is None
    assert result.frame is not None
    assert result.frame.shape == (288, 352, 3)
    assert result.frame.dtype == np.uint8
    assert 0 <= result.seconds < 20


def test_http_404_is_an_http_error(server: FakeCameraServer) -> None:
    (cam,) = server.cameras(1)
    server.serve_404(cam.id)
    (result,) = fetch.fetch_sweep([cam])
    assert (result.error, result.frame) == ("http", None)
    assert server.requests(cam.id) == 1


def test_http_500_is_an_http_error_and_not_retried() -> None:
    hits: list[int] = []

    class Broken(_Quiet):
        def do_GET(self) -> None:
            hits.append(1)
            self.send_error(500)

    with _serve(Broken) as url:
        (result,) = fetch.fetch_sweep([_camera(url)])
    assert result.error == "http"
    assert hits == [1]


def test_timeout_is_a_timeout_error(server: FakeCameraServer) -> None:
    (cam,) = server.cameras(1)
    server.serve_delay(cam.id, 5)
    started = time.monotonic()
    (result,) = fetch.fetch_sweep([cam], timeout_s=0.3)
    assert (result.error, result.frame) == ("timeout", None)
    assert result.seconds < 2
    assert time.monotonic() - started < 2
    assert server.requests(cam.id) == 1


def test_slow_body_hits_the_per_frame_deadline() -> None:
    """A server that trickles bytes never trips the socket timeout, only the deadline."""

    class Trickle(_Quiet):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"\xff" * 10)
                    self.wfile.flush()
                    time.sleep(0.1)
            except OSError:
                pass

    with _serve(Trickle) as url:
        started = time.monotonic()
        (result,) = fetch.fetch_sweep([_camera(url)], timeout_s=0.5)
        elapsed = time.monotonic() - started
    assert result.error == "timeout"
    assert elapsed < 2


def test_corrupt_jpeg_is_a_decode_error(server: FakeCameraServer) -> None:
    (cam,) = server.cameras(1)
    server.serve_corrupt(cam.id)
    (result,) = fetch.fetch_sweep([cam])
    assert (result.error, result.frame) == ("decode", None)


def test_empty_body_is_a_decode_error() -> None:
    class Empty(_Quiet):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    with _serve(Empty) as url:
        (result,) = fetch.fetch_sweep([_camera(url)])
    assert result.error == "decode"


def test_connection_refused_is_a_network_error() -> None:
    (result,) = fetch.fetch_sweep([_camera(_refused_url())])
    assert (result.error, result.frame) == ("network", None)


def test_connection_dropped_mid_response_is_a_network_error() -> None:
    class Drop(_Quiet):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "5000")
            self.end_headers()
            self.wfile.write(b"\xff\xd8\xff" + b"\0" * 100)
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)

    with _serve(Drop) as url:
        (result,) = fetch.fetch_sweep([_camera(url)])
    assert result.error == "network"


def test_oversized_body_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "MAX_FRAME_BYTES", 1000)

    class Big(_Quiet):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()  # no Content-Length: only the running count can stop it
            with contextlib.suppress(OSError):
                self.wfile.write(b"\xff" * 100_000)

    with _serve(Big) as url:
        (result,) = fetch.fetch_sweep([_camera(url)])
    assert result.error == "http"


def test_oversized_content_length_is_refused_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch, "MAX_FRAME_BYTES", 1000)

    class Declared(_Quiet):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "5000")
            self.end_headers()
            self.wfile.write(b"\xff" * 5000)

    with _serve(Declared) as url:
        (result,) = fetch.fetch_sweep([_camera(url)])
    assert result.error == "http"


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://127.0.0.1/x", "data:image/jpeg;base64,/9j/", "nonsense"]
)
def test_non_http_urls_are_never_opened(url: str) -> None:
    (result,) = fetch.fetch_sweep([_camera(url)])
    assert (result.error, result.frame) == ("network", None)


def test_redirect_to_a_file_url_is_not_followed() -> None:
    class Redirect(_Quiet):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header("Location", "file:///etc/hostname")
            self.send_header("Content-Length", "0")
            self.end_headers()

    with _serve(Redirect) as url:
        (result,) = fetch.fetch_sweep([_camera(url)])
    assert result.error == "http"
    assert result.frame is None


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_an_http_error_after_exactly_one_request(status: int) -> None:
    paths: list[str] = []

    class Redirect(_Quiet):
        def do_GET(self) -> None:
            paths.append(self.path)
            if self.path == "/x":
                self.send_response(status)
                self.send_header("Location", "/frame")  # would succeed if followed
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = _jpeg()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with _serve(Redirect) as url:
        (result,) = fetch.fetch_sweep([_camera(url)])
    assert (result.error, result.frame) == ("http", None)
    assert paths == ["/x"]


def test_trickled_headers_hit_the_wall_clock_deadline() -> None:
    """Each byte resets the socket timeout; only the per-frame deadline can end this."""

    class TrickleHeaders(_Quiet):
        def do_GET(self) -> None:
            with contextlib.suppress(OSError):
                self.wfile.write(b"HTTP/1.1 200 OK\r\n")
                self.wfile.flush()
                for _ in range(60):
                    self.wfile.write(b"X")
                    self.wfile.flush()
                    time.sleep(0.9)

    with _serve(TrickleHeaders) as url:
        started = time.monotonic()
        (result,) = fetch.fetch_sweep([_camera(url)], timeout_s=1)
        elapsed = time.monotonic() - started
    assert (result.error, result.frame) == ("timeout", None)
    assert elapsed < 1 + 1


def test_non_jpeg_body_is_refused_without_decoding(
    server: FakeCameraServer, imdecode_calls: list[int]
) -> None:
    """A Radiance HDR body would make OpenCV write it to a temporary file."""
    (cam,) = server.cameras(1)
    body = radiance_hdr()
    assert body.startswith(b"#?RADIANCE\n")
    server.serve_body(cam.id, body)
    (result,) = fetch.fetch_sweep([cam])
    assert (result.error, result.frame) == ("decode", None)
    assert imdecode_calls == []
    assert server.requests(cam.id) == 1


@pytest.mark.parametrize(
    "body",
    [b"\x89PNG\r\n\x1a\n" + bytes(64) + b"\xff\xd9", b"GIF89a\xff\xd9", b"\xff\xd8", b"\xff\xd9"],
    ids=["png", "gif", "soi-only", "eoi-only"],
)
def test_bodies_without_a_jpeg_start_are_not_decoded(
    body: bytes, server: FakeCameraServer, imdecode_calls: list[int]
) -> None:
    (cam,) = server.cameras(1)
    server.serve_body(cam.id, body)
    (result,) = fetch.fetch_sweep([cam])
    assert result.error == "decode"
    assert imdecode_calls == []


def test_decompression_bomb_is_a_decode_error_within_2_s(server: FakeCameraServer) -> None:
    (cam,) = server.cameras(1)
    body = jpeg_declaring(30000, 30000)
    assert len(body) < 2048
    server.serve_body(cam.id, body)
    started = time.monotonic()
    (result,) = fetch.fetch_sweep([cam])
    assert (result.error, result.frame) == ("decode", None)
    assert time.monotonic() - started < 2


def test_truncated_jpeg_without_end_marker_is_a_decode_error(
    server: FakeCameraServer, imdecode_calls: list[int]
) -> None:
    """Content-Length matches, but the image stops early: libjpeg would fill it with grey."""
    (cam,) = server.cameras(1)
    server.serve_body(cam.id, _jpeg()[:-200])
    (result,) = fetch.fetch_sweep([cam])
    assert (result.error, result.frame) == ("decode", None)
    assert imdecode_calls == []


def test_end_marker_followed_by_padding_is_accepted(server: FakeCameraServer) -> None:
    (cam,) = server.cameras(1)
    server.serve_body(cam.id, _jpeg() + bytes(fetch.EOI_WINDOW - 2))
    (result,) = fetch.fetch_sweep([cam])
    assert result.error is None and result.frame is not None


def test_end_marker_too_far_from_the_end_is_a_decode_error(server: FakeCameraServer) -> None:
    (cam,) = server.cameras(1)
    server.serve_body(cam.id, _jpeg() + bytes(fetch.EOI_WINDOW))
    (result,) = fetch.fetch_sweep([cam])
    assert result.error == "decode"


# A mix of everything in one sweep ------------------------------------------------------


def test_mixed_sweep_records_every_outcome_once(server: FakeCameraServer) -> None:
    cams = server.cameras(8)
    server.serve_404(cams[1].id)
    server.serve_delay(cams[2].id, 5)
    server.serve_corrupt(cams[3].id)
    server.serve_404(cams[5].id)
    cams.insert(4, _camera(_refused_url(), "refused"))
    results = fetch.fetch_sweep(cams, concurrency=4, timeout_s=0.5)
    assert [r.camera_id for r in results] == [c.id for c in cams]
    assert [r.error for r in results] == [
        None,
        "http",
        "timeout",
        "decode",
        "network",
        None,
        "http",
        None,
        None,
    ]
    assert all((r.frame is None) == (r.error is not None) for r in results)
    assert all(server.requests(c.id) == 1 for c in cams if c.id != "refused")


def test_empty_sweep() -> None:
    assert fetch.fetch_sweep([]) == []


@pytest.mark.parametrize(("concurrency", "timeout_s"), [(0, 20), (-1, 20), (1, 0), (1, -5)])
def test_invalid_arguments_are_rejected(concurrency: int, timeout_s: float) -> None:
    with pytest.raises(ValueError):
        fetch.fetch_sweep([], concurrency=concurrency, timeout_s=timeout_s)


def test_concurrency_bounds_parallel_requests(server: FakeCameraServer) -> None:
    cams = server.cameras(6)
    for cam in cams:
        server.serve_delay(cam.id, 0.3)
    started = time.monotonic()
    results = fetch.fetch_sweep(cams, concurrency=6, timeout_s=5)
    parallel = time.monotonic() - started
    started = time.monotonic()
    fetch.fetch_sweep(cams[:3], concurrency=1, timeout_s=5)
    serial = time.monotonic() - started
    assert all(r.error is None for r in results)
    assert parallel < 0.3 * 6 * 0.75  # overlapped
    assert serial >= 0.3 * 3  # one at a time


# Logging -------------------------------------------------------------------------------


def test_failures_and_summary_are_logged_without_urls(
    server: FakeCameraServer, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="wearreport")
    ok, missing = server.cameras(2)
    server.serve_404(missing.id)
    fetch.fetch_sweep([ok, missing])
    failed = [r for r in caplog.records if r.getMessage() == "frame failed"]
    assert [(getattr(r, "camera_id"), getattr(r, "error")) for r in failed] == [  # noqa: B009
        (missing.id, "http")
    ]
    (summary,) = [r for r in caplog.records if r.getMessage() == "sweep fetched"]
    assert getattr(summary, "fetched") == 1  # noqa: B009
    assert getattr(summary, "errors_http") == 1  # noqa: B009
    logged = _logged(caplog)
    assert "127.0.0.1" not in logged
    assert "/9j/" not in logged and "\\xff\\xd8" not in logged


def test_log_scan_sees_urls_and_bytes_in_extra(
    server: FakeCameraServer, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canary: a log call that leaks through `extra=` must be caught by _logged()."""
    caplog.set_level(logging.DEBUG, logger="wearreport")
    real_download = fetch._download

    def leaky_download(url: str, timeout_s: float) -> bytes:
        body = real_download(url, timeout_s)
        fetch.logger.debug(
            "frame failed",
            extra={"camera_id": "x", "url": url, "frame_b64": base64.b64encode(body).decode()},
        )
        return body

    monkeypatch.setattr(fetch, "_download", leaky_download)
    fetch.fetch_sweep(server.cameras(1))
    assert "127.0.0.1" not in caplog.text and "/9j/" not in caplog.text  # why text is not enough
    logged = _logged(caplog)
    assert server.base_url in logged
    assert "/9j/" in logged


# Command line --------------------------------------------------------------------------


def test_dry_run_prints_counts(capsys: pytest.CaptureFixture[str]) -> None:
    assert fetch.main(["--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "cameras listed: 50" in out
    assert "frames fetched: 50" in out
    assert "errors timeout: 0" in out
    assert any(line.startswith("seconds: ") for line in out)


def test_live_or_dry_run_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        fetch.main([])
    assert info.value.code == 2
    with pytest.raises(SystemExit):
        fetch.main(["--live", "--dry-run"])
