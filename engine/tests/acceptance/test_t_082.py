"""Acceptance tests for T-082: weather uses bounded, credential-safe transport.

The task contract: do not edit after its first commit. All endpoints and weather
responses are synthetic localhost fixtures. No provider or camera is contacted.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import select
import shutil
import socket
import ssl
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from wearreport import cli, detect, fetch, registry, settings, weather

AT = datetime(2026, 10, 11, 12, tzinfo=UTC)
KEY = "synthetic-weather-header"
BODY_MARKER = "synthetic-response-marker"
LIMIT = 1_000_000


def _forecast() -> bytes:
    return json.dumps(
        {
            "features": [
                {
                    "properties": {
                        "timeSeries": [
                            {
                                "time": "2026-10-11T12:00Z",
                                "screenTemperature": 12.0,
                                "feelsLikeTemperature": 10.0,
                                "totalPrecipAmount": 0.5,
                            }
                        ]
                    }
                }
            ]
        }
    ).encode()


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate a disposable localhost certificate; no saved fixture or credential."""
    root = tmp_path_factory.mktemp("weather-tls")
    cert, key = root / "cert.pem", root / "key.pem"
    openssl = shutil.which("openssl")
    assert openssl is not None
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "2",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.fixture(autouse=True)
def local_transport(
    certificate: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Retain normal proxy/CA handling, with trust and bypass for local fixtures only."""
    monkeypatch.setenv("SSL_CERT_FILE", str(certificate[0]))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    registry._tls_context.cache_clear()
    real_connect = socket.socket.connect

    def connect(sock: socket.socket, address: Any) -> None:
        if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1"}:
            raise AssertionError("a test attempted a non-local connection")
        real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    yield
    registry._tls_context.cache_clear()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def reply(self, status: int, body: bytes, **headers: str) -> None:
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        with contextlib.suppress(OSError):
            self.wfile.write(body)


@contextlib.contextmanager
def _serve(
    respond: Callable[[_Handler], None],
    certificate: tuple[Path, Path] | None,
) -> Iterator[tuple[str, list[dict[str, str]]]]:
    received: list[dict[str, str]] = []

    class Handler(_Handler):
        def do_GET(self) -> None:
            received.append(dict(self.headers.items()))
            respond(self)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.block_on_close = False
    if certificate is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*certificate)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    worker.daemon = True
    worker.start()
    scheme = "https" if certificate is not None else "http"
    try:
        yield f"{scheme}://127.0.0.1:{server.server_port}/weather", received
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


def _conditions(budget: weather.RequestBudget | None = None) -> weather.Conditions | None:
    return weather.current_conditions(
        AT,
        settings=settings.load_settings({"METOFFICE_API_KEY": KEY}),
        _budget=budget if budget is not None else weather.RequestBudget(10),
    )


def test_ac1_https_retains_api_key_and_json_accept(
    certificate: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _forecast()
    with _serve(
        lambda h: h.reply(200, body, **{"Content-Length": str(len(body))}), certificate
    ) as (
        url,
        received,
    ):
        monkeypatch.setattr(weather, "METOFFICE_URL", url)
        result = _conditions()
    assert result is not None and result.temp_c == 12.0
    assert len(received) == 1
    assert received[0]["Apikey"] == KEY
    assert received[0]["Accept"] == "application/json"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("downgrade", [False, True], ids=["https", "http"])
def test_ac1_redirect_never_reaches_another_endpoint(
    status: int,
    downgrade: bool,
    certificate: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _serve(lambda h: h.reply(200, _forecast()), None if downgrade else certificate) as (
        destination,
        destination_headers,
    ):

        def redirect(h: _Handler) -> None:
            h.reply(status, b"", Location=destination, **{"Content-Length": "0"})

        with _serve(redirect, certificate) as (url, received):
            monkeypatch.setattr(weather, "METOFFICE_URL", url)
            budget = weather.RequestBudget(10)
            assert _conditions(budget) is None
            assert budget.used == 1
    assert len(received) == 1 and received[0]["Apikey"] == KEY
    assert destination_headers == []


def test_ac1_plain_http_is_refused_without_contacting_server() -> None:
    with (
        _serve(lambda h: h.reply(200, b"{}"), None) as (url, received),
        pytest.raises(ValueError),
    ):
        weather.urllib_get(url, {"apikey": KEY}, 1)
    assert received == []


@pytest.mark.parametrize("part", ["headers", "body"])
def test_ac2_deadline_bounds_trickled_headers_and_body(
    part: str,
    certificate: tuple[Path, Path],
) -> None:
    def trickle(h: _Handler) -> None:
        with contextlib.suppress(OSError):
            if part == "headers":
                h.wfile.write(b"HTTP/1.0 200 OK\r\nX-Slow: ")
            else:
                h.send_response(200)
                h.end_headers()
            for _ in range(15):
                h.wfile.write(b"x")
                h.wfile.flush()
                time.sleep(0.04)
            if part == "headers":
                h.wfile.write(b"\r\n\r\n{}")

    with _serve(trickle, certificate) as (url, _):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            weather.urllib_get(url, {"apikey": KEY}, 0.15)
        assert time.monotonic() - started < 0.5


def test_ac2_deadline_covers_stalled_tls_handshake() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            weather.urllib_get(f"https://127.0.0.1:{listener.getsockname()[1]}/", {}, 0.15)
        assert time.monotonic() - started < 0.5


def test_ac2_connect_attempts_share_the_elapsed_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    def addresses(*args: Any, **kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 1))] * 5

    def refused(sock: socket.socket, address: object) -> None:
        timeout = sock.gettimeout()
        assert timeout is not None
        time.sleep(min(0.08, timeout))
        raise TimeoutError("synthetic stalled connect")

    monkeypatch.setattr(socket, "getaddrinfo", addresses)
    monkeypatch.setattr(socket.socket, "connect", refused)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        weather.urllib_get("https://127.0.0.1:1/", {}, 0.15)
    assert time.monotonic() - started < 0.3


def test_ac2_https_keeps_proxy_tunnel_and_configured_ca(
    certificate: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tunnels: list[str] = []

    class Proxy(_Handler):
        def do_CONNECT(self) -> None:
            tunnels.append(self.path)
            host, port = self.path.rsplit(":", 1)
            with socket.create_connection((host, int(port)), timeout=1) as upstream:
                self.connection.sendall(b"HTTP/1.0 200 Connection established\r\n\r\n")
                while ready := select.select([self.connection, upstream], [], [], 1)[0]:
                    for source in ready:
                        data = source.recv(64 * 1024)
                        if not data:
                            return
                        target = upstream if source is self.connection else self.connection
                        target.sendall(data)

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    proxy.daemon_threads = True
    worker = threading.Thread(
        target=proxy.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    worker.start()
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{proxy.server_port}")
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{proxy.server_port}")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    try:
        with _serve(lambda h: h.reply(200, b"{}", **{"Content-Length": "2"}), certificate) as (
            url,
            received,
        ):
            assert weather.urllib_get(url, {"apikey": KEY}, 1) == b"{}"
        assert len(tunnels) == 1 and received[0]["Apikey"] == KEY
    finally:
        proxy.shutdown()
        proxy.server_close()
        worker.join(timeout=1)


def test_ac2_dns_interruption_limit_is_documented() -> None:
    assert registry.bounded_get.__doc__ is not None
    assert "DNS" in registry.bounded_get.__doc__
    assert "interrupt" in registry.bounded_get.__doc__


@pytest.mark.parametrize("declared", [False, True])
def test_ac3_exact_cap_body_is_usable(
    declared: bool,
    certificate: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert weather.MAX_BODY_BYTES == LIMIT
    body = _forecast()
    body += b" " * (LIMIT - len(body))
    headers = {"Content-Length": str(len(body))} if declared else {}
    with _serve(lambda h: h.reply(200, body, **headers), certificate) as (url, _):
        assert weather.urllib_get(url, {}, 2) == body
        monkeypatch.setattr(weather, "METOFFICE_URL", url)
        assert _conditions() is not None


@pytest.mark.parametrize("declared", [False, True])
def test_ac3_oversized_response_maps_to_constant_weather_error(
    declared: bool,
    certificate: tuple[Path, Path],
) -> None:
    body = b" " * (LIMIT + 1)
    headers = {"Content-Length": str(len(body))} if declared else {}
    with (
        _serve(lambda h: h.reply(200, body, **headers), certificate) as (url, _),
        pytest.raises(weather.WeatherError) as info,
    ):
        weather.urllib_get(url, {"apikey": KEY}, 2)
    assert str(info.value) == "response body too large"
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_ac3_truncated_response_is_refused(certificate: tuple[Path, Path]) -> None:
    with (
        _serve(lambda h: h.reply(200, b"{}", **{"Content-Length": "100"}), certificate) as (
            url,
            _,
        ),
        pytest.raises(http.client.IncompleteRead),
    ):
        weather.urllib_get(url, {}, 1)


@pytest.mark.parametrize("failure", ["redirect", "oversized", "malformed", "truncated"])
def test_ac3_failure_logs_exclude_key_url_and_body(
    failure: str,
    certificate: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def response(h: _Handler) -> None:
        body = BODY_MARKER.encode()
        if failure == "redirect":
            h.reply(302, body, Location=f"http://127.0.0.1:1/{BODY_MARKER}?key={KEY}")
        else:
            length = str(LIMIT + 1) if failure == "oversized" else "100"
            headers = {} if failure == "malformed" else {"Content-Length": length}
            h.reply(200, body, **headers)

    with _serve(response, certificate) as (url, _):
        monkeypatch.setattr(weather, "METOFFICE_URL", url)
        with caplog.at_level(logging.WARNING, logger="wearreport.weather"):
            assert _conditions() is None
    assert caplog.records
    serialized = "\n".join(cli._JsonFormatter().format(record) for record in caplog.records)
    for forbidden in (KEY, url, BODY_MARKER):
        assert forbidden not in serialized


def test_ac4_timeouts_have_two_attempts_and_process_budget_is_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def timeout(url: str, headers: Mapping[str, str], seconds: float) -> bytes:
        calls.append(seconds)
        raise TimeoutError(f"{url} {headers!r} {BODY_MARKER}")

    assert weather.ATTEMPTS_PER_CALL == 2
    assert weather.PROCESS_REQUEST_CAP == 10
    budget = weather.RequestBudget(weather.PROCESS_REQUEST_CAP)
    monkeypatch.setattr(weather, "_PROCESS_BUDGET", budget)
    cfg = settings.load_settings({"METOFFICE_API_KEY": KEY})
    for expected in (2, 4, 6, 8, 10, 10, 10):
        assert weather.current_conditions(AT, settings=cfg, http_get=timeout) is None
        assert len(calls) == expected and budget.used == expected
    assert calls == [weather.TIMEOUT_S] * 10


def test_ac4_unavailable_weather_still_publishes_an_offline_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("METOFFICE_API_KEY", KEY)
    monkeypatch.delenv("WEARREPORT_DEV_WEATHER", raising=False)
    monkeypatch.setattr(weather, "METOFFICE_URL", "http://127.0.0.1:1/refused")
    monkeypatch.setattr(weather, "_PROCESS_BUDGET", weather.RequestBudget(10))
    monkeypatch.setattr(
        cli, "load_detector", lambda name: (cast(detect.Detector, object()), "0" * 64)
    )
    monkeypatch.setattr(registry, "list_cameras", lambda key: [])
    monkeypatch.setattr(fetch, "fetch_sweep", lambda cameras: [])
    assert cli.main(["sweep", "--data-dir", str(tmp_path)]) == 0
    (record_path,) = (tmp_path / "sweeps").rglob("*.json")
    record = json.loads(record_path.read_bytes())
    assert record["weather"] is None and record["cameras_listed"] == 0
    assert record["persons_total"] == 0 and record["per_camera"] == {}
    assert (tmp_path / "status.json").is_file()
    assert "weather: none" in capsys.readouterr().out
