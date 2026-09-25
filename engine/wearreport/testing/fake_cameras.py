"""A local fake JamCam image host for tests and `make sweep-dry`.

Every response is a synthetic JPEG of random noise, encoded in memory when the request
arrives. No image is read from or written to disk. URLs look like
`http://127.0.0.1:<port>/cam/<camera_id>`, with no file extension.

Per camera, the server can be told to answer 404, to wait before answering, to send
bytes that are not a decodable image, or to send a given body (for example one of the
hostile images below).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Literal

import numpy as np

from wearreport._cv import encode_jpeg
from wearreport.registry import Camera

# JamCam stills are 352x288.
FRAME_WIDTH = 352
FRAME_HEIGHT = 288
PATH_PREFIX = "/cam/"

Mode = Literal["ok", "not_found", "delay", "corrupt", "body"]


@dataclass(frozen=True, slots=True)
class _Behaviour:
    mode: Mode
    delay_s: float = 0.0
    body: bytes = b""


def synthetic_jpeg(rng: np.random.Generator) -> bytes:
    """Encode a frame of random noise as JPEG, in memory."""
    pixels = rng.integers(0, 256, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    return encode_jpeg(pixels)


def corrupt_bytes(rng: np.random.Generator) -> bytes:
    """A JPEG start-of-image marker followed by noise: looks like a JPEG, never decodes."""
    return b"\xff\xd8\xff" + rng.bytes(2048)


def radiance_hdr(width: int = 8, height: int = 8) -> bytes:
    """A flat grey Radiance HDR image: a format OpenCV decodes through a temporary file."""
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode()
    return header + bytes([128, 128, 128, 129]) * (width * height)


def jpeg_declaring(width: int, height: int) -> bytes:
    """A small valid JPEG whose frame header is rewritten to declare `width` x `height`.

    The body stays a few hundred bytes; a decoder without a pixel cap would allocate the
    declared size and fill it in.
    """
    body = bytearray(encode_jpeg(np.zeros((16, 16, 3), dtype=np.uint8)))
    sof = body.find(b"\xff\xc0")  # baseline start-of-frame
    if sof < 0:
        raise RuntimeError("no baseline start-of-frame marker")
    body[sof + 5 : sof + 9] = height.to_bytes(2, "big") + width.to_bytes(2, "big")
    return bytes(body)


class FakeCameraServer:
    """Serve synthetic camera frames on 127.0.0.1 from a background thread.

    Use as a context manager; the server stops on exit, without waiting for delayed
    responses.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._behaviours: dict[str, _Behaviour] = {}
        self._requests: dict[str, int] = {}
        self._stopping = threading.Event()
        self._httpd = _Server(("127.0.0.1", 0), _Handler)
        self._httpd.owner = self
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="fake-cameras", daemon=True
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host!s}:{port}"

    def url(self, camera_id: str) -> str:
        return f"{self.base_url}{PATH_PREFIX}{camera_id}"

    def cameras(self, count: int) -> list[Camera]:
        """Return `count` cameras whose image URLs point at this server."""
        return [
            Camera(
                id=f"Fake_{i:05d}",
                name=f"Fake camera {i}",
                lat=51.5,
                lon=-0.1,
                image_url=self.url(f"Fake_{i:05d}"),
            )
            for i in range(1, count + 1)
        ]

    def serve_404(self, camera_id: str) -> None:
        self._set(camera_id, _Behaviour("not_found"))

    def serve_delay(self, camera_id: str, seconds: float) -> None:
        """Wait `seconds` before answering with a valid frame."""
        self._set(camera_id, _Behaviour("delay", seconds))

    def serve_corrupt(self, camera_id: str) -> None:
        self._set(camera_id, _Behaviour("corrupt"))

    def serve_body(self, camera_id: str, body: bytes) -> None:
        """Answer 200 with exactly `body`."""
        self._set(camera_id, _Behaviour("body", body=body))

    def requests(self, camera_id: str) -> int:
        """How many requests the server has received for `camera_id`."""
        with self._lock:
            return self._requests.get(camera_id, 0)

    def close(self) -> None:
        self._stopping.set()  # wakes delayed handlers
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> FakeCameraServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _set(self, camera_id: str, behaviour: _Behaviour) -> None:
        with self._lock:
            self._behaviours[camera_id] = behaviour

    def _record(self, camera_id: str) -> _Behaviour:
        with self._lock:
            self._requests[camera_id] = self._requests.get(camera_id, 0) + 1
            return self._behaviours.get(camera_id, _Behaviour("ok"))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    owner: FakeCameraServer

    def handle_error(self, request: object, client_address: object) -> None:
        # A client that timed out and hung up is expected; stay quiet.
        pass


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def do_GET(self) -> None:
        owner = self.server.owner
        if not self.path.startswith(PATH_PREFIX):
            self._send(404, b"")
            return
        behaviour = owner._record(self.path.removeprefix(PATH_PREFIX))
        rng = np.random.default_rng()
        if behaviour.mode == "not_found":
            self._send(404, b"")
        elif behaviour.mode == "corrupt":
            self._send(200, corrupt_bytes(rng))
        elif behaviour.mode == "body":
            self._send(200, behaviour.body)
        else:
            if behaviour.mode == "delay" and owner._stopping.wait(behaviour.delay_s):
                return  # server is stopping
            self._send(200, synthetic_jpeg(rng))

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "image/jpeg" if status == 200 else "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass  # no request logs; the test output stays readable
