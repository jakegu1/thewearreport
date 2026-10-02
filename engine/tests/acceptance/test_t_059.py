"""Acceptance tests for T-059: attribute sessions on Austin's HD camera stills
(`spotcheck --attributes --source austin`), crops from full-resolution frames, a record
that says its source, and a summary that keeps sources apart. The task contract: do not
edit.

Every frame here is synthetic (uniform colours with a gradient, encoded in memory), the
camera list is written by the test, both are served by a local HTTP server on 127.0.0.1,
the detector and the reviewers are scripted, and the hosted model is a fake server on
127.0.0.1. Every file is written by the test or by the tool into a temporary directory,
except the committed attribute files the summary must still read. Nothing reaches the
network.
"""

from __future__ import annotations

import ast
import base64
import datetime
import hashlib
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import _cv, detect, fetch
from wearreport._cv import cv2, encode_jpeg
from wearreport.testing.fake_cameras import FakeCameraServer
from wearreport.tools import pilot_heights, spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
REAL_DIR = ROOT / "spotchecks"
README = REAL_DIR / "README.md"
REAL_FILES = ("2026-09-30.json", "2026-09-30-2.json", "2026-10-01.json", "2026-10-01-2.json")
DAY = datetime.date(2026, 10, 1)
BOTH_DAY = datetime.datetime(2026, 10, 1, 16, 0, 12, tzinfo=datetime.UTC)  # 11:00 in Austin
LONDON_DARK = datetime.datetime(2026, 10, 1, 21, 0, 40, tzinfo=datetime.UTC)  # Austin 16:00
AUSTIN_DARK = datetime.datetime(2026, 10, 1, 8, 0, 5, tzinfo=datetime.UTC)  # Austin 03:00
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
KEY_ENV = "DEEPINFRA_API_KEY"
MODEL = "di-qwen3-vl-235b"
INSIDE = (-97.745, 30.270)  # lon, lat: inside the default downtown box
OUTSIDE = (-97.818, 30.233)
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
DARK_AUSTIN = (
    "spotcheck: it is dark in Austin now (sun below -6°); attribute sessions need daylight. "
    "Use --allow-dark to run anyway."
)
DARK_LONDON = (
    "spotcheck: it is dark in London now (sun below -6°); attribute sessions need daylight. "
    "Use --allow-dark to run anyway."
)

Frame = npt.NDArray[np.uint8]
Box = tuple[float, float, float, float]


# A local Austin site: the camera list and the images ---------------------------------


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

    def endpoints(self) -> spotcheck.AustinEndpoints:
        policy = pilot_heights.UrlPolicy("http", self.netloc)
        return spotcheck.AustinEndpoints(self.url("/dataset"), policy, policy)

    def image(self, name: str, body: bytes, **kwargs: Any) -> str:
        self.routes[f"/image/{name}"] = Route(body=body, **kwargs)
        return self.url(f"/image/{name}")

    def dataset(self, records: object) -> None:
        self.routes["/dataset"] = Route(body=json.dumps(records).encode())

    def count(self, path: str) -> int:
        with self.lock:
            return self.requests.count(path)

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


def _record(url: object, lon: float = INSIDE[0], lat: float = INSIDE[1]) -> dict[str, Any]:
    return {
        "camera_id": "1",
        "screenshot_address": url,
        "location": {"type": "Point", "coordinates": [lon, lat]},
    }


class Stub:
    """boxes[key] are the person boxes in a frame whose blue channel encodes `key`;
    shapes lists the shape of every frame that reached the detector."""

    def __init__(self, boxes: dict[int, Sequence[Box]] | None = None) -> None:
        self.boxes = boxes or {}
        self.shapes: list[tuple[int, ...]] = []
        self.lock = threading.Lock()

    def detect(self, frame: Frame) -> list[detect.Detection]:
        with self.lock:
            self.shapes.append(frame.shape)
        key = int(frame[frame.shape[0] // 2, 4, 0]) // 10
        return [detect.Detection("person", 0.9, box) for box in self.boxes.get(key, [])]


# Person boxes in full-resolution pixels, by height.
B20 = (1500.0, 100.0, 1510.0, 120.0)
B90 = (900.0, 500.0, 940.0, 590.0)
B200 = (100.0, 200.0, 160.0, 400.0)
B600 = (1200.0, 300.0, 1400.0, 900.0)


class Answers:
    """An attribute reviewer answering `default` for every crop, or `by_index` by the
    crop's position; records what it saw."""

    def __init__(self, default: str | None = "ynn", by_index: Sequence[str | None] = ()) -> None:
        self.default = default
        self.by_index = list(by_index)
        self.items: list[spotcheck.ReviewItem] = []

    def attributes(
        self, items: Sequence[spotcheck.ReviewItem], deadline: float
    ) -> dict[int, str | None]:
        self.items = list(items)
        return {
            item.number: self.by_index[k] if k < len(self.by_index) else self.default
            for k, item in enumerate(items)
        }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Outside CI, offline but for 127.0.0.1, in a fresh working directory, HOME and
    temporary directory; yields the temporary directory. The London pipelines must not be
    opened."""
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    for name in (*PROXY_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    work, home, tmp = tmp_path / "work", tmp_path / "home", tmp_path / "tmp"
    for d in (work, home, tmp):
        d.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")):
            raise AssertionError(f"non-local connection attempted: {address!r}")
        real_connect(self, address)

    def no_pipeline(*args: Any, **kwargs: Any) -> spotcheck.Pipeline:
        raise AssertionError("a London pipeline must not be opened")

    monkeypatch.setattr(socket.socket, "connect", connect)
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


def _austin(
    site: Site,
    out: Path,
    args: Sequence[str] = (),
    *,
    stub: Stub | None = None,
    reviewer: Any = None,
    clock: datetime.datetime = BOTH_DAY,
    opened: list[str] | None = None,
    **kwargs: Any,
) -> int:
    argv = ["--attributes", "--source", "austin", *WINDOW, *args]
    argv += ["--reviewer", "tester", "--out-dir", str(out)]
    return spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: clock,
        reviewer=Answers() if reviewer is None else reviewer,
        austin=site.endpoints(),
        open_detector=_opener(stub or Stub(), opened),
        **kwargs,
    )


def _austin_file(out: Path, name: str = f"{DAY.isoformat()}-austin.json") -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "attributes" / name).read_text("utf-8"))
    return data


def _names(out: Path) -> list[str]:
    folder = out / "attributes"
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def _two_hd_cameras(site: Site) -> tuple[bytes, bytes]:
    first, second = _jpeg(1920, 1080, 1), _jpeg(1920, 1080, 2)
    site.dataset([_record(site.image("a", first)), _record(site.image("b", second))])
    return first, second


# AC1: the pixel cap -------------------------------------------------------------------


def test_ac1_the_cap_is_one_1080p_frame() -> None:
    assert _cv.MAX_IMAGE_PIXELS == 2_073_600 == 1920 * 1080
    assert _cv.SETTINGS["OPENCV_IO_MAX_IMAGE_PIXELS"] == "2073600"
    assert os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] == "2073600"


def test_ac1_the_header_bound_is_a_literal() -> None:
    assert pilot_heights.MAX_HEADER_PIXELS == 4_000_000
    tree = ast.parse((ROOT / "engine/wearreport/tools/pilot_heights.py").read_text("utf-8"))
    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MAX_HEADER_PIXELS" for t in node.targets)
    ]
    assert len(values) == 1
    assert isinstance(values[0], ast.Constant) and values[0].value == 4_000_000


def test_ac1_the_named_unit_assertions_hold_the_new_values() -> None:
    unit = ROOT / "engine" / "tests" / "unit"
    cv_text = (unit / "test_cv.py").read_text("utf-8")
    assert '== "2073600"' in cv_text and "MAX_IMAGE_PIXELS == 2_073_600" in cv_text
    assert "1000000" not in cv_text
    detect_text = (unit / "test_detect.py").read_text("utf-8")
    assert "MAX_IMAGE_PIXELS == 1920 * 1080" in detect_text
    assert "MAX_IMAGE_PIXELS == 1000 * 1000" not in detect_text
    heights_text = (unit / "test_pilot_heights.py").read_text("utf-8")
    assert "MAX_HEADER_PIXELS == 4_000_000" in heights_text
    assert "MAX_HEADER_PIXELS == 4 * MAX_IMAGE_PIXELS" not in heights_text


def test_ac1_the_sweep_decodes_1080p_at_full_size_and_refuses_one_pixel_more() -> None:
    hd, wider = _jpeg(1920, 1080, 1), _jpeg(1921, 1080, 1)
    with FakeCameraServer() as server:
        cameras = server.cameras(2)
        server.serve_body(cameras[0].id, hd)
        server.serve_body(cameras[1].id, wider)
        ok, refused = fetch.fetch_sweep(cameras)
    assert ok.error is None and ok.frame is not None
    assert ok.frame.shape == (1080, 1920, 3)
    assert refused.frame is None and refused.error == "decode"


def test_ac1_the_detector_takes_a_whole_1080p_frame() -> None:
    assert detect.MAX_IMAGE_PIXELS == 1920 * 1080  # type: ignore[attr-defined]


# AC2: the option ----------------------------------------------------------------------


def test_ac2_the_sources() -> None:
    assert spotcheck.SOURCES == ("london", "austin")
    args = spotcheck.build_parser().parse_args(["--n", "1"])
    assert args.source == "london"


@pytest.mark.parametrize(
    "argv",
    [
        ["--n", "5", "--source", "austin"],
        ["--n", "5", "--source", "austin", "--mode", "frames"],
        ["--n", "5", "--source", "austin", "--record-boxes"],
        ["--n", "5", "--source", "austin", "--judgements", "j.json"],
        ["--n", "5", "--source", "austin", "--dry-run"],
        ["--n", "5", "--source", "paris", "--attributes"],
        ["--n", "5", "--source", "", "--attributes"],
        ["--n", "5", "--bbox", "30.26,-97.755,30.285,-97.735", "--attributes"],
        ["--n", "5", "--source", "london", "--bbox", "30.26,-97.755,30.285,-97.735"],
    ],
)
def test_ac2_austin_outside_an_attribute_session_exits_2_with_usage(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as refused:
        spotcheck.main([*argv, "--out-dir", str(tmp_path / "out")], today=DAY)
    assert refused.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage:")
    assert not (tmp_path / "out").exists()


def test_ac2_the_usage_message_names_the_rule(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        spotcheck.main(["--n", "5", "--source", "austin"], today=DAY)
    err = capsys.readouterr().err
    assert "--source austin" in err and "--attributes" in err


@pytest.mark.parametrize(
    "bbox",
    [
        "30.26,-97.755,30.285",
        "30.285,-97.755,30.260,-97.735",
        "30.260,-97.755,nan,-97.735",
        "a,b,c,d",
    ],
)
def test_ac2_a_bad_bbox_exits_2_as_in_the_pilot(
    env: Path, site: Site, tmp_path: Path, capsys: pytest.CaptureFixture[str], bbox: str
) -> None:
    with pytest.raises(SystemExit) as refused:
        _austin(site, tmp_path / "out", ["--bbox", bbox])
    assert refused.value.code == 2
    assert site.requests == []


# London stays exactly as it was: the record and messages of one fixed fake session,
# produced by the code on main before this task.
LONDON_RECORD = (
    b'{"date": "2026-10-12", "started_at": "2026-10-12T11:22Z", "light": "day", "frames": 2, '
    b'"detector": {"model": "stub", "sha256": '
    b'"0000000000000000000000000000000000000000000000000000000000000000", "conf": 0.35}, '
    b'"min_height_px": 31, "judge": null, "crops_shown": 4, "crops_rejected": 1, '
    b'"crops": [[31, "ynu", null], [46, "nny", null], [60, "uuu", null]]}\n'
)
LONDON_BOXES = [
    [(10.0, 60.0, 30.0, 91.0), (40.0, 60.0, 60.0, 105.0)],
    [(100.0, 50.0, 120.0, 96.0), (130.0, 40.0, 150.0, 100.0), (200.0, 60.0, 220.0, 90.0)],
]


class LondonStub:
    def detect(self, frame: Frame) -> list[detect.Detection]:
        boxes = LONDON_BOXES[int(frame[0, 0, 0]) // 10]
        return [detect.Detection("person", 0.9, b) for b in boxes]


def _london_pipeline() -> spotcheck.Pipeline:
    frames = []
    for i in range(2):
        frame = np.full((288, 352, 3), 10 * i, dtype=np.uint8)
        frame[10:200, :, 1] = np.arange(352, dtype=np.uint8)[None, :]
        frame[0, 0, 0] = 10 * i
        frames.append(frame)
    return spotcheck.Pipeline(frames=lambda: frames, detector=LondonStub(), info=INFO)


class LondonAnswers:
    def attributes(
        self, items: Sequence[spotcheck.ReviewItem], deadline: float
    ) -> dict[int, str | None]:
        answers = ["ynu", None, "nny", "uuu"]
        return {item.number: answers[k % 4] for k, item in enumerate(items)}


@pytest.mark.parametrize("explicit", [False, True])
def test_ac2_london_is_byte_for_byte_as_before(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], explicit: bool
) -> None:
    out = tmp_path / "out"
    argv = ["--attributes", "--n", "5", "--min-persons", "1", "--view", "window"]
    argv += ["--reviewer", "tester", "--out-dir", str(out), "--seed", "7"]
    argv += ["--source", "london"] if explicit else []
    code = spotcheck.main(
        argv,
        pipeline=_london_pipeline(),
        reviewer=LondonAnswers(),
        today=datetime.date(2026, 10, 12),
        clock=lambda: datetime.datetime(2026, 10, 12, 11, 22, 33, tzinfo=datetime.UTC),
    )
    assert code == 0
    assert _names(out) == ["2026-10-12.json"]
    path = out / "attributes" / "2026-10-12.json"
    assert path.read_bytes() == LONDON_RECORD
    captured = capsys.readouterr()
    assert captured.err == (
        "spotcheck: detected 2 of 2\nspotcheck: opening the review: 4 crop(s)\n"
    )
    assert captured.out == (
        "Review window open: 4 crop(s). y: yes   n: no   u: cannot tell   "
        "x: not a person or nothing can be told   Backspace: back   q: stop\n"
        f"Attribute labels written to {path}: 4 crop(s) shown, 1 rejected\n"
    )


def test_ac2_london_still_refuses_in_the_london_dark(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--attributes", *WINDOW, "--reviewer", "tester", "--out-dir", str(tmp_path / "o")]
    code = spotcheck.main(argv, today=DAY, clock=lambda: LONDON_DARK, reviewer=Answers())
    assert code == 1
    assert capsys.readouterr().err == DARK_LONDON + "\n"


def test_ac2_the_dry_run_uses_a_local_fake_austin(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = Stub()
    argv = ["--attributes", "--source", "austin", "--dry-run", *WINDOW, "--min-persons", "0"]
    argv += ["--reviewer", "tester", "--out-dir", str(tmp_path / "out")]
    code = spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: BOTH_DAY,
        reviewer=Answers(),
        open_detector=_opener(stub),
    )
    assert code == 0, capsys.readouterr().err
    assert stub.shapes and set(stub.shapes) == {(1080, 1920, 3)}
    err = capsys.readouterr().err
    skipped = re.search(r"(\d+) not 1920x1080", err)
    assert skipped is not None and int(skipped.group(1)) >= 1
    assert _austin_file(tmp_path / "out")["source"] == "austin"


def test_ac2_the_dry_run_refuses_the_real_spotchecks_directory(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--attributes", "--source", "austin", "--dry-run", *WINDOW, "--out-dir", "spotchecks"]
    opened: list[str] = []
    code = spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: BOTH_DAY,
        reviewer=Answers(),
        open_detector=_opener(Stub(), opened),
    )
    assert code == 1
    assert "--out-dir" in capsys.readouterr().err
    assert opened == []


# AC3: Austin frames -------------------------------------------------------------------


def test_ac3_frames_come_through_the_pilots_list_and_policy(
    env: Path, site: Site, tmp_path: Path
) -> None:
    good = site.image("good", _jpeg(1920, 1080, 1))
    outside = site.image("outside", _jpeg(1920, 1080, 1))
    redirect = site.image("redirect", b"", status=302, headers={"Location": good})
    elsewhere = "http://203.0.113.9/image/x"  # not the pinned host: refused, never requested
    site.dataset(
        [
            _record(good),
            _record(outside, *OUTSIDE),
            _record(redirect),
            _record(elsewhere),
            {"camera_id": "broken"},
        ]
    )
    stub = Stub({1: [B90]})
    assert _austin(site, tmp_path / "out", stub=stub) == 0
    assert site.count("/dataset") == 1
    assert site.count("/image/good") == 1
    assert site.count("/image/outside") == 0
    assert site.count("/image/redirect") == 1  # one attempt, the redirect not followed
    assert len(stub.shapes) == 1


def test_ac3_bbox_selects_as_in_the_pilot(env: Path, site: Site, tmp_path: Path) -> None:
    outside = site.image("outside", _jpeg(1920, 1080, 1))
    site.dataset([_record(outside, *OUTSIDE)])
    bbox = "30.230,-97.820,30.240,-97.810"
    assert _austin(site, tmp_path / "out", ["--bbox", bbox], stub=Stub({1: [B90]})) == 0
    assert site.count("/image/outside") == 1
    assert _austin_file(tmp_path / "out")["crops"] == [[90, "ynn", None]]


def test_ac3_default_bbox_is_the_pilots() -> None:
    args = spotcheck.build_parser().parse_args(["--n", "1", "--source", "austin"])
    assert args.bbox in (None, pilot_heights.DEFAULT_BBOX)


def test_ac3_each_camera_gets_one_attempt(env: Path, site: Site, tmp_path: Path) -> None:
    good = site.image("good", _jpeg(1920, 1080, 1))
    failing = site.image("failing", b"", status=500)
    site.dataset([_record(good), _record(failing)])
    assert _austin(site, tmp_path / "out", stub=Stub({1: [B90]})) == 0
    assert site.count("/image/failing") == 1


def test_ac3_the_pilots_code_is_reused(
    env: Path, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, int] = {}

    def spy(name: str) -> None:
        real = getattr(pilot_heights, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            calls[name] = calls.get(name, 0) + 1
            return real(*args, **kwargs)

        monkeypatch.setattr(pilot_heights, name, wrapped)

    for name in ("fetch_dataset", "select_cameras", "jpeg_size", "decode_frame"):
        spy(name)
    _two_hd_cameras(site)
    assert _austin(site, tmp_path / "out", stub=Stub({1: [B90], 2: [B200]})) == 0
    assert calls["fetch_dataset"] == 1 and calls["select_cameras"] == 1
    assert calls["jpeg_size"] >= 2 and calls["decode_frame"] == 2


def test_ac3_non_1080p_frames_are_skipped_counted_and_never_shown(
    env: Path, site: Site, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hd = _jpeg(1920, 1080, 1)
    bodies = {
        "hd": hd,
        "placeholder": _jpeg(320, 176, 2),
        "small": _jpeg(1280, 720, 2),
        "wide": _jpeg(1921, 1080, 2),
        "tall": _jpeg(1920, 1081, 2),
    }
    site.dataset([_record(site.image(name, body)) for name, body in bodies.items()])
    stub = Stub({1: [B90], 2: [B200]})
    reviewer = Answers()
    assert _austin(site, tmp_path / "out", stub=stub, reviewer=reviewer) == 0
    assert stub.shapes == [(1080, 1920, 3)]
    assert len(reviewer.items) == 1
    record = _austin_file(tmp_path / "out")
    assert record["frames"] == 1 and record["crops"] == [[90, "ynn", None]]
    err = capsys.readouterr().err
    skipped = re.search(r"(\d+) not 1920x1080", err)
    assert skipped is not None and skipped.group(1) == "4"


def test_ac3_a_header_over_the_bound_is_never_decoded(
    env: Path, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wearreport.testing.fake_cameras import jpeg_declaring

    calls: list[object] = []
    real = cv2.imdecode

    def imdecode(*args: Any) -> Any:
        calls.append(args)
        return real(*args)

    site.dataset([_record(site.image("bomb", jpeg_declaring(30000, 30000)))])
    monkeypatch.setattr(cv2, "imdecode", imdecode)
    code = _austin(site, tmp_path / "out", stub=Stub())
    assert code == 1  # no frame to sample
    assert calls == []


def test_ac3_nothing_but_the_record_is_written(env: Path, site: Site, tmp_path: Path) -> None:
    def tree(root: Path) -> set[str]:
        return {str(p.relative_to(root)) for p in root.rglob("*")}

    _two_hd_cameras(site)
    work = Path.cwd()
    before = (tree(env), tree(work), tree(ROOT / "engine"))
    out = tmp_path / "out"
    assert _austin(site, out, stub=Stub({1: [B90], 2: [B200]})) == 0
    assert (tree(env), tree(work), tree(ROOT / "engine")) == before
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*")) == [
        "attributes",
        f"attributes/{DAY.isoformat()}-austin.json",
    ]


def test_ac3_the_files_view_writes_only_its_review_directory(
    env: Path, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_hd_cameras(site)
    seen: list[list[str]] = []

    class FileAnswers(Answers):
        def attributes(
            self, items: Sequence[spotcheck.ReviewItem], deadline: float
        ) -> dict[int, str | None]:
            [workdir] = [p for p in env.iterdir() if p.name.startswith(spotcheck.TEMP_PREFIX)]
            seen.append(sorted(p.name for p in workdir.iterdir()))
            return super().attributes(items, deadline)

    argv = ["--attributes", "--source", "austin", "--n", "5", "--min-persons", "1"]
    argv += ["--view", "files", "--judgements", str(tmp_path / "j.json")]
    argv += ["--reviewer", "tester", "--out-dir", str(tmp_path / "out")]
    code = spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: BOTH_DAY,
        reviewer=FileAnswers(),
        austin=site.endpoints(),
        open_detector=_opener(Stub({1: [B90], 2: [B200]})),
    )
    assert code == 0
    assert seen == [["crop-0001.png", "crop-0002.png", spotcheck.NUMBERING_FILE]]
    assert not any(p.name.startswith(spotcheck.TEMP_PREFIX) for p in env.iterdir())


# AC4: the record ----------------------------------------------------------------------


def test_ac4_an_austin_record_says_its_source(env: Path, site: Site, tmp_path: Path) -> None:
    _two_hd_cameras(site)
    out = tmp_path / "out"
    stub = Stub({1: [B90, B20], 2: [B200]})
    assert _austin(site, out, stub=stub, reviewer=Answers("nyu")) == 0
    assert _names(out) == [f"{DAY.isoformat()}-austin.json"]
    record = _austin_file(out)
    assert set(record) == FIELDS | {"source"}
    assert record["source"] == "austin"
    assert record["started_at"] == "2026-10-01T16:00Z"
    assert record["frames"] == 2
    assert record["min_height_px"] == 31
    assert record["crops"] == [[90, "nyu", None], [200, "nyu", None]]  # full-resolution px


def test_ac4_light_is_austins(env: Path, site: Site, tmp_path: Path) -> None:
    _two_hd_cameras(site)
    assert spotcheck.light_at(LONDON_DARK) == "dark"  # in London
    assert _austin(site, tmp_path / "out", stub=Stub({1: [B90]}), clock=LONDON_DARK) == 0
    assert _austin_file(tmp_path / "out")["light"] == "day"
    assert spotcheck.light_at(LONDON_DARK, spotcheck.AUSTIN) == "day"
    assert spotcheck.AUSTIN == pilot_heights.AUSTIN == (30.2672, -97.7431)


def test_ac4_light_in_the_austin_dark(env: Path, site: Site, tmp_path: Path) -> None:
    _two_hd_cameras(site)
    code = _austin(
        site, tmp_path / "out", ["--allow-dark"], stub=Stub({1: [B90]}), clock=AUSTIN_DARK
    )
    assert code == 0
    assert _austin_file(tmp_path / "out")["light"] == "dark"


def test_ac4_file_names_never_overwrite_or_take_a_london_name(
    env: Path, site: Site, tmp_path: Path
) -> None:
    _two_hd_cameras(site)
    out = tmp_path / "out"
    (out / "attributes").mkdir(parents=True)
    london = out / "attributes" / f"{DAY.isoformat()}.json"
    london.write_bytes(b"london\n")
    stub = Stub({1: [B90]})
    for _ in range(3):
        assert _austin(site, out, stub=stub) == 0
    assert _names(out) == [
        f"{DAY.isoformat()}-austin-2.json",
        f"{DAY.isoformat()}-austin-3.json",
        f"{DAY.isoformat()}-austin.json",
        f"{DAY.isoformat()}.json",
    ]
    assert london.read_bytes() == b"london\n"
    first = (out / "attributes" / f"{DAY.isoformat()}-austin.json").read_bytes()
    assert json.loads(first)["source"] == "austin"


def test_ac4_london_takes_its_own_name_beside_austin(env: Path, site: Site, tmp_path: Path) -> None:
    _two_hd_cameras(site)
    out = tmp_path / "out"
    assert _austin(site, out, stub=Stub({1: [B90]})) == 0
    argv = ["--attributes", *WINDOW, "--reviewer", "tester", "--out-dir", str(out)]
    code = spotcheck.main(
        argv,
        pipeline=_london_pipeline(),
        reviewer=Answers(),
        today=DAY,
        clock=lambda: BOTH_DAY,
    )
    assert code == 0
    assert _names(out) == [f"{DAY.isoformat()}-austin.json", f"{DAY.isoformat()}.json"]
    london = json.loads((out / "attributes" / f"{DAY.isoformat()}.json").read_text("utf-8"))
    assert set(london) == FIELDS


# AC5: daylight ------------------------------------------------------------------------


def test_ac5_the_clocks_are_what_the_tests_say() -> None:
    def sun(moment: datetime.datetime, where: tuple[float, float]) -> float:
        return pilot_heights.solar_elevation(moment, *where)

    assert sun(BOTH_DAY, spotcheck.LONDON) > 0 and sun(BOTH_DAY, pilot_heights.AUSTIN) > 0
    assert sun(LONDON_DARK, spotcheck.LONDON) < -6 < 0 < sun(LONDON_DARK, pilot_heights.AUSTIN)
    assert sun(AUSTIN_DARK, pilot_heights.AUSTIN) < -6 < 0 < sun(AUSTIN_DARK, spotcheck.LONDON)


def test_ac5_austin_refuses_in_the_austin_dark(
    env: Path, site: Site, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _two_hd_cameras(site)
    opened: list[str] = []
    assert _austin(site, tmp_path / "out", clock=AUSTIN_DARK, opened=opened) == 1
    assert capsys.readouterr().err == DARK_AUSTIN + "\n"
    assert site.requests == [] and opened == []
    assert not (tmp_path / "out").exists()


def test_ac5_austin_runs_in_the_london_dark(env: Path, site: Site, tmp_path: Path) -> None:
    _two_hd_cameras(site)
    assert _austin(site, tmp_path / "out", stub=Stub({1: [B90]}), clock=LONDON_DARK) == 0


def test_ac5_the_messages_differ_only_in_the_city() -> None:
    assert spotcheck.DARK_REFUSAL.replace("London", "Austin") == DARK_AUSTIN.removeprefix(
        "spotcheck: "
    )
    assert DARK_LONDON.removeprefix("spotcheck: ") == spotcheck.DARK_REFUSAL


def test_ac5_a_judgements_file_runs_in_the_austin_dark(
    env: Path, site: Site, tmp_path: Path
) -> None:
    _two_hd_cameras(site)
    argv = ["--attributes", "--source", "austin", "--n", "5", "--min-persons", "1"]
    argv += ["--judgements", str(tmp_path / "j.json")]
    argv += ["--reviewer", "tester", "--out-dir", str(tmp_path / "out")]
    code = spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: AUSTIN_DARK,
        reviewer=Answers(),
        austin=site.endpoints(),
        open_detector=_opener(Stub({1: [B90]})),
    )
    assert code == 0


# AC6: crops ---------------------------------------------------------------------------


def test_ac6_crops_are_cut_from_the_full_resolution_frame(
    env: Path, site: Site, tmp_path: Path
) -> None:
    first, second = _two_hd_cameras(site)
    reviewer = Answers()
    assert _austin(site, tmp_path / "out", stub=Stub({1: [B90], 2: [B600]}), reviewer=reviewer) == 0
    frames = {1: _decoded(first), 2: _decoded(second)}
    assert len(reviewer.items) == 2
    for item in reviewer.items:
        key = int(item.image[item.image.shape[0] // 2, 2, 0]) // 10
        box = B90 if key == 1 else B600
        expected = spotcheck.render_crop(
            frames[key], detect.Detection("person", 0.9, box), item.number
        )
        assert np.array_equal(item.image, expected)
    heights = sorted(item.image.shape[0] for item in reviewer.items)
    assert heights == [180, 1080]  # the existing margin, clipped to the frame


@pytest.mark.parametrize(
    ("size", "limit", "expected"),
    [
        ((100, 40), (1000, 1800), (400, 160)),  # small: enlarged by the existing factor
        ((160, 60), (5000, 5000), (480, 180)),  # never beyond the existing factor
        ((400, 200), (1000, 1800), (400, 200)),  # the existing factor is 1
        ((1080, 400), (900, 1800), (900, 333)),  # taller than the screen allows: reduced
        ((300, 1600), (900, 1000), (187, 1000)),  # wider than the screen allows: reduced
        ((60, 30), (100, 100), (60, 30)),  # enlarged by whole factors that fit only
        ((60, 30), (250, 100), (180, 90)),
    ],
)
def test_ac6_display_size(
    size: tuple[int, int], limit: tuple[int, int], expected: tuple[int, int]
) -> None:
    assert spotcheck.display_size(*size, *limit) == expected
    height, width = size
    assert expected[0] <= height * spotcheck.window_scale(height, width)
    assert expected[0] <= limit[0] and expected[1] <= limit[1]


def test_ac6_the_display_image_has_the_display_size() -> None:
    image = np.zeros((1080, 400, 3), dtype=np.uint8)
    shown = spotcheck.display_image(image, 900, 1800)
    assert shown.shape == (900, 333, 3) and shown.dtype == np.uint8
    small = np.zeros((100, 40, 3), dtype=np.uint8)
    assert spotcheck.display_image(small, 1000, 1800).shape == (400, 160, 3)


def test_ac6_the_austin_window_fits_the_screen_and_londons_does_not_change(
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
    _two_hd_cameras(site)
    argv = ["--attributes", "--source", "austin", *WINDOW]
    argv += ["--reviewer", "tester", "--out-dir", str(tmp_path / "a")]
    common: dict[str, Any] = {"today": DAY, "clock": lambda: BOTH_DAY}
    stub = Stub({1: [B90]})
    assert spotcheck.main(argv, austin=site.endpoints(), open_detector=_opener(stub), **common) == 0
    argv = ["--attributes", *WINDOW, "--reviewer", "tester", "--out-dir", str(tmp_path / "l")]
    assert spotcheck.main(argv, pipeline=_london_pipeline(), **common) == 0
    assert len(made) == 2
    assert made[0].get("fit_screen") is True
    assert not made[1].get("fit_screen", False)


@pytest.mark.parametrize(
    ("value", "ok"), [("30", False), ("31", True), ("200", True), ("201", False)]
)
def test_ac6_min_height_keeps_its_range(
    env: Path, site: Site, tmp_path: Path, value: str, ok: bool
) -> None:
    _two_hd_cameras(site)
    if ok:
        stub = Stub({1: [B200]})
        assert _austin(site, tmp_path / "out", ["--min-height", value], stub=stub) == 0
        assert _austin_file(tmp_path / "out")["min_height_px"] == int(value)
    else:
        with pytest.raises(SystemExit) as refused:
            _austin(site, tmp_path / "out", ["--min-height", value])
        assert refused.value.code == 2


def test_ac6_min_height_filters_full_resolution_heights(
    env: Path, site: Site, tmp_path: Path
) -> None:
    _two_hd_cameras(site)
    stub = Stub({1: [B90, B20], 2: [B200]})
    assert _austin(site, tmp_path / "out", ["--min-height", "100"], stub=stub) == 0
    record = _austin_file(tmp_path / "out")
    assert record["min_height_px"] == 100 and record["crops"] == [[200, "ynn", None]]


# AC7: the judge -----------------------------------------------------------------------


def _chat(text: str) -> bytes:
    body = {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 300, "completion_tokens": 12},
    }
    return json.dumps(body).encode()


class FakeModel:
    def __init__(self) -> None:
        self.bodies: list[bytes] = []
        self.lock = threading.Lock()
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                with fake.lock:
                    fake.bodies.append(body)
                reply = _chat("outer=yes legs=no umbrella=unsure")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def images(self) -> list[Frame]:
        images = []
        for body in self.bodies:
            [message] = json.loads(body)["messages"]
            [url] = [p["image_url"]["url"] for p in message["content"] if p["type"] == "image_url"]
            png = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8)
            images.append(np.asarray(cv2.imdecode(png, cv2.IMREAD_COLOR), dtype=np.uint8))
        return images

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def model() -> Iterator[FakeModel]:
    fake = FakeModel()
    try:
        yield fake
    finally:
        fake.close()


def test_ac7_the_judge_gets_the_full_resolution_crops_only(
    env: Path, site: Site, tmp_path: Path, model: FakeModel
) -> None:
    _two_hd_cameras(site)
    reviewer = Answers(by_index=["ynn", None, "nnn"])
    judge = ["--judge", MODEL, "--judge-max-requests", "10"]
    code = _austin(
        site,
        tmp_path / "out",
        judge,
        stub=Stub({1: [B90, B600], 2: [B200]}),
        reviewer=reviewer,
        judge_endpoint=model.url,
        judge_sleep=lambda seconds: None,
    )
    assert code == 0
    kept = [item for k, item in enumerate(reviewer.items) if k != 1]
    sent = model.images()
    assert len(sent) == 2
    for image, item in zip(sent, kept, strict=True):
        assert np.array_equal(image, item.image)  # PNG is lossless
        assert image.shape != (1080, 1920, 3)
    record = _austin_file(tmp_path / "out")
    assert record["judge"] == MODEL
    assert sorted(crop[2] for crop in record["crops"]) == ["ynu", "ynu"]


def test_ac7_the_request_cap_holds(env: Path, site: Site, tmp_path: Path, model: FakeModel) -> None:
    _two_hd_cameras(site)
    judge = ["--judge", MODEL, "--judge-max-requests", "1"]
    code = _austin(
        site,
        tmp_path / "out",
        judge,
        stub=Stub({1: [B90], 2: [B200]}),
        judge_endpoint=model.url,
        judge_sleep=lambda seconds: None,
    )
    assert code == 0
    assert len(model.bodies) == 1
    models = [crop[2] for crop in _austin_file(tmp_path / "out")["crops"]]
    assert sorted(models, key=str) == [None, "ynu"]


def test_ac7_the_dry_run_judge_must_be_local(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--attributes", "--source", "austin", "--dry-run", *WINDOW]
    argv += ["--judge", MODEL, "--judge-max-requests", "5", "--out-dir", str(tmp_path / "o")]
    opened: list[str] = []
    code = spotcheck.main(
        argv,
        today=DAY,
        clock=lambda: BOTH_DAY,
        reviewer=Answers(),
        open_detector=_opener(Stub(), opened),
    )
    assert code == 1
    assert "fake judge" in capsys.readouterr().err
    assert opened == []


# AC8: the summary ---------------------------------------------------------------------

VALID: dict[str, Any] = {
    "date": "2026-10-05",
    "started_at": "2026-10-05T16:00Z",
    "light": "day",
    "frames": 3,
    "detector": {"model": "yolox_m", "sha256": "0" * 64, "conf": 0.35},
    "min_height_px": 31,
    "judge": MODEL,
    "crops_shown": 3,
    "crops_rejected": 1,
    "crops": [[90, "ynn", "ynu"], [200, "uuu", None]],
}


def _raw(**changes: Any) -> bytes:
    return json.dumps({**VALID, **changes}).encode()


def _austin_raw(**changes: Any) -> bytes:
    return _raw(source="austin", **changes)


def test_ac8_an_austin_record_is_accepted() -> None:
    labelling = spotcheck_summary.parse_labelling(_austin_raw())
    assert labelling.source == "austin"
    assert labelling.crops == ((90, "ynn", "ynu"), (200, "uuu", None))
    assert spotcheck_summary.parse_labelling(_raw()).source == "london"


@pytest.mark.parametrize("value", ["london", "Austin", "AUSTIN", "", "paris", None, 1, ["austin"]])
def test_ac8_any_other_source_is_refused(value: Any) -> None:
    with pytest.raises(ValueError):
        spotcheck_summary.parse_labelling(_raw(source=value))


@pytest.mark.parametrize(
    "changes",
    [
        {"light": "dusk"},
        {"min_height_px": 30},
        {"crops_shown": 9},
        {"crops": [[20, "ynn", None], [200, "uuu", None]]},
        {"extra": 1},
    ],
)
def test_ac8_the_source_key_on_an_otherwise_invalid_record_is_refused(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        spotcheck_summary.parse_labelling(_austin_raw(**changes))


def test_ac8_a_record_missing_a_field_is_refused_with_the_source_key() -> None:
    data = json.loads(_austin_raw())
    del data["judge"]
    with pytest.raises(ValueError):
        spotcheck_summary.parse_labelling(json.dumps(data).encode())


def test_ac8_a_bad_source_is_refused_like_other_malformed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "attributes").mkdir()
    (tmp_path / "attributes" / "2026-10-05.json").write_bytes(_raw())
    (tmp_path / "attributes" / "2026-10-05-austin.json").write_bytes(_raw(source="paris"))
    code = spotcheck_summary.main(["--attributes", "--dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("summary: 2026-10-05-austin.json is not an attribute file")


# The output of `spotcheck_summary --attributes` for REAL_FILES, produced by the code on
# main before this task, with the directory written as {DIR}: its SHA-256 and first line.
LONDON_SUMMARY_SHA256 = "68dcdfa4d838c6044b5882c2fd1077398251d0d735d9735cef821ed04960c33c"
LONDON_SUMMARY_FIRST = "4 attribute file(s) in {DIR}/attributes"
LONDON_SUMMARY_LINES = 51


def _real_copy(d: Path) -> None:
    (d / "attributes").mkdir(parents=True)
    for name in REAL_FILES:
        shutil.copyfile(REAL_DIR / "attributes" / name, d / "attributes" / name)


def _summary(d: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> list[str]:
    assert spotcheck_summary.main(["--attributes", "--dir", str(d), *extra]) == 0
    return capsys.readouterr().out.replace(str(d), "{DIR}").splitlines()


def test_ac8_london_output_is_byte_identical_for_the_committed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "real"
    _real_copy(d)
    lines = _summary(d, capsys)
    assert lines[0] == LONDON_SUMMARY_FIRST
    assert len(lines) == LONDON_SUMMARY_LINES
    text = "\n".join(lines) + "\n"
    assert hashlib.sha256(text.encode()).hexdigest() == LONDON_SUMMARY_SHA256


def test_ac8_an_austin_file_adds_a_section_after_londons(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "real"
    _real_copy(d)
    london = _summary(d, capsys)
    (d / "attributes" / "2026-10-05-austin.json").write_bytes(_austin_raw())
    both = _summary(d, capsys)
    assert both[: len(london)] == london  # London first, unchanged
    austin = both[len(london) :]
    assert austin and any("Austin" in line for line in austin[:2])
    assert sum(1 for line in austin if line.startswith("crops: ")) == 1
    assert "crops: 3 shown, 1 rejected, 2 labelled, 1 with a model answer" in austin
    assert sum(1 for line in austin if line.strip().startswith("verdict: ")) == 3


def test_ac8_sources_are_never_pooled(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "attributes").mkdir()
    # London: 40 yes and 40 no and 130 more no for outer_layer, all matched: 210 labelled.
    london = [[60, "ynn", "ynn"]] * 40 + [[60, "nnn", "nnn"]] * 170
    austin = [[90, "ynn", "nnn"]] * 5
    (tmp_path / "attributes" / "2026-10-05.json").write_bytes(
        _raw(crops=london, crops_shown=210, crops_rejected=0)
    )
    (tmp_path / "attributes" / "2026-10-05-austin.json").write_bytes(
        _austin_raw(crops=austin, crops_shown=5, crops_rejected=0)
    )
    lines = _summary(tmp_path, capsys)
    split = next(k for k, line in enumerate(lines) if k and "attribute file(s)" in line)
    london_lines, austin_lines = lines[:split], lines[split:]
    assert london_lines[0] == "1 attribute file(s) in {DIR}/attributes"
    assert "Austin" in austin_lines[0]
    assert london_lines[london_lines.index("outer_layer:") + 1].startswith(
        "  reviewer: yes 40, no 170, cannot tell 0"
    )
    assert austin_lines[austin_lines.index("outer_layer:") + 1].startswith(
        "  reviewer: yes 5, no 0, cannot tell 0"
    )

    def verdicts(block: list[str]) -> list[str]:
        return [line.strip() for line in block if line.strip().startswith("verdict: ")]

    assert verdicts(london_lines)[0] == "verdict: pass"
    assert verdicts(austin_lines)[0].startswith("verdict: insufficient")
    assert austin_lines[austin_lines.index("outer_layer:") + 5] == (
        "  model: 5 paired, model cannot tell 0"
    )


def test_ac8_no_austin_section_without_an_austin_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "attributes").mkdir()
    (tmp_path / "attributes" / "2026-10-05.json").write_bytes(_raw())
    lines = _summary(tmp_path, capsys)
    assert not any("Austin" in line for line in lines)
    assert sum(1 for line in lines if "attribute file(s)" in line) == 1


def test_ac8_with_only_austin_files_london_still_comes_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "attributes").mkdir()
    (tmp_path / "attributes" / "2026-10-05-austin.json").write_bytes(_austin_raw())
    lines = _summary(tmp_path, capsys)
    assert lines[0] == "0 attribute file(s) in {DIR}/attributes"
    assert any("Austin" in line and "attribute file(s)" in line for line in lines[1:])


def test_ac8_the_austin_section_is_not_joined_to_london_rain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "attributes").mkdir()
    (tmp_path / "attributes" / "2026-10-05.json").write_bytes(_raw())
    (tmp_path / "attributes" / "2026-10-05-austin.json").write_bytes(_austin_raw())
    data = tmp_path / "data"
    data.mkdir()
    lines = _summary(tmp_path, capsys, "--data-dir", str(data))
    split = next(k for k, line in enumerate(lines) if k and "attribute file(s)" in line)
    assert any(line.startswith("  rain ") for line in lines[:split])
    assert not any(line.startswith("  rain ") for line in lines[split:])


# AC10: quality ------------------------------------------------------------------------


def test_ac10_privacy_guard_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "privacy_guard.py")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ac10_no_new_dependency() -> None:
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    lines = [line.strip().strip('",') for line in deps.splitlines() if line.strip()]
    names = sorted(line.split("=")[0].split(">")[0] for line in lines)
    assert names == ["llama-cpp-python", "numpy", "onnxruntime", "opencv-python-headless"]


def test_ac10_the_guide_documents_the_source() -> None:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Attribute session (`--attributes`)")
    end = text.index("\n## ", start + 1)
    section = text[start:end]
    assert "--source austin" in section
    assert "1920x1080" in section
