"""Acceptance tests for T-064: spot-check follow-ups. Unknown failure kinds are counted
under `other`; the size check before decoding is tested for both cities; the Windows job
runs the T-061 tests; the summary parsers refuse duplicate keys. The task contract: do
not edit.

Every still here is synthetic (a small encoded image, its frame header rewritten), every
camera list and data file is written by the test, and the stills and lists are served by
a local HTTP server on 127.0.0.1. Nothing reaches the network.
"""

from __future__ import annotations

import datetime
import io
import json
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wearreport import fetch
from wearreport._cv import encode_jpeg
from wearreport.testing.fake_cameras import jpeg_declaring
from wearreport.tools import pilot_heights, spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
SPOTCHECKS = ROOT / "spotchecks"
WINDOWS_WORKFLOW = ROOT / ".github" / "workflows" / "windows.yml"
AUSTIN_INSIDE = [-97.745, 30.270]  # lon, lat: inside Austin's default box
CALGARY_INSIDE = [-114.07, 51.045]  # inside Calgary's default box
UNKNOWN_KIND = "zz-secret-kind"


class Host:
    """Serves `routes` (path -> body) on 127.0.0.1, 404 for anything else."""

    def __init__(self) -> None:
        self.routes: dict[str, bytes] = {}
        host = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = host.routes.get(self.path)
                self.send_response(404 if body is None else 200)
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()
                self.wfile.write(body or b"")

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.netloc = "{}:{}".format(*self.httpd.server_address[:2])

    def _dataset(
        self, paths: list[str], url_field: str, point_field: str, point: list[float]
    ) -> None:
        records = [
            {
                url_field: f"http://{self.netloc}{path}",
                point_field: {"type": "Point", "coordinates": point},
            }
            for path in paths
        ]
        self.routes["/dataset"] = json.dumps(records).encode()

    def austin(self, paths: list[str]) -> spotcheck.AustinEndpoints:
        self._dataset(paths, "screenshot_address", "location", AUSTIN_INSIDE)
        policy = pilot_heights.UrlPolicy("http", self.netloc)
        return spotcheck.AustinEndpoints(f"http://{self.netloc}/dataset", policy, policy)

    def calgary(self, paths: list[str]) -> spotcheck.CalgaryEndpoints:
        self._dataset(paths, "camera_url", "point", CALGARY_INSIDE)
        policy = pilot_heights.UrlPolicy("http", self.netloc)
        return spotcheck.CalgaryEndpoints(f"http://{self.netloc}/dataset", policy, policy)


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


def _austin_pass(endpoints: spotcheck.AustinEndpoints) -> None:
    list(spotcheck.austin_frames(endpoints, pilot_heights.DEFAULT_BBOX, timeout_s=10.0))


def _calgary_pass(endpoints: spotcheck.CalgaryEndpoints) -> None:
    list(spotcheck.calgary_frames(endpoints, pilot_heights.CALGARY_BBOX, timeout_s=10.0))


def _fetched(capsys: pytest.CaptureFixture[str], run: Callable[[], None]) -> tuple[str, str]:
    """The `fetched ...` progress line of one pass, and everything it printed."""
    capsys.readouterr()
    run()
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.startswith("spotcheck: fetched ")]
    assert len(lines) == 1, err
    return lines[0], err


def _scripted_still(outcomes: dict[str, str]) -> Callable[[str, str, float], Any]:
    """A still fetcher that raises FrameFailed(kind) for the URL ending in each path."""

    def still(url: str, scheme: str, end: float) -> Any:
        for path, kind in outcomes.items():
            if url.endswith(path):
                raise pilot_heights.FrameFailed(kind)  # type: ignore[arg-type]
        raise AssertionError(url)

    return still


# AC1: unknown failure kinds -----------------------------------------------------------


def test_ac1_error_kinds_are_the_known_ones() -> None:
    assert fetch.ERROR_KINDS == ("timeout", "http", "decode", "network")
    assert "other" not in fetch.ERROR_KINDS
    assert UNKNOWN_KIND not in fetch.ERROR_KINDS


def test_ac1_austin_unknown_kinds_are_counted_as_other_and_printed_last(
    host: Host, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    still = _scripted_still({"/x1": UNKNOWN_KIND, "/t": "timeout", "/x2": "another-kind"})
    monkeypatch.setattr(spotcheck, "_austin_still", still)
    line, err = _fetched(capsys, lambda: _austin_pass(host.austin(["/x1", "/t", "/x2"])))
    assert line == (
        "spotcheck: fetched 0 1920x1080 frame(s) of 3: 0 not 1920x1080 (skipped), "
        "3 failed (timeout 1, other 2), 0 refused"
    )
    assert UNKNOWN_KIND not in err and "another-kind" not in err


def test_ac1_calgary_unknown_kinds_are_counted_as_other_and_printed_last(
    host: Host, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    still = _scripted_still({"/n": "network", "/x": UNKNOWN_KIND, "/h": "http"})
    monkeypatch.setattr(spotcheck, "_calgary_still", still)
    line, err = _fetched(capsys, lambda: _calgary_pass(host.calgary(["/n", "/x", "/h"])))
    assert line == (
        "spotcheck: fetched 0 840x630 frame(s) of 3: 0 not 840x630 (skipped), "
        "3 failed (http 1, network 1, other 1), 0 refused"
    )
    assert UNKNOWN_KIND not in err


@pytest.mark.parametrize("city", ["austin", "calgary"])
def test_ac1_only_unknown_kinds_never_print_empty_parentheses(
    city: str, host: Host, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    still = _scripted_still({"/a": UNKNOWN_KIND, "/b": UNKNOWN_KIND})
    monkeypatch.setattr(spotcheck, f"_{city}_still", still)
    if city == "austin":
        line, err = _fetched(capsys, lambda: _austin_pass(host.austin(["/a", "/b"])))
        size = "1920x1080"
    else:
        line, err = _fetched(capsys, lambda: _calgary_pass(host.calgary(["/a", "/b"])))
        size = "840x630"
    assert line == (
        f"spotcheck: fetched 0 {size} frame(s) of 2: 0 not {size} (skipped), "
        "2 failed (other 2), 0 refused"
    )
    assert "()" not in err and UNKNOWN_KIND not in err


@pytest.mark.parametrize("city", ["austin", "calgary"])
def test_ac1_per_kind_counts_add_up_to_failed(
    city: str, host: Host, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {"/1": "decode", "/2": UNKNOWN_KIND, "/3": "decode", "/4": "timeout", "/5": "x"}
    monkeypatch.setattr(spotcheck, f"_{city}_still", _scripted_still(outcomes))
    paths = list(outcomes)
    if city == "austin":
        line, _ = _fetched(capsys, lambda: _austin_pass(host.austin(paths)))
    else:
        line, _ = _fetched(capsys, lambda: _calgary_pass(host.calgary(paths)))
    match = re.search(r", (\d+) failed \(([^)]*)\), ", line)
    assert match, line
    parts = [part.rsplit(" ", 1) for part in match.group(2).split(", ")]
    assert [name for name, _ in parts] == ["timeout", "decode", "other"]
    assert sum(int(n) for _, n in parts) == int(match.group(1)) == 5


def test_ac1_known_kinds_and_no_failure_lines_are_unchanged(
    host: Host, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    host.routes["/hd"] = _jpeg(1920, 1080)
    host.routes["/small"] = _jpeg(320, 176)
    line, _ = _fetched(capsys, lambda: _austin_pass(host.austin(["/hd", "/small"])))
    assert line == (
        "spotcheck: fetched 1 1920x1080 frame(s) of 2: 1 not 1920x1080 (skipped), "
        "0 failed, 0 refused"
    )
    host.routes["/cal"] = _jpeg(840, 630)
    line, _ = _fetched(capsys, lambda: _calgary_pass(host.calgary(["/cal"])))
    assert line == (
        "spotcheck: fetched 1 840x630 frame(s) of 1: 0 not 840x630 (skipped), 0 failed, 0 refused"
    )
    monkeypatch.setattr(spotcheck, "_austin_still", _scripted_still({"/h": "http"}))
    line, _ = _fetched(capsys, lambda: _austin_pass(host.austin(["/h"])))
    assert line == (
        "spotcheck: fetched 0 1920x1080 frame(s) of 1: 0 not 1920x1080 (skipped), "
        "1 failed (http 1), 0 refused"
    )


# AC2: the size check before decoding ----------------------------------------------------


class _Decoder:
    """Stands in for pilot_heights.decode_frame: records each call and reports the
    corrupt scan data as a decode failure, as a strict decoder would."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, body: bytes) -> Any:
        self.calls += 1
        raise pilot_heights.FrameFailed("decode")


def _corrupt(width: int, height: int) -> bytes:
    """A JPEG whose frame header declares `width` x `height` over the scan data of a
    16x16 image: the header is well formed and allowed, the scan data does not fit it."""
    body = jpeg_declaring(width, height)
    assert pilot_heights.jpeg_size(body) == (width, height)
    assert width * height <= pilot_heights.MAX_HEADER_PIXELS  # an allowed size
    return body


@pytest.mark.parametrize(
    ("city", "declared", "size"),
    [
        ("austin", pilot_heights.CALGARY_FRAME_SIZE, "1920x1080"),
        ("austin", (1280, 720), "1920x1080"),
        ("calgary", pilot_heights.HD, "840x630"),
        ("calgary", (1280, 720), "840x630"),
    ],
)
def test_ac2_another_declared_size_is_skipped_before_decoding(
    city: str,
    declared: tuple[int, int],
    size: str,
    host: Host,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _Decoder()
    monkeypatch.setattr(pilot_heights, "decode_frame", decoder)
    host.routes["/odd"] = _corrupt(*declared)
    if city == "austin":
        line, _ = _fetched(capsys, lambda: _austin_pass(host.austin(["/odd"])))
    else:
        line, _ = _fetched(capsys, lambda: _calgary_pass(host.calgary(["/odd"])))
    assert line == (
        f"spotcheck: fetched 0 {size} frame(s) of 1: 1 not {size} (skipped), 0 failed, 0 refused"
    )
    assert decoder.calls == 0


@pytest.mark.parametrize("city", ["austin", "calgary"])
def test_ac2_the_city_size_still_reaches_the_decoder(
    city: str, host: Host, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: the same corrupt body declaring the city's own size is decoded, and
    so counted as a decode failure."""
    decoder = _Decoder()
    monkeypatch.setattr(pilot_heights, "decode_frame", decoder)
    if city == "austin":
        host.routes["/own"] = _corrupt(*pilot_heights.HD)
        line, _ = _fetched(capsys, lambda: _austin_pass(host.austin(["/own"])))
        size = "1920x1080"
    else:
        host.routes["/own"] = _corrupt(*pilot_heights.CALGARY_FRAME_SIZE)
        line, _ = _fetched(capsys, lambda: _calgary_pass(host.calgary(["/own"])))
        size = "840x630"
    assert line == (
        f"spotcheck: fetched 0 {size} frame(s) of 1: 0 not {size} (skipped), "
        "1 failed (decode 1), 0 refused"
    )
    assert decoder.calls == 1


# AC3: the Windows job ---------------------------------------------------------------------


def test_ac3_the_windows_test_step_runs_test_t_061() -> None:
    text = WINDOWS_WORKFLOW.read_text(encoding="utf-8")
    step = text.split("- name: Spot-check unit and acceptance tests", 1)[1].split("- name:", 1)[0]
    run = step.split("run: >-", 1)[1].split()
    assert run == [
        "uv",
        "run",
        "--no-sync",
        "pytest",
        "-v",
        "-rs",
        "--capture=sys",
        "engine/tests/unit/test_spotcheck.py",
        "engine/tests/unit/test_spotcheck_window.py",
        "engine/tests/acceptance/test_t_036.py",
        "engine/tests/acceptance/test_t_037.py",
        "engine/tests/acceptance/test_t_061.py",
    ]
    assert text.count("engine/tests/acceptance/test_t_061.py") == 1
    assert re.search(r"^permissions:\s*\n\s+contents: read\s*$", text, re.MULTILINE)
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", u) for u in uses)


# AC4: duplicate keys --------------------------------------------------------------------

_ROW = '{"person": 1, "in_vehicle": 0, "not_person": 0, "unsure": 0}'
RECORD = (
    '{"date": "2026-09-28", "boxes_shown": 10, "boxes_not_person": 2,'
    ' "notes": [{"a": 1}, {"b": 2%s}],'
    ' "judge": {"model": "judge-model"%s, "confusion": {'
    f'"person": {_ROW}, "in_vehicle": {_ROW}, "not_person": {_ROW}'
    "}}%s}"
)
SESSION = (
    '{"date": "2026-09-29", "started_at": "2026-09-28T17:45Z", "light": "day"%s,'
    ' "frames": 3, "detector": {"model": "yolox_m"%s, "runs": [{"a": 1}, {"b": 2%s}]},'
    ' "boxes": [[40, "person"]]}'
)
LABELLING = (
    '{"date": "2026-10-03", "started_at": "2026-10-02T16:21Z", "light": "day"%s,'
    ' "frames": 4, "detector": {"model": "yolox_m"%s, "runs": [{"a": 1}, {"b": 2%s}]},'
    ' "min_height_px": 46, "judge": null, "crops_shown": 1, "crops_rejected": 0,'
    ' "crops": [[50, "ynu", null]]}'
)
SWEEP = (
    '{"started_at": "2026-09-28T17:45:00Z", "cameras": [{"id": "a"}, {"id": "b"%s}],'
    ' "weather": {"precip_mm": 0.0%s}%s}'
)


def _variants(template: str, top: str, nested: str, listed: str) -> dict[str, bytes]:
    """The template with no duplicate, and with one at the top, in a nested object and
    in an object inside a list. Each duplicate comes first with a different value, so a
    reader that keeps the last value would read the file as if nothing were wrong."""
    slots = template.count("%s")
    assert slots == 3

    def fill(top_s: str = "", nested_s: str = "", listed_s: str = "") -> bytes:
        values = {"top": top_s, "nested": nested_s, "listed": listed_s}
        return (template % tuple(values[k] for k in order)).encode()

    order = _ORDER[template]
    return {
        "clean": fill(),
        "top": fill(top_s=top),
        "nested": fill(nested_s=nested),
        "listed": fill(listed_s=listed),
    }


_ORDER = {
    RECORD: ("listed", "nested", "top"),
    SESSION: ("top", "nested", "listed"),
    LABELLING: ("top", "nested", "listed"),
    SWEEP: ("listed", "nested", "top"),
}

PARSERS: dict[str, tuple[Callable[[bytes], object], dict[str, bytes]]] = {
    "parse_record": (
        spotcheck_summary.parse_record,
        _variants(RECORD, ', "boxes_shown": 10', ', "model": "judge-model"', ', "b": 3'),
    ),
    "parse_session": (
        spotcheck_summary.parse_session,
        _variants(SESSION, ', "light": "day"', ', "model": "yolox_m"', ', "b": 3'),
    ),
    "parse_labelling": (
        spotcheck_summary.parse_labelling,
        _variants(LABELLING, ', "light": "day"', ', "model": "yolox_m"', ', "b": 3'),
    ),
    "parse_sweep": (
        spotcheck_summary.parse_sweep,
        _variants(
            SWEEP, ', "started_at": "2026-09-28T17:45:00Z"', ', "precip_mm": 0.0', ', "id": "c"'
        ),
    ),
}


@pytest.mark.parametrize("name", list(PARSERS))
def test_ac4_the_clean_files_and_plain_json_read_them(name: str) -> None:
    """The control: without the duplicate each file loads, and with it plain JSON would
    silently keep one of the values."""
    parse, variants = PARSERS[name]
    parse(variants["clean"])
    for where in ("top", "nested", "listed"):
        json.loads(variants[where])  # valid JSON: only the duplicate is wrong
        assert variants[where] != variants["clean"]


@pytest.mark.parametrize("where", ["top", "nested", "listed"])
@pytest.mark.parametrize("name", list(PARSERS))
def test_ac4_a_duplicate_key_at_any_depth_is_refused(name: str, where: str) -> None:
    parse, variants = PARSERS[name]
    with pytest.raises(ValueError, match="appears twice") as info:
        parse(variants[where])
    # The malformed-file route prints a plain ValueError's message (anything else by its
    # type name), so the reason reaches the user.
    assert type(info.value) is ValueError


def test_ac4_the_loaders_name_the_file_and_the_duplicate(tmp_path: Path) -> None:
    (tmp_path / "2026-09-28.json").write_bytes(PARSERS["parse_record"][1]["listed"])
    with pytest.raises(spotcheck_summary.SummaryError) as info:
        spotcheck_summary.load(tmp_path)
    assert str(info.value) == (
        "2026-09-28.json is not a statistics file (the key 'b' appears twice in one object)"
    )
    boxes = tmp_path / "boxes"
    boxes.mkdir()
    (boxes / "2026-09-29.json").write_bytes(PARSERS["parse_session"][1]["top"])
    with pytest.raises(spotcheck_summary.SummaryError) as info:
        spotcheck_summary.load_sessions(tmp_path)
    assert str(info.value) == (
        "2026-09-29.json is not a per-box file (the key 'light' appears twice in one object)"
    )
    attributes = tmp_path / "attributes"
    attributes.mkdir()
    (attributes / "2026-10-03.json").write_bytes(PARSERS["parse_labelling"][1]["nested"])
    with pytest.raises(
        spotcheck_summary.SummaryError, match=r"^2026-10-03\.json .*'model' appears twice"
    ):
        spotcheck_summary.load_labellings(tmp_path)


def test_ac4_a_sweep_record_with_a_duplicate_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / spotcheck_summary.SWEEPS_DIR / "2026" / "09" / "28"
    folder.mkdir(parents=True)
    (folder / "20260928T1745Z.json").write_bytes(PARSERS["parse_sweep"][1]["nested"])
    moment = datetime.datetime(2026, 9, 28, 17, 45, tzinfo=datetime.UTC)
    with pytest.raises(spotcheck_summary.SummaryError) as info:
        spotcheck_summary.rain_condition(tmp_path, moment)
    assert str(info.value) == (
        "20260928T1745Z.json is not a sweep record (the key 'precip_mm' appears twice in one"
        " object)"
    )


class _PlainJson:
    """json without a duplicate check: how the summary read files before T-064."""

    JSONDecodeError = json.JSONDecodeError

    @staticmethod
    def loads(s: str | bytes, **kwargs: Any) -> Any:
        return json.loads(s)


def _summary(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = spotcheck_summary.main(argv)
    return code, out.getvalue()


@pytest.mark.parametrize("flags", [[], ["--attributes"], ["--heights"]])
def test_ac4_the_committed_files_load_and_print_the_same(
    flags: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = ["--dir", str(SPOTCHECKS), *flags]
    code, now = _summary(argv)
    assert code == 0 and now
    monkeypatch.setattr(spotcheck_summary, "json", _PlainJson)
    code_before, before = _summary(argv)
    assert code_before == 0
    assert now == before


def test_ac4_every_committed_file_loads() -> None:
    assert spotcheck_summary.load(SPOTCHECKS)
    assert spotcheck_summary.load_sessions(SPOTCHECKS)
    assert spotcheck_summary.load_labellings(SPOTCHECKS)
