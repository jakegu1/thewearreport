"""Runtime privacy guard (AGENTS.md INV-1): a sweep leaves no file behind.

The static guard (scripts/privacy_guard.py) reads source code, so it cannot see writes
made through os.open()/os.write(), getattr() dispatch or native code such as
cv2.VideoWriter. This test runs a full dry sweep and looks at what actually happened:

  - the working directory, HOME and TMPDIR (and the XDG directories, inside HOME) point
    at fresh empty directories;
  - cv2.imwrite (and its multi-image variants) raise, and so do Python-level opens for
    writing of a path with an image or video extension;
  - a Python audit hook records every file opened for writing, directory created or file
    renamed anywhere on the filesystem while the sweep runs, every process spawned and
    every connection to a host other than 127.0.0.1;
  - afterwards the three directories must still be empty, and every file under them is
    scanned for image and video bytes (leading magic of JPEG, PNG, WebP, GIF, BMP, TIFF,
    AVI, MP4/MOV `ftyp` and more, plus JPEG, PNG and base64 signatures anywhere in it);
  - pytest's base temporary directory (where a stray `../` path lands) and the repository
    are listed before and after; any file created or changed there fails the test and is
    scanned too. Shared system directories such as /tmp are not watched: other processes
    write there, which would make the test flaky.

The dry-sweep command also runs in a child process, from interpreter start to exit. A
wrapper installs the same kind of audit hook before any engine module is imported, and the
hook reports each event on fd 2 as it happens, so writes from `atexit` callbacks and
interpreter shutdown are reported too. The number of `atexit` callbacks must not change
while the engine is imported and the sweep runs.

Log output is checked through every attribute of every captured record (what a JSON
formatter would emit), not only the rendered message, so `extra=` fields are covered.

The directory checks and the byte scan are what catch native writers, which neither the
audit hook nor the patched openers can see. Out of reach: native code writing to an
absolute path outside the watched places, or writing a file and deleting it before the
sweep ends; reviewers remain responsible for those.

The canary tests prove each layer detects a leak. Set WEARREPORT_PRIVACY_CANARY=1 to make
the main test leak a JPEG through os.open() during the sweep: it must then fail.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wearreport import aggregate, benchmark, cli, detect, fetch, registry
from wearreport._cv import cv2
from wearreport.testing.fake_cameras import FakeCameraServer, synthetic_jpeg

CANARY_ENV = "WEARREPORT_PRIVACY_CANARY"
MEDIA_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".avif",
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".npy",
    ".npz",
)
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
# Signatures that may appear anywhere in a file (e.g. a frame appended to a log).
EMBEDDED = {
    "JPEG": b"\xff\xd8\xff",
    "PNG": b"\x89PNG\r\n\x1a\n",
    "base64 JPEG": b"/9j/",
    "base64 PNG": b"iVBORw0KGgo",
}


def media_kind(data: bytes) -> str | None:
    """Name the image or video format that `data` starts with, or contains, if any."""
    head = data[:16]
    leading = [
        ("JPEG", head.startswith(b"\xff\xd8\xff")),
        ("PNG", head.startswith(b"\x89PNG\r\n\x1a\n")),
        ("GIF", head.startswith((b"GIF87a", b"GIF89a"))),
        ("BMP", head.startswith(b"BM")),
        ("TIFF", head.startswith((b"II*\x00", b"MM\x00*"))),
        ("WebP", head.startswith(b"RIFF") and head[8:12] == b"WEBP"),
        ("AVI", head.startswith(b"RIFF") and head[8:12] == b"AVI "),
        ("MP4/MOV", head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free")),
        ("Matroska/WebM", head.startswith(b"\x1a\x45\xdf\xa3")),
        ("JPEG 2000", head.startswith(b"\x00\x00\x00\x0cjP  ")),
        ("NumPy array", head.startswith(b"\x93NUMPY")),
    ]
    for name, matched in leading:
        if matched:
            return name
    for name, signature in EMBEDDED.items():
        if signature in data:
            return f"embedded {name}"
    return None


@dataclass
class Report:
    """What a guarded run left behind or attempted."""

    created: list[str] = field(default_factory=list)
    elsewhere: list[str] = field(default_factory=list)
    media: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    escapes: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    output: str = ""

    def assert_clean(self) -> None:
        assert self.blocked == [], f"image writes attempted: {self.blocked}"
        assert self.writes == [], f"files written during the sweep: {self.writes}"
        assert self.escapes == [], f"processes or remote connections: {self.escapes}"
        assert self.media == [], f"image or video bytes on disk: {self.media}"
        assert self.created == [], f"files created by the sweep: {self.created}"
        assert self.elsewhere == [], f"files created or changed elsewhere: {self.elsewhere}"
        kind = media_kind(self.output.encode("utf-8", "surrogateescape"))
        assert kind is None, f"{kind} in logs or output"
        assert "\\xff\\xd8\\xff" not in self.output, "escaped JPEG bytes in logs or output"
        assert URL.search(self.output) is None, "a URL in logs or output"


URL = re.compile(r"\b[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def logged(caplog: pytest.LogCaptureFixture) -> str:
    """The rendered log text plus every attribute of every record (`extra=` included)."""
    return caplog.text + "".join(repr(vars(record)) for record in caplog.records)


# Audit hooks cannot be removed, so one hook is installed once and records only while a
# guarded run is active.
_AUDIT_LOCK = threading.Lock()
_audit_sink: tuple[list[str], list[str]] | None = None  # (writes, escapes)
_audit_installed = False
_FILE_EVENTS = frozenset(
    {"os.mkdir", "os.rename", "os.link", "os.symlink", "os.truncate"}
    | {"tempfile.mkstemp", "tempfile.mkdtemp"}
)
_PROCESS_EVENTS = frozenset(
    {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork"}
)
THREAD_GRACE_S = 5
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def _audit(event: str, args: tuple[Any, ...]) -> None:
    sinks = _audit_sink
    if sinks is None:
        return
    writes, escapes = sinks
    if event == "open":
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & WRITE_FLAGS
        )
        if writing and not isinstance(path, int):
            writes.append(f"open {path!s} mode={mode!r} flags={flags!r}")
    elif event in _FILE_EVENTS:
        writes.append(f"{event} {args[0]!s}")
    elif event in _PROCESS_EVENTS:
        escapes.append(f"{event} {args[0]!s}")
    elif event in ("socket.connect", "socket.sendto"):
        address = args[1]
        if not (isinstance(address, tuple) and address[0] in LOOPBACK):
            escapes.append(f"{event} {address!r}")


def _install_audit_hook() -> None:
    global _audit_installed
    with _AUDIT_LOCK:
        if not _audit_installed:
            sys.addaudithook(_audit)
            _audit_installed = True


def _is_media_path(path: object) -> bool:
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    return isinstance(path, str | os.PathLike) and str(path).lower().endswith(MEDIA_SUFFIXES)


REPO_ROOT = Path(__file__).resolve().parents[3]
SKIP_DIRS = frozenset(
    {".git", ".venv", ".tools", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__"}
)
Snapshot = dict[str, tuple[int, int]]


def _watched(tmp_path: Path) -> list[Path]:
    """Roots outside the three sweep directories, all private to this test run."""
    return [tmp_path.parent, REPO_ROOT]  # tmp_path.parent: pytest's base temporary dir


def _snapshot(watched: list[Path], exclude: list[Path]) -> Snapshot:
    """Map every file under the watched roots to its (mtime_ns, size)."""
    skip = {str(p) for p in exclude}
    found: Snapshot = {}
    for root in watched:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [
                d for d in dirnames if d not in SKIP_DIRS and os.path.join(dirpath, d) not in skip
            ]
            for name in dirnames + filenames:
                path = os.path.join(dirpath, name)
                with contextlib.suppress(OSError):
                    st = os.lstat(path)
                    found[path] = (st.st_mtime_ns, st.st_size)
    return found


def _diff(before: Snapshot, after: Snapshot, report: Report) -> None:
    for path, stat in sorted(after.items()):
        if before.get(path) == stat:
            continue
        report.elsewhere.append(path)
        _scan_file(Path(path), report)


def _scan_file(path: Path, report: Report) -> None:
    if path.is_symlink() or not path.is_file():
        return
    with contextlib.suppress(OSError), open(path, "rb") as fh:
        kind = media_kind(fh.read(64 * 1024 * 1024))
        if kind:
            report.media.append(f"{path} ({kind})")


def _scan(dirs: list[Path], report: Report) -> None:
    for root in dirs:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames + filenames:
                report.created.append(str(Path(dirpath, name)))
            for name in filenames:
                _scan_file(Path(dirpath, name), report)


@contextlib.contextmanager
def guarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> Iterator[Report]:
    """Run the body under the privacy conditions; fill the report when it exits."""
    work, home, tmp = tmp_path / "work", tmp_path / "home", tmp_path / "tmp"
    for d in (work, home, tmp):
        d.mkdir()
    watched = _watched(tmp_path)
    before = _snapshot(watched, exclude=[work, home, tmp])
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    for var, sub in (
        ("XDG_CACHE_HOME", ".cache"),
        ("XDG_CONFIG_HOME", ".config"),
        ("XDG_DATA_HOME", ".local/share"),
        ("XDG_STATE_HOME", ".local/state"),
        ("XDG_RUNTIME_DIR", ".run"),
    ):
        monkeypatch.setenv(var, str(home / sub))
    monkeypatch.setattr(tempfile, "tempdir", None)  # re-read TMPDIR
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # no .pyc from lazy imports

    report = Report()

    def refuse(name: str) -> Callable[..., Any]:
        def writer(*args: object, **kwargs: object) -> Any:
            report.blocked.append(name)
            raise PermissionError(f"{name} is not allowed during a sweep (INV-1)")

        return writer

    for name in ("imwrite", "imwritemulti", "imwriteanimation"):
        if hasattr(cv2, name):
            monkeypatch.setattr(cv2, name, refuse(f"cv2.{name}"))

    real_open = builtins.open
    real_os_open = os.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(c in mode for c in "wax+") and _is_media_path(file):
            report.blocked.append(f"open({file!s}, {mode!r})")
            raise PermissionError("image write blocked during a sweep (INV-1)")
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if flags & WRITE_FLAGS and _is_media_path(path):
            report.blocked.append(f"os.open({path!s})")
            raise PermissionError("image write blocked during a sweep (INV-1)")
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)

    caplog.set_level("DEBUG")
    _install_audit_hook()
    global _audit_sink
    _audit_sink = (report.writes, report.escapes)
    threads_before = set(threading.enumerate())
    try:
        yield report
    finally:
        # A thread started by the sweep could write after the scan: let each finish first.
        deadline = time.monotonic() + THREAD_GRACE_S
        for thread in set(threading.enumerate()) - threads_before:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                report.escapes.append(f"thread still running after the sweep: {thread.name}")
        _audit_sink = None
        monkeypatch.undo()  # restore openers before scanning
        _scan([work, home, tmp], report)
        _diff(before, _snapshot(watched, exclude=[work, home, tmp]), report)
        captured = capsys.readouterr()
        report.output = logged(caplog) + captured.out + captured.err


def _sweep(*, full: bool = False) -> None:
    """A sweep that exercises every failure path, after a full dry sweep if `full`."""
    if full:
        assert fetch.main(["--dry-run"]) == 0
    with FakeCameraServer() as server:
        cams = server.cameras(6)
        server.serve_404(cams[1].id)
        server.serve_corrupt(cams[2].id)
        server.serve_delay(cams[3].id, 5)
        results = fetch.fetch_sweep(cams, concurrency=6, timeout_s=0.5)
    assert [r.error for r in results] == [None, "http", "decode", "timeout", None, None]
    assert all(r.frame is not None for r in results if r.error is None)


# Leaks the canaries inject from inside the sweep --------------------------------------


def _leak_os_open() -> None:
    fd = os.open("frame", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.write(fd, synthetic_jpeg(np.random.default_rng()))
    finally:
        os.close(fd)


def _leak_getattr_dispatch() -> None:
    fd, _ = tempfile.mkstemp()  # no suffix: invisible to the static guard
    write = getattr(os, "wri" + "te")
    write(fd, synthetic_jpeg(np.random.default_rng()))
    os.close(fd)


def _leak_video_writer() -> None:
    # Native code: its writes never pass through Python's open().
    path = os.path.join(os.path.expanduser("~"), "clip.avi")
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, cv2.CAP_OPENCV_MJPEG, fourcc, 5, (64, 48))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def _leak_native_escape() -> None:
    # Native, and outside the three directories: only the before/after listing sees it.
    path = os.path.join("..", "..", "escape.avi")
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, cv2.CAP_OPENCV_MJPEG, fourcc, 5, (64, 48))
    assert writer.isOpened()
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def _leak_raw_pixels() -> None:
    # No magic bytes at all: caught because the file exists, not by its content.
    np.zeros((48, 64, 3), dtype=np.uint8).tofile("pixels")


def _leak_subprocess() -> None:
    import subprocess

    subprocess.run([sys.executable, "-c", "pass"], check=True)


def _leak_network() -> None:
    with socket.socket() as sock:
        sock.settimeout(0.05)
        sock.connect(("192.0.2.1", 443))  # TEST-NET-1: never routable


def _leak_deferred() -> None:
    # Fires after the sweep has returned, but before the scan: the join waits for it.
    threading.Timer(0.5, _leak_os_open).start()


def _leak_lingering_thread() -> None:
    # Outlives the grace period by a wide margin, so it is still alive when checked.
    wait = threading.Event().wait
    threading.Thread(target=wait, args=(THREAD_GRACE_S * 4,), name="lingering", daemon=True).start()


def _leak_imwrite() -> None:
    cv2.imwrite("frame.jpg", np.zeros((8, 8, 3), dtype=np.uint8))


def _leak_open_image_path() -> None:
    with open("frame.png", "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")


def _leak_log() -> None:
    import base64
    import logging

    frame = base64.b64encode(synthetic_jpeg(np.random.default_rng())).decode()
    logging.getLogger("wearreport.fetch").debug("frame %s", frame)


def _leak_log_extra() -> None:
    # The message is clean; the URL and the frame ride along in `extra=`.
    import base64
    import logging

    frame = base64.b64encode(synthetic_jpeg(np.random.default_rng())).decode()
    logging.getLogger("wearreport.fetch").debug(
        "frame failed",
        extra={"camera_id": "x", "url": "http://127.0.0.1:1/cam/x", "frame_b64": frame},
    )


def _leak_log_raw_bytes_extra() -> None:
    import logging

    frame = synthetic_jpeg(np.random.default_rng())
    logging.getLogger("wearreport.fetch").debug("frame failed", extra={"frame": frame[:64]})


@contextlib.contextmanager
def _canary(leak: Callable[[], None]) -> Iterator[None]:
    """Run `leak` once, from a fetch worker thread, during the next decode."""
    real_imdecode = cv2.imdecode
    done = threading.Event()

    def imdecode(*args: Any, **kwargs: Any) -> Any:
        if not done.is_set():
            done.set()
            with contextlib.suppress(OSError):
                leak()
        return real_imdecode(*args, **kwargs)

    cv2.imdecode = imdecode
    try:
        yield
    finally:
        cv2.imdecode = real_imdecode
    assert done.is_set(), "the canary never ran"


# Tests ---------------------------------------------------------------------------------


def test_sweep_creates_no_files_and_no_image_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = _canary(_leak_os_open) if os.environ.get(CANARY_ENV) == "1" else None
    with (
        guarded(tmp_path, monkeypatch, capsys, caplog) as report,
        canary or contextlib.nullcontext(),
    ):
        _sweep(full=True)
    report.assert_clean()


# The child's audit hook. It is installed before any engine module is imported and stays
# installed until the interpreter exits; each event is written straight to fd 2, so events
# from atexit callbacks and shutdown are reported as well. It never opens a file.
# It also reports an escape if onnxruntime is first imported without
# ORT_DISABLE_TELEMETRY=1.
CHILD = r"""
import os, sys

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
FILE_EVENTS = {"os.mkdir", "os.rename", "os.replace", "os.link", "os.symlink",
               "os.truncate", "tempfile.mkstemp", "tempfile.mkdtemp"}
PROCESS_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                  "os.spawn", "os.fork", "os.forkpty"}

def report(kind, text):
    os.write(2, ("\n@@privacy " + kind + " " + text.replace("\n", " ") + "\n").encode(
        "utf-8", "backslashreplace"))

ort_imported = []

def audit(event, args):
    if event == "import":
        # onnxruntime's first import: its telemetry is off only if ORT_DISABLE_TELEMETRY
        # is exactly "1" at this moment. Checked here, on every host, rather than through
        # the files onnxruntime writes on some hosts only. Never imports onnxruntime.
        name = args[0]
        if not ort_imported and (name == "onnxruntime" or name.startswith("onnxruntime.")):
            value = os.environ.get("ORT_DISABLE_TELEMETRY")
            ort_imported.append(value)
            if value != "1":
                report("escape", f"onnxruntime imported with ORT_DISABLE_TELEMETRY={value!r}")
    elif event == "open":
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & WRITE_FLAGS)
        if writing and not isinstance(path, int):
            report("write", f"open {path!s} mode={mode!r} flags={flags!r}")
    elif event in FILE_EVENTS:
        report("write", f"{event} {args[0]!s}")
    elif event in PROCESS_EVENTS:
        report("escape", f"{event} {args[0]!s}")
    elif event in ("socket.connect", "socket.sendto"):
        address = args[1]
        if not (isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")):
            report("escape", f"{event} {address!r}")

sys.addaudithook(audit)

import atexit, logging, runpy  # logging registers its own atexit callback on import

callbacks = atexit._ncallbacks()
if sys.argv[1]:
    exec(compile(sys.argv[1], "<canary>", "exec"), {})
sys.argv = ["wearreport.fetch", "--dry-run"]
try:
    runpy.run_module("wearreport.fetch", run_name="__main__", alter_sys=True)
finally:
    after = atexit._ncallbacks()
    if after != callbacks:
        report("atexit", f"{after - callbacks} callback(s) registered by the engine")
"""
MARKER = re.compile(r"^@@privacy (\w+) (.*)$", re.MULTILINE)

# The reviewer's leak: keep each frame and dump them all at exit, in text mode, to a place
# none of the watched directories cover.
ATEXIT_CANARY = r"""
import atexit
from wearreport._cv import cv2

_recent = []
_real = cv2.imdecode

def _keep(buf, flags):
    _recent.append(bytes(buf))
    return _real(buf, flags)

def _dump_recent():
    if _recent:
        with open(PATH, "a", encoding="latin-1") as fh:
            fh.write(b"".join(_recent).decode("latin-1"))

cv2.imdecode = _keep
atexit.register(_dump_recent)
"""


def _run_dry_sweep_process(tmp_path: Path, prelude: str = "") -> Report:
    """Run the real dry-sweep command under the child audit hook; report what happened.

    The child does not inherit ORT_DISABLE_TELEMETRY (this process has it, because
    wearreport.detect sets it): the engine must switch onnxruntime's telemetry off by
    itself, before the library loads, or the files it writes show up in the scan."""
    import subprocess

    work, home, tmp = tmp_path / "work", tmp_path / "home", tmp_path / "tmp"
    for d in (work, home, tmp):
        d.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("XDG_") and k != detect.TELEMETRY_ENV
    }
    env.update(HOME=str(home), TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, prelude],
        cwd=work,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"frames fetched: 50" in proc.stdout
    stderr = proc.stderr.decode(errors="replace")
    report = Report(output=proc.stdout.decode(errors="replace") + MARKER.sub("", stderr))
    for kind, text in MARKER.findall(stderr):
        (report.escapes if kind == "escape" else report.writes).append(f"{kind}: {text}")
    _scan([work, home, tmp], report)
    return report


def test_dry_sweep_process_leaves_no_files(tmp_path: Path) -> None:
    """The real command, from interpreter start to exit: covers import-time and exit-time
    writes, and atexit callbacks registered by the engine."""
    _run_dry_sweep_process(tmp_path).assert_clean()


def test_dry_sweep_process_catches_an_atexit_writer(tmp_path: Path) -> None:
    """Canary: frames dumped at exit to an unwatched directory (/dev/shm on Linux)."""
    shm = Path("/dev/shm")  # noqa: S108  (the point: a shared place nobody watches)
    outside = (shm if shm.is_dir() and os.access(shm, os.W_OK) else tmp_path) / (
        f"wearreport-canary-{os.getpid()}"
    )
    prelude = f"PATH = {str(outside)!r}\n" + ATEXIT_CANARY
    try:
        report = _run_dry_sweep_process(tmp_path, prelude)
        assert outside.exists()  # the leak really happened
    finally:
        outside.unlink(missing_ok=True)
    assert any(w.startswith("atexit: ") for w in report.writes), report.writes
    assert any(str(outside) in w and "mode='a'" in w for w in report.writes), report.writes
    assert report.created == []  # invisible to the directory scan
    with pytest.raises(AssertionError):
        report.assert_clean()


TELEMETRY_ESCAPE = "escape: onnxruntime imported with ORT_DISABLE_TELEMETRY=None"


def test_child_hook_reports_onnxruntime_imported_without_the_telemetry_variable(
    tmp_path: Path,
) -> None:
    """Control: the child does not inherit the variable, so importing onnxruntime before
    the engine does must be reported, on any host (whether or not files are written)."""
    report = _run_dry_sweep_process(tmp_path, "import onnxruntime\n")
    assert TELEMETRY_ESCAPE in report.escapes, report.escapes
    with pytest.raises(AssertionError):
        report.assert_clean()


def test_child_hook_accepts_onnxruntime_imported_by_the_engine(tmp_path: Path) -> None:
    """The engine sets the variable before it imports onnxruntime: nothing is reported."""
    report = _run_dry_sweep_process(tmp_path, "from wearreport import detect\n")
    assert not any("onnxruntime" in e for e in report.escapes), report.escapes
    report.assert_clean()


def test_child_hook_reports_writes_after_the_sweep(tmp_path: Path) -> None:
    """Canary: an exit-time os.open() with write flags, from a plain atexit callback."""
    target = tmp_path / "late"
    prelude = (
        "import atexit, os\n"
        f"atexit.register(lambda: os.close(os.open({str(target)!r}, os.O_WRONLY | os.O_CREAT)))\n"
    )
    report = _run_dry_sweep_process(tmp_path, prelude)
    assert target.exists()
    assert any(str(target) in w for w in report.writes), report.writes
    with pytest.raises(AssertionError):
        report.assert_clean()


@pytest.mark.parametrize(
    ("leak", "caught_by"),
    [
        (_leak_os_open, {"created", "media", "writes"}),
        (_leak_getattr_dispatch, {"created", "media", "writes"}),
        (_leak_video_writer, {"created", "media"}),  # native: only the scan sees it
        (_leak_native_escape, {"elsewhere", "media"}),
        (_leak_raw_pixels, {"created", "writes"}),
        (_leak_subprocess, {"escapes"}),
        (_leak_deferred, {"created", "media", "writes"}),
        (_leak_lingering_thread, {"escapes"}),
        (_leak_network, {"escapes"}),
        (_leak_imwrite, {"blocked"}),
        (_leak_open_image_path, {"blocked"}),
        (_leak_log, {"output"}),
        (_leak_log_extra, {"output"}),
        (_leak_log_raw_bytes_extra, {"output"}),
    ],
    ids=lambda v: getattr(v, "__name__", "")[len("_leak_") :],
)
def test_canary_leak_is_detected(
    leak: Callable[[], None],
    caught_by: set[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with guarded(tmp_path, monkeypatch, capsys, caplog) as report, _canary(leak):
        _sweep()
    for layer in caught_by:
        assert getattr(report, layer), f"{layer} missed the {leak.__name__} canary"
    with pytest.raises(AssertionError):
        report.assert_clean()


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (b"\xff\xd8\xff\xe0" + bytes(20), "JPEG"),
        (b"\x89PNG\r\n\x1a\n" + bytes(20), "PNG"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "WebP"),
        (b"GIF89a" + bytes(10), "GIF"),
        (b"BM" + bytes(20), "BMP"),
        (b"II*\x00" + bytes(20), "TIFF"),
        (b"MM\x00*" + bytes(20), "TIFF"),
        (b"RIFF\x00\x00\x00\x00AVI LIST", "AVI"),
        (b"\x00\x00\x00\x18ftypisom" + bytes(8), "MP4/MOV"),
        (b"\x00\x00\x00\x14ftypqt  " + bytes(8), "MP4/MOV"),
        (b"\x1a\x45\xdf\xa3" + bytes(20), "Matroska/WebM"),
        (b"\x93NUMPY\x01\x00" + bytes(20), "NumPy array"),
        (b'{"log": "x"}\n' + b"\xff\xd8\xff\xdb" + bytes(8), "embedded JPEG"),
        (b"frame=/9j/4AAQSkZJRg==", "embedded base64 JPEG"),
    ],
)
def test_media_kind_recognises_formats(data: bytes, kind: str) -> None:
    assert media_kind(data) == kind


@pytest.mark.parametrize(
    "data", [b"", b'{"sweep": 1, "persons": 3}\n', b"cameras listed: 50\n", bytes(64)]
)
def test_media_kind_ignores_text_and_zeros(data: bytes) -> None:
    assert media_kind(data) is None


# The detector benchmark (T-004) ----------------------------------------------------------
#
# The benchmark runs the model on fetched frames, so it gets the same treatment as a sweep:
# in process under `guarded`, and as a child process from start to exit. It uses the real
# models when they are downloaded; without them, and with WEARREPORT_REQUIRE_MODEL unset, a
# blank stand-in model keeps the fetch, pre-processing and reporting path covered.

REQUIRE_MODEL_ENV = "WEARREPORT_REQUIRE_MODEL"
PEOPLE_FIXTURE = REPO_ROOT / "fixtures" / "detect" / "people_street.jpg"


class _BlankModel:
    """A stand-in for YOLOX that finds nothing."""

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return [np.zeros(detect.OUTPUT_SHAPE, dtype=np.float32)]


def _benchmark_models(tmp_path: Path) -> Path | None:
    """A directory holding yolox_s and yolox_m (a link to yolox_s when -m is not
    downloaded), or None when the model is missing and not required."""
    s, m = detect.model_path("yolox_s.onnx"), detect.model_path("yolox_m.onnx")
    if not s.is_file():
        if os.environ.get(REQUIRE_MODEL_ENV):
            pytest.fail(f"yolox_s.onnx is missing and {REQUIRE_MODEL_ENV} is set")
        return None
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolox_s.onnx").symlink_to(s)
    (models / "yolox_m.onnx").symlink_to(m if m.is_file() else s)
    return models


def _benchmark_run(models: Path) -> None:
    """The synthetic benchmark, then the model comparison on fake-server frames that
    exercise every fetch outcome, some of them showing people."""
    assert benchmark.main(["--synthetic", "3", "--model-dir", str(models)]) == 0
    with FakeCameraServer() as server:
        cams = server.cameras(6)
        server.serve_body(cams[0].id, PEOPLE_FIXTURE.read_bytes())
        server.serve_body(cams[1].id, PEOPLE_FIXTURE.read_bytes())
        server.serve_404(cams[2].id)
        server.serve_corrupt(cams[3].id)
        assert benchmark._comparison(models, cams) == 0
    assert benchmark.main(["--dry-run", "--cameras", "4", "--model-dir", str(models)]) == 0


def test_benchmark_creates_no_files_and_no_image_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    models = _benchmark_models(tmp_path)
    canary = _canary(_leak_os_open) if os.environ.get(CANARY_ENV) == "1" else None
    with (
        guarded(tmp_path, monkeypatch, capsys, caplog) as report,
        canary or contextlib.nullcontext(),
    ):
        if models is None:
            monkeypatch.setattr(detect, "open_session", lambda path: _BlankModel())
            models = tmp_path / "absent"
        _benchmark_run(models)  # sessions open inside the guard: loading is covered too
    assert "| yolox_s |" in report.output and "| yolox_m |" in report.output
    report.assert_clean()


BENCHMARK_PRELUDE = r"""
import numpy as np
from wearreport import benchmark, detect
from wearreport.testing.fake_cameras import FakeCameraServer

if MODELS is None:
    class _BlankModel:
        def run(self, tensor):
            return [np.zeros(detect.OUTPUT_SHAPE, dtype=np.float32)]
    detect.open_session = lambda path: _BlankModel()
    MODELS = "absent"

assert benchmark.main(["--synthetic", "3", "--model-dir", MODELS]) == 0
with FakeCameraServer() as server:
    cams = server.cameras(4)
    with open(PEOPLE, "rb") as fh:
        server.serve_body(cams[0].id, fh.read())
    server.serve_404(cams[1].id)
    assert benchmark._comparison(detect.Path(MODELS), cams) == 0
"""


def test_benchmark_process_leaves_no_files(tmp_path: Path) -> None:
    """The benchmark in a child process from interpreter start to exit, onnxruntime's
    import, session creation and shutdown included."""
    models = _benchmark_models(tmp_path)
    prelude = (
        f"MODELS = {None if models is None else str(models)!r}\n"
        f"PEOPLE = {str(PEOPLE_FIXTURE)!r}\n" + BENCHMARK_PRELUDE
    )
    report = _run_dry_sweep_process(tmp_path, prelude)
    assert "median ms per frame: " in report.output
    assert "| yolox_s |" in report.output and "| yolox_m |" in report.output
    report.assert_clean()


# The sweep pipeline (T-005) --------------------------------------------------------------
#
# `wearreport sweep --dry-run` runs the whole pipeline (registry, fetch, detector, weather,
# record, publish) against the fake camera server, with the real detector model, in process
# under `guarded` and as a child process from start to exit. Unlike a fetch-only sweep it
# must create files: exactly one record and status.json, in the new directory it prints.
# Everything else is as strict as above, and every file is still scanned for image bytes.
# These tests need the model, which `make setup` fetches and CI provides: they fail, never
# skip, when it is missing.

PIPELINE_MODEL = "yolox_s.onnx"
PIPELINE_CAMERAS = 6
RECORD_NAME = re.compile(r"sweeps/\d{4}/\d{2}/\d{2}/\d{8}T\d{4}Z\.json")
WEATHER_ENV = ("METOFFICE_API_KEY", "WEARREPORT_DEV_WEATHER", "WEARREPORT_ENV")


def _require_pipeline_model() -> None:
    if not detect.model_path(PIPELINE_MODEL).is_file():
        pytest.fail(f"{PIPELINE_MODEL} is missing; run make setup")


def _pipeline_cameras(server: FakeCameraServer) -> list[Any]:
    """Two cameras showing people, one 404, one corrupt frame, two frames of noise."""
    cams = server.cameras(PIPELINE_CAMERAS)
    server.serve_body(cams[0].id, PEOPLE_FIXTURE.read_bytes())
    server.serve_body(cams[1].id, PEOPLE_FIXTURE.read_bytes())
    server.serve_404(cams[2].id)
    server.serve_corrupt(cams[3].id)
    return cams


def _accept_published(report: Report, tmp: Path) -> dict[str, Any]:
    """Check that the dry run left exactly its record and status.json in one new directory
    under `tmp`, take them (and only them) off the report, and return the record."""
    match = re.search(r"^directory: (.+)$", report.output, re.MULTILINE)
    assert match, report.output
    directory = Path(match.group(1).strip())
    assert directory.parent == tmp, directory
    entries = sorted(directory.rglob("*"))
    files = [p.relative_to(directory).as_posix() for p in entries if p.is_file()]
    assert len(files) == 2 and "status.json" in files, files
    (record_name,) = [f for f in files if f != "status.json"]
    assert RECORD_NAME.fullmatch(record_name), record_name
    record_file = directory / record_name
    dirs = [p for p in entries if p.is_dir()]
    assert dirs == [p for p in reversed(record_file.parents) if directory in p.parents], dirs

    published = {str(directory)} | {str(p) for p in entries}
    report.created = [c for c in report.created if c not in published]
    inside = str(directory) + os.sep
    report.writes = [
        w for w in report.writes if not (inside in w or w.endswith(" " + str(directory)))
    ]
    record: dict[str, Any] = json.loads(record_file.read_text(encoding="utf-8"))
    aggregate.check_record(record)  # counts only: exactly the schema's fields
    assert record["cameras_listed"] == PIPELINE_CAMERAS
    assert record["frames_ok"] == 4 and record["persons_total"] >= 2
    return record


def test_pipeline_dry_run_writes_only_the_record_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _require_pipeline_model()
    for var in WEATHER_ENV:  # no weather provider: no request leaves the machine
        monkeypatch.delenv(var, raising=False)
    canary = _canary(_leak_os_open) if os.environ.get(CANARY_ENV) == "1" else None
    with (
        guarded(tmp_path, monkeypatch, capsys, caplog) as report,
        canary or contextlib.nullcontext(),
        FakeCameraServer() as server,
    ):
        cams = _pipeline_cameras(server)
        monkeypatch.setattr(registry, "list_cameras", lambda app_key, **kwargs: cams)
        assert cli.main(["sweep", "--dry-run", "--model", PIPELINE_MODEL]) == 0
    record = _accept_published(report, tmp_path / "tmp")
    assert set(record["per_camera"]) == {cams[0].id, cams[1].id}
    report.assert_clean()


PIPELINE_PRELUDE = r"""
import os
for var in WEATHER_ENV:
    os.environ.pop(var, None)

from wearreport import cli, registry
from wearreport.testing.fake_cameras import FakeCameraServer

with FakeCameraServer() as server:
    cams = server.cameras(CAMERAS)
    with open(PEOPLE, "rb") as fh:
        people = fh.read()
    server.serve_body(cams[0].id, people)
    server.serve_body(cams[1].id, people)
    server.serve_404(cams[2].id)
    server.serve_corrupt(cams[3].id)
    registry.list_cameras = lambda app_key, **kwargs: cams
    assert cli.main(["sweep", "--dry-run", "--model", MODEL]) == 0
"""


def test_pipeline_process_writes_only_the_record_and_status(tmp_path: Path) -> None:
    """The sweep command in a child process from interpreter start to exit: onnxruntime,
    the JSON log handler and the publisher included."""
    _require_pipeline_model()
    prelude = (
        f"WEATHER_ENV = {WEATHER_ENV!r}\nCAMERAS = {PIPELINE_CAMERAS}\n"
        f"MODEL = {PIPELINE_MODEL!r}\nPEOPLE = {str(PEOPLE_FIXTURE)!r}\n" + PIPELINE_PRELUDE
    )
    report = _run_dry_sweep_process(tmp_path, prelude)
    assert '"message": "sweep published"' in report.output  # the JSON logs were checked too
    _accept_published(report, tmp_path / "tmp")
    report.assert_clean()
