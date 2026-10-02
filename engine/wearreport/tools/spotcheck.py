"""Spot-check detection precision on live frames, keeping only the tallies.

  python -m wearreport.tools.spotcheck --n 20 [--mode crops|frames] [--min-persons 3]
      [--seed S] [--out-dir DIR] [--reviewer NAME] [--model yolox_m.onnx]
      [--view files|window] [--judgements PATH] [--timeout SECONDS] [--dry-run]
      [--judge NAME --judge-max-requests N] [--record-boxes] [--attributes]
      [--confirm-stop] [--allow-dark] [--source london|austin|calgary] [--bbox S,W,N,E]

Lists the cameras (`wearreport.registry`), fetches one sweep in memory
(`wearreport.fetch`), runs the detector with its default thresholds, and samples up to N
frames with at least `--min-persons` person detections, at random (seeded by `--seed`).
The detections are rendered for a reviewer, either as one crop per detection with a 50%
margin (`crops`, the default) or as whole frames with numbered boxes (`frames`). With
`--view files` (the default except on Windows) the images go into a temporary directory
and the reviewer judges each one from the keyboard, or by writing a JSON file
(`--judgements`) that the tool polls for. With `--view window` (crops only; the default on
Windows) one tkinter window shows each crop straight from memory and takes one key per
crop, and no image is written anywhere. The reviewer can answer "cannot tell" (`u`) for a
box; such a box is left out of the statistics. The tool then writes one statistics file,
`<out-dir>/YYYY-MM-DD.json` (then `-2`, `-3`...). With `--record-boxes` (crops mode only)
it also writes, next to it, one per-box file with the same name,
`<out-dir>/boxes/YYYY-MM-DD.json`: each box's height in source-frame pixels and its label,
the light at the start of the sweep, and no image, position or camera id. It writes
nothing else. The formats are in `spotchecks/README.md`.

While it fetches and detects, the tool prints progress to stderr: the number of cameras
listed, then every PROGRESS_EVERY frames fetched and run through the detector, and a line
before the review opens. Only counts; never a camera id or image data.

Privacy (AGENTS.md INV-1, exception (c)). This is the only engine module that writes
images derived from camera frames, and it writes them only into a directory it creates
with `tempfile.mkdtemp(prefix=TEMP_PREFIX)` (mode 0700) and always deletes:

- after a normal run, an exception, the review timeout, or any catchable signal whose
  default action ends the process (SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGUSR1, ...; see
  HANDLED_SIGNALS). Signals are turned into exceptions, except while the directory is
  being created or deleted: a signal that arrives then is held until that step is done.
  Only the first signal is raised; later ones are dropped, so they cannot interrupt the
  clean-up the first one started;
- the directory holds a lock (flock; on Windows, msvcrt.locking on a lock file next to
  it, `.wearreport-spotcheck-*.lock`, so that no open file is ever inside the directory)
  while the tool runs. At start the tool deletes every `wearreport-spotcheck-*`
  directory of the current user in the temporary directory that is older than the
  timeout and not locked, which is what SIGKILL (which cannot be handled), a killed
  process on Windows or a power cut leaves behind. A running instance's directory stays
  locked, so a second instance never deletes it;
- image files are created with O_EXCL and O_NOFOLLOW (where the platform has it), mode
  0600, and named after their number only (`crop-0001.png`, `frame-0001.png`), never
  after a camera. Camera ids are dropped as soon as the sweep is fetched.

The window view (`--view window`) writes no image and creates no directory: each crop is
encoded to PNG bytes in memory and handed to a tkinter PhotoImage as data.

On Windows there is no SIGALRM: the review timeout is a timer thread that raises SIGINT
in the process, which the handler turns into the timeout. Standard input is read by a
thread instead of select(), and the handled signals are SIGINT, SIGTERM and SIGBREAK.

The static privacy guard exempts exactly this file from its binary-open rules
(`IMAGE_WRITE_EXEMPTION` in scripts/privacy_guard.py); every other rule still applies,
and no other engine module may import this one. The tool refuses to run when `CI` or
`GITHUB_ACTIONS` is non-empty, and when the temporary directory lies inside this
repository or any git work tree, before any network access.

With `--judge NAME` (a DeepInfra model of `judge_hosted.DEEPINFRA`; crops mode only), once
the reviewer has finished and the judgements are valid, the same crops are sent to that
model, one per request, as PNG bytes encoded in memory from the arrays the reviewer was
shown: never a whole frame, and never a file of the review directory, which is deleted by
then. The judge is never shown the reviewer's answers. At most `--judge-max-requests N`
requests are made, retries included, to https://api.deepinfra.com only
(`judge_hosted.LiveCropJudge`, the one entry point for live crops); the key comes from
DEEPINFRA_API_KEY when it is set. The statistics file then gains a `judge` block: the
reviewer x judge confusion counts, the judge's precision, its requests, tokens and cost. A
judge failure (the request limit, an HTTP error, a timeout) never loses the reviewer's
statistics: they are written with the judge's counts so far and `status` "incomplete".
Nothing about a crop is printed, logged or kept.

With `--attributes`, the tool runs an attribute session instead of a detection check. Only
person boxes at least NEAR_FIELD_MIN_HEIGHT_PX tall in the source frame are kept (and
`--min-persons` counts those), and each is shown as a crop. `--min-height N` (attribute
sessions only, NEAR_FIELD_MIN_HEIGHT_PX to MAX_ATTRIBUTE_MIN_HEIGHT_PX) keeps only boxes at
least N pixels tall instead, and the attribute file records N. For each crop the reviewer
answers ATTRIBUTE_QUESTIONS in order, one at a time: `y` yes, `n` no, `u` cannot tell, or
`x` (not a person, or nothing can be told) to reject the crop; in the window
(`--view window`), where Backspace goes back one answer, or in a JSON file
(`--judgements`). With `--judge`, the crops the reviewer did not reject then go to that
model (`judge_hosted.LiveAttributeJudge`), pixels only, under the same rules as above. The
tool writes one file, `<out-dir>/attributes/YYYY-MM-DD.json` (then `-2`, ...): each kept
crop's height and the reviewer's and the model's answers, and nothing else (no statistics
file, no per-box file). Frames mode, keyboard entry and `--record-boxes` are refused. In
the window, the session does not start when it is dark in London (sun below -6°) unless
`--allow-dark` is given; answers from a JSON file are taken at any light.

With `--source austin` (attribute sessions only; anything else is a usage error, exit 2),
the frames are one pass over the City of Austin's traffic cameras inside `--bbox` (default
downtown Austin), fetched in memory with `pilot_heights`' camera list, selection, URL
policies, single attempt and body cap, and its JPEG header reader and decoder (and so its
MAX_HEADER_PIXELS bound). Only stills whose header declares 1920x1080 are kept; the
others are counted in the progress lines and never shown. Crops are cut from the
full-resolution frame, heights are its pixels, the window fits a large crop to the
screen (`display_size`), and the daylight check and the record's `light` use Austin's
sun. The record is `<out-dir>/attributes/YYYY-MM-DD-austin.json` (then `-austin-2`, ...)
and ends with `"source": "austin"`. `--source london`, the default, is unchanged.

`--source calgary` does the same with the City of Calgary's traffic cameras (inside
`--bbox`, default downtown Calgary) and Calgary's sun: only stills whose header declares
840x630 are kept, and the record is `<out-dir>/attributes/YYYY-MM-DD-calgary.json` (then
`-calgary-2`, ...), ending with `"source": "calgary"`.

`--dry-run` sweeps a local fake camera server that serves the licensed fixture photos
in fixtures/detect/ (no network); with `--source austin`, a fake Austin camera list and
1920x1080 stills made from them, and placeholders; with `--source calgary`, the same at
840x630. Its statistics describe those photos, not the cameras, so it needs an `--out-dir`
other than spotchecks/.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import datetime
import functools
import gc
import itertools
import json
import math
import os
import queue
import random
import re
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import Counter
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Sized,
)
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Literal, Protocol, TextIO, get_args, runtime_checkable

import numpy as np
import numpy.typing as npt

from wearreport import detect, fetch, registry
from wearreport._cv import cv2, encode_jpeg
from wearreport.settings import SettingsError, load_settings
from wearreport.tools import judge_hosted, pilot_heights

if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetFileType.argtypes = [wintypes.HANDLE]
    _kernel32.GetFileType.restype = wintypes.DWORD
    _kernel32.PeekNamedPipe.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    _kernel32.PeekNamedPipe.restype = wintypes.BOOL
else:
    import fcntl
    import select

if TYPE_CHECKING:
    import tkinter

WINDOWS = sys.platform == "win32"

Mode = Literal["crops", "frames"]
MODES: tuple[Mode, ...] = get_args(Mode)
View = Literal["files", "window"]
VIEWS: tuple[View, ...] = get_args(View)
# Where the frames come from: TfL's JamCams, or (attribute sessions only) the City of
# Austin's HD traffic cameras or the City of Calgary's, fetched as `pilot_heights` fetches
# them.
Source = Literal["london", "austin", "calgary"]
SOURCES: tuple[Source, ...] = get_args(Source)
Frame = npt.NDArray[np.uint8]

TEMP_PREFIX = "wearreport-spotcheck-"
NUMBERING_FILE = "numbering.json"
IMAGE_SUFFIX = ".png"  # lossless, so the numbered boxes stay legible
DEFAULT_MODEL = "yolox_m.onnx"  # the model the sweep uses
DEFAULT_OUT_DIR = "spotchecks"
DEFAULT_TIMEOUT_S = 30 * 60
DEFAULT_MIN_PERSONS = 3
DEFAULT_REVIEWER = "unnamed"
CI_VARIABLES = ("CI", "GITHUB_ACTIONS")
# Every catchable signal whose default action ends the process, where the platform has
# it, so that none of them can end the run without clean-up. Left out: SIGALRM, which is
# the review alarm and handled as a timeout; SIGPIPE and SIGXFSZ, which Python ignores;
# and the fault signals (SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTRAP, SIGSYS, SIGABRT),
# since a Python handler cannot run while C code faults and would turn a crash into a
# hang. Like SIGKILL, what those leave behind is deleted by a later run. SIGBREAK is
# Windows' Ctrl-Break.
TERMINATING_SIGNAL_NAMES = (
    "SIGINT",
    "SIGTERM",
    "SIGBREAK",
    "SIGHUP",
    "SIGQUIT",
    "SIGUSR1",
    "SIGUSR2",
    "SIGXCPU",
    "SIGVTALRM",
    "SIGPROF",
    "SIGPOLL",
    "SIGIO",
    "SIGPWR",
    "SIGSTKFLT",
)


def _terminating_signals() -> tuple[int, ...]:
    named = {
        int(getattr(signal, name)) for name in TERMINATING_SIGNAL_NAMES if hasattr(signal, name)
    }
    low, high = getattr(signal, "SIGRTMIN", None), getattr(signal, "SIGRTMAX", None)
    realtime = set(range(int(low), int(high) + 1)) if low and high else set()
    return tuple(sorted(named | realtime))


HANDLED_SIGNALS = _terminating_signals()
# The review alarm: SIGALRM where the platform has it. Windows has none; a timer thread
# raises SIGINT instead, and the guard tells it from Ctrl-C by a flag.
ALARM_SIGNALS: tuple[int, ...] = (signal.SIGALRM,) if hasattr(signal, "SIGALRM") else ()
# Windows only: the file whose first byte holds a review directory's lock. It lies next to
# the directory, never inside it: an open file there could not be read or deleted.
LOCK_SUFFIX = ".lock"
# Extra flags for the files the tool creates, where the platform has them. O_BINARY
# matters on Windows, where a descriptor is otherwise opened in text mode.
CREATE_FLAGS = 0
for _flag in ("O_NOFOLLOW", "O_CLOEXEC", "O_BINARY", "O_NOINHERIT"):
    CREATE_FLAGS |= getattr(os, _flag, 0)
del _flag

MAX_N = 500
MAX_MIN_PERSONS = 100
MAX_TIMEOUT_S = 24 * 3600
MAX_MISSED = 1000  # per image
MAX_JUDGEMENTS_BYTES = 1024 * 1024
MAX_LINE_BYTES = 4096
MAX_FILES_PER_DAY = 1000
MAX_JUDGE_REQUESTS = 10_000
JSON_POLL_S = 0.5
PROGRESS_EVERY = 100  # frames between two progress lines, fetched or detected
REVIEWER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}")

CROP_MARGIN = 0.5  # of the box's width on each side, and of its height above and below
CROP_TARGET_HEIGHT = 240  # small crops are enlarged by a whole factor up to about this
MAX_CROP_SCALE = 8
MAX_FRAME_SCALE_WIDTH = 640  # frames narrower than this are shown at twice their size
BOX_COLOUR = (0, 255, 0)  # BGR
LABEL_COLOUR = (255, 255, 255)
LABEL_BACKGROUND = (0, 0, 0)

DRY_RUN_CAMERAS = 12
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "fixtures" / "detect"
DRY_RUN_FIXTURES = ("people_street.jpg", "umbrella_rain.jpg")

WINDOW_TARGET_HEIGHT = 480  # the window enlarges crops by a whole factor up to about this
WINDOW_MAX_WIDTH = 1200
MAX_WINDOW_SCALE = 4
# An Austin crop must fit on the screen with the window's other rows (title bar, header,
# question and legend) and a margin: these many pixels are kept free.
WINDOW_CHROME_PX = 220
WINDOW_SCREEN_MARGIN_PX = 40
WINDOW_TICK_MS = 100  # Tk hands control back to Python this often, so signals are handled
# A backstop: the tick ends a window review this long after its deadline, should the
# review's own timeout callback not have ended it.
WINDOW_DEADLINE_GRACE_S = 5.0
WINDOW_TITLE = "Spot-check"
WINDOW_LEGEND = (
    "Enter or Space: pedestrian   n: not a person   v: person in a vehicle   "
    "u: cannot tell   Backspace: back   q: stop"
)
# With --confirm-stop, the header line after a first `q`.
CONFIRM_STOP_QUESTION = (
    "Stop and discard this session? Press q again to stop, any other key to continue."
)
# Keys that only modify another (Shift for `Q`, say): they neither answer the question
# above nor dismiss it.
MODIFIER_KEYS = frozenset(
    f"{name}_{side}"
    for name in ("Shift", "Control", "Alt", "Meta", "Super", "Hyper", "Option")
    for side in "LR"
) | {"Caps_Lock", "Num_Lock", "Shift_Lock", "ISO_Level3_Shift", "Mode_switch", "Command"}

BOXES_DIR = "boxes"  # the per-box files, in the statistics directory
# The light at the start of the sweep, from the sun's elevation over central London:
# "day" at LIGHT_DAY_DEG or above, "twilight" (civil) at LIGHT_TWILIGHT_DEG or above, else
# "dark".
LONDON = (51.5074, -0.1278)  # latitude and longitude, degrees (west negative)
AUSTIN = pilot_heights.AUSTIN  # central Austin, for Austin sessions
CALGARY = pilot_heights.CALGARY  # central Calgary, for Calgary sessions
LIGHT_DAY_DEG = 0.0
LIGHT_TWILIGHT_DEG = -6.0

# The attribute session (--attributes). Only person boxes at least this tall, in source-
# frame pixels (the height the per-box file records), are shown: the near-field threshold
# that `spotcheck_summary --heights` chose on the baseline data.
NEAR_FIELD_MIN_HEIGHT_PX = 31
# --min-height N raises that threshold for one session, to at most this: crops too short to
# judge clothing are then not shown. The attribute file records the threshold used.
MAX_ATTRIBUTE_MIN_HEIGHT_PX = 200
ATTRIBUTES_DIR = "attributes"  # the attribute files, in the statistics directory
# (the key in a judgements file and in the summary, the question), in the order asked.
ATTRIBUTE_QUESTIONS = (
    ("outer_layer", "Outer layer (coat or jacket)?"),
    ("bare_legs", "Bare legs (shorts or short skirt)?"),
    ("umbrella", "Holding an open umbrella?"),
)
ATTRIBUTE_NAMES = tuple(name for name, _question in ATTRIBUTE_QUESTIONS)
ATTRIBUTE_ANSWERS = ("y", "n", "u")  # yes, no, cannot tell (that question only)
REJECT = "x"  # not a person, or nothing can be told: the crop's answers are discarded
ATTRIBUTE_LEGEND = (
    "y: yes   n: no   u: cannot tell   x: not a person or nothing can be told   "
    "Backspace: back   q: stop"
)

# Austin sessions (--source austin). Only frames whose header declares exactly this size
# are shown (the cameras' 320x176 "no image" placeholders are not), and each pass is
# bounded by one wall-clock deadline, as the pilot's is.
AUSTIN_FRAME_SIZE = pilot_heights.HD  # width, height
AUSTIN_FETCH_TIMEOUT_S = float(pilot_heights.DEFAULT_TIMEOUT_S)
AUSTIN_FILE_SUFFIX = "-austin"  # the attribute file is <date>-austin.json, then -2, ...
DRY_RUN_AUSTIN_CAMERAS = 6
DRY_RUN_PLACEHOLDER = (320, 176)  # width, height
# Calgary sessions (--source calgary): the same, at Calgary's frame size.
CALGARY_FRAME_SIZE = pilot_heights.CALGARY_FRAME_SIZE  # width, height
CALGARY_FILE_SUFFIX = "-calgary"  # the attribute file is <date>-calgary.json, then -2, ...


class SpotcheckError(RuntimeError):
    """The spot-check cannot go on; the message says why."""


class JudgementError(ValueError):
    """A judgement is invalid; the message says why, and the reviewer can try again."""


class ReviewTimeout(Exception):
    """The reviewer did not finish before the timeout."""


class ReviewAborted(Exception):
    """The reviewer stopped the review (end of input, or `q`)."""


class Interrupted(BaseException):
    """A signal asked the tool to stop. A BaseException, so `except Exception` misses it."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


# Sampling -----------------------------------------------------------------------------


class FrameDetector(Protocol):
    def detect(self, frame: Frame) -> list[detect.Detection]: ...


@dataclass(frozen=True, slots=True)
class DetectorInfo:
    model: str
    sha256: str
    conf: float


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Where frames come from and what detects people in them."""

    frames: Callable[[], Iterable[Frame]]
    detector: FrameDetector
    info: DetectorInfo


@dataclass(frozen=True, slots=True, eq=False)
class Sample:
    frame: Frame
    persons: tuple[detect.Detection, ...]


def sample(
    frames: Iterable[Frame],
    detector: FrameDetector,
    *,
    n: int,
    min_persons: int,
    seed: int | None,
    progress: Callable[[int], None] | None = None,
    min_height: int = 0,
) -> list[Sample]:
    """Up to `n` frames with at least `min_persons` person detections, chosen uniformly at
    random (reservoir sampling, so at most `n` frames are held) and returned in the order
    they came in. `progress`, if given, is called with the number of frames run through
    the detector so far, after each one. Only person boxes at least `min_height` pixels
    tall (`box_height`) count, and only they are kept.

    A frame on which the detector raises DetectorError is skipped, and the number skipped
    is printed to stderr: only the count, never a camera id or image data."""
    rng = random.Random(seed)  # noqa: S311  (sampling, not security)
    kept: list[tuple[int, Sample]] = []
    seen = skipped = done = 0
    for frame in frames:
        found: list[detect.Detection] | None = None
        try:
            found = detector.detect(frame)
        except detect.DetectorError:
            skipped += 1
        done += 1
        if progress is not None:
            progress(done)
        if found is None:
            continue
        persons = tuple(d for d in found if d.label == "person" and box_height(d.box) >= min_height)
        if len(persons) < min_persons:
            continue
        if len(kept) < n:
            kept.append((seen, Sample(frame, persons)))
        else:
            slot = rng.randint(0, seen)
            if slot < n:
                kept[slot] = (seen, Sample(frame, persons))
        seen += 1
    if skipped:
        print(
            f"spotcheck: skipped {skipped} frame(s) the detector could not read",
            file=sys.stderr,
            flush=True,
        )
    return [s for _, s in sorted(kept, key=lambda pair: pair[0])]


def _progress(text: str) -> None:
    """One progress line on stderr. Callers pass counts only."""
    print(f"spotcheck: {text}", file=sys.stderr, flush=True)


def _due(done: int, total: int | None) -> bool:
    return done % PROGRESS_EVERY == 0 or done == total


class Frames:
    """A sweep's frames, without their camera ids, in camera order. Iterating hands each
    frame on once and lets go of it; the length is the number of frames fetched."""

    def __init__(self, frames: list[Frame]) -> None:
        self._total = len(frames)
        self._frames = frames[::-1]

    def __len__(self) -> int:
        return self._total

    def __iter__(self) -> Iterator[Frame]:
        while self._frames:
            yield self._frames.pop()


def sweep_frames(cameras: Sequence[registry.Camera]) -> Frames:
    """Fetch one sweep in memory with `fetch.fetch_sweep`, PROGRESS_EVERY cameras at a
    time, printing the number of cameras, then a progress line after each batch. Camera
    ids are dropped with each result."""
    total = len(cameras)
    _progress(f"{total} cameras listed")
    frames: list[Frame] = []
    for start in range(0, total, PROGRESS_EVERY):
        batch = cameras[start : start + PROGRESS_EVERY]
        frames += [r.frame for r in fetch.fetch_sweep(batch) if r.frame is not None]
        _progress(f"fetched {start + len(batch)} of {total}")
    return Frames(frames)


_sweep_frames = sweep_frames  # the name child processes in earlier tests call


def _detector_info(path: Path, conf: float) -> DetectorInfo:
    try:
        digest = detect.sha256_of(path)
    except OSError as exc:
        raise SpotcheckError(f"cannot read model {path.name}: {exc.strerror}") from None
    names = [name for name, pinned in detect.MODEL_SHA256.items() if pinned == digest]
    if not names:
        raise SpotcheckError(f"model {path.name} does not match a pinned SHA-256")
    return DetectorInfo(model=Path(names[0]).stem, sha256=digest, conf=conf)


def _open_detector(model: str) -> tuple[detect.Detector, DetectorInfo]:
    path = detect.model_path(model)
    try:
        detector = detect.Detector(path)
    except detect.DetectorError as exc:
        raise SpotcheckError(str(exc)) from None
    return detector, _detector_info(path, detector.conf)


def live_pipeline(model: str) -> Pipeline:
    detector, info = _open_detector(model)

    def frames() -> Frames:
        try:
            cameras = registry.list_cameras(load_settings().tfl_app_key)
        except (registry.RegistryError, SettingsError) as exc:
            raise SpotcheckError(f"cannot list cameras: {exc}") from None
        return sweep_frames(cameras)

    return Pipeline(frames=frames, detector=detector, info=info)


def dry_run_pipeline(model: str) -> Pipeline:
    """A local fake camera server serving the licensed fixture photos, and the real model."""
    from wearreport.testing.fake_cameras import FakeCameraServer

    try:
        bodies = [(FIXTURE_DIR / name).read_bytes() for name in DRY_RUN_FIXTURES]
    except OSError as exc:
        raise SpotcheckError(f"cannot read the dry-run fixtures: {exc.strerror}") from None
    detector, info = _open_detector(model)

    def frames() -> Frames:
        with FakeCameraServer() as server:
            cameras = server.cameras(DRY_RUN_CAMERAS)
            for i, camera in enumerate(cameras):
                if i % 3 != 2:  # every third camera serves noise
                    server.serve_body(camera.id, bodies[i % len(bodies)])
            return sweep_frames(cameras)

    return Pipeline(frames=frames, detector=detector, info=info)


# Austin -------------------------------------------------------------------------------

DetectorOpener = Callable[[str], tuple[FrameDetector, DetectorInfo]]


@dataclass(frozen=True, slots=True)
class AustinEndpoints:
    """Where an Austin session's camera list and stills come from, and the URL policies
    they must meet (tests: a server on 127.0.0.1)."""

    dataset_url: str = pilot_heights.DATASET_URL
    dataset_policy: pilot_heights.UrlPolicy = pilot_heights.DATASET_POLICY
    image_policy: pilot_heights.UrlPolicy = pilot_heights.SCREENSHOT_POLICY


@dataclass(frozen=True, slots=True)
class CalgaryEndpoints:
    """Where a Calgary session's camera list and stills come from, and the URL policies
    they must meet (tests: a server on 127.0.0.1)."""

    dataset_url: str = pilot_heights.CALGARY_DATASET_URL
    dataset_policy: pilot_heights.UrlPolicy = pilot_heights.CALGARY_DATASET_POLICY
    image_policy: pilot_heights.UrlPolicy = pilot_heights.CALGARY_IMAGE_POLICY


CityEndpoints = AustinEndpoints | CalgaryEndpoints
StillFetcher = Callable[[str, str, float], Frame]


class _NotHD(Exception):
    """A still whose header declares a size other than its city's frame size."""


def _city_still(url: str, scheme: str, end: float, frame_size: tuple[int, int]) -> Frame:
    """One still, fetched once and decoded in memory at full resolution. Raises _NotHD
    for any declared size other than `frame_size` (before decoding), and pilot_heights'
    FrameRefused or FrameFailed as its pass would count them."""
    body = pilot_heights.download_within(url, scheme, end)
    try:
        size = pilot_heights.jpeg_size(body)
        if size is not None and size != frame_size:
            raise _NotHD
        decoded = pilot_heights.decode_frame(body)  # the header allowlist and bound, too
    finally:
        del body
    width, height = frame_size
    if decoded.frame.shape[:2] != (height, width):
        raise _NotHD  # never a reduced frame: crops come from full-resolution pixels
    return decoded.frame


def _austin_still(url: str, scheme: str, end: float) -> Frame:
    """One Austin still (see _city_still)."""
    return _city_still(url, scheme, end, AUSTIN_FRAME_SIZE)


def _calgary_still(url: str, scheme: str, end: float) -> Frame:
    """One Calgary still (see _city_still)."""
    return _city_still(url, scheme, end, CALGARY_FRAME_SIZE)


def _austin_result(future: Future[Frame], counts: Counter[str]) -> Frame | None:
    """The still `future` fetched, or None after counting why there is none."""
    try:
        return future.result()
    except _NotHD:
        counts["not_hd"] += 1
    except pilot_heights.FrameRefused:
        counts["refused"] += 1
    except pilot_heights.FrameFailed as exc:
        counts["failed"] += 1
        counts[f"failed_{exc.kind}"] += 1
    return None


def _failed_text(counts: Counter[str]) -> str:
    """`N failed`, then the non-zero counts by kind in fetch.ERROR_KINDS order, and last
    those of any other kind as `other`, so that the counts add up to N. Never the other
    kind's own text."""
    known = [(kind, counts[f"failed_{kind}"]) for kind in fetch.ERROR_KINDS]
    other = counts["failed"] - sum(n for _kind, n in known)
    kinds = ", ".join(f"{kind} {n}" for kind, n in [*known, ("other", other)] if n > 0)
    return f"{counts['failed']} failed" + (f" ({kinds})" if counts["failed"] else "")


def austin_frames(
    endpoints: AustinEndpoints,
    bbox: pilot_heights.BBox,
    *,
    timeout_s: float = AUSTIN_FETCH_TIMEOUT_S,
    concurrency: int = pilot_heights.CONCURRENCY,
) -> Generator[Frame]:
    """The 1920x1080 stills of the Austin cameras inside `bbox`, as they arrive (see
    city_frames)."""
    return city_frames(
        pilot_heights.AUSTIN_CITY,
        endpoints,
        bbox,
        _austin_still,
        timeout_s=timeout_s,
        concurrency=concurrency,
    )


def calgary_frames(
    endpoints: CalgaryEndpoints,
    bbox: pilot_heights.BBox,
    *,
    timeout_s: float = AUSTIN_FETCH_TIMEOUT_S,
    concurrency: int = pilot_heights.CONCURRENCY,
) -> Generator[Frame]:
    """The 840x630 stills of the Calgary cameras inside `bbox`, as they arrive (see
    city_frames)."""
    return city_frames(
        pilot_heights.CALGARY_CITY,
        endpoints,
        bbox,
        _calgary_still,
        timeout_s=timeout_s,
        concurrency=concurrency,
    )


def city_frames(
    city: pilot_heights.City,
    endpoints: CityEndpoints,
    bbox: pilot_heights.BBox,
    still: StillFetcher,
    *,
    timeout_s: float = AUSTIN_FETCH_TIMEOUT_S,
    concurrency: int = pilot_heights.CONCURRENCY,
) -> Generator[Frame]:
    """The stills of `city`'s cameras inside `bbox` whose header declares the city's
    frame size, fetched by `still`, as they arrive. The camera list and its record
    fields, the selection (at most pilot_heights.DEFAULT_MAX_CAMERAS cameras), the URL
    policies, one attempt per camera and the body cap are pilot_heights'. At most
    `concurrency` stills are being fetched or waiting at a time, and nothing is kept once
    handed on. Prints counts only: never a URL, a camera or image data."""
    end = time.monotonic() + timeout_s
    try:
        records = pilot_heights.fetch_dataset(
            endpoints.dataset_url,
            endpoints.dataset_policy,
            min(pilot_heights.DATASET_TIMEOUT_S, timeout_s),
        )
    except pilot_heights.PilotError as exc:
        raise SpotcheckError(f"cannot list the {city.title} cameras: {exc}") from None
    selection = pilot_heights.select_cameras(
        records,
        bbox,
        policy=endpoints.image_policy,
        max_cameras=pilot_heights.DEFAULT_MAX_CAMERAS,
        fields=(city.url_field, city.point_field),
    )
    del records
    _progress(
        f"{selection.listed} {city.title} cameras listed, {len(selection.urls)} selected "
        f"({selection.skipped} malformed record(s) skipped, {selection.refused} URL(s) refused)"
    )
    counts: Counter[str] = Counter(refused=selection.refused)
    urls = iter(selection.urls)
    scheme = endpoints.image_policy.scheme
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=city.name)
    pending: set[Future[Frame]] = set()

    def submit() -> None:
        for url in urls:
            pending.add(pool.submit(still, url, scheme, end))
            return

    try:
        for _ in range(concurrency):
            submit()
        while pending:
            wait_s = max(0.0, end - time.monotonic()) + 1.0
            done, _ = wait(pending, timeout=wait_s, return_when=FIRST_COMPLETED)
            if not done and time.monotonic() > end + 1.0:
                # Every request is bounded by the deadline; this is a backstop only.
                stuck = len(pending) + sum(1 for _url in urls)
                counts["failed"] += stuck
                counts["failed_timeout"] += stuck
                break
            for future in done:
                pending.discard(future)
                submit()
                frame = _austin_result(future, counts)
                if frame is not None:
                    counts["ok"] += 1
                    yield frame
                del frame
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    size = "{}x{}".format(*city.frame_size)
    _progress(
        f"fetched {counts['ok']} {size} frame(s) of {len(selection.urls)}: "
        f"{counts['not_hd']} not {size} (skipped), {_failed_text(counts)}, "
        f"{counts['refused']} refused"
    )


def austin_pipeline(
    model: str, bbox: pilot_heights.BBox, endpoints: AustinEndpoints, opener: DetectorOpener
) -> Pipeline:
    detector, info = opener(model)
    return Pipeline(frames=lambda: austin_frames(endpoints, bbox), detector=detector, info=info)


def calgary_pipeline(
    model: str, bbox: pilot_heights.BBox, endpoints: CalgaryEndpoints, opener: DetectorOpener
) -> Pipeline:
    detector, info = opener(model)
    return Pipeline(frames=lambda: calgary_frames(endpoints, bbox), detector=detector, info=info)


def _hd_still(body: bytes, frame_size: tuple[int, int] = AUSTIN_FRAME_SIZE) -> bytes:
    """A licensed fixture photo, enlarged 1.5 times (or less, to fit) onto a grey frame of
    `frame_size` (1920x1080 by default), as JPEG."""
    photo = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if photo is None:
        raise SpotcheckError("cannot decode a dry-run fixture")
    width, height = frame_size
    scale = min(1.5, width / photo.shape[1], height / photo.shape[0])
    size = (int(photo.shape[1] * scale), int(photo.shape[0] * scale))
    photo = cv2.resize(photo, size, interpolation=cv2.INTER_LINEAR)
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    top, left = (height - size[1]) // 2, (width - size[0]) // 2
    frame[top : top + size[1], left : left + size[0]] = photo
    return encode_jpeg(frame)


def dry_run_austin_pipeline(
    model: str, bbox: pilot_heights.BBox, opener: DetectorOpener
) -> Pipeline:
    """A local fake Austin: a camera list and stills served on 127.0.0.1, the stills made
    from the licensed fixture photos, every third one a 320x176 placeholder; no network."""
    return _dry_run_city_pipeline(pilot_heights.AUSTIN_CITY, model, bbox, opener)


def dry_run_calgary_pipeline(
    model: str, bbox: pilot_heights.BBox, opener: DetectorOpener
) -> Pipeline:
    """A local fake Calgary: as the fake Austin, with 840x630 stills."""
    return _dry_run_city_pipeline(pilot_heights.CALGARY_CITY, model, bbox, opener)


def _dry_run_city_pipeline(
    city: pilot_heights.City, model: str, bbox: pilot_heights.BBox, opener: DetectorOpener
) -> Pipeline:
    from wearreport.testing.fake_cameras import FakeCameraServer

    try:
        stills = [
            _hd_still((FIXTURE_DIR / name).read_bytes(), city.frame_size)
            for name in DRY_RUN_FIXTURES
        ]
    except OSError as exc:
        raise SpotcheckError(f"cannot read the dry-run fixtures: {exc.strerror}") from None
    width, height = DRY_RUN_PLACEHOLDER
    placeholder = encode_jpeg(np.full((height, width, 3), 64, dtype=np.uint8))
    detector, info = opener(model)
    south, west, north, east = bbox
    where = [(west + east) / 2, (south + north) / 2]  # GeoJSON: longitude, then latitude

    def frames() -> Iterator[Frame]:
        with FakeCameraServer() as server:
            records = []
            for i in range(DRY_RUN_AUSTIN_CAMERAS):
                camera = f"{city.name}-{i}"
                body = placeholder if i % 3 == 2 else stills[i % len(stills)]
                server.serve_body(camera, body)
                location = {"type": "Point", "coordinates": where}
                records.append({city.url_field: server.url(camera), city.point_field: location})
            server.serve_body("cameras", json.dumps(records).encode())
            policy = pilot_heights.UrlPolicy("http", urllib.parse.urlsplit(server.base_url).netloc)
            if city is pilot_heights.CALGARY_CITY:
                yield from calgary_frames(
                    CalgaryEndpoints(server.url("cameras"), policy, policy), bbox
                )
            else:
                yield from austin_frames(
                    AustinEndpoints(server.url("cameras"), policy, policy), bbox
                )

    return Pipeline(frames=frames, detector=detector, info=info)


# Rendering ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class ReviewItem:
    """One image to judge: its number, file name, the box numbers on it, and its pixels
    (rendered, in memory)."""

    number: int
    file: str
    boxes: tuple[int, ...]
    image: Frame


def crop_bounds(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """The box grown by CROP_MARGIN on every side, in whole pixels, clipped to the frame."""
    x1, y1, x2, y2 = box
    if not all(math.isfinite(v) for v in box) or not (x1 < x2 and y1 < y2):
        raise ValueError("a box must be finite and have a positive area")
    dx, dy = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
    left, top = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
    right, bottom = min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))
    if not (left < right and top < bottom):
        raise ValueError("the box lies outside the frame")
    return left, top, right, bottom


def _label(image: Frame, text: str, x: int, y: int) -> None:
    """Draw `text` on a dark patch just above (x, y), kept inside the image."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    height, width = image.shape[:2]
    patch_w, patch_h = tw + 4, th + baseline + 4
    left = min(max(0, x), max(0, width - patch_w))
    top = min(max(0, y - patch_h), max(0, height - patch_h))
    cv2.rectangle(image, (left, top), (left + patch_w, top + patch_h), LABEL_BACKGROUND, -1)
    cv2.putText(
        image, text, (left + 2, top + th + 2), font, scale, LABEL_COLOUR, thickness, cv2.LINE_AA
    )


def _draw_box(
    image: Frame,
    box: tuple[float, float, float, float],
    scale: float,
    dx: int,
    dy: int,
    number: int,
) -> None:
    x1, y1 = round((box[0] - dx) * scale), round((box[1] - dy) * scale)
    x2, y2 = round((box[2] - dx) * scale) - 1, round((box[3] - dy) * scale) - 1
    cv2.rectangle(image, (x1, y1), (max(x1, x2), max(y1, y2)), BOX_COLOUR, 1)
    _label(image, str(number), x1, y1)


def render_crop(frame: Frame, person: detect.Detection, number: int) -> Frame:
    height, width = frame.shape[:2]
    left, top, right, bottom = crop_bounds(person.box, width, height)
    crop = np.ascontiguousarray(frame[top:bottom, left:right])
    scale = max(1, min(MAX_CROP_SCALE, CROP_TARGET_HEIGHT // (bottom - top)))
    size = ((right - left) * scale, (bottom - top) * scale)
    image = np.asarray(cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST), dtype=np.uint8)
    _draw_box(image, person.box, scale, left, top, number)
    return image


def render_frame(
    frame: Frame, persons: Sequence[detect.Detection], numbers: Sequence[int]
) -> Frame:
    height, width = frame.shape[:2]
    scale = 2 if width < MAX_FRAME_SCALE_WIDTH else 1
    image = np.asarray(
        cv2.resize(frame, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR),
        dtype=np.uint8,
    )
    for person, number in zip(persons, numbers, strict=True):
        _draw_box(image, person.box, scale, 0, 0, number)
    return image


def _file_name(kind: str, number: int) -> str:
    return f"{kind}-{number:04d}{IMAGE_SUFFIX}"


def render(samples: Sequence[Sample], mode: Mode) -> list[ReviewItem]:
    """Render the samples in memory. Boxes are numbered 1, 2, ... across the whole check;
    in `crops` mode each crop is an image, numbered like its box."""
    items: list[ReviewItem] = []
    first = 1
    for index, s in enumerate(samples, start=1):
        numbers = tuple(range(first, first + len(s.persons)))
        first += len(s.persons)
        if mode == "frames":
            image = render_frame(s.frame, s.persons, numbers)
            items.append(ReviewItem(index, _file_name("frame", index), numbers, image))
            continue
        for person, number in zip(s.persons, numbers, strict=True):
            image = render_crop(s.frame, person, number)
            items.append(ReviewItem(number, _file_name("crop", number), (number,), image))
    return items


# The review directory -----------------------------------------------------------------


class _SignalGuard:
    """Turn signals and the review alarm into exceptions, except inside `critical()`,
    where they wait until the critical step is done.

    One-shot: once one signal (or the alarm) has been raised, or `stop()` has been
    called, every later signal is dropped, so nothing can interrupt the clean-up that
    the first one starts.
    """

    def __init__(self) -> None:
        self._depth = 0
        self._stopping = False
        self._pending: BaseException | None = None
        self._previous: dict[int, object] = {}
        self._installed = False
        self._timer: threading.Timer | None = None  # the alarm, where there is no SIGALRM
        self._alarm_due = False

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # signal handlers can only be set from the main thread
        self._installed = True
        for signum in (*HANDLED_SIGNALS, *ALARM_SIGNALS):
            self._previous[signum] = signal.signal(signum, self._handle)

    def stop(self) -> None:
        """From here on, drop every signal: the tool is on its way out."""
        self._stopping = True

    def restore(self) -> None:
        if not self._installed:
            return
        self._stopping = True
        if ALARM_SIGNALS:
            signal.setitimer(signal.ITIMER_REAL, 0)
        self._cancel_timer()
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)  # type: ignore[arg-type]
        self._installed = False

    def alarm(self, seconds: float) -> None:
        """Raise ReviewTimeout after `seconds` (0 cancels). A backstop: reviewers also
        watch their deadline."""
        if not self._installed:
            return
        if ALARM_SIGNALS:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            return
        self._cancel_timer()
        if seconds > 0:
            self._timer = threading.Timer(seconds, self._ring)
            self._timer.daemon = True
            self._timer.start()

    def _ring(self) -> None:
        """The timer thread's alarm. raise() runs Python's C-level handler, which also
        wakes a main thread that is sleeping or waiting (on Windows, only for SIGINT)."""
        if self._stopping:
            return
        self._alarm_due = True
        signal.raise_signal(signal.SIGINT)

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
            if timer is not threading.current_thread():
                timer.join()

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self._stopping:
            return
        alarm = signum in ALARM_SIGNALS or (signum == signal.SIGINT and self._alarm_due)
        exc: BaseException = ReviewTimeout("the review timed out") if alarm else Interrupted(signum)
        if self._depth:
            if self._pending is None:
                self._pending = exc
            return
        self._stopping = True  # before raising: no later signal may interrupt clean-up
        try:
            raise exc
        finally:
            # A local holding the exception, whose traceback holds this frame, is a
            # reference cycle: everything the traceback reaches (a Tk window, for one)
            # would wait for the garbage collector, which may run in another thread.
            del exc

    @contextlib.contextmanager
    def critical(self) -> Iterator[None]:
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if not self._depth and self._pending is not None:
                exc, self._pending = self._pending, None
                if not self._stopping:
                    self._stopping = True
                    try:
                        raise exc
                    finally:
                        del exc  # no reference cycle through this frame (see _handle)


def _lock(path: str | Path) -> int | None:
    """An open descriptor holding an exclusive lock on the directory, or None if another
    process holds it (or it cannot be opened as a directory)."""
    if WINDOWS:
        return _lock_windows(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _is_link_like(st: os.stat_result) -> bool:
    """A symlink, or on Windows any reparse point (a junction, for one)."""
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & reparse)


def lock_file(path: str | Path) -> Path:
    """The Windows lock file of the review directory `path`: `.<name>.lock` next to it.
    The leading dot keeps it out of `wearreport-spotcheck-*`."""
    path = Path(path)
    return path.parent / f".{path.name}{LOCK_SUFFIX}"


def _lock_windows(path: str | Path) -> int | None:
    """Windows cannot open or flock a directory: lock the first byte of the directory's
    lock file (`lock_file`) with msvcrt.locking instead. The lock goes when the descriptor
    is closed or the process ends, however it ends."""
    try:
        st = os.lstat(path)
        if not stat.S_ISDIR(st.st_mode) or _is_link_like(st):
            return None
        fd = os.open(lock_file(path), os.O_RDWR | os.O_CREAT | CREATE_FLAGS, 0o600)
    except OSError:
        return None
    try:
        if sys.platform == "win32":  # always, here; the test is for the type checker
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError:
        os.close(fd)
        return None
    return fd


def _rmtree(path: Path) -> bool:
    """Delete `path` and everything in it; True if it is gone. Never raises."""
    for _ in range(3):
        with contextlib.suppress(OSError):
            shutil.rmtree(path)
        if not os.path.lexists(path):
            return True
    return False


class ReviewDirectory:
    """The one directory the rendered images are written to, locked while it exists."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self._fd: int | None = None

    def create(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
        self.path = path
        os.chmod(path, 0o700)
        self._fd = _lock(path)
        if self._fd is None:
            raise SpotcheckError("cannot lock the review directory")
        return path

    def write_image(self, item: ReviewItem) -> None:
        if self.path is None or Path(item.file).name != item.file or item.file.startswith("."):
            raise ValueError("review images go only into the review directory")
        ok, encoded = cv2.imencode(IMAGE_SUFFIX, item.image)
        if not ok:
            raise SpotcheckError("cannot encode a review image")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | CREATE_FLAGS
        fd = os.open(self.path / item.file, flags, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(encoded.tobytes())

    def write_numbering(
        self,
        items: Sequence[ReviewItem],
        mode: Mode,
        entry: Mapping[str, object] | None = None,
    ) -> Path:
        """The numbering, and a judgements template to copy, as JSON next to the images.
        `entry` is each image's template entry (default: that of a detection check)."""
        if self.path is None:
            raise ValueError("no review directory")
        template = {
            str(item.number): _template_entry(mode) if entry is None else entry for item in items
        }
        numbering = {
            "mode": mode,
            "images": [
                {"image": item.number, "file": item.file, "boxes": list(item.boxes)}
                for item in items
            ],
            "template": template,
        }
        path = self.path / NUMBERING_FILE
        with open(path, "x", encoding="utf-8") as fh:
            json.dump(numbering, fh, indent=2)
            fh.write("\n")
        return path

    def remove(self) -> bool:
        """Delete the directory; True if it is gone (or was never made). Never raises."""
        gone = True
        if self.path is not None:
            gone = _rmtree(self.path)
            if not gone:
                print(
                    f"spotcheck: WARNING: could not delete {self.path}; delete it by hand",
                    file=sys.stderr,
                    flush=True,
                )
        self._unlock()
        return gone

    def _unlock(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None
            if WINDOWS and self.path is not None:
                _remove_lock_file(self.path)


def _remove_lock_file(path: Path) -> None:
    """Delete the Windows lock file of the review directory `path`, if no process holds it
    open (an open file cannot be deleted there). Never raises."""
    with contextlib.suppress(OSError):
        os.unlink(lock_file(path))


def remove_stale(
    tmpdir: Path, max_age_s: float, now: float | None = None, *, guard: _SignalGuard | None = None
) -> int:
    """Delete this user's review directories in `tmpdir` that are older than `max_age_s`
    and not locked by a running instance. Returns how many were deleted.

    With `guard`, each removal is a critical step: a signal that arrives during one waits
    until that directory is gone, so it cannot leave a half-deleted one behind."""
    now = time.time() if now is None else now
    removed = 0
    try:
        entries = list(os.scandir(tmpdir))
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.startswith(TEMP_PREFIX):
            continue
        try:
            # Windows' DirEntry.stat() reports st_ino as 0: ask os.lstat there.
            st = os.lstat(entry.path) if WINDOWS else entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode) or not _owned(st):
            continue
        if now - st.st_mtime <= max_age_s:
            continue
        with guard.critical() if guard is not None else contextlib.nullcontext():
            if WINDOWS:
                removed += _remove_stale_windows(Path(entry.path), st)
                continue
            fd = _lock(entry.path)
            if fd is None:
                continue  # a running instance holds it
            try:
                if os.fstat(fd).st_ino == st.st_ino and _rmtree(Path(entry.path)):
                    removed += 1
            finally:
                os.close(fd)
    return removed


def _owned(st: os.stat_result) -> bool:
    """True for this user's entry. Windows reports no owner in st_uid, but its temporary
    directory is per user; there, links and junctions are left alone instead."""
    if WINDOWS:
        return not _is_link_like(st)
    return st.st_uid == os.getuid()


def _remove_stale_windows(path: Path, st: os.stat_result) -> int:
    """1 if the unlocked directory `path`, still the one `st` describes, was deleted. Its
    lock file goes with it."""
    fd = _lock(path)
    if fd is None:
        return 0  # a running instance holds it
    try:
        same = os.lstat(path).st_ino == st.st_ino
        removed = 1 if same and _rmtree(path) else 0
    except OSError:
        removed = 0
    finally:
        os.close(fd)
    _remove_lock_file(path)
    return removed


# Judgements ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Judgement:
    """For one image: the boxes that are not a person, the boxes that are a person inside
    a vehicle, (frames mode only) how many visible people have no box, and the boxes the
    reviewer cannot tell. Every other box on the image is a pedestrian."""

    not_person: frozenset[int]
    in_vehicle: frozenset[int]
    missed: int | None
    unsure: frozenset[int] = frozenset()


@runtime_checkable
class Reviewer(Protocol):
    """Whoever judges the rendered images. `judge` returns one judgement per item, keyed by
    item number, or raises ReviewTimeout once `deadline` (time.monotonic()) has passed."""

    def judge(
        self, items: Sequence[ReviewItem], mode: Mode, deadline: float
    ) -> Mapping[int, Judgement]: ...


@runtime_checkable
class AttributeReviewer(Protocol):
    """Whoever answers the attribute questions. `attributes` returns, per item number, the
    answers in ATTRIBUTE_QUESTIONS order (e.g. "ynu"), or None for a rejected crop, or
    raises ReviewTimeout once `deadline` (time.monotonic()) has passed."""

    def attributes(
        self, items: Sequence[ReviewItem], deadline: float
    ) -> Mapping[int, str | None]: ...


# The answers a judgement gives per box, and how messages name them.
BOX_ANSWERS = ("not_person", "in_vehicle", "unsure")
BOX_ANSWER_WORDS = {
    "not_person": "not a person",
    "in_vehicle": "in a vehicle",
    "unsure": "cannot tell",
}


def _template_entry(mode: Mode) -> dict[str, object]:
    entry: dict[str, object] = {"not_person": [], "in_vehicle": [], "unsure": []}
    if mode == "frames":
        entry["missed"] = 0
    return entry


def validate(judgement: Judgement, item: ReviewItem, mode: Mode) -> Judgement:
    """Return `judgement` if it is valid for `item`, else raise JudgementError."""
    for name in BOX_ANSWERS:
        boxes = getattr(judgement, name)
        if not isinstance(boxes, frozenset) or not all(
            type(b) is int
            for b in boxes  # bool is not a box number
        ):
            raise JudgementError(f"image {item.number}: {name} must be box numbers")
        unknown = sorted(boxes - set(item.boxes))
        if unknown:
            raise JudgementError(
                f"image {item.number}: box {unknown[0]} is not on this image "
                f"(its boxes: {_numbers(item.boxes)})"
            )
    for first, second in itertools.combinations(BOX_ANSWERS, 2):
        both = sorted(getattr(judgement, first) & getattr(judgement, second))
        if both:
            raise JudgementError(
                f"image {item.number}: box {both[0]} cannot be both "
                f"{BOX_ANSWER_WORDS[first]} and {BOX_ANSWER_WORDS[second]}"
            )
    missed = judgement.missed
    if mode == "crops":
        if missed is not None:
            raise JudgementError("missed people are counted in frames mode only")
    elif type(missed) is not int or not 0 <= missed <= MAX_MISSED:
        raise JudgementError(
            f"image {item.number}: missed must be a whole number from 0 to {MAX_MISSED}"
        )
    return judgement


def validate_all(
    judgements: Mapping[int, Judgement], items: Sequence[ReviewItem], mode: Mode
) -> dict[int, Judgement]:
    by_number = {item.number: item for item in items}
    extra = sorted(set(judgements) - set(by_number), key=str)
    if extra:
        raise JudgementError(f"there is no image {extra[0]!r}")
    missing = sorted(set(by_number) - set(judgements))
    if missing:
        raise JudgementError(f"image {missing[0]} has no judgement")
    return {n: validate(judgements[n], by_number[n], mode) for n in sorted(by_number)}


def _reject_constant(name: str) -> object:
    raise JudgementError(f"{name} is not a number")


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise JudgementError(f"the key {key[:20]!r} appears twice in one object")
        seen[key] = value
    return seen


def _box_list(value: object, key: str, image: str) -> frozenset[int]:
    if not isinstance(value, list) or not all(type(b) is int for b in value):
        raise JudgementError(f"image {image}: {key} must be a list of box numbers")
    if len(set(value)) != len(value):
        raise JudgementError(f"image {image}: {key} lists a box twice")
    return frozenset(value)


def _load_judgements(raw: bytes) -> object:
    """A judgements file's JSON; raise JudgementError."""
    if len(raw) > MAX_JUDGEMENTS_BYTES:
        raise JudgementError(f"the file is larger than {MAX_JUDGEMENTS_BYTES} bytes")
    try:
        data = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except JudgementError:
        raise
    except json.JSONDecodeError as exc:
        raise JudgementError(f"not valid JSON: {exc.msg} (line {exc.lineno})") from None
    except UnicodeDecodeError:
        raise JudgementError("the file is not UTF-8 text") from None
    except (ValueError, RecursionError, OverflowError) as exc:
        raise JudgementError(f"not valid JSON: {type(exc).__name__}") from None
    return data


def parse_judgements(raw: bytes, items: Sequence[ReviewItem], mode: Mode) -> dict[int, Judgement]:
    """Parse a judgements file (format in spotchecks/README.md); raise JudgementError."""
    data = _load_judgements(raw)
    if not isinstance(data, dict):
        raise JudgementError('the file must hold one object, e.g. {"1": {...}, "2": {...}}')
    numbers = {str(item.number): item.number for item in items}
    judgements: dict[int, Judgement] = {}
    for key, entry in data.items():
        if key not in numbers:
            raise JudgementError(f"there is no image {key[:20]!r}")
        if not isinstance(entry, dict):
            raise JudgementError(f"image {key}: the judgement must be an object")
        allowed = set(BOX_ANSWERS) | ({"missed"} if mode == "frames" else set())
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise JudgementError(f"image {key}: unexpected field {unknown[0][:20]!r}")
        if mode == "frames" and "missed" not in entry:
            raise JudgementError(f"image {key}: missed is required in frames mode")
        missed = entry.get("missed")
        if mode == "frames" and type(missed) is not int:
            raise JudgementError(f"image {key}: missed must be a whole number")
        judgements[numbers[key]] = Judgement(
            _box_list(entry.get("not_person", []), "not_person", key),
            _box_list(entry.get("in_vehicle", []), "in_vehicle", key),
            missed if mode == "frames" else None,
            _box_list(entry.get("unsure", []), "unsure", key),
        )
    return validate_all(judgements, items, mode)


def _attribute_entry(entry: object, key: str) -> str | None:
    """One crop's entry of an attribute judgements file: "x", or an object with exactly
    the ATTRIBUTE_NAMES, each "y", "n" or "u". Returns the letters, or None for "x"."""
    if isinstance(entry, str) and entry == REJECT:
        return None
    if not isinstance(entry, dict) or set(entry) != set(ATTRIBUTE_NAMES):
        raise JudgementError(
            f'crop {key}: the entry must be "{REJECT}" or an object with exactly '
            + ", ".join(ATTRIBUTE_NAMES)
        )
    letters = []
    for name in ATTRIBUTE_NAMES:
        value = entry[name]
        if not isinstance(value, str) or value not in ATTRIBUTE_ANSWERS:
            raise JudgementError(f"crop {key}: {name} must be y, n or u")
        letters.append(value)
    return "".join(letters)


def parse_attribute_judgements(raw: bytes, items: Sequence[ReviewItem]) -> dict[int, str | None]:
    """Parse an attribute judgements file (format in spotchecks/README.md): one entry per
    crop, keyed by crop number. Returns each crop's letters in ATTRIBUTE_NAMES order (e.g.
    "ynu"), or None for a rejected crop; raises JudgementError naming the entry."""
    data = _load_judgements(raw)
    if not isinstance(data, dict):
        raise JudgementError('the file must hold one object, e.g. {"1": "x", "2": {...}}')
    numbers = {str(item.number): item.number for item in items}
    answers: dict[int, str | None] = {}
    for key, entry in data.items():
        if key not in numbers:
            raise JudgementError(f"there is no crop {key[:20]!r}")
        answers[numbers[key]] = _attribute_entry(entry, key)
    return validate_attributes(answers, items)


def validate_attributes(
    answers: Mapping[int, str | None], items: Sequence[ReviewItem]
) -> dict[int, str | None]:
    """`answers` checked to hold one entry per item: None (rejected), or one of
    ATTRIBUTE_ANSWERS per question. Raises JudgementError."""
    numbers = [item.number for item in items]
    extra = sorted(set(answers) - set(numbers), key=str)
    if extra:
        raise JudgementError(f"there is no crop {extra[0]!r}")
    checked: dict[int, str | None] = {}
    for number in numbers:
        if number not in answers:
            raise JudgementError(f"crop {number} has no answers")
        letters = answers[number]
        if letters is not None and not (
            isinstance(letters, str)
            and len(letters) == len(ATTRIBUTE_QUESTIONS)
            and all(letter in ATTRIBUTE_ANSWERS for letter in letters)
        ):
            raise JudgementError(f"crop {number}: the answers must be y, n or u, one per question")
        checked[number] = letters
    return checked


def attribute_state(presses: Sequence[str]) -> tuple[list[str | None], str]:
    """Replay the attribute keys pressed so far (y, n, u or x; Backspace removes the last
    one): the finished crops' answers (None for a rejected one), in order, and the current
    crop's answers so far."""
    finished: list[str | None] = []
    current = ""
    for press in presses:
        if press == REJECT:
            finished.append(None)
            current = ""
            continue
        current += press
        if len(current) == len(ATTRIBUTE_QUESTIONS):
            finished.append(current)
            current = ""
    return finished, current


LINE_TOKEN = re.compile(r"([nvum])(\d{1,6})?")


def parse_line(line: str, item: ReviewItem, mode: Mode) -> Judgement:
    """Parse one keyboard answer for `item`; raise JudgementError.

    Tokens, separated by spaces or commas: `n<box>` not a person, `v<box>` a person inside
    a vehicle, `u<box>` cannot tell, `m<count>` people missed (frames mode). In crops mode
    a bare `n`, `v` or `u` means the crop's box. An empty line: every box is a pedestrian
    and nobody is missed.
    """
    lists: dict[str, list[int]] = {"n": [], "v": [], "u": []}
    missed: int | None = None
    for token in line.replace(",", " ").lower().split():
        match = LINE_TOKEN.fullmatch(token)
        if match is None:
            raise JudgementError(
                f"cannot read {token[:20]!r}; use n<box>, v<box>, u<box> or m<count>"
            )
        kind, digits = match.group(1), match.group(2)
        if kind == "m":
            if mode != "frames":
                raise JudgementError("missed people are counted in frames mode only")
            if digits is None or missed is not None:
                raise JudgementError("give one m<count>")
            missed = int(digits)
            continue
        if digits is None:
            if mode != "crops":
                raise JudgementError(f"say which box: {kind}<box>")
            box = item.boxes[0]
        else:
            box = int(digits)
        target = lists[kind]
        if box in target:
            raise JudgementError(f"box {box} is listed twice")
        target.append(box)
    if mode == "frames" and missed is None:
        missed = 0
    judgement = Judgement(
        frozenset(lists["n"]), frozenset(lists["v"]), missed, frozenset(lists["u"])
    )
    return validate(judgement, item, mode)


def _numbers(boxes: Sequence[int]) -> str:
    return ", ".join(str(b) for b in boxes) if boxes else "none"


def _describe(item: ReviewItem) -> str:
    if len(item.boxes) == 1:
        return f"box {item.boxes[0]}"
    return f"boxes {_numbers(item.boxes)}" if item.boxes else "no boxes"


class _LineReader:
    """Lines from a file descriptor, waiting no later than a deadline."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.buffer = b""
        self.eof = False
        self.too_long = False
        self._kind: str | None = None  # Windows: "pipe", "file" or "console"
        self._chunks: queue.Queue[bytes] | None = None  # Windows consoles: filled by a thread

    def readline(self, deadline: float) -> str:
        while b"\n" not in self.buffer:
            if self.eof:
                if not self.buffer:
                    raise ReviewAborted("the input ended before every image was judged")
                self.buffer += b"\n"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReviewTimeout("the review timed out")
            chunk = self._read_chunk(remaining)
            if chunk is None:
                continue
            if not chunk:
                self.eof = True
                continue
            self.buffer += chunk
            if b"\n" not in self.buffer and len(self.buffer) > MAX_LINE_BYTES:
                self.buffer, self.too_long = b"", True
        line, _, self.buffer = self.buffer.partition(b"\n")
        if self.too_long:
            self.too_long = False
            raise JudgementError(f"the line is longer than {MAX_LINE_BYTES} bytes")
        return line.decode("utf-8", errors="replace").strip()

    def _read_chunk(self, remaining: float) -> bytes | None:
        """Up to 4096 bytes (b"" at end of input), or None if none came in `remaining`."""
        if not WINDOWS:
            ready, _, _ = select.select([self.fd], [], [], remaining)
            return os.read(self.fd, 4096) if ready else None
        # Windows cannot select() on a pipe or a console. A pipe is polled for waiting
        # bytes, so no read ever blocks (a blocked read would hold the descriptor, and a
        # close of it would wait for that read). A console, which the tool never closes,
        # is read by a thread. Waits are cut into short slices, so that signals are
        # handled between them.
        if self._kind is None:
            self._kind = _input_kind(self.fd)
        if self._kind == "file":
            return os.read(self.fd, 4096)
        if self._kind == "pipe":
            end = time.monotonic() + min(remaining, 0.1)
            while True:
                waiting = _pipe_waiting(self.fd)
                if waiting is None:
                    return b""  # the writer closed it
                if waiting:
                    return os.read(self.fd, min(waiting, 4096))
                if time.monotonic() >= end:
                    return None
                time.sleep(0.02)
        if self._chunks is None:
            self._chunks = queue.Queue()
            threading.Thread(target=self._pump, args=(self._chunks,), daemon=True).start()
        try:
            return self._chunks.get(timeout=min(remaining, 0.1))
        except queue.Empty:
            return None

    def _pump(self, chunks: queue.Queue[bytes]) -> None:
        while True:
            try:
                chunk = os.read(self.fd, 4096)
            except OSError:
                chunk = b""
            chunks.put(chunk)
            if not chunk:
                return


def _input_kind(fd: int) -> str:
    """Windows: what the descriptor reads from ("pipe", "file" or "console")."""
    if sys.platform == "win32":
        kind = _kernel32.GetFileType(msvcrt.get_osfhandle(fd))
        return {1: "file", 3: "pipe"}.get(kind, "console")  # FILE_TYPE_DISK, FILE_TYPE_PIPE
    return "file"


def _pipe_waiting(fd: int) -> int | None:
    """Windows: bytes waiting in the pipe, or None once the writer has closed it."""
    if sys.platform == "win32":
        waiting = wintypes.DWORD()
        handle = msvcrt.get_osfhandle(fd)
        if not _kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(waiting), None):
            return None  # ERROR_BROKEN_PIPE, or the pipe is unusable: its end either way
        return int(waiting.value)
    return None


class KeyboardReviewer:
    """Asks for one line per image on a terminal (or any file descriptor)."""

    def __init__(self, fd: int = 0, out: TextIO | None = None) -> None:
        self.fd = fd
        self.out = out

    def judge(
        self, items: Sequence[ReviewItem], mode: Mode, deadline: float
    ) -> Mapping[int, Judgement]:
        out = self.out or sys.stdout
        reader = _LineReader(self.fd)
        hint = "n<box> not a person, v<box> person in a vehicle, u<box> cannot tell"
        hint += ", m<count> people missed" if mode == "frames" else " (bare n, v or u: this crop)"
        print(f"One line per image: {hint}; empty line: all correct; q: stop.", file=out)
        judgements: dict[int, Judgement] = {}
        for item in items:
            while item.number not in judgements:
                print(f"{item.file} ({_describe(item)})> ", end="", file=out, flush=True)
                try:
                    line = reader.readline(deadline)
                    if line.lower() == "q":
                        raise ReviewAborted("the reviewer stopped the review")
                    judgements[item.number] = parse_line(line, item, mode)
                except JudgementError as exc:
                    print(f"rejected: {exc}", file=out, flush=True)
        return judgements


class JsonFileReviewer:
    """Waits for the reviewer to write a judgements file, re-reading it when it changes."""

    def __init__(self, path: Path, out: TextIO | None = None) -> None:
        self.path = path
        self.out = out

    def judge(
        self, items: Sequence[ReviewItem], mode: Mode, deadline: float
    ) -> Mapping[int, Judgement]:
        return self._poll(lambda raw: parse_judgements(raw, items, mode), deadline)

    def attributes(self, items: Sequence[ReviewItem], deadline: float) -> Mapping[int, str | None]:
        """The attribute answers for `items` (see parse_attribute_judgements)."""
        return self._poll(lambda raw: parse_attribute_judgements(raw, items), deadline)

    def _poll[T](self, parse: Callable[[bytes], T], deadline: float) -> T:
        """Wait for the file, and return what `parse` makes of it once it accepts it."""
        out = self.out or sys.stdout
        print(
            f"Waiting for judgements in {self.path} (format: spotchecks/README.md).",
            file=out,
            flush=True,
        )
        # The content (or, for an unreadable path, the problem) last rejected: said once.
        rejected: bytes | str | None = None
        while True:
            seen: bytes | str | None = None
            try:
                seen = self._read()
                if seen is not None and seen != rejected:
                    return parse(seen)
            except JudgementError as exc:
                seen = str(exc) if seen is None else seen
                if seen != rejected:
                    rejected = seen
                    print(
                        f"judgements rejected: {exc}. Fix the file and save it again.",
                        file=out,
                        flush=True,
                    )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReviewTimeout("the review timed out")
            time.sleep(min(JSON_POLL_S, remaining))

    def _read(self) -> bytes | None:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise JudgementError(f"cannot read the file: {exc.strerror}") from None
        if not stat.S_ISREG(st.st_mode):
            raise JudgementError("the judgements path is not a regular file")
        try:
            with open(self.path, "rb") as fh:
                return fh.read(MAX_JUDGEMENTS_BYTES + 1)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise JudgementError(f"cannot read the file: {exc.strerror}") from None


def window_scale(height: int, width: int) -> int:
    """The whole factor the window enlarges a `height` x `width` crop by: up to about
    WINDOW_TARGET_HEIGHT tall, at most WINDOW_MAX_WIDTH wide and MAX_WINDOW_SCALE times."""
    by_height = WINDOW_TARGET_HEIGHT // max(1, height)
    by_width = WINDOW_MAX_WIDTH // max(1, width)
    return max(1, min(MAX_WINDOW_SCALE, by_height, by_width))


def display_size(height: int, width: int, max_height: int, max_width: int) -> tuple[int, int]:
    """The (height, width) at which the window shows an Austin crop of `height` x `width`
    on a screen with `max_height` x `max_width` pixels free: enlarged by a whole factor,
    never beyond window_scale's and never beyond what fits; a crop that does not fit at
    its own size is reduced, keeping its proportions, until it does."""
    if min(height, width, max_height, max_width) < 1:
        raise ValueError("sizes must be at least one pixel")
    fit = min(Fraction(max_height, height), Fraction(max_width, width))
    scale = Fraction(min(window_scale(height, width), math.floor(fit))) if fit >= 1 else fit
    return max(1, math.floor(height * scale)), max(1, math.floor(width * scale))


def display_image(image: Frame, max_height: int, max_width: int) -> Frame:
    """`image` at its display_size: enlarged pixel by pixel, or reduced by area."""
    height, width = image.shape[:2]
    new_height, new_width = display_size(height, width, max_height, max_width)
    if (new_height, new_width) == (height, width):
        return image
    interpolation = cv2.INTER_NEAREST if new_height > height else cv2.INTER_AREA
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return np.asarray(resized, dtype=np.uint8)


def _png_data(image: Frame, scale: int) -> str:
    """`image` enlarged `scale` times and encoded as PNG in memory, in base64 for Tk."""
    if scale > 1:
        height, width = image.shape[:2]
        size = (width * scale, height * scale)
        image = np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_NEAREST), np.uint8)
    return _png_base64(image)


def _png_base64(image: Frame) -> str:
    """`image` encoded as PNG in memory, in base64 for Tk."""
    ok, encoded = cv2.imencode(IMAGE_SUFFIX, image)
    if not ok:
        raise SpotcheckError("cannot encode a review image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _import_tkinter() -> ModuleType:
    try:
        import tkinter
    except ImportError:
        raise SpotcheckError(
            "--view window needs tkinter, which this Python lacks; use --view files"
        ) from None
    return tkinter


class WindowReviewer:
    """Shows each crop in one tkinter window, straight from memory, and records the keys
    pressed. A detection check (`judge`) takes one key per crop: Enter or Space a
    pedestrian, `n` not a person, `v` a person in a vehicle, `u` cannot tell, Backspace back
    one crop; crops mode only. An attribute session (`attributes`) asks
    ATTRIBUTE_QUESTIONS about each crop, one at a time: `y`, `n` or `u` answers the
    question on screen, `x` rejects the crop, Backspace goes back one answer. In both, `q`
    or closing the window stop. With `confirm_stop`, a first `q` only asks, in the header
    line: a second `q` stops, and any other key dismisses the question and is ignored.

    Nothing is written: each crop is encoded to PNG in memory and given to a PhotoImage
    as data. `driver`, if given, is called with the window once it shows the first crop
    (tests use it to send key events). With `guard`, the window's teardown is a critical
    step: a signal that lands during it waits until it is done. With `fit_screen` (Austin
    sessions), each crop is shown at its display_size for the screen instead of
    window_scale's whole factor."""

    def __init__(
        self,
        driver: Callable[[tkinter.Tk], object] | None = None,
        guard: _SignalGuard | None = None,
        confirm_stop: bool = False,
        fit_screen: bool = False,
    ) -> None:
        self.driver = driver
        self.guard = guard
        self.confirm_stop = confirm_stop
        self.fit_screen = fit_screen

    def judge(
        self, items: Sequence[ReviewItem], mode: Mode, deadline: float
    ) -> Mapping[int, Judgement]:
        if mode != "crops":
            raise SpotcheckError(WINDOW_FRAMES_REFUSAL)
        if not items:
            return {}  # nothing to show: no window, and no need for tkinter
        return self._in_window(_ReviewWindow, items, deadline)

    def attributes(self, items: Sequence[ReviewItem], deadline: float) -> Mapping[int, str | None]:
        """Each crop's answers in ATTRIBUTE_QUESTIONS order (e.g. "ynu"), or None for a
        rejected crop, keyed by crop number."""
        if not items:
            return {}
        return self._in_window(_AttributeWindow, items, deadline)

    def _in_window[R](
        self, kind: type[_Window[R]], items: Sequence[ReviewItem], deadline: float
    ) -> R:
        tk = _import_tkinter()
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            raise SpotcheckError(f"cannot open the review window: {exc}") from None
        window: _Window[R] | None = None
        try:
            window = kind(tk, root, items, deadline, self.confirm_stop)
            window.fit_screen = self.fit_screen
            return window.run(self.driver)
        finally:
            # Critical: a signal raised part-way would skip the rest and leave a cycle.
            with self.guard.critical() if self.guard is not None else contextlib.nullcontext():
                if window is not None:
                    window.close()  # deletes the image now: Tk calls stay in this thread
                with contextlib.suppress(AttributeError):
                    del root.report_callback_exception  # the root's reference to the window
                # Cancel what is still scheduled, or Tcl runs it after the window is gone.
                # One suppression per callback: one that ran meanwhile must not skip the
                # rest, nor the destroy.
                pending: tuple[str, ...] = ()
                with contextlib.suppress(tk.TclError):
                    pending = root.tk.splitlist(root.tk.call("after", "info"))
                for scheduled in pending:
                    with contextlib.suppress(tk.TclError):
                        root.after_cancel(scheduled)
                with contextlib.suppress(tk.TclError):
                    root.destroy()
                # Tcl objects must be freed in this thread. Left in a reference cycle, they
                # would be freed by whichever thread next runs the garbage collector (a
                # fetch thread, say), and Tcl aborts the process when that is not this one.
                del window, root
                gc.collect()


class _Window[R]:
    """One review in one window: what both kinds share. A subclass binds its answer keys,
    shows the current crop, and says what the review returns."""

    legend = WINDOW_LEGEND
    fit_screen = False  # show each crop at its display_size for the screen (Austin)

    def __init__(
        self,
        tk: ModuleType,
        root: tkinter.Tk,
        items: Sequence[ReviewItem],
        deadline: float,
        confirm_stop: bool = False,
    ) -> None:
        self.tk, self.root, self.items, self.deadline = tk, root, list(items), deadline
        self.confirm_stop = confirm_stop
        self.asking: str | None = None  # while the stop question shows: the header it hid
        self.outcome: BaseException | None = None
        self.done = False
        self.photo: tkinter.PhotoImage | None = None
        root.title(WINDOW_TITLE)
        root.protocol("WM_DELETE_WINDOW", self._stop)
        # An exception in a callback (a signal's, too) ends the review instead of being
        # printed and forgotten by tkinter.
        root.report_callback_exception = self._callback_failed
        self.header = tk.Label(root, font=("TkDefaultFont", 14))
        self.header.pack(padx=8, pady=(8, 4))
        self.picture = tk.Label(root)
        self.picture.pack(padx=8)
        self.legend_label = tk.Label(root, text=self.legend)
        self.legend_label.pack(padx=8, pady=(4, 8))
        self._bind_answers()
        self._bind_key("BackSpace", self._back)
        for key in ("q", "Q"):
            root.bind(f"<KeyPress-{key}>", lambda _event: self._q())
        if confirm_stop:
            # Keys without a binding of their own: they only dismiss the stop question.
            root.bind("<KeyPress>", self._other_key)

    def _bind_answers(self) -> None:
        raise NotImplementedError

    def _bind_key(self, key: str, action: Callable[[], None]) -> None:
        """Bind `key` to `action`, unless the stop question shows: then the key only
        dismisses it."""

        def pressed(_event: object) -> None:
            if self.asking is not None:
                self._dismiss()
            else:
                action()

        self.root.bind(f"<KeyPress-{key}>", pressed)

    def _q(self) -> None:
        if not self.confirm_stop or self.asking is not None:
            self._stop()
        elif not self.done and self.outcome is None:
            self.asking = str(self.header.cget("text"))
            self.header.configure(text=CONFIRM_STOP_QUESTION)

    def _dismiss(self) -> None:
        if self.asking is not None:
            self.header.configure(text=self.asking)
            self.asking = None

    def _other_key(self, event: tkinter.Event[tkinter.Misc]) -> None:
        if event.keysym not in MODIFIER_KEYS:
            self._dismiss()

    def _show(self) -> None:
        raise NotImplementedError

    def _back(self) -> None:
        raise NotImplementedError

    def _result(self) -> R:
        raise NotImplementedError

    def run(self, driver: Callable[[tkinter.Tk], object] | None) -> R:
        remaining_ms = math.ceil(max(0.0, self.deadline - time.monotonic()) * 1000)
        self.root.after(remaining_ms, self._timeout)
        self.root.after(WINDOW_TICK_MS, self._tick)
        self._show()
        self.root.lift()
        self.root.focus_force()
        if driver is not None:
            driver(self.root)
        self.root.mainloop()  # a signal's exception can also end it, and propagates
        if self.outcome is not None:
            raise self.outcome
        if not self.done:
            raise ReviewAborted("the review window was closed")
        return self._result()

    def _picture(self, item: ReviewItem) -> None:
        height, width = item.image.shape[:2]
        if self.fit_screen:
            max_height = max(1, self.root.winfo_screenheight() - WINDOW_CHROME_PX)
            max_width = max(1, self.root.winfo_screenwidth() - WINDOW_SCREEN_MARGIN_PX)
            data = _png_base64(display_image(item.image, max_height, max_width))
        else:
            data = _png_data(item.image, window_scale(height, width))
        self.photo = self.tk.PhotoImage(master=self.root, data=data, format="png")
        self.picture.configure(image=self.photo)

    def _finish(self, outcome: BaseException) -> None:
        if not self.done and self.outcome is None:
            self.outcome = outcome
        self.root.quit()

    def _stop(self) -> None:
        self._finish(ReviewAborted("the reviewer stopped the review"))

    def _timeout(self) -> None:
        self._finish(ReviewTimeout("the review timed out"))

    def _tick(self) -> None:
        if time.monotonic() >= self.deadline + WINDOW_DEADLINE_GRACE_S:
            self._timeout()  # the timeout callback never ran: end the review regardless
            return
        self.root.after(WINDOW_TICK_MS, self._tick)

    def _callback_failed(self, kind: object, value: BaseException, tb: object) -> None:
        self._finish(value)

    def close(self) -> None:
        """Drop the image, and the outcome (whose traceback can lead back here), so that
        no reference cycle keeps a Tcl object alive once the review is over."""
        self.photo = None
        self.outcome = None


class _ReviewWindow(_Window[dict[int, Judgement]]):
    """A detection check: one key per crop."""

    def __init__(
        self,
        tk: ModuleType,
        root: tkinter.Tk,
        items: Sequence[ReviewItem],
        deadline: float,
        confirm_stop: bool = False,
    ) -> None:
        self.index = 0
        self.judgements: dict[int, Judgement] = {}
        super().__init__(tk, root, items, deadline, confirm_stop)

    def _bind_answers(self) -> None:
        answers = {"Return": "", "KP_Enter": "", "space": "", "n": "n", "N": "n"}
        answers |= {"v": "v", "V": "v", "u": "u", "U": "u"}
        for key, line in answers.items():
            self._bind_key(key, functools.partial(self._answer, line))

    def _result(self) -> dict[int, Judgement]:
        return self.judgements

    def _show(self) -> None:
        item = self.items[self.index]
        self._picture(item)
        self.header.configure(
            text=f"Image {item.number} ({_describe(item)}): {self.index + 1} of {len(self.items)}"
        )

    def _answer(self, line: str) -> None:
        if self.done or self.outcome is not None:
            return
        item = self.items[self.index]
        self.judgements[item.number] = parse_line(line, item, "crops")
        self.index += 1
        if self.index == len(self.items):
            self.done = True
            self.root.quit()
        else:
            self._show()

    def _back(self) -> None:
        if self.done or self.outcome is not None or self.index == 0:
            return
        self.index -= 1
        self.judgements.pop(self.items[self.index].number, None)
        self._show()


class _AttributeWindow(_Window[dict[int, str | None]]):
    """An attribute session: one key per question, the question on screen above the crop.
    The keys pressed are kept in order, and the answers are replayed from them
    (`attribute_state`), so Backspace can undo any answer, an `x` included."""

    legend = ATTRIBUTE_LEGEND

    def __init__(
        self,
        tk: ModuleType,
        root: tkinter.Tk,
        items: Sequence[ReviewItem],
        deadline: float,
        confirm_stop: bool = False,
    ) -> None:
        self.presses: list[str] = []
        self.shown: int | None = None  # the crop whose image is on screen
        super().__init__(tk, root, items, deadline, confirm_stop)
        self.question = tk.Label(root, font=("TkDefaultFont", 16))
        self.question.pack(padx=8, pady=(0, 4), before=self.picture)

    def _bind_answers(self) -> None:
        for answer in (*ATTRIBUTE_ANSWERS, REJECT):
            for key in (answer, answer.upper()):
                self._bind_key(key, functools.partial(self._press, answer))

    def _result(self) -> dict[int, str | None]:
        finished, _current = attribute_state(self.presses)
        return {item.number: answers for item, answers in zip(self.items, finished, strict=True)}

    def _show(self) -> None:
        finished, current = attribute_state(self.presses)
        index = len(finished)
        item = self.items[index]
        if self.shown != index:
            self._picture(item)
            self.shown = index
        self.header.configure(text=f"Crop {item.number}: {index + 1} of {len(self.items)}")
        _name, text = ATTRIBUTE_QUESTIONS[len(current)]
        self.question.configure(
            text=f"Question {len(current) + 1} of {len(ATTRIBUTE_QUESTIONS)}: {text}"
        )

    def _press(self, answer: str) -> None:
        if self.done or self.outcome is not None:
            return
        self.presses.append(answer)
        finished, _current = attribute_state(self.presses)
        if len(finished) == len(self.items):
            self.done = True
            self.root.quit()
        else:
            self._show()

    def _back(self) -> None:
        if self.done or self.outcome is not None or not self.presses:
            return
        self.presses.pop()
        self._show()


# The paired judge ---------------------------------------------------------------------

# The reviewer's label for a crop sent to the judge. Crops the reviewer cannot tell
# ("unsure") are not sent.
REVIEWER_LABELS = ("person", "in_vehicle", "not_person")
JUDGE_POSITIVE: tuple[judge_hosted.Answer, ...] = ("person", "in_vehicle")


@dataclass(frozen=True, slots=True)
class JudgeSetup:
    """Where the judge's requests go and how long each may take (tests: a loopback fake
    server, a short timeout and no wait between retries)."""

    endpoint: str | None = None
    timeout: float = judge_hosted.REQUEST_TIMEOUT
    sleep: Callable[[float], None] = time.sleep


def _is_loopback(endpoint: str | None) -> bool:
    if endpoint is None:
        return False
    try:
        return urllib.parse.urlsplit(endpoint).hostname in judge_hosted.LOOPBACK
    except ValueError:
        return False


def _judge_wanted(args: argparse.Namespace, mode: Mode, setup: JudgeSetup) -> bool:
    """Whether `--judge` asks for a judge; raises SpotcheckError when the options do not
    allow one."""
    if args.judge is None:
        if args.judge_max_requests is not None:
            raise SpotcheckError("--judge-max-requests needs --judge")
        return False
    if mode != "crops":
        raise SpotcheckError(
            "--judge works in crops mode only: frames mode would send whole frames"
        )
    if args.judge_max_requests is None:
        raise SpotcheckError("--judge needs --judge-max-requests N, the most requests to make")
    if args.dry_run and not _is_loopback(setup.endpoint):
        raise SpotcheckError(
            "--judge with --dry-run needs a fake judge on this machine: a dry run never "
            "calls DeepInfra"
        )
    return True


def open_judge(
    args: argparse.Namespace, mode: Mode, setup: JudgeSetup
) -> judge_hosted.LiveCropJudge | None:
    """The judge that `--judge` names, checked before anything else happens, or None.
    Raises SpotcheckError when the options do not allow one. Sends nothing."""
    if not _judge_wanted(args, mode, setup):
        return None
    try:
        return judge_hosted.LiveCropJudge(
            args.judge,
            budget=judge_hosted.RequestBudget(args.judge_max_requests),
            endpoint=setup.endpoint,
            timeout=setup.timeout,
            sleep=setup.sleep,
        )
    except judge_hosted.JudgeError as exc:
        raise SpotcheckError(f"cannot use the judge: {exc}") from None


def ask_judge(
    crops: Sequence[Frame], judge: judge_hosted.LiveCropJudge
) -> list[judge_hosted.Answer]:
    """The judge's answer on each crop, in order, until the first failure. It is given the
    pixels only, never the reviewer's answers. A failure is reported without any detail of
    a crop, and ends the judging; a signal does too. Closes the judge."""
    answers: list[judge_hosted.Answer] = []
    try:
        for image in crops:
            answers.append(judge.classify_live_crop(image))
    except judge_hosted.JudgeError as exc:  # the request limit, HTTP errors, timeouts
        _judge_stopped(str(exc))
    except Interrupted as exc:
        _judge_stopped(f"stopped by {_signal_name(exc.signum)}")
    except Exception as exc:  # any other failure must not lose the reviewer's statistics
        _judge_stopped(f"failed ({type(exc).__name__})")
    finally:
        judge.close()
    return answers


def _judge_stopped(reason: str, what: str = "its statistics are") -> None:
    print(
        f"spotcheck: the judge stopped: {reason}; {what} incomplete",
        file=sys.stderr,
        flush=True,
    )


def open_attribute_judge(
    args: argparse.Namespace, setup: JudgeSetup
) -> judge_hosted.LiveAttributeJudge | None:
    """The attribute judge that `--judge` names, checked as open_judge checks it, or None.
    Sends nothing."""
    if not _judge_wanted(args, "crops", setup):
        return None
    try:
        return judge_hosted.LiveAttributeJudge(
            args.judge,
            budget=judge_hosted.RequestBudget(args.judge_max_requests),
            endpoint=setup.endpoint,
            timeout=setup.timeout,
            sleep=setup.sleep,
        )
    except judge_hosted.JudgeError as exc:
        raise SpotcheckError(f"cannot use the judge: {exc}") from None


def ask_attribute_judge(
    crops: Sequence[Frame], judge: judge_hosted.LiveAttributeJudge
) -> list[str]:
    """The judge's answers on each crop, in order, until the first failure, as ask_judge
    asks: the pixels only, never the reviewer's answers; a failure or a signal ends the
    judging without losing them. Closes the judge."""
    answers: list[str] = []
    what = "its answers are"
    try:
        for image in crops:
            answers.append(judge.classify_attributes(image))
    except judge_hosted.JudgeError as exc:  # the request limit, HTTP errors, timeouts
        _judge_stopped(str(exc), what)
    except Interrupted as exc:
        _judge_stopped(f"stopped by {_signal_name(exc.signum)}", what)
    except Exception as exc:  # any other failure must not lose the reviewer's answers
        _judge_stopped(f"failed ({type(exc).__name__})", what)
    finally:
        judge.close()
    return answers


def box_label(box: int, judgement: Judgement) -> str:
    """The reviewer's label for `box`: person, in_vehicle, not_person or unsure."""
    if box in judgement.not_person:
        return "not_person"
    if box in judgement.in_vehicle:
        return "in_vehicle"
    return "unsure" if box in judgement.unsure else "person"


def _reviewer_label(item: ReviewItem, judgement: Judgement) -> str:
    (box,) = item.boxes  # crops mode: one box per image
    return box_label(box, judgement)


def judged_items(
    items: Sequence[ReviewItem], judgements: Mapping[int, Judgement]
) -> list[ReviewItem]:
    """The crops the reviewer did not mark unsure: the only ones the judge is shown."""
    return [item for item in items if not set(item.boxes) & judgements[item.number].unsure]


def judge_stats(
    items: Sequence[ReviewItem],
    judgements: Mapping[int, Judgement],
    answers: Sequence[judge_hosted.Answer],
    judge: judge_hosted.LiveCropJudge,
    name: str,
) -> dict[str, object]:
    """The `judge` block: the reviewer x judge confusion counts over the crops the judge
    answered (in order, from the first), and the judge's precision computed as the
    reviewer's is, on its confident answers (person, in_vehicle or not_person)."""
    confusion: dict[str, dict[judge_hosted.Answer, int]] = {
        label: dict.fromkeys(judge_hosted.ANSWERS, 0) for label in REVIEWER_LABELS
    }
    for item, answer in zip(items[: len(answers)], answers, strict=True):
        confusion[_reviewer_label(item, judgements[item.number])][answer] += 1
    positive = sum(row[a] for row in confusion.values() for a in JUDGE_POSITIVE)
    confident = positive + sum(row["not_person"] for row in confusion.values())
    usage = judge.usage
    cost = judge_hosted.cost_usd(judge.candidate, usage.input_tokens, usage.output_tokens)
    return {
        "model": name,
        "provider": "DeepInfra",
        "status": "complete" if len(answers) == len(items) else "incomplete",
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(cost, 8),
        "confusion": confusion,
        "judge_precision": _ratio(positive, confident),
    }


# Statistics ---------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def compute_stats(
    items: Sequence[ReviewItem],
    judgements: Mapping[int, Judgement],
    *,
    mode: Mode,
    frames_reviewed: int,
    reviewer: str,
    info: DetectorInfo,
    day: datetime.date,
) -> dict[str, object]:
    """The statistics record: counts, and the ratios derived from them (None on 0/0).
    Boxes the reviewer cannot tell are left out: they are not counted as shown."""
    unsure = sum(len(judgements[item.number].unsure) for item in items)
    shown = sum(len(item.boxes) for item in items) - unsure
    not_person = sum(len(judgements[item.number].not_person) for item in items)
    in_vehicle = sum(len(judgements[item.number].in_vehicle) for item in items)
    missed: int | None = None
    recall: float | None = None
    if mode == "frames":
        missed = sum(judgements[item.number].missed or 0 for item in items)
        recall = _ratio(shown - not_person, shown - not_person + missed)
    person = _ratio(shown - not_person, shown)
    pedestrian = _ratio(shown - not_person - in_vehicle, shown)
    return {
        "date": day.isoformat(),
        "reviewer": reviewer,
        "mode": mode,
        "frames_reviewed": frames_reviewed,
        "boxes_shown": shown,
        "boxes_not_person": not_person,
        "boxes_in_vehicle": in_vehicle,
        "persons_missed": missed,
        "precision_person": person,
        "precision_pedestrian": pedestrian,
        "recall_estimate": recall,
        "detector": dataclasses.asdict(info),
    }


def box_height(box: tuple[float, float, float, float]) -> int:
    """A box's height in source-frame pixels, rounded to a whole number."""
    return round(box[3] - box[1])


def box_heights(samples: Sequence[Sample]) -> dict[int, int]:
    """Each box's height in source-frame pixels, rounded, by box number (numbered as
    `render` numbers them: 1, 2, ... across the whole check)."""
    boxes = (person.box for s in samples for person in s.persons)
    return {number: box_height(box) for number, box in enumerate(boxes, start=1)}


def box_record(
    items: Sequence[ReviewItem],
    judgements: Mapping[int, Judgement],
    heights: Mapping[int, int],
    *,
    frames_reviewed: int,
    info: DetectorInfo,
    day: datetime.date,
    started_at: datetime.datetime,
) -> dict[str, object]:
    """The per-box record: each box's height and label, sorted so that the order says
    nothing about frames, and no position, width, camera id, frame index or image."""
    minute = started_at.astimezone(datetime.UTC).replace(second=0, microsecond=0)
    boxes = sorted(
        (heights[box], box_label(box, judgements[item.number]))
        for item in items
        for box in item.boxes
    )
    return {
        "date": day.isoformat(),
        "started_at": minute.strftime("%Y-%m-%dT%H:%MZ"),
        "light": light_at(minute),
        "frames": frames_reviewed,
        "detector": dataclasses.asdict(info),
        "boxes": [[height, label] for height, label in boxes],
    }


def attribute_record(
    items: Sequence[ReviewItem],
    answers: Mapping[int, str | None],
    heights: Mapping[int, int],
    models: Sequence[str],
    *,
    frames_reviewed: int,
    info: DetectorInfo,
    day: datetime.date,
    started_at: datetime.datetime,
    judge: str | None,
    min_height: int = NEAR_FIELD_MIN_HEIGHT_PX,
) -> dict[str, object]:
    """The attribute record: for each crop the reviewer did not reject, its height, the
    reviewer's answers and the model's (`models`, one per such crop in order, from the
    first; None past its end), sorted so that the order says nothing about frames, and
    `min_height`, the smallest box height shown. No position, width, camera id, frame
    index, image or free text. The light is London's (see sourced_record)."""
    minute = started_at.astimezone(datetime.UTC).replace(second=0, microsecond=0)
    kept = [item for item in items if answers[item.number] is not None]
    crops = []
    for k, item in enumerate(kept):
        (box,) = item.boxes  # crops mode: one box per image
        crops.append((heights[box], answers[item.number], models[k] if k < len(models) else None))
    crops.sort(key=lambda crop: (crop[0], crop[1] or "", crop[2] or ""))
    return {
        "date": day.isoformat(),
        "started_at": minute.strftime("%Y-%m-%dT%H:%MZ"),
        "light": light_at(minute),
        "frames": frames_reviewed,
        "detector": dataclasses.asdict(info),
        "min_height_px": min_height,
        "judge": judge,
        "crops_shown": len(items),
        "crops_rejected": len(items) - len(kept),
        "crops": [list(crop) for crop in crops],
    }


def sourced_record(
    record: dict[str, object], source: Source, started_at: datetime.datetime
) -> dict[str, object]:
    """The attribute record of a `source` session: a London one as it is; an Austin (or
    Calgary) one with the light over Austin (Calgary) at the start of the sweep and
    `"source": "austin"` (`"calgary"`) last."""
    if source == "london":
        return record
    minute = started_at.astimezone(datetime.UTC).replace(second=0, microsecond=0)
    where = CALGARY if source == "calgary" else AUSTIN
    return {**record, "light": light_at(minute, where), "source": source}


# The light ----------------------------------------------------------------------------

Light = Literal["day", "twilight", "dark"]


def solar_elevation(
    moment: datetime.datetime, latitude: float = LONDON[0], longitude: float = LONDON[1]
) -> float:
    """The sun's elevation above the horizon, in degrees, at `moment` (an aware datetime)
    seen from `latitude`, `longitude`: the geometric position of its centre, without
    refraction. NOAA's general solar position formulae (after Meeus, "Astronomical
    Algorithms"), good to a minute of time or so at this latitude."""
    moment = moment.astimezone(datetime.UTC)
    j2000 = datetime.datetime(2000, 1, 1, 12, tzinfo=datetime.UTC)
    t = (moment - j2000).total_seconds() / 86400 / 36525  # Julian centuries from J2000.0
    mean_long = math.radians((280.46646 + t * (36000.76983 + t * 0.0003032)) % 360)
    mean_anomaly = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    centre = (
        math.sin(mean_anomaly) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * mean_anomaly) * (0.019993 - 0.000101 * t)
        + math.sin(3 * mean_anomaly) * 0.000289
    )
    omega = math.radians(125.04 - 1934.136 * t)
    apparent_long = math.radians(
        math.degrees(mean_long) + centre - 0.00569 - 0.00478 * math.sin(omega)
    )
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    obliquity = math.radians(23 + (26 + seconds / 60) / 60 + 0.00256 * math.cos(omega))
    declination = math.asin(math.sin(obliquity) * math.sin(apparent_long))
    y = math.tan(obliquity / 2) ** 2
    equation_of_time = 4 * math.degrees(  # minutes
        y * math.sin(2 * mean_long)
        - 2 * eccentricity * math.sin(mean_anomaly)
        + 4 * eccentricity * y * math.sin(mean_anomaly) * math.cos(2 * mean_long)
        - 0.5 * y * y * math.sin(4 * mean_long)
        - 1.25 * eccentricity * eccentricity * math.sin(2 * mean_anomaly)
    )
    midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes = (moment - midnight).total_seconds() / 60
    hour_angle = math.radians((minutes + equation_of_time + 4 * longitude) / 4 - 180)
    lat = math.radians(latitude)
    cos_zenith = math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(
        declination
    ) * math.cos(hour_angle)
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))


def light_category(elevation: float) -> Light:
    if elevation >= LIGHT_DAY_DEG:
        return "day"
    return "twilight" if elevation >= LIGHT_TWILIGHT_DEG else "dark"


def light_at(moment: datetime.datetime, where: tuple[float, float] = LONDON) -> Light:
    """The light at `moment` over `where` (latitude, longitude; central London unless
    given): day, (civil) twilight or dark."""
    return light_category(solar_elevation(moment, *where))


def write_stats(
    stats: Mapping[str, object],
    out_dir: Path,
    day: datetime.date,
    boxes: Mapping[str, object] | None = None,
) -> Path:
    """Write `stats` to a new `<day>.json` (or `<day>-2.json`, ...), and `boxes`, if
    given, to a new file of the same name in `<out-dir>/boxes/`; never overwrite. The
    name is the first one free in both places.

    Raises SpotcheckError, with the records in the message so they are not lost, if the
    files cannot be written; then neither is left behind.
    """
    texts = [(out_dir, json.dumps(stats, indent=2, ensure_ascii=True) + "\n")]
    if boxes is not None:
        texts.append((out_dir / BOXES_DIR, json.dumps(boxes, ensure_ascii=True) + "\n"))
    try:
        return out_dir / _write_new(texts, day)
    except OSError as exc:
        lost = f"they were: {json.dumps(stats, ensure_ascii=True)}"
        if boxes is not None:
            lost += f" and {json.dumps(boxes, ensure_ascii=True)}"
        raise SpotcheckError(
            f"cannot write statistics into {out_dir} ({exc.strerror}); {lost}"
        ) from None


def write_attributes(
    record: Mapping[str, object], out_dir: Path, day: datetime.date, source: Source = "london"
) -> Path:
    """Write `record` to a new `<out-dir>/attributes/<day>.json` (or `<day>-2.json`, ...),
    or for Austin `<day>-austin.json` (or `<day>-austin-2.json`, ...), for Calgary
    `<day>-calgary.json` (...); never overwrite. Raises SpotcheckError, with the record in
    the message so it is not lost, if it cannot be written."""
    directory = out_dir / ATTRIBUTES_DIR
    text = json.dumps(record, ensure_ascii=True) + "\n"
    suffix = {"austin": AUSTIN_FILE_SUFFIX, "calgary": CALGARY_FILE_SUFFIX}.get(source, "")
    stem = day.isoformat() + suffix
    try:
        return directory / _write_new([(directory, text)], day, stem)
    except OSError as exc:
        raise SpotcheckError(
            f"cannot write the attribute file into {directory} ({exc.strerror}); it was: "
            f"{json.dumps(record, ensure_ascii=True)}"
        ) from None


def _write_new(
    texts: Sequence[tuple[Path, str]], day: datetime.date, stem: str | None = None
) -> str:
    """Each text into its directory, under one new name, the first free in all of them:
    `<stem>.json`, then `<stem>-2.json`, ... (`stem` is the day unless given); returns the
    name."""
    stem = day.isoformat() if stem is None else stem
    for directory, _text in texts:
        directory.mkdir(parents=True, exist_ok=True)
    for k in range(1, MAX_FILES_PER_DAY + 1):
        name = f"{stem}.json" if k == 1 else f"{stem}-{k}.json"
        created: list[Path] = []
        try:
            for directory, text in texts:
                path = directory / name
                fh = open(path, "x", encoding="utf-8")  # noqa: SIM115  (closed below)
                created.append(path)
                with fh:
                    fh.write(text)
        except FileExistsError:
            for path in created:
                path.unlink(missing_ok=True)
            continue
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        return name
    where = " and ".join(str(directory) for directory, _text in texts)
    raise SpotcheckError(f"{where} already holds {MAX_FILES_PER_DAY} files for {day}")


# Command line -------------------------------------------------------------------------


def _count(low: int, high: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a whole number: {text[:20]!r}") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be from {low} to {high}")
        return value

    return parse


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text[:20]!r}") from None
    if not 0 < value <= MAX_TIMEOUT_S:  # also rejects NaN
        raise argparse.ArgumentTypeError(f"must be more than 0 and at most {MAX_TIMEOUT_S}")
    return value


def _reviewer_name(text: str) -> str:
    if not REVIEWER_NAME.fullmatch(text):
        raise argparse.ArgumentTypeError("use 1-64 letters, digits, spaces, '.', '_' or '-'")
    return text


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.tools.spotcheck",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n", type=_count(1, MAX_N), required=True, help="frames to sample")
    ap.add_argument("--mode", choices=MODES, default="crops")
    ap.add_argument(
        "--min-persons",
        type=_count(0, MAX_MIN_PERSONS),
        default=DEFAULT_MIN_PERSONS,
        help="person detections a frame needs to be sampled",
    )
    ap.add_argument("--seed", type=int, default=None, help="seed for the random sample")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="where the statistics go")
    ap.add_argument("--reviewer", type=_reviewer_name, default=DEFAULT_REVIEWER)
    ap.add_argument("--model", choices=sorted(detect.MODEL_SHA256), default=DEFAULT_MODEL)
    ap.add_argument(
        "--view",
        choices=VIEWS,
        default=None,
        help="files: images in a temporary directory; window: crops in a window, nothing "
        "written (default: window on Windows in crops mode without --judgements, else files)",
    )
    ap.add_argument(
        "--judgements",
        default=None,
        help="read judgements from this JSON file (must not exist yet) instead of the keyboard",
    )
    ap.add_argument(
        "--timeout",
        type=_seconds,
        default=float(DEFAULT_TIMEOUT_S),
        help="seconds allowed for the review (default 30 minutes)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="sweep a local fake camera server serving the fixture photos (no network)",
    )
    ap.add_argument(
        "--judge",
        choices=sorted(judge_hosted.DEEPINFRA),
        default=None,
        help="after the review, send the same crops to this DeepInfra model and add a "
        "reviewer x judge agreement block to the statistics (crops mode only)",
    )
    ap.add_argument(
        "--judge-max-requests",
        type=_count(1, MAX_JUDGE_REQUESTS),
        default=None,
        help="required with --judge: stop after N requests in all (retries count)",
    )
    ap.add_argument(
        "--record-boxes",
        action="store_true",
        help="also write each box's height and label to <out-dir>/boxes/ (crops mode only)",
    )
    ap.add_argument(
        "--attributes",
        action="store_true",
        help="label the outer layer, bare legs and umbrella of near-field crops instead, and "
        "write <out-dir>/attributes/ (with --view window or --judgements)",
    )
    ap.add_argument(
        "--confirm-stop",
        action="store_true",
        help="with --view window: a first q asks before it stops the review, and any other "
        "key continues it",
    )
    ap.add_argument(
        "--min-height",
        type=_count(NEAR_FIELD_MIN_HEIGHT_PX, MAX_ATTRIBUTE_MIN_HEIGHT_PX),
        default=None,
        metavar="N",
        help="with --attributes: show only person boxes at least N px tall in the source "
        f"frame ({NEAR_FIELD_MIN_HEIGHT_PX} to {MAX_ATTRIBUTE_MIN_HEIGHT_PX}; default "
        f"{NEAR_FIELD_MIN_HEIGHT_PX}, the near-field threshold)",
    )
    ap.add_argument(
        "--allow-dark",
        action="store_true",
        help="with --attributes: start the session even when it is dark in London (refused "
        "otherwise when the answers are taken in the window)",
    )
    ap.add_argument(
        "--source",
        choices=SOURCES,
        default="london",
        help="with --attributes: austin labels crops of the City of Austin's 1920x1080 "
        "traffic cameras instead of London's, calgary those of the City of Calgary's 840x630 "
        "ones (default london)",
    )
    ap.add_argument(
        "--bbox",
        type=pilot_heights.parse_bbox,
        default=None,
        metavar="S,W,N,E",
        help="with --source austin or calgary: the cameras inside this box, degrees "
        "(default: that city's downtown)",
    )
    return ap


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command line, checked. `--source austin` (or calgary) outside an attribute
    session and `--bbox` without either are usage errors: exit 2 with the usage, as
    argparse's own."""
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.source == "austin" and not args.attributes:
        ap.error(AUSTIN_ATTRIBUTES_ONLY)
    if args.source == "calgary" and not args.attributes:
        ap.error(CALGARY_ATTRIBUTES_ONLY)
    if args.bbox is not None and args.source == "london":
        ap.error(BBOX_AUSTIN_ONLY)
    return args


WINDOW_FRAMES_REFUSAL = "--view window shows crops only; frames mode needs --view files"
CONFIRM_STOP_REFUSAL = "--confirm-stop asks in the review window; it needs --view window"
RECORD_BOXES_FRAMES_REFUSAL = "--record-boxes works in crops mode only; drop it or --mode frames"
ATTRIBUTES_FRAMES_REFUSAL = "--attributes works in crops mode only; drop --mode frames"
ATTRIBUTES_RECORD_BOXES_REFUSAL = (
    "--attributes writes its own file and no per-box file; drop --record-boxes"
)
ALLOW_DARK_REFUSAL = "--allow-dark applies to attribute sessions only; add --attributes or drop it"
MIN_HEIGHT_REFUSAL = "--min-height applies to attribute sessions only; add --attributes or drop it"
DARK_REFUSAL = (
    "it is dark in London now (sun below -6°); attribute sessions need daylight. "
    "Use --allow-dark to run anyway."
)
AUSTIN_DARK_REFUSAL = (
    "it is dark in Austin now (sun below -6°); attribute sessions need daylight. "
    "Use --allow-dark to run anyway."
)
CALGARY_DARK_REFUSAL = (
    "it is dark in Calgary now (sun below -6°); attribute sessions need daylight. "
    "Use --allow-dark to run anyway."
)
AUSTIN_ATTRIBUTES_ONLY = "--source austin works in attribute sessions only; add --attributes"
CALGARY_ATTRIBUTES_ONLY = "--source calgary works in attribute sessions only; add --attributes"
BBOX_AUSTIN_ONLY = "--bbox selects Austin cameras; it needs --source austin"
ATTRIBUTES_KEYBOARD_REFUSAL = (
    "--attributes takes its answers in the window (--view window) or from a JSON file "
    "(--judgements PATH), not from the keyboard"
)


def default_view(platform: str = sys.platform) -> View:
    """The view used when none is given: the window on Windows, files elsewhere."""
    return "window" if platform == "win32" else "files"


def resolve_view(
    view: View | None, mode: Mode, judgements: str | None, platform: str = sys.platform
) -> View:
    """The view to use. Frames and a judgements file need the files view: without
    `--view`, they get it; with `--view window`, they are refused (SpotcheckError)."""
    if view is None:
        return "files" if mode == "frames" or judgements is not None else default_view(platform)
    if view == "window" and mode == "frames":
        raise SpotcheckError(WINDOW_FRAMES_REFUSAL)
    if view == "window" and judgements is not None:
        raise SpotcheckError("--view window takes its judgements from keys; drop --judgements")
    return view


def _ci_variables() -> list[str]:
    return [name for name in CI_VARIABLES if os.environ.get(name, "") != ""]


def _check_temp_dir(tmpdir: Path) -> None:
    """Refuse a temporary directory inside this repository or any git work tree, where a
    `git add` could commit the rendered images. That includes the current directory,
    which tempfile falls back to when no usual temporary directory is writable."""
    resolved = tmpdir.resolve()
    inside = resolved.is_relative_to(REPO_ROOT)
    inside = inside or any(os.path.lexists(d / ".git") for d in (resolved, *resolved.parents))
    if inside:
        raise SpotcheckError(
            f"the temporary directory {resolved} is inside a git work tree, where the "
            "rendered images could be committed; set TMPDIR to a directory outside it"
        )


def _is_real_spotchecks(out_dir: Path) -> bool:
    """True for the repository's spotchecks/ (or a directory in it), from wherever the
    tool runs, and for the default spelling relative to the current directory. Dry-run
    statistics describe the fixtures, and must never be mixed with real ones (INV-6)."""
    resolved = out_dir.resolve()
    return resolved.is_relative_to((REPO_ROOT / DEFAULT_OUT_DIR).resolve()) or (
        resolved == Path(DEFAULT_OUT_DIR).resolve()
    )


def _check_out_dir(out_dir: Path) -> None:
    existing = out_dir
    while not os.path.lexists(existing):
        if existing.parent == existing:
            break
        existing = existing.parent
    if not existing.is_dir() or not os.access(existing, os.W_OK | os.X_OK):
        raise SpotcheckError(f"cannot write statistics into {out_dir}")


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def _print_numbering(workdir: Path, items: Sequence[ReviewItem]) -> None:
    print(f"Review directory (deleted when this tool exits): {workdir}")
    for item in items:
        print(f"  {item.file}: {_describe(item)}")
    print(f"The numbering and a judgements template are in {NUMBERING_FILE} there.", flush=True)


def _collect(
    items: Sequence[ReviewItem],
    mode: Mode,
    reviewer: Reviewer,
    timeout_s: float,
    guard: _SignalGuard,
) -> dict[int, Judgement]:
    """Ask the reviewer, under the review timeout, and check what comes back."""
    deadline = time.monotonic() + timeout_s
    guard.alarm(timeout_s)
    try:
        judgements = reviewer.judge(items, mode, deadline)
    finally:
        guard.alarm(0)
    try:
        return validate_all(judgements, items, mode)
    except JudgementError as exc:
        raise SpotcheckError(f"the reviewer returned an invalid judgement: {exc}") from None


def _collect_attributes(
    items: Sequence[ReviewItem],
    reviewer: AttributeReviewer,
    timeout_s: float,
    guard: _SignalGuard,
) -> dict[int, str | None]:
    """Ask for the attribute answers, under the review timeout, and check them."""
    deadline = time.monotonic() + timeout_s
    guard.alarm(timeout_s)
    try:
        answers = reviewer.attributes(items, deadline)
    finally:
        guard.alarm(0)
    try:
        return validate_attributes(answers, items)
    except JudgementError as exc:
        raise SpotcheckError(f"the reviewer returned invalid answers: {exc}") from None


def _review(
    items: Sequence[ReviewItem],
    mode: Mode,
    reviewer: Reviewer,
    timeout_s: float,
    guard: _SignalGuard,
    view: View = "files",
) -> dict[int, Judgement]:
    """Render to the review directory, collect the judgements, delete the directory. The
    window view writes nothing: the reviewer is given the images in memory."""
    return _shown(
        items,
        mode,
        guard,
        view,
        WINDOW_LEGEND,
        None,
        lambda: _collect(items, mode, reviewer, timeout_s, guard),
    )


def _shown[T](
    items: Sequence[ReviewItem],
    mode: Mode,
    guard: _SignalGuard,
    view: View,
    legend: str,
    entry: Mapping[str, object] | None,
    collect: Callable[[], T],
) -> T:
    """Show `items` in the view, run `collect`, and delete what the view wrote. The files
    view renders them to the review directory, with `entry` as each image's template entry
    in the numbering; the window view writes nothing."""
    if view == "window":
        print(f"Review window open: {len(items)} crop(s). {legend}", flush=True)
        return collect()
    directory = ReviewDirectory()
    try:
        try:
            with guard.critical():
                workdir = directory.create()
            for item in items:
                directory.write_image(item)
            directory.write_numbering(items, mode, entry)
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            raise SpotcheckError(f"cannot write the review directory: {reason}") from None
        _print_numbering(workdir, items)
        return collect()
    finally:
        # A signal can land before critical() has begun. The guard raises only the
        # first one, so the second attempt cannot be interrupted.
        try:
            with guard.critical():
                directory.remove()
        except (Interrupted, ReviewTimeout):
            with guard.critical():
                directory.remove()
            raise


def _run(
    args: argparse.Namespace,
    pipeline: Pipeline | None,
    reviewer: Reviewer | AttributeReviewer | None,
    day: datetime.date,
    guard: _SignalGuard,
    setup: JudgeSetup | None = None,
    clock: Callable[[], datetime.datetime] | None = None,
    austin: AustinEndpoints | None = None,
    opener: DetectorOpener | None = None,
    calgary: CalgaryEndpoints | None = None,
) -> int:
    mode: Mode = args.mode
    clock = clock or _utcnow
    if args.attributes:
        view = attribute_view(args)
        reviewer = _confirming(args, view, reviewer)
        check_light(args, clock)
        attribute_judge = open_attribute_judge(args, setup or JudgeSetup())
        try:
            return _attribute_session(
                args,
                pipeline,
                reviewer,
                day,
                guard,
                view,
                attribute_judge,
                clock,
                austin or AustinEndpoints(),
                opener or _open_detector,
                calgary,
            )
        finally:
            if attribute_judge is not None:
                attribute_judge.close()
    if args.allow_dark:
        raise SpotcheckError(ALLOW_DARK_REFUSAL)
    if args.min_height is not None:
        raise SpotcheckError(MIN_HEIGHT_REFUSAL)
    if reviewer is not None and not isinstance(reviewer, Reviewer):
        raise SpotcheckError("the reviewer given cannot judge detections")
    if args.record_boxes and mode == "frames":
        raise SpotcheckError(RECORD_BOXES_FRAMES_REFUSAL)
    view = resolve_view(args.view, mode, args.judgements)
    reviewer = _confirming(args, view, reviewer)
    judge = open_judge(args, mode, setup or JudgeSetup())
    try:
        return _check(args, pipeline, reviewer, day, guard, mode, view, judge, clock)
    finally:
        if judge is not None:
            judge.close()


def _confirming[T](args: argparse.Namespace, view: View, reviewer: T) -> T:
    """`reviewer`, asking before a stop if --confirm-stop is given (refused without the
    window view, before anything else happens)."""
    if args.confirm_stop:
        if view != "window":
            raise SpotcheckError(CONFIRM_STOP_REFUSAL)
        if isinstance(reviewer, WindowReviewer):
            reviewer.confirm_stop = True
    return reviewer


def check_light(args: argparse.Namespace, clock: Callable[[], datetime.datetime]) -> None:
    """Refuse an attribute session whose answers are taken live (not from --judgements)
    when it is dark at `clock()` where its frames come from (London, or Austin with
    `--source austin`, or Calgary with `--source calgary`), unless --allow-dark is given:
    after dark most near-field crops cannot be judged. Raises SpotcheckError before
    anything else happens."""
    if args.judgements is not None or args.allow_dark:
        return
    source = getattr(args, "source", "london")
    if source == "austin":
        if light_at(clock(), AUSTIN) == "dark":
            raise SpotcheckError(AUSTIN_DARK_REFUSAL)
    elif source == "calgary":
        if light_at(clock(), CALGARY) == "dark":
            raise SpotcheckError(CALGARY_DARK_REFUSAL)
    elif light_at(clock()) == "dark":
        raise SpotcheckError(DARK_REFUSAL)


def attribute_view(args: argparse.Namespace) -> View:
    """The view of an attribute session: crops mode, no per-box file, and the window or a
    judgements file (any view that resolve_view allows with it). Raises SpotcheckError
    for any other combination, before anything else happens."""
    if args.mode != "crops":
        raise SpotcheckError(ATTRIBUTES_FRAMES_REFUSAL)
    if args.record_boxes:
        raise SpotcheckError(ATTRIBUTES_RECORD_BOXES_REFUSAL)
    view = resolve_view(args.view, "crops", args.judgements)
    if view != "window" and args.judgements is None:
        raise SpotcheckError(ATTRIBUTES_KEYBOARD_REFUSAL)
    return view


def _prepare(args: argparse.Namespace) -> Path:
    """Check the temporary and output directories; returns the output directory."""
    _check_temp_dir(Path(tempfile.gettempdir()))
    out_dir = Path(args.out_dir)
    if args.dry_run and _is_real_spotchecks(out_dir):
        raise SpotcheckError("--dry-run needs an --out-dir other than spotchecks/")
    _check_out_dir(out_dir)
    return out_dir


def _judgements_reviewer(text: str) -> JsonFileReviewer:
    path = Path(text)
    if os.path.lexists(path):
        raise SpotcheckError(f"{path} already exists; remove it, then start again")
    return JsonFileReviewer(path)


def _open_pipeline(
    args: argparse.Namespace,
    pipeline: Pipeline | None,
    guard: _SignalGuard,
    austin: AustinEndpoints | None = None,
    opener: DetectorOpener | None = None,
    calgary: CalgaryEndpoints | None = None,
) -> Pipeline:
    """Delete what earlier runs left behind, then open the pipeline (unless given): with
    `--source austin`, Austin's stills from `austin` (a local fake one with --dry-run),
    with `--source calgary`, Calgary's from `calgary` (likewise), detected by the detector
    `opener` opens."""
    removed = remove_stale(Path(tempfile.gettempdir()), args.timeout, guard=guard)
    if removed:
        print(f"Deleted {removed} review directories left by earlier runs.")
    if pipeline is not None:
        return pipeline
    if getattr(args, "source", "london") == "austin":
        opener = opener or _open_detector
        bbox = pilot_heights.DEFAULT_BBOX if args.bbox is None else args.bbox
        if args.dry_run:
            return dry_run_austin_pipeline(args.model, bbox, opener)
        return austin_pipeline(args.model, bbox, austin or AustinEndpoints(), opener)
    if getattr(args, "source", "london") == "calgary":
        opener = opener or _open_detector
        bbox = pilot_heights.CALGARY_BBOX if args.bbox is None else args.bbox
        if args.dry_run:
            return dry_run_calgary_pipeline(args.model, bbox, opener)
        return calgary_pipeline(args.model, bbox, calgary or CalgaryEndpoints(), opener)
    return dry_run_pipeline(args.model) if args.dry_run else live_pipeline(args.model)


def _sample_sweep(
    args: argparse.Namespace, pipeline: Pipeline, min_height: int = 0
) -> list[Sample]:
    """Sweep, detect and sample, printing progress. Frames that come from a generator
    (Austin's) are closed after it, however it ends, which stops their fetching."""
    frames = pipeline.frames()
    total = len(frames) if isinstance(frames, Sized) else None

    def detected(done: int) -> None:
        if _due(done, total):
            _progress(f"detected {done} of {total}" if total is not None else f"detected {done}")

    try:
        return sample(
            frames,
            pipeline.detector,
            n=args.n,
            min_persons=args.min_persons,
            seed=args.seed,
            progress=detected,
            min_height=min_height,
        )
    finally:
        close = getattr(frames, "close", None)
        if callable(close):
            close()


def _check(
    args: argparse.Namespace,
    pipeline: Pipeline | None,
    reviewer: Reviewer | None,
    day: datetime.date,
    guard: _SignalGuard,
    mode: Mode,
    view: View,
    judge: judge_hosted.LiveCropJudge | None,
    clock: Callable[[], datetime.datetime],
) -> int:
    out_dir = _prepare(args)
    if reviewer is None:
        if args.judgements is not None:
            reviewer = _judgements_reviewer(args.judgements)
        elif view == "window":
            reviewer = WindowReviewer(guard=guard, confirm_stop=args.confirm_stop)
        else:
            reviewer = KeyboardReviewer(sys.stdin.fileno())
    pipeline = _open_pipeline(args, pipeline, guard)

    started_at = clock()
    samples = _sample_sweep(args, pipeline)
    if not samples:
        raise SpotcheckError(f"no frame had at least {args.min_persons} person detections")
    frames_reviewed = len(samples)
    heights = box_heights(samples)
    items = render(samples, mode)
    del samples  # the frames are not needed any more
    _progress(f"opening the review: {len(items)} image(s)")
    judgements = _review(items, mode, reviewer, args.timeout, guard, view) if items else {}
    stats = compute_stats(
        items,
        judgements,
        mode=mode,
        frames_reviewed=frames_reviewed,
        reviewer=args.reviewer,
        info=pipeline.info,
        day=day,
    )
    boxes: dict[str, object] | None = None
    if args.record_boxes:
        boxes = box_record(
            items,
            judgements,
            heights,
            frames_reviewed=frames_reviewed,
            info=pipeline.info,
            day=day,
            started_at=started_at,
        )
    if judge is not None:
        # Only now, with the review over and its judgements valid: the crops in memory,
        # except those the reviewer could not tell.
        judged = judged_items(items, judgements)
        answers = ask_judge([item.image for item in judged], judge)
        block = judge_stats(judged, judgements, answers, judge, args.judge)
        stats["judge"] = block
        print(
            f"Judge {args.judge}: {len(answers)} of {len(judged)} crop(s) answered "
            f"({block['status']}), {block['requests']} request(s), ${block['cost_usd']:.6f}",
            flush=True,
        )
        del judged
    del items
    path = write_stats(stats, out_dir, day, boxes)
    if boxes is None:
        print(f"Statistics written to {path}")
    else:
        print(f"Statistics written to {path}, box heights to {path.parent / BOXES_DIR / path.name}")
    return 0


# The template entry of each crop in an attribute session's numbering file: to be filled
# with "y", "n" or "u" each, or replaced by "x".
ATTRIBUTE_TEMPLATE_ENTRY: dict[str, object] = dict.fromkeys(ATTRIBUTE_NAMES, "")


def _attribute_session(
    args: argparse.Namespace,
    pipeline: Pipeline | None,
    reviewer: Reviewer | AttributeReviewer | None,
    day: datetime.date,
    guard: _SignalGuard,
    view: View,
    judge: judge_hosted.LiveAttributeJudge | None,
    clock: Callable[[], datetime.datetime],
    austin: AustinEndpoints | None = None,
    opener: DetectorOpener | None = None,
    calgary: CalgaryEndpoints | None = None,
) -> int:
    """Label the near-field crops of one sweep, then (with a judge) ask the model the same
    questions about the crops the reviewer did not reject; write the attribute file. With
    `--source austin`, the sweep is one pass over Austin's 1920x1080 stills; with
    `--source calgary`, over Calgary's 840x630 stills."""
    out_dir = _prepare(args)
    source: Source = getattr(args, "source", "london")
    labeller: AttributeReviewer
    if reviewer is None:
        if args.judgements is not None:
            labeller = _judgements_reviewer(args.judgements)
        elif source != "london":
            labeller = WindowReviewer(guard=guard, confirm_stop=args.confirm_stop, fit_screen=True)
        else:
            labeller = WindowReviewer(guard=guard, confirm_stop=args.confirm_stop)
    elif isinstance(reviewer, AttributeReviewer):
        labeller = reviewer
    else:
        raise SpotcheckError("the reviewer given cannot answer the attribute questions")
    pipeline = _open_pipeline(args, pipeline, guard, austin, opener, calgary)

    min_height = NEAR_FIELD_MIN_HEIGHT_PX if args.min_height is None else args.min_height
    started_at = clock()
    samples = _sample_sweep(args, pipeline, min_height)
    if not samples:
        raise SpotcheckError(
            f"no frame had at least {args.min_persons} near-field person detections "
            f"(boxes at least {min_height} px tall)"
        )
    frames_reviewed = len(samples)
    heights = box_heights(samples)
    items = render(samples, "crops")
    del samples  # the frames are not needed any more
    _progress(f"opening the review: {len(items)} crop(s)")
    answers: dict[int, str | None] = {}
    if items:
        answers = _shown(
            items,
            "crops",
            guard,
            view,
            ATTRIBUTE_LEGEND,
            ATTRIBUTE_TEMPLATE_ENTRY,
            functools.partial(_collect_attributes, items, labeller, args.timeout, guard),
        )
    models: list[str] = []
    if judge is not None:
        # Only now, with the labelling over and its answers valid: the crops in memory,
        # except those the reviewer rejected.
        kept = [item for item in items if answers[item.number] is not None]
        models = ask_attribute_judge([item.image for item in kept], judge)
        usage = judge.usage
        cost = judge_hosted.cost_usd(judge.candidate, usage.input_tokens, usage.output_tokens)
        print(
            f"Judge {args.judge}: {len(models)} of {len(kept)} crop(s) answered, "
            f"{usage.requests} request(s), ${cost:.6f}",
            flush=True,
        )
        del kept
    record = attribute_record(
        items,
        answers,
        heights,
        models,
        frames_reviewed=frames_reviewed,
        info=pipeline.info,
        day=day,
        started_at=started_at,
        judge=args.judge,
        min_height=min_height,
    )
    record = sourced_record(record, source, started_at)
    del items
    path = write_attributes(record, out_dir, day, source)
    print(
        f"Attribute labels written to {path}: {record['crops_shown']} crop(s) shown, "
        f"{record['crops_rejected']} rejected"
    )
    return 0


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def main(
    argv: Sequence[str] | None = None,
    *,
    pipeline: Pipeline | None = None,
    reviewer: Reviewer | AttributeReviewer | None = None,
    today: datetime.date | None = None,
    judge_endpoint: str | None = None,
    judge_timeout: float = judge_hosted.REQUEST_TIMEOUT,
    judge_sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime.datetime] | None = None,
    austin: AustinEndpoints | None = None,
    open_detector: DetectorOpener | None = None,
    calgary: CalgaryEndpoints | None = None,
) -> int:
    """Run one spot-check (or, with `--attributes`, one attribute session). `pipeline` and
    `reviewer` replace the live ones (an attribute session needs an AttributeReviewer),
    `judge_endpoint`, `judge_timeout` and `judge_sleep` the judge's origin, request timeout
    and wait between retries (tests: a fake judge on this machine), and `clock` the UTC
    clock that dates the start of the sweep. With `--source austin`, `austin` replaces
    Austin's camera list and stills (tests: a server on this machine), with
    `--source calgary`, `calgary` Calgary's, and `open_detector` the detector (`model` ->
    detector and its DetectorInfo)."""
    ci = _ci_variables()
    if ci:
        print(
            f"spotcheck: refusing to run in CI ({' and '.join(ci)} set): "
            "camera frames are never rendered there (AGENTS.md INV-1)",
            file=sys.stderr,
        )
        return 2
    args = parse_args(argv)
    guard = _SignalGuard()
    try:
        try:
            guard.install()
            setup = JudgeSetup(judge_endpoint, judge_timeout, judge_sleep)
            day = today or datetime.date.today()
            return _run(
                args, pipeline, reviewer, day, guard, setup, clock, austin, open_detector, calgary
            )
        finally:
            # Nothing may interrupt the handlers below. A signal that lands before
            # stop() is raised here, is the last one raised, and is caught below.
            guard.stop()
    except Interrupted as exc:
        name = _signal_name(exc.signum)
        print(f"spotcheck: stopped by {name}; no statistics written", file=sys.stderr)
        return 128 + exc.signum
    except ReviewTimeout:
        print("spotcheck: the review timed out; no statistics written", file=sys.stderr)
        return 3
    except (SpotcheckError, ReviewAborted) as exc:
        print(f"spotcheck: {exc}", file=sys.stderr)
        return 1
    finally:
        guard.restore()


if __name__ == "__main__":
    raise SystemExit(main())
