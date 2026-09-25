"""Spot-check detection precision on live frames, keeping only the tallies.

  python -m wearreport.tools.spotcheck --n 20 [--mode crops|frames] [--min-persons 3]
      [--seed S] [--out-dir DIR] [--reviewer NAME] [--model yolox_m.onnx]
      [--judgements PATH] [--timeout SECONDS] [--dry-run]

Lists the cameras (`wearreport.registry`), fetches one sweep in memory
(`wearreport.fetch`), runs the detector with its default thresholds, and samples up to N
frames with at least `--min-persons` person detections, at random (seeded by `--seed`).
The detections are rendered for a reviewer, either as one crop per detection with a 50%
margin (`crops`, the default) or as whole frames with numbered boxes (`frames`). The
reviewer judges each image from the keyboard, or by writing a JSON file (`--judgements`)
that the tool polls for. The tool then writes one statistics file,
`<out-dir>/YYYY-MM-DD.json` (then `-2`, `-3`...), and nothing else. The format is in
`spotchecks/README.md`.

Privacy (AGENTS.md INV-1, exception (c)). This is the only engine module that writes
images derived from camera frames, and it writes them only into a directory it creates
with `tempfile.mkdtemp(prefix=TEMP_PREFIX)` (mode 0700) and always deletes:

- after a normal run, an exception, the review timeout, or any catchable signal whose
  default action ends the process (SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGUSR1, ...; see
  HANDLED_SIGNALS). Signals are turned into exceptions, except while the directory is
  being created or deleted: a signal that arrives then is held until that step is done.
  Only the first signal is raised; later ones are dropped, so they cannot interrupt the
  clean-up the first one started;
- the directory holds a lock (flock) while the tool runs. At start the tool deletes
  every `wearreport-spotcheck-*` directory of the current user in the temporary
  directory that is older than the timeout and not locked, which is what SIGKILL (which
  cannot be handled) or a power cut leaves behind. A running instance's directory stays
  locked, so a second instance never deletes it;
- image files are created with O_EXCL and O_NOFOLLOW, mode 0600, and named after their
  number only (`crop-0001.png`, `frame-0001.png`), never after a camera. Camera ids are
  dropped as soon as the sweep is fetched.

The static privacy guard exempts exactly this file from its binary-open rules
(`IMAGE_WRITE_EXEMPTION` in scripts/privacy_guard.py); every other rule still applies,
and no other engine module may import this one. The tool refuses to run when `CI` or
`GITHUB_ACTIONS` is non-empty, and when the temporary directory lies inside this
repository or any git work tree, before any network access.

`--dry-run` sweeps a local fake camera server that serves the licensed fixture photos
in fixtures/detect/ (no network). Its statistics describe those photos, not the
cameras, so it needs an `--out-dir` other than spotchecks/.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import fcntl
import json
import math
import os
import random
import re
import select
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Literal, Protocol, TextIO, get_args, runtime_checkable

import numpy as np
import numpy.typing as npt

from wearreport import detect, fetch, registry
from wearreport._cv import cv2
from wearreport.settings import SettingsError, load_settings

Mode = Literal["crops", "frames"]
MODES: tuple[Mode, ...] = get_args(Mode)
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
# hang. Like SIGKILL, what those leave behind is deleted by a later run.
TERMINATING_SIGNAL_NAMES = (
    "SIGINT",
    "SIGTERM",
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

MAX_N = 500
MAX_MIN_PERSONS = 100
MAX_TIMEOUT_S = 24 * 3600
MAX_MISSED = 1000  # per image
MAX_JUDGEMENTS_BYTES = 1024 * 1024
MAX_LINE_BYTES = 4096
MAX_FILES_PER_DAY = 1000
JSON_POLL_S = 0.5
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
DRY_RUN_FIXTURES = ("people_aldgate.jpg", "umbrella_rain.jpg")


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
) -> list[Sample]:
    """Up to `n` frames with at least `min_persons` person detections, chosen uniformly at
    random (reservoir sampling, so at most `n` frames are held) and returned in the order
    they came in."""
    rng = random.Random(seed)  # noqa: S311  (sampling, not security)
    kept: list[tuple[int, Sample]] = []
    seen = 0
    for frame in frames:
        persons = tuple(d for d in detector.detect(frame) if d.label == "person")
        if len(persons) < min_persons:
            continue
        if len(kept) < n:
            kept.append((seen, Sample(frame, persons)))
        else:
            slot = rng.randint(0, seen)
            if slot < n:
                kept[slot] = (seen, Sample(frame, persons))
        seen += 1
    return [s for _, s in sorted(kept, key=lambda pair: pair[0])]


def _sweep_frames(cameras: Sequence[registry.Camera]) -> Iterator[Frame]:
    """Fetch one sweep in memory; yield its frames without their camera ids, and let go
    of each frame once it has been handed on."""
    results = fetch.fetch_sweep(cameras)
    results.reverse()
    while results:
        frame = results.pop().frame
        if frame is not None:
            yield frame


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

    def frames() -> Iterator[Frame]:
        try:
            cameras = registry.list_cameras(load_settings().tfl_app_key)
        except (registry.RegistryError, SettingsError) as exc:
            raise SpotcheckError(f"cannot list cameras: {exc}") from None
        yield from _sweep_frames(cameras)

    return Pipeline(frames=frames, detector=detector, info=info)


def dry_run_pipeline(model: str) -> Pipeline:
    """A local fake camera server serving the licensed fixture photos, and the real model."""
    from wearreport.testing.fake_cameras import FakeCameraServer

    try:
        bodies = [(FIXTURE_DIR / name).read_bytes() for name in DRY_RUN_FIXTURES]
    except OSError as exc:
        raise SpotcheckError(f"cannot read the dry-run fixtures: {exc.strerror}") from None
    detector, info = _open_detector(model)

    def frames() -> Iterator[Frame]:
        with FakeCameraServer() as server:
            cameras = server.cameras(DRY_RUN_CAMERAS)
            for i, camera in enumerate(cameras):
                if i % 3 != 2:  # every third camera serves noise
                    server.serve_body(camera.id, bodies[i % len(bodies)])
            yield from _sweep_frames(cameras)

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

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # signal handlers can only be set from the main thread
        self._installed = True
        for signum in (*HANDLED_SIGNALS, signal.SIGALRM):
            self._previous[signum] = signal.signal(signum, self._handle)

    def stop(self) -> None:
        """From here on, drop every signal: the tool is on its way out."""
        self._stopping = True

    def restore(self) -> None:
        if not self._installed:
            return
        self._stopping = True
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)  # type: ignore[arg-type]
        self._installed = False

    def alarm(self, seconds: float) -> None:
        """Raise ReviewTimeout after `seconds` (0 cancels). A backstop: reviewers also
        watch their deadline."""
        if self._installed:
            signal.setitimer(signal.ITIMER_REAL, seconds)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self._stopping:
            return
        exc: BaseException = (
            ReviewTimeout("the review timed out")
            if signum == signal.SIGALRM
            else Interrupted(signum)
        )
        if self._depth:
            if self._pending is None:
                self._pending = exc
            return
        self._stopping = True  # before raising: no later signal may interrupt clean-up
        raise exc

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
                    raise exc


def _lock(path: str | Path) -> int | None:
    """An open descriptor holding an exclusive lock on the directory, or None if another
    process holds it (or it cannot be opened as a directory)."""
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
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(self.path / item.file, flags, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(encoded.tobytes())

    def write_numbering(self, items: Sequence[ReviewItem], mode: Mode) -> Path:
        """The numbering, and a judgements template to copy, as JSON next to the images."""
        if self.path is None:
            raise ValueError("no review directory")
        template = {str(item.number): _template_entry(mode) for item in items}
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
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None
        return gone


def remove_stale(tmpdir: Path, max_age_s: float, now: float | None = None) -> int:
    """Delete this user's review directories in `tmpdir` that are older than `max_age_s`
    and not locked by a running instance. Returns how many were deleted."""
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
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
            continue
        if now - st.st_mtime <= max_age_s:
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


# Judgements ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Judgement:
    """For one image: the boxes that are not a person, the boxes that are a person inside
    a vehicle, and (frames mode only) how many visible people have no box."""

    not_person: frozenset[int]
    in_vehicle: frozenset[int]
    missed: int | None


@runtime_checkable
class Reviewer(Protocol):
    """Whoever judges the rendered images. `judge` returns one judgement per item, keyed by
    item number, or raises ReviewTimeout once `deadline` (time.monotonic()) has passed."""

    def judge(
        self, items: Sequence[ReviewItem], mode: Mode, deadline: float
    ) -> Mapping[int, Judgement]: ...


def _template_entry(mode: Mode) -> dict[str, object]:
    entry: dict[str, object] = {"not_person": [], "in_vehicle": []}
    if mode == "frames":
        entry["missed"] = 0
    return entry


def validate(judgement: Judgement, item: ReviewItem, mode: Mode) -> Judgement:
    """Return `judgement` if it is valid for `item`, else raise JudgementError."""
    for name in ("not_person", "in_vehicle"):
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
    both = sorted(judgement.not_person & judgement.in_vehicle)
    if both:
        raise JudgementError(
            f"image {item.number}: box {both[0]} cannot be both not a person and in a vehicle"
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


def parse_judgements(raw: bytes, items: Sequence[ReviewItem], mode: Mode) -> dict[int, Judgement]:
    """Parse a judgements file (format in spotchecks/README.md); raise JudgementError."""
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
    if not isinstance(data, dict):
        raise JudgementError('the file must hold one object, e.g. {"1": {...}, "2": {...}}')
    numbers = {str(item.number): item.number for item in items}
    judgements: dict[int, Judgement] = {}
    for key, entry in data.items():
        if key not in numbers:
            raise JudgementError(f"there is no image {key[:20]!r}")
        if not isinstance(entry, dict):
            raise JudgementError(f"image {key}: the judgement must be an object")
        allowed = {"not_person", "in_vehicle"} | ({"missed"} if mode == "frames" else set())
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
        )
    return validate_all(judgements, items, mode)


LINE_TOKEN = re.compile(r"([nvm])(\d{1,6})?")


def parse_line(line: str, item: ReviewItem, mode: Mode) -> Judgement:
    """Parse one keyboard answer for `item`; raise JudgementError.

    Tokens, separated by spaces or commas: `n<box>` not a person, `v<box>` a person inside
    a vehicle, `m<count>` people missed (frames mode). In crops mode a bare `n` or `v`
    means the crop's box. An empty line: every box is a pedestrian and nobody is missed.
    """
    not_person: list[int] = []
    in_vehicle: list[int] = []
    missed: int | None = None
    for token in line.replace(",", " ").lower().split():
        match = LINE_TOKEN.fullmatch(token)
        if match is None:
            raise JudgementError(f"cannot read {token[:20]!r}; use n<box>, v<box> or m<count>")
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
        target = not_person if kind == "n" else in_vehicle
        if box in target:
            raise JudgementError(f"box {box} is listed twice")
        target.append(box)
    if mode == "frames" and missed is None:
        missed = 0
    return validate(Judgement(frozenset(not_person), frozenset(in_vehicle), missed), item, mode)


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
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(self.fd, 4096)
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
        hint = "n<box> not a person, v<box> person in a vehicle"
        hint += ", m<count> people missed" if mode == "frames" else " (bare n or v: this crop)"
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
                    return parse_judgements(seen, items, mode)
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
    """The statistics record: counts, and the ratios derived from them (None on 0/0)."""
    shown = sum(len(item.boxes) for item in items)
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


def write_stats(stats: Mapping[str, object], out_dir: Path, day: datetime.date) -> Path:
    """Write `stats` to a new `<day>.json` (or `<day>-2.json`, ...); never overwrite.

    Raises SpotcheckError, with the statistics in the message so they are not lost, if
    the file cannot be written.
    """
    text = json.dumps(stats, indent=2, ensure_ascii=True) + "\n"
    try:
        return _write_new(text, out_dir, day)
    except OSError as exc:
        raise SpotcheckError(
            f"cannot write statistics into {out_dir} ({exc.strerror}); "
            f"they were: {json.dumps(stats, ensure_ascii=True)}"
        ) from None


def _write_new(text: str, out_dir: Path, day: datetime.date) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(1, MAX_FILES_PER_DAY + 1):
        name = f"{day.isoformat()}.json" if k == 1 else f"{day.isoformat()}-{k}.json"
        path = out_dir / name
        try:
            fh = open(path, "x", encoding="utf-8")  # noqa: SIM115  (closed below)
        except FileExistsError:
            continue
        try:
            with fh:
                fh.write(text)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path
    raise SpotcheckError(f"{out_dir} already holds {MAX_FILES_PER_DAY} files for {day}")


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
    return ap


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


def _review(
    items: Sequence[ReviewItem],
    mode: Mode,
    reviewer: Reviewer,
    timeout_s: float,
    guard: _SignalGuard,
) -> dict[int, Judgement]:
    """Render to the review directory, collect the judgements, delete the directory."""
    directory = ReviewDirectory()
    try:
        try:
            with guard.critical():
                workdir = directory.create()
            for item in items:
                directory.write_image(item)
            directory.write_numbering(items, mode)
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            raise SpotcheckError(f"cannot write the review directory: {reason}") from None
        _print_numbering(workdir, items)
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
    reviewer: Reviewer | None,
    day: datetime.date,
    guard: _SignalGuard,
) -> int:
    mode: Mode = args.mode
    _check_temp_dir(Path(tempfile.gettempdir()))
    out_dir = Path(args.out_dir)
    if args.dry_run and _is_real_spotchecks(out_dir):
        raise SpotcheckError("--dry-run needs an --out-dir other than spotchecks/")
    _check_out_dir(out_dir)
    if reviewer is None:
        if args.judgements is not None:
            path = Path(args.judgements)
            if os.path.lexists(path):
                raise SpotcheckError(f"{path} already exists; remove it, then start again")
            reviewer = JsonFileReviewer(path)
        else:
            reviewer = KeyboardReviewer(sys.stdin.fileno())
    removed = remove_stale(Path(tempfile.gettempdir()), args.timeout)
    if removed:
        print(f"Deleted {removed} review directories left by earlier runs.")
    if pipeline is None:
        pipeline = dry_run_pipeline(args.model) if args.dry_run else live_pipeline(args.model)

    samples = sample(
        pipeline.frames(), pipeline.detector, n=args.n, min_persons=args.min_persons, seed=args.seed
    )
    if not samples:
        raise SpotcheckError(f"no frame had at least {args.min_persons} person detections")
    frames_reviewed = len(samples)
    items = render(samples, mode)
    del samples  # the frames are not needed any more
    judgements = _review(items, mode, reviewer, args.timeout, guard) if items else {}
    stats = compute_stats(
        items,
        judgements,
        mode=mode,
        frames_reviewed=frames_reviewed,
        reviewer=args.reviewer,
        info=pipeline.info,
        day=day,
    )
    del items
    path = write_stats(stats, out_dir, day)
    print(f"Statistics written to {path}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    pipeline: Pipeline | None = None,
    reviewer: Reviewer | None = None,
    today: datetime.date | None = None,
) -> int:
    """Run one spot-check. `pipeline` and `reviewer` replace the live ones (tests)."""
    ci = _ci_variables()
    if ci:
        print(
            f"spotcheck: refusing to run in CI ({' and '.join(ci)} set): "
            "camera frames are never rendered there (AGENTS.md INV-1)",
            file=sys.stderr,
        )
        return 2
    args = build_parser().parse_args(argv)
    guard = _SignalGuard()
    try:
        try:
            guard.install()
            return _run(args, pipeline, reviewer, today or datetime.date.today(), guard)
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
