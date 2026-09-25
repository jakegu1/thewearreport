"""Benchmark the detector: latency on synthetic frames, and YOLOX-s against YOLOX-m live.

  python -m wearreport.benchmark --synthetic 50
      Median ms per frame for YOLOX-s on random 352x288 frames. No network.
  python -m wearreport.benchmark --live --cameras 200
      Lists cameras with `wearreport.registry`, fetches one frame from each of the first
      200 with `wearreport.fetch`, runs YOLOX-s and YOLOX-m on the same frames and prints
      a table: frames, median ms/frame, persons, umbrellas, total seconds per model.
  python -m wearreport.benchmark --dry-run --cameras 20
      The live comparison against a local fake camera server (no network).

Models come from `--model-dir` (default `.models/`, filled by `scripts/fetch_model.sh`;
YOLOX-m needs `--with-m`). Each model runs once on a blank frame before timing starts.
The table labels each row by file name; a line per model below it names the pinned model
that the file's SHA-256 matches, and a warning says when two files are the same model.

Privacy (AGENTS.md INV-1): frames exist only in memory and are dropped as soon as every
model has seen them. Only counts and timings are printed or logged.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from wearreport import detect, fetch, registry
from wearreport.settings import SettingsError, load_settings
from wearreport.testing.fake_cameras import FRAME_HEIGHT, FRAME_WIDTH

MODELS = ("yolox_s", "yolox_m")
DEFAULT_CAMERAS = 200
DEFAULT_DRY_RUN_CAMERAS = 20
MAX_COUNT = 2000  # frames or cameras; every frame is held in memory (0.3 MB each)
SEED = 0

Frame = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Result:
    model: str
    frames: int
    median_ms: float
    persons: int
    umbrellas: int
    seconds: float


def synthetic_frames(count: int, seed: int = SEED) -> list[Frame]:
    """`count` frames of uniform noise at the JamCam size."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        for _ in range(count)
    ]


def run(detector: detect.Detector, frames: Sequence[Frame]) -> Result:
    """Time `detector` on each frame after one warm-up call; count what it finds."""
    detector.detect(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
    timings: list[float] = []
    persons = umbrellas = 0
    started = time.perf_counter()
    for frame in frames:
        t0 = time.perf_counter()
        found = detector.detect(frame)
        timings.append(time.perf_counter() - t0)
        persons += sum(d.label == "person" for d in found)
        umbrellas += sum(d.label == "umbrella" for d in found)
    seconds = time.perf_counter() - started
    median_ms = 1000 * statistics.median(timings) if timings else 0.0
    return Result(detector.model_name, len(frames), median_ms, persons, umbrellas, seconds)


def compare(detectors: Sequence[detect.Detector], frames: Sequence[Frame]) -> list[Result]:
    """Run every detector on the same frames, one model after another."""
    return [run(detector, frames) for detector in detectors]


def fetched_frames(cameras: Sequence[registry.Camera]) -> tuple[list[Frame], int]:
    """Fetch one frame per camera in memory; the decoded frames and the failure count."""
    results = fetch.fetch_sweep(cameras)
    frames = [r.frame for r in results if r.frame is not None]
    return frames, len(results) - len(frames)


def table(results: Sequence[Result]) -> str:
    lines = [
        "| model | frames | median ms/frame | persons | umbrellas | total s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r.model} | {r.frames} | {r.median_ms:.1f} | {r.persons} | {r.umbrellas} "
        f"| {r.seconds:.1f} |"
        for r in results
    ]
    return "\n".join(lines)


def _sha256(path: Path) -> str | None:
    try:
        return detect.sha256_of(path)
    except OSError:
        return None


def identity(paths: Sequence[Path]) -> list[str]:
    """For each model file, the pinned model its SHA-256 matches, saying so when that is
    not the model its name claims; then a warning for each pair of files that are one
    model. The table's rows are labelled by file name, so these lines keep it honest."""
    pinned_by_digest = {digest: name for name, digest in detect.MODEL_SHA256.items()}
    digests = [(path, _sha256(path)) for path in paths]
    lines = []
    for path, digest in digests:
        pinned = pinned_by_digest.get(digest) if digest is not None else None
        if pinned is None:
            lines.append(f"sha256: {path.name} matches no pinned model")
        elif pinned == path.name:
            lines.append(f"sha256: {path.name} is the pinned {pinned}")
        else:
            lines.append(f"sha256: {path.name} is the pinned {pinned}, not {path.name}")
    for (first, a), (second, b) in itertools.combinations(digests, 2):
        if a is not None and a == b:
            lines.append(
                f"warning: {first.name} and {second.name} have the same SHA-256; "
                "their rows measure one model"
            )
    return lines


def _paths(model_dir: Path, names: Sequence[str]) -> list[Path]:
    return [model_dir / f"{name}.onnx" for name in names]


def _detectors(paths: Sequence[Path]) -> list[detect.Detector]:
    return [detect.Detector(path) for path in paths]


def _synthetic(model_dir: Path, count: int) -> int:
    paths = _paths(model_dir, MODELS[:1])
    (detector,) = _detectors(paths)
    result = run(detector, synthetic_frames(count))
    print(f"model: {result.model}")
    print("\n".join(identity(paths)))
    print(f"frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"frames: {result.frames}")
    print(f"median ms per frame: {result.median_ms:.1f}")
    print(f"total seconds: {result.seconds:.1f}")
    return 0


def _comparison(model_dir: Path, cameras: Sequence[registry.Camera]) -> int:
    paths = _paths(model_dir, MODELS)
    detectors = _detectors(paths)  # before fetching: fail fast on a bad model
    started = time.monotonic()
    frames, failed = fetched_frames(cameras)
    print(f"cameras: {len(cameras)}")
    print(f"frames fetched: {len(frames)}")
    print(f"frames failed: {failed}")
    print(f"fetch seconds: {time.monotonic() - started:.1f}")
    results = compare(detectors, frames)
    del frames
    print(table(results))
    print("\n".join(identity(paths)))
    return 0


def _count(text: str) -> int:
    value = int(text)
    if not 1 <= value <= MAX_COUNT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_COUNT}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.benchmark",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", type=_count, metavar="N", help="N random frames")
    mode.add_argument("--live", action="store_true", help="compare models on live frames")
    mode.add_argument("--dry-run", action="store_true", help="compare on a fake camera server")
    ap.add_argument("--cameras", type=_count, default=None, help="cameras to fetch")
    ap.add_argument("--model-dir", type=Path, default=detect.MODEL_DIR)
    args = ap.parse_args(argv)

    try:
        if args.synthetic is not None:
            return _synthetic(args.model_dir, args.synthetic)
        if args.dry_run:
            from wearreport.testing.fake_cameras import FakeCameraServer

            with FakeCameraServer() as server:
                count = args.cameras or DEFAULT_DRY_RUN_CAMERAS
                return _comparison(args.model_dir, server.cameras(count))
        try:
            listed = registry.list_cameras(load_settings().tfl_app_key)
        except (registry.RegistryError, SettingsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return _comparison(args.model_dir, listed[: args.cameras or DEFAULT_CAMERAS])
    except detect.DetectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
