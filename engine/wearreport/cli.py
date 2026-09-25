"""The `wearreport` command.

  wearreport sweep --dry-run [--model NAME]
      One live sweep (registry, frames, detector, weather). The record and status.json
      go into a new temporary directory, whose path is printed. No git state is touched.
  wearreport sweep --data-dir PATH [--model NAME]
      One live sweep, published into PATH, a checkout of the `data` branch. Nothing is
      committed or pushed; the scheduled workflow (T-007) does that.

The detector is YOLOX-m (`yolox_m.onnx`) unless `--model` names another pinned model.
The record's `model_sha256` is the digest of the file that was loaded, read before and
after loading. Weather comes only from `wearreport.weather.current_conditions`; a
WeatherConfigError (the dev-only weather flag set in production) stops the sweep with
exit status 1 before any camera is contacted, and nothing is written.

Exit status: 0 when the record was published (or was already there, identical), 1 when
the sweep or the publish failed, 2 for a usage error. Logs are JSON lines on stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from wearreport import aggregate, detect, publish, weather
from wearreport.settings import SettingsError

DEFAULT_MODEL = "yolox_m.onnx"
DRY_RUN_PREFIX = "wearreport-sweep-"

logger = logging.getLogger("wearreport.cli")


class _JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message and every `extra=` field."""

    _standard = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in self._standard})
        return json.dumps(payload, default=str)


def _configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. by a test runner)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def load_detector(name: str) -> tuple[detect.Detector, str]:
    """The pinned model `name` and the SHA-256 of the file that was loaded.

    The file is hashed before and after the detector opens it (the detector hashes the
    bytes it loads against the pins as well); a digest that is not the pin for `name`,
    or that changes while loading, raises DetectorError.
    """
    path = detect.model_path(name)
    before = _digest(path, name)
    if before != detect.MODEL_SHA256[name]:
        raise detect.DetectorError(f"{name} does not match its pinned SHA-256; run make setup")
    detector = detect.Detector(path)
    if _digest(path, name) != before:
        raise detect.DetectorError(f"{name} changed while it was being loaded")
    return detector, before


def _digest(path: Path, name: str) -> str:
    try:
        return detect.sha256_of(path)
    except OSError as exc:
        raise detect.DetectorError(f"cannot read {name}: {exc.strerror}; run make setup") from None


def _new_temporary_directory() -> Path:
    """A new private directory under TMPDIR (or TEMP, TMP).

    The directory is named explicitly because tempfile.gettempdir() would first probe the
    candidates by writing and deleting a test file, and the dry run writes nothing but
    its output. Without any of the three variables, tempfile picks the place.
    """
    candidates = (os.environ.get(var, "") for var in ("TMPDIR", "TEMP", "TMP"))
    base = next((c for c in candidates if c and os.path.isdir(c)), None)
    return Path(tempfile.mkdtemp(prefix=DRY_RUN_PREFIX, dir=base))


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="wearreport", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = ap.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("sweep", help="run one sweep and publish its record")
    target = sweep.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--dry-run", action="store_true", help="publish into a new temporary directory"
    )
    target.add_argument("--data-dir", type=Path, help="a checkout of the data branch")
    sweep.add_argument(
        "--model",
        choices=sorted(detect.MODEL_SHA256),
        default=DEFAULT_MODEL,
        help=f"pinned detector model (default {DEFAULT_MODEL})",
    )
    return ap


def _sweep(args: argparse.Namespace) -> int:
    if args.data_dir is not None and not args.data_dir.is_dir():
        print(f"error: data directory {args.data_dir} does not exist", file=sys.stderr)
        return 1
    try:
        detector, digest = load_detector(args.model)
        record = aggregate.run_sweep(detector, digest, model_name=Path(args.model).stem)
    except weather.WeatherConfigError as exc:
        logger.error("sweep stopped: weather configuration", extra={"error": str(exc)})
        print(f"error: weather configuration not allowed: {exc}", file=sys.stderr)
        return 1
    except (
        aggregate.SweepError,
        aggregate.RecordError,
        detect.DetectorError,
        SettingsError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    del detector

    directory = _new_temporary_directory() if args.dry_run else args.data_dir
    try:
        result = publish.publish(directory, record, now=aggregate.parse_utc(record["finished_at"]))
    except (publish.PublishError, aggregate.RecordError) as exc:
        if args.dry_run:
            with contextlib.suppress(OSError):
                shutil.rmtree(directory)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    failed = sum(record["frames_failed"].values())
    weather_source = "none" if record["weather"] is None else record["weather"]["source"]
    print(f"sweep_id: {record['sweep_id']}")
    print(f"cameras listed: {record['cameras_listed']}")
    print(f"frames ok: {record['frames_ok']}")
    print(f"frames failed: {failed}")
    print(f"persons: {record['persons_total']}")
    print(f"umbrellas: {record['umbrellas_total']}")
    print(f"weather: {weather_source}")
    print(f"record: {result.record_path}{'' if result.created else ' (already published)'}")
    print(f"status: {result.status_path}")
    if args.dry_run:
        print(f"directory: {directory}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _configure_logging()
    return _sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
