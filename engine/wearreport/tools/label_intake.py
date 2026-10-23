"""Offline intake of numeric attribute records into an immutable local directory."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from wearreport.tools import spotcheck_summary as summary

MAX_FILES = 256
MAX_FILE_BYTES = 1 << 20
MAX_TOTAL_BYTES = 32 << 20
_MODEL = re.compile(r"[A-Za-z0-9][\w.-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class IntakeError(ValueError):
    """Input or output cannot be safely accepted; messages never include payloads."""


@dataclass(frozen=True)
class _Session:
    name: str
    raw: bytes
    data: dict[str, Any]
    parsed: summary.Labelling


def _safe_name(name: str) -> bool:
    """Keep basenames portable and prevent control characters in metadata."""
    return (
        not any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} or c in '<>:"/\\|?*' for c in name)
        and not name.endswith((" ", "."))
        and name.split(".")[0].rstrip(" .").upper()
        not in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *[f"COM{i}" for i in range(1, 10)],
            *[f"LPT{i}" for i in range(1, 10)],
        }
    )


def _paths(incoming: Path, output: Path) -> tuple[Path, Path]:
    incoming = incoming.absolute()
    output = output.absolute()
    if incoming.is_symlink():
        raise IntakeError("input directory must not be a symbolic link")
    # Check lexical components as well as the resolved destination. Never traverse
    # an existing output symlink, including an otherwise identical output.
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise IntakeError("output must not traverse symbolic links")
    incoming, output = incoming.resolve(strict=True), output.resolve()
    if not incoming.is_dir():
        raise IntakeError("input must be a directory")
    if incoming == output or incoming in output.parents or output in incoming.parents:
        raise IntakeError("input and output must not overlap")
    if any((p / ".git").exists() or (p / ".git").is_symlink() for p in (output, *output.parents)):
        raise IntakeError("output must be outside a Git repository or worktree")
    return incoming, output


def _read_regular(path: Path, limit: int) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise IntakeError("input and output files must be regular files")
    if before.st_size > limit:
        raise IntakeError("file or batch byte limit exceeded")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise IntakeError("file changed while opening")
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise IntakeError("file or batch byte limit exceeded")
    return raw


def _parse(raw: bytes) -> tuple[dict[str, Any], summary.Labelling]:
    # Always use the existing parser before inspecting the additional detector
    # protocol. Its errors can contain hostile values, so callers replace them.
    parsed = summary.parse_labelling(raw)
    data = cast(dict[str, Any], json.loads(raw))
    detector = data["detector"]
    if set(detector) != {"model", "sha256", "conf"}:
        raise IntakeError("detector fields are incomplete or unknown")
    if not isinstance(detector["model"], str) or not _MODEL.fullmatch(detector["model"]):
        raise IntakeError("invalid detector model identifier")
    if not isinstance(detector["sha256"], str) or not _SHA256.fullmatch(detector["sha256"]):
        raise IntakeError("invalid detector digest")
    confidence = detector["conf"]
    if (
        type(confidence) not in (int, float)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise IntakeError("invalid detector confidence")
    return data, parsed


def _load(incoming: Path) -> tuple[list[_Session], int]:
    paths: list[Path] = []
    for path in incoming.iterdir():
        if path.is_symlink():
            raise IntakeError("symbolic link inputs are refused")
        if path.suffix != ".json":
            continue
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode) or not _safe_name(path.name):
            raise IntakeError("input requires safe regular JSON basenames")
        paths.append(path)
        if len(paths) > MAX_FILES:
            raise IntakeError("too many input JSON files")
    if not paths:
        raise IntakeError("input has no direct JSON files")
    total = duplicates = 0
    sessions: dict[tuple[str, str], _Session] = {}
    names: dict[str, tuple[str, str]] = {}
    for path in sorted(paths):
        raw = _read_regular(path, min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total))
        total += len(raw)
        data, parsed = _parse(raw)
        identity = (parsed.source, data["started_at"])
        name_key = path.name.casefold()
        if name_key in names and names[name_key] != identity:
            raise IntakeError("filename collision between sessions")
        names[name_key] = identity
        previous = sessions.get(identity)
        if previous is not None:
            if previous.data != data:
                raise IntakeError("conflicting observations for one session")
            duplicates += 1
        else:
            sessions[identity] = _Session(path.name, raw, data, parsed)
    return list(sessions.values()), duplicates


def _report(sessions: Sequence[_Session], duplicates: int) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for source in summary.SOURCES:
        group = [s for s in sessions if s.parsed.source == source]
        if not group:
            continue
        human = {name: dict.fromkeys("ynu", 0) for name in summary.ATTRIBUTE_NAMES}
        cameras: dict[str, Any] = {}
        missing = 0
        for session in group:
            for _height, answers, _model in session.parsed.crops:
                for index, name in enumerate(summary.ATTRIBUTE_NAMES):
                    human[name][answers[index]] += 1
            camera_yield = session.data.get("camera_yield")
            if camera_yield is None:
                missing += 1
                continue
            for camera, values in camera_yield.items():
                counts = cameras.setdefault(
                    camera,
                    {
                        "shown": 0,
                        "rejected": 0,
                        "outer_judgeable": 0,
                        "sessions": 0,
                    },
                )
                for key, value in zip(
                    ("shown", "rejected", "outer_judgeable"), values, strict=True
                ):
                    counts[key] += value
                counts["sessions"] += 1
        sources[source] = {
            "sessions": len(group),
            "crops": {
                "shown": sum(s.parsed.shown for s in group),
                "rejected": sum(s.parsed.rejected for s in group),
                "labelled": sum(len(s.parsed.crops) for s in group),
            },
            "human": human,
            "camera_yield": dict(sorted(cameras.items())),
            "missing_camera_yield_sessions": missing,
        }
    return {"duplicates": duplicates, "sources": sources}


def _json(data: object) -> str:
    return json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"


def _contents(sessions: Sequence[_Session], report: dict[str, Any]) -> dict[str, str]:
    entries = [
        {
            "source": s.parsed.source,
            "started_at": s.data["started_at"],
            "name": s.name,
            "sha256": hashlib.sha256(s.raw).hexdigest(),
        }
        for s in sessions
    ]
    return {
        **{f"attributes/{s.name}": s.raw.decode("utf-8") for s in sessions},
        "manifest.json": _json({"version": 1, "sessions": entries, "report": report}),
        "sessions.jsonl": "".join(_json(s.data) for s in sessions),
    }


def _matches(output: Path, contents: dict[str, str]) -> bool:
    if output.is_symlink() or not output.is_dir():
        return False
    if {p.name for p in output.iterdir()} != {"attributes", "manifest.json", "sessions.jsonl"}:
        return False
    attributes = output / "attributes"
    if attributes.is_symlink() or not attributes.is_dir():
        return False
    if {f"attributes/{p.name}" for p in attributes.iterdir()} != {
        name for name in contents if name.startswith("attributes/")
    }:
        return False
    return all(
        _read_regular(output / name, len(text.encode("utf-8"))) == text.encode("utf-8")
        for name, text in contents.items()
    )


def _write(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _commit(staging: Path, output: Path) -> None:
    """Rename the complete directory atomically, refusing concurrent destinations."""
    if os.name == "nt":
        # Windows rename already refuses an existing destination.
        os.rename(staging, output)
    elif sys.platform == "linux":
        # POSIX rename may replace an empty directory. Linux's exclusive rename
        # prevents that race without reserving a partial final directory.
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
            raise OSError(ctypes.get_errno(), "exclusive directory commit failed")
    else:
        raise IntakeError("exclusive directory commit is unsupported on this platform")


def _intake(incoming: Path, output: Path) -> dict[str, Any]:
    incoming, output = _paths(incoming, output)
    sessions, duplicates = _load(incoming)
    report = _report(sessions, duplicates)
    contents = _contents(sessions, report)
    if output.exists():
        if _matches(output, contents):
            return report
        raise IntakeError("existing output differs; nothing was overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".label-intake-", dir=output.parent))
    try:
        (staging / "attributes").mkdir()
        for name, text in contents.items():
            _write(staging / name, text)
        if not _matches(staging, contents):
            raise IntakeError("staged output failed verification")
        for line in (staging / "sessions.jsonl").read_bytes().splitlines():
            _parse(line)
        manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        if manifest["report"] != report:
            raise IntakeError("staged manifest failed verification")
        _paths(incoming, output)
        _commit(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def intake(incoming: Path, output: Path) -> dict[str, Any]:
    """Accept one bounded batch, or raise a payload-free typed error."""
    try:
        return _intake(incoming, output)
    except IntakeError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
        OverflowError,
        AttributeError,
    ) as error:
        if isinstance(error, OSError) and error.errno == errno.EEXIST:
            raise IntakeError("output already exists; nothing was overwritten") from None
        raise IntakeError(
            "cannot accept batch: invalid input or local filesystem failure"
        ) from None


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        # argparse otherwise includes arbitrary argument values in diagnostics.
        self.exit(2, "label intake: invalid command line; use --help\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = intake(args.input, args.output)
    except IntakeError as error:
        print(f"label intake: {error}", file=sys.stderr)
        return 1
    print(_json(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
