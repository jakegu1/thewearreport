#!/usr/bin/env python3
"""Fail if private project material or images would be published by this repository.

This repository is public (or about to be). Planning documents live in a private
repository, are written largely in Chinese, and use known file names. This guard
checks for those signals:

  1. private paths (OWNER.md, tasks/, seo/, research/, ...)
  2. CJK text (engine files are English-only, so CJK means a leaked private doc)
  3. private-only terms (e.g. the SEO tool's name)
  4. image files outside fixtures/ (INV-1: the engine never writes images)
  5. files larger than 5 MB (accidental dumps)
  6. with --history: commit messages (CJK text or private terms), which become public too

Usage:
  python tools/public_guard.py            # check files tracked in the working tree
  python tools/public_guard.py --history  # check every blob in every commit (before going public)

Exit code 0 = clean, 1 = findings, 2 = usage error. Standard library only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath

PRIVATE_NAMES = {"OWNER.md", "QUALITY.md", "PROTOCOL.md"}
PRIVATE_DIRS = {"tasks", "seo", "research", "decisions", "briefs", "drafts"}
PRIVATE_SUFFIXES = (".ahrefs", ".skill")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")
MAX_BYTES = 5 * 1024 * 1024

# CJK ideographs, CJK punctuation and full-width forms (built from code points so
# that this file itself stays ASCII).
CJK_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0x3000, 0x303F), (0xFF00, 0xFFEF))
CJK = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in CJK_RANGES) + "]")
PRIVATE_TERMS = re.compile(r"\bahrefs\b|keyword[ -]map|affiliate revenue", re.IGNORECASE)

# Commit messages are published with the repository; planning context must stay out of them.
MESSAGE_TERMS = re.compile(
    r"\bahrefs\b|keyword[ -]map|affiliate revenue|\bOWNER\.md\b|\bQUALITY\.md\b|\bseo/|\bH-\d{1,2}\b",
    re.IGNORECASE,
)

# Files allowed to *mention* private names/terms (they explain the rule itself).
MENTION_ALLOWED = {"AGENTS.md", "tools/public_guard.py", ".gitignore"}


def path_findings(path: str) -> list[str]:
    p = PurePosixPath(path)
    out = []
    if p.name in PRIVATE_NAMES:
        out.append("private file name")
    if any(part in PRIVATE_DIRS for part in p.parts[:-1]):
        out.append("private directory")
    if p.name.endswith(PRIVATE_SUFFIXES):
        out.append("private file type")
    if p.name == ".env" or (p.name.startswith(".env.") and p.name != ".env.example"):
        out.append("env file")
    if p.suffix.lower() in IMAGE_SUFFIXES and (not p.parts or p.parts[0] != "fixtures"):
        out.append("image outside fixtures/")
    return out


def content_findings(path: str, data: bytes) -> list[str]:
    out = []
    if len(data) > MAX_BYTES:
        out.append(f"larger than {MAX_BYTES // (1024 * 1024)} MB")
    if b"\0" in data[:8000]:
        return out  # binary: size and path checks only
    text = data.decode("utf-8", errors="replace")
    m = CJK.search(text)
    if m:
        line = text.count("\n", 0, m.start()) + 1
        out.append(f"CJK text at line {line}")
    if path not in MENTION_ALLOWED:
        m = PRIVATE_TERMS.search(text)
        if m:
            line = text.count("\n", 0, m.start()) + 1
            out.append(f"private term '{m.group(0)}' at line {line}")
    return out


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def check_tree() -> list[str]:
    findings = []
    for path in git("ls-files", "-z").decode().split("\0"):
        if not path:
            continue
        problems = path_findings(path)
        try:
            with open(path, "rb") as fh:
                problems += content_findings(path, fh.read())
        except FileNotFoundError:
            continue  # deleted in the working tree
        findings += [f"{path}: {p}" for p in problems]
    return findings


def check_history() -> list[str]:
    findings = []
    seen: set[str] = set()
    for line in git("rev-list", "--objects", "--all").decode().splitlines():
        sha, _, path = line.partition(" ")
        if not path or sha in seen:
            continue
        seen.add(sha)
        if git("cat-file", "-t", sha).strip() != b"blob":
            continue
        problems = path_findings(path) + content_findings(path, git("cat-file", "blob", sha))
        findings += [f"{path} ({sha[:8]}): {p}" for p in problems]
    return findings


def check_messages() -> list[str]:
    findings = []
    for sha in git("rev-list", "--all").decode().split():
        msg = git("log", "-1", "--format=%B", sha).decode("utf-8", errors="replace")
        if CJK.search(msg):
            findings.append(f"commit {sha[:8]}: CJK text in message")
        m = MESSAGE_TERMS.search(msg)
        if m:
            findings.append(f"commit {sha[:8]}: private term '{m.group(0)}' in message")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", action="store_true", help="scan every blob in history")
    args = ap.parse_args()
    try:
        findings = check_history() + check_messages() if args.history else check_tree()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"public_guard: must run inside a git repository ({exc})", file=sys.stderr)
        return 2
    if findings:
        print("public_guard: private or unsafe content found:")
        for f in findings:
            print(f"  - {f}")
        return 1
    print("public_guard: clean" + (" (full history)" if args.history else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
