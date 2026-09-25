#!/usr/bin/env python3
"""Fail if a Python runtime dependency is not permissively licensed (AGENTS.md INV-2).

Walks the runtime dependency closure of the installed `wearreport` distribution (dev
dependencies are not included), including every extra the project declares and every
extra requested along the way, and reads each package's licence from its metadata, in
this order: License-Expression (SPDX), then licence classifiers, then the legacy
License field. A package passes only if its licence is on the allowlist:

  MIT, BSD-2-Clause, BSD-3-Clause, 0BSD, Apache-2.0, ISC, PSF, MPL-2.0, CC0, Zlib

Other BSD variants (e.g. BSD-Protection, BSD-3-Clause-No-Nuclear-*) are rejected.

Anything else, including missing or unrecognised licence metadata, fails and needs a
human decision. `ultralytics` fails by name and must not appear anywhere in uv.lock.

Usage (inside the project environment, e.g. `uv run`):
  python scripts/license_check.py [--project NAME] [--lock PATH]

Exit code 0 = all allowed, 1 = findings.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Iterator, Sequence
from email.message import Message
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_PACKAGES = frozenset({"ultralytics"})

ALLOWED_SPDX = frozenset(
    {
        "MIT",
        "MIT-0",
        "Apache-2.0",
        "ISC",
        "PSF-2.0",
        "Python-2.0",
        "MPL-2.0",
        "0BSD",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "Zlib",
    }
)

# Classifier suffixes after "License :: " that are on the allowlist.
ALLOWED_CLASSIFIERS = frozenset(
    {
        "OSI Approved :: MIT License",
        "OSI Approved :: MIT No Attribution License (MIT-0)",
        "OSI Approved :: BSD License",
        "OSI Approved :: Apache Software License",
        "OSI Approved :: ISC License (ISCL)",
        "OSI Approved :: Python Software Foundation License",
        "OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "OSI Approved :: Zero-Clause BSD (0BSD)",
        "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    }
)

# Common free-text values of the legacy License field, lower-cased.
ALLOWED_ALIASES = frozenset(
    {
        "mit",
        "mit license",
        "bsd",
        "bsd license",
        "new bsd",
        "new bsd license",
        "3-clause bsd",
        "bsd 3-clause",
        "apache 2.0",
        "apache-2",
        "apache license 2.0",
        "apache license, version 2.0",
        "apache software license",
        "isc",
        "isc license",
        "psf",
        "psf license",
        "python software foundation license",
        "mpl 2.0",
        "mozilla public license 2.0",
    }
)

_SPDX_TOKEN = re.compile(r"\(|\)|[A-Za-z0-9.+-]+")


def spdx_allowed(identifier: str) -> bool:
    return identifier in ALLOWED_SPDX


class _SpdxParser:
    """Evaluate an SPDX licence expression: OR needs one allowed branch, AND needs all."""

    def __init__(self, expression: str) -> None:
        self.tokens = _SPDX_TOKEN.findall(expression)
        self.pos = 0

    def parse(self) -> bool:
        result = self._or()
        if self.pos != len(self.tokens):
            raise ValueError("trailing tokens")
        return result

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of expression")
        self.pos += 1
        return token

    def _or(self) -> bool:
        results = [self._and()]
        while self._peek() == "OR":
            self._take()
            results.append(self._and())
        return any(results)

    def _and(self) -> bool:
        results = [self._atom()]
        while self._peek() == "AND":
            self._take()
            results.append(self._atom())
        return all(results)

    def _atom(self) -> bool:
        part = self._take()
        if part == "(":
            result = self._or()
            if self._take() != ")":
                raise ValueError("unbalanced parentheses")
            return result
        if part in ("AND", "OR", "WITH", ")"):
            raise ValueError(f"unexpected {part}")
        if self._peek() == "WITH":  # an exception only adds permissions
            self._take()
            self._take()
        return spdx_allowed(part)


def expression_allowed(expression: str) -> bool | None:
    """True/False for a valid SPDX expression, None if it cannot be parsed."""
    try:
        return _SpdxParser(expression).parse()
    except ValueError:
        return None


def license_problem(meta: metadata.PackageMetadata | Message) -> str | None:
    """Return why a distribution's licence is not allowed, or None if it is allowed."""
    name = canonicalize_name(meta.get("Name", ""))
    if name in FORBIDDEN_PACKAGES:
        return "forbidden package (AGPL-3.0)"
    expression = meta.get("License-Expression")
    if expression:
        verdict = expression_allowed(expression)
        if verdict is None:
            return f"unparseable License-Expression '{expression}'"
        return None if verdict else f"licence '{expression}' not on the allowlist"
    classifiers = [
        c.removeprefix("License :: ")
        for c in meta.get_all("Classifier", [])
        if c.startswith("License :: ")
    ]
    if classifiers:
        # Classifiers do not say whether several licences combine with AND or OR, so
        # every one of them must be allowed.
        bad = [c for c in classifiers if c not in ALLOWED_CLASSIFIERS]
        return f"licence classifier '{bad[0]}' not on the allowlist" if bad else None
    raw = (meta.get("License") or "").strip()
    first_line = raw.splitlines()[0].strip() if raw else ""
    if first_line.lower() in ALLOWED_ALIASES or expression_allowed(first_line):
        return None
    if not first_line:
        return "no licence metadata"
    return f"licence '{first_line[:60]}' not recognised as allowed"


def runtime_distributions(project: str) -> Iterator[metadata.Distribution]:
    """Yield installed distributions in the runtime closure of `project` (excluded).

    The project is expanded with all of its declared extras. A package reached again
    with extras not yet expanded is re-expanded for those extras, so the result does
    not depend on the order of requirements. Requirements whose markers do not match
    this environment (e.g. another platform) are not followed.
    """
    root = metadata.distribution(project)
    # Extras already expanded per package; "" stands for the base requirements.
    expanded: dict[str, set[str]] = {
        canonicalize_name(project): {"", *(root.metadata.get_all("Provides-Extra") or [])}
    }
    queue: list[tuple[metadata.Distribution, frozenset[str]]] = [
        (root, frozenset(expanded[canonicalize_name(project)]))
    ]
    while queue:
        dist, extras = queue.pop()
        for line in dist.requires or []:
            req = Requirement(line)
            if req.marker is not None and not any(
                req.marker.evaluate({"extra": extra}) for extra in extras
            ):
                continue
            if req.marker is None and "" not in extras:
                continue  # base requirement, already followed
            key = canonicalize_name(req.name)
            wanted = {"", *req.extras}
            done = expanded.get(key)
            if done is not None and wanted <= done:
                continue
            child = metadata.distribution(req.name)
            if done is None:
                done = expanded[key] = set()
                yield child
            new = frozenset(wanted - done)
            done |= new
            queue.append((child, new))


def locked_packages(lock: Path) -> set[str]:
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    return {canonicalize_name(p["name"]) for p in data.get("package", [])}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--project", default="wearreport", help="distribution to check")
    ap.add_argument("--lock", type=Path, default=ROOT / "uv.lock", help="uv lockfile")
    args = ap.parse_args(argv)

    findings: list[str] = []
    lock: Path = args.lock
    for name in sorted(locked_packages(lock) & FORBIDDEN_PACKAGES):
        findings.append(f"{name}: forbidden package present in {lock.name}")
    count = 0
    try:
        for dist in runtime_distributions(args.project):
            count += 1
            meta = dist.metadata
            problem = license_problem(meta)
            label = f"{meta['Name']} {meta['Version']}"
            if problem:
                findings.append(f"{label}: {problem}")
            else:
                print(f"license_check: ok  {label}")
    except metadata.PackageNotFoundError as exc:
        findings.append(f"not installed: {exc} (run `uv sync`)")

    if findings:
        print("license_check: runtime dependencies violate INV-2:")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print(f"license_check: clean ({count} runtime dependencies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
