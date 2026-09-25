#!/usr/bin/env python3
"""Fail if engine code could write images to disk (AGENTS.md INV-1).

The engine only ever writes text (JSON Lines, logs), so binary file writes are denied by
default. Parses every Python file under engine/ (except engine/tests/) and flags:

  1. image writers: any call named imwrite/imsave/mimwrite/mimsave (cv2, imageio, ...)
  2. `.save(...)` with any argument (PIL.Image.Image.save, numpy.save, model saves...);
     `.save()` with no argument is allowed
  3. binary write modes (a mode containing "b" and one of "w", "a", "x", "+") on
     open(), io.open(), codecs.open(), gzip/bz2/lzma.open(), os.fdopen(), and on any
     `.open(mode)` method (Path.open), plus io.FileIO() in any write mode; a computed
     mode on these functions counts as binary, since it cannot be proven otherwise
  4. `.write_bytes(...)` and `.tofile(...)` on anything
  5. urlretrieve() (urllib.request, any destination), numpy savez/savez_compressed
  6. tempfile.* called with an image suffix, and tempfile.NamedTemporaryFile,
     TemporaryFile and SpooledTemporaryFile in a binary mode (their default is "w+b")
  7. an image path (see below) opened in any write mode or passed to write_text(), and
     os.path.join() whose last part ends in an image extension

Names are resolved through `import ... as`, `from ... import ... as` and simple
module-level or local rebinding (`w = cv2.imwrite`), so `w(...)` is checked as
`cv2.imwrite(...)`.

Allowlist: rule 3, rule 4's write_bytes and rule 6's binary-mode case can be allowed for a
whole module by adding its repository-relative path (e.g. "engine/wearreport/x.py") to
BINARY_WRITE_ALLOWLIST. It is empty. Adding an entry needs the maintainer's review in the
PR that adds it, with a justification in the PR description of why the module must write
binary data and why that data cannot contain frame or crop pixels. The allowlist never
exempts image writers, image paths, `.save()`, `.tofile()`, urlretrieve or numpy saves.

What it cannot see: the check is static. An image path is recognised only when the
extension is written in the source (string literals, f-strings, `/` and `+` joins,
Path(), joinpath(), with_suffix(), with_name(), str.format(), os.path.join()). It does
not see calls made through getattr() or other dynamic dispatch, writers passed in as
arguments or stored in containers, low-level os.open()/os.write() on descriptors,
writes performed inside third-party code, or a `.open(x)` method whose positional
argument is computed or not a mode string, unless the receiver is visibly an image path
(this is indistinguishable from PIL's Image.open(buffer), which only reads). Reviewers
remain responsible for those.

Usage:
  python scripts/privacy_guard.py [--root DIR]

Exit code 0 = clean, 1 = findings, 2 = usage error. Standard library only.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# Repository-relative paths of modules allowed to write binary files. See the docstring.
BINARY_WRITE_ALLOWLIST: frozenset[str] = frozenset()

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")
IMAGE_WRITERS = frozenset({"imwrite", "imsave", "mimwrite", "mimsave"})
PATH_CONSTRUCTORS = frozenset({"Path", "PurePath", "PosixPath", "PurePosixPath"})
WRITE_MODE_CHARS = frozenset("wax+")
MODE_PATTERN = re.compile(r"[rwxabtU+]{1,4}")
# Functions taking (file, mode, ...), with the default mode being read-only.
OPEN_FUNCTIONS = frozenset(
    {"open", "io.open", "codecs.open", "gzip.open", "bz2.open", "lzma.open", "os.fdopen"}
)
OS_PATH_JOIN = frozenset({"os.path.join", "posixpath.join", "ntpath.join"})
NUMPY_SAVERS = frozenset({"numpy.save", "numpy.savez", "numpy.savez_compressed"})
# tempfile functions whose default mode is "w+b".
TEMPFILE_BINARY_DEFAULT = frozenset({"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"})


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _dotted(node: ast.expr) -> str | None:
    """`a.b.c` for a chain of attributes on a name, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Map local names to the qualified names they are bound to by imports or rebinding."""
    aliases: dict[str, str] = {}
    rebinds: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], _dotted(node.value)
            if isinstance(target, ast.Name) and value:
                rebinds.append((target.id, value))
    # Resolve rebinding chains such as `w = cv2.imwrite; v = w`.
    for _ in range(len(rebinds)):
        changed = False
        for name, value in rebinds:
            resolved = _resolve(value, aliases)
            if aliases.get(name) != resolved and resolved != name:
                aliases[name] = resolved
                changed = True
        if not changed:
            break
    return aliases


def _resolve(dotted: str, aliases: dict[str, str]) -> str:
    head, _, rest = dotted.partition(".")
    base = aliases.get(head, head)
    return f"{base}.{rest}" if rest else base


class _Checker:
    def __init__(self, tree: ast.AST, allow_binary: bool) -> None:
        self.aliases = _aliases(tree)
        self.allow_binary = allow_binary

    def qualname(self, func: ast.expr) -> str | None:
        dotted = _dotted(func)
        return _resolve(dotted, self.aliases) if dotted else None

    def call_name(self, func: ast.expr) -> str | None:
        """Last segment of the resolved name, or the method name on any expression."""
        qual = self.qualname(func)
        if qual:
            return qual.rpartition(".")[2]
        return func.attr if isinstance(func, ast.Attribute) else None

    def is_image_path(self, node: ast.expr | None) -> bool:
        """True if `node` is a path expression that visibly ends in an image extension."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.lower().endswith(IMAGE_SUFFIXES)
        if isinstance(node, ast.JoinedStr):
            return bool(node.values) and self.is_image_path(node.values[-1])
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
            return self.is_image_path(node.right)
        if isinstance(node, ast.Call):
            name = self.call_name(node.func)
            if name in PATH_CONSTRUCTORS or name == "joinpath" or self.is_os_path_join(node):
                return bool(node.args) and self.is_image_path(node.args[-1])
            if name in ("with_suffix", "with_name"):
                return bool(node.args) and self.is_image_path(node.args[0])
            if name == "format" and isinstance(node.func, ast.Attribute):
                return self.is_image_path(node.func.value)
        return False

    def is_os_path_join(self, call: ast.Call) -> bool:
        return self.qualname(call.func) in OS_PATH_JOIN

    def binary_write(self, reason: str) -> str | None:
        return None if self.allow_binary else f"{reason} (binary writes are not allowed)"

    def check(self, call: ast.Call) -> str | None:
        func = call.func
        qual = self.qualname(func)
        name = self.call_name(func)

        if name in IMAGE_WRITERS:
            return f"image writer call '{name}()'"
        if qual in NUMPY_SAVERS or name in ("savez", "savez_compressed"):
            return f"numpy file writer '{name}()'"
        if name == "save" and (call.args or call.keywords):
            return "'.save()' with arguments may write an image"
        if name == "urlretrieve":
            return "'urlretrieve()' writes a download to disk"
        if name == "tofile" and isinstance(func, ast.Attribute):
            return "'.tofile()' writes array bytes to disk"
        if self.is_os_path_join(call) and call.args and self.is_image_path(call.args[-1]):
            return "'os.path.join()' builds an image path"
        if qual and qual.startswith("tempfile."):
            return self.check_tempfile(call, qual.removeprefix("tempfile."))
        if qual in OPEN_FUNCTIONS or qual == "builtins.open":
            return self.check_open(_arg(call, 0, "file"), _arg(call, 1, "mode"))
        if qual in ("io.FileIO", "_io.FileIO"):
            mode = _arg(call, 1, "mode")
            if _mode_writes(mode, binary_only=False, default=False):
                return self.binary_write("'io.FileIO()' opened for writing")
            return None
        if isinstance(func, ast.Attribute):
            if func.attr == "open":
                mode = _keyword(call, "mode")
                first = call.args[0] if call.args else None
                if mode is None and (_looks_like_mode(first) or self.is_image_path(func.value)):
                    mode = first  # Path.open(mode)
                # Otherwise the first argument is not a mode, e.g. Image.open(buffer).
                return self.check_open(func.value, mode)
            if func.attr == "write_bytes":
                if self.is_image_path(func.value):
                    return "'write_bytes()' to an image path"
                return self.binary_write("'.write_bytes()'")
            if func.attr == "write_text" and self.is_image_path(func.value):
                return "'write_text()' to an image path"
        return None

    def check_open(self, target: ast.expr | None, mode: ast.expr | None) -> str | None:
        if self.is_image_path(target) and _mode_writes(mode, binary_only=False, default=False):
            return "image path opened for writing"
        if _mode_writes(mode, binary_only=True, default=False):
            return self.binary_write("file opened in a binary write mode")
        return None

    def check_tempfile(self, call: ast.Call, func: str) -> str | None:
        for key in ("suffix", "prefix"):
            if self.is_image_path(_keyword(call, key)):
                return f"'tempfile.{func}()' with an image {key}"
        if func in TEMPFILE_BINARY_DEFAULT:
            mode = _arg(call, 0, "mode")
            if _mode_writes(mode, binary_only=True, default=True):
                return self.binary_write(f"'tempfile.{func}()' in a binary mode")
        return None


def _looks_like_mode(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and MODE_PATTERN.fullmatch(node.value) is not None
    )


def _mode_writes(node: ast.expr | None, *, binary_only: bool, default: bool) -> bool:
    """Whether a mode argument writes (and, if `binary_only`, in binary)."""
    if node is None:
        return default
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        mode = node.value
        return bool(WRITE_MODE_CHARS & set(mode)) and (not binary_only or "b" in mode)
    return True  # computed mode: cannot prove it is safe


def _keyword(call: ast.Call, keyword: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _arg(call: ast.Call, index: int, keyword: str) -> ast.expr | None:
    value = _keyword(call, keyword)
    if value is not None:
        return value
    return call.args[index] if len(call.args) > index else None


def scan_source(source: str, filename: str) -> list[Finding]:
    """Return privacy findings for one Python source file."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [Finding(filename, exc.lineno or 0, f"cannot parse: {exc.msg}")]
    checker = _Checker(tree, allow_binary=filename in BINARY_WRITE_ALLOWLIST)
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            message = checker.check(node)
            if message:
                findings.append(Finding(filename, node.lineno, message))
    return sorted(findings, key=lambda f: f.line)


def engine_files(root: Path) -> Iterator[Path]:
    engine = root / "engine"
    tests = engine / "tests"
    for path in sorted(engine.rglob("*.py")):
        if not path.is_relative_to(tests):
            yield path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: this script's repository)",
    )
    args = ap.parse_args(argv)
    root: Path = args.root
    if not (root / "engine").is_dir():
        print(f"privacy_guard: no engine/ directory under {root}", file=sys.stderr)
        return 2
    findings: list[Finding] = []
    count = 0
    for path in engine_files(root):
        count += 1
        rel = path.relative_to(root).as_posix()
        findings += scan_source(path.read_text(encoding="utf-8"), rel)
    if findings:
        print("privacy_guard: engine code may write images (INV-1):")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print(f"privacy_guard: clean ({count} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
