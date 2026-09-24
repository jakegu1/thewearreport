#!/usr/bin/env python3
"""Fail if engine code could write images to disk (AGENTS.md INV-1).

Parses every Python file under engine/ (except engine/tests/) and flags:

  1. calls to image writers: cv2.imwrite, imageio's imwrite/imsave/mimwrite/mimsave
  2. `.save(...)` calls in a module that imports PIL (covers PIL.Image.Image.save),
     and `.save(...)` anywhere when its first argument is an image path
  3. opening a path that ends in an image extension in a write mode, via open(),
     io.open(), Path.open(), Path.write_bytes() or Path.write_text()

The check is static, so it only sees extensions written in the source (literals,
f-strings, `/` and `+` joins, Path(), with_suffix()). The runtime privacy test covers
what a static check cannot.

Usage:
  python scripts/privacy_guard.py [--root DIR]

Exit code 0 = clean, 1 = findings, 2 = usage error. Standard library only.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")
IMAGE_WRITERS = frozenset({"imwrite", "imsave", "mimwrite", "mimsave"})
PATH_CONSTRUCTORS = frozenset({"Path", "PurePath", "PosixPath", "PurePosixPath"})
WRITE_MODE_CHARS = frozenset("wax+")


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_image_path(node: ast.expr | None) -> bool:
    """True if `node` is a path expression that visibly ends in an image extension."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower().endswith(IMAGE_SUFFIXES)
    if isinstance(node, ast.JoinedStr):
        return bool(node.values) and _is_image_path(node.values[-1])
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        return _is_image_path(node.right)
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name in PATH_CONSTRUCTORS or name == "joinpath":
            return bool(node.args) and _is_image_path(node.args[-1])
        if name in ("with_suffix", "with_name"):
            return bool(node.args) and _is_image_path(node.args[0])
        if name == "format" and isinstance(node.func, ast.Attribute):
            return _is_image_path(node.func.value)
    return False


def _is_write_mode(node: ast.expr | None) -> bool:
    if node is None:
        return False  # default mode is "r"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(WRITE_MODE_CHARS & set(node.value))
    return True  # computed mode: cannot prove it is read-only


def _arg(call: ast.Call, index: int, keyword: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return call.args[index] if len(call.args) > index else None


def _imports_pil(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            a.name == "PIL" or a.name.startswith("PIL.") for a in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "PIL":
            return True
    return False


def _is_builtin_open(func: ast.expr) -> bool:
    """True for `open(...)` and `io.open(...)`."""
    if isinstance(func, ast.Name):
        return func.id == "open"
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "open"
        and isinstance(func.value, ast.Name)
        and func.value.id == "io"
    )


def _check_call(call: ast.Call, uses_pil: bool) -> str | None:
    func = call.func
    name = _call_name(func)
    if name in IMAGE_WRITERS:
        return f"image writer call '{name}()'"
    if name == "save" and isinstance(func, ast.Attribute):
        if uses_pil:
            return "'.save()' in a module that imports PIL"
        if call.args and _is_image_path(call.args[0]):
            return "'.save()' to an image path"
    if _is_builtin_open(func):
        if _is_image_path(_arg(call, 0, "file")) and _is_write_mode(_arg(call, 1, "mode")):
            return "image path opened for writing"
        return None
    if isinstance(func, ast.Attribute) and _is_image_path(func.value):
        if func.attr == "open" and _is_write_mode(_arg(call, 0, "mode")):
            return "image path opened for writing"
        if func.attr in ("write_bytes", "write_text"):
            return f"'{func.attr}()' to an image path"
    return None


def scan_source(source: str, filename: str) -> list[Finding]:
    """Return privacy findings for one Python source file."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [Finding(filename, exc.lineno or 0, f"cannot parse: {exc.msg}")]
    uses_pil = _imports_pil(tree)
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            message = _check_call(node, uses_pil)
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
