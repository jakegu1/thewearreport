#!/usr/bin/env python3
"""Fail if engine code could write images or video to disk (AGENTS.md INV-1).

The engine only ever writes text (JSON Lines, logs), so binary file writes are denied by
default. Parses every Python file under engine/ (except engine/tests/) and flags:

  1. image, video and array writers, wherever their path comes from: any call named
     imwrite, imsave, mimwrite, mimsave, volwrite, mvolwrite, imwritemulti,
     imwriteanimation, VideoWriter, get_writer, savefig, print_png (and the other
     matplotlib print_<format> methods), savetxt, savez or savez_compressed;
     numpy.save, joblib.dump, urllib's URLopener, cv2.FileStorage opened for writing,
     numpy.memmap and open_memmap in a writable mode, imageio's imopen in any mode
     but read, and `.write(...)` on a cv2.VideoWriter (a name bound to one, or a
     parameter annotated as one)
  2. `.save(...)` with any argument (PIL.Image.Image.save, numpy.save, model saves...);
     `.save()` with no argument is allowed; `.show()` (PIL writes a temporary image file
     to show it), except matplotlib.pyplot.show()
  3. rule A, binary modes on any opener: for every call whose callee is named `open` or
     ends in `open` (open, io.open, codecs.open, os.fdopen, fsspec.open, x.open,
     Path.open...), the mode is read from the second positional argument or `mode=`
     (for a `.open()` method whose first argument is a mode string, from that) and a
     binary write mode (one containing "b" and one of "w", "a", "x", "+") is flagged.
     A computed mode counts as binary, since it cannot be proven otherwise. urlopen()
     and Popen() are not openers. zipfile.ZipFile/PyZipFile, tarfile.open/TarFile,
     gzip/bz2/lzma.open and GzipFile/BZ2File/LZMAFile in any write mode (w, a, x, with
     or without a compression suffix, text modes of the compressors excepted) are
     binary writes. io.FileIO in any write mode, os.open() with any flags other than
     os.O_RDONLY, shelve.open(), dbm.open() for writing and sqlite3.connect() to a file
     are binary writes too
  4. `.write_bytes(...)` and `.tofile(...)` on anything
  5. urlretrieve() (urllib.request, any destination)
  6. tempfile.* called with an image suffix, and tempfile.NamedTemporaryFile,
     TemporaryFile and SpooledTemporaryFile in a binary mode (their default is "w+b")
  7. an image or video path (see below) opened in any write mode or passed to
     write_text(), and os.path.join() whose last part ends in an image extension
  8. rule B, media path literals: any call argument (positional or keyword, including
     the constant parts of an f-string, operands of `+`, `/` and `%`, the receiver of a
     string method such as `.format()`, and the items of a list, tuple, set or dict)
     that is a string literal ending, case-insensitively, in an image or video extension
     (MEDIA_SUFFIXES). Path(...) segments and `.with_suffix(...)` are calls, so their
     literals are checked too. The only exemptions are: arguments of the readers and
     decoders in MEDIA_READERS (cv2.imread, cv2.imdecode, PIL.Image.open, imageio
     imread); the extension argument of cv2.imencode; str.endswith/str.startswith; and
     literals that start with http:// or https:// (URL templates for reads). The file
     argument of builtin open()/io.open() in a read-only mode is also exempt, because a
     read-only open() is a reader (the T-001 acceptance tests read `model.png`)
  9. a reference to a rule-1 writer, or to urlretrieve, that is not called on the spot
     (`map(cv2.imwrite, ...)`, `functools.partial(cv2.imwrite, p)`, `self.w =
     cv2.imwrite`, `getattr(cv2, "imwrite")`), since the call itself is then hidden;
     and `from ... import *`, which hides where names come from. shutil.make_archive()
     is a binary write (rule 3)
 10. outside the exempt module below: importing it, from any engine module (absolute or
     relative imports, and attribute access through its package), and call arguments
     that are string literals naming it (importlib.import_module, runpy), so its
     exemption cannot be borrowed by another module

Image-write exemption: the spot-check tool (AGENTS.md INV-1 exception (c)) renders
detections into a temporary directory it creates and always deletes. Its repository-
relative path is the one constant IMAGE_WRITE_EXEMPTION, and for that exact path, and no
other spelling of it, two checks are relaxed: rule A's binary write modes on file
openers (open(), os.fdopen() and the like) and os.open() with write flags. Everything
else still applies to it: image writers (rule 1), `.save()`/`.show()`, write_bytes and
tofile, urlretrieve, the tempfile rules, archive, database and FileIO writes, image
paths opened for writing (rule 7), media path literals (rule B) and rule 9.

Names are resolved through `import ... as`, `from ... import ... as` and simple
module-level or local rebinding (`w = cv2.imwrite`), so `w(...)` is checked as
`cv2.imwrite(...)`.

Allowlist: rule 3's binary writes, rule 4's write_bytes and rule 6's binary-mode case can
be allowed for a whole module by adding its repository-relative path (e.g.
"engine/wearreport/x.py") to BINARY_WRITE_ALLOWLIST. It is empty. Adding an entry needs
the maintainer's review in the PR that adds it, with a justification in the PR
description of why the module must write binary data and why that data cannot contain
frame or crop pixels. The allowlist never exempts writers (rule 1), image paths,
`.save()`, `.tofile()`, urlretrieve or media path literals (rule B).

What it cannot see: the check is static, so it does not follow values through
variables, containers or third-party code. Known limits, left to a runtime check that a
later task adds (it will run a sweep and assert that no image bytes reach the disk):
  - os.write() on descriptors from elsewhere (e.g. tempfile.mkstemp() without an
    image suffix), and mmap over them; os.open() itself is flagged unless read-only
  - calls made through getattr() with a computed name, or other dynamic dispatch
  - writers held on `self` (other than a VideoWriter), wrapped in functools.partial,
    stored in tuples or other containers, or passed in as arguments: a reference to a
    known writer (rule 9) is flagged where it is taken, but a writer that reaches the
    engine from a caller or a third-party object is not
  - text-mode writes of pixel data (np.array2string, json.dump(frame.tolist(), fh),
    open(p, "w").buffer.write(...)), and sys.stdout.buffer.write(...) into logs
  - ndarray.dump(p), which cannot be told apart from json.dump/pickle.dump by name
  - an image or video path held in a variable or built without a visible extension,
    passed to a function the guard does not know
  - writes performed inside third-party code or a subprocess (e.g. ffmpeg) to a path
    without a visible media extension
Reviewers remain responsible for those.

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
from pathlib import Path, PurePosixPath

# Repository-relative paths of modules allowed to write binary files. See the docstring.
BINARY_WRITE_ALLOWLIST: frozenset[str] = frozenset()
# The one module allowed to write images (INV-1 exception (c)). See the docstring.
IMAGE_WRITE_EXEMPTION = "engine/wearreport/tools/spotcheck.py"

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
)
URL_PREFIXES = ("http://", "https://")
# Writers matched on the last segment of the callee's name, whatever the receiver.
MEDIA_WRITERS = frozenset(
    {
        "imwrite",
        "imsave",
        "mimwrite",
        "mimsave",
        "volwrite",
        "volsave",
        "mvolwrite",
        "mvolsave",
        "imwritemulti",
        "imwriteanimation",
        "VideoWriter",
        "get_writer",
        "savefig",
        "savetxt",
        "savez",
        "savez_compressed",
    }
)
# matplotlib FigureCanvas.print_png(...) and friends.
PRINT_FIGURE = re.compile(r"print_(figure|png|jpe?g|tiff?|webp|gif|raw|rgba|pdf|svg|e?ps)")
# Writers matched on the fully resolved name.
QUALIFIED_WRITERS = frozenset(
    {
        "numpy.save",
        "joblib.dump",
        "joblib.numpy_pickle.dump",
        "urllib.request.URLopener",
        "urllib.request.FancyURLopener",
    }
)
# Writers that rule 9 flags when referenced without being called.
REFERENCED_WRITERS = frozenset({"urlretrieve"})
# Rule B exemptions: readers and decoders whose arguments may name media files.
MEDIA_READERS = frozenset(
    {
        "cv2.imread",
        "cv2.imdecode",
        "PIL.Image.open",
        "imageio.imread",
        "imageio.v2.imread",
        "imageio.v3.imread",
    }
)
BUILTIN_OPEN = frozenset({"open", "builtins.open", "io.open"})
# Callees ending in "open" whose second argument is not a file mode.
NOT_FILE_OPENERS = frozenset({"urlopen", "Popen"})
# Archive and compressed-file openers: every write mode writes binary data.
ARCHIVE_OPENERS = frozenset(
    {
        "zipfile.ZipFile",
        "zipfile.PyZipFile",
        "tarfile.open",
        "tarfile.TarFile",
        "tarfile.TarFile.open",
        "gzip.open",
        "gzip.GzipFile",
        "bz2.open",
        "bz2.BZ2File",
        "lzma.open",
        "lzma.LZMAFile",
    }
)
ARCHIVE_CLASSES = frozenset({"ZipFile", "PyZipFile", "TarFile", "GzipFile", "BZ2File", "LZMAFile"})
DBM_OPEN = re.compile(r"dbm(\.\w+)?\.open")
# numpy.memmap(filename, dtype, mode) and numpy.lib.format.open_memmap(filename, mode).
MEMMAP_MODE_POSITION = {"memmap": 2, "open_memmap": 1}
MEMMAP_READ_MODES = frozenset({"r", "c", "readonly", "copyonwrite"})
PATH_CONSTRUCTORS = frozenset({"Path", "PurePath", "PosixPath", "PurePosixPath"})
WRITE_MODE_CHARS = frozenset("wax+")
MODE_PATTERN = re.compile(r"[rwxabtU+]{1,4}")
OS_PATH_JOIN = frozenset({"os.path.join", "posixpath.join", "ntpath.join"})
# tempfile functions whose default mode is "w+b".
TEMPFILE_BINARY_DEFAULT = frozenset({"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"})


def _package_path(path: str) -> tuple[str, ...] | None:
    """The dotted-name parts of an engine module ('engine/a/b.py' -> ('a', 'b'))."""
    parts = PurePosixPath(path).with_suffix("").parts
    if len(parts) < 2 or parts[0] != "engine":
        return None
    return parts[1:]


# Rule 10: the exempt module's dotted name and its path as a string literal may name it.
EXEMPT_MODULE = ".".join(_package_path(IMAGE_WRITE_EXEMPTION) or ())
EXEMPT_FILE_TAIL = IMAGE_WRITE_EXEMPTION.removeprefix("engine/")


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


def _is_media_name(value: str) -> bool:
    return value.lower().endswith(MEDIA_SUFFIXES)


def _is_url(node: ast.expr) -> bool:
    """True if a string expression visibly starts with an http(s) URL literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower().startswith(URL_PREFIXES)
    if isinstance(node, ast.JoinedStr):
        return bool(node.values) and _is_url(node.values[0])
    if isinstance(node, ast.BinOp):
        return _is_url(node.left)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _is_url(node.func.value)  # "https://.../{}.jpg".format(...)
    return False


def _media_literal(node: ast.expr) -> str | None:
    """A string literal inside an argument expression that names a media file, if any."""
    if _is_url(node):
        return None
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) and _is_media_name(node.value) else None
    parts: list[ast.expr] = []
    if isinstance(node, ast.JoinedStr):
        parts = list(node.values)
    elif isinstance(node, ast.BinOp):
        parts = [node.left, node.right]
    elif isinstance(node, ast.IfExp):
        parts = [node.body, node.orelse]
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        parts = list(node.elts)
    elif isinstance(node, ast.Dict):
        parts = [k for k in node.keys if k is not None] + list(node.values)
    elif isinstance(node, ast.Starred):
        parts = [node.value]
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        receiver = node.func.value  # a string method such as "{}.jpg".format(n)
        if isinstance(receiver, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            parts = [receiver]
    for part in parts:
        found = _media_literal(part)
        if found:
            return found
    return None


def _str_constant(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _Checker:
    def __init__(self, tree: ast.AST, filename: str) -> None:
        self.aliases = _aliases(tree)
        self.allow_binary = filename in BINARY_WRITE_ALLOWLIST
        self.image_writes = filename == IMAGE_WRITE_EXEMPTION
        module = _package_path(filename)
        # The package that relative imports start from.
        self.package = None if module is None else module[:-1]
        self.video_writers = self._video_writer_names(tree)
        self.exempt = self._rule_b_exempt(tree)

    def qualname(self, func: ast.expr) -> str | None:
        dotted = _dotted(func)
        return _resolve(dotted, self.aliases) if dotted else None

    def call_name(self, func: ast.expr) -> str | None:
        """Last segment of the resolved name, or the method name on any expression."""
        qual = self.qualname(func)
        if qual:
            return qual.rpartition(".")[2]
        return func.attr if isinstance(func, ast.Attribute) else None

    def is_video_writer_call(self, node: ast.expr | None) -> bool:
        return isinstance(node, ast.Call) and self.call_name(node.func) == "VideoWriter"

    def _video_writer_names(self, tree: ast.AST) -> frozenset[str]:
        """Dotted names bound to a cv2.VideoWriter (assignment, `with`, annotation)."""
        names: set[str] = set()
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign) and self.is_video_writer_call(node.value):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign) and (
                self.is_video_writer_call(node.value) or self.names_video_writer(node.annotation)
            ):
                targets = [node.target]
            elif isinstance(node, ast.withitem) and self.is_video_writer_call(node.context_expr):
                targets = [node.optional_vars] if node.optional_vars else []
            elif isinstance(node, ast.arg) and self.names_video_writer(node.annotation):
                names.add(node.arg)
            names.update(d for d in map(_dotted, targets) if d)
        return frozenset(names)

    def names_video_writer(self, annotation: ast.expr | None) -> bool:
        if annotation is None:
            return False
        text = _str_constant(annotation) or ast.unparse(annotation)
        return "VideoWriter" in text

    def is_image_path(self, node: ast.expr | None) -> bool:
        """True if `node` is a path expression that visibly ends in a media extension."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _is_media_name(node.value)
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

    def binary_write(self, reason: str, *, opener: bool = False) -> str | None:
        """`opener`: a binary open or os.open(), which the image-write exemption allows."""
        if self.allow_binary or (opener and self.image_writes):
            return None
        return f"{reason} (binary writes are not allowed)"

    def check(self, call: ast.Call) -> str | None:
        """Rules 1 to 7; see the module docstring."""
        func = call.func
        qual = self.qualname(func)
        name = self.call_name(func)

        writer = self.check_writer(call, qual, name)
        if writer:
            return writer
        if name == "save" and (call.args or call.keywords):
            return "'.save()' with arguments may write an image"
        if name == "show" and isinstance(func, ast.Attribute) and qual != "matplotlib.pyplot.show":
            return "'.show()' writes a temporary image file"
        if name == "urlretrieve":
            return "'urlretrieve()' writes a download to disk"
        if name == "tofile" and isinstance(func, ast.Attribute):
            return "'.tofile()' writes array bytes to disk"
        if self.is_os_path_join(call) and call.args and self.is_image_path(call.args[-1]):
            return "'os.path.join()' builds an image path"
        if qual and qual.startswith("tempfile."):
            return self.check_tempfile(call, qual.removeprefix("tempfile."))
        if qual in ARCHIVE_OPENERS or name in ARCHIVE_CLASSES:
            if _archive_mode_writes(_mode_arg(call, 1, "mode")):
                return self.binary_write(f"'{name}()' opened for writing")
            return None
        if qual in ("io.FileIO", "_io.FileIO"):
            mode = _mode_arg(call, 1, "mode")
            if _mode_writes(mode, binary_only=False, default=False):
                return self.binary_write("'io.FileIO()' opened for writing")
            return None
        if qual == "shutil.make_archive":
            return self.binary_write("'shutil.make_archive()' writes an archive")
        if qual in ("shelve.open", "sqlite3.connect") or DBM_OPEN.fullmatch(qual or ""):
            return self.check_database(call, qual or "")
        if qual == "os.open":
            flags = _arg(call, 1, "flags")
            if flags is None or self.qualname(flags) != "os.O_RDONLY":
                return self.binary_write("'os.open()' with write flags", opener=True)
            return None
        if name and name.endswith("open") and name not in NOT_FILE_OPENERS:
            if qual in MEDIA_READERS:
                return None
            target, mode, _ = self.opener_args(call)
            return self.check_open(target, mode)
        if isinstance(func, ast.Attribute):
            if func.attr == "write" and (
                self.is_video_writer_call(func.value) or _dotted(func.value) in self.video_writers
            ):
                return "'.write()' on a cv2.VideoWriter"
            if func.attr == "write_bytes":
                if self.is_image_path(func.value):
                    return "'write_bytes()' to an image path"
                return self.binary_write("'.write_bytes()'")
            if func.attr == "write_text" and self.is_image_path(func.value):
                return "'write_text()' to an image path"
        return None

    def check_writer(self, call: ast.Call, qual: str | None, name: str | None) -> str | None:
        """Rule 1: calls that write images, video or arrays wherever the path comes from."""
        if name in MEDIA_WRITERS or (name and PRINT_FIGURE.fullmatch(name)):
            return f"image, video or array writer '{name}()'"
        if qual in QUALIFIED_WRITERS:
            return f"file writer '{qual}()'"
        if name == "imopen":
            io_mode = _str_constant(_arg(call, 1, "io_mode"))
            if io_mode is None or not io_mode.startswith("r"):
                return "'imopen()' not proven to be read-only"
        if name == "FileStorage" and len(call.args) + len(call.keywords) > 1:
            flags = _arg(call, 1, "flags")
            if flags is None or not (self.qualname(flags) or "").endswith("FILE_STORAGE_READ"):
                return "'cv2.FileStorage()' opened for writing"
        position = MEMMAP_MODE_POSITION.get(name or "")
        if position is not None:
            mode = _str_constant(_arg(call, position, "mode"))
            if mode not in MEMMAP_READ_MODES:  # the default "r+" writes
                return f"'{name}()' maps a file for writing"
        return None

    def check_database(self, call: ast.Call, qual: str) -> str | None:
        if qual == "sqlite3.connect":
            if _str_constant(_arg(call, 0, "database")) == ":memory:":
                return None
            return self.binary_write("'sqlite3.connect()' to a database file")
        flag = _str_constant(_arg(call, 1, "flag"))
        default = "r" if qual in ("dbm.open", "dbm.gnu.open", "dbm.ndbm.open") else "c"
        if (flag if _arg(call, 1, "flag") is not None else default) != "r":
            return self.binary_write(f"'{qual}()' opened for writing")
        return None

    def opener_args(self, call: ast.Call) -> tuple[ast.expr | None, ast.expr | None, bool]:
        """(file, mode, file_is_argument) for an `open`-like call (rule A)."""
        func = call.func
        explicit = _keyword(call, "mode")
        if isinstance(func, ast.Attribute):
            first = call.args[0] if call.args else None
            path_receiver = self.is_image_path(func.value)
            if explicit is None and (_looks_like_mode(first) or path_receiver):
                return func.value, _as_mode(first), False  # Path.open(mode)
            if path_receiver:
                return func.value, explicit, False
        # Otherwise the first argument is the file, e.g. fsspec.open(p, "wb") or
        # Image.open(buffer), which only reads.
        return _arg(call, 0, "file"), _mode_arg(call, 1, "mode"), True

    def check_open(self, target: ast.expr | None, mode: ast.expr | None) -> str | None:
        if self.is_image_path(target) and _mode_writes(mode, binary_only=False, default=False):
            return "image path opened for writing"
        if _mode_writes(mode, binary_only=True, default=False):
            return self.binary_write("file opened in a binary write mode", opener=True)
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

    def _rule_b_exempt(self, tree: ast.AST) -> frozenset[int]:
        """ids of every node inside an argument that rule B must not inspect."""
        roots: list[ast.expr] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            qual = self.qualname(node.func)
            name = self.call_name(node.func)
            if qual in MEDIA_READERS or (
                name in ("endswith", "startswith") and isinstance(node.func, ast.Attribute)
            ):
                roots += node.args + [kw.value for kw in node.keywords]
            elif qual == "cv2.imencode":
                ext = _arg(node, 0, "ext")
                roots += [ext] if ext is not None else []
            elif qual in BUILTIN_OPEN:
                target, mode, is_argument = self.opener_args(node)
                literal = mode is None or _str_constant(mode) is not None
                reads = literal and not _mode_writes(mode, binary_only=False, default=False)
                if is_argument and target is not None and reads:
                    roots.append(target)
        return frozenset(id(n) for root in roots for n in ast.walk(root))

    def check_reference(self, node: ast.expr) -> str | None:
        """Rule 9: a writer referenced without being called, e.g. map(cv2.imwrite, ...)."""
        name: str | None
        if isinstance(node, ast.Call):  # getattr(cv2, "imwrite")
            if self.qualname(node.func) not in ("getattr", "builtins.getattr"):
                return None
            name = _str_constant(_arg(node, 1, "name"))
            qual = f"{_dotted(node.args[0]) or '?'}.{name}" if node.args and name else None
        else:
            qual = self.qualname(node)
            name = qual.rpartition(".")[2] if qual else None
        if name in MEDIA_WRITERS or name in REFERENCED_WRITERS or qual in QUALIFIED_WRITERS:
            return f"reference to writer '{qual}' hides its calls"
        return None

    def check_exempt_module_use(self, node: ast.AST) -> str | None:
        """Rule 10: another module importing or naming the image-write exempt module."""
        if self.image_writes:
            return None
        message = f"uses '{EXEMPT_MODULE}', the only module allowed to write images"
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = self.import_base(node)
            names = [f"{base}.{a.name}" if base else a.name for a in node.names]
            names += [base] if base else []
        elif isinstance(node, ast.Call):
            names = [self.qualname(node.func) or ""]
            for arg in node.args + [kw.value for kw in node.keywords]:
                text = _str_constant(arg)
                if text and (EXEMPT_MODULE in text or EXEMPT_FILE_TAIL in text):
                    return message
        elif isinstance(node, (ast.Name, ast.Attribute)):
            names = [self.qualname(node) or ""]
        if any(n == EXEMPT_MODULE or n.startswith(EXEMPT_MODULE + ".") for n in names):
            return message
        return None

    def import_base(self, node: ast.ImportFrom) -> str | None:
        """The absolute module a `from ... import` reads from, if it can be resolved."""
        if node.level == 0:
            return node.module
        if self.package is None or node.level - 1 > len(self.package):
            return None
        base = list(self.package[: len(self.package) - (node.level - 1)])
        base += node.module.split(".") if node.module else []
        return ".".join(base)

    def check_media_literal(self, call: ast.Call) -> str | None:
        """Rule B: a call argument that is a literal naming an image or video file."""
        for arg in call.args + [kw.value for kw in call.keywords]:
            if id(arg) in self.exempt:
                continue
            literal = _media_literal(arg)
            if literal:
                return f"argument {literal!r} names an image or video file"
        return None


def _looks_like_mode(node: ast.expr | None) -> bool:
    value = _str_constant(node)
    return value is not None and MODE_PATTERN.fullmatch(value) is not None


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
    for position, arg in enumerate(call.args[: index + 1]):
        if isinstance(arg, ast.Starred):
            return arg  # open(*args): unknown, counts as computed
        if position == index:
            return arg
    splat = [kw.value for kw in call.keywords if kw.arg is None]
    return splat[0] if splat else None  # open(p, **kw): unknown, counts as computed


def _as_mode(node: ast.expr | None) -> ast.expr | None:
    """A non-string constant (e.g. webbrowser.open(url, 2)) is not a mode."""
    if isinstance(node, ast.Constant) and not isinstance(node.value, str):
        return None
    return node


def _mode_arg(call: ast.Call, index: int, keyword: str) -> ast.expr | None:
    return _as_mode(_arg(call, index, keyword))


def _archive_mode_writes(node: ast.expr | None) -> bool:
    """Any write mode of an archive or compressor ("w", "a:gz", "x|bz2", "wb"), not "wt"."""
    if node is None:
        return False
    value = _str_constant(node)
    if value is None:
        return True  # computed mode: cannot prove it is safe
    head = value.split(":")[0].split("|")[0]
    return head[:1] in ("w", "a", "x") and "t" not in head


def scan_source(source: str, filename: str) -> list[Finding]:
    """Return privacy findings for one Python source file."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [Finding(filename, exc.lineno or 0, f"cannot parse: {exc.msg}")]
    checker = _Checker(tree, filename)
    # Callees are checked as calls, annotations only name types, and `a.b` is checked
    # once as a whole rather than again as `a`.
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            skip.add(id(node.func))
        elif isinstance(node, ast.Attribute):
            skip.add(id(node.value))
        for annotation in _annotations(node):
            skip.update(id(n) for n in ast.walk(annotation))
    findings = []
    for node in ast.walk(tree):
        message = None
        if isinstance(node, ast.Call):
            message = (
                checker.check(node)
                or checker.check_media_literal(node)
                or checker.check_reference(node)
                or checker.check_exempt_module_use(node)
            )
        elif isinstance(node, (ast.Name, ast.Attribute)) and id(node) not in skip:
            if isinstance(node.ctx, ast.Load):
                message = checker.check_reference(node) or checker.check_exempt_module_use(node)
        elif isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            message = "'from ... import *' hides names from the privacy guard"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            message = checker.check_exempt_module_use(node)
        if message:
            findings.append(Finding(filename, getattr(node, "lineno", 0), message))
    return sorted(findings, key=lambda f: f.line)


def _annotations(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, ast.arg | ast.AnnAssign):
        return [node.annotation] if node.annotation else []
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return [node.returns] if node.returns else []
    return []


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
        print("privacy_guard: engine code may write images or video (INV-1):")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print(f"privacy_guard: clean ({count} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
