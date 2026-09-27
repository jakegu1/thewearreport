"""The spot-check judge's gold set: licensed photos, cropped and degraded in memory.

  python -m wearreport.tools.goldset [--manifest PATH] [--sources DIR]

The gold set is `fixtures/goldset/manifest.json`: openly licensed source photos (their
URL, author, licence and SHA-256) and, for each labelled item, a box in its source photo,
a label (`person`, `in_vehicle` or `not_person`), a kind (pedestrian, cyclist, pole,
depiction...) and the degradation that brings it to TfL JamCam conditions. The source
photos are downloaded by `scripts/fetch_goldset.sh` into `.goldset/` (git-ignored), and
never committed. Run as a script, this module checks every source file and every item,
and prints counts.

Degradation (`degrade`), deterministic for a given item:

1. the box is grown by CROP_MARGIN (50%) on every side and clipped to the photo, exactly
   as the spot-check tool crops a detection (an experiment may pass a wider margin);
2. that region is shrunk (area interpolation) so that the box is `height_px` high,
   15-80 px, the height of a person in a JamCam frame;
3. the result is JPEG-encoded at `jpeg_quality`, 35-60, like a JamCam frame, and decoded.

`degradation_params` draws each item's height and quality from the manifest's seed and
the item's id, and the manifest records them, so a changed manifest is caught.
`render_crop` then draws the box and its number as the spot-check tool does for a live
crop: whole-factor enlargement up to about 240 px, a 1 px green box, a number label.
The judge sees exactly that image.

Privacy (AGENTS.md INV-1): every photo is read as bytes, checked against its pinned
SHA-256 and decoded in memory; nothing here writes a file. The committed control crops
(`fixtures/goldset/controls/`) are the JPEG bytes that `degrade` produces for a few items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from wearreport._cv import cv2

Label = Literal["person", "in_vehicle", "not_person"]
LABELS: tuple[Label, ...] = get_args(Label)
Box = tuple[int, int, int, int]
FloatBox = tuple[float, float, float, float]
Image = npt.NDArray[np.uint8]

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "fixtures" / "goldset" / "manifest.json"
SOURCE_DIR = REPO_ROOT / ".goldset"

# Kinds of item, per label. The hard negatives are what the detector mistakes for people.
HARD_NEGATIVE_KINDS = frozenset({"pole", "bin", "sign", "bollard", "shadow", "depiction"})
KINDS: dict[Label, frozenset[str]] = {
    "person": frozenset({"pedestrian", "cyclist"}),
    "in_vehicle": frozenset({"in_vehicle"}),
    "not_person": HARD_NEGATIVE_KINDS | {"other"},
}

# Degradation to JamCam conditions.
CROP_MARGIN = 0.5  # the spot-check tool's margin: of the box's width and height, each side
MAX_MARGIN = 4.0  # the widest margin an experiment may ask for (T-035)
MIN_HEIGHT_PX, MAX_HEIGHT_PX = 15, 80
MIN_JPEG_QUALITY, MAX_JPEG_QUALITY = 35, 60

# Rendering, as the spot-check tool renders a crop for its reviewer.
CROP_TARGET_HEIGHT = 240
MAX_TARGET_HEIGHT = 1024  # the largest render target an experiment may ask for
MAX_CROP_SCALE = 8
BOX_COLOUR = (0, 255, 0)  # BGR
LABEL_COLOUR = (255, 255, 255)
LABEL_BACKGROUND = (0, 0, 0)

SCREEN_SIZE = 100  # the fixed screening subset
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_ITEMS = 10_000
SHA256 = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
JPEG_SUFFIX = ".jpg"
CONTROL_FILE = re.compile(r"fixtures/goldset/controls/[a-z0-9-]{1,64}" + re.escape(JPEG_SUFFIX))
LICENCE = re.compile(r"CC0 1\.0|Public domain|CC BY \d\.\d")


class GoldsetError(ValueError):
    """The manifest or a source photo is missing, malformed or does not match its pin."""


@dataclass(frozen=True, slots=True)
class Source:
    """One openly licensed source photo."""

    id: str
    url: str  # where scripts/fetch_goldset.sh downloads it
    page: str  # the photo's page, which names its author and licence
    author: str
    license: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Item:
    """One labelled box in a source photo, and how it is degraded."""

    id: str
    source: str
    box: Box  # (x1, y1, x2, y2) in source pixels
    label: Label
    kind: str
    height_px: int
    jpeg_quality: int
    control: str | None = None  # repository-relative path of its committed control crop
    control_sha256: str | None = None

    @property
    def hard_negative(self) -> bool:
        return self.kind in HARD_NEGATIVE_KINDS


@dataclass(frozen=True, slots=True)
class Manifest:
    seed: int
    sources: dict[str, Source]
    items: tuple[Item, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Degraded:
    """An item at JamCam scale: the pixels before JPEG (`small`), the JPEG bytes, the
    decoded frame, and the box in the frame's pixels."""

    small: Image
    jpeg: bytes
    frame: Image
    box: FloatBox


# Manifest --------------------------------------------------------------------------------


def _reject_constant(name: str) -> object:
    raise GoldsetError(f"{name} is not a number")


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise GoldsetError(f"the key {key[:20]!r} appears twice in one object")
        seen[key] = value
    return seen


def _str(obj: dict[str, object], key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise GoldsetError(f"{where}: {key} must be a non-empty string")
    return value


def _int(obj: dict[str, object], key: str, where: str, low: int, high: int) -> int:
    value = obj.get(key)
    if type(value) is not int or not low <= value <= high:
        raise GoldsetError(f"{where}: {key} must be a whole number from {low} to {high}")
    return value


def _object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GoldsetError(f"{where} must be an object")
    return value


def _https(obj: dict[str, object], key: str, where: str) -> str:
    value = _str(obj, key, where)
    if (
        not value.startswith("https://")
        or not value.isprintable()
        or any(c.isspace() for c in value)
    ):
        raise GoldsetError(f"{where}: {key} must be an https URL without control characters")
    return value


def _parse_source(raw: object, index: int) -> Source:
    obj = _object(raw, f"source {index}")
    where = f"source {index}"
    source_id = _str(obj, "id", where)
    if not IDENTIFIER.fullmatch(source_id):
        raise GoldsetError(f"{where}: id must be lower-case letters, digits and '-'")
    licence = _str(obj, "license", where)
    if not LICENCE.fullmatch(licence):
        raise GoldsetError(f"{where}: licence {licence[:40]!r} is not CC0, public domain or CC BY")
    sha = _str(obj, "sha256", where)
    if not SHA256.fullmatch(sha):
        raise GoldsetError(f"{where}: sha256 must be 64 lower-case hex digits")
    author = _str(obj, "author", where)
    if "|" in author or "\n" in author:
        raise GoldsetError(f"{where}: the author may not contain '|' or a line break")
    return Source(
        id=source_id,
        url=_https(obj, "url", where),
        page=_https(obj, "page", where),
        author=author,
        license=licence,
        sha256=sha,
        width=_int(obj, "width", where, 1, 100_000),
        height=_int(obj, "height", where, 1, 100_000),
    )


def _parse_box(raw: object, where: str, source: Source) -> Box:
    if not isinstance(raw, list) or len(raw) != 4 or not all(type(v) is int for v in raw):
        raise GoldsetError(f"{where}: box must be four whole numbers")
    x1, y1, x2, y2 = raw
    if not (0 <= x1 < x2 <= source.width and 0 <= y1 < y2 <= source.height):
        raise GoldsetError(f"{where}: box must lie inside its source photo")
    return x1, y1, x2, y2


_LABEL_BY_NAME: dict[str, Label] = {name: name for name in LABELS}


def _parse_item(raw: object, index: int, sources: dict[str, Source]) -> Item:
    obj = _object(raw, f"item {index}")
    where = f"item {index}"
    item_id = _str(obj, "id", where)
    if not IDENTIFIER.fullmatch(item_id):
        raise GoldsetError(f"{where}: id must be lower-case letters, digits and '-'")
    source_id = _str(obj, "source", where)
    if source_id not in sources:
        raise GoldsetError(f"{where}: unknown source {source_id[:40]!r}")
    label = _LABEL_BY_NAME.get(_str(obj, "label", where))
    if label is None:
        raise GoldsetError(f"{where}: label must be one of {', '.join(LABELS)}")
    kind = _str(obj, "kind", where)
    if kind not in KINDS[label]:
        raise GoldsetError(f"{where}: kind {kind[:20]!r} does not go with label {label}")
    control: str | None = None
    control_sha: str | None = None
    if obj.get("control") is not None:
        entry = _object(obj["control"], f"{where}: control")
        control = _str(entry, "file", where)
        control_sha = _str(entry, "sha256", where)
        if not CONTROL_FILE.fullmatch(control) or not SHA256.fullmatch(control_sha):
            raise GoldsetError(f"{where}: control needs a file under controls/ and a sha256")
    return Item(
        id=item_id,
        source=source_id,
        box=_parse_box(obj.get("box"), where, sources[source_id]),
        label=label,
        kind=kind,
        height_px=_int(obj, "height_px", where, MIN_HEIGHT_PX, MAX_HEIGHT_PX),
        jpeg_quality=_int(obj, "jpeg_quality", where, MIN_JPEG_QUALITY, MAX_JPEG_QUALITY),
        control=control,
        control_sha256=control_sha,
    )


def parse_manifest(raw: bytes) -> Manifest:
    """Parse and check a manifest; raise GoldsetError for anything malformed."""
    if len(raw) > MAX_MANIFEST_BYTES:
        raise GoldsetError(f"the manifest is larger than {MAX_MANIFEST_BYTES} bytes")
    try:
        data = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except GoldsetError:
        raise
    except json.JSONDecodeError as exc:
        raise GoldsetError(f"the manifest is not valid JSON: {exc.msg}") from None
    except UnicodeDecodeError:
        raise GoldsetError("the manifest is not UTF-8") from None
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise GoldsetError(f"the manifest is not valid JSON: {type(exc).__name__}") from None
    top = _object(data, "the manifest")
    seed = _int(top, "seed", "the manifest", 0, 2**31 - 1)
    raw_sources, raw_items = top.get("sources"), top.get("items")
    if not isinstance(raw_sources, list) or not isinstance(raw_items, list):
        raise GoldsetError("the manifest needs a list of sources and a list of items")
    if len(raw_sources) > MAX_ITEMS or len(raw_items) > MAX_ITEMS:
        raise GoldsetError(f"the manifest holds more than {MAX_ITEMS} sources or items")
    sources: dict[str, Source] = {}
    for index, entry in enumerate(raw_sources):
        source = _parse_source(entry, index)
        if source.id in sources:
            raise GoldsetError(f"source {index}: the id {source.id!r} is used twice")
        sources[source.id] = source
    items = tuple(_parse_item(entry, i, sources) for i, entry in enumerate(raw_items))
    if len({item.id for item in items}) != len(items):
        raise GoldsetError("an item id is used twice")
    controls = [item.control for item in items if item.control is not None]
    if len(set(controls)) != len(controls):
        raise GoldsetError("a control file is used twice")
    return Manifest(seed=seed, sources=sources, items=items)


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    try:
        with open(path, "rb") as fh:
            raw = fh.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise GoldsetError(f"cannot read {path.name}: {exc.strerror}") from None
    return parse_manifest(raw)


def degradation_params(seed: int, item_id: str) -> tuple[int, int]:
    """(height_px, jpeg_quality) for an item, drawn from the manifest's seed and its id."""
    rng = random.Random(f"{seed}:{item_id}")  # noqa: S311  (sampling, not security)
    height = rng.randint(MIN_HEIGHT_PX, MAX_HEIGHT_PX)
    quality = rng.randint(MIN_JPEG_QUALITY, MAX_JPEG_QUALITY)
    return height, quality


def screening_subset(manifest: Manifest, n: int = SCREEN_SIZE) -> tuple[Item, ...]:
    """A fixed subset of `n` items with each label in its gold-set proportion."""
    rng = random.Random(manifest.seed)  # noqa: S311  (sampling, not security)
    chosen: list[Item] = []
    total = len(manifest.items)
    for label in LABELS:
        pool = [item for item in manifest.items if item.label == label]
        rng.shuffle(pool)
        chosen += pool[: round(n * len(pool) / total)] if total else []
    order = {item.id: i for i, item in enumerate(manifest.items)}
    return tuple(sorted(chosen, key=lambda item: order[item.id]))


# Pixels ----------------------------------------------------------------------------------


def read_source(source: Source, source_dir: Path = SOURCE_DIR) -> Image:
    """A source photo's pixels (BGR), read as bytes, checked and decoded in memory."""
    path = source_dir / source.id
    try:
        with open(path, "rb") as fh:
            body = fh.read(MAX_SOURCE_BYTES + 1)
    except OSError as exc:
        raise GoldsetError(
            f"cannot read source {source.id}: {exc.strerror}; run scripts/fetch_goldset.sh"
        ) from None
    if len(body) > MAX_SOURCE_BYTES or hashlib.sha256(body).hexdigest() != source.sha256:
        raise GoldsetError(f"source {source.id} does not match its pinned SHA-256")
    try:
        image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:  # includes photos over wearreport._cv.MAX_IMAGE_PIXELS
        raise GoldsetError(f"source {source.id} cannot be decoded") from None
    if image is None:
        raise GoldsetError(f"source {source.id} cannot be decoded")
    if image.shape[:2] != (source.height, source.width):
        raise GoldsetError(f"source {source.id} is not a {source.width}x{source.height} image")
    return np.asarray(image, dtype=np.uint8)


def crop_bounds(
    box: FloatBox, width: int, height: int, margin: float = CROP_MARGIN
) -> tuple[int, int, int, int]:
    """The box grown by `margin` (of its width and height) on every side, in whole pixels,
    clipped to the frame. With the default margin, the spot-check tool's crop."""
    x1, y1, x2, y2 = box
    if not all(math.isfinite(v) for v in box) or not (x1 < x2 and y1 < y2):
        raise ValueError("a box must be finite and have a positive area")
    if not (math.isfinite(margin) and 0 <= margin <= MAX_MARGIN):
        raise ValueError(f"the margin must be from 0 to {MAX_MARGIN:g}")
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
    right, bottom = min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))
    if not (left < right and top < bottom):
        raise ValueError("the box lies outside the frame")
    return left, top, right, bottom


def degrade(
    image: Image,
    box: Sequence[float],
    height_px: int,
    jpeg_quality: int,
    margin: float = CROP_MARGIN,
) -> Degraded:
    """Crop `box` with `margin` (by default the spot-check margin), shrink it so that the
    box is `height_px` high, and pass it through JPEG at `jpeg_quality`. Deterministic. A
    wider margin adds surroundings, never pixels on the box."""
    if not MIN_HEIGHT_PX <= height_px <= MAX_HEIGHT_PX:
        raise ValueError(f"height_px must be from {MIN_HEIGHT_PX} to {MAX_HEIGHT_PX}")
    if not MIN_JPEG_QUALITY <= jpeg_quality <= MAX_JPEG_QUALITY:
        raise ValueError(f"jpeg_quality must be from {MIN_JPEG_QUALITY} to {MAX_JPEG_QUALITY}")
    x1, y1, x2, y2 = (float(v) for v in box)
    height, width = image.shape[:2]
    left, top, right, bottom = crop_bounds((x1, y1, x2, y2), width, height, margin)
    scale = height_px / (y2 - y1)
    size = (max(1, round((right - left) * scale)), max(1, round((bottom - top) * scale)))
    region = np.ascontiguousarray(image[top:bottom, left:right])
    small = np.asarray(cv2.resize(region, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
    ok, encoded = cv2.imencode(JPEG_SUFFIX, small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise GoldsetError("cannot encode a degraded crop")
    jpeg = encoded.tobytes()
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise GoldsetError("cannot decode a degraded crop")
    sx, sy = size[0] / (right - left), size[1] / (bottom - top)
    scaled = ((x1 - left) * sx, (y1 - top) * sy, (x2 - left) * sx, (y2 - top) * sy)
    return Degraded(small=small, jpeg=jpeg, frame=np.asarray(frame, dtype=np.uint8), box=scaled)


def _label(image: Image, text: str, x: int, y: int) -> None:
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


def render_crop(
    frame: Image,
    box: FloatBox,
    number: int,
    margin: float = CROP_MARGIN,
    target_height: int = CROP_TARGET_HEIGHT,
) -> Image:
    """The image the reviewer (and the judge) sees for a box in a frame: the spot-check
    tool's crop, enlarged by a whole factor, with the box and its number drawn on it. The
    defaults are the spot-check tool's; an experiment may widen the crop (`margin`) or
    enlarge it towards another height (`target_height`, still at most MAX_CROP_SCALE)."""
    if not 1 <= target_height <= MAX_TARGET_HEIGHT:
        raise ValueError(f"the render target must be from 1 to {MAX_TARGET_HEIGHT} pixels")
    height, width = frame.shape[:2]
    left, top, right, bottom = crop_bounds(box, width, height, margin)
    crop = np.ascontiguousarray(frame[top:bottom, left:right])
    scale = max(1, min(MAX_CROP_SCALE, target_height // (bottom - top)))
    size = ((right - left) * scale, (bottom - top) * scale)
    image = np.asarray(cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST), dtype=np.uint8)
    x1, y1 = round((box[0] - left) * scale), round((box[1] - top) * scale)
    x2, y2 = round((box[2] - left) * scale) - 1, round((box[3] - top) * scale) - 1
    cv2.rectangle(image, (x1, y1), (max(x1, x2), max(y1, y2)), BOX_COLOUR, 1)
    _label(image, str(number), x1, y1)
    return image


def degrade_item(
    manifest: Manifest,
    item: Item,
    source_dir: Path = SOURCE_DIR,
    image: Image | None = None,
    margin: float = CROP_MARGIN,
) -> Degraded:
    """Degrade one item of the manifest; `image` is its source photo if already read."""
    if image is None:
        image = read_source(manifest.sources[item.source], source_dir)
    return degrade(image, item.box, item.height_px, item.jpeg_quality, margin)


def iter_gold(
    manifest: Manifest,
    source_dir: Path = SOURCE_DIR,
    items: Iterable[Item] | None = None,
    margin: float = CROP_MARGIN,
    target_height: int = CROP_TARGET_HEIGHT,
) -> Iterator[tuple[Item, Image]]:
    """Each item (all of them, or `items`) with the rendered crop the judge sees. Items
    are numbered 1, 2, ... in the order given, like the boxes of a spot-check. `margin`
    and `target_height` are the crop variant (the defaults: the spot-check tool's)."""
    cache: dict[str, Image] = {}
    selected = manifest.items if items is None else tuple(items)
    for number, item in enumerate(selected, start=1):
        if item.source not in cache:
            cache.clear()  # items of one source are usually next to each other
            cache[item.source] = read_source(manifest.sources[item.source], source_dir)
        degraded = degrade_item(manifest, item, source_dir, cache[item.source], margin)
        yield item, render_crop(degraded.frame, degraded.box, number, margin, target_height)


# Command line ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.tools.goldset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--sources", type=Path, default=SOURCE_DIR)
    args = ap.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        count = sum(1 for _ in iter_gold(manifest, args.sources))
    except GoldsetError as exc:
        print(f"goldset: {exc}", file=sys.stderr)
        return 1
    labels = Counter(item.label for item in manifest.items)
    kinds = Counter(item.kind for item in manifest.items)
    hard = sum(item.hard_negative for item in manifest.items)
    print(f"goldset: {count} items from {len(manifest.sources)} source photos")
    print("  labels: " + ", ".join(f"{label} {labels[label]}" for label in LABELS))
    print("  kinds: " + ", ".join(f"{kind} {n}" for kind, n in sorted(kinds.items())))
    print(f"  hard negatives: {hard} ({100 * hard / max(1, count):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
