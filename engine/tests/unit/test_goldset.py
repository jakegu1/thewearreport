"""Unit tests for wearreport.tools.goldset. Every image here is synthetic."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport._cv import cv2
from wearreport.tools import goldset

SOURCE: dict[str, Any] = {
    "id": "src-1",
    "url": "https://example.org/a",
    "page": "https://example.org/page",
    "author": "Someone",
    "license": "CC BY 2.0",
    "sha256": "0" * 64,
    "width": 400,
    "height": 600,
}
ITEM: dict[str, Any] = {
    "id": "g0001",
    "source": "src-1",
    "box": [150, 80, 181, 201],
    "label": "person",
    "kind": "pedestrian",
    "height_px": 30,
    "jpeg_quality": 40,
}


def _manifest(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"seed": 29, "sources": [dict(SOURCE)], "items": [dict(ITEM)]}
    data.update(changes)
    return data


def _raw(data: object) -> bytes:
    return json.dumps(data).encode()


def _frame(height: int = 600, width: int = 400) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(3)
    frame: npt.NDArray[np.uint8] = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return frame


def test_a_valid_manifest_parses() -> None:
    manifest = goldset.parse_manifest(_raw(_manifest()))
    assert manifest.seed == 29
    assert manifest.sources["src-1"].width == 400
    (item,) = manifest.items
    assert item.box == (150, 80, 181, 201) and item.label == "person"
    assert not item.hard_negative and item.control is None


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{",
        b"\xff\xfe not utf-8",
        b"[]",
        b'{"seed": NaN, "sources": [], "items": []}',
        b'{"seed": 1, "seed": 2, "sources": [], "items": []}',
        b"[" * 100_000 + b"]" * 100_000,
        b'{"seed": 1' + b"0" * 5000 + b', "sources": [], "items": []}',
        b" " * (goldset.MAX_MANIFEST_BYTES + 1),
    ],
    ids=["empty", "truncated", "not-utf8", "list", "nan", "duplicate", "deep", "huge", "big"],
)
def test_malformed_manifests_raise_goldset_error(raw: bytes) -> None:
    with pytest.raises(goldset.GoldsetError):
        goldset.parse_manifest(raw)


@pytest.mark.parametrize(
    ("where", "key", "value"),
    [
        ("source", "license", "CC BY-SA 2.0"),
        ("source", "license", "CC BY-NC 2.0"),
        ("source", "license", "All rights reserved"),
        ("source", "url", "http://example.org/a"),
        ("source", "url", "https://example.org/a b"),
        ("source", "url", "https://example.org/a\x1bb"),
        ("source", "url", "https://example.org/a\x7f"),
        ("source", "page", "https://example.org/\x00"),
        ("source", "sha256", "abc"),
        ("source", "author", "a | b"),
        ("source", "author", ""),
        ("source", "id", "../etc"),
        ("source", "width", 0),
        ("source", "width", True),
        ("item", "box", [150, 80, 181, 700]),
        ("item", "box", [181, 80, 150, 201]),
        ("item", "box", [150.0, 80, 181, 201]),
        ("item", "box", [1, 2, 3]),
        ("item", "source", "nowhere"),
        ("item", "label", "cat"),
        ("item", "kind", "pole"),  # a pole is not a person
        ("item", "height_px", 14),
        ("item", "height_px", 81),
        ("item", "jpeg_quality", 90),
        ("item", "control", {"file": "elsewhere/x.jpg", "sha256": "0" * 64}),
        ("item", "control", "fixtures/goldset/controls/c01.jpg"),
    ],
)
def test_bad_fields_raise_goldset_error(where: str, key: str, value: object) -> None:
    data = _manifest()
    target = data["sources"][0] if where == "source" else data["items"][0]
    target[key] = value
    with pytest.raises(goldset.GoldsetError):
        goldset.parse_manifest(_raw(data))


def test_duplicate_ids_and_controls_are_refused() -> None:
    twice = _manifest(items=[dict(ITEM), dict(ITEM)])
    with pytest.raises(goldset.GoldsetError, match="twice"):
        goldset.parse_manifest(_raw(twice))
    control = {"file": "fixtures/goldset/controls/c01.jpg", "sha256": "1" * 64}
    items = [dict(ITEM, control=control), dict(ITEM, id="g0002", control=control)]
    with pytest.raises(goldset.GoldsetError, match="control"):
        goldset.parse_manifest(_raw(_manifest(items=items)))
    sources = [dict(SOURCE), dict(SOURCE)]
    with pytest.raises(goldset.GoldsetError, match="twice"):
        goldset.parse_manifest(_raw(_manifest(sources=sources)))


def test_hard_negative_follows_the_kind() -> None:
    item = dict(ITEM, label="not_person", kind="shadow")
    other = dict(ITEM, id="g0002", label="not_person", kind="other")
    manifest = goldset.parse_manifest(_raw(_manifest(items=[item, other])))
    assert [i.hard_negative for i in manifest.items] == [True, False]


def test_load_manifest_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(goldset.GoldsetError, match="cannot read"):
        goldset.load_manifest(tmp_path / "missing.json")


def test_degradation_params_are_seeded_and_in_range() -> None:
    draws = [goldset.degradation_params(29, f"g{i:04d}") for i in range(2000)]
    assert draws == [goldset.degradation_params(29, f"g{i:04d}") for i in range(2000)]
    assert draws != [goldset.degradation_params(30, f"g{i:04d}") for i in range(2000)]
    assert {h for h, _ in draws} == set(range(15, 81))
    assert {q for _, q in draws} == set(range(35, 61))


def test_screening_subset_is_fixed_and_proportional() -> None:
    items = [
        dict(ITEM, id=f"g{i:04d}", label=label, kind=kind)
        for i, (label, kind) in enumerate(
            [("person", "pedestrian")] * 200
            + [("in_vehicle", "in_vehicle")] * 60
            + [("not_person", "pole")] * 240
        )
    ]
    manifest = goldset.parse_manifest(_raw(_manifest(items=items)))
    subset = goldset.screening_subset(manifest)
    assert subset == goldset.screening_subset(manifest)
    labels = [item.label for item in subset]
    assert (labels.count("person"), labels.count("in_vehicle"), labels.count("not_person")) == (
        40,
        12,
        48,
    )
    order = [item.id for item in subset]
    assert order == sorted(order)  # in manifest order


def _write_source(tmp_path: Path, body: bytes, **changes: Any) -> goldset.Source:
    (tmp_path / "src-1").write_bytes(body)
    fields = dict(SOURCE, sha256=hashlib.sha256(body).hexdigest())
    fields.update(changes)
    return goldset.Source(**fields)


def test_read_source_decodes_in_memory(tmp_path: Path) -> None:
    ok, encoded = cv2.imencode(".png", _frame())
    assert ok
    source = _write_source(tmp_path, encoded.tobytes())
    image = goldset.read_source(source, tmp_path)
    assert image.shape == (600, 400, 3) and image.dtype == np.uint8


def test_read_source_refuses_bad_files(tmp_path: Path) -> None:
    with pytest.raises(goldset.GoldsetError, match="fetch_goldset"):
        goldset.read_source(goldset.Source(**SOURCE), tmp_path)
    source = _write_source(tmp_path, b"not an image at all")
    with pytest.raises(goldset.GoldsetError, match="decoded"):
        goldset.read_source(source, tmp_path)
    _, encoded = cv2.imencode(".png", _frame(10, 10))
    source = _write_source(tmp_path, encoded.tobytes())
    with pytest.raises(goldset.GoldsetError, match="400x600"):
        goldset.read_source(source, tmp_path)
    _, encoded = cv2.imencode(".png", np.zeros((1081, 1920, 3), np.uint8))
    big = _write_source(tmp_path, encoded.tobytes(), width=1920, height=1081)
    with pytest.raises(goldset.GoldsetError):  # over the decoder's pixel cap
        goldset.read_source(big, tmp_path)


@pytest.mark.parametrize(("height_px", "quality"), [(14, 40), (81, 40), (30, 34), (30, 61)])
def test_degrade_refuses_parameters_outside_tfl_conditions(height_px: int, quality: int) -> None:
    with pytest.raises(ValueError):
        goldset.degrade(_frame(), (150, 80, 181, 201), height_px, quality)


def test_degrade_clips_a_box_at_the_edge() -> None:
    degraded = goldset.degrade(_frame(), (0, 0, 40, 120), 20, 50)
    x1, y1, _, y2 = degraded.box
    assert (x1, y1) == (0, 0) and y2 == pytest.approx(20, abs=0.5)
    assert degraded.frame.shape[0] == pytest.approx(30, abs=1)  # the margin, below only


def test_lower_quality_means_fewer_bytes() -> None:
    frame = _frame()
    low = goldset.degrade(frame, (100, 100, 200, 400), 80, 35)
    high = goldset.degrade(frame, (100, 100, 200, 400), 80, 60)
    assert len(low.jpeg) < len(high.jpeg)


def test_render_crop_enlarges_and_draws_the_box() -> None:
    frame = np.zeros((40, 20, 3), np.uint8)
    image = goldset.render_crop(frame, (5.0, 10.0, 15.0, 30.0), 7)
    assert image.shape == (40 * 6, 20 * 6, 3)  # 240 // 40 = 6
    assert (image == goldset.BOX_COLOUR).all(axis=2).any()


def test_iter_gold_numbers_items_in_order(tmp_path: Path) -> None:
    ok, encoded = cv2.imencode(".png", _frame())
    assert ok
    body = encoded.tobytes()
    (tmp_path / "src-1").write_bytes(body)
    source = dict(SOURCE, sha256=hashlib.sha256(body).hexdigest())
    items = [dict(ITEM, id=f"g{i:04d}") for i in range(1, 4)]
    manifest = goldset.parse_manifest(_raw(_manifest(sources=[source], items=items)))
    got = list(goldset.iter_gold(manifest, tmp_path, items=manifest.items[1:]))
    assert [item.id for item, _ in got] == ["g0002", "g0003"]
    degraded = goldset.degrade_item(manifest, manifest.items[1], tmp_path)
    assert np.array_equal(got[0][1], goldset.render_crop(degraded.frame, degraded.box, 1))


def test_iter_gold_renders_a_crop_variant(tmp_path: Path) -> None:
    ok, encoded = cv2.imencode(".png", _frame())
    assert ok
    body = encoded.tobytes()
    (tmp_path / "src-1").write_bytes(body)
    source = dict(SOURCE, sha256=hashlib.sha256(body).hexdigest())
    manifest = goldset.parse_manifest(_raw(_manifest(sources=[source])))
    item = manifest.items[0]
    ((_, default),) = goldset.iter_gold(manifest, tmp_path)
    ((_, wide),) = goldset.iter_gold(manifest, tmp_path, margin=1.0, target_height=480)
    degraded = goldset.degrade_item(manifest, item, tmp_path, margin=1.0)
    expected = goldset.render_crop(degraded.frame, degraded.box, 1, 1.0, 480)
    assert np.array_equal(wide, expected)
    assert wide.shape[0] > default.shape[0]
    assert degraded.frame.shape[0] > goldset.degrade_item(manifest, item, tmp_path).frame.shape[0]


def test_degrade_refuses_a_margin_out_of_range() -> None:
    for margin in (-0.5, goldset.MAX_MARGIN + 0.1, float("nan")):
        with pytest.raises(ValueError):
            goldset.degrade(_frame(), (150, 80, 181, 201), 30, 40, margin=margin)


def test_a_zero_margin_crops_the_box_itself() -> None:
    assert goldset.crop_bounds((10.2, 20.7, 30.1, 60.0), 100, 100, margin=0.0) == (10, 20, 31, 60)


def test_main_checks_every_item_and_prints_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ok, encoded = cv2.imencode(".png", _frame())
    assert ok
    body = encoded.tobytes()
    (tmp_path / "src-1").write_bytes(body)
    data = _manifest(sources=[dict(SOURCE, sha256=hashlib.sha256(body).hexdigest())])
    data["items"].append(dict(ITEM, id="g0002", label="not_person", kind="bollard"))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    assert goldset.main(["--manifest", str(path), "--sources", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "2 items from 1 source photos" in out and "hard negatives: 1 (50.0%)" in out
    broken = copy.deepcopy(data)
    broken["sources"][0]["sha256"] = "1" * 64
    path.write_text(json.dumps(broken))
    assert goldset.main(["--manifest", str(path), "--sources", str(tmp_path)]) == 1


def test_the_committed_manifest_parses() -> None:
    manifest = goldset.load_manifest()
    assert len(manifest.items) >= 400
    kinds = {item.kind for item in manifest.items}
    assert kinds >= goldset.HARD_NEGATIVE_KINDS
