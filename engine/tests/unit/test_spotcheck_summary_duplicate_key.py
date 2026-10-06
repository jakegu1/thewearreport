"""The attribute summary's duplicate-key error names no camera id (T-078 review).

A label file's `camera_yield` is keyed by camera id, so a repeated id must be refused
without being echoed; a repeated field name of the file itself is still named.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wearreport.tools import spotcheck_summary

CAMERA = "Fake_00042"
# One attribute file as text, its camera_yield last, so that a duplicate can be appended.
HEAD = (
    '{"date": "2026-10-05", "started_at": "2026-10-05T11:22Z", "light": "day", "frames": 2,'
    ' "detector": {"model": "stub", "sha256": "' + "0" * 64 + '", "conf": 0.3},'
    ' "min_height_px": 31, "judge": null, "crops_shown": 3, "crops_rejected": 1,'
    ' "crops": [[40, "ynn", null], [50, "unn", null]],'
)
ENTRY = f'"{CAMERA}": [2, 1, 1]'
CLEAN = HEAD + f' "camera_yield": {{{ENTRY}, "Fake_00001": [1, 0, 0]}}}}'
REPEATED_ID = HEAD + f' "camera_yield": {{{ENTRY}, "Fake_00001": [1, 0, 0], {ENTRY}}}}}'

REPEATED_FIELD = CLEAN[:-1] + ', "light": "day"}'


def test_the_clean_file_parses() -> None:
    json.loads(REPEATED_ID)  # valid JSON: only the duplicate is wrong
    assert spotcheck_summary.parse_labelling(CLEAN.encode()).shown == 3


def test_a_repeated_camera_id_is_refused_without_naming_it() -> None:
    with pytest.raises(ValueError) as info:
        spotcheck_summary.parse_labelling(REPEATED_ID.encode())
    assert str(info.value) == "a key appears twice in one object"
    assert CAMERA not in str(info.value)


def test_a_repeated_field_name_is_still_named() -> None:
    with pytest.raises(ValueError) as info:
        spotcheck_summary.parse_labelling(REPEATED_FIELD.encode())
    assert str(info.value) == "the key 'light' appears twice in one object"


def test_the_summary_names_the_file_and_no_camera(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "attributes"
    folder.mkdir()
    (folder / "2026-10-05.json").write_text(REPEATED_ID + "\n", encoding="utf-8")
    code = spotcheck_summary.main(["--attributes", "--dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert "2026-10-05.json" in captured.err
    assert "appears twice" in captured.err
    assert CAMERA not in captured.err and CAMERA not in captured.out
