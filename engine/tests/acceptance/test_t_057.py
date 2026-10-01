"""Acceptance tests for T-057: the attribute summary refuses a crop below its file's
min_height_px, and a min_height_px outside the near-field threshold to 200; the judge
module's docstring states the retry behaviour. The task contract: do not edit.

Every attribute file here is synthetic and written by the test into a temporary directory,
except the three committed files the summary must still read. Nothing reaches the network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from wearreport.tools import judge, judge_hosted, spotcheck, spotcheck_summary

ROOT = Path(__file__).resolve().parents[3]
REAL_DIR = ROOT / "spotchecks"
REAL_FILES = ("2026-09-30.json", "2026-09-30-2.json", "2026-10-01.json")
MODEL = "di-qwen3-vl-235b"

VALID: dict[str, Any] = {
    "date": "2026-10-05",
    "started_at": "2026-10-05T10:00Z",
    "light": "day",
    "frames": 3,
    "detector": {"model": "yolox_m", "sha256": "0" * 64, "conf": 0.3},
    "min_height_px": 46,
    "judge": MODEL,
    "crops_shown": 3,
    "crops_rejected": 1,
    "crops": [[60, "ynn", "ynu"], [46, "uuu", None]],
}


def _raw(**changes: Any) -> bytes:
    return json.dumps({**VALID, **changes}).encode()


def _summary(d: Path, content: bytes, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    (d / "attributes").mkdir(parents=True, exist_ok=True)
    (d / "attributes" / "2026-10-05.json").write_bytes(_raw())
    (d / "attributes" / "2026-10-06-2.json").write_bytes(content)
    code = spotcheck_summary.main(["--attributes", "--dir", str(d)])
    out = capsys.readouterr()
    return code, out.out, out.err


# AC1: a crop below min_height_px ----------------------------------------------------------


@pytest.mark.parametrize(
    "crops",
    [
        pytest.param([[60, "ynn", "ynu"], [45, "uuu", None]], id="one-below"),
        pytest.param([[45, "ynn", "ynu"], [60, "uuu", None]], id="first-below"),
        pytest.param([[31, "ynn", "ynu"], [0, "uuu", None]], id="all-below"),
    ],
)
def test_ac1_a_crop_below_min_height_px_is_refused(crops: list[list[Any]]) -> None:
    with pytest.raises(ValueError) as caught:
        spotcheck_summary.parse_labelling(_raw(crops=crops))
    message = str(caught.value)
    assert "crop height" in message
    assert "min_height_px" in message


def test_ac1_a_crop_exactly_at_min_height_px_is_accepted() -> None:
    labelling = spotcheck_summary.parse_labelling(
        _raw(crops=[[46, "ynn", "ynu"], [46, "uuu", None]])
    )
    assert [crop[0] for crop in labelling.crops] == [46, 46]


def test_ac1_the_valid_example_is_accepted() -> None:
    labelling = spotcheck_summary.parse_labelling(_raw())
    assert labelling.crops == ((60, "ynn", "ynu"), (46, "uuu", None))


def test_ac1_through_load_labellings_the_file_is_named(tmp_path: Path) -> None:
    d = tmp_path / "spotchecks"
    (d / "attributes").mkdir(parents=True)
    (d / "attributes" / "2026-10-05.json").write_bytes(_raw())
    bad = _raw(crops=[[60, "ynn", "ynu"], [45, "uuu", None]])
    (d / "attributes" / "2026-10-06-2.json").write_bytes(bad)
    with pytest.raises(spotcheck_summary.SummaryError) as caught:
        spotcheck_summary.load_labellings(d)
    message = str(caught.value)
    assert "2026-10-06-2.json is not an attribute file" in message
    assert "min_height_px" in message


def test_ac1_through_the_cli_it_is_refused_like_other_malformed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _raw(crops=[[60, "ynn", "ynu"], [45, "uuu", None]])
    code, out, err = _summary(tmp_path / "spotchecks", bad, capsys)
    assert code == 1
    assert out == ""
    assert err.startswith("summary: ")
    assert "2026-10-06-2.json is not an attribute file (" in err
    assert "min_height_px" in err
    # The same path and exit code as an existing malformed case.
    code2, out2, err2 = _summary(tmp_path / "other", _raw(light="dusk"), capsys)
    assert (code2, out2) == (1, "")
    assert "2026-10-06-2.json is not an attribute file (" in err2


# AC2: min_height_px outside the near-field threshold to 200 ------------------------------


def test_ac2_the_bounds_are_the_near_field_threshold_and_200() -> None:
    assert spotcheck.NEAR_FIELD_MIN_HEIGHT_PX == 31
    assert spotcheck.MAX_ATTRIBUTE_MIN_HEIGHT_PX == 200


@pytest.mark.parametrize("value", [0, 1, 30, 201, 1000, spotcheck_summary.MAX_COUNT])
def test_ac2_a_min_height_px_out_of_range_is_refused(value: int) -> None:
    with pytest.raises(ValueError) as caught:
        spotcheck_summary.parse_labelling(
            _raw(min_height_px=value, crops=[[value, "ynn", "ynu"], [value, "uuu", None]])
        )
    assert "min_height_px" in str(caught.value)


@pytest.mark.parametrize("value", [31, 200])
def test_ac2_the_bounds_are_accepted(value: int) -> None:
    labelling = spotcheck_summary.parse_labelling(
        _raw(min_height_px=value, crops=[[value, "ynn", "ynu"], [250, "uuu", None]])
    )
    assert [crop[0] for crop in labelling.crops] == [value, 250]


@pytest.mark.parametrize("value", [30, 201])
def test_ac2_through_the_cli_it_is_refused_like_other_malformed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: int
) -> None:
    content = _raw(min_height_px=value, crops=[[250, "ynn", "ynu"], [250, "uuu", None]])
    code, out, err = _summary(tmp_path / "spotchecks", content, capsys)
    assert (code, out) == (1, "")
    assert "2026-10-06-2.json is not an attribute file (" in err
    assert "min_height_px" in err


@pytest.mark.parametrize(
    "value",
    [True, 31.0, "31", None, -31, 10**400],
    ids=["bool", "float", "str", "null", "neg", "huge"],
)
def test_ac2_a_min_height_px_that_is_not_a_whole_number_is_still_refused(value: Any) -> None:
    with pytest.raises(ValueError):
        spotcheck_summary.parse_labelling(_raw(min_height_px=value))


# AC3: the committed files ------------------------------------------------------------------


@pytest.mark.parametrize("name", REAL_FILES)
def test_ac3_the_committed_attribute_files_still_parse(name: str) -> None:
    raw = (REAL_DIR / "attributes" / name).read_bytes()
    labelling = spotcheck_summary.parse_labelling(raw)
    record = json.loads(raw)
    assert len(labelling.crops) == len(record["crops"])


def test_ac3_the_committed_directory_still_loads(capsys: pytest.CaptureFixture[str]) -> None:
    labellings = spotcheck_summary.load_labellings(REAL_DIR)
    assert len(labellings) >= len(REAL_FILES)
    assert spotcheck_summary.main(["--attributes", "--dir", str(REAL_DIR)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{len(labellings)} attribute file(s) in ")


# AC4: the judge module's docstring --------------------------------------------------------


def _flat(text: str | None) -> str:
    assert text is not None
    return re.sub(r"\s+", " ", text)


def test_ac4_the_docstring_states_what_is_retried() -> None:
    doc = _flat(judge.__doc__)
    assert "a connection that fails before any HTTP response" in doc
    assert "retried at most MAX_RETRIES times with backoff" in doc
    assert "other errors (a timeout among them) not at all" in doc
    assert "throttling" in doc.lower() and "server errors" in doc


def test_ac4_the_docstring_agrees_with_judge_hosted() -> None:
    hosted = _flat(judge_hosted.__doc__)
    assert "a connection that fails before any HTTP response" in hosted
    assert "other errors (a timeout among them) not at all" in hosted
    assert judge.MAX_RETRIES == judge_hosted.MAX_RETRIES
