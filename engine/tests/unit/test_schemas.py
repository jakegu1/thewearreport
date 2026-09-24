"""Every JSON Schema in data/schema/ is valid, and every sample validates against it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

SCHEMA_DIR = Path(__file__).resolve().parents[3] / "data" / "schema"
SCHEMAS = sorted(SCHEMA_DIR.glob("*.json"))
SAMPLES = sorted((SCHEMA_DIR / "samples").glob("*.json"))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_schema_directory_is_not_empty() -> None:
    assert SCHEMAS


@pytest.mark.parametrize("path", SCHEMAS, ids=lambda p: p.name)
def test_schema_is_valid_draft_2020_12(path: Path) -> None:
    schema = _load(path)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_sample_validates_against_its_schema(path: Path) -> None:
    schema_path = SCHEMA_DIR / path.name
    assert schema_path.exists(), f"sample {path.name} has no schema of the same name"
    jsonschema.Draft202012Validator(_load(schema_path)).validate(_load(path))


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": "example.v0"},
        {"schema_version": "example.v0", "count": -1},
        {"schema_version": "example.v1", "count": 1},
        {"schema_version": "example.v0", "count": 1, "extra": "x"},
    ],
)
def test_example_schema_rejects_invalid_documents(document: dict[str, Any]) -> None:
    validator = jsonschema.Draft202012Validator(_load(SCHEMA_DIR / "example.v0.json"))
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(document)
