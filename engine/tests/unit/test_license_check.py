from __future__ import annotations

from email.message import Message
from pathlib import Path

import pytest

import license_check


def _meta(name: str = "pkg", **fields: str | list[str]) -> Message:
    msg = Message()
    msg["Name"] = name
    msg["Version"] = "1.0"
    for key, value in fields.items():
        for item in value if isinstance(value, list) else [value]:
            msg[key.replace("_", "-")] = item
    return msg


@pytest.mark.parametrize(
    "expression",
    [
        "MIT",
        "BSD-3-Clause",
        "BSD-2-Clause",
        "0BSD",
        "ISC",
        "PSF-2.0",
        "MPL-2.0",
        "CC0-1.0",
        "Apache-2.0 OR BSD-3-Clause",
        "MIT AND Apache-2.0",
        "GPL-3.0-only OR MIT",
        "(Apache-2.0 OR MIT) AND BSD-3-Clause",
        "Apache-2.0 WITH LLVM-exception",
    ],
)
def test_allowed_expressions(expression: str) -> None:
    assert license_check.license_problem(_meta(License_Expression=expression)) is None


@pytest.mark.parametrize(
    "expression",
    [
        "AGPL-3.0-or-later",
        "GPL-2.0-only",
        "LGPL-3.0-or-later",
        "SSPL-1.0",
        "BUSL-1.1",
        "CC-BY-NC-4.0",
        "MIT AND GPL-3.0-only",
        "LicenseRef-Proprietary",
        "MIT OR",  # malformed
        "(MIT",  # malformed
    ],
)
def test_rejected_expressions(expression: str) -> None:
    assert license_check.license_problem(_meta(License_Expression=expression)) is not None


def test_classifiers_all_must_be_allowed() -> None:
    ok = _meta(Classifier=["License :: OSI Approved :: BSD License", "Programming Language :: C"])
    assert license_check.license_problem(ok) is None
    mixed = _meta(
        Classifier=[
            "License :: OSI Approved :: MIT License",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        ]
    )
    assert license_check.license_problem(mixed) is not None


def test_license_expression_takes_precedence_over_classifiers() -> None:
    meta = _meta(
        License_Expression="MIT",
        Classifier="License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    )
    assert license_check.license_problem(meta) is None


@pytest.mark.parametrize("value", ["MIT", "BSD License", "Apache License 2.0", "BSD-3-Clause"])
def test_legacy_license_field_allowed(value: str) -> None:
    assert license_check.license_problem(_meta(License=value)) is None


@pytest.mark.parametrize("value", ["GPLv3", "Proprietary", "Free for non-commercial use"])
def test_legacy_license_field_rejected(value: str) -> None:
    assert license_check.license_problem(_meta(License=value)) is not None


def test_missing_licence_metadata_is_rejected() -> None:
    assert license_check.license_problem(_meta()) == "no licence metadata"


def test_forbidden_package_in_lockfile_fails(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n\n[[package]]\nname = "Ultralytics"\nversion = "8.0.0"\n')
    assert license_check.main(["--lock", str(lock)]) == 1


def test_uninstalled_project_fails(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    assert license_check.main(["--project", "no-such-dist-xyz", "--lock", str(lock)]) == 1
