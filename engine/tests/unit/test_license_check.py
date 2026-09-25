from __future__ import annotations

from collections.abc import Sequence
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


@pytest.mark.parametrize(
    "identifier",
    ["BSD-Protection", "BSD-3-Clause-No-Nuclear-License", "BSD-4-Clause", "BSD-Anything"],
)
def test_only_enumerated_bsd_identifiers_are_allowed(identifier: str) -> None:
    assert not license_check.spdx_allowed(identifier)
    assert license_check.license_problem(_meta(License_Expression=identifier)) is not None


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


# Dependency walk over a fake site-packages -----------------------------------------------

GPL = "lcfake-gplpkg"


def _install(site: Path, name: str, licence: str = "MIT", lines: Sequence[str] = ()) -> None:
    """Create a minimal `<name>-1.0.dist-info/METADATA` in `site`."""
    info = site / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir(parents=True)
    body = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        "Version: 1.0",
        f"License-Expression: {licence}",
        *lines,
    ]
    (info / "METADATA").write_text("\n".join(body) + "\n", encoding="utf-8")


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    _install(site, GPL, licence="GPL-3.0-only")
    return site


def _run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    code = license_check.main(["--project", "lcfake-root", "--lock", str(lock)])
    return code, capsys.readouterr().out


def _walked(project: str = "lcfake-root") -> set[str]:
    return {d.metadata["Name"] for d in license_check.runtime_distributions(project)}


@pytest.mark.parametrize(
    "root_requires",
    [["lcfake-c", "lcfake-a"], ["lcfake-a", "lcfake-c"]],
    ids=["extra-reached-second", "extra-reached-first"],
)
def test_gpl_reached_through_an_extra_fails_in_any_order(
    site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], root_requires: list[str]
) -> None:
    _install(site, "lcfake-root", lines=[f"Requires-Dist: {r}" for r in root_requires])
    _install(site, "lcfake-a", lines=["Requires-Dist: lcfake-b"])
    _install(site, "lcfake-c", lines=["Requires-Dist: lcfake-b[extra1]"])
    _install(
        site,
        "lcfake-b",
        lines=["Provides-Extra: extra1", f'Requires-Dist: {GPL}; extra == "extra1"'],
    )
    assert _walked() == {"lcfake-a", "lcfake-b", "lcfake-c", GPL}
    code, out = _run(tmp_path, capsys)
    assert code == 1
    assert f"{GPL} 1.0: licence 'GPL-3.0-only' not on the allowlist" in out


def test_gpl_behind_a_root_extra_fails(
    site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(
        site,
        "lcfake-root",
        lines=["Provides-Extra: gpu", f'Requires-Dist: {GPL}; extra == "gpu"'],
    )
    assert _walked() == {GPL}
    code, out = _run(tmp_path, capsys)
    assert code == 1
    assert GPL in out


def test_transitive_gpl_dependency_fails(
    site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(site, "lcfake-root", lines=["Requires-Dist: lcfake-a"])
    _install(site, "lcfake-a", lines=["Requires-Dist: lcfake-b>=1"])
    _install(site, "lcfake-b", licence="Apache-2.0", lines=[f"Requires-Dist: {GPL}"])
    code, out = _run(tmp_path, capsys)
    assert code == 1
    assert GPL in out


def test_clean_graph_passes_and_counts_every_package(
    site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(
        site,
        "lcfake-root",
        lines=["Provides-Extra: fast", 'Requires-Dist: lcfake-a[x]; extra == "fast"'],
    )
    _install(
        site,
        "lcfake-a",
        licence="BSD-3-Clause",
        lines=[
            "Provides-Extra: x",
            "Provides-Extra: unused",
            'Requires-Dist: lcfake-b; extra == "x"',
            f'Requires-Dist: {GPL}; extra == "unused"',
            f'Requires-Dist: {GPL}; sys_platform == "no-such-platform"',
        ],
    )
    _install(site, "lcfake-b", licence="MIT OR Apache-2.0")
    code, out = _run(tmp_path, capsys)
    assert code == 0, out
    assert "clean (2 runtime dependencies)" in out


def test_missing_dependency_is_a_finding(
    site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(site, "lcfake-root", lines=["Requires-Dist: lcfake-not-installed"])
    code, out = _run(tmp_path, capsys)
    assert code == 1
    assert "not installed" in out
