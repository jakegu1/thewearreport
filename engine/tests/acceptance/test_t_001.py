"""Acceptance tests for T-001 (bootstrap). These are the task contract: do not edit."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from email.message import Message
from pathlib import Path

import jsonschema
import pytest

import license_check
import privacy_guard
from wearreport import settings

ROOT = Path(__file__).resolve().parents[3]
PRIVACY_GUARD = ROOT / "scripts" / "privacy_guard.py"
AC2_VARS = ("WEARREPORT_ENV", "TFL_APP_KEY", "METOFFICE_API_KEY", "NWS_USER_AGENT")


# AC2: typed settings read from the environment -------------------------------------


def test_ac2_settings_read_every_variable_from_environment() -> None:
    loaded = settings.load_settings(
        {
            "WEARREPORT_ENV": "production",
            "TFL_APP_KEY": "tfl-test-value",
            "METOFFICE_API_KEY": "metoffice-test-value",
            "NWS_USER_AGENT": "wearreport-test (ops@example.com)",
        }
    )
    assert loaded.env is settings.Environment.PRODUCTION
    assert loaded.tfl_app_key == "tfl-test-value"
    assert loaded.metoffice_api_key == "metoffice-test-value"
    assert loaded.nws_user_agent == "wearreport-test (ops@example.com)"


def test_ac2_settings_default_to_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEARREPORT_ENV", "test")
    monkeypatch.setenv("TFL_APP_KEY", "from-os-environ")
    monkeypatch.delenv("METOFFICE_API_KEY", raising=False)
    monkeypatch.setenv("NWS_USER_AGENT", "")
    loaded = settings.load_settings()
    assert loaded.env is settings.Environment.TEST
    assert loaded.tfl_app_key == "from-os-environ"
    assert loaded.metoffice_api_key is None
    assert loaded.nws_user_agent is None  # empty value means unset


def test_ac2_settings_missing_env_defaults_to_development() -> None:
    loaded = settings.load_settings({})
    assert loaded.env is settings.Environment.DEVELOPMENT
    assert loaded.tfl_app_key is None


def test_ac2_settings_reject_unknown_environment() -> None:
    with pytest.raises(settings.SettingsError):
        settings.load_settings({"WEARREPORT_ENV": "staging"})


def test_ac2_settings_repr_hides_secrets() -> None:
    loaded = settings.load_settings({"TFL_APP_KEY": "s3cr3t-tfl", "METOFFICE_API_KEY": "s3cr3t-mo"})
    assert "s3cr3t" not in repr(loaded)


# AC3: CI workflow shape --------------------------------------------------------------


def _jobs(workflow: str) -> list[str]:
    body = workflow.split("\njobs:\n", 1)[1]
    return re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", body, flags=re.MULTILINE)


def test_ac3_ci_workflow_triggers_permissions_and_jobs() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert re.search(r"^  pull_request:", ci, flags=re.MULTILINE)
    assert re.search(r"^  push:\n    branches: \[main\]", ci, flags=re.MULTILINE)
    assert "pull_request_target" not in ci
    assert re.search(r"^permissions:\n  contents: read\s*$", ci, flags=re.MULTILINE)
    jobs = _jobs(ci)
    assert jobs
    assert ci.count("runs-on: ubuntu-24.04") == len(jobs)
    assert ci.count("timeout-minutes:") >= len(jobs)
    assert "runs-on: ubuntu-latest" not in ci
    for needle in (
        "ruff check",
        "ruff format --check",
        "mypy --strict",
        "pytest",
        "gitleaks",
        "license_check.py",
        "privacy_guard.py",
        "test_schemas.py",
    ):
        assert needle in ci or needle in _make_recipes_used_by(ci), needle


def _make_recipes_used_by(workflow: str) -> str:
    """Return the Makefile text when the workflow delegates steps to make targets."""
    return (ROOT / "Makefile").read_text() if "make " in workflow else ""


@pytest.mark.parametrize("name", ["ci.yml", "public-guard.yml"])
def test_ac3_actions_pinned_to_full_sha_with_version_comment(name: str) -> None:
    text = (ROOT / ".github" / "workflows" / name).read_text()
    uses = re.findall(r"uses:\s*(\S+)(.*)$", text, flags=re.MULTILINE)
    assert uses
    for ref, rest in uses:
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", ref), ref
        assert re.match(r"\s+#\s*v?\d", rest), f"missing version comment: {ref}"


def test_ac3_public_guard_workflow_keeps_its_checks() -> None:
    text = (ROOT / ".github" / "workflows" / "public-guard.yml").read_text()
    assert "run: python3 tools/public_guard.py\n" in text
    assert "run: python3 tools/public_guard.py --history" in text
    assert "fetch-depth: 0" in text
    assert "runs-on: ubuntu-24.04" in text


# AC4: make check runs the public guard -----------------------------------------------


def test_ac4_make_check_runs_public_guard() -> None:
    makefile = (ROOT / "Makefile").read_text()
    assert re.search(r"^check:", makefile, flags=re.MULTILINE)
    assert "python3 tools/public_guard.py" in makefile


# AC5: static privacy guard ------------------------------------------------------------

VIOLATIONS = {
    "cv2_imwrite": "import cv2\n\ndef f(img):\n    cv2.imwrite('out.png', img)\n",
    "cv2_imwrite_from_import": "from cv2 import imwrite\n\ndef f(img):\n    imwrite(path, img)\n",
    "pil_image_save": (
        "from PIL import Image\n\ndef f(arr):\n    Image.fromarray(arr).save('crop.jpg')\n"
    ),
    "pil_image_image_save": "import PIL.Image\n\ndef f(im, p):\n    PIL.Image.Image.save(im, p)\n",
    "imageio_imwrite": "import imageio\n\ndef f(a):\n    imageio.imwrite('a.webp', a)\n",
    "imageio_v3_imwrite": "import imageio.v3 as iio\n\ndef f(a, p):\n    iio.imwrite(p, a)\n",
    "open_jpg_wb": "def f(b):\n    with open('frame.jpg', 'wb') as fh:\n        fh.write(b)\n",
    "open_png_mode_kw": "def f(b):\n    open('x.PNG', mode='w').write(b)\n",
    "open_fstring_jpeg_append": "def f(b, n):\n    open(f'/tmp/{n}.jpeg', 'ab').write(b)\n",
    "path_open_bmp": (
        "from pathlib import Path\n\ndef f(b, d):\n    (Path(d) / 'x.bmp').open('wb').write(b)\n"
    ),
}

CLEAN = (
    "import json\n"
    "from pathlib import Path\n\n"
    "def f(counts, img_bytes):\n"
    "    Path('out.jsonl').open('a').write(json.dumps(counts))\n"
    "    with open('model.png', 'rb') as fh:\n"
    "        fh.read()\n"
    "    settings.save()\n"
    "    return img_bytes\n"
)


@pytest.mark.parametrize("name", sorted(VIOLATIONS))
def test_ac5_privacy_guard_flags_violation_in_source(name: str) -> None:
    assert privacy_guard.scan_source(VIOLATIONS[name], "engine/wearreport/x.py")


def test_ac5_privacy_guard_passes_clean_source() -> None:
    assert privacy_guard.scan_source(CLEAN, "engine/wearreport/x.py") == []


def _run_guard(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PRIVACY_GUARD), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.mark.parametrize("name", sorted(VIOLATIONS))
def test_ac5_privacy_guard_cli_fails_on_each_violation(tmp_path: Path, name: str) -> None:
    _write(tmp_path, "engine/wearreport/clean.py", CLEAN)
    _write(tmp_path, "engine/wearreport/sub/bad.py", VIOLATIONS[name])
    result = _run_guard(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "bad.py" in result.stdout


def test_ac5_privacy_guard_cli_passes_clean_tree_and_ignores_tests(tmp_path: Path) -> None:
    _write(tmp_path, "engine/wearreport/clean.py", CLEAN)
    _write(tmp_path, "engine/tests/test_x.py", VIOLATIONS["cv2_imwrite"])
    result = _run_guard(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ac5_privacy_guard_passes_on_this_repository() -> None:
    result = _run_guard(ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


# AC6: licence check -------------------------------------------------------------------


def _metadata(name: str, **fields: str | list[str]) -> Message:
    msg = Message()
    msg["Name"] = name
    msg["Version"] = "1.0.0"
    for key, value in fields.items():
        for item in value if isinstance(value, list) else [value]:
            msg[key.replace("_", "-")] = item
    return msg


def test_ac6_licence_check_rejects_agpl() -> None:
    meta = _metadata("copyleft-pkg", License_Expression="AGPL-3.0")
    assert license_check.license_problem(meta) is not None


def test_ac6_licence_check_accepts_apache() -> None:
    meta = _metadata("permissive-pkg", License_Expression="Apache-2.0")
    assert license_check.license_problem(meta) is None


def test_ac6_licence_check_rejects_agpl_classifier() -> None:
    meta = _metadata(
        "copyleft-pkg",
        Classifier="License :: OSI Approved :: GNU Affero General Public License v3",
    )
    assert license_check.license_problem(meta) is not None


def test_ac6_licence_check_rejects_ultralytics_by_name() -> None:
    meta = _metadata("ultralytics", License_Expression="Apache-2.0")
    assert license_check.license_problem(meta) is not None


def test_ac6_licence_check_passes_on_this_repository() -> None:
    assert license_check.main([]) == 0


# AC7: schema validation ---------------------------------------------------------------


def test_ac7_example_schema_validates_sample_document() -> None:
    schema = json.loads((ROOT / "data" / "schema" / "example.v0.json").read_text())
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    samples = sorted((ROOT / "data" / "schema" / "samples").glob("example.v0*.json"))
    assert samples
    for sample in samples:
        validator_cls(schema).validate(json.loads(sample.read_text()))
    with pytest.raises(jsonschema.ValidationError):
        validator_cls(schema).validate({"unexpected": True})


# AC8: .env.example and fixtures/LICENSES.md -------------------------------------------


def test_ac8_env_example_lists_every_variable_empty_with_comment() -> None:
    lines = (ROOT / ".env.example").read_text().splitlines()
    for var in AC2_VARS:
        idx = lines.index(f"{var}=")  # present with an empty value
        assert idx > 0 and lines[idx - 1].startswith("# "), f"{var} needs a one-line comment"
    assigned = [ln for ln in lines if ln and not ln.startswith("#")]
    assert all(ln.endswith("=") for ln in assigned), "no values in .env.example"


def test_ac8_fixture_licenses_table_exists_and_is_empty() -> None:
    text = (ROOT / "fixtures" / "LICENSES.md").read_text()
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    assert len(rows) == 2, "header and separator only"
    header = [c.strip().lower() for c in rows[0].strip().strip("|").split("|")]
    assert header == ["source", "license", "file"]
    assert set(rows[1].replace("|", "").strip()) <= {"-", ":", " "}
