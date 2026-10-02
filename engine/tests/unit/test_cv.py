from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from wearreport import _cv
from wearreport._cv import cv2
from wearreport.testing.fake_cameras import jpeg_declaring, radiance_hdr

ENGINE = Path(__file__).resolve().parents[2] / "wearreport"


def _cv2_imports(path: Path) -> list[int]:
    """Lines of `path` that import cv2 itself, in any form."""
    lines = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            if any(a.name == "cv2" or a.name.startswith("cv2.") for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            lines += [node.lineno] if node.module.split(".")[0] == "cv2" else []
    return lines


def test_no_engine_module_but_the_shim_imports_cv2() -> None:
    modules = sorted(ENGINE.rglob("*.py"))
    assert ENGINE / "fetch.py" in modules
    found = {
        str(p.relative_to(ENGINE)): lines
        for p in modules
        if p.name != "_cv.py" and (lines := _cv2_imports(p))
    }
    assert found == {}
    assert _cv2_imports(ENGINE / "_cv.py")


def test_cv2_users_take_it_from_the_shim() -> None:
    for name in ("fetch.py", "testing/fake_cameras.py"):
        tree = ast.parse((ENGINE / name).read_text())
        sources = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "wearreport._cv" in sources, name


def test_settings_are_in_the_environment() -> None:
    assert os.environ["OPENCV_TEMP_PATH"] == _cv.TEMP_PATH
    assert os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] == "2073600"
    assert _cv.MAX_IMAGE_PIXELS == 2_073_600


def test_temp_path_can_never_be_created() -> None:
    with pytest.raises(OSError):
        os.makedirs(_cv.TEMP_PATH)
    assert not os.path.exists(_cv.TEMP_PATH)


def _python(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("OPENCV_")}
    child_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_importing_the_shim_after_a_bare_cv2_import_fails() -> None:
    proc = _python("import cv2\nimport wearreport._cv\n")
    assert proc.returncode != 0
    assert "imported before wearreport._cv" in proc.stderr


def test_importing_the_shim_first_works() -> None:
    proc = _python("import wearreport._cv\nimport cv2\nprint(cv2.__name__)\n")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "cv2"


def test_hdr_decodes_through_a_temp_file_without_the_shim(tmp_path: Path) -> None:
    """The control: without the shim's settings, OpenCV writes this body to a file."""
    code = (
        "import sys, cv2, numpy as np\n"
        "body = sys.stdin.buffer.read()\n"
        "frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)\n"
        "print(frame.shape)\n"
    )
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("OPENCV_")}
    child_env["OPENCV_TEMP_PATH"] = str(tmp_path)  # OpenCV deletes the file afterwards
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=radiance_hdr(),
        env=child_env,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == b"(8, 8, 3)"
    assert list(tmp_path.iterdir()) == []


def test_hdr_decode_fails_closed_with_the_shim() -> None:
    body = np.frombuffer(radiance_hdr(), np.uint8)
    try:
        frame = cv2.imdecode(body, cv2.IMREAD_COLOR)
    except cv2.error:
        frame = None
    assert frame is None


def test_pixel_cap_refuses_a_huge_declared_frame_quickly() -> None:
    body = jpeg_declaring(30000, 30000)
    assert len(body) < 2048
    started = time.monotonic()
    with pytest.raises(cv2.error):
        cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert time.monotonic() - started < 2


def test_frames_under_the_cap_still_decode() -> None:
    frame = cv2.imdecode(np.frombuffer(jpeg_declaring(352, 288), np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None and frame.shape == (288, 352, 3)


def test_encode_jpeg_round_trips() -> None:
    pixels = np.full((8, 8, 3), 200, dtype=np.uint8)
    body = _cv.encode_jpeg(pixels)
    assert body[:3] == b"\xff\xd8\xff" and body[-2:] == b"\xff\xd9"
    decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == (8, 8, 3)
