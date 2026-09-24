from __future__ import annotations

from pathlib import Path

import pytest

import privacy_guard


def _messages(source: str) -> list[str]:
    return [f.message for f in privacy_guard.scan_source(source, "engine/wearreport/m.py")]


@pytest.mark.parametrize(
    "source",
    [
        "import imageio\nimageio.mimsave('a.gif', frames)\n",
        "import imageio\nimageio.imsave(p, a)\n",
        "open(p + '.jpg', 'w+b')\n",
        "open('a.tiff', 'xb')\n",
        "open(file='a.png', mode='wb')\n",
        "import io\nio.open('a.png', 'wb')\n",
        "open('a.jpg', mode)\n",  # computed mode cannot be proven read-only
        "Path(d, 'a.png').write_bytes(b)\n",
        "Path(d).with_suffix('.webp').open('wb')\n",
        "Path(d).joinpath('x', 'a.gif').write_bytes(b)\n",
        "open('{}.jpg'.format(n), 'wb')\n",
        "img.save('out.png')\n",
        "from PIL import Image as I\nim.save(buf, format='JPEG')\n",
    ],
)
def test_flags_image_writes(source: str) -> None:
    assert _messages(source)


@pytest.mark.parametrize(
    "source",
    [
        "open('a.jpg')\n",
        "open('a.jpg', 'rb')\n",
        "open('out.jsonl', 'a')\n",
        "open(path, 'wb')\n",  # extension not visible statically: runtime test covers it
        "Path('a.png').read_bytes()\n",
        "Path('out.json').write_text(s)\n",
        "model.save(state)\n",
        "import numpy as np\nnp.asarray(buf)\n",
    ],
)
def test_allows_non_image_io(source: str) -> None:
    assert _messages(source) == []


def test_reports_line_numbers() -> None:
    source = "import cv2\n\n\ncv2.imwrite('a.png', x)\n"
    (finding,) = privacy_guard.scan_source(source, "engine/wearreport/m.py")
    assert finding.line == 4
    assert str(finding).startswith("engine/wearreport/m.py:4: ")


def test_unparseable_file_is_a_finding() -> None:
    assert _messages("def broken(:\n")


def test_missing_engine_directory_is_usage_error(tmp_path: Path) -> None:
    assert privacy_guard.main(["--root", str(tmp_path)]) == 2
