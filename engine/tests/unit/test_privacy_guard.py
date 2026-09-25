from __future__ import annotations

from pathlib import Path

import pytest

import privacy_guard

MODULE = "engine/wearreport/m.py"


def _messages(source: str, filename: str = MODULE) -> list[str]:
    return [f.message for f in privacy_guard.scan_source(source, filename)]


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
        "Path(d).with_suffix('.webp').open(mode)\n",
        "Path(d).joinpath('x', 'a.gif').write_bytes(b)\n",
        "open('{}.jpg'.format(n), 'wb')\n",
        "Path(d, 'a.png').write_text(s)\n",
        "img.save('out.png')\n",
        "from PIL import Image as I\nim.save(buf, format='JPEG')\n",
    ],
)
def test_flags_image_writes(source: str) -> None:
    assert _messages(source)


# Each case from the adversarial review of T-001 that the first version missed.
ADVERSARIAL_CASES = {
    "urlretrieve_attr": "import urllib.request\nurllib.request.urlretrieve(url, 'cam.jpg')\n",
    "urlretrieve_no_dest": "import urllib.request\nurllib.request.urlretrieve(url)\n",
    "urlretrieve_from_import": "from urllib.request import urlretrieve\nurlretrieve(url, p)\n",
    "os_path_join_open": "import os\nopen(os.path.join(d, 'a.jpg'), 'wb')\n",
    "os_path_join_text_mode": "import os\nopen(os.path.join(d, 'a.jpg'), 'w')\n",
    "os_path_join_alone": "import os\np = os.path.join(d, 'frame.png')\n",
    "from_import_as": "from imageio.v3 import imwrite as w\nw('a.png', f)\n",
    "import_as_module": "import cv2 as c\nc.imwrite(p, f)\n",
    "rebinding": "import cv2\nw = cv2.imwrite\nw(p, f)\n",
    "rebinding_chain": "import cv2\nw = cv2.imwrite\nv = w\nv(p, f)\n",
    "rebinding_local": (
        "import cv2\ndef f(x):\n    save_frame = cv2.imwrite\n    save_frame(p, x)\n"
    ),
    "named_temporary_file_suffix": (
        "import tempfile\ntempfile.NamedTemporaryFile(suffix='.jpg', delete=False)\n"
    ),
    "named_temporary_file_default_binary": "import tempfile\ntempfile.NamedTemporaryFile()\n",
    "temporary_file_binary_mode": "import tempfile\ntempfile.TemporaryFile('w+b')\n",
    "mkstemp_suffix": "import tempfile\ntempfile.mkstemp(suffix='.jpg')\n",
    "from_tempfile_import": "from tempfile import mkstemp\nmkstemp(suffix='.png')\n",
    "imencode_then_tofile": (
        "import cv2\nok, buf = cv2.imencode('.jpg', frame)\nbuf.tofile('x.jpg')\n"
    ),
    "imencode_then_open_wb": (
        "import cv2\nok, buf = cv2.imencode('.jpg', frame)\nopen(p, 'wb').write(buf)\n"
    ),
    "numpy_save": "import numpy as np\nnp.save(p, frame)\n",
    "numpy_savez": "import numpy as np\nnp.savez(p, frame=frame)\n",
    "numpy_savez_compressed": "import numpy as np\nnp.savez_compressed(p, frame=frame)\n",
    "numpy_from_import_save": "from numpy import save\nsave(p, frame)\n",
    "frame_tofile": "frame.tofile('f.jpg')\n",
    "frame_tofile_computed": "frame.tofile(p)\n",
    "io_fileio": "import io\nio.FileIO('a.jpg', 'w')\n",
    "io_fileio_no_extension": "import io\nio.FileIO(p, 'w')\n",
    "io_fileio_computed_mode": "import io\nio.FileIO(p, mode)\n",
    "codecs_open": "import codecs\ncodecs.open('a.jpg', 'wb')\n",
    "codecs_open_no_extension": "import codecs\ncodecs.open(p, 'wb')\n",
    "pickle_dump_open_wb": "import pickle\npickle.dump(frame, open(p, 'wb'))\n",
    "pil_save_without_pil_import": "def f(im, p):\n    im.save(p)\n",
    "gzip_open_wb": "import gzip\ngzip.open(p, 'wb')\n",
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL_CASES))
def test_flags_adversarial_review_cases(name: str) -> None:
    assert _messages(ADVERSARIAL_CASES[name])


@pytest.mark.parametrize("mode", ["wb", "ab", "xb", "w+b", "r+b", "rb+", "bw", "a+b"])
@pytest.mark.parametrize(
    "template",
    [
        "open(p, {mode!r})\n",
        "open(p, mode={mode!r})\n",
        "import io\nio.open(p, {mode!r})\n",
        "import codecs\ncodecs.open(p, {mode!r})\n",
        "p.open({mode!r})\n",
        "Path(p).open(mode={mode!r})\n",
        "with open(os_path, {mode!r}) as fh:\n    fh.write(data)\n",
    ],
)
def test_flags_any_binary_write_mode_regardless_of_path(template: str, mode: str) -> None:
    assert _messages(template.format(mode=mode))


@pytest.mark.parametrize(
    "source",
    [
        "p.write_bytes(b)\n",
        "Path(d, 'out.bin').write_bytes(b)\n",
        "open(path, mode)\n",  # computed mode could be binary
        "from io import open as o\no(p, 'wb')\n",
        "import builtins\nbuiltins.open(p, 'wb')\n",
    ],
)
def test_flags_other_binary_writes(source: str) -> None:
    assert _messages(source)


@pytest.mark.parametrize(
    "source",
    [
        "open('a.jpg')\n",
        "open('a.jpg', 'rb')\n",
        "open(p, 'r+')\n",  # text mode
        "open('out.jsonl', 'a')\n",
        "open('out.jsonl', 'w')\n",
        "open(p, 'w', encoding='utf-8')\n",
        "open(p, mode='a', encoding='utf-8')\n",
        "with open(p, 'a') as fh:\n    fh.write(json.dumps(row) + '\\n')\n",
        "import json\njson.dump(obj, open(p, 'w'))\n",
        "Path('out.jsonl').open('a')\n",
        "p.open('w', encoding='utf-8')\n",
        "Path('a.png').read_bytes()\n",
        "Path('out.json').write_text(s)\n",
        "p.write_text(s)\n",
        "settings.save()\n",
        "import numpy as np\nnp.asarray(buf)\n",
        "import cv2\nok, buf = cv2.imencode('.jpg', crop)\n",  # in-memory encode
        "from PIL import Image\nImage.open(io.BytesIO(data))\n",  # a read
        "from PIL import Image\nImage.open('bwa.png')\n",  # file name, not a mode
        "import webbrowser\nwebbrowser.open(url)\n",
        "import os\nos.path.join(d, 'sweep.jsonl')\n",
        "import tempfile\ntempfile.TemporaryDirectory()\n",
        "import tempfile\ntempfile.NamedTemporaryFile('w', suffix='.jsonl')\n",
        "import tempfile\ntempfile.mkstemp(suffix='.jsonl')\n",
        "import io\nio.FileIO(p)\n",
        "import io\nio.BytesIO(data)\n",
    ],
)
def test_allows_text_writes_and_reads(source: str) -> None:
    assert _messages(source) == []


def test_allowlisted_module_may_write_binary_but_not_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = "engine/wearreport/allowed.py"
    monkeypatch.setattr(privacy_guard, "BINARY_WRITE_ALLOWLIST", frozenset({allowed}))
    binary = "open(p, 'wb')\np.write_bytes(b)\nimport io\nio.FileIO(p, 'w')\n"
    assert _messages(binary, allowed) == []
    assert _messages(binary, MODULE)  # other modules are still checked
    for source in (
        "open('a.jpg', 'wb')\n",
        "import cv2\ncv2.imwrite(p, f)\n",
        "frame.tofile(p)\n",
        "import numpy as np\nnp.save(p, f)\n",
        "import urllib.request\nurllib.request.urlretrieve(u, p)\n",
        "im.save(p)\n",
    ):
        assert _messages(source, allowed), source


def test_allowlist_is_empty() -> None:
    assert not privacy_guard.BINARY_WRITE_ALLOWLIST


def test_reports_line_numbers() -> None:
    source = "import cv2\n\n\ncv2.imwrite('a.png', x)\n"
    (finding,) = privacy_guard.scan_source(source, MODULE)
    assert finding.line == 4
    assert str(finding).startswith(f"{MODULE}:4: ")


def test_unparseable_file_is_a_finding() -> None:
    assert _messages("def broken(:\n")


def test_missing_engine_directory_is_usage_error(tmp_path: Path) -> None:
    assert privacy_guard.main(["--root", str(tmp_path)]) == 2


def test_cli_flags_binary_write_without_image_extension(tmp_path: Path) -> None:
    module = tmp_path / "engine" / "wearreport" / "sink.py"
    module.parent.mkdir(parents=True)
    module.write_text("def f(p, b):\n    open(p, 'wb').write(b)\n")
    assert privacy_guard.main(["--root", str(tmp_path)]) == 1
