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


# Round 3 --------------------------------------------------------------------------------

# Each case from finding 1 of the round-2 re-review of T-001, as written there.
ROUND2_FINDING_CASES = {
    "videowriter_chained_write": (
        "import cv2\ncv2.VideoWriter('out.avi', fourcc, 1, (w, h)).write(frame)\n"
    ),
    "imwritemulti": "import cv2\ncv2.imwritemulti('a.tiff', frames)\n",
    "imwriteanimation": "import cv2\ncv2.imwriteanimation('a.webp', anim)\n",
    "plt_savefig_after_imshow": (
        "import matplotlib.pyplot as plt\nplt.imshow(frame)\nplt.savefig('frame.png')\n"
    ),
    "imageio_imopen_write": "import imageio.v3 as iio\niio.imopen('a.png', 'w').write(frame)\n",
    "fsspec_open_wb": "import fsspec\nfsspec.open(p, 'wb')\n",
    "tarfile_open_w": "import tarfile\ntarfile.open(p, 'w')\n",
    "zipfile_writestr": (
        "import zipfile\nzipfile.ZipFile('crops.zip', 'w').writestr('c.jpg', buf)\n"
    ),
}


@pytest.mark.parametrize("name", sorted(ROUND2_FINDING_CASES))
def test_flags_round2_finding_cases(name: str) -> None:
    assert _messages(ROUND2_FINDING_CASES[name])


# Required change 1: writers, with computed paths so rule B cannot be what catches them.
WRITER_CASES = {
    "videowriter": "import cv2\ncv2.VideoWriter(p, fourcc, 1.0, size)\n",
    "videowriter_from_import_as": "from cv2 import VideoWriter as VW\nVW(p, fourcc, 1.0, size)\n",
    "videowriter_rebinding": "import cv2\nmake = cv2.VideoWriter\nmake(p, fourcc, 1.0, size)\n",
    "videowriter_empty_then_open": (
        "import cv2\nvw = cv2.VideoWriter()\nvw.open(p, fourcc, 1.0, size)\n"
    ),
    "imwritemulti": "import cv2\ncv2.imwritemulti(p, frames)\n",
    "imwritemulti_rebinding": "import cv2\nw = cv2.imwritemulti\nw(p, frames)\n",
    "imwriteanimation": "import cv2\ncv2.imwriteanimation(p, anim)\n",
    "imwriteanimation_from_import": "from cv2 import imwriteanimation\nimwriteanimation(p, a)\n",
    "plt_savefig": "import matplotlib.pyplot as plt\nplt.savefig(p)\n",
    "plt_savefig_buffer": "import matplotlib.pyplot as plt\nplt.savefig(buf, format='png')\n",
    "figure_savefig_method": "fig.savefig(p)\n",
    "figure_savefig_unbound": (
        "from matplotlib.figure import Figure\nFigure.savefig(fig, p, dpi=72)\n"
    ),
    "savefig_from_import_as": "from matplotlib.pyplot import savefig as sf\nsf(p)\n",
    "savefig_rebinding": "import matplotlib.pyplot as plt\ns = plt.savefig\ns(p)\n",
    "canvas_print_png": "canvas.print_png(p)\n",
    "imopen_write": "import imageio.v3 as iio\niio.imopen(p, 'w')\n",
    "imopen_computed_mode": "import imageio.v3 as iio\niio.imopen(p, io_mode)\n",
    "imopen_from_import_as": "from imageio.v3 import imopen as o\no(p, 'w', extension='.png')\n",
    "imageio_get_writer": "import imageio\nimageio.get_writer(p, fps=1)\n",
    "imageio_volwrite": "import imageio\nimageio.volwrite(p, vol)\n",
    "numpy_savetxt": "import numpy as np\nnp.savetxt(p, frame.reshape(-1, 3))\n",
    "numpy_savetxt_from_import": "from numpy import savetxt\nsavetxt(p, frame)\n",
    "numpy_memmap_write": "import numpy as np\nnp.memmap(p, dtype=np.uint8, mode='w+', shape=s)\n",
    "numpy_memmap_default_mode": "import numpy as np\nnp.memmap(p, np.uint8)\n",
    "numpy_open_memmap": (
        "from numpy.lib.format import open_memmap\nopen_memmap(p, mode='w+', shape=s)\n"
    ),
    "joblib_dump": "import joblib\njoblib.dump(frame, p)\n",
    "joblib_from_import_as": "from joblib import dump as jd\njd(frame, p)\n",
    "joblib_rebinding": "import joblib\nsave_it = joblib.dump\nsave_it(frame, p)\n",
    "cv2_filestorage_write": "import cv2\ncv2.FileStorage(p, cv2.FILE_STORAGE_WRITE)\n",
    "urllib_urlopener_retrieve": (
        "import urllib.request\nurllib.request.URLopener().retrieve(u, p)\n"
    ),
    "pil_show": "from PIL import Image\nImage.fromarray(frame).show()\n",
}


@pytest.mark.parametrize("name", sorted(WRITER_CASES))
def test_flags_writers(name: str) -> None:
    assert _messages(WRITER_CASES[name])


@pytest.mark.parametrize(
    "source",
    [
        "import cv2\nvw = cv2.VideoWriter(p, f, 1.0, s)\nvw.write(frame)\n",
        "import cv2\nclass S:\n    def __init__(self):\n"
        "        self.vw = cv2.VideoWriter(p, f, 1.0, s)\n"
        "    def add(self, frame):\n        self.vw.write(frame)\n",
        "import cv2\nwith cv2.VideoWriter(p, f, 1.0, s) as vw:\n    vw.write(frame)\n",
    ],
)
def test_flags_each_videowriter_write(source: str) -> None:
    lines = {f.line for f in privacy_guard.scan_source(source, MODULE)}
    write_line = next(i for i, x in enumerate(source.splitlines(), 1) if ".write(" in x)
    assert write_line in lines


def test_flags_videowriter_write_through_annotation_alone() -> None:
    source = "import cv2\ndef add(vw: cv2.VideoWriter, frame):\n    vw.write(frame)\n"
    assert [f.line for f in privacy_guard.scan_source(source, MODULE)] == [3]


# Required change 2: rule A, binary modes on any opener, and archives in write modes.
RULE_A_CASES = {
    "fsspec_open_mode_kw": "import fsspec\nfsspec.open(p, mode='ab')\n",
    "fsspec_from_import_as": "from fsspec import open as fopen\nfopen(p, 'wb')\n",
    "smart_open": "import smart_open\nsmart_open.open(p, 'wb')\n",
    "filesystem_method_open": "fs.open(p, 'wb')\n",
    "filesystem_method_computed_mode": "fs.open(p, mode)\n",
    "os_fdopen": "import os\nos.fdopen(fd, 'wb')\n",
    "os_open_write_flags": "import os\nos.open(p, os.O_WRONLY | os.O_CREAT)\n",
    "tarfile_open_w_gz": "import tarfile\ntarfile.open(p, 'w:gz')\n",
    "tarfile_open_stream": "import tarfile\ntarfile.open(fileobj=fh, mode='w|gz')\n",
    "tarfile_open_append": "import tarfile\ntarfile.open(p, 'a')\n",
    "tarfile_open_exclusive_xz": "import tarfile\ntarfile.open(name=p, mode='x:xz')\n",
    "tarfile_open_computed_mode": "import tarfile\ntarfile.open(p, m)\n",
    "tarfile_tarfile": "import tarfile\ntarfile.TarFile(p, 'w')\n",
    "tarfile_tarfile_open": "import tarfile\ntarfile.TarFile.open(p, 'w:bz2')\n",
    "zipfile_append": "import zipfile\nzipfile.ZipFile(p, 'a')\n",
    "zipfile_exclusive": "import zipfile\nzipfile.ZipFile(p, mode='x')\n",
    "zipfile_from_import_compressed": (
        "from zipfile import ZIP_DEFLATED, ZipFile\nZipFile(p, 'w', ZIP_DEFLATED)\n"
    ),
    "zipfile_with_block": "import zipfile\nwith zipfile.ZipFile(p, 'w') as z:\n    z.write(f)\n",
    "gzip_open_w_is_binary": "import gzip\ngzip.open(p, 'w')\n",
    "gzip_gzipfile": "import gzip\ngzip.GzipFile(p, 'wb')\n",
    "bz2_file": "import bz2\nbz2.BZ2File(p, 'w')\n",
    "lzma_file": "import lzma\nlzma.LZMAFile(p, 'a')\n",
    "shelve_open_default_creates": "import shelve\nshelve.open(p)\n",
    "dbm_open_create": "import dbm\ndbm.open(p, 'c')\n",
    "dbm_dumb_open_default_creates": "import dbm.dumb\ndbm.dumb.open(p)\n",
    "sqlite_file": "import sqlite3\nsqlite3.connect(p)\n",
}


@pytest.mark.parametrize("name", sorted(RULE_A_CASES))
def test_flags_rule_a_binary_openers(name: str) -> None:
    assert _messages(RULE_A_CASES[name])


@pytest.mark.parametrize(
    "source",
    [
        "import fsspec\nfsspec.open(p, 'w')\n",
        "import fsspec\nfsspec.open(p, 'rb')\n",
        "import fsspec\nfsspec.open(p)\n",
        "fs.open(p)\n",
        "import tarfile\ntarfile.open(p)\n",
        "import tarfile\ntarfile.open(p, 'r:gz')\n",
        "import tarfile\ntarfile.open(p, 'r:xz')\n",
        "import zipfile\nzipfile.ZipFile(p)\n",
        "import zipfile\nzipfile.ZipFile(p, 'r')\n",
        "import gzip\ngzip.open('out.jsonl.gz', 'wt', encoding='utf-8')\n",
        "import gzip\ngzip.open(p, 'rb')\n",
        "import webbrowser\nwebbrowser.open(url, 2)\n",
        "import urllib.request\nurllib.request.urlopen(req, data)\n",
        "import urllib.request\nurllib.request.urlopen(req, timeout=20)\n",
        "import subprocess\nsubprocess.Popen(cmd, bufsize)\n",
        "import os\nos.open(p, os.O_RDONLY)\n",
        "import dbm\ndbm.open(p)\n",
        "import shelve\nshelve.open(p, 'r')\n",
        "import sqlite3\nsqlite3.connect(':memory:')\n",
        "import numpy as np\nnp.memmap(p, dtype=np.uint8, mode='r')\n",
        "import numpy as np\nnp.load(p, mmap_mode='r')\n",
        "import cv2\ncv2.FileStorage(p, cv2.FILE_STORAGE_READ)\n",
        "import imageio.v3 as iio\niio.imopen(p, 'r')\n",
        "import matplotlib.pyplot as plt\nplt.show()\n",
        "fh.write(json.dumps(row) + '\\n')\n",
    ],
)
def test_allows_rule_a_reads_and_text(source: str) -> None:
    assert _messages(source) == []


# Required change 3: rule B, image or video path literals in call arguments.
RULE_B_CASES = {
    "unknown_function": "store('frame.jpg', frame)\n",
    "keyword_argument_upper_case": "store(path='crop.PNG', data=b)\n",
    "fstring_last_part": "store(f'{d}/{n}.jpeg', b)\n",
    "fstring_middle_part": "store(f'{d}/{n}.mp4{suffix}', b)\n",
    "path_segment": "from pathlib import Path\nPath(d, 'a.heic')\n",
    "path_segment_nested": "from pathlib import Path\nstore(Path(d) / 'a.webm', b)\n",
    "with_suffix": "p.with_suffix('.avif')\n",
    "with_suffix_upper_case": "p.with_suffix('.MOV')\n",
    "concatenation": "store(d + '/x.mkv', b)\n",
    "percent_format": "store('%s.webm' % n, b)\n",
    "str_format": "store('{}.avi'.format(n), b)\n",
    "list_item": "import subprocess\nsubprocess.run(['ffmpeg', '-i', u, 'out.mp4'])\n",
    "tuple_item": "store(('a', 'b.gif'))\n",
    "dict_value": "store(**{'out': 'a.bmp'})\n",
    "conditional": "store('a.tif' if x else 'b.tif')\n",
    "shutil_copy": "import shutil\nshutil.copyfile(src, 'x.tiff')\n",
    "path_read_bytes": "from pathlib import Path\nPath('a.png').read_bytes()\n",
    "path_open_read": "from pathlib import Path\nPath('a.png').open('rb')\n",
    "log_message": "log.info('saved crop.jpg')\n",
    "url_without_literal_scheme": "from urllib.parse import urljoin\nurljoin(base, f'{cam}.jpg')\n",
    "requests_download": (
        "import requests\nr = requests.get(u)\nsave_bytes(r.content, f'{cam}.jpg')\n"
    ),
    "imencode_other_argument": "import cv2\ncv2.imencode('.jpg', load('x.png'))\n",
    "http_prefix_not_at_start": "store('cache/https://x.jpg')\n",
}


@pytest.mark.parametrize("name", sorted(RULE_B_CASES))
def test_flags_rule_b_media_literals(name: str) -> None:
    assert _messages(RULE_B_CASES[name])


# Rule B's exemptions, each of which must not be flagged.
RULE_B_EXEMPTIONS = {
    "cv2_imread": "import cv2\ncv2.imread('a.jpg')\n",
    "cv2_imread_nested_path": (
        "import cv2\nfrom pathlib import Path\ncv2.imread(str(Path(d, 'a.jpg')))\n"
    ),
    "cv2_imread_from_import_as": "from cv2 import imread as r\nr('a.png')\n",
    "cv2_imdecode": "import cv2\ncv2.imdecode(fetch('cam.jpg'), cv2.IMREAD_COLOR)\n",
    "pil_image_open": "from PIL import Image\nImage.open('a.jpg')\n",
    "pil_image_open_qualified": "import PIL.Image\nPIL.Image.open(Path(d) / 'x.png')\n",
    "imageio_imread": "import imageio\nimageio.imread('a.gif')\n",
    "imageio_v2_imread": "import imageio.v2 as imageio\nimageio.imread('a.bmp')\n",
    "imageio_v3_imread": "import imageio.v3 as iio\niio.imread('a.webp')\n",
    "cv2_imencode_extension": "import cv2\nok, buf = cv2.imencode('.jpg', crop)\n",
    "cv2_imencode_extension_keyword": "import cv2\ncv2.imencode(ext='.PNG', img=crop)\n",
    "str_endswith": "name.endswith('.jpg')\n",
    "str_endswith_tuple": "name.lower().endswith(('.jpg', '.png', '.mp4'))\n",
    "str_startswith": "name.startswith('thumb.jpg')\n",
    "https_literal": "import requests\nrequests.get('https://x.example/cam.jpg', timeout=10)\n",
    "https_fstring": "fetch(f'https://s3.example/{cam}.jpg')\n",
    "https_format": "fetch('https://s3.example/{}.mp4'.format(cam))\n",
    "http_concatenation": "fetch('http://x.example/' + cam + '.jpg')\n",
    "builtin_open_read": "open('model.png', 'rb')\n",
    "builtin_open_default_mode": "open('model.png')\n",
}


@pytest.mark.parametrize("name", sorted(RULE_B_EXEMPTIONS))
def test_allows_rule_b_exemptions(name: str) -> None:
    assert _messages(RULE_B_EXEMPTIONS[name]) == []


@pytest.mark.parametrize(
    "source",
    [
        "open('out.jsonl', 'w')\n",
        "open('out.jsonl', 'a', encoding='utf-8')\n",
        "with open(p, 'a') as fh:\n    fh.write(json.dumps(row) + '\\n')\n",
        "import json\njson.dump(obj, fh)\n",
        "import json\nwith open('sweep.jsonl', 'w') as fh:\n    json.dump(row, fh)\n",
        "from pathlib import Path\nPath(d, 'sweep.jsonl').open('a')\n",
        "from pathlib import Path\nPath(d).with_suffix('.jsonl').write_text(s)\n",
        "import logging\nlogging.info('frame decoded', extra={'cam': cam})\n",
    ],
)
def test_allows_text_writes(source: str) -> None:
    assert _messages(source) == []


# Bypasses found while hardening rules A and B: writers referenced without being called,
# modes hidden behind argument unpacking, star imports and archives.
HARDENING_CASES = {
    "writer_passed_to_map": "import cv2\nlist(map(cv2.imwrite, paths, frames))\n",
    "writer_in_partial": "import cv2, functools\nw = functools.partial(cv2.imwrite, p)\nw(f)\n",
    "writer_on_self": (
        "import cv2\nclass S:\n    def __init__(self):\n        self.w = cv2.imwrite\n"
    ),
    "writer_in_tuple": "import cv2\nw, r = cv2.imwrite, cv2.imread\n",
    "writer_in_conditional": "import cv2\nw = cv2.imwrite if x else None\n",
    "writer_in_dict": "import cv2\nWRITERS = {'jpg': cv2.imwrite}\n",
    "writer_through_getattr_literal": "import cv2\ngetattr(cv2, 'imwrite')(p, f)\n",
    "urlretrieve_reference": "from urllib.request import urlretrieve\nfetch = urlretrieve\n",
    "joblib_dump_reference": "import joblib\nhooks = [joblib.dump]\n",
    "open_starred_arguments": "open(*args)\n",
    "open_keyword_splat": "open(p, **kw)\n",
    "tempfile_keyword_splat": "import tempfile\ntempfile.NamedTemporaryFile(**kw)\n",
    "star_import": "from gzip import *\nopen(p, 'w')\n",
    "shutil_make_archive": "import shutil\nshutil.make_archive(base, 'zip', d)\n",
}


@pytest.mark.parametrize("name", sorted(HARDENING_CASES))
def test_flags_hardening_cases(name: str) -> None:
    assert _messages(HARDENING_CASES[name])


@pytest.mark.parametrize(
    "source",
    [
        "import cv2\ndef f(vw: cv2.VideoWriter) -> cv2.VideoWriter:\n    return vw\n",
        "import cv2\nfourcc = cv2.VideoWriter_fourcc(*'mp4v')\n",
        "import cv2\nload = cv2.imread\n",
        "import json\nsink = json.dump\n",
        "from json import dumps\n",
    ],
)
def test_allows_harmless_references(source: str) -> None:
    assert _messages(source) == []


def test_allowlist_does_not_exempt_writers_or_media_literals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = "engine/wearreport/allowed.py"
    monkeypatch.setattr(privacy_guard, "BINARY_WRITE_ALLOWLIST", frozenset({allowed}))
    assert _messages("import zipfile\nzipfile.ZipFile(p, 'w')\n", allowed) == []
    for source in (
        "store('frame.jpg', b)\n",
        "import cv2\ncv2.VideoWriter(p, f, 1.0, s)\n",
        "import joblib\njoblib.dump(frame, p)\n",
        "fig.savefig(p)\n",
    ):
        assert _messages(source, allowed), source


def test_cli_flags_media_literal(tmp_path: Path) -> None:
    module = tmp_path / "engine" / "wearreport" / "sink.py"
    module.parent.mkdir(parents=True)
    module.write_text("def f(x):\n    upload('frame.jpg', x)\n")
    assert privacy_guard.main(["--root", str(tmp_path)]) == 1


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


# The image-write exemption (T-008) ------------------------------------------------------

EXEMPT = privacy_guard.IMAGE_WRITE_EXEMPTION
EXEMPT_WRITES = [
    "import os\nfd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)\n",
    "import os\nwith os.fdopen(fd, 'wb') as fh:\n    fh.write(b)\n",
    "open(p, 'xb').write(b)\n",
]


@pytest.mark.parametrize("source", EXEMPT_WRITES)
def test_exemption_allows_binary_opens_only_in_the_exempt_file(source: str) -> None:
    assert _messages(source, EXEMPT) == []
    assert _messages(source, MODULE)
    assert _messages(source, "engine/wearreport/tools/other.py")


@pytest.mark.parametrize(
    "source",
    [
        "import io\nio.FileIO(p, 'w')\n",
        "import zipfile\nzipfile.ZipFile(p, 'w')\n",
        "import gzip\ngzip.open(p, 'wb')\n",
        "import sqlite3\nsqlite3.connect(p)\n",
        "import shelve\nshelve.open(p)\n",
        "import shutil\nshutil.make_archive(p, 'zip')\n",
        "import tempfile\ntempfile.NamedTemporaryFile()\n",
        "import tempfile\ntempfile.mkdtemp(suffix='.png')\n",
        "p.write_bytes(b)\n",
        "open(os.path.join(d, 'a.png'), 'wb')\n",
        "import os\nos.path.join(d, 'crop.png')\n",
        "Path(d, 'a.png').write_text(s)\n",
        "import cv2\ncv2.VideoWriter(p, f, 1.0, s)\n",
        "import imageio\nimageio.imwrite(p, f)\n",
        "fig.savefig(p)\n",
        "im.show()\n",
        "import cv2\nw = cv2.imwrite\n",
        "import cv2\ngetattr(cv2, 'imwrite')(p, f)\n",
        "send(name='frame.jpeg')\n",
    ],
)
def test_exemption_keeps_every_other_rule(source: str) -> None:
    assert _messages(source, EXEMPT), source


def test_exemption_is_one_constant_and_not_the_allowlist() -> None:
    assert isinstance(EXEMPT, str)
    assert EXEMPT not in privacy_guard.BINARY_WRITE_ALLOWLIST
    module = EXEMPT.removeprefix("engine/").removesuffix(".py").replace("/", ".")
    assert module == privacy_guard.EXEMPT_MODULE


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (MODULE, "import wearreport.tools.spotcheck\n"),
        (MODULE, "import wearreport.tools.spotcheck as s\n"),
        (MODULE, "from wearreport.tools import spotcheck\n"),
        (MODULE, "from wearreport.tools.spotcheck import ReviewDirectory\n"),
        (MODULE, "from .tools import spotcheck\n"),
        (MODULE, "from .tools.spotcheck import ReviewDirectory\n"),
        ("engine/wearreport/tools/other.py", "from . import spotcheck\n"),
        ("engine/wearreport/tools/other.py", "from .spotcheck import ReviewDirectory\n"),
        ("engine/wearreport/tools/sub/x.py", "from ..spotcheck import ReviewDirectory\n"),
        ("engine/wearreport/tools/__init__.py", "from .spotcheck import main\n"),
        (MODULE, "from wearreport import tools\ntools.spotcheck.ReviewDirectory()\n"),
        (MODULE, "import wearreport.tools\nd = wearreport.tools.spotcheck.ReviewDirectory\n"),
        (MODULE, "import importlib\nimportlib.import_module('wearreport.tools.spotcheck')\n"),
        (MODULE, "import runpy\nrunpy.run_path('engine/wearreport/tools/spotcheck.py')\n"),
        (MODULE, "__import__('wearreport.tools.spotcheck')\n"),
    ],
)
def test_no_other_module_may_use_the_exempt_one(path: str, source: str) -> None:
    assert _messages(source, path)


@pytest.mark.parametrize(
    "source",
    [
        "from wearreport import tools\n",
        "import wearreport.tools\n",
        "from wearreport.tools import other\n",
        "from . import spotcheck_notes\n",
        "x = 'spot check'\n",
    ],
)
def test_the_package_itself_is_not_off_limits(source: str) -> None:
    assert _messages(source, "engine/wearreport/x.py") == []


def test_the_exempt_module_may_name_itself() -> None:
    assert _messages("prog = f('python -m wearreport.tools.spotcheck')\n", EXEMPT) == []


def test_cli_symlink_to_the_exempt_file_is_checked_under_its_own_name(tmp_path: Path) -> None:
    tool = tmp_path / EXEMPT
    tool.parent.mkdir(parents=True)
    tool.write_text(EXEMPT_WRITES[2], encoding="utf-8")
    assert privacy_guard.main(["--root", str(tmp_path)]) == 0
    (tmp_path / "engine" / "wearreport" / "alias.py").symlink_to(tool)
    assert privacy_guard.main(["--root", str(tmp_path)]) == 1
