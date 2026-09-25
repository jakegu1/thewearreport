"""The engine's only import of OpenCV, with decoder limits set before it loads.

Every engine module takes `cv2` from here (`from wearreport._cv import cv2`). OpenCV
reads these settings once, when the library loads, so they must be in the environment
before the first `import cv2` in the process:

- OPENCV_TEMP_PATH points inside /dev/null, where no directory can ever be created.
  Decoders that cannot read from memory (Radiance HDR, for one) make `imdecode` write the
  body to a temporary file first; with this path that write fails and the decode fails,
  so no downloaded bytes reach disk (AGENTS.md INV-1).
- OPENCV_IO_MAX_IMAGE_PIXELS caps the size of a decoded image. JamCam frames are 352x288
  (about 0.1 megapixels); a small crafted JPEG can otherwise declare 30000x30000 and
  decode to gigabytes.

`encode_jpeg` is here, not in its one caller (the fake camera server), because the
static privacy guard recognises `cv2.imencode` by the name it is imported under.

If `cv2` was loaded before this module without these settings, importing this module
fails instead of running with them silently missing.
"""

from __future__ import annotations

import os
import sys

MAX_IMAGE_PIXELS = 1_000_000
TEMP_PATH = os.path.join(os.devnull, "opencv-temp")
SETTINGS = {
    "OPENCV_TEMP_PATH": TEMP_PATH,
    "OPENCV_IO_MAX_IMAGE_PIXELS": str(MAX_IMAGE_PIXELS),
}

if "cv2" in sys.modules and any(os.environ.get(k) != v for k, v in SETTINGS.items()):
    raise ImportError(
        "cv2 was imported before wearreport._cv, so its decoder limits are not in effect"
    )
os.environ.update(SETTINGS)

import cv2 as cv2  # noqa: E402  (must follow the settings above)
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

__all__ = ["MAX_IMAGE_PIXELS", "TEMP_PATH", "cv2", "encode_jpeg"]


def encode_jpeg(pixels: npt.NDArray[np.uint8]) -> bytes:
    """Encode an image as JPEG, in memory."""
    ok, encoded = cv2.imencode(".jpg", pixels)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()
