"""Reference spike: one full sweep of London TfL JamCams -> person/umbrella counts.

Verified 2026-09-23 on a 4-vCPU container. This is a *reference* for the engine
tasks (T-002..T-005), not production code: it has no retries, persistence, or
scheduling.

Privacy (INV-1): frames are fetched into memory, decoded, detected, and dropped.
Nothing image-derived is written anywhere; only aggregate counts are printed.

Licenses: detector is YOLOX-s (Apache-2.0). Do NOT swap in `ultralytics`
(AGPL-3.0) for runtime use; see docs/decisions/0003-detector.md.

Usage:
    curl -L -o yolox_s.onnx \
      https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
    pip install onnxruntime opencv-python-headless numpy
    python spike/sweep_detect.py --model yolox_s.onnx
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import onnxruntime as ort

TFL_JAMCAMS = "https://api.tfl.gov.uk/Place/Type/JamCam"
INPUT = (640, 640)
PERSON, UMBRELLA = 0, 25  # COCO class ids
USER_AGENT = "wearreport-spike/0.1"


def list_cameras() -> list[dict]:
    req = urllib.request.Request(TFL_JAMCAMS, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        places = json.load(resp)
    cams = []
    for p in places:
        props = {a["key"]: a["value"] for a in p.get("additionalProperties", [])}
        if props.get("available") == "true" and props.get("imageUrl"):
            cams.append({"id": p["id"], "name": p["commonName"], "url": props["imageUrl"]})
    return cams


def fetch_frame(url: str) -> np.ndarray | None:
    """Download a frame into memory and decode it. Never touches disk."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            buf = np.frombuffer(resp.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def preproc(img: np.ndarray) -> tuple[np.ndarray, float]:
    padded = np.full((INPUT[0], INPUT[1], 3), 114, dtype=np.uint8)
    r = min(INPUT[0] / img.shape[0], INPUT[1] / img.shape[1])
    resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), r


def _grids() -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for s in (8, 16, 32):
        h, w = INPUT[0] // s, INPUT[1] // s
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        g = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(g)
        strides.append(np.full((*g.shape[:2], 1), s))
    return np.concatenate(grids, 1), np.concatenate(strides, 1)


GRIDS, STRIDES = _grids()


def detect(sess: ort.InferenceSession, img: np.ndarray, conf: float = 0.35, iou: float = 0.45) -> dict[int, int]:
    """Return {class_id: count} for person and umbrella after per-class NMS."""
    x, r = preproc(img)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    out[:, :2] = (out[:, :2] + GRIDS[0]) * STRIDES[0]
    out[:, 2:4] = np.exp(out[:, 2:4]) * STRIDES[0]
    scores = out[:, 4:5] * out[:, 5:]
    cls, sc = scores.argmax(1), scores.max(1)
    counts = {PERSON: 0, UMBRELLA: 0}
    for c in counts:
        m = (cls == c) & (sc >= conf)
        if not m.any():
            continue
        b = out[m, :4] / r
        boxes = [[float(cx - w / 2), float(cy - h / 2), float(w), float(h)] for cx, cy, w, h in b]
        keep = cv2.dnn.NMSBoxes(boxes, sc[m].tolist(), conf, iou)
        counts[c] = len(np.array(keep).flatten())
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolox_s.onnx")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])

    t0 = time.time()
    cams = list_cameras()
    with ThreadPoolExecutor(args.workers) as pool:
        frames = list(pool.map(lambda c: fetch_frame(c["url"]), cams))
    t_fetch = time.time() - t0

    t1 = time.time()
    per_cam = []
    for cam, img in zip(cams, frames):
        if img is None:
            continue
        per_cam.append(detect(sess, img))
    t_detect = time.time() - t1
    del frames  # drop all image data before reporting

    persons = np.array([c[PERSON] for c in per_cam])
    summary = {
        "cameras_listed": len(cams),
        "frames_ok": len(per_cam),
        "fetch_seconds": round(t_fetch, 1),
        "detect_seconds": round(t_detect, 1),
        "ms_per_frame": round(1000 * t_detect / max(1, len(per_cam))),
        "persons_total": int(persons.sum()),
        "umbrellas_total": int(sum(c[UMBRELLA] for c in per_cam)),
        "cameras_with_persons": int((persons >= 1).sum()),
        "cameras_with_5plus": int((persons >= 5).sum()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
