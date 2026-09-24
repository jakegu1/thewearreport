# Spike: one sweep of London in under two minutes

`sweep_detect.py` is a reference implementation written by Claude to remove
ambiguity from tasks T-002 to T-005. Read it; do not import it.

## Measured (2026-09-23, 4-vCPU container, CPU only)

| Run | London time | Cameras available | Frames fetched | Fetch | Detect | ms/frame | Persons | Umbrellas |
|---|---|---|---|---|---|---|---|---|
| 1 (frames on disk, exploratory) | 09:30 Wed | 822 | 822 | 18 s | 67 s | 82 | 931 | 2 |
| 2 (this script, in memory) | 09:53 Wed | 795 | 795 | 35 s | 63 s | 79 | 822 | 2 |

Weather at the time: 18 °C, overcast, no rain — so umbrella detection was not
exercised. Run 2 wrote **zero** image files (verified with `find`).

Visual check on one frame (Kingsway/High Holborn, 17 boxes): every box was a real
person (precision ≈ 100%); about half of the visible, mostly distant, people were
missed (recall ≈ 50%). Near-field people show coat vs shirt, top color and
backpacks; sleeve length is not readable at 352×288.

About 39–40% of cameras show at least one person and 5–7% show five or more
(run 1: 318 and 58 of 822; run 2: 318 and 43 of 795): most JamCams point at
carriageways, not pavements.

## Run it

```bash
curl -L -o yolox_s.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
pip install onnxruntime opencv-python-headless numpy
python spike/sweep_detect.py --model yolox_s.onnx
```

## What production needs that the spike lacks

Retries and error categories (T-003), schema and publishing (T-005), weather
join (T-006), scheduling and alerts (T-007), measured precision (T-008), and
attribute classification (M2).
