# Spot-checks

A spot-check measures how often the detector's person boxes are right on the real
cameras, without keeping any image. The tool samples live frames, shows the detections to
a reviewer through a temporary directory that it always deletes, and keeps only the
tallies. This directory holds one small JSON file of statistics per check.

Live checks wait for the maintainer's approval. The tool refuses to run in CI.

## Running a check

```bash
sh scripts/fetch_model.sh --with-m     # YOLOX-m is the default model
uv run python -m wearreport.tools.spotcheck --n 20 --reviewer NAME
```

| Option | Default | Meaning |
|---|---|---|
| `--n N` | required | frames to sample (1 to 500) |
| `--mode crops\|frames` | `crops` | one image per detection, or one per frame |
| `--min-persons K` | 3 | person detections a frame needs to be sampled |
| `--seed S` | none | seed for the random sample, to repeat it |
| `--reviewer NAME` | `unnamed` | recorded in the statistics (letters, digits, space, `.`, `_`, `-`) |
| `--out-dir DIR` | `spotchecks` | where the statistics file goes |
| `--model FILE` | `yolox_m.onnx` | a pinned model in `.models/` |
| `--judgements PATH` | none | read judgements from this JSON file instead of the keyboard |
| `--timeout SECONDS` | 1800 | time allowed for the review |
| `--dry-run` | off | sweep a local fake camera server serving the fixture photos (no network) |

The tool lists the cameras, fetches one sweep in memory, runs the detector with its
default thresholds and picks up to N frames with at least K person detections at random.
It renders them into a new directory `$TMPDIR/wearreport-spotcheck-XXXXXXXX` (mode 0700)
and prints its path and the numbering:

- `crops` mode: one image per detection (`crop-0001.png`, ...), the box grown by half its
  width on each side and half its height above and below, clipped to the frame, and
  enlarged. The image number is the box number.
- `frames` mode: one image per frame (`frame-0001.png`, ...), with numbered boxes.

Boxes are numbered 1, 2, 3, ... across the whole check. File names never carry a camera
id. `numbering.json` in the same directory lists each image's boxes and holds a template
for the judgements file.

### Judging from the keyboard

Without `--judgements`, the tool asks for one line per image:

| Token | Meaning |
|---|---|
| `n<box>` | box is not a person |
| `v<box>` | box is a person inside a vehicle |
| `m<count>` | people visible in the frame without a box (frames mode only) |
| bare `n` or `v` | this crop's box (crops mode only) |
| empty line | every box is a pedestrian, nobody missed |
| `q` | stop without writing statistics |

Tokens are separated by spaces or commas, for example `n3 v5 m1`. An invalid line is
rejected with a message and asked again.

### Judging with a JSON file

With `--judgements PATH` (the file must not exist when the tool starts), the tool polls
for `PATH` until the timeout. Write one entry per image, keyed by image number:

```json
{
  "1": {"not_person": [2], "in_vehicle": [3], "missed": 1},
  "2": {"not_person": [], "in_vehicle": [], "missed": 0}
}
```

`not_person` and `in_vehicle` list box numbers on that image (default: none; a box cannot
be in both). `missed` is required in frames mode and not allowed in crops mode. An invalid
file is rejected with a message; fix it and save it again. The file holds only numbers
and is left in place.

## Statistics file

`<out-dir>/YYYY-MM-DD.json`; if that exists, `YYYY-MM-DD-2.json`, then `-3` and so on.
An existing file is never overwritten. The file has exactly these fields:

| Field | Type | Value |
|---|---|---|
| `date` | string | the date of the check (local time), `YYYY-MM-DD` |
| `reviewer` | string | the `--reviewer` name |
| `mode` | string | `crops` or `frames` |
| `frames_reviewed` | integer | frames sampled and shown |
| `boxes_shown` | integer | person boxes shown |
| `boxes_not_person` | integer | boxes judged not a person |
| `boxes_in_vehicle` | integer | boxes judged a person inside a vehicle |
| `persons_missed` | integer or null | people without a box; `null` in crops mode |
| `precision_person` | number or null | 1 − not_person / shown |
| `precision_pedestrian` | number or null | 1 − (not_person + in_vehicle) / shown |
| `recall_estimate` | number or null | (shown − not_person) / (shown − not_person + missed) in frames mode; `null` in crops mode |
| `detector` | object | `{"model", "sha256", "conf"}`: the model name, its pinned SHA-256 and the confidence threshold |

Ratios are rounded to 4 decimal places, and are `null` when their denominator is 0 (no
boxes shown, or nothing to recall). Recall needs whole frames, so only `frames` mode
estimates it. The counts are kept so every ratio can be recomputed.

A `--dry-run` check describes the fixture photos, not the cameras, so it needs an
`--out-dir` other than `spotchecks/`. Never commit its output here.

## Privacy and cleanup

Rendered images exist only in the temporary directory, and the tool deletes it when it
exits: after a normal run, an error, the review timeout, or any catchable signal whose
default action ends the process: Ctrl-C (SIGINT), SIGTERM, SIGHUP, SIGQUIT, SIGUSR1,
SIGUSR2, SIGXCPU (`ulimit -t`), SIGVTALRM, SIGPROF, SIGPOLL, SIGPWR, SIGSTKFLT and the
real-time signals, where the platform has them. Repeated signals (Ctrl-C twice, or a
closing terminal's SIGHUP then SIGTERM) cannot interrupt the deletion. Nothing else is
written except the statistics file.

**SIGKILL cannot be handled** (nor can a power cut, or a crash with SIGSEGV, SIGBUS or
another fault signal): the directory then stays behind.
While it runs, the tool holds a lock on its directory; at start it deletes every
`wearreport-spotcheck-*` directory of the current user that is older than the timeout
and not locked by a running instance. To clean up by hand:
`rm -rf "${TMPDIR:-/tmp}"/wearreport-spotcheck-*`.

Never open the rendered images with a tool that uploads them (for example an AI
assistant's file reader), and never copy them out of the directory.
