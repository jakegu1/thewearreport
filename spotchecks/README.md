# Spot-checks

A spot-check measures how often the detector's person boxes are right on the real
cameras, without keeping any image. The tool samples live frames, shows the detections to
a reviewer, either in a window straight from memory or through a temporary directory that
it always deletes, and keeps only the tallies. This directory holds one small JSON file of
statistics per check.

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
| `--view files\|window` | `window` on Windows, else `files` | show the images in a window from memory, or as files in a temporary directory (see below) |
| `--judgements PATH` | none | read judgements from this JSON file instead of the keyboard |
| `--timeout SECONDS` | 1800 | time allowed for the review |
| `--dry-run` | off | sweep a local fake camera server serving the fixture photos (no network) |
| `--judge NAME` | none | after the review, send the same crops to this DeepInfra model (see below; crops mode only) |
| `--judge-max-requests N` | required with `--judge` | the most requests the judge may make, retries included (1 to 10000) |

The tool lists the cameras, fetches one sweep in memory, runs the detector with its
default thresholds and picks up to N frames with at least K person detections at random.

On Windows, see [the install notes](../README.md#windows-spot-check-only) first.

### Judging in a window (`--view window`)

With `--view window` (crops mode only) the tool opens one window and shows each crop
there, enlarged by a whole factor, with its image number, "k of N" and a one-line key
legend. The crop goes from memory into the window as PNG data: no image file and no
directory is created anywhere. One key per crop:

| Key | Meaning |
|---|---|
| `Enter` or `Space` | the box is a pedestrian |
| `n` | the box is not a person |
| `v` | the box is a person inside a vehicle |
| `Backspace` | go back one crop, to change it |
| `q`, or closing the window | stop without writing statistics |

The window closes after the last crop, and the statistics are exactly those the keyboard
would give for the same answers. The review timeout closes the window too, and the tool
exits without statistics.

This is the default on Windows, unless `--mode frames` or `--judgements` is given: those
need the image files, so they use `--view files`. `--view window` with either is refused.
Elsewhere the default stays `--view files`; `--view window` works where Python has
tkinter and there is a display.

### Judging from files (`--view files`)

The tool renders the images into a new directory `$TMPDIR/wearreport-spotcheck-XXXXXXXX` (mode 0700)
and prints its path and the numbering. It refuses to start when that temporary directory
lies inside this repository or any git work tree, where a `git add` could commit the
images; set `TMPDIR` to a directory outside it.

- `crops` mode: one image per detection (`crop-0001.png`, ...), the box grown by half its
  width on each side and half its height above and below, clipped to the frame, and
  enlarged. The image number is the box number.
- `frames` mode: one image per frame (`frame-0001.png`, ...), with numbered boxes.

Boxes are numbered 1, 2, 3, ... across the whole check. File names never carry a camera
id. `numbering.json` in the same directory lists each image's boxes and holds a template
for the judgements file.

#### Judging from the keyboard

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

#### Judging with a JSON file

With `--judgements PATH` (the file must not exist when the tool starts), the tool polls
for `PATH` until the timeout. Write one entry per image, keyed by image number:

```json
{
  "1": {"not_person": [2], "in_vehicle": [3], "missed": 1},
  "2": {"not_person": [], "in_vehicle": [], "missed": 0}
}
```

`not_person` and `in_vehicle` list box numbers on that image (default: none; a box cannot
be in both). `missed` is required in frames mode and not allowed in crops mode. A key may
appear only once in each object. An invalid file is rejected with a message; fix it and save it again. The file holds only numbers
and is left in place.

### Pairing with a hosted judge (`--judge`)

No hosted vision model is good enough to replace the reviewer yet. It can, however, judge
the same crops, so that the reviewer's answers measure its error on real camera crops:

```bash
DEEPINFRA_API_KEY=... uv run python -m wearreport.tools.spotcheck --n 20 --reviewer NAME \
    --judge di-qwen3-vl-235b --judge-max-requests 200
```

`NAME` must be a model of `judge_hosted.DEEPINFRA`. The review runs as usual, with either
view. Only once the reviewer has finished and the judgements are valid does the tool send
the crops, one per request, to `https://api.deepinfra.com` (no other endpoint is
accepted): each crop is the array the reviewer was shown, encoded to PNG in memory. The
judge never sees a whole frame, a file of the review directory (deleted by then) or the
reviewer's answers. The prompt and the parsing of the answer are those of the judge
bake-off. Nothing about a crop is printed, logged or kept; only the counts go into the
statistics file.

- `--judge-max-requests N` is required, and at most N requests are made, retries
  included. Each request has a timeout; throttling and server errors are retried at most
  three times.
- Worst-case cost for the example model `di-qwen3-vl-235b` ($0.20 per million input
  tokens, $0.88 per million output tokens): about N × $0.00085, taking every request at
  the largest crop the judge accepts (2048 × 2048 pixels, about 4,200 input tokens) and
  its 10 output tokens; for example $0.17 for N = 200 and $8.50 for N = 10000.
- The key comes from `DEEPINFRA_API_KEY` only (see `.env.example`) and is sent as
  `Authorization: Bearer`. Without it, no Authorization header is sent, for an environment
  that adds the credential itself. The key is never printed, logged or put in an error,
  and a failed request is reported by its HTTP status only, never DeepInfra's message.
- A judge failure (the request limit, an HTTP error, a timeout, a malformed reply, Ctrl-C
  while the judge runs) never loses the review: the statistics are written with the
  judge's counts so far and `status` `incomplete`.
- `--judge` is refused with `--mode frames`, and with `--dry-run` unless the judge is a
  fake server on this machine (the tests' case): a dry run never calls DeepInfra.

## Statistics file

`<out-dir>/YYYY-MM-DD.json`; if that exists, `YYYY-MM-DD-2.json`, then `-3` and so on.
An existing file is never overwritten. The file has exactly these fields, and `judge`
when the check ran with `--judge` (files without it stay valid):

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
| `judge` | object | with `--judge` only: the reviewer x judge agreement, below |

Ratios are rounded to 4 decimal places, and are `null` when their denominator is 0 (no
boxes shown, or nothing to recall). Recall needs whole frames, so only `frames` mode
estimates it. The counts are kept so every ratio can be recomputed.

The `judge` block has exactly these fields:

| Field | Type | Value |
|---|---|---|
| `model` | string | the `--judge` name, e.g. `di-qwen3-vl-235b` |
| `provider` | string | `DeepInfra` |
| `status` | string | `complete` when the judge answered every crop, else `incomplete` |
| `requests` | integer | requests made, retries included |
| `input_tokens`, `output_tokens` | integer | tokens reported by the API |
| `cost_usd` | number | those tokens at the model's published prices (8 decimal places) |
| `confusion` | object | `{reviewer label: {judge answer: count}}` over the crops the judge answered |
| `judge_precision` | number or null | the judge's `person` and `in_vehicle` answers over its confident answers (`person`, `in_vehicle`, `not_person`) |

`confusion` has one row per reviewer label, `person` (a pedestrian), `in_vehicle` and
`not_person` (the reviewer has no unsure answer), and in each row one count per judge
answer: `person`, `in_vehicle`, `not_person` and `unsure`. `judge_precision` is the
reviewer's `precision_person` computed from the judge's answers instead, with its unsure
answers left out; it is `null` when the judge gave no confident answer. When `status` is
`incomplete`, the counts cover the crops the judge answered, in order, before it stopped.

A `--dry-run` check describes the fixture photos, not the cameras, so it needs an
`--out-dir` other than `spotchecks/`. Never commit its output here.

## Summary per week

```bash
uv run python -m wearreport.tools.spotcheck_summary [--dir spotchecks]
```

reads every statistics file (counts only; no network) and prints, for each ISO week and
each judge model:

- the reviewer's precision (pooled over the week's files) and n, the boxes shown;
- the judge's precision (pooled) and n, its confident answers;
- a corrected judge estimate: the week's judge answers corrected by inverting the
  reviewer x judge confusion pooled over all *earlier* weeks, Rogan-Gladen style. With
  the earlier sensitivity Se (the judge says a person when the reviewer does) and
  specificity Sp (the judge says not a person when the reviewer does), on confident
  answers, and this week's judge precision q: (q + Sp − 1) / (Se + Sp − 1), clipped to
  [0, 1]. It is `n/a` in the first week, and when Se + Sp − 1 is 0;
- the difference between that estimate and the reviewer's precision, in points, and
  whether it is within 3 points.

The last line says whether the condition holds: the two latest weeks with a corrected
estimate are both within 3 points of the reviewer's precision.

## Privacy and cleanup

With `--view window`, rendered images exist only in memory and in the window, and
nothing is written except the statistics file. With `--judge`, the crops also go, in
memory, to the DeepInfra API (and nowhere else), after the review.

With `--view files`, rendered images exist only in the temporary directory, and the tool
deletes it when it exits: after a normal run, an error, the review timeout, or any
catchable signal whose default action ends the process: Ctrl-C (SIGINT), SIGTERM, SIGHUP,
SIGQUIT, SIGUSR1, SIGUSR2, SIGXCPU (`ulimit -t`), SIGVTALRM, SIGPROF, SIGPOLL, SIGPWR,
SIGSTKFLT and the real-time signals, where the platform has them; on Windows, only
Ctrl-C and Ctrl-Break (SIGBREAK) in the tool's console. Repeated signals (Ctrl-C twice,
or a closing terminal's SIGHUP then SIGTERM) cannot interrupt the deletion. Nothing else
is written except the statistics file.

**SIGKILL cannot be handled** (nor can a power cut, or a crash with SIGSEGV, SIGBUS or
another fault signal). **On Windows, cleanup cannot run** when the process is ended from
outside: from Task Manager, by `taskkill /f`, or by any SIGTERM sent from another process
(Windows terminates the process outright). Nor can it run when the console window or
terminal tab is closed, or at log-off or shutdown: Windows ends the process before the
deletion finishes. In all these cases the directory stays behind, and a later run deletes
it once it is older than its timeout (see below).
While it runs, the tool holds a lock on its directory (on Windows, on a lock file next to
it, `.wearreport-spotcheck-*.lock`, deleted with the directory); at start it deletes every `wearreport-spotcheck-*` directory of the current user
that is older than the timeout and not locked by a running instance. To clean up by
hand: `rm -rf "${TMPDIR:-/tmp}"/wearreport-spotcheck-*`, or in PowerShell
`Remove-Item -Recurse -Force "$env:TEMP\wearreport-spotcheck-*"`.

Never open the rendered images with a tool that uploads them (for example an AI
assistant's file reader), and never copy them out of the directory.
