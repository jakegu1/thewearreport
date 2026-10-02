# Spot-checks

A spot-check measures how often the detector's person boxes are right on the real
cameras, without keeping any image. The tool samples live frames, shows the detections to
a reviewer, either in a window straight from memory or through a temporary directory that
it always deletes, and keeps only the tallies. This directory holds one small JSON file of
statistics per check and, in `boxes/`, one per-box file per check run with
`--record-boxes`: each box's height and label, and nothing else about it. In
`attributes/` it holds one attribute file per attribute session (`--attributes`, below):
counts and labels only.

Live checks wait for the maintainer's approval. The tool refuses to run in CI.

## Running a check

```bash
sh scripts/fetch_model.sh --with-m     # YOLOX-m is the default model
uv run python -m wearreport.tools.spotcheck --n 20 --reviewer NAME --record-boxes
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
| `--record-boxes` | off | also write the per-box file (below) next to the statistics file; crops mode only, refused with `--mode frames` |
| `--attributes` | off | run an attribute session instead of a detection check (see [Attribute session](#attribute-session-attributes)) |
| `--confirm-stop` | off | in the window, a first `q` asks before it stops (see below); needs `--view window`, refused otherwise |
| `--min-height N` | 31 | in an attribute session, show only person boxes at least N px tall (31 to 200; see [Box height](#box-height---min-height)); needs `--attributes`, refused otherwise |
| `--allow-dark` | off | start an attribute session in the window even when it is dark in London (see [Daylight](#daylight---allow-dark)); needs `--attributes`, refused otherwise |

The tool lists the cameras, fetches one sweep in memory, runs the detector with its
default thresholds and picks up to N frames with at least K person detections at random.
While it does, it prints progress to stderr: the number of cameras listed, then
`fetched k of N` and `detected k of N` at least every 100 frames, and a line just before
the review opens. These lines hold counts only, never a camera id or image data.

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
| `u` | cannot tell what the box is (left out of the statistics) |
| `Backspace` | go back one crop, to change it |
| `q`, or closing the window | stop without writing statistics |

With `--confirm-stop`, a single `q` never ends the review. The first `q` (or `Q`) keeps
the crop and the answers so far, and the header line shows "Stop and discard this
session? Press q again to stop, any other key to continue." A second `q` then stops
without writing statistics; any other key hides the question and is otherwise ignored (it
is not an answer and does not go back). Closing the window and the review timeout still
stop at once. Without the flag, `q` stops at once, as above.

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
| `u<box>` | cannot tell what the box is (left out of the statistics) |
| `m<count>` | people visible in the frame without a box (frames mode only) |
| bare `n`, `v` or `u` | this crop's box (crops mode only) |
| empty line | every box is a pedestrian, nobody missed |
| `q` | stop without writing statistics |

Tokens are separated by spaces or commas, for example `n3 v5 u6 m1`. An invalid line is
rejected with a message and asked again.

#### Judging with a JSON file

With `--judgements PATH` (the file must not exist when the tool starts), the tool polls
for `PATH` until the timeout. Write one entry per image, keyed by image number:

```json
{
  "1": {"not_person": [2], "in_vehicle": [3], "unsure": [4], "missed": 1},
  "2": {"not_person": [], "in_vehicle": [], "unsure": [], "missed": 0}
}
```

`not_person`, `in_vehicle` and `unsure` (cannot tell) list box numbers on that image
(default: none; a box can be in only one of them). `missed` is required in frames mode and not allowed in crops mode. A key may
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
the crops, one per request, except those the reviewer marked `u` (cannot tell), to `https://api.deepinfra.com` (no other endpoint is
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
| `boxes_shown` | integer | person boxes shown and judged (unsure boxes left out) |
| `boxes_not_person` | integer | boxes judged not a person |
| `boxes_in_vehicle` | integer | boxes judged a person inside a vehicle |
| `persons_missed` | integer or null | people without a box; `null` in crops mode |
| `precision_person` | number or null | 1 − not_person / shown |
| `precision_pedestrian` | number or null | 1 − (not_person + in_vehicle) / shown |
| `recall_estimate` | number or null | (shown − not_person) / (shown − not_person + missed) in frames mode; `null` in crops mode |
| `detector` | object | `{"model", "sha256", "conf"}`: the model name, its pinned SHA-256 and the confidence threshold |
| `judge` | object | with `--judge` only: the reviewer x judge agreement, below |

A box the reviewer marked unsure (`u`, cannot tell) is left out of every count here, and
so of every ratio: it is in none of `boxes_shown`, `boxes_not_person` and
`boxes_in_vehicle`. With `--record-boxes` it is counted in the per-box file (below);
without it, it is recorded nowhere. Ratios are rounded to 4
decimal places, and are `null` when their denominator is 0 (no
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
`not_person` (crops the reviewer marked unsure are not sent to the judge), and in each row one count per judge
answer: `person`, `in_vehicle`, `not_person` and `unsure`. `judge_precision` is the
reviewer's `precision_person` computed from the judge's answers instead, with its unsure
answers left out; it is `null` when the judge gave no confident answer. When `status` is
`incomplete`, the counts cover the crops the judge answered, in order, before it stopped.

A `--dry-run` check describes the fixture photos, not the cameras, so it needs an
`--out-dir` other than `spotchecks/`. Never commit its output here.

## Per-box file

With `--record-boxes` (crops mode only), next to each statistics file
`<out-dir>/<name>.json` the tool writes `<out-dir>/boxes/<name>.json`, with the same
name: the name is the first one free in both places, and neither file is ever
overwritten. It is one line of JSON with exactly these fields:

| Field | Type | Value |
|---|---|---|
| `date` | string | the date of the check, as in the statistics file |
| `started_at` | string | when the sweep began, UTC, to the minute: `YYYY-MM-DDTHH:MMZ` |
| `light` | string | `day`, `twilight` or `dark` over central London at `started_at` |
| `frames` | integer | frames reviewed (each from a different camera) |
| `detector` | object | as in the statistics file |
| `boxes` | array | one `[height_px, label]` per box shown, sorted |

`height_px` is the box height in source-frame pixels, rounded to a whole number. `label`
is `person` (a pedestrian), `in_vehicle`, `not_person` or `unsure`. There is no box
position or width, no camera id, no frame index and no image, and the sorted order says
nothing about which frame a box came from.

`light` comes from the sun's elevation at `started_at` over 51.5074 N, 0.1278 W (NOAA's
solar position formulae, the sun's centre, without refraction): `day` at 0° or above,
`twilight` (civil) from -6° up to 0°, `dark` below -6°. The formula agrees with the US Naval
Observatory's published sunrise, sunset and civil twilight times for London to within two
minutes.

## Near-field threshold

```bash
uv run python -m wearreport.tools.spotcheck_summary --heights [--dir spotchecks] [--data-dir PATH]
```

reads every per-box file in `<dir>/boxes/` (counts and heights only; no network) and
chooses the smallest box height worth counting. A malformed file is an error that names
it (exit 1). Boxes marked `unsure` are left out of every precision and reported as a
count. Precision is the boxes labelled `person` or `in_vehicle` over the judged ones
(`person`, `in_vehicle`, `not_person`), with a Wilson 95% interval. It prints:

- for each candidate height H, every distinct height in the files, smallest first: the
  judged boxes at least H pixels tall, their precision and interval, and the share of all
  judged person boxes (`person` or `in_vehicle`) they keep;
- for each candidate height H, the judged share: the judged boxes at least H pixels tall
  over those judged boxes and the `unsure` ones, as `judged share >= H px: judged of
  total (share)`, or `n/a` when there is no box that tall;
- the threshold: the smallest H whose boxes have a precision of at least
  `TARGET_PRECISION` (0.90) and a Wilson lower bound of at least `MIN_LOWER_BOUND` (0.85),
  over at least `MIN_BOXES_ABOVE` (100) judged boxes, with a judged share of at least
  `MIN_JUDGED_SHARE` (0.80); or `none`. A box nobody can verify must not be counted, so
  most boxes at or above the threshold must be ones the reviewer could judge: on real
  cameras most small boxes are unsure, and the few judged ones alone would pass the
  precision bar;
- at the threshold (over all boxes when there is none), the precision, n and interval per
  `light`, and with `--data-dir` per rain condition;
- a coverage line: sessions, dates, judged boxes, judged boxes per light and per rain
  condition, and `baseline complete: yes` only with at least `BASELINE_BOXES` (300)
  judged boxes, 3 sessions on at least 2 dates, and at least `MIN_CONDITION_BOXES` (50)
  judged boxes each in `day`, `twilight` and `rain` light or weather; otherwise `no` and
  what is missing. `dark` is reported but not required.

`--data-dir` is a checkout of the data branch. A session is `rain` when the published
sweep record nearest its `started_at`, within 30 minutes, has `weather.precip_mm` above 0,
`dry` when that is 0, and `unknown` otherwise (no record that close, or one without
weather). Without `--data-dir` every session is `unknown`. Only the records within 30
minutes of a session are read; a malformed one is an error that names it.

Without `--heights` the summary per week below is exactly as before, and the `boxes/`
directory is not read.

## Attribute session (`--attributes`)

The site reports the share of near-field people wearing an outer layer, with bare legs and
carrying an open umbrella. An attribute session collects the human labels those figures
are checked against and, optionally, the answers of a hosted vision model to the same
crops, so that the model's accuracy can be measured once against a fixed bar.

```bash
uv run python -m wearreport.tools.spotcheck --attributes --n 20 --min-persons 1 --view window
uv run python -m wearreport.tools.spotcheck --attributes --n 20 --judgements answers.json
```

The sweep and the sampling are those of a detection check, except that only person boxes at
least `NEAR_FIELD_MIN_HEIGHT_PX` (31) pixels tall in the source frame are kept (the height
the per-box file records, and the near-field threshold `--heights` chose on the baseline
data), and `--min-persons` counts those boxes only. Each kept box is shown as a crop,
numbered 1, 2, ... over the session.

It works in crops mode with `--view window`, or with `--judgements PATH` (then the crops
are files in the temporary review directory, as above). Any other combination is refused
before any network request: `--mode frames`, keyboard entry (`--view files` without
`--judgements`, which is also the default outside Windows) and `--record-boxes`.
`--view window` with `--judgements` stays refused, as for a detection check.

`--source austin` (attribute sessions only; anything else exits 2 with the usage) runs the
same session on the City of Austin's traffic cameras instead of London's: one pass over
the cameras in `--bbox S,W,N,E` (default downtown Austin), fetched in memory exactly as the
HD pilot (`wearreport.tools.pilot_heights`) fetches them. Only stills whose header is
exactly 1920x1080 are shown; the others (the cameras' 320x176 placeholders, for one) are
skipped and counted in the progress lines. Crops are cut from the full-resolution frame,
heights are its pixels, and the window fits large crops to the screen. The daylight check
and the file's `light` use Austin's sun, and the file is
`<out-dir>/attributes/YYYY-MM-DD-austin.json` (then `-austin-2`, ...) with the fields
below and `"source": "austin"`; the [attribute summary](#attribute-summary) reports it
apart from London's, after it, without rain lines. `--dry-run` serves a fake Austin on
127.0.0.1. `--source london`, the default, is unchanged.

### Box height (`--min-height`)

In daylight most crops under about 41 px cannot be judged for clothing, so the reviewer
would spend the session answering cannot tell. `--min-height N` shows only person boxes at
least N pixels tall in the source frame (N a whole number from 31, the near-field
threshold, to 200), and `--min-persons` then counts those boxes only. The attribute file
records N as `min_height_px`. Without the flag everything is as before: the threshold is
31. Any other value (30, 201, `4.5`, `abc`) is refused before any network request, and
`--min-height` without `--attributes` exits 1 with:

```
spotcheck: --min-height applies to attribute sessions only; add --attributes or drop it
```

The attribute threshold itself will be chosen later from the height bands of the
[attribute summary](#attribute-summary), which show where clothing becomes judgeable.

### Daylight (`--allow-dark`)

After dark most near-field crops cannot be judged, so a live session in the window does not
start when it is dark in London at its start (`light` `dark`: the sun below -6°, see
[the per-box file](#per-box-file)). The tool then exits 1 before any network request or
sweep, writes nothing, and prints one line:

```
spotcheck: it is dark in London now (sun below -6°); attribute sessions need daylight. Use --allow-dark to run anyway.
```

`--allow-dark` runs the session anyway, exactly as in daylight; the attribute file still
records `"light": "dark"`. Sessions that start in `day` or `twilight`, and sessions whose
answers come from a JSON file (`--judgements PATH`), are never refused: the check protects
a reviewer's time in the window only. `--allow-dark` without `--attributes` is refused, as
the detection check runs at any light.

### Questions and keys

For each crop the reviewer answers three questions, in this order, one at a time; the
window shows the crop and the current question above it:

1. `outer_layer`: "Outer layer (coat or jacket)?"
2. `bare_legs`: "Bare legs (shorts or short skirt)?"
3. `umbrella`: "Holding an open umbrella?"

| Key | Meaning |
|---|---|
| `y` | yes |
| `n` | no |
| `u` | cannot tell (this question only) |
| `x` | at any question: not a person, or nothing can be told; the crop's answers are discarded and it is counted as rejected |
| `Backspace` | go back one answer, also into the previous crop (an `x` is undone too) |
| `q`, or closing the window | stop without writing anything |

Capitals work as well. The window's one-line legend (`ATTRIBUTE_LEGEND`) lists these keys.
With `--confirm-stop` the first `q` only asks, as in a detection check: the header line
shows "Stop and discard this session? Press q again to stop, any other key to continue.",
a second `q` stops without writing anything, and any other key hides the question and is
not taken as an answer. The crop and the question on screen stay as they were.

With `--judgements PATH` write one entry per crop, keyed by crop number: either `"x"`, or
an object with exactly the three keys, each `"y"`, `"n"` or `"u"`:

```json
{
  "1": {"outer_layer": "y", "bare_legs": "n", "umbrella": "u"},
  "2": "x"
}
```

Anything else (a missing or extra key or crop, another value, `"X"`, a capital) is
rejected with a message naming the entry, before anything is written; fix the file and
save it again. `numbering.json` in the review directory holds a template with empty
answers, which are rejected until filled in.

### The model (`--judge`)

`--judge NAME --judge-max-requests N` work as for a detection check (both required
together, DeepInfra models only, a dry run needs a judge on this machine). Only once the
reviewer has finished are the crops the reviewer did not reject sent, one per request, as
PNG built in memory from the arrays shown: pixels only, never the reviewer's answers,
never a whole frame. Each request asks one fixed prompt, `ATTRIBUTE_PROMPT` in
`judge_hosted.py`, for exactly one line of the form `outer=yes legs=no umbrella=unsure`.
The reply parser is strict: any reply that is not exactly that line (surrounding white space
aside), within 64 characters, counts as `uuu` (cannot tell, three times). The key, the
endpoint pinning, the retries, the timeouts, the redacted errors and the request budget are
those above. A failure or the request cap ends the judging without losing the reviewer's
labels: the crops not answered get `null`. The worst-case cost is about N × $0.00087 for
`di-qwen3-vl-235b` (the largest crop, about 4,200 input tokens, and at most 32 output
tokens).

The model is tested once, with the fixed bar below. Do not tune the prompt against real
crops.

### Attribute file

`<out-dir>/attributes/YYYY-MM-DD.json`; if that exists, `-2`, `-3` and so on, never
overwritten. The session writes this one file and nothing else: no statistics file and no
per-box file. It is one line of JSON with exactly these fields:

| Field | Type | Value |
|---|---|---|
| `date` | string | the date of the session (local time), `YYYY-MM-DD` |
| `started_at` | string | when the sweep began, UTC, to the minute: `YYYY-MM-DDTHH:MMZ` |
| `light` | string | `day`, `twilight` or `dark`, as in the per-box file |
| `frames` | integer | frames sampled |
| `detector` | object | as in the statistics file |
| `min_height_px` | integer | the smallest box height shown: `--min-height N`, else `NEAR_FIELD_MIN_HEIGHT_PX` (31) |
| `judge` | string or null | the `--judge` model, or `null` |
| `crops_shown` | integer | crops shown to the reviewer |
| `crops_rejected` | integer | crops rejected with `x` |
| `crops` | array | one `[height_px, reviewer, model]` per crop not rejected, sorted |

`reviewer` is three letters, `y`, `n` or `u`, in the question order (for example `"ynn"`:
an outer layer, no bare legs, no umbrella); `model` is the same form, or `null` when the
model was not asked or did not answer. The file holds counts and labels only: no box
position or width, no camera id, no frame index, no image and no free text, and the sorted
order says nothing about which frame a crop came from.

### Attribute summary

```bash
uv run python -m wearreport.tools.spotcheck_summary --attributes [--dir spotchecks] [--data-dir PATH]
```

reads every attribute file in `<dir>/attributes/` (a malformed file is an error that
names it, exit 1) and prints, for each attribute:

- the reviewer's answers: yes, no, cannot tell, and the yes share over yes and no with its
  Wilson 95% interval; the same per `light` and, with `--data-dir`, per rain condition
  (the same nearest-record join as `--heights`);
- over the crops the reviewer answered yes or no and the model answered (`paired`): the
  model's precision on yes, its recall on yes and its specificity (recall on no), each with
  n and its Wilson 95% interval, and how many times the model said cannot tell. A model
  cannot tell (`u`) counts as wrong in recall and specificity, so a model that always
  answers cannot tell cannot pass;
- a verdict. `pass` needs, among the paired crops, at least `MIN_POSITIVES` (30) reviewer
  yes, at least `MIN_NEGATIVES` (30) reviewer no and at least `MIN_LABELLED` (200) of
  both, and precision, recall and specificity each at least `TARGET_ATTRIBUTE_ACCURACY`
  (0.85). `insufficient (...)` says which count is short; otherwise `fail (...)` names the
  measures below the bar. With no model answer the verdict is `no model`;
- after those lines, the judgeable share by height: one line per height band (31-35,
  36-40, 41-45, 46-50, 51-60 and 61+ px), not indented and starting with the attribute's
  name, with the crops in it and how many the reviewer answered yes or no (not cannot
  tell), and their share, over all files but day and twilight only: dark files are left
  out, as the line says. For example:

  ```
  outer_layer height 41-45 px, day and twilight (dark excluded): 12 crop(s), 7 answered yes or no, share 0.5833
  ```

  The attribute threshold (`--min-height`) will be chosen from these height bands.

Without `--attributes` the summary's output is exactly as before, and `attributes/` is not
read.

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
nothing is written except the statistics file (and, with `--record-boxes`, its per-box
file), or, in an attribute session, the attribute file. With `--judge`, the crops also go, in memory, to the DeepInfra API (and nowhere
else), after the review.

With `--view files`, rendered images exist only in the temporary directory, and the tool
deletes it when it exits: after a normal run, an error, the review timeout, or any
catchable signal whose default action ends the process: Ctrl-C (SIGINT), SIGTERM, SIGHUP,
SIGQUIT, SIGUSR1, SIGUSR2, SIGXCPU (`ulimit -t`), SIGVTALRM, SIGPROF, SIGPOLL, SIGPWR,
SIGSTKFLT and the real-time signals, where the platform has them; on Windows, only
Ctrl-C and Ctrl-Break (SIGBREAK) in the tool's console. Repeated signals (Ctrl-C twice,
or a closing terminal's SIGHUP then SIGTERM) cannot interrupt the deletion. Nothing else
is written except the statistics file (and, with `--record-boxes`, its per-box file).

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
