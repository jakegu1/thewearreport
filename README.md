# The Wear Report — engine

> Pre-release: the pipeline is being built and nothing is published yet. Site: thewearreport.com (not live yet).

An open pipeline that measures **what people actually wear outside**. It uses public
traffic cameras and publishes **aggregate statistics only**, for example "at 12 °C,
71% of people seen wore a coat (n = 4,210)". It never stores images, never identifies
anyone and never tracks individuals.

## How it works

1. **Fetch.** During daytime, fetch the latest still frame from each of London's
   ~800 TfL JamCam traffic cameras. Frames are held in memory only.
2. **Detect.** Find people and umbrellas on CPU with
   [YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) via ONNX Runtime.
3. **Classify.** Classify a few coarse attributes per detected person: outer layer
   (coat or jacket) yes/no, bare legs yes/no. Nothing finer than that.
4. **Aggregate.** Combine the counts with the current weather and publish per-sweep and
   daily aggregates. Images and crops are discarded as soon as they are processed.

The full method, including its known biases and limits, is in
[METHODOLOGY.md](./METHODOLOGY.md).

## Privacy

- Frames and person crops exist only in memory and are never written to disk, logs,
  caches, artifacts or git.
- No face detection, face recognition, re-identification or tracking across frames.
- Only aggregate counts are published.

[METHODOLOGY.md § Privacy](./METHODOLOGY.md#privacy) describes the rules in full. CI
enforces them: a static check blocks image-writing calls, and an end-to-end test
asserts that a sweep creates no files except its aggregate output.

## Data

Aggregates will be published on the `data` branch as one JSON record per sweep, with a
JSON Schema. The records concatenate into JSON Lines.
The data license is in [DATA-LICENSE.md](./DATA-LICENSE.md).

Powered by TfL Open Data. Powered by Met Office data. US weather from the National
Weather Service.

## Repository layout

```
engine/      Python package: registry, fetch, detect, attributes, aggregate, publish
spike/       Verified reference implementation (read, don't import)
tools/       Repository checks (public guard)
.github/     CI and the scheduled sweep
```

## Developer setup

Requires Linux x86_64 (CI runs `ubuntu-24.04`), `make`, `curl` and `git`.

```bash
make setup   # installs uv 0.12.18 if missing, Python 3.12, dependencies, gitleaks and actionlint
make check   # everything CI runs: lint, format, types, tests, licences, privacy, secrets, workflow lint, public guard
make test    # tests only
make help    # list all targets
```

`make setup` installs the pinned uv release into `~/.local/bin` when it is not already
on `PATH`, and gitleaks and actionlint (both MIT, development tools only) into
`.tools/bin/`. Every download is verified against a pinned SHA-256 checksum.

### Windows (spot-check only)

The spot-check tool ([spotchecks/README.md](./spotchecks/README.md)) also runs on native
Windows 10 or 11, without WSL. Nothing else is supported there, and the `make` targets
are for Linux. You need `git` and [uv](https://docs.astral.sh/uv/) 0.12.18; Windows ships
`curl.exe` and PowerShell.

Everything large can go to a drive other than `C:`: the clone (with its `.venv` and
`.models`), uv's cache and the Python that uv installs. Point uv at that drive before the
first `uv` command, for example `D:\uv`:

```powershell
setx UV_CACHE_DIR D:\uv\cache              # for new terminals
setx UV_PYTHON_INSTALL_DIR D:\uv\python
$env:UV_CACHE_DIR = 'D:\uv\cache'          # and for this one
$env:UV_PYTHON_INSTALL_DIR = 'D:\uv\python'
git clone <this repository's URL> D:\thewearreport
cd D:\thewearreport
uv sync --locked --no-install-package llama-cpp-python
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_model.ps1
uv run --no-sync python -m wearreport.tools.spotcheck --n 20 --reviewer NAME
```

- `--no-install-package llama-cpp-python` leaves out the judge's runtime, which has no
  Windows wheel; the spot-check never imports it. Run commands with `uv run --no-sync`:
  a plain `uv run` would install it again, which means compiling llama.cpp.
- `scripts\fetch_model.ps1` downloads YOLOX-s and YOLOX-m into `.models\` and checks
  each against the same pinned SHA-256 as `scripts/fetch_model.sh`. It fails closed: a
  file that does not match is deleted, and nothing unverified is left behind.
- On Windows the spot-check shows each crop in a window (`--view window`, the default
  there) and writes no image at all. `--mode frames` uses the temporary directory
  instead (`%TEMP%`, on `C:` by default; set `TEMP` to a folder elsewhere to move it,
  outside any git clone).

Configuration comes from environment variables, listed in [.env.example](./.env.example).
Copy it to `.env` (never committed) and fill in the values you need.

| Check | Command | Enforces |
|---|---|---|
| Static privacy guard | `make privacy` | engine code never writes images or binary files (INV-1) |
| Licence check | `make licenses` | runtime dependencies are permissively licensed (INV-2) |
| Secret scan | `make secrets` | no credentials in git history (INV-3) |
| Public guard | `make public-guard` | no private material in the repository (INV-9) |
| Schema validation | `make schemas` | `data/schema/` is valid JSON Schema and samples validate |

## Judge bake-off

The spot-check judge is chosen by a bake-off on a licensed gold set
(`engine/wearreport/tools/judge.py`, `engine/wearreport/tools/goldset.py`). No judge is
chosen yet: no candidate has passed the quality bar.

```bash
sh scripts/fetch_goldset.sh                      # the gold set's licensed source photos
sh scripts/fetch_judge_model.sh --all            # local candidates' weights (GGUF)
uv run python -m wearreport.tools.judge --bakeoff --subset screen   # local models
AWS_BEARER_TOKEN_BEDROCK=... uv run python -m wearreport.tools.judge --bakeoff \
  --backend bedrock --max-requests 1200          # models hosted by Amazon Bedrock
uv run python -m wearreport.tools.judge_deepinfra --bakeoff \
  --max-requests 1200                            # models hosted by DeepInfra
```

A model passes only if the whole gold set and the held-out items (those outside the
100-crop screening subset) both pass the bar. A hosted run needs `--max-requests`, sends
gold-set crops only and prints requests, tokens and the measured cost. It is never part
of `make check` or CI. A Bedrock run without `AWS_BEARER_TOKEN_BEDROCK` sends no
`Authorization` header, and a DeepInfra run never sends one: both rely on a credential
that the environment adds to requests for the provider's host (for example an
authenticating proxy). No DeepInfra key is read by this code.

A diagnostic experiment asks whether more context around the box, or a larger render,
helps a hosted judge. It renders the gold set with each crop variant (`m0.5`, today's crop;
`m1.0` and `m2.0`, a wider margin; `m1.0-r480`, a larger enlargement), with the degradation
unchanged, and reports each variant and model against `m0.5`. It chooses no model.

```bash
uv run python -m wearreport.tools.judge_context --variants m0.5,m1.0,m2.0,m1.0-r480 \
  --models di-qwen3-vl-235b,di-gemma-4-31b --max-requests 800   # screen subset
```

With `--subset all`, the held-out items are split by source photo: no held-out item shares
a photo with a screening item. On the screen, no variant brought either model within
5 points of the 95% bar with 15% or fewer unsure (T-035). The best result was
`di-gemma-4-31b` with `m2.0`, at 84.2% with 5% unsure.

## Operations

[`.github/workflows/sweep.yml`](./.github/workflows/sweep.yml) runs one sweep every 20
minutes in London daytime. The schedule is a plain UTC cron, `7-59/20 6-20 * * *`: minutes
7, 27 and 47 of every hour from 06 to 20 UTC. That covers 07:00–21:00 London in both GMT
and BST, and the gate skips the firings outside it, so about 42 sweeps run a day (from
07:07 to 20:47 London time). The minutes are off the top of the hour because GitHub
delays or drops scheduled runs under load, most of all at the start of every hour.
Scheduled and manual runs start from the default branch only.

- **Gate.** The first job checks the London time and skips the sweep outside
  07:00–21:00. It alone decides whether a run is in London daytime: in GMT it skips the
  06 UTC firings, in BST the 20 UTC ones, and it also skips a scheduled run that starts
  late. It also refuses a manual run started from any branch other than the default
  branch: every branch carries the workflow, and a run there would sweep and publish
  with that branch's unreviewed code.
- **One run at a time.** All runs share one concurrency group. A queued run waits, and a
  run that is publishing is never cancelled.
- **Model.** `.models/` is cached between runs. `scripts/fetch_model.sh` checks the
  SHA-256 of both models on every run, and the sweep checks it again when it loads YOLOX-m.
- **Timeout.** The sweep job stops after 15 minutes.
- **Publishing.** The sweep job checks out the `data` branch shallow and sparse:
  `status.json` and the last three UTC days of records. If none of those records is a
  success and the previous `status.json` does not report one, it adds older days until
  the newest success is in the tree, so `consecutive_failures` counts every failure since.
  If the branch does not exist, the job creates it as an orphan. The job stages only new `sweeps/**/*.json` records and
  `status.json`, checks the staged list, then commits and pushes. It is the only job with
  `contents: write`.
- **Alert.** A failed sweep is a sweep job that fails (for example, the registry is
  unreachable, so no record is written) or a published record that is not a success
  (`consecutive_failures` in `status.json`). After three failed sweeps in a row, the alert
  job opens an issue labelled `ops-alert` with the failure summary. While it is open, a
  further failure comments on it only when the failed stage changes, or when there has
  been no such note for an hour. The next successful sweep closes it. Runs skipped by the
  gate are ignored. The alert job is the only job with `issues: write`.
- **Manual run.** Actions → sweep → Run workflow. Tick `force_fail` to fail the sweep job
  before it does anything, which tests the alert without publishing. Manual runs obey the
  gate too.

Repository secrets: `METOFFICE_API_KEY` (required), `TFL_APP_KEY` (optional). The
workflow never runs on pull requests, so fork code never sees them.

## License

Code: [Apache-2.0](./LICENSE). Data: see [DATA-LICENSE.md](./DATA-LICENSE.md).

## Contributing

This project is maintained by one person and is not accepting pull requests yet.
To report a security or privacy problem, use GitHub's private vulnerability reporting
(see [SECURITY.md](./SECURITY.md)).
