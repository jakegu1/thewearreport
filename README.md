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

Configuration comes from environment variables, listed in [.env.example](./.env.example).
Copy it to `.env` (never committed) and fill in the values you need.

| Check | Command | Enforces |
|---|---|---|
| Static privacy guard | `make privacy` | engine code never writes images or binary files (INV-1) |
| Licence check | `make licenses` | runtime dependencies are permissively licensed (INV-2) |
| Secret scan | `make secrets` | no credentials in git history (INV-3) |
| Public guard | `make public-guard` | no private material in the repository (INV-9) |
| Schema validation | `make schemas` | `data/schema/` is valid JSON Schema and samples validate |

## Operations

[`.github/workflows/sweep.yml`](./.github/workflows/sweep.yml) runs one sweep every 20
minutes from 07:00 to 20:40 London time (cron `*/20 7-20 * * *` with
`timezone: "Europe/London"`, 42 runs a day). Scheduled and manual runs start from the
default branch only.

- **Gate.** The first job checks the London time again and skips the sweep outside
  07:00–21:00, for example when a scheduled run starts late. It also refuses a manual run
  started from any branch other than the default branch: every branch carries the
  workflow, and a run there would sweep and publish with that branch's unreviewed code.
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
