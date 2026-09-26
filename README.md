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
make setup   # installs uv 0.12.18 if missing, Python 3.12, dependencies and gitleaks
make check   # everything CI runs: lint, format, types, tests, licences, privacy, secrets, public guard
make test    # tests only
make help    # list all targets
```

`make setup` installs the pinned uv release into `~/.local/bin` when it is not already
on `PATH`, and gitleaks into `.tools/bin/`. Both downloads are verified against pinned
SHA-256 checksums.

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

## License

Code: [Apache-2.0](./LICENSE). Data: see [DATA-LICENSE.md](./DATA-LICENSE.md).

## Contributing

This project is maintained by one person and is not accepting pull requests yet.
To report a security or privacy problem, use GitHub's private vulnerability reporting
(see [SECURITY.md](./SECURITY.md)).
