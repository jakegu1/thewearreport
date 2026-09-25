# AGENTS.md — Engine executor handbook

Read this file completely before changing anything in this repository. It applies to
every coding agent (Claude Code, Cursor, or any other) and to humans.

## 1. What this repository is

The engine of The Wear Report: it fetches London TfL JamCam frames during
daytime, detects people and umbrellas on CPU, classifies a few coarse clothing
attributes, and publishes **aggregate counts only** to the `data` branch. See
[README.md](./README.md) and [METHODOLOGY.md](./METHODOLOGY.md).

**This repository is (or will become) public.** Planning, task specs, business
decisions and the website live in the maintainer's private repository. Two rules follow:

- Never copy private material into this repository, its commits, issues or PR
  descriptions. That includes task specs verbatim, budgets, revenue, keyword or SEO
  plans, analytics, and anything from files named `OWNER.md`, `QUALITY.md`, `tasks/`,
  `seo/` or `research/`. Refer to a task by its ID only (e.g. `T-003`). Restating an
  acceptance criterion in a PR's evidence table is fine.
- `tools/public_guard.py` enforces this in CI. Never weaken it.

## 2. How work arrives

Tasks come from the maintainer as specs with an ID (`T-NNN`), a goal, files in scope and
numbered acceptance criteria. Usually a `/next-task` session opened in the private
workspace provides them, with this repository mounted next to it. One task = one branch
= one PR.

1. Read the spec, then this file, then any referenced code in `spike/`.
2. If the spec is ambiguous, stop and ask (§7). Do not guess.
3. Branch `task/t-NNN-<slug>`.
4. Commit acceptance tests first, in their own commit
   (`engine/tests/acceptance/test_t_NNN.py`). They are the contract. Never modify them
   during implementation; if one is wrong, stop and ask.
5. Implement inside the spec's files in scope.
6. Run `make check`. It must pass.
7. Open a PR titled `T-NNN: <title>` that fills
   [the PR template](./.github/pull_request_template.md) with evidence for every
   acceptance criterion.
8. The maintainer merges. Agents never merge, push to `main`, or rewrite shared history.

## 3. Invariants (any violation blocks the PR)

**INV-1 Privacy.** Camera frames and person crops exist **only in memory**. Never write
image bytes derived from camera frames to disk, logs, git, caches, CI artifacts or
third-party services. Three exceptions only: (a) passing arrays to the local detector in
memory; (b) sending person crops (never full frames) to a vision-model API when a task
spec explicitly allows it, within its budget; (c) the spot-check tool rendering annotated
frames into a temporary directory it creates and deletes on exit, never in CI. Never
implement face detection, face recognition, re-identification, or tracking of individuals
across frames. Publish aggregates only.

**INV-2 Licenses.** Runtime dependencies must be permissively licensed (MIT, BSD,
Apache-2.0, ISC, PSF, MPL-2.0, 0BSD, Zlib, CC0). The exact list is `ALLOWED_SPDX` in
`scripts/license_check.py`. Only an OSI-approved permissive licence may be added to it, and
only with a justification in the PR. No AGPL, GPL, SSPL, BUSL or non-commercial
licenses. In particular **`ultralytics` must never be a dependency** (AGPL-3.0). Data
sources are limited to TfL, the Met Office and the US National Weather Service.
Open-Meteo may be used only in local development and tests behind an explicit flag,
never in production (its free tier is non-commercial).

**INV-3 Secrets.** Never commit secrets or tokens. Read them from environment variables
and document them in `.env.example` with empty values. In workflows, use repository
secrets, and never expose them to workflows triggered by forks.

**INV-4 Test fixtures.** Never use real camera captures as fixtures. Use synthetic or
permissively licensed images, and record the source and license of each in
`fixtures/LICENSES.md`.

**INV-5 Attribution.** Published data carries "Powered by TfL Open Data" and
"Powered by Met Office data" (and NWS credit where used). Both are the exact statements
the providers' terms ask for.

**INV-6 Honesty.** Never fabricate, impute or pad observations. Every published figure
must be traceable to the sweep records it came from.

**INV-7 Scope.** One task per PR. Changes outside the files in scope must be justified
in the PR description.

**INV-8 Cost and resilience.** No paid API usage beyond the task's stated budget. Every
external call has a timeout and a bounded retry policy.

**INV-9 Public repository.** Nothing private in this repository (§1). Workflows run with
least-privilege `permissions:` (only the data-publishing job gets `contents: write`),
pin every third-party action to a full-length commit SHA, and run on `ubuntu-24.04`.
Never use `pull_request_target` or run untrusted code with secrets. Never store AI-service
tokens (e.g. `CLAUDE_CODE_OAUTH_TOKEN`) in this repository's secrets. AI-agent workflows,
if ever added, may only be triggered by the maintainer.

## 4. Command contract

Created by the bootstrap task; later tasks must keep these working.

| Command | Does |
|---|---|
| `make setup` | Install toolchains and dependencies |
| `make check` | Everything CI runs: lint, format check, typecheck, tests, license check, privacy guard, public guard |
| `make test` | Tests only |
| `make sweep-dry` | One sweep against a local fake camera server (no network) |

## 5. Decided technology (do not relitigate inside a task)

| Area | Choice |
|---|---|
| Detector | YOLOX-s ONNX via onnxruntime (CPU) |
| Language and tools | Python 3.12, `uv`, `ruff`, `mypy --strict`, `pytest` |
| Scheduling | GitHub Actions cron during London daytime |
| Aggregate storage | One immutable JSON record per sweep on the orphan `data` branch (records concatenate into JSON Lines), validated by JSON Schema |
| Weather | Met Office DataHub (UK), NWS (US) |

If you believe a decided choice is wrong, say so with evidence in the PR. Do not change
it unilaterally.

## 6. Conventions

- Small, typed functions. Prefer the standard library. No speculative abstractions.
- Structured JSON logs. Never log image data or URLs containing tokens.
- Tests assert behaviour, including failure paths (timeouts, malformed input, partial
  failures). Do not test mocks.
- Treat external data as hostile. A parser of a network response or data file turns every
  failure into the module's typed error. That includes `ValueError`, `TypeError`, `KeyError`,
  `UnicodeDecodeError`, `RecursionError` (deep nesting) and `OverflowError` (huge numbers,
  out-of-range dates). Raise it `from None` when the original message could echo a secret.
  Cap response sizes. Tests include these pathological inputs.
- Never skip, disable or weaken a test or check to make CI pass.
- Code, comments, commits and PRs in English.

## 7. When blocked or unsure

Stop and report in exactly this format (in the session, or in the PR if one is open):

```
BLOCKED: <one sentence>
Question: <one specific question>
Options: A) ... B) ...
My default if no answer in 24h: <A or B, and why>
```

Do not work around a blocker by expanding scope or weakening a check.
