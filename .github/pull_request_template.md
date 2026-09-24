<!-- Title must be: T-NNN: <task title>. This repository is public: refer to the task
by ID only and do not paste private planning material (AGENTS.md §1). -->

## Task

T-NNN

## What changed

<!-- 3–6 bullets -->

## Acceptance criteria — evidence

<!-- One row per acceptance criterion. Evidence = command output, test name, CI link or
measured number. A row without evidence counts as not done. -->

| AC | Criterion (short) | Status | Evidence |
|---|---|---|---|
| AC1 | | ✅ / ❌ | |

## Self-review

| Item | Self-score 0–2 | Note |
|---|---|---|
| Spec conformance | | |
| Scope discipline | | |
| Correctness | | |
| Test quality | | |
| Invariants | | |
| Maintainability | | |

## Invariants

- [ ] INV-1 Privacy: no camera-derived image bytes written, persisted or sent (except the allowed exceptions)
- [ ] INV-2 Licenses: every new runtime dependency is permissive (table below); no `ultralytics`
- [ ] INV-3 No secrets committed; workflows never expose secrets to forks
- [ ] INV-4 Fixtures are synthetic or permissively licensed and recorded in `fixtures/LICENSES.md`
- [ ] INV-5 Attribution carried in published data
- [ ] INV-6 No fabricated, imputed or padded observations
- [ ] INV-7 Diff limited to the files in scope (exceptions justified below)
- [ ] INV-8 External calls have timeouts and bounded retries; spend within budget
- [ ] INV-9 Nothing private in this PR; `tools/public_guard.py` passes

## New dependencies

| Package | Version | License | Why |
|---|---|---|---|

## Out-of-scope changes and why

## Notes for the reviewer
