# Local planning and a human London pass

`label-local.cmd` is an optional Windows entry. No arguments shows offline help.
It never captures by default. Planning and preflight use local files only: they do
not update Git, install dependencies, download models, contact providers, open
images, or create reminders or operating-system tasks.

## One-time preparation

In an existing local Windows checkout, a human can update and prepare the locked
environment explicitly:

```powershell
git pull --ff-only
uv sync --locked --no-install-package llama-cpp-python
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/fetch_model.ps1
```

The model command downloads the pinned local detectors. Run preparation yourself
when needed; neither offline mode performs it. This guide does not install or
configure a remote Windows machine. Administrator access and persistent execution
policy changes are unnecessary. The batch file uses Windows PowerShell 5.1 with a
process-only execution policy and resolves its repository from its own location.

## Offline plan and preflight

Choose your own IANA display timezone, online window, and available weekdays.
These generic examples describe hypothetical availability:

```powershell
label-local.cmd -Zone Europe/London -StartDate 2026-10-25 -Days 7 -OnlineStart 09:00 -OnlineEnd 17:00 -Weekdays mon,tue,wed,thu,fri
label-local.cmd -Preflight -Zone Europe/London
```

The default online window is 00:00 through the exclusive endpoint 24:00, on every
weekday. `-Days` accepts 1 through 14. Omitting `-StartDate` uses the real current
date in `-Zone`. A start must precede its exclusive end; split an overnight window
into separate plans. A timezone is required; no machine-location assumption is
made. `-Python C:\path\to\python.exe` explicitly selects an already prepared Python;
otherwise the entry uses this repository's `.venv\Scripts\python.exe`. Offline
commands require Python 3.12 and reject other versions with preparation guidance. It never
silently searches for or installs another interpreter.

The Python module also works directly in an existing local environment:

```text
python -m wearreport.tools.local_label_plan --zone Europe/London --start-date 2026-10-25 --days 7 --online-start 09:00 --online-end 17:00 --weekdays mon,tue,wed,thu,fri
```

JSON describes **prospective opportunities**, never captures or observations.
London source-local 10:00, 12:00, and 14:00 are converted with IANA timezone rules,
including daylight-saving transitions, then filtered by the display calendar date,
weekday, and online window. Each row includes source-local, display-local, and UTC
timestamps; display date and display time are separate fields. The source dates
can differ from the display dates.

Both display and source zones load explicitly from the installed `tzdata` package
pinned by `uv.lock`, rather than the host operating system's database. JSON reports
the package version and source, and preflight verifies the installed version
against the lock. The current lock uses `tzdata` 2026.5 / IANA 2026e. Its
[official release notes](https://github.com/eggert/tz/blob/2026e/NEWS) record Alberta's
permanent UTC-06 rule from June 18, 2026: Calgary does not fall back on November 1.
The March 7/8, 2026 transition still changes Calgary noon from 19:00 to 18:00 UTC.
No fixed offsets or machine-local timezone aliases are substituted.

Calgary source-local 12:00 coverage is reported separately, including uncovered
slots and their reasons. This is an availability comparison. It does not establish
or recommend any additional morning collection protocol.

`-Preflight` checks required entries, lock and environment files, the local detector
file's presence, Python 3.12, the planner module, and installed runtime version
metadata against `uv.lock` (the unused AI judge package is excluded). It prints
paths and nonsecret versions. It does not validate detector bytes, daylight,
credentials, provider access, or a future capture. Missing files or mismatched
versions fail with preparation guidance; nothing is repaired automatically.

## Explicit human start

A human who is present and ready to review can invoke:

```powershell
label-local.cmd -StartLondon
label-local.cmd -StartLondon -OutDir "C:\labels with spaces"
```

This calls the existing absolute `label.cmd` with exactly
`-Source london -MaxPasses 1 -NoJudge`, plus `-OutDir` only when explicitly supplied.
It is a single pass, with no AI judge. The existing launcher owns its Git update,
environment preparation, daylight checks, London time gates, review window,
clipboard behavior, and output location. The new entry preserves its output and
actual exit code. It accepts no test clock or bypass options in this mode.

For `-StartLondon`, the project path, derived launcher path, and any explicit output
path must contain no shell metacharacters (`& | < > ^ % ! ( )`), double quotes, or
control characters. The entry refuses unsupported paths before starting a child
and asks you to use project/output directories without those characters. Spaces
and Unicode directory names are supported. This restriction keeps the unchanged
legacy batch launcher's extra command-parsing boundary safe.

Keep a human present throughout review. This entry is not an unattended controller,
uploader, image queue, or scheduler. A prospective calendar row does not grant
permission to collect, establish shared daily counts, or prove that collection ran.
Output files retain the existing numeric-only attribute format. T-093 planned
batching can use its own module after that change is merged; this entry has no
dependency on that module and does not perform batching or upload.

The read-only native Windows workflow always runs the T-094 planner and copied-entry
tests. It also runs T-093 intake acceptance and unit tests when both test files are
present in the checkout. Otherwise a separate step explicitly reports that intake
tests were not executed. This conditional step is not evidence of native intake
coverage until an actual run executes those tests successfully.

## Removal

Delete only the added entry files (`label-local.cmd`, `scripts/label-local.ps1`, and
`engine/wearreport/tools/local_label_plan.py`) and this guide if no longer wanted.
The optional `local-label.yml` workflow and T-094 tests can also be removed when
retiring this feature. Preserve all label outputs, the existing launchers, local
environment, and detector files. No operating-system tasks were registered.
