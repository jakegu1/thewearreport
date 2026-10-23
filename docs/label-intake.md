# Offline label intake

T-093 adds a local command for completed numeric attribute records. It does not
capture images, retain images, load a detector or judge, or contact any service.

```sh
uv run --locked python -m wearreport.tools.label_intake \
  --input /path/to/incoming-json --output /path/to/new-local-batch
```

Choose an output outside every Git repository and worktree. The input and output
must be separate, with neither inside the other. Output symlinks, including
symlinked parent directories, are refused. A `.git` directory or file anywhere
above the destination prevents intake. Keep this local output out of version
control.

The command reads direct regular `.json` files in basename order. It ignores
subdirectories and other extensions, refuses symlink entries and nonregular JSON
files, and requires at least one record. Limits are 256 files, 1 MiB per file, and
32 MiB of input bytes in total, including duplicates. Reads remain bounded if a
file grows. It checks regular-file type and inode before reading an opened file.
This is not a simultaneous snapshot of files being edited; stop other writers
before intake. It does not defend against a malicious local process replacing a
checked regular file with a FIFO during open.

Each record must pass the existing attribute parser without schema changes.
Intake additionally requires exactly `model`, `sha256`, and `conf` in `detector`:
the established model identifier syntax, a 64-character lowercase hexadecimal
digest, and a finite numeric confidence from 0 through 1. A missing `source`
means London; explicit sources remain `austin` and `calgary`. Human-only records
with `judge: null` and null model answers remain valid. Date-only, numbered,
city-suffixed, and historical timestamp basenames are preserved. Basenames with
control characters, nonportable reserved characters, or reserved device names
are refused.

A session's identity is its normalized source and `started_at`. Identical parsed
observation objects count once even when JSON whitespace, key order, or numeric
formatting differs. The first basename in input order supplies the original bytes
and filename. Duplicate files are counted separately in the report. Different
observations with one identity, or filenames that collide case-insensitively
between distinct sessions, refuse the whole batch. Detector metadata and existing
camera aggregates participate in that comparison.

Successful output contains:

- `attributes/<original basename>`: the first original's exact UTF-8 bytes,
  including its whitespace and line endings, without added fields.
- `sessions.jsonl`: one complete valid observation object per line, with no added
  metadata. An omitted London source stays omitted.
- `manifest.json`: version, session source/time identities, original basenames,
  SHA-256 digests of the originals, and the aggregate report.

The report is also printed as one JSON object. It keeps sources separate and
reports session counts, crops shown/rejected/labelled, and human `y`/`n`/`u`
counts for each attribute. For each camera already present in a record's optional
`camera_yield`, it sums the existing shown, rejected, and outer-layer judgeable
counts and counts contributing sessions. It also reports sessions missing that
optional field. Such sessions contribute human answers but no inferred camera
counts. Camera IDs are keys in aggregate dictionaries; there is no per-crop
camera association. These counts do not estimate accuracy, infer weather, rank
cameras, or recommend camera selection.

All records are validated before output. A fresh sibling staging directory holds
text files whose writes are flushed and fsynced. The command reopens and verifies
the staged files and validates the JSON Lines observations before an atomic
directory rename that refuses an existing destination. New commits use Linux's
exclusive `renameat2` or Windows' exclusive directory rename; other platforms
refuse safely. A failure removes incomplete staging and preserves existing
output, input files, and unrelated files. An existing byte-identical output is an
idempotent read-only success; any other output is refused. A new batch with
different duplicates or formatting needs a new output path. Successful output is
immutable to this command; filesystem permissions do not prevent an owner from
editing it outside the command.

Success exits 0, invalid input or a local filesystem failure exits 1, and an
invalid command line exits 2. Error messages use fixed diagnostic text and do not
echo record values or command-line payloads.
