#!/usr/bin/env sh
# Download the gold set's openly licensed source photos (fixtures/goldset/manifest.json)
# and verify each against its pinned SHA-256. The photos are never committed.
#
# Usage: sh scripts/fetch_goldset.sh [--manifest PATH] [--dest DIR]   (DIR defaults to .goldset/)
#
# Fails closed: a file is moved into DIR, under its source id, only after its checksum
# matches; a file already in DIR that does not match is deleted and downloaded again; part
# files left by an earlier, killed run are deleted first. On any failure the script exits
# non-zero and leaves no unverified file behind. Sends no credentials.
set -eu

ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
MANIFEST="${ROOT}/fixtures/goldset/manifest.json"
DEST="${ROOT}/.goldset"

while [ $# -gt 0 ]; do
  case "$1" in
    --manifest | --dest)
      [ $# -ge 2 ] || { echo "fetch_goldset: $1 needs a value" >&2; exit 2; }
      if [ "$1" = "--manifest" ]; then MANIFEST="$2"; else DEST="$2"; fi
      shift
      ;;
    *) echo "usage: fetch_goldset.sh [--manifest PATH] [--dest DIR]" >&2; exit 2 ;;
  esac
  shift
done

PART=""
LIST=""
trap '[ -z "${PART}" ] || rm -f "${PART}"; [ -z "${LIST}" ] || rm -f "${LIST}"' EXIT
trap 'exit 1' HUP INT TERM

# One line per source: id, SHA-256, URL. Ids and digests are checked here; each URL must
# be https and free of spaces and control characters.
LIST="$(mktemp)"
python3 - "${MANIFEST}" > "${LIST}" <<'PY'
import json, re, sys
with open(sys.argv[1], "rb") as fh:
    data = json.loads(fh.read(4 * 1024 * 1024).decode("utf-8"))
for s in data["sources"]:
    ok = (
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", s["id"])
        and re.fullmatch(r"[0-9a-f]{64}", s["sha256"])
        and re.fullmatch(r"https://[^\s]+", s["url"])
        and s["url"].isprintable()
    )
    if not ok:
        sys.exit(f"fetch_goldset: malformed source entry {str(s.get('id'))[:40]!r}")
    print(s["id"], s["sha256"], s["url"])
PY

mkdir -p "${DEST}"
rm -f "${DEST}"/.*.part.*

matches() { # matches FILE SHA256
  [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d ' ' -f 1)" = "$2" ]
}

count=0
while read -r id sha url; do
  target="${DEST}/${id}"
  count=$((count + 1))
  if matches "${target}" "${sha}"; then
    continue
  fi
  if [ -e "${target}" ] || [ -L "${target}" ]; then
    echo "fetch_goldset: ${id} does not match its pinned SHA-256; removing it" >&2
    rm -f "${target}"
  fi
  PART="$(mktemp "${DEST}/.${id}.part.XXXXXX")"
  if ! curl --proto '=https' --tlsv1.2 --max-time 120 --retry 3 \
    --proto-redir '=https' -fsSL -o "${PART}" "${url}"; then
    echo "fetch_goldset: download of ${id} failed" >&2
    exit 1
  fi
  if ! matches "${PART}" "${sha}"; then
    echo "fetch_goldset: ${id} failed SHA-256 verification; nothing installed" >&2
    exit 1
  fi
  chmod 0644 "${PART}"
  mv -f "${PART}" "${target}"
  PART=""
done < "${LIST}"
echo "fetch_goldset: ${count} source photos present and verified in ${DEST}"
