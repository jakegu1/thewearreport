#!/usr/bin/env sh
# Download the YOLOX ONNX models (Apache-2.0) from the official YOLOX release and verify
# each against a pinned SHA-256. YOLOX-s always; YOLOX-m too with --with-m.
#
# Usage: sh scripts/fetch_model.sh [--with-m] [--dest DIR]   (DIR defaults to .models/)
#
# Fails closed: a file is moved into DIR only after its checksum matches, and a file
# already in DIR that does not match is deleted and downloaded again. On any failure the
# script exits non-zero and leaves no unverified file behind. A run killed outright (SIGKILL)
# cannot clean up its part file (DIR/.<name>.XXXXXX), so each fetch starts by deleting
# the part files an earlier run left for that model.
set -eu

BASE_URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
SHA256_YOLOX_S="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
SHA256_YOLOX_M="21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f"

ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
DEST="${ROOT}/.models"
WITH_M=0

while [ $# -gt 0 ]; do
  case "$1" in
    --with-m) WITH_M=1 ;;
    --dest)
      [ $# -ge 2 ] || { echo "fetch_model: --dest needs a directory" >&2; exit 2; }
      DEST="$2"
      shift
      ;;
    *) echo "usage: fetch_model.sh [--with-m] [--dest DIR]" >&2; exit 2 ;;
  esac
  shift
done

PART=""
trap '[ -z "${PART}" ] || rm -f "${PART}"' EXIT
trap 'exit 1' HUP INT TERM

matches() { # matches FILE SHA256
  [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d ' ' -f 1)" = "$2" ]
}

fetch() { # fetch NAME SHA256
  name="$1"
  sha="$2"
  target="${DEST}/${name}"
  for stale in "${DEST}/.${name}."*; do
    if [ -f "${stale}" ] || [ -L "${stale}" ]; then
      rm -f "${stale}"
      echo "fetch_model: removed a part file left by an earlier run" >&2
    fi
  done
  if matches "${target}" "${sha}"; then
    echo "fetch_model: ${name} present and verified"
    return 0
  fi
  if [ -e "${target}" ] || [ -L "${target}" ]; then
    echo "fetch_model: ${name} does not match its pinned SHA-256; removing it" >&2
    rm -f "${target}"
  fi
  mkdir -p "${DEST}"
  PART="$(mktemp "${DEST}/.${name}.XXXXXX")"
  if ! curl --proto '=https' --tlsv1.2 --max-time 300 --retry 3 \
    --proto-redir '=https' -fsSL -o "${PART}" "${BASE_URL}${name}"; then
    echo "fetch_model: download of ${name} failed" >&2
    exit 1
  fi
  if ! matches "${PART}" "${sha}"; then
    echo "fetch_model: ${name} failed SHA-256 verification; nothing installed" >&2
    exit 1
  fi
  chmod 0644 "${PART}"
  mv -f "${PART}" "${target}"
  PART=""
  echo "fetch_model: installed ${name} to ${DEST}"
}

fetch yolox_s.onnx "${SHA256_YOLOX_S}"
if [ "${WITH_M}" -eq 1 ]; then
  fetch yolox_m.onnx "${SHA256_YOLOX_M}"
fi
