#!/usr/bin/env sh
# Google SigLIP 2 (Apache-2.0), converted by onnx-community, pinned int8 artifact.
# Usage: sh scripts/fetch_probe_model.sh [--dest DIR]. No credentials are needed.
set -eu

REVISION='ba1f3b0843f24bc5417d38e19c37b287d719b2f4'
SHA256='0dd31785a2713f1113ef2272472165c69d580473dae38d7b47568ac587795e70'
BYTES=94553333
NAME='siglip2-base-patch16-224-int8.onnx'
REMOTE="onnx-community/siglip2-base-patch16-224-ONNX/resolve/${REVISION}/onnx/vision_model_int8.onnx"
MANUAL_URL="https://huggingface.co/${REMOTE}"
PROBE_ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
DEST="${PROBE_ROOT}/.models"
if [ "$#" -gt 0 ]; then
  [ "$#" -eq 2 ] && [ "$1" = '--dest' ] || {
    echo 'usage: fetch_probe_model.sh [--dest DIR]' >&2
    exit 2
  }
  DEST="$2"
fi
PART=''
trap '[ -z "${PART}" ] || rm -f "${PART}"' EXIT
trap 'exit 1' HUP INT TERM

matches() {
  [ -f "$1" ] && [ "$(wc -c < "$1" | tr -d ' ')" = "${BYTES}" ] &&
    [ "$(sha256sum "$1" | cut -d ' ' -f 1)" = "${SHA256}" ]
}

target="${DEST}/${NAME}"
if matches "${target}"; then
  echo "fetch_probe_model: ${NAME} present and verified"
  exit 0
fi
mkdir -p "${DEST}"
PART="$(mktemp "${DEST}/.${NAME}.XXXXXX")"
# Each endpoint has a timeout and at most one retry (two attempts).
for origin in 'https://huggingface.co' 'https://hf-mirror.com' 'https://modelscope.cn/models'; do
  if curl --proto '=https' --tlsv1.2 --proto-redir '=https' --connect-timeout 15 \
    --max-time 120 --retry 1 --retry-max-time 240 -fsSL -o "${PART}" "${origin}/${REMOTE}" &&
    matches "${PART}"; then
    chmod 0644 "${PART}"
    mv -f "${PART}" "${target}"
    PART=''
    echo "fetch_probe_model: installed ${NAME}; SHA-256 verified"
    exit 0
  fi
done
echo 'fetch_probe_model: pinned int8 download unavailable; nothing unverified installed.' >&2
echo "Manual download: ${MANUAL_URL}" >&2
echo "Target: ${target}" >&2
echo "Expected bytes: ${BYTES}; SHA-256: ${SHA256}" >&2
exit 1
