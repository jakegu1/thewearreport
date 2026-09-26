#!/usr/bin/env sh
# Download the spot-check judge's weights (GGUF, from Hugging Face) and verify each file
# against a pinned SHA-256: with --all, every bake-off candidate; with --model NAME, one of
# them. The pins are the ones in engine/wearreport/tools/judge.py (CANDIDATES), and each URL
# names a pinned commit.
#
# No judge model is chosen: no candidate passes the quality bar (T-029 bake-off). Without
# --all or --model the script therefore downloads nothing and exits non-zero.
#
# Usage: sh scripts/fetch_judge_model.sh (--all | --model NAME) [--dest DIR]
#        (DIR defaults to .models/judge/)
#
# Fails closed: a file is moved into DIR only after its checksum matches; a file already
# in DIR that does not match is deleted and downloaded again; part files left by an
# earlier, killed run are deleted first. On any failure the script exits non-zero and
# leaves no unverified file behind. Sends no credentials, and runs no Hugging Face client.
set -eu

CHOSEN=""
CANDIDATES="qwen3.5-2b qwen3.5-4b qwen3-vl-2b internvl3.5-2b smolvlm2-2.2b"
ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
DEST="${ROOT}/.models/judge"
MODELS="${CHOSEN}"

while [ $# -gt 0 ]; do
  case "$1" in
    --all) MODELS="${CANDIDATES}" ;;
    --model | --dest)
      [ $# -ge 2 ] || { echo "fetch_judge_model: $1 needs a value" >&2; exit 2; }
      if [ "$1" = "--model" ]; then MODELS="$2"; else DEST="$2"; fi
      shift
      ;;
    *) echo "usage: fetch_judge_model.sh (--all | --model NAME) [--dest DIR]" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "${MODELS}" ]; then
  echo "fetch_judge_model: no judge model is chosen (no candidate passes the quality bar);" \
    "nothing downloaded. Use --model NAME or --all to fetch bake-off candidates." >&2
  exit 1
fi

PART=""
trap '[ -z "${PART}" ] || rm -f "${PART}"' EXIT
trap 'exit 1' HUP INT TERM

mkdir -p "${DEST}"
rm -f "${DEST}"/.*.part.*

matches() { # matches FILE SHA256
  [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d ' ' -f 1)" = "$2" ]
}

fetch() { # fetch REPO REVISION NAME SHA256
  name="$3"
  target="${DEST}/${name}"
  if matches "${target}" "$4"; then
    echo "fetch_judge_model: ${name} present and verified"
    return 0
  fi
  if [ -e "${target}" ] || [ -L "${target}" ]; then
    echo "fetch_judge_model: ${name} does not match its pinned SHA-256; removing it" >&2
    rm -f "${target}"
  fi
  PART="$(mktemp "${DEST}/.${name}.part.XXXXXX")"
  if ! curl --proto '=https' --tlsv1.2 --max-time 1800 --retry 3 \
    --proto-redir '=https' -fsSL -o "${PART}" \
    "https://huggingface.co/$1/resolve/$2/${name}"; then
    echo "fetch_judge_model: download of ${name} failed" >&2
    exit 1
  fi
  if ! matches "${PART}" "$4"; then
    echo "fetch_judge_model: ${name} failed SHA-256 verification; nothing installed" >&2
    exit 1
  fi
  chmod 0644 "${PART}"
  mv -f "${PART}" "${target}"
  PART=""
  echo "fetch_judge_model: installed ${name} to ${DEST}"
}

count=0
for model in ${MODELS}; do
  count=$((count + 1))
  case "${model}" in
  qwen3.5-2b) set -- \
    "bartowski/Qwen_Qwen3.5-2B-GGUF" "7d26695454df6de5fbcce2e58681e62dae06ce43" \
    "Qwen_Qwen3.5-2B-Q8_0.gguf" "be647507ce6cde229b838924d47bfff9763171105563f7f908670dae57c4dbe2" \
    "mmproj-Qwen_Qwen3.5-2B-f16.gguf" "044a0ea136cca70711ae16e23b24d754b44eab6f2462d187aee4d7c7a9503d36" ;;
  qwen3.5-4b) set -- \
    "bartowski/Qwen_Qwen3.5-4B-GGUF" "4168f45a16a1290d65a4ec0fa312ae917a4c15d6" \
    "Qwen_Qwen3.5-4B-Q8_0.gguf" "5c74c0ede371924357dff0cb6ba145bd67208b9b2389ded681adfff3f7608db7" \
    "mmproj-Qwen_Qwen3.5-4B-f16.gguf" "659b59dd44b73b1cd34af6cc424669484b06dc80f4340adf8ea84ad776eef813" ;;
  qwen3-vl-2b) set -- \
    "Qwen/Qwen3-VL-2B-Instruct-GGUF" "52d6c8ffea26cc873ac5ad116f8631268d7eb503" \
    "Qwen3VL-2B-Instruct-Q8_0.gguf" "1e8db19207c8ce0733ddd78c2eff8a9e22c27c82f4443df94c25792ed8fe04f2" \
    "mmproj-Qwen3VL-2B-Instruct-F16.gguf" "c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5" ;;
  internvl3.5-2b) set -- \
    "bartowski/OpenGVLab_InternVL3_5-2B-GGUF" "09023986543a68f5caaa389f64b0e0256fe22565" \
    "OpenGVLab_InternVL3_5-2B-Q8_0.gguf" "6997c6e3a1fe5920ac1429a21a3ec15d545e14eb695ee3656834859e617800b5" \
    "mmproj-OpenGVLab_InternVL3_5-2B-f16.gguf" "e83ba6e675b747f7801557dc24594f43c17a7850b6129d4972d55e3e9b010359" ;;
  smolvlm2-2.2b) set -- \
    "ggml-org/SmolVLM2-2.2B-Instruct-GGUF" "1bc3c9f74ceafd4c8d4411cc9cf188bba3798f91" \
    "SmolVLM2-2.2B-Instruct-Q8_0.gguf" "c850ffa51b0708be8911766e1d35e8e71365e987c8efb2513a7f237baade074f" \
    "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf" "db9a3a1648cab1ebc3af4a2b0c8145dd8faebf6f7dd7b16e7dc1842229f14ac4" ;;
    *) echo "fetch_judge_model: unknown model ${model}" >&2; exit 2 ;;
  esac
  fetch "$1" "$2" "$3" "$4"
  fetch "$1" "$2" "$5" "$6"
done

if [ "${count}" -eq 0 ]; then
  echo "fetch_judge_model: no model named; nothing downloaded" >&2
  exit 1
fi
