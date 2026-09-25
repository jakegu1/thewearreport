#!/usr/bin/env sh
# Download the pinned uv release archive (MIT OR Apache-2.0) for this platform, verify
# its SHA-256 and install uv and uvx into ~/.local/bin. Used by `make setup` when uv is
# missing. Fails closed on an unsupported platform or a checksum mismatch.
#
# To upgrade: change VERSION and replace every checksum with the values from the
# release's `<asset>.sha256` files, then check one by downloading and hashing it.
set -eu

VERSION="0.12.18"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    TARGET="x86_64-unknown-linux-gnu"
    SHA256="89eadd7c76fc063887959510d5ba0ab1264dfd5f1143b925ddb73021a40acf16" ;;
  Linux-aarch64 | Linux-arm64)
    TARGET="aarch64-unknown-linux-gnu"
    SHA256="afb6291f3f0a6b4521fc67b947822506c41dde5b60d2189dd8f3695b2ac8c9e7" ;;
  Darwin-x86_64)
    TARGET="x86_64-apple-darwin"
    SHA256="2e4108f5395397c8bc5d43bf83d3bdbb2d0e92b90d0efa607756be704905fa33" ;;
  Darwin-arm64)
    TARGET="aarch64-apple-darwin"
    SHA256="cf40e0c6a202190ccd9e0406dcfdd5b2d6668a9a5c779b17948963df32aafe5b" ;;
  *)
    echo "install_uv: unsupported platform $(uname -s)-$(uname -m); install uv ${VERSION} manually" >&2
    exit 1 ;;
esac

ASSET="uv-${TARGET}.tar.gz"
URL="https://github.com/astral-sh/uv/releases/download/${VERSION}/${ASSET}"
BIN_DIR="${HOME}/.local/bin"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

curl --proto '=https' --tlsv1.2 -fsSL --max-time 120 --retry 3 -o "${TMP}/${ASSET}" "${URL}"

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "${TMP}/${ASSET}" | cut -d ' ' -f 1)"
else
  ACTUAL="$(shasum -a 256 "${TMP}/${ASSET}" | cut -d ' ' -f 1)"
fi
if [ "${ACTUAL}" != "${SHA256}" ]; then
  echo "install_uv: SHA-256 mismatch for ${ASSET}: expected ${SHA256}, got ${ACTUAL}" >&2
  exit 1
fi

tar -xzf "${TMP}/${ASSET}" -C "${TMP}"
mkdir -p "${BIN_DIR}"
install -m 0755 "${TMP}/uv-${TARGET}/uv" "${TMP}/uv-${TARGET}/uvx" "${BIN_DIR}/"
echo "install_uv: installed uv ${VERSION} to ${BIN_DIR}"
