#!/usr/bin/env sh
# Download the pinned actionlint release (MIT), verify its SHA-256 and install it into
# .tools/bin/. A development tool that checks .github/workflows/; never a runtime
# dependency. Linux x86_64 only, which is what CI (ubuntu-24.04) runs.
set -eu

VERSION="1.7.12"
SHA256="8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
ASSET="actionlint_${VERSION}_linux_amd64.tar.gz"
URL="https://github.com/rhysd/actionlint/releases/download/v${VERSION}/${ASSET}"

ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
BIN_DIR="${ROOT}/.tools/bin"

if [ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]; then
  echo "install_actionlint: only Linux x86_64 is supported; install actionlint ${VERSION} manually" >&2
  exit 1
fi

if [ -x "${BIN_DIR}/actionlint" ] && "${BIN_DIR}/actionlint" -version | head -n 1 | grep -qx "${VERSION}"; then
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

curl --proto '=https' --tlsv1.2 -fsSL --max-time 120 --retry 3 -o "${TMP}/${ASSET}" "${URL}"
echo "${SHA256}  ${TMP}/${ASSET}" | sha256sum -c --quiet -
tar -xzf "${TMP}/${ASSET}" -C "${TMP}" actionlint
mkdir -p "${BIN_DIR}"
install -m 0755 "${TMP}/actionlint" "${BIN_DIR}/actionlint"
echo "install_actionlint: installed actionlint ${VERSION} to ${BIN_DIR}"
