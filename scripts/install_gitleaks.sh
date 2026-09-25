#!/usr/bin/env sh
# Download the pinned gitleaks release (MIT), verify its SHA-256 and install it into
# .tools/bin/. Linux x86_64 only, which is what CI (ubuntu-24.04) runs.
set -eu

VERSION="8.30.1"
SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
ASSET="gitleaks_${VERSION}_linux_x64.tar.gz"
URL="https://github.com/gitleaks/gitleaks/releases/download/v${VERSION}/${ASSET}"

ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
BIN_DIR="${ROOT}/.tools/bin"

if [ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]; then
  echo "install_gitleaks: only Linux x86_64 is supported; install gitleaks ${VERSION} manually" >&2
  exit 1
fi

if [ -x "${BIN_DIR}/gitleaks" ] && "${BIN_DIR}/gitleaks" version | grep -qx "${VERSION}"; then
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

curl --proto '=https' --tlsv1.2 -fsSL --max-time 120 --retry 3 -o "${TMP}/${ASSET}" "${URL}"
echo "${SHA256}  ${TMP}/${ASSET}" | sha256sum -c --quiet -
tar -xzf "${TMP}/${ASSET}" -C "${TMP}" gitleaks
mkdir -p "${BIN_DIR}"
install -m 0755 "${TMP}/gitleaks" "${BIN_DIR}/gitleaks"
echo "install_gitleaks: installed gitleaks ${VERSION} to ${BIN_DIR}"
