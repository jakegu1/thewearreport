#!/usr/bin/env sh
# Install the pinned uv release with Astral's official standalone installer
# (into ~/.local/bin by default). Used by `make setup` when uv is missing.
set -eu

UV_VERSION="${UV_VERSION:-0.12.18}"

curl --proto '=https' --tlsv1.2 -LsSf --max-time 120 --retry 3 \
  "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
