#!/usr/bin/env bash
#
# archon-build.sh — Clone and build archon-go from repoevolve at a pinned version.
#
# Usage: scripts/archon-build.sh [version]
#
# Outputs the path to the built binary on stdout.
# Requires: git, go (1.22+)

set -euo pipefail

VERSION="${1:-v0.1.0}"
ARCHON_REPO="git@github.ibm.com:ai-native-systems/repoevolve.git"
BUILD_DIR="${RUNNER_TEMP:-/tmp}/archon-build"

rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$VERSION" "$ARCHON_REPO" "$BUILD_DIR"

if [[ ! -d "$BUILD_DIR/code/archon-go" ]]; then
  echo "Error: directory code/archon-go not found in repoevolve $VERSION." >&2
  exit 1
fi

cd "$BUILD_DIR/code/archon-go"
go build -o "$BUILD_DIR/archon-go" .

echo "$BUILD_DIR/archon-go"
