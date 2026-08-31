#!/usr/bin/env bash
#
# archon-build.sh — Clone and build archon-go at a pinned version.
#
# Usage: scripts/archon-build.sh [version]
#
# Outputs the path to the built binary on stdout.
# Requires: git, go (1.26+)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION_FILE="$SCRIPT_DIR/.archon-version"

if [[ -n "${1:-}" ]]; then
  VERSION="$1"
elif [[ -f "$VERSION_FILE" ]]; then
  VERSION="$(cat "$VERSION_FILE" | tr -d '[:space:]')"
else
  echo "ERROR: No version argument and .archon-version not found at $VERSION_FILE" >&2
  exit 1
fi
ARCHON_REPO="https://github.com/AI-native-Systems-Research/archon.git"
BUILD_DIR="${RUNNER_TEMP:-/tmp}/archon-build"

echo "Building archon-go $VERSION..." >&2
rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$VERSION" "$ARCHON_REPO" "$BUILD_DIR" >&2

cd "$BUILD_DIR"
go build -o "$BUILD_DIR/archon-go" . >&2
echo "Built: $BUILD_DIR/archon-go" >&2

echo "$BUILD_DIR/archon-go"
