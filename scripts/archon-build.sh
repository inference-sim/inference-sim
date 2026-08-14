#!/usr/bin/env bash
#
# archon-build.sh — Clone and build archon-go at a pinned version.
#
# Usage: scripts/archon-build.sh [version]
#
# Outputs the path to the built binary on stdout.
# Requires: git, go (1.26+)

set -euo pipefail

VERSION="${1:-v0.2.0}"
ARCHON_REPO="https://github.com/AI-native-Systems-Research/archon.git"
BUILD_DIR="${RUNNER_TEMP:-/tmp}/archon-build"

rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$VERSION" "$ARCHON_REPO" "$BUILD_DIR" >&2

cd "$BUILD_DIR"
go build -o "$BUILD_DIR/archon-go" . >&2

echo "$BUILD_DIR/archon-go"
