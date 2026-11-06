#!/bin/bash
# Regenerate the hash-locked Python requirement files from their .in sources (X21).
# Every Python CI job installs a .txt file with pip install --require-hashes, so a
# dependency that is not pinned, or whose archive changed, fails the install.
#
#   tools/pylock.sh           write every lock file
#   tools/pylock.sh --check   fail if a lock file differs from what it would write
#
# The lock files are resolved for CPython on Linux, the platform CI runs, at each job's
# Python version. uv pip compile is pip-compile's resolver; --generate-hashes writes every
# archive's sha256. A range in a .in file resolves to the newest matching release at the
# time of locking; --check therefore compares against a fresh resolution, and fails when a
# new release appears in a range. CI does not run --check; it installs the committed lock.
set -euo pipefail
cd "$(dirname "$0")/.."
# in-file  out-file  python-version
locks=(
  "catalog/scripts/requirements-dev.in catalog/scripts/requirements-dev.txt 3.12"
  "registry/scripts/requirements.in registry/scripts/requirements.txt 3.11"
  "registry/scripts/requirements-fit.in registry/scripts/requirements-fit.txt 3.11"
  "docs/requirements.in docs/requirements.txt 3.12"
)
check=false
[ "${1:-}" = "--check" ] && check=true
status=0
for l in "${locks[@]}"; do
  read -r in out py <<<"$l"
  tmp=$(mktemp)
  (cd "$(dirname "$in")" && uv pip compile --quiet --generate-hashes --no-header \
    --python-version "$py" --python-platform x86_64-unknown-linux-gnu \
    "$(basename "$in")") > "$tmp.body"
  {
    echo "# Generated from $(basename "$in") by tools/pylock.sh for CPython $py on Linux; do not edit."
    echo "# Install with: pip install --require-hashes -r $(basename "$out")"
    cat "$tmp.body"
  } > "$tmp"
  rm -f "$tmp.body"
  if $check; then
    if ! cmp -s "$tmp" "$out"; then echo "$out differs from what tools/pylock.sh writes" >&2; status=1; fi
    rm -f "$tmp"
  else
    mv "$tmp" "$out"
  fi
done
exit $status
