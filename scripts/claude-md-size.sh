#!/usr/bin/env bash
# Enforce the CLAUDE.md size ceiling. CLAUDE.md is agent operating context, not a changelog
# (see the charter at the top of the file); this fails when it grows to changelog scale.
#
# Extracted from ci.yml so scripts/claude_md_size_test.go can exercise it against over-ceiling
# and MISSING-file inputs (#1818). Fails CLOSED: a missing/unreadable file or a non-numeric size
# is an error, never a silent pass — an earlier inline version (`set -uo pipefail`, no `-e`) let a
# branch that DELETED CLAUDE.md reach exit 0, because the empty `size` made the `-gt` test error
# out and fall through.
#
# Usage: claude-md-size.sh [file] [ceiling-bytes]   (defaults: CLAUDE.md 40000)
set -uo pipefail

file="${1:-CLAUDE.md}"
ceiling="${2:-40000}"

if [ ! -f "$file" ]; then
  echo "::error::$file is missing or not a regular file — the size guard cannot run. A delivery must not remove it."
  exit 1
fi

size=$(wc -c < "$file" | tr -d '[:space:]')
# `| tr -d [:space:]` strips the leading padding BSD/macOS `wc -c` prints (e.g. " 14500"), which
# would otherwise fail the numeric check below on those platforms (GNU/CI does not pad).
# wc on an existing file yields a number, but guard explicitly rather than trust it: a non-numeric
# value must fail, not be treated as "under the ceiling" by a broken comparison.
if ! [[ "$size" =~ ^[0-9]+$ ]]; then
  echo "::error::could not determine the byte size of $file (got '$size')"
  exit 1
fi

echo "$file is $size bytes (ceiling $ceiling)"
if [ "$size" -gt "$ceiling" ]; then
  echo "::error file=$file::$file is $size bytes, over the $ceiling-byte ceiling."
  echo "CLAUDE.md is agent operating context, not a changelog (see the charter at the top of the"
  echo "file). Per-PR rationale, migration notes and byte-identity evidence belong in the commit"
  echo "body and the relevant docs/ guide — not here. Move the new content there and leave a"
  echo "pointer, or raise the ceiling in ci.yml only with a reason."
  exit 1
fi
