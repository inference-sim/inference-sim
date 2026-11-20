#!/usr/bin/env bash
# golden-change-check.sh BASE HEAD < messages
#
# A change that moves a byte-gate golden (testdata/bytegate/golden/) is a behaviour change. The
# commit messages of the change, read from standard input, must say what moved and why on a line that starts
# "Golden change:" and is followed by that statement (on the same line or the lines below it up to
# the next blank line). A diff that leaves the goldens alone needs nothing. Exit 1 names the moved
# goldens when the statement is missing or empty.
set -euo pipefail
base=${1:?base commit}
head=${2:?head commit}
body=$(cat)

moved=$(git diff --name-only "$base" "$head" -- testdata/bytegate/golden/)
if [ -z "$moved" ]; then
  echo "golden-change: no byte-gate golden moved"
  exit 0
fi

statement=$(printf '%s\n' "$body" | awk '
  found && /^[[:space:]]*$/ { exit }
  found { print; next }
  /^[[:space:]]*Golden change:/ { found = 1; sub(/^[[:space:]]*Golden change:[[:space:]]*/, ""); print }
')
words=$(printf '%s' "$statement" | wc -w | tr -d ' ')
if [ "$words" -lt 8 ]; then
  echo "::error::these byte-gate goldens moved, and no commit message has a 'Golden change:' statement of what moved and why (at least 8 words):"
  printf '%s\n' "$moved"
  exit 1
fi
echo "golden-change: $(printf '%s\n' "$moved" | wc -l | tr -d ' ') golden file(s) moved, stated"
