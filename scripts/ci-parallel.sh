#!/usr/bin/env bash
# ci-parallel.sh: run the parts of one sharded command side by side, in one job.
#
# Usage: ci-parallel.sh COUNT COMMAND
#
# Runs `bash -c COMMAND _ PART` for every PART in 0..COUNT-1 at once, so COMMAND reads its part as
# "$1" (an exported shell function can be its body). That shell does not inherit set -e, so a
# COMMAND of several steps sets its own options. Each part's output goes to its own log, printed
# whole (as a collapsed group in a GitHub log) after every part has finished, so the parts' lines
# never interleave. The exit status is 0 only when every part exited 0; a part that fails does not
# stop the others, and each part's status is printed. The oracle workflows use it to run the
# shards that were once a job each.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: ci-parallel.sh COUNT COMMAND" >&2
  exit 2
fi
count="$1"; command="$2"
case "$count" in *[!0-9]*|''|0) echo "ci-parallel: COUNT must be a positive integer" >&2; exit 2 ;; esac

logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT
pids=()
for ((part = 0; part < count; part++)); do
  bash -c "$command" _ "$part" > "$logs/$part.log" 2>&1 &
  pids+=("$!")
done
failed=0
for ((part = 0; part < count; part++)); do
  if wait "${pids[$part]}"; then status=0; else status=$?; failed=1; fi
  echo "::group::part $part of $count: exit $status"
  cat "$logs/$part.log"
  echo "::endgroup::"
done
[ "$failed" -eq 0 ] || { echo "ci-parallel: a part failed (exit statuses above)" >&2; exit 1; }
