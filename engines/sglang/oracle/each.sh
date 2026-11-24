#!/usr/bin/env bash
# Original to blis-sglang.
# each.sh: run COMMAND once per FILE, several at once, on the CPUs this job may use.
#
# Usage: oracle/each.sh COMMAND FILE...
#
# Runs `bash -c COMMAND _ FILE` for every FILE, ORACLE_JOBS at once (else what forkpool.py prints:
# a pod's CPU limit, not the node's count nproc gives). COMMAND reads its file as "$1"; it runs in
# a child shell, which sees exported variables but not set -e. Each FILE's command is a process of
# its own writing a file of its own, so the files run in any order. The exit status is 0 only when
# every command exited 0; one that fails does not stop the others. No FILE runs nothing.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: oracle/each.sh COMMAND FILE..." >&2
  exit 2
fi
command="$1"
shift
if [ "$#" -eq 0 ]; then
  exit 0
fi
jobs="${ORACLE_JOBS:-$(python3 -I "$(dirname "$0")/forkpool.py")}"
case "$jobs" in *[!0-9]*|''|0) echo "each.sh: ORACLE_JOBS must be a positive integer, got '$jobs'" >&2; exit 2 ;; esac
if ! printf '%s\0' "$@" | xargs -0 -n 1 -P "$jobs" bash -c "$command" _; then
  echo "each.sh: a command failed (its output is above)" >&2
  exit 1
fi
