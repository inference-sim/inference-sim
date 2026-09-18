#!/usr/bin/env bash
# Dispatch a workflow with exponential-backoff retry (#1757).
#
# Usage: dispatch-with-retry.sh <workflow> --repo <repo> --ref <ref> -f key=val ...
#
# Retries up to 3 times with 5s/10s backoff. Exits 0 on success, 1 after exhaustion.
# All arguments after the script name are forwarded to `gh workflow run`.
set -uo pipefail

max_attempts=3
for (( attempt = 1; attempt <= max_attempts; attempt++ )); do
  if gh workflow run "$@"; then
    echo "Dispatched (attempt $attempt)."
    exit 0
  fi
  if (( attempt < max_attempts )); then
    delay=$(( 5 * (1 << (attempt - 1)) ))
    echo "::warning::dispatch attempt $attempt failed; retrying in ${delay}s"
    sleep "$delay"
  fi
done
echo "::error::all $max_attempts dispatch attempts failed"
exit 1
