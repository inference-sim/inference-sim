#!/usr/bin/env bash
# Fails when any tracked Go file under the current git repository is not gofmt-clean, naming each
# one, or when the repository tracks no Go file (a check that finds nothing proves nothing).
# Run by .github/workflows/gofmt.yml and by scripts/gofmt_workflow_test.go.
set -euo pipefail
files=()
while IFS= read -r f; do files+=("$f"); done < <(git ls-files '*.go')
if [ ${#files[@]} -eq 0 ]; then
  echo "::error::no tracked Go file found"
  exit 1
fi
bad=$(gofmt -l "${files[@]}")
if [ -n "$bad" ]; then
  echo "::error::these Go files are not gofmt-clean (run gofmt -w):"
  echo "$bad"
  # shellcheck disable=SC2086
  gofmt -d $bad
  exit 1
fi
echo "${#files[@]} Go files are gofmt-clean"
