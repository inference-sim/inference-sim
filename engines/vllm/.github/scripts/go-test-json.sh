#!/usr/bin/env bash
# Run `go test -json ARGS...`, keep the results, record the name and result of every test, and on a failure print
# which tests failed and the output of the failing packages.
#
# Usage: go-test-json.sh NAME ARGS...     (the results: $RUNNER_TEMP/gotest-NAME/results.json and names.tsv)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="$1"; shift
out="${RUNNER_TEMP:-$(mktemp -d)}/gotest-$name"
rm -rf "$out"; mkdir -p "$out"
status=0
go test -json "$@" > "$out/results.json" || status=$?
python3 -I "$here/shard.py" names -results "$out/results.json" > "$out/names.tsv"
echo "$(wc -l < "$out/names.tsv") test results recorded in $out/names.tsv"
if [ "$status" -ne 0 ]; then
	echo "::group::failing tests"
	jq -r 'select(.Action=="fail" and .Test) | "FAIL \(.Package) \(.Test)"' "$out/results.json"
	echo "::endgroup::"
	for p in $(jq -r 'select(.Action=="fail" and (.Test|not)) | .Package' "$out/results.json" | sort -u); do
		echo "::group::output of $p"
		jq -r --arg p "$p" 'select(.Package==$p and .Action=="output") | .Output' "$out/results.json" | tail -n 300
		echo "::endgroup::"
	done
	echo "go test failed (exit $status)" >&2
	exit 1
fi
