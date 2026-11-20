#!/usr/bin/env bash
# Run one shard of a Go test group, and prove it ran exactly its share.
#
# Usage: run-go-shard.sh [-race] [-exclude REGEX] [-split FILE] -- NAME OF INDEX PATTERN...
#   NAME     names the group in messages and in the results file
#   OF INDEX the shard count and this shard's number (0-based)
#   PATTERN  go list patterns the group covers (./... for every package)
#
# The shard lists the group's items (go-test-items.sh), checks that the shards partition them
# (shard.py check), takes its share by the measured weights (.github/shards/weights.txt), runs the
# whole packages of its share in one go test and each split package's tests in a go test of their
# own, side by side, and then checks (shard.py verify) that every item of the share has a result.
# The results are go test -json, left in $RUNNER_TEMP/shard-NAME-INDEX/results.json with the
# test names in names.tsv.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/../.."
race=()
fargs=()
while [[ "${1:-}" == -* && "$1" != "--" ]]; do
	case "$1" in
	-race) race=(-race); shift ;;
	-exclude) fargs+=(-exclude "$2"); shift 2 ;;
	-split) fargs+=(-split "$2"); shift 2 ;;
	*) echo "run-go-shard: unknown flag $1" >&2; exit 2 ;;
	esac
done
[[ "${1:-}" == "--" ]] && shift
[[ $# -ge 4 ]] || { echo "usage: run-go-shard.sh [flags] -- NAME OF INDEX PATTERN..." >&2; exit 2; }
name="$1"; of="$2"; idx="$3"; shift 3
module="$(go list -m)"
out="${RUNNER_TEMP:-$(mktemp -d)}/shard-$name-$idx"
rm -rf "$out"; mkdir -p "$out"
weights=.github/shards/weights.txt
shard="python3 -I $here/shard.py"

"$here/go-test-items.sh" "${fargs[@]+"${fargs[@]}"}" "$@" > "$out/all.txt"
$shard check -of "$of" -min-items 1 -weights "$weights" -default 1 < "$out/all.txt"
$shard select -of "$of" -index "$idx" -weights "$weights" -default 1 < "$out/all.txt" > "$out/mine.txt"
echo "group $name, shard $idx of $of: $(wc -l < "$out/mine.txt") of $(wc -l < "$out/all.txt") items"
cat "$out/mine.txt"

# Whole packages in one go test; each split package's tests in a go test of their own.
grep -v ' ' "$out/mine.txt" | sed 's|^\.$|.|; s|^\([^.]\)|./\1|' > "$out/pkgs.txt" || true
grep ' ' "$out/mine.txt" | awk '{print $1}' | sort -u > "$out/splitpkgs.txt" || true
pids=()
files=()
n=0
if [[ -s "$out/pkgs.txt" ]]; then
	f="$out/run-$n.json"; files+=("$f"); n=$((n + 1))
	# shellcheck disable=SC2046
	( go test "${race[@]+"${race[@]}"}" -count=1 -timeout 30m -json $(cat "$out/pkgs.txt") > "$f" ) &
	pids+=($!)
fi
while read -r pkg; do
	[[ -n "$pkg" ]] || continue
	re="^($(grep "^$pkg " "$out/mine.txt" | awk '{print $2}' | paste -sd'|' -))\$"
	f="$out/run-$n.json"; files+=("$f"); n=$((n + 1))
	( go test "${race[@]+"${race[@]}"}" -count=1 -timeout 30m -json -run "$re" "./$pkg" > "$f" ) &
	pids+=($!)
done < "$out/splitpkgs.txt"
status=0
for p in "${pids[@]}"; do wait "$p" || status=$?; done
cat "${files[@]}" > "$out/results.json"
python3 -I "$here/shard.py" names -results "$out/results.json" > "$out/names.tsv"
echo "$(wc -l < "$out/names.tsv") test results recorded in $out/names.tsv"
if [[ "$status" -ne 0 ]]; then
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
grep -v ' ' "$out/mine.txt" > "$out/mine-pkgs.txt" || true
grep ' ' "$out/mine.txt" > "$out/mine-tests.txt" || true
if [[ -s "$out/mine-pkgs.txt" ]]; then $shard verify -results "$out/results.json" -mode package -module "$module" < "$out/mine-pkgs.txt"; fi
if [[ -s "$out/mine-tests.txt" ]]; then $shard verify -results "$out/results.json" -mode test -module "$module" < "$out/mine-tests.txt"; fi
