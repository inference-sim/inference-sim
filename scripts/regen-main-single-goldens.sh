#!/usr/bin/env bash
# Regenerates cmd/testdata/golden/main-single-*: the stdout and per-request metrics of single-instance
# runs made with the binary of a BLIS commit that routes natively (default: 0d092d4e, the main commit
# llm-d-r started from). TestSingleInstance_MatchesMainGolden compares this branch's output with them.
#
#   scripts/regen-main-single-goldens.sh [commit]
#
# Run it from the repository root; it builds the commit in a temporary worktree and removes it.
set -euo pipefail
commit=${1:-0d092d4e}
root=$(git rev-parse --show-toplevel)
cd "$root"
tmp=$(mktemp -d)
trap 'git worktree remove --force "$tmp/src" >/dev/null 2>&1 || true; rm -rf "$tmp"' EXIT
git worktree add -q --detach "$tmp/src" "$commit"
(cd "$tmp/src" && go build -o "$tmp/blis" main.go)
kernel=$(go list -m -f '{{.Dir}}' github.com/inference-sim/blis-latency-kernel)
for scenario in llama-3.1-70b-instruct-h200-fp8-vllm-tp4 gpt-oss-120b-h200-fp4-vllm-tp4; do
  out=cmd/testdata/golden/main-single-$scenario
  "$tmp/blis" run --scenario "$scenario.yaml" --scenarios "$kernel/testdata/aisimulate" \
    --catalog testdata/catalog --registry testdata/registry --defaults-filepath defaults.yaml \
    --num-instances 1 --rate 10 --num-requests 200 --seed 42 \
    --metrics-path "$out.metrics.json" > "$out.stdout"
  echo "wrote $out.{stdout,metrics.json} from $commit"
done
