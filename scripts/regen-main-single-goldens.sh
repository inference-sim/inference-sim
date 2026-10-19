#!/usr/bin/env bash
# Regenerates cmd/testdata/golden/main-single-*: the stdout and per-request metrics of single-instance
# runs made with the binary of a BLIS commit that routes natively (default: 0d092d4e, the main commit
# llm-d-r started from). TestSingleInstance_MatchesMainGolden compares this branch's output with them.
# The current goldens were made from main 63ffc020 with this branch's kernel bump applied (see the
# test's comment); the script refuses a commit that pins another kernel or schemas.
#
#   scripts/regen-main-single-goldens.sh [commit]
#
# Run it from the repository root; it builds the commit in a temporary worktree and removes it.
# The commit must pin the same blis-latency-kernel and blis-schemas as this branch, or its goldens
# would price with another kernel; when it does not, commit the same bump (go.mod, re-vendored
# testdata, adapter changes) onto it in a throwaway worktree, never on main, and pass that commit.
set -euo pipefail
commit=${1:-0d092d4e}
root=$(git rev-parse --show-toplevel)
cd "$root"
tmp=$(mktemp -d)
trap 'git worktree remove --force "$tmp/src" >/dev/null 2>&1 || true; rm -rf "$tmp"' EXIT
git worktree add -q --detach "$tmp/src" "$commit"
for dep in github.com/inference-sim/blis-latency-kernel github.com/inference-sim/blis-schemas; do
  want=$(grep -E "^[[:space:]]*$dep " go.mod || true)
  got=$(grep -E "^[[:space:]]*$dep " "$tmp/src/go.mod" || true)
  if [[ "$want" != "$got" ]]; then
    echo "regen-main-single-goldens: $commit pins '${got:-nothing}' for $dep, this branch '${want}'; bump it first (see above)" >&2
    exit 1
  fi
done
for vendored in testdata/catalog testdata/registry; do
  if ! cmp -s "$vendored/.upstream-commit" "$tmp/src/$vendored/.upstream-commit"; then
    echo "regen-main-single-goldens: $commit vendors a different $vendored than this branch; re-vendor it first (see above)" >&2
    exit 1
  fi
done
if [[ -n "$(git -C "$tmp/src" status --porcelain)" ]]; then
  echo "regen-main-single-goldens: the $commit checkout is not clean" >&2
  exit 1
fi
(cd "$tmp/src" && go build -o "$tmp/blis" main.go)
kernel=$(go list -m -f '{{.Dir}}' github.com/inference-sim/blis-latency-kernel)
# Run from inside the generation commit, so the catalog the binary reads, and the provenance it
# records ("revision", "dirty": false), are that clean commit's own testdata, not this checkout's.
for scenario in llama-3.1-70b-instruct-h200-fp8-vllm-tp4 gpt-oss-120b-h200-fp4-vllm-tp4; do
  out="$root/cmd/testdata/golden/main-single-$scenario"
  (cd "$tmp/src" && "$tmp/blis" run --scenario "$scenario.yaml" --scenarios "$kernel/testdata/aisimulate" \
    --catalog testdata/catalog --registry testdata/registry --defaults-filepath defaults.yaml \
    --num-instances 1 --rate 10 --num-requests 200 --seed 42 \
    --metrics-path "$out.metrics.json" > "$out.stdout")
  echo "wrote $out.{stdout,metrics.json} from $commit"
done
