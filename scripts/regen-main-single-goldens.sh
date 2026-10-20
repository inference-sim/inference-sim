#!/usr/bin/env bash
# Regenerates cmd/testdata/golden/main-single-*: the stdout and per-request metrics of single-instance
# runs made with the binary of a BLIS commit that routes natively (a main commit).
# TestSingleInstance_MatchesMainGolden compares this branch's output with them.
#
#   scripts/regen-main-single-goldens.sh <commit>
#
# <commit> is required: no published commit both routes natively and prices with this branch's
# kernel. The current goldens were made from main 63ffc020 (pinning blis-latency-kernel v0.1.0)
# with this branch's kernel bump applied as a local throwaway commit:
#
#   scripts/regen-main-single-goldens.sh 63ffc020
#
# When <commit> pins another blis-latency-kernel or blis-schemas than this branch, the script
# applies that same bump itself, in a temporary worktree, as one commit with a fixed identity and
# date (never on a branch, never pushed). The bump is exactly what produced the current goldens:
#   - go.mod / go.sum: this branch's blis-latency-kernel and blis-schemas requirements and sums;
#   - testdata/catalog and testdata/registry: this branch's vendored releases;
#   - the adapter's typed max_model_len change: this branch's sim/kernelmodel/settings.go and
#     sim/kernelmodel/harness/enginesettings.go.
# It is verified for 63ffc020 only: re-running the line above reproduces the committed goldens
# byte for byte except the catalog provenance "revision", the hash of the bump commit. The
# committed goldens record 8bd6b061, the hand-made bump commit they were produced from; this
# script's bump commit has the same built files but a fixed identity and date, so its hash is
# stable from run to run yet not 8bd6b061. The test masks the revision. For another commit the overlay may
# not compile; the script then stops at the build. A commit that already pins this branch's
# kernel and schemas is built as it is.
#
# Run it from the repository root; it builds in a temporary worktree and removes it.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: scripts/regen-main-single-goldens.sh <commit>   (the current goldens: 63ffc020)" >&2
  exit 2
fi
commit=$1
root=$(git rev-parse --show-toplevel)
cd "$root"
tmp=$(mktemp -d)
# Cleanup on every exit, failure included: remove the worktree, then prune its administrative
# entry in case the removal itself failed, so no orphaned worktree is left behind.
trap 'git worktree remove --force "$tmp/src" >/dev/null 2>&1 || true; rm -rf "$tmp"; git worktree prune' EXIT
git worktree add -q --detach "$tmp/src" "$commit"
deps=(github.com/inference-sim/blis-latency-kernel github.com/inference-sim/blis-schemas)
pinned() { grep -E "^[[:space:]]*$2 " "$1/go.mod" | awk '{print $2}' || true; }
bump=false
for dep in "${deps[@]}"; do
  [[ "$(pinned "$root" "$dep")" == "$(pinned "$tmp/src" "$dep")" ]] || bump=true
done
if $bump; then
  echo "regen-main-single-goldens: $commit pins another kernel/schemas; applying this branch's bump to it" >&2
  for dep in "${deps[@]}"; do
    ver=$(pinned "$root" "$dep")
    (cd "$tmp/src" && go mod edit -require="$dep@$ver")
    grep -E "^$dep $ver[ /]" go.sum >> "$tmp/src/go.sum"
  done
  for dir in testdata/catalog testdata/registry; do
    rm -rf "${tmp:?}/src/$dir"
    cp -R "$dir" "$tmp/src/$dir"
  done
  for f in sim/kernelmodel/settings.go sim/kernelmodel/harness/enginesettings.go; do
    cp "$f" "$tmp/src/$f"
  done
  git -C "$tmp/src" add -A
  GIT_AUTHOR_NAME=regen GIT_AUTHOR_EMAIL=regen@localhost GIT_AUTHOR_DATE="1970-01-01T00:00:00Z" \
    GIT_COMMITTER_NAME=regen GIT_COMMITTER_EMAIL=regen@localhost GIT_COMMITTER_DATE="1970-01-01T00:00:00Z" \
    git -C "$tmp/src" commit -q --no-verify -m "throwaway kernel bump for regen-main-single-goldens (not pushed)"
fi
for dep in "${deps[@]}"; do
  if [[ "$(pinned "$root" "$dep")" != "$(pinned "$tmp/src" "$dep")" ]]; then
    echo "regen-main-single-goldens: $commit still pins another $dep after the bump" >&2
    exit 1
  fi
done
for vendored in testdata/catalog testdata/registry; do
  if ! cmp -s "$vendored/.upstream-commit" "$tmp/src/$vendored/.upstream-commit"; then
    echo "regen-main-single-goldens: $commit vendors a different $vendored than this branch" >&2
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
  out="${REGEN_OUT_DIR:-$root/cmd/testdata/golden}/main-single-$scenario"
  (cd "$tmp/src" && "$tmp/blis" run --scenario "$scenario.yaml" --scenarios "$kernel/testdata/aisimulate" \
    --catalog testdata/catalog --registry testdata/registry --defaults-filepath defaults.yaml \
    --num-instances 1 --rate 10 --num-requests 200 --seed 42 \
    --metrics-path "$out.metrics.json" > "$out.stdout")
  echo "wrote $out.{stdout,metrics.json} from $commit"
done
