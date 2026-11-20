#!/usr/bin/env bash
# test-selected-deps.sh — the pre-release test of the router fork's packages (design §3.2, LIF-2).
#
# Go builds every module in a program at the newest version any of them requires, so the
# llm-d router packages this binary links run against the versions that inference-sim selects,
# not the versions the fork's own go.mod pins. This script runs their tests at exactly those
# versions: in a scratch copy of inference-sim's module, with the test dependencies of those
# packages added, it runs go test over each router package the binary links. The engines and the
# substrate are directories of this module and are tested by `go test ./...`.
#
# Usage: scripts/test-selected-deps.sh [go test flags...]
# Run from anywhere; it works in a temporary directory and changes nothing in the checkout.
#
# Sharding (CI runs the packages as parallel jobs): set SELECTED_DEPS_SHARD and CI_SHARD_OUT.
#   SELECTED_DEPS_SHARD=rest:I/N         the router packages, package I of N by import path
# CI_SHARD_OUT is the directory the evidence files go to (list.txt, assigned.txt, ran.txt), which
# scripts/ci-shard-verify.sh reads after all shards finish. Unset, every package runs.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

cp "$root/go.mod" "$root/go.sum" "$scratch/"
# A package of the main module that imports the binary's entry point, so the scratch module
# selects exactly what the binary selects.
mkdir -p "$scratch/selected"
cat > "$scratch/selected/selected.go" <<'GO'
package selected

import _ "github.com/inference-sim/inference-sim/cmd"
GO
# The main module's own packages come from the checkout.
placeholder_version=v0.0.0-00010101000000-000000000000
echo "replace github.com/inference-sim/inference-sim => $root" >> "$scratch/go.mod"
sed -i.bak 's#^module github.com/inference-sim/inference-sim$#module blis.test/selected#' "$scratch/go.mod"
rm -f "$scratch/go.mod.bak"
(cd "$scratch" && go mod edit -require="github.com/inference-sim/inference-sim@${placeholder_version}")

cd "$scratch"
pkgs="$(go list -deps ./selected | grep -E '^github.com/llm-d/llm-d-router(/|$)' || true)"
if [ -z "$pkgs" ]; then
  echo "ERROR: the binary links no llm-d router package; the scratch module is wrong" >&2
  exit 1
fi
# Test-only dependencies of those packages are added at whatever version their modules
# require; the versions the binary links do not move. A replaced module is compared by its
# replacement, which is what the build uses (LIF-2): the router's require line carries a
# placeholder version that go get may raise.
selected() {
  go list -m -f '{{.Path}} {{if .Replace}}=> {{.Replace.Path}} {{.Replace.Version}}{{else}}{{.Version}}{{end}}' all |
    grep -E '^github.com/inference-sim/|^github.com/llm-d/llm-d-router '
}
before="$(selected)"
# shellcheck disable=SC2086
go get -t $pkgs >/dev/null
after="$(selected)"
if [ "$before" != "$after" ]; then
  echo "ERROR: adding the test dependencies moved a BLIS module:" >&2
  diff <(echo "$before") <(echo "$after") >&2 || true
  exit 1
fi
echo "testing $(echo "$pkgs" | wc -l | tr -d ' ') packages at the versions inference-sim selects"
# The router tests that read the substrate's data (the catalog and the registry) find it through
# the checkout this script is run from: the substrate is a directory of this module, so the data is
# the data the binary embeds. A value the caller sets wins.
substrate_dir="$root/substrate"
if [ ! -d "$substrate_dir/catalog" ]; then
  echo "ERROR: the checkout has no substrate/catalog directory (got '$substrate_dir')" >&2
  exit 1
fi
export BLIS_CATALOG="${BLIS_CATALOG:-$substrate_dir/catalog}" BLIS_REGISTRY="${BLIS_REGISTRY:-$substrate_dir/registry}"
echo "substrate data: $substrate_dir"

shard="${SELECTED_DEPS_SHARD:-}"
if [ -z "$shard" ]; then
  # shellcheck disable=SC2086
  go test "$@" $pkgs
  exit
fi

out="${CI_SHARD_OUT:?SELECTED_DEPS_SHARD needs CI_SHARD_OUT, the evidence directory}"
mkdir -p "$out"
case "$out" in /*) ;; *) out="$root/$out" ;; esac
kind="${shard%%:*}"; frac="${shard#*:}"; index="${frac%/*}"; count="${frac#*/}"
case "$kind" in
  rest)
    # Package-level partition; the evidence files are the list, the assigned packages, and the ones
    # go test reported on.
    echo "$pkgs" | LC_ALL=C sort -u > "$out/list.txt"
    : > "$out/assigned.txt"
    while IFS= read -r pkg; do
      sum="$(printf '%s' "$pkg" | cksum | cut -d' ' -f1)"
      if [ $(( sum % count + 1 )) -eq "$index" ]; then printf '%s\n' "$pkg" >> "$out/assigned.txt"; fi
    done < "$out/list.txt"
    echo "selected-deps: shard $index of $count owns $(wc -l < "$out/assigned.txt" | tr -d ' ') of $(wc -l < "$out/list.txt" | tr -d ' ') packages"
    status=0
    # shellcheck disable=SC2046
    go test "$@" $(cat "$out/assigned.txt") 2>&1 | tee "$out/go-test.log" || status=$?
    # Each package go test starts ends in one "ok", "FAIL" or "?" line (a package without test files).
    sed -n -E 's/^(ok|FAIL|\?)[[:space:]]+([^[:space:]]+).*/\2/p' "$out/go-test.log" | LC_ALL=C sort -u > "$out/ran.txt"
    rm -f "$out/go-test.log"
    [ "$status" -eq 0 ] || exit "$status"
    diff <(LC_ALL=C sort -u "$out/assigned.txt") "$out/ran.txt" >&2 \
      || { echo "ERROR: go test reported on a different set of packages than shard $index of $count owns" >&2; exit 1; }
    ;;
  *)
    echo "ERROR: SELECTED_DEPS_SHARD '$shard' is not rest:I/N" >&2
    exit 2
    ;;
esac
