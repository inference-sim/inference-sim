#!/usr/bin/env bash
# test-selected-deps.sh — the pre-release test of engine and router packages (design §3.2, LIF-2).
#
# Go builds every module in a program at the newest version any of them requires, so the
# blis-vllm and llm-d router packages this binary links run against the blis-substrate (and
# every other module) that inference-sim selects, not the versions their own go.mod files
# pin. This script runs their tests at exactly those versions: in a scratch copy of
# inference-sim's module, with the test dependencies of those packages added, it runs
# go test over each package the binary links from blis-vllm and from the router fork.
#
# Usage: scripts/test-selected-deps.sh [go test flags...]
# Run from anywhere; it works in a temporary directory and changes nothing in the checkout.
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
pkgs="$(go list -deps ./selected | grep -E '^(github.com/inference-sim/blis-vllm|github.com/llm-d/llm-d-router)(/|$)' || true)"
if [ -z "$pkgs" ]; then
  echo "ERROR: the binary links no blis-vllm or llm-d router package; the scratch module is wrong" >&2
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
# shellcheck disable=SC2086
go test "$@" $pkgs
