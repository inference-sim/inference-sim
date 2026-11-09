#!/bin/bash
# Reproduce the price-change report (docs/evidence/price-change-report.md).
#
#   scripts/kernel/price-report.sh OLD_REPO OUT_DIR
#
# OLD_REPO is a checkout of the v0.15.0 release of inference-sim (the last release that runs the old fixtures;
# `git worktree add OLD_REPO v0.15.0`; the script writes scripts/kernel/priceprobe into it, so use
# a throwaway worktree). REPORT_BASE is the commit whose testdata/kernel holds the old fixtures: the
# parent of the commit that regenerated them. OUT_DIR receives the probe outputs and the report.
set -euo pipefail
OLD_REPO=${1:?path to a checkout of the v0.15.0 release of inference-sim}
OUT=${2:?output directory}
BASE=${REPORT_BASE:?commit holding the old testdata/kernel (the parent of the regeneration)}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT/old-fixtures" "$OUT/attr"
SUB=$(cd "$HERE" && go list -m -f '{{.Dir}}' github.com/inference-sim/blis-substrate)
(cd "$HERE" && git archive "$BASE" testdata/kernel) | tar -x -C "$OUT/old-fixtures"
# v0.15.0's probe: the same probe.go, with repos.go naming directories (its adapter takes them).
mkdir -p "$OLD_REPO/scripts/kernel/priceprobe"
cp "$HERE/scripts/kernel/priceprobe/probe.go" "$OLD_REPO/scripts/kernel/priceprobe/probe.go"
cat > "$OLD_REPO/scripts/kernel/priceprobe/repos.go" <<'GO'
package main

import "github.com/inference-sim/blis-vllm/compat/kernelmodel"

func reposFor(scenarios, catalog, registry string) (kernelmodel.Repos, error) {
	return kernelmodel.Repos{Scenarios: scenarios, Catalog: catalog, Registry: registry}, nil
}
GO
(cd "$OLD_REPO" && go run ./scripts/kernel/priceprobe -scenarios "$OUT/old-fixtures/testdata/kernel" -catalog "$SUB/catalog" -registry "$SUB/registry") > "$OUT/before.jsonl"
(cd "$HERE" && go run ./scripts/kernel/priceprobe -scenarios testdata/kernel) > "$OUT/after.jsonl"
python3 -I "$HERE/scripts/kernel/price_change_report.py" "$OUT/before.jsonl" "$OUT/after.jsonl" "$OUT/old-fixtures/testdata/kernel" > "$OUT/report.md"
# The one-at-a-time attribution: five fixtures, each old setting stated back.
python3 -I "$HERE/scripts/kernel/attribution_variants.py" "$HERE/testdata/kernel/aisimulate" "$OUT/attr"
for d in "$OUT"/attr/*/; do
  n=$(basename "$d")
  (cd "$HERE" && go run ./scripts/kernel/priceprobe -scenarios "$d") > "$OUT/attr/probe.$n.jsonl"
done
python3 -I "$HERE/scripts/kernel/price_change_report.py" --attribute "$OUT/attr/probe" >> "$OUT/report.md"
echo "report: $OUT/report.md"
