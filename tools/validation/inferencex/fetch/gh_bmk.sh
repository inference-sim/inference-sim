#!/bin/bash
# gh_bmk.sh: the post-dump inputs of build_inventory.py --bmk-dir. Lists SemiAnalysisAI/InferenceX "Run Sweep"
# runs (workflow 214438218) created 2026-09-27..2026-10-16 with conclusion success or null, downloads each
# bmk_* artifact zip, and extracts its one JSON summary. Writes $IX_DATA_DIR/gh-bmk-postdump/{index.tsv,json/,zips/}.
# index.tsv: run_id, created date, artifact id, name, sha256(zip). Needs `gh` logged in (the artifacts API
# requires auth even for public repos). Artifacts expire after the repo's retention period: an expired one
# cannot be fetched, and the inventory then differs (check index.tsv against the pinned list below).
# Downloads are UNTRUSTED: only `unzip -p` of a *.json member, never executed.
set -uo pipefail
export LC_ALL=C
DATA=${IX_DATA_DIR:-./ix-data}; D=$DATA/gh-bmk-postdump; mkdir -p "$D/zips" "$D/json"
R=SemiAnalysisAI/InferenceX
runs=$D/runsweep.tsv; : > "$runs"
for d in 2026-09-{27..30} 2026-10-0{1..9} 2026-10-{10..16}; do
  for p in $(seq 1 10); do
    r=$(gh api "repos/$R/actions/workflows/214438218/runs?per_page=100&page=$p&created=$d" \
        --jq '.workflow_runs[]|"\(.id)\t\(.created_at[:10])\t\(.conclusion)"' 2>/dev/null) || break
    [ -z "$r" ] && break; echo "$r" >> "$runs"; [ "$(echo "$r" | wc -l)" -lt 100 ] && break
  done
done
arts=$D/artifacts.tsv; : > "$arts"
awk -F'\t' '$3=="success" || $3=="null"' "$runs" | while IFS=$'\t' read -r run d _; do
  gh api "repos/$R/actions/runs/$run/artifacts?per_page=100" \
    --jq ".artifacts[]|select(.name|test(\"^bmk_\"))|\"$run\t$d\t\(.id)\t\(.name)\t\(.expired)\"" >> "$arts" 2>/dev/null
done
while IFS=$'\t' read -r run d id name exp; do
  case $id in *[!0-9]*) continue;; esac
  [ "$exp" = true ] && { echo "expired $id $name" >&2; continue; }
  [ -s "$D/json/$id.json" ] && continue
  gh api "repos/$R/actions/artifacts/$id/zip" > "$D/zips/$id.zip" 2>/dev/null || { rm -f "$D/zips/$id.zip"; continue; }
  f=$(unzip -Z1 "$D/zips/$id.zip" 2>/dev/null | grep -m1 '^[^/]*\.json$')
  [ -n "$f" ] && unzip -p "$D/zips/$id.zip" "$f" > "$D/json/$id.json"
  printf '%s\t%s\t%s\t%s\t%s\n' "$run" "$d" "$id" "$name" "$(shasum -a 256 "$D/zips/$id.zip" | cut -c1-64)" >> "$D/index.tsv"
done < "$arts"
ls "$D/json" | wc -l
