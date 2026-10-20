#!/bin/bash
# recipes.sh <nolog_shas.txt>: for each pinned SemiAnalysisAI/InferenceX commit (one 40-hex sha per line;
# partition/nolog_shas.txt lists the 41 behind the g_nolog rows), write the launcher scripts and workflow/config
# files reconstruct.py reads into $IX_RECIPES/<sha>/ (default $IX_DATA_DIR/recipes). Files are written with
# `git show`, never sourced or run: they are UNTRUSTED shell text that reconstruct.py parses as text.
set -uo pipefail
DATA=${IX_DATA_DIR:-./ix-data}; O=${IX_RECIPES:-$DATA/recipes}; SRC=${IX_INFERENCEX_SRC:-$DATA/src/InferenceX}
mkdir -p "$O"; O=$(cd "$O" && pwd)
[ -d "$SRC/.git" ] || git clone -q --filter=blob:none --no-checkout https://github.com/SemiAnalysisAI/InferenceX "$SRC"
while read -r sha; do
  [[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "skip $sha" >&2; continue; }
  [ -s "$O/$sha.files" ] && continue
  git -C "$SRC" fetch -q origin "$sha" 2>/dev/null
  git -C "$SRC" ls-tree -r --name-only "$sha" -- benchmarks .github > "$O/$sha.files" || continue
  grep -E '^benchmarks/[^/]*(gptoss|70b|llama)[^/]*\.sh$|^\.github/workflows/.*\.yml$|^\.github/configs/.*\.ya?ml$' "$O/$sha.files" |
  while read -r f; do mkdir -p "$O/$sha/$(dirname "$f")"; git -C "$SRC" show "$sha:$f" > "$O/$sha/$f"; done
done < "$1"
ls "$O" | grep -vc files
