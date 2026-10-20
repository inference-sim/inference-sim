#!/bin/bash
# corpora.sh: the two aiperf trace corpora the agentic InferenceX runs replay, at pinned HF commits, into
# $IX_CORPORA/<name>/traces.jsonl (default $IX_DATA_DIR/corpora), verified against the pinned sha256
# (the HF LFS oid). The corpora are UNTRUSTED JSON lines, read by BLIS as data.
set -euo pipefail
DATA=${IX_DATA_DIR:-./ix-data}; O=${IX_CORPORA:-$DATA/corpora}
while read -r name repo rev sha; do
  d=$O/$name; mkdir -p "$d"; f=$d/traces.jsonl
  if ! { [ -s "$f" ] && [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" = "$sha" ]; }; then
    curl -fL --retry 3 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} -o "$f.part" \
      "https://huggingface.co/datasets/$repo/resolve/$rev/traces.jsonl"
    got=$(shasum -a 256 "$f.part" | cut -d' ' -f1)
    [ "$got" = "$sha" ] || { echo "sha256 mismatch for $name: $got" >&2; rm -f "$f.part"; exit 1; }
    mv "$f.part" "$f"
  fi
  echo "$sha  traces.jsonl" > "$d/traces.sha256"
done <<'PINS'
weka-062126 semianalysisai/cc-traces-weka-062126 23f152f6f0f9399a85901b89a6458def0ef16729 29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276
weka-256k semianalysisai/cc-traces-weka-062126-256k 8fecd2fc56694469f758f0afbbb6335ad3043740 e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef
PINS
