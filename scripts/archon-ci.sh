#!/usr/bin/env bash
#
# archon-ci.sh — Run Archon architectural analysis on a PR.
#
# Usage: scripts/archon-ci.sh <repo-dir> <base-sha> <head-sha>
#
# Outputs formatted markdown to stdout. Exit 0 on success, 1 on error.
# The caller (GitHub Action) is responsible for posting the output as a PR comment.

set -euo pipefail

REPO_DIR="${1:-}"
BASE_SHA="${2:-}"
HEAD_SHA="${3:-}"

if [[ -z "$REPO_DIR" || -z "$BASE_SHA" || -z "$HEAD_SHA" ]]; then
  echo "Usage: archon-ci.sh <repo-dir> <base-sha> <head-sha>" >&2
  exit 1
fi

ARCHON="${ARCHON_BIN:-archon-go}"

if ! command -v "$ARCHON" &>/dev/null; then
  echo "Error: archon-go binary not found. Set ARCHON_BIN or add to PATH." >&2
  exit 1
fi

# Step 1: Run delta (JSON) to determine if architecture changed.
DELTA_JSON=$("$ARCHON" delta --json "$REPO_DIR" "$BASE_SHA" "$HEAD_SHA" 2>&1) || {
  echo "## Archon Error"
  echo ""
  echo "Failed to compute architectural delta:"
  echo '```'
  echo "$DELTA_JSON"
  echo '```'
  exit 1
}

EMPTY=$(echo "$DELTA_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print('true' if d.get('emptyAtPackageAltitude', False) else 'false')")

if [[ "$EMPTY" == "true" ]]; then
  echo "## Archon Architectural Review"
  echo ""
  echo "**No architectural change detected.** Internal-only PR — fast-track eligible."
  echo ""
  # Still report invariant changes if any.
  INVARIANTS=$(echo "$DELTA_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
invs = d.get('invariants', [])
if not invs:
    sys.exit(0)
print('### Invariants Touched')
print('')
for inv in invs:
    pkg = inv['package'].split('/')[-1]
    for a in inv.get('added', []):
        print(f'- + {pkg}.{a}')
    for r in inv.get('removed', []):
        print(f'- - {pkg}.{r}')
    for m in inv.get('modified', []):
        print(f'- ~ {pkg}.{m}')
" 2>/dev/null || true)
  if [[ -n "$INVARIANTS" ]]; then
    echo "$INVARIANTS"
  fi
  exit 0
fi

# Step 2: Non-empty delta. Gather full analysis.
echo "## Archon Architectural Review"
echo ""

# Human-readable delta.
echo "### Architectural Delta"
echo ""
echo '```'
"$ARCHON" delta "$REPO_DIR" "$BASE_SHA" "$HEAD_SHA" 2>/dev/null || echo "(delta render failed)"
echo '```'
echo ""

# Get the Go module path from go.mod.
GO_MODULE=$(head -1 "$REPO_DIR/go.mod" | awk '{print $2}')

# Extract changed internal packages for impact analysis.
CHANGED_PKGS=$(echo "$DELTA_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
mod = sys.argv[1]
pkgs = set()
for e in d.get('edgesAdded', []) + d.get('edgesRemoved', []):
    for k in ('from', 'to'):
        if e[k].startswith(mod):
            pkgs.add(e[k])
for s in d.get('surface', []):
    if s['package'].startswith(mod):
        pkgs.add(s['package'])
for b in d.get('boxesAdded', []):
    if b.startswith(mod):
        pkgs.add(b)
for b in d.get('boxesRemoved', []):
    if b.startswith(mod):
        pkgs.add(b)
for p in sorted(pkgs):
    print(p)
" "$GO_MODULE" 2>/dev/null)

# Blast radius per changed package.
if [[ -n "$CHANGED_PKGS" ]]; then
  echo "### Blast Radius"
  echo ""
  echo '```'
  while IFS= read -r pkg; do
    "$ARCHON" impact "$REPO_DIR" "$pkg" "$HEAD_SHA" 2>/dev/null || true
    echo ""
  done <<< "$CHANGED_PKGS"
  echo '```'
  echo ""
fi

# Contract evidence.
echo "### Contract Evidence"
echo ""
echo '```'
"$ARCHON" evidence "$REPO_DIR" "$HEAD_SHA" 2>/dev/null || echo "(evidence analysis failed)"
echo '```'
