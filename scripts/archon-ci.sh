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

if ! command -v python3 &>/dev/null; then
  echo "Error: python3 is required but not found on PATH." >&2
  exit 1
fi

ARCHON="${ARCHON_BIN:-archon-go}"

if [[ ! -x "$ARCHON" ]]; then
  echo "Error: archon-go binary not found or not executable at '$ARCHON'." >&2
  echo "Set ARCHON_BIN to the path of the archon-go binary." >&2
  exit 1
fi

if [[ ! -f "$REPO_DIR/go.mod" ]]; then
  echo "## Archon Error"
  echo ""
  echo "No \`go.mod\` found in \`$REPO_DIR\`. Archon requires a Go module."
  exit 1
fi

GO_MODULE=$(awk '/^module /{print $2; exit}' "$REPO_DIR/go.mod")
if [[ -z "$GO_MODULE" ]]; then
  echo "## Archon Error"
  echo ""
  echo "Could not parse module path from \`go.mod\`."
  exit 1
fi

# Step 1: Run delta (JSON) to determine if architecture changed.
# Capture stdout only — archon may print warnings to stderr (e.g. partial extraction)
# which would corrupt the JSON if merged. Stderr goes to the GHA log.
ARCHON_EXIT=0
DELTA_JSON=$("$ARCHON" delta --json "$REPO_DIR" "$BASE_SHA" "$HEAD_SHA") || ARCHON_EXIT=$?
if [[ $ARCHON_EXIT -ne 0 ]]; then
  echo "## Archon Error"
  echo ""
  echo "Failed to compute architectural delta (archon-go exited $ARCHON_EXIT)."
  echo "Check the workflow log for details."
  exit 1
fi

EMPTY=$(echo "$DELTA_JSON" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print('true' if d.get('emptyAtPackageAltitude', False) else 'false')
except (json.JSONDecodeError, AttributeError) as e:
    print(f'Error parsing delta JSON: {e}', file=sys.stderr)
    sys.exit(1)
") || {
  echo "## Archon Error"
  echo ""
  echo "Failed to parse architectural delta JSON."
  exit 1
}

if [[ "$EMPTY" == "true" ]]; then
  echo "## Archon Architectural Review"
  echo ""
  echo "**No architectural change detected.** Internal-only PR — fast-track eligible."
  echo ""
  # Still report invariant changes if any.
  INVARIANTS=$(echo "$DELTA_JSON" | python3 -c "
import json, sys
try:
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
except Exception as e:
    print(f'### Invariants\n\n(Failed to parse invariant data: {e})')
" 2>&1) || true
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
"$ARCHON" delta "$REPO_DIR" "$BASE_SHA" "$HEAD_SHA" 2>&1 || echo "(delta render failed)"
echo '```'
echo ""

# Extract changed internal packages for impact analysis.
# Only include packages that exist at HEAD (skip removed packages — impact can't resolve them).
CHANGED_PKGS=$(echo "$DELTA_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
mod = sys.argv[1]
pkgs = set()
removed = set()
for e in d.get('edgesAdded', []) + d.get('edgesRemoved', []):
    for k in ('from', 'to'):
        if e[k].startswith(mod):
            pkgs.add(e[k])
for s in d.get('surface', []):
    if s['package'].startswith(mod):
        pkgs.add(s['package'])
for p in d.get('packagesAdded', []):
    path = p.get('path', '') if isinstance(p, dict) else p
    if path.startswith(mod):
        pkgs.add(path)
for p in d.get('packagesRemoved', []):
    path = p.get('path', '') if isinstance(p, dict) else p
    if path.startswith(mod):
        removed.add(path)
for p in sorted(pkgs - removed):
    print(p)
" "$GO_MODULE") || CHANGED_PKGS=""

# Blast radius per changed package.
if [[ -n "$CHANGED_PKGS" ]]; then
  echo "### Blast Radius"
  echo ""
  echo '```'
  while IFS= read -r pkg; do
    "$ARCHON" impact "$REPO_DIR" "$pkg" "$HEAD_SHA" 2>&1 || echo "(impact analysis failed for $pkg)"
    echo ""
  done <<< "$CHANGED_PKGS"
  echo '```'
  echo ""
fi

# Contract evidence.
echo "### Contract Evidence"
echo ""
echo '```'
"$ARCHON" evidence "$REPO_DIR" "$HEAD_SHA" 2>&1 || echo "(evidence analysis failed)"
echo '```'
