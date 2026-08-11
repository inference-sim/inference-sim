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

# Step 1: Run delta (human-readable) for triage and reporting.
ARCHON_EXIT=0
DELTA_OUTPUT=$("$ARCHON" delta "$REPO_DIR" "$BASE_SHA" "$HEAD_SHA") || ARCHON_EXIT=$?
if [[ $ARCHON_EXIT -ne 0 ]]; then
  echo "## Archon Error"
  echo ""
  echo "Failed to compute architectural delta (archon-go exited $ARCHON_EXIT)."
  echo "Check the workflow log for details."
  exit 1
fi

# Step 2: Triage — fast-track only if structurally empty AND no schema/contract changes.
if echo "$DELTA_OUTPUT" | grep -q "empty at package altitude" \
   && ! echo "$DELTA_OUTPUT" | grep -q "SCHEMA CHANGED" \
   && ! echo "$DELTA_OUTPUT" | grep -q "CONTRACT COVERAGE"; then
  echo "## Archon Architectural Review"
  echo ""
  echo "**No architectural change detected.** Internal-only PR — fast-track eligible."
  echo ""
  # Show invariant/other info from the delta if present (after the first line).
  REST=$(echo "$DELTA_OUTPUT" | tail -n +3)
  if [[ -n "$REST" ]]; then
    echo '```'
    echo "$REST"
    echo '```'
  fi
  exit 0
fi

# Step 3: Non-empty delta. Gather full analysis.
echo "## Archon Architectural Review"
echo ""

echo "### Architectural Delta"
echo ""
echo '```'
echo "$DELTA_OUTPUT"
echo '```'
echo ""

# Extract changed internal packages for blast radius.
GO_MODULE=$(awk '/^module /{print $2; exit}' "$REPO_DIR/go.mod")
if [[ -n "$GO_MODULE" ]] && command -v python3 &>/dev/null; then
  DELTA_JSON=$("$ARCHON" delta --json "$REPO_DIR" "$BASE_SHA" "$HEAD_SHA") || DELTA_JSON=""
  if [[ -z "$DELTA_JSON" ]]; then
    echo "### Blast Radius"
    echo ""
    echo "_Unavailable — delta JSON extraction failed. See workflow log._"
    echo ""
  elif [[ -n "$DELTA_JSON" ]]; then
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
for s in d.get('schema', []):
    if s.get('package', '').startswith(mod):
        pkgs.add(s['package'])
for c in d.get('contracts', []):
    iface = c.get('interface', '')
    pkg = iface.rsplit('.', 1)[0]
    if pkg.startswith(mod):
        pkgs.add(pkg)
for p in sorted(pkgs - removed):
    print(p)
" "$GO_MODULE" 2>/dev/null) || CHANGED_PKGS=""

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
  fi
fi

# Contract evidence.
echo "### Contract Evidence"
echo ""
echo '```'
"$ARCHON" evidence "$REPO_DIR" "$HEAD_SHA" 2>&1 || echo "(evidence analysis failed)"
echo '```'
