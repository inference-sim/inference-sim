#!/usr/bin/env bash
#
# restrict-write.sh — Claude Code PreToolUse hook for the blis external review.
#
# The reviewer is granted a bare Write tool (a path-qualified Write allow-rule
# refuses silently in the action), so THIS hook is the real boundary: it permits
# a write ONLY to $PR_REVIEW_VERDICT_FILE and refuses every other path. Exit 0
# allows; exit 2 blocks (Claude Code treats a PreToolUse exit 2 as "deny" and
# feeds stderr back to the model).
#
# Fails closed: an unreadable payload, an empty allowed path, or any mismatch is
# a refusal.
set -uo pipefail

allowed="${PR_REVIEW_VERDICT_FILE:-}"
input=$(cat)

path=$(printf '%s' "$input" \
  | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print((d.get("tool_input") or {}).get("file_path",""))
except Exception:
    print("")' 2>/dev/null || echo "")

if [[ -z "$allowed" ]]; then
  echo "restrict-write: PR_REVIEW_VERDICT_FILE is unset; refusing all writes" >&2
  exit 2
fi
if [[ "$path" == "$allowed" ]]; then
  exit 0
fi
echo "restrict-write: write to '$path' refused; this review may only write $allowed" >&2
exit 2
