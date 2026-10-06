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

if [[ -z "$allowed" ]]; then
  echo "restrict-write: PR_REVIEW_VERDICT_FILE is unset; refusing all writes" >&2
  exit 2
fi

# Decide on the RESOLVED target, not its spelling: canonicalize both the write
# path and the allowed path (symlinks, `..`, relative) so a path alias cannot
# slip past a literal string compare. python3 is already required to parse the
# payload and is portable across the Linux runner and macOS tests (`realpath -m`
# is GNU-only). Prints OK iff the target resolves to the allowed verdict file;
# any parse error, missing path, or mismatch prints DENY (fail closed).
verdict=$(printf '%s' "$input" | ALLOWED="$allowed" python3 -c 'import sys, os, json
try:
    d = json.load(sys.stdin)
    p = (d.get("tool_input") or {}).get("file_path", "")
except Exception:
    p = ""
if not p:
    print("DENY")
elif os.path.realpath(p) == os.path.realpath(os.environ["ALLOWED"]):
    print("OK")
else:
    print("DENY")' 2>/dev/null || echo "DENY")

if [[ "$verdict" == "OK" ]]; then
  exit 0
fi
echo "restrict-write: write refused; this review may only write $allowed" >&2
exit 2
