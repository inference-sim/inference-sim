#!/usr/bin/env bash
#
# deliver-conflict-check.sh — did a correction round actually resolve the conflict? (#1781)
#
#   $1  the delivery branch name (deliver/issue-<N>)
#
# Prints two lines and exits 0:
#   state=clean|conflicting|unknown
#   files=<comma-separated conflicting paths; empty unless state=conflicting>
#
#   clean        the branch merges into main with no conflict
#   conflicting  it still conflicts, and `files` names the paths a human must resolve
#   unknown      it could not be determined (origin unreachable, or git could not describe the
#                merge). NOT an escalation: the caller falls back to GitHub's own mergeable_state.
#
# Exit 2 only on a usage error, for the same reason as deliver-update-branch.sh: this runs at the
# END of a correction round, and failing the phase would report "the correct phase errored"
# instead of the thing the round exists to establish.
#
# WHY IT EXISTS. #1758's acceptance criterion is that a conflicting delivery PR is either
# (a) auto-updated and re-verified to a real verdict, or (b) explicitly flagged `needs-human`
# NAMING THE CONFLICT. On PR #1778 neither happened: the correction round no-op'd, posted nothing,
# handed back, and the next verify stopped for a human citing a missing review marker. Nothing
# named the conflict, and a human had to go and find it. This is the check that makes (b) hold
# WITHOUT depending on the agent having done anything at all — the caller posts the naming comment
# from `files` and withholds the hand-back on `conflicting`.
#
# WHY IT READS THE REMOTE REFS. The correction agent pushes from inside claude-code-action, so the
# workflow's own checkout may be behind whatever the branch now points at. Both ends are fetched
# fresh into their remote-tracking refs before the question is asked.
#
# WHY IT ONLY ESCALATES ON POSITIVE EVIDENCE. conflicting-files.sh is best-effort by contract
# (exit 3 = could not determine), and treating "could not tell" as a conflict would stop healthy
# deliveries at needs-human. So `unknown` hands the decision on to the phase that reads GitHub's
# mergeable_state, and only a git-confirmed conflict stops the loop here.

set -uo pipefail

branch="${1-}"
if [[ -z "$branch" ]]; then
  echo "usage: $0 <delivery-branch>" >&2
  exit 2
fi

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

emit() {
  printf 'state=%s\nfiles=%s\n' "$1" "${2-}"
  exit 0
}

# Explicit refspecs rather than `git fetch origin main "$branch"`: with two refs FETCH_HEAD holds
# two lines and `git rev-parse FETCH_HEAD` silently resolves the FIRST — main — so the check would
# compare main against itself and always look clean. `+` forces the update in case of a force-push.
if ! git fetch --no-tags --quiet origin \
       "+refs/heads/main:refs/remotes/origin/main" \
       "+refs/heads/$branch:refs/remotes/origin/$branch"; then
  echo "::warning::could not fetch origin/main and origin/$branch; cannot tell whether the branch still conflicts" >&2
  emit unknown
fi

if ! files=$("$here/conflicting-files.sh" "origin/main" "origin/$branch"); then
  echo "::warning::could not determine whether origin/$branch still conflicts with main" >&2
  emit unknown
fi

if [[ -z "$files" ]]; then
  echo "origin/$branch merges cleanly into main" >&2
  emit clean
fi

# One line, comma-joined: the value is consumed as a workflow step output and pasted into a PR
# comment, and an embedded newline would corrupt both.
files=$(tr '\n' ',' <<< "$files" | sed 's/,$//')
echo "origin/$branch still conflicts with main in: $files" >&2
emit conflicting "$files"
