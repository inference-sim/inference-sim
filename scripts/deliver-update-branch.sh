#!/usr/bin/env bash
#
# deliver-update-branch.sh — bring a delivery branch up to date with main, deterministically (#1781).
#
#   $1  the delivery branch name (deliver/issue-<N>), the ref this pushes to
#
# Prints two lines and exits 0:
#   state=merged|current|conflicting|unknown
#   files=<conflicting paths, NEWLINE-delimited (one per line) via GITHUB_OUTPUT's multiline form;
#          empty unless state=conflicting>
#
#   merged       main was merged cleanly and the merge commit was PUSHED
#   current      the branch was already up to date with main; nothing was pushed
#   conflicting  the merge conflicts; it was ABORTED and `files` names the paths
#   unknown      the update could not be completed (origin unreachable, or the push lost a
#                race); the caller must not assume the branch is either updated or conflicting
#
# Exit 2 only on a usage error. Every other outcome is a state, because this runs inside a
# delivery round whose job is to make progress, and failing the phase here would report
# "the correct phase errored" rather than what actually happened to the branch.
#
# WHY THIS IS A SCRIPT AND NOT INLINE WORKFLOW YAML — the same reason deliver-gate.sh and
# map-merge-state.sh are. On PR #1778 the branch update was a PROMPT INSTRUCTION to the
# correction agent ("run git merge origin/main"), and the round completed `success` having made
# no commit and posted no comment: whether the merge was never attempted, was abandoned, or its
# push was refused, the outcome was a success with no progress and no diagnostic — the silent
# stall #1758 set out to remove, one layer deeper. An instruction an agent may quietly not follow
# is not a mechanism. Ordinary drift is the common case and needs no judgment, so it belongs in
# code that runs every round and can be tested (scripts/deliver_update_branch_test.go); only a
# REAL content conflict needs an agent's read of intent.
#
# WHY MERGE AND NOT REBASE OR GITHUB'S UPDATE-BRANCH API: a merge preserves the delivery branch's
# existing commits (no force-push, which the loop's push-triggered re-verify and the agent's own
# checkout would fight), and it is the one form that can carry a CONFLICTING update through at
# all — the update-branch API only fast-forwards a clean branch.

set -uo pipefail

branch="${1-}"
if [[ -z "$branch" ]]; then
  echo "usage: $0 <delivery-branch>" >&2
  exit 2
fi

# The bot identity, supplied per-command with `-c` rather than written into the repository config:
# this runs in a persistent self-hosted workspace shared by every delivery, so a config write
# would outlive the round that made it. The merge commit is the automation's, not the agent's.
git_as_bot() {
  git -c user.name='github-actions[bot]' \
      -c user.email='41898282+github-actions[bot]@users.noreply.github.com' \
      -c commit.gpgsign=false "$@"
}

# `files` is emitted NEWLINE-delimited via GITHUB_OUTPUT's multiline form, not a single
# `files=<comma-joined>` line: a git path may legally contain spaces AND commas (git forbids only
# NUL), so no single-character join is lossless (#1781 G4). `state` stays a single line.
#
# The heredoc delimiter is RANDOM per call (GitHub's own recommendation for untrusted multiline
# output): a fixed delimiter that happened to equal a conflicting path's whole line would let that
# path close the block early and inject further GITHUB_OUTPUT keys. Randomising makes a collision
# impossible. The caller need not know the delimiter — GitHub parses the heredoc into the step
# output, and the consumers that DO parse it manually (the tests, deliver-correct.yml) read the
# delimiter off the `files<<` line rather than hardcoding it.
emit() {
  printf 'state=%s\n' "$1"
  local d
  d="BLIS_FILES_$(od -An -N8 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n')"
  [[ "$d" != "BLIS_FILES_" ]] || d="BLIS_FILES_${RANDOM}${RANDOM}${RANDOM}"
  printf 'files<<%s\n%s\n%s\n' "$d" "${2-}" "$d"
  exit 0
}

if ! git fetch --no-tags --quiet origin main; then
  echo "::warning::could not fetch origin/main; leaving the branch update to the agent" >&2
  emit unknown
fi

# G6 — refuse to touch a DIRTY worktree. The push-race path below runs `git reset --hard`, which
# would destroy uncommitted TRACKED changes; a merge against a dirty index can also fail
# confusingly. In the delivery workflow the checkout is always clean, but this script must be safe
# to invoke anywhere, so a dirty tracked worktree is reported as `unknown` (hand the update back to
# the agent, and to verify's own mergeable_state) rather than risking data loss. Untracked files
# are left alone deliberately — neither the merge nor the reset touches them, and a fresh checkout
# routinely carries build scratch that is not this script's to police.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "::warning::the worktree has uncommitted tracked changes; refusing to merge or reset and leaving the update to the agent" >&2
  emit unknown
fi

before=$(git rev-parse HEAD) || emit unknown

if git_as_bot merge --no-edit origin/main; then
  after=$(git rev-parse HEAD)
  if [[ "$after" == "$before" ]]; then
    echo "the branch is already up to date with main" >&2
    emit current
  fi
  # A GITHUB_TOKEN push starts no workflow run, so this cannot double-trigger the verify phase;
  # the correct phase's own hand-back dispatch is what re-verifies.
  if git push origin "HEAD:refs/heads/$branch"; then
    echo "merged origin/main cleanly and pushed $after" >&2
    emit merged
  fi
  # The push lost a race with the remote. RESET rather than leave the merge commit behind: the
  # agent's `gh pr checkout` would then meet a diverged local branch, and a confusing checkout
  # failure is a worse outcome than handing it an untouched tree to merge itself.
  echo "::warning::merged origin/main cleanly but could not push; resetting so the agent starts from the remote state and does the update itself" >&2
  git reset --hard "$before" >/dev/null 2>&1 || true
  emit unknown
fi

# A real conflict: record the paths, abort, and leave the resolution to the agent — the one
# participant that can weigh this PR's intent against main's.
#
# `git ls-files -u` prints "<mode> <object> <stage>\t<path>", so the path is the second
# TAB-separated field. Read from the INDEX rather than from a name-only diff, so add/add and
# delete/modify conflicts are named too. Kept NEWLINE-delimited (no comma-join) so a path
# containing a space or comma survives intact through emit's multiline output (#1781 G4).
files=$(git ls-files -u | cut -f2 | sort -u)
git merge --abort >/dev/null 2>&1 || true

# A failed merge that recorded no conflicted path is not a conflict we can describe (a refusal to
# merge unrelated histories, say). Reporting it as `conflicting` with an empty list would send a
# human looking for files git never named, so it is `unknown` — the caller then falls back to
# GitHub's own mergeable_state, which is the authority this only approximates.
if [[ -z "$files" ]]; then
  echo "::warning::merging origin/main failed but recorded no conflicted path; treating the update state as unknown" >&2
  emit unknown
fi

echo "conflicts with main in: $files" >&2
emit conflicting "$files"
