#!/usr/bin/env bash
#
# conflicting-files.sh — name the paths that conflict when BASE is merged into HEAD (#1781).
#
#   $1  BASE committish (what would be merged IN, normally origin/main)
#   $2  HEAD committish (what it would be merged INTO, normally the delivery branch head)
#
# Prints one path per line, sorted, on stdout. Diagnostics go to stderr.
#
#   exit 0  determined — the printed list is complete. NO output means the merge is clean.
#   exit 3  could not determine — the caller must not read the absence of output as "clean".
#
# WHY THIS EXISTS. #1758's acceptance criterion is that a conflicting delivery PR is either
# auto-resolved or "explicitly flagged needs-human NAMING THE CONFLICT". On PR #1778 it stopped
# for a human with the reason "no DELIVER-VERDICT marker" and the conflict was never named at all
# — a human had to go and find it. Naming it needs the paths, and neither GitHub's
# `mergeable_state` (a single word) nor the gate (a pure function of its environment) can supply
# them. This is the one place that computes them.
#
# WHY A THROWAWAY WORKTREE, and not a merge in the caller's tree: the verify phase must be able to
# ask this question WITHOUT mutating its checkout (later steps in that job run archon against it),
# and this script must behave identically in both phases (R23) rather than being safe in one and
# destructive in the other. A `--detach` worktree under a temporary directory is trial-merged and
# then removed, so the caller's index, working tree and HEAD are untouched either way.
#
# WHY A TRIAL MERGE, and not `git merge-tree --write-tree`: the porcelain that reports conflicted
# paths directly requires git >= 2.38, and the runners here are on 2.34. A trial merge in a
# throwaway worktree works on every version this repository can encounter, and reads the paths out
# of the index (`git ls-files -u`), which covers add/add and delete/modify conflicts that a
# name-only diff can miss.
#
# BEST-EFFORT BY CONTRACT. This is a DIAGNOSTIC, never the authority on whether a branch
# conflicts: verify's authority is GitHub's `mergeable_state` and the correct phase's is its own
# real merge. So every failure mode here — a git too old, an object not fetched, a worktree that
# will not materialise — exits 3 with a warning rather than failing its caller. A delivery must
# not stop because the loop could not pretty-print a path list (R1: it says so, it does not
# pretend the merge was clean).

set -uo pipefail

base="${1-}"
head="${2-}"

if [[ -z "$base" || -z "$head" ]]; then
  echo "usage: $0 <base-committish> <head-committish>" >&2
  exit 3
fi

undetermined() {
  echo "conflicting-files.sh: could not determine the conflicting paths ($1)" >&2
  exit 3
}

# Both ends must actually be present locally. Resolving them first turns "the caller forgot to
# fetch" into a named diagnostic instead of a confusing worktree failure.
git rev-parse --verify --quiet "$base^{commit}" >/dev/null || undetermined "cannot resolve '$base'"
git rev-parse --verify --quiet "$head^{commit}" >/dev/null || undetermined "cannot resolve '$head'"

wt=$(mktemp -d "${TMPDIR:-/tmp}/conflicting-files.XXXXXX") || undetermined "cannot create a temporary directory"
# mktemp -d created it; `git worktree add` insists on a path that does not exist yet.
rmdir "$wt" 2>/dev/null || undetermined "cannot clear the temporary worktree path"

cleanup() {
  git worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
  git worktree prune >/dev/null 2>&1 || true
}
trap cleanup EXIT

git worktree add --detach "$wt" "$head" >/dev/null 2>&1 \
  || undetermined "cannot materialise '$head' into a temporary worktree"

# --no-commit --no-ff: we want the merge ATTEMPTED and its result left in the index, never
# recorded. No committer identity is configured in this worktree and none is needed, which is
# also why --no-commit matters beyond tidiness.
if git -C "$wt" merge --no-commit --no-ff "$base" >/dev/null 2>&1; then
  # Clean merge: determined, and the list is empty.
  git -C "$wt" merge --abort >/dev/null 2>&1 || true
  exit 0
fi

# `git ls-files -u` prints "<mode> <object> <stage>\t<path>" per unmerged index entry, so the
# path is the second TAB-separated field. Deduplicated because a single conflicted path has one
# entry per stage.
paths=$(git -C "$wt" ls-files -u | cut -f2 | sort -u)
git -C "$wt" merge --abort >/dev/null 2>&1 || true

# A non-zero `git merge` with NO unmerged entries is not a conflict we can describe — the merge
# refused for some other reason (a bad committish, an unrelated-histories refusal). Reporting an
# empty list as "determined, clean" would tell the caller the opposite of the truth, so it is
# undetermined instead.
[[ -n "$paths" ]] || undetermined "the trial merge of '$base' into '$head' failed without recording any conflicted path"

printf '%s\n' "$paths"
