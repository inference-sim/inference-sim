#!/usr/bin/env bash
#
# deliver-issue-refinements.sh — print the DESIGN REFINEMENTS made in an issue's comment thread
# that carry authority over its body.
#
# Usage:
#   scripts/deliver-issue-refinements.sh <issue-number>
#   scripts/deliver-issue-refinements.sh --render <payload-file>
#
# The first form reads the thread from GitHub. The second renders a payload that has already been
# assembled (`{"comments": [ … , {"writeAccess": true|false} ]}`) and makes no network call — it is
# how scripts/deliver_issue_refinements_test.go exercises the selection law, and it is also the
# form to reach for when reproducing a decision offline.
#
# WHY THIS EXISTS (#1782). An implement phase — automated or human — that reads an issue's BODY and
# nothing else faithfully builds an out-of-date spec whenever the design was refined in the comment
# thread after the body was written. That is not a hypothetical: issue #1706's body proposes
# "extend the block commit past endIndex", and a comment on it later replaces that with the
# vLLM-faithful shape ("fold the external credit in BEFORE the chunk/budget clamps") — which is what
# was actually built. A delivery reading the body alone would have implemented the superseded plan
# and the divergence would only have surfaced in review, an agent hour later.
#
# ── The authority rule this script implements ─────────────────────────────────────────────────
#
# It answers exactly one question — WHICH COMMENTS COUNT — and deliberately not the second one,
# what to do when one contradicts the body. That second half is judgement and lives in prose
# (docs/contributing/pr-workflow.md Step 1.5, docs/contributing/automated-delivery.md), because a
# script cannot reconcile two English sentences about scope. What a script CAN do, and what is
# worth making mechanical, is the trust boundary: this repository is public, so anyone can comment
# on any issue, and an agent that treats every comment as spec is steerable by anyone.
#
# A comment counts iff its author holds ADMIN, WRITE or MAINTAIN permission on this repository —
# the same boundary `/approve-issue-for-pr-delivery` and the review triggers already use. Resolved
# per distinct author against `repos/{owner}/{repo}/collaborators/{login}/permission`.
#
# `authorAssociation` is NOT used for this, and the reason is a measurement rather than a
# preference: on #1782 the maintainer who issues every delivery command reports
# `authorAssociation: CONTRIBUTOR`, because GitHub reports "has had a PR merged" in preference to
# collaborator status. Trusting OWNER/MEMBER/COLLABORATOR would have dropped exactly the comments
# this exists to read, while still admitting anyone whose PR has ever been merged.
#
# ── Failure is loud, never silent (R1) ───────────────────────────────────────────────────────
#
# When the thread could not be WEIGHED AT ALL this prints a digest whose FIRST LINE is
# `REFINEMENT-READ-FAILED`, explains what went wrong, and exits 3. The one outcome that must never
# occur is empty output, which reads exactly like "this issue has no refinements" — the failure mode
# the whole script exists to end. That covers a missing/broken jq or filter, an unreadable payload, a
# failed `gh issue view`, a failed author extraction, and the case where there was at least one
# author to weigh and NOT ONE of their permissions could be established.
#
# PARTIAL permission resolution deliberately does NOT degrade. If some authors resolve and some do
# not, the unresolved ones are dropped — each named on stderr — and the run continues at exit 0.
# Degrading on any single lookup failure would let one deleted account or one renamed login block a
# delivery, which is worse than dropping a comment that is probably not a refinement. The reasoning
# is recorded under Consequences in docs/contributing/issue-comment-authority.md.
#
# Exit codes: 0 = read (with or without refinements, and possibly with some authors dropped),
# 2 = usage, 3 = degraded (marker printed).
#
# `set -e` is off deliberately, matching the sibling scripts: a legitimate "no match" is a non-zero
# exit from grep/jq and is expected control flow here.

set -uo pipefail

SELF="$0"
FILTER="$(cd "$(dirname "$SELF")" && pwd)/deliver-issue-refinements.jq"

# Printed on stdout so it reaches whoever reads the digest, with the cause on stderr too so it
# reaches a workflow log. Both, deliberately: a reader who only sees one of the two channels must
# still learn that a refinement may have been missed.
degrade() {
  printf 'REFINEMENT-READ-FAILED\n\n'
  printf 'The issue comment thread could not be read: %s\n\n' "$1"
  printf 'Work from the issue body alone, and SAY SO — a reader must learn that a design\n'
  printf 'refinement made in a comment may have been missed, rather than assume none existed.\n'
  echo "$SELF: $1" >&2
  exit 3
}

usage() {
  echo "usage: $SELF <issue-number> | $SELF --render <payload-file>" >&2
  exit 2
}

if [[ ! -r "$FILTER" ]]; then
  degrade "the selection filter $FILTER is missing"
fi
for tool in jq; do
  command -v "$tool" >/dev/null 2>&1 || degrade "$tool is not on PATH"
done
# Proved USABLE, not merely present: `set -e` is off, so a jq that is installed but broken would
# otherwise turn into empty output at exit 0 — the silent-empty class deliver-seed-refs.sh
# documents at length.
printf '{}' | jq -e . >/dev/null 2>&1 || degrade "jq is present but not usable"

render() {
  local payload="$1" out
  if ! out=$(jq -r -f "$FILTER" < "$payload"); then
    degrade "the selection filter failed on $payload"
  fi
  if [[ -z "${out//[[:space:]]/}" ]]; then
    echo "No design refinements were made in the comment thread."
    echo "The issue body is the whole specification."
  else
    printf '%s\n' "$out"
  fi
  exit 0
}

case "${1:-}" in
  --render)
    [[ $# -eq 2 ]] || usage
    [[ -r "$2" ]] || degrade "cannot read the payload file $2"
    render "$2"
    ;;
  '' | -*)
    usage
    ;;
esac

[[ $# -eq 1 ]] || usage
ISSUE="$1"
[[ "$ISSUE" =~ ^[0-9]+$ ]] || usage

command -v gh >/dev/null 2>&1 || degrade "gh is not on PATH"

# Bound every GitHub call so a stalled request cannot hang the whole delivery job until its
# 60/120-minute timeout. The permission lookup below is the likeliest culprit, but `gh issue view`
# carries the same risk, so the deadline wraps all of them here rather than at one call site.
# `timeout` (coreutils — present on the Linux CI and self-hosted runners that actually run
# deliveries) or `gtimeout` (macOS with coreutils) enforces it where present; where neither is
# installed the portable `run_bounded` watchdog below enforces the same deadline, so NO execution
# path runs `gh` unbounded. A deadline expiry exits non-zero, so resolve_permission and the
# `gh issue view` check treat it exactly like any other "could not ask" failure — fail-closed, never
# mistaken for a definitive answer. GH_DEADLINE_SECONDS is overridable so the tests can force a
# short deadline.
GH_DEADLINE_SECONDS="${GH_DEADLINE_SECONDS:-30}"
_GH_BIN="$(command -v gh)"
if command -v timeout >/dev/null 2>&1; then
  _GH_TIMEOUT="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  _GH_TIMEOUT="gtimeout"
else
  _GH_TIMEOUT=""
fi

# Portable fallback deadline, used when neither `timeout` nor `gtimeout` is installed, so there is NO
# execution path on which a stalled `gh` call runs unbounded — not even on a host without coreutils.
# A watchdog subshell kills the call after the deadline. Its stdout/stderr go to /dev/null so it can
# never hold the command-substitution pipe open: were it to, `out=$(gh …)` would block on the
# watchdog's own sleep instead of returning when the call does, defeating the bound.
run_bounded() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  ( sleep "$secs"; kill -TERM "$pid" 2>/dev/null ) >/dev/null 2>&1 &
  local watcher=$!
  wait "$pid" 2>/dev/null
  local rc=$?
  kill -TERM "$watcher" 2>/dev/null
  wait "$watcher" 2>/dev/null
  return "$rc"
}

gh() {
  if [[ -n "$_GH_TIMEOUT" ]]; then
    "$_GH_TIMEOUT" "$GH_DEADLINE_SECONDS" "$_GH_BIN" "$@"
  else
    run_bounded "$GH_DEADLINE_SECONDS" "$_GH_BIN" "$@"
  fi
}

REPO="${GH_REPO:-${GITHUB_REPOSITORY:-}}"
if [[ -z "$REPO" ]]; then
  REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner) \
    || degrade "could not determine the repository (set GH_REPO or GITHUB_REPOSITORY)"
fi

TMP=$(mktemp -d) || degrade "could not create a temporary directory"
trap 'rm -rf "$TMP"' EXIT

# `gh issue view --json comments` paginates internally and already carries every field the filter
# reads except write access: id, author.login, authorAssociation, body, createdAt, url, isMinimized.
if ! gh issue view "$ISSUE" --repo "$REPO" --json comments > "$TMP/raw.json"; then
  degrade "gh issue view $ISSUE failed"
fi

# Bot logins are skipped BEFORE the permission lookup, not after: the delivery loop's own bot has
# write access, so asking would return `true` and the filter's bot term would then be the only
# thing keeping the loop's own refusal comments out of its next agent's spec.
if ! LOGINS=$(jq -r '
      [ .comments[]?.author.login // "" ]
      | map(select(. != "" and ((. | test("\\[bot\\]$")) | not)))
      | unique | .[]' "$TMP/raw.json"); then
  degrade "could not list comment authors for #$ISSUE"
fi

# Prints the author's permission and returns 0 when the answer is DEFINITIVE; returns 1 when the
# lookup could not be made at all.
#
# The distinction is the difference between two outcomes that must not be conflated. A 404 is a real
# answer — GitHub says this login is not a collaborator (it is also what a non-user login such as
# `github-actions` returns) — whereas a 401/403/5xx/network failure, or a deadline expiry from the
# bounded `gh` wrapper above, means the caller could not ask.
# Verified against this repository: a genuine non-collaborator returns 200 with `read`, so "no write
# access" normally arrives as a successful lookup and a failure really is a failure.
#
# Why it matters: `GET /repos/{owner}/{repo}/collaborators/{login}/permission` needs PUSH access, so
# a contributor with read-only access running this script gets a failure for EVERY author. Without
# this split that reads as "nobody has write access" and the digest reports "no design refinements" —
# reintroducing, one level down, the exact silent failure #1782 is about.
resolve_permission() {
  local login="$1" out
  if out=$(gh api "repos/$REPO/collaborators/$login/permission" --jq '.permission' 2>"$TMP/err"); then
    printf '%s' "$out"
    return 0
  fi
  if grep -qi 'HTTP 404' "$TMP/err"; then
    printf 'none'
    return 0
  fi
  printf '%s' "$(tr '\n' ' ' < "$TMP/err")"
  return 1
}

ACCESS='{}'
attempted=0
resolved=0
while IFS= read -r login; do
  [[ -n "$login" ]] || continue
  attempted=$((attempted + 1))
  ok=false
  if perm=$(resolve_permission "$login"); then
    resolved=$((resolved + 1))
    case "$perm" in
      admin | write | maintain) ok=true ;;
      *) echo "$SELF: @$login has no write access (permission='$perm') — their comments carry no authority" >&2 ;;
    esac
  else
    # Unresolved, so the comment is dropped: under-trusting costs a missed refinement, over-trusting
    # hands the spec to an unverified author. Named on stderr so it is not invisible.
    echo "$SELF: could not establish @$login's repository permission ($perm) — their comments are being dropped" >&2
  fi
  ACCESS=$(jq -c --arg l "$login" --argjson v "$ok" '. + {($l): $v}' <<< "$ACCESS") \
    || degrade "could not record write access for @$login"
done <<< "$LOGINS"

# Not one author's authority could be established, and there was at least one to establish. Reporting
# "no refinements" here would be a lie of exactly the kind this script exists to end — the thread was
# read, but nothing in it could be weighed. The likeliest cause is a caller without push access,
# which the permission endpoint requires.
if [[ "$attempted" -gt 0 && "$resolved" -eq 0 ]]; then
  degrade "none of the $attempted comment author(s)' repository permissions could be established (the permission endpoint requires push access)"
fi

if ! jq --argjson access "$ACCESS" \
      '.comments |= ((. // []) | map(. + {writeAccess: ($access[.author.login // ""] == true)}))' \
      "$TMP/raw.json" > "$TMP/payload.json"; then
  degrade "could not attach write access to the comments of #$ISSUE"
fi

render "$TMP/payload.json"
