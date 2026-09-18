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
# On any failure this prints a digest whose FIRST LINE is `REFINEMENT-READ-FAILED`, explains what
# went wrong, and exits 3. The one outcome that must never occur is empty output, which reads
# exactly like "this issue has no refinements" — the failure mode the whole script exists to end.
#
# Exit codes: 0 = read (with or without refinements), 2 = usage, 3 = degraded (marker printed).
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

ACCESS='{}'
while IFS= read -r login; do
  [[ -n "$login" ]] || continue
  # A lookup that fails means "not a collaborator", which is the SAFE reading: the comment becomes
  # unauthoritative rather than authoritative. A transient API failure therefore under-trusts, and
  # the count printed below is what tells a reader it happened.
  perm=$(gh api "repos/$REPO/collaborators/$login/permission" --jq '.permission' 2>/dev/null) || perm=""
  case "$perm" in
    admin | write | maintain) ok=true ;;
    *) ok=false; echo "$SELF: @$login has no write access (permission='${perm:-unknown}') — their comments carry no authority" >&2 ;;
  esac
  ACCESS=$(jq -c --arg l "$login" --argjson v "$ok" '. + {($l): $v}' <<< "$ACCESS") \
    || degrade "could not record write access for @$login"
done <<< "$LOGINS"

if ! jq --argjson access "$ACCESS" \
      '.comments |= ((. // []) | map(. + {writeAccess: ($access[.author.login // ""] == true)}))' \
      "$TMP/raw.json" > "$TMP/payload.json"; then
  degrade "could not attach write access to the comments of #$ISSUE"
fi

render "$TMP/payload.json"
