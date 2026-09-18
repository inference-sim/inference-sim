#!/usr/bin/env bash
#
# deliver-gate.sh — decide what happens next in an L1 delivery round.
#
# Environment (all required):
#   CI_STATUS      success | failure | unknown
#   PLAN_GATE      pass | regression | conflicts | unverified | absent
#   AGENT_VERDICT  GREEN | NOT-GREEN | MISSING
#   QA_VERDICT     PASS | BLOCK | MISSING  (the cross-vendor qa-review pass, #1715 — a second
#                                           reviewer from a different model family than the
#                                           implementer and AGENT_VERDICT's reviewer)
#   DISMISSALS     none | open | unknown   (open = a correction dismissed a finding that the
#                                           reviewer has not accepted; unknown = unreadable)
#   MERGE_STATE    mergeable | conflicting | unknown
#                                          (conflicting = the branch has merge conflicts with
#                                           main, REST mergeable_state "dirty"; unknown = the
#                                           mergeability could not be determined. The caller maps
#                                           GitHub's mergeable_state into this domain — "behind"
#                                           and every other non-dirty state map to mergeable,
#                                           since only a true conflict blocks the merge, #1758.)
#   ROUND          correction rounds already spent (non-negative integer)
#   MAX_ROUNDS     hard cap on correction rounds (non-negative integer)
#
# Optional:
#   CONFLICT_FILES the paths that conflict with main, whitespace- or comma-separated, for the
#                  reason string only (#1781). Read ONLY when MERGE_STATE is conflicting, and
#                  absent by default: a caller that cannot compute the list still gets a reason
#                  that names the conflict, just not the files. Supplied by
#                  scripts/conflicting-files.sh, which is best-effort by design.
#
# Prints two lines and exits 0:
#   decision=ready|correct|needs-human|recheck
#   reason=<one line, safe to paste into a PR comment>
#
# `recheck` is the one NON-TERMINAL decision: every signal is green but the branch's
# mergeability could not be read this run (a transient API blip, #1758 G1). It is not `ready`
# (an unverified mergeability must not merge) and not the terminal `needs-human` (a re-checkable
# blip must not stop the delivery for a human). The caller re-verifies on the next event and
# leaves no terminal label; the stall sweep is the bounded backstop if it never resolves.
#
# Exit 2 only on a wiring error — an unset input or a non-integer counter.
#
# WHY THIS IS A SCRIPT AND NOT WORKFLOW YAML: this is the one place in the delivery loop
# where a bug can mark broken code ready to merge, so it is the one place that needs tests.
# scripts/deliver_gate_test.go drives every combination of the declared input domains.
#
# The caller is responsible for mapping raw GitHub values into the domains above. That
# mapping is the likeliest thing to drift (GitHub has eight check conclusions, not two), so
# an out-of-domain value is not trusted here: it takes the catch-all and stops the delivery.
#
# `set -e` is deliberately off: every path ends in an explicit emit or usage call, and a
# non-zero exit from a comparison is normal control flow.

set -uo pipefail

usage() {
  echo "usage: $0 (required environment variable $1 is unset, empty, or malformed)" >&2
  exit 2
}

# conflict_clause — the ONE phrasing of "this branch conflicts with main", naming the files when
# the caller could compute them (#1781). One function rather than a literal at each emit site, so
# every decision says the same thing and a future row cannot invent a variant spelling.
conflict_clause() {
  local clause="the branch has merge conflicts with main that must be resolved"
  # Whitespace/comma-separated in, comma-space separated out, so a multi-line list from
  # scripts/conflicting-files.sh reads as one line in a PR comment.
  local files
  files=$(tr ',' ' ' <<< "${CONFLICT_FILES:-}" | tr -s '[:space:]' '\n' | sed '/^$/d' | paste -sd, - | sed 's/,/, /g')
  [[ -z "$files" ]] || clause="$clause (conflicting files: $files)"
  printf '%s' "$clause"
}

# decorate_conflict is false until MERGE_STATE has been validated, because the domain checks
# themselves emit and must not append a clause derived from a value they just rejected.
decorate_conflict=false

# emit <decision> <reason> — the single exit point for a computed decision. Every branch
# below ends here, which is what makes "the gate always decides" structural rather than a
# property to be re-checked on every edit.
#
# It is also where the conflict clause is PREPENDED (#1781). Doing it here rather than at each
# call site is what makes "a conflicting branch's reason always names the conflict" structural:
# the observed dead-end on PR #1778 was a row that fired first (no DELIVER-VERDICT marker) and
# reported its own condition while the conflict — the actual cause, and the only actionable thing
# — went unmentioned. It leads rather than trails because it is the thing a human must act on.
emit() {
  local reason="$2"
  if [[ "$decorate_conflict" == true && "$MERGE_STATE" == conflicting ]]; then
    reason="$(conflict_clause), and $reason"
  fi
  printf 'decision=%s\nreason=%s\n' "$1" "$reason"
  exit 0
}

for var in CI_STATUS PLAN_GATE AGENT_VERDICT QA_VERDICT DISMISSALS MERGE_STATE ROUND MAX_ROUNDS; do
  [[ -n "${!var:-}" ]] || usage "$var"
done

# A non-integer counter means the round-label parsing upstream is broken. That is a wiring
# bug rather than an unrecognised signal, so it is loud: a delivery that silently restarted
# its round count at zero would loop forever.
[[ "$ROUND" =~ ^[0-9]+$ ]] || usage ROUND
[[ "$MAX_ROUNDS" =~ ^[0-9]+$ ]] || usage MAX_ROUNDS

# Domain checks come before any decision. An unmapped GitHub check conclusion reaching the
# gate must stop the delivery, not silently miss every branch: `cancelled` is neither
# `success` nor `failure`, and without this it would match no row at all.
case "$CI_STATUS" in
  success | failure | unknown) ;;
  *) emit needs-human "unrecognised CI_STATUS '$CI_STATUS' — the CI derivation step needs to map this to success, failure, or unknown" ;;
esac
case "$PLAN_GATE" in
  pass | regression | conflicts | unverified | absent) ;;
  *) emit needs-human "unrecognised PLAN_GATE '$PLAN_GATE' — expected pass, regression, conflicts, unverified, or absent" ;;
esac
case "$AGENT_VERDICT" in
  GREEN | NOT-GREEN | MISSING) ;;
  *) emit needs-human "unrecognised AGENT_VERDICT '$AGENT_VERDICT' — expected GREEN, NOT-GREEN, or MISSING" ;;
esac
case "$QA_VERDICT" in
  PASS | BLOCK | MISSING) ;;
  *) emit needs-human "unrecognised QA_VERDICT '$QA_VERDICT' — expected PASS, BLOCK, or MISSING" ;;
esac
case "$DISMISSALS" in
  none | open | unknown) ;;
  *) emit needs-human "unrecognised DISMISSALS '$DISMISSALS' — expected none, open, or unknown" ;;
esac
case "$MERGE_STATE" in
  mergeable | conflicting | unknown) ;;
  *) emit needs-human "unrecognised MERGE_STATE '$MERGE_STATE' — the mergeability derivation step needs to map GitHub's mergeable_state to mergeable, conflicting, or unknown" ;;
esac
# From here on MERGE_STATE is in-domain, so every reason may name a conflict (#1781).
decorate_conflict=true

# Rows 1 and 2: no usable evidence. Checked before anything else so that a GREEN review can
# never stand in for a signal that was never read. Both reviews are treated alike here: a
# qa-review that never produced a marker is missing evidence, not a pass — a qa-review run
# that crashed, timed out or lost its model would otherwise wave the delivery through.
[[ "$CI_STATUS" != unknown ]] \
  || emit needs-human "CI status could not be determined for this head commit, so no verdict can be trusted"

# #1781 — a CONFLICT OUTRANKS AN UNREADABLE REVIEW MARKER, and this ordering is the whole fix.
#
# A `dirty` branch has no merge ref, so nothing was verified against main and the review markers
# are not evidence of anything: verify now deliberately SKIPS both agent reviews on such a branch
# (deliver-verify.yml), which makes MISSING the expected reading rather than an anomaly. Before
# this, the missing-marker rows below fired first and PR #1778 stopped for a human with the reason
# "the verify phase posted no DELIVER-VERDICT marker" — reporting the symptom, hiding the cause,
# and leaving the conflict unnamed. That is the silent-stall class #1758 set out to remove.
#
# The conflict is also the one thing a correction round can act on, so this routes to `correct`
# (#1758(a): resolve and re-verify) by falling through to the round cap, which turns an
# unresolvable conflict into a `needs-human` that NAMES it (#1758(b)). It can never reach `ready`:
# `ready` is emitted only from MERGE_STATE=mergeable.
conflict_over_missing=false
if [[ "$MERGE_STATE" == conflicting ]] \
   && { [[ "$AGENT_VERDICT" == MISSING ]] || [[ "$QA_VERDICT" == MISSING ]]; }; then
  conflict_over_missing=true
  reason="the review verdicts could not be read (review '$AGENT_VERDICT', qa-review '$QA_VERDICT'), which is expected on a branch with no merge ref"
fi

if [[ "$conflict_over_missing" != true ]]; then
  [[ "$AGENT_VERDICT" != MISSING ]] \
    || emit needs-human "the verify phase posted no DELIVER-VERDICT marker, so its verdict could not be read"
  [[ "$QA_VERDICT" != MISSING ]] \
    || emit needs-human "the verify phase posted no QA-VERDICT marker, so the qa-review verdict could not be read"
fi

# Collect every objective signal that blocks a merge. Both are reported when both apply —
# a human reading the PR should not have to re-run the gate to discover the second reason.
blocking=""
add_blocking() { blocking="${blocking:+$blocking; }$1"; }

[[ "$CI_STATUS" != failure ]] || add_blocking "CI is failing"
case "$PLAN_GATE" in
  regression) add_blocking "archon plan distance increased" ;;
  conflicts) add_blocking "archon plan verdict is CONFLICTS" ;;
  # The PR declared an archon-plan but the plan check did not evaluate it. That is MISSING
  # EVIDENCE, not a pass: archon-review.sh exits 0 and falls back to a plan-less delta review
  # when plan resolution fails, so without this the dist ratchet would silently not apply and
  # a GREEN review could carry the PR to ready. Distinct from `absent`, which means the PR
  # never claimed a plan at all and is legitimately gated on CI plus the review alone.
  unverified) add_blocking "the PR declares an archon-plan but the plan check did not run, so the dist ratchet is unverified" ;;
esac

# Collect every REVIEW that is asking for a correction. Two reviewers, one signal each,
# reported together for the same reason both objective blockers are: a human reading the PR
# comment should see every reason the delivery did not pass, not just the first.
#
# Both markers are non-MISSING on every path that CONSULTS `findings`, so an empty `findings`
# means both reviews came back clean — which is what makes the ready row and the disagreement row
# below exhaustive. (A MISSING marker does reach this point in the #1781 conflict case, where the
# markers are known untrustworthy; that case takes the FIRST branch of the decision chain below and
# never reads `findings` or `blocking`, so the property the two rows rely on still holds.)
findings=""
add_finding() { findings="${findings:+$findings; }$1"; }

[[ "$AGENT_VERDICT" != NOT-GREEN ]] || add_finding "the review returned NOT-GREEN with open findings"
[[ "$QA_VERDICT" != BLOCK ]] || add_finding "the cross-vendor qa-review returned BLOCK with open findings"

# The decision itself. `reason` is only set on paths that fall through to the round cap;
# every terminal path emits directly.
if [[ "$conflict_over_missing" == true ]]; then
  # #1781: `reason` was set above and the conflict clause is prepended by `emit`. Deliberately
  # FIRST in this chain — the review-derived rows below are computed from markers this branch has
  # already established are not trustworthy, so letting them speak would restate the symptom.
  # Falls through to the round cap, exactly like any other correctable round.
  :
elif [[ -n "$blocking" ]]; then
  # Row 3 — the guardrail, generalised over both reviewers (#1715). A review claiming the code
  # is fine against a failing objective signal is a disagreement, and resolving it is a human's
  # call: correcting would ask the agent to fix findings both reviewers said do not exist, and
  # readying would trust them over the evidence.
  #
  # The test is now "is NEITHER reviewer asking for a correction", not "did the Anthropic
  # reviewer say GREEN". With two reviewers the original test would send an objective failure
  # that qa-review DID find fault with to a human, when there are named findings a correction
  # round can act on — and acting on them is the whole reason the loop has correction rounds.
  [[ -n "$findings" ]] \
    || emit needs-human "both reviews came back clean (review GREEN, qa-review PASS) but $blocking — a human needs to resolve this disagreement"
  reason="$blocking, and $findings"
elif [[ -n "$findings" ]]; then
  reason="$findings"
else
  # Row 5 — the only path to ready. Reached only with CI success, a plan signal of pass or
  # absent, an explicit GREEN from the review, a PASS from qa-review, no dismissal the reviewer
  # has left unaccepted, AND a branch that is actually mergeable. (AGENT_VERDICT is necessarily
  # GREEN and QA_VERDICT necessarily PASS here — a NOT-GREEN or BLOCK went into `findings` above,
  # and MISSING/unknown emitted earlier — so the only axes left are dismissals and mergeability.)
  #
  # Dismissals are checked here rather than alongside CI and the plan signal on purpose: an
  # outstanding dismissal should withhold the TERMINAL verdict, not divert an honest set of
  # review findings away from the correction round that would resolve them. `unknown` is
  # treated as outstanding — an unreadable dismissal state is not evidence that there is
  # nothing to accept.
  case "$DISMISSALS" in
    none)
      # The one place the merge state changes the outcome. A conflicting branch cannot be merged
      # (GitHub cannot compute its merge ref), so `ready-for-merge` on it is a stale label nobody
      # can act on — #1758. It is NOT the clean-reviews-vs-blocking disagreement above: the
      # reviews approved the code, not the mergeability, so it routes to a correction round (the
      # agent merges main + resolves) by falling through to the round cap, rather than stopping at
      # needs-human. `unknown` withholds the terminal verdict loudly rather than trusting an
      # unverified mergeability.
      case "$MERGE_STATE" in
        mergeable)   emit ready "CI passed, plan signal '$PLAN_GATE', the review returned GREEN, and qa-review returned PASS" ;;
        # The conflict itself is phrased once, by `emit`'s prepended clause (#1781) — so this reason
        # states only what the reviews said, and a change to the wording (or to the file list)
        # cannot leave two spellings of the same fact in one comment.
        conflicting) reason="the reviews are clean (review GREEN, qa-review PASS)" ;;
        # Non-terminal: an unread mergeability on an otherwise-green PR is a re-checkable blip,
        # not a reason to stop for a human. `recheck` re-verifies on the next event (#1758 G1).
        *)           emit recheck "every signal is green, but the branch's mergeability against main could not be determined this run; re-verifying on the next event rather than stopping" ;;
      esac
      ;;
    open) emit needs-human "every signal is green, but a correction dismissed a finding that the review has not accepted — a human needs to decide whether the dismissal stands" ;;
    *)    emit needs-human "every signal is green, but the dismissal state could not be read, so it is not known whether a dismissed finding is outstanding" ;;
  esac
fi

# Defence in depth for "the gate always decides": the branches above are exhaustive over the
# declared domains, so an unset `reason` here means a domain gained a value nothing routes.
# Better a named stop than a correction round with an empty explanation.
[[ -n "${reason:-}" ]] \
  || emit needs-human "AGENT_VERDICT '$AGENT_VERDICT', QA_VERDICT '$QA_VERDICT', and MERGE_STATE '$MERGE_STATE' reached the decision chain unhandled"

# Row 7 — the cap applies only to a correction. A delivery that is genuinely ready stays
# ready at any round; the cap bounds correction attempts, not the delivery.
(( ROUND < MAX_ROUNDS )) \
  || emit needs-human "$reason, and the correction round cap of $MAX_ROUNDS is reached after $ROUND round(s)"

emit correct "$reason"
