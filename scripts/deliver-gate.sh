#!/usr/bin/env bash
#
# deliver-gate.sh — decide what happens next in an L1 delivery round.
#
# Environment (all required):
#   CI_STATUS      success | failure | unknown
#   PLAN_GATE      pass | regression | conflicts | unverified | absent
#   AGENT_VERDICT  GREEN | NOT-GREEN | MISSING
#   ROUND          correction rounds already spent (non-negative integer)
#   MAX_ROUNDS     hard cap on correction rounds (non-negative integer)
#
# Prints two lines and exits 0:
#   decision=ready|correct|needs-human
#   reason=<one line, safe to paste into a PR comment>
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

# emit <decision> <reason> — the single exit point for a computed decision. Every branch
# below ends here, which is what makes "the gate always decides" structural rather than a
# property to be re-checked on every edit.
emit() {
  printf 'decision=%s\nreason=%s\n' "$1" "$2"
  exit 0
}

for var in CI_STATUS PLAN_GATE AGENT_VERDICT ROUND MAX_ROUNDS; do
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

# Rows 1 and 2: no usable evidence. Checked before anything else so that a GREEN review can
# never stand in for a signal that was never read.
[[ "$CI_STATUS" != unknown ]] \
  || emit needs-human "CI status could not be determined for this head commit, so no verdict can be trusted"
[[ "$AGENT_VERDICT" != MISSING ]] \
  || emit needs-human "the verify phase posted no DELIVER-VERDICT marker, so its verdict could not be read"

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

# The decision itself. `reason` is only set on paths that fall through to the round cap;
# every terminal path emits directly.
if [[ -n "$blocking" ]]; then
  # Row 3 — the guardrail. A review claiming GREEN against a failing objective signal is a
  # disagreement, and resolving it is a human's call: correcting would ask the agent to fix
  # findings it just said do not exist, and readying would trust it over the evidence.
  [[ "$AGENT_VERDICT" != GREEN ]] \
    || emit needs-human "the review returned GREEN but $blocking — a human needs to resolve this disagreement"
  reason="$blocking"
else
  case "$AGENT_VERDICT" in
    # Row 5 — the only path to ready. Reached only with CI success, a plan signal of pass or
    # absent, and an explicit GREEN.
    GREEN) emit ready "CI passed, plan signal '$PLAN_GATE', and the review returned GREEN" ;;
    NOT-GREEN) reason="the review returned NOT-GREEN with open findings" ;;
    *) emit needs-human "AGENT_VERDICT '$AGENT_VERDICT' reached the decision chain unhandled" ;;
  esac
fi

# Row 7 — the cap applies only to a correction. A delivery that is genuinely ready stays
# ready at any round; the cap bounds correction attempts, not the delivery.
(( ROUND < MAX_ROUNDS )) \
  || emit needs-human "$reason, and the correction round cap of $MAX_ROUNDS is reached after $ROUND round(s)"

emit correct "$reason"
