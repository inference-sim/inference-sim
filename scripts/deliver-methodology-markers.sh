#!/usr/bin/env bash
#
# deliver-methodology-markers.sh — decide whether a blis-pr-review verdict comment shows that
# the review methodology actually ran, or only improvised a verdict.
#
# Reads the review comment body on STDIN. Prints two lines and exits 0:
#   markers=present|absent
#   reason=<one line, safe to paste into a workflow log>
#
# Exit 2 only on a wiring error — empty input, which means the caller failed to capture the
# comment body and no judgement is possible.
#
# WHY THIS IS A SCRIPT AND NOT WORKFLOW YAML: a GREEN verdict from a review that skipped the
# methodology is indistinguishable, to scripts/deliver-gate.sh, from a thorough one — the gate
# reads only the DELIVER-VERDICT marker. This predicate is what lets deliver-verify.yml downgrade
# such a GREEN to NOT-GREEN before the gate ever sees it, so it is the one piece of that logic
# worth unit-testing. scripts/deliver_markers_test.go drives it.
#
# WHAT COUNTS AS "RAN THE METHODOLOGY". Two markers, both required, chosen because each is emitted
# only by a phase that cannot be summarised away:
#
#   1. Q/A phase — the skill (SKILL.md:82-99) spawns a subagent that answers each probing question
#      tagged CONFIDENT / FLAW_FOUND / CANNOT_ANSWER. Those tags appear nowhere else, so their
#      absence means the Q/A phase did not run (not merely that it went unprinted).
#   2. Findings table — a "Findings Summary" heading. The skill's blocking signal is FLAW_FOUND;
#      the table is where findings become visible to the correction phase and the dismissal ledger
#      rather than dissolving into prose. Its absence is what let two real findings self-dismiss
#      inline on #1725.
#
# `set -e` is deliberately off: every path ends in an explicit emit or usage call, and a non-zero
# grep is normal (it means "marker absent"), not an error.

set -uo pipefail

emit() {
  printf 'markers=%s\nreason=%s\n' "$1" "$2"
  exit 0
}

body=$(cat)

# Empty input is a wiring bug in the caller (it read no comment body), not an absent-markers
# verdict. Fail loud rather than silently reporting `absent` and downgrading a review that may
# have been fine — the caller must fix how it captures the body.
grep -q '[^[:space:]]' <<< "$body" \
  || { echo "usage: $0 reads the review comment body on stdin (received empty input)" >&2; exit 2; }

# A Q/A tag: emitted only by the Q/A phase's per-question answers.
qa=no
grep -qE 'CONFIDENT|FLAW_FOUND|CANNOT_ANSWER' <<< "$body" && qa=yes

# The findings table heading. Accept it as a markdown heading (`## Findings Summary`, any
# depth) OR a bold line (`**Findings Summary**`) — both are shapes a real review uses, and a
# false "absent" on a genuine review would cost a needless correction round. Requiring the line
# to START with `#`/`**` keeps prose like "see the findings summary below" from matching.
table=no
grep -qiE '^[[:space:]]*(#{1,6}[[:space:]]*|\*\*)Findings Summary' <<< "$body" && table=yes

if [[ "$qa" == yes && "$table" == yes ]]; then
  emit present "the review shows a Q/A phase and a findings table"
fi

missing=""
add_missing() { missing="${missing:+$missing and }$1"; }
[[ "$qa" == yes ]] || add_missing "the Q/A phase (no CONFIDENT/FLAW_FOUND/CANNOT_ANSWER tags)"
[[ "$table" == yes ]] || add_missing "a findings table (no 'Findings Summary' heading)"
emit absent "the review is missing $missing — the methodology did not run whole"
