#!/usr/bin/env bash
#
# map-merge-state.sh — map GitHub's REST `mergeable_state` into deliver-gate.sh's MERGE_STATE
# domain (#1758).
#
#   $1  the raw mergeable_state string ("" when the caller could not read it)
#
# Prints one of: mergeable | conflicting | unknown
#
# WHY THIS IS A SCRIPT AND NOT INLINE WORKFLOW YAML: the same reason deliver-gate.sh is one —
# a bug here maps a branch that conflicts with main to `mergeable`, which the gate then carries
# to `ready-for-merge`, the exact silent stall #1758 fixes. So this mapping is tested directly
# (scripts/map_merge_state_test.go), not just proven by the gate rejecting raw values.
#
# The caller (deliver-verify.yml) owns the async part — polling REST until GitHub has computed
# mergeability — and passes the final value here. This script is the pure mapping only, so it
# needs no network, no state, and is exhaustively testable across GitHub's whole state surface.
#
# Conflicts-only scope: `dirty` is the sole state that blocks the merge. `behind` and every
# other non-dirty state are `mergeable` — a behind branch still has a merge ref, is reviewable,
# and the correction phase merges main every round regardless (deliver-correct.yml STEP 1). An
# empty or `unknown` read is `unknown`, on which the gate withholds `ready` rather than trusting
# an unverified mergeability.

set -uo pipefail

case "${1-}" in
  dirty)      echo conflicting ;;
  "" | unknown) echo unknown ;;
  *)          echo mergeable ;;
esac
