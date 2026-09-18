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
# Conflicts-only scope: `dirty` is the sole state that blocks the merge. The other known states
# GitHub reports are `mergeable` — a `behind` branch still has a merge ref and is reviewable
# (and the correction phase merges main every round regardless, deliver-correct.yml STEP 1);
# `blocked`/`unstable`/`has_hooks`/`draft`/`clean` all still have a merge ref. An empty or
# `unknown` read is `unknown`, on which the gate withholds `ready` rather than trusting an
# unverified mergeability.
#
# The states are ENUMERATED and the wildcard fails closed to `unknown`, NOT open to `mergeable`
# (G4). This mirrors deliver-gate.sh's own rule that an out-of-domain value is not trusted: if
# GitHub ever adds a new state, this loop must not assume it is safe to merge — it stops for a
# human (recoverable on the next run) until the new state is classified here deliberately.

set -uo pipefail

case "${1-}" in
  dirty)                                          echo conflicting ;;
  clean | behind | blocked | unstable | has_hooks | draft) echo mergeable ;;
  *)                                              echo unknown ;;   # "", "unknown", and any unrecognised/future state — fail closed
esac
