# Fix Documentation Drift (INV-10/INV-11 + Event Ordering) Implementation Plan

**Goal:** Sync CLAUDE.md invariants and architecture.md event ordering description with their canonical sources.
**Source:** GitHub issues #702 and #733.
**Closes:** `Fixes #702, fixes #733`

## Behavioral Contracts

BC-1: CLAUDE.md invariants list completeness
- GIVEN the canonical `docs/contributing/standards/invariants.md` defines INV-1 through INV-11
- WHEN a contributor reads the CLAUDE.md invariants working copy
- THEN they see summaries for all 11 invariants (INV-1 through INV-11)

BC-2: Architecture event ordering accuracy
- GIVEN per-instance events use `(timestamp, priority, seqID)` ordering (`sim/event.go`, `sim/simulator.go`)
- WHEN a contributor reads `docs/concepts/architecture.md` ordering rules
- THEN all three ordering levels (cluster-first, instance-index tie-break, per-instance three-key) are accurately described

## Tasks

### Task 1: Add INV-10 and INV-11 to CLAUDE.md (BC-1)

**Files:** modify `CLAUDE.md`

**Impl:**
After the INV-9 bullet (line 126), add:
```
- **INV-10 Session causality**: For all rounds N in a closed-loop session: `round[N+1].ArrivalTime >= round[N].CompletionTime + ThinkTimeUs`. See `docs/contributing/standards/invariants.md`.
- **INV-11 Session completeness**: Every session reaches exactly one terminal state: completed, cancelled, or horizon-interrupted. No session is silently abandoned. See `docs/contributing/standards/invariants.md`.
```

**Verify:** Visual inspection — count 11 invariant bullets in CLAUDE.md.
**Lint:** `golangci-lint run ./...` (no Go changes, but confirm no pre-existing issues)
**Commit:** `docs(claude-md): add INV-10 and INV-11 session invariants (BC-1)`

### Task 2: Fix event ordering description in architecture.md (BC-2)

**Files:** modify `docs/concepts/architecture.md`

**Impl:**
Replace line 65:
```
- Within a single instance, events are ordered by timestamp only
```
With:
```
- Within a single instance, events are ordered by `(timestamp, priority, seqID)` — the same three-key scheme used at the cluster level
```

**Verify:** Visual inspection — confirm the description matches `sim/simulator.go` EventQueue.Less().
**Lint:** N/A (markdown only)
**Commit:** `docs(architecture): fix per-instance event ordering description (BC-2)`

## Sanity Checklist

- [x] R4 (construction sites): No struct changes
- [x] R1 (silent continue): No code changes
- [x] R2 (determinism): No code changes
- [x] Source-of-truth map: CLAUDE.md working copy updated to match invariants.md (this IS the fix)
- [x] No new interfaces, types, CLI flags, or behavioral changes
