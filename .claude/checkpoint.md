# Fix Audit Bugs — Session Checkpoint

**Date:** 2026-02-22
**Branch:** `fix-audit-bugs`
**Worktree:** `.worktrees/fix-audit-bugs/`
**Plan:** `docs/plans/fix-audit-bugs-plan.md`

## Status: Step 4.5 (Code Review) — All 10 tasks committed, 4 review passes running

### What's Done

1. **14 GitHub issues filed:** #352–#365 (all labeled `bug`)
2. **Plan written and converged:** 5 review passes + convergence re-run, all 0 CRITICAL / 0 IMPORTANT
3. **Rebased on upstream/main** after PR #371 (BatchFormation extraction)
4. **All 10 tasks committed** (9 commits, 14 issues):

| Commit | Task | Issues | Contracts |
|--------|------|--------|-----------|
| `a083c4e` | 1: Phantom block fix | #352 | BC-1 |
| `50600a3` | 2: Rollback order + dead test | #353, #354 | BC-2, BC-3 |
| `f4cdcc4` | 3: NaN/Inf validation + roofline | #355, #356 | BC-4, BC-5 |
| `94e8474` | 4: ComputedTokens cleanup | #357 | BC-6 |
| `66164ca` | 5: Block size threading | #359 | BC-7 |
| `4f5ace6` | 6: PrefixAffinity block hashing | #358 | BC-8 |
| `bcbb349` | 7-8: Sampler guards + param validation | #360, #361 | BC-9, BC-10 |
| `93264d8` | 9: Strict YAML + CSV errors | #362, #363 | BC-11, BC-12 |
| `ca4faad` | 10: JainFairness + per-request units | #364, #365 | BC-13, BC-14 |

5. **Full test suite passes** (`go test ./...` + `golangci-lint run ./...` = 0 issues)
6. **Step 4.5 in progress:** 4 parallel code review agents running

### Task 2 Resolution

The rollback ordering bug (BC-3, #354) was confirmed REAL:
- **Root cause:** `rollbackAllocation` iterated `cachedMutations` in forward order, but blocks were removed in forward order too. Appending in forward order reversed the tail positions.
- **Fix:** Reverse-iterate `cachedMutations` so blocks are appended to the tail in their original order. `newlyAllocated` still prepends in reverse (already correct).
- **Test:** `TestAllocateKVBlocks_Rollback_PreservesFreeListOrder` verifies both free-head identity AND second-block identity match pre-allocation state.

### Key Design Decisions

- **Task 5 (block size threading):** Added `blockSize int64` param to `NewRoutingPolicy`. Used Python script for mass test-site updates (~44 call sites).
- **Task 6 (PrefixAffinity):** Added standalone `computeBlockHashes` function in `routing.go` instead of creating a `PrefixCacheIndex` per route call. Reuses `hashBlock` from `prefix_cache_index.go`.
- **Task 10 (per-request units):** Divides `ITL` and `SchedulingDelay` by 1e3 in `SaveResults`. Only affects `--results-path` JSON, not stdout aggregates. Golden tests unaffected.

### Remaining Workflow Steps

- **Step 4.5:** 4 code review passes (running) → fix issues between passes → convergence re-run
- **Step 4.75:** Self-audit (9 dimensions)
- **Step 5:** commit-push-pr skill
