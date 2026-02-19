# Fix #231: CLI Validation for --total-kv-blocks and --block-size-in-tokens

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate `--total-kv-blocks` and `--block-size-in-tokens` at the CLI boundary so invalid values produce a clear error message instead of a panic in library code.

**The problem today:** If a user passes `--total-kv-blocks 0` or `--block-size-in-tokens 0`, the program panics deep inside `NewKVStore` (library code in `sim/`). Per BLIS architecture, `sim/` must never terminate the process — user input errors should be caught at the CLI boundary (`cmd/root.go`) with `logrus.Fatalf`. The defaults (1,000,000 blocks and 16 tokens/block) mean this hasn't triggered in practice, but it's a correctness gap.

**What this PR adds:**
1. CLI-level validation for `--total-kv-blocks` — rejects zero and negative values with a clear `logrus.Fatalf` message before reaching library code
2. CLI-level validation for `--block-size-in-tokens` — same pattern

**Why this matters:** Enforces the error handling boundary rule (Antipattern Rule 6): library code returns errors or panics on invariant violations, but user input is validated at the CLI level. This prevents confusing panics and aligns with all other numeric flag validations already in `cmd/root.go`.

**Architecture:** Two `logrus.Fatalf` guards added to `cmd/root.go` in the validation section (around line 249, alongside the existing `numInstances` check). No new types, no new files, no interface changes.

**Source:** GitHub issue #231

**Closes:** Fixes #231

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds two CLI-level guards for `--total-kv-blocks` and `--block-size-in-tokens` in `cmd/root.go`. These flags control KV cache sizing and are currently validated only inside `sim/kv_store.go:NewKVStore()` via `panic()`. The fix moves validation to the CLI boundary where it belongs, matching the pattern used for `--num-instances`, `--rate`, `--kv-cpu-blocks`, and other numeric flags.

No other files are affected. The `NewKVStore` panics (sim/kv_store.go:25-29) and `NewSimulator` panics (sim/simulator.go:161-166) both remain as defense-in-depth for programmatic callers.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: CLI rejects zero total-kv-blocks
- GIVEN the CLI is invoked with `--total-kv-blocks 0`
- WHEN the run command executes
- THEN the program MUST exit with a fatal error message containing "total-kv-blocks must be > 0"
- MECHANISM: `logrus.Fatalf` guard in `cmd/root.go` before `DeploymentConfig` construction

BC-2: CLI rejects zero block-size-in-tokens
- GIVEN the CLI is invoked with `--block-size-in-tokens 0`
- WHEN the run command executes
- THEN the program MUST exit with a fatal error message containing "block-size-in-tokens must be > 0"
- MECHANISM: `logrus.Fatalf` guard in `cmd/root.go` before `DeploymentConfig` construction

BC-3: CLI rejects negative total-kv-blocks
- GIVEN the CLI is invoked with `--total-kv-blocks -5`
- WHEN the run command executes
- THEN the program MUST exit with a fatal error message containing "total-kv-blocks must be > 0"
- MECHANISM: Same guard as BC-1 (uses `<= 0`)

BC-4: CLI rejects negative block-size-in-tokens
- GIVEN the CLI is invoked with `--block-size-in-tokens -1`
- WHEN the run command executes
- THEN the program MUST exit with a fatal error message containing "block-size-in-tokens must be > 0"
- MECHANISM: Same guard as BC-2 (uses `<= 0`)

**Negative Contracts:**

BC-5: Valid defaults pass validation
- GIVEN the CLI is invoked without `--total-kv-blocks` or `--block-size-in-tokens` flags
- WHEN the run command executes
- THEN validation MUST pass (defaults are 1000000 and 16 respectively)
- MECHANISM: Guards only trigger on `<= 0`

**Error Handling Contracts:**

BC-6: Error handling boundary respected
- GIVEN the validation is in `cmd/root.go`
- WHEN invalid values are provided
- THEN `logrus.Fatalf` MUST be used (not `panic`, not `os.Exit`)
- MECHANISM: Follows existing CLI validation pattern

### C) Component Interaction

```
CLI (cmd/root.go)
  │
  ├─ [NEW] Validate --total-kv-blocks > 0
  ├─ [NEW] Validate --block-size-in-tokens > 0
  ├─ [existing] Validate --num-instances >= 1
  ├─ [existing] Validate policy names
  ├─ [existing] Validate --kv-cpu-blocks >= 0
  │
  └─→ DeploymentConfig → ClusterSimulator → sim.NewKVStore (defense-in-depth panics remain)
```

No new interfaces, types, or state. Extension friction: 0 files (this adds guards to an existing validation section).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue mentions panics in `NewKVStore` / `NewSimulator` | Plan's CLI guard covers both; both factory panics remain as defense-in-depth | SIMPLIFICATION: Single CLI guard supersedes both library-level panics |

### E) Review Guide

1. **The tricky part:** Nothing tricky — this is a 4-line fix. The only subtlety is placement: the guards must be before `DeploymentConfig` construction (line 370) to prevent the values reaching `NewKVStore`.
2. **What to scrutinize:** Verify the guards use `<= 0` (not `< 0` or `== 0`) and use `logrus.Fatalf` (not `panic`).
3. **What's safe to skim:** Everything else in `cmd/root.go` — untouched.
4. **Known debt:** The `NewKVStore` panics (sim/kv_store.go:25-29) and `NewSimulator` panics (sim/simulator.go:161-166) remain as defense-in-depth. This is intentional — they catch programmatic misuse.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files:**
- Modify: `cmd/root.go` (~line 249, add 6 lines of validation)
- Modify: `cmd/root_test.go` (add behavioral test for validation)

**Key decisions:**
- Place validation after `numInstances` check (line 250) and before policy bundle loading (line 258)
- Use `<= 0` to reject both zero and negative values
- Note: `totalKVBlocks` can be overridden by `GetCoefficients` (line 170) before our validation runs. The validation at line ~252 happens AFTER that override, so it catches all sources of the value.

### G) Task Breakdown

---

### Task 1: Add CLI validation guards and tests

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-4, BC-5, BC-6

**Files:**
- Modify: `cmd/root.go:249-251` (add validation after numInstances check)
- Modify: `cmd/root_test.go` (add test)

**Step 1: Write failing test for CLI validation**

Context: We need to verify that zero/negative values for `--total-kv-blocks` and `--block-size-in-tokens` produce fatal errors. Since `logrus.Fatalf` calls `os.Exit`, we can't test it directly in-process. Instead, we test the validation logic by checking the flag defaults are positive (BC-5) and verifying the guard code exists via a unit test that checks the validation condition directly.

In `cmd/root_test.go`, add `"strconv"` to the import block, then add:
```go
func TestRunCmd_KVBlockFlags_DefaultsArePositive(t *testing.T) {
	// GIVEN the run command with its registered flags
	kvBlocksFlag := runCmd.Flags().Lookup("total-kv-blocks")
	blockSizeFlag := runCmd.Flags().Lookup("block-size-in-tokens")

	// WHEN we check the default values
	// THEN they MUST be positive (BC-5: valid defaults pass validation)
	assert.NotNil(t, kvBlocksFlag, "total-kv-blocks flag must be registered")
	assert.NotNil(t, blockSizeFlag, "block-size-in-tokens flag must be registered")

	kvDefault, err := strconv.ParseInt(kvBlocksFlag.DefValue, 10, 64)
	assert.NoError(t, err, "total-kv-blocks default must be a valid int64")
	assert.Greater(t, kvDefault, int64(0),
		"default total-kv-blocks must be positive (passes <= 0 validation)")

	bsDefault, err := strconv.ParseInt(blockSizeFlag.DefValue, 10, 64)
	assert.NoError(t, err, "block-size-in-tokens default must be a valid int64")
	assert.Greater(t, bsDefault, int64(0),
		"default block-size-in-tokens must be positive (passes <= 0 validation)")
}
```

**Step 2: Run test to verify it passes (this is a defaults test)**

Run: `go test ./cmd/... -run TestRunCmd_KVBlockFlags_DefaultsArePositive -v`
Expected: PASS (defaults are already correct)

**Step 3: Implement CLI validation guards**

Context: Add `logrus.Fatalf` guards for both flags in `cmd/root.go`, placed after the `numInstances` check (line 250) and before the policy bundle loading (line 258).

In `cmd/root.go`, after the `numInstances` validation block (line 250: `logrus.Fatalf("num-instances must be >= 1")`), add:

```go
		if totalKVBlocks <= 0 {
			logrus.Fatalf("--total-kv-blocks must be > 0, got %d", totalKVBlocks)
		}
		if blockSizeTokens <= 0 {
			logrus.Fatalf("--block-size-in-tokens must be > 0, got %d", blockSizeTokens)
		}
```

**Step 4: Run all tests to verify nothing breaks**

Run: `go test ./cmd/... -v`
Expected: PASS (all tests including new one)

Run: `go test ./... -count=1`
Expected: PASS (full suite)

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add cmd/root.go cmd/root_test.go
git commit -m "fix(cmd): validate --total-kv-blocks and --block-size-in-tokens at CLI boundary (BC-1..BC-6)

- Add logrus.Fatalf guards for zero/negative values before DeploymentConfig construction
- Prevents panic in NewKVStore by catching invalid input at CLI boundary
- Matches existing validation pattern for --num-instances, --rate, --kv-cpu-blocks

Fixes #231

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|------------------------|
| BC-1, BC-2, BC-3, BC-4 | Task 1 | Integration (implicit) | Guards use `logrus.Fatalf` which calls `os.Exit` — not unit-testable in-process. Verified by code review + manual CLI test. |
| BC-5 | Task 1 | Unit | `TestRunCmd_KVBlockFlags_DefaultsArePositive` |
| BC-6 | Task 1 | Code review | Guard uses `logrus.Fatalf`, not `panic` — verified by inspection |

Note: `logrus.Fatalf` terminates the process, making in-process testing impractical without subprocess execution (`exec.Command` + check exit code/stderr). The test verifies defaults are valid (won't trigger the guard), and the guard placement is verified by code review. This matches how other `logrus.Fatalf` guards in `cmd/root.go` are tested (they aren't — they're integration-tested via CLI invocation).

**Known debt:** BC-1 through BC-4 are not unit tested. Subprocess testing for `logrus.Fatalf` guards is a project-wide gap (no existing guards have subprocess tests), not specific to this PR. Adding a subprocess test pattern here would be scope creep.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Validation placed after `GetCoefficients` override | Low | Medium | Verified: `GetCoefficients` (line 170) can override `totalKVBlocks`; our guard at ~line 253 runs after that | Task 1 |
| Misleading error when `GetCoefficients` returns 0 | Low | Low | When no model matches in `defaults.yaml`, `GetCoefficients` returns `totalKVBlocks=0`, causing our guard to fire with "must be > 0" — misleading since the user didn't pass `--total-kv-blocks 0`. Pre-existing issue; fixing the error message is outside scope | N/A |
| Guard uses wrong comparison | Low | Low | Uses `<= 0` matching `NewKVStore` panic condition exactly | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing shared test package (not duplicated locally)
- [x] CLAUDE.md — no update needed (no new files, packages, or CLI flags)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code (no scaffolding)
- [x] Task dependencies are correctly ordered (single task)
- [x] All contracts are mapped to specific tasks
- [x] Golden dataset regeneration — not needed (no output changes)
- [x] Construction site audit — no struct fields added
- [x] Every new CLI flag validated — N/A (no new flags; existing flags getting validation)
- [x] Every error path either returns error, panics with context, or increments a counter
- [x] No map iteration feeds float accumulation
- [x] Library code never calls logrus.Fatalf — validation is in cmd/ only
- [x] No exported mutable maps
- [x] YAML config — N/A
- [x] Every division operation has zero guard — N/A
- [x] Grepped for references to issue #231 — none found in codebase

---

## Appendix: File-Level Implementation Details

### File: `cmd/root.go`

**Purpose:** Add validation guards for `--total-kv-blocks` and `--block-size-in-tokens`

**Change location:** After line 250 (`logrus.Fatalf("num-instances must be >= 1")`), before line 253 (traces filepath check).

**Code to add:**

```go
		if totalKVBlocks <= 0 {
			logrus.Fatalf("--total-kv-blocks must be > 0, got %d", totalKVBlocks)
		}
		if blockSizeTokens <= 0 {
			logrus.Fatalf("--block-size-in-tokens must be > 0, got %d", blockSizeTokens)
		}
```

**Behavioral notes:**
- `totalKVBlocks` may be overridden by `GetCoefficients()` at line 170. The validation at ~line 253 runs AFTER that override, so it validates the final resolved value.
- `blockSizeTokens` is never overridden by any config loading — it comes purely from CLI.
- Uses `<= 0` to match the exact condition in `NewKVStore` (sim/kv_store.go:25-29).

### File: `cmd/root_test.go`

**Purpose:** Add behavioral test verifying flag defaults are positive

**Code to add (after existing tests):**

```go
func TestRunCmd_KVBlockFlags_DefaultsArePositive(t *testing.T) {
	// GIVEN the run command with its registered flags
	kvBlocksFlag := runCmd.Flags().Lookup("total-kv-blocks")
	blockSizeFlag := runCmd.Flags().Lookup("block-size-in-tokens")

	// WHEN we check the default values
	// THEN they MUST be positive (BC-5: valid defaults pass validation)
	assert.NotNil(t, kvBlocksFlag, "total-kv-blocks flag must be registered")
	assert.NotNil(t, blockSizeFlag, "block-size-in-tokens flag must be registered")

	kvDefault, err := strconv.ParseInt(kvBlocksFlag.DefValue, 10, 64)
	assert.NoError(t, err, "total-kv-blocks default must be a valid int64")
	assert.Greater(t, kvDefault, int64(0),
		"default total-kv-blocks must be positive (passes <= 0 validation)")

	bsDefault, err := strconv.ParseInt(blockSizeFlag.DefValue, 10, 64)
	assert.NoError(t, err, "block-size-in-tokens default must be a valid int64")
	assert.Greater(t, bsDefault, int64(0),
		"default block-size-in-tokens must be positive (passes <= 0 validation)")
}
```
