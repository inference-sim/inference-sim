# H22: Zero KV Blocks -- CLI Validation Boundary

**Status:** Confirmed
**Resolution:** Clean confirmation -- defense-in-depth validated
**Family:** Robustness/failure-mode
**VV&UQ:** Verification
**Tier:** Edge case robustness
**Type:** Deterministic
**Date:** 2026-02-22
**Rounds:** 1

## Hypothesis

> Running with `--total-kv-blocks 0` (or other zero/negative KV configurations) should produce a clean CLI error (`logrus.Fatalf`), not a panic or stack trace from `sim/`.

## Experiment Design

**Classification:** Deterministic -- each invalid input has exactly one expected outcome.

**Test cases:**

| # | Flags | Expected behavior |
|---|-------|-------------------|
| 1 | `--total-kv-blocks 0` | `logrus.Fatalf` with message mentioning flag name and value |
| 2 | `--block-size-in-tokens 0` | `logrus.Fatalf` with message mentioning flag name and value |
| 3 | `--total-kv-blocks -1` | `logrus.Fatalf` with message mentioning flag name and value |
| 4 | `--total-kv-blocks 0 --kv-cpu-blocks 100` | `logrus.Fatalf` (GPU blocks validated before CPU tier) |
| 5 | `--kv-cpu-blocks -1` | `logrus.Fatalf` with message mentioning flag name and value |
| C | `--total-kv-blocks 2048 --block-size-in-tokens 16` | Exit 0, normal completion (control) |

**Controlled variables:** `--model meta-llama/llama-3.1-8b-instruct`, `--num-requests 5`, `--rate 100`

**Verification criteria per test case:**
1. Exit code is non-zero (1)
2. stderr contains NO panic/stack trace (no `goroutine N`, no `runtime error`)
3. stderr contains a logrus fatal message (`level=fatal` in non-TTY mode)
4. The error message mentions the specific flag name and invalid value

## Results

All 6 test cases passed. Every invalid configuration is caught at the CLI boundary with a clean, user-friendly error message.

| Test case | Exit code | Panic? | logrus fatal? | Message correct? | Result |
|-----------|-----------|--------|---------------|------------------|--------|
| `--total-kv-blocks 0` | 1 | No | Yes | `--total-kv-blocks must be > 0, got 0` | PASS |
| `--block-size-in-tokens 0` | 1 | No | Yes | `--block-size-in-tokens must be > 0, got 0` | PASS |
| `--total-kv-blocks -1` | 1 | No | Yes | `--total-kv-blocks must be > 0, got -1` | PASS |
| `--total-kv-blocks 0 --kv-cpu-blocks 100` | 1 | No | Yes | `--total-kv-blocks must be > 0, got 0` | PASS |
| `--kv-cpu-blocks -1` | 1 | No | Yes | `--kv-cpu-blocks must be >= 0, got -1` | PASS |
| Valid config (control) | 0 | No | No | N/A | PASS |

## Root Cause Analysis

The CLI boundary has a two-layer defense-in-depth structure:

### Layer 1: CLI validation (`cmd/root.go`)

The `run` command validates KV-related flags before constructing `SimConfig`:

- **`cmd/root.go:280-281`**: `if totalKVBlocks <= 0 { logrus.Fatalf("--total-kv-blocks must be > 0, got %d", totalKVBlocks) }`
- **`cmd/root.go:283-284`**: `if blockSizeTokens <= 0 { logrus.Fatalf("--block-size-in-tokens must be > 0, got %d", blockSizeTokens) }`
- **`cmd/root.go:354-355`**: `if kvCPUBlocks < 0 { logrus.Fatalf("--kv-cpu-blocks must be >= 0, got %d", kvCPUBlocks) }`

These use `logrus.Fatalf`, which logs a fatal-level message to stderr and calls `os.Exit(1)`. This is the correct boundary pattern per R6 (no `Fatalf` in library code; CLI owns process termination).

Note: The `<= 0` check covers both zero and negative values in a single guard, which is why `--total-kv-blocks -1` produces the same message format as `--total-kv-blocks 0`.

### Layer 2: Library panics (`sim/`)

If CLI validation were removed, the library layer would still catch invalid values, but via `panic()` instead of clean error messages:

- **`sim/simulator.go:166-167`**: `panic("SimConfig.TotalKVBlocks must be > 0, got %d")`
- **`sim/simulator.go:169-170`**: `panic("SimConfig.BlockSizeTokens must be > 0, got %d")`
- **`sim/kv_store.go:25-26`**: `panic("KVStore: TotalKVBlocks must be > 0, got %d")`
- **`sim/kv_store.go:28-29`**: `panic("KVStore: BlockSizeTokens must be > 0, got %d")`

These panics are defense-in-depth (R6 compliant -- library code panics on programming errors, not user input errors). They would produce stack traces, which is acceptable for programmatic callers but not for CLI users.

### Validation ordering

Test case 4 (`--total-kv-blocks 0 --kv-cpu-blocks 100`) confirms that GPU block validation at line 280 fires before the code reaches CPU tier configuration at line 354. This is correct -- the primary resource must be validated before any tier composition logic.

## Devil's Advocate (RCV-5)

**Counter-argument:** "The `<= 0` checks are trivially correct; this experiment adds no value."

**Rebuttal:** The value is threefold:
1. **Regression guard**: Without explicit verification, a refactor could move validation after construction, allowing panics to reach users. This experiment documents the expected boundary.
2. **Defense-in-depth documentation**: The experiment reveals the two-layer structure (CLI `Fatalf` + library `panic`). If the CLI layer is bypassed (e.g., programmatic callers using `NewSimulator` directly), the library layer catches the error.
3. **Message quality**: The experiment verifies that error messages include the flag name and the invalid value, which is essential for user debugging. A check that just says "invalid configuration" would be harder to act on.

**Counter-argument:** "What about `--total-kv-blocks` values that are positive but pathologically small (e.g., 1 block for 1000 requests)?"

**Rebuttal:** That is a different hypothesis class (capacity planning / resource exhaustion), not a validation boundary test. Small-but-valid values should run and produce degraded metrics, not errors.

## Findings Classification

| ID | Finding | Classification | Severity |
|----|---------|---------------|----------|
| F1 | All zero/negative KV block configs caught at CLI with `logrus.Fatalf` | Clean confirmation | N/A |
| F2 | CLI validation fires before `sim/` construction | Expected behavior | N/A |
| F3 | Library-layer panics exist as defense-in-depth backup | By design (R6) | N/A |
| F4 | Error messages include flag name and invalid value | Good practice (R3) | N/A |

## Standards Audit

| Standard | Compliance | Evidence |
|----------|-----------|----------|
| R3 (Validate CLI flags) | Compliant | `cmd/root.go:280-285, 354-355` validate zero, negative |
| R6 (No Fatalf in library) | Compliant | `sim/simulator.go:166-171`, `sim/kv_store.go:25-30` use `panic()`, not `Fatalf` |
| R11 (Guard division) | Compliant | Zero `BlockSizeTokens` caught before any division by block size |
| ED-1 (Control experiment) | Valid config control included | Test case C exits 0 |
| ED-2 (Rate awareness) | N/A | Deterministic validation, not rate-dependent |
| RCV-1 (Cite file:line) | All citations verified | See Root Cause Analysis |

## Scope and Limitations (RCV-6)

**Tested:**
- Zero and negative values for `--total-kv-blocks`, `--block-size-in-tokens`, `--kv-cpu-blocks`
- Combination of zero GPU blocks with non-zero CPU tier
- Valid configuration as positive control

**Not tested:**
- NaN/Inf values for integer flags (Go's flag parser rejects these before our validation)
- Extremely large values (e.g., `--total-kv-blocks 9999999999999`) -- memory exhaustion, not validation
- `--kv-offload-threshold` boundary values (covered by separate validation at `cmd/root.go:357-358`)
- `--kv-transfer-bandwidth` zero/negative when CPU tier is enabled (covered at `cmd/root.go:360-361`)
- Interaction with `--model-config-folder` (roofline mode) -- orthogonal config axis
- Programmatic callers bypassing CLI (library panics are the safety net, not tested here)

## Evidence Quality

| Aspect | Assessment |
|--------|-----------|
| Reproducibility | Deterministic -- same binary, same flags, same result every time |
| Sample size | N/A (deterministic, not statistical) |
| Confounds | None -- each test case varies exactly one flag |
| Measurement validity | Exit code and stderr content are the direct observables |
| Observer effect | None -- validation runs before simulation starts |

## Implications for Users

1. **Zero KV blocks is never valid.** The simulator requires at least 1 GPU KV block. The error message explicitly says `> 0`.
2. **Negative values are caught by the same `<= 0` guard** -- no separate "negative" error message exists.
3. **CPU tier with zero GPU blocks is rejected** because GPU block validation runs first (line 280) before CPU tier config (line 354).
4. **Error output format varies by terminal:** TTY shows `FATA[0000]`, redirected stderr shows `level=fatal msg="..."`. Both contain the same message text.

## Reproducing

```bash
cd hypotheses/h22-zero-blocks
./run.sh          # builds binary if needed
./run.sh --rebuild  # force rebuild
```
