# Plan: Rename `--results-path` on `blis run` to `--metrics-path`

**Goal:** Fix the footgun where `blis run --results-path` and `blis replay --results-path` write incompatible JSON schemas under the same flag name. After this PR, `blis run` accepts `--metrics-path` (MetricsOutput JSON) and `blis replay` accepts `--results-path` ([]SimResult JSON). The flag names now match their output schemas.

**Source:** [Issue #864](https://github.com/inference-sim/inference-sim/issues/864)

**Closes:** #864

**PR Size Tier:** Small (3 files changed, no new interfaces or types, mechanical CLI rename)

**Deviation Log:**
- `CORRECTION`: The Usage override added to `replayCmd` in `replay.go:258–260` (a bandaid that documented the schema divergence) is removed — it is redundant once the flags have distinct names with accurate descriptions.
- `CLARIFICATION`: The issue offered three options (A/B/C). This plan implements Option A (flag rename) as the narrowest correct fix. Option C's stated benefit ("directly pass blis run output to blis calibrate") was found to be incorrect during analysis — `blis calibrate` requires request identity that only `blis replay` provides.

---

## Behavioral Contracts

### BC-1: `blis run` accepts `--metrics-path`, not `--results-path`

**GIVEN** the `blis run` command
**WHEN** `--help` is invoked
**THEN** the flag `--metrics-path` appears in the flag list with a description mentioning "MetricsOutput JSON"
**AND** the flag `--results-path` does NOT appear in the flag list

### BC-2: `blis replay` accepts `--results-path`, not `--metrics-path`

**GIVEN** the `blis replay` command
**WHEN** `--help` is invoked
**THEN** the flag `--results-path` appears in the flag list with a description mentioning "SimResult JSON" and "blis calibrate"
**AND** the flag `--metrics-path` does NOT appear in the flag list

### BC-3: `blis run --metrics-path` writes MetricsOutput JSON

**GIVEN** a completed simulation run with `--metrics-path out.json`
**WHEN** `out.json` is parsed
**THEN** it unmarshals as `sim.MetricsOutput` (has `instance_id` string field, `ttft_mean_ms` float64 field — distinguishable from `[]workload.SimResult` which has `request_id` int and `ttft_us` float64)
**AND** it does NOT unmarshal as `[]workload.SimResult`

### BC-4: `blis replay --results-path` continues to write []SimResult JSON

**GIVEN** a completed replay with `--results-path out.json`
**WHEN** `out.json` is parsed
**THEN** it unmarshals as `[]workload.SimResult` (has `request_id`, `ttft_us`, `e2e_us` fields)
**AND** `blis calibrate` can consume it (existing behavior, regression guard)

---

## Tasks

### Task 1 — Write failing behavioral tests for flag presence/absence

**What:** Write two table-driven tests in `cmd/root_test.go`:
1. `TestRunCmd_HasMetricsPathFlag` — asserts `runCmd` has `--metrics-path`, does NOT have `--results-path`
2. `TestReplayCmd_HasResultsPathFlag` — asserts `replayCmd` has `--results-path`, does NOT have `--metrics-path`

**Why first:** These tests will fail before the implementation and pass after — that's the TDD red/green proof that the rename actually happened.

**Exact test code to add in `cmd/root_test.go`:**

```go
// TestRunCmd_HasMetricsPathFlag verifies BC-1: blis run exposes --metrics-path,
// not --results-path.
func TestRunCmd_HasMetricsPathFlag(t *testing.T) {
	if runCmd.Flags().Lookup("metrics-path") == nil {
		t.Error("BC-1: runCmd missing --metrics-path flag")
	}
	if runCmd.Flags().Lookup("results-path") != nil {
		t.Error("BC-1: runCmd must NOT have --results-path flag (schema footgun)")
	}
}

// TestReplayCmd_HasResultsPathFlag verifies BC-2: blis replay exposes --results-path,
// not --metrics-path.
func TestReplayCmd_HasResultsPathFlag(t *testing.T) {
	if replayCmd.Flags().Lookup("results-path") == nil {
		t.Error("BC-2: replayCmd missing --results-path flag")
	}
	if replayCmd.Flags().Lookup("metrics-path") != nil {
		t.Error("BC-2: replayCmd must NOT have --metrics-path flag")
	}
}
```

**Run to confirm red:**
```bash
cd /Users/sri/Documents/Projects/inference-sim/.worktrees/pr-rename-metrics-path
go test ./cmd/... -run "TestRunCmd_HasMetricsPathFlag|TestReplayCmd_HasResultsPathFlag" -v
```
Expected: both tests FAIL (flag names not yet changed).

**Commit:** `test(cmd): add failing BC-1/BC-2 tests for metrics-path/results-path flag separation`

---

### Task 2 — Implement the flag rename

**What:** Four edits across two files.

#### Edit 1 — `cmd/root.go`: Add `metricsPath` var

In the `var (...)` block near line 152, alongside `resultsPath`, add:

```go
metricsPath  string // File to write MetricsOutput JSON for blis run (--metrics-path)
```

Update the existing `resultsPath` comment to:
```go
resultsPath  string // File to write []SimResult JSON for blis replay (--results-path)
```

#### Edit 2 — `cmd/root.go`: Remove `--results-path` from `registerSimConfigFlags`

Delete these two lines from `registerSimConfigFlags` (around line 935):
```go
// Results path
cmd.Flags().StringVar(&resultsPath, "results-path", "", "File to save BLIS results to")
```

> **Ordering constraint:** Edit 2 must be applied **before** Edit 4, or both applied **simultaneously** (same edit session). Do NOT apply Edit 4 before Edit 2: if Edit 4 replaces lines 258-260 first (while `registerSimConfigFlags` still registers `--results-path`), cobra panics at startup — duplicate flag name, two registrations. Conversely, applying Edit 2 alone (without Edit 4) panics because the old `Lookup("results-path")` at line 260 returns nil. The safest approach: apply both edits before any build or test run.

#### Edit 3 — `cmd/root.go`: Wire `metricsPath` into `runCmd` Run function and init

In `runCmd`'s Run function, find the line that uses `resultsPath` for saving metrics (around line 1449):
```go
if err := cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, resultsPath); err != nil {
```
Change `resultsPath` → `metricsPath`.

In `runCmd`'s `init()` function (near line 1658, after `registerSimConfigFlags(runCmd)`), add the new flag:
```go
runCmd.Flags().StringVar(&metricsPath, "metrics-path", "", "File to write MetricsOutput JSON (aggregate P50/P95/P99 TTFT, E2E, throughput stats). Use --results-path on blis replay for per-request SimResult JSON.")
```

#### Edit 4 — `cmd/replay.go`: Register `--results-path` explicitly and remove the bandaid

In `replayCmd`'s `init()` function, **replace** the two-line bandaid (lines 258–260):
```go
// Override --results-path description for replay: schema differs from blis run.
// blis run writes MetricsOutput JSON; blis replay writes []SimResult JSON.
replayCmd.Flags().Lookup("results-path").Usage = "File to write []SimResult JSON ..."
```
**With** a direct registration (no Lookup override needed):
```go
replayCmd.Flags().StringVar(&resultsPath, "results-path", "", "File to write []SimResult JSON (request_id, ttft_us, e2e_us, input_tokens, output_tokens) for blis calibrate consumption.")
```

> **Note on `resultsPath` variable:** The package-level `resultsPath` variable (`cmd/root.go:152`) is **retained** — it is still used by `replayCmd`'s Run function at `replay.go:238–247`. Only its flag registration moves from `registerSimConfigFlags` to `replayCmd`'s `init()`. Do NOT remove the `resultsPath` var.

#### Edit 5 — `cmd/replay_test.go`: Remove `--results-path` from `ParseFlags` args

`TestReplayCmd_EndToEnd_BlackboxMode` at line 513 passes `"--results-path", resultsFilePath` to `testCmd.ParseFlags`. After Edit 2, `registerSimConfigFlags(testCmd)` no longer registers `--results-path`, so `ParseFlags` will fail with "unknown flag: --results-path".

The fix is safe: the `resultsPath` package-level var is already set directly at line 480 (`resultsPath = resultsFilePath`). Remove the `--results-path` entry from the `ParseFlags` call:

```go
// BEFORE (line 513 area):
if err := testCmd.ParseFlags([]string{
    "--model", "test-model",
    "--latency-model", "blackbox",
    "--beta-coeffs", "10000.0,1.0,1.0",
    "--alpha-coeffs", "0.0,0.0,0.0",
    "--total-kv-blocks", "1000",
    "--trace-header", headerPath,
    "--trace-data", dataPath,
    "--results-path", resultsFilePath,  // ← REMOVE THIS LINE
}); err != nil {

// AFTER:
if err := testCmd.ParseFlags([]string{
    "--model", "test-model",
    "--latency-model", "blackbox",
    "--beta-coeffs", "10000.0,1.0,1.0",
    "--alpha-coeffs", "0.0,0.0,0.0",
    "--total-kv-blocks", "1000",
    "--trace-header", headerPath,
    "--trace-data", dataPath,
}); err != nil {
```

The `resultsPath = resultsFilePath` assignment at line 480 already ensures the per-request output is written; the `ParseFlags` entry was redundant and is now broken.

#### Edit 6 — `cmd/replay_test.go`: Update comment at line 61

After Edit 4, `--results-path` is no longer registered via `registerSimConfigFlags` on `replayCmd` — it is registered directly in `replayCmd`'s `init()`. Update the comment at `replay_test.go:61`:

```go
// BEFORE:
// registerSimConfigFlags: results
"results-path",

// AFTER:
// replay-specific: results
"results-path",
```

**Run to confirm green:**
```bash
go test ./cmd/... -run "TestRunCmd_HasMetricsPathFlag|TestReplayCmd_HasResultsPathFlag" -v
```
Expected: both tests PASS.

**Run full suite:**
```bash
go test ./... -count=1
```
Expected: all packages pass.

**Lint:**
```bash
golangci-lint run ./cmd/...
```
Expected: zero issues.

**Commit:** `fix(cmd): rename --results-path to --metrics-path on blis run (BC-1, BC-2, BC-4)`

---

### Task 3 — Write BC-3 regression test: `--metrics-path` writes MetricsOutput

**What:** Add a test in `cmd/root_test.go` that runs `runCmd` with `--metrics-path` and verifies the output file unmarshals as `MetricsOutput`.

**Why:** Without this, a future change could silently swap what `--metrics-path` writes without breaking BC-3. The flag presence test (BC-1) catches naming; this test catches schema.

**Test approach:** There is **no** existing `runCmd.Run` test in `cmd/root_test.go`. Use `TestReplayCmd_EndToEnd_BlackboxMode` (`cmd/replay_test.go:347`) as the structural template.

**Critical requirements — do not skip:**
1. **Use `testCmd + ParseFlags`**, not `runCmd.Run(runCmd, ...)` directly. The Run closure calls `resolveLatencyConfig(cmd)` which uses `cmd.Flags().Changed("latency-model")`. Without `ParseFlags`, `Changed` always returns false, causing roofline mode which needs HuggingFace config files on disk → `logrus.Fatalf` → test binary dies.
2. **Save and restore all ~28 package-level vars** mutated by the run (copy the save/restore block from `TestReplayCmd_EndToEnd_BlackboxMode:387-434`). Add `metricsPath` to the list. Missing any var causes state leak that silently breaks subsequent tests.
3. **Pass `--latency-model blackbox`** in `ParseFlags` to avoid roofline HF config fetch.

**Structural template (fill in full var list from replay_test.go:387-434):**

```go
// TestRunCmd_MetricsPath_WritesMetricsOutput verifies BC-3: --metrics-path on
// blis run produces MetricsOutput JSON (instance_id string ≠ SimResult request_id int).
// NOTE: Do NOT use t.Parallel() — mutates package-level vars.
func TestRunCmd_MetricsPath_WritesMetricsOutput(t *testing.T) {
	outFile := filepath.Join(t.TempDir(), "metrics.json")

	// --- Save and restore ALL package-level vars (copy from replay_test.go:387-434) ---
	origMetrics := metricsPath
	origModel := model
	origBackend := latencyModelBackend
	// ... copy all origXxx lines from TestReplayCmd_EndToEnd_BlackboxMode
	defer func() {
		metricsPath = origMetrics
		model = origModel
		latencyModelBackend = origBackend
		// ... restore all
	}()

	// --- Set run vars (must be explicit — runCmd.Run has Fatalf guards on zero/invalid values) ---
	metricsPath = outFile
	workloadType = "distribution"  // avoids preset-path Fatalf("Undefined workload")
	rate         = 1.0             // required by distribution rate-mode path
	numRequests  = 1               // minimal run
	// Token distribution defaults (Fatalf if min=0 or mean=0)
	promptTokensMean   = 512; promptTokensStdev  = 256
	promptTokensMin    = 2;   promptTokensMax    = 7000
	outputTokensMean   = 512; outputTokensStdev  = 256
	outputTokensMin    = 2;   outputTokensMax    = 7000

	// --- Build testCmd with Changed() tracking ---
	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	// Register run-only workload flags (--num-requests, --rate, etc.) that runCmd.Run reads
	testCmd.Flags().IntVar(&numRequests, "num-requests", 0, "")
	// ... add other run-only flags used in ParseFlags below
	if err := testCmd.ParseFlags([]string{
		"--model", "qwen/qwen3-14b",
		"--latency-model", "blackbox",  // avoid roofline HF config fetch
		"--beta-coeffs", "10000.0,1.0,1.0",
		"--alpha-coeffs", "0.0,0.0,0.0",
		"--total-kv-blocks", "1000",
		"--num-requests", "1",
		"--seed", "42",
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	runCmd.Run(testCmd, nil)

	data, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatalf("BC-3: metrics file not written: %v", err)
	}
	var out sim.MetricsOutput
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("BC-3: not MetricsOutput JSON: %v\nraw: %s", err, data)
	}
	// MetricsOutput.InstanceID is "cluster" (string); SimResult.RequestID is an int
	if out.InstanceID == "" {
		t.Error("BC-3: InstanceID empty — wrong schema or SaveResults not called")
	}
}
```

> **Implementation notes:**
> 1. **Save/restore list is incomplete in replay_test.go** — `TestReplayCmd_EndToEnd_BlackboxMode:387-434` saves ~28 shared SimConfig vars, but `runCmd.Run` also reads run-only workload vars NOT in that list: `workloadType`, `rate`, `numRequests`, `concurrency`, `thinkTimeMs`, `prefixTokens`, `promptTokensMean`, `promptTokensStdev`, `promptTokensMin`, `promptTokensMax`, `outputTokensMean`, `outputTokensStdev`, `outputTokensMin`, `outputTokensMax`, `workloadSpecPath`, `traceOutput`, `logLevel`. Add save/restore for all of these in addition to copying the replay template. If any is left unreset, subsequent tests may fail with corrupted state or crash with `logrus.Fatalf`.
> 2. **`metricsPath` is not in the replay template** — Add `origMetrics := metricsPath` / `metricsPath = origMetrics` explicitly (this var does not exist yet in the replay test).
> 3. **Initialization of run-only vars** — In the ParseFlags block, pass `--num-requests 1` (registered manually on testCmd) and `--latency-model blackbox` (skips HF fetch). Also ensure `rate`, `promptTokensMean`, `promptTokensMin`, `outputTokensMean`, `outputTokensMin` are set to valid non-zero values before `runCmd.Run` — the distribution code has `logrus.Fatalf` guards if token bounds are zero or invalid.
> 4. The skeleton above shows structural intent — do not implement from scratch.

**Run:**
```bash
go test ./cmd/... -run "TestRunCmd_MetricsPath_WritesMetricsOutput" -v
go test ./... -count=1
golangci-lint run ./cmd/...
```

**Commit:** `test(cmd): add BC-3 regression test for --metrics-path MetricsOutput schema`

---

### Task 4 — Update user-facing documentation

**What:** Five files need changes. All other mentions of `--results-path` in docs are in the context of `blis replay` or `blis calibrate`, where the flag name is correct and unchanged.

#### `docs/guide/results.md`

**Line 74** — replace `--results-path` with `--metrics-path`:
```
When `--metrics-path` is set, the JSON output includes a `per_model` key ...
```

**Line 135** — change the example invocation:
```bash
  --rate 100 --num-requests 500 --metrics-path results.json
```

#### `docs/reference/configuration.md`

**Line 98** — replace the `--results-path` row in the Simulation Control table. Add a "(blis run only)" qualifier because this flag is no longer in `registerSimConfigFlags` (no longer shared with `blis replay`):
```markdown
| `--metrics-path` | string | "" | File path to write MetricsOutput JSON (aggregate P50/P95/P99 TTFT, E2E, throughput stats). blis run only — blis replay uses `--results-path` instead. Empty = no file output. |
```

**Line 481** — in the Top-level SimConfig grouping row, replace `--results-path` with `--metrics-path`. Note: this row reflects flags registered via `registerSimConfigFlags`, which is shared by both `blis run` and `blis replay`. After this PR, `--results-path` is no longer in `registerSimConfigFlags` — it belongs exclusively to `blis replay`. Add a parenthetical to make this explicit:
```markdown
| **Top-level** | `--seed`, `--horizon`, `--log`, `--metrics-path` (run only), `--trace-output`, `--policy-config`, `--fitness-weights`, `--summarize-trace` |
```
(The `--results-path` flag for replay is documented in the Replay-Specific Flags table below.)

**Lines 543–558** — in the `blis replay` section, the current "Semantic Difference" callout documents the old footgun. Remove that entire section and add `--results-path` as a first-class row in the Replay-Specific Flags table:

Replace the existing Replay-Specific Flags table and Semantic Difference section:
```markdown
### Replay-Specific Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-header` | string | "" | Path to TraceV2 header YAML file (required). |
| `--trace-data` | string | "" | Path to TraceV2 data CSV file (required). |
| `--results-path` | string | "" | File to write `[]SimResult` JSON (fields: `request_id`, `ttft_us`, `e2e_us`, `input_tokens`, `output_tokens`) for `blis calibrate` consumption. |
```
(Remove the `### --results-path Semantic Difference` heading and the table under it entirely — they are no longer needed.)

#### Additional files with `blis run` context (also need updating)

**`docs/contributing/extension-recipes.md` line 150** — The recipe for adding per-request metric fields to `RequestMetrics` (part of `MetricsOutput`, written by `blis run`) says "appears in `--results-path` output". Change to `--metrics-path`:
```
To add a new field to per-request JSON output (appears in `--metrics-path` output):
```

**`specs/001-infra-nodes-gpus-instances/quickstart.md` line 126** — References `--results-path` in the context of `blis run`'s `per_model` key output (MetricsOutput). Change to `--metrics-path`:
```
When using `--metrics-path`, the output JSON includes a `per_model` key ...
```

**`docs/contributing/templates/hypothesis.md` line 131** — The hypothesis experiment template's `blis_run` shell wrapper uses `--results-path`. Change to `--metrics-path`:
```bash
#       --results-path "$RESULTS_DIR/config_a_results.json"
```
→
```bash
#       --metrics-path "$RESULTS_DIR/config_a_results.json"
```

**Files verified correct and unchanged** (all reference `blis replay` or `blis calibrate` which keep `--results-path`):
- `README.md` lines 137/141: `blis replay` context → correct, leave as-is
- `docs/guide/observe-replay-calibrate.md` lines 10/310: `blis replay` context → leave as-is
- `docs/concepts/architecture.md` line 261: `blis replay` mermaid arrow → leave as-is
- `docs/reference/project-structure.md` lines 16/17: `blis replay` and calibrate → leave as-is
- `cmd/calibrate.go` lines 27/159: reference `blis replay --results-path` → correct, leave as-is
- `cmd/replay.go` line 34 Long description: references `--results-path` for replay → correct, leave as-is
- `docs/plans/` files: historical planning artifacts → leave as-is

**Run after edits:**
```bash
go test ./... -count=1
golangci-lint run ./...
```
(Docs-only edits don't affect tests, but run the suite anyway to confirm no accidental changes.)

**Commit:** `docs: update --results-path → --metrics-path in user-facing docs (BC-1)`

---

## Sanity Checklist

Before marking this PR ready:

- [ ] `runCmd.Flags().Lookup("results-path")` returns nil (flag is gone from run)
- [ ] `runCmd.Flags().Lookup("metrics-path")` returns non-nil (new flag present)
- [ ] `replayCmd.Flags().Lookup("results-path")` returns non-nil (flag still on replay)
- [ ] `replayCmd.Flags().Lookup("metrics-path")` returns nil (not on replay)
- [ ] The Usage override / Lookup-based description hack in `replay.go` init() is gone
- [ ] `resultsPath` var is still used by replay's Run function unchanged
- [ ] `metricsPath` var is used by run's Run function in place of `resultsPath`
- [ ] `docs/guide/results.md` references `--metrics-path` (not `--results-path`) for `blis run`
- [ ] `docs/reference/configuration.md` Simulation Control table has `--metrics-path`
- [ ] `docs/reference/configuration.md` Top-level SimConfig row has `--metrics-path (run only)` note
- [ ] `docs/reference/configuration.md` Replay-Specific Flags table has `--results-path` as a first-class row
- [ ] `docs/reference/configuration.md` "Semantic Difference" section is removed
- [ ] `docs/contributing/extension-recipes.md:150` says `--metrics-path`
- [ ] `specs/001-infra-nodes-gpus-instances/quickstart.md:126` says `--metrics-path`
- [ ] `docs/contributing/templates/hypothesis.md:131` says `--metrics-path`
- [ ] `cmd/replay_test.go:61` comment updated from `// registerSimConfigFlags: results` to `// replay-specific: results` (Edit 6)
- [ ] `cmd/replay_test.go:513` no longer passes `"--results-path"` to ParseFlags (Edit 5)
- [ ] `go test ./... -count=1` — all green
- [ ] `golangci-lint run ./...` — zero issues
- [ ] `go build ./...` — clean build
