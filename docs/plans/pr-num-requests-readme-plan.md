# Rename --max-prompts to --num-requests + README Updates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `--workload-spec` silently ignoring `--max-prompts`, rename the flag to `--num-requests` for consistency, add a request count cap to the workload generator, and refresh the README.

**The problem today:** When using `--workload-spec`, the `--max-prompts` flag is silently ignored. The workload generator uses the simulation horizon to determine request count, and the default horizon is `math.MaxInt64` (~9.2x10^18 us). Running `--workload-spec examples/servegen-language.yaml --max-prompts 500` without `--horizon` causes the generator to attempt creating ~10^17 requests, hanging indefinitely. Additionally, the flag name "max-prompts" uses legacy language — these are requests, not prompts. The README intro paragraph and ServeGen workload section are also out of date.

**What this PR adds:**
1. **Request count cap for workload-spec** — `GenerateRequests` now accepts a `maxRequests` parameter that stops generation when the count is reached, whichever limit (count or horizon) is hit first.
2. **Consistent naming** — CLI flag renamed from `--max-prompts` to `--num-requests`, YAML field `num_requests` added to `WorkloadSpec`, Go variable renamed to `numRequests`.
3. **Unbounded generation guard** — if both `maxRequests == 0` and `horizon == MaxInt64`, the CLI refuses to run (prevents hangs).
4. **README refresh** — updated intro section, added ServeGen citation.

**Why this matters:** This is a usability bug that causes the CLI to hang silently, making `--workload-spec` unusable without a `--horizon` workaround.

**Architecture:** Pure CLI/config change. Touches `cmd/root.go` (flag rename + wiring), `sim/simulator.go` (struct field rename), `sim/workload/spec.go` (new field), `sim/workload/generator.go` (new parameter + count guard), `sim/internal/testutil/golden.go` (JSON tag rename), `testdata/goldendataset.json` (key rename), `README.md` (content updates), `CLAUDE.md` (flag reference updates).

**Source:** GitHub issue #228 + user request for README/ServeGen citation updates

**Closes:** Fixes #228

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a silent hang when using `--workload-spec` without `--horizon` by:
1. Adding a `maxRequests` parameter to the workload generator's `GenerateRequests` function
2. Renaming the CLI flag from `--max-prompts` to `--num-requests` (breaking change, no alias)
3. Adding a `num_requests` field to `WorkloadSpec` YAML
4. Guarding against unbounded generation at the CLI level

Adjacent blocks: `sim/workload/generator.go` (generation loop), `cmd/root.go` (CLI wiring), `sim/simulator.go` (`GuideLLMConfig` struct), `sim/cluster/workload.go` (distribution-based generation using `NumRequests`).

No deviations from the approved design in issue #228.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Request count cap in workload generator
- GIVEN a `WorkloadSpec` with `maxRequests=50` and a long horizon
- WHEN `GenerateRequests` is called
- THEN the returned slice contains at most 50 requests
- MECHANISM: A global counter across all clients; per-client loop breaks when `len(allRequests) >= maxRequests`

BC-2: Horizon-only generation preserved
- GIVEN `maxRequests=0` (unlimited) and a finite horizon
- WHEN `GenerateRequests` is called
- THEN generation stops at horizon as before (backward compatible)
- MECHANISM: Zero maxRequests means "no count limit" — only horizon applies

BC-3: CLI flag rename
- GIVEN a user running `./simulation_worker run --num-requests 500`
- WHEN the simulation runs
- THEN exactly 500 requests are generated (for distribution workload)
- MECHANISM: CLI variable renamed from `maxPrompts` to `numRequests`, flag from `--max-prompts` to `--num-requests`

BC-4: YAML `num_requests` field
- GIVEN a workload spec YAML with `num_requests: 200`
- WHEN `LoadWorkloadSpec` is called
- THEN `spec.NumRequests` equals 200
- MECHANISM: New `int64` field with `yaml:"num_requests,omitempty"` tag

BC-5: CLI overrides YAML `num_requests`
- GIVEN a workload spec YAML with `num_requests: 200` and CLI `--num-requests 500`
- WHEN the simulation runs
- THEN 500 requests are generated (CLI wins)
- MECHANISM: `cmd.Flags().Changed("num-requests")` check, same pattern as `--horizon`

BC-6: Whichever-limit-first semantics
- GIVEN `maxRequests=50` and a horizon that would produce 1000 requests
- WHEN `GenerateRequests` is called
- THEN exactly 50 requests are returned (count limit hit first)

**Negative Contracts:**

BC-7: Unbounded generation guard
- GIVEN `maxRequests == 0` AND `horizon == math.MaxInt64` AND `--workload-spec` is set
- WHEN the CLI validates inputs
- THEN `logrus.Fatalf` is called with a clear error message
- MECHANISM: Guard in `cmd/root.go` workload-spec path before calling `GenerateRequests`

BC-8: Determinism preserved
- GIVEN the same spec, horizon, and maxRequests
- WHEN `GenerateRequests` is called twice
- THEN identical results are returned
- MECHANISM: Count guard doesn't affect RNG state — it only stops early

**Error Handling Contracts:**

BC-9: Zero `num_requests` in YAML is valid
- GIVEN a workload spec YAML with `num_requests: 0` (or omitted)
- WHEN the spec is loaded and validated
- THEN validation passes (0 means "unlimited, use horizon only")
- MECHANISM: `int64` zero value = omitted = unlimited

### C) Component Interaction

```
cmd/root.go (CLI boundary)
    │
    ├─ resolves maxRequests: spec.NumRequests overridden by --num-requests if Changed()
    ├─ guards: fatalf if maxRequests==0 && horizon==MaxInt64 && workload-spec
    │
    └─► workload.GenerateRequests(spec, horizon, maxRequests)
            │
            ├─ per-client loop: break if len(allRequests) >= maxRequests
            └─ existing horizon check preserved
```

**API changes:**
- `GenerateRequests(spec *WorkloadSpec, horizon int64)` → `GenerateRequests(spec *WorkloadSpec, horizon int64, maxRequests int64)`
- `WorkloadSpec` gains `NumRequests int64` field
- `GuideLLMConfig.MaxPrompts` renamed to `NumRequests`

**Extension friction:** Adding `NumRequests` to `WorkloadSpec` touches 1 file (spec.go). Renaming `GuideLLMConfig.MaxPrompts` is a cross-cutting rename touching ~15 files but it's mechanical.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue says "Add `maxRequests` parameter to `GenerateRequests`" with `int64` type | Same | No deviation |
| Issue says "Rename variable `maxPrompts` → `numRequests`" | Also renames `GuideLLMConfig.MaxPrompts` field + golden dataset JSON keys + README references | ADDITION: Complete rename across all artifacts for consistency |
| Issue doesn't mention README intro/ServeGen citation | Plan includes README refresh + ServeGen BibTeX | ADDITION: User request in PR description |

### E) Review Guide

**The tricky part:** The `GenerateRequests` count guard must work across multiple clients without changing RNG state. The per-client RNG seeds are derived before the loop, so early termination is safe for determinism. But verify the guard doesn't introduce off-by-one errors (we want `<=maxRequests`, with truncation after sort+merge).

**What to scrutinize:** BC-7 (unbounded guard) — make sure the guard fires in the right code path (workload-spec only, not distribution). BC-5 (CLI override) — the `Changed()` pattern.

**What's safe to skim:** The rename (Task 1-2) is mechanical. Documentation (Task 6).

**Known debt:** `GuideLLMConfig` still uses `int` for `NumRequests` (not `int64`). The issue suggests `int64` for the workload-spec path but the distribution path uses `int`. We keep the `int` type for `GuideLLMConfig.NumRequests` to avoid a larger refactor of the distribution codepath.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/simulator.go` — rename `GuideLLMConfig.MaxPrompts` → `NumRequests`
- `sim/workload_config.go` — update field reference
- `sim/workload/spec.go` — add `NumRequests int64` field
- `sim/workload/generator.go` — add `maxRequests` parameter + count guard
- `sim/workload/generator_test.go` — update signature + add cap test
- `cmd/root.go` — rename flag, wiring, unbounded guard
- `cmd/default_config.go` — rename parameter
- `sim/internal/testutil/golden.go` — rename JSON tag
- `testdata/goldendataset.json` — rename JSON keys
- `examples/servegen-language.yaml` — add `num_requests: 500`
- `examples/weighted-routing.yaml` — rename `--max-prompts` in comments
- `README.md` — intro update + ServeGen citation
- `CLAUDE.md` — flag reference updates
- Multiple test files — rename `MaxPrompts` → `NumRequests`

**Key decisions:**
- Breaking rename with no alias (per issue #228 approved design)
- `maxRequests=0` means unlimited (backward compat)
- Count guard applied after per-client generation but before sort (simpler, same result since we truncate)

### G) Task Breakdown

---

### Task 1: Rename GuideLLMConfig.MaxPrompts to NumRequests

**Contracts Implemented:** BC-3 (partially — struct rename)

**Files:**
- Modify: `sim/simulator.go:60`
- Modify: `sim/workload_config.go:128`
- Modify: `cmd/root.go:40,232,241,522`
- Modify: `cmd/default_config.go:52,67`
- Modify: `sim/internal/testutil/golden.go:26`
- Modify: `sim/simulator_test.go` (multiple lines)
- Modify: `sim/cluster/workload.go:42`
- Modify: `sim/cluster/workload_test.go`
- Modify: `sim/cluster/cluster_test.go`
- Modify: `sim/cluster/cluster_trace_test.go`
- Modify: `sim/cluster/instance_test.go`
- Modify: `sim/cluster/pending_requests_test.go`
- Modify: `sim/cluster/evaluation_test.go`

**Step 1: Rename the struct field and all Go references**

In `sim/simulator.go`:
```go
// Change line 60 from:
MaxPrompts         int     // Number of requests
// To:
NumRequests        int     // Number of requests
```

In `sim/workload_config.go`:
```go
// Change line 128 from:
for currentTime < sim.Horizon && reqIdx < sim.guideLLMConfig.MaxPrompts {
// To:
for currentTime < sim.Horizon && reqIdx < sim.guideLLMConfig.NumRequests {
```

In `cmd/root.go`:
```go
// Change line 40 from:
maxPrompts                int       // Number of requests
// To:
numRequests               int       // Number of requests

// Change line 232 from:
guideLLMConfig = &sim.GuideLLMConfig{Rate: rate / 1e6, MaxPrompts: maxPrompts,
// To:
guideLLMConfig = &sim.GuideLLMConfig{Rate: rate / 1e6, NumRequests: numRequests,

// Change line 241 from:
guideLLMConfig = GetWorkloadConfig(defaultsFilePath, workloadType, rate/1e6, maxPrompts)
// To:
guideLLMConfig = GetWorkloadConfig(defaultsFilePath, workloadType, rate/1e6, numRequests)

// Change line 522 from:
runCmd.Flags().IntVar(&maxPrompts, "max-prompts", 100, "Number of requests")
// To:
runCmd.Flags().IntVar(&numRequests, "num-requests", 100, "Number of requests to generate")
```

In `cmd/default_config.go`:
```go
// Change line 52 from:
func GetWorkloadConfig(workloadFilePath string, workloadType string, rate float64, maxPrompts int) *sim.GuideLLMConfig {
// To:
func GetWorkloadConfig(workloadFilePath string, workloadType string, rate float64, numRequests int) *sim.GuideLLMConfig {

// Change line 67 from:
return &sim.GuideLLMConfig{Rate: rate, MaxPrompts: maxPrompts,
// To:
return &sim.GuideLLMConfig{Rate: rate, NumRequests: numRequests,
```

In `sim/internal/testutil/golden.go`:
```go
// Change line 26 from:
MaxPrompts                int           `json:"max-prompts"`
// To:
NumRequests               int           `json:"num-requests"`
```

In `sim/cluster/workload.go`:
```go
// Change line 42 from:
for currentTime < horizon && reqIdx < cfg.MaxPrompts {
// To:
for currentTime < horizon && reqIdx < cfg.NumRequests {
```

Then rename `MaxPrompts` → `NumRequests` in all test file construction sites:
- `sim/simulator_test.go`: all `MaxPrompts:` → `NumRequests:`
- `sim/cluster/cluster_test.go`: all `MaxPrompts:` → `NumRequests:` (including `newTestWorkload`)
- `sim/cluster/instance_test.go`: all `MaxPrompts:` → `NumRequests:` (including `smallWorkload`)
- `sim/cluster/cluster_trace_test.go`: all `MaxPrompts:` → `NumRequests:`
- `sim/cluster/pending_requests_test.go`: all `MaxPrompts:` → `NumRequests:`
- `sim/cluster/evaluation_test.go`: all `MaxPrompts:` → `NumRequests:`
- `sim/cluster/workload_test.go`: all `MaxPrompts:` → `NumRequests:`

**Step 2: Update golden dataset JSON keys**

In `testdata/goldendataset.json`, for ALL test entries:
- Replace `"max-prompts"` with `"num-requests"` (field key)
- In `"blis-cmd"` strings: replace `--max-prompts` with `--num-requests`

**Step 3: Run tests to verify rename is complete**

Run: `go test ./... -count=1`
Expected: All tests PASS (pure rename, no behavioral change)

**Step 4: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(cmd): rename --max-prompts to --num-requests (BC-3)

- Rename GuideLLMConfig.MaxPrompts to NumRequests across all packages
- Rename CLI flag --max-prompts to --num-requests (breaking, no alias)
- Rename CLI variable maxPrompts to numRequests
- Update golden dataset JSON keys and blis-cmd strings
- Update GoldenTestCase struct tag

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add NumRequests to WorkloadSpec + YAML support

**Contracts Implemented:** BC-4, BC-9

**Files:**
- Modify: `sim/workload/spec.go`
- Modify: `sim/workload/spec_test.go`

**Step 1: Write failing test for NumRequests YAML parsing**

In `sim/workload/spec_test.go`, add:
```go
func TestWorkloadSpec_NumRequests_ParsedFromYAML(t *testing.T) {
	// BC-4: YAML num_requests field is parsed
	yamlData := `
version: "1"
seed: 42
category: language
aggregate_rate: 10.0
num_requests: 200
clients:
  - id: "c1"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 100
    output_distribution:
      type: exponential
      params:
        mean: 50
`
	spec, err := parseWorkloadSpecFromBytes([]byte(yamlData))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.NumRequests != 200 {
		t.Errorf("NumRequests = %d, want 200", spec.NumRequests)
	}
}

func TestWorkloadSpec_NumRequestsOmitted_DefaultsToZero(t *testing.T) {
	// BC-9: omitted num_requests defaults to 0 (unlimited)
	yamlData := `
version: "1"
seed: 42
category: language
aggregate_rate: 10.0
clients:
  - id: "c1"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 100
    output_distribution:
      type: exponential
      params:
        mean: 50
`
	spec, err := parseWorkloadSpecFromBytes([]byte(yamlData))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.NumRequests != 0 {
		t.Errorf("NumRequests = %d, want 0 (default unlimited)", spec.NumRequests)
	}
}
```

Also add the helper function if it doesn't exist:
```go
func parseWorkloadSpecFromBytes(data []byte) (*WorkloadSpec, error) {
	var spec WorkloadSpec
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&spec); err != nil {
		return nil, err
	}
	return &spec, nil
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run TestWorkloadSpec_NumRequests -v`
Expected: FAIL (NumRequests field doesn't exist yet, or YAML strict parsing rejects unknown key `num_requests`)

**Step 3: Add NumRequests field to WorkloadSpec**

In `sim/workload/spec.go`, add to `WorkloadSpec` struct after `Horizon`:
```go
NumRequests   int64        `yaml:"num_requests,omitempty"` // 0 = unlimited (use horizon only)
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/workload/... -run TestWorkloadSpec_NumRequests -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/spec.go sim/workload/spec_test.go
git commit -m "feat(workload): add NumRequests field to WorkloadSpec (BC-4, BC-9)

- Add num_requests YAML field (int64, omitempty, 0=unlimited)
- Add YAML parsing tests for explicit and omitted values

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add maxRequests parameter to GenerateRequests + count guard

**Contracts Implemented:** BC-1, BC-2, BC-6, BC-8

**Files:**
- Modify: `sim/workload/generator.go`
- Modify: `sim/workload/generator_test.go`
- Modify: `sim/workload/servegen_test.go`
- Modify: `cmd/root.go:211`

**Step 1: Write failing test for request count cap**

In `sim/workload/generator_test.go`, add:
```go
func TestGenerateRequests_MaxRequests_CapsOutput(t *testing.T) {
	// BC-1, BC-6: maxRequests caps total output even with long horizon
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	horizon := int64(100e6) // 100 seconds — would produce ~10000 requests
	maxReqs := int64(50)

	requests, err := GenerateRequests(spec, horizon, maxReqs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if int64(len(requests)) > maxReqs {
		t.Errorf("len(requests) = %d, want <= %d", len(requests), maxReqs)
	}
	if len(requests) == 0 {
		t.Error("expected at least some requests")
	}
}

func TestGenerateRequests_ZeroMaxRequests_UsesHorizonOnly(t *testing.T) {
	// BC-2: maxRequests=0 means unlimited — horizon controls
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	horizon := int64(1e6) // 1 second

	requests, err := GenerateRequests(spec, horizon, 0) // 0 = unlimited
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Error("expected requests with unlimited maxRequests and finite horizon")
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run "TestGenerateRequests_MaxRequests|TestGenerateRequests_ZeroMaxRequests" -v`
Expected: FAIL (wrong number of arguments to GenerateRequests)

**Step 3: Update GenerateRequests signature and add count guard**

In `sim/workload/generator.go`:

Update function signature:
```go
// GenerateRequests creates a request sequence from a WorkloadSpec.
// Deterministic given the same spec, seed, and maxRequests.
// maxRequests caps the total number of requests (0 = unlimited, use horizon only).
// Returns requests sorted by ArrivalTime with sequential IDs.
func GenerateRequests(spec *WorkloadSpec, horizon int64, maxRequests int64) ([]*sim.Request, error) {
```

Add count guard inside the per-client loop. Replace the existing loop (line 86):
```go
		// Generate requests for this client
		currentTime := int64(0)
		for currentTime < horizon {
			// Count guard: stop if we've reached the global cap
			if maxRequests > 0 && int64(len(allRequests)) >= maxRequests {
				break
			}

			iat := arrivalSampler.SampleIAT(clientRNG)
```

Also add the same guard before appending reasoning requests (after the `GenerateReasoningRequests` call):
```go
			if err != nil {
				return nil, fmt.Errorf("client %q reasoning: %w", client.ID, err)
			}
			// Apply count cap to reasoning requests too
			if maxRequests > 0 && int64(len(allRequests)+len(reasoningReqs)) > maxRequests {
				remaining := maxRequests - int64(len(allRequests))
				if remaining > 0 {
					reasoningReqs = reasoningReqs[:remaining]
				} else {
					reasoningReqs = nil
				}
			}
			allRequests = append(allRequests, reasoningReqs...)
```

**Step 4: Update all existing callers**

In `cmd/root.go` line 211, update the call:
```go
// Temporary: pass 0 (unlimited) — proper wiring in Task 4
reqs, err := workload.GenerateRequests(spec, simulationHorizon, 0)
```

In all existing test calls in `sim/workload/generator_test.go` and `sim/workload/servegen_test.go`, add `, 0` as the third argument to `GenerateRequests`:
- `GenerateRequests(spec, horizon)` → `GenerateRequests(spec, horizon, 0)`
- `GenerateRequests(spec, 10e6)` → `GenerateRequests(spec, 10e6, 0)`
- `GenerateRequests(spec, 1e6)` → `GenerateRequests(spec, 1e6, 0)`
- `GenerateRequests(spec, 0)` → `GenerateRequests(spec, 0, 0)`

**Step 5: Run tests**

Run: `go test ./... -count=1`
Expected: All PASS

**Step 6: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/workload/generator.go sim/workload/generator_test.go sim/workload/servegen_test.go cmd/root.go
git commit -m "feat(workload): add maxRequests parameter to GenerateRequests (BC-1, BC-2, BC-6, BC-8)

- Add maxRequests int64 parameter (0=unlimited, uses horizon only)
- Add per-client count guard that breaks when global cap reached
- Cap reasoning requests to remaining budget
- Add tests for count cap and unlimited behaviors

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: CLI wiring — precedence rules + unbounded generation guard

**Contracts Implemented:** BC-5, BC-7

**Files:**
- Modify: `cmd/root.go`
- Modify: `cmd/root_test.go` (if exists, otherwise behavioral tests via integration)

**Step 1: Implement CLI wiring and unbounded guard**

In `cmd/root.go`, replace the workload-spec block (lines 199-219):

```go
		// --workload-spec takes precedence over --workload if set
		if workloadSpecPath != "" {
			spec, err := workload.LoadWorkloadSpec(workloadSpecPath)
			if err != nil {
				logrus.Fatalf("Failed to load workload spec: %v", err)
			}
			if err := spec.Validate(); err != nil {
				logrus.Fatalf("Invalid workload spec: %v", err)
			}
			// Apply spec horizon as default; CLI --horizon flag overrides via Changed().
			if spec.Horizon > 0 && !cmd.Flags().Changed("horizon") {
				simulationHorizon = spec.Horizon
			}

			// Resolve maxRequests: spec.NumRequests as default, CLI --num-requests overrides
			maxRequests := spec.NumRequests
			if cmd.Flags().Changed("num-requests") {
				maxRequests = int64(numRequests)
			}

			// BC-7: Guard against unbounded generation
			if maxRequests <= 0 && simulationHorizon == math.MaxInt64 {
				logrus.Fatalf("--workload-spec requires either num_requests (in YAML or --num-requests) or --horizon to bound generation")
			}

			reqs, err := workload.GenerateRequests(spec, simulationHorizon, maxRequests)
			if err != nil {
				logrus.Fatalf("Failed to generate workload: %v", err)
			}
			preGeneratedRequests = reqs
			// Set a placeholder GuideLLMConfig to satisfy constructor validation.
			// The actual requests come from preGeneratedRequests.
			guideLLMConfig = &sim.GuideLLMConfig{Rate: spec.AggregateRate / 1e6}
			logrus.Infof("Generated %d requests from workload spec", len(reqs))
```

**Step 2: Run tests**

Run: `go test ./... -count=1`
Expected: All PASS

**Step 3: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 4: Commit**

```bash
git add cmd/root.go
git commit -m "fix(cmd): wire --num-requests to workload-spec path with unbounded guard (BC-5, BC-7)

- Resolve maxRequests from spec.NumRequests, CLI --num-requests overrides
- Guard: fatalf when both maxRequests=0 and horizon=MaxInt64
- Pass resolved maxRequests to GenerateRequests

Fixes #228

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Update example YAML + README + ServeGen citation

**Contracts Implemented:** (documentation)

**Files:**
- Modify: `examples/servegen-language.yaml`
- Modify: `README.md`

**Step 1: Add num_requests to example YAML**

In `examples/servegen-language.yaml`, add `num_requests: 500` after the `aggregate_rate` line:

```yaml
version: "1"
seed: 42
category: language
aggregate_rate: 100.0  # 100 requests/second total
num_requests: 500      # Generate at most 500 requests
```

**Step 2: Update README intro section**

Replace the current intro (lines 1-7) with:

```markdown
# Blackbox Inference Simulator (BLIS)

A discrete-event simulator for LLM inference serving systems. BLIS models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients or analytical roofline estimates.

The simulator is CPU-only, deterministic, and designed for capacity planning, policy optimization research, and performance prediction across model/GPU/TP configurations without requiring real GPUs.
```

**Step 3: Add ServeGen citation**

In the README, after the `ServeGen-Informed Workload Generation` section (after the line about `examples/servegen-language.yaml`), add:

```markdown
The workload generation module is informed by the ServeGen characterization framework:

> Xiang et al., "ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production," arXiv:2505.09999, 2025. [[paper](https://arxiv.org/abs/2505.09999)] [[code](https://github.com/alibaba/ServeGen)]
```

**Step 4: Update README CLI reference and all --max-prompts references**

In README, replace all `--max-prompts` references with `--num-requests`:
- Line 107: `--max-prompts 300` → `--num-requests 300`
- Line 180, 188: `--max-prompts 500` → `--num-requests 500`
- Line 616: CLI reference table row for `--max-prompts` → `--num-requests`
- `examples/weighted-routing.yaml` comment lines: `--max-prompts` → `--num-requests`

**Step 5: Run build to verify no broken references**

Run: `go build ./...`
Expected: Build succeeds

**Step 6: Commit**

```bash
git add examples/servegen-language.yaml examples/weighted-routing.yaml README.md
git commit -m "docs: update README intro, add ServeGen citation, rename --max-prompts references

- Refresh intro to reflect multi-instance cluster capabilities
- Add ServeGen paper citation with arXiv link
- Rename all --max-prompts to --num-requests in docs and examples
- Add num_requests: 500 to servegen-language.yaml example

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update CLAUDE.md + documentation references

**Contracts Implemented:** (documentation)

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

In `CLAUDE.md`:
- Replace all `--max-prompts` references with `--num-requests`
- In the CLI flags section of "Build and Run Commands", update the example:
  `--rate 10 --max-prompts 100` → `--rate 10 --num-requests 100`
- In the File Organization section, verify `generator.go` description is current

**Step 2: Grep for any remaining stale references**

Run: `grep -r "max-prompts\|max_prompts\|MaxPrompts\|maxPrompts" --include="*.go" --include="*.yaml" --include="*.json" .`
Expected: Zero matches in source code (*.go, *.yaml, *.json). Frozen plan docs (docs/plans/*.md) will still contain historical references — those should NOT be updated.

**Step 3: Run full test suite**

Run: `go test ./... -count=1`
Expected: All PASS

**Step 4: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for --num-requests rename

- Replace all --max-prompts references with --num-requests
- Update example commands

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1     | Task 3 | Unit | TestGenerateRequests_MaxRequests_CapsOutput |
| BC-2     | Task 3 | Unit | TestGenerateRequests_ZeroMaxRequests_UsesHorizonOnly |
| BC-3     | Task 1 | Regression | All existing tests pass after rename |
| BC-4     | Task 2 | Unit | TestWorkloadSpec_NumRequests_ParsedFromYAML |
| BC-5     | Task 4 | Integration | Verified by CLI wiring (manual test: `--workload-spec ... --num-requests 50`) |
| BC-6     | Task 3 | Unit | TestGenerateRequests_MaxRequests_CapsOutput (subsumes) |
| BC-7     | Task 4 | Integration | Guard in cmd/root.go (CLI boundary — testing requires process-level invocation; deferred to PR16 integration tests) |
| BC-8     | Task 3 | Unit | Existing TestGenerateRequests_Deterministic_SameSeedSameOutput (with maxRequests=0) |
| BC-9     | Task 2 | Unit | TestWorkloadSpec_NumRequestsOmitted_DefaultsToZero |

**Golden dataset update:** Task 1 renames JSON keys only — values unchanged, no regeneration needed.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Missed rename site causes compilation error | Low | Low | `go build ./...` after Task 1 catches all | Task 1 |
| Count guard changes RNG sequence | Low | High | Guard only breaks loop early, doesn't skip RNG calls. Existing determinism test validates. | Task 3 |
| Golden dataset key rename breaks CI | Low | Medium | Task 1 updates both struct tag and JSON file atomically | Task 1 |
| Stale `--max-prompts` in user scripts | Medium | Low | Breaking change is intentional (issue #228 approved design) | N/A |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions.
- [x] No feature creep beyond PR scope.
- [x] No unexercised flags or interfaces.
- [x] No partial implementations.
- [x] No breaking changes without explicit contract updates. (Breaking rename is intentional per #228)
- [x] No hidden global state impact.
- [x] All new code will pass golangci-lint.
- [x] Shared test helpers used from existing shared test package (not duplicated locally).
- [x] CLAUDE.md updated if: new files/packages added, file organization changed, plan milestone completed, new CLI flags added. (Flag renamed)
- [x] No stale references left in CLAUDE.md.
- [x] Deviation log reviewed — no unresolved deviations.
- [x] Each task produces working, testable code (no scaffolding).
- [x] Task dependencies are correctly ordered.
- [x] All contracts are mapped to specific tasks.
- [x] Golden dataset regeneration documented (if needed). (Key rename only, no value change)
- [x] Construction site audit completed — `WorkloadSpec{}` constructed in ~20 places (tests, scenarios, plan docs). Only spec.go needs the new field; test sites use zero-value (unlimited) which is correct default.
- [x] Every new CLI flag validated for: zero, negative, NaN, Inf, empty string. (No new numeric flag — renamed existing one. Validation unchanged.)
- [x] Every error path either returns error, panics with context, or increments a counter.
- [x] No map iteration feeds float accumulation without sorted keys.
- [x] Library code never calls logrus.Fatalf. (Guard is in cmd/root.go only)
- [x] No exported mutable maps.
- [x] YAML config uses strict parsing (existing).
- [x] Grepped for references to this PR number — N/A (no PR number yet).

---

## Appendix: File-Level Implementation Details

### File: `sim/simulator.go`

**Purpose:** Rename `MaxPrompts` field to `NumRequests` in `GuideLLMConfig`.

**Change:** Line 60: `MaxPrompts int` → `NumRequests int`

### File: `sim/workload/spec.go`

**Purpose:** Add `NumRequests` field to `WorkloadSpec`.

**Change:** Add after `Horizon` field:
```go
NumRequests   int64        `yaml:"num_requests,omitempty"` // 0 = unlimited (use horizon only)
```

### File: `sim/workload/generator.go`

**Purpose:** Add `maxRequests` parameter and count guard.

**Signature change:**
```go
func GenerateRequests(spec *WorkloadSpec, horizon int64, maxRequests int64) ([]*sim.Request, error)
```

**Negative guard** at function top (after horizon check):
```go
if maxRequests < 0 {
    return nil, fmt.Errorf("maxRequests must be non-negative, got %d", maxRequests)
}
```

**Count guard** in per-client loop (before IAT sampling):
```go
if maxRequests > 0 && int64(len(allRequests)) >= maxRequests {
    break
}
```

### File: `cmd/root.go`

**Purpose:** Rename flag, wire precedence, add unbounded guard.

**Key changes:**
1. Variable: `maxPrompts` → `numRequests`
2. Flag: `--max-prompts` → `--num-requests`
3. Workload-spec path: resolve `maxRequests` from spec + CLI override
4. Guard: `logrus.Fatalf` if unbounded

### File: `sim/internal/testutil/golden.go`

**Purpose:** Rename JSON tag for golden dataset struct.

**Change:** Line 26: `json:"max-prompts"` → `json:"num-requests"`

### File: `testdata/goldendataset.json`

**Purpose:** Rename JSON keys.

**Change:** All `"max-prompts"` keys → `"num-requests"`, all `--max-prompts` in `blis-cmd` strings → `--num-requests`

### File: `examples/servegen-language.yaml`

**Purpose:** Add `num_requests` so example works out of the box.

**Change:** Add `num_requests: 500` after `aggregate_rate`

### File: `README.md`

**Purpose:** Refresh intro, add ServeGen citation, rename flag references.

**Changes:**
1. Replace intro paragraph (lines 3-6)
2. Add ServeGen citation after workload generation section
3. Replace all `--max-prompts` with `--num-requests`

### File: `CLAUDE.md`

**Purpose:** Update flag references.

**Change:** Replace all `--max-prompts` with `--num-requests`
