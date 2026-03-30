# Plan: Add --workload Preset Support to blis observe

**Goal:** Add `--workload <preset>` to `blis observe` so real servers can be driven with
the same named workload presets as `blis run` (chatbot, summarization, contentgen, multidoc).

**Source:** GitHub Issue #865

**Closes:** #865

**PR size tier:** Small (2 files modified: `cmd/observe_cmd.go`, `cmd/observe_cmd_test.go`)

**Read-only dependency:** `cmd/convert.go` — provides `loadPresetWorkload` (same `cmd` package, no changes needed)

**Deviation Log:**

| ID | Source says | Plan does | Reason |
|----|------------|-----------|--------|
| D-1 | "`--workload`, `--workload-spec`, and `--rate` remain mutually exclusive" | `--workload` REQUIRES `--rate`; it is mutually exclusive with `--workload-spec` and `--concurrency` only | CORRECTION: preset mode calls `SynthesizeFromPreset(rate, ...)` — `rate` is a required parameter, not an excluded one. This matches `blis run` behavior (root.go:1160). |
| D-2 | No mention of `--defaults-filepath` | Adds `--defaults-filepath` flag (default `"defaults.yaml"`) to `observeCmd` | CORRECTION: loading a preset requires reading `defaults.yaml`; without this flag the preset path cannot work. Follows `runCmd` pattern exactly. |
| D-3 | `--workload` described as equivalent to run's flag | `--workload` on observe does NOT accept `"distribution"` | CLARIFICATION: `"distribution"` is already served by `--rate` on observe; accepting it would be a silent no-op. Valid preset names: chatbot, summarization, contentgen, multidoc. |

---

## Behavioral Contracts

**BC-1: Preset-spec parity**
GIVEN `blis observe --workload chatbot --rate 5 --num-requests 100 …`
WHEN the command builds the WorkloadSpec
THEN the spec has the same preset distribution shape as `blis run --workload chatbot --rate 5 --num-requests 100` (same PromptTokensMean, PromptTokensStdev, OutputTokensMean, OutputTokensStdev, PrefixTokens — all loaded from defaults.yaml under key "chatbot")

**BC-2: Preset requires `--rate`**
GIVEN `blis observe --workload chatbot` (no `--rate`)
WHEN the command validates flags
THEN the command exits with a fatal error containing "requires --rate"

**BC-3: Preset exclusive with `--workload-spec`**
GIVEN `blis observe --workload chatbot --rate 5 --workload-spec spec.yaml …`
WHEN the command validates flags
THEN the command exits with a fatal error containing "--workload and --workload-spec are mutually exclusive"

**BC-4: Preset exclusive with `--concurrency`**
GIVEN `blis observe --workload chatbot --concurrency 50 …`
WHEN the command validates flags
THEN the command exits with a fatal error containing "--workload and --concurrency are mutually exclusive"

**BC-5: Unknown preset name rejected**
GIVEN `blis observe --workload unknown-preset --rate 5 …`
WHEN the command runs
THEN the command exits with a fatal error listing valid preset names (chatbot, summarization, contentgen, multidoc)

**BC-6: Flag registration**
GIVEN the `observe` command
THEN it has a `--workload` flag (default `""`) and a `--defaults-filepath` flag (default `"defaults.yaml"`)

**BC-7: Workload input guard updated**
GIVEN `blis observe` with none of `--workload`, `--workload-spec`, `--rate`, `--concurrency` provided
THEN the command exits with a fatal error listing all four input modes

---

## Design Note: `buildPresetSpec` Helper

The preset synthesis logic is extracted into an unexported helper `buildPresetSpec`:

```go
func buildPresetSpec(preset, defaultsPath string, rate float64, numRequests int) (*workload.WorkloadSpec, string)
```

It returns `(spec, "")` on success, or `(nil, errMsg)` on unknown preset.
`runObserve` calls it and then `logrus.Fatalf`s on non-empty errMsg.

**Why extract it:** `logrus.Fatalf` calls `os.Exit`, making the synthesis path in `runObserve`
impossible to unit-test directly. Extracting the preset-name-resolution logic into a helper
allows BC-1 and BC-5 to be tested without process exit. Same pattern as
`validateObserveWorkloadFlags` (also extracted for testability per R14).

**Scope of testability:** `buildPresetSpec` catches only "unknown preset name" errors. File I/O
and YAML parse errors from `loadPresetWorkload` → `loadDefaultsConfig` call `logrus.Fatalf`
directly — those are CLI-boundary fatals, consistent with all other defaults.yaml reads in
this codebase (e.g., `blis run --workload chatbot`).

---

## Tasks

All tasks execute inside `.worktrees/pr-observe-workload-preset/`. Every task follows TDD:
write failing test → run to confirm failure → implement → run to confirm pass → lint.

```bash
go test ./cmd/... -run <TestName> -v    # test command
golangci-lint run ./cmd/...             # lint command
```

---

### Task 1 — Flag Registration Tests + Flags

**Step 1.1 — Write failing tests (BC-6)**

Add to `cmd/observe_cmd_test.go`:

```go
func TestObserveCmd_WorkloadFlag_Exists(t *testing.T) {
	f := observeCmd.Flags().Lookup("workload")
	if f == nil {
		t.Fatal("missing expected flag --workload on observeCmd")
	}
	if f.DefValue != "" {
		t.Errorf("--workload default: got %q, want %q (empty — no default preset)", f.DefValue, "")
	}
}

func TestObserveCmd_DefaultsFilepathFlag_Exists(t *testing.T) {
	f := observeCmd.Flags().Lookup("defaults-filepath")
	if f == nil {
		t.Fatal("missing expected flag --defaults-filepath on observeCmd")
	}
	if f.DefValue != "defaults.yaml" {
		t.Errorf("--defaults-filepath default: got %q, want %q", f.DefValue, "defaults.yaml")
	}
}
```

**Step 1.2 — Run to confirm failure**

```
go test ./cmd/... -run "TestObserveCmd_WorkloadFlag_Exists|TestObserveCmd_DefaultsFilepathFlag_Exists" -v
# Expected: FAIL — missing expected flag --workload on observeCmd
```

**Step 1.3 — Implement: add variables and flags**

In `cmd/observe_cmd.go`, add to the `var (...)` block (after `observeThinkTimeMs`):

```go
observeWorkload         string
observeDefaultsFilePath string
```

In `init()`, after the `--workload-spec` line:

```go
observeCmd.Flags().StringVar(&observeWorkload, "workload", "", "Workload preset name (chatbot, summarization, contentgen, multidoc); requires --rate")
observeCmd.Flags().StringVar(&observeDefaultsFilePath, "defaults-filepath", "defaults.yaml", "Path to defaults.yaml (for preset workload definitions)")
```

**Step 1.4 — Run to confirm pass**

```
go test ./cmd/... -run "TestObserveCmd_WorkloadFlag_Exists|TestObserveCmd_DefaultsFilepathFlag_Exists" -v
# Expected: PASS
```

**Step 1.5 — Lint**

```
golangci-lint run ./cmd/...
# Expected: 0 issues
```

**Step 1.6 — Commit**

```
git add cmd/observe_cmd.go cmd/observe_cmd_test.go
git commit -m "test(cmd): add flag-existence tests for --workload and --defaults-filepath on observe

Implements BC-6. Tests fail until flags are registered in observe_cmd.go init()."
```

---

### Task 2 — Validation Tests + Validation Logic

**Step 2.1 — Write failing tests (BC-2, BC-3, BC-4, BC-7)**

Add to `cmd/observe_cmd_test.go` (needs `"strings"` — already imported):

```go
func TestValidateObserveWorkloadFlags_PresetRequiresRate(t *testing.T) {
	// BC-2: preset without --rate is rejected
	msg := validateObserveWorkloadFlags("chatbot", "", false, 0)
	if msg == "" {
		t.Fatal("expected validation error for preset without --rate, got none")
	}
	if !strings.Contains(msg, "--rate") {
		t.Errorf("error should mention --rate, got: %q", msg)
	}
}

func TestValidateObserveWorkloadFlags_PresetExclusiveWithSpec(t *testing.T) {
	// BC-3: preset + --workload-spec is rejected
	msg := validateObserveWorkloadFlags("chatbot", "spec.yaml", true, 0)
	if msg == "" {
		t.Fatal("expected validation error for --workload + --workload-spec, got none")
	}
	if !strings.Contains(msg, "--workload-spec") {
		t.Errorf("error should mention --workload-spec, got: %q", msg)
	}
}

func TestValidateObserveWorkloadFlags_PresetExclusiveWithConcurrency(t *testing.T) {
	// BC-4: preset + --concurrency is rejected
	msg := validateObserveWorkloadFlags("chatbot", "", true, 50)
	if msg == "" {
		t.Fatal("expected validation error for --workload + --concurrency, got none")
	}
	if !strings.Contains(msg, "--concurrency") {
		t.Errorf("error should mention --concurrency, got: %q", msg)
	}
}

func TestValidateObserveWorkloadFlags_ValidPreset_Accepted(t *testing.T) {
	// Precondition for BC-1: valid preset + rate is accepted by validator
	msg := validateObserveWorkloadFlags("chatbot", "", true, 0)
	if msg != "" {
		t.Errorf("expected no error for valid preset+rate combination, got: %q", msg)
	}
}

func TestValidateObserveWorkloadFlags_EmptyPreset_NoOp(t *testing.T) {
	// Non-preset modes must not be affected by the validator
	msg := validateObserveWorkloadFlags("", "", false, 0)
	if msg != "" {
		t.Errorf("expected no error when --workload is not set, got: %q", msg)
	}
}

// TestObserveCmd_WorkloadInputGuard_IncludesPreset verifies BC-7:
// the required-input error message lists --workload as an option.
// Uses source-level scan (acceptable here: tests the exact error message text
// that users see — a behavioral property of the CLI contract).
func TestObserveCmd_WorkloadInputGuard_IncludesPreset(t *testing.T) {
	data, err := os.ReadFile("observe_cmd.go")
	if err != nil {
		t.Fatalf("cannot read observe_cmd.go: %v", err)
	}
	content := string(data)
	// The required-input guard message must list all four input modes.
	// We check for the specific error string text, not just any occurrence of "--workload".
	wantText := "Either --workload, --workload-spec, --rate, or --concurrency is required"
	if !strings.Contains(content, wantText) {
		t.Errorf("required-input guard in observe_cmd.go must contain:\n  %q\nnot found in file", wantText)
	}
}
```

**Step 2.2 — Run to confirm failure**

```
go test ./cmd/... -run "TestValidateObserveWorkloadFlags|TestObserveCmd_WorkloadInputGuard" -v
# Expected: FAIL — undefined: validateObserveWorkloadFlags
```

**Step 2.3 — Implement**

Add `"fmt"` to the import block in `cmd/observe_cmd.go` (not currently imported).

Add the validator function before `runObserve`:

```go
// validateObserveWorkloadFlags checks preset-mode flag constraints.
// Returns a non-empty error string if the combination is invalid, empty string if valid.
// Called from runObserve; extracted for unit testability (R14).
func validateObserveWorkloadFlags(preset, workloadSpec string, rateChanged bool, concurrency int) string {
	if preset == "" {
		return "" // no preset — nothing to validate
	}
	if workloadSpec != "" {
		return "--workload and --workload-spec are mutually exclusive"
	}
	if concurrency > 0 {
		return "--workload and --concurrency are mutually exclusive; define concurrency in the spec file"
	}
	if !rateChanged {
		return fmt.Sprintf("--workload %q requires --rate (preset synthesis needs a request rate)", preset)
	}
	return ""
}
```

In `runObserve`, update the workload-input guard (currently on line ~149). The new guard must
come BEFORE the existing `--concurrency and --rate are mutually exclusive` check so that
preset-specific error messages take priority:

```go
// BC-7: at least one workload input mode must be provided
if observeWorkload == "" && observeWorkloadSpec == "" && !cmd.Flags().Changed("rate") && observeConcurrency <= 0 {
	logrus.Fatalf("Either --workload, --workload-spec, --rate, or --concurrency is required")
}
// BC-2/3/4: preset-mode constraint check (extracted for testability, R14).
// Runs before the existing concurrency/rate exclusion so preset errors are shown first.
if msg := validateObserveWorkloadFlags(observeWorkload, observeWorkloadSpec, cmd.Flags().Changed("rate"), observeConcurrency); msg != "" {
	logrus.Fatalf("%s", msg)
}
```

**Step 2.4 — Run to confirm pass**

```
go test ./cmd/... -run "TestValidateObserveWorkloadFlags|TestObserveCmd_WorkloadInputGuard" -v
# Expected: PASS (6 tests)
```

**Step 2.5 — Lint**

```
golangci-lint run ./cmd/...
# Expected: 0 issues
```

**Step 2.6 — Commit**

```
git add cmd/observe_cmd.go cmd/observe_cmd_test.go
git commit -m "feat(cmd): add --workload flag validation to blis observe

Adds validateObserveWorkloadFlags helper (BC-2/3/4) and updates
required-input guard to include --workload path (BC-7)."
```

---

### Task 3 — Preset Synthesis Tests + `buildPresetSpec` Helper + Synthesis Path

**Step 3.1 — Write failing tests (BC-1, BC-5)**

The test for BC-1 calls `buildPresetSpec` (the helper we will add in step 3.3) directly with
a temp defaults.yaml. This is the wiring test — it verifies that the helper correctly maps
preset YAML fields to `SynthesizeFromPreset` arguments.

> **YAML key note (R10):** The `Workload` struct uses `yaml:"prompt_tokens"` and
> `yaml:"output_tokens"` for the mean fields (not `prompt_tokens_mean`). The temp YAML
> must use exactly these keys or `loadDefaultsConfig`'s `KnownFields(true)` will fatal.
> See `cmd/default_config.go:15,19`.

Add to `cmd/observe_cmd_test.go` (needs `"path/filepath"` — check if already imported;
if not, add it):

```go
// TestBuildPresetSpec_MatchesPresetDefinition verifies BC-1:
// buildPresetSpec loads token distribution from defaults.yaml and passes it
// to SynthesizeFromPreset, producing a spec with correct rate and token means.
//
// This is the wiring test: it exercises the actual buildPresetSpec code path
// in observe_cmd.go, not just the workload package independently.
func TestBuildPresetSpec_MatchesPresetDefinition(t *testing.T) {
	dir := t.TempDir()
	defaultsPath := filepath.Join(dir, "defaults.yaml")
	// YAML keys must match Workload struct tags exactly (R10: KnownFields(true)).
	// prompt_tokens → PromptTokensMean, output_tokens → OutputTokensMean (see default_config.go:15,19)
	defaultsContent := `workloads:
  chatbot:
    prefix_tokens: 0
    prompt_tokens: 512
    prompt_tokens_stdev: 100
    prompt_tokens_min: 50
    prompt_tokens_max: 1024
    output_tokens: 256
    output_tokens_stdev: 50
    output_tokens_min: 10
    output_tokens_max: 512
`
	if err := os.WriteFile(defaultsPath, []byte(defaultsContent), 0600); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

	const testRate = 5.0
	const testNumRequests = 10
	spec, errMsg := buildPresetSpec("chatbot", defaultsPath, testRate, testNumRequests)
	if errMsg != "" {
		t.Fatalf("buildPresetSpec returned error: %q", errMsg)
	}
	if spec == nil {
		t.Fatal("buildPresetSpec returned nil spec")
	}
	if len(spec.Clients) == 0 {
		t.Fatal("spec has no clients")
	}
	client := spec.Clients[0]
	if client.ArrivalProcess == nil {
		t.Fatal("spec client has nil ArrivalProcess")
	}
	// Invariant: arrival rate matches requested rate
	if client.ArrivalProcess.Rate != testRate {
		t.Errorf("ArrivalProcess.Rate: got %v, want %v", client.ArrivalProcess.Rate, testRate)
	}
	// Invariant: token means come from preset YAML, not distribution defaults
	if client.PromptTokensMean != 512 {
		t.Errorf("PromptTokensMean: got %v, want 512 (from chatbot preset)", client.PromptTokensMean)
	}
	if client.OutputTokensMean != 256 {
		t.Errorf("OutputTokensMean: got %v, want 256 (from chatbot preset)", client.OutputTokensMean)
	}
	// Invariant: num-requests bound propagated into spec
	if spec.NumRequests != int64(testNumRequests) {
		t.Errorf("spec.NumRequests: got %d, want %d", spec.NumRequests, testNumRequests)
	}
}

// TestBuildPresetSpec_UnknownPreset_ReturnsError verifies BC-5:
// buildPresetSpec returns a non-empty error for an undefined preset name.
// Error message must list valid preset names.
func TestBuildPresetSpec_UnknownPreset_ReturnsError(t *testing.T) {
	dir := t.TempDir()
	defaultsPath := filepath.Join(dir, "defaults.yaml")
	// Minimal valid defaults.yaml — no workloads section means all presets are undefined
	if err := os.WriteFile(defaultsPath, []byte("version: test\n"), 0600); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

	spec, errMsg := buildPresetSpec("unknown-preset", defaultsPath, 5.0, 10)
	if spec != nil {
		t.Error("expected nil spec for unknown preset, got non-nil")
	}
	if errMsg == "" {
		t.Fatal("expected error for unknown preset, got empty message")
	}
	// Invariant: error lists valid preset names so users know what to pass
	for _, name := range []string{"chatbot", "summarization", "contentgen", "multidoc"} {
		if !strings.Contains(errMsg, name) {
			t.Errorf("error message should list valid preset %q, got: %q", name, errMsg)
		}
	}
}
```

**Step 3.2 — Run to confirm failure**

```
go test ./cmd/... -run "TestBuildPresetSpec" -v
# Expected: FAIL — undefined: buildPresetSpec
```

**Step 3.3 — Implement `buildPresetSpec` helper and synthesis path**

Add the helper function in `cmd/observe_cmd.go` (before `runObserve`):

```go
// buildPresetSpec loads the named preset from defaults.yaml and synthesizes a WorkloadSpec.
// Returns (nil, errMsg) if the preset is not defined; (spec, "") on success.
// Extracted from runObserve for unit testability (R14).
func buildPresetSpec(preset, defaultsPath string, rate float64, numRequests int) (*workload.WorkloadSpec, string) {
	wl := loadPresetWorkload(defaultsPath, preset)
	if wl == nil {
		return nil, fmt.Sprintf("Undefined workload %q. Use one among (chatbot, summarization, contentgen, multidoc) or --workload-spec", preset)
	}
	spec := workload.SynthesizeFromPreset(preset, workload.PresetConfig{
		PrefixTokens:      wl.PrefixTokens,
		PromptTokensMean:  wl.PromptTokensMean,
		PromptTokensStdev: wl.PromptTokensStdev,
		PromptTokensMin:   wl.PromptTokensMin,
		PromptTokensMax:   wl.PromptTokensMax,
		OutputTokensMean:  wl.OutputTokensMean,
		OutputTokensStdev: wl.OutputTokensStdev,
		OutputTokensMin:   wl.OutputTokensMin,
		OutputTokensMax:   wl.OutputTokensMax,
	}, rate, numRequests)
	return spec, ""
}
```

Update the workload generation block in `runObserve`. Currently:

```go
// Generate workload
var spec *workload.WorkloadSpec
if observeWorkloadSpec != "" {
    ...
} else {
    // Distribution or concurrency synthesis
    spec = workload.SynthesizeFromDistribution(...)
    spec.Seed = observeSeed
}
```

Replace with a three-way branch:

```go
// Generate workload
var spec *workload.WorkloadSpec
if observeWorkloadSpec != "" {
	if observeConcurrency > 0 {
		logrus.Fatalf("--concurrency cannot be used with --workload-spec; " +
			"define concurrency in the spec file using clients[].concurrency instead")
	}
	var err error
	spec, err = workload.LoadWorkloadSpec(observeWorkloadSpec)
	if err != nil {
		logrus.Fatalf("Failed to load workload spec: %v", err)
	}
	if cmd.Flags().Changed("seed") {
		spec.Seed = observeSeed
	}
} else if observeWorkload != "" {
	// Preset synthesis — BC-1: same token distribution as blis run --workload <preset>
	// Rate was validated finite+positive by line ~170 (defense-in-depth: also guarded
	// by validateObserveWorkloadFlags above, which requires rateChanged to be true).
	// Use separate errMsg var + = (not :=) to avoid shadowing the outer spec variable.
	var errMsg string
	spec, errMsg = buildPresetSpec(observeWorkload, observeDefaultsFilePath, observeRate, observeNumRequests)
	if errMsg != "" {
		logrus.Fatalf("%s", errMsg)
	}
	spec.Seed = observeSeed
} else {
	// Distribution or concurrency synthesis
	spec = workload.SynthesizeFromDistribution(workload.DistributionParams{
		Rate:               observeRate,
		Concurrency:        observeConcurrency,
		ThinkTimeMs:        observeThinkTimeMs,
		NumRequests:        observeNumRequests,
		PrefixTokens:       observePrefixTokens,
		PromptTokensMean:   observePromptTokens,
		PromptTokensStdDev: observePromptStdDev,
		PromptTokensMin:    observePromptMin,
		PromptTokensMax:    observePromptMax,
		OutputTokensMean:   observeOutputTokens,
		OutputTokensStdDev: observeOutputStdDev,
		OutputTokensMin:    observeOutputMin,
		OutputTokensMax:    observeOutputMax,
	})
	spec.Seed = observeSeed
}
```

Also update the `Long` description in `var observeCmd`. In `cmd/observe_cmd.go` around line 66,
replace the existing input paths line:

```
Supports --workload-spec (YAML), --rate (distribution synthesis), or --concurrency
```

with:

```
Supports --workload-spec (YAML), --workload <preset> (named preset; requires --rate),
--rate (distribution synthesis), or --concurrency
```

And add one new example to the Examples block, after the `--rate 10` example and before the
`--concurrency` example. Insert this block (with the blank line separator):

```
  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --workload chatbot --rate 10 --num-requests 100 \
    --trace-header trace.yaml --trace-data trace.csv
```

**Step 3.4 — Run to confirm pass**

```
go test ./cmd/... -run "TestBuildPresetSpec" -v
# Expected: PASS (2 tests)
```

**Step 3.5 — Run full test suite**

```
go test ./cmd/... -count=1
# Expected: all tests pass
```

**Step 3.6 — Lint**

```
golangci-lint run ./cmd/...
# Expected: 0 issues
```

**Step 3.7 — Commit**

```
git add cmd/observe_cmd.go cmd/observe_cmd_test.go
git commit -m "feat(cmd): implement --workload preset synthesis for blis observe

Adds buildPresetSpec helper (BC-1/5) and wires --workload preset path
in runObserve. --workload chatbot --rate 10 now produces the same
WorkloadSpec shape as blis run --workload chatbot --rate 10."
```

---

### Task 4 — Final Verification Gate

**Step 4.1 — Full test suite**

```
go test ./... -count=1
```

Expected output: all packages pass. Report exact pass/fail counts.

**Step 4.2 — Build**

```
go build ./...
```

Expected: exit code 0.

**Step 4.3 — Lint**

```
golangci-lint run ./...
```

Expected: 0 issues.

Report all three results. Wait for user approval before Step 5.

---

## Sanity Checklist

- [ ] `--workload chatbot --rate 5` produces the same preset spec shape as run (BC-1)
- [ ] Missing `--rate` with preset produces a fatal error mentioning `--rate` (BC-2)
- [ ] `--workload` + `--workload-spec` together produce a fatal error (BC-3)
- [ ] `--workload` + `--concurrency` together produce a fatal error (BC-4)
- [ ] Unknown preset name returns an error listing chatbot/summarization/contentgen/multidoc (BC-5)
- [ ] Both `--workload` and `--defaults-filepath` flags exist with correct defaults (BC-6)
- [ ] Required-input guard updated to include `--workload` with exact message text (BC-7)
- [ ] `go test ./...` passes
- [ ] `go build ./...` passes
- [ ] `golangci-lint run ./...` passes
- [ ] Long description and `--help` mention `--workload` with example
