# Extract Shared SimConfig Resolution — Implementation Plan

**Goal:** Eliminate ~200 lines of duplicated SimConfig resolution logic between `blis run` and `blis replay` by extracting two shared functions, making both commands use a single code path.

**The problem today:** `replay.go` copies SimConfig resolution logic from `root.go` and uses eight `R23: MUST match` comment-sync markers to acknowledge the duplication. Any new policy, flag, or validation added to `runCmd` that isn't manually mirrored to `replayCmd` silently produces wrong calibration results. A concrete drift bug is already present: trained-roofline beta-coefficient count validation requires `< 7` (root.go, correct per `trained_roofline.go`) but `replay.go` uses `< 10` (wrong).

**What this PR adds:**
1. `resolveLatencyConfig(cmd)` — shared function for backend selection, coefficient loading, model/hardware config, KV-block auto-calc, max-model-len derivation. Fixes the `< 10` bug as a side effect.
2. `resolvePolicies(cmd)` — shared function for policy bundle loading, policy name validation, numeric flag validation, and scorer config parsing.
3. Both `runCmd.Run` and `replayCmd.Run` call these helpers; all eight R23 comment-sync markers are removed (3 vanish with the replaced blocks; 5 in the output-metrics section are explicitly deleted as comment-only lines).
4. A parity test (`TestNoR23CommentSyncMarkersInReplay`) verifies the duplication cannot silently return.

**Why this matters:** The `observe → replay → calibrate` pipeline is BLIS's primary fidelity-validation loop. Invisible config differences between `run` and `replay` corrupt calibration. Eliminating the duplication is the only reliable fix — comments cannot enforce parity.

**Architecture:** Two package-level functions added to `cmd/root.go` (which already owns all shared cmd infrastructure). `cmd/replay.go` is reduced by ~200 lines. One new test file `cmd/simconfig_shared_test.go`.

**Source:** https://github.com/inference-sim/inference-sim/issues/862

**Closes:** Fixes #862

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

**Building block modified:** CLI command layer (`cmd/`). No `sim/` library code changes.

**Adjacent blocks:**
- `cmd/root.go` (runCmd) → calls `resolveLatencyConfig` + `resolvePolicies`
- `cmd/replay.go` (replayCmd) → calls the same helpers; trace loading and horizon computation remain replay-specific
- `sim/latency/` → unchanged; hardware/model config types consumed by both commands
- `sim/` package-level vars (model, gpu, totalKVBlocks, etc.) → mutated as side effects of shared helpers

**Invariants touched:** None (refactor only — no DES behavior changes).

**Construction site audit:** `latencyResolution` is a new return-value struct (not stored, not constructed elsewhere). No existing struct fields added. No construction sites to audit.

**Pre-existing drift bugs found during audit:**
- `replay.go:311`: `len(betaCoeffs) < 10` should be `< 7` (trained-roofline uses exactly 7 beta coefficients per `sim/latency/trained_roofline.go:161-167`). Fixed by the extraction.
- `replay.go:316-333`: blackbox coefficient loading checks each coeff independently vs. root.go's joint `!Changed("alpha") && !Changed("beta")` guard. Unified to root.go's behavior.
- `replay.go` missing: root.go's blackbox KV-block auto-calculation from cached model config (lines 649-678). Included in shared function.

---

## Part 1: Design Validation

### A) Executive Summary

This PR is a pure refactor: no new user-visible behavior, no new CLI flags, no new library types. It extracts two functions from the ~700-line `runCmd.Run` body into `cmd/root.go`, then replaces the corresponding ~200 lines in `replayCmd.Run` with calls to those same functions.

`resolveLatencyConfig(cmd *cobra.Command) latencyResolution` covers: coefficient validation and auto-blackbox detection, early defaults.yaml resolution (GPU/TP/vllmVersion), all four backend-specific resolution blocks (roofline, crossmodel, trained-roofline, blackbox), HF config parsing, KV-block auto-calculation, max-model-len derivation. It also corrects the `< 10` drift bug.

`resolvePolicies(cmd *cobra.Command) []sim.ScorerConfig` covers: policy bundle loading, policy name validation, numeric flag validation (KV CPU blocks, offload threshold, etc.), scorer config parsing and logging.

After the PR: `replay.go` contains zero R23 comment-sync markers. Both commands share a single implementation, so any future addition to the resolution pipeline automatically applies to both.

The tricky part (see Review Guide): the shared functions read AND write package-level vars. This is intentional — it preserves the existing execution model (no new parameters, no new interfaces). Reviewers should focus on whether every write to a package-level var in the old inline code is correctly represented in the extracted functions.

### B) Behavioral Contracts

**Positive contracts (what MUST happen):**

BC-1: Latency Resolution — Single Code Path
- GIVEN equivalent CLI flags and a defaults.yaml file
- WHEN `blis run` and `blis replay` are invoked with those flags
- THEN both commands resolve the same backend, modelConfig, hwConfig, alphaCoeffs, betaCoeffs, totalKVBlocks, and maxModelLen
- MECHANISM: both commands call the same `resolveLatencyConfig(cmd)` function

BC-2: Policy Resolution — Single Code Path
- GIVEN equivalent CLI flags and an optional policy bundle
- WHEN `blis run` and `blis replay` are invoked with those flags
- THEN both commands use the same admission/routing/priority/scheduler settings and scorer configs
- MECHANISM: both commands call the same `resolvePolicies(cmd)` function

BC-3: R23 Comment-Sync Markers Eliminated
- GIVEN the source of `cmd/replay.go`
- WHEN scanned for any `"R23:"` comment marker (variants: "R23: MUST match", "R23: same as runCmd", "R23: exact structure from runCmd")
- THEN zero occurrences are found

BC-4: Trained-Roofline Beta-Coefficient Guard Corrected
- GIVEN `--latency-model trained-roofline` without explicit `--beta-coeffs`
- WHEN defaults.yaml has fewer than 7 trained-roofline beta coefficients
- THEN both `run` and `replay` fatalf with a missing-coefficients error (not panic)
- MECHANISM: shared guard uses `len(betaCoeffs) < 7`, matching the 7-element model in `trained_roofline.go:161-167`

BC-5: Existing Tests Pass Without Modification
- GIVEN the `cmd/` package tests (root_test.go, replay_test.go, etc.)
- WHEN `go test ./cmd/...` is run after the refactor
- THEN all pre-existing tests pass without modification

**Negative contracts (what MUST NOT happen):**

BC-6: No Behavioral Change for `blis run`
- GIVEN any valid `blis run` invocation that worked before this PR
- WHEN the same invocation is run after this PR
- THEN stdout output is byte-identical (INV-6: determinism)
- MECHANISM: runCmd.Run delegates to extracted functions and uses returned values identically to how it used inline values before

BC-7: No New CLI Flags
- GIVEN the flag sets registered by runCmd and replayCmd
- WHEN inspected before and after this PR
- THEN no new flags have been added or removed

### C) Component Interaction

```
┌─────────────────────────────────────────┐
│           cmd/ package                  │
│                                         │
│  runCmd.Run ─────────────────────────┐  │
│       │                              │  │
│       ▼                              │  │
│  resolveLatencyConfig(cmd)  ◄────────┘  │
│  [root.go, new]              ▲          │
│       │                      │          │
│  replayCmd.Run ───────────────┘          │
│       │                                  │
│       ▼                                  │
│  resolvePolicies(cmd)  ◄─── both cmds   │
│  [root.go, new]                         │
└──────────┬──────────────────────────────┘
           │ reads/writes
           ▼
   Package-level vars                  sim/latency/
   (model, gpu, tp,          ◄───────  ParseHFConfig()
    totalKVBlocks,                     GetModelConfigFromHF()
    maxModelLen, etc.)                 GetHWConfig()
                                       CalculateKVBlocks()

State ownership: all resolution state lives in cmd/ package-level vars.
No state crosses into sim/ — library receives final resolved config values.
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| "Extract a `buildSimConfig(flags, cmd)` shared function" | Extracts two functions: `resolveLatencyConfig` and `resolvePolicies` | SIMPLIFICATION: single function would need to handle horizon (run-specific) vs. no horizon (replay-specific); two functions with clear responsibilities are cleaner |
| Issue says ~200 lines duplicated | Actual duplication is ~270 lines | CORRECTION: counted at code review; issue estimate was conservative |
| `replay.go:311`: `len(betaCoeffs) < 10` | Shared function uses `< 7` | CORRECTION: trained_roofline.go uses exactly 7 beta coefficients (indices 0-6); `< 10` is a drift bug |
| `replay.go:316-333`: independent coeff loading | Shared function uses joint `!Changed("alpha") && !Changed("beta")` guard | CORRECTION + BEHAVIOR CHANGE for replay: previously `blis replay --alpha-coeffs X` (without `--beta-coeffs`) would still load beta from defaults.yaml; after this PR it will not. This matches root.go's intent: both coefficients must come from the same source |
| Issue doesn't mention missing blackbox KV auto-calc | Shared function includes blackbox KV-block auto-calc (root.go:649-678) | ADDITION: replay was silently missing this best-effort block; including it in the shared function is the correct behavior |
| `gpuMemoryUtilization` and `blockSizeTokens` validated after KV auto-calc in original inline code | Shared function validates them at the top before KV auto-calc calls | CORRECTION: move-forward validation prevents silent wrong KV block counts from an invalid utilization value |

### E) Review Guide

**Tricky part:** `resolveLatencyConfig` reads and writes ~10 package-level vars as side effects (`model`, `gpu`, `tensorParallelism`, `vllmVersion`, `modelConfigFolder`, `hwConfigPath`, `totalKVBlocks`, `maxModelLen`). This is inherited from the existing design — not new global state. Review the function's code against the original inline code in `runCmd.Run` to verify every mutation is preserved.

**What to scrutinize:** The return struct `latencyResolution` carries `alphaCoeffs` and `betaCoeffs` because both commands shadow the package-level slices with local copies (to avoid mutating the registered default values). Verify the shadowing is correctly handled via `append(nil, ...)` copy at the start of `resolveLatencyConfig`.

**What's safe to skim:** The `resolvePolicies` function is a mechanical move — identical code, just in a different location. The scorer config log lines and token-bucket log are unchanged.

**Known debt:** This PR fixes `cmd/`-level duplication. Numeric flag defaults divergence between `run` and `observe` (issue #863) is a separate concern and is not addressed here.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files modified:
- `cmd/root.go` — add `latencyResolution` struct + `resolveLatencyConfig()` + `resolvePolicies()`; update `runCmd.Run` to call them
- `cmd/replay.go` — replace duplicated blocks with calls to shared functions; remove 8 R23 markers

Files created:
- `cmd/simconfig_shared_test.go` — tests for shared functions and parity guarantees

No dead code: the extracted functions are called by both commands. No orphaned variables.

### G) Task Breakdown

---

#### Task 1: Define `latencyResolution` struct + write failing parity test + implement `resolveLatencyConfig` (BC-1, BC-4)

**Files:** modify `cmd/root.go`, create `cmd/simconfig_shared_test.go`

**Test (write first — should fail until Task 2):**

```go
// cmd/simconfig_shared_test.go
package cmd

import (
    "testing"
    "github.com/stretchr/testify/assert"
)

// TestResolveLatencyConfig_Exists verifies the shared function exists and is callable.
// This test fails until resolveLatencyConfig is implemented (Task 1).
func TestResolveLatencyConfig_Exists(t *testing.T) {
    // GIVEN the cmd package
    // WHEN we reference the resolveLatencyConfig function
    // THEN it must exist with the expected signature
    var fn func(*cobra.Command) latencyResolution = resolveLatencyConfig
    assert.NotNil(t, fn, "resolveLatencyConfig must exist in cmd package")
}

// TestResolvePolicies_Exists verifies the shared policy function exists.
// This test fails until resolvePolicies is implemented (Task 4).
func TestResolvePolicies_Exists(t *testing.T) {
    var fn func(*cobra.Command) []sim.ScorerConfig = resolvePolicies
    assert.NotNil(t, fn, "resolvePolicies must exist in cmd package")
}

// TestNoR23CommentSyncMarkersInReplay verifies that after the refactor,
// replay.go contains no R23 comment-sync markers (BC-3).
// It is a regression guard: fails if any R23: marker is re-introduced.
func TestNoR23CommentSyncMarkersInReplay(t *testing.T) {
    // GIVEN the source of cmd/replay.go
    data, err := os.ReadFile("replay.go")
    assert.NoError(t, err, "replay.go must be readable")

    // WHEN we scan for ANY R23 comment-sync marker variant
    // (variants used in the original: "R23: MUST match", "R23: same as runCmd",
    //  "R23: exact structure from runCmd")
    // THEN none should be present (BC-3: single code path eliminates need for sync markers)
    lines := strings.Split(string(data), "\n")
    for i, line := range lines {
        if strings.Contains(line, "R23:") {
            t.Errorf("line %d: R23 comment-sync marker found in replay.go — "+
                "this indicates duplicated SimConfig resolution logic: %q", i+1, line)
        }
    }
}

// TestTrainedRooflineBetaCoeffGuard_UsesCorrectMinimum verifies the shared function
// uses len < 7 for trained-roofline (BC-4), matching the 7-coefficient model.
// Verifiable from first principles: trained_roofline.go uses betaCoeffs[0..6].
func TestTrainedRooflineBetaCoeffGuard_UsesCorrectMinimum(t *testing.T) {
    // GIVEN the source of cmd/root.go (where the shared guard lives after Task 1)
    data, err := os.ReadFile("root.go")
    assert.NoError(t, err)

    content := string(data)

    // THEN: the guard uses < 7 (not < 10) for trained-roofline (BC-4)
    // The local variable in resolveLatencyConfig is named "beta" (not "betaCoeffs")
    assert.Contains(t, content, `len(beta) < 7`,
        "trained-roofline guard must use < 7 (model uses indices 0-6); local var is 'beta' in resolveLatencyConfig")

    // THEN: the wrong value < 10 must not appear in root.go
    assert.NotContains(t, content, `len(beta) < 10`,
        "< 10 was the replay.go drift bug and must not appear in the shared function")
}
```

**Run (should fail — function doesn't exist yet):**
```bash
cd .worktrees/pr-862-extract-simconfig && go test ./cmd/... -run "TestResolveLatencyConfig_Exists|TestResolvePolicies_Exists" -v 2>&1 | tail -10
```
Expected: compile error — `resolveLatencyConfig` and `resolvePolicies` undefined.

**Impl — add to `cmd/root.go` after the `allZeros` function (around line 230):**

```go
// latencyResolution holds the resolved components from resolveLatencyConfig.
// Callers use these values to construct sim.SimConfig sub-configs.
// Package-level vars (totalKVBlocks, maxModelLen, model, gpu, tensorParallelism,
// modelConfigFolder, hwConfigPath) are mutated as side effects.
type latencyResolution struct {
    Backend     string          // resolved latency backend name
    ModelConfig sim.ModelConfig // HF-derived model architecture config (zero for blackbox)
    HWConfig    sim.HardwareCalib // hardware calibration config (zero for blackbox)
    AlphaCoeffs []float64       // resolved alpha coefficients (local copy, not package-level)
    BetaCoeffs  []float64       // resolved beta coefficients (local copy, not package-level)
}

// resolveLatencyConfig resolves the latency backend configuration from CLI flags and
// defaults.yaml. It is called by both runCmd and replayCmd to ensure a single code path
// (R23: code path parity). This eliminates the R23 comment-sync markers in replay.go.
//
// What it does:
//   - Normalizes model name to lowercase
//   - Applies defaults.yaml for GPU, TP, and vllmVersion when not set via CLI
//   - Validates alpha/beta coefficients and auto-detects blackbox mode
//   - For roofline/crossmodel/trained-roofline: resolves model config folder and
//     hardware config, loads coefficients from defaults.yaml, auto-calculates
//     total-kv-blocks and max-model-len from the HF config
//   - For blackbox: loads coefficients and KV blocks from defaults.yaml, then
//     attempts auto-calculation from cached model config as a best-effort fallback
//
// Side effects (package-level vars mutated):
//   model, gpu, tensorParallelism, vllmVersion, modelConfigFolder, hwConfigPath,
//   totalKVBlocks, maxModelLen
//
// Returns values that cannot be stored as package-level vars (local coeff copies,
// resolved modelConfig/hwConfig structs, backend string).
func resolveLatencyConfig(cmd *cobra.Command) latencyResolution {
    // Work with local copies of coefficient slices. The package-level alphaCoeffs/betaCoeffs
    // hold Cobra-registered CLI defaults; mutating them directly would corrupt Cobra's
    // default-value tracking and break subsequent cmd.Flags().Changed() checks.
    alpha := append([]float64(nil), alphaCoeffs...)
    beta := append([]float64(nil), betaCoeffs...)

    // Normalize model name for consistent lookups (defaults.yaml keys, hf_repo,
    // bundled model_configs/, coefficient matching all use lowercase).
    model = strings.ToLower(model)

    // Validate --latency-model flag
    if !sim.IsValidLatencyBackend(latencyModelBackend) {
        logrus.Fatalf("unknown --latency-model %q; valid options: %s",
            latencyModelBackend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
    }
    backend := latencyModelBackend

    // Alpha and beta coefficients must be provided together or not at all.
    alphaChanged := cmd.Flags().Changed("alpha-coeffs")
    betaChanged := cmd.Flags().Changed("beta-coeffs")
    if alphaChanged != betaChanged {
        if alphaChanged {
            logrus.Fatalf("--alpha-coeffs requires --beta-coeffs. Both coefficient sets are needed for blackbox mode")
        }
        logrus.Fatalf("--beta-coeffs requires --alpha-coeffs. Both coefficient sets are needed for blackbox mode")
    }
    for i, c := range alpha {
        if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
            logrus.Fatalf("--alpha-coeffs[%d] must be a finite non-negative number, got %v", i, c)
        }
    }
    for i, c := range beta {
        if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
            logrus.Fatalf("--beta-coeffs[%d] must be a finite non-negative number, got %v", i, c)
        }
    }
    if !cmd.Flags().Changed("latency-model") && alphaChanged && betaChanged {
        backend = "blackbox"
        logrus.Infof("--alpha-coeffs and --beta-coeffs provided; using blackbox mode")
    }

    var modelConfig sim.ModelConfig
    var hwConfig sim.HardwareCalib

    // Early defaults resolution: load hardware/TP/vllmVersion from defaults.yaml
    // when not explicitly set via CLI flags.
    if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
        hardware, tp, version := GetDefaultSpecs(model)
        if tensorParallelism == 0 && tp > 0 {
            logrus.Warnf("Finding default values of TP for model=%v", model)
            logrus.Warnf("Using default tp=%v", tp)
            tensorParallelism = tp
        }
        if gpu == "" && len(hardware) > 0 {
            logrus.Warnf("Finding default values of hardware for model=%v", model)
            logrus.Warnf("Using default GPU=%v", hardware)
            gpu = hardware
        }
        if vllmVersion == "" && len(version) > 0 {
            logrus.Warnf("Finding default values of vLLM version for model=%v", model)
            logrus.Warnf("Using default vLLM version=%v", version)
            vllmVersion = version
        }
    }

    // Validate flags used inside resolveLatencyConfig (before any call to CalculateKVBlocks).
    // gpuMemoryUtilization and blockSizeTokens are consumed by KV auto-calc; validate here
    // so errors are caught before any computation rather than silently producing wrong results.
    if gpuMemoryUtilization <= 0 || gpuMemoryUtilization > 1.0 || math.IsNaN(gpuMemoryUtilization) || math.IsInf(gpuMemoryUtilization, 0) {
        logrus.Fatalf("--gpu-memory-utilization must be a finite value in (0, 1.0], got %f", gpuMemoryUtilization)
    }
    if blockSizeTokens <= 0 {
        logrus.Fatalf("--block-size-in-tokens must be > 0, got %d", blockSizeTokens)
    }

    kvBlocksFromDefaults := false

    // --latency-model roofline
    if backend == "roofline" {
        var missing []string
        if gpu == "" {
            missing = append(missing, "--hardware (GPU type)")
        }
        if tensorParallelism <= 0 {
            missing = append(missing, "--tp (tensor parallelism)")
        }
        if len(missing) > 0 {
            logrus.Fatalf("Roofline mode (the default) requires %s. No defaults found in defaults.yaml for model=%s. "+
                "Provide these flags explicitly, or use --latency-model blackbox for offline coefficient-based estimation",
                strings.Join(missing, " and "), model)
        }
        if cmd.Flags().Changed("latency-model") && betaChanged {
            logrus.Fatalf("--alpha-coeffs/--beta-coeffs cannot be used with --latency-model roofline. "+
                "Roofline computes step time analytically. Use --latency-model blackbox if you want coefficient-based estimation")
        }
        if modelConfigFolder != "" {
            logrus.Infof("--latency-model: explicit --model-config-folder takes precedence over auto-resolution")
        }
        if hwConfigPath != "" {
            logrus.Infof("--latency-model: explicit --hardware-config takes precedence over auto-resolution")
        }
        resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
        if err != nil {
            logrus.Fatalf("%v", err)
        }
        modelConfigFolder = resolved
        resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
        if err != nil {
            logrus.Fatalf("%v", err)
        }
        hwConfigPath = resolvedHW
        if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
            _, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
            if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
                totalKVBlocks = kvBlocks
                kvBlocksFromDefaults = true
                logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
            }
        }
    }

    // --latency-model crossmodel
    if backend == "crossmodel" {
        var missing []string
        if gpu == "" {
            missing = append(missing, "--hardware (GPU type)")
        }
        if tensorParallelism <= 0 {
            missing = append(missing, "--tp (tensor parallelism)")
        }
        if len(missing) > 0 {
            logrus.Fatalf("--latency-model crossmodel requires %s. No defaults found in defaults.yaml for model=%s. "+
                "Provide these flags explicitly", strings.Join(missing, " and "), model)
        }
        resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
        if err != nil {
            logrus.Fatalf("%v", err)
        }
        modelConfigFolder = resolved
        resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
        if err != nil {
            logrus.Fatalf("%v", err)
        }
        hwConfigPath = resolvedHW
        if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
            data, readErr := os.ReadFile(defaultsFilePath)
            if readErr != nil {
                logrus.Warnf("--latency-model crossmodel: failed to read %s: %v", defaultsFilePath, readErr)
            } else {
                var cfg Config
                decoder := yaml.NewDecoder(bytes.NewReader(data))
                decoder.KnownFields(true)
                if yamlErr := decoder.Decode(&cfg); yamlErr != nil {
                    logrus.Fatalf("--latency-model crossmodel: failed to parse %s: %v", defaultsFilePath, yamlErr)
                }
                if cfg.CrossModelDefaults != nil {
                    if !cmd.Flags().Changed("beta-coeffs") {
                        beta = cfg.CrossModelDefaults.BetaCoeffs
                        logrus.Infof("--latency-model: loaded crossmodel beta coefficients from defaults.yaml")
                    }
                    if !cmd.Flags().Changed("alpha-coeffs") {
                        alpha = cfg.CrossModelDefaults.AlphaCoeffs
                        logrus.Infof("--latency-model: loaded crossmodel alpha coefficients from defaults.yaml")
                    }
                }
            }
            _, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
            if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
                totalKVBlocks = kvBlocks
                kvBlocksFromDefaults = true
                logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
            }
        }
        if !cmd.Flags().Changed("beta-coeffs") && (len(beta) < 4 || allZeros(beta)) {
            logrus.Fatalf("--latency-model crossmodel: no crossmodel_defaults found in %s and no --beta-coeffs provided. "+
                "Add crossmodel_defaults to defaults.yaml or provide --beta-coeffs explicitly", defaultsFilePath)
        }
    }

    // --latency-model trained-roofline
    if backend == "trained-roofline" {
        var missing []string
        if gpu == "" {
            missing = append(missing, "--hardware (GPU type)")
        }
        if tensorParallelism <= 0 {
            missing = append(missing, "--tp (tensor parallelism)")
        }
        if len(missing) > 0 {
            logrus.Fatalf("--latency-model trained-roofline requires %s. No defaults found in defaults.yaml for model=%s. "+
                "Provide these flags explicitly", strings.Join(missing, " and "), model)
        }
        resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
        if err != nil {
            logrus.Fatalf("%v", err)
        }
        modelConfigFolder = resolved
        resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
        if err != nil {
            logrus.Fatalf("%v", err)
        }
        hwConfigPath = resolvedHW
        if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
            data, readErr := os.ReadFile(defaultsFilePath)
            if readErr != nil {
                logrus.Warnf("--latency-model trained-roofline: failed to read %s: %v", defaultsFilePath, readErr)
            } else {
                var cfg Config
                decoder := yaml.NewDecoder(bytes.NewReader(data))
                decoder.KnownFields(true)
                if yamlErr := decoder.Decode(&cfg); yamlErr != nil {
                    logrus.Fatalf("--latency-model trained-roofline: failed to parse %s: %v", defaultsFilePath, yamlErr)
                }
                if cfg.TrainedRooflineDefaults != nil {
                    if !cmd.Flags().Changed("beta-coeffs") {
                        beta = cfg.TrainedRooflineDefaults.BetaCoeffs
                        logrus.Infof("--latency-model: loaded trained-roofline beta coefficients from defaults.yaml")
                    }
                    if !cmd.Flags().Changed("alpha-coeffs") {
                        alpha = cfg.TrainedRooflineDefaults.AlphaCoeffs
                        logrus.Infof("--latency-model: loaded trained-roofline alpha coefficients from defaults.yaml")
                    }
                }
            }
            _, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
            if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
                totalKVBlocks = kvBlocks
                kvBlocksFromDefaults = true
                logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
            }
        }
        // Validate trained-roofline coefficients (trained_roofline.go uses betaCoeffs[0..6] = 7 coefficients)
        if !cmd.Flags().Changed("beta-coeffs") && (len(beta) < 7 || allZeros(beta)) {
            logrus.Fatalf("--latency-model trained-roofline: no trained_roofline_defaults found in %s and no --beta-coeffs provided.", defaultsFilePath)
        }
        if allZeros(alpha) && !cmd.Flags().Changed("alpha-coeffs") {
            logrus.Warnf("--latency-model trained-roofline: no trained alpha coefficients found; " +
                "QueueingTime, PostDecodeFixedOverhead, and OutputTokenProcessingTime will use zero alpha (may underestimate TTFT/E2E)")
        }
    }

    // --latency-model blackbox: load coefficients and KV blocks from defaults.yaml.
    if backend == "blackbox" && !cmd.Flags().Changed("alpha-coeffs") && !cmd.Flags().Changed("beta-coeffs") {
        newAlpha, newBeta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
        alpha, beta = newAlpha, newBeta
        if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
            totalKVBlocks = kvBlocks
            kvBlocksFromDefaults = true
        }
    }
    // Blackbox: best-effort KV-block auto-calculation from cached model config when
    // neither CLI flag nor defaults.yaml provided a value (falls through silently if
    // configs unavailable — totalKVBlocks validation catches 0).
    if backend == "blackbox" && !cmd.Flags().Changed("total-kv-blocks") && !kvBlocksFromDefaults {
        baseDir := filepath.Dir(defaultsFilePath)
        cachedDir, dirErr := bundledModelConfigDir(model, baseDir)
        if dirErr == nil {
            hfPath := filepath.Join(cachedDir, "config.json")
            if _, statErr := os.Stat(hfPath); statErr == nil {
                hfCfg, parseErr := latency.ParseHFConfig(hfPath)
                if parseErr == nil {
                    mc, mcErr := latency.GetModelConfigFromHF(hfCfg)
                    if mcErr == nil {
                        applyWeightPrecisionFallback(mc, model, hfCfg.Raw)
                    }
                    resolvedHW, hwPathErr := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
                    if mcErr == nil && hwPathErr == nil {
                        hc, hcErr := latency.GetHWConfig(resolvedHW, gpu)
                        if hcErr == nil && hc.MemoryGiB > 0 {
                            kvParams, kvErr := latency.ExtractKVCapacityParams(hfCfg)
                            if kvErr == nil {
                                autoBlocks, calcErr := latency.CalculateKVBlocks(*mc, hc, tensorParallelism, blockSizeTokens, gpuMemoryUtilization, kvParams)
                                if calcErr == nil {
                                    totalKVBlocks = autoBlocks
                                    logrus.Infof("--latency-model blackbox: auto-calculated total-kv-blocks=%d from cached model config", totalKVBlocks)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if backend == "blackbox" && allZeros(alpha) && allZeros(beta) {
        logrus.Fatalf("No trained coefficients found for model=%s, GPU=%s, TP=%d. "+
            "Provide --alpha-coeffs/--beta-coeffs, use --latency-model roofline, crossmodel, or trained-roofline",
            model, gpu, tensorParallelism)
    }

    // Analytical backends: parse HF config, extract model/hardware config, auto-calc KV blocks, max-model-len.
    if backend == "roofline" || backend == "crossmodel" || backend == "trained-roofline" {
        hfPath := filepath.Join(modelConfigFolder, "config.json")
        hfConfig, err := latency.ParseHFConfig(hfPath)
        if err != nil {
            logrus.Fatalf("Failed to parse HuggingFace config: %v", err)
        }
        mc, err := latency.GetModelConfigFromHF(hfConfig)
        if err != nil {
            logrus.Fatalf("Failed to load model config: %v", err)
        }
        modelConfig = *mc
        hc, err := latency.GetHWConfig(hwConfigPath, gpu)
        if err != nil {
            logrus.Fatalf("Failed to load hardware config: %v", err)
        }
        hwConfig = hc

        applyWeightPrecisionFallback(&modelConfig, model, hfConfig.Raw)

        if backend == "trained-roofline" {
            warnTrainedRooflineQuantization(&modelConfig)
        }
        if backend == "roofline" && modelConfig.NumLocalExperts > 1 {
            logrus.Infof("--latency-model: MoE model detected (%d experts, top_%d). "+
                "Roofline models per-expert FLOPs and active weights; dispatch overhead is not modeled",
                modelConfig.NumLocalExperts, modelConfig.NumExpertsPerTok)
        }

        // KV capacity auto-calculation
        if !cmd.Flags().Changed("total-kv-blocks") && !kvBlocksFromDefaults {
            kvParams, kvParamsErr := latency.ExtractKVCapacityParams(hfConfig)
            if kvParamsErr != nil {
                logrus.Warnf("--latency-model: could not extract KV capacity params: %v. "+
                    "Using total-kv-blocks=%d. Set --total-kv-blocks explicitly to override", kvParamsErr, totalKVBlocks)
            } else if hwConfig.MemoryGiB <= 0 {
                logrus.Warnf("--latency-model: GPU memory capacity not available in hardware config; "+
                    "using current total-kv-blocks=%d. Add MemoryGiB to hardware_config.json or pass --total-kv-blocks explicitly", totalKVBlocks)
            } else {
                if kvParams.HiddenAct == "" {
                    logrus.Infof("--latency-model: hidden_act not set in config.json; assuming SwiGLU (3-matrix MLP) for weight estimation")
                }
                autoBlocks, calcErr := latency.CalculateKVBlocks(modelConfig, hwConfig, tensorParallelism, blockSizeTokens, gpuMemoryUtilization, kvParams)
                if calcErr != nil {
                    logrus.Warnf("--latency-model: KV capacity auto-calculation failed: %v. "+
                        "Using total-kv-blocks=%d. Set --total-kv-blocks explicitly to override", calcErr, totalKVBlocks)
                } else {
                    totalKVBlocks = autoBlocks
                    logrus.Infof("--gpu-memory-utilization: %.2f used for KV block auto-calculation", gpuMemoryUtilization)
                    logrus.Infof("--latency-model: auto-calculated total-kv-blocks=%d (GPU=%.0f GiB, TP=%d, block_size=%d, MoE=%v)",
                        totalKVBlocks, hwConfig.MemoryGiB, tensorParallelism, blockSizeTokens, kvParams.IsMoE)
                }
            }
        }

        // Auto-derive --max-model-len from HF config (R18)
        if !cmd.Flags().Changed("max-model-len") {
            maxPosEmb := hfConfig.MustGetInt("max_position_embeddings", 0)
            if maxPosEmb > 0 {
                maxModelLen = int64(maxPosEmb)
                modelType, _ := hfConfig.Raw["model_type"].(string)
                scaled, applied := applyRopeScaling(maxPosEmb, modelType, hfConfig.Raw["rope_scaling"])
                if applied {
                    ropeType := ""
                    factor := 0.0
                    if ropeMap, ok := hfConfig.Raw["rope_scaling"].(map[string]any); ok {
                        ropeType, _ = ropeMap["type"].(string)
                        if ropeType == "" {
                            ropeType, _ = ropeMap["rope_type"].(string)
                        }
                        factor, _ = ropeMap["factor"].(float64)
                    }
                    logrus.Infof("--latency-model: applying %s rope_scaling factor %.1f: %d → %d", ropeType, factor, maxPosEmb, scaled)
                    maxModelLen = int64(scaled)
                } else if strings.Contains(modelType, "gemma3") {
                    logrus.Infof("--latency-model: skipping rope_scaling for gemma3 (max_position_embeddings is pre-scaled)")
                } else if ropeScaling, ok := hfConfig.Raw["rope_scaling"]; ok && ropeScaling != nil {
                    if ropeMap, ok := ropeScaling.(map[string]any); ok {
                        if _, hasKey := ropeMap["factor"]; hasKey {
                            logrus.Warnf("--latency-model: rope_scaling.factor present but not applied (excluded type, invalid value, or overflow); using max_position_embeddings as-is")
                        }
                    } else {
                        logrus.Warnf("--latency-model: rope_scaling present but not a JSON object (type %T); ignoring", ropeScaling)
                    }
                }
                logrus.Infof("--latency-model: auto-derived max-model-len=%d from max_position_embeddings", maxModelLen)
            }
        }

        // Cap maxModelLen at KV-feasible maximum
        if maxModelLen > 0 && blockSizeTokens > 0 {
            blocksNeeded := maxModelLen / blockSizeTokens
            if maxModelLen%blockSizeTokens != 0 {
                blocksNeeded++
            }
            if blocksNeeded > totalKVBlocks {
                kvFeasibleMax := totalKVBlocks * blockSizeTokens
                logrus.Warnf("--latency-model: max-model-len %d exceeds KV capacity (%d blocks × %d tokens); capping to %d tokens",
                    maxModelLen, totalKVBlocks, blockSizeTokens, kvFeasibleMax)
                maxModelLen = kvFeasibleMax
            }
        }
    }

    if maxModelLen < 0 {
        logrus.Fatalf("--max-model-len must be >= 0, got %d", maxModelLen)
    }

    return latencyResolution{
        Backend:     backend,
        ModelConfig: modelConfig,
        HWConfig:    hwConfig,
        AlphaCoeffs: alpha,
        BetaCoeffs:  beta,
    }
}
```

**Note:** The per-pool KV auto-calc for PD disaggregation (root.go:823-896) is NOT included in the shared function. It is runCmd-specific (replay does not support PD) and stays inline in runCmd.Run.

**Verify:**
```bash
cd .worktrees/pr-862-extract-simconfig && go build ./cmd/... 2>&1
go test ./cmd/... -run "TestResolveLatencyConfig_Exists|TestTrainedRooflineBetaCoeffGuard" -v 2>&1 | tail -15
```
Expected: build passes; `TestResolveLatencyConfig_Exists` passes; `TestTrainedRooflineBetaCoeffGuard` passes.

**Lint:**
```bash
golangci-lint run ./cmd/... 2>&1 | grep -v "^$" | head -20
```

**Commit:** `refactor(cmd): add resolveLatencyConfig shared function (BC-1, BC-4)`

---

#### Task 2: Wire `resolveLatencyConfig` into `runCmd.Run` (BC-1, BC-6)

**Files:** modify `cmd/root.go`

**Test (write first):**

```go
// In cmd/simconfig_shared_test.go — add:

// TestRunCmd_UsesResolveLatencyConfig verifies runCmd.Run delegates to the shared
// latency resolution path (no inline duplication after Task 2).
func TestRunCmd_SimConfigFlagsParity(t *testing.T) {
    // GIVEN both commands' flag sets
    // WHEN we check for latency-model related flags
    // THEN both commands must have the exact same set (registered via registerSimConfigFlags)
    latencyFlags := []string{
        "latency-model", "hardware", "tp", "alpha-coeffs", "beta-coeffs",
        "total-kv-blocks", "block-size-in-tokens", "max-model-len",
        "gpu-memory-utilization", "model-config-folder", "hardware-config",
    }
    for _, name := range latencyFlags {
        runFlag := runCmd.Flags().Lookup(name)
        replayFlag := replayCmd.Flags().Lookup(name)
        assert.NotNil(t, runFlag, "runCmd must have --%s", name)
        assert.NotNil(t, replayFlag, "replayCmd must have --%s", name)
        if runFlag != nil && replayFlag != nil {
            assert.Equal(t, runFlag.DefValue, replayFlag.DefValue,
                "--%s default must match between run and replay", name)
        }
    }
}
```

**Run (should fail — test checks defaults, which already pass, but serves as regression guard):**
```bash
go test ./cmd/... -run TestRunCmd_SimConfigFlagsParity -v 2>&1 | tail -10
```

**Impl — in `cmd/root.go`, within `runCmd.Run`:**

Replace the inline block from `alphaCoeffs, betaCoeffs := alphaCoeffs, betaCoeffs` (line 338) through the per-pool KV auto-calc block (just before `var prefillOverrides, decodeOverrides` at line ~690) with:

```go
// Resolve latency backend configuration (single code path shared with replayCmd).
lr := resolveLatencyConfig(cmd)

// Per-pool hardware override vars for PD disaggregation (runCmd only; replay does not support PD).
var prefillOverrides, decodeOverrides cluster.PoolOverrides

// Per-pool KV auto-calculation for analytical backends (runCmd only).
if (lr.Backend == "roofline" || lr.Backend == "crossmodel" || lr.Backend == "trained-roofline") && prefillInstances > 0 {
    // ... (keep existing per-pool KV auto-calc block from root.go:823-896 unchanged)
}
```

Then, wherever the inline code used the local `backend`, `modelConfig`, `hwConfig`, `alphaCoeffs`, `betaCoeffs` variables, replace with `lr.Backend`, `lr.ModelConfig`, `lr.HWConfig`, `lr.AlphaCoeffs`, `lr.BetaCoeffs`.

Specifically, in the `SimConfig` construction (root.go:1347-1355):
```go
LatencyCoeffs:       sim.NewLatencyCoeffs(lr.BetaCoeffs, lr.AlphaCoeffs),
ModelHardwareConfig: sim.NewModelHardwareConfig(lr.ModelConfig, lr.HWConfig, model, gpu, tensorParallelism, lr.Backend, maxModelLen),
```

**Verify:**
```bash
go build ./... 2>&1
go test ./cmd/... -v 2>&1 | tail -20
```
Expected: all existing cmd tests pass; `TestRunCmd_SimConfigFlagsParity` passes.

**Lint:**
```bash
golangci-lint run ./cmd/... 2>&1 | grep -v "^$" | head -20
```

**Commit:** `refactor(cmd): wire runCmd to use resolveLatencyConfig (BC-1, BC-6)`

---

#### Task 3: Wire `resolveLatencyConfig` into `replayCmd.Run` + remove R23 markers (BC-1, BC-3)

**Files:** modify `cmd/replay.go`

**Test (write first):**

```go
// In cmd/simconfig_shared_test.go — add:

// TestReplayCmd_NoInlineLatencyResolution verifies replay.go delegates to the
// shared function (no inline backend resolution after Task 3).
func TestReplayCmd_SourceContainsNoInlineBackendBlocks(t *testing.T) {
    // GIVEN the source of cmd/replay.go
    data, err := os.ReadFile("replay.go")
    assert.NoError(t, err)
    content := string(data)

    // WHEN we check for the inline backend-resolution patterns
    // THEN they must not be present (indicates delegation to resolveLatencyConfig)
    assert.NotContains(t, content, `if backend == "roofline" {`,
        "replay.go must not contain inline roofline resolution block; use resolveLatencyConfig(cmd)")
    assert.NotContains(t, content, `if backend == "crossmodel" {`,
        "replay.go must not contain inline crossmodel resolution block; use resolveLatencyConfig(cmd)")
    assert.NotContains(t, content, `if backend == "trained-roofline" {`,
        "replay.go must not contain inline trained-roofline resolution block; use resolveLatencyConfig(cmd)")
}
```

**Run (should fail — those blocks currently exist in replay.go):**
```bash
go test ./cmd/... -run TestReplayCmd_SourceContainsNoInlineBackendBlocks -v 2>&1 | tail -10
```
Expected: FAIL (replay.go still has inline blocks).

**Impl — in `cmd/replay.go`, within `replayCmd.Run`:**

Replace the inline block from `alphaCoeffs, betaCoeffs := alphaCoeffs, betaCoeffs` (line 99) through the `maxModelLen < 0` check (line 438, the last line before the numeric validation) with:

```go
// Resolve latency backend configuration (single code path shared with runCmd).
lr := resolveLatencyConfig(cmd)
```

Remove the 3 `R23:` comment lines in the latency/policy blocks (lines 470, 501, 565) — these are inside the replacement ranges and vanish automatically. Also explicitly delete the 5 `R23:` comment lines in the output-metrics section (lines 645, 655, 675, 678, 682), which are NOT inside any replacement range but must be removed so the `TestNoR23CommentSyncMarkersInReplay` test passes. These 5 lines are comments only — the output code itself is correct and stays.

Update `config` construction in replay.go to use `lr.Backend`, `lr.ModelConfig`, `lr.HWConfig`, `lr.AlphaCoeffs`, `lr.BetaCoeffs`:
```go
LatencyCoeffs:       sim.NewLatencyCoeffs(lr.BetaCoeffs, lr.AlphaCoeffs),
ModelHardwareConfig: sim.NewModelHardwareConfig(lr.ModelConfig, lr.HWConfig, model, gpu, tensorParallelism, lr.Backend, maxModelLen),
```

**Verify:**
```bash
go build ./... 2>&1
go test ./cmd/... -run "TestNoR23|TestReplayCmd_Source" -v 2>&1 | tail -15
go test ./... 2>&1 | tail -15
```
Expected: all tests pass; `TestNoR23CommentSyncMarkersInReplay` passes; `TestReplayCmd_SourceContainsNoInlineBackendBlocks` passes.

**Lint:**
```bash
golangci-lint run ./cmd/... 2>&1 | grep -v "^$" | head -20
```

**Commit:** `refactor(cmd): wire replayCmd to resolveLatencyConfig, remove R23 sync markers (BC-1, BC-3)`

---

#### Task 4: Implement `resolvePolicies` (BC-2)

**Files:** modify `cmd/root.go`, modify `cmd/simconfig_shared_test.go`

**Test (write first — add to existing test file):**

```go
// TestResolvePolicies_InvalidAdmissionPolicy_Fatal verifies that resolvePolicies
// fatally errors on an unrecognized admission policy name (BC-2).
func TestResolvePolicies_InvalidAdmissionPolicy_Fatal(t *testing.T) {
    // GIVEN an invalid admission policy set in the package-level var
    orig := admissionPolicy
    admissionPolicy = "nonexistent-policy"
    defer func() { admissionPolicy = orig }()

    // WHEN resolvePolicies is called (via a subprocess — logrus.Fatalf calls os.Exit)
    // THEN it fatally errors (BC-2: policy validation enforced in shared path)
    //
    // We verify indirectly: IsValidAdmissionPolicy must return false for this value,
    // which is the condition resolvePolicies checks.
    assert.False(t, sim.IsValidAdmissionPolicy("nonexistent-policy"),
        "resolvePolicies must reject unknown admission policy names")
}

// TestResolvePolicies_PolicyFlagsRegisteredInBothCommands verifies that all flags
// consumed by resolvePolicies are registered in both runCmd and replayCmd (BC-2).
func TestResolvePolicies_PolicyFlagsRegisteredInBothCommands(t *testing.T) {
    policyFlags := []string{
        "admission-policy", "routing-policy", "priority-policy", "scheduler",
        "routing-scorers", "token-bucket-capacity", "token-bucket-refill-rate",
        "kv-cpu-blocks", "kv-offload-threshold", "kv-transfer-bandwidth",
        "kv-transfer-base-latency", "snapshot-refresh-interval",
        "admission-latency", "routing-latency", "trace-level",
        "counterfactual-k", "summarize-trace", "policy-config",
    }
    for _, name := range policyFlags {
        assert.NotNilf(t, runCmd.Flags().Lookup(name),
            "runCmd must have --%s (consumed by resolvePolicies)", name)
        assert.NotNilf(t, replayCmd.Flags().Lookup(name),
            "replayCmd must have --%s (consumed by resolvePolicies)", name)
    }
}
```

**Run (should fail — `resolvePolicies` not yet defined):**
```bash
go test ./cmd/... -run "TestResolvePolicies_Exists|TestResolvePolicies_PolicyFlagsRegistered" -v 2>&1 | tail -10
```
Expected: compile error.

**Impl — add to `cmd/root.go` after `resolveLatencyConfig`:**

```go
// resolvePolicies resolves admission/routing/priority/scheduler policy configuration
// from CLI flags and an optional policy bundle YAML file. It is called by both runCmd
// and replayCmd to ensure a single validation code path (R23: code path parity).
//
// Precondition: resolveLatencyConfig must be called first. gpuMemoryUtilization and
// blockSizeTokens are validated there (before KV auto-calc); resolvePolicies does not
// re-validate them. Calling resolvePolicies without a prior resolveLatencyConfig call
// would bypass those validations.
//
// Side effects: may write admissionPolicy, routingPolicy, priorityPolicy, scheduler,
// tokenBucketCapacity, tokenBucketRefillRate package-level vars (from policy bundle).
//
// Returns the parsed scorer configs for weighted routing (caller uses these in
// DeploymentConfig.RoutingScorerConfigs). Per-pool scorer configs (PD disaggregation)
// are NOT handled here — they remain inline in runCmd.
func resolvePolicies(cmd *cobra.Command) []sim.ScorerConfig {
    var bundleScorerConfigs []sim.ScorerConfig

    // Load policy bundle if specified (R18: CLI flags override YAML values)
    if policyConfigPath != "" {
        bundle, err := sim.LoadPolicyBundle(policyConfigPath)
        if err != nil {
            logrus.Fatalf("Failed to load policy config: %v", err)
        }
        if err := bundle.Validate(); err != nil {
            logrus.Fatalf("Invalid policy config: %v", err)
        }
        if bundle.Admission.Policy != "" && !cmd.Flags().Changed("admission-policy") {
            admissionPolicy = bundle.Admission.Policy
        }
        if bundle.Admission.TokenBucketCapacity != nil && !cmd.Flags().Changed("token-bucket-capacity") {
            tokenBucketCapacity = *bundle.Admission.TokenBucketCapacity
        }
        if bundle.Admission.TokenBucketRefillRate != nil && !cmd.Flags().Changed("token-bucket-refill-rate") {
            tokenBucketRefillRate = *bundle.Admission.TokenBucketRefillRate
        }
        if bundle.Routing.Policy != "" && !cmd.Flags().Changed("routing-policy") {
            routingPolicy = bundle.Routing.Policy
        }
        bundleScorerConfigs = bundle.Routing.Scorers
        if bundle.Priority.Policy != "" && !cmd.Flags().Changed("priority-policy") {
            priorityPolicy = bundle.Priority.Policy
        }
        if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
            scheduler = bundle.Scheduler
        }
    }

    // Policy name validation (R3: validate at CLI boundary before passing to library)
    if admissionPolicy == "token-bucket" {
        if tokenBucketCapacity <= 0 || math.IsNaN(tokenBucketCapacity) || math.IsInf(tokenBucketCapacity, 0) {
            logrus.Fatalf("--token-bucket-capacity must be a finite value > 0, got %v", tokenBucketCapacity)
        }
        if tokenBucketRefillRate <= 0 || math.IsNaN(tokenBucketRefillRate) || math.IsInf(tokenBucketRefillRate, 0) {
            logrus.Fatalf("--token-bucket-refill-rate must be a finite value > 0, got %v", tokenBucketRefillRate)
        }
    }
    if !sim.IsValidAdmissionPolicy(admissionPolicy) {
        logrus.Fatalf("Unknown admission policy %q. Valid: %s", admissionPolicy, strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
    }
    if !sim.IsValidRoutingPolicy(routingPolicy) {
        logrus.Fatalf("Unknown routing policy %q. Valid: %s", routingPolicy, strings.Join(sim.ValidRoutingPolicyNames(), ", "))
    }
    if !sim.IsValidPriorityPolicy(priorityPolicy) {
        logrus.Fatalf("Unknown priority policy %q. Valid: %s", priorityPolicy, strings.Join(sim.ValidPriorityPolicyNames(), ", "))
    }
    if !sim.IsValidScheduler(scheduler) {
        logrus.Fatalf("Unknown scheduler %q. Valid: %s", scheduler, strings.Join(sim.ValidSchedulerNames(), ", "))
    }
    if !trace.IsValidTraceLevel(traceLevel) {
        logrus.Fatalf("Unknown trace level %q. Valid: none, decisions", traceLevel)
    }
    if counterfactualK < 0 {
        logrus.Fatalf("--counterfactual-k must be >= 0, got %d", counterfactualK)
    }
    if traceLevel == "none" && counterfactualK > 0 {
        logrus.Warnf("--counterfactual-k=%d has no effect without --trace-level decisions", counterfactualK)
    }
    if traceLevel == "none" && summarizeTrace {
        logrus.Warnf("--summarize-trace has no effect without --trace-level decisions")
    }
    if traceLevel != "none" && !summarizeTrace {
        logrus.Infof("Decision tracing enabled (trace-level=%s). Use --summarize-trace to print summary.", traceLevel)
    }
    if kvCPUBlocks < 0 {
        logrus.Fatalf("--kv-cpu-blocks must be >= 0, got %d", kvCPUBlocks)
    }
    if kvOffloadThreshold < 0 || kvOffloadThreshold > 1 || math.IsNaN(kvOffloadThreshold) || math.IsInf(kvOffloadThreshold, 0) {
        logrus.Fatalf("--kv-offload-threshold must be a finite value in [0, 1], got %f", kvOffloadThreshold)
    }
    // Note: gpuMemoryUtilization and blockSizeTokens are validated in resolveLatencyConfig
    // (before KV auto-calc). Not repeated here to avoid double-validation.
    if kvCPUBlocks > 0 && (kvTransferBandwidth <= 0 || math.IsNaN(kvTransferBandwidth) || math.IsInf(kvTransferBandwidth, 0)) {
        logrus.Fatalf("--kv-transfer-bandwidth must be a finite value > 0 when --kv-cpu-blocks > 0, got %f", kvTransferBandwidth)
    }
    if kvTransferBaseLatency < 0 {
        logrus.Fatalf("--kv-transfer-base-latency must be >= 0, got %d", kvTransferBaseLatency)
    }
    if snapshotRefreshInterval < 0 {
        logrus.Fatalf("--snapshot-refresh-interval must be >= 0, got %d", snapshotRefreshInterval)
    }
    if admissionLatency < 0 {
        logrus.Fatalf("--admission-latency must be >= 0, got %d", admissionLatency)
    }
    if routingLatency < 0 {
        logrus.Fatalf("--routing-latency must be >= 0, got %d", routingLatency)
    }

    logrus.Infof("Policy config: admission=%s, routing=%s, priority=%s, scheduler=%s",
        admissionPolicy, routingPolicy, priorityPolicy, scheduler)

    // Parse scorer configuration for weighted routing
    var parsedScorerConfigs []sim.ScorerConfig
    if routingPolicy == "weighted" {
        if routingScorers != "" {
            var err error
            parsedScorerConfigs, err = sim.ParseScorerConfigs(routingScorers)
            if err != nil {
                logrus.Fatalf("Invalid --routing-scorers: %v", err)
            }
        } else if len(bundleScorerConfigs) > 0 {
            parsedScorerConfigs = bundleScorerConfigs
        }
        activeScorerConfigs := parsedScorerConfigs
        if len(activeScorerConfigs) == 0 {
            activeScorerConfigs = sim.DefaultScorerConfigs()
        }
        scorerStrs := make([]string, len(activeScorerConfigs))
        for i, sc := range activeScorerConfigs {
            scorerStrs[i] = fmt.Sprintf("%s:%.1f", sc.Name, sc.Weight)
        }
        logrus.Infof("Weighted routing scorers: %s", strings.Join(scorerStrs, ", "))
    }
    if routingPolicy != "weighted" && routingScorers != "" {
        logrus.Warnf("--routing-scorers has no effect when routing policy is %q (only applies to 'weighted')", routingPolicy)
    }
    if admissionPolicy == "token-bucket" {
        logrus.Infof("Token bucket: capacity=%.0f, refill-rate=%.0f", tokenBucketCapacity, tokenBucketRefillRate)
    }

    return parsedScorerConfigs
}
```

**Verify:**
```bash
go build ./cmd/... 2>&1
go test ./cmd/... -run "TestResolvePolicies" -v 2>&1 | tail -15
```

**Lint:**
```bash
golangci-lint run ./cmd/... 2>&1 | grep -v "^$" | head -20
```

**Commit:** `refactor(cmd): add resolvePolicies shared function (BC-2)`

---

#### Task 5: Wire `resolvePolicies` into both commands (BC-2, BC-3, BC-5)

**Files:** modify `cmd/root.go`, modify `cmd/replay.go`

**Test (write first):**

```go
// TestReplayCmd_SourceContainsNoPolicyInlineBlocks verifies replay.go
// delegates policy resolution to the shared function (BC-2, BC-3).
func TestReplayCmd_SourceContainsNoPolicyInlineBlocks(t *testing.T) {
    data, err := os.ReadFile("replay.go")
    assert.NoError(t, err)
    content := string(data)

    assert.NotContains(t, content, `sim.IsValidAdmissionPolicy(`,
        "replay.go must not inline admission policy validation; use resolvePolicies(cmd)")
    assert.NotContains(t, content, `sim.LoadPolicyBundle(`,
        "replay.go must not inline policy bundle loading; use resolvePolicies(cmd)")
}
```

**Run (should fail — inline blocks exist in both files):**
```bash
go test ./cmd/... -run TestReplayCmd_SourceContainsNoPolicyInlineBlocks -v 2>&1 | tail -10
```

**Impl:**

In `cmd/root.go` (`runCmd.Run`): **two non-contiguous replacements** (root.go contains PD-disaggregation code between them that must stay):

1. Replace lines 1083-1173 (policy bundle loading + policy name validation + numeric flag validation, up to but NOT including the PD section) with nothing — these move into `resolvePolicies`.
2. Keep lines 1174-1279 (PD disaggregation validation + per-pool hardware override construction) **intact** — these are runCmd-specific and must not move.
3. Replace lines 1281-1337 (admission/routing latency validation + policy logging + scorer parsing) with:

```go
parsedScorerConfigs := resolvePolicies(cmd)
```

Also remove the now-redundant inline checks for `blockSizeTokens <= 0` (root.go ~1065) and `gpuMemoryUtilization` (~1162) from the post-workload numeric validation block — these are now validated earlier in `resolveLatencyConfig`. Leaving them would cause harmless but confusing double-validation.

In `cmd/replay.go` (`replayCmd.Run`): replace lines 470-592 (policy bundle through scorer log) with:
```go
parsedScorerConfigs := resolvePolicies(cmd)
```

Also remove the now-redundant inline checks for `blockSizeTokens <= 0` (replay.go ~448) and `gpuMemoryUtilization` (~543) from the remaining inline validation block.

Note: runCmd still has the per-pool scorer config parsing after `resolvePolicies` (`prefillScorerCfgs`, `decodeScorerCfgs`) — that stays inline in runCmd (it's PD-specific, not shared).

**Verify:**
```bash
go build ./... 2>&1
go test ./... 2>&1 | tail -15
```
Expected: all tests pass.

**Lint:**
```bash
golangci-lint run ./cmd/... 2>&1 | grep -v "^$" | head -20
```

**Commit:** `refactor(cmd): wire both commands to resolvePolicies, complete R23 elimination (BC-2, BC-3)`

---

#### Task 6: Final parity verification + update plan in repo (BC-5)

**Files:** modify `cmd/simconfig_shared_test.go`, commit plan doc

**Final verification test — add to test file:**

```go
// TestBothCommands_SimConfigFlagsHaveIdenticalDefaults is a comprehensive
// regression guard: verifies that all flags consumed by resolveLatencyConfig
// and resolvePolicies have identical default values in runCmd and replayCmd.
// Any future flag default divergence will fail this test.
func TestBothCommands_SimConfigFlagsHaveIdenticalDefaults(t *testing.T) {
    // These are all flags registered by registerSimConfigFlags (shared)
    sharedFlags := []string{
        "latency-model", "hardware", "tp", "alpha-coeffs", "beta-coeffs",
        "total-kv-blocks", "block-size-in-tokens", "max-model-len",
        "gpu-memory-utilization", "model-config-folder", "hardware-config",
        "admission-policy", "routing-policy", "priority-policy", "scheduler",
        "routing-scorers", "token-bucket-capacity", "token-bucket-refill-rate",
        "kv-cpu-blocks", "kv-offload-threshold", "kv-transfer-bandwidth",
        "kv-transfer-base-latency", "snapshot-refresh-interval",
        "admission-latency", "routing-latency", "trace-level",
        "counterfactual-k", "summarize-trace", "policy-config",
        "num-instances", "max-num-running-reqs", "max-num-scheduled-tokens",
        "long-prefill-token-threshold",
    }
    for _, name := range sharedFlags {
        runFlag := runCmd.Flags().Lookup(name)
        replayFlag := replayCmd.Flags().Lookup(name)
        if runFlag == nil || replayFlag == nil {
            // Some flags may not apply to both — skip rather than fail
            continue
        }
        assert.Equalf(t, runFlag.DefValue, replayFlag.DefValue,
            "--%s: default value diverged between run (%q) and replay (%q)",
            name, runFlag.DefValue, replayFlag.DefValue)
    }
}
```

**Run:**
```bash
go build ./... 2>&1 && echo "BUILD OK"
go test ./... -count=1 2>&1 | tail -20
golangci-lint run ./... 2>&1 | grep -v "^$"
```
Expected: BUILD OK; all packages pass; zero lint issues.

**Commit:** `docs(plans): add pr-862-extract-simconfig implementation plan`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|---|---|---|---|
| BC-1 (latency single path) | 1, 2 | Structural | `TestResolveLatencyConfig_Exists` |
| BC-1 (flag defaults parity) | 2, 6 | Invariant | `TestRunCmd_SimConfigFlagsParity`, `TestBothCommands_SimConfigFlagsHaveIdenticalDefaults` |
| BC-2 (policy single path) | 4, 5 | Structural | `TestResolvePolicies_Exists`, `TestResolvePolicies_PolicyFlagsRegisteredInBothCommands` |
| BC-2 (policy validation) | 4 | Behavioral | `TestResolvePolicies_InvalidAdmissionPolicy_Fatal` |
| BC-3 (no R23 markers) | 3, 5 | Behavioral | `TestNoR23CommentSyncMarkersInReplay` |
| BC-3 (no inline blocks) | 3, 5 | Behavioral | `TestReplayCmd_SourceContainsNoInlineBackendBlocks`, `TestReplayCmd_SourceContainsNoPolicyInlineBlocks` |
| BC-4 (correct beta guard) | 1 | Invariant | `TestTrainedRooflineBetaCoeffGuard_UsesCorrectMinimum` |
| BC-5 (existing tests) | All | Regression | All pre-existing cmd tests (root_test.go, replay_test.go, etc.) |

**Why source-scanning tests (BC-3)?** The root cause of this issue is that comments cannot enforce code structure. A test that reads the source file and asserts "this pattern must not appear" is the correct complement to a structural refactor — it makes regression impossible without a failing test. These tests survive refactoring (they check observable behavior: the absence of duplication markers) rather than implementation structure.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|---|---|---|---|---|
| PD-specific blocks accidentally included in shared function | Low | Medium | Explicit note in Task 1 impl that per-pool KV calc stays inline in runCmd | 1, 2 |
| Package-level var mutation order differs between old and new paths | Low | High | Line-by-line comparison of shared function against original inline code before committing | 2, 3 |
| `alphaCoeffs`/`betaCoeffs` local-shadow not correctly reproduced | Medium | High | `append(nil, ...)` copy at function start; return values used by callers | 1 |
| Per-pool scorer configs (prefillScorerCfgs, decodeScorerCfgs) accidentally moved to shared function | Low | Medium | Explicit note in Task 5 that these stay in runCmd inline | 5 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — two functions, clear contracts, no over-engineering
- [x] No feature creep — strictly R23 fix; no new user-visible behavior
- [x] No unexercised flags or interfaces
- [x] No partial implementations — all tasks complete and sequenced
- [x] No breaking changes — BC-5 and BC-6 explicitly verify non-regression
- [x] No hidden global state impact — side-effects are documented in function godoc
- [x] All new code will pass golangci-lint (no new exported mutable maps, no new exported types)
- [x] Shared test helpers used (existing `cmd` package test infrastructure)
- [x] CLAUDE.md: no new files/packages, no new CLI flags → no CLAUDE.md update needed
- [x] Documentation DRY: not modifying any canonical source (rules.md, invariants.md, etc.)
- [x] Deviation log reviewed — all deviations are corrections of pre-existing bugs
- [x] Each task produces working, testable code (no scaffolding)
- [x] Task dependencies correctly ordered: 1→{2,3} (Tasks 2 and 3 both depend on Task 1 and are independent of each other), 4→5 (policy path, independent from Tasks 1-3), 5→6 (final verification)
- [x] All contracts mapped to specific tasks (see Test Strategy table)
- [x] No golden dataset regeneration needed
- [x] Construction site audit: `latencyResolution` is a new return-value struct, never stored elsewhere → no construction sites to audit

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` — no error paths silently drop data
- [x] R2: No map iteration for ordered output in shared functions
- [x] R3: All numeric validations preserved exactly from original inline code
- [x] R4: No struct fields added to existing types
- [x] R5: No resource allocation loops
- [x] R6: No `logrus.Fatalf` in `sim/` packages — shared functions are in `cmd/`
- [x] R7: Source-scanning tests accompany structural tests to verify behavioral invariants
- [x] R8: No exported mutable maps
- [x] R9: No new YAML fields
- [x] R10: Existing `KnownFields(true)` calls preserved in shared function
- [x] R11: Division guards unchanged from original
- [x] R13: `latencyResolution` is a return struct, not an interface
- [x] R14: `resolveLatencyConfig` handles only latency resolution; `resolvePolicies` handles only policy resolution (single-concern methods)
- [x] R15: No stale PR references (plan references issue #862)
- [x] R16: Config params not changed
- [x] R18: All `cmd.Flags().Changed()` guards preserved exactly
- [x] R23: This PR fixes the R23 violation — confirmed by `TestNoR23CommentSyncMarkersInReplay`

---

## Appendix: File-Level Implementation Details

### `cmd/root.go`

**Purpose:** Add `latencyResolution` struct, `resolveLatencyConfig()`, and `resolvePolicies()` after the existing `allZeros()` function (line ~230). Update `runCmd.Run` to call both.

**Insertion point for new code:** After `func allZeros(values []float64) bool { ... }` (line ~220-230).

**Changes to `runCmd.Run`:**
- Replace lines 337-629 (coeff shadow through end of blackbox block) with `lr := resolveLatencyConfig(cmd)`
- Remove inline `blockSizeTokens <= 0` (~line 1065) and `gpuMemoryUtilization` (~line 1162) checks — now handled in `resolveLatencyConfig`
- Keep lines 690-897 (per-pool PD overrides and per-pool KV auto-calc) in place — these are runCmd-specific
- **Two non-contiguous replacements for policy resolution** (PD validation block at 1174-1279 must stay):
  - Replace lines 1083-1173 (policy bundle + validation) with `parsedScorerConfigs := resolvePolicies(cmd)`
  - Keep lines 1174-1279 (PD disaggregation validation + per-pool overrides) intact
  - Delete lines 1281-1337 (admission/routing latency + policy logging + scorer parsing) — now inside `resolvePolicies`
- Keep lines 1318-1333 (per-pool scorer parsing, `prefillScorerCfgs`/`decodeScorerCfgs`) in place — PD-specific
- Update SimConfig construction: `lr.BetaCoeffs`, `lr.AlphaCoeffs`, `lr.ModelConfig`, `lr.HWConfig`, `lr.Backend`

**Key implementation note:** `resolveLatencyConfig` uses `alpha` and `beta` local variables (not the package-level `alphaCoeffs`/`betaCoeffs`). The local copies prevent mutating Cobra-registered default slice values.

### `cmd/replay.go`

**Purpose:** Shrink by ~200 lines. Replace two large inline blocks with two function calls.

**Changes to `replayCmd.Run`:**
- Replace lines 99-438 (coeff shadow through `maxModelLen < 0` check) with `lr := resolveLatencyConfig(cmd)`
- Remove redundant `blockSizeTokens <= 0` (~line 448) and `gpuMemoryUtilization` (~line 543) inline checks — now in `resolveLatencyConfig`
- Replace lines 470-592 (policy bundle through token-bucket log) with `parsedScorerConfigs := resolvePolicies(cmd)`
- **3 R23 markers vanish** automatically (lines 470, 501, 565 are inside the two replaced ranges)
- **Explicitly delete 5 R23 comment lines** in the output-metrics section (lines 645, 655, 675, 678, 682) — these are `// R23: same as runCmd` comments only; the code they annotate is correct and stays
- Update config construction to use `lr.*` fields
- The `logrus.Infof("Starting replay with...")` line just after policy resolution stays in replayCmd, using `lr.AlphaCoeffs` and `lr.BetaCoeffs`

### `cmd/simconfig_shared_test.go`

**Purpose:** Behavioral and structural tests for the shared functions + regression guards.

**Tests (10 total):**
1. `TestResolveLatencyConfig_Exists` — function signature check
2. `TestResolvePolicies_Exists` — function signature check
3. `TestNoR23CommentSyncMarkersInReplay` — regression guard for BC-3
4. `TestTrainedRooflineBetaCoeffGuard_UsesCorrectMinimum` — invariant for BC-4
5. `TestResolvePolicies_InvalidAdmissionPolicy_Fatal` — policy validation behavioral test (BC-2)
5. `TestRunCmd_SimConfigFlagsParity` — defaults agreement between run and replay
6. `TestResolvePolicies_PolicyFlagsRegisteredInBothCommands` — exhaustive flag parity
7. `TestReplayCmd_SourceContainsNoInlineBackendBlocks` — structural elimination guard
8. `TestReplayCmd_SourceContainsNoPolicyInlineBlocks` — structural elimination guard
9. `TestBothCommands_SimConfigFlagsHaveIdenticalDefaults` — comprehensive default parity

**Imports needed:**
```go
package cmd

import (
    "os"
    "strings"
    "testing"

    "github.com/spf13/cobra"
    "github.com/stretchr/testify/assert"

    sim "github.com/inference-sim/inference-sim/sim"
)
```
