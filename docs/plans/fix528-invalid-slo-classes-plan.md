# Fix #528: Replace Invalid SLO Classes in Example Workload YAMLs

- **Goal:** Fix three example workload YAML files that use deprecated SLO class names, replacing them with canonical v2 tier names so the source YAML matches the valid set without relying on auto-upgrade.
- **The problem today:** PR #485 introduced three regression workload examples (`cache_warmup`, `load_spikes`, `multiturn`) that use `"interactive"` and `"realtime"` as SLO class values. While these currently *pass* validation (because `UpgradeV1ToV2()` auto-maps them before `Validate()` runs), the source YAML files contain deprecated v1 tier names that: (a) emit deprecation warnings at load time, (b) confuse users reading the examples who see names not in the documented valid set `{critical, standard, sheddable, batch, background}`, and (c) break if auto-upgrade is ever removed.
- **What this PR adds:**
  1. Replaces `"interactive"` → `"standard"` in all three files (6 occurrences) — "standard" is the closest valid tier for interactive workloads.
  2. Replaces `"realtime"` → `"critical"` in all three files (3 occurrences) — "critical" is the highest-priority valid tier, matching the intent of "realtime".
- **Why this matters:** Example files are the first thing users try. Deprecated names in examples erode trust, emit confusing warnings, and teach non-canonical patterns.
- **Architecture:** Pure YAML content fix — no Go code changes. The valid SLO class set is enforced by `validateClient()` in `sim/workload/spec.go`.
- **Source:** [Issue #528](https://github.com/inference-sim/inference-sim/issues/528)
- **Closes:** Fixes #528
- **Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** Example workload YAML files in `examples/`.
2. **Adjacent blocks:** `sim/workload/spec.go` (validation), `cmd/root.go` (`--workload-spec` flag loading).
3. **Invariants touched:** None — this is a data-only fix.
4. **Construction Site Audit:** N/A — no struct fields added.

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes 9 deprecated SLO class names across 3 example workload YAML files introduced in PR #485. The mapping is `"interactive"` → `"standard"` (6 occurrences) and `"realtime"` → `"critical"` (3 occurrences), matching the `v1ToV2SLOClasses` auto-upgrade mapping already in `spec.go`. The files currently pass validation via auto-upgrade, but the source YAML uses non-canonical names that emit deprecation warnings and confuse users. After this fix, the source files use canonical v2 tier names directly. No Go production code changes. No behavioral changes to the simulator.

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: Example YAML files use canonical v2 SLO class names
- GIVEN any example workload YAML file in examples/
- WHEN the raw slo_class values are extracted from the YAML source
- THEN every slo_class value passes IsValidSLOClass() without auto-upgrade
```

```
BC-2: SLO class semantics preserved
- GIVEN the original intent of "interactive" (moderate-priority user-facing) and "realtime" (highest-priority latency-sensitive)
- WHEN replaced with valid SLO classes
- THEN "interactive" maps to "standard" and "realtime" maps to "critical", preserving the priority ordering intent
```

**Negative contracts:**

```
BC-3: No behavioral changes to simulator
- GIVEN the fix modifies only example YAML files
- WHEN existing tests are run
- THEN all tests pass with identical results (no golden dataset changes)
```

### C) Component Interaction

```
examples/*.yaml  ──(loaded by)──>  sim/workload/spec.go:LoadWorkloadSpec()
                                          │
                                          ▼
                                   validateClient() checks slo_class
                                   against validSLOClasses map
                                          │
                                          ▼
                                   ERROR if not in {critical, standard,
                                   sheddable, batch, background, ""}
```

This PR only modifies the left side (YAML files). The validation code is unchanged.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue suggests `"interactive"` → `"standard"` or equivalent | Uses `"standard"` | SIMPLIFICATION — "standard" is the closest semantic match for interactive workloads |
| Issue suggests `"realtime"` → `"critical"` or equivalent | Uses `"critical"` | SIMPLIFICATION — "critical" is the highest-priority tier, matching "realtime" intent |

### E) Review Guide

**Tricky part:** None — this is a mechanical string replacement.
**Scrutinize:** Verify every occurrence of `"interactive"` and `"realtime"` is replaced across all three files.
**Safe to skim:** File comments (unchanged).
**Known debt:** YAML comments and client IDs (e.g., `"realtime-api"`, `"light-prefix-interactive"`) still reference the old terminology. These describe workload *intent* (e.g., "realtime traffic" = latency-sensitive use case), not the SLO class value, so they remain accurate as prose. Client IDs are arbitrary identifiers with no validation. No update needed.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to modify (3):
- `examples/regression_workload_cache_warmup.yaml` — replace 2× `"interactive"` → `"standard"`, 1× `"realtime"` → `"critical"`
- `examples/regression_workload_load_spikes.yaml` — replace 1× `"interactive"` → `"standard"`, 1× `"realtime"` → `"critical"`
- `examples/regression_workload_multiturn.yaml` — replace 3× `"interactive"` → `"standard"`, 1× `"realtime"` → `"critical"`

File to add (1):
- `sim/workload/spec_test.go` — add `TestExampleWorkloadFiles_AllValid` (glob-based, validates ALL example YAMLs)

No new Go production code. No dead code.

### G) Task Breakdown

#### Task 1: Replace invalid SLO classes in all three YAML files (BC-1, BC-2)

**Step 1:** Replace all `"interactive"` → `"standard"` and `"realtime"` → `"critical"` in the three files.

**Step 2:** Write a validation test that loads each example file and confirms it passes validation.

```go
// In sim/workload/spec_test.go
func TestExampleWorkloadFiles_AllValid(t *testing.T) {
    // Glob all example YAML files — catches future regressions too.
    files, err := filepath.Glob("../../examples/*.yaml")
    require.NoError(t, err)
    require.NotEmpty(t, files, "no example YAML files found — check relative path")
    for _, path := range files {
        t.Run(filepath.Base(path), func(t *testing.T) {
            spec, err := LoadWorkloadSpec(path)
            require.NoError(t, err, "failed to load %s", path)
            require.NoError(t, spec.Validate(), "validation failed for %s", path)
        })
    }
}

func TestExampleWorkloadFiles_CanonicalSLOClasses(t *testing.T) {
    // BC-1: Verify raw YAML slo_class values are canonical v2 names,
    // not deprecated v1 names that rely on auto-upgrade.
    files, err := filepath.Glob("../../examples/*.yaml")
    require.NoError(t, err)
    require.NotEmpty(t, files)
    for _, path := range files {
        t.Run(filepath.Base(path), func(t *testing.T) {
            data, err := os.ReadFile(path)
            require.NoError(t, err)
            var raw struct {
                Clients []struct {
                    SLOClass string `yaml:"slo_class"`
                } `yaml:"clients"`
            }
            require.NoError(t, yaml.Unmarshal(data, &raw))
            for i, c := range raw.Clients {
                require.True(t, IsValidSLOClass(c.SLOClass),
                    "client[%d] slo_class %q is not a canonical v2 tier name in %s",
                    i, c.SLOClass, filepath.Base(path))
            }
        })
    }
}
```

Note: `go test` sets CWD to the package directory, so `../../examples/` reliably resolves to the repo root's `examples/` from `sim/workload/`. The first test verifies load+validate works. The second test verifies raw YAML values are canonical — this test would FAIL before the fix (because "interactive" and "realtime" are not in `validSLOClasses`).

**Step 3:** Run tests:
```bash
go test ./sim/workload/... -run TestExampleWorkloadFiles
```
Expected: PASS

**Step 4:** Run lint:
```bash
golangci-lint run ./sim/workload/...
```
Expected: 0 issues

**Step 5:** Run full test suite:
```bash
go test ./...
```
Expected: All tests pass (no golden dataset changes)

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestExampleWorkloadFiles_CanonicalSLOClasses |
| BC-1 | Task 1 | Integration | TestExampleWorkloadFiles_AllValid |
| BC-2 | Task 1 | Manual verification | Inspect YAML diffs |
| BC-3 | Task 1 | Regression | `go test ./...` (full suite) |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Wrong semantic mapping (e.g., "interactive" should be "critical") | Low | Low | Issue #528 confirms the mapping; comments in YAML describe intent | Task 1 |
| Missed occurrence | Low | Medium | Grep for all occurrences before/after | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions.
- [x] No feature creep beyond PR scope.
- [x] No unexercised flags or interfaces.
- [x] No partial implementations.
- [x] No breaking changes without explicit contract updates.
- [x] No hidden global state impact.
- [x] All new code will pass golangci-lint.
- [x] Shared test helpers used (LoadWorkloadSpec from existing package).
- [x] CLAUDE.md update not needed (no new files/packages/flags).
- [x] No stale references in CLAUDE.md.
- [x] Documentation DRY: no canonical sources modified.
- [x] Deviation log reviewed — no unresolved deviations.
- [x] Each task produces working, testable code.
- [x] Task dependencies are correctly ordered (single task).
- [x] All contracts are mapped to specific tasks.
- [x] Golden dataset regeneration not needed.
- [x] Construction site audit N/A.
- [x] Not part of a macro plan.

**Antipattern rules:** N/A — no Go production code changes. The only new code is a test.

---

## Appendix: File-Level Implementation Details

### File: `examples/regression_workload_cache_warmup.yaml`
- **Purpose:** Example workload for testing cache warmup behavior.
- **Changes:** Line 40: `"interactive"` → `"standard"`. Line 62: `"interactive"` → `"standard"`. Line 82: `"realtime"` → `"critical"`.

### File: `examples/regression_workload_load_spikes.yaml`
- **Purpose:** Example workload for testing load spike handling.
- **Changes:** Line 69: `"realtime"` → `"critical"`. Line 91: `"interactive"` → `"standard"`.

### File: `examples/regression_workload_multiturn.yaml`
- **Purpose:** Example workload for testing multi-turn session affinity.
- **Changes:** Line 51: `"interactive"` → `"standard"`. Line 79: `"interactive"` → `"standard"`. Line 107: `"interactive"` → `"standard"`. Line 135: `"realtime"` → `"critical"`.
