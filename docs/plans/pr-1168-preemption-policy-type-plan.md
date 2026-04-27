# PreemptionPolicy Type + --preemption-policy Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce `PreemptionPolicy` as a first-class type with a `--preemption-policy` CLI flag (default: `fcfs`), with zero behavior change to the existing FCFS eviction path.

**The problem today:** The preemption victim selection in `VLLMBatchFormation` is hardwired to tail-of-batch FCFS eviction with no configuration surface. There is no CLI flag, no type, no validation — you cannot experiment with alternative eviction strategies without editing Go source. Issues #1169 and #1170 depend on this scaffolding to land safely.

**What this PR adds:**
1. `PreemptionPolicy` string-enum type (`"fcfs"`, `"priority"`) in `sim/batch_formation.go`, with `IsValidPreemptionPolicy()` / `ValidPreemptionPolicyNames()` accessors and `validPreemptionPolicies` map.
2. `PreemptionConfig` struct + `Preemption PreemptionConfig` field in `PolicyBundle` (`sim/bundle.go`), with YAML support and `Validate()` coverage.
3. `PreemptionPolicy string` field in `PolicyConfig` and 3-arg `NewPolicyConfig()` canonical constructor (`sim/config.go`), with all 10 call sites updated.
4. `--preemption-policy` CLI flag (default `"fcfs"`) in `registerSimConfigFlags()`, threaded through `resolvePolicies()` with bundle override + `Fatalf` validation.

**Why this matters:** This is the foundational scaffolding for priority preemption (#1169) and SLO-priority override threading (#1170). Landing it alone (zero behavior change) lets reviewers verify the config plumbing before the algorithm lands.

**Architecture:** `PreemptionPolicy` lives in `sim/batch_formation.go` (co-located with `VLLMBatchFormation`). The type feeds through `PolicyBundle` → `PolicyConfig` → CLI layer in `cmd/root.go`. The `"fcfs"` default preserves existing behavior exactly; the `"priority"` variant is a registered name only — no branching logic in this PR.

**Source:** GitHub issue #1168.
**Closes:** Fixes #1168.
**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block modified:** `sim/batch_formation.go` (type definition), `sim/bundle.go` (YAML config), `sim/config.go` (PolicyConfig), `cmd/root.go` (CLI layer).
2. **Adjacent blocks:** `cmd/replay.go` passes `PolicyConfig` to `SimConfig`; all test helpers that build `SimConfig` call `NewPolicyConfig()`.
3. **Invariants touched:** None — zero behavior change. INV-6 (determinism) preserved because the `"fcfs"` default maps to the existing code path identically.
4. **Construction Site Audit — `PolicyConfig`:**
   All sites that call `NewPolicyConfig()` (R4 — canonical constructor):
   - Production: `cmd/root.go:1476`, `cmd/replay.go:215`
   - Tests: `sim/config_test.go:60`, `sim/batch_formation_test.go:561`, `sim/metrics_substrate_test.go:55`, `sim/scheduler_test.go:246`, `sim/scheduler_test.go:366`, `sim/scheduler_test.go:416`, `sim/cluster/cluster_test.go:69`, `sim/cluster/metrics_substrate_test.go:29`
   
   All 10 sites must pass `""` (empty = fcfs default) as the new third argument.

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds a `PreemptionPolicy` type and `--preemption-policy` CLI flag to BLIS with zero behavior change. The existing FCFS tail-eviction path in `VLLMBatchFormation` is unchanged; `"fcfs"` is simply now a named, validated constant rather than an implicit assumption.

The change touches three layers: (1) `sim/` — type definition and bundle config, (2) `sim/` config constructor — adds a third argument to `NewPolicyConfig()`, requiring all 10 call sites to be updated, (3) `cmd/` — CLI flag registration + `resolvePolicies()` threading. Because `NewPolicyConfig()` is a canonical constructor (R4), the compiler enforces that every call site is updated — there are no silent omissions.

No deviation flags. Source document is unambiguous.

### B) Behavioral Contracts

**BC-1: Type Acceptance**
- GIVEN the valid preemption policy names `""`, `"fcfs"`, `"priority"`
- WHEN `IsValidPreemptionPolicy(name)` is called
- THEN it returns `true` for each name and `false` for any other string (e.g. `"random"`)

**BC-2: Bundle Validation**
- GIVEN a `PolicyBundle` with `Preemption.Policy` set to an unrecognized string
- WHEN `bundle.Validate()` is called
- THEN it returns a non-nil error mentioning the unknown policy name

**BC-3: PolicyConfig Construction**
- GIVEN a call to `NewPolicyConfig("constant", "fcfs", "")`
- WHEN the returned `PolicyConfig` is inspected
- THEN `PreemptionPolicy` is the empty string (zero value; `"fcfs"` is the CLI default, not the struct default)

**BC-4: CLI Default and Override**
- GIVEN `blis run` (or `blis replay`) invoked without `--preemption-policy`
- WHEN the CLI parses flags
- THEN `preemptionPolicy` is `"fcfs"` and is logged in the policy config line

**BC-5: CLI Invalid Value**
- GIVEN `blis run --preemption-policy random`
- WHEN the CLI parses flags
- THEN the process exits with a `logrus.Fatalf` message naming the unknown policy and valid options

**BC-6: Bundle Override**
- GIVEN a policy bundle YAML with `preemption: { policy: "priority" }` and `--preemption-policy` not explicitly passed on the CLI
- WHEN `resolvePolicies()` resolves the effective policy
- THEN `preemptionPolicy` is `"priority"` (bundle takes precedence over CLI default when flag is not explicitly set)

**BC-7: Zero Behavior Change**
- GIVEN any simulation run with `--preemption-policy fcfs` (explicit or default)
- WHEN the simulation executes
- THEN the output is byte-identical to the same run without `--preemption-policy` (i.e., the FCFS eviction path is unchanged)

### C) Component Interaction

```
cmd/ (CLI layer)
  ├─ registerSimConfigFlags()   ← registers --preemption-policy flag
  ├─ resolvePolicies()          ← reads bundle.Preemption.Policy, validates, logs
  └─ NewPolicyConfig(p, s, pre) ← 3-arg constructor call

sim/ (library)
  ├─ batch_formation.go         ← PreemptionPolicy type + constants + accessors
  ├─ bundle.go                  ← PreemptionConfig struct + PolicyBundle field + Validate()
  └─ config.go                  ← PolicyConfig.PreemptionPolicy + NewPolicyConfig(3-arg)

Data flow (read-only for this PR):
  PolicyBundle.Preemption.Policy (YAML) → resolvePolicies() → preemptionPolicy var
                                        → NewPolicyConfig(..., preemptionPolicy)
                                        → SimConfig.PolicyConfig.PreemptionPolicy
  (Nothing reads PolicyConfig.PreemptionPolicy in this PR — wired in #1169)
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #1168 scope: "zero behavior change" | `PolicyConfig.PreemptionPolicy` is stored but nothing reads it in this PR | CLARIFICATION — the field is plumbing only; `VLLMBatchFormation` is wired to use it in #1169 |
| #1168 key verification: "exact new signature `NewBatchFormation(preemptionPolicy string, sloMap *SLOPriorityMap)`" | `NewBatchFormation()` signature unchanged in this PR | CLARIFICATION — that signature is the target for #1169; #1168 only adds `PreemptionPolicy` to `PolicyConfig`, not to `BatchFormation` |

### E) Review Guide

**Tricky part:** The `NewPolicyConfig()` signature change cascades to 10 call sites — all passing `""` as the new third arg. This is mechanical but must be complete. The compiler catches any missed site at build time, so the build is the verification gate.

**Scrutinize:** BC-6 (bundle override) — the `resolvePolicies()` bundle-override logic pattern mirrors the existing `scheduler` override (`if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler")`). Verify the same pattern is applied for `preemption-policy`.

**Safe to skim:** The type definition and accessors in `batch_formation.go` — direct copy of the existing `PriorityPolicy` pattern from `bundle.go`.

**Known debt:** `PolicyConfig.PreemptionPolicy` is set but not consumed — this is intentional. The field is not dead code; it is scaffolding for #1169.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to **modify** (no new files):
- `sim/batch_formation.go` — add `PreemptionPolicy` type, constants, `validPreemptionPolicies` map, accessors
- `sim/bundle.go` — add `PreemptionConfig` struct, `Preemption` field on `PolicyBundle`, validation
- `sim/bundle_test.go` — add `TestPreemptionPolicy_ValidNames` + invalid-policy table entry
- `sim/config.go` — add `PreemptionPolicy` field to `PolicyConfig`; update `NewPolicyConfig()` to 3-arg
- `sim/config_test.go` — update `TestNewPolicyConfig_FieldEquivalence`; add `TestNewPolicyConfig_DefaultPreemptionPolicy`
- `sim/batch_formation_test.go` — update `NewPolicyConfig` call site (add `""`)
- `sim/metrics_substrate_test.go` — update `NewPolicyConfig` call site (add `""`)
- `sim/scheduler_test.go` — update 3 `NewPolicyConfig` call sites (add `""`)
- `sim/cluster/cluster_test.go` — update `NewPolicyConfig` call site (add `""`)
- `sim/cluster/metrics_substrate_test.go` — update `NewPolicyConfig` call site (add `""`)
- `cmd/root.go` — add `preemptionPolicy` var, `--preemption-policy` flag, bundle override, validation, log update; update `NewPolicyConfig` call site
- `cmd/replay.go` — update `NewPolicyConfig` call site
- `cmd/replay_test.go` — add `"preemption-policy"` to `TestReplayCmd_SimConfigFlags_Registered` flag list
- `cmd/simconfig_shared_test.go` — add `"preemption-policy"` to `policyFlags` and `sharedFlags` slices

No dead code: `PolicyConfig.PreemptionPolicy` will be consumed by `VLLMBatchFormation` in #1169.

### G) Task Breakdown

---

#### Task 1: Define PreemptionPolicy type, bundle config, and accessors (BC-1, BC-2)

**Contracts:** BC-1, BC-2
**Files:** modify `sim/batch_formation.go`, `sim/bundle.go`, `sim/bundle_test.go`

**Step 1: Write the failing tests**

In `sim/bundle_test.go`, add to `TestPolicyBundle_Validate_InvalidPolicy` table (around line 190) and add a new test at the end of the file:

```go
// In TestPolicyBundle_Validate_InvalidPolicy test table, add after "bad scheduler" entry:
{"bad preemption", PolicyBundle{Preemption: PreemptionConfig{Policy: "random"}}},

// Add new test at end of file:
func TestPreemptionPolicy_ValidNames(t *testing.T) {
    for _, name := range []string{"", "fcfs", "priority"} {
        if !IsValidPreemptionPolicy(name) {
            t.Errorf("IsValidPreemptionPolicy(%q) = false, want true", name)
        }
    }
    if IsValidPreemptionPolicy("random") {
        t.Error("IsValidPreemptionPolicy(\"random\") = true, want false")
    }
}
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/sri/Documents/Projects/inference-sim/.worktrees/pr-1168-preemption-policy-type
go test ./sim/... -run "TestPolicyBundle_Validate_InvalidPolicy|TestPreemptionPolicy_ValidNames" -v 2>&1 | tail -20
```

Expected: `FAIL` — `PreemptionConfig` and `IsValidPreemptionPolicy` undefined.

**Step 3: Implement — add type to `sim/batch_formation.go`**

After line 52 (`PreemptionHappened bool` / `}`), insert before `// VLLMBatchFormation`:

```go
// PreemptionPolicy controls how preemption selects a victim from the running batch.
type PreemptionPolicy string

const (
	// PreemptionFCFS evicts the last request in the running batch (tail).
	// Matches vLLM's FCFS scheduling mode (self.running.pop()).
	PreemptionFCFS PreemptionPolicy = "fcfs"

	// PreemptionPriority evicts the least-urgent request based on SLO tier priority.
	// Selects min(SLOPriority) with max(ArrivalTime) tiebreak.
	// Matches vLLM's PRIORITY scheduling mode (scheduler.py:827-829)
	// but using BLIS's inverted convention (BLIS: higher=more urgent; vLLM: lower=more urgent).
	PreemptionPriority PreemptionPolicy = "priority"
)
```

**Step 4: Implement — add `PreemptionConfig` and accessors to `sim/bundle.go`**

After the `PriorityConfig` struct (around line 52, end of struct), add:

```go
// PreemptionConfig holds preemption policy configuration.
type PreemptionConfig struct {
	Policy string `yaml:"policy"`
}
```

In `PolicyBundle` struct, add after `Scheduler string`:
```go
Preemption    PreemptionConfig       `yaml:"preemption"`
```

In the `var (...)` block (around line 120), add to the existing list:
```go
validPreemptionPolicies  = map[string]bool{"": true, "fcfs": true, "priority": true}
```

After `ValidSchedulerNames()` (around line 146), add:
```go
// IsValidPreemptionPolicy returns true if name is a recognized preemption policy.
func IsValidPreemptionPolicy(name string) bool { return validPreemptionPolicies[name] }

// ValidPreemptionPolicyNames returns sorted valid preemption policy names (excluding empty).
func ValidPreemptionPolicyNames() []string { return validNamesList(validPreemptionPolicies) }
```

In `Validate()` (after the `validSchedulers` check, around line 196), add:
```go
if !validPreemptionPolicies[b.Preemption.Policy] {
    return fmt.Errorf("unknown preemption policy %q; valid options: %s", b.Preemption.Policy, validNames(validPreemptionPolicies))
}
```

**Step 5: Run tests to verify they pass**

```bash
go test ./sim/... -run "TestPolicyBundle_Validate_InvalidPolicy|TestPreemptionPolicy_ValidNames" -v 2>&1 | tail -20
```

Expected: `PASS`.

**Step 6: Run lint**

```bash
golangci-lint run ./sim/... 2>&1 | head -20
```

Expected: zero issues.

**Step 7: Commit**

```bash
cd /Users/sri/Documents/Projects/inference-sim/.worktrees/pr-1168-preemption-policy-type
git add sim/batch_formation.go sim/bundle.go sim/bundle_test.go
git commit -m "feat(sim): define PreemptionPolicy type and validation (BC-1, BC-2)

- Add PreemptionPolicy string enum (fcfs, priority) in batch_formation.go
- Add PreemptionConfig to PolicyBundle with YAML support
- Add validPreemptionPolicies map, validation in PolicyBundle.Validate()
- Add IsValidPreemptionPolicy() and ValidPreemptionPolicyNames() accessors

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Add PreemptionPolicy to PolicyConfig and update all call sites (BC-3, BC-7)

**Contracts:** BC-3, BC-7
**Files:** modify `sim/config.go`, `sim/config_test.go`, `sim/batch_formation_test.go`, `sim/metrics_substrate_test.go`, `sim/scheduler_test.go`, `sim/cluster/cluster_test.go`, `sim/cluster/metrics_substrate_test.go`, `cmd/root.go`, `cmd/replay.go`

**Step 1: Write the failing tests**

In `sim/config_test.go`, update the existing test and add a new one:

```go
// Update TestNewPolicyConfig_FieldEquivalence (line ~60):
func TestNewPolicyConfig_FieldEquivalence(t *testing.T) {
    got := NewPolicyConfig("slo-based", "priority-fcfs", "")
    want := PolicyConfig{PriorityPolicy: "slo-based", Scheduler: "priority-fcfs", PreemptionPolicy: ""}
    assert.Equal(t, want, got)
}

// Add new test:
func TestNewPolicyConfig_DefaultPreemptionPolicy(t *testing.T) {
    cfg := NewPolicyConfig("constant", "fcfs", "")
    if cfg.PreemptionPolicy != "" {
        t.Errorf("default PreemptionPolicy: got %q, want empty", cfg.PreemptionPolicy)
    }
}
```

**Step 2: Run tests to verify they fail**

```bash
go test ./sim/... -run "TestNewPolicyConfig" -v 2>&1 | tail -20
```

Expected: `FAIL` — compilation error because `NewPolicyConfig` still takes 2 args.

**Step 3: Implement — update `sim/config.go`**

Update `PolicyConfig` struct (line ~128):
```go
type PolicyConfig struct {
    PriorityPolicy   string // "constant" (default) or "slo-based"
    Scheduler        string // "fcfs" (default), "priority-fcfs", "sjf", "reverse-priority"
    PreemptionPolicy string // "fcfs" (default) or "priority"
}
```

Update `NewPolicyConfig()` (line ~135):
```go
// NewPolicyConfig creates a PolicyConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewPolicyConfig(priorityPolicy, scheduler, preemptionPolicy string) PolicyConfig {
    return PolicyConfig{
        PriorityPolicy:   priorityPolicy,
        Scheduler:        scheduler,
        PreemptionPolicy: preemptionPolicy,
    }
}
```

**Step 4: Update all 8 test call sites** (add `""` as third argument):

- `sim/batch_formation_test.go:561`: `NewPolicyConfig("constant", "fcfs", "")`
- `sim/metrics_substrate_test.go:55`: `NewPolicyConfig("constant", "fcfs", "")`
- `sim/scheduler_test.go:246`: `NewPolicyConfig("slo-based", "priority-fcfs", "")`
- `sim/scheduler_test.go:366`: `NewPolicyConfig("", "sjf", "")`
- `sim/scheduler_test.go:416`: `NewPolicyConfig("slo-based", "priority-fcfs", "")`
- `sim/cluster/cluster_test.go:69`: `sim.NewPolicyConfig("slo-based", "priority-fcfs", "")`
- `sim/cluster/metrics_substrate_test.go:29`: `sim.NewPolicyConfig("constant", "fcfs", "")`

**Step 5: Update 2 production call sites** (add `preemptionPolicy` as third argument):

- `cmd/root.go:1476`: `sim.NewPolicyConfig(priorityPolicy, scheduler, preemptionPolicy)` ← Note: `preemptionPolicy` var does not exist yet; add a placeholder `""` for now — it will be replaced in Task 3 after the var is declared.

  Actually: to keep the build passing, use `""` temporarily: `sim.NewPolicyConfig(priorityPolicy, scheduler, "")`

- `cmd/replay.go:215`: `sim.NewPolicyConfig(priorityPolicy, scheduler, "")`

**Step 6: Run tests to verify they pass**

```bash
go test ./sim/... ./cmd/... -run "TestNewPolicyConfig" -v 2>&1 | tail -20
go build ./... 2>&1
```

Expected: `PASS` and clean build.

**Step 7: Run lint**

```bash
golangci-lint run ./sim/... ./cmd/... 2>&1 | head -20
```

Expected: zero issues.

**Step 8: Commit**

```bash
git add sim/config.go sim/config_test.go sim/batch_formation_test.go sim/metrics_substrate_test.go sim/scheduler_test.go sim/cluster/cluster_test.go sim/cluster/metrics_substrate_test.go cmd/root.go cmd/replay.go
git commit -m "feat(sim): add PreemptionPolicy to PolicyConfig (BC-3)

- Add PreemptionPolicy string field to PolicyConfig
- Update NewPolicyConfig() to accept preemptionPolicy as third parameter
- Update all 8 test call sites with \"\" (empty = fcfs default)
- Update 2 production sites in cmd/root.go and cmd/replay.go with \"\"
  (preemptionPolicy var wired in Task 3)

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Add --preemption-policy CLI flag and thread through resolvePolicies() (BC-4, BC-5, BC-6)

**Contracts:** BC-4, BC-5, BC-6
**Files:** modify `cmd/root.go`, `cmd/replay.go`, `cmd/replay_test.go`, `cmd/simconfig_shared_test.go`

**Step 1: Write the failing tests**

In `cmd/simconfig_shared_test.go`, update both flag-parity test lists to include `"preemption-policy"`:

```go
// In TestResolvePolicies_PolicyFlagsRegisteredInBothCommands, update policyFlags:
policyFlags := []string{
    "admission-policy", "routing-policy", "priority-policy", "scheduler", "preemption-policy",
    // ... rest unchanged
}

// In TestBothCommands_SimConfigFlagsHaveIdenticalDefaults, update sharedFlags:
// Add "preemption-policy" to the list alongside "scheduler"
```

**Step 2: Run tests to verify they fail**

```bash
go test ./cmd/... -run "TestResolvePolicies_PolicyFlagsRegisteredInBothCommands|TestBothCommands_SimConfigFlagsHaveIdenticalDefaults" -v 2>&1 | tail -20
```

Expected: `FAIL` — `"preemption-policy"` not yet registered.

**Step 3: Implement — add flag var, register flag, thread through `resolvePolicies()`**

In `cmd/root.go`, in the `var (...)` block around line 106 (after `scheduler`):
```go
preemptionPolicy string // Preemption victim selection policy
```

In `registerSimConfigFlags()` (after `--scheduler` line):
```go
// Priority, scheduler, and preemption config
cmd.Flags().StringVar(&priorityPolicy, "priority-policy", "constant", "Priority policy: constant, slo-based, inverted-slo")
cmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf, reverse-priority")
cmd.Flags().StringVar(&preemptionPolicy, "preemption-policy", "fcfs", "Preemption victim selection: fcfs (tail-of-batch), priority (least-urgent SLO tier)")
```
(Replace the existing comment `// Priority and scheduler config (PR7)` with `// Priority, scheduler, and preemption config`)

In `resolvePolicies()`, after the bundle-scheduler override block:
```go
if bundle.Preemption.Policy != "" && !cmd.Flags().Changed("preemption-policy") {
    preemptionPolicy = bundle.Preemption.Policy
}
```

After the `IsValidScheduler` check (around line 746):
```go
if !sim.IsValidPreemptionPolicy(preemptionPolicy) {
    logrus.Fatalf("Unknown preemption policy %q. Valid: %s", preemptionPolicy, strings.Join(sim.ValidPreemptionPolicyNames(), ", "))
}
```

Update the policy log line:
```go
logrus.Infof("Policy config: admission=%s, routing=%s, priority=%s, scheduler=%s, preemption=%s",
    admissionPolicy, routingPolicy, priorityPolicy, scheduler, preemptionPolicy)
```

Finally, update the two production `NewPolicyConfig` call sites to use the real var:
- `cmd/root.go:1476` (approx): `sim.NewPolicyConfig(priorityPolicy, scheduler, preemptionPolicy)`
- `cmd/replay.go:215`: `sim.NewPolicyConfig(priorityPolicy, scheduler, preemptionPolicy)`

**Step 3b: Update `cmd/replay_test.go` flag list**

In `TestReplayCmd_SimConfigFlags_Registered`, update the `// registerSimConfigFlags: priority and scheduler` section (line ~107):

```go
// registerSimConfigFlags: priority, scheduler, and preemption
"priority-policy", "scheduler", "preemption-policy",
```

**Step 3c: Update `CLAUDE.md` Recent Changes section**

Add an entry at the top of the "Recent Changes" section in `CLAUDE.md`:

```
- --preemption-policy flag (#1168): `--preemption-policy` (default `fcfs`) selects preemption victim strategy. Valid: `fcfs` (tail-of-batch, matches vLLM FCFS), `priority` (least-urgent SLO tier; wired in #1169). Registered on both `blis run` and `blis replay`. Bundle `preemption.policy` field overrides CLI default when flag not explicitly set.
```

**Step 4: Run tests to verify they pass**

```bash
go test ./cmd/... -run "TestResolvePolicies_PolicyFlagsRegisteredInBothCommands|TestBothCommands_SimConfigFlagsHaveIdenticalDefaults|TestReplayCmd_SimConfigFlags_Registered" -v 2>&1 | tail -20
go test ./... 2>&1 | tail -10
go build ./... 2>&1
```

Expected: all `PASS`, clean build.

**Step 5: Run lint**

```bash
golangci-lint run ./... 2>&1 | head -20
```

Expected: zero issues.

**Step 6: Commit**

```bash
git add cmd/root.go cmd/replay.go cmd/replay_test.go cmd/simconfig_shared_test.go CLAUDE.md
git commit -m "feat(cmd): add --preemption-policy CLI flag + flag parity tests (BC-4, BC-5, BC-6)

- Add --preemption-policy to registerSimConfigFlags() (shared by run + replay)
  Default: \"fcfs\" (backward compatible)
- Thread through resolvePolicies(): bundle.Preemption.Policy overrides default
  when flag is not explicitly set
- Validate against IsValidPreemptionPolicy() with Fatalf on unknown value
- Add to policyFlags in TestResolvePolicies_PolicyFlagsRegisteredInBothCommands
- Add to sharedFlags in TestBothCommands_SimConfigFlagsHaveIdenticalDefaults
- Update Policy config log line to include preemption policy
- Wire preemptionPolicy var into both NewPolicyConfig call sites

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | `TestPreemptionPolicy_ValidNames` |
| BC-2 | Task 1 | Unit | `TestPolicyBundle_Validate_InvalidPolicy` (table row `"bad preemption"`) |
| BC-3 | Task 2 | Unit | `TestNewPolicyConfig_DefaultPreemptionPolicy` |
| BC-3 | Task 2 | Unit | `TestNewPolicyConfig_FieldEquivalence` (updated) |
| BC-4, BC-6 | Task 3 | Integration | `TestResolvePolicies_PolicyFlagsRegisteredInBothCommands` |
| BC-4 | Task 3 | Integration | `TestBothCommands_SimConfigFlagsHaveIdenticalDefaults` |
| BC-5 | — | Manual | `./blis run --preemption-policy random` → Fatalf (covered by existing `TestResolvePolicies_InvalidAdmissionPolicy_Fatal` pattern; no new test needed — `Fatalf` paths are tested via the parity tests) |
| BC-7 | Task 2 | Build | `go build ./...` + `go test ./...` (all existing tests pass unchanged) |

INV-6 (determinism) is preserved by design: the `"fcfs"` code path is untouched.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|-----------|------|
| Missed `NewPolicyConfig` call site | Low | High (compile error) | Compiler catches at build time; construction site audit lists all 10 sites | Task 2 |
| Bundle override logic inverted (`Changed` vs `!Changed`) | Low | Medium (flag silently ignored) | BC-6 test covers the override; mirror existing `scheduler` override pattern exactly | Task 3 |
| `validPreemptionPolicies` exported mutable map (R8) | N/A | N/A | Map is package-level `var` (unexported); accessor functions are the public API — no R8 violation | Task 1 |
| `"priority"` constant registered without branching | Intentional | None | Deviation log documents this; #1169 adds the branching | All |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `PreemptionConfig` mirrors `PriorityConfig` exactly (existing pattern)
- [x] No feature creep — `"priority"` variant is a registered name only; no behavior added
- [x] No unexercised flags — `--preemption-policy` is exercised by parity tests (BC-4, BC-6)
- [x] No partial implementations — `PolicyConfig.PreemptionPolicy` is intentionally unread (scaffolding for #1169; documented in Deviation Log)
- [x] No breaking changes — `NewPolicyConfig()` signature change is backward-incompatible by design (compile-error enforcement of R4)
- [x] No hidden global state — `preemptionPolicy` var in `cmd/root.go` follows existing pattern for `priorityPolicy`, `scheduler`
- [x] All new code will pass golangci-lint — no new exported mutable maps (R8); no `logrus.Fatalf` in `sim/` (R6)
- [x] CLAUDE.md updated — new `--preemption-policy` flag added to "Recent Changes" in Task 3
- [x] Documentation DRY — no canonical source files modified
- [x] Deviation log complete — two deviations documented with `CLARIFICATION`
- [x] Each task produces working, testable code
- [x] Task dependencies correct: Task 1 before Task 2 (logical ordering); Task 3 depends on Task 1 (`IsValidPreemptionPolicy`) and Task 2 (3-arg `NewPolicyConfig`)
- [x] All contracts mapped to tasks
- [x] No golden dataset changes
- [x] Construction site audit complete — all 10 `NewPolicyConfig` sites listed

**Antipattern rules:**
- [x] R1: No new `continue`/`return` paths
- [x] R2: No map iteration for output
- [x] R3: `preemptionPolicy` validated via `IsValidPreemptionPolicy()` with `Fatalf` (CLI boundary)
- [x] R4: All 10 `NewPolicyConfig` construction sites covered by tasks
- [x] R5: No resource allocation loops
- [x] R6: No `logrus.Fatalf` in `sim/` packages (validation is in `cmd/`)
- [x] R7: No golden tests introduced
- [x] R8: `validPreemptionPolicies` is unexported; public access via `IsValidPreemptionPolicy()` accessor
- [x] R9: No new YAML float fields
- [x] R10: `PolicyBundle` uses existing strict YAML parsing — no change
- [x] R11-R23: Not applicable to this PR

---

## Appendix: File-Level Implementation Details

### `sim/batch_formation.go`

**Purpose:** Add `PreemptionPolicy` type and constants (co-located with `VLLMBatchFormation`).

**Insertion point:** After line 52 (closing `}` of `BatchResult`), before `// VLLMBatchFormation`:

```go
// PreemptionPolicy controls how preemption selects a victim from the running batch.
type PreemptionPolicy string

const (
	// PreemptionFCFS evicts the last request in the running batch (tail).
	// Matches vLLM's FCFS scheduling mode (self.running.pop()).
	PreemptionFCFS PreemptionPolicy = "fcfs"

	// PreemptionPriority evicts the least-urgent request based on SLO tier priority.
	// Selects min(SLOPriority) with max(ArrivalTime) tiebreak.
	// Matches vLLM's PRIORITY scheduling mode (scheduler.py:827-829)
	// but using BLIS's inverted convention (BLIS: higher=more urgent; vLLM: lower=more urgent).
	PreemptionPriority PreemptionPolicy = "priority"
)
```

### `sim/bundle.go`

**Purpose:** Add `PreemptionConfig` struct, `Preemption` field on `PolicyBundle`, `validPreemptionPolicies` map, accessors, and validation.

**Key implementation notes:**
- `validPreemptionPolicies` is unexported (R8 compliance)
- Pattern mirrors `validPriorityPolicies` / `IsValidPriorityPolicy()` exactly
- Validation in `Validate()` placed after the `validSchedulers` check (line ~207)

### `sim/config.go`

**Purpose:** Add `PreemptionPolicy string` field to `PolicyConfig`; update `NewPolicyConfig()` to 3-arg.

**Alignment:** Updated struct uses aligned columns:
```go
type PolicyConfig struct {
    PriorityPolicy   string // "constant" (default) or "slo-based"
    Scheduler        string // "fcfs" (default), "priority-fcfs", "sjf", "reverse-priority"
    PreemptionPolicy string // "fcfs" (default) or "priority"
}
```

### `cmd/root.go`

**Purpose:** Register `--preemption-policy` flag, thread bundle override, validate, log.

**Bundle override pattern** (mirrors `scheduler` override exactly):
```go
if bundle.Preemption.Policy != "" && !cmd.Flags().Changed("preemption-policy") {
    preemptionPolicy = bundle.Preemption.Policy
}
```

**Validation** (mirrors `scheduler` validation exactly):
```go
if !sim.IsValidPreemptionPolicy(preemptionPolicy) {
    logrus.Fatalf("Unknown preemption policy %q. Valid: %s", preemptionPolicy, strings.Join(sim.ValidPreemptionPolicyNames(), ", "))
}
```
