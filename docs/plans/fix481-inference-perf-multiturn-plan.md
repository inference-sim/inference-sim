# fix(workload): inference-perf multi-turn semantic mismatch — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the inference-perf converter so `enable_multi_turn_chat` no longer creates multi-turn sessions with invented parameters, producing independent requests that match real inference-perf training data.

**The problem today:** When `enable_multi_turn_chat: true`, BLIS's `ExpandInferencePerfSpec` creates a `ReasoningSpec` with hardcoded `MaxRounds=5`, `ThinkTimeUs=500000`, and `ContextGrowth="accumulate"`. These values are invented — they have no correspondence to inference-perf's configuration. Real inference-perf training data shows constant input tokens (~574), not accumulated context. Additionally, BLIS's generator spawns multiple sessions per client via the arrival loop, each with `MaxRounds` rounds — architecturally incompatible with inference-perf's model of one long-lived session per user. The result: workloads produce dramatically inflated input tokens and catastrophically wrong TTFT (H30 #480 found 2000x TTFT error across the queueing phase boundary).

**What this PR adds:**
1. **Ignore the flag** — `enable_multi_turn_chat: true` no longer creates a `ReasoningSpec`. Clients produce independent requests with constant tokens, matching real inference-perf data. The flag is silently accepted for YAML compatibility (R6: no logging in library code).
2. **Follow-up issue** — file an issue for proper single-session-per-client multi-turn support in the generator.

**Why this matters:** The inference-perf converter is the bridge between real benchmarking data and BLIS simulation. The hardcoded mapping produces dramatically wrong capacity predictions. This fix ensures the converter produces faithful workloads.

**Architecture:** Changes are confined to `sim/workload/inference_perf.go` (remove hardcoded reasoning block, remove category override), its tests, and docs. No new interfaces, packages, or architectural changes.

**Source:** GitHub issue #481

**Closes:** Fixes #481

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR removes the broken multi-turn mapping from the inference-perf converter. When `enable_multi_turn_chat: true`, the converter previously created a `ReasoningSpec` with invented hardcoded values (5 rounds, 500ms think time, context accumulation). This produced workloads catastrophically different from real inference-perf data (constant input tokens across all requests).

The fix removes the `ReasoningSpec` creation entirely and produces independent requests with constant tokens. The flag is silently accepted for YAML compatibility (R6: library code must not log or terminate). A follow-up issue will be filed for proper single-session multi-turn support that requires generator changes.

**Adjacent components:** `ExpandInferencePerfSpec` → `ClientSpec.Reasoning` field → `GenerateReasoningRequests` (reasoning.go). After this fix, inference-perf clients never enter the reasoning path.

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: Multi-turn flag produces independent requests**
- GIVEN an `InferencePerfSpec` with `enable_multi_turn_chat: true`
- WHEN `ExpandInferencePerfSpec` is called
- THEN every client's `Reasoning` field is nil (no multi-turn session generation)

**BC-2: No multi-turn without flag (unchanged)**
- GIVEN `enable_multi_turn_chat: false` (or omitted)
- WHEN `ExpandInferencePerfSpec` is called
- THEN every client's `Reasoning` field is nil

**BC-3: Category is always "language"**
- GIVEN any `InferencePerfSpec` (regardless of `enable_multi_turn_chat`)
- WHEN `ExpandInferencePerfSpec` is called
- THEN `WorkloadSpec.Category` is `"language"` (not "reasoning")

**BC-4: Multi-stage + multi-turn validation removed**
- GIVEN `enable_multi_turn_chat: true` with 2+ stages
- WHEN `validateInferencePerfSpec` is called
- THEN it succeeds (the flag is now a no-op, so there's no conflict with multi-stage)

**BC-7: Flag is a true no-op end-to-end**
- GIVEN two identical `InferencePerfSpec` differing only in `enable_multi_turn_chat` (true vs false)
- WHEN both are expanded and passed through `GenerateRequests` with the same seed and horizon
- THEN they produce identical request sequences (same seed + same expansion = deterministic)
- MECHANISM: Since `Reasoning` is nil in both cases, both take the same generator path

#### Backward Compatibility

**BC-5: Existing specs without multi-turn unaffected**
- GIVEN any `InferencePerfSpec` with `enable_multi_turn_chat: false`
- WHEN `ExpandInferencePerfSpec` is called
- THEN identical output to current behavior

**BC-6: YAML strict parsing still catches field typos**
- GIVEN a YAML file with `enable_multi_turn_chat` (correct spelling)
- WHEN `LoadWorkloadSpec` is called
- THEN it parses successfully (field still exists in struct, R10 satisfied)

### C) Component Interaction

```
SharedPrefixSpec { EnableMultiTurnChat: true }
  │
  ▼
ExpandInferencePerfSpec()
  │  CHANGED: no longer creates ReasoningSpec
  │  Flag silently accepted for YAML compat (R6: no logging in library)
  ▼
ClientSpec { Reasoning: nil }  ← always nil now for inference-perf clients
  │
  ▼
GenerateRequests() → normal (non-reasoning) arrival path
```

**State changes:** None. Removes state creation (ReasoningSpec) rather than adding it.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue #481 Option B: "Add explicit multi-turn fields" | Option A: ignore the flag | CORRECTION: Plan review discovered BLIS's generator architecture (many-sessions-per-client) is incompatible with inference-perf's model (one-session-per-user). Computing multi-turn params produces 1000x request overgeneration. Real data shows constant input tokens. Ignoring the flag matches reality and avoids the architectural mismatch. |
| Issue #481 Option C: "Add a warning when flag is used" | Silently accept the flag | The flag is a true no-op (produces identical output regardless of value), so a runtime warning would add noise without actionable information. Note: while R6 discourages logging in library code, pre-existing `logrus.Warnf` calls exist in `sim/workload/` (generator.go:38, arrival.go, spec.go). The primary rationale is pragmatic: warning about a no-op confuses users. |
| Issue #481 Option A sub-item: "Add chat_template_overhead tokens (~27)" | Not implemented | DEFERRAL: Chat template overhead is model-specific (~27 for Llama-2, different for others). Adding it would change token counts from user-specified values and complicate the mapping. The ~27 token difference is <5% and negligible for capacity planning. |
| Issue #481: "Multi-turn request generation does not respect lifecycle windows" (validation guard) | Remove the multi-stage + multi-turn guard | SIMPLIFICATION: Since the flag is now a no-op, the guard is unnecessary. Note: the underlying generator bug remains for non-inference-perf reasoning clients (reasoning path at generator.go:116-143 skips `isInActiveWindow` check). Documented in follow-up issue. |

### E) Review Guide

**The tricky part:** Backward compatibility — removing the multi-turn mapping changes the behavior of existing YAML files with `enable_multi_turn_chat: true`. But those files already produced catastrophically wrong results. The doc comment and workload-spec reference document the change.

**What to scrutinize:** BC-1 (multi-turn flag ignored), BC-4 (validation guard removed), BC-7 (end-to-end no-op regression anchor).

**What's safe to skim:** BC-2/BC-5 (unchanged paths), test construction site updates.

**Known debt:** (1) Proper single-session multi-turn support requires generator changes — filed as follow-up issue. (2) The generator's reasoning path (generator.go:116-143) does not check lifecycle windows — pre-existing bug, documented in follow-up issue.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/workload/inference_perf.go` — Remove reasoning block, remove multi-stage guard, remove category override, update doc comment
- `sim/workload/inference_perf_test.go` — Replace multi-turn tests, add e2e regression anchor, remove obsolete tests
- `docs/reference/workload-spec.md` — Update `enable_multi_turn_chat` description
- `examples/inference-perf-shared-prefix.yaml` — Update comment

**Key decisions:**
- `EnableMultiTurnChat` field stays on `SharedPrefixSpec` for YAML compatibility (strict parsing, R10)
- Flag is silently accepted — R6 prohibits logging/termination in `sim/` library code; doc comment and changelog explain the change

### G) Task Breakdown

---

### Task 1: Remove multi-turn mapping and update tests

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-4, BC-5, BC-6, BC-7

**Files:**
- Modify: `sim/workload/inference_perf.go`
- Modify: `sim/workload/inference_perf_test.go`

**Step 1: Write failing test for BC-1 (multi-turn flag produces no ReasoningSpec)**

Context: Replace the existing test that expects a ReasoningSpec with hardcoded values. The new test verifies the flag is a no-op.

Replace `TestExpandInferencePerfSpec_MultiTurn_MapsToReasoning` (line 527) with:

```go
func TestExpandInferencePerfSpec_MultiTurn_ProducesIndependentRequests(t *testing.T) {
	// BC-1: enable_multi_turn_chat is ignored — no ReasoningSpec created.
	// Real inference-perf data shows constant input tokens (~574).
	// BLIS's generator architecture (many-sessions-per-client) is incompatible
	// with inference-perf's model (one-session-per-user). See #481.
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 3,
			SystemPromptLen:         100,
			QuestionLen:             50,
			OutputLen:               25,
			EnableMultiTurnChat:     true,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, c := range ws.Clients {
		if c.Reasoning != nil {
			t.Errorf("client %q: Reasoning should be nil (multi-turn flag is ignored for inference-perf)", c.ID)
		}
	}
	// BC-3: category should be "language" (not "reasoning")
	if ws.Category != "language" {
		t.Errorf("category = %q, want language", ws.Category)
	}
}
```

**Step 2: Write test for BC-4 (multi-stage + multi-turn now allowed) with lifecycle window verification**

```go
func TestExpandInferencePerfSpec_MultiStageMultiTurn_Succeeds(t *testing.T) {
	// BC-4: multi-stage + multi-turn no longer rejected since the flag is a no-op.
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 600},
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
			EnableMultiTurnChat:     true,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ws.Clients) != 2 {
		t.Fatalf("client count = %d, want 2 (1 per stage)", len(ws.Clients))
	}
	// BC-1 also holds for multi-stage path
	for _, c := range ws.Clients {
		if c.Reasoning != nil {
			t.Errorf("client %q: Reasoning should be nil", c.ID)
		}
	}
	// Verify expanded spec passes full validation
	if err := ws.Validate(); err != nil {
		t.Fatalf("expanded spec validation failed: %v", err)
	}
	// Verify lifecycle windows are correctly assigned (stage 0: [0, 600s), stage 1: [600s, 1200s))
	lc0 := ws.Clients[0].Lifecycle
	if lc0 == nil || len(lc0.Windows) != 1 {
		t.Fatal("stage 0 client should have exactly 1 lifecycle window")
	}
	if lc0.Windows[0].StartUs != 0 || lc0.Windows[0].EndUs != 600_000_000 {
		t.Errorf("stage 0 window = [%d, %d), want [0, 600000000)",
			lc0.Windows[0].StartUs, lc0.Windows[0].EndUs)
	}
	lc1 := ws.Clients[1].Lifecycle
	if lc1 == nil || len(lc1.Windows) != 1 {
		t.Fatal("stage 1 client should have exactly 1 lifecycle window")
	}
	if lc1.Windows[0].StartUs != 600_000_000 || lc1.Windows[0].EndUs != 1_200_000_000 {
		t.Errorf("stage 1 window = [%d, %d), want [600000000, 1200000000)",
			lc1.Windows[0].StartUs, lc1.Windows[0].EndUs)
	}
}
```

**Step 2b: Write test for BC-2 (false path explicitly tested)**

```go
func TestExpandInferencePerfSpec_MultiTurnFalse_NoReasoning(t *testing.T) {
	// BC-2: enable_multi_turn_chat=false produces nil Reasoning (explicit false-path test)
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
			EnableMultiTurnChat:     false,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ws.Clients[0].Reasoning != nil {
		t.Error("Reasoning should be nil when enable_multi_turn_chat is false")
	}
	if ws.Category != "language" {
		t.Errorf("category = %q, want language", ws.Category)
	}
}
```

**Step 2c: Write end-to-end regression anchor test for BC-7**

```go
func TestExpandInferencePerfSpec_MultiTurnFlag_IsNoOpEndToEnd(t *testing.T) {
	// BC-7: enable_multi_turn_chat=true produces the same request count as false.
	// This is the quantitative regression anchor proving the flag is truly a no-op.
	base := &SharedPrefixSpec{
		NumUniqueSystemPrompts:  2,
		NumUsersPerSystemPrompt: 2,
		SystemPromptLen:         50,
		QuestionLen:             100,
		OutputLen:               50,
	}
	horizon := int64(10_000_000) // 10 seconds

	// With flag = false
	specFalse := &InferencePerfSpec{
		Stages:       []StageSpec{{Rate: 10.0, Duration: 10}},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  base.NumUniqueSystemPrompts,
			NumUsersPerSystemPrompt: base.NumUsersPerSystemPrompt,
			SystemPromptLen:         base.SystemPromptLen,
			QuestionLen:             base.QuestionLen,
			OutputLen:               base.OutputLen,
			EnableMultiTurnChat:     false,
		},
	}
	wsFalse, err := ExpandInferencePerfSpec(specFalse, 42)
	if err != nil {
		t.Fatalf("expand false: %v", err)
	}
	rFalse, err := GenerateRequests(wsFalse, horizon, 0)
	if err != nil {
		t.Fatalf("generate false: %v", err)
	}

	// With flag = true
	specTrue := &InferencePerfSpec{
		Stages:       []StageSpec{{Rate: 10.0, Duration: 10}},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  base.NumUniqueSystemPrompts,
			NumUsersPerSystemPrompt: base.NumUsersPerSystemPrompt,
			SystemPromptLen:         base.SystemPromptLen,
			QuestionLen:             base.QuestionLen,
			OutputLen:               base.OutputLen,
			EnableMultiTurnChat:     true,
		},
	}
	wsTrue, err := ExpandInferencePerfSpec(specTrue, 42)
	if err != nil {
		t.Fatalf("expand true: %v", err)
	}
	rTrue, err := GenerateRequests(wsTrue, horizon, 0)
	if err != nil {
		t.Fatalf("generate true: %v", err)
	}

	if len(rFalse) != len(rTrue) {
		t.Errorf("request count differs: false=%d, true=%d (flag should be no-op)", len(rFalse), len(rTrue))
	}
}
```

**Step 3: Run tests to verify they fail**

Run: `cd /Users/sri/Documents/Projects/inference-sim/.worktrees/fix481-inference-perf-multiturn && go test ./sim/workload/... -run "TestExpandInferencePerfSpec_MultiTurn_ProducesIndependentRequests|TestExpandInferencePerfSpec_MultiStageMultiTurn_Succeeds|TestExpandInferencePerfSpec_MultiTurnFlag_IsNoOpEndToEnd" -v`
Expected: FAIL — expansion still creates ReasoningSpec with hardcoded values, and multi-stage+multi-turn is still rejected by validation

**Step 4: Implement the fix**

In `sim/workload/inference_perf.go`:

**Note:** Line numbers below refer to the pre-edit state. Use content-based matching (old_string/new_string) for edits, not line numbers, since earlier deletions shift subsequent lines.

**4a. Update doc comment** (lines 8-13):
```go
// InferencePerfSpec defines an inference-perf style workload using a compact
// format. It is expanded into a standard WorkloadSpec via ExpandInferencePerfSpec().
//
// Stage-based rates: sequential rate/duration pairs that produce lifecycle windows.
// Shared prefix: auto-generates N*M clients with prefix groups.
// Note: enable_multi_turn_chat is accepted but ignored — real inference-perf data
// shows constant input tokens (the flag controls chat template formatting, not context
// accumulation). BLIS's generator would also need single-session-per-client support
// to model multi-turn correctly. Token counts may differ from real inference-perf
// by a model-dependent number of chat template tokens (e.g., ~27 for Llama-3).
```

**4b. Remove multi-stage + multi-turn validation guard** (lines 71-77):
Delete this block:
```go
	// Multi-turn request generation (generator.go reasoning path) does not check
	// lifecycle windows, so multi-stage + multi-turn would silently ignore stage
	// boundaries. Reject explicitly until the generator supports this combination.
	if len(spec.Stages) > 1 && sp.EnableMultiTurnChat {
		return fmt.Errorf("inference_perf: multi-stage with enable_multi_turn_chat is not supported; " +
			"multi-turn request generation does not respect lifecycle windows")
	}
```

**4c. Remove reasoning block and category override** (lines 102-121):
Replace:
```go
	// Build optional reasoning spec for multi-turn
	var reasoning *ReasoningSpec
	if sp.EnableMultiTurnChat {
		reasoning = &ReasoningSpec{
			ReasonRatioDist: DistSpec{
				Type:   "constant",
				Params: map[string]float64{"value": 0},
			},
			MultiTurn: &MultiTurnSpec{
				MaxRounds:     5,
				ThinkTimeUs:   500000, // 500ms
				ContextGrowth: "accumulate",
			},
		}
	}

	category := "language"
	if sp.EnableMultiTurnChat {
		category = "reasoning"
	}
```

With:
```go
	category := "language"
```

**4d. Remove `Reasoning: reasoning` from both client construction sites** (lines 146 and 185):
Delete the `Reasoning: reasoning,` line from both the single-stage and multi-stage client construction loops.

**Step 5: Remove old tests**

- **Delete** `TestValidateInferencePerfSpec_MultiStageMultiTurn_ReturnsError` — validation guard removed
- **Delete** `TestValidateInferencePerfSpec_SingleStageMultiTurn_NoError` — all specs valid regardless of flag
- **Delete** `TestExpandInferencePerfSpec_MultiTurn_CategoryIsReasoning` — category is always "language" now
- **Delete** `TestExpandInferencePerfSpec_NoMultiTurn_NoReasoning` — replaced by `TestExpandInferencePerfSpec_MultiTurnFalse_NoReasoning`

**Step 6: Run all workload tests**

Run: `cd /Users/sri/Documents/Projects/inference-sim/.worktrees/fix481-inference-perf-multiturn && go test ./sim/workload/... -v`
Expected: All PASS

**Step 7: Run full project tests**

Run: `cd /Users/sri/Documents/Projects/inference-sim/.worktrees/fix481-inference-perf-multiturn && go test ./...`
Expected: All PASS

**Step 8: Run lint**

Run: `cd /Users/sri/Documents/Projects/inference-sim/.worktrees/fix481-inference-perf-multiturn && golangci-lint run ./...`
Expected: No new issues.

**Step 9: Commit**

```bash
git add sim/workload/inference_perf.go sim/workload/inference_perf_test.go
git commit -m "fix(workload): ignore enable_multi_turn_chat in inference-perf converter (#481)

- Remove hardcoded ReasoningSpec creation (MaxRounds=5, ThinkTimeUs=500000)
  that produced dramatically inflated tokens and wrong TTFT vs real data
- enable_multi_turn_chat is now accepted but ignored: real inference-perf
  data shows constant input tokens, and BLIS's generator architecture
  (many-sessions-per-client) is incompatible with inference-perf's model
  (one-session-per-user)
- Remove multi-stage + multi-turn validation guard (flag is a no-op)
- Category is always 'language' for inference-perf workloads

Fixes #481

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Update documentation

**Contracts Implemented:** documentation for BC-1 through BC-7

**Files:**
- Modify: `docs/reference/workload-spec.md`
- Modify: `examples/inference-perf-shared-prefix.yaml`

**Step 1: Update workload-spec reference**

In `docs/reference/workload-spec.md`, update the `enable_multi_turn_chat` row:

Replace:
```markdown
| `enable_multi_turn_chat` | bool | Enable multi-turn chat mode |
```

With:
```markdown
| `enable_multi_turn_chat` | bool | Accepted but ignored. Real inference-perf data shows constant input tokens; BLIS's generator requires architectural changes for single-session multi-turn. See #481. |
```

**Step 2: Update example YAML comment**

In `examples/inference-perf-shared-prefix.yaml`, update the comment:

Replace:
```yaml
    enable_multi_turn_chat: false
```

With:
```yaml
    enable_multi_turn_chat: false  # When true: accepted but ignored (see #481)
```

Also update the translation comment at line 19:

Replace:
```yaml
#   data.shared_prefix.enable_multi_turn_chat --> reasoning.multi_turn
```

With:
```yaml
#   data.shared_prefix.enable_multi_turn_chat --> ignored (see #481)
```

**Step 3: Run build**

Run: `cd /Users/sri/Documents/Projects/inference-sim/.worktrees/fix481-inference-perf-multiturn && go build ./...`
Expected: Success

**Step 4: Commit**

```bash
git add docs/reference/workload-spec.md examples/inference-perf-shared-prefix.yaml
git commit -m "docs(workload): update enable_multi_turn_chat documentation (#481)

- Mark field as accepted but ignored in workload-spec reference
- Update example YAML comments

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | `TestExpandInferencePerfSpec_MultiTurn_ProducesIndependentRequests` |
| BC-2 | Task 1 | Unit | `TestExpandInferencePerfSpec_MultiTurnFalse_NoReasoning` |
| BC-3 | Task 1 | Unit | `TestExpandInferencePerfSpec_MultiTurn_ProducesIndependentRequests` (checks category) |
| BC-4 | Task 1 | Unit | `TestExpandInferencePerfSpec_MultiStageMultiTurn_Succeeds` |
| BC-5 | Existing | Unit | `TestExpandInferencePerfSpec_SharedPrefix_GeneratesNxMClients` etc. |
| BC-6 | Existing | Unit | `TestLoadWorkloadSpec_InferencePerfSpec_StrictParsing` |
| BC-7 | Task 1 | Integration | `TestExpandInferencePerfSpec_MultiTurnFlag_IsNoOpEndToEnd` |

**Golden dataset:** Not affected — golden dataset uses `--rate` CLI mode, not inference-perf YAML.

**Invariant tests:** Existing `TestInferencePerf_Determinism_SameSeedIdenticalOutput` and `TestInferencePerf_Causality_ArrivalTimesMonotonic` cover INV-5/INV-6 and don't use multi-turn.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Breaking YAML files with `enable_multi_turn_chat: true` | Medium | Low | Those files already produced catastrophically wrong results; now they produce correct (constant token) results | Task 1 |
| Silent behavioral change surprises upgrading users | Medium | Low | Documented in doc comment, workload-spec.md, example YAML, and PR changelog. R6 prevents runtime warnings from library code. Risk explicitly accepted — see deviation log. | Task 1+2 |
| `ReasoningSpec` type appears unused in `inference_perf.go` after removal | N/A | None | `ReasoningSpec` is defined in `spec.go` (same package) and used by `reasoning.go` and `generator.go`. No import involved — same-package type. | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep — this is a targeted removal of broken mapping
- [x] No unexercised flags — `EnableMultiTurnChat` stays for YAML compat but is documented as ignored
- [x] No partial implementations
- [x] Breaking change documented — doc comment, workload-spec.md, example YAML, changelog
- [x] No hidden global state impact
- [x] R1: No silent data loss — field is parsed, just not acted upon. Deviation log explicitly accepts the silent behavioral change with R6 rationale.
- [x] R4: `Reasoning: reasoning,` removed from both client construction sites (lines 146, 185)
- [x] R6: No logrus.Fatalf or logrus.Warn in library code
- [x] R10: `EnableMultiTurnChat` field stays on struct → strict YAML parsing preserved
- [x] CLAUDE.md update: Not needed — no new files, packages, or CLI flags

---

## Appendix: File-Level Implementation Details

### File: `sim/workload/inference_perf.go`

**Purpose:** Inference-perf format conversion.

**Changes (all removals):**

1. **Doc comment** (lines 8-13): Replace "Multi-turn: maps to BLIS reasoning.multi_turn" with "enable_multi_turn_chat is accepted but ignored" + rationale including ~27 token chat template overhead note.

2. **Validation** (lines 71-77): Delete the multi-stage + multi-turn guard block entirely.

3. **Expansion** (lines 102-121): Delete the `reasoning` variable, the `ReasoningSpec` construction, and the category override. Replace with just `category := "language"`.

4. **Client construction** (lines 146, 185): Remove `Reasoning: reasoning,` from both single-stage and multi-stage client construction sites.

### File: `sim/workload/inference_perf_test.go`

**Purpose:** Tests for expansion behavior.

**Changes:**

1. **Replace** `TestExpandInferencePerfSpec_MultiTurn_MapsToReasoning` with `TestExpandInferencePerfSpec_MultiTurn_ProducesIndependentRequests` (BC-1, BC-3)
2. **Replace** `TestValidateInferencePerfSpec_MultiStageMultiTurn_ReturnsError` with `TestExpandInferencePerfSpec_MultiStageMultiTurn_Succeeds` (BC-4, with lifecycle window verification)
3. **Replace** `TestExpandInferencePerfSpec_NoMultiTurn_NoReasoning` with `TestExpandInferencePerfSpec_MultiTurnFalse_NoReasoning` (BC-2, explicit false-path)
4. **Add** `TestExpandInferencePerfSpec_MultiTurnFlag_IsNoOpEndToEnd` (BC-7, end-to-end regression anchor)
5. **Delete** `TestValidateInferencePerfSpec_SingleStageMultiTurn_NoError` (redundant — all specs valid regardless of flag)
6. **Delete** `TestExpandInferencePerfSpec_MultiTurn_CategoryIsReasoning` (category is always "language")

### File: `docs/reference/workload-spec.md`

**Changes:** Update `enable_multi_turn_chat` description to "Accepted but ignored."

### File: `examples/inference-perf-shared-prefix.yaml`

**Changes:** Update comments to note the flag is ignored when true, with issue reference.

### Follow-up Issue

File a GitHub issue for proper multi-turn support:

**Title:** `enhancement: support inference-perf-style single-session multi-turn in generator`

**Body:** Three distinct multi-turn patterns exist in LLM serving:

**(a) Chat template formatting** — inference-perf's `enable_multi_turn_chat` wraps prompts in multi-turn chat template format (`<|user|>...<|assistant|>...`). This is cosmetic for BLIS since it operates on token counts, not templated text. Currently ignored per #481.

**(b) Client-side context accumulation** — each turn's prompt includes all prior turns' input+output. BLIS's `ReasoningSpec` with `ContextGrowth: "accumulate"` models this, but the generator's arrival loop (`GenerateRequests`, reasoning branch) spawns **multiple sessions per client**, each with `MaxRounds` rounds. Inference-perf models **one session per user** where the load generator cycles through sessions. To support this, the generator needs a single-session mode.

**(c) Server-side session caching** (e.g., SGLang RadixAttention) — the server retains KV cache across requests in the same session. BLIS does not model this; it is orthogonal.

This issue tracks support for pattern (b): adding a single-session-per-client mode to the generator where each client creates exactly 1 session with computed `MaxRounds = ceil(rate × duration / sessions)` and `ThinkTimeUs = sessions / rate × 1e6`.

**Additional note:** The reasoning path in `GenerateRequests` (generator.go, `if client.Reasoning != nil && client.Reasoning.MultiTurn != nil`) does not check lifecycle windows via `isInActiveWindow`, unlike the standard path. This is a pre-existing bug that would need fixing before multi-turn + multi-stage can be properly supported.
