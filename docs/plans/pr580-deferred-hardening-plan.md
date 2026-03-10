# PR #580 Deferred Items — Hardening Follow-Up

- **Goal:** Complete the 7 deferred hardening items from issue #580: rope_scaling corrections, unit tests, type alignment, glossary entries, documentation refinements, and two new behavioral tests.
- **The problem today:** The `mrope` rope_scaling type has no test coverage documenting that it correctly falls through the blacklist (vLLM normalizes it to "default" and applies the factor); rope_scaling logic is embedded in `cmd/root.go` with no unit tests; `MaxModelLen` uses `int` while adjacent fields use `int64`; glossary lacks entries for MaxModelLen and oracle knowledge boundary; rope_scaling docs are vague about which types are excluded; cluster-mode and chunked-prefill interaction tests are missing.
- **What this PR adds:**
  1. Correct `mrope` handling in rope_scaling (normalize to "default", apply factor — matching vLLM behavior).
  2. Extract rope_scaling logic to a pure function with table-driven unit tests covering 10+ cases.
  3. Change `MaxModelLen` from `int` to `int64` for consistency with `ProgressIndex`, `TotalKVBlocks`, `BlockSizeTokens`.
  4. Add glossary entries for `MaxModelLen` and `Oracle Knowledge Boundary`.
  5. Refine rope_scaling documentation with explicit blacklist details.
  6. Add cluster-mode test verifying DroppedUnservable surfaces when MaxModelLen drops requests.
  7. Add chunked prefill + MaxModelLen interaction test (no spurious force-completions during multi-chunk prefill).
- **Why this matters:** Completes the hardening wave from #580, closing validation gaps and improving test coverage for critical control-plane paths.
- **Architecture:** Changes span `cmd/root.go` (rope_scaling extraction + int→int64), `sim/config.go` + `sim/simulator.go` (int→int64), `sim/internal/testutil/golden.go` + `sim/workload/tracev2.go` (int→int64), `docs/` (glossary + reference), `sim/simulator_test.go` + `sim/cluster/cluster_test.go` (new tests). 11 files total, no new packages or interfaces.
- **Source:** Issue #580 deferred items (handoff summary from PR #587).
- **Closes:** Related to #580 (deferred items from PR #587). No auto-close — #580 is already closed.
- **Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** rope_scaling logic in `cmd/root.go` (CLI layer); `MaxModelLen` field in `sim/config.go` → `sim/simulator.go`; metrics pipeline through cluster.
2. **Adjacent blocks:** HuggingFace config resolution (`cmd/hfconfig.go`), latency model factory (`sim/latency/`), cluster simulation (`sim/cluster/`), batch formation (`sim/batch_formation.go`).
3. **Invariants touched:** INV-1 (conservation in cluster test), INV-9 (oracle knowledge boundary glossary).
4. **Construction Site Audit:**
   - `ModelHardwareConfig.MaxModelLen` (int → int64): canonical constructor `NewModelHardwareConfig` in `sim/config.go:88-102`. All production callers use the constructor (R4 enforced by #393).
   - `Simulator.maxModelLen` (private, int → int64): assigned in `NewSimulator` at `sim/simulator.go:140`. Lines 277, 287: add `int64()` widening casts on the `int` side (`len()`, `MaxOutputLen`). Line 501: REMOVE existing `int64(sim.maxModelLen)` cast (now redundant). Lines 112-113: REMOVE `int64(cfg.MaxModelLen)` casts.
   - `cmd/root.go:54` — `maxModelLen int` variable → `int64`. Line 973: `IntVar` → `Int64Var` (Cobra flag binding).
   - `cmd/root.go:489` — `kvFeasibleMax := int(...)` → `int64(...)` (KV cap computation).
   - `sim/internal/testutil/golden.go:41` — `MaxModelLen int` in `GoldenTestCase` → `int64` (JSON deserialization, passed to constructor).
   - `sim/workload/tracev2.go:36` — `MaxModelLen int` in `TraceServerConfig` → `int64` (YAML deserialization).
   - Test call sites (e.g., `sim/cluster/instance_test.go:53,107`) pass `tc.MaxModelLen` from `GoldenTestCase` → safe after golden.go changes. All other test sites pass untyped integer literals (e.g., `0`, `100`, `4096`) which are compatible with both `int` and `int64` in Go.

---

## Part 1: Design Validation

### A) Executive Summary

This PR closes 7 deferred hardening items from #580. The changes are a mix of: (1) a behavioral fix (`mrope` rope type handling), (2) a refactoring extraction (rope_scaling → pure function + tests), (3) a type alignment (`int` → `int64`), (4) documentation additions (glossary, reference docs), and (5) two new behavioral tests (cluster-mode MaxModelLen drops, chunked prefill interaction). No new interfaces or modules. The type change is the highest-risk item due to construction site breadth.

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: mrope Normalization
- GIVEN a HuggingFace config with rope_scaling.type == "mrope" and factor > 1.0
- WHEN auto-deriving max-model-len
- THEN the scaling factor is applied (same as "default" type)
- MECHANISM: applyRopeScaling() normalizes "mrope" → applies factor, matching vLLM behavior.

BC-2: Rope Scaling Blacklist
- GIVEN a HuggingFace config with rope_scaling.type in {"su", "longrope", "llama3"}
- WHEN auto-deriving max-model-len
- THEN the scaling factor is NOT applied (max_position_embeddings used as-is)

BC-3: Gemma3 Model Type Exclusion
- GIVEN a HuggingFace config with model_type == "gemma3"
- WHEN auto-deriving max-model-len
- THEN rope_scaling is skipped entirely (max_position_embeddings is pre-scaled)

BC-4: Yarn Original Max Position
- GIVEN a HuggingFace config with rope_scaling.type == "yarn" and original_max_position_embeddings present
- WHEN auto-deriving max-model-len
- THEN original_max_position_embeddings is used as base (not max_position_embeddings)

BC-5: MaxModelLen int64 Type Consistency
- GIVEN any code path using MaxModelLen
- WHEN comparing or computing with ProgressIndex, TotalKVBlocks, or BlockSizeTokens
- THEN MaxModelLen is int64, eliminating int64(cfg.MaxModelLen) casts in NewSimulator and processCompletions
- NOTE: len() and MaxOutputLen remain int, so EnqueueRequest comparisons still need int64() widening casts on the int side (not on MaxModelLen)
- MECHANISM: MaxModelLen field changed from int to int64 in ModelHardwareConfig, Simulator, and 3 auxiliary structs.

BC-6: Cluster DroppedUnservable Surfaces
- GIVEN a cluster with MaxModelLen configured and requests that exceed it
- WHEN simulation completes
- THEN aggregated DroppedUnservable count matches the number of oversized requests
- AND INV-1 conservation holds across all instances

BC-7: Chunked Prefill + MaxModelLen No Spurious Cap
- GIVEN LongPrefillTokenThreshold > 0 and MaxModelLen configured
- WHEN a request with input tokens < MaxModelLen undergoes multi-chunk prefill
- THEN the request completes normally (no force-completion triggered)
- AND LengthCappedRequests == 0

BC-8: Rope Scaling Warning on Invalid Input
- GIVEN rope_scaling with a non-object value or non-float factor
- WHEN auto-deriving max-model-len
- THEN a warning is logged and the value is ignored (no crash)
```

**Error handling contracts:**

```
BC-9: Rope Scaling Pure Function Returns
- GIVEN any combination of inputs to applyRopeScaling
- WHEN the function is called
- THEN it returns (scaledMaxPosEmb int, applied bool) without side effects
- AND the function never panics
```

### C) Component Interaction

```
cmd/root.go                       sim/config.go                    sim/simulator.go
┌─────────────────┐              ┌──────────────────┐             ┌──────────────────┐
│ applyRopeScaling│─(extracted)──│ModelHardwareConfig│────────────│ maxModelLen int64 │
│ (pure function) │              │ MaxModelLen int64 │             │ EnqueueRequest()  │
└────────┬────────┘              └──────────────────┘             │ processCompletion │
         │                              │                          └──────────────────┘
         │                              │ via NewModelHardwareConfig()        │
         │                              │                                     │
    ┌────▼────┐                  ┌──────▼──────┐                  ┌──────────▼──────┐
    │cmd tests│                  │cluster tests│                  │ sim tests       │
    │(unit)   │                  │(integration)│                  │ (behavioral)    │
    └─────────┘                  └─────────────┘                  └─────────────────┘
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Handoff says "Add mrope to rope_scaling handling" | mrope already falls through correctly (not in blacklist); we add explicit test case + comment documenting intentional non-exclusion | CORRECTION — no behavioral change needed, just test coverage |
| Handoff says "Requires extracting rope_scaling logic from cmd/root.go into a pure function" | Extract `applyRopeScaling(maxPosEmb int, modelType string, ropeScaling any) (int, bool)` | ADDITION — returning `bool` allows callers to log context |
| Handoff lists int64 change as "Medium scope" | We do it early (Task 2) since later tasks depend on it. Construction sites: config.go, simulator.go, cmd/root.go (IntVar→Int64Var), golden.go, tracev2.go | REORDER + SCOPE_CHANGE |
| N/A | `applyRopeScaling` keeps `int` parameter/return despite MaxModelLen becoming `int64` — the function operates on `max_position_embeddings` (always fits in `int`); the widening from `int` to `int64` happens at the assignment `maxModelLen = int64(scaled)` in the caller | ADDITION — intentional type boundary |
| N/A | Add CLAUDE.md update step to Task 2 noting MaxModelLen is now int64 | ADDITION |

### E) Review Guide

**Tricky:** The int→int64 change touches `sim/config.go`, `sim/simulator.go`, `cmd/root.go`, and every test that constructs `NewModelHardwareConfig` with a `MaxModelLen` argument (but the canonical constructor handles the type so callers just pass the same literal). The `maxModelLen` private field on `Simulator` also changes.

**Scrutinize:** BC-1 (mrope) — verify vLLM truly normalizes mrope to default and applies factor. BC-7 (chunked prefill) — verify the test actually exercises multi-chunk path.

**Safe to skim:** Glossary entries, docs changes.

**Known debt:** BLIS lacks a proactive `FormBatch`-level MaxModelLen token cap (vLLM uses `min(numNewTokens, maxModelLen - 1 - numComputedTokens)`). This causes a 1-token overshoot vs vLLM (BLIS produces `MaxModelLen - inputLen` output tokens, vLLM produces `MaxModelLen - 1 - inputLen`). Not blocking for this PR since the runtime cap in processCompletions catches it. **Follow-up issue to file during this PR:** Add `numNewTokens = min(numNewTokens, maxModelLen - 1 - req.ProgressIndex)` in `sim/batch_formation.go` (both Phase 1 running-request path ~line 88 and Phase 2 new-request path ~line 122) when `maxModelLen > 0`. Title: "Proactive FormBatch MaxModelLen token cap to match vLLM scheduler semantics".

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files modified:
- `cmd/root.go` — extract `applyRopeScaling()`, add mrope comment, `IntVar` → `Int64Var`, int → int64
- `cmd/root_test.go` — add table-driven rope_scaling unit tests (file already exists)
- `sim/config.go` — `MaxModelLen int` → `int64`, update constructor parameter type
- `sim/simulator.go` — `maxModelLen int` → `int64`, update all comparison sites, remove int64 casts
- `sim/internal/testutil/golden.go` — `MaxModelLen int` → `int64` in GoldenTestCase
- `sim/workload/tracev2.go` — `MaxModelLen int` → `int64` in TraceServerConfig
- `sim/simulator_test.go` — add chunked prefill + MaxModelLen test (BC-7)
- `sim/cluster/cluster_test.go` — add cluster MaxModelLen drop test (BC-6)
- `docs/concepts/glossary.md` — add MaxModelLen and Oracle Knowledge Boundary entries
- `docs/guide/latency-models.md` — refine rope_scaling docs
- `docs/reference/configuration.md` — refine rope_scaling docs

No dead code. No new packages.

### G) Task Breakdown

#### Task 1: Extract applyRopeScaling and add mrope handling (BC-1, BC-2, BC-3, BC-4, BC-8, BC-9)

**Step 1: Write test** — Add table-driven tests for `applyRopeScaling` to existing `cmd/root_test.go`.

**Step 2: Run test** — `go test ./cmd/... -run TestApplyRopeScaling` → fails (function doesn't exist).

**Step 3: Implement** — Extract rope_scaling logic from `cmd/root.go` into `applyRopeScaling(maxPosEmb int, modelType string, ropeScaling any) (scaled int, applied bool)`. No behavioral change needed for `mrope`: it already falls through correctly because it's not in the blacklist `{su, longrope, llama3}`. Add a comment documenting that mrope is intentionally not excluded (vLLM normalizes mrope → "default" and applies the factor). Add overflow guards for `int(float64(base) * factor)` and `int(orig)` conversions. Add NaN/Inf check on factor. Then replace the inline logic in `cmd/root.go` with a call to `applyRopeScaling`, keeping the log messages in the caller (the pure function returns `applied` bool for the caller to decide what to log). Note: `applyRopeScaling` intentionally keeps `int` parameters/return — `max_position_embeddings` always fits in `int`. After Task 2, the caller assignment becomes `maxModelLen = int64(scaled)` (widening, always safe).

**Step 4: Run test** — `go test ./cmd/... -run TestApplyRopeScaling` → passes.

**Step 5: Lint** — `golangci-lint run ./cmd/...`

**Step 6: Commit** — `feat(cmd): extract applyRopeScaling with mrope support (BC-1..BC-4, BC-8, BC-9)`

---

#### Task 2: MaxModelLen int → int64 (BC-5)

**Step 1: Write test** — No new test needed; existing tests validate behavior. The type change is mechanical.

**Step 2: Change `sim/config.go`** — `MaxModelLen int` → `MaxModelLen int64` in `ModelHardwareConfig`. Update `NewModelHardwareConfig` parameter from `maxModelLen int` to `maxModelLen int64`.

**Step 3: Change `sim/simulator.go`** — `maxModelLen int` → `maxModelLen int64` on `Simulator` struct. Specific comparison sites:
- Line 104: `cfg.MaxModelLen < 0` — works (both int64 now)
- Line 112-113: `int64(cfg.MaxModelLen)` casts → remove (already int64)
- Line 118: `cfg.MaxModelLen` in fmt → works
- Line 140: `maxModelLen: cfg.MaxModelLen` — works (both int64)
- Line 277: `len(r.InputTokens) >= sim.maxModelLen` → `int64(len(r.InputTokens)) >= sim.maxModelLen`
- Line 286-287: `totalSeqLen := len(r.InputTokens) + r.MaxOutputLen; totalSeqLen > sim.maxModelLen` → `totalSeqLen := int64(len(r.InputTokens)) + int64(r.MaxOutputLen); totalSeqLen > sim.maxModelLen` (compute in int64 to prevent intermediate int overflow on 32-bit)
- Line 501: `req.ProgressIndex >= int64(sim.maxModelLen)` → `req.ProgressIndex >= sim.maxModelLen` (remove cast)

**Step 4: Change `cmd/root.go`** — Six changes:
- Line 54: `maxModelLen int` → `maxModelLen int64`
- Line 427: `maxModelLen = maxPosEmb` → `maxModelLen = int64(maxPosEmb)` (MustGetInt returns int)
- After Task 1's extraction: `maxModelLen = int64(scaled)` (applyRopeScaling returns int, caller widens)
- Lines 484-485: remove redundant `int64(maxModelLen)` casts (already int64)
- Line 489: `kvFeasibleMax := int(totalKVBlocks * blockSizeTokens)` → `kvFeasibleMax := totalKVBlocks * blockSizeTokens` (both operands already int64, no cast needed). Also update or remove the stale safety comment "fits in int" — the result is now int64.
- Line 973: `IntVar(&maxModelLen, ...)` → `Int64Var(&maxModelLen, ...)`

**Step 5: Change `sim/internal/testutil/golden.go`** — Line 41: `MaxModelLen int` → `MaxModelLen int64` in `GoldenTestCase`.

**Step 6: Change `sim/workload/tracev2.go`** — Line 36: `MaxModelLen int` → `MaxModelLen int64` in `TraceServerConfig`.

**Step 7: Build & test** — `go build ./... && go test ./...` → all pass.

**Step 8: Update CLAUDE.md** — In the `ModelHardwareConfig` description under File Organization, note `MaxModelLen int64`. Update the `sim/config.go` description to mention the type change.

**Step 9: Lint** — `golangci-lint run ./...`

**Step 10: Commit** — `refactor(sim): MaxModelLen int → int64 for type consistency (BC-5)`

---

#### Task 3: Cluster-mode MaxModelLen drop test (BC-6)

**Step 1: Write test** in `sim/cluster/cluster_test.go`:
- Configure DeploymentConfig with MaxModelLen set, 2 instances, alpha=[0,0,0] (zero alpha for deterministic event timing).
- Inject mix of requests: some with input < MaxModelLen (fit), some with input >= MaxModelLen (dropped via Guard 1a), and at least one with input < MaxModelLen but input + MaxOutputLen > MaxModelLen (dropped via Guard 1b).
- Run simulation.
- Assert aggregated DroppedUnservable matches count of oversized requests.
- Assert INV-1 conservation: `completed + still_queued + still_running + dropped == injected`.
- Assert post-simulation inFlightRequests drains: `inFlightRequests[instID] == 0` for each instance (direct assertion, not just the proxy StillQueued + StillRunning == 0).
- Assert aggregated Metrics.Requests map excludes dropped request IDs (map-level cleanup, not just counter arithmetic).
- Note: This test differs from existing `TestClusterSimulator_InFlightRequests_DroppedUnservable_Decrements` (which tests KV-capacity Guard 2) by exercising the MaxModelLen Guard 1 path with a mixed fit/oversized workload.

**Step 2: Run test** — `go test ./sim/cluster/... -run TestClusterSimulator_MaxModelLen_DroppedUnservable` → passes (existing code already works).

**Step 3: Lint** — `golangci-lint run ./sim/cluster/...`

**Step 4: Commit** — `test(cluster): MaxModelLen drop test with conservation (BC-6)`

---

#### Task 4: Chunked prefill + MaxModelLen interaction test (BC-7)

**Step 1: Write test** in `sim/simulator_test.go`:
- Configure with LongPrefillTokenThreshold > 0 (e.g., 64) and MaxModelLen (e.g., 500).
- Inject request with input=200, output=50 (total 250 < MaxModelLen).
- Input is 200 tokens → with threshold=64, prefill splits into multiple chunks.
- Run simulation.
- Assert CompletedRequests == 1, LengthCappedRequests == 0.
- Assert TTFT recorded (req.FirstTokenTime > 0) — verifies TTFT fires on final prefill chunk, not intermediate chunks.
- Assert TotalOutputTokens == 49 (decode runs PI 201→249 = 49 tokens; normal completion fires at PI == 200 + max(50,1) - 1 = 249).

**Step 2: Run test** — `go test ./sim/... -run TestSimulator_ChunkedPrefill_MaxModelLen_NoSpuriousCap` → passes.

**Step 3: Lint** — `golangci-lint run ./sim/...`

**Step 4: Commit** — `test(sim): chunked prefill + MaxModelLen interaction (BC-7)`

---

#### Task 5: Glossary entries (BC-5 docs)

**Step 1: Add entries** to `docs/concepts/glossary.md`:
- **MaxModelLen**: Maximum total sequence length (input + output) for a single request, in tokens. Mirrors vLLM's `--max-model-len`. When set, requests whose input alone fills the context window (input >= MaxModelLen) or whose input + output budget exceeds it are dropped before entering the wait queue. A runtime cap force-completes any request whose progress reaches this limit. Set to 0 for unlimited. Auto-derived from `max_position_embeddings` in roofline/crossmodel modes.
- **Oracle Knowledge Boundary (INV-9)**: The principle that control-plane decisions (admission, routing, scheduling, priority) must not read `Request.OutputTokens`. Output token count is oracle knowledge — known only after generation. The control plane uses `MaxOutputLen` (client-declared budget) or input-only checks. Only the execution engine (batch step, completion detection) may access `OutputTokens`.

**Step 2: Commit** — `docs(glossary): add MaxModelLen and Oracle Knowledge Boundary entries`

---

#### Task 6: Rope scaling documentation refinement

**Step 1: Update** `docs/guide/latency-models.md` — expand the rope_scaling paragraph to explicitly list excluded types and the gemma3 model_type exclusion.

**Step 2: Update** `docs/reference/configuration.md` — refine the `--max-model-len` row to mention the blacklist.

**Step 3: Commit** — `docs(latency): refine rope_scaling exclusion documentation`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestApplyRopeScaling (mrope case) |
| BC-2 | Task 1 | Unit | TestApplyRopeScaling (su, longrope, llama3 cases) |
| BC-3 | Task 1 | Unit | TestApplyRopeScaling (gemma3 case) |
| BC-4 | Task 1 | Unit | TestApplyRopeScaling (yarn + original case) |
| BC-5 | Task 2 | Existing | All existing MaxModelLen tests pass unchanged |
| BC-6 | Task 3 | Integration | TestClusterSimulator_MaxModelLen_DroppedUnservable |
| BC-7 | Task 4 | Behavioral | TestSimulator_ChunkedPrefill_MaxModelLen_NoSpuriousCap |
| BC-8 | Task 1 | Unit | TestApplyRopeScaling (invalid factor, non-object cases) |
| BC-9 | Task 1 | Unit | TestApplyRopeScaling (all cases return without panic) |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| int→int64 breaks construction sites | Low | Medium | R4 audit: canonical constructor + `go build ./...` | Task 2 |
| Rope scaling extraction changes behavior | Low | Medium | Table-driven tests cover all existing code paths | Task 1 |
| Chunked prefill test doesn't actually exercise multi-chunk | Medium | Low | Verify LongPrefillTokenThreshold < input tokens, check step count | Task 4 |
| Golden dataset affected by int64 change | Low | High | Golden dataset uses MaxModelLen=4096. int→int64 type change does not alter JSON deserialization or runtime behavior — golden tests pass unchanged | Task 2 |
| applyRopeScaling float-to-int overflow | Low | High | Overflow guard checks product > math.MaxInt before int() cast; test case 19 validates | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions (applyRopeScaling is the minimal extraction).
- [x] No feature creep beyond PR scope.
- [x] No unexercised flags or interfaces.
- [x] No partial implementations.
- [x] No breaking changes without explicit contract updates.
- [x] No hidden global state impact.
- [x] All new code will pass golangci-lint.
- [x] CLAUDE.md updated: MaxModelLen type note.
- [x] No stale references left in CLAUDE.md.
- [x] Documentation DRY: glossary is the canonical source for these terms.
- [x] Deviation log reviewed — no unresolved deviations.
- [x] Each task produces working, testable code.
- [x] Task dependencies correctly ordered (Task 2 before Tasks 3-4 since they use int64).
- [x] All contracts mapped to specific tasks.
- [x] Golden dataset NOT regenerated (golden uses MaxModelLen=4096; int→int64 doesn't change JSON deser or runtime values).
- [x] Construction site audit completed.

**Antipattern rules:**
- [x] R1: applyRopeScaling returns bool for callers to log, no silent drops.
- [x] R3: MaxModelLen validated in both NewModelHardwareConfig (panic) and NewSimulator (error).
- [x] R4: Construction site audit for MaxModelLen type change — canonical constructor updated.
- [x] R6: No logrus.Fatalf in sim/ (new tests only).
- [x] R7: Cluster test has INV-1 companion assertion.
- [x] R11: Division by zero not introduced (existing guard on BlockSizeTokens).
- [x] R23: applyRopeScaling handles all code paths equivalently.
- [x] All other rules: N/A or already satisfied.

---

## Appendix: File-Level Implementation Details

### File: `cmd/root.go`

**Purpose:** Extract rope_scaling logic to testable pure function, add mrope handling.

**Function signature:**
```go
// applyRopeScaling applies rope_scaling factor to maxPosEmb if applicable.
// Returns the (possibly scaled) value and whether scaling was applied.
// modelType is the HuggingFace model_type string (empty if not present).
// ropeScaling is the raw rope_scaling value from config.json (nil if not present).
func applyRopeScaling(maxPosEmb int, modelType string, ropeScaling any) (scaled int, applied bool) {
```

**Key logic:** Same as existing inline code, with:
- `mrope` documented as intentionally not excluded (vLLM normalizes mrope → "default" and applies factor)
- Overflow guard: after `float64(base) * factor`, check product > `math.MaxInt` before `int()` cast (prevents silent sign-flip on absurd configs). Same guard for `int(orig)` on `original_max_position_embeddings`.
- Factor=nil (JSON null): improve warning to include `%T` of actual value for debugging
- NaN/Inf guard: check `math.IsNaN(factor) || math.IsInf(factor, 0)` before `> 1.0` comparison (defense-in-depth for non-standard JSON sources)
- Warnings returned via the `applied` bool + caller logs. Caller should log distinct messages for: (a) scaling applied, (b) excluded type, (c) no factor/invalid factor, (d) overflow detected. The `applied` bool alone can't distinguish (b)-(d), but the caller can check `ropeScaling != nil && ropeMap exists && factor > 1.0 && !applied` to detect overflow specifically.
- maxPosEmb <= 0 guard: return (maxPosEmb, false) immediately (R3: validate numeric parameters)
- gemma3 model_type check first (early return)

### File: `cmd/root_test.go`

**Purpose:** Table-driven tests for applyRopeScaling.

**Test cases:**
1. No rope_scaling (nil) → maxPosEmb unchanged, applied=false
2. Linear + factor 4.0 → maxPosEmb * 4, applied=true
3. Yarn + original_max_position_embeddings → original * factor, applied=true
4. Yarn without original → maxPosEmb * factor, applied=true
5. Dynamic + factor 2.0 → maxPosEmb * 2, applied=true
6. su (excluded) → unchanged, applied=false
7. longrope (excluded) → unchanged, applied=false
8. llama3 (excluded) → unchanged, applied=false
9. gemma3 model_type → unchanged, applied=false (early return)
10. mrope + factor 8.0 → maxPosEmb * 8, applied=true (BC-1)
11. default + factor 2.0 → maxPosEmb * 2, applied=true
12. Non-object rope_scaling (string) → unchanged, applied=false
13. Factor not float64 → unchanged, applied=false
14. Factor <= 1.0 → unchanged, applied=false
15. Empty type string + factor > 1.0 → factor IS applied, applied=true (empty string not in blacklist; treated as "default")
16. No factor key at all → unchanged, applied=false
17. JSON array rope_scaling ([]any) → unchanged, applied=false (not a map)
18. Null type key ({"type": null, "factor": 8.0}) → factor applied, applied=true (null type → empty string → not excluded)
19. Overflow: maxPosEmb=math.MaxInt/2, factor=4.0 → capped at maxPosEmb, applied=false (overflow guard fires)
20. original_max_position_embeddings overflow: yarn + orig=1e18 + factor=2 → capped, applied=false
21. maxPosEmb=0, ropeScaling with factor > 1.0 → unchanged, applied=false (degenerate base guard)
22. maxPosEmb=-1, ropeScaling with factor > 1.0 → unchanged, applied=false (negative base guard)

### File: `sim/config.go`

**Purpose:** Change `MaxModelLen int` to `MaxModelLen int64`.

**Changes:**
- `ModelHardwareConfig.MaxModelLen` field: `int` → `int64`
- `NewModelHardwareConfig` parameter: `maxModelLen int` → `maxModelLen int64`

### File: `sim/simulator.go`

**Purpose:** Change private `maxModelLen int` to `int64`, remove casts.

**Changes:**
- `Simulator.maxModelLen` field: `int` → `int64`
- `NewSimulator`: remove `int64(cfg.MaxModelLen)` casts (lines 112-113)
- `EnqueueRequest`: `len(r.InputTokens) >= sim.maxModelLen` → `int64(len(r.InputTokens)) >= sim.maxModelLen` (line 277); `totalSeqLen` computed as `int64(len(r.InputTokens)) + int64(r.MaxOutputLen)`, then `totalSeqLen > sim.maxModelLen` (line 286-287)
- `processCompletions`: `req.ProgressIndex >= int64(sim.maxModelLen)` → `req.ProgressIndex >= sim.maxModelLen` (line 501)

### File: `sim/internal/testutil/golden.go`

**Purpose:** Align GoldenTestCase.MaxModelLen with int64.

**Changes:** Line 41: `MaxModelLen int` → `MaxModelLen int64`

### File: `sim/workload/tracev2.go`

**Purpose:** Align TraceServerConfig.MaxModelLen with int64.

**Changes:** Line 36: `MaxModelLen int` → `MaxModelLen int64`

### File: `sim/simulator_test.go`

**Purpose:** Add chunked prefill + MaxModelLen interaction test.

### File: `sim/cluster/cluster_test.go`

**Purpose:** Add cluster-mode MaxModelLen drop test with conservation check.

### File: `docs/concepts/glossary.md`

**Purpose:** Add MaxModelLen and Oracle Knowledge Boundary glossary entries.

### File: `docs/guide/latency-models.md`

**Purpose:** Expand rope_scaling paragraph with explicit blacklist details.

### File: `docs/reference/configuration.md`

**Purpose:** Refine --max-model-len description with blacklist.
