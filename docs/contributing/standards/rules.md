# BLIS Antipattern Rules

Every rule traces to a real bug, design failure, or hypothesis finding. Rules are enforced at three checkpoints:
- **PR template** — checklist before merge
- **Micro-plan Phase 8** — checklist before implementation
- **Self-audit Step 4.75** — deliberate critical thinking before commit

For the full process, see [docs/contributing/pr-workflow.md](../pr-workflow.md).

## Priority Tiers

New contributors: focus on **Critical** rules first. These protect correctness — violating them produces wrong results or crashes. **Important** rules protect code quality and maintainability. **Hygiene** rules keep the codebase clean over time.

| Tier | Rules | Why |
|------|-------|-----|
| **Critical** (correctness) | R1, R4, R5, R6, R11, R19, R21 | Violations produce silent data loss, panics, conservation invariant breaks, or infinite loops |
| **Important** (quality) | R2, R3, R7, R8, R9, R10, R13, R14, R17, R18, R20, R22, R23 | Violations produce non-determinism, validation gaps, silent misconfig, interface debt, or undetected anomalies |
| **Hygiene** (maintenance) | R12, R15, R16 | Violations produce stale references, config sprawl, or misleading test baselines |

All 23 rules apply to every PR. The tiers help you prioritize during review — check Critical rules first.

## Rules

### R1: No silent data loss

Every error path must either return an error, panic with context, or increment a counter. A `continue` or early `return` that silently drops a request, metric, or allocation is a correctness bug.

**Evidence:** Issue #183 — a KV allocation failure silently dropped a request. The golden test perpetuated the bug for months because it captured "499 completions" as the expected value.

**Additional evidence:** H14 hypothesis experiment — HOL blocking detector silently returns 0 instead of flagging the most extreme imbalance case when `always-busiest` routes all traffic to one instance (bug #291).

**Check:** For every `continue` or early `return` in new code, verify the error is propagated, counted, or documented as safe.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 9.

---

### R2: Sort map keys before float accumulation

Go map iteration is non-deterministic. Any `for k, v := range someMap` that feeds a running sum (`total += v`) or determines output ordering must sort keys first. Unsorted iteration violates the determinism invariant (INV-6).

**Evidence:** Five sites iterated Go maps to accumulate floats or determine output ordering, violating determinism.

**Check:** For every `range` over a map, check if the loop body accumulates floats or produces ordered output. If so, sort keys first.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 3.

---

### R3: Validate ALL numeric parameters

Every numeric flag (`--rate`, `--fitness-weights`, `--kv-cpu-blocks`, etc.) must be validated for: zero, negative, NaN, Inf, and empty string. Missing validation causes infinite loops (Rate=0) or wrong results (NaN weights).

**Evidence:** `--rate 0` caused an infinite loop deep in the simulation. `--snapshot-refresh-interval` was added without validation (#281).

**Additional evidence:** Issues #508, #509, #382, #383, #384 (fix #520) — `NewTieredKVCache` accepted `cpuBlocks <= 0`, `NewKVStore` did not validate `KVOffloadThreshold` range or `KVTransferBandwidth > 0`, and `NewSimulator` accepted `MaxRunningReqs <= 0`. Library callers bypass CLI validation entirely.

**Check:** For every new numeric parameter — whether CLI flag or library constructor argument — add validation. CLI: `logrus.Fatalf` in `cmd/root.go`. Library: `panic` or `error` return in the constructor. Validation must appear before the first consumption site (validate-before-use).

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 6.

---

### R4: Construction site audit

Before adding a field to a struct, find every place that struct is constructed as a literal. If there are multiple sites, either add a canonical constructor or update every site. Missing a site causes silent field-zero bugs.

**Evidence:** Issue #181 — adding `InstanceID` to per-request metrics required changes in 4 files. Three construction sites for `RequestMetrics` existed, and one was missed initially.

**Check:** `grep 'StructName{' across the codebase`. List every site. Update all or refactor to canonical constructor.

**Enforced:** PR template, micro-plan Phase 0 + Phase 8, self-audit dimension 8.

---

### R5: Resource allocation must not leave partial state on failure

Any code that allocates resources (blocks, slots, counters) must ensure no partial mutations remain on failure. Two strategies are valid: (a) **check-then-act** — verify capacity before any mutation, so failure returns before state is touched (preferred; used by KV cache `AllocateKVBlocks`, matching vLLM's `kv_cache_manager.py:334-336`); (b) **rollback** — undo all mutations from prior iterations on mid-loop failure.

**Evidence:** KV block allocation originally used rollback, but `rollbackAllocation` had a bug (#1061) that deleted `RequestMap` for continuing requests, orphaning blocks and causing deadlocks. The check-then-act refactor eliminates this bug class entirely.

**Additional evidence:** H12 hypothesis experiment — preemption loop in `sim/simulator.go:383` accesses `RunningBatch.Requests[len-1]` without bounds check. When all running requests are evicted and the batch is empty, the code panics with index out of range (bug #293).

**Check:** For every loop that mutates state, verify either: (a) a pre-check gate prevents entry when capacity is insufficient, or (b) the failure path rolls back all mutations.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 9.

---

### R6: No logrus.Fatalf in library code

The `sim/` package tree must never terminate the process — return errors so callers can handle them. Only `cmd/` may terminate. This enables embedding, testing, and adapters.

**Evidence:** Library code that called `logrus.Fatalf` prevented test isolation and made the simulator non-embeddable.

**Check:** `grep -r 'logrus.Fatal\|os.Exit' sim/` must return zero results.

**Enforced:** PR template, micro-plan Phase 8.

---

### R7: Invariant tests alongside golden tests

Golden tests (comparing against known-good output) are regression freezes, not correctness checks. If a bug exists when the golden values are captured, the golden test perpetuates the bug. Every subsystem that has golden tests must also have invariant tests that verify conservation laws, causality, and determinism.

**Evidence:** Issue #183 — the codellama golden dataset expected 499 completions because one request was silently dropped. A conservation invariant test would have caught it on day one.

**Check:** For every golden test, ask: "If this expected value were wrong, would any other test catch it?" If no, add an invariant test.

**Enforced:** PR template, micro-plan Phase 6 + Phase 8, self-audit dimension 7.

---

### R8: No exported mutable maps

Validation lookup maps (e.g., `validRoutingPolicies`) must be unexported. Expose through `IsValid*()` accessor functions. Exported maps allow callers to mutate global state, breaking encapsulation and enabling hard-to-trace bugs.

**Evidence:** Exported mutable maps were found during hardening audit — callers could silently add entries to validation maps.

**Check:** `grep -r 'var [A-Z].*map\[' sim/` must return zero mutable map results.

**Enforced:** PR template, micro-plan Phase 8.

---

### R9: Pointer types for YAML zero-value ambiguity

YAML config structs must use `*float64` (pointer) for fields where zero is a valid user-provided value, to distinguish "not set" (nil) from "set to zero" (0.0). Using bare `float64` causes silent misconfiguration when users intentionally set a value to zero.

**Evidence:** YAML fields with bare `float64` couldn't distinguish "user set this to 0" from "user didn't set this."

**Check:** For every new YAML config field where zero is meaningful, use a pointer type.

**Enforced:** Micro-plan Phase 8.

---

### R10: Strict YAML parsing

Use `yaml.KnownFields(true)` or equivalent strict parsing for all YAML config loading. Typos in field names must cause parse errors, not silent acceptance of malformed config.

**Evidence:** YAML typos in field names were silently accepted, producing default behavior instead of the user's intended configuration.

**Check:** Every `yaml.Unmarshal` or decoder usage must enable strict/known-fields mode.

**Enforced:** Micro-plan Phase 8.

---

### R11: Guard division in runtime computation

Any division where the denominator derives from runtime state (batch size, block count, request count, bandwidth) must guard against zero. CLI validation (R3) catches input zeros at the boundary; this rule catches intermediate zeros that arise during simulation.

**Evidence:** `utilization = usedBlocks / totalBlocks` when no blocks are configured; `avgLatency = sum / count` when count is zero.

**Check:** For every division, verify the denominator is either (a) guarded by an explicit zero check, or (b) proven non-zero by a documented invariant.

**Enforced:** Micro-plan Phase 8.

---

### R12: Golden dataset regenerated when output changes

When a PR changes output format, metrics, or default behavior, the golden dataset must be regenerated and the regeneration command documented. Golden tests that pass with stale expected values provide false confidence.

**Evidence:** Present in CONTRIBUTING.md and PR template but not in CLAUDE.md's numbered rules — an inconsistency this consolidation resolves.

**Check:** If `go test ./sim/... -run Golden` fails after your changes, regenerate and document the command.

**Enforced:** PR template, micro-plan Phase 8.

---

### R13: Interfaces accommodate multiple implementations

New interfaces must accommodate at least two implementations (even if only one exists today). No methods that only make sense for one backend.

**Evidence:** `KVStore` interface has methods exposing block-level semantics. A distributed KV cache like LMCache thinks in tokens and layers, not blocks. The interface encodes vLLM's implementation model rather than an abstract behavioral contract.

**Check:** For every new interface, ask: "Could a second backend implement this without dummy methods?"

**Enforced:** Micro-plan Phase 8.

*Previously: principle in CLAUDE.md "Interface design" section. Promoted to numbered rule for checkability.*

---

### R14: No multi-module methods

No method should span multiple module responsibilities (scheduling + latency estimation + metrics in one function). Extract each concern into its module's interface.

**Evidence:** `Simulator.Step()` is 134 lines mixing scheduling, latency estimation, token generation, completion, and metrics. Impossible to swap the latency model without modifying this method.

**Check:** If a method touches >1 module's concern, extract each concern.

**Enforced:** Micro-plan Phase 8.

*Previously: principle in CLAUDE.md "Interface design" section. Promoted to numbered rule for checkability.*

---

### R15: Resolve stale PR references

After completing a PR, grep for references to that PR number (`planned for PR N`, `TODO.*PR N`) in the codebase. Resolve all stale references.

**Evidence:** Multiple stale comments referencing completed PRs accumulated over time, misleading future developers about what was implemented vs planned.

**Check:** `grep -rn 'planned for PR\|TODO.*PR' --include='*.go' --include='*.md'` for the current PR number.

**Enforced:** Micro-plan Phase 8.

---

### R16: Group configuration by module

Configuration parameters must be grouped by module — not added to a monolithic config struct mixing unrelated concerns. Each module's config should be independently specifiable and validatable.

**Evidence:** `SimConfig` previously combined hardware identity, model parameters, simulation parameters, and policy choices in 23 flat fields. Resolved in #350: `SimConfig` now embeds 6 module-scoped sub-configs (`KVCacheConfig`, `BatchConfig`, `LatencyCoeffs`, `ModelHardwareConfig`, `PolicyConfig`, `WorkloadConfig`). Factory signatures accept the narrowest sub-config (e.g., `NewKVStore(KVCacheConfig)`).

**Check:** New config parameters go into the appropriate sub-config in `sim/config.go`, not directly into `SimConfig`.

**Enforced:** Micro-plan Phase 8.

*Previously: principle in CLAUDE.md "Configuration design" section. Promoted to numbered rule for checkability.*

---

### R17: Document signal freshness for routing inputs

Routing snapshot signals have different freshness guarantees due to DES event ordering. Scorer authors must understand which signals are synchronously fresh and which are stale. Any scorer intended for high-rate routing must either use a synchronously-fresh signal or be combined with one that does.

**Evidence:** H3 hypothesis experiment (#279) — kv-utilization scorer produced 200x worse distribution uniformity than queue-depth at rate=5000. See issues #282, #283.

**Freshness hierarchy:**
- **Synchronous (cluster-owned):** InFlightRequests — always fresh (gateway counter)
- **Immediate/Periodic (instance-owned):** QueueDepth, BatchSize, KVUtilization, CacheHitRate — Immediate when `--snapshot-refresh-interval=0`, Periodic when `>0`

**Check:** When writing a new scorer, identify which snapshot fields it reads and their freshness. If using only Periodic signals, document why or combine with a synchronous scorer (InFlightRequests via EffectiveLoad). The `queue-depth` scorer reads only QueueDepth (Periodic) for GIE parity; the `load-balance` scorer reads EffectiveLoad (includes synchronous InFlightRequests).

**Enforced:** Design review, scorer implementation review.

---

### R18: CLI flag precedence over defaults

When the CLI binary loads default values from `defaults.yaml`, it must not silently overwrite user-provided flag values. Always check `cmd.Flags().Changed("<flag>")` before applying a default. A user who explicitly passes `--total-kv-blocks 50` must get 50, not the model's default of 132,139.

**Evidence:** H9 hypothesis experiment — `GetCoefficients()` unconditionally overwrote `totalKVBlocks` with the model default, silently destroying the CLI flag value. The entire H9 Experiment 3 (cache capacity independence) produced invalid results. Bug #285, fix cbb0de7.

**Check:** For every assignment from `defaults.yaml` to a CLI-parsed variable, verify `cmd.Flags().Changed()` is checked first. Grep for `GetCoefficients` and `defaults.yaml` assignment patterns.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 6.

---

### R19: Livelock protection for unbounded retry loops

Loops where the exit condition depends on resource availability that may never be satisfied (e.g., preempt → requeue → schedule → preempt) must have a circuit breaker: maximum iteration count, progress assertion, or bounded retry with error escalation. An infinite loop in a deterministic simulator is indistinguishable from a hang.

**Evidence:** H8 hypothesis experiment — with total KV blocks below ~1000 (insufficient for any single request), the preempt-requeue cycle ran indefinitely with no termination condition, no max-retry limit, and no progress check.

**Additional evidence:** Issue #349, fix #519 — Go `range` over `RunningBatch.Requests` captured the slice header at loop entry. When `preemptForTokens` evicted tail requests mid-iteration, the range loop visited phantom elements, causing 102K+ cascading preemptions with zero completions. This is a distinct livelock mechanism from resource-retry loops — the loop itself is not a retry, but it processes stale elements that trigger resource exhaustion. See also R21.

**Check:** For every loop that retries an operation after a resource failure, verify there is an explicit bound or progress check. Pay special attention to preemption, eviction, and reallocation loops.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 4.

---

### R20: Degenerate input handling in detectors and analyzers

Anomaly detectors and metric analyzers must explicitly handle degenerate inputs: empty sample sets, single-instance concentration, all-zero distributions, and cross-class comparisons. The degenerate case is often the most important one to detect — a detector that returns "no anomaly" when all traffic hits one instance is worse than useless.

**Evidence:** H14 hypothesis experiment — two detector failures: (1) HOL blocking detector requires ≥2 instances with samples, but `always-busiest` routes ALL traffic to one instance, leaving 3 empty — detector returns 0 for the most extreme HOL case (bug #291). (2) Priority inversion detector uses a 2x threshold that conflates workload heterogeneity with scheduling unfairness — 7,463 false positives with normal configs (bug #292).

**Check:** For every detector or analyzer, identify what happens when one or more inputs are empty, zero, or maximally skewed. Write tests for these degenerate cases.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 9.

---

### R21: No `range` over mutable slices

Go's `range` captures the slice header (pointer, length, capacity) at loop entry. If the loop body — or any function it calls — removes elements from the slice (e.g., preemption evicting tail requests), `range` still visits the original indices, accessing stale or evicted elements. Use index-based `for i := 0; i < len(slice); i++` when the slice can shrink during iteration.

**Evidence:** Issue #349, fix #519 — `FormBatch` Phase 1 used `for _, req := range result.RunningBatch.Requests`. When `preemptForTokens` evicted tail requests, the range loop still visited them at original indices. Evicted requests had `ProgressIndex=0`, triggering full re-prefill allocation → cascading preemptions (102K+ preemptions, 0 completions). Fix: index-based loop with per-iteration `len()` re-evaluation.

**Check:** For every `range` over a slice, check if the loop body (or any callee) can modify the slice's length. If so, use index-based iteration. Pay special attention to preemption, eviction, and reallocation callees. Note: read-only `range` loops that don't shrink the iterated slice are safe (e.g., zeroing fields on existing elements).

**Enforced:** Micro-plan Phase 8, self-audit dimension 1.

---

### R22: Pre-check consistency with guarded operation

When a capacity pre-check guards an expensive operation (e.g., "do we have enough free blocks before attempting allocation?"), the pre-check must be at least as permissive as the actual operation. A pre-check that uses a simplified formula (e.g., `ceil(tokens/blockSize)`) while the actual path accounts for additional factors (e.g., partial last-block fill) causes false rejections — the pre-check says "no" when the operation would succeed.

**Evidence:** Issue #492, fix #502 — `AllocateKVBlocks` pre-check computed `ceil(newTokens/blockSize)` without accounting for tokens absorbed into the request's partially-filled last block. Over-estimated by up to 1 block, causing false KV allocation failures and spurious preemptions under tight KV pressure with chunked prefill.

**Check:** For every fast-path pre-check that guards an allocation or resource operation, verify the pre-check's estimate is consistent with (at least as permissive as) the actual operation's accounting. Diff the two formulas explicitly.

**Enforced:** Micro-plan Phase 8, self-audit dimension 1.

---

### R23: Parallel code path transformation parity

When multiple code paths produce the same output type (e.g., standard requests, reasoning single-session, reasoning multi-session), all paths must apply the same set of transformations. A transformation present in one path but missing from another causes silent data corruption — requests with wrong token counts, missing lifecycle filtering, or absent prefix tokens.

**Evidence:** Issue #515 (missing lifecycle filtering), #516 (missing prefix token prepend), fix #530 — `GenerateRequests` had three paths producing `sim.Request` objects: standard, reasoning single-session, reasoning multi-session. The standard path applied three transformations: (1) prefix token prepend, (2) per-round lifecycle filtering, (3) horizon check. Both reasoning paths were missing transformations (1) and (2), causing reasoning requests to have shorter inputs than expected and to leak past lifecycle window boundaries.

**Check:** When adding a new code path that generates the same output type as existing paths, diff the transformation steps. List all transformations applied by the reference path side-by-side with the new path and verify each is present (or explicitly documented as inapplicable). When a function has multiple branches producing the same return type, write a table-driven test where each row exercises one path and asserts the same observable properties.

**Enforced:** Micro-plan Phase 8, self-audit dimension 1.

---

## Quick Reference Checklist

For PR authors — check each rule before submitting:

- [ ] **R1:** No silent `continue`/`return` dropping data
- [ ] **R2:** Map keys sorted before float accumulation or ordered output
- [ ] **R3:** Every new numeric parameter validated (CLI flags AND library constructors)
- [ ] **R4:** All struct construction sites audited for new fields
- [ ] **R5:** Resource allocation uses check-then-act pre-check or rollback on failure
- [ ] **R6:** No `logrus.Fatalf` or `os.Exit` in `sim/` packages
- [ ] **R7:** Invariant tests alongside any golden tests
- [ ] **R8:** No exported mutable maps
- [ ] **R9:** `*float64` for YAML fields where zero is valid
- [ ] **R10:** YAML strict parsing (`KnownFields(true)`)
- [ ] **R11:** Division by runtime-derived denominators guarded
- [ ] **R12:** Golden dataset regenerated if output changed
- [ ] **R13:** New interfaces work for 2+ implementations
- [ ] **R14:** No method spans multiple module responsibilities
- [ ] **R15:** Stale PR references resolved
- [ ] **R16:** Config params grouped by module
- [ ] **R17:** Routing scorer signals documented for freshness tier
- [ ] **R18:** CLI flag values not silently overwritten by defaults.yaml
- [ ] **R19:** Unbounded retry/requeue loops have circuit breakers
- [ ] **R20:** Detectors and analyzers handle degenerate inputs (empty, skewed, zero)
- [ ] **R21:** No `range` over slices that can shrink during iteration
- [ ] **R22:** Pre-check estimates consistent with actual operation accounting
- [ ] **R23:** Parallel code paths apply equivalent transformations

---

## Rule Lifecycle

Rules are born from real bugs and live as long as they prevent real bugs. As the codebase evolves, some rules may become automated, consolidated, or no longer applicable.

### Lifecycle States

| State | Meaning | Action |
|-------|---------|--------|
| **Active** | Rule prevents a class of bugs that can still occur | Check in every PR review |
| **Automated** | Rule is enforced by CI (linter, test, build) | Note the enforcement mechanism; keep for documentation but skip manual checks |
| **Consolidated** | Rule merged into a broader rule | Redirect to the parent rule; remove from checklist |
| **Retired** | The class of bugs is no longer possible (e.g., the vulnerable code path was removed) | Move to a "Retired Rules" appendix with rationale |

### When to Consolidate

If two rules address the same root principle and checking one always catches the other, consolidate them. Example: if a linter rule were added that caught all R2 violations (unsorted map iteration), R2 could move to "Automated" state.

### Quarterly Review

Every ~10 PRs or quarterly (whichever comes first), scan the rule list:
1. Can any rule be automated by a linter or CI check?
2. Are any two rules always checked together and catching the same class of bugs?
3. Has the code path that motivated any rule been removed?

File an issue for each proposed state change. Do not retire rules silently.

### Current State

All 23 rules (R1-R23) are **Active** as of 2026-03-05. No rules have been automated, consolidated, or retired.
