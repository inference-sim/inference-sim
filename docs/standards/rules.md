# BLIS Antipattern Rules

Every rule traces to a real bug, design failure, or hypothesis finding. Rules are enforced at three checkpoints:
- **PR template** — checklist before merge
- **Micro-plan Phase 8** — checklist before implementation
- **Self-audit Step 4.75** — deliberate critical thinking before commit

For the full process, see [docs/process/pr-workflow.md](../process/pr-workflow.md).

## Priority Tiers

New contributors: focus on **Critical** rules first. These protect correctness — violating them produces wrong results or crashes. **Important** rules protect code quality and maintainability. **Hygiene** rules keep the codebase clean over time.

| Tier | Rules | Why |
|------|-------|-----|
| **Critical** (correctness) | R1, R4, R5, R6, R11, R19 | Violations produce silent data loss, panics, conservation invariant breaks, or infinite loops |
| **Important** (quality) | R2, R3, R7, R8, R9, R10, R13, R14, R17, R18, R20 | Violations produce non-determinism, validation gaps, silent misconfig, interface debt, or undetected anomalies |
| **Hygiene** (maintenance) | R12, R15, R16 | Violations produce stale references, config sprawl, or misleading test baselines |

All 20 rules apply to every PR. The tiers help you prioritize during review — check Critical rules first.

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

### R3: Validate ALL numeric CLI flags

Every numeric flag (`--rate`, `--fitness-weights`, `--kv-cpu-blocks`, etc.) must be validated for: zero, negative, NaN, Inf, and empty string. Missing validation causes infinite loops (Rate=0) or wrong results (NaN weights).

**Evidence:** `--rate 0` caused an infinite loop deep in the simulation. `--snapshot-refresh-interval` was added without validation (#281).

**Check:** For every new CLI flag, add validation in `cmd/root.go` with `logrus.Fatalf` for invalid values.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 6.

---

### R4: Construction site audit

Before adding a field to a struct, find every place that struct is constructed as a literal. If there are multiple sites, either add a canonical constructor or update every site. Missing a site causes silent field-zero bugs.

**Evidence:** Issue #181 — adding `InstanceID` to per-request metrics required changes in 4 files. Three construction sites for `RequestMetrics` existed, and one was missed initially.

**Check:** `grep 'StructName{' across the codebase`. List every site. Update all or refactor to canonical constructor.

**Enforced:** PR template, micro-plan Phase 0 + Phase 8, self-audit dimension 8.

---

### R5: Transactional state mutation

Any loop that allocates resources (blocks, slots, counters) must handle mid-loop failure by rolling back all mutations from previous iterations. A partial allocation that returns `false` without cleanup violates conservation invariants.

**Evidence:** KV block allocation (`AllocateKVBlocks`) had a mid-loop failure path that didn't roll back previously allocated blocks, violating KV conservation (INV-4).

**Additional evidence:** H12 hypothesis experiment — preemption loop in `sim/simulator.go:383` accesses `RunningBatch.Requests[len-1]` without bounds check. When all running requests are evicted and the batch is empty, the code panics with index out of range (bug #293).

**Check:** For every loop that mutates state, verify the failure path rolls back all mutations.

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
- **Tier 1 — Synchronously fresh (cluster-owned):** PendingRequests
- **Tier 2 — Stale within tick (instance-owned):** QueueDepth, BatchSize
- **Tier 3 — Stale across batch steps (instance-owned):** KVUtilization, CacheHitRate

**Check:** When writing a new scorer, identify which snapshot fields it reads and their freshness tier. If using only Tier 3 signals, document why or combine with a Tier 1 scorer.

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

**Check:** For every loop that retries an operation after a resource failure, verify there is an explicit bound or progress check. Pay special attention to preemption, eviction, and reallocation loops.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 4.

---

### R20: Degenerate input handling in detectors and analyzers

Anomaly detectors and metric analyzers must explicitly handle degenerate inputs: empty sample sets, single-instance concentration, all-zero distributions, and cross-class comparisons. The degenerate case is often the most important one to detect — a detector that returns "no anomaly" when all traffic hits one instance is worse than useless.

**Evidence:** H14 hypothesis experiment — two detector failures: (1) HOL blocking detector requires ≥2 instances with samples, but `always-busiest` routes ALL traffic to one instance, leaving 3 empty — detector returns 0 for the most extreme HOL case (bug #291). (2) Priority inversion detector uses a 2x threshold that conflates workload heterogeneity with scheduling unfairness — 7,463 false positives with normal configs (bug #292).

**Check:** For every detector or analyzer, identify what happens when one or more inputs are empty, zero, or maximally skewed. Write tests for these degenerate cases.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 9.

---

## Quick Reference Checklist

For PR authors — check each rule before submitting:

- [ ] **R1:** No silent `continue`/`return` dropping data
- [ ] **R2:** Map keys sorted before float accumulation or ordered output
- [ ] **R3:** Every new CLI flag validated (zero, negative, NaN, Inf)
- [ ] **R4:** All struct construction sites audited for new fields
- [ ] **R5:** Resource allocation loops handle mid-loop failure with rollback
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

All 20 rules (R1-R20) are **Active** as of 2026-02-26. No rules have been automated, consolidated, or retired.
