# BLIS-to-llm-d Promotion Pipeline Design

**Status:** Draft
**Date:** 2026-02-26
**Species:** System overview

---

## Problem Statement

We have a simulation-driven algorithm discovery pipeline (OpenEvolve + BLIS) that produces evolved Go functions for routing, admission, and scheduling. We need an automated pipeline that takes these evolved functions and promotes them to the real llm-d system — translating code, generating hypotheses, running benchmarks, and opening PRs on a controlled fork — with human approval only at critical gates.

The core challenge: BLIS and llm-d share the same *concepts* (scorers, filters, queue depth, KV utilization) but have different interfaces, signal access patterns, and real-world concerns that simulation doesn't model. The pipeline must bridge that gap reliably and repeatedly.

**Inputs:** An evolved Go function conforming to a BLIS interface (`RoutingPolicy`, `AdmissionPolicy`, or `InstanceScheduler`).

**Outputs:** A PR on the llm-d fork with translated code, tests, benchmark evidence showing the algorithm's improvement direction matches BLIS predictions, and hypothesis FINDINGS documenting what transferred and what didn't.

**Constraints:**
- Three policy types: routing (-> llm-d scorer/filter), admission (-> llm-d filter), scheduling (-> llm-d scorer)
- Full GPU cluster available for real benchmarks via llm-d-benchmark
- Human gates at stage transitions, not within stages
- One algorithm promoted at a time
- Independent review agent (fresh context, no author bias) before every human gate

---

## Pipeline Overview

### 7 Stages, 3 Human Gates

```
Stage 1: Analyze          (one-time foundation — interface/signal map)
    |
Stage 2: Translate        (BLIS evolved function -> llm-d scorer/filter)
    | -- GATE 1: Review agent + human approval
Stage 3: Hypothesize      (generate Track A/B/C hypotheses for llm-d)
    | -- GATE 2: Review agent + human approval
Stage 4: Implement Tests  (unit + integration tests + benchmark configs)
    |
Stage 5: Run & Validate   (go test + llm-d-benchmark + hypothesis evaluation)
    | -- GATE 3: Review agent + human approval
Stage 6: Document          (FINDINGS per hypothesis, promotion report)
    |
Stage 7: PR               (open PR on fork with full artifact set)
```

### Gate Structure

Each gate has two phases — independent review before human decision:

```
Author agent    ->  produces artifact
                        |
Review agent    ->  independent agent reviews artifact
                    (fresh context, no access to author reasoning)
                        |
Human           ->  sees artifact + review report, approves/rejects
```

The independent reviewer is a separate Claude session that receives ONLY the artifact, reference docs, and a gate-specific checklist. It does NOT receive the author agent's chain-of-thought or design rationale. This prevents the reviewer from anchoring on the author's reasoning.

The reviewer produces a structured report with CRITICAL / IMPORTANT / SUGGESTION findings (same severity levels as the BLIS convergence protocol). If any CRITICAL or IMPORTANT findings exist, the author agent fixes them before presenting to the human. The human always sees a pre-converged artifact.

---

## Stage 1: Interface and Signal Mapping (One-Time Foundation)

This stage produces a translation dictionary that all future promotions reference. Claude builds it by reading both codebases.

### Map A — Interface Translation

| BLIS Interface | BLIS Method | llm-d Equivalent | Target Repo |
|---|---|---|---|
| `RoutingPolicy` | `Route(*RouterState) RoutingDecision` | Scorer plugin (filter->score->pick pipeline) | llm-d-inference-scheduler |
| `AdmissionPolicy` | `Admit(*RouterState, *Request) bool` | Filter plugin (pre-scoring gate) | llm-d-inference-scheduler |
| `InstanceScheduler` | `OrderQueue([]*Request, clock)` | Scorer with queue-position weighting | llm-d-inference-scheduler |

### Map B — Signal Translation

| BLIS Signal | Source | llm-d Signal | Source | Freshness Gap |
|---|---|---|---|---|
| `QueueDepth` | `RoutingSnapshot` | Pod metrics via ext-proc headers | Envoy | ~same (request-level) |
| `KVUtilization` | `RoutingSnapshot` | `llm-d-kv-cache` Indexer | ZMQ events | llm-d is event-driven vs BLIS batch-stale |
| `PrefixMatch` | Router-side LRU | `kvblock.Scorer` longest-match | Centralized index | llm-d is more accurate (real block hashes) |
| `PendingRequests` | `RoutingSnapshot` | Pod queue length header | Envoy | ~same |
| `BatchSize` | `RoutingSnapshot` | Running batch count header | Envoy | ~same |
| `EffectiveLoad()` | QueueDepth + BatchSize + PendingRequests | Composite of above | Computed | Same formula, different freshness per component |

### Map C — Gap Inventory

Signals and concerns that exist in llm-d but NOT in BLIS. The evolved algorithm cannot have optimized for these — the pipeline must flag them as "untested dimensions."

| llm-d Concern | Why BLIS Can't Model It | Risk Level |
|---|---|---|
| LoRA adapter affinity | BLIS has no LoRA concept | Medium — may conflict with routing decisions |
| Pod health / readiness | BLIS instances never fail | High — evolved scorer may route to unhealthy pods |
| Network topology / latency | BLIS routing is instantaneous | Low — unless algorithm is latency-sensitive |
| Prefill/decode role split | BLIS instances are homogeneous | High — scorer must respect role filters |
| Pod heterogeneity (mixed GPUs) | BLIS instances are identical | Medium — KV capacity assumptions may break |
| Real concurrency / race conditions | BLIS is deterministic DES | Medium — concurrent scoring may see inconsistent state |

This dictionary is committed to the fork as a reference doc and updated as either codebase evolves.

---

## Stage 2: Translate

**Input:** An evolved Go function from BLIS (e.g., a modified `WeightedScoring` with conditional logic in `sim/routing.go`).

**Claude's actions:**
1. Read the evolved BLIS function and identify which signals/interfaces it uses
2. Look up each signal in the Stage 1 translation dictionary (Maps A, B, C)
3. Generate the llm-d equivalent — for routing this means a new Scorer plugin in `llm-d-inference-scheduler` conforming to its plugin interface
4. For signals with a **freshness gap** (Map B), insert a comment: `// BLIS assumes batch-stale KV; llm-d has event-driven KV — verify via Track B hypothesis`
5. For signals that **don't exist in llm-d** (Map C gaps), flag as untranslatable and propose a fallback or omission with justification

**Output:** A Go file in the llm-d-inference-scheduler repo (e.g., `pkg/plugins/scorers/evolved_scorer.go`) plus a translation report documenting every mapping decision.

### Gate 1: Translation Review

**Review agent receives:** Translated Go file, BLIS original, translation dictionary.

**Review checklist:**
- Does each BLIS signal map to the correct llm-d signal?
- Are freshness gap annotations present for all Map B mismatches?
- Are Map C gaps flagged, with fallback or omission justified?
- Does the generated code follow llm-d's plugin conventions (registration, interface compliance)?
- Are there signals the evolved function uses that aren't in the translation dictionary?

**Human reviews:** The artifact + review report. Approves, rejects, or requests changes.

---

## Stage 3: Hypothesize

**Input:** The translated code, the BLIS hypotheses that originally validated the algorithm, and the gap inventory.

### Track A — Transfer Hypotheses (BLIS -> llm-d)

Take each BLIS hypothesis that validated the evolved algorithm and rewrite it for llm-d.

**Translation rules:**
- Metric names: BLIS `TTFT` -> llm-d-benchmark `time_to_first_token`
- Operating points: BLIS request rates -> llm-d-benchmark request rates (calibrated to real GPU throughput)
- Infrastructure: BLIS `run.sh` -> llm-d-benchmark profile YAML
- Magnitude: Relax exact thresholds — assert **direction** holds, not exact improvement factor
- Family/type: Preserve from BLIS original

**Example:**
- BLIS: "Prefix-affinity routing produces 2.45x better TTFT than load-only for prefix-heavy workloads" (Cross-policy, Statistical/Dominance)
- llm-d: "Evolved scorer produces better TTFT than default scorer pipeline for prefix-heavy workloads under llm-d-benchmark" (Cross-policy, Statistical/Dominance)

### Track B — Gap Hypotheses (sim-to-real differences)

For each gap in Map C, generate a hypothesis using the family sentence patterns from `docs/standards/experiments.md`.

**Examples:**
- Structural model: "ZMQ event propagation delay of up to 100ms should not change the evolved scorer's instance ranking vs default, verified by TTFT comparison under prefix-heavy workload"
- Robustness: "Under pod failure (1 of 4 pods restarted mid-benchmark), the evolved scorer should recover routing within 10s and not produce worse TTFT than default"
- Structural model: "The evolved scorer's conditional logic based on `EffectiveLoad()` produces the same routing decisions when llm-d signals have different freshness than BLIS signals"

### Track C — Regression Hypotheses

- Cross-policy/Equivalence: "Under workloads where the default scorer pipeline already performs well, the evolved scorer matches within 5%"
- For each existing llm-d-benchmark baseline scenario: "The evolved scorer does not regress [metric] beyond 5% tolerance"

**Output:** A set of hypothesis documents, each with family classification, VV&UQ category, type, and diagnostic clause.

### Gate 2: Hypothesis Review

**Review agent receives:** Hypothesis docs, BLIS originals, gap inventory, `docs/standards/experiments.md`.

**Review checklist:**
- Track A: Are transferred claims faithful to BLIS originals? Are magnitude expectations relaxed appropriately?
- Track B: Does every Map C gap have at least one hypothesis? Are diagnostic clauses actionable ("if this fails, it indicates...")?
- Track C: Are regression baselines comprehensive (all existing benchmark scenarios covered)?
- Classification: Is each hypothesis correctly classified (family, VV&UQ category, type)?
- Coverage: Are there obvious gaps — claims the evolved algorithm relies on that have no hypothesis?

**Human reviews:** The hypothesis set + review report. Approves, rejects, or requests additions.

---

## Stage 4: Implement Tests

**Input:** Approved translated code + approved hypothesis set.

**Claude's actions:**
1. **Unit tests** — Go tests in the llm-d-inference-scheduler test framework verifying the scorer/filter interface contract
2. **Integration tests** — Tests that wire the evolved scorer into the full filter->score->pick pipeline
3. **Benchmark profile YAMLs** — llm-d-benchmark configurations matching the hypothesis operating points (one profile per Track A/B hypothesis)
4. **Hypothesis evaluation scripts** — Scripts that parse benchmark output and evaluate each hypothesis verdict (confirmed/refuted/inconclusive), following the same evidence standards as BLIS experiments

**Output:** Test files, benchmark configs, evaluation scripts — all committed to the fork branch.

No human gate here — the tests are validated by execution in Stage 5.

---

## Stage 5: Run and Validate

**Automated execution:**

1. `go test ./...` in the llm-d-inference-scheduler fork (unit + integration)
2. Deploy the fork to the GPU cluster via llm-d-benchmark standup phase
3. Run each benchmark profile (one per hypothesis)
4. Evaluate hypothesis verdicts against benchmark results
5. Collect: test results, benchmark data, hypothesis verdicts

**Confidence bar:** Tests pass AND benchmarks show the algorithm's improvement direction matches BLIS predictions. Exact magnitude match is not required.

### Gate 3: Validation Review

**Review agent receives:** Test results, benchmark data, hypothesis predictions vs actual results.

**Review checklist:**
- Do all unit and integration tests pass?
- Track A: Does the improvement **direction** match BLIS predictions? (magnitude differences are expected and documented)
- Track B: Are gap hypotheses resolved? Any refuted gap hypotheses that indicate a real sim-to-real problem?
- Track C: Do any existing benchmarks regress beyond 5% tolerance?
- Hypothesis resolution: Is each verdict (confirmed/refuted/inconclusive) consistent with the data?

**Human reviews:** The validation report + review agent findings. This is the final gate before the PR.

---

## Stage 6: Document

**Claude's actions:**
1. **Per-hypothesis FINDINGS** — For each hypothesis (Tracks A/B/C), write a findings document following the BLIS template structure: status, resolution, root cause analysis, scope and limitations
2. **Promotion report** — A summary document covering:
   - What transferred cleanly (Track A confirmed)
   - What didn't transfer (Track A refuted) — with analysis of why and implications for BLIS fidelity
   - Sim-to-real gaps discovered (Track B findings)
   - Regression status (Track C)
   - **Feedback loop recommendations**: specific suggestions for BLIS improvements (new structural model hypotheses, signal modeling changes, new OpenEvolve fitness dimensions)

**Output:** FINDINGS docs + promotion report, committed to the fork branch.

---

## Stage 7: PR

**Claude opens a PR on the fork** containing:
- Translated scorer/filter Go code
- Unit and integration tests
- Benchmark profile YAMLs
- Hypothesis evaluation scripts
- Per-hypothesis FINDINGS docs
- Promotion report
- Translation report (from Stage 2)

**PR description includes:**
- Algorithm origin (which OpenEvolve run, which BLIS fitness score)
- Track A summary (N/M claims transferred)
- Track B summary (gap findings)
- Track C summary (regression status)
- Feedback loop recommendations for BLIS

---

## Feedback Loop

The promotion pipeline is not one-directional. Findings flow back to BLIS:

```
OpenEvolve + BLIS  ──(evolved algorithm)──>  Promotion Pipeline  ──(PR)──>  llm-d fork
       ^                                            |
       |                                            |
       └────────(feedback: BLIS fidelity gaps,      |
                 new hypotheses, signal modeling     |
                 changes, new fitness dimensions)────┘
```

**When Track A hypotheses are refuted:** The BLIS prediction didn't hold in llm-d. This triggers:
1. A new BLIS structural model hypothesis: "BLIS signal X diverges from llm-d signal X under condition Y"
2. Potentially a BLIS code change to improve signal fidelity
3. Re-evolution with updated BLIS model

**When Track B hypotheses are refuted:** A sim-to-real gap matters. This triggers:
1. A BLIS enhancement issue to model the missing concern (e.g., pod failure, LoRA affinity)
2. New OpenEvolve fitness dimensions that penalize algorithms sensitive to the gap

**When Track C hypotheses are refuted:** The evolved algorithm causes regressions. This triggers:
1. Re-evolution with additional regression constraints in the fitness function
2. Or scoping the algorithm to specific workload profiles where it doesn't regress

---

## Repositories and Artifacts

| Artifact | Location | Repo |
|---|---|---|
| This design doc | `docs/plans/2026-02-26-promotion-pipeline-design.md` | BLIS (inference-sim) |
| Translation dictionary (Maps A/B/C) | `docs/promotion/translation-dictionary.md` | llm-d fork |
| Translated scorer/filter code | `pkg/plugins/scorers/evolved_*.go` | llm-d fork (llm-d-inference-scheduler) |
| Hypothesis documents | `docs/promotion/hypotheses/` | llm-d fork |
| Benchmark profiles | `benchmarks/promotion/` | llm-d fork (llm-d-benchmark) |
| Per-hypothesis FINDINGS | `docs/promotion/findings/` | llm-d fork |
| Promotion report | `docs/promotion/report.md` | llm-d fork |
| Feedback issues | GitHub issues | BLIS (inference-sim) |

---

## Gate-Specific Review Checklists (Summary)

| Gate | Review Agent Receives | Checks |
|---|---|---|
| Gate 1 (Translation) | Translated Go file + BLIS original + translation dictionary | Signal mapping correctness, llm-d convention compliance, gap handling |
| Gate 2 (Hypotheses) | Hypothesis docs + BLIS originals + gap inventory + experiments.md | Claim faithfulness, gap coverage, classification correctness, diagnostic clause quality |
| Gate 3 (Validation) | Test results + benchmark data + hypothesis predictions | Direction match, regression detection, hypothesis resolution consistency |

---

## Automation Level

| Stage | Automation | Human Involvement |
|---|---|---|
| 1. Analyze | Fully automated (one-time) | Review the dictionary once |
| 2. Translate | Fully automated | Gate 1 approval |
| 3. Hypothesize | Fully automated | Gate 2 approval |
| 4. Implement Tests | Fully automated | None |
| 5. Run & Validate | Fully automated (cluster deploy + benchmark) | Gate 3 approval |
| 6. Document | Fully automated | None (covered by Gate 3) |
| 7. PR | Fully automated | Standard PR review |

The human touches the pipeline at 3 points (plus final PR review). Everything else is Claude end-to-end.

---

## Scope

**In scope (v1):**
- Routing policies (BLIS `RoutingPolicy` -> llm-d Scorer)
- Admission policies (BLIS `AdmissionPolicy` -> llm-d Filter)
- Scheduling policies (BLIS `InstanceScheduler` -> llm-d Scorer)
- One algorithm at a time

**Out of scope (future):**
- Auto-scaling policies (no BLIS interface yet)
- Batch promotion of multiple algorithms
- Automatic upstreaming from fork to llm-d main
- OpenEvolve integration (triggering promotion automatically after evolution completes)

---

## References

- BLIS hypothesis process: `docs/process/hypothesis.md`
- BLIS experiment standards: `docs/standards/experiments.md`
- BLIS hypothesis template: `docs/templates/hypothesis.md`
- llm-d inference scheduler: https://github.com/llm-d/llm-d-inference-scheduler
- llm-d KV cache: https://github.com/llm-d/llm-d-kv-cache
- llm-d benchmark: https://github.com/llm-d/llm-d-benchmark
- OpenEvolve + BLIS pipeline: https://github.com/toslali-ibm/openevolve/tree/blis/examples/blis_router
