# Agentic Strategy Evolution: A Dual-Loop Methodology for Mechanism Discovery

> **Status:** Draft for review — feedback welcome on structure, framing, and open questions.

---

## What This Document Is

This document describes an evolved form of [Strategy Evolution](strategy-evolution.md) that addresses three limitations of the original single-loop design:

1. **No frontier management.** The original methodology picks one candidate per iteration and commits to it. A hypothesis frontier allows the search to maintain and score competing bundles, enabling best-first exploration rather than greedy commitment.

2. **No sim-to-real discipline.** The original loop stays entirely in simulation. Real-system validation was informal and opportunistic. The evolved methodology makes sim-to-real transfer a first-class, cost-aware outer loop.

3. **Simulator treated as fixed.** BLIS is used as an evaluation harness, but its fidelity is never systematically improved. Co-evolving the simulator alongside the mechanism search produces a progressively more trustworthy scientific instrument.

The result is a three-loop methodology: an **inner loop** for rapid hypothesis search in simulation, an **outer loop** for real-world experiment selection governed by a value-of-information calculus, and a **simulator evolution loop** that co-evolves the simulator's structure, parameters, and trust map alongside the mechanism catalog.

---

## The Core Idea (Unchanged)

The central idea from Strategy Evolution remains: **a strategy is a hypothesis bundle.** Every candidate mechanism is formulated as a set of testable, falsifiable predictions — designed *before* any code is written. Prediction errors, not just fitness scores, are the primary signal for learning.

What changes is the architecture around that core: how bundles are managed, how experiments are selected, and how the simulator improves.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INNER LOOP                                   │
│                  (simulation, fast, cheap)                           │
│                                                                      │
│  Hypothesis Frontier  →  Bundle Design  →  Sim Evaluation           │
│        ↑                                        │                   │
│   Principles +         ←─────────────────  Prediction Error         │
│   Frontier Update                              Analysis              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Promote candidates
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTER LOOP                                   │
│              (real system, expensive, VoI-governed)                  │
│                                                                      │
│  VoI Selector  →  Real Experiment  →  Transfer Analysis             │
│       ↑         (Type A/B/C)               │                        │
│  Coverage + Cost   ←───────────────  Discrepancy Attribution        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Simulator update signals
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SIMULATOR EVOLUTION LOOP                           │
│              (BLIS structural + parametric + data)                   │
│                                                                      │
│  Structure Updates  │  Coefficient Fitting  │  Fidelity Map         │
│  (physics / logic)  │  (α/β parameters)    │  (where to trust)     │
└─────────────────────────────────────────────────────────────────────┘
                               │ Updated simulator + trust map
                               ▼
                    ┌──────────────────────┐
                    │   Knowledge Stores   │
                    │  • Mechanism prncpls │
                    │  • Fidelity prncpls  │
                    │  • Transfer ledger   │
                    │  • Frontier archive  │
                    └──────────┬───────────┘
                               │ Constrains
                               ▼
                         (back to inner loop)
```

---

## Inner Loop: Hypothesis Frontier Search

The inner loop is the original [Strategy Evolution](strategy-evolution.md) five-phase cycle, extended with a frontier manager and a verifier gate.

### What's New: The Hypothesis Frontier

Instead of generating one candidate per iteration and immediately committing, maintain a **scored frontier** of partially-specified hypothesis bundles:

```
frontier_score(bundle) =
    expected_value        # predicted performance gain if confirmed
  + uncertainty_bonus     # prefer bundles with high epistemic value
  + novelty_bonus         # prefer mechanistically unexplored regions
  - eval_cost             # simulator budget required to evaluate
  - principle_violation   # penalty for contradicting known principles
```

The frontier is expanded by generating new bundle candidates (via `/research-ideas` or `/brainstorming`), scored, and pruned when bundles are dominated or refuted. The highest-scoring bundle is selected for the current iteration.

**Why this matters:** Sequential commitment causes re-exploration of similar mechanisms when early candidates succeed. A frontier explicitly tracks what has been tried, scores alternatives, and prevents the search from collapsing to a narrow mechanism family prematurely.

### What's New: The Verifier Gate

Before committing to full simulation, run lightweight verifiers on each candidate bundle:

| Verifier | What it checks |
|----------|----------------|
| Bundle consistency | H-main, ablations, and controls are mutually coherent |
| Principle violations | Bundle doesn't contradict an established principle without justification |
| Control validity | Negative-control conditions are achievable |
| Plausibility | Predicted effect sizes are within physically plausible range |
| Analyzer sanity | `analyze.py` can be written before running experiments |

Bundles that fail a verifier are repaired or pruned before expensive simulation runs.

### The Five Phases (Unchanged)

The original five phases of Strategy Evolution apply unchanged within each inner-loop iteration:

1. **Phase 1: Problem Framing** — `problem.md`, baseline, workload design, success criteria
2. **Phase 2: Hypothesis Bundle Design** — multi-judge candidate review, bundle decomposition, Design Review, human approval gate
3. **Phase 3: Implement and Verify** — strategy code, experiment code, Code Review, execute, compare predictions to outcomes, FINDINGS Review, ledger
4. **Phase 4: Bayesian Parameter Optimization** — Gaussian process over confirmed mechanisms
5. **Phase 5: Principle Extraction and Iteration** — principles from confirmed *and* refuted predictions, fast-fail rules, stopping criterion

The detailed workflow for each phase is documented in [Strategy Evolution](strategy-evolution.md).

### What Changes: Diversity Preservation

Without diversity management, frontier search collapses onto a narrow mechanism family. Maintain a **diversity archive** indexed by:

- **Mechanistic family** — e.g., admission-heavy, queue-depth-dominant, prefix-affinity family
- **Regime behavior** — e.g., works only under KV pressure, works at saturation, regime-independent
- **Performance frontier** — Pareto-optimal bundles across TTFT × throughput tradeoffs

When selecting the next bundle to evaluate, prefer nodes that are both high-scoring *and* mechanistically distinct from recently evaluated bundles. This prevents super-additivity blindness — a family of good mechanisms can eclipse a different family that is also good but unexplored.

---

## Outer Loop: Real-World Experiment Selection

The outer loop governs when, what kind, and in what order to run real-system experiments. Real GPU-hours are scarce; the outer loop makes their use explicit and principled.

### The Core Principle: VoI-Governed Selection

At each decision point, choose the real-world experiment with the highest **expected research value per unit cost**:

```
ROI(e) = E[VoI(e)] / Cost(e)

where:

VoI(e) = w_m · ΔU_mechanism    # reduces uncertainty about whether promoted candidate works
       + w_s · ΔU_simulator    # helps calibrate or repair the simulator
       + w_r · ΔU_ranking      # helps rank frontier candidates correctly
       + w_b · ΔU_boundary     # identifies where a mechanism starts helping or failing
       + w_d · ΔU_deployment   # reduces uncertainty for a likely production choice

Cost(e) = gpu_hours
        + λ · eng_hours        # engineering setup / analysis time
        + μ · calendar_delay   # queueing delay for scarce cluster access
```

This reframes real experimentation from "validate the current best candidate" to "choose the most informative experiment available."

### Three Experiment Types

| Type | What it tests | When to prefer |
|------|--------------|----------------|
| **A — Transfer run** | Does the simulated win transfer end-to-end to reality? | Uncertainty is about whether a promoted mechanism survives real-system effects |
| **B — Calibration probe / microbenchmark** | Is a specific simulator submodel accurate? | Uncertainty is concentrated in one shared submodel that affects many candidates |
| **C — Coverage experiment** | Does the simulator behave correctly in a new regime? | No type-A or type-B bug, but a workload family / concurrency band is uncharacterized |

**Why microbenchmarks often have the highest ROI:** A single type-A transfer run informs one promoted policy very well. A type-B calibration probe that resolves one shared simulator uncertainty (e.g., decode-phase CPU overhead) can improve rank-order prediction for dozens of future comparisons. When a simulator uncertainty is globally shared across many frontier candidates, probes have better amortized value.

**Operating rule:**
- Prefer type-B when uncertainty is in a shared simulator submodel.
- Prefer type-A when uncertainty is about whether a promoted mechanism survives end-to-end.
- Prefer type-C when the primary gap is regime coverage.

### The Transfer Bundle

For every promoted candidate, produce a **transfer bundle** — the analogue of a hypothesis bundle for the outer loop:

```yaml
mechanism_claim: "SLO-gated admission reduces critical TTFT by preventing
                  low-value work from saturating decode capacity."

transfer_claim: "Improvement direction transfers; absolute magnitude may
                 shrink under real backpressure."

fidelity_expectation:
  rank_order: high         # simulator should correctly rank admission > baseline
  direction: high          # improvement direction should hold
  magnitude: medium        # absolute latency values may differ
  boundary: medium         # load threshold where gains appear may shift

simulator_blind_spots:
  - network contention between prefill and decode nodes
  - token-level burst amplification from chunked prefill
  - cache fragmentation under concurrent sessions

diagnostic_mismatch_clause: "If simulator shows gain but real system does not,
  inspect admission-delay overhead and per-token burst amplification. If only
  magnitude differs, re-fit admission threshold coefficient."

data_capture_plan:
  required_observables:
    - per-step latency breakdown (prefill / decode / comm)
    - queue depth evolution over time
    - admission accept/reject rates
  optional_observables:
    - per-token scheduling events
    - KV cache occupancy trace
  calibration_targets:
    - decode step time coefficient (β_decode)
    - admission overhead constant
  coverage_tags: [admission, near-saturation, mixed-SLO, bursty-arrival]
```

### Dual-Purpose Real Runs

**Sim-to-real transfer experiments should be treated as multi-purpose scientific probes** — not just validation checkpoints. Every outer-loop run can simultaneously:

1. **Validate** the promoted mechanism
2. **Test** simulator fidelity hypotheses
3. **Collect** data for coefficient fitting and workload modeling

This avoids paying for GPU time twice. The `data_capture_plan` in the transfer bundle ensures the run is instrumented appropriately upfront.

**When transfer experiments are not enough:** Transfer candidates are selected because they look promising — creating selection bias in the collected data. If the simulator needs calibration on corner cases or specific submodels not covered by promoted candidates, dedicated type-B probes remain necessary.

### Fidelity Hypothesis Classes

In addition to mechanism hypotheses, define **fidelity hypotheses** that explicitly test what the simulator gets right:

| Arm | What it tests |
|-----|--------------|
| **H-fidelity-rank** | Simulator correctly ranks top-k candidates |
| **H-fidelity-direction** | Simulator preserves improvement direction over baseline |
| **H-fidelity-boundary** | Simulator correctly predicts the load regime where gains appear or vanish |
| **H-fidelity-gap** | Simulator overestimates or underestimates magnitude by no more than X% |

Every serious outer-loop campaign should test both a mechanism hypothesis and the corresponding fidelity hypothesis simultaneously.

---

## Simulator Evolution Loop

The simulator is not a fixed evaluation harness — it is a **scientific instrument that co-evolves** with the mechanisms it is used to study.

### The Key Reframing

> Stop asking: "Is the simulator accurate?"
>
> Start asking: "Where is the simulator reliable enough for which decisions?"

A simulator does not need to be perfect. It needs to be **trustworthy for specific decisions**:
- Inner-loop pruning of frontier candidates requires rank-order fidelity.
- Publication claims require directional fidelity.
- Deployment recommendations may require regime-boundary fidelity.
- Absolute latency SLO targets require magnitude fidelity.

The fidelity map records these distinctions.

### Five Update Types

| Layer | What changes | When to update |
|-------|-------------|----------------|
| **Structural model** | Physics, logic, causal mechanisms — queueing disciplines, batching model, KV cache eviction, communication model | Systematic directional mismatch; wrong regime boundary; missing interaction effect |
| **Parametric model** | Fitted coefficients — α/β step-time terms, network latency multipliers, CPU overhead constants | Correct direction but wrong magnitude; consistent bias |
| **Data model** | Workload distributions — arrival processes, token length histograms, burstiness, cache reuse | Simulator works on synthetic workloads but fails on real traces |
| **Observability** | Traces, counters, per-step breakdowns, event logs | Cannot explain discrepancies; multiple causal explanations possible |
| **Fidelity map** | Trust regions — where the simulator is reliable for which decision types | After outer-loop validation runs; accumulated evidence |

### Discrepancy Attribution

Every sim-to-real mismatch requires attribution before acting on it:

| Root cause | Implication |
|-----------|-------------|
| Mechanism theory wrong | Strategy was wrong; update mechanism frontier, extract principle |
| Simulator missing causal factor | Structural model update needed |
| Coefficients mis-calibrated | Parametric re-fitting needed |
| Experiment mapping poor | Real run under different conditions than intended |
| Real deployment introduced hidden constraint | Transfer bundle blind-spot; update list for future |

**Never attribute discrepancy before investigation.** The transfer bundle's diagnostic mismatch clause guides this.

### Three Knowledge Stores

All three loops write to a shared knowledge state, which feeds back into the inner loop's frontier scoring:

**Mechanism knowledge store:**
- Confirmed and refuted hypothesis bundles
- Extracted design principles (RP-N, S-N)
- Regime-specific applicability boundaries

**Simulator knowledge store:**
- Validated fidelity claims per submodel
- Known blind spots and missing factors
- Calibration dataset versions
- Fidelity map (where to trust, where to be cautious)

**Transfer ledger:**
- One row per outer-loop experiment
- Simulation prediction, real-system outcome, discrepancy, attribution, update triggered

---

## Positioning: What This Is and Isn't

### What Problem This Solves

In systems with multiple interacting policy layers — routing, scheduling, memory management, admission control — the optimal configuration cannot be derived analytically or discovered by tuning numeric parameters alone. Interactions produce non-obvious emergent behaviors: super-additive effects, signal cancellation, and regime-dependent dominance.

**The target is not a faster artifact. The target is a validated understanding of which mechanisms work, why they work, where they break, and how to transfer that understanding to real systems.**

### Not a World Model

Recent agentic search systems (e.g., K-Search) use large language models as **world models** — learned predictors of which actions are likely to produce better program artifacts. The LLM estimates the value of search actions and guides the search frontier accordingly.

This methodology uses a different construct: a **predictive causal model** built through structured experimentation.

| Property | LLM World Model (K-Search style) | Predictive Causal Model (this work) |
|----------|----------------------------------|--------------------------------------|
| Representation | Learned, implicit, neural | Explicit, structured, auditable |
| Update signal | Fitness score (performance) | Prediction error + causal attribution |
| Primary output | Better search prioritization | Better causal understanding |
| Persistent artifact | Updated search tree | Principles catalog + fidelity map |
| Falsifiability | Not directly | Designed in (negative controls, ablations) |

The predictive causal model is also distinct because it explicitly encodes **what should fail** — the negative control conditions, the regime boundaries where a mechanism should stop working, and the ablations that isolate individual components. A world model does not naturally represent "where this mechanism should not help."

### Not Parameter Search

Bayesian optimization (Phase 4) is one component of this methodology, but the methodology is not reducible to it. Bayesian optimization tunes numeric values given a confirmed mechanism. This methodology discovers which mechanisms are worth tuning in the first place — and specifically separates mechanism design from parameter tuning so that comparisons are fair.

### Not RL

Reinforcement learning over the policy search problem has an obvious appeal but two significant problems in this domain:

1. The state space is discontinuous — routing policy logic, admission predicates, and scheduling disciplines are not parameterized by a common continuous space.
2. High fitness does not imply understanding. An RL agent can discover a high-performing policy that exploits simulator artifacts or overfits to a specific workload, without the methodology having learned *why* it works or *where* it will fail.

This methodology prioritizes epistemic quality over search speed. An iteration that ends in a refuted H-main with a clear diagnostic clause is more valuable than a confirmed iteration whose mechanism is not understood.

---

## Open Questions for Review

The following are genuine open questions where feedback is welcome:

**1. Frontier scoring calibration.**
The frontier score combines expected value, uncertainty, novelty, cost, and principle penalty. In practice, how should these weights be set? Should they be learned from the ledger, or specified manually per domain?

**2. Fidelity hypothesis coverage.**
The methodology proposes H-fidelity-rank, H-fidelity-direction, H-fidelity-boundary, and H-fidelity-gap. Are these the right axes? What fidelity property matters most for inner-loop pruning decisions?

**3. Promotion policy for the outer loop.**
Which simulated wins should be promoted to real-system validation? The document sketches criteria (strong simulation win, high uncertainty, high upside, likely to expose simulator weakness) but a concrete promotion rule is not yet specified.

**4. Bundle overhead at scale.**
The hypothesis bundle structure adds significant evaluation cost (multiple arms, multiple reviews). How do the fast-fail and tiered-review mechanisms interact with the frontier — specifically, when a bundle's H-main is refuted after a verifier passes it, how should the frontier be updated?

**5. Adversarial workload generation.**
The document briefly mentions using a red-team agent to find workload conditions where a mechanism should fail. This would tighten H-control-negative and H-robustness arms. Is this the right level of rigor for all iterations, or only for mechanisms being promoted to the outer loop?

**6. Simulator co-evolution scope.**
Should structural simulator updates be gated by a formal SCP (Simulator Change Proposal) artifact with its own review cycle? Or should they be treated as fast engineering changes triggered by discrepancy analysis?

**7. Multi-fidelity calibration hierarchy.**
The document proposes type-A transfer runs, type-B probes, and type-C coverage experiments. Should there be a fourth level — real-system A/B tests for production deployment validation — that sits above type-A?

---

## Relationship to Existing Workflow

| Skill / Process | Role in Extended Methodology |
|---|---|
| [Strategy Evolution](strategy-evolution.md) | Inner loop (all five phases) |
| [Hypothesis bundles](hypothesis-bundles.md) | Bundle design in Phase 2; mechanism arm structure |
| `/convergence-review` | Design Review (Phase 2c), Code Review (Phase 3c), FINDINGS Review (Phase 3g) |
| `/research-ideas` | Frontier candidate generation (Phase 2a) |
| `blis observe` + `blis calibrate` | Outer loop — transfer runs and calibration data collection |
| `ledger.md` | Inner loop ledger (one row per iteration, prediction accuracy column) |
| Transfer ledger | Outer loop ledger (one row per real experiment, discrepancy + attribution) |
| Simulator knowledge store | Fidelity map, known blind spots, calibration dataset versions |

---

## Summary

This methodology evolved from a single loop into a system of three coupled loops because:

- **Simulation is fast but not ground truth** — a sim-to-real outer loop is needed.
- **Sequential iteration is greedy** — a frontier over bundles enables more principled exploration.
- **The simulator is a scientific instrument** — its fidelity should co-evolve with the knowledge it helps produce.

The durable outputs are:
1. A winning strategy (or strategy family) validated both in simulation and in reality
2. A mechanism principles catalog that constrains future search
3. A simulator fidelity map that characterizes where the simulator can be trusted
4. A transfer ledger that makes the sim-to-real campaign auditable

The methodology is domain-agnostic. The instantiation described here was developed for LLM inference serving using BLIS, but the three-loop structure, the hypothesis bundle, and the VoI-governed outer loop apply to any complex system with multiple interacting policy layers and a simulation-based evaluation harness.
