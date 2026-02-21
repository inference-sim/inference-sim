# BLIS Experiment Standards

Hypothesis-driven experimentation is a first-class activity in BLIS — equal in rigor to implementation and design. The experiment framework is grounded in Verification, Validation, and Uncertainty Quantification (VV&UQ) from simulation science.

### VV&UQ Framing

Every hypothesis falls into one of three VV&UQ categories. This determines what kind of evidence is needed.

| Category | Question | Evidence required | Examples |
|----------|----------|-------------------|---------|
| **Verification** | Does the code implement the intended math/logic? | Exact invariant checks. Failure = bug. | H12 (conservation), H13 (determinism), H22 (input validation) |
| **Validation** | Does the model match expected system behavior? | Statistical comparison against analytical baselines or real data within a pre-specified accuracy interval. | Cross-validation against M/M/k; H19 (roofline vs blackbox) |
| **Uncertainty Quantification** | How confident are we in the region where a finding holds? | Confidence intervals on thresholds; probability statements on properties. | H8 (preemption cliff at 2100±? blocks); H10 (28% improvement at this operating point — what about others?) |

Most current experiments are **Verification** (invariant checking) or informal **Validation** (metric comparison). Future experiments should increasingly incorporate **UQ** — every threshold finding should include a confidence interval, every "confirmed" result should quantify the probability of holding under parameter variation.

### Three purposes

1. **Verification** — confirm that code implements intended math correctly (scheduler invariants, conservation laws)
2. **Validation** — confirm that model outputs match expected behavior within acceptable accuracy intervals
3. **Discovery** — surface bugs, design gaps, and undocumented limitations

### How to choose your VV&UQ category

```
Is your hypothesis about whether the CODE is correct?
  → Yes: Verification (e.g., "conservation holds," "deterministic output")
  → No: ↓
Is your hypothesis comparing the MODEL's output to expected behavior?
  → Yes: Validation (e.g., "policy A beats B," "TTFT ∝ input tokens")
  → No: ↓
Is your hypothesis about the BOUNDARIES or CONFIDENCE of a finding?
  → Yes: UQ (e.g., "preemption cliff at 2100±100 blocks," "P(stable) > 0.95")
```

The VV&UQ category determines what counts as evidence:
- **Verification:** Exact invariant checks. One failure = bug. Single seed sufficient.
- **Validation:** Statistical comparison within pre-specified accuracy interval. 3+ seeds. Formal tests (KS, Mann-Whitney U).
- **UQ:** Confidence intervals on thresholds. Parameter sweeps. Sensitivity analysis.

### Formal statistical rigor

Experiments involving statistical claims must use proper hypothesis tests, not ad-hoc thresholds:

- **Distribution validation** (workload/arrival family): [Kolmogorov-Smirnov test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html) — compares a sample against a theoretical CDF. Reject if p < 0.05. In Python: `from scipy.stats import kstest; stat, p = kstest(samples, 'expon', args=(0, 1/rate))`.
- **Metric comparison** (cross-policy family): [Mann-Whitney U test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html) — non-parametric comparison of two independent samples. Report effect size AND confidence interval, not just "X% better." In Python: `from scipy.stats import mannwhitneyu; stat, p = mannwhitneyu(a_values, b_values)`.
- **Threshold estimation** (performance-regime family): Report thresholds with confidence intervals. "Preemption cliff at 2100 blocks" → "Preemption cliff at 2100 ± 100 blocks (95% CI across seeds 42, 123, 456)."
- **Invariant probability** (scheduler invariants family): For stochastic invariants, estimate P(invariant holds) with a confidence interval, not just "holds for 3 seeds."

**Note on scipy:** The tests above use `scipy.stats`. Install with `pip install scipy` if needed. For experiments that only use standard-library Python, the legacy thresholds (below) remain acceptable.

Legacy thresholds (still valid for experiments without scipy):
- **>20% improvement** consistent across all seeds = significant
- **<10% in any seed** = inconclusive
- **Within 5%** across all seeds = equivalent (for equivalence tests)

These thresholds were chosen pragmatically — 20% ensures the effect is visible above seed-to-seed variance in typical BLIS experiments; 5% accounts for floating-point and timing noise. They are not derived from formal power analysis. New experiments should prefer formal tests where scipy is available.

### Cross-validation against analytical models

Where applicable, validate DES outputs against analytically-tractable models under matching assumptions. This grounds the simulator in theory.

- **M/M/k baseline**: [M/M/k](https://en.wikipedia.org/wiki/M/M/c_queue) is the standard queueing model with Markovian (Poisson) arrivals, Markovian (exponential) service times, and k servers. Under matching assumptions, compare DES queue length distribution against the M/M/k analytical solution. **Caveat:** BLIS uses batching and deterministic service times (alpha/beta coefficients), so exact M/M/k matching is not possible. The comparison requires configuring BLIS with `--max-batch-size 1` and interpreting the service time distribution as approximately exponential. Divergence may indicate modeling errors OR fundamental architectural differences from M/M/k assumptions.
- **Little's Law**: For any stable configuration, verify L = λW (average queue length = arrival rate × average wait time). This is a universal law that must hold. **In BLIS terms:** L = mean `still_queued` from per-instance metrics; λ = `injected_requests / (sim_duration_us / 1e6)`; W = mean scheduling delay from `scheduling_delay_p99_ms` (approximate). Extract from JSON output.
- **Phase structure**: Verify that prefill time ∝ prompt tokens and decode time ∝ output tokens by fitting linear models and checking R² > 0.95. **In BLIS terms:** prefill time ≈ TTFT (time to first token); decode time ≈ E2E - TTFT. Vary `input_distribution` mean while holding `output_distribution` constant, and vice versa.

## Experiment Classification

Every hypothesis must be classified before designing the experiment. The classification determines rigor requirements.

### Type 1: Deterministic Experiments

**Definition:** Verify exact properties — invariants, conservation laws, error handling boundaries. Same seed = same result, guaranteed.

**Requirements:**
- Single seed sufficient (determinism is the point)
- Pass/fail is exact — the invariant holds or it doesn't
- Failure is ALWAYS a bug (never noise)
- No statistical analysis needed

**Examples:**
- H12: Request conservation (INV-1) holds across 10 policy configurations (67 checks)
- H13: Same seed produces byte-identical output
- H22: Zero KV blocks panics at CLI boundary, not deep in simulation

**Pass criteria:** The invariant holds for every configuration tested. One failure = bug.

---

### Type 2: Statistical Experiments

**Definition:** Compare metrics (TTFT, throughput, distribution uniformity) across configurations. Results vary by seed.

**Requirements:**
- **Minimum 3 seeds** (42, 123, 456) for each configuration
- **Effect size thresholds:**
  - **Significant:** >20% improvement consistent across ALL seeds
  - **Inconclusive:** <10% in any seed
  - **Equivalent:** within 5% across all seeds (for equivalence tests)
- **Directional consistency:** the predicted direction must hold across ALL seeds. One contradicting seed = hypothesis not confirmed
- **Report:** mean, min, max across seeds for primary metric. Include per-seed values for transparency.

**Subtypes:**

#### Dominance
A is strictly better than B on metric M.

- **Analysis:** Compare metric M for A vs B across all seeds. Compute ratio per seed.
- **Pass:** A beats B on M for all seeds, with >20% effect size in every seed.
- **Examples:** H3 — queue-depth TTFT is 1.7-2.8x better than kv-utilization across 3 seeds. H14 — `always-busiest` routing produces 4.6x worse TTFT and routes all 500 requests to a single instance.

#### Monotonicity
Increasing X should monotonically increase/decrease Y.

- **Analysis:** Run at >=3 values of X. Verify Y changes monotonically.
- **Pass:** Y is strictly monotonic in X across all seeds. No inversions.
- **Example:** H8 — reducing total KV blocks increases preemption frequency. H9 — increasing prefix_length decreases TTFT.

#### Equivalence
A ~ B within tolerance (baseline sanity checks).

- **Analysis:** Compare metric M for A vs B. Compute percentage difference per seed.
- **Pass:** |A - B| / max(A, B) < 5% across all seeds.
- **Example:** H4 — round-robin ~ least-loaded for uniform workloads at low rates. H23 — all policies equivalent at near-zero load.

#### Pareto
No single configuration dominates all metrics simultaneously.

- **Analysis:** Run N configurations, measure multiple metrics. Identify Pareto-optimal set.
- **Pass:** At least 2 configurations are Pareto-optimal (each best on >=1 metric).
- **Example:** H17 — different scorer weights optimize for different objectives (TTFT vs throughput).

---

## Hypothesis Formation

Hypotheses must be **conceptual and behavioral**, not code-grounded. This is the experimental analogue of behavioral vs structural testing.

### Conceptual hypotheses test system behavior

A good hypothesis is an intuitive claim about system behavior: "burst smoothing should reduce tail latency," "tiered storage should reduce preemptions," "same seed should produce identical output." These claims are based on systems thinking, not on reading the implementation.

### Do NOT read the code before forming hypotheses

Reading the code before hypothesizing is like writing structural tests — you end up testing the implementation, not the behavior. The value of hypothesis-driven experimentation is that *conceptual claims failing against the implementation* surfaces design limitations that code-aware experiments would avoid.

Evidence: If H5 had read `admission.go:45` before hypothesizing, the experimenter would have designed a "correct" experiment with cap=100K, confirmed a tiny effect, and **missed the discovery** that the per-input-token cost model makes burst smoothing structurally impossible at practical parameters. The conceptual hypothesis exposed a design limitation that a code-grounded hypothesis would have sidestepped.

### "Mechanism not plausible" is a valid resolution

When a conceptual hypothesis fails because the implementation doesn't support the assumed mechanism, this is the resolution "Refuted — mechanism not plausible." This is a **design limitation finding**, not an experimenter error. The hypothesis did its job — it revealed a gap between how users think the system works and how it actually works.

---

## Hypothesis Families

Every hypothesis belongs to a **family** (what domain is being tested) AND a **type** (how rigor is assessed). These are orthogonal — a scheduler invariant can be deterministic or statistical; a cross-policy comparison is always statistical.

### The six families

| Family | Tests | Hypothesis shape | Typical type | Examples |
|--------|-------|-----------------|-------------|---------|
| **Workload/arrival** | Input generation: distributions, rates, burstiness, mix proportions | "Generator X produces arrivals matching distribution D within tolerance T" | Statistical | H16, H20 |
| **Scheduler invariants (safety/liveness)** | Conservation, determinism, lifecycle, livelock protection | "For ALL configurations, property P holds" (universally quantified) | Deterministic | H12 (conservation), H13 (determinism), H25 |
| **Performance-regime (scaling laws)** | Saturation curves, throughput-latency tradeoffs, horizontal scaling | "Metric M is monotonic/convex in parameter P" | Statistical/Monotonicity | H7 (scaling), H8 (KV pressure), H11 (batch formation) |
| **Structural model** | DES model assumptions: phase structure, KV mechanics, signal freshness, prefix caching | "Component C behaves according to model assumption A" | Mixed | H3 (signal freshness), H9 (prefix caching), H10 (tiered KV), H26 |
| **Robustness/failure-mode** | Overload, misconfiguration, degenerate inputs, pathological policies | "Under stress condition S, the system exhibits defined behavior B (not undefined state)" | Deterministic or Statistical | H5 (token-bucket), H14 (pathological), H21, H22, H24 |
| **Cross-policy comparative** | Policy ordering, Pareto frontiers, robustness to workload shifts | "There EXISTS a workload where policy A beats B on metric M" (existentially quantified) | Statistical/Dominance or Pareto | H1, H2, H4, H6, H15, H17, H18, H19, H23 |

### Family-specific hypothesis sentence patterns

Use these templates when generating new hypotheses. Each family has a characteristic sentence shape that ensures testability. See also `docs/process/hypothesis.md` for the full generation guide.

| Family | Sentence pattern | Example |
|--------|-----------------|---------|
| **Workload/arrival** | "Generator G with parameters P should produce distribution D with property X within tolerance T" | "Gamma sampler with CV=3.5 should produce inter-arrival times with CV within 10% of 3.5 over 10K samples" |
| **Scheduler invariants** | "For ALL configurations C, invariant I holds at simulation end" | "For all routing × scheduling × admission combinations, injected == completed + queued + running" |
| **Performance-regime** | "Metric M should be monotonically non-decreasing/non-increasing in parameter P across range [a, b]" | "TTFT P99 should be monotonically non-decreasing in offered load from 500 to 5000 req/s" |
| **Structural model** | "Component C should behave according to assumption A, verified by observable O" | "Prefill time should be proportional to input token count (R² > 0.95 for linear fit)" |
| **Robustness** | "Under stress condition S, the system should exhibit behavior B and NOT exhibit behavior X" | "Under 10x overload, the system should reject excess requests and NOT deadlock or panic" |
| **Cross-policy** | "Under workload W, policy A should produce better metric M than policy B because of mechanism Z" | "Under mixed-SLO workload, priority-FCFS should produce lower realtime TTFT than FCFS because realtime requests get scheduled first" |

### Family × Type matrix

| | Deterministic | Statistical/Dominance | Statistical/Monotonicity | Statistical/Equivalence | Statistical/Pareto |
|---|---|---|---|---|---|
| Workload/arrival | Seed reproducibility | Distribution match | Rate scaling | — | — |
| Scheduler invariants | **Primary** (INV-1 through INV-6) | — | — | — | — |
| Performance-regime | — | — | **Primary** (scaling curves) | Baseline sanity | Knee behavior |
| Structural model | Phase structure | Signal freshness | Cache effectiveness | — | — |
| Robustness | Input validation | Overload behavior | — | — | — |
| Cross-policy | — | **Primary** (A vs B) | — | Low-load equivalence | Multi-scorer tradeoffs |

### Family determines rigor requirements

- **Scheduler invariants**: Single seed sufficient. Pass/fail is exact. One failure = bug.
- **Cross-policy comparative**: 3+ seeds minimum. Must control confounding variables (ED-1, ED-6).
- **Performance-regime**: Sweep points (≥3 values of the independent variable), not just pairwise comparison.
- **Workload/arrival**: Statistical tests on generated distributions. Long runs for accurate rate estimation.
- **Structural model**: Code-level verification (RCV-1, RCV-4) is essential — these test implementation assumptions.
- **Robustness**: Must test BOTH the defined behavior AND verify no undefined states (deadlock, panic, data loss).

### Relationship to existing invariants and rules

| Family | Related invariants | Related rules |
|--------|-------------------|---------------|
| Scheduler invariants | INV-1 (conservation), INV-2 (lifecycle), INV-3 (clock monotonicity), INV-5 (causality), INV-6 (determinism) | R1 (no silent data loss), R5 (transactional mutation) |
| Structural model | INV-4 (KV conservation), INV-7 (signal freshness) | R2 (sort map keys), R11 (guard division), R17 (signal freshness) |
| Robustness | — | R3 (validate CLI flags), R19 (livelock protection), R20 (degenerate inputs) |
| Cross-policy | — | R18 (CLI flag precedence) |

---

## Experiment Design Rules

### ED-1: Controlled comparison
Vary exactly one dimension between configurations. Everything else held constant (same model, same instances, same workload, same seed). If the experiment requires varying multiple dimensions, decompose into separate sub-experiments.

### ED-2: Rate awareness
Many effects are rate-dependent (e.g., signal freshness only matters at high rates). When the hypothesis involves load-dependent behavior:
- Run at the target rate where the effect is expected
- Also run at a rate where the effect should vanish (to confirm the mechanism, not just the outcome)
- Document the rate-dependent transition point if observed

### ED-3: Precondition verification
Before comparing configurations, verify the experiment preconditions hold. Examples:
- Testing SJF vs FCFS? Verify queue depth exceeds batch size (otherwise both produce identical batches).
- Testing cache hit benefit? Verify KV blocks are large enough to hold the prefix (otherwise LRU eviction destroys it).

Document the precondition check in the experiment script (not just in prose).

### ED-4: Workload seed independence
**Resolved (#284):** CLI `--seed` now overrides the workload-spec YAML `seed:` field when explicitly passed. Behavior:
- `--seed N --workload-spec w.yaml` → workload uses seed N (CLI override)
- `--workload-spec w.yaml` (no `--seed`) → workload uses YAML `seed:` value (backward compatible)
- CLI-generated workloads (`--rate`, `--num-requests`) → `--seed` controls everything (unchanged)

For multi-seed experiments: simply vary `--seed` on the command line. No need to generate per-seed YAML copies.

**Note:** The YAML `seed:` field still serves as the default seed for the workload when `--seed` is not explicitly specified. This enables the "shareable workload" pattern — distributing a YAML file that always produces the same workload by default.

### ED-5: Reproducibility
Every experiment must be reproducible from its artifacts alone:
- `run.sh` must build the binary and run all variants
- Exact seed values documented
- Exact commit hash recorded (or the experiment is tied to a specific branch/PR)
- No manual steps between script invocation and results

### ED-6: Config diff against reference experiments
When an experiment reuses calibration data from a prior experiment (e.g., "H8 found the preemption cliff at 2100 blocks, so we use 2100"), **diff every CLI flag and YAML field** between the two experiments. Document any differences. Even a single changed flag (e.g., routing policy) can invalidate the calibration.

Evidence: H10 used `--routing-policy least-loaded` while H8 used the default `round-robin`. This shifted the preemption cliff, producing zero preemptions where H8 found 11%. The mismatch was not caught until post-publication code review.

---

## Root Cause Verification

After analyzing results (step 7) and before classifying findings (step 8), every experiment MUST verify its causal explanations. This step exists because plausible narratives can pass review without being correct.

### RCV-1: Every causal claim must cite `file:line`

A root cause analysis that says "the tiered cache increases total capacity" without citing the code that does this is a *hypothesis about the root cause*, not a verified root cause. Trace the claim through the code:
- Which function implements the claimed behavior?
- What are the exact conditions under which it fires?
- Does the claimed mechanism actually change the measured metric?
- **Tracing depth**: The citation must trace to the code that *directly modifies the measured metric*, not just to the constructor or factory that creates the relevant object. Citing `NewKVStore` and claiming "this creates a tiered cache with more capacity" is insufficient — you must verify that the GPU block count actually changes in the created object.

Evidence: H10 claimed "CPU tier increases total effective capacity" — but `NewKVStore` (`kv_store.go:31-36`) does not change GPU block count. The actual mechanism was `maybeOffload` preserving prefix hashes (`kvcache_tiered.go:214-224`).

### RCV-2: Every "surprise" must have a first-principles calculation

Before labeling a result as "surprising," compute the expected value from the system's parameters. If the result matches the calculation, it is not a surprise — it is the expected outcome of a mechanism you didn't initially consider.

Evidence: H5 labeled 96% rejection as a "surprise." But `admission.go:45` charges `len(req.InputTokens)` per request (mean=512). Token demand (1,024,000 tokens/s) exceeds supply (400 tokens/s) by 2,560x. The 96% rejection is the mathematically inevitable steady state.

### RCV-3: Check the mechanism, not just the direction

Confirming that "A is better than B" is necessary but not sufficient. The root cause analysis must explain *why* through a specific code path. A correct directional result with an incorrect explanation is a ticking time bomb — the explanation will mislead future experiments.

**Paradox flag**: If the proposed mechanism predicts the *opposite* direction of what would be intuitive (e.g., "fewer cache hits improving performance"), treat this as a red flag. Before accepting a paradoxical explanation, independently verify the underlying data. In H10, the claim "fewer cache hits → better TTFT" survived two rounds because the data (from a buggy analyzer) appeared to support it. The corrected data showed cache hits *increased*, resolving the paradox. When mechanism and intuition disagree, verify the data first.

### RCV-4: Validate causal claims with control experiments

When a mechanism is proposed (e.g., "`maybeOffload` causes the TTFT improvement"), design a control experiment that disables **only that mechanism** (e.g., `--kv-offload-threshold 1.0`). If the effect vanishes, the mechanism is confirmed. If it persists, the explanation is wrong.

Evidence: H10 proposed `maybeOffload` as the mechanism. The control experiment (threshold=1.0) produced output byte-identical to single-tier, confirming `maybeOffload` as the sole cause. Without this control, the mechanism question ("does maybeOffload cause the TTFT improvement?") would have remained unverified.

### RCV-5: Confirmation bias guard (Devil's Advocate)

Before sending FINDINGS.md to external review, the experimenter must write a **Devil's Advocate** section: 2-3 sentences arguing the **opposite** of the conclusion. This is a pre-review self-check that forces consideration of alternative interpretations.

```markdown
## Devil's Advocate

**If this is "Confirmed," argue why it might be Refuted:**
The 69x TTFT improvement could be entirely from load shedding (96% rejection)
rather than burst smoothing. A firewall that blocks all traffic also has great
latency for the requests that pass.

**If this is "Refuted," argue why it might be Confirmed:**
The calibrated bucket (cap=100K) showed a 4% improvement — small but consistent
across 2 of 3 seeds. This might be a real but tiny burst-smoothing effect masked
by workload noise.
```

The reviewers see both the conclusion AND the counter-argument. This prevents the failure mode where the experimenter writes "Confirmed" and the reviewers are anchored by that label.

Evidence: H5 was labeled "Confirmed" for three rounds. Nobody argued the alternative until Round 3's honest reassessment. A Devil's Advocate section in Round 1 would have surfaced "could this be load shedding?" immediately.

### RCV-6: Mandatory Scope and Limitations

Every FINDINGS.md must include a **Scope and Limitations** section documenting:
- Exact operating point tested (blocks, rate, seeds, instances, routing)
- Parameters the findings depend on
- What was NOT tested that could change the conclusion
- Whether the finding generalizes or is specific to the tested configuration

Evidence: H10's "28% TTFT improvement" is specific to GPU=2100 blocks near the preemption cliff. Without the scope section, this number would be cited as a general property of tiered KV caching.

---

## Iterative Review Protocol

Every hypothesis experiment iterates **until convergence** (max 10 rounds), with **three parallel Opus 4.6 reviews per round**. No minimum round count — convergence in Round 1 is valid if all three reviewers agree.

Each round runs three parallel reviews with different focus areas (see `docs/process/hypothesis.md` for the full protocol):
- **Reviewer A (Mechanism):** Code-level verification of causal claims (RCV-1, RCV-3, RCV-4)
- **Reviewer B (Design):** Confounds, missing controls, parameter calibration (ED-1 through ED-6)
- **Reviewer C (Rigor):** First-principles calculations, sample size, scope of claims (RCV-2)

**Convergence:** All three reviewers have no remaining items that require a new experiment. Max 10 rounds as a safety valve.

**Why three parallel reviewers instead of sequential rounds:** PR #310's multi-model reviews showed no single reviewer caught all issues. Gemini 3 Pro caught the queue-time split, Opus 4.6 caught the confound matrix need, GPT-4o caught scope limitations. Three parallel reviewers maximize coverage per round.

---

## Hypothesis Resolution

Every hypothesis resolves to a **status** (did the prediction hold?) and a **resolution** (what do we do about it?). These are distinct — a "confirmed" hypothesis can still have a wrong-mechanism resolution that changes user guidance entirely.

### Status (the prediction)

| Status | Definition | Example |
|--------|-----------|---------|
| **Confirmed** | The predicted directional outcome holds across all seeds | H13: same seed → byte-identical output |
| **Confirmed with nuance** | The prediction holds but the mechanism or practical implications differ from expected | H5: token-bucket reduces TTFT 69x but via 96% load shedding, not burst smoothing; no practical sweet spot |
| **Partially confirmed** | Some predictions hold, others don't, or the experiment tested something different than intended | H14: routing pathological confirmed, scheduling showed double-inversion cancellation |
| **Refuted** | The predicted outcome does not hold across seeds | *(not yet observed — the refutation IS the value)* |
| **Inconclusive** | Effect is within noise (<10% in any seed) or parameter-dependent | H5 exp4: calibrated bucket shows <5% TTFT improvement |

### Resolution (what we learned and what to do)

| Resolution | Definition | Action | Example |
|-----------|-----------|--------|---------|
| **Clean confirmation** | Hypothesis holds, mechanism matches prediction | Document. No further action. | H13, H3, H8 |
| **Confirmation with wrong mechanism** | Prediction holds directionally but the underlying cause differs | Correct the explanation. May change user guidance entirely. | H5: improvement is load shedding, not burst smoothing |
| **Confirmation with bug discovery** | Prediction holds but experiment surfaces code defects | File issues (`--label bug`). Fix in separate PRs. | H12: conservation holds but preemption panics. H14: routing works but 3 detector bugs. H10: tiered KV confirmed but analyzer bug masked preemptions for 2 rounds. |
| **Partial confirmation with surprise** | Some predictions fail; unexpected useful insights emerge | Document surprise. May spawn new hypotheses. | *(use when the experiment finds something valuable but different from what was hypothesized)* |
| **Refuted — mechanism not plausible** | The hypothesis assumed a mechanism that the implementation doesn't support | File design issue if the mechanism *should* exist but doesn't. Document the actual mechanism. | H5: hypothesis assumed burst smoothing, but per-input-token cost model (`admission.go:45`) makes burst smoothing structurally impossible at practical parameters |
| **Refuted — system design flaw** | Prediction fails because system doesn't work as designed | File design issue (`--label design`). May require architectural change. | *(not yet observed)* |
| **Refuted — wrong mental model** | Prediction fails because experimenter's assumptions were wrong | Correct understanding. Document what the system actually does. | *(not yet observed)* |
| **Inconclusive — parameter-dependent** | Effect exists at some parameters but not others | Document the parameter boundary. May need recalibration. | H5 exp4: <5% effect with calibrated bucket |
| **Converged to open question** | Mechanism identified but directional explanation requires different tooling | Mark as open. Propose specific tooling needed. | *(use when remaining questions require code instrumentation, not more experiment sweeps)* |

### Choosing status vs resolution

The status answers "did the number go the way we predicted?" The resolution answers "do we understand why and what to do about it?" Always report both:

```
**Status:** Confirmed with nuance
**Resolution:** Confirmation with wrong mechanism — token-bucket reduces TTFT
via load shedding (96% rejection), not burst smoothing. No practical sweet spot
under Gamma CV=3.5.
```

A common mistake is declaring "Confirmed" and stopping. The resolution is where the real value lives.

---

## Findings Classification

Every experiment produces individual findings. Each finding MUST be classified independently of the hypothesis status:

| Finding Type | Definition | Action Required |
|-------------|------------|-----------------|
| **Confirmation** | The hypothesis holds; the system works as designed | Document in FINDINGS.md. No issues needed. |
| **Bug discovery** | The hypothesis failed due to a code defect | File GitHub issue with `--label bug`. Fix in separate PR. |
| **New rule** | The experiment revealed a pattern that should be checked in all future PRs | Add to `docs/standards/rules.md` with evidence. File issue with `--label enhancement` if code changes needed. |
| **New invariant** | The experiment revealed a property that must always hold | Add to `docs/standards/invariants.md`. |
| **Design limitation** | The system works as coded but has an undocumented behavioral limitation | Document in FINDINGS.md + file issue with `--label design` for design doc update. |
| **Surprise** | An unexpected result that doesn't fit other categories | Document in FINDINGS.md. May spawn new hypotheses. |
| **Open question** | Mechanism identified but explanation incomplete; requires different tooling to resolve | Mark explicitly in FINDINGS.md with proposed tooling/experiment. |

### The Audit Step

After analyzing results, EVERY experiment MUST audit findings against `docs/standards/`:

1. Do any findings reveal violations of existing rules or principles?
2. Do any findings suggest a new rule, invariant, or principle is needed?
3. Do any findings confirm that existing rules/invariants hold under new conditions?

This audit is what makes experiments a feedback loop into the standards. Example: H3 confirmed that the llm-d default config is robust (confirmation) AND revealed that KV utilization is stale at high rates (design limitation -> new rule R17 + new invariant INV-7 + 3 issues).

---

## Experiment Artifacts

Each hypothesis experiment lives in `hypotheses/<name>/` with:

| File | Purpose |
|------|---------|
| `run.sh` | Self-contained script: builds binary, runs all variants, calls analyzer |
| `analyze.py` | Output parser producing formatted comparison tables |
| `FINDINGS.md` | Results, root cause analysis, findings classification, standards audit |
| `*.yaml` (optional) | Custom workload specs for this experiment |

Scripts must be reproducible — running `./run.sh` on the same commit produces deterministic output.
