# BLIS Experiment Standards

Hypothesis-driven experimentation is a first-class activity in BLIS — equal in rigor to implementation and design. Experiments serve three purposes:

1. **Validation** — confirm that implemented features work as designed (e.g., prefix-affinity produces 2.4x TTFT improvement for multi-turn chat)
2. **Discovery** — surface bugs, design gaps, and undocumented limitations (e.g., H3 revealed KV utilization signal staleness -> 3 new issues, 1 new rule, 1 new invariant)
3. **Documentation** — each experiment becomes a reproducible artifact that helps users understand when to use which configuration

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

Evidence: H10 claimed "CPU tier increases total effective capacity" — but `NewKVStore` (`kv_store.go:31-36`) does not change GPU block count. The actual mechanism was `maybeOffload` stripping prefix hashes (`kvcache_tiered.go:224`).

### RCV-2: Every "surprise" must have a first-principles calculation

Before labeling a result as "surprising," compute the expected value from the system's parameters. If the result matches the calculation, it is not a surprise — it is the expected outcome of a mechanism you didn't initially consider.

Evidence: H5 labeled 96% rejection as a "surprise." But `admission.go:45` charges `len(req.InputTokens)` per request (mean=512). Token demand (1,024,000 tokens/s) exceeds supply (400 tokens/s) by 2,560x. The 96% rejection is the mathematically inevitable steady state.

### RCV-3: Check the mechanism, not just the direction

Confirming that "A is better than B" is necessary but not sufficient. The root cause analysis must explain *why* through a specific code path. A correct directional result with an incorrect explanation is a ticking time bomb — the explanation will mislead future experiments.

### RCV-4: Validate causal claims with control experiments

When a mechanism is proposed (e.g., "`maybeOffload` causes the TTFT improvement"), design a control experiment that disables **only that mechanism** (e.g., `--kv-offload-threshold 1.0`). If the effect vanishes, the mechanism is confirmed. If it persists, the explanation is wrong.

Evidence: H10 proposed `maybeOffload` as the mechanism. The control experiment (threshold=1.0) produced output byte-identical to single-tier, confirming `maybeOffload` as the sole cause. Without this control, the directional question ("why do fewer cache hits help?") would have remained unresolved.

---

## Iterative Review Protocol

Every hypothesis experiment must go through **three rounds** of experimentation interleaved with external LLM review (see `docs/process/hypothesis.md` for the full protocol). This is enforced because:

1. **Round 1** produces confident but often wrong root cause analyses
2. **Round 2** (post-review) identifies confounds, missing controls, and logical gaps
3. **Round 3** (post-review) produces definitive evidence via targeted experiments

The external review uses frontier LLMs (default: Claude Opus 4.6 via `/review-plan aws/claude-opus-4-6`) as independent technical reviewers. Their value is in catching:
- Confounding variables invisible to the experimenter (ED-1 violations)
- Logical gaps in causal chains (RCV-3 violations)
- Missing first-principles calculations (RCV-2 violations)
- Configuration assumptions that need verification (ED-6 violations)

Evidence: PR #310 required 4 rounds across H5/H10 to reach correct conclusions. The confound matrix (Round 3) was designed entirely from Opus 4.6 review feedback and produced the definitive result.

---

## Findings Classification

Every experiment produces findings. Each finding MUST be classified:

| Finding Type | Definition | Action Required |
|-------------|------------|-----------------|
| **Confirmation** | The hypothesis holds; the system works as designed | Document in FINDINGS.md. No issues needed. |
| **Bug discovery** | The hypothesis failed due to a code defect | File GitHub issue with `--label bug`. Fix in separate PR. |
| **New rule** | The experiment revealed a pattern that should be checked in all future PRs | Add to `docs/standards/rules.md` with evidence. File issue with `--label enhancement` if code changes needed. |
| **New invariant** | The experiment revealed a property that must always hold | Add to `docs/standards/invariants.md`. |
| **Design limitation** | The system works as coded but has an undocumented behavioral limitation | Document in FINDINGS.md + file issue with `--label design` for design doc update. |
| **Surprise** | An unexpected result that doesn't fit other categories | Document in FINDINGS.md. May spawn new hypotheses. |

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
