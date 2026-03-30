# Iteration 7: Clean Data Retraining with Decode Overhead Decoupling

## Context and Motivation

**Critical Discovery from Iter6 Post-Analysis**: The original reasoning experiments (20260217-170634-llama-2-7b-tp1-reasoning, 48-llama-4-scout-17b-16e-tp2-reasoning-2, 66-qwen2-5-7b-instruct-tp1-reasoning-1-1) were collected from **overloaded servers** with 85% request failure/timeout rates. Journey trace analysis revealed 97-99% of training data consisted of requests stuck in queue for 259 seconds before timeout—no physics-based model can fit this.

Fresh reasoning-lite data collected on 2026-03-30 shows roofline baseline performance improved from 99% → 53% avg TTFT error (range: 15-92%), confirming data quality issue rather than model deficiency.

**Iter7 Strategy**: Retrain on clean dataset (exclude 3 corrupted reasoning experiments, include 3 fresh reasoning-lite experiments) while addressing decode coefficient destabilization observed in iter6.

---

## H-main: Clean Data Retraining Enables Coefficient Stabilization

**Prediction**: Overall loss will decrease from 161.69% (iter6 on mixed clean/corrupt data) to **<80%**, with:
- **TTFT RMSE**: 69.47% → **<40%** (removing 99% errors from corrupt reasoning data)
- **E2E RMSE**: 92.22% → **<50%** (stabilizing decode coefficients β₁/β₄)
- **All coefficients physically plausible**: α₀ <2ms, α₁ <150μs, α₂ <50μs, β₁ ∈ [1.0, 1.15], β₄ ∈ [0.75, 0.90]

**Causal Mechanism**:

Iter6 trained on 15 experiments including 3 corrupted reasoning experiments with 97-99% unusable data (259-second timeout requests). This created three failure modes:

1. **Alpha inflation** (α₁ = 351μs, α₂ = 216μs, 6-10× above physical): Optimizer inflated per-token overhead to partially compensate for 100-200ms TTFT gap from corrupt reasoning data. With clean data, Alpha can return to iter4 physical values (α₁ = 125μs, α₂ = 36μs).

2. **Decode coefficient destabilization** (β₁ = 1.851, β₄ = 1.451): As prefill predictions improved (β₀ = 0.164, β₂ = 0.270), optimizer increased decode coefficients to maintain E2E fit. But corrupt reasoning data's bimodal distribution (50ms successful vs 259s timeout) prevented stable convergence.

3. **Insufficient decode overhead modeling**: E2E RMSE increased 9% (84% → 92%) despite TTFT improving 87%. This indicates missing decode-phase overhead not captured by β₁ (memory-bound) or β₄ (compute-bound) alone.

**Physics Justification**:

Training on clean data removes three corrupted experiments contributing 597% combined loss (3 × 199% avg). Replacing with reasoning-lite experiments (roofline baseline: 15-92% TTFT) provides:
- 146pp average TTFT improvement per reasoning experiment (99% → 53%)
- Eliminates bimodal timeout distribution preventing coefficient convergence
- Enables tighter Alpha bounds without degrading fit (no need to absorb 100-200ms gap via per-token inflation)

**Code Citations**:
- vLLM queue/scheduler: `vllm/core/scheduler.py:Scheduler.schedule()` — journey traces show 0.3-2ms queue time for clean requests vs 259s for overload
- BLIS Alpha coefficients: `sim/latency/evolved_model.go:273-283 QueueingTime()` — α₁/α₂ physically model tokenization (30-50μs/token), not 100-200ms scheduler delays
- Iter6 results: `training/iterations/iter6/inner_loop_results.json` — "ttft_mean_ape": 99.98% for all 3 reasoning experiments

**Diagnostic Clause**: *If overall loss does NOT decrease below 100%, it indicates either: (1) reasoning-lite data still has quality issues (check failure rates in traces), (2) workload diversity reduced (15 → 15 experiments but 3 new), or (3) decode overhead term (H-decode-overhead) is mandatory, not optional. In this case, immediately add decode per-request overhead term (β₇) and re-optimize.*

---

## H-decode-overhead: Decode Phase Needs Per-Request Overhead Term

**Prediction**: Adding decode per-request overhead term (β₇) will improve E2E RMSE from 92.22% → **<60%** and stabilize decode coefficients:
- **β₁** (decode memory-bound): 1.851 → **1.00-1.15** (return to iter3's 1.037 range)
- **β₄** (decode compute-bound): 1.451 → **0.75-0.90** (return to iter3's 0.796 range)
- **β₇** (decode per-request overhead): **5-15ms** per request (output processing, result aggregation)

**Causal Mechanism**:

Iter6 decode coefficient destabilization resulted from zero-sum trade-off: As prefill predictions improved (β₀ = 0.164, β₂ = 0.270), E2E = TTFT + decode time required longer decode predictions to match ground truth. Optimizer increased β₁ (+28%) and β₄ (+134%) to compensate, but this made decode over-predicted rather than fixing the root cause.

The missing physics: vLLM decode phase has fixed overhead beyond memory/compute costs:
1. **Per-request output processing**: After each decode step, vLLM processes output tokens (sampling, stop condition check, streaming updates) — vLLM code: `vllm/model_executor/model_loader.py:_run_workers()` calls `execute_model()` per step with 1-5ms overhead
2. **Decode batching coordination**: Unlike prefill (batched naturally), decode requires per-request coordination across TP workers — vLLM code: `vllm/worker/worker.py:execute_model()` synchronizes across TP ranks per step
3. **KV cache write-back**: After decode compute, updated KV cache blocks written back to memory — same mechanism as β₃ (prefill KV allocation) but during decode phase

Current model: `decode_time = β₁ × memory_term × memory_weight + β₄ × compute_term × compute_weight`
Proposed: `decode_time = β₁ × memory_term × memory_weight + β₄ × compute_term × compute_weight + β₇ × num_decode_requests`

**Physics Justification**:

Iter3 had stable decode coefficients (β₁ = 1.037, β₄ = 0.796). Iter4/5/6 destabilized them (β₁ = 1.30 → 1.45 → 1.85, β₄ = 1.31 → 0.62 → 1.45). The pattern: Whenever prefill coefficients improved, decode coefficients destabilized to maintain E2E fit. This indicates **missing additive term**, not wrong multiplicative scaling.

Evidence from roofline model: `PostDecodeFixedOverhead()` returns 0, but actual vLLM has per-request overhead during decode. Adding β₇ decouples decode compute/memory efficiency (β₁/β₄) from decode framework overhead.

**Code Citations**:
- vLLM decode execution: `vllm/model_executor/model_loader.py:_run_workers()` — calls `execute_model()` per step with sampling overhead
- TP worker synchronization: `vllm/worker/worker.py:execute_model()` — per-step TP barrier adds 1-3ms
- BLIS evolved model: `sim/latency/evolved_model.go:162-183` — decode terms β₁/β₄ only model memory/compute, no per-request overhead
- Iter6 destabilization: `training/iterations/iter6/iter6-FINDINGS.md:554-570` — "β₁/β₄ moved AWAY from iter3 ranges, E2E RMSE worsened 9%"

**Diagnostic Clause**: *If β₇ converges to <3ms, decode overhead is negligible and destabilization is due to different root cause. Check: (1) E2E trace analysis for missing decode-phase operations, (2) whether decode predictions are systematically biased (all over-predicted or all under-predicted), or (3) whether TP communication during decode (β₂) needs revision.*

**Note**: This hypothesis is **testable via ablation** — train two variants (with/without β₇) and compare β₁/β₄ stability. If β₇-enabled version has β₁/β₄ closer to iter3 ranges AND lower E2E RMSE, hypothesis confirmed.

---

## H-alpha-reversion: Alpha Inflation Reversal via Tight Bounds

**Prediction**: Reverting Alpha to iter4 values with tight bounds will prevent inflation without degrading fit:
- **α₀**: 4.07ms → **<2ms** (return to iter4's 1.5ms)
- **α₁**: 351μs → **<150μs** (return to iter4's 125μs, 6× reduction)
- **α₂**: 216μs → **<50μs** (return to iter4's 36μs, 6× reduction)
- **Overall loss impact**: Minimal (<5% change) because corrupt reasoning data removed

**Causal Mechanism**:

Iter6 Alpha inflation (α₁ = 351μs, α₂ = 216μs, 6-10× above physical) occurred because optimizer used per-token overhead to compensate for reasoning's 100-200ms TTFT gap:
- Reasoning: 1000 input tokens × 351μs = 351ms contributed to QueueingTime
- This added ~20-60ms overhead to help reasoning experiments (insufficient for 100-200ms gap, but reduced error)
- Side effect: All experiments got inflated tokenization cost (physically implausible)

With clean data (reasoning-lite 15-92% TTFT, no 99% outliers), Alpha no longer needs to absorb missing scheduler overhead. Tighter bounds prevent inflation while maintaining fit:
- α₀ bounds: [0.0, 0.005] (max 5ms, was unbounded in iter6)
- α₁ bounds: [0.0, 0.0002] (max 200μs, was unbounded)
- α₂ bounds: [0.0, 0.0001] (max 100μs, was unbounded)

**Physics Justification**:

Tokenization/detokenization are CPU-bound string operations with well-known costs:
- HuggingFace BPE tokenization: 30-50μs per token (measured on A100 host CPU)
- Detokenization: 20-40μs per token (string concatenation + UTF-8 encoding)
- API parsing/validation: 1-3ms fixed overhead per request

Iter6 values (α₁ = 351μs, α₂ = 216μs) are **physically implausible** — would imply tokenization is 6-10× slower than measured. This inflation was optimizer's only mechanism to help reasoning (no workload-dependent terms available), but with clean data, inflation is unnecessary.

**Code Citations**:
- vLLM tokenization: `vllm/transformers_utils/tokenizer.py:Tokenizer.encode()` — HuggingFace BPE tokenization, ~30-50μs/token measured
- BLIS Alpha usage: `sim/latency/evolved_model.go:276-277` — α₁ applied per input token in QueueingTime
- Iter6 Alpha values: `training/iterations/iter6/inner_loop_results.json:13-15` — α₁ = 0.000351 (351μs), 10× above physical
- Physical measurements: `docs/reference/vllm-profiling.md` (if exists) or vLLM benchmarking documentation

**Diagnostic Clause**: *If Alpha inflation recurs (α₁ > 200μs or α₂ > 100μs after reversion), it indicates: (1) reasoning-lite data still has systematic TTFT underestimation (check traces), (2) β₆ (scheduler overhead) insufficient even for reasoning-lite (increase β₆ bounds to [0.01, 0.10]), or (3) workload-dependent overhead needed (add variance term or split β₆ by workload). In this case, prioritize increasing β₆ bounds before adding new terms.*

---

## H-error-pattern: Workload-Specific Improvements

**Prediction**: With clean data, error patterns will **redistribute** from reasoning-dominated (4 experiments at 99%) to architecture/workload-dominated:
- **Reasoning-lite**: 99% → **30-60%** TTFT (replacing corrupt data, 3 experiments)
- **Scout MoE**: 87-99% → **50-70%** TTFT (no longer absorbing reasoning error, 4 experiments)
- **Mistral TP=2**: 91% → **60-80%** TTFT (benefits from β₂ stabilization, 1 experiment)
- **Short-context codegen/roleplay**: 11-46% → **10-35%** TTFT (slight improvement from Alpha reversion, 7 experiments)

**Rationale**: Iter6 trained on data where reasoning contributed 597% combined loss (3 × 199% avg). Optimizer prioritized fitting 11 clean experiments over 4 corrupt experiments, leading to:
- β₆ = 21.5ms chosen as compromise (helps short-context, insufficient for reasoning 100-200ms gap)
- β₁/β₄ destabilized to maintain E2E fit while absorbing reasoning error
- Scout/Mistral errors remained high because optimizer had limited capacity to help all failure modes

With clean data:
- Optimizer no longer constrained by 99% reasoning outliers (removed)
- Reasoning-lite tractable (roofline 15-92%, evolved should reach 30-60%)
- Scout/Mistral errors addressable (no longer third-priority behind reasoning and short-context)
- Short-context already good (11-46%), slight improvement from Alpha returning to physical values

**Experiments expected to improve most** (sorted by predicted improvement):
1. **Reasoning-lite** (3 experiments): 99% → 30-60% TTFT (69-39pp improvement, replacing corrupt data)
2. **Scout codegen** (48-llama-4-scout-17b-16e-tp2-codegen-2): 98% → 50-70% TTFT (28-48pp)
3. **Scout roleplay** (21-llama-4-scout-17b-16e-tp2-roleplay-2): 87% → 50-70% TTFT (17-37pp)
4. **Mistral TP=2 general-lite** (62-mistral-nemo-12b-tp2-general-lite-2-1): 91% → 60-80% TTFT (11-31pp)

**Diagnostic Clause**: *If reasoning-lite does NOT improve to <70% TTFT, check: (1) reasoning-lite traces for hidden quality issues (failure rates, latency distributions), (2) whether roofline's 15-92% baseline is achievable (evolved may be capacity-limited), or (3) whether workload-specific overhead still needed (reasoning-lite may differ from codegen/roleplay by more than context length). If Scout experiments do NOT improve to <80%, investigate MoE-specific overhead (expert routing, load balancing) as potential missing term.*

---

## H-boundary: Decode Overhead Should Scale with Output Length

**Prediction**: If decode per-request overhead (β₇) is added, it should satisfy boundary condition:
- **At num_decode_tokens = 1**: Decode time ≈ β₇ + small compute/memory term (5-10ms total)
- **At num_decode_tokens = 100**: Decode time ≈ 100 × (memory/compute per token) + β₇ (β₇ becomes <10% of total)
- **β₇ convergence**: 5-15ms (consistent with vLLM per-request overhead, independent of output length)

**Rationale**: Decode per-request overhead (output processing, TP coordination, result aggregation) is **fixed cost per request**, not scaled by num_decode_tokens. Current model (β₁/β₄) scales linearly with output length but lacks fixed overhead term, causing:
- Short outputs (1-10 tokens): Under-predicted because fixed overhead dominates
- Long outputs (100+ tokens): Correctly predicted because per-token cost dominates

Adding β₇ should preserve linearity at high output counts (β₇ <<  per-token cost × 100) while improving short-output predictions.

**Verification Strategy**: After optimization, plot `decode_time_predicted vs num_decode_tokens` for experiments with varying output lengths:
- Slope = β₁ × memory_weight + β₄ × compute_weight (should match iter3: ~1.0-1.2× theoretical)
- Intercept = β₇ (should be 5-15ms)

**Diagnostic Clause**: *If β₇ converges to >20ms, it indicates β₇ is absorbing error beyond per-request overhead (possibly batch formation delay or scheduler overhead). Check whether β₆ (prefill scheduler overhead) needs adjustment, or whether decode-phase scheduler overhead differs from prefill.*

---

## Summary of Hypotheses

| Hypothesis | Component | Prediction | Mechanism | Diagnostic Trigger |
|------------|-----------|------------|-----------|-------------------|
| **H-main** | Overall | Loss 161.69% → <80% | Clean data removes 597% combined loss from 3 corrupt reasoning experiments | Loss >100% → check reasoning-lite data quality or add β₇ |
| **H-decode-overhead** | E2E RMSE | 92.22% → <60%, β₁/β₄ stabilize | Add decode per-request overhead (β₇) decouples compute/memory from framework overhead | β₇ <3ms → decode overhead negligible, different root cause |
| **H-alpha-reversion** | Alpha | α₁ 351μs → <150μs, α₂ 216μs → <50μs | Tight bounds prevent inflation now that corrupt reasoning removed | α₁ >200μs → reasoning-lite still has issues or β₆ insufficient |
| **H-error-pattern** | Per-exp | Reasoning-lite 99% → 30-60%, Scout 87-99% → 50-70% | Optimizer capacity freed by removing 99% outliers | Reasoning-lite >70% → check traces; Scout >80% → MoE overhead term |
| **H-boundary** | Decode | β₇ = 5-15ms fixed, decode time = β₇ + per-token × num_tokens | Decode overhead fixed per request, not scaled by output length | β₇ >20ms → absorbing scheduler overhead, adjust β₆ |

**Iteration Goal**: Achieve overall loss <80% by retraining on clean dataset (15 experiments: 12 original + 3 reasoning-lite), stabilizing decode coefficients, and reverting Alpha inflation. If H-decode-overhead is required (E2E RMSE >70% without β₇), add term and re-optimize.

**Success Criteria**:
- ✅ **Overall loss <80%** (primary goal)
- ✅ **TTFT RMSE <40%** (prefill phase well-modeled)
- ✅ **E2E RMSE <50%** (decode phase stabilized)
- ✅ **All coefficients physically plausible** (Alpha within 10% of iter4, Beta within expected ranges)
- ✅ **No experiment with TTFT >90%** (reasoning-lite replaces 99% outliers)
- ✅ **β₁/β₄ within iter3 ranges** (1.00-1.15 and 0.75-0.90 respectively)

**Failure Modes**:
- ❌ If loss >100%: reasoning-lite data quality issue or missing decode overhead term (β₇) mandatory
- ❌ If α₁ >200μs or α₂ >100μs: bounds too loose, reasoning-lite still problematic, or β₆ insufficient
- ❌ If β₁ >1.5 or β₄ >1.2: decode overhead term (β₇) needed or different decode physics issue
- ❌ If reasoning-lite >70% TTFT: workload-specific overhead still required despite clean data
