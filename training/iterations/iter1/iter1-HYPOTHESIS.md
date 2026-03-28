# Iteration 1: Additive Overhead Terms + Prefill Chunking

## Executive Summary

Iteration 0 achieved 200.544% loss with a single-crossover roofline model `max(compute, memory)`, revealing fundamental structural inadequacy. The inverted coefficients (β₀=0.308 vs expected 0.6-0.8, β₁=1.548 vs expected 0.4-0.6) indicate the optimizer is compensating for missing additive terms by distorting the prefill/decode relationship.

Iteration 1 hypothesis: **Real vLLM execution has additive overhead terms beyond the max() bottleneck**. The model structure must change from `max(compute, memory)` to `max(compute, memory) + communication + KV_mgmt + chunking + scheduler`.

This iteration adds 5 new basis functions (bringing total from 3 to 8 Beta terms) to capture: TP communication, KV cache management, prefill chunking overhead, decode batch-size regime, and MoE gating overhead.

---

## H-main: Additive Overhead Mechanism

**Prediction**: Overall loss will decrease from 200.544% (iter0) to **<80%**, with:
- TTFT RMSE reducing from 111.07% to **<40%** (64% reduction)
- E2E RMSE reducing from 89.47% to **<40%** (55% reduction)

Target breakdown:
- ttft_rmse < 40% (currently 111.07%)
- e2e_rmse < 40% (currently 89.47%)
- overall_loss = ttft_rmse + e2e_rmse < 80% (currently 200.54%)

**Causal Mechanism**:

vLLM step execution time has additive overhead components that cannot be expressed by `max(compute_time, memory_time)` alone:

1. **TP Communication Overhead** (~5-15% of step time when TP > 1): After each transformer layer, vLLM performs ring all-reduce to synchronize activations across tensor-parallel GPUs. This overhead is additive (not max-ed with compute) and scales with `log₂(TP) × num_layers × bytes_per_layer / NVLink_BW`. Iter0 has no TP term, causing systematic underprediction for TP=2 and TP=4 experiments.

2. **Prefill Chunking Overhead** (~10-20% of prefill time for long sequences): vLLM splits prefills longer than 2048 tokens into chunks, introducing per-chunk boundary overhead: kernel launch (~50μs), partial KV cache write, scheduler re-entry. Iter0 computes total FLOPs but ignores `num_chunks × chunk_overhead`. This explains why codegen experiments (long prompts → more chunks) have highest TTFT errors: 181%, 109%, 96%, 63%.

3. **KV Cache Management Overhead** (~10-50μs per request): vLLM's PagedAttention allocates/deallocates KV blocks per request. Iter0 has no per-request overhead term in StepTime (only in QueueingTime via α₀). This explains why roleplay experiments (long contexts → many KV blocks) have catastrophic errors: 269.6%, 237.6%.

4. **Decode Batch-Size Regime** (affects 87.5% ITL error): Iter0 treats decode as uniformly memory-bound. But at typical vLLM batch sizes (16-32 requests), decode becomes compute-bound due to tensor core utilization. The optimizer set β₁=1.548 (2.6× expected) trying to compensate. Splitting decode into two regimes (small-batch memory-bound vs large-batch compute-bound) allows proper modeling.

5. **MoE Gating Overhead** (~1-5% for MoE models): Iter0's MoE formula accounts for sparse FLOPs and unique expert bandwidth, but NOT gating network compute (routing probability for all experts). This explains why Scout experiments have 12.8% systematic overhead above dense models.

**Code Citations**:

- **vLLM chunking**: `vllm/worker/model_runner.py:_prepare_model_input()` splits prefill into `max_tokens_per_chunk=2048` chunks
- **TP communication**: `vllm/model_executor/layers/linear.py:ColumnParallelLinear.forward()` calls `torch.distributed.all_reduce()` after each layer when TP > 1
- **KV block management**: `vllm/core/block_manager.py:BlockSpaceManager.allocate()` allocates blocks per request
- **BLIS iter0 model**: `sim/latency/evolved_model.go:StepTime()` lines 90-122 — no additive terms beyond β₂ constant

**Diagnostic Clause**:

*If this fails (loss remains > 100%), it indicates:*
- Missing terms: Scheduler overhead (β₂ currently ≈0μs, suggesting constant overhead doesn't fit variable batch compositions), activation bandwidth (weights+KV only, missing residual connections and attention outputs), kernel launch latency per operation
- Wrong regime boundaries: The batch-size split for decode (currently 8 requests) may be incorrect; actual transition could be at different batch size or context length
- Formula errors: The TP communication formula assumes ring all-reduce with `log₂(TP)` scaling, but vLLM may use different reduction topology

**Expected Outcomes**:

If the additive overhead hypothesis is correct:
1. **β₀ should rise** from 0.308 to ~0.5-0.6 (prefill efficiency returns to physical range after chunking term absorbs overhead)
2. **β₁ should drop** from 1.548 to ~0.5-0.7 (decode memory-bound term returns to physical range after β₆ captures large-batch compute)
3. **β₃ (TP communication)** should converge to ~0.8-1.2 (near-linear scaling with TP)
4. **β₅ (chunking)** should converge to ~50-200μs per chunk (kernel launch + KV write overhead)
5. **Codegen experiments** (long prompts → many chunks) should see largest TTFT improvement (from 112% mean APE to <40%)
6. **Scout experiments** (MoE) should drop from 194.3% mean to <100% after β₇ captures gating overhead

---

## H-ablation-chunking: Prefill Chunking Term Importance

**Prediction**: Removing the chunking term β₅ (reverting to iter0's 3-term model but keeping TP/KV/decode/MoE terms) will increase TTFT RMSE by >15% compared to the full 8-term model.

**Causal Mechanism**: Codegen and reasoning experiments have long prompts (>2048 tokens in many cases), causing vLLM to chunk prefills. Without the chunking term, the model will systematically underpredict TTFT for these experiments, just like iter0 did (codegen TTFT errors: 181%, 109%, 96%, 63%).

**Diagnostic Clause**: *If removing β₅ causes <5% TTFT RMSE increase, chunking overhead is negligible (<5% of total prefill time) or the chunk size assumption (2048 tokens) is wrong.*

---

## H-ablation-tp-comm: TP Communication Term Importance

**Prediction**: Removing the TP communication term β₃ (reverting to no TP modeling) will increase overall loss by >10% for TP=2 and TP=4 experiments, while TP=1 experiments remain unchanged (<2% difference).

**Causal Mechanism**: TP communication adds ~5-15% overhead per step for TP > 1 due to all-reduce after each layer. Iter0 has no TP term, causing uniform underprediction for all TP configs. With β₃ added, removing it should specifically harm TP=2/4 experiments.

**Diagnostic Clause**: *If removing β₃ harms TP=1 experiments equally, there's a confounded variable (model size, hardware) accidentally correlated with TP in the dataset.*

---

## H-ablation-kv-mgmt: KV Management Term Importance

**Prediction**: Removing the KV management term β₄ will increase E2E RMSE by >10%, with largest impact on long-context experiments (roleplay workload).

**Causal Mechanism**: KV cache block allocation/deallocation is per-request overhead. Roleplay experiments have long contexts (1000+ tokens) requiring many KV blocks, making management overhead significant. Iter0's worst outlier was Llama-2-7B roleplay (269.6% error).

**Diagnostic Clause**: *If removing β₄ causes <5% E2E RMSE increase, KV management overhead is constant (absorbed by α₀) or negligible compared to compute/memory costs.*

---

## H-boundary-decode: Decode Regime Transition Point

**Prediction**: At batch_size < 8 requests, decode will be memory-bound (β₁ dominates). At batch_size ≥ 8 requests, decode will be compute-bound (β₆ dominates). The transition should be visible as a coefficient ratio flip.

**Causal Mechanism**: Small-batch decode has insufficient parallelism for tensor core utilization → memory-bound. Large-batch decode saturates compute resources → compute-bound. The 8-request threshold is estimated from literature on attention kernel efficiency.

**Diagnostic Clause**: *If β₆ ≈ 0 after optimization, decode is uniformly memory-bound at all batch sizes, and the regime split is unnecessary. If β₁ ≈ 0, decode is uniformly compute-bound.*

---

## H-error-pattern-improvement: Per-Experiment Gains

**Prediction**: The 5 largest APE reductions (compared to iter0) will be:
1. **Llama-2-7B roleplay** (currently 269.6%) → <150% (KV management term captures long-context overhead)
2. **Llama-3.1-70B codegen** (currently 249.0%) → <140% (chunking term captures long-prompt overhead)
3. **Qwen2.5 roleplay** (currently 237.6%) → <130% (KV management + chunking)
4. **Llama-2-7B reasoning** (currently 198.8%) → <120% (TP=1 benefits from all additive terms)
5. **Scout general** (currently 199.8%) → <110% (MoE gating term captures expert overhead)

**Causal Mechanism**: These experiments have characteristics that maximize the missing terms in iter0:
- Roleplay: Long contexts → high KV management overhead
- Codegen: Long prompts → high chunking overhead
- Scout: MoE architecture → gating network overhead
- TP=1 small models: All overhead terms matter more (less amortization over large compute)

**Diagnostic Clause**: *If improvement is uniform across experiments (all ~40% reduction), the new terms are capturing global inefficiency, not experiment-specific overhead. Check for formula errors or incorrect feature extraction.*

---

## H-robustness-moe: MoE Generalization

**Prediction**: After adding β₇ (MoE gating overhead), Scout experiments will have mean APE within 5% of dense model mean (currently 12.8% gap).

**Causal Mechanism**: Iter0's 12.8% MoE overhead is explained by gating network compute not captured in sparse FLOPs formula. Adding β₇ should close this gap.

**Diagnostic Clause**: *If gap remains > 8% after optimization, MoE has additional overhead beyond gating (e.g., expert context switching, load balancing auxiliary loss) that needs separate term.*

---

## Summary of Hypotheses

| Hypothesis | Predicted Outcome | Key Metric | Diagnostic Signal |
|------------|------------------|------------|-------------------|
| **H-main** | Loss < 80% (from 200.54%) | overall_loss, ttft_rmse, e2e_rmse | If loss > 100%, missing terms or wrong formulas |
| **H-ablation-chunking** | TTFT RMSE +15% without β₅ | ttft_rmse | If <5%, chunking negligible |
| **H-ablation-tp-comm** | Overall loss +10% for TP>1 without β₃ | per-experiment APE (TP=2/4 vs TP=1) | If TP=1 harmed equally, confounded variable |
| **H-ablation-kv-mgmt** | E2E RMSE +10% without β₄ | e2e_rmse, roleplay experiments | If <5%, KV mgmt absorbed by α₀ |
| **H-boundary-decode** | β₁ vs β₆ regime flip at batch_size=8 | coefficient values, per-batch-size APE | If one coefficient ≈0, uniform regime |
| **H-error-pattern** | Top 5 experiments improve most | per-experiment APE deltas | If uniform, global not specific overhead |
| **H-robustness-moe** | Scout APE within 5% of dense mean | Scout vs dense model mean APE | If gap > 8%, additional MoE overhead |

---

## Iteration 1 Success Criteria

**Primary**: Overall loss < 80% (ttft_rmse < 40%, e2e_rmse < 40%)

**Secondary**:
- β₀ rises from 0.308 to 0.5-0.6 range (prefill efficiency physical)
- β₁ drops from 1.548 to 0.5-0.7 range (decode memory-bound physical)
- Codegen mean TTFT APE drops from 112% to <40%
- Roleplay mean E2E APE drops from ~170% to <80%
- Scout mean APE within 5% of dense model mean

**If < 80% but ≥ 50%**: Partial success, proceed to iter2 with refinements

**If ≥ 80%**: Structural failure, reconsider basis function formulas or add new terms
