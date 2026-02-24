# CPU Overhead Formula Refinements (Post-H2b)

**Status:** Research ideas — not yet formalized as hypotheses
**Context:** H2b (`100μs × layers / tp`) reduced TPOT MAPE from 36.1% to 17.3%, but one experiment worsens (7B TP=1 chatsweep: +8.2pp) and 70B codesweep undershoots. These ideas explore no-training refinements to the overhead formula.
**Constraint:** Tier 0 only — all parameters must come from `config.json` or hardware specs, no fitting to ground truth.

---

## Where the remaining error budget sits

After H1 (BW efficiency) + H2b (model-scaled overhead), the residual errors break into regimes:

| Regime | Signed error | Magnitude | Fixable by overhead? |
|--------|-------------|-----------|---------------------|
| 7B TP=4 | -37% to -48% | Huge | **No** — step time ~1ms, overhead 0.8ms. Compute model error (H3/H4 territory) |
| 7B TP=1 chat | +30.5% | Large | **Yes** — 3.2ms overshoots |
| 70B codesweep | -21.2% | Medium | **Maybe** — fixed 5ms did better (3.9%), wants more than 2ms |
| 7B TP=1 code | +15.2% | Medium | **Yes** — same overshoot as chatsweep |
| Everything else | -10% to +10% | Small | Diminishing returns |

The fundamental tension: **7B TP=1 needs less overhead (overshoots), 70B TP=4 needs more (undershoots)**. Since `layers/tp` is 32 for 7B-TP1 and 20 for 70B-TP4, any single-term formula proportional to `layers/tp` cannot move them in opposite directions.

---

## Idea A: Two-component model (fixed base + per-layer scaling)

**Claim:** The overhead has a model-independent floor (Python interpreter frame, CUDA driver sync, output sampling) plus a model-dependent component (block table updates, tensor dispatch).

**Formula:** `overhead_us = framework_base + per_layer × layers / tp`

**Reasoning:** Not all scheduler work scales with layers. The `torch.cuda.synchronize()`, Python `asyncio` event loop tick, and `sample()` call at batch=1 are constants. Decomposing into two components lets the per-layer rate be smaller, reducing overshoot for small models while the base lifts the floor for large models.

**The problem:** Even this faces the same tension. For example, `800 + 75 × layers/tp` gives 3200μs for 7B-TP1 (same as current) and 2300μs for 70B-TP4 (slightly better). To significantly reduce 7B-TP1, you need a smaller per_layer constant, which makes 70B even worse. The two parameters trade off against each other on the `layers/tp` axis.

**Verdict:** Modest improvement potential. Helps at the margin but doesn't resolve the fundamental tension because both terms still collapse to a single `layers/tp` dimension.

---

## Idea B: Sublinear scaling — `sqrt(layers / tp)`

**Claim:** Block table updates use batched tensor operations, not per-layer Python loops. The actual CPU cost scales sublinearly with layer count due to CPU cache effects and batched CUDA memcpy.

**Formula:** `overhead_us = base × sqrt(layers / tp)`

**Example values (base = 700):**

| Config | layers/tp | Linear (100×) | Sqrt (700×√) | Direction needed |
|--------|-----------|---------------|-------------|-----------------|
| 7B TP=1 | 32 | 3200μs | 3960μs | Lower |
| 7B TP=4 | 8 | 800μs | 1980μs | N/A (compute error) |
| 34B TP=2 | 24 | 2400μs | 3430μs | ~OK |
| 70B TP=4 | 20 | 2000μs | 3130μs | Higher |

**The problem:** Sqrt *increases* the 7B-TP1 value (3960 vs 3200), making the overshoot worse. It compresses the range from both ends — lifts the low end (good for 70B) but also lifts the high end (bad for 7B-TP1). Would need a lower base to compensate, which then undershoots 70B again.

**Verdict:** Wrong direction for the primary failure mode. Might work as part of a two-component model (Idea A + B combined) but not standalone.

---

## Idea C: Vocab-size additive component (MOST PROMISING)

**Claim:** Output logit sampling is O(vocab_size) per step. Qwen models (152K vocab) incur ~4.75× more sampling overhead than Llama models (32K vocab). This cost is independent of `layers/tp` and is currently unmodeled.

**Formula:** `overhead_us = per_layer × layers / tp + vocab_factor × vocab_size / 32000`

**Why this is the most promising:**

1. **Orthogonal to layers/tp.** Vocab size varies independently of layer count and TP. This is the only proposed axis that can break the 7B-TP1 vs 70B-TP4 tension, because it adds a dimension to the model.

2. **Directly testable from existing ground truth.** The 13 experiments include:
   - Llama2-7B: 32K vocab, 32 layers
   - Qwen2.5-7B: 152K vocab, 28 layers
   - Qwen3-14B: 152K vocab, 40 layers
   - CodeLlama-34B: 32K vocab, 48 layers
   - Llama2-70B: 32K vocab, 80 layers

   Compare H2b residuals for Qwen vs Llama at similar layer counts. If Qwen consistently underpredicts more than Llama after H2b, vocab is the missing factor.

3. **Meaningful magnitude.** At batch=1, vLLM runs `Sampler.forward()` which includes top-p/top-k filtering over the full logit vector. For 152K vocab on CPU, this is 300-800μs — not negligible relative to the 2-4ms overhead range. For 32K vocab, it's ~60-170μs.

4. **Fully Tier 0.** `vocab_size` is in `config.json`, no training needed.

**Quick validation check:** From H2b signed TPOT errors:
- Qwen2.5-7B TP=1 (152K vocab, 28 layers): +9.1% → mild overprediction
- Llama2-7B TP=1 (32K vocab, 32 layers): +30.5% / +15.2% → strong overprediction
- Qwen3-14B TP=1 (152K vocab, 40 layers): +5.0% → mild overprediction

The Qwen models are *less* overpredicted than Llama, even though they have more layers (40 vs 32 for Qwen3 vs Llama2). This is the **opposite** of what a layers-only model predicts. It's consistent with Qwen needing a larger overhead (from vocab) that partially compensates for the per-layer overshoot. This is suggestive — not definitive — evidence that vocab size is a missing factor.

**Implementation:** Add `vocab_factor × vocab_size / 32000` to the overhead. With `vocab_factor ≈ 150μs` (estimated sampling cost at 32K baseline), a 152K-vocab model gets an extra `150 × 152/32 = 712μs`. This could also let the per_layer constant drop from 100μs to ~80μs, reducing the 7B-TP1 overshoot.

---

## Idea D: KV-head-aware scaling

**Claim:** Per-layer CPU work depends on `num_kv_heads / tp` — the number of KV head groups managed per GPU. GQA models (8 KV heads) have less per-layer bookkeeping than MHA models (32 heads).

**Formula:** `overhead_us = base × layers × max(1, kv_heads / tp) / reference_factor`

**Reasoning:** Block table management, cache slot tracking, and attention metadata preparation scale with the number of distinct KV head groups. GQA models should show smaller per-layer residuals than MHA models.

**The problem:** vLLM's block table is per-sequence-per-layer, not per-head. Each layer has one block table row regardless of KV head count. The KV head count affects the *size* of each cache block (bytes), not the *number* of block table entries. So the CPU scheduling cost may not actually vary with KV heads — the GPU memory traffic does, but that's already in `stepHardwareS`.

**Testable:** Compare H2b residuals for MHA models (Llama2-7B: 32 KV heads) vs GQA models (CodeLlama-34B: 8 KV heads, Llama2-70B: 8 KV heads) at similar `layers/tp` values. But we don't have any pair with the same `layers/tp` but different KV head counts, so it's confounded.

**Verdict:** Theoretically appealing but likely wrong about the mechanism. Block tables don't scale with KV heads. Low testability from existing data.

---

## Idea E: Separate prefill/decode scaling formulas

**Claim:** Prefill overhead scales with `prompt_tokens / chunk_size` (chunked-prefill planning), not `layers / tp`. Decode overhead scales with `layers / tp` (block table updates). Using the same `layers/tp` basis for both is wrong.

**Formula:**
- `decode_overhead_us = per_layer_decode × layers / tp`
- `prefill_overhead_us = base_prefill + per_chunk × ceil(prompt_tokens / chunk_size)`

**Reasoning:** Prefill scheduling involves fundamentally different CPU work than decode scheduling:
- Chunked prefill planning: how many tokens to process this step → O(prompt_tokens / chunk_size)
- Initial block allocation: allocate blocks for the full input → O(input_tokens / block_size)
- Prompt prefix matching: hash computation → O(prefix_length)

None of these scale with `layers`. The current formula (`500 × layers / tp` for prefill) is using layers as a proxy for model complexity, but the actual cost driver is prompt length.

**Testable:** Compare E2E improvement for chatsweep (prompt=70, decode-heavy) vs codesweep (prompt=2048, prefill-heavy). If prefill overhead formula is wrong, codesweep E2E should show a different error pattern.

From H2b E2E results:
- Chatsweep E2E improvement: variable, +4.6pp to +39.2pp
- Codesweep E2E improvement: also variable, -28.8pp to +38.0pp

The codesweep E2E is indeed more erratic — some experiments get much worse (7B TP=1: -28.8pp E2E). This is consistent with the prefill overhead formula being wrong for long-prompt workloads.

**Verdict:** Moderately promising. The `prompt_tokens / chunk_size` scaling is principled. But `chunk_size` is a vLLM config parameter (default 512) that BLIS doesn't currently model — adding it introduces a new dependency.

---

## Recommended priority

1. **Idea C (vocab size)** — most promising. Orthogonal axis, testable from existing data, meaningful magnitude, fully Tier 0. Quick experiment: add `vocab_factor × vocab_size / 32000` to the H2b formula and check if it reduces the Llama-vs-Qwen residual asymmetry.

2. **Idea A+C combined** — two-component model with vocab. `overhead = base + per_layer × layers / tp + vocab_factor × vocab_size / 32000`. Three analytically-derivable constants, zero training. This is the most expressive Tier 0 formula that uses only `config.json` parameters.

3. **Idea E (separate prefill)** — worth investigating if E2E MAPE remains high after C. The codesweep E2E erraticism suggests the prefill formula needs work.

4. **Ideas B and D** — lower priority. Sqrt scaling is wrong direction; KV-head scaling has a weak mechanism.
