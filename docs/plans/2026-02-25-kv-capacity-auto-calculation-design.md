# Design: Auto-Calculate `total_kv_blocks` from Model and Hardware Config in Roofline Mode

**Date:** 2026-02-25
**Status:** Draft
**Issue:** #432
**Species:** Specification
**Extension Type:** Configuration enhancement — below the extension threshold (no new interfaces, no new module boundaries, no new events). Does not map to the four canonical extension types (policy template, subsystem module, backend swap, tier composition) because this is a pre-simulation configuration derivation, not a simulation-time module.

## 1. Motivation

When running BLIS in roofline mode, the user has already provided everything needed to derive the correct KV cache capacity: a HuggingFace `config.json` (model architecture) and `hardware_config.json` (GPU specs). Yet BLIS still requires a manual `--total-kv-blocks` flag or falls back to a hardcoded 1,000,000-block default — an unrealistically large value that suppresses all KV eviction and memory pressure effects, producing misleading simulation results.

The llm-d-benchmark [capacity_planner.py](https://github.com/llm-d/llm-d-benchmark/blob/e724935c533d468b7ab02c542ed2047790f91d6b/config_explorer/src/config_explorer/capacity_planner.py#L596) already computes this value. BLIS must match it for consistency with the llm-d ecosystem.

## 2. Scope

### In scope
- Add GPU memory capacity (in GiB) to the hardware calibration data
- Implement the KV block capacity calculation matching the reference
- Integrate auto-calculation into the roofline mode path in the CLI
- Validate against empirical `defaults.yaml` values

### Explicitly out of scope
- Changes to blackbox (alpha/beta coefficient) mode
- Pipeline parallelism (BLIS does not model PP; hardcode `pp=1`)
- Data parallelism (BLIS does not model DP; hardcode `dp=1`)
- Dynamic KV cache resizing during simulation
- CUDA graph memory estimation (the reference returns 0; already included in activation profiling)
- Heterogeneous GPU deployments — all instances share the same simulation configuration, including auto-calculated KV blocks. Mixed GPU types or TP configurations within a cluster are not modeled.

### Deferred
- Quantized-weight-aware model memory estimation (see Decision 5)
- User-configurable activation memory constants
- MLA (Multi-Latent Attention) KV format (e.g., DeepSeek v3)
- Capacity-aware admission control (currently `AlwaysAdmit`/`TokenBucket` do not examine KV state; realistic KV capacity makes this relevant — see Extension Points)
- CPU tier auto-sizing (`--kv-cpu-blocks` remains manual; auto-calculation applies to GPU tier only)
- KV cache dtype decoupling from model dtype — a `--kv-cache-dtype` flag to support FP8 KV cache deployments (see Decision 8, Extension Points)

## 3. Modeling Decisions

| Aspect | Decision | Justification |
|--------|----------|---------------|
| **GPU memory utilization** | Fixed at 0.9 (90%) | Matches reference `gpu_mem_util=0.9` default; represents vLLM's standard reservation |
| **Activation memory** | Fixed constants: 5.5 GiB (dense), 8.0 GiB (MoE) | Matches reference `ACTIVATION_MEMORY_BASE_DENSE_GIB` / `MOE` constants; empirical vLLM measurements |
| **Non-torch overhead** | 0.15 GiB (TP=1), 0.6 GiB (TP>=2) | Matches reference; accounts for NCCL buffers and CUDA context |
| **Block size** | Uses `--block-size-in-tokens` CLI flag (default: 16) | Consistent with existing BLIS config and vLLM default |
| **Pipeline parallelism** | Hardcoded `pp=1` | BLIS does not model PP |
| **CUDA graph memory** | Omitted (0.0 GiB) | Reference returns 0 — already captured in activation profiling |
| **Model weight bytes** | Computed from architecture parameters (layers + embeddings + lm_head), with MoE expert multiplication and `tie_word_embeddings` handling | Approximates reference `model_params_by_dtype()`; see Decisions 1, 9, 10 |
| **KV cache precision** | Uses `precision_bytes` derived from `torch_dtype` (e.g., 2 for bfloat16) | Correct for standard models; known limitation for quantized models (see Decision 5) |
| **KV cache dtype** | Same as model `torch_dtype` (e.g., 2 bytes for bfloat16) | Matches reference `inference_dtype_byte()` for standard models; see Decision 8 |
| **MoE detection** | Presence of standard HF MoE indicator fields in config | Covers known MoE architectures; see Decision 3 for exact field list |
| **MoE weight scaling** | MLP weights multiplied by `num_local_experts` for MoE models | Required because BLIS computes from architecture, not safetensors metadata; see Decision 9 |
| **Tied embeddings** | When `tie_word_embeddings: true`, `lm_head` weight is omitted (shared with embedding) | Prevents double-counting shared weight tensors; see Decision 10 |
| **Activation memory TP-scaling** | Activation memory is per-replica constant (not multiplied by TP) | Matches reference behavior: activation memory is a per-DP-replica constant; since BLIS hardcodes `dp=1`, the value is used directly regardless of TP |
| **SwiGLU activation assumption** | MLP weight formula uses 3 matrices (gate + up + down) | All validation targets use SwiGLU; non-SwiGLU models detected and rejected via BC-19 |
| **`max_model_len` interaction** | Not modeled — raw block count computed (matching reference) | vLLM's `max_model_len` can reduce usable blocks; BLIS computes hardware-level maximum. The reference `capacity_planner.py` also ignores this. |

## 4. Invariants

- **KV-CAP-1 (Override precedence):** The precedence chain for KV block count is: (1) explicit `--total-kv-blocks` CLI flag (always wins, R18), (2) auto-calculated roofline value (when roofline mode is active), (3) defaults.yaml empirical value (blackbox mode), (4) CLI default (1M). Auto-calculation must run AFTER roofline config is loaded (model config + hardware config available) but BEFORE the total KV blocks validation. The existing CLI flag-change detection mechanism distinguishes user-provided values from defaults.yaml/default values. When the blackbox defaults.yaml path sets a KV block count and then roofline fallback activates, auto-calculation replaces the defaults.yaml value.
- **KV-CAP-2 (Blackbox unchanged):** Blackbox mode behavior is completely unchanged by this feature.

**Precedence scenario table** (KV-CAP-1 concrete examples):

| `--total-kv-blocks` | `--model-config-folder` | defaults.yaml has KV? | GPU memory in HW config? | Expected behavior |
|---|---|---|---|---|
| Explicit | Yes | Yes | Yes | Uses explicit value (BC-2) |
| Explicit | No | Yes | — | Uses explicit value (BC-2) |
| Not provided | Yes | Yes | Yes | Auto-calculates, replaces defaults.yaml value (BC-1) |
| Not provided | Yes | No | Yes | Auto-calculates (BC-1) |
| Not provided | Yes | Yes | No | Falls back to defaults.yaml value with warning (GPU memory missing) |
| Not provided | Yes | No | No | Falls back to 1M default with warning (GPU memory missing) |
| Not provided | No | Yes | — | Uses defaults.yaml value (blackbox, BC-3) |
| Not provided | No | No | — | Uses 1M default (BC-3) |

**Backward compatibility note:** Rows 5-6 handle existing roofline users whose hardware config JSON lacks the new GPU memory field. Rather than erroring (which would be a breaking change), auto-calculation falls back gracefully with a warning directing the user to either add GPU memory capacity to their hardware config or provide `--total-kv-blocks` explicitly. This preserves backward compatibility: existing roofline users without the new field continue working as before (with 1M default or defaults.yaml value), while new users get auto-calculation.
- **KV-CAP-3 (Non-negative result):** The calculation must never produce zero or negative block counts. If model weights exceed GPU budget, return an error.
- **KV-CAP-4 (Division safety):** All denominators are validated before division (R11). Specifically: TP > 0, block size > 0, per-block bytes > 0, and `num_attention_heads > 0` (for head_dim derivation). Each zero-valued input returns a descriptive error.
- **KV-CAP-5 (Determinism):** Auto-calculated value is a pure function of inputs — no randomness, no external state (INV-6).
- **KV-CAP-6 (Observability):** Auto-calculated value is logged at info level with key intermediate values (model weight GiB, activation GiB, allocatable GiB, total blocks) so users can diagnose unexpected results. This is especially important given Decision 5's known limitation with quantized models and for FP8 KV cache deployments (see Decision 8).
- **KV-CAP-7 (Input validation):** The calculation function validates all inputs independently — it does not rely on the roofline latency model's validation function, which validates latency-model-specific fields only. Validated inputs: GPU memory capacity > 0, precision bytes > 0, TP > 0, block size > 0, `num_attention_heads > 0`, `num_layers > 0`, `hidden_dim > 0`. Additionally, degenerate numerator inputs that produce silently wrong results (R20) are validated: `intermediate_dim > 0` (MLP weight term) and `vocab_size > 0` (embedding/lm_head terms). A zero `num_layers` would produce zero per-token KV bytes and zero layer weights, producing a massively inflated block count. A zero `hidden_dim` would cascade to `head_dim = 0` and `per_token_kv_bytes = 0`.
- **KV-CAP-8 (INV-4 preservation):** The auto-calculated value is assigned once at initialization (before KV store construction) and remains immutable throughout the simulation. INV-4 (`allocated_blocks + free_blocks = total_blocks`) is preserved identically to the manual-flag path because `total_blocks` does not change after construction.
- **KV-CAP-9 (CPU tier independence):** Auto-calculation applies to GPU-tier KV blocks only. The `--kv-cpu-blocks` flag for CPU-tier offload capacity is unaffected — it remains manually specified. Auto-calculated GPU blocks are independent of CPU tier configuration. This means tiered KV cache users get auto-calculation for the GPU tier but must still manually size the CPU tier.

**Downstream invariant impact note:** While this feature does not modify any simulation-time code, it fundamentally changes simulation behavior by switching from the 1M-block default to realistic capacity (~100K–500K blocks). This is a significant behavioral change — KV eviction, preemption, and memory pressure become operationally real for roofline-mode users who were previously running with effectively unlimited KV cache. This has downstream consequences for existing invariants:
- **INV-1 (Request conservation):** `dropped_unservable` requests may increase when KV blocks are exhausted. The conservation equation still holds; the distribution across its terms changes. Per the H8 experiment findings (INV-4 operational note), KV pressure exhibits a "sharp cliff" — for supported model/GPU/TP combinations, auto-calculated values should be well above this cliff. Decision 5 (quantized model limitation) could push values closer to the cliff for quantized models; those users should use `--total-kv-blocks` explicitly.
- **INV-7 (Signal freshness):** The `kv-utilization` routing scorer and the free-KV-blocks snapshot field (both Tier 3 stale — refreshed periodically, not per-batch-step) become operationally significant. With the previous 1M default, utilization was always near zero; with realistic capacity it varies meaningfully. Any future capacity-aware admission or routing policy must account for this staleness.
- **INV-8 (Work-conserving):** Under KV exhaustion, the wait queue may have pending requests but KV allocation fails for all of them. The work-conserving property still holds (a step event exists to retry), but throughput drops to zero until KV blocks are freed via completion. This is pre-existing behavior now made reachable.
- **INV-2 (Request lifecycle):** Unaffected. Preemption (running → queued recomputation) is an existing state transition; this design does not change transition rules, only makes the preemption path exercised more frequently.
- **INV-3 (Clock monotonicity), INV-5 (Causality):** Unaffected. No new events, no clock changes. Configuration-time computation only.
- **INV-6 (Determinism):** Unaffected. The auto-calculated value is a pure function of inputs (KV-CAP-5). The simulation remains deterministic: same seed + same config (including auto-calculated KV blocks) = byte-identical stdout. The end-to-end test should verify determinism by running the same roofline configuration twice and confirming identical output.

## 5. Reference Formula

The implementation must replicate the calculation from `capacity_planner.py:total_kv_cache_blocks()`. BLIS simplifies with `pp=1, dp=1`. The following table maps each reference variable to its BLIS simplification:

| Reference variable | Reference formula | BLIS (pp=1, dp=1) |
|---|---|---|
| `gpu_count` | `tp * pp * dp` | `tp` |
| `available_memory` | `gpu_mem * util * gpu_count` | `gpu_mem * util * tp` |
| `model_size` | `model_memory_req() * dp` | `model_memory_req()` |
| `activation_memory` | `activation_const * dp` | `activation_const` |
| `non_torch_memory` | `non_torch_per_gpu * gpu_count` | `non_torch_per_gpu * tp` |
| `per_token_memory_per_gpu` | `per_token_bytes / (tp * pp)` | `per_token_bytes / tp` |

### Step 1 — Per-token KV memory (before TP sharding)

For standard attention (non-MLA):

```
per_token_kv_bytes = num_layers * 2 * head_dim * num_kv_heads * precision_bytes
```

The factor of 2 accounts for separate K and V caches.

Where:
- `head_dim = hidden_dim / num_attention_heads`
- `precision_bytes` = bytes per parameter from `torch_dtype` (e.g., 2 for bfloat16)

### Step 2 — Account for tensor parallelism

```
per_token_kv_bytes_per_gpu = per_token_kv_bytes / tp
```

### Step 3 — Per-block memory

```
per_block_bytes = per_token_kv_bytes_per_gpu * block_size
```

### Step 4 — Allocatable KV cache memory

Total memory budget across all TP GPUs:

```
total_available = gpu_memory_gib * gpu_mem_util * tp
```

Memory consumed (total across all GPUs):

```
model_weight_gib    = total_model_bytes / (1024^3)
activation_gib      = 5.5  (dense) or 8.0 (MoE)
non_torch_total_gib = non_torch_per_gpu * tp
    where non_torch_per_gpu = 0.15 (TP=1) or 0.6 (TP>=2)

overhead_gib = model_weight_gib + activation_gib + non_torch_total_gib

# Check BEFORE clamping: if overhead exceeds budget, return specific error (KV-CAP-3, BC-8)
if overhead_gib > total_available:
    error("model weights (X GiB) + activation (Y GiB) + overhead (Z GiB) exceed available GPU memory (W GiB)")

allocatable_gib = total_available - overhead_gib
```

Note on model weights: TP distributes weights across GPUs. With TP=2, each GPU holds half the weights. The total memory consumed by model weights across all GPUs equals the total model size (not multiplied by TP). This is why the formula uses `total_available` (sum across GPUs) minus `model_weight_gib` (total, not per-GPU).

Note on activation memory: The 5.5/8.0 GiB constant is a per-replica quantity from empirical vLLM profiling. It is **not** multiplied by TP because activation memory does not scale with tensor parallelism — each TP rank processes a shard of the activations. The reference multiplies by `dp` (data parallelism), which is 1 in BLIS. Since BLIS hardcodes `dp=1`, the activation constant is used directly.

Note on non-torch overhead: There is a discontinuity at the TP=1 to TP=2 boundary: total non-torch jumps from 0.15 GiB (TP=1) to 1.2 GiB (TP=2). This matches the reference behavior — NCCL buffers and additional CUDA context allocations are significantly larger with multi-GPU communication. The per-GPU constant (0.6 GiB for TP>=2) is then multiplied by total GPU count.

### Step 5 — Total blocks

```
total_blocks = floor(allocatable_bytes / per_block_bytes)
```

Where `allocatable_bytes = allocatable_gib * 1024^3`.

### Total model weight bytes computation

The reference uses HuggingFace safetensors metadata for exact parameter counts by dtype. BLIS computes from architecture parameters, which requires explicit handling of MoE expert multiplication and tied embeddings that the reference handles implicitly via stored parameter metadata.

```
embeddings   = vocab_size * hidden_dim * bytes_per_param
per_layer    = attention_weights + mlp_weights + layer_norm_weights
  attention  = hidden_dim * (hidden_dim + 2 * kv_dim) + hidden_dim * hidden_dim
             = (Q projection) + (K + V projections) + (output projection)
  kv_dim     = num_kv_heads * head_dim

  # Dense models (SwiGLU activation — gate + up + down = 3 matrices):
  mlp        = 3 * hidden_dim * intermediate_dim
  # Note: The factor of 3 assumes SwiGLU/GeGLU activation (gate_proj + up_proj + down_proj).
  # Standard ReLU/GELU MLPs use 2 matrices. All BLIS validation targets (Llama, Qwen, Mixtral)
  # use SwiGLU. If a non-SwiGLU model is added, this factor needs adjustment.
  # MoE models:
  mlp        = num_local_experts * 3 * hidden_dim * intermediate_dim + router_weights
  router_weights = num_local_experts * hidden_dim   (per-layer expert routing gate; negligible)

  layer_norm = 2 * hidden_dim                      (negligible but included)
all_layers   = per_layer * num_layers * bytes_per_param

# Tied embeddings: when tie_word_embeddings is true, lm_head shares the embedding tensor
lm_head      = 0                                    if tie_word_embeddings is true
lm_head      = vocab_size * hidden_dim * bytes_per_param   if tie_word_embeddings is false

final_norm   = hidden_dim * bytes_per_param

total_model_bytes = embeddings + all_layers + lm_head + final_norm
```

Where `bytes_per_param` = `precision_bytes` from `torch_dtype` (same value used in Step 1's KV formula).

Note: Bias terms are omitted — modern architectures (Llama, Qwen, Mixtral) use `attention_bias: false` and `mlp_bias: false`. The bias contribution (`hidden_dim` per layer) is negligible relative to weight matrices and well within the 10% tolerance band.

Note: MoE shared experts (e.g., DeepSeek v2's `n_shared_experts`) are not handled. Shared experts are always-active MLP blocks whose weights are added to the per-layer total in addition to the routed expert weights. This omission under-counts model weight memory for shared-expert architectures, producing optimistic (too many) KV blocks. No models in the current `defaults.yaml` use shared experts. If shared-expert models are added, extend the MLP formula: `mlp = num_local_experts * expert_mlp + n_shared_experts * shared_mlp + router_weights`. The MoE detection already checks for `n_shared_experts` in the HF config (Decision 3).

Note: This architecture-based computation differs from the reference's safetensors-based count. The reference naturally handles MoE expert weights and tied embeddings via stored metadata. BLIS must handle these explicitly because it computes from `config.json` architecture parameters alone. See Decision 1 for rationale and Decisions 9-10 for MoE and tied embedding handling.

## 6. Decisions with Trade-offs

### Decision 1: Model weight computation — architecture-based vs. external parameter count

**Problem:** The reference uses `model_params_by_dtype()` which queries HuggingFace for exact parameter counts. BLIS has only the HF `config.json` architecture parameters.

**Decision:** Compute total model bytes from architecture parameters (layers + embeddings + lm_head).

**Rationale:**
- BLIS already parses all needed fields from `config.json`
- No new external dependency or API calls
- For standard transformer architectures, the result matches HF parameter counts exactly (verified: Llama-3.1-8B yields 8,030,261,248 params)

**Alternatives considered:**
- *Reuse existing roofline per-step weight formula*: Only computes per-layer attention + MLP. Misses embeddings (~525M params) and lm_head (~525M params) for Llama-3.1-8B — a ~1 GiB error at bfloat16, causing ~1000 extra blocks. Rejected for accuracy.
- *Query HuggingFace API for param counts*: Would match the reference exactly but introduces network dependency in a library function, violating separation of concerns (R6). Rejected.
- *Add a `total_params` field to config*: Requires manual maintenance. Rejected for fragility.

**What breaks if wrong:** If architecture-based computation misses parameters for non-standard architectures (e.g., models with extra projection heads), the KV block estimate will be slightly too high (by the proportion of missed parameters). The BC-4 tolerance band (10%) provides margin. Key special cases handled explicitly: MoE expert weight multiplication (Decision 9) and tied embeddings (Decision 10).

### Decision 2: Activation memory constants — hardcoded vs. configurable

**Problem:** The reference uses fixed constants (5.5 GiB dense, 8.0 GiB MoE) that represent empirical vLLM profiling results. These may change across vLLM versions.

**Decision:** Hardcode the constants to match the reference. Do not expose as CLI flags.

**Rationale:**
- Matches the reference implementation exactly
- Avoids exposing internal vLLM implementation details to users
- Constants are stable across vLLM versions (they changed once in 2 years)
- If `--total-kv-blocks` is passed explicitly, these constants are irrelevant (KV-CAP-1)

**Alternatives considered:**
- *Make configurable via CLI flags*: Exposes implementation details that users shouldn't need to know. Rejected for usability.
- *Make configurable via defaults.yaml*: Adds config surface without clear user benefit. Rejected.

**What breaks if wrong:** If vLLM changes its activation memory significantly, the auto-calculated value will drift from empirical. The 10% tolerance band provides margin, and explicit `--total-kv-blocks` provides an escape hatch.

### Decision 3: MoE detection from HF config

**Problem:** Different activation memory constants and weight formulas apply to MoE vs. dense models. Need to distinguish at config time for both activation memory selection and expert weight multiplication.

**Decision:** Detect MoE by checking for any of the standard HuggingFace MoE indicator fields: `n_routed_experts`, `n_shared_experts`, `num_experts`, `num_experts_per_tok`, `num_local_experts` in the HF config.

**Rationale:** Covers all known MoE indicator fields across HuggingFace model configs. The reference `is_moe()` checks `n_routed_experts`, `n_shared_experts`, `num_experts`, `num_experts_per_tok`. BLIS additionally checks `num_local_experts` (used by Mixtral) for robustness. The union of both sets ensures detection of all known MoE architectures.

**Data flow:** The MoE detection check requires access to the raw HF config fields, which are not currently exposed through the model architecture configuration type. The calculation function will accept a boolean MoE indicator (and `num_local_experts` count for weight multiplication) derived from the raw HF config at parse time. This avoids modifying the model config struct and its ~12+ construction sites.

**Coupling risk mitigation:** Both the model config extraction and the MoE/tied-embedding extraction must parse the same HF config file. To prevent silent divergence, both should share the same parsed HF config instance (one parse, two extraction passes). The micro plan should prescribe that the CLI layer calls a single parse function and passes the result to both the model config factory and the capacity function.

**Failure mode — unrecognized MoE architecture:** If a new MoE model uses indicator fields not in the detection set, it would be misclassified as dense, causing: (a) 5.5 GiB activation instead of 8.0 GiB, and (b) no expert multiplication on MLP weights — potentially tens of GiB undercount. The result would be wildly over-estimated KV blocks (far beyond the 20% tolerance), and users would see KV exhaustion much sooner than expected. This failure mode is detectable via the fidelity validation (auto-calculated value deviates > 20% from empirical). No silent data loss occurs because the deviation would be large enough to fail the automated assertion.

**Alternatives considered:**
- *Check `model_type` field*: Not all MoE models have a distinguishing type string. Fragile.
- *Always use dense constant*: Would under-estimate memory usage for MoE models by 2.5 GiB and massively under-count model weights. Rejected.

### Decision 4: GPU memory capacity in hardware calibration

**Problem:** GPU memory capacity is not currently in the hardware calibration data. Needed for KV capacity calculation.

**Decision:** Add a GPU memory capacity field to the hardware calibration type. Validation of this field lives **only** in the KV capacity calculation function (KV-CAP-7), **not** in the roofline latency model's validation function. This avoids breaking existing roofline users who provide `--total-kv-blocks` explicitly but have hardware config files without the new field. The roofline latency model does not need GPU memory capacity (it only needs FLOP/bandwidth specs); only the KV capacity calculation does.

**GPU memory values for supported GPUs:** H100 (SXM, 80 GiB) = 80.0 GiB. A100-SXM (80 GB variant) = 80.0 GiB. Note: the hardware config JSON currently uses "A100-SXM" while `defaults.yaml` uses "A100-80" for the same GPU. PR 1 must reconcile this naming — either by adding an "A100-80" alias entry to the hardware config JSON, or by mapping the names in the lookup code. See BC-14 for the behavioral contract.

**R9 consideration:** Zero is not a valid value for GPU memory capacity, so a plain float (not a pointer type) is correct. A missing field in the JSON defaults to 0.0, which KV-CAP-7 catches.

**Rationale:** Natural home — the hardware calibration type already captures GPU specifications. Validated alongside other hardware fields.

**Construction site audit (R4):** The hardware calibration struct has construction sites in latency test fixtures (3 files), the hardware config JSON data file, and a zero-value initialization in the CLI layer. Additionally, ~12 test files across the simulator and cluster packages construct zero-value hardware calibration structs via canonical constructor calls. These zero-value sites are intentionally not updated: zero GPU memory capacity means no auto-calculation occurs (blackbox mode path), which is safe because auto-calculation is only invoked when roofline mode is active and the GPU memory capacity validation (KV-CAP-7) catches zero values.

All non-zero construction sites (latency test fixtures and hardware config JSON) must be updated when the field is added.

**Extension friction:** Adding the GPU memory field requires updating: (1) the hardware calibration type definition (1 file), (2) latency test fixtures with non-zero hardware calibration values (~3 files), (3) the hardware config JSON data file (1 file). Total: ~5 file touches. Zero-value construction sites (~12 test files) need no changes.

### Decision 5: Quantized model weight memory (known limitation)

**Problem:** For quantized models (w4a16, w8a8, fp8), model weights consume less memory than `total_params * precision_bytes` suggests. The `precision_bytes` value (derived from `torch_dtype`) reflects the compute precision, not the weight storage precision. This means the formula over-estimates model weight memory for quantized models, producing a lower (more conservative) KV block count.

**Decision:** Use `precision_bytes` (from `torch_dtype`) for both model weights and KV cache in v1. Document as a known limitation.

**Rationale:**
- The reference implementation uses sophisticated per-dtype parameter counting that requires HF API access — not available in BLIS
- Over-estimating model weight memory produces *conservative* KV block counts (better to under-estimate capacity than over-estimate)
- The `--total-kv-blocks` explicit override provides an escape hatch for quantized models
- Future work can add a `weight_bytes_per_param` field when needed

**Impact:** For `w4a16` models, the formula over-estimates model weight memory by ~3.5x for the weight portion. For a 70B model at w4a16, this means ~30 GiB over-estimation, resulting in significantly fewer auto-calculated blocks. Users running quantized models in roofline mode should pass `--total-kv-blocks` explicitly until this is addressed.

**Alternatives considered:**
- *Infer weight precision from model name (e.g., "w4a16")*: Fragile, not all model names follow this convention. Deferred.
- *Add `quant_config` parsing from HF config*: Significant complexity. Deferred to a follow-up issue.

### Decision 6: Calculation placement — configuration-time pure function

**Problem:** Where should the calculation live?

**Decision:** Exported pure function that accepts model config, hardware config, TP, block size, and MoE/tied-embedding indicators. Placement in the latency module is natural (alongside existing roofline pure functions) but the micro plan may choose a different location if warranted. CLI integration in the CLI layer.

**Rationale:**
- This is a **configuration-time computation**, not a simulation-time module. It runs once before the simulation starts, produces a single integer, and has no event interactions, no state ownership during simulation, and no extension friction in the module sense. No behavioral contract (6-aspect template) is needed because no mutable state, events, or interface are introduced.
- Consistent with existing pattern (roofline step time estimation is a pure function in the same module)
- Testable without CLI scaffolding
- Does not violate separation of concerns — the calculation is config derivation, not simulation logic

**Note on analytical vs. runtime estimation:** vLLM profiles GPU memory at runtime (load model, run forward pass, measure `torch.cuda.mem_get_info()`). The reference `capacity_planner.py` is itself an analytical approximation of this profiling. This function is therefore an approximation of an approximation. Divergence from actual vLLM KV block counts is expected due to: (a) PyTorch allocator fragmentation, (b) CUDA graph memory (reference returns 0, but may be non-zero in newer vLLM), (c) model-specific memory allocations not captured by architecture parameters. The 10% tolerance in BC-4 covers this gap.

### Decision 7: Existing roofline per-step weight formula reuse

**Problem:** The existing roofline latency model's per-step memory access function computes per-step weight memory (attention + MLP only). The KV capacity calculation needs *total* model bytes (including embeddings, lm_head, norms).

**Decision:** Do NOT reuse the existing formula. Write a separate total-model-bytes computation.

**Rationale:**
- The existing per-step formula intentionally omits embeddings and lm_head because they're not loaded per-step in the roofline latency model
- The KV capacity formula needs the full model footprint in GPU memory
- For Llama-3.1-8B, the difference is ~3 GiB (~1.05B params for embeddings + lm_head at bfloat16)
- Mixing purposes would couple the latency model to the capacity model inappropriately

### Decision 8: KV cache dtype — use model dtype (matches reference for standard models)

**Problem:** KV cache precision affects the per-token KV bytes calculation and therefore the total block count. Need to determine the correct precision to use.

**Decision:** Use the model's `torch_dtype` precision for KV cache bytes. This matches both vLLM's `--kv-cache-dtype auto` default and the reference's `inference_dtype_byte()` for standard models.

**Rationale:**
- The reference's `KVCacheDetail` sets `precision_in_bytes = inference_dtype_byte(model_config)`, which returns 2 for bfloat16 models (via `precision_to_byte('bfloat16')`). The `DEFAULT_KV_CACHE_DTYPE_BYTES = 1` is only a last-resort fallback when the model's dtype cannot be resolved — it does NOT apply to standard bfloat16/float16 models
- For all BLIS validation targets (Llama, Qwen, Mixtral), both BLIS and the reference use 2 bytes per KV element. There is no divergence for standard models
- The empirical `defaults.yaml` values were collected from vLLM with `--kv-cache-dtype auto` (bfloat16), so using model dtype precision produces the correct comparison target
- Verified: using bfloat16 (2 bytes) for KV produces 5.2% deviation from empirical for Llama-3.1-8B/H100/TP=2; using FP8 (1 byte) would produce ~90% deviation, confirming empirical values used bfloat16 KV

**What breaks if wrong:** If a model has an unresolvable `torch_dtype` (rare), the reference falls back to FP8 (1 byte) while BLIS would fail at precision validation. The BC-7 error path for `precision_bytes=0` handles this case.

**FP8 KV cache deployments:** Users deploying vLLM with `--kv-cache-dtype fp8` will have approximately 2x more KV blocks than BLIS auto-calculates. This is a known gap addressed by the deferred `--kv-cache-dtype` flag in Extension Points. The `--total-kv-blocks` explicit override provides an immediate escape hatch.

Note: This creates an implicit coupling — the KV cache precision matches the roofline latency model's memory access calculations. If a future change decouples KV cache precision from model precision, both the capacity formula and the roofline memory bandwidth estimates would need updating.

### Decision 9: MoE expert weight multiplication

**Problem:** BLIS computes model weight bytes from architecture parameters. For MoE models, each transformer layer has `num_local_experts` separate MLP weight sets. The reference avoids this by querying safetensors metadata (which naturally includes all expert weights). BLIS must handle this explicitly.

**Decision:** For MoE models, multiply the MLP weight term by `num_local_experts`. Add a small router weight term (`num_local_experts * hidden_dim` per layer) for completeness.

**Rationale:**
- For Mixtral-8x7B (`num_local_experts=8`): dense MLP per layer = `3 * 4096 * 14336 * 2 ≈ 0.33 GiB`. MoE MLP per layer = `8 * 0.33 GiB ≈ 2.6 GiB`. Across 32 layers, the difference is ~73 GiB — ignoring this would massively over-estimate free memory and produce far too many KV blocks.
- Router weights are negligible (~2 MiB for Mixtral) but included for correctness.

**What breaks if wrong:** Without expert multiplication, a Mixtral-8x7B at bfloat16 would undercount model weight memory by ~73 GiB, producing wildly over-estimated KV block counts.

### Decision 10: Handling `tie_word_embeddings`

**Problem:** Some HF models (e.g., Qwen2.5-3B, Qwen2.5-1.5B) set `tie_word_embeddings: true`, meaning the input embedding and output projection (`lm_head`) share the same weight tensor. The architecture-based formula must not double-count this.

**Decision:** Check `tie_word_embeddings` from the HF config. When true, omit the `lm_head` term from the total model bytes computation.

**Rationale:**
- For Qwen2.5-3B (`vocab_size=151936, hidden_dim=2048, bfloat16`): the `lm_head` term is `151936 * 2048 * 2 ≈ 0.58 GiB`. Double-counting over-estimates model weight by 0.58 GiB, producing ~36,000 fewer KV blocks at block_size=16.
- For small models, this over-estimation can exceed the 10% tolerance band.
- The reference avoids this implicitly because safetensors metadata counts each stored tensor once.

**Data flow:** The `tie_word_embeddings` field is available in the raw HF config JSON. It will be passed as a boolean parameter alongside the MoE indicators.

## 7. Real-System Correspondence

| BLIS concept | vLLM equivalent | Reference (`capacity_planner.py`) equivalent |
|---|---|---|
| Auto-calculated `total_kv_blocks` | `num_gpu_blocks` from `CacheEngine.get_cache_block_size()` + `Worker.determine_num_available_blocks()` | `total_kv_cache_blocks()` return value |
| `precision_bytes` from `torch_dtype` | `--kv-cache-dtype auto` → model dtype | `inference_dtype_byte(model_config)` |
| `gpu_memory_gib * gpu_mem_util` | `torch.cuda.mem_get_info()` * `gpu_memory_utilization` | `gpu_mem * gpu_mem_util` |
| 5.5 GiB / 8.0 GiB activation constant | Measured via `torch.cuda.mem_get_info()` after model load + forward pass | `ACTIVATION_MEMORY_BASE_DENSE_GIB` / `_MOE` |
| 0.15 / 0.6 GiB non-torch | NCCL buffers + CUDA context (measured by vLLM profiling) | `NON_TORCH_MEMORY_*_PER_GPU_GIB` constants |
| Architecture-based weight counting | `model.state_dict()` actual parameter sizes | `model_params_by_dtype()` via HuggingFace API |
| Raw block count (no watermark) | `num_gpu_blocks` minus `watermark_blocks` (1% reserved) | Raw `total_kv_cache_blocks()` (no watermark) |

Note: vLLM's `BlockSpaceManagerV2` reserves a small watermark (default 1%) of blocks for priority scheduling. BLIS computes the raw block count without this watermark, matching the reference. The 1% difference is well within the 10% tolerance band.

## 8. Extension Points

- **New GPU types:** Add GPU memory capacity to the GPU's entry in the hardware config JSON. No code changes needed. Extension friction: 1 touch point.
- **Activation memory evolution:** If vLLM changes activation constants, update the two constants in the KV capacity function. Single location, zero touch-point friction.
- **MLA attention format:** The per-token KV formula would change for MLA (DeepSeek v3 style). The function should detect MLA from HF config and apply the alternative formula. Deferred — no MLA models currently in `defaults.yaml`.
- **Weight quantization awareness:** A future `weight_bytes_per_param` field (from HF `quant_config`) would replace the current `precision_bytes` usage for the weight portion only, decoupling weight precision from KV cache precision.
- **KV cache dtype decoupling:** A `--kv-cache-dtype` flag would allow using FP8 (1 byte) KV cache instead of the model dtype. This would affect both the per-token KV bytes formula (Step 1) and should be coordinated with the roofline latency model's memory access calculations.
- **GPU memory utilization configurability:** The hardcoded 0.9 `gpu_mem_util` matches vLLM's default. A future `--gpu-memory-utilization` flag (matching vLLM's `--gpu-memory-utilization` CLI flag) would allow tuning this value. Extension friction: 1 constant → 1 CLI flag + 1 config field.
- **Capacity-aware admission control:** With accurate KV block counts, admission policies can now meaningfully consider cluster-wide KV utilization. A future capacity-aware admission policy could reject requests when aggregate free KV blocks falls below a threshold, preventing instance-level KV allocation failure cascades. **Operational note:** With the current `AlwaysAdmit` policy, realistic KV capacity creates a sharp edge — all requests are admitted regardless of KV pressure, relying entirely on instance-level preemption as backpressure. Users running high-load scenarios with auto-calculated capacity should be aware that throughput can drop sharply when KV blocks are exhausted (INV-8 note above). A capacity-aware admission policy would smooth this behavior.
- **Tiered KV cache interaction:** Auto-calculation applies to GPU-tier KV blocks only. When tiered KV caching is enabled (`--kv-cpu-blocks`), the CPU tier block count remains manually specified. The GPU tier auto-calculated value is independent of the CPU tier — changing GPU blocks does not affect CPU blocks or the offload/reload thresholds. Users combining roofline mode with tiered KV caching should be aware that only the GPU tier benefits from auto-calculation.
- **Stale snapshot impact:** Accurate KV capacity amplifies the impact of snapshot staleness on the `kv-utilization` routing scorer. With the previous 1M-block default, utilization stayed near zero; with realistic capacity, utilization varies meaningfully and the periodic refresh interval (`--snapshot-refresh-interval`) becomes operationally significant. The existing H3 experiment findings on snapshot refresh interval sensitivity should be re-validated with realistic KV capacity — H3 was conducted with the 1M-block default, so its conclusions about the `kv-utilization` scorer's sensitivity may change. The default weighted scoring profile (`prefix-affinity:3,queue-depth:2,kv-utilization:2`) will produce more differentiated routing decisions under realistic capacity.

## 9. Validation Strategy

### Verification (correctness)

- **Unit tests on the calculation function** with known-good inputs:
  - Llama-3.1-8B / H100 (80 GiB) / TP=2 → result within 10% of empirical 132,139 blocks (BC-4)
  - Llama-3.1-8B / H100 / TP=4 → result within 10% of empirical 559,190 blocks
  - Llama-2-70B / H100 / TP=4 → validates large model path (70B params at float16). Note: `defaults.yaml` empirical value for this configuration should be verified before use as a tolerance baseline; if no empirical value is available, validate against `capacity_planner.py` reference output.
  - Mixtral-8x7B / H100 / TP=2 → validates MoE weight formula and 8.0 GiB activation constant within 20% of empirical (BC-4, BC-9, BC-11)
  - At least one A100-based configuration (e.g., Llama-3.1-8B / A100-80 / TP=2) → validates A100 GPU memory path and GPU name resolution (BC-14)
- **Monotonicity tests:** Higher TP → more KV blocks (BC-5), tested across the TP=1→TP=2 discontinuity (non-torch overhead step function). Higher GPU memory → more blocks. Precondition: monotonicity applies for TP values where the calculation succeeds (no error from model too large for GPU).
- **Edge case tests:** Model weights exceeding GPU budget → error (KV-CAP-3). Zero TP → error (KV-CAP-4). Zero GPU memory → error (KV-CAP-7). Zero precision bytes → error (KV-CAP-7). Block size=0 → error (KV-CAP-4). `num_kv_heads % TP != 0` with `num_kv_heads >= TP` → error (BC-23). Auto-calculated result < 1000 → warning logged (BC-24). `num_local_experts=1` → classified as dense, not MoE (BC-25).
- **Invariant tests:** auto-calculated value is pure (same inputs → same output, KV-CAP-5)
- **Override tests:** Explicit `--total-kv-blocks` always wins (KV-CAP-1). Blackbox mode unchanged (KV-CAP-2).
- **Tied embedding tests:** Qwen2.5-3B with `tie_word_embeddings: true` → `lm_head` not double-counted (BC-12)
- **End-to-end cluster test:** Run a cluster simulation in roofline mode without `--total-kv-blocks` and verify: (a) simulation completes without error, (b) KV eviction events occur (the auto-calculated value is realistic, not 1M), (c) INV-4 holds throughout the run, (d) INV-1 request conservation equation holds at simulation end, (e) INV-6 determinism — running the same configuration twice with the same seed produces byte-identical stdout. This validates the CLI integration path (KV-CAP-1) through to simulation behavior and confirms the newly-exercised code paths preserve all system invariants.

### Validation (fidelity)

- **Automated assertion** (not just logging): Compare auto-calculated values against all `total_kv_blocks` entries in `defaults.yaml` for non-quantized models with available HF configs in `model_configs/`. Assert deviation ≤ 10% for dense models and ≤ 20% for MoE models (R7: invariant test alongside golden test).
- For quantized models, document expected over-estimation due to Decision 5 and skip assertion.
- **Cross-validation against reference:** Run the reference `capacity_planner.py` with the same model/GPU/TP inputs for at least 3 configurations (one dense, one large, one MoE) and compare output to BLIS. This validates the stated goal of "consistency with the llm-d ecosystem." Cross-validation is a **one-time verification** during implementation: capture the reference output as hardcoded expected values in the test suite. This avoids a CI dependency on the Python reference while still ensuring formula parity. If the reference implementation changes, re-run the cross-validation manually and update expected values.

### Model config coverage

Not all models in `defaults.yaml` have corresponding HF configs in `model_configs/`. Auto-calculation can only be validated against empirical values for models with available config files. The current `model_configs/` directory covers: Llama-3.1-8B-Instruct, Llama-2-70B, Llama-2-7B, Meta-Llama-3-8B, CodeLlama-34B, Mixtral-8x7B, Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Qwen3-14B.

Notable gaps:
- **Llama-3.3-70B and Qwen2.5-72B:** Not available in `model_configs/`. Their empirical values in `defaults.yaml` cannot be auto-validated. HF configs should be added during implementation.
- **Llama-4 MoE models:** `defaults.yaml` includes Llama-4-Maverick-17B-128E (FP8, 128 experts) and Llama-4-Scout-17B-16E (16 experts). Neither has an HF config in `model_configs/`. Maverick is FP8 (excluded by Decision 5's quantization limitation — auto-calculation would be conservative). Scout has 16 experts — the MoE weight formula should handle it correctly (`num_local_experts=16`) but cannot be validated without an HF config. Users running Llama-4 models in roofline mode should use `--total-kv-blocks` explicitly until validation configs are available.
- **Empirical data vLLM version spread:** Most `defaults.yaml` entries use vLLM v0.8.4, while some entries (e.g., `openai/gpt-oss-*`, `redhatai/smollm3-*`) use v0.10.1.1. The hardcoded constants (5.5/8.0 GiB activation, 0.15/0.6 GiB non-torch) are calibrated against the reference's constants, which were measured on a specific vLLM version. If vLLM v0.10.1 changed memory profiling behavior, tolerance comparisons against mixed-version empiricals could mask systematic errors. The cross-validation step should use empiricals from a consistent vLLM version where possible.

### Tolerance band justification

**Dense models (10%):** The tolerance accounts for: (a) difference between architecture-based weight counting and safetensors-based counting (minor parameter differences for non-standard layers), (b) gap between analytical estimation and vLLM's runtime GPU memory profiling (PyTorch allocator fragmentation, CUDA graph memory), (c) the `tie_word_embeddings` and bias term omissions for models where they apply. For non-quantized dense transformer models, the expected deviation is 2-5%.

**MoE models (20%):** MoE models have a wider tolerance because: (a) the 8.0 GiB activation constant is a coarser approximation (covers diverse expert gating patterns), (b) router weight memory is estimated from architecture parameters and may diverge from actual vLLM allocation, (c) expert parameter count from architecture may not perfectly match safetensors metadata for all MoE architectures. For Mixtral-8x7B, the expected deviation from empirical is ~15-18%. If the deviation exceeds 20%, the architecture-based computation or activation constant should be investigated.

### Sensitivity analysis — activation memory constants

The activation memory constants (5.5 GiB dense, 8.0 GiB MoE) are the least principled inputs — they are empirical vLLM profiling results, not derived from architecture parameters. A back-of-envelope sensitivity analysis for Llama-3.1-8B / H100 / TP=2:

- **Baseline:** total_available ≈ 144 GiB (80 × 0.9 × 2). Model weights ≈ 16 GiB. Activation = 5.5 GiB. Non-torch = 1.2 GiB. Allocatable ≈ 121.3 GiB.
- **If activation changes by ±1 GiB** (5.5 → 4.5 or 6.5): allocatable changes by ∓1 GiB. At ~920 bytes/block (Llama-3.1-8B, block_size=16, TP=2), this is ±1,140 blocks — about ±0.9% of the ~132K empirical total. Well within 10% tolerance.
- **If activation changes by ±3 GiB** (5.5 → 2.5 or 8.5): ±3,420 blocks, ~2.6%. Still within 10% tolerance.
- **Breakeven:** The activation constant would need to be wrong by ~13 GiB to breach the 10% tolerance for this configuration.
- **For smaller models or lower TP**, the sensitivity is higher because the KV block count is smaller — the same 1 GiB error represents a larger percentage.

**Conclusion:** The activation constant has low sensitivity for the primary validation targets at TP>=2. The tolerance band is robust to activation constant drift of ±3 GiB at TP=2.

**TP=1 worst case:** For Llama-3.1-8B / H100 / TP=1, the auto-calculated result is approximately 26,312 blocks vs. empirical 29,205 — a 9.9% deviation, consuming nearly all of the 10% budget. At TP=1, the total block count is much smaller (~26K vs. ~132K at TP=2), so the same absolute GiB error represents a larger percentage. A ±1 GiB activation constant error at TP=1 represents ±3.9% of the total (vs. ±0.9% at TP=2). This is the tightest validation target. The TP=1 case passes the 10% band but with near-zero margin; if any constant drifts, it could breach the threshold. Users running TP=1 in production scenarios should verify auto-calculated values against their empirical measurements.

**Non-torch overhead sensitivity:** For the TP=1→TP=2 discontinuity (0.15 GiB → 1.2 GiB total), the overhead increase of 1.05 GiB is always dominated by the additional GPU memory (0.9 × 80 = 72 GiB for H100). The non-torch constants would need to change by ~13 GiB (TP=2) or ~3.4 GiB (TP=1) to breach the 10% tolerance. The constants are stable across NCCL versions and CUDA drivers. Low sensitivity overall.

**Conclusion:** The activation constant has low sensitivity for TP>=2 validation targets. TP=1 is the tightest case — users should be aware that the 10% band has minimal margin at TP=1. If vLLM significantly changes activation profiling, constants should be updated, but the 10%/20% bands provide margin for TP>=2.

### Golden dataset impact (R12)

The existing golden dataset (`testdata/goldendataset.json`) uses blackbox mode, which is completely unchanged by this feature (KV-CAP-2). No golden dataset regeneration is needed. If future golden tests are added for roofline mode, they should use explicit `--total-kv-blocks` to avoid coupling test baselines to the auto-calculation formula.

### Falsification criteria

The tolerance bands would need revision if:
1. Any non-quantized dense model deviates > 10% from empirical `defaults.yaml` values — indicates the architecture-based weight formula or activation constant is wrong
2. Any non-quantized MoE model deviates > 20% from empirical — indicates the MoE weight formula or 8.0 GiB constant is wrong
3. Cross-validation against `capacity_planner.py` shows systematic bias (e.g., all models deviate in the same direction by > 5%) — indicates a formula translation error
4. A new vLLM version changes KV block counts by > 10% for the same configuration — indicates the hardcoded constants need updating

## 10. DES Design Review Checklist

| Question | Answer |
|----------|--------|
| What analysis questions does this design help answer? | Enables realistic memory pressure modeling in roofline mode without manual block count input |
| What is modeled, simplified, and deliberately omitted? | See Section 3 (Modeling Decisions table) |
| What events are introduced or modified? | No new events introduced. However, the auto-calculated capacity value changes the *frequency* of existing events: KV allocation failures, preemption events, and eviction events become reachable under normal workloads (previously suppressed by the 1M default). See "Downstream invariant impact note" in Section 4. |
| How do new events interact with existing tie-breaking rules? | N/A — no events introduced. This is a pre-simulation configuration derivation. |
| What new state is introduced? Who owns it? | No new simulation-time state. A GPU memory capacity field is added to the hardware calibration data (configuration, not mutable state). It is consumed once at initialization and does not participate in event processing. |
| What new metrics are derived? | None. The auto-calculated block count is a configuration input, not a simulation output. |
| How will correctness be verified? | Unit tests against empirical values from `defaults.yaml`; see Section 9 |
| How will fidelity be validated? | Cross-reference with llm-d-benchmark `capacity_planner.py` for the same inputs; see Section 9 |
| Does this introduce new randomness? | No |
| What is the simplest version that answers the same questions? | This is already the simplest — a pure function with hardcoded constants matching the reference |

## 11. Behavioral Contracts

| # | Contract |
|---|----------|
| BC-1 | Roofline mode + no explicit `--total-kv-blocks` → uses auto-calculated value, not 1M default. If the blackbox defaults.yaml path already set a value before roofline fallback, auto-calculation replaces it. |
| BC-2 | Explicit `--total-kv-blocks` always wins, even in roofline mode (R18) |
| BC-3 | Blackbox mode completely unchanged |
| BC-4 | Result within 10% of empirical `defaults.yaml` value for dense models (e.g., Llama-3.1-8B / H100 / TP=2 = 132,139 blocks), and within 20% for MoE models (e.g., Mixtral-8x7B) |
| BC-5 | GIVEN the same model and GPU, WHEN TP increases and both TP values produce successful calculations (no error) and `num_kv_heads >= TP` for both values, THEN the KV block count increases (monotonicity). Tested across the TP=1→TP=2 non-torch overhead discontinuity. **Proof sketch:** increasing TP by 1 adds `0.9 * gpu_mem` GiB to the available budget while non-torch overhead increases by at most 0.6 GiB. Since `0.9 * 80 = 72 >> 0.6`, the net available memory always increases. Per-block bytes decrease (divided by larger TP), further ensuring monotonicity. Excluded: `num_kv_heads < TP` regime where the formula is known to be inaccurate (see BC-20). |
| BC-6 | Auto-calculated value logged at info level with intermediate values (model weight GiB, activation GiB, allocatable GiB) |
| BC-7 | Zero/negative denominators return error, never panic (R11). Specifically: GIVEN TP=0, THEN error. GIVEN block_size=0, THEN error. GIVEN num_attention_heads=0, THEN error. GIVEN per_block_bytes=0, THEN error. GIVEN GPU memory=0, THEN error. GIVEN precision_bytes=0, THEN error. Note: `num_kv_heads=0` in HF config means MHA (Multi-Head Attention) — the formula falls back to `num_kv_heads = num_attention_heads`. After fallback, `num_kv_heads > 0` is guaranteed by the `num_attention_heads > 0` check. |
| BC-8 | Model weights exceeding GPU memory returns error, never negative blocks |
| BC-9 | MoE models use 8.0 GiB activation constant; dense models use 5.5 GiB |
| BC-10 | If GPU memory capacity is 0 or missing in roofline mode, returns error with clear message directing user to add the field to their hardware config |
| BC-11 | MoE models: MLP weight term multiplied by `num_local_experts`. Dense formula produces wrong results for Mixtral-8x7B (off by ~73 GiB). |
| BC-12 | When `tie_word_embeddings: true`, `lm_head` weight is omitted from total model bytes (shared with embedding layer) |
| BC-13 | Mixtral-8x7B / H100 / TP=2: auto-calculated value within 20% of empirical `defaults.yaml` value. This is the primary MoE validation target. |
| BC-14 | GPU name resolution: GIVEN a `defaults.yaml` GPU name (e.g., "A100-80"), WHEN looking up GPU memory capacity from the hardware config JSON, THEN the lookup resolves to the correct GPU entry (e.g., "A100-SXM") or returns a clear "unrecognized GPU" error. No silent fallback to wrong GPU memory. |
| BC-15 | Intermediate computation overflow: GIVEN models up to 405B parameters at bfloat16, WHEN computing total model weight bytes, THEN intermediate products do not overflow 64-bit integers. (405B * 2 bytes = 810 GB, well within int64 range of ~9.2 × 10^18.) |
| BC-16 | NaN/Inf input rejection: GIVEN any floating-point input (GPU memory capacity, precision_bytes) that is NaN or Inf, WHEN the calculation function is invoked, THEN it returns a descriptive error — never silently propagates NaN/Inf to the result. |
| BC-17 | `head_dim` divisibility: GIVEN `hidden_dim` that is not evenly divisible by `num_attention_heads`, WHEN computing `head_dim = hidden_dim / num_attention_heads`, THEN the function returns a descriptive error rather than silently truncating. In practice, all known transformer architectures guarantee divisibility. |
| BC-18 | Preemption interaction: Auto-calculated KV blocks represent the same physical GPU KV budget that vLLM manages. Under memory pressure, BLIS's batch formation step uses the existing recomputation-based preemption path (vLLM v2 default). This design does not change preemption behavior — it only makes realistic capacity reachable. |
| BC-19 | SwiGLU activation guard: GIVEN a model config with `hidden_act` not in {`silu`, `swiglu`, `geglu`}, WHEN computing total model weight bytes, THEN the function returns a descriptive error rather than silently applying the 3-matrix MLP assumption. GIVEN `hidden_act` is absent from the HF config, THEN assume SwiGLU with a warning (all known transformer models either use SwiGLU or include the field). All current validation targets (Llama, Qwen, Mixtral) use SwiGLU (`silu`). Standard ReLU/GELU MLPs use 2 matrices — the 3-matrix formula would over-estimate MLP weights by 50%. The allowlist may need extension as new model architectures appear. |
| BC-20 | `num_kv_heads < TP` warning: GIVEN a configuration where `num_kv_heads < TP`, WHEN auto-calculating KV blocks, THEN the function logs a warning explaining that the result is optimistic (too many blocks) because vLLM replicates KV heads across ranks, and recommends `--total-kv-blocks` explicit override. The calculation still completes (not an error) because the result is usable as an upper bound. |
| BC-21 | Quantized model warning: GIVEN a model name containing known quantization suffixes (`fp8`, `gptq`, `awq`, `bnb`) or an HF config containing a `quantization_config` field, WHEN auto-calculating KV blocks, THEN the function logs a warning that the result is conservative (too few blocks) due to Decision 5's known limitation, and recommends `--total-kv-blocks` explicit override. |
| BC-22 | Floor-zero guard: GIVEN inputs where `allocatable_bytes > 0` but `floor(allocatable_bytes / per_block_bytes) == 0` (overhead leaves less than one block's worth of memory), THEN the function returns a descriptive error (not zero blocks). This extends KV-CAP-3 to cover the gap between "overhead exceeds budget" (explicit error at BC-8) and "budget barely positive but insufficient for one block." |
| BC-23 | TP divisibility guard: GIVEN `num_kv_heads >= TP` AND `num_kv_heads % TP != 0`, WHEN auto-calculating KV blocks, THEN the function returns a descriptive error. vLLM rejects this configuration at startup (KV heads cannot be evenly sharded across TP ranks). Computing a block count for an impossible configuration would be misleading. |
| BC-24 | Livelock-zone warning: GIVEN auto-calculated `total_blocks < 1000`, WHEN the calculation completes successfully, THEN the function logs a warning referencing the H8 experiment findings that below ~1000 blocks the preempt-requeue cycle can produce near-zero throughput (R19). The calculation still succeeds (not an error) because the value may be correct for the hardware — the warning alerts the user to the operational consequence. |
| BC-25 | `num_local_experts = 1` classification: GIVEN an HF config with `num_local_experts = 1` and no other MoE indicator fields with values > 1, WHEN classifying the model as MoE or dense, THEN it is classified as dense (5.5 GiB activation, no expert multiplication). A single expert is functionally identical to a dense MLP. Misclassifying as MoE would add 2.5 GiB activation overhead unnecessarily. |

## 12. PR Decomposition

### PR 1: Add GPU memory capacity to hardware config

**Goal:** Extend hardware calibration data with GPU memory capacity.

**Changes:**
- Add GPU memory capacity field to the hardware calibration type
- Add GPU memory data for supported GPUs to the hardware config data file
- GPU memory > 0 validated by the KV capacity calculation function (KV-CAP-7), not by the roofline latency model's validation (see Decision 4)
- Update all non-zero-value test construction sites (R4; see Decision 4 for audit)

### PR 2: Auto-calculate KV blocks in roofline mode (depends on PR 1)

**Goal:** Derive KV block count from model architecture and GPU memory, matching the llm-d-benchmark reference.

**Changes:**
- New exported calculation function in the latency module — pure function accepting config types and MoE/tied-embedding indicators, returning block count or error
- MoE detection helper (checks HF config for MoE indicator fields per Decision 3)
- Total model weight bytes computation (from architecture, with MoE expert multiplication and tied-embedding handling per Decisions 9-10)
- CLI integration — invoke auto-calculation when in roofline mode and user has not explicitly provided a KV block count (KV-CAP-1)
- Comprehensive test suite (validation against empirical values, monotonicity across TP=1→TP=2 discontinuity, MoE model validation, tied-embedding test, edge cases)

### Key risks

| Risk | Mitigation |
|------|------------|
| Formula accuracy for quantized models | Document as known limitation (Decision 5); `--total-kv-blocks` override available |
| Activation memory constants become stale | 10% tolerance band; constants are stable across vLLM versions |
| Missing GPU memory in user's hardware config | Roofline validation catches this with a clear error (PR 1, KV-CAP-7) |
| TP sharding edge case: `num_kv_heads < TP` | Per-token KV computation uses float division (not integer truncation); validate `TP > 0`. When `num_kv_heads < TP`, vLLM replicates KV heads across ranks — each GPU stores all KV heads, so total KV memory across all GPUs is `TP * per_head_kv` rather than just `num_kv_heads * per_head_kv`. The formula divides total KV bytes by TP, which underestimates per-GPU KV memory in this case, producing *optimistic* (too many) KV blocks. For all models in `defaults.yaml`, `num_kv_heads >= TP` holds, so this edge case does not affect validation targets. If encountered, users should pass `--total-kv-blocks` explicitly. A future refinement could detect this case and apply `per_gpu_kv = max(num_kv_heads, TP) * head_dim * 2 * precision_bytes * num_layers / TP`. |
| MoE weight undercounting | Expert multiplication applied per Decision 9; validated against Mixtral empirical values |
| Tied embedding double-counting | `tie_word_embeddings` check per Decision 10; validated against Qwen2.5-3B |

## Quality Gates

- [x] Extension type identified (configuration enhancement — no new interfaces)
- [x] DES checklist completed (Section 10)
- [x] No prohibited content: no Go structs, no method implementations, no file:line references. Formulas use domain variables (`precision_bytes`, `bytes_per_param`), not Go field names. Section 12 uses module descriptions, not code-level types.
- [x] Every non-obvious decision has alternatives listed with rationale (Section 6, Decisions 1-10)
- [x] Validation strategy specified (Section 9) with automated assertions, cross-validation (one-time verification with hardcoded expected values), model config coverage acknowledgment, sensitivity analysis, and falsification criteria
- [x] Real-system correspondence table included (Section 7)
- [x] Behavioral contracts BC-1 through BC-25 cover: override precedence, tolerance bands, edge cases, GPU name resolution, overflow safety, NaN/Inf rejection, head_dim divisibility, preemption interaction, SwiGLU guard, `num_kv_heads < TP` warning, quantized model warning, floor-zero guard, TP divisibility guard, livelock-zone warning, and `num_local_experts=1` classification
- [x] Extension friction documented for hardware calibration type change (~5 file touches)
- [x] Precedence scenario table with backward compatibility analysis (Section 4)
- [x] All 8 system invariants (INV-1 through INV-8) explicitly addressed in downstream impact note
- [x] Sensitivity analysis covers activation constants, non-torch overhead, and TP=1 worst case (Section 9)
- [x] Model config coverage gaps documented including Llama-4 MoE models and vLLM version spread
- [x] Golden dataset impact assessed (R12) — blackbox golden tests unaffected (KV-CAP-2)
