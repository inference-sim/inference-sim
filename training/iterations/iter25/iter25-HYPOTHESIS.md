# Iteration 25: β₈ Interleaved Architecture Scaling

## Context

Iteration 24 achieved 39.18% loss on Scout experiments by including `β₈·nMoELayers` (427.3 µs/MoE-layer), which captures MoE-specific overhead absent in dense models. However, when evaluating the iter24 formula on Mixtral-8x7B and Mixtral-8x22B experiments, the predictions systematically overestimated latency, particularly for decode-heavy workloads (ITL and E2E metrics).

**Key observation**: Between iter16 (no β₈) and iter24 (β₈=427µs), Mixtral performs:
- **Better** on TTFT (time-to-first-token)
- **Worse** on ITL (inter-token latency) and E2E (end-to-end latency)

The β₈ term helps prefill but compounds excessively during decode, where many tokens accumulate the per-step overhead.

## Architectural Difference: Scout vs Mixtral

### Scout (where β₈ was calibrated)
- **N=16 experts**, k=1 expert/token
- **Interleaved architecture**: `InterleaveMoELayerStep=1` → MoE/dense/MoE/dense alternating
- **24 MoE layers + 24 dense layers** (48 total)
- Dense layers use different FFN dimensions (16384 vs 8192)
- **EP=TP=2** (experts distributed across GPUs)

### Mixtral
- **N=8 experts**, k=2 experts/token
- **Uniform architecture**: All 32 layers are MoE (no interleaving)
- **32 MoE layers** (out of 32 total), 0 dense layers
- **TP=1 or TP=2** (all experts can fit on 1-2 GPUs)

## H-main: β₈ Applies Only to Interleaved MoE Architectures

**Prediction**: The β₈=427µs overhead represents **interleaved architecture synchronization cost** — the overhead of alternating between MoE and dense layer execution within a single forward pass. Uniform MoE architectures (all layers MoE) do not incur this cost.

**Physical Mechanism**:
1. **Kernel switching overhead**: MoE layers use expert-parallel kernels; dense layers use standard GEMM kernels. Switching between these kernel types has overhead:
   - Cache invalidation (different memory access patterns)
   - Kernel launch latency (different dispatch paths)
   - Scheduler state transitions (different parallelization strategies)

2. **Memory pattern disruption**: Interleaving creates:
   - MoE layers: spiky, gather/scatter memory access
   - Dense layers: sequential memory access
   - Alternating between them causes cache thrashing and memory fragmentation

3. **Per-step synchronization**: Each MoE→dense or dense→MoE transition requires:
   - Intermediate activation synchronization
   - Expert routing state cleanup before dense computation
   - Framework overhead managing two different execution paths

**Prediction**: Mixtral (uniform MoE, no interleaving) should have `moeScaling=0.0`, completely dropping the β₈ term. Scout (interleaved) keeps `moeScaling=1.0`.

**Implementation**:
```go
// Check if model has interleaved MoE/dense architecture
hasInterleavedMoE := hw.ModelConfig.InterleaveMoELayerStep > 0

// In StepTime calculation:
var moeScaling float64
if hasInterleavedMoE {
    moeScaling = 1.0  // Apply β₈ for Scout-style interleaved architectures
} else {
    moeScaling = 0.0  // Skip β₈ for uniform MoE (Mixtral, DeepSeek, etc.)
}

moeOverhead := β₈ × moeScaling × nMoELayers
```

**Diagnostic Clause**: If dropping β₈ for Mixtral does NOT improve ITL/E2E predictions:
- The overhead may scale with `k` (experts-per-token) rather than interleaving
- Mixtral's k=2 architecture may have different overhead characteristics
- The β₈ term may be capturing multiple effects (routing + communication + synchronization)

## H-scout-unchanged: Scout Predictions Remain Identical

**Prediction**: All Scout experiment APE values remain byte-identical to iter24. Since Scout has `InterleaveMoELayerStep=1`, the `moeScaling=1.0` preserves the β₈ term exactly as before.

**Mathematical proof**: For Scout,
- iter24: `β₈ × 24 = 427.3 × 24 = 10,255.2 µs`
- iter25: `β₈ × 1.0 × 24 = 427.3 × 1.0 × 24 = 10,255.2 µs` (identical)

**Diagnostic Clause**: If Scout loss changes from 39.179725505241215 (iter24 exact), the implementation has a bug — the scaling factor should be exactly 1.0 for interleaved architectures.

## H-mixtral-improved: Mixtral Predictions Improve Significantly

**Prediction**: Dropping the β₈ term (10,560 µs overhead for Scout's 24 MoE layers) will significantly improve Mixtral ITL and E2E predictions. For Mixtral's 32 MoE layers, iter24 adds:
- `β₈ × 32 = 427.3 × 32 = 13,674 µs = 13.7ms` per step

This is a substantial overhead that compounds across decode tokens. For a 100-token decode:
- Total β₈ overhead: `13.7ms × 100 = 1.37 seconds`

**Expected improvement**: Mixtral ITL/E2E APE should drop by 10-30 percentage points, bringing predictions closer to iter16 levels (which had good Mixtral performance).

**Diagnostic Clause**: If Mixtral predictions do not improve:
- Check that `InterleaveMoELayerStep` is correctly 0 for Mixtral configs
- Verify the binary is rebuilt with the new scaling logic
- Consider alternative hypotheses (k-scaling, EP-based, model-family-specific)

---

## Alternative Hypotheses (Not Implemented)

These approaches were analyzed but not chosen for iter25:

### Hypothesis 2: Single-Expert Routing (k=1) Overhead

**Theory**: k=1 routing has special load-balancing constraints that k≥2 doesn't need.

**Physical mechanism**: k=1 requires each token map to exactly one expert (hard constraint), triggering auxiliary balancing logic. k=2+ has more flexibility (each token → multiple experts).

**Scaling**: `moeScaling = 1/k`
- Scout (k=1): 427µs × 1.0 = 427µs ✓
- Mixtral (k=2): 427µs × 0.5 = 213µs
- DeepSeek-V3 (k=8): 427µs × 0.125 = 53µs

**Why not chosen**:
- Requires understanding functional form of k-dependence
- May over-correct for high-k models (DeepSeek k=8 → 53µs seems too small)
- Not as clearly physics-motivated as interleaved architecture hypothesis

### Hypothesis 3: Dense-MoE Co-Execution Overhead

**Theory**: Running dense and MoE layers in the same forward pass creates memory/compute contention.

**Physical mechanism**: MoE spiky memory access + dense sequential access → cache thrashing.

**Prediction**: Same as H-main (uniform MoE → no thrashing → β₈=0).

**Why not chosen**: Mechanistically similar to H-main but less testable. Cache profiling would be needed to distinguish from kernel switching overhead.

### Hypothesis 4: Scout-Specific vLLM Implementation

**Theory**: vLLM's Scout implementation uses custom kernels not present in standard MoE.

**Evidence**: `DenseIntermediateDim` and `InterleaveMoELayerStep` fields exist specifically for Scout.

**Why not chosen**: Too implementation-specific. Hypothesis 1 (interleaved architecture) is more general and testable across frameworks.

---

## Alternative Scaling Functions (Not Implemented)

These deterministic scaling approaches were considered:

### Option A: Inverse-k Scaling (`f(k) = 1/k`)

**Rationale**: β₈ overhead is per-token (router, permutation), not per-activation. Higher k means more expert activations but same routing overhead per token.

**Scaling**:
- Scout (k=1): 427µs × 1.0 = 427µs ✓
- Mixtral (k=2): 427µs × 0.5 = 213µs
- DeepSeek-V3 (k=8): 427µs × 0.125 = 53µs

**Pros**: Physically motivated (per-token overhead), simple, preserves Scout calibration

**Cons**: Unvalidated heuristic, may over-correct for high-k models

### Option B: Per-Model-Family Coefficients

**Approach**: Train separate coefficient sets for each architecture family:
- `evolved-scout`: β₈=427µs (interleaved)
- `evolved-mixtral`: β₈=0 or β₈=TBD (uniform, k=2)
- `evolved-deepseek`: β₈=TBD (uniform, k=8)

**Pros**: Most accurate per architecture, no unprincipled scaling

**Cons**: Requires maintaining multiple coefficient sets, doesn't generalize to unseen architectures

**Note**: This is effectively what iter25 implements via the binary scaling flag, but with a single coefficient set.

### Option C: Bounded Inverse-k Scaling

**Formula**: `moeScaling = max(0.5, 1.0/k)`

**Effect**: Conservative — prevents scaling below 50% of original overhead.
- Scout (k=1): 427µs × 1.0 = 427µs ✓
- Mixtral (k=2): 427µs × 0.5 = 213µs
- DeepSeek (k=8): 427µs × 0.5 = 213µs (floor at 50%)

**Why not chosen**: Still unvalidated, adds arbitrary floor parameter.

---

## Why Hypothesis 1 (Interleaved Architecture) Was Chosen

1. **Physically grounded**: Architectural feature (interleaving vs uniform) with clear kernel-switching mechanism
2. **Falsifiable**: If non-interleaved MoE models also need β₈, hypothesis fails
3. **Simple to implement**: Binary flag based on `InterleaveMoELayerStep > 0`
4. **Testable prediction**: Mixtral (uniform) should improve, Scout (interleaved) should be unchanged
5. **No new parameters**: Still uses β₈=427µs from iter24, just scales it by 0/1
6. **Conservative**: Binary decision avoids unprincipled fractional scaling

**Long-term path**: After validating Hypothesis 1, collect profiling data (kernel timings, cache miss rates, synchronization points) to refine the scaling function if needed.
