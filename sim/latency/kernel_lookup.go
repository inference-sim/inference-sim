package latency

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// KernelLookupModel predicts step time using per-layer latency lookup tables
// from aiconfigurator's measured GPU kernel database, corrected by learned γ factors.
//
// Step-time formula (8-term, fused GEMM):
//
//	γ₁·T_gemm + γ₂·T_pf_attn + γ₃·T_dc_attn
//	+ γ₅·T_allreduce + γ₆·T_moe
//	+ γ₇·numLayers + γ₈·batchSize + γ₉
//
// γ₁-γ₃ and γ₅-γ₆: dimensionless corrections (expected ~1.0 at warm start)
// γ₄: unused (was decode GEMM before fusion; kept for coefficient index alignment)
// γ₇: µs/layer overhead; γ₈: µs/request overhead; γ₉: µs/step overhead
//
// Why fused GEMM: In vLLM continuous batching, prefill and decode tokens are
// concatenated into the SAME GEMM call per layer. Splitting them into separate
// lookups double-counts kernel launch overhead (~150µs/layer), causing 1.5-2x
// overestimation for mixed batches. Attention stays split because vLLM uses
// different kernels (FlashAttention for prefill, PagedAttention for decode).
//
// Basis function conventions (must match Python profile generation script):
//   - T_gemm: fused GEMM, interpolated by totalTokens (prefill + decode), × numLayers
//   - T_pf_attn: FlashAttention, interpolated by (numPrefillRequests, avgFullS), × numLayers,
//     then multiplied by prefixCorrection = Σ(full_s²−prefix²)/Σ(full_s²) where
//     full_s = ProgressIndex+NumNewTokens and prefix = ProgressIndex.
//     Matches aiconfigurator's (full_s²−prefix²)/full_s² scaling: only the s=NumNewTokens
//     new tokens generate attention outputs while attending the full_s KV context.
//     For non-chunked prefill (ProgressIndex=0) correction=1; for chunked prefill < 1.
//   - T_dc_attn: PagedAttention, interpolated by (totalDecodeTokens, avgDecodeCtx), × numLayers
//   - T_allreduce: bandwidth-only component (overhead subtracted), × allReduceUnits; 0 when TP=1
//     Raw measurement includes ~constant NCCL kernel launch overhead (amortized in CUDA graphs);
//     allReduceOverhead (latency at tokens=1) is subtracted before multiplying.
//   - T_moe: expert FFN computation, interpolated by totalTokens, × numMoELayers
type KernelLookupModel struct {
	gamma [10]float64 // gamma[0]=γ₁(gemm), [1]=γ₂(pf_attn), [2]=γ₃(dc_attn),
	//                   [3]=γ₄(unused), [4]=γ₅(allreduce), [5]=γ₆(moe),
	//                   [6]=γ₇(per-layer), [7]=γ₈(per-req), [8]=γ₉(per-step)
	//                   [9] reserved
	alpha [3]float64 // α₀ (queueing), α₁ (post-decode), α₂ (per-output-token)

	// Pre-loaded lookup tables (per-layer µs)
	gemm           Lookup1D // fused GEMM: total tokens → per-layer latency
	contextAttn    Lookup2D // FlashAttention: (batch_size, ISL) → per-layer latency
	generationAttn Lookup2D // PagedAttention: (decode_tokens, context) → per-layer latency
	allreduce      Lookup1D // AllReduce: total tokens → per-invocation latency
	moeCompute     *Lookup1D

	// Architecture (from kernel profile)
	numLayers      int
	numMoELayers   int
	numDenseLayers int
	allReduceUnits    int     // 2·numDenseLayers + 1·numMoELayers
	allReduceOverhead float64 // per-call NCCL kernel overhead (µs), measured at tokens=1
	// The allreduce profile includes two components:
	//   (a) NVLink data-transfer time: scales with tokens × hidden_dim → kept
	//   (b) NCCL kernel launch overhead: ~constant per call → subtracted
	// In vLLM CUDA graphs, (b) is amortized to near-zero across all layer
	// kernels. We subtract the t=1 measurement as a proxy for (b) so that
	// the effective allreduce term models only the bandwidth-limited component.
}

// NewKernelLookupModel creates a KernelLookupModel from BLIS config types.
// Called by the NewLatencyModel factory when hw.Backend == "kernel-lookup".
func NewKernelLookupModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (sim.LatencyModel, error) {
	if hw.KernelProfilePath == "" {
		return nil, fmt.Errorf("kernel-lookup: KernelProfilePath must be set; " +
			"use hw.WithKernelProfilePath(path) or --kernel-profile flag")
	}
	if len(coeffs.BetaCoeffs) < 10 {
		return nil, fmt.Errorf("kernel-lookup: requires 10 gamma coefficients "+
			"(BetaCoeffs[0..9] = γ₁-γ₁₀), got %d", len(coeffs.BetaCoeffs))
	}
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("kernel-lookup: requires 3 alpha coefficients, got %d",
			len(coeffs.AlphaCoeffs))
	}
	if err := validateCoeffs("BetaCoeffs/gamma", coeffs.BetaCoeffs[:10]); err != nil {
		return nil, err
	}

	profile, err := LoadKernelProfile(hw.KernelProfilePath)
	if err != nil {
		return nil, fmt.Errorf("kernel-lookup: %w", err)
	}

	if hw.TP > 0 && hw.TP != profile.TP {
		return nil, fmt.Errorf("kernel-lookup: runtime TP=%d does not match profile TP=%d (profile: %q)",
			hw.TP, profile.TP, hw.KernelProfilePath)
	}

	var gamma [10]float64
	copy(gamma[:], coeffs.BetaCoeffs[:10])
	var alpha [3]float64
	copy(alpha[:], coeffs.AlphaCoeffs[:3])

	numDense := profile.NumDenseLayers
	numMoE := profile.NumMoELayers
	if numDense == 0 && numMoE == 0 {
		numDense = profile.NumLayers
	}

	// Compute per-call NCCL overhead as the allreduce latency at the smallest
	// measured token count (tokens=1). This is almost entirely kernel launch
	// overhead — in CUDA graph mode it is amortized to near-zero, so we subtract
	// it from every allreduce lookup to leave only the bandwidth-scaling component.
	arOverhead := float64(0)
	if len(profile.AllReduce.Tokens) > 0 {
		arOverhead = profile.AllReduce.LatencyUs[0] // latency at tokens[0] ≈ 1
	}

	return &KernelLookupModel{
		gamma:             gamma,
		alpha:             alpha,
		gemm:              profile.Gemm,
		contextAttn:       profile.ContextAttention,
		generationAttn:    profile.GenerationAttention,
		allreduce:         profile.AllReduce,
		moeCompute:        profile.MoECompute,
		numLayers:         profile.NumLayers,
		numMoELayers:      numMoE,
		numDenseLayers:    numDense,
		allReduceUnits:    2*numDense + numMoE,
		allReduceOverhead: arOverhead,
	}, nil
}

// StepTime computes step time via interpolated kernel lookups.
// Single O(batch_size) pass, zero heap allocations.
func (m *KernelLookupModel) StepTime(batch []*sim.Request) int64 {
	if len(batch) == 0 {
		return 1
	}

	var (
		totalPrefillTokens  float64
		numPrefillRequests  float64
		sumPrefillFullS     float64 // Σ full_s = Σ(ProgressIndex + NumNewTokens)
		sumPrefillFullSsq   float64 // Σ full_s², denominator for prefix correction
		sumPrefillAttended  float64 // Σ(full_s² − prefix²), numerator for prefix correction
		totalDecodeTokens   float64
		sumDecodeCtx        float64
	)
	batchSize := float64(len(batch))
	L := float64(m.numLayers)

	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			newT := float64(req.NumNewTokens)
			prefix := float64(req.ProgressIndex)
			fullS := prefix + newT
			totalPrefillTokens += newT
			numPrefillRequests++
			sumPrefillFullS += fullS
			sumPrefillFullSsq += fullS * fullS
			// attended = full_s² − prefix² = NumNewTokens × (2·prefix + NumNewTokens)
			// Matches aiconfigurator's prefix_correction = (full_s²−prefix²)/full_s²,
			// which accounts for only the s new tokens generating attention outputs
			// while attending to the full full_s KV context.
			sumPrefillAttended += newT * (2*prefix + newT)
		} else if len(req.OutputTokens) > 0 {
			totalDecodeTokens++
			sumDecodeCtx += float64(req.ProgressIndex)
		}
	}

	avgPrefillFullS := float64(0)
	// prefixCorrection = Σ(full_s²−prefix²) / Σ(full_s²).
	// = 1.0 for non-chunked prefill (all ProgressIndex=0), so no cost for the common case.
	// < 1.0 for chunked prefill, discounting attention by the fraction of KV context
	// that belongs to already-processed prefix tokens (matching aiconfigurator semantics).
	prefixCorrection := float64(1)
	if numPrefillRequests > 0 {
		avgPrefillFullS = sumPrefillFullS / numPrefillRequests
		if sumPrefillFullSsq > 0 {
			prefixCorrection = sumPrefillAttended / sumPrefillFullSsq
		}
	}
	avgDecodeCtx := float64(0)
	if totalDecodeTokens > 0 {
		avgDecodeCtx = sumDecodeCtx / totalDecodeTokens
	}
	totalTokens := totalPrefillTokens + totalDecodeTokens

	// γ₁·T_gemm: FUSED GEMM for all tokens (prefill + decode concatenated) × numLayers
	// Matches vLLM continuous batching where one GEMM processes the entire token batch.
	var tGemm float64
	if totalTokens > 0 {
		tGemm = clampPositive(m.gemm.Interp1D(totalTokens)) * L
	}

	// γ₂·T_pf_attn: FlashAttention for prefill tokens × numLayers, prefix-corrected.
	// Lookup at avgPrefillFullS (= avg context length = prefix + new_tokens), then
	// scale by prefixCorrection = Σ(full_s²−prefix²)/Σ(full_s²) to match aiconfigurator's
	// semantics: only the s new tokens generate outputs, but they attend to full_s context.
	var tPfAttn float64
	if numPrefillRequests > 0 {
		tPfAttn = clampPositive(m.contextAttn.Interp2D(numPrefillRequests, avgPrefillFullS)) * L * prefixCorrection
	}

	// γ₃·T_dc_attn: PagedAttention for decode tokens × numLayers
	var tDcAttn float64
	if totalDecodeTokens > 0 {
		tDcAttn = clampPositive(m.generationAttn.Interp2D(totalDecodeTokens, avgDecodeCtx)) * L
	}

	// γ₅·T_allreduce: bandwidth-limited component only, × allReduceUnits; 0 when TP=1.
	// The raw measurement includes per-call NCCL kernel overhead (~constant, amortized
	// in CUDA graphs). Subtract allReduceOverhead (latency at tokens=1 ≈ launch cost)
	// to isolate the NVLink data-transfer component that actually scales with batch size.
	var tAllReduce float64
	if m.allReduceUnits > 0 && totalTokens > 0 {
		rawAR := m.allreduce.Interp1D(totalTokens)
		tAllReduce = clampPositive(rawAR-m.allReduceOverhead) * float64(m.allReduceUnits)
	}

	// γ₆·T_moe: MoE expert computation × numMoELayers
	var tMoE float64
	if m.moeCompute != nil && m.numMoELayers > 0 && totalTokens > 0 {
		tMoE = clampPositive(m.moeCompute.Interp1D(totalTokens)) * float64(m.numMoELayers)
	}

	stepTime := m.gamma[0]*tGemm +
		m.gamma[1]*tPfAttn +
		m.gamma[2]*tDcAttn +
		// gamma[3] = γ₄ unused (was split decode GEMM, now fused into γ₁)
		m.gamma[4]*tAllReduce +
		m.gamma[5]*tMoE +
		m.gamma[6]*L +
		m.gamma[7]*batchSize +
		m.gamma[8]
		// gamma[9] reserved

	return max(1, clampToInt64(stepTime))
}

// QueueingTime returns per-request API processing overhead (ARRIVED → QUEUED).
func (m *KernelLookupModel) QueueingTime(_ *sim.Request) int64 {
	return clampToInt64(m.alpha[0])
}

// OutputTokenProcessingTime returns per-output-token streaming/detokenization overhead.
func (m *KernelLookupModel) OutputTokenProcessingTime() int64 {
	return clampToInt64(m.alpha[2])
}

// PostDecodeFixedOverhead returns fixed per-request overhead applied at completion.
func (m *KernelLookupModel) PostDecodeFixedOverhead() int64 {
	return clampToInt64(m.alpha[1])
}
