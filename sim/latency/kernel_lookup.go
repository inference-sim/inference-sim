package latency

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// KernelLookupModel predicts step time using per-layer latency lookup tables
// from aiconfigurator's measured GPU kernel database, corrected by learned γ factors.
//
// Step-time formula:
//
//	γ₁·T_pf_gemm + γ₂·T_pf_attn + γ₃·T_dc_gemm + γ₄·T_dc_attn
//	+ γ₆·T_allreduce + γ₇·T_moe
//	+ γ₈·numLayers + γ₉·batchSize + γ₁₀
//
// γ₁-γ₄ and γ₆-γ₇: dimensionless corrections (expected ~1.0 at warm start)
// γ₅: intentionally unused (weight loading removed — double-counts GEMM memory access)
// γ₈: µs/layer overhead; γ₉: µs/request overhead; γ₁₀: µs/step overhead
//
// Basis function conventions (must match Python profile generation script):
//   - T_pf_gemm: context GEMM, interpolated by totalPrefillTokens, × numLayers
//   - T_pf_attn: context attention, interpolated by (numPrefillRequests, avgISL), × numLayers
//   - T_dc_gemm: generation GEMM, interpolated by totalDecodeTokens, × numLayers
//   - T_dc_attn: generation attention, interpolated by (totalDecodeTokens, avgDecodeCtx), × numLayers
//   - T_allreduce: per-step, interpolated by totalTokens, × allReduceUnits
//     where allReduceUnits = 2·numDenseLayers + 1·numMoELayers
//   - T_moe: expert FFN computation, interpolated by totalTokens, × numMoELayers
type KernelLookupModel struct {
	gamma [10]float64 // γ₁-γ₁₀ (gamma[4]=γ₅ unused, kept for index alignment)
	alpha [3]float64  // α₀ (queueing), α₁ (post-decode), α₂ (per-output-token)

	// Pre-loaded lookup tables (per-layer µs)
	contextGemm    Lookup1D
	contextAttn    Lookup2D
	generationGemm Lookup1D
	generationAttn Lookup2D
	allreduce      Lookup1D
	moeCompute     *Lookup1D

	// Architecture (from kernel profile)
	numLayers      int
	numMoELayers   int
	numDenseLayers int
	allReduceUnits int // 2·numDenseLayers + 1·numMoELayers
}

// NewKernelLookupModel creates a KernelLookupModel from BLIS config types.
// Called by the NewLatencyModel factory when hw.Backend == "kernel-lookup".
//
// Requires:
//   - hw.KernelProfilePath != ""  (set via hw.WithKernelProfilePath(path))
//   - len(coeffs.BetaCoeffs) >= 10  (γ₁-γ₁₀; gamma[4]=γ₅ must be 0.0)
//   - len(coeffs.AlphaCoeffs) >= 3  (α₀-α₂)
//   - hw.TP must match profile.TP (when hw.TP > 0)
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

	// Validate runtime TP matches profile TP.
	// hw.TP == 0 means caller didn't specify TP (e.g., blackbox tests) — skip validation.
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
		numDense = profile.NumLayers // dense model fallback
	}

	return &KernelLookupModel{
		gamma:          gamma,
		alpha:          alpha,
		contextGemm:    profile.ContextGemm,
		contextAttn:    profile.ContextAttention,
		generationGemm: profile.GenerationGemm,
		generationAttn: profile.GenerationAttention,
		allreduce:      profile.AllReduce,
		moeCompute:     profile.MoECompute,
		numLayers:      profile.NumLayers,
		numMoELayers:   numMoE,
		numDenseLayers: numDense,
		allReduceUnits: 2*numDense + numMoE,
	}, nil
}

// StepTime computes step time via interpolated kernel lookups.
// Single O(batch_size) pass, zero heap allocations.
func (m *KernelLookupModel) StepTime(batch []*sim.Request) int64 {
	if len(batch) == 0 {
		return 1
	}

	var (
		totalPrefillTokens float64
		numPrefillRequests float64
		sumPrefillISL      float64
		totalDecodeTokens  float64
		sumDecodeCtx       float64
	)
	batchSize := float64(len(batch))
	L := float64(m.numLayers)

	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			totalPrefillTokens += float64(req.NumNewTokens)
			numPrefillRequests++
			sumPrefillISL += float64(len(req.InputTokens))
		} else if len(req.OutputTokens) > 0 {
			totalDecodeTokens++
			sumDecodeCtx += float64(req.ProgressIndex)
		}
	}

	avgPrefillISL := float64(0)
	if numPrefillRequests > 0 {
		avgPrefillISL = sumPrefillISL / numPrefillRequests
	}
	avgDecodeCtx := float64(0)
	if totalDecodeTokens > 0 {
		avgDecodeCtx = sumDecodeCtx / totalDecodeTokens
	}
	totalTokens := totalPrefillTokens + totalDecodeTokens

	// γ₁·T_pf_gemm: context GEMM latency × numLayers (table is per-layer)
	var tPfGemm float64
	if totalPrefillTokens > 0 {
		tPfGemm = clampPositive(m.contextGemm.Interp1D(totalPrefillTokens)) * L
	}

	// γ₂·T_pf_attn: context attention × numLayers
	// Primary axis = numPrefillRequests (batch_size in aiconfigurator)
	// Secondary axis = avgPrefillISL
	var tPfAttn float64
	if numPrefillRequests > 0 {
		tPfAttn = clampPositive(m.contextAttn.Interp2D(numPrefillRequests, avgPrefillISL)) * L
	}

	// γ₃·T_dc_gemm: generation GEMM × numLayers
	var tDcGemm float64
	if totalDecodeTokens > 0 {
		tDcGemm = clampPositive(m.generationGemm.Interp1D(totalDecodeTokens)) * L
	}

	// γ₄·T_dc_attn: generation attention × numLayers
	// Primary axis = totalDecodeTokens (== decode batch size, 1 token/request)
	var tDcAttn float64
	if totalDecodeTokens > 0 {
		tDcAttn = clampPositive(m.generationAttn.Interp2D(totalDecodeTokens, avgDecodeCtx)) * L
	}

	// γ₅ is intentionally skipped (gamma[4] must be 0.0).

	// γ₆·T_allreduce: × allReduceUnits (2·dense + 1·MoE); 0 when TP=1
	var tAllReduce float64
	if m.allReduceUnits > 0 && totalTokens > 0 {
		tAllReduce = clampPositive(m.allreduce.Interp1D(totalTokens)) * float64(m.allReduceUnits)
	}

	// γ₇·T_moe: MoE expert computation × numMoELayers
	var tMoE float64
	if m.moeCompute != nil && m.numMoELayers > 0 && totalTokens > 0 {
		tMoE = clampPositive(m.moeCompute.Interp1D(totalTokens)) * float64(m.numMoELayers)
	}

	stepTime := m.gamma[0]*tPfGemm +
		m.gamma[1]*tPfAttn +
		m.gamma[2]*tDcGemm +
		m.gamma[3]*tDcAttn +
		// gamma[4] = γ₅ intentionally unused (set to 0.0 in training)
		m.gamma[5]*tAllReduce +
		m.gamma[6]*tMoE +
		m.gamma[7]*L +
		m.gamma[8]*batchSize +
		m.gamma[9]

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
