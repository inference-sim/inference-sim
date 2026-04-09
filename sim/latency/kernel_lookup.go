package latency

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// KernelLookupModel predicts step time using per-layer latency lookup tables
// from aiconfigurator's measured GPU kernel database, corrected by learned γ factors.
//
// # Step-time formula (iter35, fixed overhead terms):
//
//	γ₁·T_gemm + γ₂·T_pf_attn + γ₃·T_dc_attn
//	+ γ₅·T_allreduce + γ₆·T_moe
//	+ γ₇_pf·L·numPrefillSeqs + γ₇_dc·L + γ₈·batchSize + γ₉
//
// # Overhead term history:
//
//	iter33: γ₇_dc × L / √decodeTokens  — UNPHYSICAL: total overhead
//	  decreased as decode batch grew (more work = less time).
//	iter34: retired γ₇_dc, added floor subtraction + γ_wt — revealed
//	  that γ₇_dc was providing necessary per-step overhead.
//	iter35 (current): γ₇_dc × L — constant per-layer per-step overhead,
//	  physically sensible: fixed scheduling cost per transformer layer,
//	  independent of batch size. Model-depth dependent (more layers = more).
//
// # Coefficient index map
//
//	gamma[0] = γ₁    GEMM + logits correction          warm ~0.086
//	gamma[1] = γ₂    FlashAttention correction          warm ~0.302
//	gamma[2] = γ₃    PagedAttention correction          warm ~0.939
//	gamma[3]         RESERVED (was γ₄ unused, then γ_wt in iter34)
//	gamma[4] = γ₅    AllReduce (bandwidth only)         warm ~0.009
//	gamma[5] = γ₆    MoE expert compute                 warm ~1.0
//	gamma[6] = γ₇_pf per-layer per-prefill-seq (µs)    warm ~15 µs/L
//	gamma[7] = γ₈    per-request overhead (µs)          warm ~86 µs
//	gamma[8] = γ₉    per-step constant overhead (µs)    warm ~52 µs
//	gamma[9] = γ₇_dc per-layer constant overhead (µs)  warm ~0–50 µs/L
//
// # Basis function conventions
//
//   - T_gemm: fused GEMM, interpolated by totalTokens (prefill + decode), × numLayers
//   - T_pf_attn: FlashAttention, (numPrefillRequests, avgFullS) × numLayers × prefixCorrection.
//     First step (ProgressIndex=0): full_s = len(InputTokens) (covers KV prefix cache hits).
//     Chunked second+ step (ProgressIndex>0): full_s = ProgressIndex + NumNewTokens.
//   - T_dc_attn: PagedAttention, (totalDecodeTokens, avgDecodeCtx) × numLayers
//   - T_allreduce: floor-subtracted (allReduceOverhead), × allReduceUnits; 0 when TP=1
//   - T_moe: expert FFN, × numMoELayers
type KernelLookupModel struct {
	gamma [10]float64
	alpha [3]float64

	// Pre-loaded lookup tables (per-layer µs)
	gemm           Lookup1D
	contextAttn    Lookup2D
	generationAttn Lookup2D
	allreduce      Lookup1D
	moeCompute     *Lookup1D
	logitsGemm     *Lookup1D

	// Architecture (from kernel profile)
	numLayers      int
	numMoELayers   int
	numDenseLayers int
	allReduceUnits    int
	allReduceOverhead float64
}

// NewKernelLookupModel creates a KernelLookupModel from BLIS config types.
func NewKernelLookupModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (sim.LatencyModel, error) {
	if hw.KernelProfilePath == "" {
		return nil, fmt.Errorf("kernel-lookup: KernelProfilePath must be set; " +
			"use hw.WithKernelProfilePath(path) or --kernel-profile flag")
	}
	if len(coeffs.BetaCoeffs) < 10 {
		return nil, fmt.Errorf("kernel-lookup: requires 10 gamma coefficients "+
			"(BetaCoeffs[0..9]), got %d", len(coeffs.BetaCoeffs))
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

	// AllReduce overhead: latency at tokens=1, subtracted to isolate bandwidth component.
	arOverhead := float64(0)
	if len(profile.AllReduce.Tokens) > 0 {
		arOverhead = profile.AllReduce.LatencyUs[0]
	}

	return &KernelLookupModel{
		gamma:             gamma,
		alpha:             alpha,
		gemm:              profile.Gemm,
		contextAttn:       profile.ContextAttention,
		generationAttn:    profile.GenerationAttention,
		allreduce:         profile.AllReduce,
		moeCompute:        profile.MoECompute,
		logitsGemm:        profile.LogitsGemm,
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
		sumPrefillFullS     float64
		sumPrefillFullSsq   float64
		sumPrefillAttended  float64
		totalDecodeTokens   float64
		sumDecodeCtx        float64
	)
	batchSize := float64(len(batch))
	L := float64(m.numLayers)

	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			newT := float64(req.NumNewTokens)

			// full_s = full KV context; prefix = already-in-KV tokens.
			// ProgressIndex==0 (first prefill step): use len(InputTokens) so
			// KV prefix cache hits are correctly modelled (iter33 fix, preserved).
			var prefix, fullS float64
			if req.ProgressIndex == 0 {
				fullS  = float64(util.Len64(req.InputTokens))
				prefix = fullS - newT
			} else {
				prefix = float64(req.ProgressIndex)
				fullS  = prefix + newT
			}

			totalPrefillTokens += newT
			numPrefillRequests++
			sumPrefillFullS += fullS
			sumPrefillFullSsq += fullS * fullS
			sumPrefillAttended += newT * (2*prefix + newT)
		} else if len(req.OutputTokens) > 0 {
			totalDecodeTokens++
			sumDecodeCtx += float64(req.ProgressIndex)
		}
	}

	avgPrefillFullS := float64(0)
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

	// γ₁·T_gemm: fused GEMM for all tokens × numLayers.
	var tGemm float64
	if totalTokens > 0 {
		tGemm = clampPositive(m.gemm.Interp1D(totalTokens)) * L
	}

	// γ₁·T_logits: vocabulary projection GEMM (once per step, not per-layer).
	var tLogits float64
	if m.logitsGemm != nil && totalTokens > 0 {
		tLogits = clampPositive(m.logitsGemm.Interp1D(totalTokens))
	}

	// γ₂·T_pf_attn: FlashAttention × numLayers, prefix-corrected.
	var tPfAttn float64
	if numPrefillRequests > 0 {
		tPfAttn = clampPositive(m.contextAttn.Interp2D(numPrefillRequests, avgPrefillFullS)) * L * prefixCorrection
	}

	// γ₃·T_dc_attn: PagedAttention × numLayers.
	var tDcAttn float64
	if totalDecodeTokens > 0 {
		tDcAttn = clampPositive(m.generationAttn.Interp2D(totalDecodeTokens, avgDecodeCtx)) * L
	}

	// γ₅·T_allreduce: bandwidth-limited × allReduceUnits; 0 when TP=1.
	var tAllReduce float64
	if m.allReduceUnits > 0 && totalTokens > 0 {
		rawAR := m.allreduce.Interp1D(totalTokens)
		tAllReduce = clampPositive(rawAR-m.allReduceOverhead) * float64(m.allReduceUnits)
	}

	// γ₆·T_moe: MoE expert computation × numMoELayers.
	var tMoE float64
	if m.moeCompute != nil && m.numMoELayers > 0 && totalTokens > 0 {
		tMoE = clampPositive(m.moeCompute.Interp1D(totalTokens)) * float64(m.numMoELayers)
	}

	// γ₇_pf (gamma[6]): per-layer overhead per prefill sequence in batch.
	tPfOverhead := m.gamma[6] * L * numPrefillRequests

	// γ₇_dc (gamma[9]): constant per-layer overhead when decode is active (µs/layer).
	// Fixed from the unphysical 1/√batch form used in iter33.
	// Physically: fixed CUDA scheduling cost per layer during decode steps,
	// independent of how many decode requests are in the batch.
	// Conditional on decode being present — avoids inflating prefill TTFT.
	var tDcOverhead float64
	if totalDecodeTokens > 0 {
		tDcOverhead = m.gamma[9] * L
	}

	stepTime := m.gamma[0]*(tGemm+tLogits) +
		m.gamma[1]*tPfAttn +
		m.gamma[2]*tDcAttn +
		// gamma[3] reserved (was γ₄ unused; γ_wt in iter34 experiment)
		m.gamma[4]*tAllReduce +
		m.gamma[5]*tMoE +
		tPfOverhead +
		tDcOverhead +
		m.gamma[7]*batchSize +
		m.gamma[8]

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

