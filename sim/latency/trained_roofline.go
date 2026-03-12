package latency

import (
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// TrainedRooflineLatencyModel estimates latency using analytical roofline basis functions
// with learned correction coefficients fitted from real vLLM traces.
//
// Step-time formula (from training/DESIGN.md):
//
//	β₁·max(T_pf_compute, T_pf_kv) + β₂·max(T_dc_compute, T_dc_kv)
//	+ β₃·T_weight + β₄·T_tp + β₅·L + β₆·batchSize + β₇
//
// Where β₁-β₄ are dimensionless corrections to the roofline model (analytical prior ≈ 1.0),
// β₅ is µs/layer, β₆ is µs/request, β₇ is µs/step. Basis functions compute µs from model
// architecture + hardware specs + batch composition.
//
// Coefficients are from training/output/fit/coefficients.json, fitted via 3-phase NNLS
// from 13 experiments across 4 architectures (137K requests, 7% MAPE GPU combined).
//
// Key differences from the pure RooflineLatencyModel:
//   - No MFU scaling: β₁/β₂ ARE the MFU corrections. Applying MfuPrefill/MfuDecode would double-count.
//   - 3-matrix SwiGLU (6·d·d_ff for FLOPs, 3·d·d_ff for weight bytes), NOT roofline.go's
//     2-matrix convention (mlpMatrixCount()=2). See R23 documented exception.
//   - MoE weight loading uses min(N, max(k, B·k)) effective experts, not all N.
//   - T_tp hardcoded to 0 (β₄=0.0 in current fit; TP communication absorbed into β₅·L).
//
// All fields are frozen at construction. StepTime is allocation-free and uses a single
// O(batch_size) pass over the batch.
type TrainedRooflineLatencyModel struct {
	betaCoeffs  []float64 // [β₁..β₇] from trained_roofline_defaults
	alphaCoeffs []float64 // [α₀, α₁, α₂]

	// Pre-computed architecture features (frozen at construction)
	numLayers  int
	hiddenDim  int // d (hidden_size)
	numHeads   int // H (num_attention_heads)
	headDim    int // d_h = d / H
	dKV        int // kv_heads * d_h (NOT d; differs for GQA)
	dFF        int // intermediate_size (= IntermediateDim, NOT MoEExpertFFNDim)
	kEff       int // max(1, NumExpertsPerTok) — FFN FLOPs multiplier
	numExperts int // NumLocalExperts (0 for dense)
	isMoE      bool
	tp         int

	// Pre-converted hardware specs (FLOP/µs and bytes/µs) for hot-path efficiency.
	// flopsPeakUs = TFlopsPeak * 1e6; bwHbmUs = BwPeakTBs * 1e6.
	flopsPeakUs float64 // FLOP/µs (divide FLOPs by this → µs)
	bwHbmUs     float64 // bytes/µs (divide bytes by this → µs)
}

// bytesPerElement is FP16 = 2 bytes, matching training pipeline's _BYTES_PER_ELEMENT = 2.
const bytesPerElement = 2.0

func (m *TrainedRooflineLatencyModel) StepTime(batch []*sim.Request) int64 {
	if len(batch) == 0 {
		return 1
	}

	// Single-pass accumulation: classify prefill/decode, accumulate all aggregate values.
	// Zero heap allocations — all arithmetic uses stack-local float64 values.
	var (
		totalPrefillTokens float64
		totalDecodeTokens  float64
		sumCtx             float64 // Σ ProgressIndex for decode requests
		prefillAttnFlops   float64 // per-request attention FLOPs sum
	)
	batchSize := float64(len(batch))
	L := float64(m.numLayers)
	d := float64(m.hiddenDim)
	dKV := float64(m.dKV)
	dH := float64(m.headDim)
	dFF := float64(m.dFF)
	tp := float64(m.tp)
	kEff := float64(m.kEff)
	hPerGPU := float64(m.numHeads) / tp

	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			// Prefill: t_i = NumNewTokens, s_i = len(InputTokens) (total prompt length,
			// matching training's entry.prompt_tokens — see Deviation 3)
			ti := float64(req.NumNewTokens)
			si := float64(len(req.InputTokens))
			totalPrefillTokens += ti
			// Per-request attention FLOPs: 4 * (H/TP) * t_i * (s_i + t_i/2) * d_h
			// Accounts for causal masking: average context length is s_i + t_i/2.
			prefillAttnFlops += 4 * hPerGPU * ti * (si + ti/2) * dH
		} else if len(req.OutputTokens) > 0 {
			// Decode: context_length maps to ProgressIndex in BLIS
			// (= inputLen + outputSoFar at StepTime call time, before ProgressIndex++)
			totalDecodeTokens++
			sumCtx += float64(req.ProgressIndex)
		}
	}

	// --- Basis function computation (O(1) from pre-accumulated aggregates) ---

	// T_pf_compute: prefill compute time (µs)
	// FLOPs_proj = L * 2 * T_pf * d * (2*d + 2*d_kv) / TP
	// FLOPs_attn = L * prefillAttnFlops (already accumulated per-request above)
	// FLOPs_ffn  = L * T_pf * k_eff * 6 * d * d_ff / TP
	var tPfCompute float64
	if totalPrefillTokens > 0 {
		flopsProj := L * 2 * totalPrefillTokens * d * (2*d + 2*dKV) / tp
		flopsAttn := L * prefillAttnFlops
		flopsFfn := L * totalPrefillTokens * kEff * 6 * d * dFF / tp
		tPfCompute = (flopsProj + flopsAttn + flopsFfn) / m.flopsPeakUs
	}

	// T_pf_kv: prefill KV cache write bandwidth (µs)
	// Bytes = L * 2 * (kv_heads/TP) * d_h * T_pf * 2
	// Using dKV/TP = (kvHeads * dH) / TP, which is exact (factory validates divisibility).
	var tPfKv float64
	if totalPrefillTokens > 0 {
		bytesPfKv := L * 2 * (dKV / tp) * totalPrefillTokens * bytesPerElement
		tPfKv = bytesPfKv / m.bwHbmUs
	}

	// T_dc_compute: decode compute time (µs)
	// FLOPs_proj = L * 2 * T_dc * d * (2*d + 2*d_kv) / TP
	// FLOPs_attn = L * 4 * (H/TP) * sum_ctx * d_h   (each decode token attends to its context)
	// FLOPs_ffn  = L * T_dc * k_eff * 6 * d * d_ff / TP
	var tDcCompute float64
	if totalDecodeTokens > 0 {
		flopsProj := L * 2 * totalDecodeTokens * d * (2*d + 2*dKV) / tp
		flopsAttn := L * 4 * hPerGPU * sumCtx * dH
		flopsFfn := L * totalDecodeTokens * kEff * 6 * d * dFF / tp
		tDcCompute = (flopsProj + flopsAttn + flopsFfn) / m.flopsPeakUs
	}

	// T_dc_kv: decode KV cache read+write bandwidth (µs)
	// Bytes = L * 2 * (kv_heads/TP) * d_h * 2 * (sum_ctx + T_dc)
	var tDcKv float64
	if totalDecodeTokens > 0 {
		bytesDcKv := L * 2 * (dKV / tp) * bytesPerElement * (sumCtx + totalDecodeTokens)
		tDcKv = bytesDcKv / m.bwHbmUs
	}

	// T_weight: weight loading time (µs)
	// Dense: nEff = 1. MoE: nEff = min(N, max(k, B*k)) where B = total tokens.
	// Uses 3-matrix SwiGLU for FFN weights (R23 exception vs roofline.go's mlpMatrixCount()=2).
	nEff := 1.0
	if m.isMoE {
		B := totalPrefillTokens + totalDecodeTokens
		nEff = math.Min(float64(m.numExperts), math.Max(kEff, B*kEff))
	}
	bytesAttn := L * d * (2*d + 2*dKV) * bytesPerElement / tp
	bytesFfn := L * nEff * 3 * d * dFF * bytesPerElement / tp
	tWeight := (bytesAttn + bytesFfn) / m.bwHbmUs

	// T_tp: TP communication time (µs) — hardcoded to 0.
	// β₄=0.0 in current fit; TP communication cost absorbed into β₅·L (H100 NVLink specific).
	// No NVLink bandwidth data in HardwareCalib. See Deviation 2.
	tTp := 0.0

	// 7-term step-time formula
	stepTime := m.betaCoeffs[0]*math.Max(tPfCompute, tPfKv) +
		m.betaCoeffs[1]*math.Max(tDcCompute, tDcKv) +
		m.betaCoeffs[2]*tWeight +
		m.betaCoeffs[3]*tTp +
		m.betaCoeffs[4]*L +
		m.betaCoeffs[5]*batchSize +
		m.betaCoeffs[6]

	return max(1, int64(stepTime))
}

func (m *TrainedRooflineLatencyModel) QueueingTime(req *sim.Request) int64 {
	// α₀ = API processing overhead (ARRIVED → QUEUED), constant per-request.
	// Unlike other backends, this does NOT scale with input length.
	return int64(m.alphaCoeffs[0])
}

func (m *TrainedRooflineLatencyModel) OutputTokenProcessingTime() int64 {
	// α₂ = per-output-token detokenization cost (µs/token).
	return int64(m.alphaCoeffs[2])
}

func (m *TrainedRooflineLatencyModel) PostDecodeFixedOverhead() int64 {
	// α₁ = fixed per-request post-decode overhead (µs).
	// Added to E2E in recordRequestCompletion, NOT to TTFT.
	return int64(m.alphaCoeffs[1])
}
