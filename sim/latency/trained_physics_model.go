package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// TrainedPhysicsModel implements physics-informed latency model with learned correction
// coefficients.
//
// Iteration 29: Sequential golden section search optimization (loss: 34.57%).
//
// BACKGROUND: Iterations 0-14 used a compute-only decode formula that dropped the
// memory-bandwidth floor from the roofline model. This caused per-request overhead
// terms (β₃, β₇) to inflate 100-1000× to compensate, producing O(N×B) accumulated
// phantom overhead in predicted E2E latency. The trained-roofline backend (7% MAPE
// on 13 experiments) already solves this with max(compute, memory) basis functions.
//
// ITER29 STRATEGY: Use trained-roofline's proven 7-term formula with three
// dataset-specific enhancements:
//
//  1. InterleaveMoELayerStep/DenseIntermediateDim: Split FLOPs and weight bytes
//     between MoE and dense layers for Scout (#877). The trained-roofline treats
//     all layers identically, which is wrong for interleaved MoE/dense architectures.
//
//  2. EffectiveWeightBytesPerParam: Use quantization-aware weight precision instead
//     of hardcoded FP16. Scout uses FP8 (1 byte/param vs 2), which halves weight
//     loading time. The trained-roofline hardcodes bytesPerElement=2.0.
//
//  3. FP8 peak FLOPS: Select TFlopsFP8 for FP8 models on GPUs with native FP8
//     tensor cores (H100), matching roofline.go's logic.
//
// Step-time formula (up to 10-term, extends trained-roofline with MoE overhead
// and optional prefill/decode compute/memory splits):
//
// With 8 betas (default):
//
//	β₁·max(T_pf_compute, T_pf_kv) + β₂·max(T_dc_compute, T_dc_kv)
//	+ β₃·T_weight + β₄·T_tp + β₅·L + β₆·batchSize + β₇
//	+ β₈·nMoELayers
//
// With 9 betas (prefill split — prefill is compute-dominated):
//
//	β₁ₐ·T_pf_compute + β₁ᵦ·T_pf_kv + β₂·max(T_dc_compute, T_dc_kv)
//	+ β₃·T_weight + β₄·T_tp + β₅·L + β₆·batchSize + β₇
//	+ β₈·nMoELayers
//
// With 10 betas (prefill + decode split — decode is memory-dominated):
//
//	β₁ₐ·T_pf_compute + β₁ᵦ·T_pf_kv + β₂ₐ·T_dc_compute + β₂ᵦ·T_dc_kv
//	+ β₃·T_weight + β₄·T_tp + β₅·L + β₆·batchSize + β₇
//	+ β₈·nMoELayers
//
// Physical insight: prefill is compute-bound (FlashAttention), decode is
// memory-bound (single-token bandwidth). Each uses only its bottleneck term.
//
// Where β₁/β₁ₐ-β₃ are dimensionless roofline corrections (analytical prior ≈ 1.0),
// β₁ᵦ (β₉) is prefill memory correction, β₄ is TP All-Reduce correction (absorbs NVLink/HBM bandwidth ratio; ~0.27 on H100),
// β₅ is µs/layer, β₆ is µs/request, β₇ is µs/step,
// β₈ is µs/MoE-layer (router gating + token permutation + EP communication overhead;
// zero for dense models since nMoELayers=0).
//
// Alpha coefficients:
//   - α₀: QueueingTime — fixed per-request API processing overhead (µs)
//   - α₁: PostDecodeFixedOverhead — fixed per-request post-decode overhead (µs)
//   - α₂: OutputTokenProcessingTime — per-output-token overhead (µs/token)
type TrainedPhysicsModel struct {
	Alpha [3]float64 // [α₀, α₁, α₂]
	Beta  []float64  // [β₁..β₁₀] — 7-10 coefficients (7→β₈=0, 8→MoE, 9→pf split, 10→dc split)

	// Mode flags.
	prefillSplit bool // true when ≥9 betas: β₁ₐ·compute + β₁ᵦ·kv instead of β₁·max
	decodeSplit  bool // true when ≥10 betas: β₂ₐ·compute + β₂ᵦ·kv instead of β₂·max

	// Pre-computed architecture features (frozen at construction).
	numLayers         int
	numMoELayers      int // Interleaved MoE layers (0 for dense models)
	numDenseLayers    int // Dense layers (= numLayers for dense models)
	hiddenDim         int
	numHeads          int
	headDim           int     // d_h = hiddenDim / numHeads
	dKV               int     // kvHeads * d_h (differs from hiddenDim for GQA)
	dFF               int     // Default FFN intermediate dim
	dFFMoE            int     // MoE expert FFN dim (may differ from dFF)
	dFFDense          int     // Dense layer FFN dim (may differ for interleaved archs)
	kEff              int     // max(1, NumExpertsPerTok)
	numExperts        int     // NumLocalExperts (0 for dense)
	isMoE             bool    // NumLocalExperts > 0
	hasInterleavedMoE bool    // InterleaveMoELayerStep > 0 (Scout-style alternating MoE/dense)
	tp                int     // Tensor parallelism degree
	weightBPP         float64 // EffectiveWeightBytesPerParam (FP8-aware)

	// Pre-converted hardware specs for hot-path efficiency.
	flopsPeakUs float64 // FLOP/µs (divide FLOPs by this → µs)
	bwHbmUs     float64 // bytes/µs (divide bytes by this → µs)
}

// bytesPerKVElement is 2 bytes (FP16) for KV cache, matching vLLM's default.
// KV cache uses FP16 regardless of weight quantization.
const bytesPerKVElement = 2.0

// StepTime computes vLLM step execution time using roofline basis functions
// with learned correction coefficients.
//
// Single O(batch_size) pass, zero heap allocations.
func (m *TrainedPhysicsModel) StepTime(batch []*sim.Request) int64 {
	if len(batch) == 0 {
		return 1
	}

	// Single-pass accumulation: classify prefill/decode, accumulate aggregates.
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
	tp := float64(m.tp)
	kEff := float64(m.kEff)
	hPerGPU := float64(m.numHeads) / tp

	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			// Prefill
			ti := float64(req.NumNewTokens)
			si := float64(len(req.InputTokens))
			totalPrefillTokens += ti
			prefillAttnFlops += 4 * hPerGPU * ti * (si + ti/2) * dH
		} else if len(req.OutputTokens) > 0 {
			// Decode
			totalDecodeTokens++
			sumCtx += float64(req.ProgressIndex)
		}
	}

	// ─── Basis function computation ────────────────────────────────────

	// T_pf_compute: prefill compute time (µs)
	// Enhancement: split FLOPs between MoE and dense layers for interleaved architectures.
	var tPfCompute float64
	if totalPrefillTokens > 0 {
		flopsProj := L * 2 * totalPrefillTokens * d * (2*d + 2*dKV) / tp
		flopsAttn := L * prefillAttnFlops

		// MLP FLOPs: split between MoE and dense layers (#877 fix)
		var flopsFfn float64
		if m.numMoELayers > 0 {
			flopsFfn += float64(m.numMoELayers) * totalPrefillTokens * kEff * 6 * d * float64(m.dFFMoE) / tp
		}
		if m.numDenseLayers > 0 {
			flopsFfn += float64(m.numDenseLayers) * totalPrefillTokens * 1 * 6 * d * float64(m.dFFDense) / tp
		}

		tPfCompute = (flopsProj + flopsAttn + flopsFfn) / m.flopsPeakUs
	}

	// T_pf_kv: prefill KV cache write bandwidth (µs)
	var tPfKv float64
	if totalPrefillTokens > 0 {
		bytesPfKv := L * 2 * (dKV / tp) * totalPrefillTokens * bytesPerKVElement
		tPfKv = bytesPfKv / m.bwHbmUs
	}

	// T_dc_compute: decode compute time (µs)
	// Enhancement: split FLOPs between MoE and dense layers.
	var tDcCompute float64
	if totalDecodeTokens > 0 {
		flopsProj := L * 2 * totalDecodeTokens * d * (2*d + 2*dKV) / tp
		flopsAttn := L * 4 * hPerGPU * sumCtx * dH

		var flopsFfn float64
		if m.numMoELayers > 0 {
			flopsFfn += float64(m.numMoELayers) * totalDecodeTokens * kEff * 6 * d * float64(m.dFFMoE) / tp
		}
		if m.numDenseLayers > 0 {
			flopsFfn += float64(m.numDenseLayers) * totalDecodeTokens * 1 * 6 * d * float64(m.dFFDense) / tp
		}

		tDcCompute = (flopsProj + flopsAttn + flopsFfn) / m.flopsPeakUs
	}

	// T_dc_kv: decode KV cache read+write bandwidth (µs)
	var tDcKv float64
	if totalDecodeTokens > 0 {
		bytesDcKv := L * 2 * (dKV / tp) * bytesPerKVElement * (sumCtx + totalDecodeTokens)
		tDcKv = bytesDcKv / m.bwHbmUs
	}

	// T_weight: weight loading time (µs)
	// Enhancement: use EffectiveWeightBytesPerParam (FP8-aware) and split MoE/dense.
	// MoE: nEff = min(N, max(k, B*k)) effective experts per step.
	nEff := 1.0
	if m.isMoE {
		B := totalPrefillTokens + totalDecodeTokens
		nEff = math.Min(float64(m.numExperts), math.Max(kEff, B*kEff))
	}
	bpp := m.weightBPP
	bytesAttn := L * d * (2*d + 2*dKV) * bpp / tp

	// MoE and dense layers have different FFN dims and different weight loading
	var bytesFfn float64
	if m.numMoELayers > 0 {
		bytesFfn += float64(m.numMoELayers) * nEff * 3 * d * float64(m.dFFMoE) * bpp / tp
	}
	if m.numDenseLayers > 0 {
		bytesFfn += float64(m.numDenseLayers) * 1 * 3 * d * float64(m.dFFDense) * bpp / tp
	}
	tWeight := (bytesAttn + bytesFfn) / m.bwHbmUs

	// T_tp: TP All-Reduce communication time (µs)
	//
	// Each transformer layer performs All-Reduces over NVLink for the attention
	// sublayers. Dense layers also All-Reduce their FFN; MoE layers use EP
	// All-to-All instead (captured by β₈). We count All-Reduce "units" as:
	//   dense layer → 2 units (attention + FFN)
	//   MoE layer   → 1 unit  (attention only; FFN replaced by EP All-to-All)
	//
	// Volume per unit: totalTokens × hiddenDim × 2 bytes (BF16) × 2 (ring phases)
	// Denominator: bwHbmUs normalises to µs; β₄ absorbs NVLink/HBM ratio (~0.27 on H100)
	//
	// Generalisation:
	//   TP=1 → (TP-1)/TP = 0 → tTp = 0 (no communication)
	//   Dense-only model → numMoELayers=0 → units = 2·numDenseLayers
	//   Mixtral (all MoE) → numDenseLayers=0 → units = numMoELayers (half of dense equivalent)
	var tTp float64
	if m.tp > 1 {
		totalTokens := totalPrefillTokens + totalDecodeTokens
		allReduceUnits := float64(2*m.numDenseLayers + m.numMoELayers)
		tpFactor := float64(m.tp-1) / float64(m.tp)
		tTp = allReduceUnits * totalTokens * float64(m.hiddenDim) * 2.0 * 2.0 * tpFactor / m.bwHbmUs
	}

	// ─── Step-time formula ─────────────────────────────────────────────
	//
	// Prefill term: β₁·max(compute, kv) when 8 betas,
	//               β₁ₐ·compute + β₁ᵦ·kv when 9 betas (prefill split).
	var prefillTerm float64
	if m.prefillSplit {
		prefillTerm = m.Beta[0]*tPfCompute + m.Beta[8]*tPfKv
	} else {
		prefillTerm = m.Beta[0] * math.Max(tPfCompute, tPfKv)
	}

	// Decode term: β₂·max(compute, kv) when ≤9 betas,
	//              β₂ₐ·compute + β₂ᵦ·kv when 10 betas (decode is memory-dominated).
	var decodeTerm float64
	if m.decodeSplit {
		decodeTerm = m.Beta[1]*tDcCompute + m.Beta[9]*tDcKv
	} else {
		decodeTerm = m.Beta[1] * math.Max(tDcCompute, tDcKv)
	}

	// β₈ MoE overhead: Applies only to interleaved MoE architectures.
	// Hypothesis: β₈=427µs represents interleaved MoE/dense synchronization overhead:
	//   - Kernel switching between MoE (expert-parallel) and dense (GEMM) layers
	//   - Cache effects from alternating memory access patterns
	//   - Scheduler state transitions between different layer types
	// Scout (InterleaveMoELayerStep=1): 24 MoE + 24 dense → β₈ applies
	// Mixtral (uniform MoE, no interleaving): All layers MoE → β₈ does not apply
	// Physics-motivated: Uniform architectures avoid kernel switching overhead.
	var moeScaling float64
	if m.hasInterleavedMoE {
		moeScaling = 1.0
	} else {
		moeScaling = 0.0
	}

	stepTime := prefillTerm +
		decodeTerm +
		m.Beta[2]*tWeight +
		m.Beta[3]*tTp +
		m.Beta[4]*L +
		m.Beta[5]*batchSize +
		m.Beta[6] +
		m.Beta[7]*moeScaling*float64(m.numMoELayers) // β₈: per-MoE-layer overhead (interleaved archs only)

	return max(1, clampToInt64(stepTime))
}

// QueueingTime computes request-level overhead (ARRIVED → QUEUED).
// Constant per-request, matching trained-roofline convention.
//
// α₀ = API processing overhead (HTTP parsing, request validation, queue insertion).
func (m *TrainedPhysicsModel) QueueingTime(req *sim.Request) int64 {
	return clampToInt64(m.Alpha[0])
}

// OutputTokenProcessingTime returns per-output-token post-processing overhead.
// α₂ = streaming detokenization cost per output token (µs/token).
func (m *TrainedPhysicsModel) OutputTokenProcessingTime() int64 {
	return clampToInt64(m.Alpha[2])
}

// PostDecodeFixedOverhead returns fixed per-request overhead at completion.
// α₁ = post-decode overhead (µs), applied ONCE per request in recordRequestCompletion.
//
// This is the key structural fix from iter15: per-request overhead belongs here
// (applied once at completion), NOT in StepTime (where it would accumulate O(N×B)
// over N decode steps × B batch size).
func (m *TrainedPhysicsModel) PostDecodeFixedOverhead() int64 {
	return clampToInt64(m.Alpha[1])
}

// NewTrainedPhysicsModel creates an TrainedPhysicsModel with validation.
// Called by NewLatencyModel() when hw.Backend == "trained-physics".
func NewTrainedPhysicsModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*TrainedPhysicsModel, error) {
	// Validate coefficient counts (at least 7 beta required; 8th is optional MoE term)
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("trained-physics model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 7 {
		return nil, fmt.Errorf("trained-physics model: BetaCoeffs requires at least 7 elements, got %d (expected β₁-β₇, optionally β₈)", len(coeffs.BetaCoeffs))
	}

	// Backward compatible: 7→β₈=0, 8→no prefill split, 9→prefill split active
	betaSlice := make([]float64, 10)
	copy(betaSlice, coeffs.BetaCoeffs[:min(10, len(coeffs.BetaCoeffs))])

	// Validate hardware config
	if hw.TP <= 0 {
		return nil, fmt.Errorf("trained-physics model: TP must be > 0, got %d", hw.TP)
	}
	if hw.ModelConfig.NumLayers <= 0 {
		return nil, fmt.Errorf("trained-physics model: NumLayers must be > 0, got %d", hw.ModelConfig.NumLayers)
	}
	if hw.ModelConfig.NumHeads <= 0 {
		return nil, fmt.Errorf("trained-physics model: NumHeads must be > 0, got %d", hw.ModelConfig.NumHeads)
	}
	if hw.ModelConfig.HiddenDim <= 0 {
		return nil, fmt.Errorf("trained-physics model: HiddenDim must be > 0, got %d", hw.ModelConfig.HiddenDim)
	}
	if hw.ModelConfig.IntermediateDim <= 0 {
		return nil, fmt.Errorf("trained-physics model: IntermediateDim must be > 0, got %d", hw.ModelConfig.IntermediateDim)
	}
	if hw.ModelConfig.NumHeads%hw.TP != 0 {
		return nil, fmt.Errorf("trained-physics model: NumHeads (%d) must be divisible by TP (%d)", hw.ModelConfig.NumHeads, hw.TP)
	}
	numKVHeads := hw.ModelConfig.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = hw.ModelConfig.NumHeads // MHA fallback
	}
	if numKVHeads%hw.TP != 0 {
		return nil, fmt.Errorf("trained-physics model: NumKVHeads (%d) must be divisible by TP (%d)", numKVHeads, hw.TP)
	}
	if hw.HWConfig.TFlopsPeak <= 0 || math.IsNaN(hw.HWConfig.TFlopsPeak) || math.IsInf(hw.HWConfig.TFlopsPeak, 0) {
		return nil, fmt.Errorf("trained-physics model: TFlopsPeak must be valid positive, got %v", hw.HWConfig.TFlopsPeak)
	}
	if hw.HWConfig.BwPeakTBs <= 0 || math.IsNaN(hw.HWConfig.BwPeakTBs) || math.IsInf(hw.HWConfig.BwPeakTBs, 0) {
		return nil, fmt.Errorf("trained-physics model: BwPeakTBs must be valid positive, got %v", hw.HWConfig.BwPeakTBs)
	}

	// Validate coefficients (no NaN, Inf, or negative)
	if err := validateCoeffs("AlphaCoeffs", coeffs.AlphaCoeffs); err != nil {
		return nil, err
	}
	if err := validateCoeffs("BetaCoeffs", coeffs.BetaCoeffs); err != nil {
		return nil, err
	}

	headDim := hw.ModelConfig.HiddenDim / hw.ModelConfig.NumHeads

	// Determine MoE/dense layer split (#877)
	numMoELayers := 0
	numDenseLayers := hw.ModelConfig.NumLayers
	if hw.ModelConfig.InterleaveMoELayerStep > 0 && hw.ModelConfig.NumLocalExperts > 1 {
		step := hw.ModelConfig.InterleaveMoELayerStep
		numMoELayers = hw.ModelConfig.NumLayers / (step + 1)
		numDenseLayers = hw.ModelConfig.NumLayers - numMoELayers
	} else if hw.ModelConfig.NumLocalExperts > 1 {
		numMoELayers = hw.ModelConfig.NumLayers
		numDenseLayers = 0
	}

	// Determine FFN dimensions for MoE and dense layers
	dFF := hw.ModelConfig.IntermediateDim
	dFFMoE := dFF
	if hw.ModelConfig.MoEExpertFFNDim > 0 {
		dFFMoE = hw.ModelConfig.MoEExpertFFNDim
	}
	dFFDense := dFF
	if hw.ModelConfig.DenseIntermediateDim > 0 {
		dFFDense = hw.ModelConfig.DenseIntermediateDim
	}

	// Select compute throughput: FP8 for 1-byte-per-param models on FP8-capable GPUs
	peakFlops := hw.HWConfig.TFlopsPeak * 1e6 // TFLOPS → FLOP/µs
	weightBPP := hw.ModelConfig.EffectiveWeightBytesPerParam()
	if weightBPP == 1.0 && hw.HWConfig.TFlopsFP8 > 0 {
		peakFlops = hw.HWConfig.TFlopsFP8 * 1e6
	}

	return &TrainedPhysicsModel{
		Alpha:             [3]float64{coeffs.AlphaCoeffs[0], coeffs.AlphaCoeffs[1], coeffs.AlphaCoeffs[2]},
		Beta:              betaSlice,
		prefillSplit:      len(coeffs.BetaCoeffs) >= 9,
		decodeSplit:       len(coeffs.BetaCoeffs) >= 10,
		numLayers:         hw.ModelConfig.NumLayers,
		numMoELayers:      numMoELayers,
		numDenseLayers:    numDenseLayers,
		hiddenDim:         hw.ModelConfig.HiddenDim,
		numHeads:          hw.ModelConfig.NumHeads,
		headDim:           headDim,
		dKV:               numKVHeads * headDim,
		dFF:               dFF,
		dFFMoE:            dFFMoE,
		dFFDense:          dFFDense,
		kEff:              max(1, hw.ModelConfig.NumExpertsPerTok),
		numExperts:        hw.ModelConfig.NumLocalExperts,
		hasInterleavedMoE: hw.ModelConfig.InterleaveMoELayerStep > 0,
		isMoE:             hw.ModelConfig.NumLocalExperts > 0,
		tp:                hw.TP,
		weightBPP:         weightBPP,
		flopsPeakUs:       peakFlops,
		bwHbmUs:           hw.HWConfig.BwPeakTBs * 1e6,
	}, nil
}
