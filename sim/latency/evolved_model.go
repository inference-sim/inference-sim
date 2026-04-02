package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// EvolvedModel implements physics-informed latency model with learned correction
// coefficients, based on the proven trained-roofline architecture.
//
// Iteration 15: Adopt trained-roofline architecture + dataset-specific enhancements.
//
// BACKGROUND: Iterations 0-14 used a compute-only decode formula that dropped the
// memory-bandwidth floor from the roofline model. This caused per-request overhead
// terms (β₃, β₇) to inflate 100-1000× to compensate, producing O(N×B) accumulated
// phantom overhead in predicted E2E latency. The trained-roofline backend (7% MAPE
// on 13 experiments) already solves this with max(compute, memory) basis functions.
//
// ITER15 STRATEGY: Use trained-roofline's proven 7-term formula with three
// dataset-specific enhancements for the current 15-experiment training set:
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
// Step-time formula (8-term, extends trained-roofline with MoE overhead):
//
//	β₁·max(T_pf_compute, T_pf_kv) + β₂·max(T_dc_compute, T_dc_kv)
//	+ β₃·T_weight + β₄·T_tp + β₅·L + β₆·batchSize + β₇
//	+ β₈·nMoELayers
//
// Where β₁-β₃ are dimensionless roofline corrections (analytical prior ≈ 1.0),
// β₄ is TP communication correction, β₅ is µs/layer, β₆ is µs/request, β₇ is µs/step,
// β₈ is µs/MoE-layer (router gating + token permutation + EP communication overhead;
// zero for dense models since nMoELayers=0).
//
// Alpha coefficients:
//   - α₀: QueueingTime — fixed per-request API processing overhead (µs)
//   - α₁: PostDecodeFixedOverhead — fixed per-request post-decode overhead (µs)
//   - α₂: OutputTokenProcessingTime — per-output-token overhead (µs/token)
type EvolvedModel struct {
	Alpha [3]float64 // [α₀, α₁, α₂]
	Beta  []float64  // [β₁..β₈] — 8 coefficients (β₈=0 when 7 provided for backward compat)

	// Pre-computed architecture features (frozen at construction).
	numLayers      int
	numMoELayers   int // Interleaved MoE layers (0 for dense models)
	numDenseLayers int // Dense layers (= numLayers for dense models)
	hiddenDim      int
	numHeads       int
	headDim        int     // d_h = hiddenDim / numHeads
	dKV            int     // kvHeads * d_h (differs from hiddenDim for GQA)
	dFF            int     // Default FFN intermediate dim
	dFFMoE         int     // MoE expert FFN dim (may differ from dFF)
	dFFDense       int     // Dense layer FFN dim (may differ for interleaved archs)
	kEff           int     // max(1, NumExpertsPerTok)
	numExperts     int     // NumLocalExperts (0 for dense)
	isMoE          bool    // NumLocalExperts > 0
	tp             int     // Tensor parallelism degree
	weightBPP      float64 // EffectiveWeightBytesPerParam (FP8-aware)

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
func (m *EvolvedModel) StepTime(batch []*sim.Request) int64 {
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

	// T_tp: TP communication time (µs)
	// Currently zeroed (β₄=0 in trained-roofline fit; TP cost absorbed into β₅·L).
	tTp := 0.0

	// ─── 8-term step-time formula ──────────────────────────────────────
	stepTime := m.Beta[0]*math.Max(tPfCompute, tPfKv) +
		m.Beta[1]*math.Max(tDcCompute, tDcKv) +
		m.Beta[2]*tWeight +
		m.Beta[3]*tTp +
		m.Beta[4]*L +
		m.Beta[5]*batchSize +
		m.Beta[6] +
		m.Beta[7]*float64(m.numMoELayers) // β₈: per-MoE-layer overhead (µs/MoE-layer; 0 for dense)

	return max(1, clampToInt64(stepTime))
}

// QueueingTime computes request-level overhead (ARRIVED → QUEUED).
// Constant per-request, matching trained-roofline convention.
//
// α₀ = API processing overhead (HTTP parsing, request validation, queue insertion).
func (m *EvolvedModel) QueueingTime(req *sim.Request) int64 {
	return clampToInt64(m.Alpha[0])
}

// OutputTokenProcessingTime returns per-output-token post-processing overhead.
// α₂ = streaming detokenization cost per output token (µs/token).
func (m *EvolvedModel) OutputTokenProcessingTime() int64 {
	return clampToInt64(m.Alpha[2])
}

// PostDecodeFixedOverhead returns fixed per-request overhead at completion.
// α₁ = post-decode overhead (µs), applied ONCE per request in recordRequestCompletion.
//
// This is the key structural fix from iter15: per-request overhead belongs here
// (applied once at completion), NOT in StepTime (where it would accumulate O(N×B)
// over N decode steps × B batch size).
func (m *EvolvedModel) PostDecodeFixedOverhead() int64 {
	return clampToInt64(m.Alpha[1])
}

// NewEvolvedModel creates an EvolvedModel with validation.
// Called by NewLatencyModel() when hw.Backend == "evolved".
func NewEvolvedModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*EvolvedModel, error) {
	// Validate coefficient counts (at least 7 beta required; 8th is optional MoE term)
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 7 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 7 elements, got %d (expected β₁-β₇, optionally β₈)", len(coeffs.BetaCoeffs))
	}

	// Backward compatible: 7 betas still work (β₈ defaults to 0 = no MoE correction)
	betaSlice := make([]float64, 8)
	copy(betaSlice, coeffs.BetaCoeffs[:min(8, len(coeffs.BetaCoeffs))])

	// Validate hardware config
	if hw.TP <= 0 {
		return nil, fmt.Errorf("evolved model: TP must be > 0, got %d", hw.TP)
	}
	if hw.ModelConfig.NumLayers <= 0 {
		return nil, fmt.Errorf("evolved model: NumLayers must be > 0, got %d", hw.ModelConfig.NumLayers)
	}
	if hw.ModelConfig.NumHeads <= 0 {
		return nil, fmt.Errorf("evolved model: NumHeads must be > 0, got %d", hw.ModelConfig.NumHeads)
	}
	if hw.ModelConfig.HiddenDim <= 0 {
		return nil, fmt.Errorf("evolved model: HiddenDim must be > 0, got %d", hw.ModelConfig.HiddenDim)
	}
	if hw.ModelConfig.IntermediateDim <= 0 {
		return nil, fmt.Errorf("evolved model: IntermediateDim must be > 0, got %d", hw.ModelConfig.IntermediateDim)
	}
	if hw.ModelConfig.NumHeads%hw.TP != 0 {
		return nil, fmt.Errorf("evolved model: NumHeads (%d) must be divisible by TP (%d)", hw.ModelConfig.NumHeads, hw.TP)
	}
	numKVHeads := hw.ModelConfig.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = hw.ModelConfig.NumHeads // MHA fallback
	}
	if numKVHeads%hw.TP != 0 {
		return nil, fmt.Errorf("evolved model: NumKVHeads (%d) must be divisible by TP (%d)", numKVHeads, hw.TP)
	}
	if hw.HWConfig.TFlopsPeak <= 0 || math.IsNaN(hw.HWConfig.TFlopsPeak) || math.IsInf(hw.HWConfig.TFlopsPeak, 0) {
		return nil, fmt.Errorf("evolved model: TFlopsPeak must be valid positive, got %v", hw.HWConfig.TFlopsPeak)
	}
	if hw.HWConfig.BwPeakTBs <= 0 || math.IsNaN(hw.HWConfig.BwPeakTBs) || math.IsInf(hw.HWConfig.BwPeakTBs, 0) {
		return nil, fmt.Errorf("evolved model: BwPeakTBs must be valid positive, got %v", hw.HWConfig.BwPeakTBs)
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

	return &EvolvedModel{
		Alpha:          [3]float64{coeffs.AlphaCoeffs[0], coeffs.AlphaCoeffs[1], coeffs.AlphaCoeffs[2]},
		Beta:           betaSlice,
		numLayers:      hw.ModelConfig.NumLayers,
		numMoELayers:   numMoELayers,
		numDenseLayers: numDenseLayers,
		hiddenDim:      hw.ModelConfig.HiddenDim,
		numHeads:       hw.ModelConfig.NumHeads,
		headDim:        headDim,
		dKV:            numKVHeads * headDim,
		dFF:            dFF,
		dFFMoE:         dFFMoE,
		dFFDense:       dFFDense,
		kEff:           max(1, hw.ModelConfig.NumExpertsPerTok),
		numExperts:     hw.ModelConfig.NumLocalExperts,
		isMoE:          hw.ModelConfig.NumLocalExperts > 0,
		tp:             hw.TP,
		weightBPP:      weightBPP,
		flopsPeakUs:    peakFlops,
		bwHbmUs:        hw.HWConfig.BwPeakTBs * 1e6,
	}, nil
}
