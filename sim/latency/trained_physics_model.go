package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// TrainedPhysicsModel implements a physics-informed latency model that combines
// analytical roofline performance bounds with learned correction coefficients.
//
// # Model Architecture
//
// The model uses roofline analysis as a physics-based prior, computing analytical
// bounds for compute (FLOPs) and memory bandwidth (HBM transfers) for each operation:
// attention, MLP, weight loading, KV cache access, and TP communication. Learned
// coefficients (α, β) correct these analytical estimates to match observed latencies.
//
// # Step-Time Formula
//
// The model supports 8, 9, or 10 beta coefficients for increasing fidelity:
//
// 8-beta (default):
//   T_step = β₁·max(T_pf_compute, T_pf_kv) + β₂·max(T_dc_compute, T_dc_kv)
//            + β₃·T_weight + β₄·T_tp + β₅·L + β₆·B + β₇ + β₈·nMoE
//
// 9-beta (prefill split):
//   T_step = β₁ₐ·T_pf_compute + β₁ᵦ·T_pf_kv + β₂·max(T_dc_compute, T_dc_kv)
//            + β₃·T_weight + β₄·T_tp + β₅·L + β₆·B + β₇ + β₈·nMoE
//
// 10-beta (prefill + decode split):
//   T_step = β₁ₐ·T_pf_compute + β₁ᵦ·T_pf_kv + β₂ₐ·T_dc_compute + β₂ᵦ·T_dc_kv
//            + β₃·T_weight + β₄·T_tp + β₅·L + β₆·B + β₇ + β₈·nMoE
//
// Where:
//   - T_pf_compute: Prefill compute time (FlashAttention FLOPs + MLP FLOPs)
//   - T_pf_kv: Prefill KV cache write bandwidth
//   - T_dc_compute: Decode compute time (single-token attention + MLP)
//   - T_dc_kv: Decode KV cache read bandwidth (past tokens)
//   - T_weight: Model weight loading bandwidth (per-step fixed cost)
//   - T_tp: Tensor-parallel All-Reduce communication time
//   - L: Number of transformer layers
//   - B: Batch size (number of requests)
//   - nMoE: Number of MoE layers (0 for dense models)
//
// # Beta Coefficients (Roofline Corrections + Overheads)
//
// β₁ (or β₁ₐ): Prefill compute correction (dimensionless, ~1.0)
//   Corrects analytical FlashAttention + MLP FLOP estimates. Accounts for kernel
//   efficiency, memory access patterns, and instruction-level parallelism.
//
// β₁ᵦ (β₉): Prefill memory correction (dimensionless, ~0.0 when compute-bound)
//   Corrects KV cache write bandwidth. Typically zero since prefill is compute-bound.
//
// β₂ (or β₂ₐ): Decode compute correction (dimensionless, ~0.0 when memory-bound)
//   Corrects single-token attention + MLP FLOPs. Typically zero since decode is
//   memory-bound (bandwidth-limited by KV cache reads).
//
// β₂ᵦ (β₁₀): Decode memory correction (dimensionless, ~1.0-2.0)
//   Corrects KV cache read bandwidth. Primary decode bottleneck.
//
// β₃: Weight loading correction (dimensionless, ~1.0-1.5)
//   Corrects model weight bandwidth (loaded once per step). Accounts for cache
//   effects, prefetching, and HBM contention with KV cache traffic.
//
// β₄: TP All-Reduce correction (dimensionless, ~0.3-0.8)
//   Corrects tensor-parallel communication overhead. Absorbs NVLink/HBM bandwidth
//   ratio and collective communication efficiency (ring, tree, etc.).
//
// β₅: Per-layer overhead (µs/layer)
//   Fixed overhead per transformer layer: kernel launch latency, CUDA graph overhead,
//   residual connections, layer normalization. Typically ~30-60 µs/layer.
//
// β₆: Per-request overhead (µs/request)
//   Scheduling and dispatch overhead per request in batch: queue management, attention
//   mask construction, token ID lookup. Typically ~3-5 µs/request.
//
// β₇: Per-step constant overhead (µs/step)
//   Fixed overhead per step independent of batch/model size: CUDA synchronization,
//   sampler invocation, logging. Typically ~100-200 µs/step.
//
// β₈: MoE-layer overhead (µs/MoE-layer, architecture-aware)
//   Per-MoE-layer overhead: router gating, token permutation, expert-parallel
//   communication. Applies only to interleaved architectures (InterleaveMoELayerStep > 0).
//   Zero for uniform MoE (all layers are MoE) and dense models. Typically ~400-500 µs/layer.
//
// # Alpha Coefficients (API/Framework Overheads)
//
// α₀: QueueingTime (µs)
//   Fixed per-request API processing: HTTP parsing, request validation, queue
//   insertion. Independent of model/batch size. Typically ~15,000 µs (15ms).
//
// α₁: PostDecodeFixedOverhead (µs)
//   Fixed per-request post-decode overhead: detokenization setup, finish reason
//   determination, response serialization. Typically ~800 µs.
//
// α₂: OutputTokenProcessingTime (µs/token)
//   Per-output-token overhead: streaming token transmission, incremental detokenization.
//   Typically ~50 µs/token.
//
// # Architecture-Aware Features
//
// 1. Interleaved MoE/Dense Layers: Models like Scout alternate MoE and dense layers.
//    The model splits FLOPs and weight bandwidth calculations by layer type, using
//    DenseIntermediateDim for dense layers and MoEExpertFFNDim for MoE layers.
//    β₈ overhead applies only to MoE layers in interleaved architectures.
//
// 2. Quantization-Aware Weights: Uses EffectiveWeightBytesPerParam for weight
//    bandwidth (e.g., 1 byte for FP8, 0.5 for W4A16), while KV cache always uses
//    FP16 (2 bytes). Automatically selects TFlopsFP8 for FP8 models on H100.
//
// 3. Tensor Parallelism Scaling: All compute and bandwidth terms are divided by TP
//    degree, while β₄ captures All-Reduce communication overhead explicitly.
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
	dFFMoE            int     // MoE expert FFN dim
	dFFDense          int     // Dense layer FFN dim (may differ for interleaved archs)
	kEff              int     // max(1, NumExpertsPerTok)
	numExperts        int     // NumLocalExperts (0 for dense)
	isMoE             bool    // NumLocalExperts > 1
	hasInterleavedMoE bool    // InterleaveMoELayerStep > 0 && NumLocalExperts > 1 (Scout-style alternating MoE/dense)
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
// Constant per-request.
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

	// Validate MoE consistency (same check as ValidateRooflineConfig)
	if hw.ModelConfig.NumLocalExperts > 1 && hw.ModelConfig.NumExpertsPerTok <= 0 {
		return nil, fmt.Errorf("trained-physics model: MoE config invalid - NumLocalExperts=%d but NumExpertsPerTok must be > 0", hw.ModelConfig.NumLocalExperts)
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
		dFFMoE:            dFFMoE,
		dFFDense:          dFFDense,
		kEff:              max(1, hw.ModelConfig.NumExpertsPerTok),
		numExperts:        hw.ModelConfig.NumLocalExperts,
		hasInterleavedMoE: hw.ModelConfig.InterleaveMoELayerStep > 0 && hw.ModelConfig.NumLocalExperts > 1,
		isMoE:             hw.ModelConfig.NumLocalExperts > 1,
		tp:                hw.TP,
		weightBPP:         weightBPP,
		flopsPeakUs:       peakFlops,
		bwHbmUs:           hw.HWConfig.BwPeakTBs * 1e6,
	}, nil
}
