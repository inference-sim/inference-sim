package latency

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// CrossModelLatencyModel estimates latency using physics-informed features derived from
// model architecture (config.json). A single set of 4 beta coefficients works across
// model architectures via architecture-specific feature scaling.
//
// StepTime formula:
//
//	β₀·numLayers + β₁·decodeTokens·kvDimScaled + β₂·(prefillTokens+decodeTokens)·isMoE + β₃·isTP
//
// Beta coefficients (globally-fitted via OLS from 13 vLLM experiments; see defaults.yaml crossmodel_defaults):
//
//	β₀ = per-layer CUDA kernel dispatch overhead (µs/layer)
//	β₁ = KV cache bandwidth cost (µs per scaled KV unit)
//	β₂ = MoE expert routing + dispatch/gather cost (µs per MoE token)
//	β₃ = fixed TP synchronization barrier (µs per step, TP > 1 only)
//
// Architecture features are computed once at construction and frozen:
//
//	kvDimScaled = numLayers × numKVHeads × headDim / TP × 1e-6
//	isMoE       = 1.0 if NumLocalExperts > 0, else 0.0
//	isTP        = 1.0 if TP > 1, else 0.0
type CrossModelLatencyModel struct {
	betaCoeffs  []float64 // [per_layer, kv_bw, moe_dispatch, tp_sync]
	alphaCoeffs []float64 // [pre_sched_fixed, pre_sched_per_tok, output_per_tok]

	// Pre-computed architecture features (frozen at construction)
	numLayers   int
	kvDimScaled float64 // L × kvHeads × headDim / TP × 1e-6
	isMoE       float64 // 1.0 if NumLocalExperts > 0
	isTP        float64 // 1.0 if TP > 1
}

func (m *CrossModelLatencyModel) StepTime(batch []*sim.Request) int64 {
	var totalPrefillTokens, totalDecodeTokens int64
	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			totalPrefillTokens += int64(req.NumNewTokens)
		} else if len(req.OutputTokens) > 0 {
			totalDecodeTokens += int64(req.NumNewTokens)
		}
	}
	// β₁ term uses only decode tokens (not prefill) because decode is memory-bandwidth-bound
	// on H100: each decode token reads its accumulated KV cache from HBM. Prefill KV write
	// cost is absorbed into β₀ (per-layer term) where it overlaps with compute via GPU pipelining.
	stepTime := m.betaCoeffs[0]*float64(m.numLayers) +
		m.betaCoeffs[1]*float64(totalDecodeTokens)*m.kvDimScaled +
		m.betaCoeffs[2]*float64(totalPrefillTokens+totalDecodeTokens)*m.isMoE +
		m.betaCoeffs[3]*m.isTP
	return max(1, int64(stepTime))
}

func (m *CrossModelLatencyModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return int64(totalProcessingTime)
}

func (m *CrossModelLatencyModel) OutputTokenProcessingTime() int64 {
	return int64(m.alphaCoeffs[2])
}

func (m *CrossModelLatencyModel) PostDecodeFixedOverhead() int64 { return 0 }
