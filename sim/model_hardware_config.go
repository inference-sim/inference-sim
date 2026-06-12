package sim

// ModelConfig holds model architecture parameters parsed from a HuggingFace config.json.
// Used by the roofline and cross-model latency models for step time estimation.
// Parsing functions are in sim/latency/config.go.
type ModelConfig struct {
	NumLayers           int     `json:"num_hidden_layers"`
	HiddenDim           int     `json:"hidden_size"`
	NumHeads            int     `json:"num_attention_heads"`
	NumKVHeads          int     `json:"num_key_value_heads"`
	VocabSize           int     `json:"vocab_size"`
	BytesPerParam       float64 `json:"bytes_per_param"`
	IntermediateDim     int     `json:"intermediate_size"`
	NumLocalExperts     int     `json:"num_local_experts"`                // 0 = dense model (MoE: number of experts)
	NumExpertsPerTok    int     `json:"num_experts_per_tok"`              // 0 = dense model (MoE: active experts per token)
	MoEExpertFFNDim     int     `json:"moe_intermediate_size"`            // Per-routed-expert FFN dim; 0 = use IntermediateDim (Mixtral convention)
	SharedExpertFFNDim  int     `json:"shared_expert_intermediate_size"`  // Total shared-expert FFN dim; 0 = no shared experts
	InterleaveMoELayerStep int  `json:"interleave_moe_layer_step"`        // Layer interleave pattern: 0 = uniform (all same type), 1 = alternate MoE/dense, 2 = every 3rd layer is MoE, etc. Used for Scout-style hybrid architectures.
	DenseIntermediateDim int    `json:"intermediate_size_mlp"`            // Dense layer FFN dimension; 0 = use IntermediateDim. For models like Scout where dense layers have different FFN size than MoE expert FFN.
	HiddenAct           string  `json:"hidden_act"`                       // Activation function (e.g. "silu", "gelu", "relu"); used by KV capacity (3-matrix SwiGLU detection), reserved for future roofline per-activation tuning
	WeightBytesPerParam float64 `json:"weight_bytes_per_param,omitempty"` // Quantized weight precision (bytes/param); 0 = not set, use BytesPerParam. Auto-detected from quantization_config or model name conventions.
}

// EffectiveWeightBytesPerParam returns the bytes-per-parameter to use for
// weight memory calculations. Returns WeightBytesPerParam when explicitly set
// (> 0), otherwise falls back to BytesPerParam (the compute/activation dtype).
// This decouples weight bandwidth (often quantized, e.g. 0.5 for W4A16) from
// KV cache and activation memory (which use the compute dtype, e.g. 2.0 for bfloat16).
func (mc ModelConfig) EffectiveWeightBytesPerParam() float64 {
	if mc.WeightBytesPerParam > 0 {
		return mc.WeightBytesPerParam
	}
	return mc.BytesPerParam
}

// MoEMinExperts is the minimum NumLocalExperts for a model to be treated as MoE.
// It is the single source of truth for the MoE-vs-dense boundary across BLIS:
// the detection predicate (IsMoE), the parse-time expert-count resolver
// (latency.HFConfig.ResolveNumExperts), and the KV-capacity MoE branch all key
// off this constant.
//
// Single-expert configs (NumLocalExperts == 1) are dense-equivalent in BLIS. The
// MoE weight/FLOP formulas (sim/latency/kv_capacity.go, roofline.go,
// trained_physics_model.go) read NumLocalExperts as a multiplier/divisor and would
// MISESTIMATE at N=1 (e.g. MoE weight 3·h·f_expert·1 + router ≠ dense 3·h·f_dense).
// The >= 2 threshold keeps N=1 out of those formulas.
//
// This is an intentional, documented divergence from vLLM, whose is_moe is
// get_num_experts() > 0 and which builds a well-defined degenerate 1-expert
// FusedMoE kernel. On every real model the two thresholds agree — no real HF config
// has NumLocalExperts == 1 — so keeping >= 2 loses no parity and preserves BLIS's
// analytic correctness.
const MoEMinExperts = 2

// IsMoE reports whether the model is a mixture-of-experts model
// (NumLocalExperts >= MoEMinExperts). See MoEMinExperts for the threshold rationale
// and the vLLM divergence note. This is the canonical MoE-detection predicate;
// callers must not re-encode the threshold inline.
func (mc ModelConfig) IsMoE() bool {
	return mc.NumLocalExperts >= MoEMinExperts
}

// HardwareCalib holds GPU hardware calibration parameters.
// Used by the roofline latency model for compute/memory bandwidth estimation.
// Parsing functions are in sim/latency/config.go.
type HardwareCalib struct {
	TFlopsPeak float64 `json:"TFlopsPeak"` // Tera (10^12) FLOP/s for FP16/BF16 compute
	TFlopsFP8  float64 `json:"TFlopsFP8"`  // Tera (10^12) FLOP/s for FP8 compute; 0 = no native FP8 support
	BwPeakTBs  float64 `json:"BwPeakTBs"`  // in TB/s
	MfuPrefill float64 `json:"mfuPrefill"`
	MfuDecode  float64 `json:"mfuDecode"`
	MemoryGiB  float64 `json:"MemoryGiB"` // GPU memory capacity in GiB
}
