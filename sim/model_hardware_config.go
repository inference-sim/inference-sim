package sim

// ModelConfig holds model architecture parameters parsed from a HuggingFace config.json.
// Used by the roofline and cross-model latency models for step time estimation.
// Parsing functions are in sim/latency/config.go.
type ModelConfig struct {
	NumLayers          int     `json:"num_hidden_layers"`
	HiddenDim          int     `json:"hidden_size"`
	NumHeads           int     `json:"num_attention_heads"`
	NumKVHeads         int     `json:"num_key_value_heads"`
	VocabSize          int     `json:"vocab_size"`
	BytesPerParam      float64 `json:"bytes_per_param"`
	IntermediateDim    int     `json:"intermediate_size"`
	NumLocalExperts    int     `json:"num_local_experts"`               // 0 = dense model (MoE: number of experts)
	NumExpertsPerTok   int     `json:"num_experts_per_tok"`             // 0 = dense model (MoE: active experts per token)
	MoEExpertFFNDim    int     `json:"moe_intermediate_size"`           // Per-routed-expert FFN dim; 0 = use IntermediateDim (Mixtral convention)
	SharedExpertFFNDim int     `json:"shared_expert_intermediate_size"` // Total shared-expert FFN dim; 0 = no shared experts
	HiddenAct           string  `json:"hidden_act"`                      // Activation function (e.g. "silu", "gelu", "relu"); used by KV capacity (3-matrix SwiGLU detection), reserved for future roofline per-activation tuning
	WeightBytesPerParam float64 `json:"weight_bytes_per_param,omitempty"` // Quantized weight precision (bytes/param); 0 = not set, use BytesPerParam. Parsed from quantization_config or --weight-bytes-per-param CLI flag.
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

// HardwareCalib holds GPU hardware calibration parameters.
// Used by the roofline latency model for compute/memory bandwidth estimation.
// Parsing functions are in sim/latency/config.go.
type HardwareCalib struct {
	TFlopsPeak float64 `json:"TFlopsPeak"` // Tera (10^12) FLOP/s
	BwPeakTBs  float64 `json:"BwPeakTBs"`  // in TB/s
	MfuPrefill float64 `json:"mfuPrefill"`
	MfuDecode  float64 `json:"mfuDecode"`
	MemoryGiB  float64 `json:"MemoryGiB"` // GPU memory capacity in GiB
}
