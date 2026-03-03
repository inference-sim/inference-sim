package sim

// ModelConfig holds model architecture parameters parsed from a HuggingFace config.json.
// Used by the roofline and cross-model latency models for step time estimation.
// Parsing functions are in sim/latency/config.go.
type ModelConfig struct {
	NumLayers        int     `json:"num_hidden_layers"`
	HiddenDim        int     `json:"hidden_size"`
	NumHeads         int     `json:"num_attention_heads"`
	NumKVHeads       int     `json:"num_key_value_heads"`
	VocabSize        int     `json:"vocab_size"`
	BytesPerParam    float64 `json:"bytes_per_param"`
	IntermediateDim  int     `json:"intermediate_size"`
	NumLocalExperts  int     `json:"num_local_experts"`   // 0 = dense model (MoE: number of experts)
	NumExpertsPerTok int     `json:"num_experts_per_tok"` // 0 = dense model (MoE: active experts per token)
}

// HardwareCalib holds GPU hardware calibration parameters.
// Used by the roofline latency model for compute/memory bandwidth estimation.
// Parsing functions are in sim/latency/config.go.
type HardwareCalib struct {
	TFlopsPeak       float64 `json:"TFlopsPeak"`      // Tera (10^12) FLOP/s
	BwPeakTBs        float64 `json:"BwPeakTBs"`       // in TB/s
	BwEffConstant    float64 `json:"BwEffConstant"`    // scaling factor to convert Peak BW to Effective BW
	TOverheadMicros  float64 `json:"TOverheadMicros"`  // Per-step Overheads unaccounted for
	PerLayerOverhead float64 `json:"perLayerOverhead"`
	MfuPrefill       float64 `json:"mfuPrefill"`
	MfuDecode        float64 `json:"mfuDecode"`
	AllReduceLatency float64 `json:"allReduceLatency"`
	MemoryGiB        float64 `json:"MemoryGiB"` // GPU memory capacity in GiB
}
