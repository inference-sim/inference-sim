package sim

// ModelConfig holds model architecture parameters parsed from a HuggingFace config.json.
// Used by the roofline latency model for FLOPs/bandwidth estimation.
// Parsing functions are in sim/latency/config.go.
type ModelConfig struct {
	NumLayers       int     `json:"num_hidden_layers"`
	HiddenDim       int     `json:"hidden_size"`
	NumHeads        int     `json:"num_attention_heads"`
	NumKVHeads      int     `json:"num_key_value_heads"`
	VocabSize       int     `json:"vocab_size"`
	BytesPerParam   float64 `json:"bytes_per_param"`
	IntermediateDim int     `json:"intermediate_size"`
}

// HardwareCalib holds GPU hardware calibration parameters.
// Used by the roofline latency model for compute/memory bandwidth estimation.
// Parsing functions are in sim/latency/config.go.
type HardwareCalib struct {
	TFlopsPeak float64 `json:"TFlopsPeak"` // Peak FP16 TFLOP/s (e.g. 989.5 for H100 SXM)
	BwPeakTBs  float64 `json:"BwPeakTBs"`  // Peak HBM bandwidth in TB/s (e.g. 3.35 for H100 SXM)

	// BwEfficiencyFactor is the sustained-to-peak HBM bandwidth ratio.
	// Accounts for the gap between datasheet peak and achievable sustained BW
	// (measured via STREAM or similar). 0 means disabled (use raw peak).
	// Reference: 0.82 for H100 SXM (H1 hypothesis), 0.80 for A100 SXM.
	BwEfficiencyFactor float64 `json:"bwEfficiencyFactor"`

	// PerLayerCPUOverhead is the per-layer CPU scheduling overhead in microseconds.
	// Per-step overhead = PerLayerCPUOverhead × (num_layers / tp).
	//
	// Models vLLM scheduler CPU work that scales with transformer layers per GPU:
	// block-table management (O(layers × batch)), per-layer tensor dispatch,
	// and input metadata preparation.
	//
	// Reference: 100 μs/layer for H100 (H2b hypothesis, TPOT MAPE 17.3%).
	// Sources: vAttention (block-table 10-30% of decode), Wuklab (up to 50%
	// overhead for small models), BROS (7-10% per iteration).
	PerLayerCPUOverhead float64 `json:"perLayerOverhead"`
}
