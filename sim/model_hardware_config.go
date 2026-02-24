package sim

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

type ModelConfig struct {
	NumLayers       int     `json:"num_hidden_layers"`
	HiddenDim       int     `json:"hidden_size"`
	NumHeads        int     `json:"num_attention_heads"`
	NumKVHeads      int     `json:"num_key_value_heads"`
	VocabSize       int     `json:"vocab_size"`
	BytesPerParam   float64 `json:"bytes_per_param"`
	IntermediateDim int     `json:"intermediate_size"`
}

// --- Hardware Data Structures ---

// HardwareCalib contains GPU specifications for the roofline latency model.
// All fields map to hardware_config_roofline_valid.json.
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

// HFConfig represents a flexible JSON object with dynamic fields.
type HFConfig struct {
	// Raw holds the entire JSON as a dynamic map.
	Raw map[string]any
}

func parseHWConfig(HWConfigFilePath string) (map[string]HardwareCalib, error) {
	data, err := os.ReadFile(HWConfigFilePath)
	if err != nil {
		return nil, fmt.Errorf("read hardware config %q: %w", HWConfigFilePath, err)
	}

	var HardwareList map[string]HardwareCalib
	if err := json.Unmarshal(data, &HardwareList); err != nil {
		return nil, fmt.Errorf("parse hardware config JSON: %w", err)
	}
	return HardwareList, nil
}

// GetHWConfig returns hardware calibration data for the specified GPU.
// Returns an error if the config file cannot be read/parsed or if the GPU is not found.
func GetHWConfig(HWConfigFilePath string, GPU string) (HardwareCalib, error) {
	hwConfig, err := parseHWConfig(HWConfigFilePath)
	if err != nil {
		return HardwareCalib{}, fmt.Errorf("get hardware config: %w", err)
	}
	config, ok := hwConfig[GPU]
	if !ok {
		available := make([]string, 0, len(hwConfig))
		for k := range hwConfig {
			available = append(available, k)
		}
		sort.Strings(available)
		return HardwareCalib{}, fmt.Errorf("GPU %q not found in hardware config (available: %v)", GPU, available)
	}
	return config, nil
}

// parseHFConfig parses arbitrary JSON into HFConfig.
func parseHFConfig(HFConfigFilePath string) (*HFConfig, error) {
	data, err := os.ReadFile(HFConfigFilePath)
	if err != nil {
		return nil, fmt.Errorf("read HF config %q: %w", HFConfigFilePath, err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse HF config JSON: %w", err)
	}
	// Check if this is a multimodal/composite config
	if textCfg, ok := m["text_config"].(map[string]any); ok {
		// We only care about text config, we "pivot" to the inner map.
		for k, v := range textCfg {
			m[k] = v
		}
	}
	return &HFConfig{Raw: m}, nil
}

// GetModelConfig parses a HuggingFace config.json and extracts model parameters.
// Returns an error if the config file cannot be read or parsed.
func GetModelConfig(hfConfigPath string) (*ModelConfig, error) {
	hf, err := parseHFConfig(hfConfigPath)
	if err != nil {
		return nil, fmt.Errorf("get model config: %w", err)
	}
	getInt := func(key string) int {
		if val, ok := hf.Raw[key].(float64); ok {
			return int(val)
		}
		return 0
	}

	// Extract heads first to handle the KV heads default logic
	numHeads := getInt("num_attention_heads")
	numKVHeads := getInt("num_key_value_heads")

	// Implementation logic: If num_key_value_heads is missing (0),
	// it typically defaults to num_attention_heads (MHA).
	if numKVHeads == 0 {
		numKVHeads = numHeads
	}

	// Extract precision and infer bytes per parameter
	precisionToBytesPerParam := map[string]int{
		"float32":  4,
		"float16":  2,
		"bfloat16": 2,
		"int8":     1,
		"uint8":    1,
		"fp8":      1,
		"int4":     1, // Often stored in 1-byte containers or packed
		"nf4":      1,
	}

	// Safely extract torch_dtype - defaults to 0 bytes if missing or invalid
	var bytesPerParam int
	if dtype, ok := hf.Raw["torch_dtype"].(string); ok {
		bytesPerParam = precisionToBytesPerParam[dtype]
	}

	modelConfig := &ModelConfig{
		// From HFConfig.Raw
		NumLayers:       getInt("num_hidden_layers"),
		HiddenDim:       getInt("hidden_size"),
		VocabSize:       getInt("vocab_size"),
		IntermediateDim: getInt("intermediate_size"),
		NumHeads:        numHeads,
		NumKVHeads:      numKVHeads,
		BytesPerParam:   float64(bytesPerParam),
	}
	return modelConfig, nil
}

// GetString returns a string value for a key if present and of the right type.
func (c *HFConfig) GetString(key string) (string, bool) {
	if v, ok := c.Raw[key]; ok {
		if s, ok := v.(string); ok {
			return s, true
		}
	}
	return "", false
}

// GetFloat returns a float64 for a key, handling JSON numbers (which decode as float64).
func (c *HFConfig) GetFloat(key string) (float64, bool) {
	if v, ok := c.Raw[key]; ok {
		switch x := v.(type) {
		case float64:
			return x, true
		case json.Number:
			f, err := x.Float64()
			if err == nil {
				return f, true
			}
		}
	}
	return 0, false
}

// GetInt tries to coerce a JSON number to int.
func (c *HFConfig) GetInt(key string) (int, bool) {
	if v, ok := c.Raw[key]; ok {
		switch x := v.(type) {
		case float64:
			return int(x), true
		case json.Number:
			i, err := x.Int64()
			if err == nil {
				return int(i), true
			}
		}
	}
	return 0, false
}

// GetBool returns a bool for a key.
func (c *HFConfig) GetBool(key string) (bool, bool) {
	if v, ok := c.Raw[key]; ok {
		if b, ok := v.(bool); ok {
			return b, true
		}
	}
	return false, false
}

// GetStrings returns []string, coercing []any → []string when possible.
func (c *HFConfig) GetStrings(key string) ([]string, bool) {
	v, ok := c.Raw[key]
	if !ok {
		return nil, false
	}
	switch arr := v.(type) {
	case []any:
		out := make([]string, 0, len(arr))
		for _, elem := range arr {
			if s, ok := elem.(string); ok {
				out = append(out, s)
			} else {
				return nil, false
			}
		}
		return out, true
	case []string:
		// Rarely produced directly by encoding/json; included for completeness.
		return arr, true
	default:
		return nil, false
	}
}

// MustGetString returns the string or a default.
func (c *HFConfig) MustGetString(key, def string) string {
	if s, ok := c.GetString(key); ok {
		return s
	}
	return def
}

// MustGetInt returns the int or a default.
func (c *HFConfig) MustGetInt(key string, def int) int {
	if i, ok := c.GetInt(key); ok {
		return i
	}
	return def
}

// Marshal returns the canonical JSON (preserving unknown fields).
func (c *HFConfig) Marshal() ([]byte, error) {
	if c.Raw == nil {
		return nil, errors.New("nil HFConfig")
	}
	return json.MarshalIndent(c.Raw, "", "  ")
}

// invalidPositiveFloat returns true if v is not a valid positive float64
// (i.e., v <= 0, NaN, or Inf). Used to validate roofline config denominators.
func invalidPositiveFloat(v float64) bool {
	return v <= 0 || math.IsNaN(v) || math.IsInf(v, 0)
}

// ValidateRooflineConfig checks that all fields required by the roofline latency
// model are valid positive values. Returns an error listing all invalid fields, or nil if valid.
func ValidateRooflineConfig(mc ModelConfig, hc HardwareCalib) error {
	var problems []string

	if mc.NumHeads <= 0 {
		problems = append(problems, fmt.Sprintf("ModelConfig.NumHeads must be > 0, got %d", mc.NumHeads))
	}
	if mc.NumLayers <= 0 {
		problems = append(problems, fmt.Sprintf("ModelConfig.NumLayers must be > 0, got %d", mc.NumLayers))
	}
	if mc.HiddenDim <= 0 {
		problems = append(problems, fmt.Sprintf("ModelConfig.HiddenDim must be > 0, got %d", mc.HiddenDim))
	}
	if invalidPositiveFloat(mc.BytesPerParam) {
		problems = append(problems, fmt.Sprintf("ModelConfig.BytesPerParam must be a valid positive number, got %v", mc.BytesPerParam))
	}

	if invalidPositiveFloat(hc.TFlopsPeak) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.TFlopsPeak must be a valid positive number, got %v", hc.TFlopsPeak))
	}
	if invalidPositiveFloat(hc.BwPeakTBs) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.BwPeakTBs must be a valid positive number, got %v", hc.BwPeakTBs))
	}

	// BwEfficiencyFactor: optional (0 = no correction), but if set must be in (0, 1.0]
	if hc.BwEfficiencyFactor != 0 {
		if math.IsNaN(hc.BwEfficiencyFactor) || math.IsInf(hc.BwEfficiencyFactor, 0) || hc.BwEfficiencyFactor < 0 || hc.BwEfficiencyFactor > 1.0 {
			problems = append(problems, fmt.Sprintf("HardwareCalib.BwEfficiencyFactor must be in (0, 1.0] or 0 (disabled), got %v", hc.BwEfficiencyFactor))
		}
	}

	if len(problems) > 0 {
		return fmt.Errorf("invalid roofline config: %s", strings.Join(problems, "; "))
	}
	return nil
}
