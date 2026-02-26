package latency

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

// HFConfig represents a flexible JSON object with dynamic fields.
type HFConfig struct {
	// Raw holds the entire JSON as a dynamic map.
	Raw map[string]any
}

func parseHWConfig(HWConfigFilePath string) (map[string]sim.HardwareCalib, error) {
	data, err := os.ReadFile(HWConfigFilePath)
	if err != nil {
		return nil, fmt.Errorf("read hardware config %q: %w", HWConfigFilePath, err)
	}

	var HardwareList map[string]sim.HardwareCalib
	if err := json.Unmarshal(data, &HardwareList); err != nil {
		return nil, fmt.Errorf("parse hardware config JSON: %w", err)
	}
	return HardwareList, nil
}

// GetHWConfig returns hardware calibration data for the specified GPU.
// Returns an error if the config file cannot be read/parsed or if the GPU is not found.
func GetHWConfig(HWConfigFilePath string, GPU string) (sim.HardwareCalib, error) {
	hwConfig, err := parseHWConfig(HWConfigFilePath)
	if err != nil {
		return sim.HardwareCalib{}, fmt.Errorf("get hardware config: %w", err)
	}
	config, ok := hwConfig[GPU]
	if !ok {
		available := make([]string, 0, len(hwConfig))
		for k := range hwConfig {
			available = append(available, k)
		}
		sort.Strings(available)
		return sim.HardwareCalib{}, fmt.Errorf("GPU %q not found in hardware config (available: %v)", GPU, available)
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
func GetModelConfig(hfConfigPath string) (*sim.ModelConfig, error) {
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

	// getIntWithFallbacks tries multiple field names, returning the first non-zero value.
	getIntWithFallbacks := func(keys ...string) int {
		for _, k := range keys {
			if v := getInt(k); v != 0 {
				return v
			}
		}
		return 0
	}

	// Extract heads first to handle the KV heads default logic.
	// Fallback field names: Falcon uses "num_kv_heads", GLM uses "multi_query_group_num".
	numHeads := getInt("num_attention_heads")
	numKVHeads := getIntWithFallbacks("num_key_value_heads", "num_kv_heads", "multi_query_group_num")

	// If all KV head fields are missing (0), default to num_attention_heads (MHA).
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

	// Safely extract torch_dtype - defaults to 0 bytes if missing or invalid.
	// Some models (e.g. GLM-5) use "dtype" instead of "torch_dtype".
	var bytesPerParam int
	if dtype, ok := hf.Raw["torch_dtype"].(string); ok {
		bytesPerParam = precisionToBytesPerParam[dtype]
	} else if dtype, ok := hf.Raw["dtype"].(string); ok {
		bytesPerParam = precisionToBytesPerParam[dtype]
	}

	// Intermediate dim: Falcon/GLM use "ffn_hidden_size" instead of "intermediate_size".
	intermediateDim := getIntWithFallbacks("intermediate_size", "ffn_hidden_size")

	modelConfig := &sim.ModelConfig{
		// From HFConfig.Raw
		NumLayers:       getInt("num_hidden_layers"),
		HiddenDim:       getInt("hidden_size"),
		VocabSize:       getInt("vocab_size"),
		IntermediateDim: intermediateDim,
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
func ValidateRooflineConfig(mc sim.ModelConfig, hc sim.HardwareCalib) error {
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
	// BwEfficiencyFactor is optional (0 = disabled, use raw peak BW).
	// When set, it must be a valid positive number in (0, 1].
	if hc.BwEfficiencyFactor != 0 {
		if math.IsNaN(hc.BwEfficiencyFactor) || math.IsInf(hc.BwEfficiencyFactor, 0) || hc.BwEfficiencyFactor < 0 || hc.BwEfficiencyFactor > 1 {
			problems = append(problems, fmt.Sprintf("HardwareCalib.BwEfficiencyFactor must be in (0, 1] or 0 (disabled), got %v", hc.BwEfficiencyFactor))
		}
	}

	// PerLayerCPUOverhead is optional (0 = no CPU overhead).
	// When set, it must be non-negative and finite.
	if hc.PerLayerCPUOverhead != 0 {
		if math.IsNaN(hc.PerLayerCPUOverhead) || math.IsInf(hc.PerLayerCPUOverhead, 0) || hc.PerLayerCPUOverhead < 0 {
			problems = append(problems, fmt.Sprintf("HardwareCalib.PerLayerCPUOverhead must be >= 0 and finite, got %v", hc.PerLayerCPUOverhead))
		}
	}

	if len(problems) > 0 {
		return fmt.Errorf("invalid roofline config: %s", strings.Join(problems, "; "))
	}
	return nil
}
