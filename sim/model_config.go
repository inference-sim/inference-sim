package sim

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

type ParamsConfig struct {
	NumParams         int64   `json:"total_params"`
	InferredPrecision string  `json:"precision"`
	BytesPerParam     float64 `json:"bytes_per_param"`
}

type ModelConfig struct {
	NumParams         int64   `json:"total_params"`
	NumLayers         int     `json:"num_hidden_layers"`
	HiddenDim         int     `json:"hidden_size"`
	NumHeads          int     `json:"num_attention_heads"`
	NumKVHeads        int     `json:"num_key_value_heads"`
	VocabSize         int     `json:"vocab_size"`
	InferredPrecision string  `json:"precision"`
	IntermediateDim   int     `json:"intermediate_size"`
	BytesPerParam     float64 `json:"bytes_per_param"`
}

// HFConfig represents a flexible JSON object with dynamic fields.
type HFConfig struct {
	// Raw holds the entire JSON as a dynamic map.
	Raw map[string]any
}

// parseHFConfig parses arbitrary JSON into HFConfig.
func parseHFConfig(HFConfigFilePath string) (*HFConfig, error) {
	data, err := os.ReadFile(HFConfigFilePath)
	if err != nil {
		panic(err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse HFConfig: %w", err)
	}
	return &HFConfig{Raw: m}, nil
}

// parseParamsConfig parses model parameters JSON into ParamsConfig.
func parseParamsConfig(paramsConfigFilePath string) (*ParamsConfig, error) {
	data, err := os.ReadFile(paramsConfigFilePath)
	if err != nil {
		return nil, err
	}

	var params ParamsConfig
	if err := json.Unmarshal(data, &params); err != nil {
		return nil, err
	}

	return &params, nil
}

func GetModelConfig(hfConfigPath string, paramsConfigPath string) *ModelConfig {
	hf, _ := parseHFConfig(hfConfigPath)
	params, _ := parseParamsConfig(paramsConfigPath)
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

	return &ModelConfig{
		// From ParamsConfig
		NumParams:         params.NumParams,
		InferredPrecision: params.InferredPrecision,
		BytesPerParam:     params.BytesPerParam,

		// From HFConfig.Raw
		NumLayers:       getInt("num_hidden_layers"),
		HiddenDim:       getInt("hidden_size"),
		VocabSize:       getInt("vocab_size"),
		IntermediateDim: getInt("intermediate_size"),
		NumHeads:        numHeads,
		NumKVHeads:      numKVHeads,
	}
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
