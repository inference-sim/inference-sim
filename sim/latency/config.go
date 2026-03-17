package latency

import (
	"encoding/json"
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

// GetString returns a string value for a key if present and of the right type.
func (c *HFConfig) GetString(key string) (string, bool) {
	if v, ok := c.Raw[key]; ok {
		if s, ok := v.(string); ok {
			return s, true
		}
	}
	return "", false
}

// GetInt tries to coerce a JSON number to int.
func (c *HFConfig) GetInt(key string) (int, bool) {
	if v, ok := c.Raw[key]; ok {
		if f, ok := v.(float64); ok {
			return int(f), true
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

// ParseHFConfig parses a HuggingFace config.json file into an HFConfig.
func ParseHFConfig(HFConfigFilePath string) (*HFConfig, error) {
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
	hf, err := ParseHFConfig(hfConfigPath)
	if err != nil {
		return nil, fmt.Errorf("get model config: %w", err)
	}
	return GetModelConfigFromHF(hf)
}

// GetModelConfigFromHF extracts model parameters from a pre-parsed HFConfig.
// Use this when you already have a parsed HFConfig to avoid re-reading the file.
func GetModelConfigFromHF(hf *HFConfig) (*sim.ModelConfig, error) {
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

	// MoE expert count: extended resolution chain (design D4).
	// Threshold is > 1: single-expert models (num_local_experts=1) are dense-equivalent.
	// This matches ExtractKVCapacityParams semantics (R23 code path parity).
	numLocalExperts := getInt("num_local_experts")
	if numLocalExperts <= 1 {
		for _, key := range []string{"num_routed_experts", "n_routed_experts", "num_experts"} {
			if v := getInt(key); v > 1 {
				numLocalExperts = v
				break
			}
		}
	}
	numExpertsPerTok := getInt("num_experts_per_tok")

	// MoE per-expert FFN dimension (design Section 4.2)
	// When present and nonzero, takes precedence over general intermediate dim.
	moeExpertFFNDim := getInt("moe_intermediate_size")

	// Shared expert FFN dimension resolution (design D3, D5)
	// Priority: explicit shared_expert_intermediate_size > n_shared_experts × per-expert dim
	var sharedExpertFFNDim int
	if v := getInt("shared_expert_intermediate_size"); v > 0 {
		sharedExpertFFNDim = v
	} else if nShared := getInt("n_shared_experts"); nShared > 0 {
		// Compute total shared dim from count × per-expert dim
		perExpert := moeExpertFFNDim
		if perExpert == 0 {
			perExpert = intermediateDim // Mixtral convention
		}
		sharedExpertFFNDim = nShared * perExpert
	}

	// Activation function: used by KV capacity for SwiGLU detection (3-matrix weight estimation).
	// Roofline step time currently uses 2-matrix for all activations (see mlpMatrixCount).
	hiddenAct := hf.MustGetString("hidden_act", "")

	// Extract quantized weight precision from quantization_config (if present).
	// torch_dtype reports the compute/activation dtype (e.g. bfloat16=2 bytes), but
	// quantized models store weights at lower precision (e.g. W4A16=0.5 bytes/param).
	// WeightBytesPerParam=0 means "not quantized, use BytesPerParam".
	var weightBytesPerParam float64
	if qcRaw, ok := hf.Raw["quantization_config"]; ok {
		if qc, ok := qcRaw.(map[string]any); ok {
			quantMethod, _ := qc["quant_method"].(string)
			bits := 0
			if bitsRaw, ok := qc["bits"].(float64); ok {
				bits = int(bitsRaw)
			}
			if bits > 0 {
				weightBytesPerParam = float64(bits) / 8.0
			} else if strings.EqualFold(quantMethod, "fp8") {
				weightBytesPerParam = 1.0
			}
		}
	}

	modelConfig := &sim.ModelConfig{
		NumLayers:           getInt("num_hidden_layers"),
		HiddenDim:           getInt("hidden_size"),
		VocabSize:           getInt("vocab_size"),
		IntermediateDim:     intermediateDim,
		NumHeads:            numHeads,
		NumKVHeads:          numKVHeads,
		BytesPerParam:       float64(bytesPerParam),
		NumLocalExperts:     numLocalExperts,
		NumExpertsPerTok:    numExpertsPerTok,
		MoEExpertFFNDim:     moeExpertFFNDim,
		SharedExpertFFNDim:  sharedExpertFFNDim,
		HiddenAct:           hiddenAct,
		WeightBytesPerParam: weightBytesPerParam,
	}
	return modelConfig, nil
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
	if invalidPositiveFloat(hc.MfuPrefill) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.MfuPrefill must be a valid positive number, got %v", hc.MfuPrefill))
	}
	if invalidPositiveFloat(hc.MfuDecode) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.MfuDecode must be a valid positive number, got %v", hc.MfuDecode))
	}

	// MoE consistency checks (design Section 4.6)
	if mc.NumLocalExperts < 0 {
		problems = append(problems, fmt.Sprintf(
			"MoE: NumLocalExperts must be >= 0, got %d", mc.NumLocalExperts))
	}
	if mc.NumLocalExperts > 1 && mc.NumExpertsPerTok <= 0 {
		problems = append(problems, fmt.Sprintf(
			"MoE: NumLocalExperts=%d but active experts per token (NumExpertsPerTok) must be > 0",
			mc.NumLocalExperts))
	}
	if mc.NumExpertsPerTok > mc.NumLocalExperts && mc.NumLocalExperts > 1 {
		problems = append(problems, fmt.Sprintf(
			"MoE: NumExpertsPerTok (%d) cannot exceed NumLocalExperts (%d)",
			mc.NumExpertsPerTok, mc.NumLocalExperts))
	}
	if mc.NumLocalExperts == 0 && mc.NumExpertsPerTok > 0 {
		problems = append(problems, fmt.Sprintf(
			"MoE: NumExpertsPerTok=%d but NumLocalExperts=0 (inconsistent)",
			mc.NumExpertsPerTok))
	}
	if mc.MoEExpertFFNDim < 0 {
		problems = append(problems, fmt.Sprintf(
			"MoE: MoEExpertFFNDim must be >= 0, got %d", mc.MoEExpertFFNDim))
	}
	if mc.SharedExpertFFNDim < 0 {
		problems = append(problems, fmt.Sprintf(
			"MoE: SharedExpertFFNDim must be >= 0, got %d", mc.SharedExpertFFNDim))
	}

	// MemoryGiB is optional (0 = no auto-calculation).
	// When set, it must be a valid positive number.
	if hc.MemoryGiB != 0 {
		if math.IsNaN(hc.MemoryGiB) || math.IsInf(hc.MemoryGiB, 0) || hc.MemoryGiB < 0 {
			problems = append(problems, fmt.Sprintf("HardwareCalib.MemoryGiB must be > 0 and finite when set, got %v", hc.MemoryGiB))
		}
	}

	// WeightBytesPerParam is optional (0 = not set, fall back to BytesPerParam).
	// When set, it must be a valid positive number. No upper-bound check is enforced:
	// WeightBytesPerParam > BytesPerParam is unusual but not invalid (e.g., INT4 KV cache
	// with FP32 weights). Callers should not assume weight precision <= compute precision.
	if mc.WeightBytesPerParam != 0 {
		if mc.WeightBytesPerParam < 0 || math.IsNaN(mc.WeightBytesPerParam) || math.IsInf(mc.WeightBytesPerParam, 0) {
			problems = append(problems, fmt.Sprintf(
				"ModelConfig.WeightBytesPerParam must be positive when set, got %v",
				mc.WeightBytesPerParam))
		}
	}

	if len(problems) > 0 {
		return fmt.Errorf("invalid roofline config: %s", strings.Join(problems, "; "))
	}
	return nil
}
