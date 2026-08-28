package latency

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

const bitsPerByte = 8.0

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

// mustGetIntFallback returns the first of keys that resolves to a non-zero int
// (tried in order), else def. It centralizes multi-spelling field resolution
// (e.g. vendor-specific MoE activation-count names) so GetModelConfigFromHF and
// ExtractKVCapacityParams resolve identical spellings and cannot desync (R23
// code-path parity). Unexported — used only within package latency.
func (c *HFConfig) mustGetIntFallback(def int, keys ...string) int {
	for _, k := range keys {
		if v, ok := c.GetInt(k); ok && v != 0 {
			return v
		}
	}
	return def
}

// moeExpertCountFields lists the HF config field names that carry the total
// routed-expert count, in the resolution order used by vLLM's get_num_experts
// (vllm/transformers_utils/model_arch_config_convertor.py): num_experts (Jamba),
// moe_num_experts (Dbrx), n_routed_experts (DeepSeek), num_local_experts (Mixtral).
// num_routed_experts is a BLIS-historical alias retained at the end for
// compatibility. NumExpertsPerTok / n_shared_experts are activation counts, NOT
// totals, and are deliberately excluded.
var moeExpertCountFields = []string{
	"num_experts",       // Jamba
	"moe_num_experts",   // Dbrx
	"n_routed_experts",  // DeepSeek
	"num_local_experts", // Mixtral
	"num_routed_experts", // BLIS-historical alias
}

// moeActiveExpertFields and moeSharedExpertFields list the accepted HF spellings
// for the MoE *activation* counts (experts active per token; shared experts),
// tried in order. DeepSeek/GLM spell them num_experts_per_tok / n_shared_experts;
// Kimi-K3 (transformers/vLLM) spells them num_experts_per_token / num_shared_experts
// (#1634). Shared as package vars so GetModelConfigFromHF and ExtractKVCapacityParams
// resolve the same spellings and cannot desync (R23 code-path parity).
var moeActiveExpertFields = []string{"num_experts_per_tok", "num_experts_per_token"}
var moeSharedExpertFields = []string{"n_shared_experts", "num_shared_experts"}

// ResolveNumExperts returns the total routed-expert count for the model, trying the
// known architecture-specific field names (moeExpertCountFields) in order and
// returning the first value that meets the MoE threshold (sim.MoEMinExperts).
// Returns 0 for dense models — including single-expert configs, which are
// dense-equivalent in BLIS — so the count fed downstream never enters the MoE
// weight/FLOP formulas at N < 2 (see sim.MoEMinExperts for why N=1 must not).
//
// This is the single source of truth for expert-count resolution, shared by
// GetModelConfigFromHF and ExtractKVCapacityParams so the two cannot desync
// (R23 code-path parity).
//
// Parity with vLLM: the field set and order match vLLM's get_num_experts. The one
// intentional difference is the selection rule — vLLM returns the first field that
// EXISTS (then classifies via is_moe == count > 0), whereas BLIS returns the first
// field that is >= MoEMinExperts. On every real model the two rules pick the same
// field, because no real HF config sets a total-count field to 0 or 1 (verified
// against vLLM's config classes and fixtures). BLIS's threshold rule additionally
// protects its analytic formulas from a degenerate N=1, which vLLM does not need.
func (c *HFConfig) ResolveNumExperts() int {
	for _, key := range moeExpertCountFields {
		if v := c.MustGetInt(key, 0); v >= sim.MoEMinExperts {
			return v
		}
	}
	return 0
}

// LinearAttnFullLayerCount returns the number of full-attention (KV-cache-bearing)
// layers declared under a hybrid-attention model's linear_attn_config, i.e.
// len(linear_attn_config.full_attn_layers). ParseHFConfig pivots text_config onto
// the top-level map, so linear_attn_config is reachable as a top-level nested value.
//
// Kimi-K3 is a hybrid model: linear_attn_config lists 24 full_attn_layers (full
// Multi-head Latent Attention, which store a per-token KV cache) and 69 kda_layers
// (Kimi Delta Attention — linear attention with a fixed-size recurrent + short-conv
// state and no growing KV) of 93 total. Sizing the KV cache over the full-attention
// count alone corrects the per-token KV footprint (issue #1635).
//
// Returns 0 when the model is not hybrid — no linear_attn_config, or a block that
// lacks a full_attn_layers list (or carries one of the wrong type) — so callers
// fall back to NumLayers (every standard-MHA and non-hybrid MLA model, INV-6). Only
// the list length is read (elements may be any JSON number type), so the value type
// of individual entries is irrelevant.
func (c *HFConfig) LinearAttnFullLayerCount() int {
	lac, ok := c.Raw["linear_attn_config"].(map[string]any)
	if !ok {
		return 0 // not a hybrid-attention model
	}
	// linear_attn_config IS present, so this is a hybrid model. A missing / empty /
	// wrong-typed full_attn_layers would silently fall back to all-layers KV sizing —
	// reinstating the very ~3.9x over-count #1635 fixes — so warn loudly (R1) instead
	// of degrading silently.
	full, ok := lac["full_attn_layers"].([]any)
	if !ok || len(full) == 0 {
		logrus.Warnf("linear_attn_config present but full_attn_layers is missing/empty/non-list; " +
			"falling back to all-layers KV sizing for this hybrid model — KV capacity may be over-counted")
		return 0
	}
	return len(full)
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

// parseQuantizationConfig extracts quantized weight precision from quantization_config.
// Returns 0 if no quantization is detected or if parsing fails.
// torch_dtype reports the compute/activation dtype (e.g. bfloat16=2 bytes), but
// quantized models store weights at lower precision (e.g. W4A16=0.5 bytes/param).
func parseQuantizationConfig(qc map[string]any) float64 {
	quantMethod, _ := qc["quant_method"].(string)
	bits := 0

	// Try to extract bits from quantization_config.bits (float64 or string)
	if bitsRaw, ok := qc["bits"].(float64); ok {
		bits = int(bitsRaw)
	} else if bitsStr, ok := qc["bits"].(string); ok {
		if parsed, err := strconv.Atoi(bitsStr); err == nil {
			bits = parsed
		} else {
			logrus.Debugf("quantization_config.bits: invalid string value %q (expected integer)", bitsStr)
		}
	}

	if bits > 0 {
		return float64(bits) / bitsPerByte
	}

	// FP8 quantization
	if strings.EqualFold(quantMethod, "fp8") {
		return 1.0
	}

	// compressed-tensors: extract from config_groups.*.weights.num_bits
	if strings.EqualFold(quantMethod, "compressed-tensors") {
		// Keys are sorted for deterministic iteration (INV-6).
		// First-match semantics: the first valid num_bits found (in sorted key order) is used.
		if cg, ok := qc["config_groups"].(map[string]any); ok {
			keys := make([]string, 0, len(cg))
			for k := range cg {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				if gm, ok := cg[k].(map[string]any); ok {
					if w, ok := gm["weights"].(map[string]any); ok {
						if nb, ok := w["num_bits"].(float64); ok && nb > 0 {
							return nb / bitsPerByte
						}
					}
				}
			}
		} else {
			logrus.Debugf("compressed-tensors: config_groups structure does not match expected schema (expected map[string]any)")
		}
	}

	return 0
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
	getIntWithFallbacks := func(keys ...string) int { return hf.mustGetIntFallback(0, keys...) }

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

	// MoE expert count: resolved via the shared chain (R23 code-path parity with
	// ExtractKVCapacityParams). Single-expert models are dense-equivalent.
	numLocalExperts := hf.ResolveNumExperts()
	// Active experts per token: DeepSeek/GLM spell it num_experts_per_tok; Kimi-K3
	// (transformers/vLLM) spells it num_experts_per_token (#1634). Missing this on a
	// detected-MoE model is fatal (trips the MoE-consistency guard at latency-model
	// construction), not merely inaccurate.
	numExpertsPerTok := getIntWithFallbacks(moeActiveExpertFields...)

	// MoE per-expert FFN dimension (design Section 4.2)
	// When present and nonzero, takes precedence over general intermediate dim.
	moeExpertFFNDim := getInt("moe_intermediate_size")

	// Shared expert FFN dimension resolution (design D3, D5)
	// Priority: explicit shared_expert_intermediate_size > n_shared_experts × per-expert dim
	var sharedExpertFFNDim int
	if v := getInt("shared_expert_intermediate_size"); v > 0 {
		sharedExpertFFNDim = v
	} else if nShared := getIntWithFallbacks(moeSharedExpertFields...); nShared > 0 {
		// DeepSeek/GLM spell it n_shared_experts; Kimi-K3 spells it
		// num_shared_experts (#1634). Missing this is a silent weight under-count
		// (shared experts are optional, so no guard trips).
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
	// WeightBytesPerParam=0 means "not quantized, use BytesPerParam".
	var weightBytesPerParam float64
	if qcRaw, ok := hf.Raw["quantization_config"]; ok {
		if qc, ok := qcRaw.(map[string]any); ok {
			weightBytesPerParam = parseQuantizationConfig(qc)
		}
	}

	// Interleaved MoE architecture (Scout-style): alternate MoE/dense layers
	// 0 = uniform (all same type), 1 = alternate MoE/dense, 2 = every 3rd is MoE, etc.
	interleaveMoELayerStep := getInt("interleave_moe_layer_step")

	// Dense layer FFN dimension (for models with different dense vs MoE FFN sizes)
	// 0 = use IntermediateDim for both MoE and dense layers
	denseIntermediateDim := getInt("intermediate_size_mlp")

	// Explicit attention head dimension (F1, #1527). Modern MLA/GQA models declare
	// a head_dim that differs from hidden/heads (e.g. GLM-5.2: 192 vs 6144/64=96).
	// 0 = absent → EffectiveHeadDim falls back to HiddenDim/NumHeads (INV-6).
	headDim := getInt("head_dim")

	// MLA compressed-KV latent shape (F2, #1527). kv_lora_rank > 0 marks a
	// Multi-head Latent Attention model (DeepSeek-V2/V3, Kimi-K3, GLM-5.2); the KV
	// cache then stores a compressed latent of kv_lora_rank + qk_rope_head_dim
	// scalars per token per layer. Both 0 for standard MHA/GQA (INV-6).
	kvLoraRank := getInt("kv_lora_rank")
	qkRopeHeadDim := getInt("qk_rope_head_dim")

	// Dense-layer prefix count for MoE models (F3, #1527). first_k_dense_replace = K
	// means the first K layers are dense and the remainder are MoE (a prefix split,
	// distinct from InterleaveMoELayerStep's every-Nth interleave). 0 = no dense
	// prefix (INV-6: all-MoE weight accounting unchanged when absent).
	firstKDenseReplace := getInt("first_k_dense_replace")

	// KV-bearing (full-attention) layer count for hybrid-attention models (#1635).
	// Kimi-K3 interleaves 24 full-attention (MLA, KV-bearing) layers with 69
	// linear-attention (KDA, fixed-size recurrent state, no growing KV) layers of 93
	// total; only the full-attention layers store a per-token KV cache. The count is
	// len(linear_attn_config.full_attn_layers). 0 for every non-hybrid model →
	// EffectiveKVBearingLayers falls back to NumLayers, so the KV footprint is
	// byte-identical there (INV-6). Scoped to the KV-capacity path — KDA weights
	// (#1638) and KDA step time (#1636) are out of scope and still use all NumLayers.
	kvBearingLayers := hf.LinearAttnFullLayerCount()

	// Reject negative values for the shape fields parsed above (#1527). getInt
	// returns the raw JSON number, so a negative would otherwise pass silently: a
	// negative kv_lora_rank would fall through to the standard MHA path (wrong KV
	// capacity, no error), and a negative first_k_dense_replace would clamp to 0. A
	// negative head_dim / qk_rope_head_dim is equally nonsensical. Fail fast at
	// parse time (R1: no silent acceptance of bad input) rather than at use time.
	for _, f := range []struct {
		name string
		val  int
	}{
		{"head_dim", headDim},
		{"kv_lora_rank", kvLoraRank},
		{"qk_rope_head_dim", qkRopeHeadDim},
		{"first_k_dense_replace", firstKDenseReplace},
	} {
		if f.val < 0 {
			return nil, fmt.Errorf("GetModelConfigFromHF: %s must be >= 0, got %d", f.name, f.val)
		}
	}

	modelConfig := &sim.ModelConfig{
		NumLayers:              getInt("num_hidden_layers"),
		HiddenDim:              getInt("hidden_size"),
		VocabSize:              getInt("vocab_size"),
		IntermediateDim:        intermediateDim,
		NumHeads:               numHeads,
		NumKVHeads:             numKVHeads,
		BytesPerParam:          float64(bytesPerParam),
		NumLocalExperts:        numLocalExperts,
		NumExpertsPerTok:       numExpertsPerTok,
		MoEExpertFFNDim:        moeExpertFFNDim,
		SharedExpertFFNDim:     sharedExpertFFNDim,
		InterleaveMoELayerStep: interleaveMoELayerStep,
		DenseIntermediateDim:   denseIntermediateDim,
		HiddenAct:              hiddenAct,
		WeightBytesPerParam:    weightBytesPerParam,
		HeadDim:                headDim,
		KVLoraRank:             kvLoraRank,
		QKRopeHeadDim:          qkRopeHeadDim,
		FirstKDenseReplace:     firstKDenseReplace,
		KVBearingLayers:        kvBearingLayers,
	}
	return modelConfig, nil
}

// Compiled regexes for model name quantization detection.
var (
	// Matches wXaY patterns (e.g. w4a16, W8A8) — X is weight bits.
	reWxAy = regexp.MustCompile(`(?i)(?:^|[\.\-_/])w(\d+)a\d+(?:$|[\.\-_])`)
	// Matches fp8 keyword (e.g. FP8-dynamic, fp8).
	reFP8Name = regexp.MustCompile(`(?i)(?:^|[\.\-_/])fp8(?:$|[\.\-_])`)
)

// InferWeightBytesFromModelName attempts to infer quantized weight precision
// from naming conventions in HuggingFace model identifiers (e.g. "w4a16" → 0.5,
// "FP8" → 1.0). Returns 0 if no quantization pattern is detected.
// Used as a fallback when quantization_config parsing does not yield a result.
func InferWeightBytesFromModelName(name string) float64 {
	// Explicit wXaY pattern — weight bits are unambiguous.
	if m := reWxAy.FindStringSubmatch(name); m != nil {
		if bits, err := strconv.Atoi(m[1]); err == nil && bits > 0 {
			return float64(bits) / bitsPerByte
		}
	}
	// FP8 keyword — always 8-bit weights.
	if reFP8Name.MatchString(name) {
		return 1.0
	}
	return 0
}

// KVCacheDtypeToBytes maps a --kv-cache-dtype value to the KV-cache storage
// precision in bytes per element, mirroring vLLM's CacheConfig.cache_dtype
// resolution (#1565, vLLM v0.11.0). It returns (bytes, true) for a recognized value
// and (0, false) for an unrecognized one, so the CLI can fail loudly (R1).
//
//   - "auto" (and the empty string) → (0, true): the KV cache follows the
//     compute/activation dtype (ModelConfig.BytesPerParam via EffectiveKVBytesPerParam).
//     This is the default and is byte-identical to a build without the flag (INV-6).
//   - "fp8" / "fp8_e4m3" / "fp8_e5m2" → (1.0, true): vLLM maps every fp8 variant to
//     torch.uint8 (1 byte/element). fp8 KV under bf16 compute halves KV memory.
//   - "bf16" / "bfloat16" / "fp16" / "float16" / "half" → (2.0, true)
//   - "fp32" / "float32" / "float" → (4.0, true)
//
// KV storage precision is independent of weight quantization (WeightBytesPerParam) —
// they are separate vLLM engine args. Matching is case-insensitive and trims spaces.
func KVCacheDtypeToBytes(dtype string) (float64, bool) {
	switch strings.ToLower(strings.TrimSpace(dtype)) {
	case "", "auto":
		return 0, true
	case "fp8", "fp8_e4m3", "fp8_e5m2", "fp8e4m3", "fp8e5m2":
		return 1.0, true
	case "bf16", "bfloat16", "fp16", "float16", "half":
		return 2.0, true
	case "fp32", "float32", "float":
		return 4.0, true
	default:
		return 0, false
	}
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

	// TFlopsFP8 is optional (0 = no native FP8 support).
	// When set, it must be a valid positive number.
	if hc.TFlopsFP8 != 0 {
		if math.IsNaN(hc.TFlopsFP8) || math.IsInf(hc.TFlopsFP8, 0) || hc.TFlopsFP8 < 0 {
			problems = append(problems, fmt.Sprintf("HardwareCalib.TFlopsFP8 must be > 0 and finite when set, got %v", hc.TFlopsFP8))
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
		// Warn if weight precision exceeds compute precision (unusual but valid)
		if mc.WeightBytesPerParam > mc.BytesPerParam {
			logrus.Warnf("WeightBytesPerParam (%.2f) > BytesPerParam (%.2f): weight precision exceeds compute precision (unusual but valid, e.g., FP32 weights with INT4 KV cache)",
				mc.WeightBytesPerParam, mc.BytesPerParam)
		}
	}

	if len(problems) > 0 {
		return fmt.Errorf("invalid roofline config: %s", strings.Join(problems, "; "))
	}
	return nil
}
