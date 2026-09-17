package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sirupsen/logrus"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

const (
	hfConfigFile    = "config.json"
	modelConfigsDir = "model_configs"
)

// resolveModelConfig finds a HuggingFace config.json for the given model in the catalog.
// Resolution order: explicit --model-config-folder > the catalog's model_configs/<short-name>/.
// Returns the path to a directory containing config.json.
// Paths are resolved relative to defaultsFile's directory (consistent with resolveHardwareConfig).
//
// NS-6 (#1733): a model runs if and only if it is in the catalog. This function READS the
// catalog and never writes to it — a model whose config.json is absent is refused, naming
// the path the entry belongs at. It used to download config.json from HuggingFace and write
// it into model_configs/, which made "run an unknown model" silently ADD a catalog entry;
// that fetch (and every path that could create or modify a catalog file) is gone.
func resolveModelConfig(model, explicitFolder, defaultsFile string) (string, error) {
	// 1. Explicit override takes precedence
	if explicitFolder != "" {
		return explicitFolder, nil
	}

	// Derive the catalog's model_configs/<short-name>/ path relative to defaults.yaml
	// location (consistent with resolveHardwareConfig using filepath.Dir(defaultsFile))
	baseDir := filepath.Dir(defaultsFile)
	localDir, err := bundledModelConfigDir(model, baseDir)
	if err != nil {
		return "", fmt.Errorf("--latency-model: invalid model name %q: %w", model, err)
	}

	// 2. Read the catalog entry. Absent or non-HuggingFace-shaped is a hard error (R1):
	// there is no fallback that could supply the config, and inventing one is what NS-6
	// forbids. The file is never rewritten or deleted — a user-provided config with
	// non-standard field names is reported, not overwritten.
	localPath := filepath.Join(localDir, hfConfigFile)
	data, readErr := os.ReadFile(localPath)
	if readErr != nil {
		return "", fmt.Errorf(
			"model %q is not in the catalog: no %s at %s (%v).\n"+
				"  BLIS does not fetch model configs at run time — a model runs only if it is catalogued.\n"+
				"  Add the entry at %s, or pass --model-config-folder pointing at a directory that contains %s",
			model, hfConfigFile, localPath, readErr, localPath, hfConfigFile,
		)
	}
	if !json.Valid(data) || !isHFConfig(data) {
		return "", fmt.Errorf(
			"catalog entry for model %q at %s is not a HuggingFace config.json: it lacks the expected "+
				"fields (num_hidden_layers, hidden_size, layers_block_type, or the same fields under text_config).\n"+
				"  Fix that catalog entry, or pass --model-config-folder pointing at a directory that contains a valid %s",
			model, localPath, hfConfigFile,
		)
	}
	logrus.Infof("--latency-model: using config from %s", localDir)
	return localDir, nil
}

// resolveHardwareConfig finds the hardware config JSON file.
// Returns the explicit path if provided, or the bundled default.
func resolveHardwareConfig(explicitPath, defaultsFile string) (string, error) {
	if explicitPath != "" {
		return explicitPath, nil
	}

	// Derive bundled path from defaults.yaml location
	defaultsDir := filepath.Dir(defaultsFile)
	bundledPath := filepath.Join(defaultsDir, "hardware_config.json")
	if _, err := os.Stat(bundledPath); err == nil {
		logrus.Infof("--latency-model: using bundled hardware config at %s", bundledPath)
		return bundledPath, nil
	}

	return "", fmt.Errorf(
		"--latency-model: bundled hardware config not found at %q. Provide --hardware-config explicitly",
		bundledPath,
	)
}

// isHFConfig checks whether JSON bytes represent a HuggingFace transformer
// config.json. It looks for num_hidden_layers, hidden_size, or a non-empty
// layers_block_type list at the top level (text-only models) or nested inside
// text_config (multimodal models such as Llama4ForConditionalGeneration). This
// rejects an empty JSON {} or unrelated JSON that passes json.Valid as a catalog entry.
//
// layers_block_type is accepted because latency.GetModelConfigFromHF derives the layer
// count from its length when no num_hidden_layers scalar is declared (#1729 / NS-4). A
// config the parser can read must not be rejected one layer up as "not a HuggingFace
// config" — that would reject a perfectly good catalogued config. An empty or non-list
// value is NOT accepted: the parser cannot count it either, so the two paths agree on
// exactly what counts as usable evidence.
func isHFConfig(data []byte) bool {
	var m map[string]interface{}
	// Defensive: callers currently pre-validate with json.Valid, but retain this guard for future call sites.
	if err := json.Unmarshal(data, &m); err != nil {
		return false
	}

	hasLayerCountEvidence := func(cfg map[string]interface{}) bool {
		if _, ok := cfg["num_hidden_layers"]; ok {
			return true
		}
		if _, ok := cfg["hidden_size"]; ok {
			return true
		}
		blocks, ok := cfg[latency.LayersBlockTypeField].([]interface{})
		return ok && len(blocks) > 0
	}

	// Top-level fields cover text-only transformer configs.
	if hasLayerCountEvidence(m) {
		return true
	}

	// Fall back to text_config.* for multimodal models (Llama4ForConditionalGeneration, etc.)
	if textCfg, ok := m["text_config"].(map[string]interface{}); ok {
		return hasLayerCountEvidence(textCfg)
	}

	return false
}

// applyWeightPrecisionFallback applies model-name-based weight precision detection
// when quantization_config parsing didn't yield a result, and logs diagnostic messages.
// mc is modified in place. hfRaw is the parsed HFConfig.Raw map used for the
// quantization_config presence check.
func applyWeightPrecisionFallback(mc *sim.ModelConfig, model string, hfRaw map[string]any) {
	// Model name fallback: if quantization_config parsing didn't yield weight
	// precision, try to infer from naming conventions (e.g. w4a16, FP8).
	if mc.WeightBytesPerParam == 0 {
		mc.WeightBytesPerParam = latency.InferWeightBytesFromModelName(model)
	}

	// Log quantization info when weight precision differs from compute precision
	if mc.WeightBytesPerParam > 0 && mc.WeightBytesPerParam != mc.BytesPerParam {
		logrus.Infof("quantized model detected — weight precision: %.2f bytes/param, compute/activation precision: %.1f bytes/param",
			mc.WeightBytesPerParam, mc.BytesPerParam)
	} else if mc.WeightBytesPerParam == 0 {
		// Warn if quantization_config detected but neither parser nor name yielded precision
		if _, hasQC := hfRaw["quantization_config"]; hasQC {
			logrus.Warnf("HuggingFace config has quantization_config but weight precision could not be determined")
		} else if mc.BytesPerParam > 0 && mc.BytesPerParam <= 1 {
			logrus.Warnf("model reports %.0f byte(s)/param (possible quantization); "+
				"step time estimates may be inaccurate for quantized models",
				mc.BytesPerParam)
		}
	}
}

// applyKVCacheDtype resolves the --kv-cache-dtype flag to a KV-cache storage
// precision (bytes/param) and records it on mc.KVBytesPerParam (#1565). It mirrors
// applyWeightPrecisionFallback on the KV axis and is called at the same sites, so KV
// precision rides alongside weight precision wherever a ModelConfig is resolved.
//
// "auto" (the default) maps to 0, leaving KVBytesPerParam unset so
// EffectiveKVBytesPerParam falls back to the compute/activation dtype (BytesPerParam)
// — byte-identical to a build without the flag (INV-6). An explicit fp8 KV dtype under
// bf16 compute sets 1.0, halving per-token KV bytes (~2x KV block capacity), matching
// vLLM's --kv-cache-dtype fp8. KV precision is independent of weight quantization. mc
// is modified in place; an unrecognized value is a hard error (R1, CLI boundary).
func applyKVCacheDtype(mc *sim.ModelConfig, kvCacheDtype string) {
	bytes, ok := latency.KVCacheDtypeToBytes(kvCacheDtype)
	if !ok {
		logrus.Fatalf("--kv-cache-dtype %q is not recognized; valid values: auto, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, bf16, bfloat16, fp16, fp32", kvCacheDtype)
	}
	if bytes <= 0 {
		return // "auto": follow the compute dtype (KVBytesPerParam stays 0), INV-6.
	}
	mc.KVBytesPerParam = bytes
	if mc.BytesPerParam > 0 && bytes != mc.BytesPerParam {
		logrus.Infof("--kv-cache-dtype %q: KV cache stored at %.2f byte(s)/param vs compute/activation %.1f byte(s)/param (independent of weight precision)",
			kvCacheDtype, bytes, mc.BytesPerParam)
	}
}

// bundledModelConfigDir returns the expected catalog path for a model's config.
// Model names like "meta-llama/llama-3.1-8b-instruct" map to "<baseDir>/model_configs/llama-3.1-8b-instruct/".
// When baseDir is empty, returns a relative path (resolved relative to CWD).
// Returns an error if the model name contains path traversal sequences.
//
// Note (I12): The org prefix is stripped, so different orgs with identical model names
// (e.g., "org-a/llama" and "org-b/llama") would share the same directory. This matches
// the existing model_configs/ convention and is acceptable because HuggingFace model
// names are unique within orgs, and BLIS uses hf_repo for case-sensitive HF API calls.
func bundledModelConfigDir(model, baseDir string) (string, error) {
	// Use the part after the org prefix (after the /)
	parts := strings.SplitN(model, "/", 2)
	shortName := model
	if len(parts) == 2 {
		shortName = parts[1]
	}

	// Reject path traversal attempts (Clean first to normalize sequences like "a/./b")
	shortName = filepath.Clean(shortName)
	if strings.Contains(shortName, "..") || filepath.IsAbs(shortName) {
		return "", fmt.Errorf("model name %q contains invalid path components", model)
	}

	if baseDir != "" {
		return filepath.Join(baseDir, modelConfigsDir, shortName), nil
	}
	return filepath.Join(modelConfigsDir, shortName), nil
}
