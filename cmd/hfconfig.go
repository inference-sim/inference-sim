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
	hfConfigFile = "config.json"
	// catalogEnvVar names the environment variable that locates the model catalog when
	// --catalog is not given (#1731, R1/S4). It is read exactly like HF_TOKEN was — the
	// only environment variables cmd/ consults.
	catalogEnvVar = "BLIS_CATALOG"
)

// catalogRootFrom picks the model-catalog root from the --catalog flag value and the
// BLIS_CATALOG environment value (#1731). The flag wins when both are set (an explicit
// CLI input beats the environment), and the override is announced on stderr so the choice
// is visible in the run's history. With NEITHER there is no default, no search path and
// no remote fetch: the caller is refused, naming both forms. The retired working-directory
// default (model_configs/ resolved against --defaults-filepath's directory) meant
// `blis run` silently worked only from the repository root, and inferring a catalog
// location is exactly what NS-6 forbids — nothing about the model is inferred.
//
// Pure with respect to process state: both inputs are supplied by the caller, so the
// precedence law is table-testable without touching globals or the environment.
func catalogRootFrom(flagValue, envValue string) (string, error) {
	if flagValue != "" {
		if envValue != "" && envValue != flagValue {
			// Warnf, not Infof: both commands default --log to warn, so an Infof precedence
			// notice would be silent under normal invocation. This choice overrides an
			// explicit BLIS_CATALOG, so the announcement must be visible in the run's
			// history at the default log level (qa-review G3, #1731).
			logrus.Warnf("--catalog %q takes precedence over %s=%q", flagValue, catalogEnvVar, envValue)
		}
		return flagValue, nil
	}
	if envValue != "" {
		logrus.Infof("model catalog located via %s=%q", catalogEnvVar, envValue)
		return envValue, nil
	}
	return "", fmt.Errorf("no model catalog was supplied: pass --catalog <path> or set the %s "+
		"environment variable (there is no default and no search path; a catalog holds one "+
		"directory per model, each with that model's %s)", catalogEnvVar, hfConfigFile)
}

// resolveCatalogRoot resolves the catalog root from the --catalog flag and the
// BLIS_CATALOG environment variable, then verifies it is a readable directory. An
// unreadable or non-directory catalog is refused naming both forms (R1) rather than
// deferred into a per-model "not in the catalog" message that would blame the model for
// a mistyped catalog path.
func resolveCatalogRoot() (string, error) {
	root, err := catalogRootFrom(catalogPath, os.Getenv(catalogEnvVar))
	if err != nil {
		return "", err
	}
	info, statErr := os.Stat(root)
	if statErr != nil {
		return "", fmt.Errorf("model catalog %q is not readable: %w (set --catalog or %s to the catalog root)",
			root, statErr, catalogEnvVar)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("model catalog %q is not a directory (set --catalog or %s to the catalog root)",
			root, catalogEnvVar)
	}
	return root, nil
}

// resolveModelConfig finds a HuggingFace config.json for the given model inside the
// catalog located by --catalog / BLIS_CATALOG. Returns the path to the catalog entry
// directory containing config.json.
//
// NS-6 (#1733): a model runs if and only if it is in the catalog. This function READS the
// catalog and never writes to it — a model whose config.json is absent is refused, naming
// the path the entry belongs at. It used to download config.json from HuggingFace and write
// it into model_configs/, which made "run an unknown model" silently ADD a catalog entry;
// that fetch (and every path that could create or modify a catalog file) is gone.
//
// S4 (#1731): the catalog's LOCATION is now an explicit input (--catalog / BLIS_CATALOG)
// rather than a working-directory-relative default, and the retired per-model folder flag
// is subsumed by it — pointing --catalog at a scratch directory covers the same
// "use my own config" need, so there is exactly one way to supply a model config.
func resolveModelConfig(model string) (string, error) {
	catalog, err := resolveCatalogRoot()
	if err != nil {
		return "", err
	}
	// Record the catalog root that produced this run's model config, so the results
	// file can attribute the result to it (#1732, R1/S5). A side effect in the same
	// spirit as modelConfigDir: the root is not otherwise recoverable at the emit site,
	// and re-resolving there would re-emit the precedence announcement.
	resolvedCatalogRoot = catalog
	return resolveModelConfigInCatalog(model, catalog)
}

// resolveModelConfigInCatalog is resolveModelConfig with the catalog root supplied
// explicitly (the injectable core; production callers use resolveModelConfig, which
// resolves the root from the flag and the environment).
func resolveModelConfigInCatalog(model, catalog string) (string, error) {
	entryDir, err := catalogModelDir(model, catalog)
	if err != nil {
		return "", fmt.Errorf("--latency-model: invalid model name %q: %w", model, err)
	}

	// Read the catalog entry. Absent or non-HuggingFace-shaped is a hard error (R1):
	// there is no fallback that could supply the config, and inventing one is what NS-6
	// forbids. The file is never rewritten or deleted — a user-provided config with
	// non-standard field names is reported, not overwritten.
	entryPath := filepath.Join(entryDir, hfConfigFile)
	data, readErr := os.ReadFile(entryPath)
	if readErr != nil {
		return "", fmt.Errorf(
			"model %q is not in the catalog: no %s at %s (%v).\n"+
				"  BLIS does not fetch model configs at run time — a model runs only if it is catalogued.\n"+
				"  Add the entry at %s, or point --catalog / %s at a catalog that has it",
			model, hfConfigFile, entryPath, readErr, entryPath, catalogEnvVar,
		)
	}
	if !json.Valid(data) || !isHFConfig(data) {
		return "", fmt.Errorf(
			"catalog entry for model %q at %s is not a HuggingFace config.json: it lacks the expected "+
				"fields (num_hidden_layers, hidden_size, layers_block_type, or the same fields under text_config).\n"+
				"  Fix that catalog entry, or point --catalog / %s at a catalog that has a valid %s",
			model, entryPath, catalogEnvVar, hfConfigFile,
		)
	}
	logrus.Infof("--latency-model: using config from %s", entryDir)
	return entryDir, nil
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

// catalogModelDir returns the catalog entry directory for a model. Model names like
// "meta-llama/llama-3.1-8b-instruct" map to "<catalog>/llama-3.1-8b-instruct/".
// Returns an error if the catalog root is empty or the model name contains path
// traversal sequences.
//
// Note (I12): The org prefix is stripped, so different orgs with identical model names
// (e.g., "org-a/llama" and "org-b/llama") would share the same directory. This matches
// the catalog's one-directory-per-model convention and is acceptable because HuggingFace
// model names are unique within orgs, and BLIS uses hf_repo for case-sensitive HF API calls.
func catalogModelDir(model, catalog string) (string, error) {
	// Defensive: production callers get a non-empty root from resolveCatalogRoot, which
	// refuses when neither --catalog nor BLIS_CATALOG is set. An empty root here would
	// silently reintroduce the retired working-directory default (#1731).
	if catalog == "" {
		return "", fmt.Errorf("model catalog root is empty; pass --catalog or set %s", catalogEnvVar)
	}
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

	return filepath.Join(catalog, shortName), nil
}
