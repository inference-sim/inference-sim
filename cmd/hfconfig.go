package cmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"

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
	// catalogModelsSubdir is the models namespace inside a catalog CLONE ROOT (#1774).
	// --catalog / BLIS_CATALOG names the clone root, and model entries are read from
	// <catalog>/models/<short-name>/config.json. The authoritative blis-catalog
	// repository stores configs there, with workloads/, devices/, hardware/ and
	// networks/ as SIBLING namespaces under the same root — those siblings only compose
	// off a single root, which is why the root (not the models directory) is the
	// contract every downstream reader inherits.
	catalogModelsSubdir = "models"
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
		"environment variable (there is no default and no search path; both name the catalog "+
		"CLONE ROOT, whose model entries live at <catalog>/%s/<short-name>/%s. A relative path "+
		"is resolved against the current working directory)",
		catalogEnvVar, catalogModelsSubdir, hfConfigFile)
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
	candidates, err := catalogModelDirs(model, catalog)
	if err != nil {
		return "", fmt.Errorf("--latency-model: invalid model name %q: %w", model, err)
	}

	// Read the catalog entry. Absent from every layout, or non-HuggingFace-shaped, is a
	// hard error (R1): nothing outside the catalog could supply the config, and inventing
	// one is what NS-6 forbids. The file is never rewritten or deleted — a user-provided
	// config with non-standard field names is reported, not overwritten.
	entryDir, entryPath, data, err := readCatalogEntry(model, candidates)
	if err != nil {
		return "", err
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

// readCatalogEntry reads the model's config.json from the first candidate entry
// directory that has one, returning that directory, the file path it read, and the
// bytes. candidates come from catalogModelDirs in resolution order (canonical
// clone-root layout first, transition fallback second).
//
// Only ABSENCE advances to the next candidate: presence is decided by a stat, and a
// config.json that IS there but cannot be read is reported naming that path rather than
// silently bypassed by the transition fallback — a broken entry must never resolve to a
// different model's config (R1, NS-6). A malformed but readable entry is judged by the
// caller, for the same reason.
//
// "Absence" is specifically ENOENT (nothing at the path) or ENOTDIR (a non-directory
// sits on the path). A flat catalog is a directory of entries, so a root that holds an
// unrelated FILE named `models` makes the canonical candidate
// <catalog>/models/<name>/config.json surface ENOTDIR — not a broken entry, just no entry
// in THIS layout — and it must fall through to the flat candidate. Every OTHER stat
// failure (EACCES on the entry directory, an I/O error) is NOT absence: it is reported
// naming the path, never swallowed as absence, because collapsing it into the fallback
// could resolve a DIFFERENT model's config for an entry that is present but unstatable —
// the same hazard the readErr branch guards (R1, NS-6).
//
// Absent from every layout is a refusal that names every path looked at, plus the
// canonical path an entry belongs at — the operator-actionable half of NS-6, which is
// why the message is built here rather than signalled as a bare not-found.
func readCatalogEntry(model string, candidates []string) (entryDir, entryPath string, data []byte, err error) {
	// Defensive: catalogModelDirs always returns a non-empty list or an error. An empty
	// list here must not index-panic on candidates[0].
	if len(candidates) == 0 {
		return "", "", nil, fmt.Errorf("model %q: no catalog entry directory to look in", model)
	}
	lookedAt := make([]string, 0, len(candidates))
	for _, dir := range candidates {
		path := filepath.Join(dir, hfConfigFile)
		lookedAt = append(lookedAt, "    "+path)
		info, statErr := os.Stat(path)
		if statErr != nil {
			// Only ABSENCE advances to the transition fallback. ENOENT and ENOTDIR both
			// mean "no entry in this layout"; any other stat failure (EACCES, EIO) is
			// reported naming the path rather than letting the fallback silently resolve
			// a different model's config for an entry that is there but unstatable.
			if os.IsNotExist(statErr) || errors.Is(statErr, syscall.ENOTDIR) {
				continue
			}
			return "", "", nil, fmt.Errorf(
				"catalog entry for model %q at %s cannot be checked: %w.\n"+
					"  Fix that catalog path, or point --catalog / %s at a catalog whose %s is accessible",
				model, path, statErr, catalogEnvVar, hfConfigFile,
			)
		}
		// A directory sitting where config.json should be is "no entry in this
		// layout", not a broken entry — deliberately grouped with the ENOENT/ENOTDIR
		// absence cases above (it is not a plausible catalogued config, so the
		// stale-flat-shadowing hazard the errno classification guards does not apply).
		if info.IsDir() {
			continue
		}
		content, readErr := os.ReadFile(path)
		if readErr != nil {
			return "", "", nil, fmt.Errorf(
				"catalog entry for model %q at %s exists but is not readable: %w.\n"+
					"  Fix that catalog entry, or point --catalog / %s at a catalog that has a readable %s",
				model, path, readErr, catalogEnvVar, hfConfigFile,
			)
		}
		return dir, path, content, nil
	}

	canonical := filepath.Join(candidates[0], hfConfigFile)
	return "", "", nil, fmt.Errorf(
		"model %q is not in the catalog: no %s at any of\n%s\n"+
			"  BLIS does not fetch model configs at run time — a model runs only if it is catalogued.\n"+
			"  Add the entry at %s (--catalog / %s names the catalog CLONE ROOT; model entries live "+
			"under its %s/ namespace), or point --catalog / %s at a catalog that has it",
		model, hfConfigFile, strings.Join(lookedAt, "\n"),
		canonical, catalogEnvVar, catalogModelsSubdir, catalogEnvVar,
	)
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

// catalogModelDirs returns the candidate catalog entry directories for a model, in
// resolution order. Model names like "meta-llama/llama-3.1-8b-instruct" map to
//
//	<catalog>/models/llama-3.1-8b-instruct/   (canonical clone-root layout, #1774)
//	<catalog>/llama-3.1-8b-instruct/          (transition fallback, removed by #1771)
//
// Returns an error if the catalog root is empty or the model name contains path
// traversal sequences.
//
// #1774 settles what --catalog points at. BLIS used to implement ENTRIES-ROOT semantics
// (no models/ level) while the authoritative blis-catalog repository stores configs at
// <clone-root>/models/<name>/config.json with workloads/, devices/, hardware/ and
// networks/ as sibling namespaces — so an operator had to pass <clone-root>/models while
// every document described the clone root. The clone root is now the contract, because
// the sibling namespaces (which upcoming readers consult) only compose off a single root.
//
// The flat fallback is a TRANSITION affordance, not a second location to search: the
// bundled model_configs/ tree and every in-repo test catalog have no models/ level, so
// without it this change would instantly break `export BLIS_CATALOG=$PWD/model_configs`
// (INV-6). It also keeps the pre-#1774 invocation `--catalog <clone-root>/models`
// working, since <clone-root>/models/models/<name> is absent and resolution falls back.
// #1771 deletes the bundled tree and this fallback together.
//
// This function is PURE — it derives paths and touches no filesystem. Which candidate is
// the entry is decided by readCatalogEntry, the one place that reads. Relative and
// absolute roots are preserved as given: a relative candidate is resolved against the
// process working directory by the OS at read time, an absolute one is used as-is.
//
// Note (I12): The org prefix is stripped, so different orgs with identical model names
// (e.g., "org-a/llama" and "org-b/llama") would share the same directory. This matches
// the catalog's one-directory-per-model convention and is acceptable because HuggingFace
// model names are unique within orgs, and BLIS uses hf_repo for case-sensitive HF API calls.
func catalogModelDirs(model, catalog string) ([]string, error) {
	// Defensive: production callers get a non-empty root from resolveCatalogRoot, which
	// refuses when neither --catalog nor BLIS_CATALOG is set. An empty root here would
	// silently reintroduce the retired working-directory default (#1731).
	if catalog == "" {
		return nil, fmt.Errorf("model catalog root is empty; pass --catalog or set %s", catalogEnvVar)
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
		return nil, fmt.Errorf("model name %q contains invalid path components", model)
	}

	return []string{
		filepath.Join(catalog, catalogModelsSubdir, shortName),
		filepath.Join(catalog, shortName),
	}, nil
}
