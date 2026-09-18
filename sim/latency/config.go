package latency

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"regexp"
	"slices"
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
	"num_experts",        // Jamba
	"moe_num_experts",    // Dbrx
	"n_routed_experts",   // DeepSeek
	"num_local_experts",  // Mixtral
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

// LayersBlockTypeField is the HF config key whose list length expresses a model's
// total layer count when no num_hidden_layers scalar is declared (#1729 / NS-4).
// Exported so the CLI's HF-config presence detection recognizes exactly the key the
// parser can consume — the two must not disagree about what counts as a usable config.
const LayersBlockTypeField = "layers_block_type"

// numHiddenLayersField is the HF config key holding the layer-count scalar. Named once
// so the resolver that reads it and the diagnostic that names it cannot drift apart.
const numHiddenLayersField = "num_hidden_layers"

// BlockTypeLayerCount returns the number of layers declared as the length of the
// model's per-layer block-type list, i.e. len(layers_block_type). ParseHFConfig
// pivots text_config onto the top-level map, so the key is reachable as a top-level
// value for multimodal configs too.
//
// Some configs (Nemotron-3-Ultra-550B) omit the num_hidden_layers scalar entirely and
// express the layer count only as this list — one entry per layer, naming its block
// type ("attention", "mamba", …). BLIS read the scalar, got 0, and aborted before any
// simulation with "NumLayers must be > 0" (#1729), even though the count was right
// there in the config.
//
// Returns 0 when the key is absent, holds an empty list, or holds a non-list value, so
// the caller keeps the existing layer-count resolution (the scalar, and ultimately the
// loud validator failure) rather than a silent 0. Only the LENGTH is read — element
// types are irrelevant, mirroring LinearAttnFullLayerCount. Counting is deliberately
// all this does: per-type tallies / layer groups are a later release (R4a), so a
// derived count is a total layer count and nothing more.
func (c *HFConfig) BlockTypeLayerCount() int {
	blocks, ok := c.Raw[LayersBlockTypeField].([]any)
	if !ok {
		return 0
	}
	return len(blocks)
}

// ResolveNumLayers returns the model's total transformer-layer count: the
// num_hidden_layers scalar when it is present and non-zero, else the length of the
// block-type list (#1729 / NS-4). Returns 0 when neither source has an answer, so the
// per-backend validators still fail loudly (never a silent 0).
//
// The scalar wins whenever it answers, which makes the array a pure fallback: every
// config that declares num_hidden_layers — i.e. every currently-catalogued model —
// resolves exactly as it did before this fallback existed (INV-6). A present-but-
// negative scalar is returned as-is rather than overridden: it is bad input, and the
// validators' "NumLayers must be > 0" is the right response, not a second opinion.
//
// This is the single source of truth for layer-count resolution (R23 code-path
// parity); ExtractKVCapacityParams derives no layer count of its own.
func (c *HFConfig) ResolveNumLayers() int {
	if n, answered := c.numLayersScalar(); answered {
		return n
	}
	return c.BlockTypeLayerCount()
}

// numLayersScalar reports the num_hidden_layers scalar and whether it ANSWERS the
// layer-count question: present AND non-zero. It is the single predicate both
// ResolveNumLayers and the derivation warning in GetModelConfigFromHF read, so the
// two can never disagree about whether the fallback fired.
//
// Keeping them in one place is the point. The two predicates were briefly separate —
// the resolver falling back on "absent or 0" while the warning gate fired only on
// key ABSENCE — which let a config carrying "num_hidden_layers": 0 alongside a
// block-type list derive its count with no stderr line at all: exactly the silent
// reinterpretation of an operator's config that R1 forbids.
func (c *HFConfig) numLayersScalar() (int, bool) {
	n, ok := c.GetInt(numHiddenLayersField)
	return n, ok && n != 0
}

// jsonValueKind names the JSON type of a decoded config value in operator-facing terms,
// so a "wrong type" diagnostic can say WHAT the key holds instead of only that it was
// unusable. json.Unmarshal into `any` produces exactly these Go types.
func jsonValueKind(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "a boolean"
	case float64:
		return "a number"
	case string:
		return "a string"
	case []any:
		return "a list"
	case map[string]any:
		return "an object"
	default:
		return fmt.Sprintf("a %T", v)
	}
}

// layerCountEvidence describes what each of the two keys ResolveNumLayers consults
// actually contributes, one clause per key. PRECONDITION: ResolveNumLayers returned 0,
// i.e. neither key answered — the clauses are phrased for that state.
//
// It exists because the per-backend validators' "ModelConfig.NumLayers must be > 0"
// names a Go STRUCT FIELD, not a config key: it reports the number BLIS ended up with
// and says nothing about where BLIS looked for it. Since #1729 there are two places to
// look, and a present-but-malformed layers_block_type (a bare string instead of a list,
// say) reads as "absent" to the resolver — so an operator could be staring at a config
// that visibly states its layer count while being told the count is missing. Naming both
// keys, and saying what is wrong with each, is the difference between a one-edit fix and
// a source dive.
func (c *HFConfig) layerCountEvidence() []string {
	clauses := make([]string, 0, 2)

	// The type tests mirror the readers exactly: GetInt accepts only a JSON number
	// (float64 after json.Unmarshal), and BlockTypeLayerCount only a []any. Asserted on
	// the type directly rather than by comparing jsonValueKind's prose, which would make
	// the branch depend on the wording of a message.
	switch v, present := c.Raw[numHiddenLayersField]; {
	case !present:
		clauses = append(clauses, fmt.Sprintf("%q is absent", numHiddenLayersField))
	default:
		if _, isNumber := v.(float64); !isNumber {
			clauses = append(clauses, fmt.Sprintf(
				"%q holds %s, not a number", numHiddenLayersField, jsonValueKind(v)))
		} else {
			// Present and numeric, yet the scalar did not answer ⇒ it is 0
			// (numLayersScalar treats 0 as no answer, deliberately: 0 layers is not a model).
			clauses = append(clauses, fmt.Sprintf("%q is 0", numHiddenLayersField))
		}
	}

	switch v, present := c.Raw[LayersBlockTypeField]; {
	case !present:
		clauses = append(clauses, fmt.Sprintf("%q is absent", LayersBlockTypeField))
	default:
		if _, isList := v.([]any); !isList {
			clauses = append(clauses, fmt.Sprintf(
				"%q holds %s, not a list (only a list's LENGTH is read)", LayersBlockTypeField, jsonValueKind(v)))
		} else {
			clauses = append(clauses, fmt.Sprintf("%q holds an empty list", LayersBlockTypeField))
		}
	}

	return clauses
}

func parseHWConfig(HWConfigFilePath string) (map[string]sim.HardwareCalib, error) {
	data, err := os.ReadFile(HWConfigFilePath)
	if err != nil {
		return nil, fmt.Errorf("read hardware config %q: %w", HWConfigFilePath, err)
	}

	// #1694: reject the pre-#1694 per-COLLECTIVE key. A legacy "InterNodeLatencyUs" is
	// caught by the generic unknown-key check below too (#1728), but this guard runs
	// FIRST so the operator gets the migration message instead: the value is NOT a
	// drop-in rename — the unit changed from µs-per-collective to µs-per-hop, so it must
	// be re-divided by the hop count. A bare "unknown key" would lose that hint, and
	// silently dropping the value to 0 would discard a calibrated number with no warning
	// at all (the bandwidths alone already satisfy HasInterconnectCalibration, so
	// warnIfCrossNodeUnpriced stays quiet) — the R1 "never silent" case.
	if err := rejectLegacyInterNodeLatencyKey(data); err != nil {
		return nil, err
	}

	// #1728: parse strictly. This was the last permissively-parsed config path in BLIS
	// (every YAML path uses decoder.KnownFields(true)), which meant an arbitrary
	// misspelling of a numeric key — IntraNodeBandwidthGBps for IntraNodeBwGBps, MemoryGB
	// for MemoryGiB — decoded to 0 and produced a plausible-but-wrong result with no
	// diagnostic. A blocklist (the legacy-key guard above) catches one known key; only an
	// allowlist catches an arbitrary typo.
	if err := rejectUnknownHardwareCalibKeys(data); err != nil {
		return nil, err
	}

	var HardwareList map[string]sim.HardwareCalib
	if err := json.Unmarshal(data, &HardwareList); err != nil {
		return nil, fmt.Errorf("parse hardware config JSON: %w", err)
	}
	return HardwareList, nil
}

// hardwareCalibProvenanceKeys are the documentation-only keys a hardware-config GPU
// entry may carry alongside its numeric fields. The bundled hardware_config.json uses
// both to record where each calibration came from (Discussion #589 for the MFU values,
// the datasheet reasoning for the interconnect bandwidths) — provenance that belongs
// next to the numbers it explains, since a reader checking a value looks at the entry,
// not at a doc page. They are accepted and ignored: the value decode never reads them.
// These exact spellings are what a config must use — like the calibration fields, a
// case-only variant ("_Comment") is rejected with the canonical spelling named.
var hardwareCalibProvenanceKeys = []string{"_comment", "_comment_interconnect"}

// hardwareCalibKnownKeys maps the ASCII-lowercased form of every JSON key
// parseHWConfig accepts on a GPU entry to its canonical spelling. It is derived from
// sim.HardwareCalib's struct tags rather than hand-listed, so a field added to the
// struct is accepted with no parser change and the accepted set cannot drift from the
// fields the decoder actually populates (R23 code-path parity).
//
// The folded form is kept so a key that differs from a declared field ONLY in letter
// case can be diagnosed as such (encoding/json would silently accept it — see
// rejectUnknownHardwareCalibKeys).
var hardwareCalibKnownKeys = buildHardwareCalibKnownKeys()

func buildHardwareCalibKnownKeys() map[string]string {
	typ := reflect.TypeOf(sim.HardwareCalib{})
	keys := make(map[string]string, typ.NumField()+len(hardwareCalibProvenanceKeys))
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if !f.IsExported() {
			continue // unexported fields are invisible to encoding/json
		}
		name := f.Name
		if tag, ok := f.Tag.Lookup("json"); ok {
			tagName := strings.Split(tag, ",")[0]
			if tagName == "-" {
				continue // explicitly not part of the JSON surface
			}
			if tagName != "" {
				name = tagName
			}
		}
		keys[strings.ToLower(name)] = name
	}
	for _, p := range hardwareCalibProvenanceKeys {
		keys[strings.ToLower(p)] = p
	}
	return keys
}

// hardwareCalibFieldKeyList returns the canonical calibration-field keys (i.e. excluding
// the provenance keys, which are listed separately in the diagnostic) in sorted order.
// Sorted so the message is byte-identical run to run (INV-6).
func hardwareCalibFieldKeyList() []string {
	names := make([]string, 0, len(hardwareCalibKnownKeys))
	for _, canonical := range hardwareCalibKnownKeys {
		if slices.Contains(hardwareCalibProvenanceKeys, canonical) {
			continue
		}
		names = append(names, canonical)
	}
	sort.Strings(names)
	return names
}

// rejectUnknownHardwareCalibKeys scans a hardware-config file for keys that are not
// fields of sim.HardwareCalib (nor one of the ignored provenance keys) and returns a
// fatal, actionable error listing every one of them with the GPU entry it appears under
// (#1728).
//
// It is the allowlist equivalent of json.DisallowUnknownFields, chosen over the decoder
// flag for three reasons: the decoder reports only `unknown field "X"` without saying
// WHICH GPU entry it came from (useless in a file with a dozen entries); it would reject
// the provenance keys the bundled file depends on; and it inherits encoding/json's
// case-insensitive field matching, which hides a whole class of near-miss key (below).
//
// Two offender classes, both errors, deliberately distinguished:
//
//   - UNKNOWN: no declared field matches even case-insensitively. This is the silent-zero
//     case the fix exists for — the decoder drops the key and the field reads 0.
//   - CASE MISMATCH: the key matches a declared field when folded but is not spelled
//     canonically (e.g. IntraNodeBwGbps for IntraNodeBwGBps). encoding/json's fallback
//     accepts these and reads the value CORRECTLY, so they never produced a wrong number —
//     but relying on that leaves the file one letter away from a genuine typo, and two
//     spellings of one field in the same entry resolve last-wins. Requiring the canonical
//     spelling costs an operator one trivially-actionable error naming the exact key to use.
//
// The scan is file-wide rather than scoped to the GPU being run — unlike GetHWConfig's
// per-GPU ValidateInterconnect call, which checks whether a PRESENT value is usable and so
// only matters for the entry in use. A misspelled key is different in kind: it is a
// file-integrity defect that nothing else will ever surface, which is exactly why it needs
// to fail here. This also matches the sibling rejectLegacyInterNodeLatencyKey guard, which
// has always been file-wide.
func rejectUnknownHardwareCalibKeys(data []byte) error {
	var raw map[string]map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		// Not the expected shape; the main decode in parseHWConfig will produce the real
		// parse error rather than a confusing "unknown key" one.
		return nil //nolint:nilerr // defer the diagnostic to the primary json.Unmarshal
	}

	var offenders []string
	for gpu, fields := range raw {
		for key := range fields {
			canonical, folds := hardwareCalibKnownKeys[strings.ToLower(key)]
			switch {
			case folds && canonical == key:
				continue // canonical spelling of a declared field (or a provenance key)
			case folds:
				offenders = append(offenders, fmt.Sprintf(
					"GPU %q key %q (differs from the declared key %q only in letter case — use the canonical spelling)",
					gpu, key, canonical))
			default:
				offenders = append(offenders, fmt.Sprintf("GPU %q key %q (unknown)", gpu, key))
			}
		}
	}
	if len(offenders) == 0 {
		return nil
	}
	// Sorted so a multi-offender diagnostic is byte-identical across runs (INV-6) rather
	// than following Go's randomized map order.
	sort.Strings(offenders)
	return fmt.Errorf("hardware config declares unrecognized key(s): %s. Parsing is strict: an "+
		"unrecognized key is almost always a misspelling, and accepting it would leave the "+
		"intended field at 0 — a plausible-but-wrong bandwidth, MFU or memory capacity with no "+
		"diagnostic anywhere. Valid keys are %v, plus the ignored provenance keys %v",
		strings.Join(offenders, "; "), hardwareCalibFieldKeyList(), hardwareCalibProvenanceKeys)
}

// rejectLegacyInterNodeLatencyKey scans a hardware-config file for the pre-#1694
// per-collective "InterNodeLatencyUs" key on any GPU entry and returns a fatal,
// actionable error if present. Named per-GPU so the operator knows where to look.
func rejectLegacyInterNodeLatencyKey(data []byte) error {
	var raw map[string]map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		// Not the expected shape; the main decode below will produce the real parse error.
		return nil //nolint:nilerr // defer the diagnostic to the primary json.Unmarshal
	}
	var offenders []string
	for gpu, fields := range raw {
		if _, ok := fields["InterNodeLatencyUs"]; ok {
			offenders = append(offenders, gpu)
		}
	}
	if len(offenders) == 0 {
		return nil
	}
	sort.Strings(offenders)
	return fmt.Errorf("hardware config declares the removed per-collective key %q on GPU(s) %v; "+
		"it was replaced by the per-HOP key %q in #1694 (the charge is now n_steps·α_hop, so the "+
		"unit changed from µs-per-collective to µs-per-hop). This is a RECALIBRATION, not a rename: "+
		"divide the old value by the collective's cross-node hop count before setting %q — do not "+
		"copy it verbatim. Remove the old key to proceed",
		"InterNodeLatencyUs", offenders, "InterNodeHopLatencyUs", "InterNodeHopLatencyUs")
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
	// #1530: the optional interconnect calibration is validated here, at the load
	// boundary, so a malformed hardware config fails identically regardless of which
	// latency backend will consume it — the roofline backend ignores these fields, and
	// silently accepting a typo under roofline while rejecting it under trained-physics
	// would be a confusing asymmetry (R23). Only the requested GPU is checked, so an
	// unrelated malformed entry elsewhere in the file does not block an unrelated run.
	if err := config.ValidateInterconnect(); err != nil {
		return sim.HardwareCalib{}, fmt.Errorf("hardware config %q, GPU %q: %w", HWConfigFilePath, GPU, err)
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

	// Total layer count: the num_hidden_layers scalar when it answers, else the length of
	// the block-type list (#1729 / NS-4). Some configs (Nemotron-3-Ultra-550B) declare no
	// scalar at all, and BLIS used to abort before any simulation with "NumLayers must be
	// > 0"; the count was in the config, just phrased as a list. The scalar still wins
	// whenever it answers, so every config that declares it parses exactly as before
	// (INV-6).
	//
	// The warning gate reads the SAME numLayersScalar predicate the resolver does, so it
	// fires for every derivation — including the "num_hidden_layers": 0 case, which the
	// resolver treats as no answer. Gating on mere key absence here would let that config
	// be silently reinterpreted (R1).
	numLayers := hf.ResolveNumLayers()
	// Neither key answered. Fail HERE, naming both keys the resolver consults, rather
	// than deferring to the per-backend "ModelConfig.NumLayers must be > 0" validators:
	// that message names a struct field the operator's config does not contain, and since
	// #1729 there are two config keys that could have supplied the count — one of which
	// (a present-but-wrong-typed layers_block_type) is indistinguishable from absent by
	// the time the validator runs. The backend validators keep their own check as
	// defense in depth, for a ModelConfig built by any other route.
	//
	// Scope: only the no-evidence case (numLayers == 0). A present-but-NEGATIVE scalar
	// ANSWERED — it is bad input rather than missing input, and ResolveNumLayers
	// deliberately passes it through so the validators report the actual value; second-
	// guessing it here would replace a precise "got -5" with a "cannot determine" that is
	// simply untrue.
	if numLayers == 0 {
		return nil, fmt.Errorf("GetModelConfigFromHF: cannot determine the model's transformer-layer "+
			"count: %s. Set %q to the model's total layer count, or declare %q as a list with one entry "+
			"per layer", strings.Join(hf.layerCountEvidence(), ", and "), numHiddenLayersField, LayersBlockTypeField)
	}
	if _, scalarAnswered := hf.numLayersScalar(); !scalarAnswered && numLayers > 0 {
		// Never silent (R1), and warn rather than inform: the count is derived, and every
		// entry is counted as one transformer layer whatever type it names — so a hybrid
		// block list (Nemotron's "attention"/"mamba" mix) prices its non-attention layers
		// as full attention. Pessimistic, and worth one stderr line at the default log
		// level, exactly as the hybrid-attention detection is (#1635/#1636). Only fires for
		// a config whose scalar has no answer, so catalogued models stay quiet.
		logrus.Warnf("HuggingFace config declares no usable num_hidden_layers (absent or 0); derived NumLayers=%d "+
			"from len(%s). Every entry is counted as one transformer layer regardless of the block type it names, "+
			"so a hybrid (e.g. attention/mamba) block list is priced as all-attention — pessimistic. Per-type "+
			"layer groups are not modeled",
			numLayers, LayersBlockTypeField)
	}

	modelConfig := &sim.ModelConfig{
		NumLayers:              numLayers,
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
//   - "fp8" / "fp8_e4m3" / "fp8_e5m2" / "fp8_inc" → (1.0, true): vLLM maps every fp8
//     variant to 1 byte/element (fp8_inc is Intel Gaudi fp8). fp8 KV under bf16 compute
//     halves KV memory.
//   - "bf16" / "bfloat16" / "fp16" / "float16" / "half" → (2.0, true)
//   - "fp32" / "float32" / "float" → (4.0, true)
//
// KV storage precision is independent of weight quantization (WeightBytesPerParam) —
// they are separate vLLM engine args. Matching is case-insensitive and trims spaces.
func KVCacheDtypeToBytes(dtype string) (float64, bool) {
	switch strings.ToLower(strings.TrimSpace(dtype)) {
	case "", "auto":
		return 0, true
	case "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8e4m3", "fp8e5m2":
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
