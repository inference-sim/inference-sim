package latency_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// TestBlockTypeLayerCount covers the #1729 (NS-4) block-type-array reader directly,
// including the robustness cases the end-to-end parse tests do not reach: an absent
// key, an empty list, and a non-list value all yield 0, so the caller keeps the loud
// layer-count failure rather than a silent 0. Since #1777 that failure is
// GetModelConfigFromHF's parse-boundary refusal naming both keys (asserted by
// TestGetModelConfig_NoLayerEvidenceIsRefusedNamingBothKeys), with the per-backend
// "NumLayers must be > 0" validators retained as defense in depth (the original AC-4
// claim, now asserted by TestBothBackendsRejectZeroLayerModelConfig).
//
// Only the list LENGTH is read, so element types are irrelevant — a config may spell
// its blocks as strings ("attention"/"mamba"), numbers, or a mixture.
func TestBlockTypeLayerCount(t *testing.T) {
	cases := []struct {
		name string
		raw  map[string]any
		want int
	}{
		{"absent", map[string]any{"num_hidden_layers": float64(52)}, 0},
		{
			"string_elements",
			map[string]any{"layers_block_type": []any{"attention", "mamba", "mamba", "attention"}},
			4,
		},
		{
			"numeric_elements",
			map[string]any{"layers_block_type": []any{float64(0), float64(1), float64(1)}},
			3,
		},
		{
			"mixed_elements",
			map[string]any{"layers_block_type": []any{"attention", float64(1), nil}},
			3,
		},
		{"empty_list", map[string]any{"layers_block_type": []any{}}, 0},
		{"wrong_type_scalar", map[string]any{"layers_block_type": float64(52)}, 0},
		{"wrong_type_string", map[string]any{"layers_block_type": "attention,mamba"}, 0},
		{"wrong_type_map", map[string]any{"layers_block_type": map[string]any{"0": "attention"}}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hf := &latency.HFConfig{Raw: tc.raw}
			if got := hf.BlockTypeLayerCount(); got != tc.want {
				t.Errorf("BlockTypeLayerCount() = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestResolveNumLayers is the BC-1/BC-2/BC-3 resolution law: the
// num_hidden_layers scalar wins whenever it is present and non-zero (so every
// currently-catalogued model is byte-identical, INV-6); the block-type array is a
// pure fallback consulted only when the scalar has no answer; and with neither, the
// resolver returns 0 — which since #1777 GetModelConfigFromHF refuses at the parse
// boundary rather than passing to the per-backend validators (both retain their own
// NumLayers > 0 check as defense in depth).
func TestResolveNumLayers(t *testing.T) {
	cases := []struct {
		name string
		raw  map[string]any
		want int
	}{
		{"scalar_only", map[string]any{"num_hidden_layers": float64(52)}, 52},
		{
			"array_only",
			map[string]any{"layers_block_type": []any{"a", "m", "m", "a", "m"}},
			5,
		},
		{
			// BC-2: the scalar wins even when an array of a DIFFERENT length is present.
			"scalar_wins_over_array",
			map[string]any{
				"num_hidden_layers": float64(52),
				"layers_block_type": []any{"a", "m", "m"},
			},
			52,
		},
		{
			// A scalar of 0 has no answer, so the array is consulted (the issue's
			// "absent (or 0)").
			"zero_scalar_falls_back",
			map[string]any{
				"num_hidden_layers": float64(0),
				"layers_block_type": []any{"a", "m", "m"},
			},
			3,
		},
		{
			// A negative scalar is bad input, not a reason to consult a second source:
			// it is returned as-is and keeps failing loudly at the validators.
			"negative_scalar_not_overridden",
			map[string]any{
				"num_hidden_layers": float64(-4),
				"layers_block_type": []any{"a", "m", "m"},
			},
			-4,
		},
		{"neither", map[string]any{"hidden_size": float64(4096)}, 0},
		{
			"unusable_array_and_no_scalar",
			map[string]any{"hidden_size": float64(4096), "layers_block_type": []any{}},
			0,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hf := &latency.HFConfig{Raw: tc.raw}
			if got := hf.ResolveNumLayers(); got != tc.want {
				t.Errorf("ResolveNumLayers() = %d, want %d", got, tc.want)
			}
		})
	}
}

// nemotronStyleConfig is a Nemotron-3-Ultra-550B-shaped config: the layer count is
// expressed ONLY as the length of layers_block_type (52 entries), with no top-level
// num_hidden_layers scalar. Every other field is present and ordinary so the only
// thing under test is the layer-count derivation.
const nemotronStyleConfig = `{
	"architectures": ["NemotronForCausalLM"],
	"layers_block_type": [
		"attention", "mamba", "mamba", "mamba", "attention", "mamba", "mamba", "mamba",
		"attention", "mamba", "mamba", "mamba", "attention", "mamba", "mamba", "mamba",
		"attention", "mamba", "mamba", "mamba", "attention", "mamba", "mamba", "mamba",
		"attention", "mamba", "mamba", "mamba", "attention", "mamba", "mamba", "mamba",
		"attention", "mamba", "mamba", "mamba", "attention", "mamba", "mamba", "mamba",
		"attention", "mamba", "mamba", "mamba", "attention", "mamba", "mamba", "mamba",
		"attention", "mamba", "mamba", "mamba"
	],
	"hidden_size": 4096,
	"num_attention_heads": 32,
	"num_key_value_heads": 8,
	"intermediate_size": 14336,
	"vocab_size": 32000,
	"hidden_act": "silu",
	"torch_dtype": "bfloat16"
}`

// parseConfigJSON writes body to a temp config.json and returns the parsed ModelConfig.
func parseConfigJSON(t *testing.T, body string) *sim.ModelConfig {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	mc, err := latency.GetModelConfig(path)
	if err != nil {
		t.Fatalf("GetModelConfig: %v", err)
	}
	return mc
}

// TestGetModelConfig_LayersBlockTypeFallback is BC-1 at the parse boundary: a
// Nemotron-style config whose only layer-count evidence is layers_block_type yields
// NumLayers == len(layers_block_type) instead of the 0 that aborted the run (#1729).
func TestGetModelConfig_LayersBlockTypeFallback(t *testing.T) {
	mc := parseConfigJSON(t, nemotronStyleConfig)
	if mc.NumLayers != 52 {
		t.Errorf("NumLayers = %d, want 52 (len layers_block_type)", mc.NumLayers)
	}
	// The scope guard: this PR counts the array, it does not introduce layer groups or
	// per-type tallies (R4a). A hybrid KV-bearing count comes from linear_attn_config
	// (#1635), which this config does not declare, so it must stay 0 → all layers.
	if mc.KVBearingLayers != 0 {
		t.Errorf("KVBearingLayers = %d, want 0 (no linear_attn_config; the block-type array must not be tallied by type)", mc.KVBearingLayers)
	}
	if got := mc.EffectiveKVBearingLayers(); got != 52 {
		t.Errorf("EffectiveKVBearingLayers() = %d, want 52 (falls back to the derived NumLayers)", got)
	}
}

// TestGetModelConfig_ScalarWinsOverBlockTypeArray is BC-2 / INV-6: a config that
// declares num_hidden_layers is unaffected by the new fallback even when a
// layers_block_type of a different length is also present. This is the guard that
// keeps every currently-catalogued model byte-identical — the 30B Nemotron sibling
// declares 52 layers, and the scalar must remain authoritative.
func TestGetModelConfig_ScalarWinsOverBlockTypeArray(t *testing.T) {
	body := `{
		"num_hidden_layers": 52,
		"layers_block_type": ["attention", "mamba", "mamba"],
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"intermediate_size": 14336,
		"vocab_size": 32000,
		"torch_dtype": "bfloat16"
	}`
	if got := parseConfigJSON(t, body).NumLayers; got != 52 {
		t.Errorf("NumLayers = %d, want 52 (the declared scalar, not len(layers_block_type)=3)", got)
	}
}

// TestGetModelConfig_DerivationWarning is the R1 gate: whenever the layer count is
// DERIVED from the block-type array, the operator gets one stderr line saying so — and
// whenever the num_hidden_layers scalar is what answered, they get silence.
//
// The two must agree, which is the whole point of the test. A warning gated on the
// scalar KEY being absent, rather than on the scalar having no ANSWER, disagrees with
// the resolver on exactly one input: "num_hidden_layers": 0 alongside a valid array.
// That config derives its count (correctly) and used to do it silently — a
// non-standard reinterpretation of an operator's config at the default log level,
// which R1 forbids. The zero_scalar case below is the regression guard.
func TestGetModelConfig_DerivationWarning(t *testing.T) {
	cases := []struct {
		name        string
		body        string
		wantNumLays int
		wantWarn    bool
	}{
		{
			// The scalar is present but has no answer, so the array is consulted:
			// derived ⇒ warn.
			name: "zero_scalar_derives_and_warns",
			body: `{
				"num_hidden_layers": 0,
				"layers_block_type": ["attention", "mamba", "mamba"],
				"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
				"intermediate_size": 14336, "vocab_size": 32000, "torch_dtype": "bfloat16"
			}`,
			wantNumLays: 3,
			wantWarn:    true,
		},
		{
			name:        "absent_scalar_derives_and_warns",
			body:        nemotronStyleConfig,
			wantNumLays: 52,
			wantWarn:    true,
		},
		{
			// The scalar answered, so nothing was derived: catalogued models stay quiet
			// (INV-6 — the stderr stream is part of the byte-identity claim).
			name: "scalar_answers_stays_quiet",
			body: `{
				"num_hidden_layers": 52,
				"layers_block_type": ["attention", "mamba", "mamba"],
				"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
				"intermediate_size": 14336, "vocab_size": 32000, "torch_dtype": "bfloat16"
			}`,
			wantNumLays: 52,
			wantWarn:    false,
		},
		{
			// A negative scalar DOES answer (it is bad input, not a reason to consult a
			// second source): nothing is derived, so no derivation warning — the
			// backends' "NumLayers must be > 0" is the loud response instead.
			name: "negative_scalar_stays_quiet",
			body: `{
				"num_hidden_layers": -4,
				"layers_block_type": ["attention", "mamba", "mamba"],
				"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
				"intermediate_size": 14336, "vocab_size": 32000, "torch_dtype": "bfloat16"
			}`,
			wantNumLays: -4,
			wantWarn:    false,
		},
		// The "neither source answers" case is NOT in this table: since #1777 the parse
		// REFUSES rather than returning a zero-layer ModelConfig, so there is no
		// NumLayers to assert here. Its companion claim — that nothing was derived, so
		// no derivation warning fires — is asserted on the error path by
		// TestGetModelConfig_NoLayerEvidenceIsRefusedNamingBothKeys.
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			prev := logrus.StandardLogger().Out
			logrus.SetOutput(&buf)
			t.Cleanup(func() { logrus.SetOutput(prev) })

			mc := parseConfigJSON(t, tc.body)
			if mc.NumLayers != tc.wantNumLays {
				t.Errorf("NumLayers = %d, want %d", mc.NumLayers, tc.wantNumLays)
			}

			warned := strings.Contains(buf.String(), "derived NumLayers=")
			if warned != tc.wantWarn {
				t.Errorf("derivation warning emitted = %v, want %v; log output: %q",
					warned, tc.wantWarn, buf.String())
			}
			if tc.wantWarn && !strings.Contains(buf.String(),
				fmt.Sprintf("derived NumLayers=%d from len(%s)", tc.wantNumLays, latency.LayersBlockTypeField)) {
				t.Errorf("warning must name the derived count and its source, got: %q", buf.String())
			}
		})
	}
}

// TestGetModelConfig_BlockTypeArrayUnderTextConfig covers the multimodal shape:
// ParseHFConfig pivots text_config onto the top-level map, so a block-type array
// declared there is reachable by the same reader.
func TestGetModelConfig_BlockTypeArrayUnderTextConfig(t *testing.T) {
	body := `{
		"text_config": {
			"layers_block_type": ["attention", "mamba", "mamba", "attention"],
			"hidden_size": 4096,
			"num_attention_heads": 32,
			"num_key_value_heads": 8,
			"intermediate_size": 14336,
			"vocab_size": 32000,
			"torch_dtype": "bfloat16"
		}
	}`
	if got := parseConfigJSON(t, body).NumLayers; got != 4 {
		t.Errorf("NumLayers = %d, want 4 (len text_config.layers_block_type)", got)
	}
}

// blockTypeHW is a minimal valid hardware calibration for the backend-acceptance
// tests below: only the fields both validators require are set.
func blockTypeHW() sim.HardwareCalib {
	return sim.HardwareCalib{
		TFlopsPeak: 989.5,
		BwPeakTBs:  3.35,
		MfuPrefill: 0.5,
		MfuDecode:  0.5,
		MemoryGiB:  80.0,
	}
}

// blockTypeCoeffs is a minimal valid trained-physics coefficient set (3 alpha, 7 beta).
func blockTypeCoeffs() sim.LatencyCoeffs {
	return sim.LatencyCoeffs{
		AlphaCoeffs: []float64{1, 1, 1},
		BetaCoeffs:  []float64{1, 1, 1, 1, 1, 1, 1},
	}
}

// TestBothBackendsAcceptDerivedLayerCount is BC-1 / AC-3: the derived count clears
// BOTH latency backends' NumLayers validation — the roofline check in
// ValidateRooflineConfig and the trained-physics check in NewTrainedPhysicsModel.
// A config whose only layer-count evidence is the array is no longer a fatal.
func TestBothBackendsAcceptDerivedLayerCount(t *testing.T) {
	mc := parseConfigJSON(t, nemotronStyleConfig)
	if mc.NumLayers != 52 {
		t.Fatalf("precondition: NumLayers = %d, want 52", mc.NumLayers)
	}

	if err := latency.ValidateRooflineConfig(*mc, blockTypeHW()); err != nil {
		t.Errorf("roofline backend rejected the derived layer count: %v", err)
	}

	hw := sim.NewModelHardwareConfig(*mc, blockTypeHW(), "nvidia/nemotron-style", "H100",
		1, 1, false, "", sim.LatencyBackendTrainedPhysics, 0)
	if _, err := latency.NewTrainedPhysicsModel(blockTypeCoeffs(), hw); err != nil {
		t.Errorf("trained-physics backend rejected the derived layer count: %v", err)
	}
}

// shapeFieldsForLayerEvidence is the rest of a minimally-valid HF config: everything a
// parse needs EXCEPT layer-count evidence, so a case below is isolating that evidence.
const shapeFieldsForLayerEvidence = `"hidden_size": 4096, "num_attention_heads": 32, ` +
	`"num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 32000, "torch_dtype": "bfloat16"`

// TestGetModelConfig_NoLayerEvidenceIsRefusedNamingBothKeys is #1777 item 1: a config
// that supplies NO usable layer count is refused AT THE PARSE BOUNDARY, and the
// diagnostic names both keys the resolver consults — num_hidden_layers and
// layers_block_type — plus what is wrong with each.
//
// It replaces the deferral to the per-backend "ModelConfig.NumLayers must be > 0", which
// names a Go struct field rather than either config key. The malformed-type rows are the
// reason the message has to name both keys explicitly: a layers_block_type holding a bare
// string or a number is INDISTINGUISHABLE from an absent one by the time a backend
// validator runs, so an operator could be looking at a config that visibly states its
// layer count while being told the count could not be determined.
func TestGetModelConfig_NoLayerEvidenceIsRefusedNamingBothKeys(t *testing.T) {
	cases := []struct {
		name string
		body string
		// wantClauses are substrings the diagnostic must contain — the per-key verdicts.
		wantClauses []string
	}{
		{
			name: "both_keys_absent",
			body: `{` + shapeFieldsForLayerEvidence + `}`,
			wantClauses: []string{
				`"num_hidden_layers" is absent`,
				`"layers_block_type" is absent`,
			},
		},
		{
			name: "scalar_absent_list_empty",
			body: `{"layers_block_type": [], ` + shapeFieldsForLayerEvidence + `}`,
			wantClauses: []string{
				`"num_hidden_layers" is absent`,
				`"layers_block_type" holds an empty list`,
			},
		},
		{
			name: "scalar_zero_list_empty",
			body: `{"num_hidden_layers": 0, "layers_block_type": [], ` + shapeFieldsForLayerEvidence + `}`,
			wantClauses: []string{
				`"num_hidden_layers" is 0`,
				`"layers_block_type" holds an empty list`,
			},
		},
		{
			// The malformed-type case the issue calls out: the key is present and even
			// carries the block names, but as a string rather than a list. Only a list's
			// LENGTH is read, so this contributes nothing — and the operator has to be
			// told that, not told the key is missing.
			name: "list_is_a_string",
			body: `{"layers_block_type": "attention", ` + shapeFieldsForLayerEvidence + `}`,
			wantClauses: []string{
				`"num_hidden_layers" is absent`,
				`"layers_block_type" holds a string, not a list`,
			},
		},
		{
			name: "list_is_a_number",
			body: `{"layers_block_type": 52, ` + shapeFieldsForLayerEvidence + `}`,
			wantClauses: []string{
				`"num_hidden_layers" is absent`,
				`"layers_block_type" holds a number, not a list`,
			},
		},
		{
			name: "list_is_an_object",
			body: `{"layers_block_type": {"0": "attention"}, ` + shapeFieldsForLayerEvidence + `}`,
			wantClauses: []string{
				`"num_hidden_layers" is absent`,
				`"layers_block_type" holds an object, not a list`,
			},
		},
		{
			// A scalar of the wrong type does not answer either (GetInt only accepts a
			// JSON number), and must be diagnosed as a type problem rather than as absent.
			name: "scalar_is_a_string",
			body: `{"num_hidden_layers": "52", ` + shapeFieldsForLayerEvidence + `}`,
			wantClauses: []string{
				`"num_hidden_layers" holds a string, not a number`,
				`"layers_block_type" is absent`,
			},
		},
		{
			name: "scalar_is_null",
			body: `{"num_hidden_layers": null, "layers_block_type": [], ` + shapeFieldsForLayerEvidence + `}`,
			wantClauses: []string{
				`"num_hidden_layers" holds null, not a number`,
				`"layers_block_type" holds an empty list`,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var log bytes.Buffer
			prev := logrus.StandardLogger().Out
			logrus.SetOutput(&log)
			t.Cleanup(func() { logrus.SetOutput(prev) })

			path := filepath.Join(t.TempDir(), "config.json")
			if err := os.WriteFile(path, []byte(tc.body), 0644); err != nil {
				t.Fatalf("write config: %v", err)
			}
			mc, err := latency.GetModelConfig(path)
			if err == nil {
				t.Fatalf("a config with no usable layer count must be refused, got NumLayers=%d", mc.NumLayers)
			}
			for _, want := range tc.wantClauses {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("diagnostic must contain %s\ngot: %v", want, err)
				}
			}
			// Both keys named regardless of which one failed how, so the operator learns
			// there are two ways to supply the count.
			for _, key := range []string{"num_hidden_layers", latency.LayersBlockTypeField} {
				if !strings.Contains(err.Error(), key) {
					t.Errorf("diagnostic must name %q (an operator has two ways to fix this)\ngot: %v", key, err)
				}
			}
			// Nothing was derived, so the derivation warning must NOT fire: the refusal is
			// the whole message, and a warning about a count that does not exist would be
			// noise contradicting it.
			if strings.Contains(log.String(), "derived NumLayers=") {
				t.Errorf("derivation warning must not fire when nothing was derived; log: %q", log.String())
			}
		})
	}
}

// TestGetModelConfigFromHF_JSONNumberScalarIsDiagnosedAsADecoderViolation covers the one
// way the refusal above can be reached WITHOUT a defective config: HFConfig.Raw is
// exported and GetModelConfigFromHF takes an already-built *HFConfig, so a caller decoding
// with json.Decoder.UseNumber() supplies json.Number where every reader expects float64.
// The scalar then does not answer and the parse refuses — correctly, since accepting it in
// GetInt alone would half-decode the config (see the HFConfig.Raw contract).
//
// What must NOT happen is the refusal calling a genuine JSON number "not a number": that
// reads as a defect in the config file and sends the caller editing a value that is already
// right, when the fix is one line away in their decoder. So the diagnostic has to name the
// decoder — the same "say what is actually wrong with which key" standard the config-side
// clauses are held to.
func TestGetModelConfigFromHF_JSONNumberScalarIsDiagnosedAsADecoderViolation(t *testing.T) {
	// Built the way a caller reaching for UseNumber would, rather than by hand-writing
	// json.Number literals, so the test exercises the actual route into this state.
	dec := json.NewDecoder(strings.NewReader(
		`{"num_hidden_layers": 52, ` + shapeFieldsForLayerEvidence + `}`))
	dec.UseNumber()
	var raw map[string]any
	if err := dec.Decode(&raw); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := raw["num_hidden_layers"].(json.Number); !ok {
		t.Fatalf("precondition: UseNumber must yield json.Number, got %T", raw["num_hidden_layers"])
	}

	hf := &latency.HFConfig{Raw: raw}
	if got := hf.ResolveNumLayers(); got != 0 {
		t.Fatalf("precondition: a json.Number scalar does not answer, want 0, got %d", got)
	}

	_, err := latency.GetModelConfigFromHF(hf)
	if err == nil {
		t.Fatal("a Raw map violating the float64 contract must be refused, not silently parsed")
	}
	// The clause must name the offending key, the type it holds, and the decoder setting
	// that produced it — everything needed to fix this without reading BLIS's source.
	for _, want := range []string{`"num_hidden_layers"`, "json.Number", "UseNumber"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("diagnostic must mention %q\ngot: %v", want, err)
		}
	}
	// The regression this pins: reporting a JSON number as "not a number".
	if strings.Contains(err.Error(), "not a number") {
		t.Errorf("a json.Number is a JSON number; the diagnostic must not call it "+
			"\"not a number\" and send the caller editing a correct config value\ngot: %v", err)
	}
}

// TestJSONValueKind_NamesAJSONNumberANumber is the companion unit claim: jsonValueKind
// reports json.Number as "a number", so the OTHER key's clause — where a number genuinely
// is the wrong type — reads "holds a number, not a list" rather than leaking the Go type
// name at an operator. Asserted through the layers_block_type clause because jsonValueKind
// is unexported; that clause is its only other caller.
func TestJSONValueKind_NamesAJSONNumberANumber(t *testing.T) {
	dec := json.NewDecoder(strings.NewReader(
		`{"layers_block_type": 52, ` + shapeFieldsForLayerEvidence + `}`))
	dec.UseNumber()
	var raw map[string]any
	if err := dec.Decode(&raw); err != nil {
		t.Fatalf("decode: %v", err)
	}

	_, err := latency.GetModelConfigFromHF(&latency.HFConfig{Raw: raw})
	if err == nil {
		t.Fatal("a scalar-less config whose block-type key is a number must be refused")
	}
	want := `"layers_block_type" holds a number, not a list`
	if !strings.Contains(err.Error(), want) {
		t.Errorf("diagnostic must contain %s\ngot: %v", want, err)
	}
}

// TestGetModelConfig_NegativeScalarStillDefersToTheBackendValidators is the scope
// boundary of the refusal above: a NEGATIVE num_hidden_layers ANSWERS the layer-count
// question with bad input rather than leaving it unanswered, so the parse passes the
// value through (ResolveNumLayers' documented decision) and the backends report the
// actual value. "Cannot determine the layer count" would be plainly untrue there — the
// config determined it, at -4.
func TestGetModelConfig_NegativeScalarStillDefersToTheBackendValidators(t *testing.T) {
	body := `{"num_hidden_layers": -4, ` + shapeFieldsForLayerEvidence + `}`
	mc := parseConfigJSON(t, body)
	if mc.NumLayers != -4 {
		t.Fatalf("NumLayers = %d, want -4 (a negative scalar is passed through, not refused at parse)", mc.NumLayers)
	}

	err := latency.ValidateRooflineConfig(*mc, blockTypeHW())
	if err == nil || !strings.Contains(err.Error(), "NumLayers must be > 0") {
		t.Errorf("roofline backend must reject a negative layer count, got err=%v", err)
	}
	hw := sim.NewModelHardwareConfig(*mc, blockTypeHW(), "nvidia/nemotron-style", "H100",
		1, 1, false, "", sim.LatencyBackendTrainedPhysics, 0)
	if _, err := latency.NewTrainedPhysicsModel(blockTypeCoeffs(), hw); err == nil ||
		!strings.Contains(err.Error(), "NumLayers must be > 0") {
		t.Errorf("trained-physics backend must reject a negative layer count, got err=%v", err)
	}
}

// TestBothBackendsRejectZeroLayerModelConfig is the surviving half of the original
// BC-3 / AC-4 contract: both backends keep their own zero-layer check as defense in
// depth. The parse boundary is now the loud one (above), but a ModelConfig can reach a
// backend by other routes — the Go API, a test, a future config source — so neither
// validator may be relaxed on the strength of the parse-time refusal.
//
// Asserted against a DIRECTLY-CONSTRUCTED ModelConfig rather than a parsed one, because
// the parser can no longer produce a zero-layer config: routing this through GetModelConfig
// is exactly what would make the check untestable the day the parse guard changed.
func TestBothBackendsRejectZeroLayerModelConfig(t *testing.T) {
	mc := sim.ModelConfig{
		NumLayers:     0,
		HiddenDim:     4096,
		NumHeads:      32,
		NumKVHeads:    8,
		VocabSize:     32000,
		BytesPerParam: 2,
	}

	err := latency.ValidateRooflineConfig(mc, blockTypeHW())
	if err == nil || !strings.Contains(err.Error(), "NumLayers must be > 0") {
		t.Errorf("roofline backend must still reject a zero layer count, got err=%v", err)
	}

	hw := sim.NewModelHardwareConfig(mc, blockTypeHW(), "nvidia/nemotron-style", "H100",
		1, 1, false, "", sim.LatencyBackendTrainedPhysics, 0)
	if _, err := latency.NewTrainedPhysicsModel(blockTypeCoeffs(), hw); err == nil ||
		!strings.Contains(err.Error(), "NumLayers must be > 0") {
		t.Errorf("trained-physics backend must still reject a zero layer count, got err=%v", err)
	}
}
