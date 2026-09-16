package latency_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// TestBlockTypeLayerCount covers the #1729 (NS-4) block-type-array reader directly,
// including the robustness cases the end-to-end parse tests do not reach: an absent
// key, an empty list, and a non-list value all yield 0 so the caller keeps the
// existing loud NumLayers failure (AC-4) rather than a silent 0.
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
// resolver returns 0 so the backend validators still fail loudly.
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

// TestBothBackendsRejectNoLayerEvidence is BC-3 / AC-4: a config with NEITHER
// num_hidden_layers NOR a usable layers_block_type still fails loudly in both
// backends — the fallback must not convert a missing layer count into a silent 0.
func TestBothBackendsRejectNoLayerEvidence(t *testing.T) {
	body := `{
		"layers_block_type": [],
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"intermediate_size": 14336,
		"vocab_size": 32000,
		"torch_dtype": "bfloat16"
	}`
	mc := parseConfigJSON(t, body)
	if mc.NumLayers != 0 {
		t.Fatalf("precondition: NumLayers = %d, want 0 (no usable evidence)", mc.NumLayers)
	}

	err := latency.ValidateRooflineConfig(*mc, blockTypeHW())
	if err == nil || !strings.Contains(err.Error(), "NumLayers must be > 0") {
		t.Errorf("roofline backend must still reject a missing layer count, got err=%v", err)
	}

	hw := sim.NewModelHardwareConfig(*mc, blockTypeHW(), "nvidia/nemotron-style", "H100",
		1, 1, false, "", sim.LatencyBackendTrainedPhysics, 0)
	_, err = latency.NewTrainedPhysicsModel(blockTypeCoeffs(), hw)
	if err == nil || !strings.Contains(err.Error(), "NumLayers must be > 0") {
		t.Errorf("trained-physics backend must still reject a missing layer count, got err=%v", err)
	}
}
