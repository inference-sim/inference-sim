// Original to blis-vllm.

package quantization

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

const (
	mxfp8Config = "a3698fff231a6cd92f632dcbe50f0b1dbda716994f29ccde9f0fa9d2a87698bf" // minimax-m3-mxfp8
	mixedConfig = "440d22038ffa0bcb93fa01e8d2d555e3840d939167c08eace81b9cce386fb117" // minimax-m3-nvfp4
)

// recorded answers with the layout of one checkpoint rewritten, as a recipe that assigned its layers
// differently would have been recorded by the oracle.
func withClasses(t *testing.T, sha string, edit func(classes map[string]map[string]int)) map[string]checkpoint {
	t.Helper()
	by, err := load()
	if err != nil {
		t.Fatal(err)
	}
	cp := by[sha]
	var l layout
	raw, _ := json.Marshal(cp.Layout)
	if err := json.Unmarshal(raw, &l); err != nil {
		t.Fatal(err)
	}
	edit(l.Classes)
	cp.Layout = &l
	by[sha] = cp
	return by
}

// The per-layer mapping decides the layout: the same lowering over a recipe whose routed experts are
// MXFP8 states fp8 experts, over the recorded one (NVFP4 experts) states nvfp4. The dense layers of
// both are MXFP8 with one scale per 32 weights, and the head, which both recipes exclude, is in the
// model dtype.
func TestTheLayoutFollowsTheRecipesPerLayerMapping(t *testing.T) {
	by, err := load()
	if err != nil {
		t.Fatal(err)
	}
	mixed, err := lower(by, mixedConfig, "modelopt_mixed")
	if err != nil || mixed == nil {
		t.Fatalf("modelopt_mixed on the mixed-precision checkpoint: %v, %v", mixed, err)
	}
	if got := mixed.Served; got.Linear != "fp8" || got.LinearScaleElements != 32 || got.RoutedExperts != "nvfp4" ||
		got.Head != "bf16" || got.Embedding != "bf16" || got.Router != "fp32" {
		t.Errorf("mixed-precision layout %+v", got)
	}
	plain, err := lower(by, mxfp8Config, "mxfp8")
	if err != nil || plain == nil {
		t.Fatalf("mxfp8 on the MXFP8 checkpoint: %v, %v", plain, err)
	}
	if plain.Served.RoutedExperts != "fp8" || plain.Served.Linear != "fp8" || plain.Served.LinearScaleElements != 32 {
		t.Errorf("MXFP8 layout %+v", plain.Served)
	}

	flipped := withClasses(t, mixedConfig, func(c map[string]map[string]int) {
		c["routed_experts"] = map[string]int{"MXFP8": 57}
	})
	got, err := lower(flipped, mixedConfig, "modelopt_mixed")
	if err != nil {
		t.Fatal(err)
	}
	if got.Served.RoutedExperts != "fp8" {
		t.Errorf("a recipe with MXFP8 routed experts states %q for them", got.Served.RoutedExperts)
	}
}

// The kernel prices one format per class of layers, so a recipe that splits a class between two
// formats is refused by naming the class, not priced at one of them.
func TestARecipeThatSplitsAClassIsRefusedByName(t *testing.T) {
	for _, class := range []string{"linear", "routed_experts", "head"} {
		by := withClasses(t, mixedConfig, func(c map[string]map[string]int) {
			c[class] = map[string]int{"MXFP8": 50, "NVFP4": 7}
		})
		_, err := lower(by, mixedConfig, "modelopt_mixed")
		var r *Refusal
		if !errors.As(err, &r) || !strings.Contains(r.Message, class) || !strings.Contains(r.Message, "MXFP8, NVFP4") {
			t.Errorf("%s split between two formats: %v", class, err)
		}
	}
}

// A format with no weight width the kernel prices is refused by name.
func TestAnAlgorithmWithoutAPricedWidthIsRefused(t *testing.T) {
	by := withClasses(t, mixedConfig, func(c map[string]map[string]int) {
		c["linear"] = map[string]int{"GGUF_Q4": 240}
	})
	_, err := lower(by, mixedConfig, "")
	var r *Refusal
	if !errors.As(err, &r) || !strings.Contains(r.Message, "GGUF_Q4") {
		t.Errorf("an unpriced algorithm: %v", err)
	}
}

// A stated value vLLM raises on is refused with vLLM's message, the stated value in it; a value the
// oracle did not list is answered as vLLM answers a string it does not name.
func TestAValueVLLMRefusesIsRefusedWithItsMessage(t *testing.T) {
	by, _ := load()
	for _, stated := range []string{"fp8", "awq", "modelopt_mixed", "a_value_nobody_listed"} {
		_, err := lower(by, mxfp8Config, stated)
		var r *Refusal
		if !errors.As(err, &r) {
			t.Fatalf("%q on the MXFP8 checkpoint: %v", stated, err)
		}
		want := "Quantization method specified in the model config (mxfp8) does not match the quantization method " +
			"specified in the `quantization` argument (" + stated + ")."
		if !strings.Contains(r.Message, want) {
			t.Errorf("%q: refusal %q lacks vLLM's message %q", stated, r.Message, want)
		}
	}
}

// vLLM serves a mixed-precision ModelOpt checkpoint with its own method whatever is stated, and the
// replaced value is reported; humming is the one value it honours, and it has no layout here.
func TestAClaimedRecipeReportsTheReplacedValue(t *testing.T) {
	by, _ := load()
	for _, tc := range []struct {
		stated, requested string
	}{{"", ""}, {"modelopt_mixed", ""}, {"modelopt_fp4", "modelopt_fp4"}, {"fp8", "fp8"}} {
		got, err := lower(by, mixedConfig, tc.stated)
		if err != nil || got == nil {
			t.Fatalf("%q: %v, %v", tc.stated, got, err)
		}
		if got.Method != "modelopt_mixed" || got.Requested != tc.requested {
			t.Errorf("%q: method %q, requested %q; want modelopt_mixed, %q", tc.stated, got.Method, got.Requested, tc.requested)
		}
	}
	_, err := lower(by, mixedConfig, "humming")
	var r *Refusal
	if !errors.As(err, &r) || !strings.Contains(r.Message, `"humming"`) {
		t.Errorf("humming: %v", err)
	}
}

// A checkpoint the oracle did not record a layout for is not lowered: nil, nil, and the caller keeps
// the stated value.
func TestAnUncoveredCheckpointIsLeftAlone(t *testing.T) {
	by, _ := load()
	var doc table
	if err := json.Unmarshal(recorded, &doc); err != nil {
		t.Fatal(err)
	}
	v4 := doc.Checkpoints["deepseek-v4-pro"].ConfigSHA256 // recorded for its test, lowered by the kernel
	if v4 == "" {
		t.Fatal("the recorded answers have no deepseek-v4-pro entry")
	}
	for _, sha := range []string{"", "0000", v4} {
		got, err := lower(by, sha, "mxfp8")
		if got != nil || err != nil {
			t.Errorf("config %q: %v, %v", sha, got, err)
		}
	}
}

func TestTheRecordedAnswersAreForThePinnedRelease(t *testing.T) {
	var t0 table
	if err := json.Unmarshal(recorded, &t0); err != nil {
		t.Fatal(err)
	}
	if t0.VLLM != Upstream {
		t.Errorf("resolved.json is for %s, the package for %s", t0.VLLM, Upstream)
	}
	if _, err := parse([]byte(strings.Replace(string(recorded), Upstream, strings.Repeat("0", 40), 1))); err == nil {
		t.Error("answers recorded at another vLLM commit were accepted")
	}
}
