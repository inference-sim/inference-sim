// Original to blis-vllm.

package kernelmodel

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"

	"github.com/inference-sim/blis-vllm/compat/quantization"
	"github.com/inference-sim/blis-vllm/internal/substratedata"
)

// m3Oracle is a MiniMax-M3 entry of compat/quantization/resolved.json: what vLLM's own ModelConfig
// (vllm/config/model.py:1370-1449@5df63a72) resolves `quantization` to on the catalog checkpoint, for
// every method name vLLM lists, or the text it refuses with.
type m3Oracle struct {
	ConfigSHA256 string `json:"config_sha256"`
	OwnMethod    string `json:"resolved_without_quantization"`
	Cases        []struct {
		Stated   *string `json:"stated"`
		Resolved *string `json:"resolved"`
		Error    *string `json:"error"`
	} `json:"cases"`
}

func loadM3Oracle(t *testing.T, model string) m3Oracle {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "quantization", "resolved.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Checkpoints map[string]m3Oracle `json:"checkpoints"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	o, ok := doc.Checkpoints[model]
	if !ok || len(o.Cases) < 20 {
		t.Fatalf("the oracle file has no usable entry for %s", model)
	}
	return o
}

// openM3 opens a MiniMax-M3 checkpoint on a chip with `quantization` set as stated ("" leaves it out),
// the way the dashboard's points state it (tp 8, language_model_only).
func openM3(t *testing.T, model, chip, stated string) (*Model, error) {
	t.Helper()
	y := "kind: Scenario\nname: q\nengine_version: \"0.32.0\"\nmodel: " + model + "\n" +
		"coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory]\n" +
		"cluster:\n  hardware: " + chip + "\n  nodes: 1\n  gpus_per_node: 8\n---\nkind: Deployment\nname: q\npools:\n" +
		"  - role: colocated\n    nodes: 1\n    parallel: {tp: 8, pp: 1, dp: 1}\n    engine:\n" +
		"      block_size: 128\n      max_num_batched_tokens: 2048\n      max_num_seqs: 64\n      max_model_len: 2304\n" +
		"      language_model_only: true\n      enable_prefix_caching: false\n"
	if stated != "" {
		y += "      quantization: " + stated + "\n"
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "q.yaml"), []byte(y), 0o644); err != nil {
		t.Fatal(err)
	}
	return Open("q.yaml", r)
}

func mustOpenM3(t *testing.T, model, chip, stated string) *Model {
	t.Helper()
	m, err := openM3(t, model, chip, stated)
	if err != nil {
		t.Fatalf("%s on %s stating quantization %q: %v", model, chip, stated, err)
	}
	return m
}

// The value each MiniMax-M3 checkpoint's points state is the method vLLM resolves its recipe to
// (mxfp8 for the MXFP8 checkpoint, modelopt_mixed for the mixed-precision one, whose recipe a
// ModelOpt class claims whatever is stated). vLLM serves the checkpoint's own formats, so the plan
// for the stated value is the plan for none stated, on every chip the points ran on.
func TestM3StatedQuantizationGivesTheUnstatedPlan(t *testing.T) {
	for _, c := range []struct{ model, stated string }{
		{"minimax-m3-mxfp8", "mxfp8"},
		{"minimax-m3-nvfp4", "modelopt_mixed"},
	} {
		o := loadM3Oracle(t, c.model)
		if o.OwnMethod != c.stated {
			t.Fatalf("vLLM resolves %s to %q; this test states %q", c.model, o.OwnMethod, c.stated)
		}
		for _, chip := range []string{"h100", "h200", "b200", "b300"} {
			t.Run(c.model+"/"+chip, func(t *testing.T) {
				want := planOf(mustOpenM3(t, c.model, chip, ""))
				if got := planOf(mustOpenM3(t, c.model, chip, c.stated)); got != want {
					t.Errorf("plan with %s stated differs from the unstated one:\n got %s\nwant %s", c.stated, got, want)
				}
			})
		}
	}
}

// Every value vLLM lists, on both checkpoints: where vLLM resolves it to the checkpoint's own method the
// kernel prices the unstated plan (and records the replacement when the value differed); where vLLM
// raises, the open is refused with vLLM's own text; where vLLM resolves to another method (humming) the
// open is refused by name.
func TestM3EveryValueVLLMListsIsPricedOrRefusedAsVLLMDoes(t *testing.T) {
	for _, model := range []string{"minimax-m3-mxfp8", "minimax-m3-nvfp4"} {
		o := loadM3Oracle(t, model)
		want := planOf(mustOpenM3(t, model, "h200", ""))
		var priced, raised, other int
		for _, c := range o.Cases {
			stated := ""
			if c.Stated != nil {
				stated = *c.Stated
			}
			if stated == "" || stated == "not_a_quantization_method" {
				continue
			}
			m, err := openM3(t, model, "h200", stated)
			switch {
			case c.Error != nil:
				raised++
				if err == nil || !strings.Contains(err.Error(), *c.Error) {
					t.Errorf("%s stating %q: vLLM raises %q; got %v", model, stated, *c.Error, err)
				}
			case *c.Resolved != o.OwnMethod:
				other++
				if err == nil || !strings.Contains(err.Error(), "resolves to \""+*c.Resolved+"\"") {
					t.Errorf("%s stating %q: vLLM serves %q, which has no layout here; got %v", model, stated, *c.Resolved, err)
				}
			default:
				priced++
				if err != nil {
					t.Errorf("%s stating %q: vLLM serves the checkpoint's own %q; got %v", model, stated, *c.Resolved, err)
					continue
				}
				if got := planOf(m); got != want {
					t.Errorf("%s stating %q: plan differs from the unstated one", model, stated)
				}
				subs := m.Provenance().Substitutions
				if stated == o.OwnMethod && len(subs) != 0 {
					t.Errorf("%s stating its own method %q recorded %+v", model, stated, subs)
				}
				if stated != o.OwnMethod && (len(subs) != 1 || subs[0].Setting != "quantization" || subs[0].Stated != stated || subs[0].Priced != o.OwnMethod) {
					t.Errorf("%s stating %q: substitutions %+v, want one from %q to %q", model, stated, subs, stated, o.OwnMethod)
				}
			}
		}
		// The test can fail both ways: a checkpoint whose recorded answers are all priced, or all
		// refused, proves nothing.
		if priced == 0 || raised+other == 0 {
			t.Errorf("%s: %d values priced, %d raised, %d served by another method", model, priced, raised, other)
		}
	}
}

// The recipe's per-layer mapping is what the engine states: the MXFP8 checkpoint serves its routed
// experts in fp8, the mixed-precision one in nvfp4 (its quantized_layers map the experts to NVFP4 and
// the dense layers to MXFP8), both with one scale per 32 weights for the dense layers, an unquantized
// head, and a float32 router; and the stated value is cleared, because the flag names no weight width.
func TestM3LoweringStatesThePerLayerLayout(t *testing.T) {
	for _, c := range []struct {
		model, stated string
		want          enginesettings.ServedWeights
	}{
		{"minimax-m3-mxfp8", "mxfp8", enginesettings.ServedWeights{Linear: "fp8", LinearScaleElements: 32,
			RoutedExperts: "fp8", Router: "fp32", Head: "bf16", Embedding: "bf16"}},
		{"minimax-m3-nvfp4", "modelopt_mixed", enginesettings.ServedWeights{Linear: "fp8", LinearScaleElements: 32,
			RoutedExperts: "nvfp4", Router: "fp32", Head: "bf16", Embedding: "bf16"}},
	} {
		g := mustOpenM3(t, c.model, "h200", c.stated).graph
		s := enginesettings.Settings{Quantization: c.stated}
		if _, err := lowerQuantization(g, c.stated, &s); err != nil {
			t.Fatalf("%s: %v", c.model, err)
		}
		if s.Quantization != "" {
			t.Errorf("%s: the flag %q was kept", c.model, s.Quantization)
		}
		if s.ServedWeights == nil || *s.ServedWeights != c.want {
			t.Errorf("%s: served layout %+v, want %+v", c.model, s.ServedWeights, c.want)
		}
	}
}

// The two checkpoints differ only in the recipe's mapping, and they price differently: the lowering
// reaches the kernel's weight bytes (the nvfp4 experts are a quarter of the fp8 ones).
func TestM3CheckpointsWithDifferentMappingsPriceTheirOwnWeights(t *testing.T) {
	a := mustOpenM3(t, "minimax-m3-mxfp8", "h200", "mxfp8").Kernel().FixedBytes().Weights
	b := mustOpenM3(t, "minimax-m3-nvfp4", "h200", "modelopt_mixed").Kernel().FixedBytes().Weights
	if a <= b {
		t.Errorf("fp8 experts weigh %d per rank, nvfp4 experts %d", a, b)
	}
}

// A value the recipe's class does not cover still prices a checkpoint the oracle has no recipe for as
// before: the bfloat16 minimax-m3 states mxfp8 as online quantization, which the kernel does not price.
func TestAnUncoveredCheckpointKeepsItsStatedQuantization(t *testing.T) {
	_, err := openM3(t, "minimax-m3", "h200", "mxfp8")
	if err == nil || !strings.Contains(err.Error(), "is not one this kernel prices") {
		t.Errorf("minimax-m3 (bfloat16) stating mxfp8: %v", err)
	}
}

// The digests the package keys on are those of the pinned catalog's config.json files, and every catalog
// checkpoint whose recipe this package would have to lower is either lowered or named here with the reason
// it is not.
func TestTheRecordedCheckpointsAreTheCatalogsAndNoneIsLeftOut(t *testing.T) {
	dir, err := substratedata.Catalog()
	if err != nil {
		t.Fatal(err)
	}
	covered, err := quantization.Checkpoints()
	if err != nil {
		t.Fatal(err)
	}
	if len(covered) != 2 {
		t.Fatalf("lowered checkpoints %v", covered)
	}
	for name, sha := range covered {
		raw, err := os.ReadFile(filepath.Join(dir, "models", name, "config.json"))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(raw)
		if hex.EncodeToString(sum[:]) != sha {
			t.Errorf("%s: config.json has digest %x, the recorded answers are for %s", name, sum, sha)
		}
		g, err := repos().Data.Model(name)
		if err != nil {
			t.Fatal(err)
		}
		if g.DerivedFrom.SHA256 != sha {
			t.Errorf("%s: the graph was derived from %s, the recorded answers are for %s", name, g.DerivedFrom.SHA256, sha)
		}
	}
	// ModelOpt mixed-precision and MXFP8 recipes the lowering reads by MiniMax-M3's module names only.
	notLowered := map[string]string{
		"nemotron-3-ultra-550b-a55b-nvfp4":     "a mixed-precision recipe over Nemotron's module names, which no oracle layout covers",
		"nemotron-3.5-lightning-30b-a3b-nvfp4": "a mixed-precision recipe over Nemotron's module names, which no oracle layout covers",
	}
	entries, err := os.ReadDir(filepath.Join(dir, "models"))
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		raw, err := os.ReadFile(filepath.Join(dir, "models", e.Name(), "config.json"))
		if err != nil {
			continue
		}
		var cfg struct {
			QuantRaw  map[string]any `json:"quantization_config"`
			TextQuant struct {
				Q map[string]any `json:"quantization_config"`
			} `json:"text_config"`
		}
		if err := json.Unmarshal(raw, &cfg); err != nil {
			t.Fatalf("%s: %v", e.Name(), err)
		}
		q := cfg.QuantRaw
		if q == nil {
			q = cfg.TextQuant.Q
		}
		method, _ := q["quant_method"].(string)
		algo, _ := q["quant_algo"].(string)
		if method != "mxfp8" && (method != "modelopt" || algo != "MIXED_PRECISION") {
			continue
		}
		if _, ok := covered[e.Name()]; !ok && notLowered[e.Name()] == "" {
			t.Errorf("%s states a %s %s recipe that is neither lowered nor listed as not lowered", e.Name(), method, algo)
		}
	}
}
