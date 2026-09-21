package latency_test

import (
	"path/filepath"
	"reflect"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// TestCatalogConfigEquivalence is the R1 migration gate for blis-catalog PR #1
// (design decision recorded there; tracked by issue #1748).
//
// The catalog ships fresh, verbatim-from-vendor config.json files. For three
// models the upstream config is *fuller* than the older trimmed model_configs/
// fixture it replaced (extra tokenizer/runtime/quantization keys the vendor
// ships). R1 acceptance test #3 requires byte-identical stdout after the
// migration, which holds only if BLIS parses an equivalent model shape from the
// fuller config. This test proves that: for each model, the catalog config must
// parse (through the real loader latency.GetModelConfig) to the golden
// *sim.ModelConfig below.
//
// #1771 removed the bundled model_configs/ tree, so — exactly as this test's
// predecessor prescribed ("once S6 lands and the fixtures are gone, this test is
// replaced by a golden *ModelConfig keyed directly off the catalog config") —
// the comparison is now against a hard-coded golden rather than against the
// deleted fixture. The golden values ARE what the retired model_configs/ fixture
// parsed to (the migration gate's original claim), captured before deletion; a
// change to either the catalog config or the loader that moves the parsed shape
// fails here.
//
// The one read key that differs between the (deleted) fixture and upstream —
// llama-3.1's rope_scaling (type "llama3") — is NOT read by GetModelConfig; it is
// consumed only by cmd.applyRopeScaling, where "llama3" is blacklisted (a no-op)
// and is already covered permanently by cmd/root_test.go:TestApplyRopeScaling. So
// ModelConfig equality is the complete equivalence claim at this layer.
func TestCatalogConfigEquivalence(t *testing.T) {
	// name -> golden *sim.ModelConfig the vendor-verbatim catalog config must parse to.
	// These are the values the retired model_configs/<name>/config.json fixtures parsed
	// to at the time of deletion (#1771).
	goldens := map[string]sim.ModelConfig{
		"glm-5.2-fp8": {
			NumLayers: 78, HiddenDim: 6144, NumHeads: 64, NumKVHeads: 64, VocabSize: 154880,
			BytesPerParam: 2, IntermediateDim: 12288, NumLocalExperts: 256, NumExpertsPerTok: 8,
			MoEExpertFFNDim: 2048, SharedExpertFFNDim: 2048, HiddenAct: "silu",
			WeightBytesPerParam: 1, HeadDim: 192, KVLoraRank: 512, QKRopeHeadDim: 64,
			FirstKDenseReplace: 3,
		},
		"llama-2-7b-hf": {
			NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 32, VocabSize: 32000,
			BytesPerParam: 2, IntermediateDim: 11008, HiddenAct: "silu",
		},
		"llama-3.1-8b-instruct": {
			NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 8, VocabSize: 128256,
			BytesPerParam: 2, IntermediateDim: 14336, HiddenAct: "silu",
		},
	}

	for name, want := range goldens {
		want := want
		t.Run(name, func(t *testing.T) {
			catalogPath := filepath.Join("testdata", "catalog_configs", name+".json")

			catalogMC, err := latency.GetModelConfig(catalogPath)
			if err != nil {
				t.Fatalf("GetModelConfig(catalog %s): %v", catalogPath, err)
			}

			if !reflect.DeepEqual(*catalogMC, want) {
				t.Errorf("parsed ModelConfig differs from the golden for %s\n"+
					"the vendor-verbatim catalog config must parse to the shape BLIS reads\n"+
					"catalog: %+v\ngolden:  %+v", name, *catalogMC, want)
			}
		})
	}
}
