package latency_test

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim/latency"
)

// TestCatalogConfigEquivalence is the R1 migration gate for blis-catalog PR #1
// (design decision recorded there; tracked by issue #1748).
//
// The catalog ships fresh, verbatim-from-vendor config.json files. For three
// models the upstream config is *fuller* than the older trimmed model_configs/
// fixture it replaces (extra tokenizer/runtime/quantization keys the vendor
// ships). R1 acceptance test #3 requires byte-identical stdout after the
// migration, which holds only if BLIS parses an equivalent model shape from the
// fuller config. This test proves that: for each model, the catalog config and
// the retained fixture must parse to an identical *sim.ModelConfig through the
// real loader (latency.GetModelConfig).
//
// This is intentionally a *migration* gate. It compares the catalog config
// against the model_configs/ fixture that S6 will later remove; once S6 lands
// and the fixtures are gone, this test is replaced by a golden *ModelConfig
// keyed directly off the catalog config (which needs no fixture). Until then it
// is the automated guard that the fresh-fetch migration is behaviour-preserving.
//
// The one read key that differs between fixture and upstream — llama-3.1's
// rope_scaling (type "llama3") — is NOT read by GetModelConfig; it is consumed
// only by cmd.applyRopeScaling, where "llama3" is blacklisted (a no-op) and is
// already covered permanently by cmd/root_test.go:TestApplyRopeScaling. So
// ModelConfig equality is the complete equivalence claim at this layer.
func TestCatalogConfigEquivalence(t *testing.T) {
	// name -> retained model_configs/ fixture (relative to repo root; this test
	// runs from sim/latency, hence ../../).
	fixtures := map[string]string{
		"glm-5.2-fp8":           "../../model_configs/glm-5.2-fp8/config.json",
		"llama-2-7b-hf":         "../../model_configs/llama-2-7b-hf/config.json",
		"llama-3.1-8b-instruct": "../../model_configs/llama-3.1-8b-instruct/config.json",
	}

	for name, fixturePath := range fixtures {
		t.Run(name, func(t *testing.T) {
			catalogPath := filepath.Join("testdata", "catalog_configs", name+".json")

			catalogMC, err := latency.GetModelConfig(catalogPath)
			if err != nil {
				t.Fatalf("GetModelConfig(catalog %s): %v", catalogPath, err)
			}
			fixtureMC, err := latency.GetModelConfig(fixturePath)
			if err != nil {
				t.Fatalf("GetModelConfig(fixture %s): %v", fixturePath, err)
			}

			if !reflect.DeepEqual(catalogMC, fixtureMC) {
				t.Errorf("parsed ModelConfig differs between catalog and fixture for %s\n"+
					"the fuller upstream config must parse to the same shape BLIS reads\n"+
					"catalog: %+v\nfixture: %+v", name, catalogMC, fixtureMC)
			}
		})
	}
}
