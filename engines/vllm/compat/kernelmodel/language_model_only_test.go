package kernelmodel

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/internal/substratedata"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// languageModelOnlyFixtures are one committed 0.32.0 deployment of each model class the rule
// distinguishes: two multimodal checkpoints whose catalog graph holds the text decoder only, and a
// text-only model.
var languageModelOnlyFixtures = []struct {
	name, path string
	towerless  bool
	// chip, when set, replaces the fixture's hardware, for a fixture whose weights do not fit its own.
	chip string
}{
	{"minimax-m3", filepath.Join("kerneltests", "testdata", "aisimulate", "minimax-m3-b200-fp4-vllm-tp2.yaml"), true, ""},
	{"kimi-k3", filepath.Join("kerneltests", "testdata", "kimi-k3-h100-nospec.yaml"), true, "b300"},
	{"llama-3.1-70b-instruct", filepath.Join("kerneltests", "testdata", "aisimulate", "llama-3.1-70b-instruct-h200-fp8-vllm-tp2.yaml"), false, ""},
}

// openWithLanguageModelOnly opens a fixture with engine.language_model_only set to stated
// ("", true or false) in its first pool's engine block.
func openWithLanguageModelOnly(t *testing.T, path, chip, stated string) *Model {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", path))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if chip != "" {
		re := regexp.MustCompile(`(?m)^(\s*hardware:\s*)\S+`)
		if !re.MatchString(body) {
			t.Fatalf("%s states no hardware", path)
		}
		body = re.ReplaceAllString(body, "${1}"+chip)
	}
	if stated != "" {
		re := regexp.MustCompile(`(?m)^(\s*)block_size:`)
		if !re.MatchString(body) {
			t.Fatalf("%s states no block_size line to put the key before", path)
		}
		body = re.ReplaceAllString(body, "${1}language_model_only: "+stated+"\n${1}block_size:")
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open("v.yaml", r)
	if err != nil {
		t.Fatalf("%s with language_model_only %q: %v", path, stated, err)
	}
	return m
}

// engine.language_model_only over {a multimodal checkpoint whose graph holds no tower, a
// text-only model} x {unset, true, false}. Every value builds. True is what the catalog graph
// prices, so it discloses nothing. Unset and false are vLLM's default, which keeps the tower
// resident, and for the two multimodal checkpoints the engine discloses the gap; a text-only
// model builds no MultiModalConfig in vLLM, so nothing differs and nothing is disclosed.
func TestLanguageModelOnlyIsDisclosedWhereTheGraphPricesNoTower(t *testing.T) {
	for _, f := range languageModelOnlyFixtures {
		for _, stated := range []string{"", "true", "false"} {
			t.Run(f.name+"/"+map[string]string{"": "unset"}[stated]+stated, func(t *testing.T) {
				got := openWithLanguageModelOnly(t, f.path, f.chip, stated).EngineAssumptions()
				wantDisclosed := f.towerless && stated != "true"
				if (len(got) == 1) != wantDisclosed || len(got) > 1 {
					t.Fatalf("assumptions %q, want one: %t", got, wantDisclosed)
				}
				if wantDisclosed {
					for _, frag := range []string{"language_model_only", f.name, "vision tower", "language_model_only: true"} {
						if !strings.Contains(got[0], frag) {
							t.Errorf("assumption %q does not name %q", got[0], frag)
						}
					}
				}
			})
		}
	}
}

// For a multimodal checkpoint whose graph holds no tower, stating language_model_only: true
// prices nothing differently: the same per-rank fixed bytes, KV budget and blocks, KV layers and
// step times as the deployment without the key. This is the identity the rule rests on: the cost
// model already prices the text decoder alone. A text-only model is the control.
func TestLanguageModelOnlyTrueChangesNoPrice(t *testing.T) {
	shapes := [][]ReqShape{
		{{Scheduled: 512, Computed: 0, PromptLen: 512}},
		{{Scheduled: 1, Computed: 700, PromptLen: 700}, {Scheduled: 1, Computed: 90, PromptLen: 90}},
		{{Scheduled: 256, Computed: 256, PromptLen: 1024}, {Scheduled: 1, Computed: 3000, PromptLen: 3000}},
	}
	for _, f := range languageModelOnlyFixtures {
		t.Run(f.name, func(t *testing.T) {
			base, with := openWithLanguageModelOnly(t, f.path, f.chip, ""), openWithLanguageModelOnly(t, f.path, f.chip, "true")
			bb, err := base.KVBudget()
			if err != nil {
				t.Fatal(err)
			}
			wb, err := with.KVBudget()
			if err != nil {
				t.Fatal(err)
			}
			if bb.FixedBytes <= 0 || bb.PerRankBlocks <= 0 {
				t.Fatalf("the base deployment prices no weights or no blocks: %+v", bb)
			}
			if !reflect.DeepEqual(bb, wb) {
				t.Errorf("KV budget differs:\n unset %+v\n true  %+v", bb, wb)
			}
			bl, err1 := base.KVLayers()
			wl, err2 := with.KVLayers()
			// Kimi-K3's recurrent layers have no KV spec in this package, so both refuse alike.
			if fmt.Sprint(err1) != fmt.Sprint(err2) || !reflect.DeepEqual(bl, wl) {
				t.Errorf("KV layers differ: %v %v\n unset %+v\n true  %+v", err1, err2, bl, wl)
			}
			for i, s := range shapes {
				if b, w := base.StepTimeOfShapes(s), with.StepTimeOfShapes(s); b != w {
					t.Errorf("step %d: %d ticks unset, %d with language_model_only true", i, b, w)
				}
			}
		})
	}
}

// towerlessMultimodalConfigs names catalog models by the digest of their config.json. Each digest
// is the one the catalog's graph was derived from, that config.json has a vision_config, and the
// graph holds no node for a vision tower, projector or patch merge (which is what makes true the
// identity above). The listed names also cover the catalog id of each model the rule is about.
func TestTowerlessMultimodalConfigsNameTheCatalogModels(t *testing.T) {
	dir, err := substratedata.Catalog()
	if err != nil {
		t.Fatal(err)
	}
	data := repos().Data
	tower := regexp.MustCompile(`(?i)vision|tower|projector|merger|patch_merge|vit`)
	seen := map[string]bool{}
	for sha, name := range towerlessMultimodalConfigs {
		g, err := data.Model(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if g.DerivedFrom.SHA256 != sha {
			t.Errorf("%s: the catalog graph was derived from %s, the list has %s", name, g.DerivedFrom.SHA256, sha)
		}
		cfg, err := os.ReadFile(filepath.Join(dir, "models", name, "config.json"))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(cfg), `"vision_config"`) {
			t.Errorf("%s: config.json has no vision_config, so it is not a multimodal checkpoint", name)
		}
		graph, err := os.ReadFile(filepath.Join(dir, "models", name, "graph.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if m := tower.Find(graph); m != nil {
			t.Errorf("%s: the graph names %q, so it may price a tower and true is not the identity", name, m)
		}
		seen[name] = true
	}
	for _, want := range []string{"minimax-m3", "minimax-m3-mxfp8", "kimi-k3"} {
		if !seen[want] {
			t.Errorf("%s is not in towerlessMultimodalConfigs", want)
		}
	}
	if languageModelOnlyAssumptions(nil, engineconfigStating(true)) != nil {
		t.Error("a nil graph was given an assumption")
	}
}

func engineconfigStating(v bool) engineconfig.Engine {
	return engineconfig.Engine{LanguageModelOnly: &v}
}
