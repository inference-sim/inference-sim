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
// text-only model. (All fourteen multimodal checkpoints of the catalog are in the same class; the
// table test below covers every one, and these fixtures are the ones with a committed deployment.)
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
// model already prices the text decoder alone. A text-only model is the control. This shows the
// engine ignores the key; it does not compare the weights with vLLM's under true (see
// TestTowerlessMultimodalConfigsNameTheCatalogModels for that limit).
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

// Every catalog model whose config.json has a vision_config is in exactly one of
// towerlessMultimodalConfigs and pricedTowerConfigs, so a multimodal checkpoint cannot be left
// without a decision. Each digest is the one the catalog's graph was derived from. A listed
// model's graph holds no node for a vision tower, projector or patch merge (the name pattern
// below), which is what makes true the identity TestLanguageModelOnlyTrueChangesNoPrice shows; a
// model in pricedTowerConfigs must match the pattern.
//
// The limit: this is a reading of the catalog graph's node names. It is not a comparison with the
// weights vLLM keeps resident under language_model_only true, which would need vLLM's model
// classes built on a meta device, and they do not build without their GPU-side parallel groups
// on the CPU build the oracle runs on. The catalog holds no checkpoint tensor index to count
// against either. So the test shows the graph names no tower, not that its weight bytes equal
// vLLM's.
func TestTowerlessMultimodalConfigsNameTheCatalogModels(t *testing.T) {
	dir, err := substratedata.Catalog()
	if err != nil {
		t.Fatal(err)
	}
	data := repos().Data
	tower := regexp.MustCompile(`(?i)vision|tower|projector|merger|patch_merge|vit`)
	entries, err := os.ReadDir(filepath.Join(dir, "models"))
	if err != nil {
		t.Fatal(err)
	}
	multimodal := map[string]bool{}
	for _, e := range entries {
		cfg, err := os.ReadFile(filepath.Join(dir, "models", e.Name(), "config.json"))
		if err != nil {
			continue // a catalog entry without a vendor config (a drafter, say) has no vision_config to find
		}
		if strings.Contains(string(cfg), `"vision_config"`) {
			multimodal[e.Name()] = true
		}
	}
	if len(multimodal) < 15 {
		t.Fatalf("%d catalog models have a vision_config, the pinned catalog has 15", len(multimodal))
	}
	listed := map[string]bool{}
	for _, table := range []struct {
		lists map[string]string
		tower bool
	}{{towerlessMultimodalConfigs, false}, {pricedTowerConfigs, true}} {
		for sha, name := range table.lists {
			if listed[name] {
				t.Errorf("%s is in both lists", name)
			}
			listed[name] = true
			g, err := data.Model(name)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if g.DerivedFrom.SHA256 != sha {
				t.Errorf("%s: the catalog graph was derived from %s, the list has %s", name, g.DerivedFrom.SHA256, sha)
			}
			if !multimodal[name] {
				t.Errorf("%s: config.json has no vision_config, so it is not a multimodal checkpoint", name)
			}
			graph, err := os.ReadFile(filepath.Join(dir, "models", name, "graph.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			m := tower.Find(graph)
			if table.tower && m == nil {
				t.Errorf("%s: listed as pricing a tower, and the graph names none", name)
			}
			if !table.tower && m != nil {
				t.Errorf("%s: the graph names %q, so it may price a tower and true is not the identity", name, m)
			}
			// A listed model is disclosed when the key is unset or false, and not when it is true.
			got := languageModelOnlyAssumptions(g, engineconfig.Engine{})
			if !table.tower && (len(got) != 1 || !strings.Contains(got[0], name)) {
				t.Errorf("%s: unset gives assumptions %q, want one naming the model", name, got)
			}
			if table.tower && got != nil {
				t.Errorf("%s: a model that prices a tower was given an assumption %q", name, got)
			}
			f := false
			if got := languageModelOnlyAssumptions(g, engineconfig.Engine{LanguageModelOnly: &f}); !table.tower && (len(got) != 1 || !strings.Contains(got[0], "= false")) {
				t.Errorf("%s: false gives assumptions %q, want one stating false", name, got)
			}
			if got := languageModelOnlyAssumptions(g, engineconfigStating(true)); got != nil {
				t.Errorf("%s: true gives assumptions %q, want none", name, got)
			}
		}
	}
	for name := range multimodal {
		if !listed[name] {
			t.Errorf("%s has a vision_config and is in neither towerlessMultimodalConfigs nor pricedTowerConfigs", name)
		}
	}
	for _, want := range []string{"minimax-m3", "minimax-m3-mxfp8", "minimax-m3-nvfp4", "kimi-k3", "deepseek-v4.1-flash", "glm-5.3-flash", "inkling",
		"kimi-k2.5", "kimi-k2.6", "llama-4-scout-17b-16e-instruct-fp8-dynamic", "muse-spark-open-816b-a42b",
		"muse-spark-open-816b-a42b-nvfp4", "qwen3.5-397b-a17b", "qwen3.5-397b-a17b-fp8", "qwen3.6-35b-a3b"} {
		if !listed[want] {
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
