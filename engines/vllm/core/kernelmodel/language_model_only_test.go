package kernelmodel

import (
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/engineconfig"
	"github.com/inference-sim/blis-vllm/internal/substratedata"
)

// languageModelOnlyFixtures are one committed 0.32.0 deployment of each model class the rule
// distinguishes: two multimodal checkpoints whose catalog graph holds the text decoder only, and a
// text-only model. (All fourteen multimodal checkpoints of the catalog are in the same class; the
// table test below covers every one, and these fixtures are the ones with a committed deployment.)
var languageModelOnlyFixtures = []struct {
	name, path string
	// towered says the catalog graph prices a vision tower, so the key decides whether it is resident.
	towered bool
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

// engine.language_model_only over {a multimodal checkpoint whose graph prices its tower, a text-only
// model} x {unset, true, false}. Every value builds, and none is a substitution: a priced tower is
// resident or not as the key says, and a text-only model builds no MultiModalConfig, so nothing differs.
// (A graph that prices no tower is still disclosed: TestPricedAndTowerlessConfigsFollowTheCatalogGraphs.)
func TestLanguageModelOnlyIsNotASubstitutionWhereTheGraphPricesItsTower(t *testing.T) {
	for _, f := range languageModelOnlyFixtures {
		for _, stated := range []string{"", "true", "false"} {
			t.Run(f.name+"/"+map[string]string{"": "unset"}[stated]+stated, func(t *testing.T) {
				m := openWithLanguageModelOnly(t, f.path, f.chip, stated)
				for _, a := range m.EngineAssumptions() {
					if strings.HasPrefix(a, "language_model_only") {
						t.Errorf("assumption %q: the graph prices its tower, so the key is a setting, not a gap", a)
					}
				}
				prov := m.Provenance()
				for _, sub := range prov.Substitutions {
					if sub.Setting == "language_model_only" {
						t.Errorf("substitution %+v for a graph that prices its tower", sub)
					}
				}
			})
		}
	}
}

// The key decides whether a priced tower is resident. Unset and false (vLLM's default) hold the tower:
// the same per-rank fixed bytes and KV budget. True holds none: fewer fixed bytes by at most the tower's
// stored bytes and at least their tensor-parallel share, and no fewer KV blocks. A text-only model is the
// control: every value prices alike. No value changes a text-only step: the tower runs only for images.
func TestLanguageModelOnlyTrueDropsThePricedTowerFromResidentWeights(t *testing.T) {
	shapes := [][]ReqShape{
		{{Scheduled: 512, Computed: 0, PromptLen: 512}},
		{{Scheduled: 1, Computed: 700, PromptLen: 700}, {Scheduled: 1, Computed: 90, PromptLen: 90}},
		{{Scheduled: 256, Computed: 256, PromptLen: 1024}, {Scheduled: 1, Computed: 3000, PromptLen: 3000}},
	}
	data := repos().Data
	for _, f := range languageModelOnlyFixtures {
		t.Run(f.name, func(t *testing.T) {
			unset, off, on := openWithLanguageModelOnly(t, f.path, f.chip, ""), openWithLanguageModelOnly(t, f.path, f.chip, "false"),
				openWithLanguageModelOnly(t, f.path, f.chip, "true")
			bu, err := unset.KVBudget()
			if err != nil {
				t.Fatal(err)
			}
			bf, err := off.KVBudget()
			if err != nil {
				t.Fatal(err)
			}
			bt, err := on.KVBudget()
			if err != nil {
				t.Fatal(err)
			}
			if bu.FixedBytes <= 0 || bu.PerRankBlocks <= 0 {
				t.Fatalf("the base deployment prices no weights or no blocks: %+v", bu)
			}
			if !reflect.DeepEqual(bu, bf) {
				t.Errorf("unset and false are vLLM's one default but price differently:\n unset %+v\n false %+v", bu, bf)
			}
			if !f.towered {
				if !reflect.DeepEqual(bu, bt) {
					t.Errorf("a text-only model's KV budget differs under true:\n unset %+v\n true  %+v", bu, bt)
				}
			} else {
				g, err := data.Model(f.name)
				if err != nil {
					t.Fatal(err)
				}
				tw, ok := g.ImageTower()
				if !ok {
					t.Fatalf("%s: the graph prices no image tower", f.name)
				}
				saved := bu.FixedBytes - bt.FixedBytes
				whole := int64(tw.WeightBytes())
				if saved <= 0 || saved > whole {
					t.Errorf("true saves %d bytes per rank; the tower stores %d", saved, whole)
				}
				if tp := int64(on.Kernel().Resolved().TensorParallel()); saved < whole/tp-tp {
					t.Errorf("true saves %d bytes per rank, under the tower's 1/%d share %d", saved, tp, whole/tp)
				}
				if bt.PerRankBlocks < bu.PerRankBlocks {
					t.Errorf("true leaves %d blocks, fewer than the %d with the tower resident", bt.PerRankBlocks, bu.PerRankBlocks)
				}
			}
			for i, s := range shapes {
				if b, w := unset.StepTimeOfShapes(s), on.StepTimeOfShapes(s); b != w {
					t.Errorf("step %d: %d ticks unset, %d with language_model_only true", i, b, w)
				}
			}
		})
	}
}

// The kernel prices a resident vision tower at pipeline-parallel width 1 only (the tower sits on the
// first stage). A Muse Spark deployment at PP2 x TP8 that leaves language_model_only unset therefore
// still builds, is priced as language_model_only true (the same KV budget and step times), and says so
// in its provenance, naming the setting as unset; stating true discloses nothing, because then no tower
// is built.
func TestPipelineParallelMuseSparkIsPricedWithoutItsTowerAndSaysSo(t *testing.T) {
	pp2 := filepath.Join("kerneltests", "testdata", "muse-h100-pp2tp8.yaml")
	unset, on := openWithLanguageModelOnly(t, pp2, "", ""), openWithLanguageModelOnly(t, pp2, "", "true")
	bu, err := unset.KVBudget()
	if err != nil {
		t.Fatal(err)
	}
	bt, err := on.KVBudget()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(bu, bt) {
		t.Errorf("at pp 2 the tower is not priced, so unset must price as true:\n unset %+v\n true  %+v", bu, bt)
	}
	var subs []string
	for _, sub := range unset.Provenance().Substitutions {
		if sub.Setting == "language_model_only" {
			if sub.Stated != "unset" || sub.Priced != "true" {
				t.Errorf("substitution %+v, want unset priced as true", sub)
			}
			subs = append(subs, sub.Reason)
		}
	}
	if len(subs) != 1 || !strings.Contains(subs[0], "2 pipeline stages") {
		t.Errorf("provenance substitutions %q, want one naming the 2 pipeline stages", subs)
	}
	for _, sub := range on.Provenance().Substitutions {
		if sub.Setting == "language_model_only" {
			t.Errorf("language_model_only: true was given a substitution %+v", sub)
		}
	}
}

// A catalog graph that prices at least one vision tower is in pricedTowerConfigs; one that lists its
// only tower as omitted is in towerlessMultimodalConfigs and is disclosed when the key is unset or
// false. Both lists are keyed by the digest of the config the graph was derived from, and every catalog
// graph with an encoder section is in exactly one of them, so a multimodal checkpoint cannot be left
// without a decision. A committed vendor config.json with a vision_config must have an encoder section.
func TestPricedAndTowerlessConfigsFollowTheCatalogGraphs(t *testing.T) {
	dir, err := substratedata.Catalog()
	if err != nil {
		t.Fatal(err)
	}
	data := repos().Data
	entries, err := os.ReadDir(filepath.Join(dir, "models"))
	if err != nil {
		t.Fatal(err)
	}
	listed := map[string]bool{}
	for _, table := range []struct {
		lists  map[string]string
		priced bool
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
			if g.Encoder == nil {
				t.Errorf("%s: the graph states no encoder section, so it is not a multimodal checkpoint", name)
				continue
			}
			if g.PricesEncoder() != table.priced {
				t.Errorf("%s: the graph prices a tower: %t, the list says %t", name, g.PricesEncoder(), table.priced)
			}
			unset := languageModelOnlyAssumptions(g, engineconfig.Engine{}, 1)
			f := false
			stated := languageModelOnlyAssumptions(g, engineconfig.Engine{LanguageModelOnly: &f}, 1)
			if table.priced && (unset != nil || stated != nil) {
				t.Errorf("%s: a graph that prices its tower was given assumptions %q %q", name, unset, stated)
			}
			if !table.priced {
				if len(unset) != 1 || !strings.Contains(unset[0], name) {
					t.Errorf("%s: unset gives assumptions %q, want one naming the model", name, unset)
				}
				if len(stated) != 1 || !strings.Contains(stated[0], "= false") {
					t.Errorf("%s: false gives assumptions %q, want one stating false", name, stated)
				}
			}
			if got := languageModelOnlyAssumptions(g, engineconfigStating(true), 1); got != nil {
				t.Errorf("%s: true gives assumptions %q, want none", name, got)
			}
		}
	}
	for _, e := range entries {
		g, err := data.Model(e.Name())
		if err != nil || g.Encoder == nil {
			continue
		}
		if !listed[e.Name()] {
			t.Errorf("%s has an encoder section and is in neither towerlessMultimodalConfigs nor pricedTowerConfigs", e.Name())
		}
	}
	for _, e := range entries {
		cfg, err := os.ReadFile(filepath.Join(dir, "models", e.Name(), "config.json"))
		if err != nil || !strings.Contains(string(cfg), `"vision_config"`) {
			continue
		}
		if g, err := data.Model(e.Name()); err != nil || g.Encoder == nil {
			t.Errorf("%s: config.json has a vision_config and the graph states no encoder section", e.Name())
		}
	}
	for _, want := range []string{"minimax-m3", "kimi-k3", "kimi-k2.5", "qwen3.5-397b-a17b", "gemma-4-31b-it", "glm-5.3-flash"} {
		g, err := data.Model(want)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := pricedTowerConfigs[g.DerivedFrom.SHA256]; !ok {
			t.Errorf("%s is not in pricedTowerConfigs", want)
		}
	}
	for _, want := range []string{"deepseek-v4.1-flash", "inkling"} {
		g, err := data.Model(want)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := towerlessMultimodalConfigs[g.DerivedFrom.SHA256]; !ok {
			t.Errorf("%s is not in towerlessMultimodalConfigs", want)
		}
	}
	if languageModelOnlyAssumptions(nil, engineconfigStating(true), 1) != nil {
		t.Error("a nil graph was given an assumption")
	}
}

func engineconfigStating(v bool) engineconfig.Engine {
	return engineconfig.Engine{LanguageModelOnly: &v}
}
