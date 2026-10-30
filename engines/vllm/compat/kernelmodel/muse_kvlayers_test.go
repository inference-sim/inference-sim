package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// Muse-Spark-Open-816B-A42B's KV layers, as vLLM v0.32.0 registers them: 16 full-attention and
// 46 sliding-window (2048) layers in layer_types order; with its DSpark drafter named by
// speculative.draft_model, five sliding-window draft layers after them; and under pipeline
// parallelism, each stage only its own share, the drafter's on the last.
func TestMuseSparkKVLayersAreTheEngines(t *testing.T) {
	r := repos()
	r.Scenarios = filepath.Join(repoRoot(), "testdata", "scenarios")
	count := func(runs []KVLayerRun) (full, sliding int) {
		for _, x := range runs {
			switch x.Spec {
			case "full":
				full += x.Count
			case "sliding_window":
				if x.SlidingWindow != 2048 {
					t.Errorf("a sliding window of %d", x.SlidingWindow)
				}
				sliding += x.Count
			}
		}
		return full, sliding
	}
	for _, c := range []struct {
		scenario      string
		full, sliding int
		stages        [][2]int
	}{
		{"muse-h100-pp2tp8.yaml", 16, 46, [][2]int{{8, 23}, {8, 23}}},
		{"muse-h100-pp2tp8-dspark.yaml", 16, 51, [][2]int{{8, 23}, {8, 28}}},
	} {
		m, err := Open(c.scenario, r)
		if err != nil {
			t.Fatalf("%s: %v", c.scenario, err)
		}
		runs, err := m.KVLayers()
		if err != nil {
			t.Fatalf("%s: %v", c.scenario, err)
		}
		if f, s := count(runs); f != c.full || s != c.sliding {
			t.Errorf("%s: %d full and %d sliding layers, want %d and %d", c.scenario, f, s, c.full, c.sliding)
		}
		if m.PipelineStages() != len(c.stages) {
			t.Fatalf("%s: %d stages", c.scenario, m.PipelineStages())
		}
		for s, want := range c.stages {
			runs, err := m.KVLayersOfStage(s)
			if err != nil {
				t.Fatalf("%s stage %d: %v", c.scenario, s, err)
			}
			if f, sl := count(runs); f != want[0] || sl != want[1] {
				t.Errorf("%s stage %d: %d full and %d sliding, want %d and %d", c.scenario, s, f, sl, want[0], want[1])
			}
		}
	}
}

// Each stage's layers are vLLM's own split: get_pp_indices at v0.32.0
// (vllm/distributed/utils.py:126-169@5df63a72), printed from the pinned source, for layer
// counts with and without a remainder.
func TestPipelineStagesSplitLayersAsVLLMDoes(t *testing.T) {
	for _, c := range []struct {
		layers, pp int
		want       [][2]int
	}{
		{5, 2, [][2]int{{0, 3}, {3, 5}}},
		{5, 3, [][2]int{{0, 2}, {2, 4}, {4, 5}}},
		{5, 4, [][2]int{{0, 1}, {1, 2}, {2, 4}, {4, 5}}},
		{7, 2, [][2]int{{0, 4}, {4, 7}}},
		{7, 3, [][2]int{{0, 2}, {2, 5}, {5, 7}}},
		{7, 4, [][2]int{{0, 2}, {2, 4}, {4, 6}, {6, 7}}},
		{30, 2, [][2]int{{0, 15}, {15, 30}}},
		{30, 3, [][2]int{{0, 10}, {10, 20}, {20, 30}}},
		{30, 4, [][2]int{{0, 7}, {7, 15}, {15, 23}, {23, 30}}},
		{30, 8, [][2]int{{0, 3}, {3, 7}, {7, 11}, {11, 15}, {15, 19}, {19, 23}, {23, 27}, {27, 30}}},
		{61, 2, [][2]int{{0, 31}, {31, 61}}},
		{61, 3, [][2]int{{0, 20}, {20, 41}, {41, 61}}},
		{61, 4, [][2]int{{0, 15}, {15, 30}, {30, 46}, {46, 61}}},
		{61, 8, [][2]int{{0, 7}, {7, 14}, {14, 22}, {22, 30}, {30, 38}, {38, 46}, {46, 54}, {54, 61}}},
		{62, 2, [][2]int{{0, 31}, {31, 62}}},
		{62, 3, [][2]int{{0, 21}, {21, 42}, {42, 62}}},
		{62, 4, [][2]int{{0, 15}, {15, 31}, {31, 47}, {47, 62}}},
		{62, 8, [][2]int{{0, 7}, {7, 15}, {15, 23}, {23, 31}, {31, 39}, {39, 47}, {47, 55}, {55, 62}}},
		{78, 2, [][2]int{{0, 39}, {39, 78}}},
		{78, 3, [][2]int{{0, 26}, {26, 52}, {52, 78}}},
		{78, 4, [][2]int{{0, 19}, {19, 39}, {39, 59}, {59, 78}}},
		{78, 8, [][2]int{{0, 9}, {9, 19}, {19, 29}, {29, 39}, {39, 49}, {49, 59}, {59, 69}, {69, 78}}},
		{94, 2, [][2]int{{0, 47}, {47, 94}}},
		{94, 3, [][2]int{{0, 31}, {31, 63}, {63, 94}}},
		{94, 4, [][2]int{{0, 23}, {23, 47}, {47, 71}, {71, 94}}},
		{94, 8, [][2]int{{0, 11}, {11, 23}, {23, 35}, {35, 47}, {47, 59}, {59, 71}, {71, 83}, {83, 94}}},
	} {
		for s, w := range c.want {
			if lo, hi := ppIndices(c.layers, c.pp, s); lo != w[0] || hi != w[1] {
				t.Errorf("%d layers over pp %d, stage %d: [%d, %d), vLLM gives %v", c.layers, c.pp, s, lo, hi, w)
			}
		}
	}
}

// RolePools indexes a P/D deployment's pools by role, as engine.Spec.Pool selects them.
func TestRolePoolsIndexesEachRole(t *testing.T) {
	r := repos()
	r.Scenarios = "../../vllmengine/testdata"
	got, err := RolePools("muse-h100-roce-1p1d.yaml", r)
	if err != nil {
		t.Fatal(err)
	}
	if got[deployment.RolePrefill] != 0 || got[deployment.RoleDecode] != 1 || len(got) != 2 {
		t.Fatalf("pools by role %v, want prefill 0 and decode 1", got)
	}
}

// RolePools refuses a deployment stating one role in more than one pool: one engine layout per
// role is what a simulated pool runs, and Spec.Pool could select only one of them.
func TestRolePoolsRefusesARepeatedRole(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "vllmengine", "testdata", "muse-h100-roce-1p1d.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	twoPrefill := strings.Replace(string(src), "  - role: decode", "  - role: prefill", 1)
	if err := os.WriteFile(filepath.Join(dir, "two.yaml"), []byte(twoPrefill), 0o644); err != nil {
		t.Fatal(err)
	}
	r := repos()
	r.Scenarios = dir
	if _, err := RolePools("two.yaml", r); err == nil || !strings.Contains(err.Error(), "more than one prefill pool") {
		t.Fatalf("two prefill pools: %v, want refused", err)
	}
}
