package vllmengine

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// roots are the repository's vendored catalog and registry, which hold the dashboard models.
func roots(t *testing.T, scenario string) map[string]string {
	t.Helper()
	abs, err := filepath.Abs(filepath.Join("testdata", scenario))
	if err != nil {
		t.Fatal(err)
	}
	cat, _ := filepath.Abs(filepath.Join("..", "testdata", "catalog"))
	reg, _ := filepath.Abs(filepath.Join("..", "testdata", "registry"))
	return map[string]string{RootCatalog: cat, RootRegistry: reg, RootScenario: abs}
}

// poolEngine is the scenario's first pool's engine block, as a host hands it over: unopened.
func poolEngine(t *testing.T, scenario string) yaml.Node {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", scenario))
	if err != nil {
		t.Fatal(err)
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	for {
		var doc struct {
			Kind  string `yaml:"kind"`
			Pools []struct {
				Engine yaml.Node `yaml:"engine"`
			} `yaml:"pools"`
		}
		if err := dec.Decode(&doc); err != nil {
			t.Fatalf("%s: no Deployment document: %v", scenario, err)
		}
		if doc.Kind == "Deployment" {
			return doc.Pools[0].Engine
		}
	}
}

// compatBlocks is the compatibility engine's KV block count for the same deployment at 0.31.0.
func compatBlocks(t *testing.T, scenario031 string) int64 {
	t.Helper()
	r := roots(t, scenario031)
	m, err := kernelmodel.OpenPool(filepath.Base(r[RootScenario]), kernelmodel.Repos{
		Scenarios: filepath.Dir(r[RootScenario]), Catalog: r[RootCatalog], Registry: r[RootRegistry]}, 0)
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	return s.KVBlocks
}

// A dashboard deployment at 0.32.0 builds through blis-schemas' registry with the latency
// kernel's pricing and KV sizing, runs requests to completion through the engine.Engine
// contract, and holds exactly the compatibility engine's block count for the same deployment.
func TestDashboardDeploymentsBuildAndRun(t *testing.T) {
	Register()
	for _, c := range []struct{ v032, v031 string }{
		{"d058-032.yaml", "d058-031.yaml"}, // kimi-k2.5, b200, tp4
		{"d071-032.yaml", "d071-031.yaml"}, // minimax-m2.5, b200, tp2
	} {
		t.Run(c.v032, func(t *testing.T) {
			spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Config: poolEngine(t, c.v032),
				Roots: roots(t, c.v032), Pool: 0}
			e, err := engine.New(spec)
			if err != nil {
				t.Fatal(err)
			}
			want := compatBlocks(t, c.v031)
			if got := e.Snapshot().KVBlocksTotal; got != want-1 {
				// Snapshot excludes the null block; compat's count includes it.
				t.Errorf("KV blocks: %d usable (%d with the null block), compat sizes %d", got, got+1, want)
			}
			var steps []engine.Duration
			tr, err := enginetest.Drive(func() (engine.Engine, error) { return engine.New(spec) },
				[]enginetest.Arrival{
					{At: 0, Req: req("a", 512, 64)},
					{At: 5000, Req: req("b", 1024, 32)},
				}, enginetest.Options{})
			if err != nil {
				t.Fatal(err)
			}
			for _, st := range tr.Steps {
				steps = append(steps, st.End-st.Start)
			}
			if len(tr.Finished) != 2 {
				t.Fatalf("finished %d of 2 requests", len(tr.Finished))
			}
			for i, d := range steps {
				if d <= 1 {
					t.Errorf("step %d took %d us: not kernel-priced", i, d)
				}
			}
			if a, ok := e.(interface{ Assumptions() []string }); !ok || len(a.Assumptions()) == 0 {
				t.Errorf("the engine discloses no pricing assumption; the kernel's 0.31.0 tuning prices 0.32.0")
			}
		})
	}
}

func req(id string, prompt, output int) engine.Request {
	r := engine.Request{ID: id, Prompt: make([]int32, prompt), Output: make([]int32, output)}
	for i := range r.Prompt {
		r.Prompt[i] = int32(len(id)*1000 + i + 1)
	}
	for i := range r.Output {
		r.Output[i] = int32(9000 + i)
	}
	return r
}

// A 0.32.0 deployment that needs a piece this engine does not build yet is refused, naming the
// piece, rather than run without it: hybrid KV groups (a sliding-window, indexer or recurrent
// cache beside full attention), an MTP drafter, or data parallelism across ranks.
func TestDashboardDeploymentsNeedingAPieceAreRefusedNamingIt(t *testing.T) {
	Register()
	cat, _ := filepath.Abs(filepath.Join("..", "testdata", "catalog"))
	reg, _ := filepath.Abs(filepath.Join("..", "testdata", "registry"))
	for _, c := range []struct {
		scenario string
		want     []string
	}{
		{"d036-032.yaml", []string{"hybrid KV groups"}},                            // gpt-oss-120b: sliding window
		{"d001-032.yaml", []string{"hybrid KV groups", "MTP drafter"}},             // deepseek-v4-pro with mtp
		{"d116-032.yaml", []string{"hybrid KV groups", "data parallelism (dp 4)"}}, // minimax-m3 dp 4
		{"d079-032.yaml", []string{"data parallelism (dp 2)"}},                     // minimax-m2.5 dp 2
	} {
		abs, _ := filepath.Abs(filepath.Join("testdata", c.scenario))
		_, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0",
			Roots: map[string]string{RootCatalog: cat, RootRegistry: reg, RootScenario: abs}})
		if err == nil {
			t.Errorf("%s: built, want a refusal naming %v", c.scenario, c.want)
			continue
		}
		for _, w := range c.want {
			if !strings.Contains(err.Error(), w) {
				t.Errorf("%s: %v, want it to name %q", c.scenario, err, w)
			}
		}
	}
}
