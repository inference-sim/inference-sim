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
	"github.com/inference-sim/blis-vllm/core/vllm"
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
// cache beside full attention) or data parallelism across ranks. A drafter is not such a piece:
// d001's MTP and d027's DSpark are refused for their KV groups alone.
func TestDashboardDeploymentsNeedingAPieceAreRefusedNamingIt(t *testing.T) {
	Register()
	cat, _ := filepath.Abs(filepath.Join("..", "testdata", "catalog"))
	reg, _ := filepath.Abs(filepath.Join("..", "testdata", "registry"))
	for _, c := range []struct {
		scenario string
		want     []string
	}{
		{"d036-032.yaml", []string{"hybrid KV groups"}},                            // gpt-oss-120b: sliding window
		{"d001-032.yaml", []string{"hybrid KV groups"}},                            // deepseek-v4-pro with mtp
		{"d027-032.yaml", []string{"hybrid KV groups"}},                            // deepseek-v4.1-flash with dspark
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
		if strings.Contains(err.Error(), "drafter") {
			t.Errorf("%s: %v, refused for its drafter, which the engine builds", c.scenario, err)
		}
	}
}

// openPool opens a scenario's pool as open does, without its refusal of the pieces the engine
// does not build yet, so a drafter's block can be checked on a pool still refused for another.
func openPool(t *testing.T, scenario string) pool {
	t.Helper()
	r := roots(t, scenario)
	m, err := kernelmodel.OpenPool(scenario, kernelmodel.Repos{Scenarios: filepath.Dir(r[RootScenario]),
		Catalog: r[RootCatalog], Registry: r[RootRegistry]}, 0)
	if err != nil {
		t.Fatal(err)
	}
	st, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	node, pp, err := poolBlock(r[RootScenario], 0)
	if err != nil {
		t.Fatal(err)
	}
	return pool{model: m, blocks: st.KVBlocks, engine: node, pp: pp}
}

// stated is the value a completed engine block gives key, "" when it states none.
func stated(block yaml.Node, key string) string {
	for i := 0; i+1 < len(block.Content); i += 2 {
		if block.Content[i].Value == key {
			return block.Content[i+1].Value
		}
	}
	return ""
}

// Prepare states an MTP drafter's layer count from the catalog: the model graph's speculator
// stack, whose layers are the checkpoint's num_nextn_predict_layers (deepseek-v4-pro: one), so
// the factory resolves the prefill lookahead vLLM reserves (vllm/config/vllm.py:644-664@5df63a72).
// A block that states the field keeps its own value, and a pool drafting with DSpark states none.
func TestPrepareStatesTheMTPDrafterLayers(t *testing.T) {
	d001 := openPool(t, "d001-032.yaml")
	if v := stated(d001.complete(yaml.Node{}), "mtp_nextn_predict_layers"); v != "1" {
		t.Errorf("d001 (deepseek-v4-pro, mtp): mtp_nextn_predict_layers %q, want the graph's 1", v)
	}
	var own yaml.Node
	if err := yaml.Unmarshal([]byte("speculative: {method: mtp, num_spec_tokens: 2}\nmtp_nextn_predict_layers: 3\n"), &own); err != nil {
		t.Fatal(err)
	}
	if v := stated(d001.complete(own), "mtp_nextn_predict_layers"); v != "3" {
		t.Errorf("a block stating 3: mtp_nextn_predict_layers %q, want its own 3", v)
	}
	if v := stated(openPool(t, "d027-032.yaml").complete(yaml.Node{}), "mtp_nextn_predict_layers"); v != "" {
		t.Errorf("d027 (deepseek-v4.1-flash, dspark): mtp_nextn_predict_layers %q, want none", v)
	}
}

// d001's MTP and d027's DSpark engine blocks, completed by Prepare, build through the factory
// with their drafters and the kernel's pricing, and run requests to completion through the batch
// queue, verifying drafts each decode and accepting the outcome's. The pools themselves stay
// refused for their hybrid KV groups until those are built
// (TestDashboardDeploymentsNeedingAPieceAreRefusedNamingIt): this checks the drafter is not
// what refuses them, and runs the KV pool as one full-attention group.
func TestDashboardDrafterBlocksBuild(t *testing.T) {
	for _, c := range []struct {
		scenario, method string
	}{{"d001-032.yaml", "mtp"}, {"d027-032.yaml", "dspark"}} {
		t.Run(c.scenario, func(t *testing.T) {
			p := openPool(t, c.scenario)
			block := p.complete(yaml.Node{})
			f := vllm.Factory{Pricer: func(engine.Spec) (vllm.Pricer, error) { return kernelPricer{m: p.model}, nil }}
			spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Config: block}
			e, err := f.New(spec)
			if err != nil {
				t.Fatalf("%s's engine block with its %s drafter: refused: %v", c.scenario, c.method, err)
			}
			if got := e.InFlightLimit(); got != 2 {
				t.Errorf("InFlightLimit %d, want 2: async scheduling stays on with a %s drafter", got, c.method)
			}
			tr, err := enginetest.Drive(func() (engine.Engine, error) { return f.New(spec) },
				[]enginetest.Arrival{{At: 0, Req: req("a", 512, 64)}, {At: 5000, Req: req("b", 1024, 32)}},
				enginetest.Options{Accepted: func(_ engine.Batch, s engine.Scheduled) int { return s.Drafts / 2 }})
			if err != nil {
				t.Fatal(err)
			}
			drafts := 0
			for _, st := range tr.Steps {
				for _, s := range st.Batch.Reqs {
					drafts += s.Drafts
				}
			}
			if len(tr.Finished) != 2 || drafts == 0 {
				t.Errorf("finished %d of 2 requests, verifying %d drafts", len(tr.Finished), drafts)
			}
		})
	}
}
