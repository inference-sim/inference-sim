package vllmengine

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/core/kv"
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

// No dashboard deployment in testdata needs a piece this engine refuses: data parallelism
// builds a DPGroup with or without each rank's batch queue, and a drafter and the DeepSeek and
// MiniMax KV cache groups build. d116 (minimax-m3, dp 4) and d079 (minimax-m2.5, dp 2), which
// were refused for running each rank's batch queue, build as stated, and the build never names
// data parallelism as a missing piece.
func TestDashboardDataParallelDeploymentsAreNotRefused(t *testing.T) {
	Register()
	cat, _ := filepath.Abs(filepath.Join("..", "testdata", "catalog"))
	reg, _ := filepath.Abs(filepath.Join("..", "testdata", "registry"))
	for _, scenario := range []string{"d116-032.yaml", "d079-032.yaml", "d088-032.yaml", "d103-032.yaml"} {
		abs, _ := filepath.Abs(filepath.Join("testdata", scenario))
		eng, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0",
			Roots: map[string]string{RootCatalog: cat, RootRegistry: reg, RootScenario: abs}})
		if err != nil {
			if strings.Contains(err.Error(), "data parallelism") {
				t.Errorf("%s: %v refuses data parallelism, which the engine builds", scenario, err)
			} else {
				t.Errorf("%s: %v", scenario, err)
			}
			continue
		}
		if g, ok := eng.(*vllm.DPGroup); !ok || g.InFlightLimit() != 2 {
			t.Errorf("%s built %T, want a DPGroup with a batch queue of 2 on each rank", scenario, eng)
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
	node, par, err := poolBlock(r[RootScenario], 0)
	if err != nil {
		t.Fatal(err)
	}
	return pool{model: m, blocks: st.KVBlocks, engine: node, pp: max(par.PP, 1), dp: max(par.DP, 1)}
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

// The data-parallel dashboard deployments whose KV cache is one full-attention group build, as
// stated, a DPGroup through engine.New and run requests to completion through the engine
// contract, every lockstep step priced by the kernel's rule: minimax-m2.5 at dp 2 with expert
// parallelism on one node, on B200 (d079) and B300 (d088, and d103 at an 8192-token prompt).
// vLLM resolves async scheduling on for each, so every rank runs its batch queue of two.
func TestDashboardDataParallelDeploymentsRun(t *testing.T) {
	Register()
	for _, name := range []string{"d079", "d088", "d103"} {
		t.Run(name, func(t *testing.T) {
			r := roots(t, name+"-032.yaml")
			_, par, err := poolBlock(r[RootScenario], 0)
			if err != nil {
				t.Fatal(err)
			}
			if par.DP != 2 {
				t.Fatalf("%s states dp %d, want 2", name, par.DP)
			}
			spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: r}
			eng, err := engine.New(spec)
			if err != nil {
				t.Fatal(err)
			}
			g, ok := eng.(*vllm.DPGroup)
			if !ok {
				t.Fatalf("%s (dp 2) built %T, not a DPGroup", name, eng)
			}
			if g.Ranks() != 2 || g.InFlightLimit() != 2 {
				t.Fatalf("%s built %d ranks with %d batches in flight, want 2 ranks, each with async scheduling's "+
					"queue of 2", name, g.Ranks(), g.InFlightLimit())
			}
			prompt := 300
			if name == "d103" {
				prompt = 8192
			}
			var reqs []enginetest.Arrival
			for i := 0; i < 2; i++ {
				rq := engine.Request{ID: fmt.Sprintf("r%d", i), Prompt: make([]int32, prompt), Output: make([]int32, 20)}
				for j := range rq.Prompt {
					rq.Prompt[j] = int32(i*100000 + j + 1)
				}
				for j := range rq.Output {
					rq.Output[j] = int32(900000 + j)
				}
				reqs = append(reqs, enginetest.Arrival{At: engine.Time(i * 1000), Req: rq})
			}
			tr := enginetest.TestEngine(t, func() (engine.Engine, error) { return engine.New(spec) }, reqs, enginetest.Options{})
			if len(tr.Finished) != 2 {
				t.Fatalf("finished %d of 2 requests", len(tr.Finished))
			}
			for _, s := range tr.Steps {
				if s.End-s.Start <= 1 && !s.Batch.Empty() {
					t.Errorf("batch %d priced at %d µs; the kernel prices a real forward pass", s.Batch.ID, s.End-s.Start)
				}
			}
		})
	}
}

// The hybrid-KV dashboard models whose layers vLLM groups at one block size build through
// engine.New with vLLM's groups and run requests to completion: gpt-oss-120b's windowed and
// full layers are two groups, window first, and MiniMax-M3's full-attention and indexer layers
// are one, as vLLM v0.32.0's get_kv_cache_groups groups them (core/sched's dashboard-* diff
// scenarios).
func TestDashboardHybridKVDeploymentsBuildAndRun(t *testing.T) {
	Register()
	for _, c := range []struct {
		scenario string
		want     []kv.Group
	}{
		{"d036-032.yaml", []kv.Group{{Kind: kv.SlidingWindow, Window: 128}, {Kind: kv.FullAttention}}}, // gpt-oss-120b
		{"d106-032.yaml", []kv.Group{{Kind: kv.FullAttention}}},                                        // minimax-m3
	} {
		t.Run(c.scenario, func(t *testing.T) {
			spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, c.scenario)}
			done, err := prepare(spec)
			if err != nil {
				t.Fatal(err)
			}
			var block struct {
				Layers []vllm.KVLayerRun `yaml:"kv_layers"`
			}
			if err := done.Config.Decode(&block); err != nil {
				t.Fatal(err)
			}
			got, err := vllm.KVGroups(block.Layers, false, false)
			if err != nil || !reflect.DeepEqual(got, c.want) {
				t.Errorf("KV groups %v (%v), want %v", got, err, c.want)
			}
			tr, err := enginetest.Drive(func() (engine.Engine, error) { return engine.New(spec) },
				[]enginetest.Arrival{{At: 0, Req: req("a", 512, 64)}, {At: 5000, Req: req("b", 1024, 32)}},
				enginetest.Options{})
			if err != nil {
				t.Fatal(err)
			}
			if len(tr.Finished) != 2 {
				t.Fatalf("finished %d of 2 requests", len(tr.Finished))
			}
		})
	}
}

// The KV pool holds what vLLM's would for the same memory: the memory left for KV over the widest
// KV cache group's bytes per block (get_kv_cache_config_from_groups,
// vllm/v1/core/kv_cache_utils.py:1622-1628@5df63a72). gpt-oss-120b's 36 layers are two groups of
// 18, a window and full attention, so a block is 18 layers' pages: the kernel's per-block bytes,
// which are its 18 full-attention layers' (the window layers' rings are bounded per sequence, not
// paged with context), and half its 36 layers' page bytes. A model of one group keeps the
// compatibility engine's count: MiniMax-M3 (full attention and its indexers, one uniform-type
// group) and kimi-k2.5 (full attention only).
func TestKVPoolBlocksAreVLLMsOverTheKVCacheGroups(t *testing.T) {
	Register()
	for _, c := range []struct {
		scenario string
		want     func(compat kernelmodel.KVBudget) int64
	}{
		{"d036-032.yaml", func(b kernelmodel.KVBudget) int64 { return b.AllocatableBytes / b.PerBlockBytes }}, // gpt-oss-120b
		{"d106-032.yaml", func(b kernelmodel.KVBudget) int64 { return b.PerRankBlocks }},                      // minimax-m3
		{"d058-032.yaml", func(b kernelmodel.KVBudget) int64 { return b.PerRankBlocks }},                      // kimi-k2.5
		// minimax-m3 with cache_dtype fp8: the attention layers' KV at fp8, the indexer's keys at
		// bf16 (its own dtype), so the pool holds what vLLM sizes for one uniform-type group of
		// two page sizes: 72893 blocks (oracle/scenarios/dashboard-minimax-m3-fp8-kv-probe, at
		// d106's bytes, and dashboard-minimax-m3-fp8-kv-pool at the dashboard's b200 TP8).
		{"d106-fp8kv-032.yaml", func(kernelmodel.KVBudget) int64 { return 72893 }},
	} {
		t.Run(c.scenario, func(t *testing.T) {
			b, err := openPool(t, c.scenario).model.KVBudget()
			if err != nil {
				t.Fatal(err)
			}
			e, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, c.scenario)})
			if err != nil {
				t.Fatal(err)
			}
			want := c.want(b)
			if got := e.Snapshot().KVBlocksTotal + 1; got != want {
				// Snapshot excludes the null block.
				t.Errorf("KV pool of %d blocks, want %d (compat sizes %d of %d bytes each from %d bytes)",
					got, want, b.PerRankBlocks, b.PerBlockBytes, b.AllocatableBytes)
			}
		})
	}
}

// vLLM v0.32.0 refuses to start an engine whose KV pool cannot hold one max_model_len request
// across every KV cache group (get_kv_cache_configs, vllm/v1/core/kv_cache_utils.py:2627-2646@5df63a72),
// and so does the factory. Dashboard d025 (deepseek-v4-pro, h200, tp1 dp8, max_model_len 9472)
// has 3.01 GiB of KV per rank, 2250 blocks. vLLM resolves async scheduling on for it, two
// batches in flight, so each sliding window's admission cap holds window - 1 + 2 x 8192 tokens:
// one request needs 5.2 GiB, and vLLM's own check raises with these numbers (4.53 GiB with
// async scheduling off; oracle/scenarios/dashboard-deepseek-v4-pro-startup*.json). The same
// deployment at max_model_len 4096 fits and builds.
func TestADeploymentVLLMRefusesAtStartupIsRefusedWithVLLMsMessage(t *testing.T) {
	Register()
	_, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, "d025-032.yaml")})
	want := "To serve at least one request with the model's max seq len (9472), (5.2 GiB KV cache is needed, " +
		"which is larger than the available KV cache memory (3.01 GiB). Based on the available memory, the " +
		"estimated maximum model length is 5468."
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("d025: err %v, want vLLM's start-up refusal %q", err, want)
	}
	dir := t.TempDir()
	body, rerr := os.ReadFile(filepath.Join("testdata", "d025-032.yaml"))
	if rerr != nil {
		t.Fatal(rerr)
	}
	fits := strings.Replace(string(body), "max_model_len: 9472", "max_model_len: 4096", 1)
	if err := os.WriteFile(filepath.Join(dir, "d025-4096.yaml"), []byte(fits), 0o644); err != nil {
		t.Fatal(err)
	}
	r := roots(t, "d025-032.yaml")
	r[RootScenario] = filepath.Join(dir, "d025-4096.yaml")
	tr, err := enginetest.Drive(func() (engine.Engine, error) {
		return engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: r})
	}, []enginetest.Arrival{{At: 0, Req: req("a", 2048, 16)}}, enginetest.Options{})
	if err != nil {
		t.Fatalf("d025 at max_model_len 4096: %v", err)
	}
	if len(tr.Finished) != 1 {
		t.Errorf("d025 at max_model_len 4096 finished %d of 1 requests", len(tr.Finished))
	}
}

// Muse-Spark-Open-816B-A42B's hybrid KV (16 full, 46 sliding-window layers at 2048) is checked
// at start-up over its groups, as vLLM checks it, not as one full-attention group: a DP8
// rank on H200 holds a 262,144-token request (a full-attention group at full context and three
// sliding-window groups at their admission caps), which every layer at full context would not.
// Each arm runs: H100 RoCE PP2 x TP8 with and without the DSpark drafter, H200 InfiniBand
// DP8 x EP8, and GB200 TP8 x EP8 and DP8 x EP8 with DSpark over two trays of one NVL72 rack.
func TestMuseSparkDeploymentsBuildAndRun(t *testing.T) {
	Register()
	for _, s := range []string{"muse-h100-roce-armB.yaml", "muse-h100-roce-armC.yaml",
		"muse-h200-ib-dep8.yaml", "muse-h200-ib-dep8-dspark-128k.yaml", "muse-gb200-nvl-tep8.yaml",
		"muse-gb200-nvl-dep8-dspark.yaml"} {
		t.Run(s, func(t *testing.T) {
			spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, s)}
			done, err := prepare(spec)
			if err != nil {
				t.Fatal(err)
			}
			var block struct {
				Layers []vllm.KVLayerRun `yaml:"kv_layers"`
			}
			if err := done.Config.Decode(&block); err != nil {
				t.Fatal(err)
			}
			if groups, err := vllm.KVGroups(block.Layers, false, false); err != nil || len(groups) < 2 {
				t.Errorf("%d KV groups (%v): want the hybrid model's several", len(groups), err)
			}
			tr, err := enginetest.Drive(func() (engine.Engine, error) { return engine.New(spec) },
				[]enginetest.Arrival{{At: 0, Req: req("a", 4096, 16)}, {At: 5000, Req: req("b", 8192, 8)}},
				enginetest.Options{Accepted: func(engine.Batch, engine.Scheduled) int { return 2 }})
			if err != nil {
				t.Fatal(err)
			}
			if len(tr.Finished) != 2 {
				t.Fatalf("finished %d of 2 requests", len(tr.Finished))
			}
		})
	}
}

// What vLLM refuses to start, the engine refuses, with vLLM's message: H200 DP8 x EP8 with the
// DSpark drafter at a 262,144-token window leaves a rank 16,124 blocks, and one request needs
// cdiv(262144, 16) = 16,384 for its full-attention group plus 4 x 1,153 for its sliding-window
// groups (get_kv_cache_configs' check,
// vllm/v1/core/kv_cache_utils.py:2627-2647@5df63a72). The same arm at 131,072 runs (above).
func TestMuseSparkDeploymentVLLMRefusesIsRefused(t *testing.T) {
	Register()
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, "muse-h200-ib-dep8-dspark.yaml")}
	_, err := engine.New(spec)
	if err == nil || !strings.Contains(err.Error(), "To serve at least one request with the model's max seq len (262144)") {
		t.Errorf("err %v, want vLLM's start-up refusal", err)
	}
}

// Muse Spark's P/D arm, 1P1D with TP16 + EP16 on each side over RoCE H100 (neither side at PP, so
// NIXL's worker accepts it): the production factory gives the prefill pool NIXL as kv_producer
// and the decode pool NIXL as kv_consumer, from the deployment's pd_transfer and each pool's
// role. A disaggregated request then prefills on one side, is handed off, pulls on the other and
// decodes there.
func TestMuseSparkPDPoolsRunThroughTheEngine(t *testing.T) {
	Register()
	const s = "muse-h100-roce-1p1d.yaml"
	var engines [2]engine.Engine
	for i, want := range []string{"kv_producer", "kv_consumer"} {
		spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, s), Pool: i}
		done, err := prepare(spec)
		if err != nil {
			t.Fatal(err)
		}
		var block struct {
			KV struct {
				Connector string `yaml:"connector"`
				Role      string `yaml:"kv_role"`
			} `yaml:"kv_transfer"`
		}
		if err := done.Config.Decode(&block); err != nil {
			t.Fatal(err)
		}
		if block.KV.Connector != "NixlConnector" || block.KV.Role != want {
			t.Fatalf("pool %d's kv_transfer is %+v, want NixlConnector as %s", i, block.KV, want)
		}
		if engines[i], err = engine.New(spec); err != nil {
			t.Fatal(err)
		}
	}
	p, d := engines[0], engines[1]
	r := req("a", 8192, 32)
	pre := r
	pre.RemoteDecode, pre.MaxOutputTokens = true, 1
	if err := p.Add(pre, 0); err != nil {
		t.Fatal(err)
	}
	now := engine.Time(0)
	var held []engine.Handoff
	for len(held) == 0 {
		b, err := p.Schedule(now)
		if err != nil || b.Empty() {
			t.Fatalf("prefill schedule: %v, empty %v", err, b.Empty())
		}
		now += p.StepTime(b)
		res, err := p.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		held = res.Handoffs
	}
	dec := r
	dec.RemotePrefill = true
	if err := d.Add(dec, now); err != nil {
		t.Fatal(err)
	}
	b, err := d.Schedule(now)
	if err != nil || len(b.TransfersStarted) != 1 || b.TransfersStarted[0].Tokens != 8192 {
		t.Fatalf("decode schedule: %v, transfers %+v, want one pull of 8192 tokens", err, b.TransfersStarted)
	}
	now += 5000
	if err := d.TransferDone(b.TransfersStarted[0].ID, true, now); err != nil {
		t.Fatal(err)
	}
	if err := p.TransferDone(held[0].Transfer, true, now); err != nil {
		t.Fatal(err)
	}
	emitted := 0
	for steps := 0; emitted < 32 && steps < 200; steps++ {
		b, err := d.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			now += 1000
			continue
		}
		for _, sc := range b.Reqs {
			if sc.Computed == 0 && sc.Tokens > 1 {
				t.Fatalf("the decoder computes the prompt itself: %+v", sc)
			}
		}
		now += d.StepTime(b)
		res, err := d.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		emitted += res.Emitted["a"]
	}
	if emitted != 32 {
		t.Fatalf("the decoder emitted %d of 32 tokens", emitted)
	}
}
