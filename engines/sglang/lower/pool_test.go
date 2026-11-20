// Original to blis-sglang.

package lower

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

func readYAML(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := yaml.Unmarshal(b, v); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func graph(t *testing.T, name string) *model.Graph {
	t.Helper()
	var g model.Graph
	readYAML(t, "testdata/"+name+".graph.yaml", &g)
	return &g
}

func h200(t *testing.T) *hardware.Chip {
	t.Helper()
	var c hardware.Chip
	readYAML(t, "testdata/h200.yaml", &c)
	c.Name = "h200"
	return &c
}

// pool is a one-pool deployment whose engine block is block (YAML) at tensor-parallel width tp.
func pool(t *testing.T, tp int, block string) *deployment.Deployment {
	t.Helper()
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(block), &n); err != nil {
		t.Fatal(err)
	}
	return &deployment.Deployment{Pools: []deployment.Pool{{
		Role: "colocated", Nodes: 1, Parallel: deployment.Parallelism{TP: tp, PP: 1, DP: 1},
		Engine: deployment.NewEngineBlock(n.Content[0]),
	}}}
}

func lowerPool(t *testing.T, g string, tp int, block string) (*plan.Lowered, error) {
	t.Helper()
	return Pool(PoolInput{Deployment: pool(t, tp, block), Model: graph(t, g), Chip: h200(t)})
}

// The facts the lowering hands config.Resolve for a catalog model are the facts SGLang's own
// ModelConfig derives for the same checkpoint, which the oracle records for each model it resolves.
func TestModelFactsMatchTheOracle(t *testing.T) {
	a := readAnswers(t)
	for _, tc := range []struct{ graph, oracleModel string }{
		{"deepseek-r1-0528", "DeepseekV3ForCausalLM"},
		{"glm-5", "GlmMoeDsaForCausalLM"},
	} {
		want := a.facts(t, tc.oracleModel)
		got, err := ModelFactsOf(graph(t, tc.graph))
		if err != nil {
			t.Fatal(err)
		}
		// The oracle models a checkpoint (deepseek-v3, glm-5.3); the catalog graph is a sibling
		// checkpoint of the same architecture. The architecture's own facts must agree.
		got.HiddenSize, want.HiddenSize = nil, nil
		// An empty list of expert quant_algos is the same fact as none (the oracle records [], JSON
		// of an older answer file omits it).
		got.MixedPrecisionMoEQuantAlgos, want.MixedPrecisionMoEQuantAlgos = nil, nil
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: facts %+v, SGLang derives %+v", tc.graph, got, want)
		}
	}
}

type answerSet struct{ cases []config.ModelFacts }

func (a answerSet) facts(t *testing.T, arch string) config.ModelFacts {
	t.Helper()
	for _, f := range a.cases {
		if f.Architecture == arch {
			return f
		}
	}
	t.Fatalf("the oracle resolved no %s", arch)
	return config.ModelFacts{}
}

func readAnswers(t *testing.T) answerSet {
	t.Helper()
	b, err := os.ReadFile("../config/testdata/serverargs.json")
	if err != nil {
		t.Fatal(err)
	}
	var d struct {
		Cases []struct {
			ModelFacts config.ModelFacts `json:"model_facts"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatal(err)
	}
	var out answerSet
	for _, c := range d.Cases {
		out.cases = append(out.cases, c.ModelFacts)
	}
	return out
}

// The chip memory the lowering states is the figure the oracle's cases state for the chip.
func TestChipMemoryIsTheOraclesFigure(t *testing.T) {
	b, err := os.ReadFile("../oracle/serverargs/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var d struct {
		Chips map[string]config.Chip `json:"chips"`
	}
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatal(err)
	}
	for name, mib := range chipMemoryMiB {
		if d.Chips[name].MemoryMiB != mib {
			t.Errorf("%s: %v MiB, the oracle states %v", name, mib, d.Chips[name].MemoryMiB)
		}
	}
	if _, err := ChipOf(&hardware.Chip{Name: "mystery", ComputeCapability: "9.0"}); err == nil {
		t.Error("a chip with no stated nvidia-smi memory was accepted")
	}
}

// DeepSeek-R1 on an H200 at SGLang's defaults: bf16-width KV (an MLA latent cell), Triton's fused
// MoE for block-fp8 experts, a full decode graph and breakable prefill graphs, the overlap
// schedule, and the custom all-reduce at width 8.
func TestPoolLowersDeepSeekOnH200(t *testing.T) {
	low, err := lowerPool(t, "deepseek-r1-0528", 8, "mem_fraction_static: 0.82\nmax_running_requests: 512\ndisable_radix_cache: true\n")
	if err != nil {
		t.Fatal(err)
	}
	p := low.Plan
	if p.WeightFormat != "" || p.Cache.ElementBytes != 2 || p.Cache.Record != "" {
		t.Errorf("weights %q, KV element %v bytes record %q; want the checkpoint's own, 2, the model's own", p.WeightFormat, p.Cache.ElementBytes, p.Cache.Record)
	}
	if p.MoE.Experts != plan.ExpertsTriton || p.MoE.Routing != plan.RoutingSeparate {
		t.Errorf("experts %+v routing %q, want Triton's fused_moe with a separate selection", p.MoE.Experts, p.MoE.Routing)
	}
	if p.Graphs.Decode != plan.LaunchGraph || p.Graphs.Other != plan.LaunchPiecewise || p.Graphs.Mode != plan.GraphByBatch {
		t.Errorf("graphs %+v, want a full decode graph by batch and breakable prefill", p.Graphs)
	}
	if p.Batch.Requests != 512 || p.MemoryFraction != 0.82 || p.PageTokens != 1 || p.InFlight != 2 {
		t.Errorf("batch %+v fraction %v page %v in-flight %v", p.Batch, p.MemoryFraction, p.PageTokens, p.InFlight)
	}
	if got := p.Graphs.Buckets[len(p.Graphs.Buckets)-1]; got != 512 {
		t.Errorf("largest decode graph %d, want 512 (max_running_requests bounds the capture)", got)
	}
	if !low.Reported.SMAllReduce {
		t.Error("tp 8 should use SGLang's custom all-reduce")
	}
	// Resolution auto-enables the FlashInfer all-reduce fusion at tp 8 on SM90 (overrides.py:984-1001):
	// two norms a layer fold into the all-reduce for steps of up to 2048 tokens.
	if len(p.Launches) != 1 || p.LaunchChangeAt(1) != -2 || p.LaunchChangeAt(2048) != -2 || p.LaunchChangeAt(2049) != 0 || p.LaunchChangeAt(0) != 0 {
		t.Errorf("launch changes %+v, want -2 a layer for 1..2048 tokens", p.Launches)
	}
	// At tp 1 there is no all-reduce to fuse into and no custom all-reduce.
	one, err := lowerPool(t, "deepseek-r1-0528", 1, "mem_fraction_static: 0.82\n")
	if err != nil {
		t.Fatal(err)
	}
	if len(one.Plan.Launches) != 0 || one.Reported.SMAllReduce {
		t.Errorf("tp 1: launches %+v custom all-reduce %v", one.Plan.Launches, one.Reported.SMAllReduce)
	}
}

// GLM-5 resolves to the DSA attention backend, 64-token pages and a bf16 KV cache.
func TestPoolLowersGLMOnH200(t *testing.T) {
	low, err := lowerPool(t, "glm-5", 8, "mem_fraction_static: 0.85\n")
	if err != nil {
		t.Fatal(err)
	}
	if low.Plan.PageTokens != 64 || low.Plan.Cache.ElementBytes != 2 {
		t.Errorf("page %v element %v; SGLang's DSA layout pages by 64 at 2 bytes", low.Plan.PageTokens, low.Plan.Cache.ElementBytes)
	}
	var found bool
	for _, n := range low.Notes {
		found = found || n.Name == "max_running_requests"
	}
	if !found {
		t.Error("an unstated max_running_requests is a Note: SGLang sizes it from the pool")
	}
}

// Each setting the lowering does not price is refused naming itself, never priced as if it were
// the default.
func TestPoolRefusesWhatItDoesNotLower(t *testing.T) {
	for _, tc := range []struct{ block, name string }{
		{"kv_cache_dtype: fp8_e5m2x\n", "kv_cache_dtype"},
		{"moe_a2a_backend: deepep\n", "moe_a2a_backend"},
	} {
		_, err := lowerPool(t, "deepseek-r1-0528", 8, tc.block)
		var nl *NotLowered
		var nm *config.NotModelled
		var up *config.UpstreamRefusal
		if !errors.As(err, &nl) && !errors.As(err, &nm) && !errors.As(err, &up) && !strings.Contains(strings.ToLower(errString(err)), tc.name) {
			t.Errorf("%s: got %v, want a refusal naming it", tc.block, err)
		}
	}
	d := pool(t, 8, "mem_fraction_static: 0.8\n")
	d.Pools[0].Parallel.PP = 2
	if _, err := Pool(PoolInput{Deployment: d, Model: graph(t, "deepseek-r1-0528"), Chip: h200(t)}); err == nil {
		t.Error("pipeline parallelism was lowered")
	}
	d = pool(t, 8, "tp_size: 4\n")
	if _, err := Pool(PoolInput{Deployment: d, Model: graph(t, "deepseek-r1-0528"), Chip: h200(t)}); err == nil || !strings.Contains(err.Error(), "agree") {
		t.Errorf("a block and a pool that disagree on tp_size: %v", err)
	}
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// TestInertFieldsChangeNoDecision is the proof fields/status.json cites for the fields the
// recipes state that no decision reads. For each, resolving a pool with the field set to a probe
// value gives every other ServerArgs field the value it has with the field unset, and lowers to the
// same plan: the field reaches neither the scheduler's settings nor the cost model's. The reads in
// SGLang itself are listed with each field in the status file (the HTTP server and the
// tokenizer, detokenizer and metrics-label code; stream_interval only batches the tokens the
// detokenizer is sent, output_streamer.py:455-470@297e54fd).
func TestInertFieldsChangeNoDecision(t *testing.T) {
	probes := map[string][]string{
		"host":                 {"0.0.0.0", "127.0.0.1"},
		"port":                 {"8888", "30000"},
		"trust_remote_code":    {"true", "false"},
		"served_model_name":    {"glm-5-fp8", "m"},
		"tool_call_parser":     {"glm47", "qwen"},
		"reasoning_parser":     {"glm45", "qwen3"},
		"stream_interval":      {"10", "1", "64"},
		"tokenizer_worker_num": {"4", "8"},
	}
	for _, g := range []string{"deepseek-r1-0528", "glm-5"} {
		base := "mem_fraction_static: 0.82\n"
		want, err := lowerPool(t, g, 8, base)
		if err != nil {
			t.Fatal(err)
		}
		wantRes, err := Resolve(PoolInput{Deployment: pool(t, 8, base), Model: graph(t, g), Chip: h200(t)})
		if err != nil {
			t.Fatal(err)
		}
		for field, vals := range probes {
			for _, v := range vals {
				block := base + field + ": " + v + "\n"
				got, err := lowerPool(t, g, 8, block)
				if err != nil {
					t.Fatalf("%s %s=%s: %v", g, field, v, err)
				}
				if !reflect.DeepEqual(got, want) {
					t.Errorf("%s %s=%s changes the lowered plan", g, field, v)
				}
				res, err := Resolve(PoolInput{Deployment: pool(t, 8, block), Model: graph(t, g), Chip: h200(t)})
				if err != nil {
					t.Fatal(err)
				}
				for _, name := range config.FieldNames() {
					if name == field {
						continue
					}
					a, _ := res.Value(name)
					b, _ := wantRes.Value(name)
					if !reflect.DeepEqual(a, b) {
						t.Errorf("%s %s=%s changes the resolved %s: %v, was %v", g, field, v, name, a, b)
					}
				}
			}
		}
	}
}

// attention_backend nsa is SGLang's deprecated alias of dsa (layers/attention/attention_registry.py
// :166-176@297e54fd), which a GLM-5 pool defaults to: a pool that states nsa, or dsa, lowers to the
// plan of a pool that states neither, and a Note says which DSA kernels SGLang resolved and that
// the cost model's attention lane is a transfer to them.
func TestPoolLowersNSAAndDSAAsTheDefaultDSABackend(t *testing.T) {
	base, err := lowerPool(t, "glm-5", 8, "mem_fraction_static: 0.85\n")
	if err != nil {
		t.Fatal(err)
	}
	for _, backend := range []string{"nsa", "dsa"} {
		got, err := lowerPool(t, "glm-5", 8, "mem_fraction_static: 0.85\nattention_backend: "+backend+"\n")
		if err != nil {
			t.Fatalf("attention_backend %s: %v", backend, err)
		}
		if !reflect.DeepEqual(got.Plan, base.Plan) {
			t.Errorf("attention_backend %s lowers to a different plan than the default DSA backend", backend)
		}
	}
	// The plan states the backends SGLang resolved as an override: stated once, with the fact that the
	// price is a transfer from the vLLM-measured kernel, not a measurement of SGLang's.
	got := map[string]plan.Override{}
	for _, o := range base.Overrides {
		got[o.Field] = o
	}
	for field, want := range map[string]string{"dsa_prefill_backend": "flashmla_sparse", "dsa_decode_backend": "fa3"} {
		o, ok := got[field]
		if !ok || o.Requested != want {
			t.Fatalf("%s: override %+v, want the resolved backend %s on an H200 with a bf16 cache", field, o, want)
		}
		if !strings.Contains(o.Reason, "transfer") {
			t.Errorf("%s: the reason does not say the price is a transfer: %q", field, o.Reason)
		}
	}
	// A dense-MLA model (no sparse_mla layer) states no DSA kernels.
	dense, err := lowerPool(t, "deepseek-r1-0528", 8, "mem_fraction_static: 0.85\n")
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range dense.Overrides {
		if o.Field == "dsa_prefill_backend" || o.Field == "dsa_decode_backend" {
			t.Errorf("DeepSeek-R1 has no DSA layer, yet overrides %+v", o)
		}
	}
}
