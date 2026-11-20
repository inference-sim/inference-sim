package contracts

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
)

// This file is the coverage suite: it asserts that the schemas express the
// deployments the three design documents work through and the ones a published
// report corpus measured. A schema that validates its own examples proves little;
// these cases come from outside it.

func digest() string { return strings.Repeat("c", 64) }

// graniteMoE is the 230B preview: a uniform 72-layer MoE stack, 224 experts,
// GQA over 8 KV heads. It is NOT hybrid, which is why the stack has one layer kind.
func graniteMoE() *model.Graph {
	return &model.Graph{
		Kind: "ModelGraph", Name: "granite-5-230b",
		DerivedFrom: model.Derivation{Format: "hf_config_json",
			Path: "config.json", SHA256: digest(), DeriverVersion: 1},
		Global: model.GlobalShape{HiddenSize: 3072, VocabSize: 100352,
			WeightDType: model.DTypeFP8},
		LayerKinds: []model.LayerKind{{ID: "attn_moe", Nodes: []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			{Op: model.OpGEMM, Role: "qkv_proj", N: 4096, K: 3072},
			{Op: model.OpAttention, HoldsEngineKV: model.Bool(true), AttentionKind: model.AttentionGQA,
				NumQHeads: 48, NumKVHeads: 8, HeadDim: 64},
			{Op: model.OpGEMM, Role: "o_proj", N: 3072, K: 3072},
			{Op: model.OpAllReduce, Role: "attn_out", Emit: model.EmitTensorParallel},
			{Op: model.OpGroupedGEMM, Shared: model.Bool(false), Role: "experts", N: 1536, K: 3072,
				Experts: 224, TopK: 8},
			{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
			{Op: model.OpAllReduce, Role: "mlp_out", Emit: model.EmitTensorParallelUnlessSequenceParallelMoE},
		}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}}}},
		Stack: model.Stack{Pattern: []string{"attn_moe"}, Repeat: 72},
	}
}

// hybridMLAKDA is a Kimi-shaped stack: latent attention interleaved with linear
// attention, three recurrent layers to one attention layer.
func hybridMLAKDA() *model.Graph {
	attn := model.LayerKind{ID: "mla_moe", Nodes: []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		{Op: model.OpAttention, HoldsEngineKV: model.Bool(true), AttentionKind: model.AttentionMLA,
			NumQHeads: 64, NumKVHeads: 1, HeadDim: 576,
			KVLoRARank: 512, QKRopeHeadDim: 64},
		{Op: model.OpGroupedGEMM, Shared: model.Bool(false), Role: "experts", N: 2048, K: 7168,
			Experts: 256, TopK: 8},
		{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
	}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}}}
	kda := model.LayerKind{ID: "kda_moe", Nodes: []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentKDA,
			NumHeads: 64, StateSize: 128, StateDType: model.DTypeFP32},
		{Op: model.OpGroupedGEMM, Shared: model.Bool(false), Role: "experts", N: 2048, K: 7168,
			Experts: 256, TopK: 8},
		{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
	}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}}}
	return &model.Graph{
		Kind: "ModelGraph", Name: "kimi-k3",
		DerivedFrom: model.Derivation{Format: "hf_config_json",
			Path: "config.json", SHA256: digest(), DeriverVersion: 1},
		Global: model.GlobalShape{HiddenSize: 7168, VocabSize: 163840,
			WeightDType: model.DTypeFP8},
		LayerKinds: []model.LayerKind{kda, attn},
		Stack: model.Stack{
			Pattern: []string{"kda_moe", "kda_moe", "kda_moe", "mla_moe"},
			Repeat:  15},
		Speculator: &model.Speculator{Method: "kimi_k3_mtp", NumSpec: 1,
			Stack: model.Stack{Pattern: []string{"mla_moe"}, Repeat: 1}},
	}
}

// mambaHybrid is a Granite-hybrid-shaped stack: Mamba2 state-space layers, whose
// state is two tensors rather than one.
func mambaHybrid() *model.Graph {
	g := graniteMoE()
	g.Name = "granite-hybrid"
	g.LayerKinds = append(g.LayerKinds, model.LayerKind{
		ID: "mamba_moe", Nodes: []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2,
				NumHeads: 128, StateSize: 128, NumGroups: 8, ConvKernel: 4,
				IntermediateSize: 8192, StateDType: model.DTypeFP32},
			{Op: model.OpGroupedGEMM, Shared: model.Bool(false), Role: "experts", N: 1536, K: 3072,
				Experts: 224, TopK: 8},
			{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
		}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}}})
	g.Stack = model.Stack{
		Pattern: []string{"mamba_moe", "mamba_moe", "mamba_moe", "attn_moe"},
		Repeat:  18}
	return g
}

// TestModelShapesAreExpressible covers the architecture families the three design
// documents and the report corpus span: uniform MoE, latent-plus-linear hybrid, and
// state-space hybrid.
func TestModelShapesAreExpressible(t *testing.T) {
	for _, g := range []*model.Graph{graniteMoE(), hybridMLAKDA(), mambaHybrid()} {
		t.Run(g.Name, func(t *testing.T) {
			if p := g.Validate(); !p.OK() {
				t.Fatalf("%s is not expressible:\n%s", g.Name, p.Error())
			}
		})
	}
	if got := graniteMoE().Stack.Layers(); got != 72 {
		t.Errorf("granite layers = %d, want 72", got)
	}
	if got := hybridMLAKDA().Stack.Layers(); got != 60 {
		t.Errorf("kimi layers = %d, want 60", got)
	}
	if got := mambaHybrid().Stack.Layers(); got != 72 {
		t.Errorf("granite-hybrid layers = %d, want 72", got)
	}
}

// corpus builds a Scenario+Deployment pair from the parts a corpus report states: the
// immutable problem (model, available-hardware inventory, coefficient/engine refs) and
// the mutable layout (pools, offload, PD transfer) chosen against it.
func corpus(name string, nodes, gpusPerNode int, fabric string,
	pools []deployment.Pool,
	mutate func(*scenario.Scenario, *deployment.Deployment)) Bundle {
	s := &scenario.Scenario{
		Kind: "Scenario", Name: name, Model: "granite-5-230b",
		Coefficients:  []string{"cost-model-primitives-h200"},
		EngineVersion: "1.0.0",
		Cluster: scenario.Cluster{Hardware: "h200", Fabric: fabric,
			Nodes: nodes, GPUsPerNode: gpusPerNode},
	}
	d := &deployment.Deployment{Kind: "Deployment", Name: name, Pools: pools}
	if mutate != nil {
		mutate(s, d)
	}
	return Bundle{Scenario: s, Deployment: d, Rules: openAnyEngine}
}

func pool(role deployment.Role, nodes int, pl deployment.Parallelism,
	e deployment.EngineBlock) deployment.Pool {
	return deployment.Pool{Role: role, Nodes: nodes, Parallel: pl, Engine: e}
}

// TestCorpusDeploymentsAreExpressible walks the deployment shapes the published
// reports measured and the design documents work through. Each case names the knob
// it is testing, so a failure says which capability was lost.
func TestCorpusDeploymentsAreExpressible(t *testing.T) {
	tp8 := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	cases := []struct {
		what string
		b    Bundle
	}{
		{
			what: "single node, tp8, fp8 KV cache, large batched-token bound",
			b: corpus("granite-h100-tp8", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, engineBlock("{cache_dtype: fp8, block_size: 16, max_num_batched_tokens: 32768, max_num_seqs: 1024, gpu_memory_utilization: 0.9, cudagraph_mode: FULL_AND_PIECEWISE}")),
			}, nil),
		},
		{
			what: "CPU offload with a 1 TiB tier and a lazy connector",
			b: corpus("granite-h200-offload", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, engineBlock("{cache_dtype: fp8, block_size: 16, gpu_memory_utilization: 0.9}")),
			}, func(s *scenario.Scenario, d *deployment.Deployment) {
				d.Offload = &deployment.Offload{
					Connector: "OffloadingConnector", Spec: "CPUOffloadingSpec",
					EvictionPolicy: "lru", PrefetchDepth: 2,
					Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}}
			}),
		},
		{
			what: "multi-tier offload: CPU above NVMe",
			b: corpus("granite-h200-tiered", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, engineBlock("{cache_dtype: fp8, block_size: 16, gpu_memory_utilization: 0.9}")),
			}, func(s *scenario.Scenario, d *deployment.Deployment) {
				d.Offload = &deployment.Offload{Spec: "TieringOffloadingSpec",
					EvictionPolicy: "arc", PrefetchDepth: 3,
					Tiers: []deployment.Tier{
						{Device: "cpu_dram", Bytes: 512 << 30},
						{Device: "nvme_gen4", Bytes: 4 << 40}}}
			}),
		},
		{
			what: "prefill/decode disaggregation across 60 nodes, wide EP",
			b: corpus("glm-5.3-60node-pd", 60, 8, "ib-400g", []deployment.Pool{
				pool(deployment.RolePrefill, 36, deployment.Parallelism{TP: 1, PP: 1,
					DP: 8, DPLocal: 8, EnableExpertParallel: true},
					engineBlock("{all2all_backend: deepep_high_throughput, cudagraph_mode: PIECEWISE, cache_dtype: fp8, block_size: 64, gpu_memory_utilization: 0.9, speculative: {method: glm4_moe_mtp, num_spec_tokens: 1}}")),
				pool(deployment.RoleDecode, 24, deployment.Parallelism{TP: 1, PP: 1,
					DP: 16, DPLocal: 8, EnableExpertParallel: true},
					engineBlock("{all2all_backend: deepep_low_latency, cudagraph_mode: FULL_AND_PIECEWISE, cache_dtype: fp8, block_size: 64, gpu_memory_utilization: 0.9, dbo: {enable_dbo: true, dbo_decode_token_threshold: 32, dbo_prefill_token_threshold: 512}, eplb: {enable_eplb: true, num_redundant_experts: 32, window_size: 1000, step_interval: 3000}}")),
			}, func(s *scenario.Scenario, d *deployment.Deployment) {
				d.PDTransfer = &deployment.PDTransfer{Connector: "nixl"}
			}),
		},
		{
			what: "aggregated 32-GPU serving, the alternative PD is compared against",
			b: corpus("glm-5.3-flash-32gpu", 4, 8, "ib-400g", []deployment.Pool{
				pool(deployment.RoleColocated, 4, deployment.Parallelism{TP: 1, PP: 1,
					DP: 32, DPLocal: 8, EnableExpertParallel: true},
					engineBlock("{all2all_backend: deepep_low_latency, cache_dtype: fp8, block_size: 64, gpu_memory_utilization: 0.9}")),
			}, nil),
		},
		{
			what: "eager execution with the custom all-reduce disabled",
			b: corpus("nemotron-h100-eager", 2, 8, "roce-200g", []deployment.Pool{
				pool(deployment.RoleColocated, 2, deployment.Parallelism{TP: 8, PP: 1,
					DP: 2, DPLocal: 1, EnableExpertParallel: true},
					engineBlock("{cudagraph_mode: NONE, allreduce_backend: nccl, cache_dtype: auto, all2all_backend: deepep_high_throughput, gpu_memory_utilization: 0.9}")),
			}, nil),
		},
		{
			what: "hybrid stack: mamba cache mode and both state dtypes",
			b: corpus("granite-hybrid-h200", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, engineBlock("{cache_dtype: fp8, mamba_cache_dtype: auto, mamba_ssm_cache_dtype: auto, mamba_cache_mode: align, max_model_len: 131072, gpu_memory_utilization: 0.9}")),
			}, nil),
		},
		{
			what: "priority scheduling with async scheduling declined",
			b: corpus("granite-priority", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, engineBlock("{scheduling_policy: priority, async_scheduling: false, cache_dtype: fp8, gpu_memory_utilization: 0.9}")),
			}, nil),
		},
		{
			// From an internal llm-d deployment (a glm-5.3 H100 canary): a
			// LeaderWorkerSet of 4 pods, one vLLM per pod, prefill at DP 4 x PCP 8 with
			// DCP 8 at TP 1 — an expert group of 32 — and decode at DP 16 x TP 2. The
			// canary ran an engine that supports PCP with DP, which v0.29.0 does not, so
			// this case asserts field validation only: the v0.29 rule refuses it, and
			// should, for a scenario pinned to that release.
			what: "llm-d prefill-context parallelism with data parallelism",
			b: corpus("glm-5.3-pcp-canary", 8, 8, "ib-400g", []deployment.Pool{
				pool(deployment.RolePrefill, 4, deployment.Parallelism{TP: 1, PP: 1,
					DP: 4, DPLocal: 1, PCP: 8, DCP: 8, EnableExpertParallel: true},
					engineBlock("{all2all_backend: deepep_high_throughput, cache_dtype: fp8, block_size: 64, gpu_memory_utilization: 0.9, dcp_comm_backend: ag_rs}")),
				pool(deployment.RoleDecode, 4, deployment.Parallelism{TP: 2, PP: 1,
					DP: 16, DPLocal: 4, EnableExpertParallel: true},
					engineBlock("{all2all_backend: deepep_low_latency, cache_dtype: fp8, block_size: 64, gpu_memory_utilization: 0.9}")),
			}, func(s *scenario.Scenario, d *deployment.Deployment) {
				d.PDTransfer = &deployment.PDTransfer{Connector: "nixl"}
			}),
		},
		{
			what: "GB200-class three-tier topology",
			b: corpus("dsr1-gb200", 18, 4, "ib-400g", []deployment.Pool{
				pool(deployment.RoleColocated, 18, deployment.Parallelism{TP: 1, PP: 1,
					DP: 72, DPLocal: 4, EnableExpertParallel: true},
					engineBlock("{all2all_backend: deepep_low_latency, cache_dtype: fp8, gpu_memory_utilization: 0.9, eplb: {enable_eplb: true, num_redundant_experts: 32}}")),
			}, func(s *scenario.Scenario, d *deployment.Deployment) {
				s.Cluster.GPUsPerRack = 72
			}),
		},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			if rep := Validate(c.b); !rep.Field.OK() {
				t.Fatalf("not expressible:\n%s", rep.Field.Error())
			}
		})
	}
}

// anyEngine is an engine's rules that open any engine block: what the corpus cases need, whose
// engine blocks are written for a release this schema does not know. Field expressibility is
// the schema's to check; whether a release accepts the block is the release's rules'.
type anyEngine struct{ fakeRules }

func openAnyEngine(v string) rules.EngineRules { return anyEngine{fakeRules{version: v}} }

func (anyEngine) Check(rules.Input) *rules.Problems { return nil }
