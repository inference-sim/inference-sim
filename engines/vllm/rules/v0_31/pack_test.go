package v0_31

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

func graniteGraph() *model.Graph {
	return &model.Graph{
		Kind: "ModelGraph", Name: "granite-5-230b",
		DerivedFrom: model.Derivation{Format: "hf_config_json", Path: "config.json",
			SHA256:         strings.Repeat("a", 64),
			DeriverVersion: 1},
		Global: model.GlobalShape{HiddenSize: 3072, VocabSize: 100352,
			WeightDType: model.DTypeFP8},
		LayerKinds: []model.LayerKind{{ID: "attn_moe", Nodes: []model.Node{
			{Op: model.OpAttention, AttentionKind: model.AttentionGQA,
				NumQHeads: 48, NumKVHeads: 8, HeadDim: 64},
			{Op: model.OpGroupedGEMM, N: 1536, K: 3072, Experts: 224, TopK: 8},
		}}},
		Stack: model.Stack{Pattern: []string{"attn_moe"}, Repeat: 72},
	}
}

func dep(pools ...deployment.Pool) *deployment.Deployment {
	return &deployment.Deployment{Kind: "Deployment", Name: "t", Pools: pools}
}

func colocated(pl deployment.Parallelism, e deployment.Engine) deployment.Pool {
	return deployment.Pool{Role: deployment.RoleColocated, Nodes: 1, Parallel: pl, Engine: e}
}

// run fires the pack's rules over a deployment, pairing it with the scenario its pool
// node counts imply so the cluster-aware rules (custom-allreduce-reachable) have the
// inventory they read.
func run(t *testing.T, d *deployment.Deployment, g *model.Graph) *rules.Problems {
	t.Helper()
	nodes := 0
	for _, pl := range d.Pools {
		nodes += pl.Nodes
	}
	s := &scenario.Scenario{
		Kind: "Scenario", Name: "t", Model: "granite-5-230b",
		Coefficients: []string{"c"}, EngineVersion: Version,
		Cluster: scenario.Cluster{Hardware: "h200", Nodes: nodes, GPUsPerNode: 8},
	}
	p := &rules.Problems{}
	for _, r := range Pack().Rules {
		r.Check(rules.Input{Scenario: s, Deployment: d, Model: g}, p)
	}
	return p
}

func fired(p *rules.Problems, rule string) bool {
	for _, it := range p.All() {
		if it.Rule == rule {
			return true
		}
	}
	return false
}

// TestPackRuleSet pins the rule names. A rule silently dropped during a refactor
// would otherwise leave a passing suite and no coverage.
func TestPackRuleSet(t *testing.T) {
	want := []string{
		"all2all-backend-known",
		"allreduce-backend-known",
		"cache-record-dtypes-known",
		"cascade-attention-is-opt-in",
		"connector-known",
		"cp-kv-cache-interleave-fits-block",
		"cudagraph-capture-sizes",
		"custom-allreduce-reachable",
		"data-parallel-load-balancing",
		"dbo-thresholds-at-least-microbatches",
		"dbo-thresholds-stated-when-enabled",
		"dcp-comm-backend-known",
		"deepseek-v41-refuses-mtp",
		"engram-config-valid",
		"engram-excludes-dbo",
		"engram-residency-default",
		"enum-values-known",
		"eviction-policy-known",
		"expert-divisibility-under-eplb",
		"expert-imbalance-without-eplb",
		"long-prefill-threshold-within-max-model-len",
		"mamba-cache-mode-matches-model",
		"max-model-len-within-context-window",
		"model-sequence-parallel-collectives",
		"moonep-requirements",
		"offload-spec-known",
		"quantization-known",
		"sequence-parallel-architecture-known",
		"sequence-parallel-moe-implied",
		"speculative-method-known",
	}
	got := Pack().Names()
	if len(got) != len(want) {
		t.Fatalf("rule count = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("rule[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestEveryRuleStatesItsReason: a rule a reader cannot evaluate is a rule they will
// waive blindly.
func TestEveryRuleStatesItsReason(t *testing.T) {
	for _, r := range Pack().Rules {
		if r.Because == "" {
			t.Errorf("rule %q states no reason", r.Name)
		}
		if r.Check == nil {
			t.Errorf("rule %q has no check", r.Name)
		}
	}
}

func TestUnknownBackendsRejected(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{All2AllBackend: "telepathy", AllReduceBackend: "osmosis"}))
	p := run(t, s, graniteGraph())
	for _, rule := range []string{"all2all-backend-known", "allreduce-backend-known"} {
		if !fired(p, rule) {
			t.Errorf("%s did not fire on an unknown backend", rule)
		}
	}
}

func TestKnownBackendsAccepted(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{All2AllBackend: "allgather_reducescatter",
			AllReduceBackend: "nccl", CUDAGraphMode: "FULL",
			CacheDType: "fp8", SchedulingPolicy: "priority"}))
	if p := run(t, s, graniteGraph()); !p.OK() {
		t.Fatalf("accepted values were rejected:\n%s", p.Error())
	}
}

// TestExpertDivisibility is the Granite case: 224 experts do not divide an
// expert-parallel width of 72, and the rule must say how many redundant experts
// would fix it.
func TestExpertDivisibility(t *testing.T) {
	pl := deployment.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}
	// With load balancing on, an indivisible layout is an error.
	s := dep(colocated(pl, deployment.Engine{
		EPLB: &deployment.EPLB{Enabled: true, NumRedundantExperts: 32}}))
	p := run(t, s, graniteGraph())
	if !fired(p, "expert-divisibility-under-eplb") {
		t.Fatal("divisibility rule did not fire on 224+32 over 72")
	}
	if !strings.Contains(p.Error(), "32 more redundant") {
		t.Errorf("message should name the shortfall; got:\n%s", p.Error())
	}
	// 224 + 64 = 288 = 4 x 72.
	ok := dep(colocated(pl, deployment.Engine{
		EPLB: &deployment.EPLB{Enabled: true, NumRedundantExperts: 64}}))
	if p := run(t, ok, graniteGraph()); fired(p, "expert-divisibility-under-eplb") {
		t.Errorf("divisibility rule fired on a divisible layout:\n%s", p.Error())
	}
}

// TestExpertImbalanceWithoutEPLB: the same layout is legal without load balancing,
// and carries a static imbalance the estimate must account for.
func TestExpertImbalanceWithoutEPLB(t *testing.T) {
	pl := deployment.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}
	s := dep(colocated(pl, deployment.Engine{}))
	p := run(t, s, graniteGraph())
	if !fired(p, "expert-imbalance-without-eplb") {
		t.Fatal("imbalance rule did not fire on 224 over 72")
	}
	// Legal, so a warning rather than an error.
	if !p.OK() {
		t.Errorf("an indivisible split without EPLB should warn, not fail:\n%s", p.Error())
	}
	if !strings.Contains(p.Error(), "1.33x") {
		t.Errorf("message should quantify the imbalance; got:\n%s", p.Error())
	}
}

// TestExpertParallelWidthExceedsExperts is an error rather than a warning: a rank
// with no expert cannot run the layer at all.
func TestExpertParallelWidthExceedsExperts(t *testing.T) {
	pl := deployment.Parallelism{TP: 1, PP: 1, DP: 256, EnableExpertParallel: true}
	s := dep(colocated(pl, deployment.Engine{}))
	p := run(t, s, graniteGraph())
	if p.OK() {
		t.Fatal("expected an error when width exceeds the expert count")
	}
}

func TestSequenceParallelMoEImplied(t *testing.T) {
	// The default backend with both tp and dp above one implies sequence-parallel
	// MoE, which changes which collectives the graph must emit.
	pl := deployment.Parallelism{TP: 2, PP: 1, DP: 8, EnableExpertParallel: true}
	s := dep(colocated(pl, deployment.Engine{All2AllBackend: "allgather_reducescatter"}))
	if p := run(t, s, graniteGraph()); !fired(p, "sequence-parallel-moe-implied") {
		t.Fatal("rule did not fire for tp=2, dp=8 on the default backend")
	}
	// tp=1 does not trigger it, which is why the design's own examples do not.
	pl1 := deployment.Parallelism{TP: 1, PP: 1, DP: 8, EnableExpertParallel: true}
	s1 := dep(colocated(pl1, deployment.Engine{All2AllBackend: "allgather_reducescatter"}))
	if p := run(t, s1, graniteGraph()); fired(p, "sequence-parallel-moe-implied") {
		t.Error("rule fired at tp=1, where the engine does not apply it")
	}
}

func TestCustomAllReduceReachability(t *testing.T) {
	// A world size outside the supported set falls back.
	s := dep(colocated(deployment.Parallelism{TP: 3, PP: 1, DP: 1},
		deployment.Engine{AllReduceBackend: "custom"}))
	if p := run(t, s, graniteGraph()); !fired(p, "custom-allreduce-reachable") {
		t.Error("rule did not fire on tp=3")
	}
	// Spanning nodes needs multi-node NVLink.
	s2 := dep(
		colocated(deployment.Parallelism{TP: 16, PP: 1, DP: 1},
			deployment.Engine{AllReduceBackend: "custom"}),
	)
	if p := run(t, s2, graniteGraph()); !fired(p, "custom-allreduce-reachable") {
		t.Error("rule did not fire on tp=16 across 8-GPU nodes")
	}
	// tp=8 inside one node is reachable.
	s3 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{AllReduceBackend: "custom"}))
	if p := run(t, s3, graniteGraph()); fired(p, "custom-allreduce-reachable") {
		t.Errorf("rule fired on a reachable layout:\n%s", p.Error())
	}
}

func TestMambaCacheModeMatchesModel(t *testing.T) {
	// A mode on a model with no recurrent layer has no effect.
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{MambaCacheMode: "align"}))
	if p := run(t, s, graniteGraph()); !fired(p, "mamba-cache-mode-matches-model") {
		t.Error("rule did not fire for a mode on a non-hybrid model")
	}
	// A hybrid model with no mode leaves the state unsized.
	hybrid := graniteGraph()
	hybrid.LayerKinds[0].Nodes = append(hybrid.LayerKinds[0].Nodes, model.Node{
		Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2,
		NumHeads: 128, StateSize: 128, NumGroups: 8, ConvKernel: 4,
		IntermediateSize: 8192})
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	if p := run(t, s2, hybrid); !fired(p, "mamba-cache-mode-matches-model") {
		t.Error("rule did not fire for a hybrid model with no mode")
	}
}

func TestOffloadSpecKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	s.Offload = &deployment.Offload{Spec: "MagicOffloadingSpec",
		Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}}
	if p := run(t, s, graniteGraph()); !fired(p, "offload-spec-known") {
		t.Fatal("rule did not fire on an unregistered spec")
	}
	s.Offload.Spec = "CPUOffloadingSpec"
	if p := run(t, s, graniteGraph()); fired(p, "offload-spec-known") {
		t.Error("rule fired on a registered spec")
	}
}

func TestDBOThresholdsStatedWhenEnabled(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{DBO: &deployment.DBO{Enabled: true}}))
	p := run(t, s, graniteGraph())
	if !fired(p, "dbo-thresholds-stated-when-enabled") {
		t.Fatal("rule did not fire when thresholds were omitted")
	}
	if !strings.Contains(p.Error(), "32") || !strings.Contains(p.Error(), "512") {
		t.Errorf("message should name both defaults; got:\n%s", p.Error())
	}
}

func TestSpeculativeMethodKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{Speculative: &deployment.Speculative{
			Method: "crystal_ball", NumSpecTokens: 2}}))
	if p := run(t, s, graniteGraph()); !fired(p, "speculative-method-known") {
		t.Fatal("rule did not fire on an unknown method")
	}
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{Speculative: &deployment.Speculative{
			Method: "glm4_moe_mtp", NumSpecTokens: 1}}))
	if p := run(t, s2, graniteGraph()); fired(p, "speculative-method-known") {
		t.Error("rule fired on an accepted method")
	}
}

// TestQuantizationKnown: an in-tree method passes, a typo warns. It is a warning rather
// than an error because the engine admits out-of-tree methods the pack cannot enumerate,
// so the rule flags a probable typo without rejecting a possibly-valid name.
func TestQuantizationKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{Quantization: "fp8_but_misspelled"}))
	p := run(t, s, graniteGraph())
	if !fired(p, "quantization-known") {
		t.Fatal("rule did not fire on an unrecognized quantization method")
	}
	// The warning must enumerate the valid set (#19 item 3): naming an in-tree method the
	// author could have meant is what makes the diagnostic self-correcting.
	if !strings.Contains(p.Error(), "compressed-tensors") {
		t.Errorf("the warning should enumerate in-tree methods; got:\n%s", p.Error())
	}
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{Quantization: "compressed-tensors"}))
	if p := run(t, s2, graniteGraph()); fired(p, "quantization-known") {
		t.Error("rule fired on an in-tree quantization method")
	}
}

// TestConnectorKnown checks both connector sites — the offload block and the pd_transfer
// block — since a deployment can name a connector in either. An unknown name warns; a
// registered one is quiet.
func TestConnectorKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	s.Offload = &deployment.Offload{Connector: "TelepathyConnector",
		Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}}
	s.PDTransfer = &deployment.PDTransfer{Connector: "NixlConnector"}
	p := run(t, s, graniteGraph())
	if !fired(p, "connector-known") {
		t.Fatal("rule did not fire on an unregistered offload connector")
	}
	if !strings.Contains(p.Error(), "offload.connector") {
		t.Errorf("the warning should name the offload connector site; got:\n%s", p.Error())
	}
	// The warning enumerates the registered connectors (#19 item 3).
	if !strings.Contains(p.Error(), "NixlConnector") {
		t.Errorf("the warning should enumerate registered connectors; got:\n%s", p.Error())
	}

	// Both connectors registered: quiet.
	s.Offload.Connector = "OffloadingConnector"
	if p := run(t, s, graniteGraph()); fired(p, "connector-known") {
		t.Errorf("rule fired on two registered connectors:\n%s", p.Error())
	}

	// A bad pd_transfer connector is caught on its own.
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	s2.PDTransfer = &deployment.PDTransfer{Connector: "CarrierPigeonConnector"}
	if p := run(t, s2, graniteGraph()); !fired(p, "connector-known") {
		t.Fatal("rule did not fire on an unregistered pd_transfer connector")
	}
}

// TestEvictionPolicyKnown: an in-tree policy (lru/arc) passes, a typo warns and the warning
// enumerates the in-tree set. A warning, not an error, because the CachePolicyFactory admits
// out-of-tree policies — the same shape as the connector and quantization rules (#19).
func TestEvictionPolicyKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	s.Offload = &deployment.Offload{EvictionPolicy: "most_recently_used_typo",
		Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}}
	p := run(t, s, graniteGraph())
	if !fired(p, "eviction-policy-known") {
		t.Fatal("rule did not fire on an unrecognized eviction policy")
	}
	if !strings.Contains(p.Error(), "lru") || !strings.Contains(p.Error(), "arc") {
		t.Errorf("the warning should enumerate in-tree policies lru and arc; got:\n%s", p.Error())
	}

	// An in-tree policy is quiet.
	s.Offload.EvictionPolicy = "arc"
	if p := run(t, s, graniteGraph()); fired(p, "eviction-policy-known") {
		t.Errorf("rule fired on an in-tree policy:\n%s", p.Error())
	}
}

// TestPCPWithDataParallelismIsAccepted: v0.31.0 dropped v0.29.0's startup refusal of
// prefill-context parallelism combined with data parallelism, so no rule may report it.
func TestPCPWithDataParallelismIsAccepted(t *testing.T) {
	p := run(t, dep(colocated(deployment.Parallelism{TP: 1, PP: 1, DP: 4, PCP: 8},
		deployment.Engine{})), nil)
	if !p.OK() || len(p.All()) != 0 {
		t.Fatalf("DP 4 x PCP 8 should draw no finding under 0.31.0:\n%s", p.Error())
	}
}

// TestDCPCommBackendKnown: the engine types the setting as a closed literal, so a name
// outside it does not start. Unset is the engine default and is fine.
func TestDCPCommBackendKnown(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1, DCP: 8}
	for _, c := range []struct {
		backend string
		want    bool
	}{{"", false}, {"ag_rs", false}, {"a2a", false}, {"allgather", true}, {"A2A", true}} {
		p := run(t, dep(colocated(pl, deployment.Engine{DCPCommBackend: c.backend})), nil)
		if got := fired(p, "dcp-comm-backend-known"); got != c.want {
			t.Errorf("dcp_comm_backend %q: rule fired=%v, want %v\n%s", c.backend, got, c.want, p.Error())
		}
		if c.want && !strings.Contains(p.Error(), "a2a, ag_rs") {
			t.Errorf("message should name the accepted set: %s", p.Error())
		}
	}
}

// hybridGraph is graniteGraph with one recurrent layer, which lets the platform re-align
// a stated block size.
func hybridGraph() *model.Graph {
	g := graniteGraph()
	g.LayerKinds[0].Nodes = append(g.LayerKinds[0].Nodes,
		model.Node{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2})
	return g
}

// TestCPKVCacheInterleaveFitsBlock: with DCP on and no NixlConnector, this release asserts
// at startup that block_size is at least, and divisible by, the interleave size. The rule
// fires only where that assert is certain to run on the values stated.
func TestCPKVCacheInterleaveFitsBlock(t *testing.T) {
	pool := func(dcp, size, block int) deployment.Pool {
		return colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1, DCP: dcp},
			deployment.Engine{CPKVCacheInterleaveSize: size, BlockSize: block})
	}
	cases := []struct {
		name string
		d    *deployment.Deployment
		g    *model.Graph
		want bool
	}{
		{"interleave above block", dep(pool(8, 128, 64)), graniteGraph(), true},
		{"interleave does not divide block", dep(pool(8, 24, 64)), graniteGraph(), true},
		{"interleave divides block", dep(pool(8, 16, 64)), graniteGraph(), false},
		{"interleave equals block", dep(pool(8, 64, 64)), graniteGraph(), false},
		// Where the assert does not run, or does not run on these values.
		{"dcp off", dep(pool(1, 128, 64)), graniteGraph(), false},
		{"interleave unstated", dep(pool(8, 0, 64)), graniteGraph(), false},
		{"block size unstated", dep(pool(8, 128, 0)), graniteGraph(), false},
		{"model unknown", dep(pool(8, 128, 64)), nil, false},
		{"hybrid model", dep(pool(8, 128, 64)), hybridGraph(), false},
		// Only NIXL skips the assert in this release; a MultiConnector may hide one.
		{"nixl pd transfer", func() *deployment.Deployment {
			d := dep(pool(8, 128, 64))
			d.PDTransfer = &deployment.PDTransfer{Connector: "NixlConnector"}
			return d
		}(), graniteGraph(), false},
		{"multi connector", func() *deployment.Deployment {
			d := dep(pool(8, 128, 64))
			d.PDTransfer = &deployment.PDTransfer{Connector: "MultiConnector"}
			return d
		}(), graniteGraph(), false},
		// Unlike v0.29.0, any other connector leaves the assert in force.
		{"mooncake pd transfer", func() *deployment.Deployment {
			d := dep(pool(8, 128, 64))
			d.PDTransfer = &deployment.PDTransfer{Connector: "MooncakeConnector"}
			return d
		}(), graniteGraph(), true},
		{"offload connector", func() *deployment.Deployment {
			d := dep(pool(8, 128, 64))
			d.Offload = &deployment.Offload{Connector: "OffloadingConnector",
				Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1}}}
			return d
		}(), graniteGraph(), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := run(t, c.d, c.g)
			if got := fired(p, "cp-kv-cache-interleave-fits-block"); got != c.want {
				t.Fatalf("rule fired=%v, want %v\n%s", got, c.want, p.Error())
			}
			if c.want && !strings.Contains(p.Error(), "block_size 64") {
				t.Errorf("message should name the block size: %s", p.Error())
			}
		})
	}
}

func TestDBOThresholdsAtLeastMicrobatches(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	for _, c := range []struct {
		name string
		dbo  *deployment.DBO
		want bool
	}{
		{"off", &deployment.DBO{Enabled: false, DecodeTokenThreshold: intp(1)}, false},
		{"defaults", &deployment.DBO{Enabled: true}, false},
		{"two", &deployment.DBO{Enabled: true, DecodeTokenThreshold: intp(2), PrefillTokenThreshold: intp(2)}, false},
		{"decode one", &deployment.DBO{Enabled: true, DecodeTokenThreshold: intp(1)}, true},
		{"prefill one", &deployment.DBO{Enabled: true, PrefillTokenThreshold: intp(1)}, true},
		// A stated 0 is not the default: the engine refuses it once DBO is on.
		{"decode zero", &deployment.DBO{Enabled: true, DecodeTokenThreshold: intp(0)}, true},
	} {
		p := run(t, dep(colocated(pl, deployment.Engine{DBO: c.dbo})), graniteGraph())
		if got := fired(p, "dbo-thresholds-at-least-microbatches"); got != c.want {
			t.Errorf("%s: rule fired=%v, want %v\n%s", c.name, got, c.want, p.Error())
		}
	}
}

// engramGraph is graniteGraph with an Engram lookup, which is how a graph says it is a
// DeepSeek-V4.1 with engram layers.
func engramGraph() *model.Graph {
	g := graniteGraph()
	g.Global.Architecture = rules.ArchDeepseekV41
	g.LayerKinds[0].Nodes = append(g.LayerKinds[0].Nodes, model.Node{
		Op: model.OpGather, Role: "engram_lookup", Rows: 384006168, RowBytes: 264,
		RowsPerToken: 24, OutBytesPerToken: 12288, Residency: model.ResidencyHost,
		ResidencyControl: model.ResidencyEngramCPUOffload, Shard: model.ShardHeads})
	return g
}

func TestEngramExcludesDBO(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	on := deployment.Engine{DBO: &deployment.DBO{Enabled: true}}
	if p := run(t, dep(colocated(pl, on)), engramGraph()); !fired(p, "engram-excludes-dbo") {
		t.Errorf("DBO beside Engram should be refused:\n%s", p.Error())
	}
	if p := run(t, dep(colocated(pl, on)), graniteGraph()); fired(p, "engram-excludes-dbo") {
		t.Errorf("DBO without Engram is allowed:\n%s", p.Error())
	}
	if p := run(t, dep(colocated(pl, deployment.Engine{})), engramGraph()); fired(p, "engram-excludes-dbo") {
		t.Errorf("Engram without DBO is allowed:\n%s", p.Error())
	}
	// The engine's test is the architecture: another model EngramConfig governs runs DBO.
	other := engramGraph()
	other.Global.Architecture = "Qwen4ExpForCausalLM"
	if p := run(t, dep(colocated(pl, on)), other); fired(p, "engram-excludes-dbo") {
		t.Errorf("DBO refused for an architecture the engine allows it on:\n%s", p.Error())
	}
	// An Engram graph that names no architecture cannot be decided, so it warns.
	anon := engramGraph()
	anon.Global.Architecture = ""
	p := run(t, dep(colocated(pl, on)), anon)
	if !fired(p, "engram-excludes-dbo") || !p.OK() {
		t.Errorf("want a warning, not an error, without an architecture:\n%s", p.Error())
	}
}

func TestModelSequenceParallel(t *testing.T) {
	p := Pack()
	v41 := rules.ArchDeepseekV41
	for _, c := range []struct {
		name string
		arch string
		pl   deployment.Parallelism
		moe  string
		want bool
	}{
		{"dp above one", v41, deployment.Parallelism{TP: 4, PP: 1, DP: 2, EnableExpertParallel: true}, "", true},
		{"mega-MoE at dp one", v41, deployment.Parallelism{TP: 8, PP: 1, DP: 1, EnableExpertParallel: true}, "deep_gemm_mega_moe", true},
		{"auto at dp one", v41, deployment.Parallelism{TP: 8, PP: 1, DP: 1, EnableExpertParallel: true}, "", false},
		{"no expert parallelism", v41, deployment.Parallelism{TP: 4, PP: 1, DP: 2}, "", false},
		{"tp one", v41, deployment.Parallelism{TP: 1, PP: 1, DP: 8, EnableExpertParallel: true}, "", false},
		{"pipeline", v41, deployment.Parallelism{TP: 4, PP: 2, DP: 2, EnableExpertParallel: true}, "", false},
		{"other architecture", "DeepseekV3ForCausalLM", deployment.Parallelism{TP: 4, PP: 1, DP: 2, EnableExpertParallel: true}, "", false},
	} {
		if got := p.ModelSequenceParallel(c.arch, c.pl, c.moe); got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}

// A graph asks about the model's sequence parallelism through a collective's emit
// condition or a node's split under the mode. Either one, on an architecture with no rule,
// is a question nothing answers.
func TestSequenceParallelArchitectureKnown(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	for name, n := range map[string]model.Node{
		"emit":                    {Op: model.OpAllGather, Emit: model.EmitSequenceParallel},
		"under_sequence_parallel": {Op: model.OpElementwise, Role: "mhc_pre", TP: model.TPReplicated, UnderSequenceParallel: model.SPTokenSharded},
	} {
		g := graniteGraph()
		g.LayerKinds[0].Nodes = append(g.LayerKinds[0].Nodes, n)
		g.Global.Architecture = "GraniteMoeForCausalLM"
		p := run(t, dep(colocated(pl, deployment.Engine{})), g)
		if !fired(p, "sequence-parallel-architecture-known") || !strings.Contains(p.Error(), "."+name+": ") {
			t.Errorf("%s: an architecture with no rule should warn, naming the field:\n%s", name, p.Error())
		}
		g.Global.Architecture = rules.ArchDeepseekV41
		if p := run(t, dep(colocated(pl, deployment.Engine{})), g); fired(p, "sequence-parallel-architecture-known") {
			t.Errorf("%s: V4.1 has a rule in this release:\n%s", name, p.Error())
		}
	}
}

func TestEnumValuesForMambaAndMoE(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	for _, c := range []struct {
		e    deployment.Engine
		want bool
	}{
		{deployment.Engine{MambaSSMCacheDType: "float32"}, false},
		{deployment.Engine{MambaCacheDType: "fp8"}, true},
		{deployment.Engine{MoEBackend: "deep_gemm_mega_moe"}, false},
		{deployment.Engine{MoEBackend: "rdna3"}, false},
		{deployment.Engine{MoEBackend: "megamoe"}, true},
	} {
		if got := fired(run(t, dep(colocated(pl, c.e)), graniteGraph()), "enum-values-known"); got != c.want {
			t.Errorf("%+v: rule fired=%v, want %v", c.e, got, c.want)
		}
	}
}

func TestSequenceParallelMoEOnDefaultBackend(t *testing.T) {
	pl := deployment.Parallelism{TP: 4, PP: 1, DP: 2, EnableExpertParallel: true}
	if p := run(t, dep(colocated(pl, deployment.Engine{})), graniteGraph()); !fired(p, "sequence-parallel-moe-implied") {
		t.Errorf("the default backend, allgather_reducescatter, is in the subset:\n%s", p.Error())
	}
}

func TestMoonEPRequirements(t *testing.T) {
	pl := deployment.Parallelism{TP: 1, PP: 1, DP: 8, EnableExpertParallel: true}
	bf16 := graniteGraph()
	bf16.Global.WeightDType = model.DTypeBF16
	mixed := graniteGraph()
	mixed.Global.WeightDType = model.DTypeBF16
	mixed.LayerKinds[0].Nodes[1].WeightDType = model.DTypeMXFP4
	moon := deployment.Engine{All2AllBackend: "moonep"}
	for _, c := range []struct {
		name string
		e    deployment.Engine
		g    *model.Graph
		want bool
	}{
		{"bf16", moon, bf16, false},
		{"fp16", moon, func() *model.Graph { g := graniteGraph(); g.Global.WeightDType = model.DTypeFP16; return g }(), true},
		{"fp8 checkpoint", moon, graniteGraph(), true},
		{"mxfp4 experts", moon, mixed, true},
		{"served quantized", deployment.Engine{All2AllBackend: "moonep", Quantization: "fp8"}, bf16, true},
		{"eplb", deployment.Engine{All2AllBackend: "moonep",
			EPLB: &deployment.EPLB{Enabled: true}}, bf16, true},
		{"other backend", deployment.Engine{All2AllBackend: "deepep_low_latency"}, graniteGraph(), false},
	} {
		p := run(t, dep(colocated(pl, c.e)), c.g)
		if got := fired(p, "moonep-requirements"); got != c.want {
			t.Errorf("%s: rule fired=%v, want %v\n%s", c.name, got, c.want, p.Error())
		}
	}
}

func TestEngramConfigValid(t *testing.T) {
	yes, no := true, false
	cases := []struct {
		name string
		dp   int
		ec   *deployment.EngramConfig
		g    *model.Graph
		want bool
	}{
		{"defaults", 1, &deployment.EngramConfig{}, engramGraph(), false},
		{"hbm", 1, &deployment.EngramConfig{CPUOffload: &no}, engramGraph(), false},
		{"shared with dp", 4, &deployment.EngramConfig{DPSharedMemory: &yes}, engramGraph(), false},
		{"model without engram", 1, &deployment.EngramConfig{}, graniteGraph(), true},
		{"shared without offload", 4, &deployment.EngramConfig{CPUOffload: &no, DPSharedMemory: &yes}, engramGraph(), true},
		{"shared with one rank", 1, &deployment.EngramConfig{DPSharedMemory: &yes}, engramGraph(), true},
		{"thp per rank", 1, &deployment.EngramConfig{UseTHP: &yes}, engramGraph(), false},
		{"thp shared", 4, &deployment.EngramConfig{UseTHP: &yes, DPSharedMemory: &yes}, engramGraph(), true},
		{"thp in hbm", 1, &deployment.EngramConfig{UseTHP: &yes, CPUOffload: &no}, engramGraph(), true},
	}
	for _, c := range cases {
		pl := deployment.Parallelism{TP: 2, PP: 1, DP: c.dp}
		p := run(t, dep(colocated(pl, deployment.Engine{EngramConfig: c.ec})), c.g)
		if got := fired(p, "engram-config-valid"); got != c.want {
			t.Errorf("%s: rule fired=%v, want %v\n%s", c.name, got, c.want, p.Error())
		}
	}
}

func TestMaxModelLenWithinContextWindow(t *testing.T) {
	g := graniteGraph()
	g.Global.ContextWindow = 131072
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	for _, c := range []struct {
		m    deployment.MaxModelLen
		want bool
	}{{0, false}, {deployment.MaxModelLenAuto, false}, {131072, false}, {131073, true}} {
		p := run(t, dep(colocated(pl, deployment.Engine{MaxModelLen: c.m})), g)
		if got := fired(p, "max-model-len-within-context-window"); got != c.want {
			t.Errorf("max_model_len %d: rule fired=%v, want %v\n%s", c.m, got, c.want, p.Error())
		}
	}
	// A graph that states no window gives the rule nothing to compare.
	if p := run(t, dep(colocated(pl, deployment.Engine{MaxModelLen: 1 << 30})), graniteGraph()); fired(p, "max-model-len-within-context-window") {
		t.Errorf("fired without a window:\n%s", p.Error())
	}
}

func intp(v int) *int { return &v }

// layoutGrid is every combination of the widths and flags the sequence-parallel decisions
// read, unset widths included.
func layoutGrid() []deployment.Parallelism {
	var out []deployment.Parallelism
	for _, tp := range []int{0, 1, 2, 4, 8} {
		for _, pp := range []int{0, 1, 2} {
			for _, dp := range []int{0, 1, 2, 8} {
				for _, ep := range []bool{false, true} {
					out = append(out, deployment.Parallelism{TP: tp, PP: pp, DP: dp, EnableExpertParallel: ep})
				}
			}
		}
	}
	return out
}

// Over every layout and backend: an unstated backend behaves exactly as the default stated
// does, and the sequence-parallel-moe-implied rule fires exactly when SequenceParallelMoE
// says the engine makes the MoE input sequence-parallel, so a cost model reading the
// method and a reader seeing the rule never get different answers.
func TestSequenceParallelMoEAgreesWithItsRule(t *testing.T) {
	p := Pack()
	backends := append(sortedKeys(p.All2AllBackends), "")
	for _, pl := range layoutGrid() {
		if p.SequenceParallelMoE("", pl.EnableExpertParallel, pl.TP, pl.DP) !=
			p.SequenceParallelMoE(p.DefaultAll2AllBackend, pl.EnableExpertParallel, pl.TP, pl.DP) {
			t.Fatalf("%+v: an unstated backend differs from the default %q", pl, p.DefaultAll2AllBackend)
		}
		for _, b := range backends {
			pl := pl
			if pl.TP < 1 || pl.PP < 1 || pl.DP < 1 {
				continue // a field error, which stops rules before they run
			}
			want := p.SequenceParallelMoE(b, pl.EnableExpertParallel, pl.TP, pl.DP)
			got := fired(run(t, dep(colocated(pl, deployment.Engine{All2AllBackend: b})), graniteGraph()),
				"sequence-parallel-moe-implied")
			if got != want {
				t.Fatalf("%+v backend %q: rule fired=%v, method says %v", pl, b, got, want)
			}
		}
	}
}

// The properties DeepSeek-V4.1's own rule must have, checked over every layout and MoE
// backend: it needs pipeline parallelism of one, expert parallelism and more than one
// tensor-parallel rank; with those, more than one data-parallel rank turns it on whatever
// the backend, and at one rank only a mega-MoE backend does; and raising dp never turns it
// off. An architecture without a rule is never on.
func TestModelSequenceParallelProperties(t *testing.T) {
	p := Pack()
	v41 := rules.ArchDeepseekV41
	moes := append(sortedKeys(p.MoEBackends), "")
	for _, pl := range layoutGrid() {
		for _, moe := range moes {
			on := p.ModelSequenceParallel(v41, pl, moe)
			base := pl.PP <= 1 && pl.EnableExpertParallel && pl.TP > 1
			switch {
			case !base && on:
				t.Fatalf("%+v %q: on without pp 1, expert parallelism and tp above 1", pl, moe)
			case base && pl.DP > 1 && !on:
				t.Fatalf("%+v %q: off with dp above 1", pl, moe)
			case base && pl.DP <= 1 && on != p.MegaMoEBackends[moe]:
				t.Fatalf("%+v %q: at dp 1 it should follow the mega-MoE backend", pl, moe)
			}
			wider := pl
			wider.DP = 8
			if on && !p.ModelSequenceParallel(v41, wider, moe) {
				t.Fatalf("%+v %q: raising dp turned it off", pl, moe)
			}
			if p.ModelSequenceParallel("DeepseekV3ForCausalLM", pl, moe) {
				t.Fatalf("an architecture with no rule is on: %+v", pl)
			}
		}
	}
}

// Every MTP-family method is refused for DeepSeek-V4.1, through the deployment or the
// graph, and accepted for an architecture the engine serves MTP on; dspark is accepted.
func TestDeepseekV41RefusesMTP(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	v41 := graniteGraph()
	v41.Global.Architecture = rules.ArchDeepseekV41
	v3 := graniteGraph()
	v3.Global.Architecture = "DeepseekV3ForCausalLM"
	for _, m := range mtpMethods {
		e := deployment.Engine{Speculative: &deployment.Speculative{Method: m, NumSpecTokens: 1}}
		if p := run(t, dep(colocated(pl, e)), v41); !fired(p, "deepseek-v41-refuses-mtp") {
			t.Errorf("%s on V4.1 accepted", m)
		}
		if p := run(t, dep(colocated(pl, e)), v3); fired(p, "deepseek-v41-refuses-mtp") {
			t.Errorf("%s refused for another architecture", m)
		}
	}
	e := deployment.Engine{Speculative: &deployment.Speculative{Method: "dspark", NumSpecTokens: 5}}
	if p := run(t, dep(colocated(pl, e)), v41); fired(p, "deepseek-v41-refuses-mtp") {
		t.Errorf("dspark refused on V4.1:\n%s", p.Error())
	}
	g := graniteGraph()
	g.Global.Architecture = rules.ArchDeepseekV41
	g.Speculator = &model.Speculator{Method: "mtp", NumSpec: 1, Stack: g.Stack}
	if p := run(t, dep(colocated(pl, deployment.Engine{})), g); !fired(p, "deepseek-v41-refuses-mtp") {
		t.Errorf("an MTP speculator in the graph accepted on V4.1:\n%s", p.Error())
	}
}

// The Engram config is accepted only for the architectures the engine lists, and a graph
// that names none draws a warning rather than a pass or an error.
func TestEngramConfigArchitectures(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	e := deployment.Engine{EngramConfig: &deployment.EngramConfig{}}
	for arch, want := range map[string]string{
		rules.ArchDeepseekV41: "", "Qwen4ExpForCausalLM": "", "GraniteMoeForCausalLM": "error", "": "warning",
	} {
		g := engramGraph()
		g.Global.Architecture = arch
		p := run(t, dep(colocated(pl, e)), g)
		got := ""
		for _, pr := range p.All() {
			if pr.Rule == "engram-config-valid" {
				got = string(pr.Severity)
			}
		}
		if got != want {
			t.Errorf("architecture %q: got %q, want %q\n%s", arch, got, want, p.Error())
		}
	}
	// With no model there is nothing to judge the config against.
	if p := run(t, dep(colocated(pl, e)), nil); fired(p, "engram-config-valid") {
		t.Errorf("fired with no model:\n%s", p.Error())
	}
}

// The graph states the release's default residency: host, since cpu_offload defaults to
// true. Stating device, by word or by default, is refused; a table no setting moves may
// live anywhere.
func TestEngramResidencyDefault(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	for _, c := range []struct {
		r    model.Residency
		ctl  model.ResidencyControl
		want bool
	}{
		{model.ResidencyHost, model.ResidencyEngramCPUOffload, false},
		{model.ResidencyDevice, model.ResidencyEngramCPUOffload, true},
		{"", model.ResidencyEngramCPUOffload, true},
		{"", model.ResidencyFixed, false},
		{model.ResidencyHost, model.ResidencyFixed, false},
	} {
		g := engramGraph()
		n := &g.LayerKinds[0].Nodes[len(g.LayerKinds[0].Nodes)-1]
		n.Residency, n.ResidencyControl = c.r, c.ctl
		if got := fired(run(t, dep(colocated(pl, deployment.Engine{})), g), "engram-residency-default"); got != c.want {
			t.Errorf("%q under %q: fired=%v, want %v", c.r, c.ctl, got, c.want)
		}
	}
}

// An offload connector reaches the NIXL skip the same way a PD-transfer one does.
func TestNixlOffloadSkipsInterleaveAssert(t *testing.T) {
	d := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1, DCP: 8},
		deployment.Engine{CPKVCacheInterleaveSize: 128, BlockSize: 64}))
	d.Offload = &deployment.Offload{Connector: "NixlConnector", Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1}}}
	if p := run(t, d, graniteGraph()); fired(p, "cp-kv-cache-interleave-fits-block") {
		t.Errorf("a NIXL offload connector should skip the assert:\n%s", p.Error())
	}
}

// A LongRoPE model's shorter starting window does not lower the limit: a stated length up
// to the full window passes.
func TestMaxModelLenAgainstLongRoPEWindow(t *testing.T) {
	g := graniteGraph()
	g.Global.ContextWindow, g.Global.DefaultContextWindow = 131072, 4096
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	if p := run(t, dep(colocated(pl, deployment.Engine{MaxModelLen: 65536})), g); fired(p, "max-model-len-within-context-window") {
		t.Errorf("a length within the window fired:\n%s", p.Error())
	}
}

// A per-dtype record is read under the dtype the engine runs, so a key this release does
// not accept is refused, in a cache and in an override alike; the release's own names pass.
func TestCacheRecordDTypesKnown(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	withRecord := func(key string, inOverride bool) *model.Graph {
		g := graniteGraph()
		n := &g.LayerKinds[0].Nodes[0]
		n.Caches = []model.Cache{{Name: "kv", Kind: model.CacheKV, Owned: true, BytesPerEntry: 256}}
		if inOverride {
			g.ArchOverrides = []model.ArchOverride{{CapabilityFamily: 10,
				Caches: []model.CacheOverride{{Name: "kv", BytesByCacheDType: map[string]int{key: 128}}}}}
		} else {
			n.Caches[0].BytesByCacheDType = map[string]int{key: 128}
		}
		return g
	}
	for _, inOverride := range []bool{false, true} {
		for key, want := range map[string]bool{"nvfp4_ds_mla": false, "fp8": false, "nvfp4_ds_ml": true, "e4m3": true, "auto": true} {
			p := run(t, dep(colocated(pl, deployment.Engine{})), withRecord(key, inOverride))
			if got := fired(p, "cache-record-dtypes-known"); got != want {
				t.Errorf("%q (override %v): fired=%v, want %v\n%s", key, inOverride, got, want, p.Error())
			}
		}
	}
}

// On an architecture with its own sequence parallelism, a decoder-layer reduction keyed
// to the engine-wide MoE rule is an error and a plain tensor-parallel all-reduce a
// warning; the same graph on an architecture without one draws neither, and neither does
// a head collective, which runs after the stream is gathered back.
func TestModelSequenceParallelCollectives(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	graph := func(arch string, emit model.EmitCondition, inHead bool) *model.Graph {
		g := graniteGraph()
		g.Global.Architecture = arch
		n := model.Node{Op: model.OpAllReduce, Role: "mlp_out", Emit: emit}
		if inHead {
			g.Head = append(g.Head, n)
		} else {
			g.LayerKinds[0].Nodes = append(g.LayerKinds[0].Nodes, n)
		}
		return g
	}
	severity := func(g *model.Graph) string {
		for _, pr := range run(t, dep(colocated(pl, deployment.Engine{})), g).All() {
			if pr.Rule == "model-sequence-parallel-collectives" {
				return string(pr.Severity)
			}
		}
		return ""
	}
	v41 := rules.ArchDeepseekV41
	for _, c := range []struct {
		arch   string
		emit   model.EmitCondition
		inHead bool
		want   string
	}{
		{v41, model.EmitTensorParallelUnlessSequenceParallelMoE, false, "error"},
		{v41, model.EmitTensorParallel, false, "warning"},
		{v41, model.EmitTensorParallelUnlessSequenceParallel, false, ""},
		{v41, model.EmitTensorParallel, true, ""},
		{"DeepseekV3ForCausalLM", model.EmitTensorParallelUnlessSequenceParallelMoE, false, ""},
		{"DeepseekV3ForCausalLM", model.EmitTensorParallel, false, ""},
	} {
		if got := severity(graph(c.arch, c.emit, c.inHead)); got != c.want {
			t.Errorf("%s %q (head %v): got %q, want %q", c.arch, c.emit, c.inHead, got, c.want)
		}
	}
}

// The draft's vocab-parallel lookup reduces before the draft shards its stream, as the
// target's does. Stated in the speculator's embedding it draws nothing; the same nodes at
// the head of the draft's layer kind read as a decoder-layer reduction and draw the
// warning, so the slot, not an exemption, is what tells the two apart.
func TestDraftEmbeddingReductionIsNotADecoderLayerOne(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	lookup := []model.Node{
		{Op: model.OpGather, Role: "draft_embed_tokens", Rows: 49155, RowBytes: 3072, RowsPerToken: 1,
			OutBytesPerToken: 3072, Shard: model.ShardRows, Aliases: "embedding.embed_tokens"},
		{Op: model.OpAllReduce, Role: "draft_embed_out", Emit: model.EmitTensorParallel},
	}
	graph := func(inSlot bool) *model.Graph {
		g := graniteGraph()
		g.Global.Architecture = rules.ArchDeepseekV41
		g.Embedding = []model.Node{{Op: model.OpGather, Role: "embed_tokens", Rows: 49155,
			RowBytes: 3072, RowsPerToken: 1, OutBytesPerToken: 3072, Shard: model.ShardRows}}
		draft := model.LayerKind{ID: "draft", Nodes: []model.Node{
			{Op: model.OpGEMM, Role: "draft_proj", N: 3072, K: 3072, TP: model.TPReplicated}}}
		g.Speculator = &model.Speculator{Method: "dspark", NumSpec: 5, Drafting: model.DraftingParallelBlock,
			Stack: model.Stack{Pattern: []string{"draft"}, Repeat: 1}}
		if inSlot {
			g.Speculator.Embedding = append([]model.Node(nil), lookup...)
		} else {
			draft.Nodes = append(append([]model.Node(nil), lookup...), draft.Nodes...)
		}
		g.LayerKinds = append(g.LayerKinds, draft)
		return g
	}
	for _, inSlot := range []bool{true, false} {
		g := graph(inSlot)
		if p := g.Validate(); !p.OK() {
			t.Fatalf("in slot %v: invalid graph:\n%s", inSlot, p.Error())
		}
		p := run(t, dep(colocated(pl, deployment.Engine{})), g)
		if got := fired(p, "model-sequence-parallel-collectives"); got == inSlot {
			t.Errorf("in slot %v: rule fired %v:\n%s", inSlot, got, p.Error())
		}
	}
}

// pplx and naive are accepted and run as allgather_reducescatter, so they make the MoE
// input sequence-parallel exactly where that backend does, and the rule names what runs.
func TestRenamedAll2AllBackends(t *testing.T) {
	p := Pack()
	for _, pl := range layoutGrid() {
		want := p.SequenceParallelMoE("allgather_reducescatter", pl.EnableExpertParallel, pl.TP, pl.DP)
		for _, b := range []string{"pplx", "naive"} {
			if got := p.SequenceParallelMoE(b, pl.EnableExpertParallel, pl.TP, pl.DP); got != want {
				t.Fatalf("%+v %q: %v, but allgather_reducescatter gives %v", pl, b, got, want)
			}
		}
	}
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	r := run(t, dep(colocated(pl, deployment.Engine{All2AllBackend: "pplx"})), graniteGraph())
	if !fired(r, "all2all-backend-known") || !r.OK() || !strings.Contains(r.Error(), "allgather_reducescatter") {
		t.Errorf("pplx should warn that allgather_reducescatter runs:\n%s", r.Error())
	}
}

// vLLM spells the MoE backend in lower case with underscores before validating it.
func TestMoEBackendIsNormalized(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	for _, b := range []string{"deep-gemm-mega-moe", "DEEP_GEMM_MEGA_MOE", "Deep-Gemm"} {
		if r := run(t, dep(colocated(pl, deployment.Engine{MoEBackend: b})), graniteGraph()); fired(r, "enum-values-known") {
			t.Errorf("%q is accepted by the engine once normalized:\n%s", b, r.Error())
		}
	}
	v4 := deployment.Parallelism{TP: 8, PP: 1, DP: 1, EnableExpertParallel: true}
	if !Pack().ModelSequenceParallel(rules.ArchDeepseekV4, v4, "deep-gemm-mega-moe") {
		t.Error("a dashed mega-MoE backend should count as one")
	}
}

// A model that decides its own sequence parallelism never consults the engine-wide rule,
// so the rule stays silent for it on a layout where the engine-wide rule is on.
func TestSequenceParallelMoEImpliedSkipsOwnRule(t *testing.T) {
	pl := deployment.Parallelism{TP: 2, PP: 2, DP: 2, EnableExpertParallel: true}
	for arch, want := range map[string]bool{rules.ArchDeepseekV4: false, "DeepseekV3ForCausalLM": true} {
		g := graniteGraph()
		g.Global.Architecture = arch
		if got := fired(run(t, dep(colocated(pl, deployment.Engine{})), g), "sequence-parallel-moe-implied"); got != want {
			t.Errorf("%s: fired=%v, want %v", arch, got, want)
		}
	}
}

// This release lowers the checked block size to the smallest prefix-cacheable group's. A
// kv cache paged at 32 always takes part, so the failure is certain; a window cache paged
// at 32 takes part only without bounded replay, so the failure is possible and warns.
func TestInterleaveChecksTheEffectiveBlock(t *testing.T) {
	pool := colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1, DCP: 8},
		deployment.Engine{CPKVCacheInterleaveSize: 64, BlockSize: 256})
	severity := func(kind model.CacheKind) string {
		g := graniteGraph()
		c := model.Cache{Name: "c", Kind: kind, Owned: true, BytesPerEntry: 584, BlockSize: 32}
		if kind == model.CacheWindow {
			c.Window = 128
			g.LayerKinds[0].Nodes[0].Window = 128
		}
		g.LayerKinds[0].Nodes[0].Caches = []model.Cache{c}
		for _, pr := range run(t, dep(pool), g).All() {
			if pr.Rule == "cp-kv-cache-interleave-fits-block" {
				return string(pr.Severity)
			}
		}
		return ""
	}
	if got := severity(model.CacheKV); got != "error" {
		t.Errorf("a kv cache paged at 32: got %q, want error", got)
	}
	if got := severity(model.CacheWindow); got != "warning" {
		t.Errorf("a window cache paged at 32: got %q, want warning", got)
	}
	if got := severity(model.CacheState); got != "" {
		t.Errorf("a state cache never takes part in prefix caching: got %q", got)
	}
	// Under prefill-context parallelism bounded replay is off, so the window cache
	// certainly takes part and the failure is certain.
	pcpPool := colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1, DCP: 8, PCP: 2},
		deployment.Engine{CPKVCacheInterleaveSize: 64, BlockSize: 256})
	g := graniteGraph()
	g.LayerKinds[0].Nodes[0].Window = 128
	g.LayerKinds[0].Nodes[0].Caches = []model.Cache{{Name: "w", Kind: model.CacheWindow, Owned: true, BytesPerEntry: 584, Window: 128, BlockSize: 32}}
	if p := run(t, dep(pcpPool), g); p.OK() {
		t.Errorf("under pcp 2 the window cache's 32 is certain:\n%s", p.Error())
	}
}

// On a moonep pool an FP16 checkpoint may be cast to BF16 by --dtype, which a deployment
// does not record, so it warns where a quantized checkpoint is refused.
func TestMoonEPOnFP16Warns(t *testing.T) {
	pl := deployment.Parallelism{TP: 1, PP: 1, DP: 8, EnableExpertParallel: true}
	g := graniteGraph()
	g.Global.WeightDType = model.DTypeFP16
	p := run(t, dep(colocated(pl, deployment.Engine{All2AllBackend: "moonep"})), g)
	if !fired(p, "moonep-requirements") || !p.OK() {
		t.Errorf("want a warning, not an error:\n%s", p.Error())
	}
}

// The cache dtype that runs on DeepSeek-V4.1: on SM100 mega attention's NVFP4 record
// wherever tp divides 8, FlashMLA's MXFP8 record otherwise, FlashMLA's FP8 record on SM90;
// a stated packed dtype runs as stated; other architectures run what they request.
func TestResolveCacheDType(t *testing.T) {
	p := Pack()
	v41 := rules.ArchDeepseekV41
	for _, c := range []struct {
		arch            string
		family, tp      int
		requested, want string
	}{
		{v41, 10, 8, "", "nvfp4_ds_mla"},
		{v41, 10, 4, "fp8", "nvfp4_ds_mla"},
		{v41, 10, 1, "auto", "nvfp4_ds_mla"},
		{v41, 10, 16, "auto", "fp8_ds_mla"},
		{v41, 10, 3, "auto", "fp8_ds_mla"},
		{v41, 10, 8, "fp8_ds_mla", "fp8_ds_mla"},
		{v41, 9, 8, "auto", "fp8_ds_mla"},
		{v41, 9, 8, "fp8", "fp8_ds_mla"},
		{"DeepseekV3ForCausalLM", 10, 8, "auto", "auto"},
	} {
		if got := p.ResolveCacheDType(c.arch, c.family, c.tp, c.requested); got != c.want {
			t.Errorf("%s sm%d0 tp %d %q: %q, want %q", c.arch, c.family, c.tp, c.requested, got, c.want)
		}
	}
}

// The window-pages formula: a 128-token window at 32-token pages holds 5 pages with one
// token in flight and 6 with DSpark's 1 + 5; max_model_len caps it; a non-window cache has none.
func TestWindowPagesPerRequest(t *testing.T) {
	p := Pack()
	w := model.Cache{Kind: model.CacheWindow, Window: 128, BlockSize: 32}
	for _, c := range []struct{ inFlight, extra, maxLen, want int }{
		{1, 0, 0, 5}, {6, 0, 0, 6}, {6, 0, 64, 3}, {1, 4, 0, 6},
	} {
		if got := p.WindowPagesPerRequest(w, 64, c.inFlight, c.extra, c.maxLen); got != c.want {
			t.Errorf("%+v: %d pages, want %d", c, got, c.want)
		}
	}
	if got := p.WindowPagesPerRequest(model.Cache{Kind: model.CacheKV, BlockSize: 32}, 64, 6, 0, 0); got != 0 {
		t.Errorf("a kv cache is not a window: %d", got)
	}
	// The engine block applies where the cache states none.
	if got := p.WindowPagesPerRequest(model.Cache{Kind: model.CacheWindow, Window: 128}, 64, 1, 0, 0); got != 3 {
		t.Errorf("at 64-token pages: %d, want 3", got)
	}
}

// The capture knobs are refused as a pair the engine refuses, and the values it changes
// at startup are named with the value that runs.
func TestCUDAGraphCaptureSizesRule(t *testing.T) {
	n := func(v int) *int { return &v }
	for _, c := range []struct {
		name string
		e    deployment.Engine
		sev  rules.Severity // empty: the rule stays silent
	}{
		{"unstated", deployment.Engine{MaxNumBatchedTokens: 8192}, ""},
		{"on the stride", deployment.Engine{MaxNumBatchedTokens: 8192, MaxCUDAGraphCaptureSize: n(64)}, ""},
		{"consistent pair", deployment.Engine{MaxNumBatchedTokens: 8192, MaxCUDAGraphCaptureSize: n(32),
			CUDAGraphCaptureSizes: []int{8, 32}}, ""},
		{"unstated max_num_batched_tokens", deployment.Engine{MaxCUDAGraphCaptureSize: n(100)}, ""},
		{"inconsistent pair", deployment.Engine{MaxNumBatchedTokens: 8192, MaxCUDAGraphCaptureSize: n(64),
			CUDAGraphCaptureSizes: []int{8, 32}}, rules.SeverityError},
		{"capped", deployment.Engine{MaxNumBatchedTokens: 512, MaxCUDAGraphCaptureSize: n(1024)}, rules.SeverityWarning},
		{"truncated", deployment.Engine{MaxNumBatchedTokens: 8192, MaxCUDAGraphCaptureSize: n(100)}, rules.SeverityWarning},
		{"dropped sizes", deployment.Engine{MaxNumBatchedTokens: 16, CUDAGraphCaptureSizes: []int{8, 32}}, rules.SeverityWarning},
		{"ignored under NONE", deployment.Engine{CUDAGraphMode: "NONE", MaxCUDAGraphCaptureSize: n(64)}, rules.SeverityWarning},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := run(t, dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, c.e)), graniteGraph())
			var got rules.Severity
			for _, it := range p.All() {
				if it.Rule == "cudagraph-capture-sizes" && (got == "" || it.Severity == rules.SeverityError) {
					got = it.Severity
				}
			}
			if got != c.sev {
				t.Errorf("severity %q, want %q\n%s", got, c.sev, p.Error())
			}
		})
	}
}

func TestLongPrefillThresholdWithinMaxModelLen(t *testing.T) {
	for _, c := range []struct {
		e    deployment.Engine
		fire bool
	}{
		{deployment.Engine{LongPrefillTokenThreshold: 8192, MaxModelLen: 131072}, false},
		{deployment.Engine{LongPrefillTokenThreshold: 8192, MaxModelLen: 8192}, false},
		{deployment.Engine{LongPrefillTokenThreshold: 8193, MaxModelLen: 8192}, true},
		{deployment.Engine{LongPrefillTokenThreshold: 1 << 20, MaxModelLen: deployment.MaxModelLenAuto}, false},
		{deployment.Engine{LongPrefillTokenThreshold: 1 << 20}, false},
	} {
		p := run(t, dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, c.e)), graniteGraph())
		if fired(p, "long-prefill-threshold-within-max-model-len") != c.fire {
			t.Errorf("%+v: fired %v, want %v\n%s", c.e, !c.fire, c.fire, p.Error())
		}
	}
}

// The load-balancing mode is refused where the engine refuses it, and a hybrid mode the
// engine runs as another is named with the mode that runs.
func TestDataParallelLoadBalancing(t *testing.T) {
	dense := graniteGraph()
	dense.LayerKinds[0].Nodes = dense.LayerKinds[0].Nodes[:1]
	pl := func(mode deployment.DPLoadBalancing, dp, local int) deployment.Parallelism {
		return deployment.Parallelism{TP: 1, PP: 1, DP: dp, DPLocal: local, DataParallelLoadBalancing: mode}
	}
	for _, c := range []struct {
		name string
		pl   deployment.Parallelism
		g    *model.Graph
		sev  rules.Severity
		runs deployment.DPLoadBalancing
	}{
		{"internal", pl("", 8, 0), graniteGraph(), "", deployment.DPLBInternal},
		{"external on a MoE", pl(deployment.DPLBExternal, 8, 8), graniteGraph(), "", deployment.DPLBExternal},
		{"external on a dense model", pl(deployment.DPLBExternal, 8, 8), dense, rules.SeverityError, deployment.DPLBExternal},
		{"hybrid across nodes", pl(deployment.DPLBHybrid, 16, 8), graniteGraph(), "", deployment.DPLBHybrid},
		{"hybrid without dp_local", pl(deployment.DPLBHybrid, 16, 0), graniteGraph(), rules.SeverityError, deployment.DPLBHybrid},
		{"hybrid with one rank per node", pl(deployment.DPLBHybrid, 8, 1), graniteGraph(), rules.SeverityWarning, deployment.DPLBExternal},
		{"hybrid with one rank per node, dense", pl(deployment.DPLBHybrid, 8, 1), dense, rules.SeverityError, deployment.DPLBExternal},
		{"hybrid on one node", pl(deployment.DPLBHybrid, 8, 8), graniteGraph(), rules.SeverityWarning, deployment.DPLBInternal},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := Pack().ResolveDPLoadBalancing(c.pl); got != c.runs {
				t.Errorf("runs %s, want %s", got, c.runs)
			}
			pool := colocated(c.pl, deployment.Engine{})
			pool.Nodes = max(1, c.pl.DP/max(1, c.pl.DPLocal))
			p := run(t, dep(pool), c.g)
			var got rules.Severity
			for _, it := range p.All() {
				if it.Rule == "data-parallel-load-balancing" && (got == "" || it.Severity == rules.SeverityError) {
					got = it.Severity
				}
			}
			if got != c.sev {
				t.Errorf("severity %q, want %q\n%s", got, c.sev, p.Error())
			}
		})
	}
}
