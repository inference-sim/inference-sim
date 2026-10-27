package v0_29

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
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

func colocated(pl deployment.Parallelism, e engineconfig.Engine) deployment.Pool {
	b, err := engineconfig.Encode(e)
	if err != nil {
		panic(err)
	}
	return deployment.Pool{Role: deployment.RoleColocated, Nodes: 1, Parallel: pl, Engine: b}
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
		"dbo-thresholds-stated-when-enabled",
		"dcp-comm-backend-known",
		"engram-config-unsupported",
		"enum-values-known",
		"eviction-policy-known",
		"expert-divisibility-under-eplb",
		"expert-imbalance-without-eplb",
		"long-prefill-threshold-within-max-model-len",
		"mamba-cache-mode-matches-model",
		"model-sequence-parallel-collectives",
		"offload-spec-known",
		"pcp-excludes-data-parallelism",
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
		engineconfig.Engine{All2AllBackend: "telepathy", AllReduceBackend: "osmosis"}))
	p := run(t, s, graniteGraph())
	for _, rule := range []string{"all2all-backend-known", "allreduce-backend-known"} {
		if !fired(p, rule) {
			t.Errorf("%s did not fire on an unknown backend", rule)
		}
	}
}

func TestKnownBackendsAccepted(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		engineconfig.Engine{All2AllBackend: "allgather_reducescatter",
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
	s := dep(colocated(pl, engineconfig.Engine{
		EPLB: &engineconfig.EPLB{Enabled: true, NumRedundantExperts: 32}}))
	p := run(t, s, graniteGraph())
	if !fired(p, "expert-divisibility-under-eplb") {
		t.Fatal("divisibility rule did not fire on 224+32 over 72")
	}
	if !strings.Contains(p.Error(), "32 more redundant") {
		t.Errorf("message should name the shortfall; got:\n%s", p.Error())
	}
	// 224 + 64 = 288 = 4 x 72.
	ok := dep(colocated(pl, engineconfig.Engine{
		EPLB: &engineconfig.EPLB{Enabled: true, NumRedundantExperts: 64}}))
	if p := run(t, ok, graniteGraph()); fired(p, "expert-divisibility-under-eplb") {
		t.Errorf("divisibility rule fired on a divisible layout:\n%s", p.Error())
	}
}

// TestExpertImbalanceWithoutEPLB: the same layout is legal without load balancing,
// and carries a static imbalance the estimate must account for.
func TestExpertImbalanceWithoutEPLB(t *testing.T) {
	pl := deployment.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}
	s := dep(colocated(pl, engineconfig.Engine{}))
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
	s := dep(colocated(pl, engineconfig.Engine{}))
	p := run(t, s, graniteGraph())
	if p.OK() {
		t.Fatal("expected an error when width exceeds the expert count")
	}
}

func TestSequenceParallelMoEImplied(t *testing.T) {
	// The default backend with both tp and dp above one implies sequence-parallel
	// MoE, which changes which collectives the graph must emit.
	pl := deployment.Parallelism{TP: 2, PP: 1, DP: 8, EnableExpertParallel: true}
	s := dep(colocated(pl, engineconfig.Engine{All2AllBackend: "allgather_reducescatter"}))
	if p := run(t, s, graniteGraph()); !fired(p, "sequence-parallel-moe-implied") {
		t.Fatal("rule did not fire for tp=2, dp=8 on the default backend")
	}
	// tp=1 does not trigger it, which is why the design's own examples do not.
	pl1 := deployment.Parallelism{TP: 1, PP: 1, DP: 8, EnableExpertParallel: true}
	s1 := dep(colocated(pl1, engineconfig.Engine{All2AllBackend: "allgather_reducescatter"}))
	if p := run(t, s1, graniteGraph()); fired(p, "sequence-parallel-moe-implied") {
		t.Error("rule fired at tp=1, where the engine does not apply it")
	}
}

func TestCustomAllReduceReachability(t *testing.T) {
	// A world size outside the supported set falls back.
	s := dep(colocated(deployment.Parallelism{TP: 3, PP: 1, DP: 1},
		engineconfig.Engine{AllReduceBackend: "custom"}))
	if p := run(t, s, graniteGraph()); !fired(p, "custom-allreduce-reachable") {
		t.Error("rule did not fire on tp=3")
	}
	// Spanning nodes needs multi-node NVLink.
	s2 := dep(
		colocated(deployment.Parallelism{TP: 16, PP: 1, DP: 1},
			engineconfig.Engine{AllReduceBackend: "custom"}),
	)
	if p := run(t, s2, graniteGraph()); !fired(p, "custom-allreduce-reachable") {
		t.Error("rule did not fire on tp=16 across 8-GPU nodes")
	}
	// tp=8 inside one node is reachable.
	s3 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		engineconfig.Engine{AllReduceBackend: "custom"}))
	if p := run(t, s3, graniteGraph()); fired(p, "custom-allreduce-reachable") {
		t.Errorf("rule fired on a reachable layout:\n%s", p.Error())
	}
}

func TestMambaCacheModeMatchesModel(t *testing.T) {
	// A mode on a model with no recurrent layer has no effect.
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		engineconfig.Engine{MambaCacheMode: "align"}))
	if p := run(t, s, graniteGraph()); !fired(p, "mamba-cache-mode-matches-model") {
		t.Error("rule did not fire for a mode on a non-hybrid model")
	}
	// A hybrid model with no mode leaves the state unsized.
	hybrid := graniteGraph()
	hybrid.LayerKinds[0].Nodes = append(hybrid.LayerKinds[0].Nodes, model.Node{
		Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2,
		NumHeads: 128, StateSize: 128, NumGroups: 8, ConvKernel: 4,
		IntermediateSize: 8192})
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, engineconfig.Engine{}))
	if p := run(t, s2, hybrid); !fired(p, "mamba-cache-mode-matches-model") {
		t.Error("rule did not fire for a hybrid model with no mode")
	}
}

func TestOffloadSpecKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, engineconfig.Engine{}))
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
		engineconfig.Engine{DBO: &engineconfig.DBO{Enabled: true}}))
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
		engineconfig.Engine{Speculative: &engineconfig.Speculative{
			Method: "crystal_ball", NumSpecTokens: 2}}))
	if p := run(t, s, graniteGraph()); !fired(p, "speculative-method-known") {
		t.Fatal("rule did not fire on an unknown method")
	}
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		engineconfig.Engine{Speculative: &engineconfig.Speculative{
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
		engineconfig.Engine{Quantization: "fp8_but_misspelled"}))
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
		engineconfig.Engine{Quantization: "compressed-tensors"}))
	if p := run(t, s2, graniteGraph()); fired(p, "quantization-known") {
		t.Error("rule fired on an in-tree quantization method")
	}
}

// TestConnectorKnown checks both connector sites — the offload block and the pd_transfer
// block — since a deployment can name a connector in either. An unknown name warns; a
// registered one is quiet.
func TestConnectorKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, engineconfig.Engine{}))
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
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, engineconfig.Engine{}))
	s2.PDTransfer = &deployment.PDTransfer{Connector: "CarrierPigeonConnector"}
	if p := run(t, s2, graniteGraph()); !fired(p, "connector-known") {
		t.Fatal("rule did not fire on an unregistered pd_transfer connector")
	}
}

// TestEvictionPolicyKnown: an in-tree policy (lru/arc) passes, a typo warns and the warning
// enumerates the in-tree set. A warning, not an error, because the CachePolicyFactory admits
// out-of-tree policies — the same shape as the connector and quantization rules (#19).
func TestEvictionPolicyKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, engineconfig.Engine{}))
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

// TestPCPExcludesDataParallelism: this release refuses the combination at startup, so a
// v0.29 scenario that asks for it is reported — by the version's own rule, because the
// schema admits it for releases that run it. Each axis alone is fine.
func TestPCPExcludesDataParallelism(t *testing.T) {
	cases := []struct {
		name string
		pl   deployment.Parallelism
		want bool
	}{
		{"pcp with dp", deployment.Parallelism{TP: 1, PP: 1, DP: 4, PCP: 8}, true},
		{"pcp alone", deployment.Parallelism{TP: 1, PP: 1, DP: 1, PCP: 8}, false},
		{"dp alone", deployment.Parallelism{TP: 1, PP: 1, DP: 8}, false},
		{"pcp unset with dp", deployment.Parallelism{TP: 1, PP: 1, DP: 8, PCP: 0}, false},
		{"pcp 1 with dp", deployment.Parallelism{TP: 1, PP: 1, DP: 8, PCP: 1}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := run(t, dep(colocated(c.pl, engineconfig.Engine{})), nil)
			if got := fired(p, "pcp-excludes-data-parallelism"); got != c.want {
				t.Fatalf("%+v: rule fired=%v, want %v\n%s", c.pl, got, c.want, p.Error())
			}
			if c.want && !strings.Contains(p.Error(), "pcp 8 with dp 4") {
				t.Errorf("message should name both widths: %s", p.Error())
			}
		})
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
		p := run(t, dep(colocated(pl, engineconfig.Engine{DCPCommBackend: c.backend})), nil)
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

// TestCPKVCacheInterleaveFitsBlock: with DCP on and no KV connector, this release asserts
// at startup that block_size is at least, and divisible by, the interleave size. The rule
// fires only where that assert is certain to run on the values stated.
func TestCPKVCacheInterleaveFitsBlock(t *testing.T) {
	pool := func(dcp, size, block int) deployment.Pool {
		return colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1, DCP: dcp},
			engineconfig.Engine{CPKVCacheInterleaveSize: size, BlockSize: block})
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
		{"pd transfer configured", func() *deployment.Deployment {
			d := dep(pool(8, 128, 64))
			d.PDTransfer = &deployment.PDTransfer{Connector: "nixl"}
			return d
		}(), graniteGraph(), false},
		{"offload connector configured", func() *deployment.Deployment {
			d := dep(pool(8, 128, 64))
			d.Offload = &deployment.Offload{Connector: "OffloadingConnector",
				Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1}}}
			return d
		}(), graniteGraph(), false},
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
			got := fired(run(t, dep(colocated(pl, engineconfig.Engine{All2AllBackend: b})), graniteGraph()),
				"sequence-parallel-moe-implied")
			if got != want {
				t.Fatalf("%+v backend %q: rule fired=%v, method says %v", pl, b, got, want)
			}
		}
	}
}

// In 0.29.0 DeepSeek-V4 decides its own sequence parallelism, by the condition 0.31.0
// keeps, and DeepSeek-V4.1 does not exist: over every layout and backend V4 follows the
// condition and V4.1 is never on.
func TestModelSequenceParallelIn029(t *testing.T) {
	p := Pack()
	for _, pl := range layoutGrid() {
		for _, moe := range append(sortedKeys(p.MoEBackends), "") {
			if p.ModelSequenceParallel(rules.ArchDeepseekV41, pl, moe) {
				t.Fatalf("%+v %q: V4.1 is on in a release that predates it", pl, moe)
			}
			base := pl.PP <= 1 && pl.EnableExpertParallel && pl.TP > 1
			want := base && (pl.DP > 1 || p.MegaMoEBackends[moe])
			if got := p.ModelSequenceParallel(rules.ArchDeepseekV4, pl, moe); got != want {
				t.Fatalf("%+v %q: V4 got %v, want %v", pl, moe, got, want)
			}
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
	r := run(t, dep(colocated(pl, engineconfig.Engine{All2AllBackend: "pplx"})), graniteGraph())
	if !fired(r, "all2all-backend-known") || !r.OK() || !strings.Contains(r.Error(), "allgather_reducescatter") {
		t.Errorf("pplx should warn that allgather_reducescatter runs:\n%s", r.Error())
	}
}

// vLLM spells the MoE backend in lower case with underscores before validating it.
func TestMoEBackendIsNormalized(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	for _, b := range []string{"deep-gemm-mega-moe", "DEEP_GEMM_MEGA_MOE", "Deep-Gemm"} {
		if r := run(t, dep(colocated(pl, engineconfig.Engine{MoEBackend: b})), graniteGraph()); fired(r, "enum-values-known") {
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
		if got := fired(run(t, dep(colocated(pl, engineconfig.Engine{})), g), "sequence-parallel-moe-implied"); got != want {
			t.Errorf("%s: fired=%v, want %v", arch, got, want)
		}
	}
}

// This release lowers the checked block size to the smallest page size of every KV cache
// group, so a cache paged at 32 tokens makes a stated 256 with interleave 64 fail.
func TestInterleaveChecksTheEffectiveBlock(t *testing.T) {
	pool := colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1, DCP: 8},
		engineconfig.Engine{CPKVCacheInterleaveSize: 64, BlockSize: 256})
	g := graniteGraph()
	if p := run(t, dep(pool), g); fired(p, "cp-kv-cache-interleave-fits-block") {
		t.Fatalf("256 is divisible by 64:\n%s", p.Error())
	}
	g.LayerKinds[0].Nodes[0].Caches = []model.Cache{{Name: "kv", Kind: model.CacheKV, Owned: true, BytesPerEntry: 256, BlockSize: 32}}
	p := run(t, dep(pool), g)
	if !fired(p, "cp-kv-cache-interleave-fits-block") || p.OK() || !strings.Contains(p.Error(), "checks 32, not the stated 256") {
		t.Errorf("the engine checks 32, which 64 does not fit:\n%s", p.Error())
	}
}

func TestEngramConfigUnsupported(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	if p := run(t, dep(colocated(pl, engineconfig.Engine{EngramConfig: &engineconfig.EngramConfig{}})), graniteGraph()); !fired(p, "engram-config-unsupported") {
		t.Errorf("0.29.0 has no Engram config:\n%s", p.Error())
	}
	if p := run(t, dep(colocated(pl, engineconfig.Engine{})), graniteGraph()); fired(p, "engram-config-unsupported") {
		t.Errorf("fired without an Engram config:\n%s", p.Error())
	}
}

// The Mamba state dtypes are their own literal, which accepts float32 and refuses fp8, and
// the MoE backend is checked against its literal.
func TestEnumValuesForMambaAndMoE(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	for _, c := range []struct {
		e    engineconfig.Engine
		want bool
	}{
		{engineconfig.Engine{MambaSSMCacheDType: "float32"}, false},
		{engineconfig.Engine{MambaCacheDType: "fp8"}, true},
		{engineconfig.Engine{MoEBackend: "deep_gemm_mega_moe"}, false},
		{engineconfig.Engine{MoEBackend: "rdna3"}, true}, // added in 0.31.0
	} {
		if got := fired(run(t, dep(colocated(pl, c.e)), graniteGraph()), "enum-values-known"); got != c.want {
			t.Errorf("%+v: rule fired=%v, want %v", c.e, got, c.want)
		}
	}
}

// 0.29.0 has no nvfp4_ds_mla, so a V4.1 record keyed by it can never be read there.
func TestCacheRecordDTypesKnownIn029(t *testing.T) {
	pl := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	for key, want := range map[string]bool{"fp8_ds_mla": false, "nvfp4_ds_mla": true, "auto": true} {
		g := graniteGraph()
		g.LayerKinds[0].Nodes[0].Caches = []model.Cache{{Name: "kv", Kind: model.CacheKV, Owned: true,
			BytesPerEntry: 256, BytesByCacheDType: map[string]int{key: 128}}}
		if got := fired(run(t, dep(colocated(pl, engineconfig.Engine{})), g), "cache-record-dtypes-known"); got != want {
			t.Errorf("%q: fired=%v, want %v", key, got, want)
		}
	}
}

// The capture knobs are refused as a pair the engine refuses, and the values it changes
// at startup are named with the value that runs.
func TestCUDAGraphCaptureSizesRule(t *testing.T) {
	n := func(v int) *int { return &v }
	for _, c := range []struct {
		name string
		e    engineconfig.Engine
		sev  rules.Severity // empty: the rule stays silent
	}{
		{"unstated", engineconfig.Engine{MaxNumBatchedTokens: 8192}, ""},
		{"on the stride", engineconfig.Engine{MaxNumBatchedTokens: 8192, MaxCUDAGraphCaptureSize: n(64)}, ""},
		{"consistent pair", engineconfig.Engine{MaxNumBatchedTokens: 8192, MaxCUDAGraphCaptureSize: n(32),
			CUDAGraphCaptureSizes: []int{8, 32}}, ""},
		{"unstated max_num_batched_tokens", engineconfig.Engine{MaxCUDAGraphCaptureSize: n(100)}, ""},
		{"inconsistent pair", engineconfig.Engine{MaxNumBatchedTokens: 8192, MaxCUDAGraphCaptureSize: n(64),
			CUDAGraphCaptureSizes: []int{8, 32}}, rules.SeverityError},
		{"capped", engineconfig.Engine{MaxNumBatchedTokens: 512, MaxCUDAGraphCaptureSize: n(1024)}, rules.SeverityWarning},
		{"truncated", engineconfig.Engine{MaxNumBatchedTokens: 8192, MaxCUDAGraphCaptureSize: n(100)}, rules.SeverityWarning},
		{"dropped sizes", engineconfig.Engine{MaxNumBatchedTokens: 16, CUDAGraphCaptureSizes: []int{8, 32}}, rules.SeverityWarning},
		{"ignored under NONE", engineconfig.Engine{CUDAGraphMode: "NONE", MaxCUDAGraphCaptureSize: n(64)}, rules.SeverityWarning},
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
		e    engineconfig.Engine
		fire bool
	}{
		{engineconfig.Engine{LongPrefillTokenThreshold: 8192, MaxModelLen: 131072}, false},
		{engineconfig.Engine{LongPrefillTokenThreshold: 8192, MaxModelLen: 8192}, false},
		{engineconfig.Engine{LongPrefillTokenThreshold: 8193, MaxModelLen: 8192}, true},
		{engineconfig.Engine{LongPrefillTokenThreshold: 1 << 20, MaxModelLen: engineconfig.MaxModelLenAuto}, false},
		{engineconfig.Engine{LongPrefillTokenThreshold: 1 << 20}, false},
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
			pool := colocated(c.pl, engineconfig.Engine{})
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
