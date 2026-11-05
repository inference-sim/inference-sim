package resolve

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
)

// A stub rules pack. The tests below are about the resolver's behaviour, so the engine
// facts are stated here rather than read from a real pack: a test that failed because a
// pack changed would not be testing the resolver.
type stubRules struct {
	spBackends   map[string]bool
	customWidths map[int]bool
}

func newStubRules() stubRules {
	return stubRules{
		spBackends:   map[string]bool{"allgather_reducescatter": true, "deepep_low_latency": true},
		customWidths: map[int]bool{2: true, 4: true, 6: true, 8: true, 16: true},
	}
}

func (r stubRules) SequenceParallelMoE(backend string, ep bool, tp, dp int) bool {
	return ep && tp > 1 && dp > 1 && r.spBackends[backend]
}
func (r stubRules) ModelSequenceParallel(string, deployment.Parallelism, string) bool {
	return false
}
func (r stubRules) EffectiveAll2AllBackend(b string) string { return b }
func (r stubRules) ResolveCacheDType(_ string, _, _ int, requested string) string {
	return requested
}
func (r stubRules) WindowPagesPerRequest(model.Cache, int, int, int, int) int       { return 0 }
func (r stubRules) CheckEngineBlock(string, deployment.EngineBlock) *rules.Problems { return nil }
func (r stubRules) DraftTokens(deployment.EngineBlock) int                          { return 0 }
func (r stubRules) CheckConnectorOptions(string, deployment.EngineBlock) *rules.Problems {
	return nil
}
func (r stubRules) CUDAGraphCaptureSizes(rules.CUDAGraphCaptureInput) (rules.CUDAGraphCapture, bool, error) {
	return rules.CUDAGraphCapture{}, false, nil
}
func (r stubRules) KVCacheGroups(rules.KVCacheGroupsInput) (rules.KVCacheGrouping, error) {
	return rules.KVCacheGrouping{}, nil
}
func (r stubRules) CustomAllReduceSupportsWidth(tp int) bool { return r.customWidths[tp] }
func (r stubRules) TritonFetchWithholdsSMs(pageBytes int) int {
	if pageBytes > 0 && pageBytes < 28*1024 {
		return 12
	}
	return 0
}
func (r stubRules) DBOEngages(enabled bool, tokens int, uniform bool) bool {
	if !enabled {
		return false
	}
	if uniform {
		return tokens >= 32
	}
	return tokens >= 512
}

// The rest of rules.EngineRules: facts the resolver does not read, answered inertly so a
// test here cannot pass by reading one.
func (r stubRules) EngineVersion() string                          { return "stub" }
func (r stubRules) Check(rules.Input) *rules.Problems              { return &rules.Problems{} }
func (r stubRules) RuleNames() []string                            { return nil }
func (r stubRules) SpeculativeMethodAccepted(string) bool          { return false }
func (r stubRules) SpeculativeMethodNames() []string               { return nil }
func (r stubRules) AsyncCompatibleSpeculative(string) (bool, bool) { return false, false }
func (r stubRules) EngramOnHostByDefault() bool                    { return true }
func (r stubRules) DCPCommBackendNames() map[string]bool           { return nil }
func (r stubRules) MegaMoEBackend(string) bool                     { return false }
func (r stubRules) ResolveDPLoadBalancing(deployment.Parallelism) deployment.DPLoadBalancing {
	return deployment.DPLBInternal
}

// scenarioWith builds the pair a resolver needs: the immutable problem and the
// deployment applied to it. They are two documents in the schema, so a helper that
// returned only a scenario would leave every caller to construct the other half.
func scenarioWith(nodes, gpusPerNode int,
	pools ...deployment.Pool) (*scenario.Scenario, *deployment.Deployment) {
	return &scenario.Scenario{
			Kind: "Scenario", Name: "t", Model: "m",
			Coefficients: []string{"c"}, EngineVersion: "0.29.0",
			Cluster: scenario.Cluster{
				Hardware: "h", Nodes: nodes, GPUsPerNode: gpusPerNode,
			},
		}, &deployment.Deployment{
			Kind: "Deployment", Name: "t", Pools: pools,
		}
}

// pool is a colocated pool with the parallel block pl. Its engine block is unstated: resolve
// takes the engine settings it reads separately (enginesettings.Settings).
func pool(pl deployment.Parallelism) deployment.Pool {
	return deployment.Pool{Role: deployment.RoleColocated, Nodes: 1, Parallel: pl}
}

// TestEmitsAnswersEveryConditionTheSchemaAllows is the property that makes the enum safe.
// The schema's condition set and this switch must be the same closed set: a condition a
// graph can state and the resolver cannot answer would silently drop a node, removing a
// cost with nothing reporting it.
func TestEmitsAnswersEveryConditionTheSchemaAllows(t *testing.T) {
	all := []model.EmitCondition{
		model.EmitAlways,
		model.EmitTensorParallel,
		model.EmitExpertParallel,
		model.EmitTensorParallelUnlessSequenceParallelMoE,
		model.EmitSequenceParallel,
		model.EmitTensorParallelUnlessSequenceParallel,
		model.EmitTensorParallelWithoutExpertParallel,
		model.EmitExpertParallelWithoutDispatch,
	}
	for _, c := range all {
		if !c.Valid() {
			t.Errorf("%q is in the test's list but the schema rejects it", c)
		}
		if !Recognizes(c) {
			t.Errorf("the schema allows %q but the resolver does not recognize it", c)
		}
	}
	// And the reverse: a condition the resolver would answer must be one the schema
	// allows, or the enum has drifted.
	if Recognizes("sp_moe_only") {
		t.Error("the resolver recognizes a condition the schema does not define")
	}
}

// TestEmitsUnderEachLayout states the behaviour a cost model depends on: which collectives
// exist at which parallelism. Table-driven over layouts rather than over internal state,
// so the test says what a deployment gets rather than how the resolver stores it.
func TestEmitsUnderEachLayout(t *testing.T) {
	cases := []struct {
		what   string
		layout Layout
		want   map[model.EmitCondition]bool
	}{
		{
			what:   "single rank: no collective runs",
			layout: Layout{TP: 1, ExpertWidth: 1},
			want: map[model.EmitCondition]bool{
				model.EmitAlways:                                  true,
				model.EmitTensorParallel:                          false,
				model.EmitExpertParallel:                          false,
				model.EmitTensorParallelUnlessSequenceParallelMoE: false,
			},
		},
		{
			what:   "tensor parallel only: the reductions run, the dispatch does not",
			layout: Layout{TP: 8, ExpertWidth: 1},
			want: map[model.EmitCondition]bool{
				model.EmitTensorParallel:                          true,
				model.EmitExpertParallel:                          false,
				model.EmitTensorParallelUnlessSequenceParallelMoE: true,
			},
		},
		{
			what:   "expert parallel only: the dispatch runs, the reductions do not",
			layout: Layout{TP: 1, ExpertWidth: 16},
			want: map[model.EmitCondition]bool{
				model.EmitTensorParallel:                          false,
				model.EmitExpertParallel:                          true,
				model.EmitTensorParallelUnlessSequenceParallelMoE: false,
			},
		},
		{
			what:   "sequence-parallel MoE replaces the MLP reduction",
			layout: Layout{TP: 2, ExpertWidth: 16, SequenceParallelMoE: true},
			want: map[model.EmitCondition]bool{
				// The attention reduction still runs: sequence-parallel MoE changes the
				// MLP path only.
				model.EmitTensorParallel:                          true,
				model.EmitExpertParallel:                          true,
				model.EmitTensorParallelUnlessSequenceParallelMoE: false,
			},
		},
		{
			what:   "expert parallel at dp 1: the experts run on replicated tokens, no dispatch",
			layout: Layout{TP: 8, DP: 1, ExpertWidth: 8},
			want: map[model.EmitCondition]bool{
				model.EmitTensorParallel: true,
				model.EmitExpertParallel: false,
			},
		},
		{
			what:   "expert parallel at dp 1 under the model's own sequence parallelism dispatches",
			layout: Layout{TP: 8, DP: 1, ExpertWidth: 8, ModelSequenceParallel: true},
			want: map[model.EmitCondition]bool{
				model.EmitExpertParallel: true,
			},
		},
		{
			// An MoE that reduces a different tensor per mode (Muse Spark): the TP-sharded
			// experts' own reduction, or the combine of EP partials over replicated tokens.
			what:   "tensor parallel without expert parallelism: the sharded experts reduce",
			layout: Layout{TP: 8, DP: 1, ExpertWidth: 1},
			want: map[model.EmitCondition]bool{
				model.EmitTensorParallelWithoutExpertParallel: true,
				model.EmitExpertParallelWithoutDispatch:       false,
			},
		},
		{
			what:   "expert parallel over replicated tokens: the partial outputs combine",
			layout: Layout{TP: 8, DP: 1, ExpertWidth: 8},
			want: map[model.EmitCondition]bool{
				model.EmitTensorParallelWithoutExpertParallel: false,
				model.EmitExpertParallelWithoutDispatch:       true,
			},
		},
		{
			what:   "expert parallel with a dispatch: the all-to-all combines, neither reduction runs",
			layout: Layout{TP: 8, DP: 2, ExpertWidth: 16},
			want: map[model.EmitCondition]bool{
				model.EmitExpertParallel:                      true,
				model.EmitTensorParallelWithoutExpertParallel: false,
				model.EmitExpertParallelWithoutDispatch:       false,
			},
		},
		{
			what:   "no expert parallelism at dp 2: the naive dispatch runs over the dp ranks",
			layout: Layout{TP: 4, DP: 2, ExpertWidth: 1, MoEGroupWidth: 8},
			want: map[model.EmitCondition]bool{
				model.EmitExpertParallel: true,
			},
		},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			for cond, want := range c.want {
				if got := c.layout.Emits(cond); got != want {
					t.Errorf("Emits(%q) = %v, want %v", cond, got, want)
				}
			}
		})
	}
}

// TestUnrecognizedConditionDoesNotSilentlyDropANode: returning false for an unknown
// condition would remove a cost invisibly, so the resolver must be askable about what it
// can answer.
func TestUnrecognizedConditionDoesNotSilentlyDropANode(t *testing.T) {
	l := Layout{TP: 8, ExpertWidth: 16}
	if l.Emits("invented_condition") {
		t.Error("an unknown condition should not emit a node")
	}
	if Recognizes("invented_condition") {
		t.Error("Recognizes must report that the condition is unknown, so a caller can " +
			"reject the graph rather than price it with a node missing")
	}
}

func TestNodesSpannedFollowsTheWidestGroup(t *testing.T) {
	cases := []struct {
		what        string
		gpusPerNode int
		pl          deployment.Parallelism
		rack        int
		rackDomain  bool
		want        int
	}{
		{"tp 8 on 8-GPU nodes stays on one node", 8,
			deployment.Parallelism{TP: 8, PP: 1, DP: 1}, 0, false, 1},
		{"tp 16 on 8-GPU nodes spans two", 8,
			deployment.Parallelism{TP: 16, PP: 1, DP: 1}, 0, false, 2},
		{"expert width is the widest group", 8,
			deployment.Parallelism{TP: 1, PP: 1, DP: 16, EnableExpertParallel: true}, 0, false, 2},
		{"multi-node NVLink makes a rack one domain", 4,
			deployment.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}, 72, true, 1},
		{"a rack that is not one domain still spans nodes", 4,
			deployment.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}, 72, false, 18},
		// A two-rank prefill-context group at tp=8 has a member on each of two nodes: its
		// ranks are tp apart. Expert parallelism is off, so no wider group hides it.
		{"a prefill-context group spans its tp x pcp block", 8,
			deployment.Parallelism{TP: 8, PP: 1, DP: 1, PCP: 2}, 0, false, 2},
		{"a prefill-context group inside one node stays on it", 8,
			deployment.Parallelism{TP: 2, PP: 1, DP: 1, PCP: 4}, 0, false, 1},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			s, d := scenarioWith(1, c.gpusPerNode, pool(c.pl))
			s.Cluster.GPUsPerRack = c.rack
			l, err := ResolveLayout(s, d.Pools[0], enginesettings.Settings{},
				Fabric{RackIsOneDomain: c.rackDomain}, newStubRules())
			if err != nil {
				t.Fatalf("ResolveLayout: %v", err)
			}
			if l.NodesSpanned != c.want {
				t.Errorf("NodesSpanned = %d, want %d", l.NodesSpanned, c.want)
			}
		})
	}
}

// TestCustomAllReduceRequestIsResolvedNotObeyed: the scenario states a request, and the
// layout may decline it. A prediction that assumed the request was honoured would charge
// SM occupancy that does not occur and omit NIC traffic that does.
func TestCustomAllReduceRequestIsResolvedNotObeyed(t *testing.T) {
	no := false
	cases := []struct {
		what        string
		gpusPerNode int
		tp          int
		rackDomain  bool
		wantCustom  bool
		wantReason  string
	}{
		{"tp 8 within a node: honoured", 8, 8, false, true, ""},
		{"tp 3 is an unsupported width: declined", 8, 3, false, false, "rank counts"},
		{"tp 16 spans nodes without multi-node NVLink: declined", 8, 16, false, false,
			"spans more than"},
		{"tp 16 spans nodes with multi-node NVLink: honoured", 8, 16, true, true, ""},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			s, d := scenarioWith(1, c.gpusPerNode, pool(
				deployment.Parallelism{TP: c.tp, PP: 1, DP: 1}))
			l, err := ResolveLayout(s, d.Pools[0], enginesettings.Settings{DisableCustomAllReduce: &no},
				Fabric{RackIsOneDomain: c.rackDomain}, newStubRules())
			if err != nil {
				t.Fatalf("ResolveLayout: %v", err)
			}
			if l.CustomAllReduce != c.wantCustom {
				t.Fatalf("CustomAllReduce = %v, want %v", l.CustomAllReduce, c.wantCustom)
			}
			if c.wantReason == "" {
				if len(l.Overrides) != 0 {
					t.Errorf("an honoured request should record no override, got %+v", l.Overrides)
				}
				return
			}
			if len(l.Overrides) == 0 {
				t.Fatal("a declined request must record why")
			}
			if !strings.Contains(l.Overrides[0].Reason, c.wantReason) {
				t.Errorf("reason %q does not mention %q", l.Overrides[0].Reason, c.wantReason)
			}
		})
	}
}

func TestFabricRatio(t *testing.T) {
	cases := []struct {
		intra, inter, want float64
	}{
		{450, 50, 9},  // NVLink 4 against InfiniBand NDR
		{450, 25, 18}, // the same chip behind RoCE
		{900, 900, 1}, // multi-node NVLink: no step at the boundary
		{450, 0, 1},   // no fabric stated: single node, nothing crosses
		{0, 50, 1},    // no chip bandwidth: not a division by an unknown
	}
	for _, c := range cases {
		f := Fabric{IntraNodeBwGBps: c.intra, InterNodeBwGBps: c.inter}
		if got := f.Ratio(); got != c.want {
			t.Errorf("Ratio(%v, %v) = %v, want %v", c.intra, c.inter, got, c.want)
		}
	}
}
