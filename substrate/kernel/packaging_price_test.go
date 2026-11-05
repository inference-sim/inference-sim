package latencykernel

import (
	"fmt"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// inertRules answers every engine-release question with the engine-neutral answer, so a
// price below differs only by what the packaging changes. The kernel's tests carry no
// engine model (the engine repositories test their own releases end to end).
type inertRules struct{}

func (inertRules) EngineVersion() string                                           { return "inert" }
func (inertRules) Check(rules.Input) *rules.Problems                               { return &rules.Problems{} }
func (inertRules) RuleNames() []string                                             { return nil }
func (inertRules) CheckEngineBlock(string, deployment.EngineBlock) *rules.Problems { return nil }
func (inertRules) CheckConnectorOptions(string, deployment.EngineBlock) *rules.Problems {
	return nil
}
func (inertRules) DraftTokens(deployment.EngineBlock) int          { return 0 }
func (inertRules) SpeculativeMethodAccepted(string) bool           { return false }
func (inertRules) SpeculativeMethodNames() []string                { return nil }
func (inertRules) AsyncCompatibleSpeculative(string) (bool, bool)  { return false, false }
func (inertRules) EngramOnHostByDefault() bool                     { return true }
func (inertRules) DCPCommBackendNames() map[string]bool            { return nil }
func (inertRules) MegaMoEBackend(string) bool                      { return false }
func (inertRules) SequenceParallelMoE(string, bool, int, int) bool { return false }
func (inertRules) ModelSequenceParallel(string, deployment.Parallelism, string) bool {
	return false
}
func (inertRules) CustomAllReduceSupportsWidth(int) bool                     { return true }
func (inertRules) TritonFetchWithholdsSMs(int) int                           { return 0 }
func (inertRules) DBOEngages(bool, int, bool) bool                           { return false }
func (inertRules) EffectiveAll2AllBackend(b string) string                   { return b }
func (inertRules) ResolveCacheDType(_ string, _, _ int, r string) string     { return r }
func (inertRules) WindowPagesPerRequest(model.Cache, int, int, int, int) int { return 0 }
func (inertRules) KVCacheGroups(rules.KVCacheGroupsInput) (rules.KVCacheGrouping, error) {
	return rules.KVCacheGrouping{}, nil
}
func (inertRules) CUDAGraphCaptureSizes(rules.CUDAGraphCaptureInput) (rules.CUDAGraphCapture, bool, error) {
	return rules.CUDAGraphCapture{}, false, nil
}
func (inertRules) ResolveDPLoadBalancing(deployment.Parallelism) deployment.DPLoadBalancing {
	return deployment.DPLBInternal
}

// priceOnGB200 prices one decode step of Muse Spark (a model with a decode-sharded
// projection gated by min_tokens_across_nodes) at tensor-parallel width tp over a GB200
// NVL72 pool. legacy true is the packaging before the catalog owned it: 8 GPUs to a node
// and no rack tier, built from a copy of the entry (the scenario then states 8, which the
// real entry would refuse). legacy false is the entry's own 4 per tray and 72 per rack, with
// the scenario stating neither.
func priceOnGB200(t *testing.T, tp, tokens int, legacy bool) (kernel.StepEstimate, *Kernel) {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	perNode, stated := 4, ""
	if legacy {
		perNode, stated = 8, "  gpus_per_node: 8\n"
	}
	nodes := tp / perNode
	body := fmt.Sprintf(`kind: Scenario
name: gb200-packaging
model: muse-spark-open-816b-a42b-nvfp4
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]
engine_version: "0.32.0"
cluster:
  hardware: gb200-nvl72
  fabric: ib-400g
  nodes: %d
%s---
kind: Deployment
name: tp%d
pools:
  - role: colocated
    nodes: %d
    parallel: {tp: %d, pp: 1, dp: 1}
`, nodes, stated, tp, nodes, tp)
	r, name := openRepos(t, d, body)
	r.Rules = func(string) rules.EngineRules { return inertRules{} }
	r.EngineSettings = func(*deployment.Deployment, int) (enginesettings.Settings, error) {
		return enginesettings.Settings{BlockSize: 16, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192,
			MaxModelLen: 8192}, nil
	}
	in, err := OpenInputs(name, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	if legacy {
		chip := *in.Chip
		chip.GPUsPerNode, chip.GPUsPerRack, chip.IntraRackBwGBps = 8, 0, 0
		in.Chip = &chip
	}
	k, err := New(in)
	if err != nil {
		t.Fatalf("tp %d legacy %v: %v", tp, legacy, err)
	}
	b := kernel.Batch{DecodeThreshold: 1}
	for range tokens {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096})
	}
	return k.StepTime(b), k
}

// Muse Spark's decode-sharded projections shard from min_tokens_across_nodes tokens when the
// TP group spans nodes (the catalog graph states 32). A batch of this many decode requests
// is at or above it; fewTokens is below it.
const (
	manyTokens = 64
	fewTokens  = 4
)

// ON A GB200 NVL72 POOL, TP8 PRICES THE SAME UNDER THE OLD PACKAGING AS UNDER THE ENTRY'S.
// The old computation put 8 GPUs on a node with no rack tier, so TP8 sat on one node; the
// entry's 4 per tray and 72 per rack put it on two trays of one NVLink rack. Both are NVLink at
// 900 GB/s, so the group's collectives cost the same and nothing reaches the NIC. Held at a
// batch that is not gated by min_tokens_across_nodes (see the test below for the gated one).
func TestTP8OnAnNVL72PoolPricesTheSameUnderEitherPackaging(t *testing.T) {
	old, kOld := priceOnGB200(t, 8, manyTokens, true)
	now, kNow := priceOnGB200(t, 8, manyTokens, false)
	if kOld.tpSpansHosts() || !kNow.tpSpansHosts() {
		t.Fatalf("tpSpansHosts: old packaging %v, entry's %v; want false then true (the setup changed)",
			kOld.tpSpansHosts(), kNow.tpSpansHosts())
	}
	if old.NoOverlap != now.NoOverlap || old.Overlap != now.Overlap {
		t.Errorf("TP8 prices %v/%v (no overlap/overlap) on 8 per node and %v/%v on 4 per tray in a 72-GPU rack; want equal",
			old.NoOverlap, old.Overlap, now.NoOverlap, now.Overlap)
	}
	if nic := now.PerResource[kernel.ResourceNIC]; nic != 0 {
		t.Errorf("TP8 inside one NVLink rack charged %v to the NIC", nic)
	}
}

// TP16 AND WIDER ARE THE DIFFERENCE. On 8 GPUs to a node with no rack, TP16 crosses nodes and
// its collectives price on the NIC. On the entry's 72-GPU NVLink rack the same group stays on
// NVLink and the NIC term is zero.
func TestTP16OnAnNVL72PoolLeavesTheNICForNVLink(t *testing.T) {
	old, _ := priceOnGB200(t, 16, manyTokens, true)
	now, _ := priceOnGB200(t, 16, manyTokens, false)
	if old.PerResource[kernel.ResourceNIC] == 0 {
		t.Fatal("under the old packaging TP16 charged nothing to the NIC; the contrast proves nothing")
	}
	if nic := now.PerResource[kernel.ResourceNIC]; nic != 0 {
		t.Errorf("TP16 inside one NVLink rack charged %v to the NIC", nic)
	}
	if old.NoOverlap == now.NoOverlap {
		t.Errorf("TP16 prices %v under both packagings; want the NVLink rack to differ from the NIC", old.NoOverlap)
	}
	t.Logf("TP16, %d decode tokens: %v (NIC %v) on 8 per node, %v on 4 per tray in a 72-GPU rack",
		manyTokens, old.NoOverlap, old.PerResource[kernel.ResourceNIC], now.NoOverlap)
}

// tpSpansHosts BECOMES TRUE AT TP8, AND THAT GATES min_tokens_across_nodes. The engine tests
// a TP group's hosts, not its links (tp_group_spans_nodes), so two trays of one rack span
// nodes. A step below the graph's min_tokens_across_nodes then does not shard the replicated
// projections, where on 8 per node it did. A step at or above the minimum is unaffected
// (the equal-price test above).
func TestTP8OnAnNVL72PoolGatesDecodeShardingOnMinTokensAcrossNodes(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	g, err := d.Model("muse-spark-open-816b-a42b-nvfp4")
	if err != nil {
		t.Fatal(err)
	}
	var min int
	for _, l := range g.LayerKinds {
		for _, n := range l.Nodes {
			if n.ShardedAtDecode != nil && n.ShardedAtDecode.MinTokensAcrossNodes > min {
				min = n.ShardedAtDecode.MinTokensAcrossNodes
			}
		}
	}
	if min <= fewTokens || min > manyTokens {
		t.Fatalf("the graph's min_tokens_across_nodes is %d; the test's %d and %d tokens must straddle it",
			min, fewTokens, manyTokens)
	}
	old, _ := priceOnGB200(t, 8, fewTokens, true)
	now, _ := priceOnGB200(t, 8, fewTokens, false)
	if old.NoOverlap == now.NoOverlap {
		t.Errorf("a %d-token TP8 step below min_tokens_across_nodes %d prices %v under both packagings; "+
			"want the spanning group to lose the decode sharding", fewTokens, min, old.NoOverlap)
	}
	t.Logf("TP8, %d decode tokens (min_tokens_across_nodes %d): %v on 8 per node, %v on 4 per tray",
		fewTokens, min, old.NoOverlap, now.NoOverlap)
}
