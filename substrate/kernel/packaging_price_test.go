package latencykernel

import (
	"fmt"
	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/enginefacts"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/load"
)

// inertRules answers every engine-release question with the engine-neutral answer, so a
// price below differs only by what the packaging changes. The kernel's tests carry no
// engine model (the engine repositories test their own releases end to end).
type inertRules struct{}

func (inertRules) EngineVersion() string                                              { return "inert" }
func (inertRules) CheckEngineBlock(string, deployment.EngineBlock) *validate.Problems { return nil }
func (inertRules) CheckConnectorOptions(string, deployment.EngineBlock) *validate.Problems {
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
func (inertRules) KVCacheGroups(enginefacts.KVCacheGroupsInput) (enginefacts.KVCacheGrouping, error) {
	return enginefacts.KVCacheGrouping{}, nil
}
func (inertRules) CUDAGraphCaptureSizes(enginefacts.CUDAGraphCaptureInput) (enginefacts.CUDAGraphCapture, bool, error) {
	return enginefacts.CUDAGraphCapture{}, false, nil
}
func (inertRules) ResolveDPLoadBalancing(deployment.Parallelism) deployment.DPLoadBalancing {
	return deployment.DPLBInternal
}

// spansHosts reports whether a pool's tensor-parallel group sits on more than one node (host),
// whatever links join its ranks: the packaging fact the engine reads to decide its layout.
func spansHosts(k *Kernel) bool {
	return k.layout.GPUsPerNode > 0 && max(k.layout.TP, 1) > k.layout.GPUsPerNode
}

// priceOnGB200 prices one decode step of Muse Spark at tensor-parallel width tp over a GB200
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
	r.Rules = func(string) enginefacts.EngineRules { return inertRules{} }
	r.DraftModel, r.Lower = noDrafter, lowerTo(8192, 16, 0)
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

// The decode batches the packaging tests price: a few requests and many.
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
	if spansHosts(kOld) || !spansHosts(kNow) {
		t.Fatalf("tpSpansHosts: old packaging %v, entry's %v; want false then true (the setup changed)",
			spansHosts(kOld), spansHosts(kNow))
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
