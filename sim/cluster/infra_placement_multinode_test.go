// infra_placement_multinode_test.go — BDD/TDD tests for multi-node TP placement (#1529, PR A).
// An instance may reserve its TP GPUs across more than one node of a single pool, but ONLY
// when no single Ready node in any matching pool can satisfy tpDegree (single-node placement
// is strictly preferred to spanning). See docs/plans/multi-node-tp-placement-plan.md.
package cluster

import (
	"bytes"
	"sort"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"
)

// ─── Task 1: distinctNodesForGPUs helper ────────────────────────────────────

// TestDistinctNodesForGPUs verifies the GPU-ID → distinct-node resolver returns a
// sorted, de-duplicated node list. Uses REAL GPU IDs from the manager so the test
// survives any change to the GPU-ID string format (BC-2, BC-5 support).
func TestDistinctNodesForGPUs(t *testing.T) {
	// Pool with 2 nodes × 8 GPUs so we have two distinct nodes to draw from.
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 2)})

	// Collect the two nodes' GPU IDs (nodes are index-ordered in nodesByPool).
	nodes := pm.nodesByPool["h100"]
	if len(nodes) != 2 {
		t.Fatalf("precondition: expected 2 nodes, got %d", len(nodes))
	}
	nodeA, nodeB := nodes[0], nodes[1]

	cases := []struct {
		name    string
		gpuIDs  []string
		wantLen int
		want    []string
	}{
		{
			name:    "two GPUs from the same node → one distinct node",
			gpuIDs:  []string{nodeA.GPUs[0].ID, nodeA.GPUs[1].ID},
			wantLen: 1,
			want:    []string{nodeA.ID},
		},
		{
			name:    "GPUs from two different nodes → two distinct nodes, sorted",
			gpuIDs:  []string{nodeB.GPUs[0].ID, nodeA.GPUs[7].ID},
			wantLen: 2,
			want:    sortedStrings(nodeA.ID, nodeB.ID),
		},
		{
			name:    "empty input → empty result",
			gpuIDs:  []string{},
			wantLen: 0,
			want:    []string{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := pm.distinctNodesForGPUs(tc.gpuIDs)
			if len(got) != tc.wantLen {
				t.Fatalf("distinctNodesForGPUs(%v) len = %d, want %d (got %v)", tc.gpuIDs, len(got), tc.wantLen, got)
			}
			if !sort.StringsAreSorted(got) {
				t.Errorf("result not sorted: %v", got)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Errorf("result[%d] = %q, want %q (full: %v)", i, got[i], tc.want[i], got)
				}
			}
		})
	}
}

func sortedStrings(s ...string) []string {
	out := append([]string(nil), s...)
	sort.Strings(out)
	return out
}

// captureLogWarn redirects logrus output to a buffer for the duration of fn and
// returns the captured text. Restores the previous output afterward.
func captureLogWarn(t *testing.T, fn func()) string {
	t.Helper()
	var buf bytes.Buffer
	prevOut := logrus.StandardLogger().Out
	prevLevel := logrus.GetLevel()
	logrus.SetOutput(&buf)
	logrus.SetLevel(logrus.WarnLevel)
	defer func() {
		logrus.SetOutput(prevOut)
		logrus.SetLevel(prevLevel)
	}()
	fn()
	return buf.String()
}

func countSubstr(haystack, needle string) int {
	return strings.Count(haystack, needle)
}

// ─── Task 2: two-pass PlaceInstance (single-node preferred; span as fallback) ─

// TestPlaceInstance_SingleNodeUnchanged (BC-1): when a single node fits, placement
// lands on exactly one node with tpDegree GPUs all from that node.
func TestPlaceInstance_SingleNodeUnchanged(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})
	nid, gpus, gt, err := pm.PlaceInstance("inst-0", "model-a", "H100", 4)
	if err != nil {
		t.Fatalf("PlaceInstance: %v", err)
	}
	if gt != "H100" {
		t.Errorf("matchedGPUType = %q, want H100", gt)
	}
	if len(gpus) != 4 {
		t.Fatalf("got %d GPUs, want 4", len(gpus))
	}
	if span := pm.distinctNodesForGPUs(gpus); len(span) != 1 || span[0] != nid {
		t.Errorf("expected single-node placement on %q, got span %v", nid, span)
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestPlaceInstance_SpansTwoNodes (BC-2): when no single node fits but the pool
// has aggregate capacity, the instance spans exactly the needed nodes.
func TestPlaceInstance_SpansTwoNodes(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 2)})
	_, gpus, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 16)
	if err != nil {
		t.Fatalf("PlaceInstance (span): %v", err)
	}
	if len(gpus) != 16 {
		t.Fatalf("got %d GPUs, want 16", len(gpus))
	}
	// Exactly 16 distinct GPUs (no duplicate allocation).
	uniq := map[string]struct{}{}
	for _, g := range gpus {
		uniq[g] = struct{}{}
	}
	if len(uniq) != 16 {
		t.Errorf("got %d distinct GPU IDs, want 16 (duplicate allocation?)", len(uniq))
	}
	if span := pm.distinctNodesForGPUs(gpus); len(span) != 2 {
		t.Errorf("expected 2-node span, got %v", span)
	}
	for _, n := range pm.nodesByPool["h100"] {
		if free := pm.FreeGPUCount(n.ID); free != 0 {
			t.Errorf("node %s free = %d after spanning tp=16, want 0", n.ID, free)
		}
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestPlaceInstance_SpanNeededButInsufficient (BC-3): aggregate capacity < tpDegree
// → error, and NO GPU is mutated (select-then-commit atomicity, R5).
func TestPlaceInstance_SpanNeededButInsufficient(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 2)}) // 16 GPUs total
	_, _, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 24)
	if err == nil {
		t.Fatal("expected error placing tp=24 with only 16 GPUs, got nil")
	}
	// Both nodes must remain fully free — no partial allocation.
	for _, n := range pm.nodesByPool["h100"] {
		if free := pm.FreeGPUCount(n.ID); free != 8 {
			t.Errorf("node %s free = %d after failed placement, want 8 (partial allocation!)", n.ID, free)
		}
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestPlaceInstance_PrefersSingleNodeOverSpanning (BC-1/BC-2 boundary): with a
// two-node pool and tp=8, a single node fits, so no spanning.
func TestPlaceInstance_PrefersSingleNodeOverSpanning(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 2)})
	_, gpus, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 8)
	if err != nil {
		t.Fatalf("PlaceInstance: %v", err)
	}
	if span := pm.distinctNodesForGPUs(gpus); len(span) != 1 {
		t.Errorf("tp=8 on 8-GPU nodes must NOT span; got span %v", span)
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestPlaceInstance_SingleNodeFitInLaterPoolBeatsSpanningEarlierPool (BC-2, the
// CRITICAL-1 regression guard): pool A (declared first) can only satisfy tp=16 by
// spanning; pool B (declared second) fits tp=16 on one node. With gpuType="" (match
// any), single-node placement in pool B MUST win over spanning pool A.
func TestPlaceInstance_SingleNodeFitInLaterPoolBeatsSpanningEarlierPool(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{
		newTestPool("poolA", "H100", 8, 2),  // 2×8: can only reach 16 by spanning
		newTestPool("poolB", "H200", 16, 1), // 1×16: fits tp=16 on one node
	})
	nid, gpus, gt, err := pm.PlaceInstance("inst-0", "model-a", "", 16)
	if err != nil {
		t.Fatalf("PlaceInstance: %v", err)
	}
	span := pm.distinctNodesForGPUs(gpus)
	if len(span) != 1 {
		t.Errorf("expected single-node placement (pool B), got %d-node span %v", len(span), span)
	}
	if gt != "H200" {
		t.Errorf("expected placement in poolB (H200), got gpuType %q on node %q", gt, nid)
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// ─── Task 5: one-time optimistic-latency warning (BC-6) ──────────────────────

// TestPlaceInstance_SpanningWarnsOnce (BC-6): a spanning placement emits exactly one
// stderr warning naming #1530, no matter how many instances span. Captured-log
// assertion (behavioral, survives a sync.Once refactor — does not read spanWarned).
func TestPlaceInstance_SpanningWarnsOnce(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 4)}) // 4×8 = 32 GPUs
	out := captureLogWarn(t, func() {
		// Two spanning instances (each tp=16 spans two 8-GPU nodes).
		if _, _, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 16); err != nil {
			t.Fatalf("first spanning PlaceInstance: %v", err)
		}
		if _, _, _, err := pm.PlaceInstance("inst-1", "model-a", "H100", 16); err != nil {
			t.Fatalf("second spanning PlaceInstance: %v", err)
		}
	})
	if n := countSubstr(out, "#1530"); n != 1 {
		t.Errorf("expected exactly 1 warning mentioning #1530, got %d\ncaptured:\n%s", n, out)
	}
}

// TestPlaceInstance_SingleNodeNoWarning (BC-6): single-node placements emit no
// spanning warning.
func TestPlaceInstance_SingleNodeNoWarning(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 2)})
	out := captureLogWarn(t, func() {
		if _, _, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 8); err != nil {
			t.Fatalf("PlaceInstance: %v", err)
		}
	})
	if n := countSubstr(out, "#1530"); n != 0 {
		t.Errorf("single-node placement must not warn about spanning; got %d mentions of #1530\ncaptured:\n%s", n, out)
	}
}

// TestPlaceInstance_DoesNotSpanAcrossPools (BC-2 contract guard): spanning stays
// within a single pool. Two single-node Ready pools each with 8 free GPUs cannot be
// combined into one tp=16 placement — the request must fail, and both pools stay
// fully free. Guards against a future refactor hoisting the Pass-2 `selected` slice
// out of the per-pool loop (which would wrongly combine GPUs across pools).
func TestPlaceInstance_DoesNotSpanAcrossPools(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{
		newTestPool("poolA", "H100", 8, 1),
		newTestPool("poolB", "H200", 8, 1),
	})
	// gpuType="" matches both pools; aggregate free = 16 but split across two pools.
	_, _, _, err := pm.PlaceInstance("inst-0", "model-a", "", 16)
	if err == nil {
		t.Fatal("expected error: tp=16 cannot be satisfied within any single pool (8+8 across two pools)")
	}
	for _, pool := range []string{"poolA", "poolB"} {
		for _, n := range pm.nodesByPool[pool] {
			if free := pm.FreeGPUCount(n.ID); free != 8 {
				t.Errorf("pool %s node %s free = %d after failed cross-pool placement, want 8 (no partial/cross-pool allocation)", pool, n.ID, free)
			}
		}
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestPlaceInstance_SpansThreeNodes (BC-2, multi-hop): tp=24 over 3×8 nodes spans
// exactly three nodes, and the primary node is the lowest-index contributor.
func TestPlaceInstance_SpansThreeNodes(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 3)})
	primary, gpus, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 24)
	if err != nil {
		t.Fatalf("PlaceInstance (3-node span): %v", err)
	}
	if len(gpus) != 24 {
		t.Fatalf("got %d GPUs, want 24", len(gpus))
	}
	span := pm.distinctNodesForGPUs(gpus)
	if len(span) != 3 {
		t.Errorf("expected 3-node span, got %v", span)
	}
	// Primary node is the lowest-index touched node = first in the sorted span.
	if primary != span[0] {
		t.Errorf("primary node = %q, want lowest-index touched node %q", primary, span[0])
	}
	for _, n := range pm.nodesByPool["h100"] {
		if free := pm.FreeGPUCount(n.ID); free != 0 {
			t.Errorf("node %s free = %d after tp=24 span, want 0", n.ID, free)
		}
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestPlaceInstance_DoesNotSpanOntoDrainingNode (IMPORTANT-1 guard): a Draining
// node's free GPUs must not be packed into a spanning placement.
func TestPlaceInstance_DoesNotSpanOntoDrainingNode(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 2)})
	nodes := pm.nodesByPool["h100"]
	// Drain the second node (it holds 8 free GPUs but is not Ready).
	if err := pm.DrainNode(nodes[1].ID, func() {}); err != nil {
		t.Fatalf("DrainNode: %v", err)
	}
	// Only 8 Ready free GPUs remain; tp=16 must fail rather than span onto the draining node.
	_, _, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 16)
	if err == nil {
		t.Fatal("expected error: only 8 Ready GPUs available, tp=16 must not use the draining node")
	}
	// Draining node's GPUs untouched.
	if free := pm.FreeGPUCount(nodes[1].ID); free != 8 {
		t.Errorf("draining node free = %d, want 8 (was packed into!)", free)
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// ─── Task 3: ReleaseInstance multi-node drain-callback fix (BC-4) ─────────────

// TestReleaseInstance_SpanningFiresDrainPerNode (BC-4): releasing a spanning
// instance frees GPUs on every node AND fires the drain callback for each node
// that becomes fully free. The pre-#1529 code checked only the last node touched.
func TestReleaseInstance_SpanningFiresDrainPerNode(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 2)})
	// Place a spanning instance across both nodes (tp=16).
	_, gpus, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 16)
	if err != nil {
		t.Fatalf("PlaceInstance (span): %v", err)
	}
	spanNodes := pm.distinctNodesForGPUs(gpus)
	if len(spanNodes) != 2 {
		t.Fatalf("precondition: expected 2-node span, got %v", spanNodes)
	}

	// Drain both nodes; each records its own node ID when its callback fires.
	fired := map[string]bool{}
	for _, nid := range spanNodes {
		nid := nid
		if err := pm.DrainNode(nid, func() { fired[nid] = true }); err != nil {
			t.Fatalf("DrainNode(%s): %v", nid, err)
		}
	}
	// While the instance holds GPUs, no callback should have fired yet.
	if len(fired) != 0 {
		t.Fatalf("drain callbacks fired before release: %v", fired)
	}

	// Release the instance — this should free all 16 GPUs and fire BOTH callbacks.
	if err := pm.ReleaseInstance("inst-0"); err != nil {
		t.Fatalf("ReleaseInstance: %v", err)
	}
	for _, nid := range spanNodes {
		if !fired[nid] {
			t.Errorf("drain callback did NOT fire for node %s after releasing spanning instance", nid)
		}
		if free := pm.FreeGPUCount(nid); free != 8 {
			t.Errorf("node %s free = %d after release, want 8", nid, free)
		}
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// ─── Task 4: spanning-instance cost = nodes-spanned × cost_per_hour (BC-5) ───

// TestInstanceCostPerHour (BC-5): the cost rule maps a placed GPU set + pool cost
// to distinct-nodes × cost. Exercised through real placement so the GPU set is a
// genuine placement result, not a hand-built slice.
func TestInstanceCostPerHour(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{
		{Name: "h100", GPUType: "H100", GPUsPerNode: 8, InitialNodes: 2, MinNodes: 0, MaxNodes: 2, GPUMemoryGiB: 80, CostPerHour: 10},
	})

	// Single-node placement (tp=8) → 1 × 10 = 10.
	_, single, _, err := pm.PlaceInstance("inst-single", "m", "H100", 8)
	if err != nil {
		t.Fatalf("single-node PlaceInstance: %v", err)
	}
	if got := pm.InstanceCostPerHour(single, 10); got != 10 {
		t.Errorf("single-node cost = %v, want 10 (1 node × 10)", got)
	}
	if err := pm.ReleaseInstance("inst-single"); err != nil {
		t.Fatalf("release: %v", err)
	}

	// Spanning placement (tp=16 across 2 nodes) → 2 × 10 = 20.
	_, span, _, err := pm.PlaceInstance("inst-span", "m", "H100", 16)
	if err != nil {
		t.Fatalf("spanning PlaceInstance: %v", err)
	}
	if n := pm.distinctNodesForGPUs(span); len(n) != 2 {
		t.Fatalf("precondition: expected 2-node span, got %v", n)
	}
	if got := pm.InstanceCostPerHour(span, 10); got != 20 {
		t.Errorf("spanning cost = %v, want 20 (2 nodes × 10)", got)
	}
}

// TestStartupPlacement_SpanningInstanceCost (BC-5): the observable an autoscaler
// consumes — a placed instance's CostPerHour — reflects the node span through the
// real startup placement path.
func TestStartupPlacement_SpanningInstanceCost(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "h100", GPUType: "H100", GPUsPerNode: 8, InitialNodes: 2, MinNodes: 2, MaxNodes: 2, GPUMemoryGiB: 80, CostPerHour: 10},
	}
	// TP=16 forces the single startup instance to span both 8-GPU nodes.
	cfg := deploymentForPlacement(1, false, pools, 9999)
	cfg.TP = 16
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)

	if len(cs.instances) != 1 {
		t.Fatalf("expected 1 placed instance, got %d", len(cs.instances))
	}
	inst := cs.instances[0]
	if got := inst.CostPerHour; got != 20 {
		t.Errorf("spanning instance CostPerHour = %v, want 20 (2 nodes × 10)", got)
	}
}

// TestReleaseInstance_SingleNodeStillFires (BC-4 regression): the common single-node
// drain-completion path must keep working after the multi-node fix.
func TestReleaseInstance_SingleNodeStillFires(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})
	_, _, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 8)
	if err != nil {
		t.Fatalf("PlaceInstance: %v", err)
	}
	nid := pm.nodesByPool["h100"][0].ID
	fired := false
	if err := pm.DrainNode(nid, func() { fired = true }); err != nil {
		t.Fatalf("DrainNode: %v", err)
	}
	if fired {
		t.Fatal("callback fired before release")
	}
	if err := pm.ReleaseInstance("inst-0"); err != nil {
		t.Fatalf("ReleaseInstance: %v", err)
	}
	if !fired {
		t.Error("single-node drain callback did not fire after release")
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}
