// infra_placement_multinode_test.go — BDD/TDD tests for multi-node TP placement (#1529, PR A).
// An instance may occupy whole nodes across a single pool for multi-node tensor parallelism,
// but ONLY when tpDegree > GPUsPerNode and tpDegree % GPUsPerNode == 0 (an evenly-divisible
// whole-node span) and no single Ready node fits. Single-node placement is strictly preferred
// to spanning; the fragmentation case (tpDegree ≤ GPUsPerNode) never spans. See PR #1537 / #1529.
package cluster

import (
	"bytes"
	"sort"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
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

// TestDistinctNodesForGPUs_UnknownIDLogsAndSkips (MIN-2, I-2 branch): an unknown GPU
// ID resolves to no node and emits an error log — never silently dropped.
func TestDistinctNodesForGPUs_UnknownIDLogsAndSkips(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})
	out := captureLogWarn(t, func() {
		nodes := pm.distinctNodesForGPUs([]string{"does-not-exist-gpu-0"})
		if len(nodes) != 0 {
			t.Errorf("unknown GPU ID should resolve to 0 nodes, got %v", nodes)
		}
	})
	if countSubstr(out, "not found in index") == 0 {
		t.Errorf("expected an error log for the unknown GPU ID; captured:\n%s", out)
	}
}

// TestInstanceCostPerHour_EmptySetLogsAndReturnsZero (MIN-2, I-3 branch): an empty GPU
// set logs a placement-path bug and returns 0 (not a plausible 1× cost).
func TestInstanceCostPerHour_EmptySetLogsAndReturnsZero(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})
	var got float64
	out := captureLogWarn(t, func() {
		got = pm.InstanceCostPerHour(nil, 10)
	})
	if got != 0 {
		t.Errorf("InstanceCostPerHour(nil) = %v, want 0", got)
	}
	if countSubstr(out, "placement-path bug") == 0 {
		t.Errorf("expected an error log for the empty GPU set; captured:\n%s", out)
	}
}

// TestPlaceInstance_UnsatisfiableTPReportsShapeError (IMP-1): a tpDegree that exceeds
// gpus_per_node but is not a whole multiple of it can never place on that pool shape;
// the error must say so (distinct from a transient capacity shortfall).
func TestPlaceInstance_UnsatisfiableTPReportsShapeError(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 4)}) // plenty of nodes
	_, _, _, err := pm.PlaceInstance("inst-0", "m", "H100", 12)          // 12 > 8, 12 % 8 = 4
	if err == nil {
		t.Fatal("tp=12 on 8-GPU nodes must fail (not a whole multiple)")
	}
	if !strings.Contains(err.Error(), "unsatisfiable") {
		t.Errorf("error should flag the request as structurally unsatisfiable, got: %v", err)
	}
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
	// T-3: two sequential spanning placements are the case most likely to expose a
	// double-allocation bug — assert GPU-level conservation.
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
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
// fully free. Guards against a future refactor hoisting the Pass-2 `selectedNodes`
// slice out of the per-pool loop (which would wrongly combine nodes across pools).
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
	// Primary is the lowest-INDEX touched node; span is lexicographically sorted, so
	// asserting primary == span[0] would encode a false "index order == lex order"
	// invariant (diverges at ≥10 nodes, e.g. "h100-10" < "h100-2"). primary is
	// logging-only; assert it is one of the spanned nodes.
	found := false
	for _, n := range span {
		if n == primary {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("primary node %q not among spanned nodes %v", primary, span)
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

// TestPlaceInstance_FragmentationDoesNotSpan (I-1 guard): when tpDegree ≤ GPUsPerNode
// but the pool is fragmented so no single node has room, the instance must NOT span
// (that would fabricate an asymmetric TP group). It stays unplaceable/pending.
// Setup: 2×4-GPU nodes. Place a tp=3 on each node (Pass 1 first-fits node-0, then
// node-1 — leaving 1 free GPU on each). A tp=2 then fits on neither node (1 free each);
// since tp=2 ≤ GPUsPerNode(4) Pass 2 is ineligible, so it must NOT be packed as 1+1.
func TestPlaceInstance_FragmentationDoesNotSpan(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 4, 2)})
	// tp=3 on node-0 (1 free left), tp=3 on node-1 (1 free left).
	if _, _, _, err := pm.PlaceInstance("a", "m", "H100", 3); err != nil {
		t.Fatalf("place a: %v", err)
	}
	if _, _, _, err := pm.PlaceInstance("b", "m", "H100", 3); err != nil {
		t.Fatalf("place b: %v", err)
	}
	// Confirm the fragmentation: each node has exactly 1 free GPU.
	for _, n := range pm.nodesByPool["h100"] {
		if free := pm.FreeGPUCount(n.ID); free != 1 {
			t.Fatalf("precondition: node %s free = %d, want 1 (fragmented)", n.ID, free)
		}
	}
	// tp=2 fits on neither node (1 free each). tp=2 ≤ GPUsPerNode(4) → Pass 2 ineligible.
	_, gpus, _, err := pm.PlaceInstance("c", "m", "H100", 2)
	if err == nil {
		t.Fatalf("tp=2 must NOT span a fragmented 1+1-free pool (would be an asymmetric 1+1 TP group ≤ GPUsPerNode); got placement %v", gpus)
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestPlaceInstance_NonDivisibleSpanRejected (I-1 guard): tpDegree > GPUsPerNode but
// NOT an exact multiple → no whole-node span (ranks would not divide evenly). Setup:
// 3×8-GPU nodes, tp=20 (20 > 8 but 20 % 8 = 4). Must fail rather than span.
func TestPlaceInstance_NonDivisibleSpanRejected(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 3)})
	_, _, _, err := pm.PlaceInstance("inst-0", "m", "H100", 20)
	if err == nil {
		t.Fatal("tp=20 on 8-GPU nodes must NOT span (20 % 8 != 0 — uneven ranks per node)")
	}
	// No node touched.
	for _, n := range pm.nodesByPool["h100"] {
		if free := pm.FreeGPUCount(n.ID); free != 8 {
			t.Errorf("node %s free = %d after rejected non-divisible span, want 8", n.ID, free)
		}
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestPlaceInstance_SpanRequiresFullyFreeNodes (T-4, BC-3 atomicity with pre-allocated
// GPUs): a partially-used node cannot contribute to a whole-node span. Setup: 2×8
// nodes, place a tp=2 on node-0 (now partially used), then a tp=16 must fail (only one
// fully-free node remains) with NO partial allocation.
func TestPlaceInstance_SpanRequiresFullyFreeNodes(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 2)})
	// Occupy 2 GPUs on the first node.
	if _, _, _, err := pm.PlaceInstance("small", "m", "H100", 2); err != nil {
		t.Fatalf("place small: %v", err)
	}
	freeBefore := map[string]int{}
	for _, n := range pm.nodesByPool["h100"] {
		freeBefore[n.ID] = pm.FreeGPUCount(n.ID)
	}
	// tp=16 needs two fully-free nodes; only one remains fully free → must fail.
	if _, _, _, err := pm.PlaceInstance("big", "m", "H100", 16); err == nil {
		t.Fatal("tp=16 must fail: only one fully-free node available (the other is partially used)")
	}
	// No GPU state changed by the failed attempt.
	for _, n := range pm.nodesByPool["h100"] {
		if got := pm.FreeGPUCount(n.ID); got != freeBefore[n.ID] {
			t.Errorf("node %s free changed from %d to %d after failed span (partial mutation!)", n.ID, freeBefore[n.ID], got)
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

// TestStartupPlacement_UnplacedEmitsOneSummaryWarning: when instances cannot be
// placed at startup, NewClusterSimulator emits exactly ONE summary warning (not one
// per instance), and it carries the actionable first error — here the structurally-
// unsatisfiable shape message (tp=12 is not a whole multiple of gpus_per_node=8),
// not a generic capacity message. Guards the "first error, once" behavior.
func TestStartupPlacement_UnplacedEmitsOneSummaryWarning(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "h100", GPUType: "H100", GPUsPerNode: 8, InitialNodes: 4, MinNodes: 4, MaxNodes: 4, GPUMemoryGiB: 80, CostPerHour: 10},
	}
	cfg := deploymentForPlacement(3, false, pools, 9999)
	cfg.TP = 12 // 12 > 8 and 12 % 8 != 0 → structurally unsatisfiable on this pool
	var cs *ClusterSimulator
	out := captureLogWarn(t, func() {
		cs = NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	})
	if len(cs.instances) != 0 {
		t.Fatalf("expected 0 placed instances (tp=12 unsatisfiable), got %d", len(cs.instances))
	}
	// Exactly one summary line, not one per instance.
	if n := countSubstr(out, "not placed at startup"); n != 1 {
		t.Errorf("expected exactly 1 summary warning, got %d\ncaptured:\n%s", n, out)
	}
	// It reports the count across all instances.
	if countSubstr(out, "3 of 3") == 0 {
		t.Errorf("summary should report 3 of 3 unplaced; captured:\n%s", out)
	}
	// The reported error is the actionable structural one, not a generic capacity error.
	if countSubstr(out, "unsatisfiable") == 0 {
		t.Errorf("summary should carry the structurally-unsatisfiable error; captured:\n%s", out)
	}
}

// TestDeferredPlacement_SpanningInstanceCost (T-1, BC-5): the deferred NodeReadyEvent
// path computes spanning cost too. InitialNodes=0 defers the tp=16 instance; once two
// nodes are Ready, RetryPendingInstances places it spanning both, and its CostPerHour
// reflects the 2-node span. Guards infra_lifecycle_event.go from silently reverting to
// the flat pool cost.
func TestDeferredPlacement_SpanningInstanceCost(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "h100", GPUType: "H100", GPUsPerNode: 8, InitialNodes: 0, MinNodes: 0, MaxNodes: 2, GPUMemoryGiB: 80, CostPerHour: 10},
	}
	cfg := deploymentForPlacement(1, false, pools, 9999)
	cfg.TP = 16
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	if len(cs.instances) != 0 {
		t.Fatalf("precondition: expected 0 instances before nodes ready (InitialNodes=0), got %d", len(cs.instances))
	}

	// Provision and ready two nodes; the second NodeReadyEvent makes the tp=16 span placeable.
	for i := 0; i < 2; i++ {
		node, _ := cs.placement.ProvisionNode("h100", 0)
		(&NodeReadyEvent{timestamp: 0, nodeID: node.ID}).Execute(cs)
	}
	if len(cs.instances) != 1 {
		t.Fatalf("expected 1 deferred instance placed after 2 nodes ready, got %d", len(cs.instances))
	}
	if got := cs.instances[0].CostPerHour; got != 20 {
		t.Errorf("deferred spanning instance CostPerHour = %v, want 20 (2 nodes × 10)", got)
	}
	if err := cs.placement.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestAutoscalerScaleUp_SpanningInstanceCost (T-2, BC-5): the DirectActuator.scaleUp
// path computes spanning cost too. A scale-up of a TP=16 variant onto a pool of 8-GPU
// nodes spans two nodes; the resulting instance's CostPerHour = 2 × pool cost.
func TestAutoscalerScaleUp_SpanningInstanceCost(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "h100", GPUType: "H100", GPUsPerNode: 8, InitialNodes: 2, MinNodes: 2, MaxNodes: 2, GPUMemoryGiB: 80, CostPerHour: 10},
	}
	// TP=16 matches the scale-up variant (realizable config); the startup instance
	// spans both nodes, so release it and clear the list to give scale-up a clean pool.
	cfg := deploymentForPlacement(1, false, pools, 9999)
	cfg.Model = "test-model"
	cfg.TP = 16
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	for _, inst := range cs.instances {
		if err := cs.placement.ReleaseInstance(inst.ID()); err != nil {
			t.Fatalf("release startup instance: %v", err)
		}
	}
	cs.instances = []*InstanceSimulator{}

	actuator := NewDirectActuator(cs)
	if err := actuator.Apply([]ScaleDecision{
		{ModelID: "test-model", Variant: NewVariantSpec("H100", 16), Delta: 1},
	}); err != nil {
		t.Fatalf("scale-up Apply: %v", err)
	}
	if len(cs.instances) != 1 {
		t.Fatalf("expected 1 instance after scale-up, got %d", len(cs.instances))
	}
	if got := cs.instances[0].CostPerHour; got != 20 {
		t.Errorf("autoscaler spanning instance CostPerHour = %v, want 20 (2 nodes × 10)", got)
	}
	if err := cs.placement.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// TestRetryPendingInstances_SpanningPlacement (T-5): the pending → NodeReady →
// RetryPendingInstances integration produces a correct whole-node span. A tp=16
// instance is added pending (no capacity), then two nodes are readied; the retry must
// place it across both nodes with 16 GPUs.
func TestRetryPendingInstances_SpanningPlacement(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{
		{Name: "h100", GPUType: "H100", GPUsPerNode: 8, InitialNodes: 0, MinNodes: 0, MaxNodes: 2, GPUMemoryGiB: 80},
	})
	pm.AddPending("inst-0", "model-a", "H100", 16, sim.SimConfig{})

	// First node ready: still cannot span (needs 2 whole nodes).
	n0, _ := pm.ProvisionNode("h100", 0)
	if err := pm.MarkNodeReady(n0.ID); err != nil {
		t.Fatalf("MarkNodeReady n0: %v", err)
	}
	if placed := pm.RetryPendingInstances(); len(placed) != 0 {
		t.Fatalf("tp=16 must stay pending with only 1 node ready, got %d placed", len(placed))
	}

	// Second node ready: now the span is placeable.
	n1, _ := pm.ProvisionNode("h100", 0)
	if err := pm.MarkNodeReady(n1.ID); err != nil {
		t.Fatalf("MarkNodeReady n1: %v", err)
	}
	placed := pm.RetryPendingInstances()
	if len(placed) != 1 {
		t.Fatalf("expected 1 instance placed after 2nd node ready, got %d", len(placed))
	}
	if len(placed[0].gpuIDs) != 16 {
		t.Errorf("placed spanning instance has %d GPUs, want 16", len(placed[0].gpuIDs))
	}
	if span := pm.distinctNodesForGPUs(placed[0].gpuIDs); len(span) != 2 {
		t.Errorf("expected 2-node span, got %v", span)
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
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
