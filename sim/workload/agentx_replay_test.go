package workload

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// The hand-checked scenarios below pin the snapshot instant with an explicit trajectory
// window (min_ratio == max_ratio, which aiperf honours and which needs no RNG draw), use a
// fixed 1 s service time, and spell out the issue timeline aiperf would produce.

const sec = int64(1_000_000)

// threeTurnTrace: one stream, t = 0, 5, 10 s, each extending the previous hash prefix.
func threeTurnTrace(t *testing.T, id string) string {
	return wekaLine(t, id,
		wreq(0, 2, 128, 10, 1, 2),
		wreq(5, 1, 192, 10, 1, 2, 3),
		wreq(10, 1, 256, 10, 1, 2, 3, 4))
}

// TestAgentXClosedLoopWarmupAndRecycle: t* = 5 s (half of [0, 10] s). The warm-up sends
// turn 0 (the last request before t*) with max_tokens 1; profiling starts when it returns
// (1 s) and turn 1 fires at its offset from t* (0). Each later turn waits its end-to-start
// gap max(0, t_k - t_{k-1} - api_{k-1}) after the previous one completes: turn 2 at
// 2 + (10-5-1) = 6 s. The drained tree recycles the lane into the next root at turn 0
// (7 s), whose turn 1 waits 5-0-2 = 3 s. Nothing is sent at or after the window end
// (1 + 20 = 21 s), where turn 1 of the third tree was due.
func TestAgentXClosedLoopWarmupAndRecycle(t *testing.T) {
	c := agxCorpus(t, threeTurnTrace(t, "A"))
	h := runAgentX(t, c, agxCfg(1, 0.5, 20), fixedService(sec))
	checkIssues(t, h, []agxWant{
		{0, "A", 0, sim.PhaseWarmup},
		{1 * sec, "A", 1, sim.PhaseProfile},
		{6 * sec, "A", 2, sim.PhaseProfile},
		{7 * sec, "A", 0, sim.PhaseProfile},
		{11 * sec, "A", 1, sim.PhaseProfile},
		{16 * sec, "A", 2, sim.PhaseProfile},
		{17 * sec, "A", 0, sim.PhaseProfile},
	})
	if len(h.issued) != 7 {
		t.Errorf("%d issues, want 7", len(h.issued))
	}
	if h.issued[0].out != 1 || h.issued[1].out != 10 {
		t.Errorf("max_tokens: warm-up %d (want 1), profile %d (want 10)", h.issued[0].out, h.issued[1].out)
	}
	for i, tree := range []int{0, 0, 0, 1, 1, 1, 2} {
		if h.issued[i].tag.Tree != tree {
			t.Errorf("issue %d in tree %d, want %d", i, h.issued[i].tag.Tree, tree)
		}
	}
	if s, e := h.x.Window(); s != 1*sec || e != 21*sec {
		t.Errorf("window [%d, %d], want [1 s, 21 s]", s, e)
	}
	st := h.x.Stats()
	if st.TreesStarted != 3 || st.TreesCompleted != 2 || st.WarmupRequests != 1 || st.MainRequests != 6 {
		t.Errorf("stats %+v", st)
	}
}

// TestAgentXPromptsShareOnlyWithinATree: prompts are exactly `in` tokens; within one tree
// a turn's leading hash blocks are byte-identical to the previous turn's, and the partial
// tail is not; across tree instances of the same trace nothing is shared (cache bust).
func TestAgentXPromptsShareOnlyWithinATree(t *testing.T) {
	c := agxCorpus(t, threeTurnTrace(t, "A"))
	h := runAgentX(t, c, agxCfg(1, 0.5, 20), fixedService(sec))
	tok := func(i int) []sim.TokenID { return h.issued[i].req.InputTokens }
	for i, want := range []int{128, 192, 256, 128, 192, 256, 128} {
		if h.issued[i].in != want {
			t.Errorf("issue %d: %d prompt tokens, want %d", i, h.issued[i].in, want)
		}
	}
	same := func(a, b []sim.TokenID, n int) bool {
		for i := 0; i < n; i++ {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}
	// tree 0: turn 1 (192 = 3 blocks) and turn 2 (4 blocks) share hash ids [1,2,3].
	if !same(tok(1), tok(2), 192) {
		t.Error("turn 2 does not reuse turn 1's three hash blocks within the tree")
	}
	// warm-up turn 0 (128 tokens, hash [1,2]) shares its two blocks with turn 1.
	if !same(tok(0), tok(1), 128) {
		t.Error("turn 1 does not reuse the warm-up's two hash blocks")
	}
	// tree 1's turn 0 shares nothing with tree 0's turn 0.
	for i := 0; i < 128; i += 64 {
		if tok(0)[i] == tok(3)[i] {
			t.Errorf("block %d shared across tree instances", i/64)
		}
	}
}

// TestAgentXGlobalIdleCap: a 99 s recorded gap with nothing else in flight is cut so the
// system idles at most 10 s: turn 1 fires 10 s after turn 0 returns (1 s), at 11 s.
func TestAgentXGlobalIdleCap(t *testing.T) {
	c := agxCorpus(t, wekaLine(t, "B", wreq(0, 1, 128, 5, 1, 2), wreq(100, 1, 192, 5, 1, 2, 3)))
	h := runAgentX(t, c, agxCfg(1, 0, 60), fixedService(sec))
	checkIssues(t, h, []agxWant{
		{0, "B", 0, sim.PhaseProfile},
		{11 * sec, "B", 1, sim.PhaseProfile},
		{12 * sec, "B", 0, sim.PhaseProfile},
		{23 * sec, "B", 1, sim.PhaseProfile},
	})
	if st := h.x.Stats(); st.IdleCapJumps == 0 || st.IdleCapSkippedUs != int64(st.IdleCapJumps)*89*sec {
		t.Errorf("idle cap stats %+v", st)
	}
	// Without the cap the gap is honoured in full.
	cfg := agxCfg(1, 0, 200)
	cfg.IdleCapUs = 0
	h = runAgentX(t, c, cfg, fixedService(sec))
	checkIssues(t, h, []agxWant{{0, "B", 0, sim.PhaseProfile}, {100 * sec, "B", 1, sim.PhaseProfile}})
}

// subagentTrace: the root's turn 0 spawns a subagent (recorded 1-6 s) that the root's
// turn 1 (t = 10 s) joins.
func subagentTrace(t *testing.T, id string) string {
	return wekaLine(t, id,
		wreq(0, 1, 128, 5, 1, 2),
		wsub(1, "a1", 5000,
			wreq(1, 2, 128, 5, 50, 51),
			wreq(4, 2, 192, 5, 50, 51, 52)),
		wreq(10, 1, 192, 5, 1, 2, 3))
}

// TestAgentXSubagentJoin: the child starts when the spawning turn returns (1 s; its
// recorded start equals the branch start, offset 0), runs its own closed loop (turn 1
// after (4-1)-2 = 1 s), and the root's turn 1, gated on the join, fires the instant the
// child's last turn returns (4 s), not after its own recorded delay.
func TestAgentXSubagentJoin(t *testing.T) {
	c := agxCorpus(t, subagentTrace(t, "C"))
	h := runAgentX(t, c, agxCfg(1, 0, 30), fixedService(sec))
	checkIssues(t, h, []agxWant{
		{0, "C", 0, sim.PhaseProfile},
		{1 * sec, "C::sa:a1", 0, sim.PhaseProfile},
		{3 * sec, "C::sa:a1", 1, sim.PhaseProfile},
		{4 * sec, "C", 1, sim.PhaseProfile},
		{5 * sec, "C", 0, sim.PhaseProfile},
	})
	if h.issued[1].tag.Kind != "subagent" || h.issued[1].tag.Tree != 0 {
		t.Errorf("child tag %+v", h.issued[1].tag)
	}
	if st := h.x.Stats(); st.JoinSuspensions == 0 || st.SubagentRequests == 0 {
		t.Errorf("stats %+v", st)
	}
}

// flatTrace: a worker request w0 (t = 0.5 s) forks the root's context while the root's
// turn 0 is still running, so chain detection splits it into a flattened worker chain.
// Its branch started before turn 0's recorded end, so it spawns as turn 0 goes on the wire,
// 0.5 s later; the root's turn 1 joins it and is also ordered after it by the replay
// barrier (w0 ended before turn 1's recorded start).
func flatTrace(t *testing.T, id string) string {
	return wekaLine(t, id,
		wreq(0, 2, 128, 5, 1, 2),
		wreq(0.5, 1, 192, 5, 1, 2, 9),
		wreq(3, 1, 192, 5, 1, 2, 3))
}

func TestAgentXFlatChainOverlapSpawnAndBarrier(t *testing.T) {
	c := agxCorpus(t, flatTrace(t, "F"))
	tr := c.Traces[0]
	if len(tr.convs) != 2 || tr.convs[1].sessionID != "F::aux:000" {
		t.Fatalf("chain split: %d conversations", len(tr.convs))
	}
	if p := tr.convs[0].turns[1].preds; len(p) != 1 || p[0] != (agxRef{1, 0}) {
		t.Fatalf("root turn 1 predecessors %v, want the worker's turn 0", p)
	}
	h := runAgentX(t, c, agxCfg(1, 0, 30), fixedService(sec))
	checkIssues(t, h, []agxWant{
		{0, "F", 0, sim.PhaseProfile},
		{sec / 2, "F::aux:000", 0, sim.PhaseProfile},
		{3 * sec / 2, "F", 1, sim.PhaseProfile},
		{5 * sec / 2, "F", 0, sim.PhaseProfile},
	})
	if h.issued[1].tag.Kind != "flat" {
		t.Errorf("worker tag %+v", h.issued[1].tag)
	}
}
