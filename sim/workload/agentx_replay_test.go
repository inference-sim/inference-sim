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

// TestAgentXPromptsShareOnlyWithinATree: a prompt is its message list rendered through the
// chat template -- BOS, a role header per message, the tree's marker at the head of the first
// user message, the generation prompt -- around exactly the recorded `in` content tokens.
// Within one tree, turn k+1 re-sends turn k's messages, so it shares all of turn k's prompt
// but the generation prompt; across tree instances of the same trace the marker differs,
// so the prompts share only BOS and the first header.
func TestAgentXPromptsShareOnlyWithinATree(t *testing.T) {
	c := agxCorpus(t, threeTurnTrace(t, "A"))
	h := runAgentX(t, c, agxCfg(1, 0.5, 20), fixedService(sec))
	tok := func(i int) []sim.TokenID { return h.issued[i].req.InputTokens }
	common := func(a, b []sim.TokenID) int {
		n := 0
		for n < len(a) && n < len(b) && a[n] == b[n] {
			n++
		}
		return n
	}
	// Turns 0, 1, 2 hold 1, 2 and 3 user messages: [h1 h2], [h3], [h4] (each new block is
	// user content: the trailing-user rule hands back the one assistant block).
	for i, w := range []struct{ in, msgs int }{{128, 1}, {192, 2}, {256, 3}, {128, 1}, {192, 2}, {256, 3}, {128, 1}} {
		if got := agxContentLen(tok(i)); got != w.in {
			t.Errorf("issue %d: %d content tokens, want %d", i, got, w.in)
		}
		if want := w.in + 2 + w.msgs + agxMarkerTokens; len(tok(i)) != want {
			t.Errorf("issue %d: %d prompt tokens, want %d", i, len(tok(i)), want)
		}
	}
	if n := common(tok(1), tok(2)); n != len(tok(1))-1 {
		t.Errorf("turn 2 shares %d tokens with turn 1, want all %d but its generation prompt", n, len(tok(1))-1)
	}
	if n := common(tok(0), tok(1)); n != len(tok(0))-1 {
		t.Errorf("turn 1 shares %d tokens with the warm-up's turn 0, want %d", n, len(tok(0))-1)
	}
	if n := common(tok(0), tok(3)); n != 2 {
		t.Errorf("two tree instances share %d leading tokens, want 2 (BOS and the user header)", n)
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

// TestAgentXPerTraceIdleCap is the per-trace idle watchdog (aiperf
// --trace-idle-gap-cap-seconds): with two lanes and the global cap off, tree A idles 100 s
// between turn 0 and turn 1 while tree B stays active (3 s cadence). The per-trace cap (20 s)
// fires on A alone -- A's turn 1 is pulled from 100 s to 21 s (A#0 returned at 1 s + 20 s) --
// while B's recorded cadence is untouched. At concurrency >= 2 the global cap can never do
// this: tree B is always in flight across A's gap, so the system is never system-idle.
func TestAgentXPerTraceIdleCap(t *testing.T) {
	a := wekaLine(t, "A", wreq(0, 1, 128, 5, 1, 2), wreq(100, 1, 192, 5, 1, 2, 3))
	b := wekaLine(t, "B",
		wreq(0, 1, 128, 5, 10, 11),
		wreq(3, 1, 160, 5, 10, 11, 12),
		wreq(6, 1, 192, 5, 10, 11, 12, 13))
	c := agxCorpus(t, a, b)

	// Baseline: no per-trace cap, global cap off -> A's 100 s gap is honoured in full, so
	// A#1 falls outside the 25 s window and never sends.
	base := agxCfg(2, 0, 25)
	base.IdleCapUs = 0
	hb := runAgentX(t, c, base, fixedService(sec))
	checkIssues(t, hb, []agxWant{
		{0, "A", 0, sim.PhaseProfile},
		{0, "B", 0, sim.PhaseProfile},
		{3 * sec, "B", 1, sim.PhaseProfile},
		{6 * sec, "B", 2, sim.PhaseProfile},
		{7 * sec, "A", 0, sim.PhaseProfile}, // A's tree drained and recycled
	})
	if st := hb.x.Stats(); st.TraceIdleCapJumps != 0 {
		t.Errorf("baseline must not fire the per-trace cap: %+v", st)
	}

	// With the per-trace cap, A#1 is pulled in to 21 s; B's cadence is byte-identical.
	cfg := agxCfg(2, 0, 25)
	cfg.IdleCapUs = 0
	cfg.TraceIdleCapUs = 20 * sec
	h := runAgentX(t, c, cfg, fixedService(sec))
	checkIssues(t, h, []agxWant{
		{0, "A", 0, sim.PhaseProfile},
		{0, "B", 0, sim.PhaseProfile},
		{3 * sec, "B", 1, sim.PhaseProfile},
		{6 * sec, "B", 2, sim.PhaseProfile},
		{7 * sec, "A", 0, sim.PhaseProfile},
		{21 * sec, "A", 1, sim.PhaseProfile}, // pulled in by the per-trace cap (1 + 20)
		{22 * sec, "B", 0, sim.PhaseProfile}, // B's tree recycles after its window
	})
	if st := h.x.Stats(); st.TraceIdleCapJumps == 0 {
		t.Errorf("the per-trace cap should have fired: %+v", st)
	}
	// Tree B's turns are identical with and without the cap: the watchdog advanced only A.
	for _, w := range []agxWant{
		{0, "B", 0, sim.PhaseProfile}, {3 * sec, "B", 1, sim.PhaseProfile}, {6 * sec, "B", 2, sim.PhaseProfile},
	} {
		if !agxHasIssue(hb, w) || !agxHasIssue(h, w) {
			t.Errorf("tree B turn %d at %d s must be unchanged by the per-trace cap", w.turn, w.at/sec)
		}
	}
}

// TestAgentXPerTraceIdleCapArmsInitialWork: the per-trace cap must be armed at profiling
// start (aiperf's observe_idle_root after the initial timers are installed), so a tree whose
// FIRST profiling request -- not a follow-up after a return -- is scheduled past the cap is
// pulled in. t* = 1 s (ratio 0.01 of the 100 s span) makes turn 0 the warmup primer and
// turn 1 the first profiling request, scheduled 99 s out.
func TestAgentXPerTraceIdleCapArmsInitialWork(t *testing.T) {
	c := agxCorpus(t, wekaLine(t, "A", wreq(0, 1, 128, 5, 1, 2), wreq(100, 1, 192, 5, 1, 2, 3)))

	// Without the cap, profiling starts at 1 s but the first request (turn 1) is 99 s out, so
	// nothing is issued inside a 50 s window.
	base := agxCfg(1, 0.01, 50)
	base.IdleCapUs = 0
	hb := runAgentX(t, c, base, fixedService(sec))
	checkIssues(t, hb, []agxWant{{0, "A", 0, sim.PhaseWarmup}})
	if st := hb.x.Stats(); st.TraceIdleCapJumps != 0 {
		t.Errorf("baseline must not fire the per-trace cap: %+v", st)
	}

	// With a 20 s cap, the first profiling request is pulled in to 21 s (warmup returns at
	// 1 s, + 20 s cap) -- armed at profiling start, before any return.
	cfg := agxCfg(1, 0.01, 50)
	cfg.IdleCapUs = 0
	cfg.TraceIdleCapUs = 20 * sec
	h := runAgentX(t, c, cfg, fixedService(sec))
	checkIssues(t, h, []agxWant{
		{0, "A", 0, sim.PhaseWarmup},
		{21 * sec, "A", 1, sim.PhaseProfile}, // first profiling request, armed at start
		{22 * sec, "A", 0, sim.PhaseProfile},
	})
	if st := h.x.Stats(); st.TraceIdleCapJumps == 0 {
		t.Errorf("the per-trace cap should have armed the initial timer: %+v", st)
	}
}

// sixTurnTrace is one stream of six turns, each extending the prior hash prefix, used to
// exercise an accelerated warmup that stops mid-stream.
func sixTurnTrace(t *testing.T, id string) string {
	return wekaLine(t, id,
		wreq(0, 1, 100, 5, 1),
		wreq(1, 1, 110, 5, 1, 2),
		wreq(2, 1, 120, 5, 1, 2, 3),
		wreq(3, 1, 130, 5, 1, 2, 3, 4),
		wreq(4, 1, 140, 5, 1, 2, 3, 4, 5),
		wreq(5, 1, 150, 5, 1, 2, 3, 4, 5, 6))
}

// TestAgentXAcceleratedWarmupAdvancesTurns: the request-bounded cache-pressure warmup
// (aiperf --warmup-requests-per-lane) advances each live trajectory through its REAL next
// turns with zero idle delay and max_tokens 1 -- not a repeated primer -- and profiling
// resumes from the advanced position, not the snapshot. t* = 0 (ratio 0), quota 2: accel
// sends turns 0 and 1, defers at turn 2, and profiling resumes at turn 2.
func TestAgentXAcceleratedWarmupAdvancesTurns(t *testing.T) {
	c := agxCorpus(t, sixTurnTrace(t, "R"))
	cfg := agxCfg(1, 0, 60)
	cfg.WarmupRequestsPerLane = 2
	h := runAgentX(t, c, cfg, fixedService(sec))

	// The two accelerated turns are the stream's real turns 0 and 1 (growing prompts), warmup
	// phase, one token each -- not the same primer twice.
	checkIssues(t, h, []agxWant{
		{0, "R", 0, sim.PhaseWarmup},
		{1 * sec, "R", 1, sim.PhaseWarmup},
		{4 * sec, "R", 2, sim.PhaseProfile}, // profiling resumes from the ADVANCED turn 2
		{5 * sec, "R", 3, sim.PhaseProfile},
	})
	if h.issued[0].in >= h.issued[1].in {
		t.Errorf("accelerated turns must advance (turn 1 prompt %d should exceed turn 0 prompt %d)", h.issued[1].in, h.issued[0].in)
	}
	for i := 0; i < 2; i++ {
		if h.issued[i].out != 1 {
			t.Errorf("accelerated warmup turn %d max_tokens %d, want 1", i, h.issued[i].out)
		}
	}
	// The first measured (profiling) turn is the advanced turn 2, never the snapshot turn 0.
	var firstProfile int
	for _, is := range h.issued {
		if is.phase == sim.PhaseProfile {
			firstProfile = is.tag.Turn
			break
		}
	}
	if firstProfile != 2 {
		t.Errorf("profiling resumed at turn %d, want the advanced turn 2", firstProfile)
	}
	if st := h.x.Stats(); st.WarmupRequests != 2 {
		t.Errorf("warmup requests %d, want 2 accelerated extras (no baseline primer at t*=0)", st.WarmupRequests)
	}
}

// TestAgentXAcceleratedWarmupNoBaselinePrimer: a lane whose first request is at t* has no
// baseline primer, but the accelerated cache-pressure stage still runs (aiperf
// _execute_warmup: `if not prepared: _start_accelerated_warmup`). Profiling does not start
// until the quota is spent.
func TestAgentXAcceleratedWarmupNoBaselinePrimer(t *testing.T) {
	c := agxCorpus(t, threeTurnTrace(t, "A")) // ratio 0 -> turn 0 is AT t*, nothing precedes it
	cfg := agxCfg(1, 0, 20)
	cfg.WarmupRequestsPerLane = 2
	h := runAgentX(t, c, cfg, fixedService(sec))
	checkIssues(t, h, []agxWant{
		{0, "A", 0, sim.PhaseWarmup}, // accelerated, not a baseline primer
		{1 * sec, "A", 1, sim.PhaseWarmup},
	})
	st := h.x.Stats()
	if st.WarmupRequests != 2 {
		t.Errorf("warmup requests %d, want 2 (accelerated extras, no baseline primer)", st.WarmupRequests)
	}
	if ps, _ := h.x.Window(); ps == 0 {
		t.Errorf("profiling must start after the accelerated stage, not at time 0")
	}
}

// TestAgentXAcceleratedWarmupPerLane: the quota is per lane. Two lanes, quota 2 each, so four
// accelerated warmup requests in all, and a lane with no baseline primer still gets its quota.
func TestAgentXAcceleratedWarmupPerLane(t *testing.T) {
	a := wekaLine(t, "A", wreq(0, 1, 128, 5, 1, 2), wreq(10, 1, 192, 5, 1, 2, 3), wreq(20, 1, 256, 5, 1, 2, 3, 4))
	b := wekaLine(t, "B", wreq(0, 1, 128, 5, 10, 11), wreq(10, 1, 192, 5, 10, 11, 12), wreq(20, 1, 256, 5, 10, 11, 12, 13))
	c := agxCorpus(t, a, b)
	cfg := agxCfg(2, 0, 900)
	cfg.WarmupRequestsPerLane = 2
	h := runAgentX(t, c, cfg, fixedService(sec))
	warm := 0
	for _, is := range h.issued {
		if is.phase == sim.PhaseWarmup {
			warm++
		}
	}
	if warm != 4 {
		t.Errorf("%d warmup requests issued, want 4 (2 per lane x 2 lanes)", warm)
	}
	if st := h.x.Stats(); st.WarmupRequests != 4 {
		t.Errorf("WarmupRequests stat %d, want 4", st.WarmupRequests)
	}
}

// agxHasIssue reports whether the harness issued a send matching w (stream, turn, time).
func agxHasIssue(h *agxHarness, w agxWant) bool {
	for _, is := range h.issued {
		if is.at == w.at && is.tag.Stream == w.stream && is.tag.Turn == w.turn {
			return true
		}
	}
	return false
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
