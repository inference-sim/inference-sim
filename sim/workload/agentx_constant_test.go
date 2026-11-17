package workload

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// chainTrace is a root whose turns run back to back: every turn starts the instant the
// previous one ends at 1 s of service, so the recorded end-to-start delay is 0.
func chainTrace(t *testing.T, id string, turns int) string {
	reqs := make([]map[string]any, turns)
	for k := range reqs {
		reqs[k] = wreq(float64(k), 1, 128+64*k, 5, 1, 2, int64(3+k))
	}
	return wekaLine(t, id, reqs...)
}

func constantCfg(slots int, durS int64, c AgentXConstantLoad) AgentXReplayConfig {
	return AgentXReplayConfig{Concurrency: slots, Seed: 42, DurationUs: durS * sec, GraceUs: 30 * sec, Constant: &c}
}

// rootsInFlight counts the root requests on the wire at time at.
func rootsInFlight(h *agxHarness, at int64) int {
	n := 0
	for _, i := range h.issued {
		if i.tag.Kind == "root" && i.at <= at && (i.doneAt < 0 || at < i.doneAt) {
			n++
		}
	}
	return n
}

// liveSessions counts the root sessions at time at: started, and not yet past the return of the
// root's final turn (finalTurn maps a trace to its last root turn).
func liveSessions(h *agxHarness, finalTurn map[string]int, at int64) int {
	started, ended := map[int]bool{}, map[int]bool{}
	for _, i := range h.issued {
		if i.tag.Kind != "root" {
			continue
		}
		if i.tag.Turn == 0 && i.at <= at {
			started[i.tag.Tree] = true
		}
		if i.tag.Turn == finalTurn[i.tag.TraceID] && i.doneAt >= 0 && i.doneAt <= at {
			ended[i.tag.Tree] = true
		}
	}
	return len(started) - len(ended)
}

// probes are the half-second marks of the measured window, which fall between the
// integer-second hand-offs of a 1 s service time.
func probes(h *agxHarness, durS int64) []int64 {
	start, _ := h.x.Window()
	var out []int64
	for at := start + sec/2; at < start+durS*sec; at += sec {
		out = append(out, at)
	}
	return out
}

// Constant load holds exactly N root requests in flight for the whole measured window while
// sessions are short enough to recycle many times: a session ending hands its slot to the next
// root at the same instant (the CONCURRENCY_BURST interval is zero).
func TestConstantLoadHoldsExactlyNInFlight(t *testing.T) {
	const n = 8
	c := agxCorpus(t, chainTrace(t, "A", 3), chainTrace(t, "B", 2), chainTrace(t, "C", 4))
	h := runAgentX(t, c, constantCfg(n, 40, AgentXConstantLoad{}), fixedService(sec))
	ps := probes(h, 40)
	if len(ps) < 30 {
		t.Fatalf("probed only %d instants", len(ps))
	}
	for _, at := range ps {
		if got := rootsInFlight(h, at); got != n {
			t.Fatalf("at %.1fs root requests in flight = %d, want %d", float64(at)/1e6, got, n)
		}
	}
	if st := h.x.Stats(); st.TreesStarted <= n {
		t.Errorf("no slot was ever recycled: %d trees for %d slots", st.TreesStarted, n)
	}
	for _, i := range h.issued {
		if i.phase == sim.PhaseWarmup {
			t.Errorf("a warmup request %+v with no warmup configured", i.tag)
		}
	}
}

// Sub-agent requests are kept and take no slot: N live sessions throughout, with their
// sub-agents' requests on top. A session waiting on its sub-agent's join holds its slot with no
// root request in flight, which is why the count here is of sessions and not of requests.
func TestConstantLoadKeepsSubagentsAndHoldsNSessions(t *testing.T) {
	const n = 6
	c := agxCorpus(t, subagentTrace(t, "S"), chainTrace(t, "B", 2), flatTrace(t, "F"))
	h := runAgentX(t, c, constantCfg(n, 60, AgentXConstantLoad{}), fixedService(sec))
	final := map[string]int{"S": 1, "B": 1, "F": 1}
	for _, at := range probes(h, 60) {
		if got := liveSessions(h, final, at); got != n {
			t.Fatalf("at %.1fs live sessions = %d, want %d", float64(at)/1e6, got, n)
		}
	}
	st := h.x.Stats()
	if st.SubagentRequests == 0 || st.JoinSuspensions == 0 {
		t.Errorf("sub-agents not replayed: %+v", st)
	}
	kinds := map[string]int{}
	for _, i := range h.issued {
		if i.phase == sim.PhaseProfile {
			kinds[i.tag.Kind]++
		}
	}
	if kinds["subagent"] == 0 || kinds["flat"] == 0 || kinds["root"] == 0 {
		t.Errorf("requests by kind: %v, want root, subagent and flat all present", kinds)
	}
}

// A slot frees when its root's final turn returns, even if a background sub-agent is still
// running (credit/callback_handler.py:620-623). Here the root's only request ends at 1 s while
// its sub-agent runs to 12 s: the next session must start at 1 s, not at 12 s.
func TestConstantLoadSlotFreesAtRootFinalTurn(t *testing.T) {
	bg := wekaLine(t, "BG",
		wreq(0, 1, 128, 5, 1, 2),
		wsub(0.5, "a1", 12000, wreq(0.5, 6, 128, 5, 50, 51), wreq(7, 5, 192, 5, 50, 51, 52)))
	c := agxCorpus(t, bg)
	h := runAgentX(t, c, constantCfg(1, 20, AgentXConstantLoad{}), fixedService(sec))
	var rootStarts []int64
	for _, i := range h.issued {
		if i.tag.Kind == "root" && i.tag.Turn == 0 {
			rootStarts = append(rootStarts, i.at)
		}
	}
	if len(rootStarts) < 2 || rootStarts[1] != 1*sec {
		t.Fatalf("root session starts at %v, want the second at 1s (the root's final return)", rootStarts)
	}
	childEnd := int64(0)
	for _, i := range h.issued {
		if i.tag.Kind == "subagent" && i.doneAt > childEnd {
			childEnd = i.doneAt
		}
	}
	if childEnd <= rootStarts[1] {
		t.Errorf("the sub-agent ended at %d, before the next session started: nothing shows the slot outlived the tree", childEnd)
	}
}

// The per-trace idle cap cuts a long gap between a trace's requests as the loader does
// (weka_trace.py:320-345): turn 1 is recorded 100 s after turn 0 (server time 1 s), and a 3 s
// cap leaves the first 3 s of that gap, so the end-to-start delay is 3 - 1 = 2 s and turn 1 goes
// out at 1 + 2 = 3 s. Uncapped, it waits the recorded 99 s.
func TestConstantLoadHonoursTheTraceIdleCap(t *testing.T) {
	line := wekaLine(t, "G", wreq(0, 1, 128, 5, 1, 2), wreq(100, 1, 192, 5, 1, 2, 3))
	second := func(capUs int64) int64 {
		cfg := constantCfg(1, 400, AgentXConstantLoad{})
		cfg.TraceIdleCapUs = capUs
		h := runAgentX(t, agxCorpus(t, line), cfg, fixedService(sec))
		for _, i := range h.issued {
			if i.tag.Turn == 1 {
				return i.at
			}
		}
		t.Fatal("turn 1 never sent")
		return 0
	}
	if got := second(3 * sec); got != 3*sec {
		t.Errorf("capped at 3s: turn 1 sent at %d µs, want %d", got, 3*sec)
	}
	if got := second(0); got != 100*sec {
		t.Errorf("uncapped: turn 1 sent at %d µs, want the recorded %d", got, 100*sec)
	}
}

// The warmup runs the same load, stops sending at its length, and the measured phase starts
// when its last request has returned, with fresh sessions drawn on from where the warmup's left
// off in the sampler's order.
func TestConstantLoadWarmupThenMeasuredPhase(t *testing.T) {
	c := agxCorpus(t, chainTrace(t, "A", 2), chainTrace(t, "B", 2), chainTrace(t, "C", 2), chainTrace(t, "D", 2))
	cfg := constantCfg(2, 20, AgentXConstantLoad{WarmupUs: 5*sec + sec/2})
	h := runAgentX(t, c, cfg, fixedService(sec))
	start, _ := h.x.Window()
	var lastWarm int64
	var firstProfile int64 = -1
	var warmRoots []string
	for _, i := range h.issued {
		switch i.phase {
		case sim.PhaseWarmup:
			if i.at >= 5*sec+sec/2 {
				t.Errorf("a warmup request sent at %d, past the warmup's sending window", i.at)
			}
			lastWarm = max(lastWarm, i.doneAt)
			if i.tag.Turn == 0 {
				warmRoots = append(warmRoots, i.tag.TraceID)
			}
		case sim.PhaseProfile:
			if firstProfile < 0 {
				firstProfile = i.at
			}
			if i.at < lastWarm {
				t.Errorf("a measured request at %d before the warmup's last return at %d", i.at, lastWarm)
			}
		}
	}
	if len(warmRoots) == 0 || firstProfile != start || start != lastWarm {
		t.Fatalf("warm roots %v, first measured send %d, window start %d, last warmup return %d", warmRoots, firstProfile, start, lastWarm)
	}
	// The first measured root continues the sampler's order (A B C D wrapping).
	order := []string{"A", "B", "C", "D"}
	want := order[len(warmRoots)%4]
	for _, i := range h.issued {
		if i.phase == sim.PhaseProfile && i.tag.Turn == 0 {
			if i.tag.TraceID != want {
				t.Errorf("first measured root = %s, want %s after %d warmup roots", i.tag.TraceID, want, len(warmRoots))
			}
			break
		}
	}
}

// --num-sessions bounds the measured phase's session starts; sessions already started finish.
func TestConstantLoadMeasuredSessionCap(t *testing.T) {
	c := agxCorpus(t, chainTrace(t, "A", 2), chainTrace(t, "B", 2), chainTrace(t, "C", 2))
	h := runAgentX(t, c, constantCfg(2, 60, AgentXConstantLoad{MeasuredSessions: 5}), fixedService(sec))
	roots := 0
	for _, i := range h.issued {
		if i.phase == sim.PhaseProfile && i.tag.Kind == "root" && i.tag.Turn == 0 {
			roots++
		}
	}
	if roots != 5 {
		t.Errorf("measured sessions started = %d, want 5", roots)
	}
	if st := h.x.Stats(); st.Completed != 10 {
		t.Errorf("completed %d requests, want 10 (5 two-turn sessions)", st.Completed)
	}
}

// Recycling rules: sequential walks the corpus in order and wraps; shuffle visits every root
// once per pass; random is reproducible under the seed.
func TestConstantLoadRecyclingRules(t *testing.T) {
	c := agxCorpus(t, chainTrace(t, "A", 1), chainTrace(t, "B", 1), chainTrace(t, "C", 1))
	roots := func(rule string, seed int64) []string {
		cfg := constantCfg(1, 12, AgentXConstantLoad{Recycling: rule})
		cfg.Seed = seed
		h := runAgentX(t, c, cfg, fixedService(sec))
		var out []string
		for _, i := range h.issued {
			if i.tag.Turn == 0 {
				out = append(out, i.tag.TraceID)
			}
		}
		return out
	}
	if got := roots(AgentXRecycleSequential, 1); fmt.Sprint(got[:7]) != "[A B C A B C A]" {
		t.Errorf("sequential: %v", got)
	}
	sh := roots(AgentXRecycleShuffle, 1)
	for p := 0; p+3 <= len(sh); p += 3 {
		seen := map[string]bool{sh[p]: true, sh[p+1]: true, sh[p+2]: true}
		if len(seen) != 3 {
			t.Errorf("shuffle pass %d = %v, not a permutation", p/3, sh[p:p+3])
		}
	}
	if a, b := roots(AgentXRecycleRandom, 7), roots(AgentXRecycleRandom, 7); fmt.Sprint(a) != fmt.Sprint(b) {
		t.Error("random recycling is not reproducible under one seed")
	}
	if _, err := NewAgentXReplay(c, constantCfg(1, 12, AgentXConstantLoad{Recycling: "lifo"})); err == nil {
		t.Error("an unknown recycling rule was accepted")
	}
}
