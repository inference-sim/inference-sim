package workload

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"pgregory.net/rapid"

	"github.com/inference-sim/inference-sim/sim"
)

// Property and metamorphic tests of the AgentX replay over randomly generated Weka v7
// corpora with subagents and flattened worker chains. Each law is stated in
// docs/contributing/standards/invariants.md (INV-AX1 ... INV-AX4).

// msGrid draws a time in seconds on the millisecond grid Weka records, so no two times
// fall within aiperf's absolute 1e-6 s comparison epsilon of each other by accident.
func msGrid(t *rapid.T, lo, hi float64, label string) float64 {
	return math.Round(rapid.Float64Range(lo, hi).Draw(t, label)*1000) / 1000
}

// genTrace draws one Weka v7 trace: a root stream whose turns mostly extend the previous
// hash prefix (sometimes overlapping, which splits off flat worker chains), with subagent
// groups between root turns.
func genTrace(t *rapid.T, id string) string {
	nRoot := rapid.IntRange(1, 6).Draw(t, "nRoot")
	var reqs []map[string]any
	tm := 0.0
	hash := []int64{1, 2}
	next := int64(100)
	for k := 0; k < nRoot; k++ {
		if k > 0 {
			tm += msGrid(t, 0, 40, "gap")
		}
		api := msGrid(t, 0, 8, "api")
		switch rapid.IntRange(0, 4).Draw(t, "shape") {
		case 0: // fork a sibling context off the base prefix
			hash = []int64{1, 2, next}
		default:
			hash = append(append([]int64(nil), hash...), next)
		}
		next++
		in := len(hash)*64 + rapid.IntRange(0, 63).Draw(t, "partial")
		out := rapid.IntRange(0, 300).Draw(t, "out")
		reqs = append(reqs, map[string]any{"type": "s", "t": tm, "api_time": api, "in": in, "out": out, "model": "m", "hash_ids": hash})
		// Subagents after any root turn: after the last one they outlive the root (a
		// background branch); several may share a gap and overlap.
		for ns := rapid.IntRange(0, 2).Draw(t, "nSub"); ns > 0; ns-- {
			st := tm + msGrid(t, 0, 3, "saStart")
			var inner []map[string]any
			ih := []int64{next, next + 1}
			next += 2
			it := st
			end := st
			for j := 0; j < rapid.IntRange(1, 3).Draw(t, "nInner"); j++ {
				ia := msGrid(t, 0, 5, "innerApi")
				inner = append(inner, map[string]any{"type": "s", "t": it, "api_time": ia, "in": len(ih)*64 + 10, "out": rapid.IntRange(1, 200).Draw(t, "innerOut"), "model": "m", "hash_ids": ih})
				end = math.Max(end, it+ia)
				it += ia + msGrid(t, 0, 5, "innerGap")
				ih = append(append([]int64(nil), ih...), next)
				next++
			}
			reqs = append(reqs, map[string]any{"type": "subagent", "t": st, "agent_id": fmt.Sprintf("a%d_%d", k, ns), "subagent_type": "Explore",
				"duration_ms": (end - st) * 1000, "status": "completed", "models": []string{"m"}, "requests": inner})
		}
	}
	return mustWekaLine(id, reqs)
}

func mustWekaLine(id string, reqs []map[string]any) string {
	b, err := json.Marshal(map[string]any{"id": id, "models": []string{"m"}, "block_size": 64, "hash_id_scope": "local", "requests": reqs})
	if err != nil {
		panic(err)
	}
	return string(b)
}

// scaleTimes multiplies every recorded time of a trace line (t, api_time, duration_ms) by f.
func scaleTimes(line string, f float64) string {
	var v map[string]any
	if err := json.Unmarshal([]byte(line), &v); err != nil {
		panic(err)
	}
	var walk func(rs []any)
	walk = func(rs []any) {
		for _, r := range rs {
			m := r.(map[string]any)
			for _, k := range []string{"t", "api_time", "duration_ms"} {
				if x, ok := m[k].(float64); ok {
					m[k] = x * f
				}
			}
			if inner, ok := m["requests"].([]any); ok {
				walk(inner)
			}
		}
	}
	walk(v["requests"].([]any))
	b, _ := json.Marshal(v)
	return string(b)
}

func genCorpus(t *rapid.T) []string {
	n := rapid.IntRange(1, 4).Draw(t, "nTraces")
	lines := make([]string, n)
	for i := range lines {
		lines[i] = genTrace(t, fmt.Sprintf("T%d", i))
	}
	return lines
}

func genConfig(t *rapid.T) AgentXReplayConfig {
	cfg := AgentXReplayConfig{
		Concurrency: rapid.IntRange(1, 5).Draw(t, "C"),
		Seed:        int64(rapid.IntRange(0, 1000).Draw(t, "seed")),
		DurationUs:  int64(rapid.IntRange(20, 300).Draw(t, "dur")) * sec,
		GraceUs:     int64(rapid.IntRange(0, 60).Draw(t, "grace")) * sec,
		IdleCapUs:   10 * sec,
		MinRatio:    0, MaxRatio: 1,
	}
	if rapid.Bool().Draw(t, "timeouts") {
		cfg.RequestTimeoutUs = int64(rapid.IntRange(1, 6).Draw(t, "timeoutS")) * sec
	}
	if rapid.Bool().Draw(t, "overflows") {
		cfg.MaxModelLen = int64(rapid.IntRange(150, 700).Draw(t, "maxModelLen"))
	}
	return cfg
}

// genReject has the cluster refuse a pseudo-random share of requests (none, or 1 in n).
func genReject(t *rapid.T) func(*sim.Request) bool {
	n := rapid.SampledFrom([]int{0, 3, 7}).Draw(t, "rejectEvery")
	if n == 0 {
		return nil
	}
	return func(r *sim.Request) bool { return agxMix(uint64(len(r.ID)), uint64(r.ArrivalTime))%uint64(n) == 0 }
}

// agxCoverage counts what the random runs exercised, so a property cannot pass vacuously.
var agxCoverage = map[string]int{}

// genService draws a service time that depends on the request's size and lane.
func genService(t *rapid.T) func(*sim.Request) int64 {
	base := int64(rapid.IntRange(1, 3000).Draw(t, "baseMs")) * 1000
	perLane := int64(rapid.IntRange(0, 2000).Draw(t, "laneMs")) * 1000
	return func(r *sim.Request) int64 {
		return base + int64(r.Agentic.Lane)*perLane + int64(len(r.OutputTokens))*1000
	}
}

type agxKey struct {
	tree   int
	stream string
	inst   int
	turn   int
}

// checkAgentXInvariants asserts INV-AX1 ... INV-AX4 on one run.
func checkAgentXInvariants(t interface {
	Fatalf(string, ...any)
	Errorf(string, ...any)
}, h *agxHarness, cfg AgentXReplayConfig) {
	x := h.x
	start, end := x.Window()
	windowEnd := x.windowEndUs
	sent := map[agxKey]agxIssue{} // profiling sends
	streamSends := map[[2]any][]agxIssue{}
	byConv := map[[2]any][]int{} // (tree, stream) -> instances sent
	for _, is := range h.issued {
		k := agxKey{is.tag.Tree, is.tag.Stream, is.tag.StreamInstance, is.tag.Turn}
		// INV-AX4: phases and the sending window.
		switch is.phase {
		case sim.PhaseWarmup:
			if x.phase >= agxProfile && is.at > start {
				t.Errorf("INV-AX4: warm-up send at %d after profiling start %d", is.at, start)
			}
			if is.out != 1 {
				t.Errorf("INV-AX4: warm-up send with max_tokens %d", is.out)
			}
			continue
		case sim.PhaseProfile:
			if is.at < start || is.at >= windowEnd {
				t.Errorf("INV-AX4: profile send at %d outside [%d, %d)", is.at, start, windowEnd)
			}
		default:
			t.Errorf("INV-AX4: send with phase %q", is.phase)
		}
		if _, dup := sent[k]; dup {
			t.Errorf("INV-AX2: %v sent twice", k)
		}
		sent[k] = is
		sk := [2]any{is.tag.Tree, is.tag.StreamInstance}
		streamSends[sk] = append(streamSends[sk], is)
		byConv[[2]any{is.tag.Tree, is.tag.Stream}] = append(byConv[[2]any{is.tag.Tree, is.tag.Stream}], is.tag.StreamInstance)
	}
	// INV-AX1: every send ends exactly once -- completed, an error (timeout or refusal), a
	// context overflow, or cancelled because it was still in flight at the grace end.
	ended := map[string]int{}
	for _, is := range h.issued {
		o := is.req.Agentic.Outcome
		switch {
		case is.doneAt >= 0 && (o == sim.AgenticCompleted || o == sim.AgenticError || o == sim.AgenticContextOverflow):
			if is.phase != sim.PhaseWarmup && is.doneAt > x.graceEndUs {
				t.Errorf("INV-AX1: a request ended at %d after the grace end %d", is.doneAt, x.graceEndUs)
			}
		case o == sim.AgenticCancelled && (is.doneAt < 0 || is.refused):
			// In flight at the stop -- or refused, with the error response still on its way.
		default:
			t.Errorf("INV-AX1: send %s done=%d outcome %q", is.req.ID, is.doneAt, o)
		}
		ended[o]++
		agxCoverage["outcome:"+o]++
	}
	st := x.Stats()
	if ended[sim.AgenticCompleted] != st.Completed || ended[sim.AgenticError] != st.Errors ||
		ended[sim.AgenticContextOverflow] != st.ContextOverflows || ended[sim.AgenticCancelled] != st.Cancelled ||
		x.InFlight() != st.Cancelled {
		t.Errorf("INV-AX1: outcomes %v disagree with the replay's %+v (in flight %d)", ended, st, x.InFlight())
	}
	if end < windowEnd && x.phase >= agxProfile && end >= 0 {
		t.Errorf("INV-AX1: phase completion %d before the window end %d", end, windowEnd)
	}
	// INV-AX2: per-stream order and dependencies.
	insts := map[int]*agxInstance{}
	for _, in := range x.instances {
		insts[in.id] = in
	}
	for sk, ss := range streamSends {
		sort.SliceStable(ss, func(i, j int) bool { return ss[i].at < ss[j].at })
		for i := 1; i < len(ss); i++ {
			agxCoverage["stream continued"]++
			if ss[i-1].req.Agentic.Outcome == sim.AgenticContextOverflow {
				t.Errorf("INV-AX2: stream %v sent turn %d after a context overflow", sk, ss[i].tag.Turn)
			}
			if ss[i-1].req.Agentic.Outcome == sim.AgenticError {
				agxCoverage["continued after error"]++
			}
			if ss[i].tag.Turn != ss[i-1].tag.Turn+1 {
				t.Errorf("INV-AX2: stream %v turns %d then %d", sk, ss[i-1].tag.Turn, ss[i].tag.Turn)
			}
			if ss[i-1].doneAt < 0 || ss[i-1].doneAt > ss[i].at {
				t.Errorf("INV-AX2: stream %v turn %d sent at %d before turn %d completed (%d)", sk, ss[i].tag.Turn, ss[i].at, ss[i-1].tag.Turn, ss[i-1].doneAt)
			}
		}
	}
	for _, is := range h.issued {
		if is.phase != sim.PhaseProfile {
			continue
		}
		in := insts[is.tag.Tree]
		tr := in.trace
		conv := int32(-1)
		for ci := range tr.convs {
			if tr.convs[ci].sessionID == is.tag.Stream {
				conv = int32(ci)
			}
		}
		tn := tr.convs[conv].turns[is.tag.Turn]
		var hist map[int32]int32
		if in.id < len(x.lanes) {
			hist = x.lanes[in.lane].completed
		}
		for _, p := range tn.preds {
			pk := agxKey{in.id, tr.convs[p.conv].sessionID, -1, int(p.turn)}
			var ps agxIssue
			ok := false
			for _, inst := range byConv[[2]any{in.id, pk.stream}] {
				if q, found := sent[agxKey{in.id, pk.stream, inst, pk.turn}]; found && (!ok || q.doneAt < ps.doneAt) {
					ps, ok = q, true
				}
			}
			switch {
			case ok:
				agxCoverage["barrier predecessor checked"]++
				if ps.doneAt < 0 || ps.doneAt > is.at {
					t.Errorf("INV-AX2: %v sent at %d before its barrier predecessor %v completed (%d)", is.tag, is.at, pk, ps.doneAt)
				}
			case hist != nil && p.turn < hist[p.conv]:
			default:
				t.Errorf("INV-AX2: %v sent with barrier predecessor %v neither completed nor history", is.tag, pk)
			}
		}
		if conv != 0 {
			continue
		}
		for _, b := range tn.gates {
			for _, c := range tr.branches[b].children {
				for _, inst := range uniqueInts(byConv[[2]any{in.id, tr.convs[c].sessionID}]) {
					cs := streamSends[[2]any{in.id, inst}]
					if len(cs) == 0 {
						continue
					}
					last := cs[len(cs)-1]
					agxCoverage["join checked"]++
					if (int(last.tag.Turn) == len(tr.convs[c].turns)-1 || last.req.Agentic.Outcome == sim.AgenticContextOverflow) && (last.doneAt < 0 || last.doneAt > is.at) {
						t.Errorf("INV-AX2: gated root turn %v sent at %d before child %s finished (%d)", is.tag, is.at, tr.convs[c].sessionID, last.doneAt)
					}
					if int(last.tag.Turn) < len(tr.convs[c].turns)-1 && last.req.Agentic.Outcome != sim.AgenticContextOverflow {
						t.Errorf("INV-AX2: gated root turn %v sent while child %s has turns left", is.tag, tr.convs[c].sessionID)
					}
				}
			}
		}
	}
	// INV-AX3: one tree per lane at a time -- a lane's next tree starts only after the
	// previous one drained -- so at most C trees are live.
	type span struct {
		first, last int64
		tree        int
	}
	byLane := map[int][]span{}
	spans := map[int]*span{}
	for _, is := range h.issued {
		sp, ok := spans[is.tag.Tree]
		if !ok {
			sp = &span{first: is.at, last: is.at, tree: is.tag.Tree}
			spans[is.tag.Tree] = sp
		}
		fin := is.doneAt
		if fin < 0 {
			fin = math.MaxInt64
		}
		sp.last = max(sp.last, fin)
	}
	for _, in := range x.instances {
		if sp, ok := spans[in.id]; ok {
			byLane[in.lane] = append(byLane[in.lane], *sp)
		}
	}
	if len(byLane) > cfg.Concurrency {
		t.Errorf("INV-AX3: %d lanes for concurrency %d", len(byLane), cfg.Concurrency)
	}
	for lane, sps := range byLane {
		sort.Slice(sps, func(i, j int) bool { return sps[i].tree < sps[j].tree })
		for i := 1; i < len(sps); i++ {
			if sps[i].first < sps[i-1].last {
				t.Errorf("INV-AX3: lane %d tree %d starts at %d before tree %d drained (%d)", lane, sps[i].tree, sps[i].first, sps[i-1].tree, sps[i-1].last)
			}
		}
	}
}

// issueLog is a run's sends as comparable values (INV-6).
func issueLog(h *agxHarness) []string {
	var out []string
	for _, is := range h.issued {
		out = append(out, fmt.Sprintf("%d %d %s %d %s %d %d %d %v", is.at, is.tag.Tree, is.tag.Stream, is.tag.Turn, is.phase, is.in, is.out, is.doneAt, is.req.InputTokens[:min(8, len(is.req.InputTokens))]))
	}
	return out
}

func runRapid(t *rapid.T, lines []string, cfg AgentXReplayConfig, svc func(*sim.Request) int64, reject ...func(*sim.Request) bool) *agxHarness {
	c, err := ReadAgentXCorpus(strings.NewReader(strings.Join(lines, "\n")), 0, true)
	if err != nil {
		t.Fatalf("corpus: %v", err)
	}
	var rj func(*sim.Request) bool
	if len(reject) > 0 {
		rj = reject[0]
	}
	h := runAgentXWith(t, c, cfg, svc, rj)
	for _, is := range h.issued {
		if is.tag.Kind != "root" {
			agxCoverage["child sent"]++
		}
	}
	return h
}

// TestAgentXInvariantsHoldOverRandomCorpora: INV-AX1 ... INV-AX4 and INV-6 over random
// corpora, concurrencies, windows, grace periods and service times.
func TestAgentXInvariantsHoldOverRandomCorpora(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		lines := genCorpus(t)
		cfg := genConfig(t)
		svc := genService(t)
		rj := genReject(t)
		h := runRapid(t, lines, cfg, svc, rj)
		checkAgentXInvariants(t, h, cfg)
		if again := runRapid(t, lines, cfg, svc, rj); !reflect.DeepEqual(issueLog(h), issueLog(again)) {
			t.Fatalf("INV-6: two runs of the same corpus and config sent different requests")
		}
	})
	// Not vacuous: the runs sent children, checked barriers and joins, and ended requests
	// every way a request can end.
	for _, k := range []string{"child sent", "barrier predecessor checked", "join checked", "continued after error",
		"outcome:" + sim.AgenticCompleted, "outcome:" + sim.AgenticError, "outcome:" + sim.AgenticContextOverflow, "outcome:" + sim.AgenticCancelled} {
		if agxCoverage[k] == 0 {
			t.Errorf("the random runs never exercised %q", k)
		}
	}
}

// TestAgentXLongerGraceKeepsEveryCompletion (metamorphic): lengthening the grace period
// only lets more in-flight requests finish; every request completed under grace G is
// sent identically and completed under 2G.
func TestAgentXLongerGraceKeepsEveryCompletion(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		lines := genCorpus(t)
		cfg := genConfig(t)
		svc := genService(t)
		a := runRapid(t, lines, cfg, svc)
		cfg.GraceUs *= 2
		b := runRapid(t, lines, cfg, svc)
		if len(b.issued) != len(a.issued) {
			t.Fatalf("grace changed the sends: %d vs %d", len(a.issued), len(b.issued))
		}
		for i, is := range a.issued {
			o := b.issued[i]
			if is.at != o.at || is.tag.Stream != o.tag.Stream || is.tag.Turn != o.tag.Turn {
				t.Fatalf("send %d differs under a longer grace", i)
			}
			if is.doneAt >= 0 && o.doneAt != is.doneAt {
				t.Fatalf("send %d completed at %d under grace G but %d under 2G", i, is.doneAt, o.doneAt)
			}
		}
	})
}

// TestAgentXServiceTimesDoNotReorderAStream (metamorphic): whatever the service times --
// here, lanes' service multipliers permuted -- each stream resumes at the same turn and
// sends its turns in order; only how far it gets and when change.
func TestAgentXServiceTimesDoNotReorderAStream(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		lines := genCorpus(t)
		cfg := genConfig(t)
		mult := make([]int64, cfg.Concurrency)
		for i := range mult {
			mult[i] = int64(rapid.IntRange(1, 4000).Draw(t, "mult")) * 1000
		}
		perm := rapid.Permutation(mult).Draw(t, "perm")
		svc := func(m []int64) func(*sim.Request) int64 {
			return func(r *sim.Request) int64 { return m[r.Agentic.Lane] + int64(len(r.OutputTokens))*500 }
		}
		a := runRapid(t, lines, cfg, svc(mult))
		b := runRapid(t, lines, cfg, svc(perm))
		first := func(h *agxHarness) map[[2]any]int {
			m := map[[2]any]int{}
			for _, is := range h.issued {
				k := [2]any{is.tag.Lane, is.tag.Stream}
				if is.phase == sim.PhaseProfile && is.tag.Tree < cfg.Concurrency {
					if _, ok := m[k]; !ok {
						m[k] = is.tag.Turn
					}
				}
			}
			return m
		}
		fa, fb := first(a), first(b)
		for k, v := range fa {
			if w, ok := fb[k]; ok && w != v {
				t.Fatalf("stream %v resumes at turn %d under one service assignment and %d under its permutation", k, v, w)
			}
		}
		checkAgentXInvariants(t, b, cfg)
	})
}

// TestAgentXTimeScalingKeepsTheTree (metamorphic): scaling every recorded time of a trace
// (starts, api times, subagent durations) by a factor rebuilds the same session tree --
// the same streams, branches, joins and barriers -- with delays scaled by the factor.
func TestAgentXTimeScalingKeepsTheTree(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		line := genTrace(t, "S")
		f := rapid.SampledFrom([]float64{0.5, 2, 4}).Draw(t, "factor")
		a, err := ReadAgentXCorpus(strings.NewReader(line), 0, true)
		if err != nil {
			t.Fatal(err)
		}
		b, err := ReadAgentXCorpus(strings.NewReader(scaleTimes(line, f)), 0, true)
		if err != nil {
			t.Fatal(err)
		}
		ta, tb := a.Traces[0], b.Traces[0]
		if len(ta.convs) != len(tb.convs) || len(ta.branches) != len(tb.branches) {
			t.Fatalf("scaling by %v changed the tree: %d/%d conversations, %d/%d branches", f, len(ta.convs), len(tb.convs), len(ta.branches), len(tb.branches))
		}
		for ci := range ta.convs {
			ca, cb := ta.convs[ci], tb.convs[ci]
			if ca.sessionID != cb.sessionID || len(ca.turns) != len(cb.turns) {
				t.Fatalf("conversation %d differs", ci)
			}
			for k := range ca.turns {
				x, y := ca.turns[k], cb.turns[k]
				if !reflect.DeepEqual(x.preds, y.preds) || !reflect.DeepEqual(x.spawns, y.spawns) || !reflect.DeepEqual(x.gates, y.gates) {
					t.Fatalf("%s turn %d: dependencies changed under scaling", ca.sessionID, k)
				}
				if math.Abs(y.delayMs-f*x.delayMs) > 1e-6*math.Max(1, y.delayMs) {
					t.Fatalf("%s turn %d: delay %v scaled by %v gave %v", ca.sessionID, k, x.delayMs, f, y.delayMs)
				}
			}
		}
	})
}

// TestAgentXMaxContextFilter: the filter drops exactly the traces whose peak
// in + max(out, 1) exceeds the limit, keeps the others in row order, and a corpus whose
// every trace exceeds it is refused, as aiperf refuses an empty pool.
func TestAgentXMaxContextFilter(t *testing.T) {
	small := wekaLine(t, "small", wreq(0, 1, 1000, 0, 1)) // peak 1000 + max(0, 1) = 1001
	big := wekaLine(t, "big", wreq(0, 1, 1000, 5, 1), wsub(0.5, "a", 1000, wreq(0.5, 1, 4000, 100, 7)))
	big2 := strings.Replace(big, `"id":"big"`, `"id":"big2"`, 1)
	corpus := strings.Join([]string{big, small, big2}, "\n")
	// big's peak is its subagent's 4000 + 100: subagent requests count, uncapped.
	for _, tc := range []struct {
		limit int
		kept  []string
	}{
		{0, []string{"big", "small", "big2"}},
		{4100, []string{"big", "small", "big2"}},
		{4099, []string{"small"}},
		{1001, []string{"small"}},
	} {
		c, err := ReadAgentXCorpus(strings.NewReader(corpus), tc.limit, true)
		if err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, tr := range c.Traces {
			ids = append(ids, tr.ID)
		}
		if !reflect.DeepEqual(ids, tc.kept) || c.FilteredByMaxCtx != 3-len(tc.kept) {
			t.Errorf("limit %d: kept %v (filtered %d), want %v", tc.limit, ids, c.FilteredByMaxCtx, tc.kept)
		}
	}
	path := filepath.Join(t.TempDir(), "big.jsonl")
	if err := os.WriteFile(path, []byte(big+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadAgentXCorpus(path, "", 1000, true); err == nil || !strings.Contains(err.Error(), "no trace fits") {
		t.Errorf("a corpus with no trace under the limit was accepted: %v", err)
	}
	// Filtered traces never reach a lane: every lane replays "small".
	c, _ := ReadAgentXCorpus(strings.NewReader(corpus), 4099, true)
	h := runAgentX(t, c, agxCfg(3, 0, 30), fixedService(sec))
	for _, is := range h.issued {
		if is.tag.TraceID != "small" {
			t.Errorf("a filtered trace was replayed: %s", is.tag.TraceID)
		}
	}
}

// TestAgentXCorpusDigestIsEnforced: a wrong --corpus-sha256 is refused.
func TestAgentXCorpusDigestIsEnforced(t *testing.T) {
	if _, err := LoadAgentXCorpus("testdata/agentx/weka_v7_slice.jsonl", strings.Repeat("0", 64), 0, true); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("mismatched digest accepted: %v", err)
	}
	c, err := LoadAgentXCorpus("testdata/agentx/weka_v7_slice.jsonl", "", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := LoadAgentXCorpus("testdata/agentx/weka_v7_slice.jsonl", strings.ToUpper(c.SHA256), 0, true); err != nil {
		t.Fatalf("matching digest refused: %v", err)
	}
}

func uniqueInts(v []int) []int {
	seen := map[int]bool{}
	var out []int
	for _, x := range v {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	return out
}
