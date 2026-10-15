package workload

import (
	"container/heap"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// agxHarness is a minimal discrete-event stand-in for the cluster: an arrival runs the
// replay's ArrivalGate exactly as ClusterArrivalEvent does, and a sent request completes
// after service(req). It lets the replay's issue order and timing be checked by hand.
type agxHarness struct {
	x       *AgentXReplay
	service func(*sim.Request) int64
	events  agxEvHeap
	seq     int64
	horizon int64
	issued  []agxIssue
	done    []agxIssue
	inFl    map[*sim.Request]bool
	maxInFl int
}

type agxIssue struct {
	at     int64
	req    *sim.Request
	tag    sim.AgenticTag
	phase  sim.Phase
	in     int
	out    int
	doneAt int64
}

type agxEv struct {
	at, seq int64
	arrive  bool
	req     *sim.Request
}

type agxEvHeap []agxEv

func (h agxEvHeap) Len() int { return len(h) }
func (h agxEvHeap) Less(i, j int) bool {
	if h[i].at != h[j].at {
		return h[i].at < h[j].at
	}
	// cluster events (arrivals) before instance events (completions) at the same time
	if h[i].arrive != h[j].arrive {
		return h[i].arrive
	}
	return h[i].seq < h[j].seq
}
func (h agxEvHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *agxEvHeap) Push(x any)   { *h = append(*h, x.(agxEv)) }
func (h *agxEvHeap) Pop() any {
	o := *h
	v := o[len(o)-1]
	*h = o[:len(o)-1]
	return v
}

func (h *agxHarness) push(at int64, arrive bool, r *sim.Request) {
	h.seq++
	heap.Push(&h.events, agxEv{at, h.seq, arrive, r})
}

func runAgentX(t *testing.T, c *AgentXCorpus, cfg AgentXReplayConfig, service func(*sim.Request) int64) *agxHarness {
	t.Helper()
	x, err := NewAgentXReplay(c, cfg)
	if err != nil {
		t.Fatal(err)
	}
	h := &agxHarness{x: x, service: service, horizon: math.MaxInt64, inFl: map[*sim.Request]bool{}}
	x.SetHorizon = func(v int64) { h.horizon = v }
	for _, r := range x.Start() {
		h.push(r.ArrivalTime, true, r)
	}
	idx := map[*sim.Request]int{}
	stop := int64(0)
	for h.events.Len() > 0 {
		e := heap.Pop(&h.events).(agxEv)
		if e.at > h.horizon {
			break
		}
		stop = e.at
		if e.arrive {
			send, more := x.Gate(e.req, e.at)
			for _, m := range more {
				h.push(m.ArrivalTime, true, m)
			}
			if !send {
				continue
			}
			r := e.req
			if r.ArrivalTime != e.at {
				t.Fatalf("sent request %s carries ArrivalTime %d at %d", r.ID, r.ArrivalTime, e.at)
			}
			idx[r] = len(h.issued)
			h.issued = append(h.issued, agxIssue{at: e.at, req: r, tag: *r.Agentic, phase: r.Phase, in: len(r.InputTokens), out: len(r.OutputTokens), doneAt: -1})
			h.inFl[r] = true
			h.maxInFl = max(h.maxInFl, len(h.inFl))
			h.push(e.at+h.service(r), false, r)
			continue
		}
		delete(h.inFl, e.req)
		e.req.State = sim.StateCompleted
		h.issued[idx[e.req]].doneAt = e.at
		h.done = append(h.done, h.issued[idx[e.req]])
		for _, m := range x.OnComplete(e.req, e.at) {
			h.push(m.ArrivalTime, true, m)
		}
	}
	x.Finish(stop)
	return h
}

// wekaLine encodes a hand-written trace as a Weka v7 JSONL row.
func wekaLine(t *testing.T, id string, reqs ...map[string]any) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{"id": id, "models": []string{"m"}, "block_size": 64, "hash_id_scope": "local", "requests": reqs})
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func wreq(tSec, api float64, in, out int, hash ...int64) map[string]any {
	if hash == nil {
		hash = []int64{}
	}
	return map[string]any{"type": "s", "t": tSec, "api_time": api, "in": in, "out": out, "model": "m", "hash_ids": hash}
}

func wsub(tSec float64, agent string, durationMs float64, inner ...map[string]any) map[string]any {
	return map[string]any{"type": "subagent", "t": tSec, "agent_id": agent, "subagent_type": "Explore", "duration_ms": durationMs, "status": "completed", "models": []string{"m"}, "requests": inner}
}

func agxCorpus(t *testing.T, lines ...string) *AgentXCorpus {
	t.Helper()
	c, err := ReadAgentXCorpus(strings.NewReader(strings.Join(lines, "\n")+"\n"), 0, true)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func agxCfg(c int, ratio float64, durS int64) AgentXReplayConfig {
	return AgentXReplayConfig{Concurrency: c, Seed: 42, DurationUs: durS * 1e6, GraceUs: 30e6, IdleCapUs: 10e6, MinRatio: ratio, MaxRatio: ratio}
}

func fixedService(us int64) func(*sim.Request) int64 { return func(*sim.Request) int64 { return us } }

type agxWant struct {
	at     int64 // µs
	stream string
	turn   int
	phase  sim.Phase
}

func checkIssues(t *testing.T, h *agxHarness, want []agxWant) {
	t.Helper()
	n := min(len(want), len(h.issued))
	for i := 0; i < n; i++ {
		g, w := h.issued[i], want[i]
		if g.at != w.at || g.tag.Stream != w.stream || g.tag.Turn != w.turn || g.phase != w.phase {
			t.Errorf("issue %d: got t=%d %s#%d %s, want t=%d %s#%d %s", i, g.at, g.tag.Stream, g.tag.Turn, g.phase, w.at, w.stream, w.turn, w.phase)
		}
	}
	if len(want) > len(h.issued) {
		t.Errorf("%d issues, want at least %d", len(h.issued), len(want))
	}
}
