package workload

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/trace"

	"github.com/inference-sim/inference-sim/sim"
)

// PhaseGate runs a workload in drain-gated phases (inference-sim#1926, blis-substrate#156). The
// first phase's requests are sent at the start; each later phase is held until the clock the
// previous phase drains (every one of its requests has reached its end) plus the phase's offset,
// or sent from a stated clock. The engines, the KV caches and the prefix state are the cluster's
// own and run on across phases, so the gate only decides when each request is sent.
//
// A phase with a concurrency limit sends that many of its requests at its start and one more
// each time one of its own ends, as a benchmark client with a fixed number of workers does; its
// requests then arrive when sent, not at their recorded arrival. Without a limit each request of
// a phase arrives at the phase's start plus its recorded arrival less the phase's earliest.
type PhaseGate struct {
	phases []*gatePhase
	of     map[string]int // request ID -> phase index
	ended  map[string]bool
}

type gatePhase struct {
	spec    trace.Phase
	pending []*sim.Request // not yet sent, in workload order
	total   int
	ended   int
	base    int64 // earliest recorded arrival
	started bool
	start   int64
	lastEnd int64
}

// NewPhaseGate groups requests by their Phase label into the header's phases. It returns the
// requests to send at the start of the run: the first phase's, and those of every phase that
// starts at a clock of its own. A phase no request names, or a request naming a phase the list
// lacks, is an error.
func NewPhaseGate(phases []trace.Phase, requests []*sim.Request) (*PhaseGate, []*sim.Request, error) {
	g := &PhaseGate{of: map[string]int{}, ended: map[string]bool{}}
	index := map[string]int{}
	for i, ph := range phases {
		g.phases = append(g.phases, &gatePhase{spec: ph, base: -1})
		index[ph.Name] = i
	}
	for _, r := range requests {
		i, ok := index[string(r.Phase)]
		if !ok {
			return nil, nil, fmt.Errorf("request %s is in phase %q, which the phase list does not name", r.ID, r.Phase)
		}
		p := g.phases[i]
		p.pending = append(p.pending, r)
		p.total++
		if p.base < 0 || r.ArrivalTime < p.base {
			p.base = r.ArrivalTime
		}
		g.of[r.ID] = i
	}
	for _, p := range g.phases {
		if p.total == 0 {
			return nil, nil, fmt.Errorf("phase %q names no request", p.spec.Name)
		}
	}
	var initial []*sim.Request
	for i, p := range g.phases {
		switch {
		case p.spec.Start == trace.StartAt:
			initial = append(initial, g.begin(i, p.spec.OffsetUs)...)
		case i == 0:
			initial = append(initial, g.begin(0, p.base+p.spec.OffsetUs)...)
		}
	}
	return g, initial, nil
}

// begin starts phase i at clock start and returns the requests it sends then.
func (g *PhaseGate) begin(i int, start int64) []*sim.Request {
	p := g.phases[i]
	p.started, p.start = true, start
	n := len(p.pending)
	if c := p.spec.Concurrency; c > 0 && c < n {
		n = c
	}
	out := p.take(n, start, p.spec.Concurrency > 0)
	return out
}

// take removes the next n pending requests and sets their arrival: at the clock when the phase
// is concurrency-limited (the request is sent then), else its recorded offset into the phase.
func (p *gatePhase) take(n int, now int64, sentNow bool) []*sim.Request {
	out := p.pending[:n:n]
	p.pending = p.pending[n:]
	for _, r := range out {
		if sentNow {
			r.ArrivalTime = now
		} else {
			r.ArrivalTime = p.start + (r.ArrivalTime - p.base)
		}
	}
	return out
}

// Ended is called when a request reaches its end at the given clock. It returns the requests to
// send now: the next one of the request's own concurrency-limited phase, and the first requests of
// the next phase when this one has drained.
func (g *PhaseGate) Ended(req *sim.Request, at int64) []*sim.Request {
	i, ok := g.of[req.ID]
	if !ok || g.ended[req.ID] {
		return nil
	}
	g.ended[req.ID] = true
	p := g.phases[i]
	p.ended++
	if at > p.lastEnd {
		p.lastEnd = at
	}
	var out []*sim.Request
	if p.spec.Concurrency > 0 && len(p.pending) > 0 {
		out = append(out, p.take(1, at, true)...)
	}
	if p.ended == p.total && i+1 < len(g.phases) {
		if next := g.phases[i+1]; !next.started && next.spec.Start != trace.StartAt {
			out = append(out, g.begin(i+1, p.lastEnd+next.spec.OffsetUs)...)
		}
	}
	return out
}

// Unsent returns the requests the run ended before sending. Each is a request the workload
// planned and never sent, so it ends not started (SIM-8). Its arrival is stated as the clock the
// run ended at, the earliest the plan allows.
func (g *PhaseGate) Unsent(at int64) []sim.PlannedRound {
	var out []sim.PlannedRound
	for _, p := range g.phases {
		for _, r := range p.pending {
			out = append(out, sim.PlannedRound{ID: r.ID, SessionID: r.SessionID, RoundIndex: r.RoundIndex,
				Model: r.Model, Adapter: r.Adapter, SLOClass: r.SLOClass, TenantID: r.TenantID,
				PromptTokens: len(r.InputTokens), Phase: r.Phase, ArrivalUs: at, HasArrival: true})
		}
	}
	return out
}
