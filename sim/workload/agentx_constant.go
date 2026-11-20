package workload

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	sim "github.com/inference-sim/inference-sim/sim"
)

// This file is the constant-load mode of AgentXReplay: aiperf 0.12.0 run without --scenario
// over a Weka agentic corpus (the substrate contracts/spec/workload ConstantLoad,
// inference-sim#2124). The harness then runs PhaseType.CONCURRENCY through RequestRateStrategy
// (timing/config.py:47; timing/strategies/request_rate.py), not the agentic strategy, so:
//
//   - there are N session slots, one per live root conversation (credit/issuer.py:320-330); a
//     slot frees when the root's final turn returns, whatever its sub-agents are still doing
//     (credit/callback_handler.py:620-623), and the freed slot starts the next root at once
//     (the CONCURRENCY_BURST interval is zero);
//   - sub-agent requests take no slot, so requests in flight can exceed N, and a session in a
//     think-time gap holds its slot with nothing in flight, so they can fall below it;
//   - a session always starts at its root's first turn: no snapshot, no primers, no replay
//     barrier (timing/phase/runner.py:187-193 builds it for agentic replay only), and no global
//     idle cap (config/phases.py:338-352);
//   - the next root comes from the dataset sampler over the corpus's roots, sequential for Weka
//     (dataset/dataset_samplers.py:46-63), shared by the warmup and measured phases;
//   - the per-trace idle-gap cap is a compression of the trace's request-start timeline made
//     when the corpus is loaded (dataset/loader/weka_trace.py:320-345, 836-880), not a runtime
//     watchdog.
//
// The warmup phase runs the same load for WarmupUs of sending, stops sending, and the measured
// phase begins when the last warmup request has returned, with fresh sessions and the server's
// cache as the warmup left it. (All citations are to aiperf v0.12.0, src/aiperf/.)

// Session recycling rules of AgentXConstantLoad.Recycling.
const (
	AgentXRecycleSequential = "sequential"
	AgentXRecycleShuffle    = "shuffle"
	AgentXRecycleRandom     = "random"
)

// AgentXConstantLoad switches AgentXReplay to the constant-load mode. Concurrency is then the
// number of session slots.
type AgentXConstantLoad struct {
	// Recycling picks the next corpus root when a slot frees: sequential (exact), or shuffle and
	// random, which draw from the named stream workload/agentx/recycle (DEV-6) and so match
	// aiperf's rule but not its numpy draws. Empty is sequential.
	Recycling string
	// MeasuredSessions caps the sessions the measured phase may start (--num-sessions); 0 is
	// bounded by the window alone.
	MeasuredSessions int
	// WarmupUs is the warmup phase's sending length; 0 has no warmup phase.
	WarmupUs int64
}

// agxConstant is the mode's run state.
type agxConstant struct {
	cfg      AgentXConstantLoad
	roots    []*AgentXTrace // the sampler's pool with empty roots dropped
	next     int            // sequential and shuffled cursor
	order    []int          // shuffle: the current pass
	rng      *rand.Rand
	measured int // sessions started in the measured phase
}

func newAgxConstant(cfg AgentXConstantLoad, pool []*AgentXTrace, seed int64) (*agxConstant, error) {
	switch cfg.Recycling {
	case "":
		cfg.Recycling = AgentXRecycleSequential
	case AgentXRecycleSequential, AgentXRecycleShuffle, AgentXRecycleRandom:
	default:
		return nil, fmt.Errorf("session recycling %q is not sequential, shuffle or random", cfg.Recycling)
	}
	if cfg.MeasuredSessions < 0 || cfg.WarmupUs < 0 {
		return nil, fmt.Errorf("measured sessions and warmup must not be negative")
	}
	c := &agxConstant{cfg: cfg}
	for _, t := range pool {
		if len(t.convs[0].turns) > 0 {
			c.roots = append(c.roots, t)
		}
	}
	if len(c.roots) == 0 {
		return nil, fmt.Errorf("no root conversation with any turn in the pool")
	}
	if cfg.Recycling != AgentXRecycleSequential {
		c.rng = sim.NewStream(seed, sim.StreamName{Component: "workload/agentx", Purpose: "session-recycle"})
	}
	return c, nil
}

// nextRoot is the dataset sampler's next conversation.
func (c *agxConstant) nextRoot() *AgentXTrace {
	switch c.cfg.Recycling {
	case AgentXRecycleRandom:
		return c.roots[c.rng.Intn(len(c.roots))]
	case AgentXRecycleShuffle:
		if c.next >= len(c.order) {
			c.order = c.rng.Perm(len(c.roots))
			c.next = 0
		}
		t := c.roots[c.order[c.next]]
		c.next++
		return t
	}
	if c.next >= len(c.roots) {
		c.next = 0
	}
	t := c.roots[c.next]
	c.next++
	return t
}

// startConstant is Start for the constant-load mode: every slot begins a session at time 0.
func (x *AgentXReplay) startConstant() {
	if x.constant.cfg.WarmupUs > 0 {
		x.phase = agxWarmup
		// The warmup's sending window is a timer the idle cap never moves, like the measured one.
		x.push(&agxDispatch{state: agxTimer, warmupEnd: true}, x.constant.cfg.WarmupUs)
		x.seedConstant()
		return
	}
	x.startProfiling(0)
}

// seedConstant starts a fresh session in every slot.
func (x *AgentXReplay) seedConstant() {
	for li := range x.laneInst {
		x.startSession(li)
	}
}

// startSession puts the sampler's next root in slot lane and sends its first turn.
func (x *AgentXReplay) startSession(lane int) {
	tr := x.constant.nextRoot()
	in := x.newInstance(lane, tr)
	in.originMs = tr.convs[0].turns[0].tsMs
	x.laneInst[lane] = in
	if x.phase == agxProfile {
		x.constant.measured++
	}
	x.submit(&agxDispatch{inst: in, stream: in.root, turn: 0, start: true})
}

// sessionAllowed is SessionCountStopCondition.can_start_new_session
// (timing/phase/stop_conditions.py:187-193): the measured phase starts no more than
// --num-sessions sessions.
func (x *AgentXReplay) sessionAllowed() bool {
	c := x.constant
	return x.phase != agxProfile || c.cfg.MeasuredSessions == 0 || c.measured < c.cfg.MeasuredSessions
}

// releaseSlot frees the slot of in when its root's final turn has returned, and starts the
// slot's next session at once. It is the legacy per-root release (credit/callback_handler.py:
// 620-623), not the agentic tree-drain release.
func (x *AgentXReplay) releaseSlot(in *agxInstance) {
	if x.laneInst[in.lane] != in || in.recycled {
		return
	}
	in.recycled = true
	if !x.canSend() || !x.sessionAllowed() {
		return
	}
	x.startSession(in.lane)
}

// endWarmup stops the warmup's sending and waits for its requests to return.
func (x *AgentXReplay) endWarmup() {
	x.closeWindow(agxWarmDrain)
}

// agxWarpTrace returns tr with every request-start gap longer than capMs cut to capMs, as the
// Weka loader does when --trace-idle-gap-cap-seconds is set (weka_trace.py:320-345, 836-880):
// the first capMs after a request are kept, the rest of the gap is removed, and every later
// timestamp moves earlier by what was removed. The gaps are those between consecutive request
// starts over the whole trace, root and sub-agents together. Each stream's end-to-start delay
// is rederived from the warped starts (max(0, start gap - the previous request's recorded
// server time)), and branch start times move onto the same timeline. The input is unchanged.
func agxWarpTrace(tr *AgentXTrace, capMs float64) *AgentXTrace {
	var starts []float64
	for ci := range tr.convs {
		for _, t := range tr.convs[ci].turns {
			starts = append(starts, t.tsMs)
		}
	}
	sort.Float64s(starts)
	type gap struct{ from, to, before, cap float64 }
	var gaps []gap
	shift := 0.0
	for i := 1; i < len(starts); i++ {
		if g := starts[i] - starts[i-1]; g > capMs {
			gaps = append(gaps, gap{starts[i-1], starts[i], shift, capMs})
			shift += g - capMs
		}
	}
	warp := func(t float64) float64 {
		for _, g := range gaps {
			switch {
			case t < g.from:
				return t - g.before
			case t < g.to:
				if t-g.from <= g.cap {
					return t - g.before
				}
				return g.from - g.before + g.cap
			}
		}
		return t - shift
	}
	out := *tr
	out.convs = make([]agxConv, len(tr.convs))
	for ci := range tr.convs {
		c := tr.convs[ci]
		c.turns = append([]agxTurn(nil), c.turns...)
		for k := range c.turns {
			c.turns[k].tsMs = warp(c.turns[k].tsMs)
			if k > 0 {
				c.turns[k].delayMs = math.Max(0, c.turns[k].tsMs-c.turns[k-1].tsMs-c.turns[k-1].apiMs)
				c.turns[k].hasDelay = true
			}
		}
		out.convs[ci] = c
	}
	out.branches = append([]agxBranch(nil), tr.branches...)
	for i := range out.branches {
		out.branches[i].startMs = warp(out.branches[i].startMs)
	}
	out.minMs, out.maxMs = warp(tr.minMs), warp(tr.maxMs)
	return &out
}
