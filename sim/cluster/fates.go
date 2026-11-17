package cluster

// The cluster's half of DEV-9 and SIM-8 (sim/fate.go): it creates every request in the fate
// ledger at the one place requests enter the cluster (pushArrival, or the arrival gate's
// admission), writes each request's fate at the site that ends it, before the gateway or a
// session driver is told, and gives every request still unended at the horizon its fate. At
// the end of a run the ledger's count of requests created must equal the fates written, and the
// fates must agree with INV-1's terminal buckets.

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// Fates returns the run's fate ledger. Every request the cluster created has exactly one fate
// once Run has returned without error.
func (c *ClusterSimulator) Fates() *sim.FateLedger { return c.fates }

// createRequest registers req in the fate ledger. A drain redirect re-pushes a request the
// ledger already holds; a placeholder an arrival gate has not yet admitted is not a request
// (ArrivalGate), so with a gate the request is created when the gate admits it.
func (c *ClusterSimulator) createRequest(req *sim.Request) {
	if req.Redirected || c.arrivalGate != nil {
		return
	}
	c.noteFateErr(c.fates.Create(req.ID))
	c.noteFacts(req)
}

// RequestFact is what a request's Result row states that no metric keeps: the request as the
// workload produced it.
type RequestFact struct {
	ArrivalUs    int64
	PromptTokens int
	Model        string
	Adapter      string
	SLOClass     string
	TenantID     string
	SessionID    string
	RoundIndex   int
	Phase        sim.Phase
	Agentic      *sim.AgenticTag
}

// RequestFacts returns the facts of every created request, by ID.
func (c *ClusterSimulator) RequestFacts() map[string]RequestFact { return c.requestFacts }

func (c *ClusterSimulator) noteFacts(req *sim.Request) {
	if c.requestFacts == nil {
		c.requestFacts = make(map[string]RequestFact)
	}
	c.requestFacts[req.ID] = RequestFact{ArrivalUs: req.ArrivalTime, PromptTokens: int(req.InputLen()),
		Model: req.Model, Adapter: req.Adapter, SLOClass: req.SLOClass, TenantID: req.TenantID,
		SessionID: req.SessionID, RoundIndex: req.RoundIndex, Phase: req.Phase, Agentic: req.Agentic}
}

// arriveRequest records that req reached the gateway.
func (c *ClusterSimulator) arriveRequest(req *sim.Request) {
	if req.Redirected {
		return
	}
	if c.arrivalGate != nil {
		c.noteFateErr(c.fates.Create(req.ID))
		c.noteFacts(req)
	}
	c.noteFateErr(c.fates.Arrive(req.ID))
	c.liveRequests[req.ID] = req
}

// SetPlannedRounds installs the source of the rounds sessions planned and never issued. At the
// end of the run each is created and ends not started, with a row (SIM-8). Must be called
// before Run().
func (c *ClusterSimulator) SetPlannedRounds(f func() []sim.PlannedRound) {
	if c.hasRun {
		panic("ClusterSimulator: SetPlannedRounds must be called before Run()")
	}
	c.plannedRounds = f
}

// endUnissuedRounds creates each round the sessions planned and never issued, and ends it not
// started at the instant the run ended (the horizon, or the clock of a run that drained first).
func (c *ClusterSimulator) endUnissuedRounds(at int64) {
	if c.plannedRounds == nil && c.gate == nil {
		return
	}
	// A planned round arrives, by its session's plan, its think times after the clock its
	// session's last issued round ended at.
	type tail struct {
		round   int
		endedAt int64
	}
	last := map[string]tail{}
	for _, rec := range c.fates.Records() {
		f, ok := c.requestFacts[rec.ID]
		if !ok || f.SessionID == "" {
			continue
		}
		if t, seen := last[f.SessionID]; !seen || f.RoundIndex > t.round {
			last[f.SessionID] = tail{f.RoundIndex, rec.At}
		}
	}
	var planned []sim.PlannedRound
	if c.plannedRounds != nil {
		planned = c.plannedRounds()
	}
	if c.gate != nil {
		planned = append(planned, c.gate.Unsent(at)...)
	}
	for _, r := range planned {
		c.noteFateErr(c.fates.CreatePlanned(r.ID))
		c.noteFateErr(c.fates.End(r.ID, sim.FateNotStarted, "", at))
		if c.requestFacts == nil {
			c.requestFacts = make(map[string]RequestFact)
		}
		c.requestFacts[r.ID] = RequestFact{Model: r.Model, Adapter: r.Adapter, SLOClass: r.SLOClass,
			TenantID: r.TenantID, SessionID: r.SessionID, RoundIndex: r.RoundIndex,
			ArrivalUs: last[r.SessionID].endedAt + r.ThinkUs, PromptTokens: r.PromptTokens, Phase: r.Phase}
		if r.HasArrival {
			c.requestFacts[r.ID] = withArrival(c.requestFacts[r.ID], r.ArrivalUs)
		}
	}
}

// advanceRequest moves req along the state machine; a move with no edge fails the run.
func (c *ClusterSimulator) advanceRequest(req *sim.Request, to sim.LifecycleState) {
	if req == nil {
		return
	}
	c.noteFateErr(c.fates.Advance(req.ID, to))
}

// noteRunState hears an engine start running a request or preempt it, and moves the request the
// client sent: a colocated request itself, a P/D request through its parent, on the stage the
// sub-request is on.
func (c *ClusterSimulator) noteRunState(req *sim.Request, running bool) {
	target := req
	to := sim.LifecycleRunning
	if parent := c.parentOf(req); parent != nil {
		target = parent.OriginalRequest
		if req.IsDecodeSubRequest {
			to = sim.LifecycleDecoding
		} else {
			to = sim.LifecyclePrefilling
		}
	}
	if !running {
		to = sim.LifecyclePreempted
	}
	c.advanceRequest(target, to)
}

// endRequest writes req's fate at the current site. It is called before the gateway, the
// session driver or any hook hears of the end (SIM-8).
func (c *ClusterSimulator) endRequest(req *sim.Request, f sim.Fate, reason string, at int64) {
	if req == nil {
		return
	}
	c.noteFateErr(c.fates.End(req.ID, f, reason, at))
	delete(c.liveRequests, req.ID)
	if c.gate != nil && !c.horizonReached {
		// A phase drains when every request of it has its fate; the next phase is sent from that
		// clock. A request cannot arrive before the clock it is sent at.
		for _, next := range c.gate.Ended(req, at) {
			c.pushArrival(next, max(next.ArrivalTime, c.clock), &c.workloadSender)
		}
	}
}

// withArrival returns f with its arrival set.
func withArrival(f RequestFact, at int64) RequestFact {
	f.ArrivalUs = at
	return f
}

// RequestGate sequences a workload's phases: it hears each request end and returns the requests
// to send as a result (sim/workload.PhaseGate).
type RequestGate interface {
	Ended(req *sim.Request, at int64) []*sim.Request
	Unsent(at int64) []sim.PlannedRound
}

// SetGate installs the gate that sends the later phases. Must be called before Run().
func (c *ClusterSimulator) SetGate(g RequestGate) {
	if c.hasRun {
		panic("ClusterSimulator: SetGate must be called before Run()")
	}
	c.gate = g
}

// endByOutcome writes the fate a DoneOutcome names; reason is the drop reason for DoneDropped.
func (c *ClusterSimulator) endByOutcome(req *sim.Request, o DoneOutcome, reason string, at int64) {
	switch o {
	case DoneCompleted:
		c.endRequest(req, sim.FateCompleted, "", at)
	case DoneTimedOut:
		c.endRequest(req, sim.FateTimedOut, "", at)
	case DoneDropped:
		c.endRequest(req, sim.FateDropped, reason, at)
	case DoneEvicted:
		c.endRequest(req, sim.FateDropped, sim.DropGatewayEvicted, at)
	default:
		c.noteFateErr(fmt.Errorf("request %s ended with unknown outcome %d", req.ID, o))
	}
}

// rejectionReason is the drop reason of a gateway rejection; a client timeout is not a drop.
func rejectionFate(why Rejection) (sim.Fate, string) {
	switch why {
	case RejectedAdmission:
		return sim.FateDropped, sim.DropAdmission
	case RejectedRouting:
		return sim.FateDropped, sim.DropRouting
	case RejectedGatewayCapacity:
		return sim.FateDropped, sim.DropGatewayCapacity
	case RejectedGatewayExpired:
		return sim.FateDropped, sim.DropGatewayExpired
	case RejectedClientTimeout:
		return sim.FateTimedOut, ""
	case RejectedUnservable:
		return sim.FateDropped, sim.DropUnservable
	}
	panic(fmt.Sprintf("ClusterSimulator.Reject: unknown rejection %d", why))
}

func (c *ClusterSimulator) noteFateErr(err error) {
	if err != nil && c.fateErr == nil {
		c.fateErr = err
	}
}

// endAtHorizon gives every request without a fate its fate when the run stops at the horizon:
// one created whose arrival never ran is not started; one the cluster holds has the fate its
// state gives (sim.HorizonFate).
func (c *ClusterSimulator) endAtHorizon() {
	c.horizonReached = true // the requests the horizon ends release nothing
	at := c.config.Horizon
	for _, id := range c.fates.Unended(sim.LifecycleCreated) {
		c.noteFateErr(c.fates.End(id, sim.FateNotStarted, "", at))
	}
	for _, id := range c.fates.Held() {
		req := c.liveRequests[id]
		if req == nil {
			c.noteFateErr(fmt.Errorf("request %s arrived and has no fate, and the cluster no longer holds it", id))
			continue
		}
		st, _ := c.fates.State(id)
		f, _ := sim.HorizonFate(st)
		c.endRequest(req, f, "", at)
	}
}

// stalled reports a run that ended before its horizon with requests the cluster still holds:
// no event remained that could move them (SIM-7). Their fate is unknown, so the run has no
// result.
func (c *ClusterSimulator) stalled() error {
	held := c.fates.Held()
	if len(held) == 0 {
		return nil
	}
	n := len(held)
	if len(held) > 5 {
		held = append(held[:5], "...")
	}
	return fmt.Errorf("%w: the run ended at %d µs, before its horizon, with %d request(s) held and no event that can move them: %v",
		sim.ErrNoProgress, c.clock, n, held)
}

// checkFates is DEV-9's end-of-run check: every request created has one fate, and the fates
// agree with INV-1's terminal buckets, which other code counts at other sites. Called after the
// aggregated metrics are final.
func (c *ClusterSimulator) checkFates() error {
	if c.fateErr != nil {
		return c.fateErr
	}
	if err := c.fates.Check(); err != nil {
		return err
	}
	t := c.fates.Tally()
	m := c.aggregatedMetrics
	held := m.StillQueued + m.StillRunning + c.GatewayQueueDepth()
	pairs := []struct {
		name         string
		fates, other int
	}{
		{"completed", t.ByFate[sim.FateCompleted], m.CompletedRequests},
		{"timed_out", t.ByFate[sim.FateTimedOut], m.TimedOutRequests},
		{"dropped admission (rejected)", t.Drops[sim.DropAdmission], c.rejectedRequests},
		{"dropped routing (routing_rejections)", t.Drops[sim.DropRouting], c.routingRejections},
		{"dropped gateway_capacity (gateway_queue_rejected)", t.Drops[sim.DropGatewayCapacity], c.gatewayRejected},
		{"dropped gateway_expired", t.Drops[sim.DropGatewayExpired], c.gatewayExpired},
		{"dropped gateway_evicted", t.Drops[sim.DropGatewayEvicted], c.gatewayEvicted},
		{"dropped unservable and decode_kv (dropped_unservable)",
			t.Drops[sim.DropUnservable] + t.Drops[sim.DropDecodeKV], m.DroppedUnservable},
		{"in flight and queued at the horizon (still_queued + still_running + gateway_queue_depth)",
			t.ByFate[sim.FateInFlightAtHorizon] + t.ByFate[sim.FateQueuedAtHorizon], held},
	}
	for _, p := range pairs {
		if p.fates != p.other {
			return fmt.Errorf("request conservation (DEV-9): %d requests have fate %s, and the INV-1 bucket counts %d",
				p.fates, p.name, p.other)
		}
	}
	return nil
}
