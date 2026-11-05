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
}

// arriveRequest records that req reached the gateway.
func (c *ClusterSimulator) arriveRequest(req *sim.Request) {
	if req.Redirected {
		return
	}
	if c.arrivalGate != nil {
		c.noteFateErr(c.fates.Create(req.ID))
	}
	c.noteFateErr(c.fates.Arrive(req.ID))
	c.liveRequests[req.ID] = req
}

// endRequest writes req's fate at the current site. It is called before the gateway, the
// session driver or any hook hears of the end (SIM-8).
func (c *ClusterSimulator) endRequest(req *sim.Request, f sim.Fate, reason string, at int64) {
	if req == nil {
		return
	}
	c.noteFateErr(c.fates.End(req.ID, f, reason, at))
	delete(c.liveRequests, req.ID)
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
// one created whose arrival never ran is not started; one the cluster holds is in flight (an
// engine runs it, or its KV is moving between a prefill and a decode engine) or queued (in the
// gateway, the event pipeline or an engine's queue).
func (c *ClusterSimulator) endAtHorizon() {
	at := c.config.Horizon
	for _, id := range c.fates.Unended(sim.LifecycleCreated) {
		c.noteFateErr(c.fates.End(id, sim.FateNotStarted, "", at))
	}
	for _, id := range c.fates.Unended(sim.LifecycleArrived) {
		req := c.liveRequests[id]
		if req == nil {
			c.noteFateErr(fmt.Errorf("request %s arrived and has no fate, and the cluster no longer holds it", id))
			continue
		}
		c.endRequest(req, c.horizonFate(req), "", at)
	}
}

// horizonFate is the fate of a request the cluster holds when the horizon stops the run.
func (c *ClusterSimulator) horizonFate(req *sim.Request) sim.Fate {
	if p := c.parentRequests[req.ID]; p != nil && p.OriginalRequest == req {
		if _, prefilling := c.pendingPrefillCompletions[p.PrefillSubReqID]; prefilling {
			return runningFate(p.PrefillSubReq)
		}
		if _, waiting := c.remotePulls[p.DecodeSubReqID]; waiting {
			return sim.FateQueuedAtHorizon // its decode engine has not started the pull
		}
		if d := p.DecodeSubReq; d != nil && d.State == sim.StateQueued {
			return sim.FateQueuedAtHorizon // landed on the decode engine, waiting (or preempted)
		}
		return sim.FateInFlightAtHorizon // its KV is being handed off, or it is decoding
	}
	return runningFate(req)
}

func runningFate(req *sim.Request) sim.Fate {
	if req != nil && req.State == sim.StateRunning {
		return sim.FateInFlightAtHorizon
	}
	return sim.FateQueuedAtHorizon
}

// stalled reports a run that ended before its horizon with requests the cluster still holds:
// no event remained that could move them (SIM-7). Their fate is unknown, so the run has no
// result.
func (c *ClusterSimulator) stalled() error {
	held := c.fates.Unended(sim.LifecycleArrived)
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
