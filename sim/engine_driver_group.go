package sim

import (
	"errors"
	"fmt"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// A data-parallel group behind the contract with one instance per rank. Each rank's instance has
// its own EngineDriver over the rank's view of the group engine. Rank 0's driver, the leader,
// steps the group: its schedule runs every rank's share of the lockstep step, and each request's
// share is applied by the driver that holds the request, so every instance keeps its own report.
// A member's arrivals and timeouts reach the group through its own view, and its asks for a step
// wake the leader.

// JoinGroup makes d, the driver of rank k's instance, a member of the group led by leader; the
// leader is rank 0's driver. Members must join before the run starts.
func (d *EngineDriver) JoinGroup(leader *EngineDriver, k int) {
	if k == 0 {
		if d != leader {
			panic("engine driver: rank 0's driver leads its group")
		}
		return
	}
	d.leader = leader
	leader.members = append(leader.members, d)
	if len(leader.members) != k {
		panic(fmt.Sprintf("engine driver: rank %d joined its group out of order", k))
	}
}

// ShareAdmissionStage makes the drivers of one endpoint's ranks queue on one admission stage:
// the stage of the first driver, which every rank's engine states alike. It is for the ranks
// behind one balancer, whose endpoint admits every request before the balancer picks a rank
// (INV-R1). A no-op when the engine states no queue.
func ShareAdmissionStage(drivers []*EngineDriver) error {
	if len(drivers) == 0 || drivers[0].stage == nil {
		return nil
	}
	for _, d := range drivers {
		if d.stage == nil {
			return fmt.Errorf("ShareAdmissionStage: the ranks of one endpoint must all state a queue")
		}
		d.stage = drivers[0].stage
	}
	return nil
}

// PickRank is the rank the engine's data-parallel balancer sends a request reaching the replica's
// endpoint at now to: an engine whose group picks the rank (a PickRank query), given the front
// end's count of its unfinished requests on each rank. It reports false for an engine that picks
// none.
func (d *EngineDriver) PickRank(now int64, inFlight []int) (int, bool, error) {
	p, ok := d.eng.(interface {
		PickRank(engine.Time, []int) (int, error)
	})
	if !ok {
		return 0, false, nil
	}
	k, err := p.PickRank(now, inFlight)
	return k, true, err
}

// LeadsGroupEngine reports whether the driver's engine can lead a data-parallel group's lockstep
// step — it implements the group-step surface (StepDue/LockstepSteps/LastActive). The cluster
// wires the sim leader/member path (JoinGroup) only for such an engine, the shared-group external
// build; a replica whose ranks are separately-built engines is coordinated at the cluster level
// instead (sim/cluster eventMembers).
func (d *EngineDriver) LeadsGroupEngine() bool {
	_, ok := d.eng.(groupEngine)
	return ok
}

// groupOf is the driver whose request id is, among the leader and its members.
func (d *EngineDriver) groupOf(id string) *EngineDriver {
	if _, ok := d.reqs[id]; ok {
		return d
	}
	for _, m := range d.members {
		if _, ok := m.reqs[id]; ok {
			return m
		}
	}
	return nil
}

// drainMembers runs, on every member, the events due at now that order before a step (arrival,
// queueing): the group's step sits in the leader's queue and the cluster breaks a time tie by the
// lowest instance index, so without this a request queued on rank k > 0 at the step's own tick
// would wait a whole step that one queued on rank 0 at that tick joins.
func (d *EngineDriver) drainMembers(now int64) {
	for _, m := range d.members {
		for len(m.eventQueue) > 0 {
			head := m.eventQueue[0].event
			if head.Timestamp() != now || head.Priority() >= PriorityStep {
				break
			}
			m.ProcessNextEvent()
		}
	}
}

// splitGroupBatch is the leader's share of a group step and each member's: a batch per driver
// holding the requests it runs, and its preemptions.
func (d *EngineDriver) splitGroupBatch(b engine.Batch) map[*EngineDriver]engine.Batch {
	out := map[*EngineDriver]engine.Batch{}
	for _, s := range b.Reqs {
		if o := d.groupOf(s.ID); o != nil {
			ob := out[o]
			ob.ID = b.ID
			ob.Reqs = append(ob.Reqs, s)
			out[o] = ob
		}
	}
	for _, id := range b.Preempted {
		if o := d.groupOf(id); o != nil {
			ob := out[o]
			ob.Preempted = append(ob.Preempted, id)
			out[o] = ob
		}
	}
	return out
}

// splitGroupResult is each driver's share of a group step's result.
func (d *EngineDriver) splitGroupResult(res engine.Result) map[*EngineDriver]engine.Result {
	out := map[*EngineDriver]engine.Result{}
	get := func(id string) (*EngineDriver, engine.Result) {
		o := d.groupOf(id)
		r := out[o]
		if r.Emitted == nil {
			r.Emitted = map[string]int{}
		}
		return o, r
	}
	for id, n := range res.Emitted {
		if o, r := get(id); o != nil {
			r.Emitted[id] = n
			out[o] = r
		}
	}
	for _, id := range res.FirstTokens {
		if o, r := get(id); o != nil {
			r.FirstTokens = append(r.FirstTokens, id)
			out[o] = r
		}
	}
	for _, f := range res.Finished {
		if o, r := get(f.ID); o != nil {
			r.Finished = append(r.Finished, f)
			out[o] = r
		}
	}
	for _, h := range res.Handoffs {
		if o, r := get(h.ID); o != nil {
			r.Handoffs = append(r.Handoffs, h)
			out[o] = r
		}
	}
	return out
}

// groupEngine is the leader's engine: a data-parallel group's rank 0 view, which steps the whole
// group. No engine in this tree implements it now (the rank view it was written against left
// with the retired compatibility engine), so a group's leader reaches kickGroup's panic.
type groupEngine interface {
	StepDue(now engine.Time) bool
	LockstepSteps() (steps, dummy int)
	LastActive() []bool
}

// groupBatch is one lockstep step the leader keeps in flight: the contract batch, each driver's
// share of it (from splitGroupBatch, so the completion applies to the same drivers that ran it),
// and when the step ends. It is the group analog of driverBatch.
type groupBatch struct {
	b      engine.Batch
	shares map[*EngineDriver]engine.Batch
	end    int64
}

// kickGroup is the leader's kick: the group's lockstep steps at now, with every member's
// same-instant arrivals delivered first. It mirrors the normal driver's kick (engine_driver.go):
// the leader schedules a step, keeps its batch in flight and completes it at the step's end
// (driverGroupStepEnd) rather than applying the result at the step's start, and loops to schedule
// as many steps as the group's batch queue admits — under async scheduling a rank's queue holds
// two steps (InFlightLimit 2), so the leader fills the queue in one instant exactly as the normal
// kick does over this engine (BC-H2-5); the engine's own ErrNoCapacity / open / busyUntil guards
// stop the loop. Each rank with work takes the step's instant as its clock; when rank 0 runs in no
// step this instant, the leader's clock stays at its last real event. The step-end event re-kicks
// the leader, and the dummy / nothing-due cases re-wake through NextWake (rearmWake).
func (d *EngineDriver) kickGroup(now int64) {
	g, ok := d.eng.(groupEngine)
	if !ok {
		panic(fmt.Sprintf("engine driver: the engine %T leads a group but does not step one", d.eng))
	}
	if !g.StepDue(now) {
		// No step is due (the queue is full, one is mid-dummy, or no rank has work): the leader's
		// rank, idle, admits its waiting handoffs at once, then re-arms its wake.
		d.admitIdle(now)
		d.rearmWake(now)
		return
	}
	d.drainMembers(now)
	drivers := append([]*EngineDriver{d}, d.members...)
	// Each idle rank first admits its waiting handoffs, as the cluster's idle-decode admission does
	// at the end of the step executing, before the group's next step at that instant.
	for _, o := range drivers {
		o.admitIdle(now)
	}
	prev := d.preEventClock
	rank0Active := false
	for {
		before, _ := g.LockstepSteps()
		b, err := d.eng.Schedule(now)
		if errors.Is(err, engine.ErrNoCapacity) {
			break
		}
		if err != nil {
			panic(fmt.Sprintf("engine driver: the group's Schedule at %d: %v", now, err))
		}
		if after, _ := g.LockstepSteps(); after == before {
			// Schedule began no iteration (its open / busyUntil / not-stepping guards): nothing more
			// runs this instant.
			break
		}
		// The pulls the step's schedules admitted, each started by the driver that holds its request.
		for _, t := range b.TransfersStarted {
			for _, o := range drivers {
				if _, ok := o.remote[t.Request]; ok {
					o.startTransfers([]engine.Transfer{t}, now)
					break
				}
			}
		}
		active := g.LastActive()
		for k, a := range active {
			if a {
				drivers[k].Clock = max(drivers[k].Clock, now)
				drivers[k].stepIdx++
				drivers[k].noteStep(now)
			}
		}
		rank0Active = rank0Active || active[0]
		shares := d.splitGroupBatch(b)
		for _, o := range drivers {
			if ob, ran := shares[o]; ran {
				o.notePreempted(ob)
			}
		}
		for _, f := range b.Finished {
			// A request the step ended without running it (engine.Batch.Finished) ends on the rank
			// that holds it, at the step's start, as the normal driver finishes such a request in kick.
			if o := d.groupOf(f.ID); o != nil {
				o.finishUnrun([]engine.Finished{f}, now)
			}
		}
		if b.Empty() {
			// A dummy-only step runs no forward pass and is not in flight: the engine sets busyUntil
			// and the group wakes when the dummy batches end (NextWake). No more runs this instant.
			break
		}
		// A real step runs a forward pass: time it (gpuFreeAt serialises steps exactly as the normal
		// driver's timeBatch does over this same engine), admit each driver's share at the step's
		// start, record each rank busy, and keep the batch in flight to complete at its end.
		start, end := d.timeBatch(b, now)
		for _, o := range drivers {
			if ob, ran := shares[o]; ran {
				o.admit(ob, now, start)
				o.Metrics.Busy.Record(d.eng, ob, start, end)
			}
		}
		d.groupInFlight = append(d.groupInFlight, groupBatch{b: b, shares: shares, end: end})
		d.schedule(&driverGroupStepEnd{at: end})
	}
	if !rank0Active {
		// Rank 0 ran in no step this instant: its driver's clock stays at its last real event, as the
		// host loop's group leaves an idle rank 0.
		d.Clock = prev
	}
	d.rearmWake(now)
	for _, m := range d.members {
		m.rearmWake(now)
	}
}

// driverGroupStepEnd completes the oldest in-flight lockstep step at its step's end, splits its
// result across the ranks' drivers so each keeps its own report, delivers each rank's KV events to
// that rank's endpoint, and re-kicks the leader for the next step — the group analog of
// driverStepEnd's Complete/apply/kick.
type driverGroupStepEnd struct {
	at int64
}

func (e *driverGroupStepEnd) Timestamp() int64    { return e.at }
func (e *driverGroupStepEnd) Priority() int       { return PriorityStep }
func (e *driverGroupStepEnd) run(d *EngineDriver) { d.groupStepEnd(e.at) }

func (d *EngineDriver) groupStepEnd(end int64) {
	if len(d.groupInFlight) == 0 {
		panic(fmt.Sprintf("engine driver: a group step ended at %d with no batch in flight", end))
	}
	gb := d.groupInFlight[0]
	d.groupInFlight = d.groupInFlight[1:]
	res, err := d.eng.Complete(gb.b.ID, d.groupOutcome(gb.shares), end)
	if err != nil {
		panic(fmt.Sprintf("engine driver: the group's Complete(%d) at %d: %v", gb.b.ID, end, err))
	}
	results := d.splitGroupResult(res)
	drivers := append([]*EngineDriver{d}, d.members...)
	// Each rank's KV events reach only its own endpoint (the gateway keys delivery by instance id):
	// splitGroupResult cannot place a Removed or Cleared event (it names no request), so the events
	// are taken from the producing rank the engine recorded (LastKVEventsByRank), index r being
	// rank r's driver.
	if kv, ok := d.eng.(interface {
		LastKVEventsByRank() [][]engine.KVEvent
	}); ok {
		for rank, evs := range kv.LastKVEventsByRank() {
			if rank >= len(drivers) || len(evs) == 0 {
				continue
			}
			o := drivers[rank]
			r := results[o]
			r.KVEvents = evs
			results[o] = r
		}
	}
	for _, o := range drivers {
		o.apply(gb.shares[o], results[o], end)
		for range results[o].Finished {
			// A finished request leaves its rank's instance when the step ends (the host loop's
			// RequestLeftEvent), which is the instance's last event if nothing follows it.
			o.schedule(&driverLeftEvent{at: end})
		}
	}
	// Schedule the next lockstep step while this one has just freed its slot, as driverStepEnd
	// re-kicks the normal driver.
	d.kickGroup(end)
}

// groupOutcome is the group step's acceptance outcome, each request's accepted drafts computed on
// the driver that holds it (its own Acceptance and carry), the per-endpoint form of the normal
// driver's outcome. Acceptance zero (the common case) yields the empty outcome on every driver, so
// this leaves a non-speculating run unchanged.
func (d *EngineDriver) groupOutcome(shares map[*EngineDriver]engine.Batch) engine.Outcome {
	var out engine.Outcome
	for o, ob := range shares {
		oo := o.outcome(ob)
		for id, n := range oo.Accepted {
			if out.Accepted == nil {
				out.Accepted = map[string]int{}
			}
			out.Accepted[id] = n
		}
	}
	return out
}

// admitIdle starts the pulls d's rank admits while it has no other work (an AdmitIdle query).
func (d *EngineDriver) admitIdle(now int64) {
	if a, ok := d.eng.(interface {
		AdmitIdle(engine.Time) []engine.Transfer
	}); ok {
		d.startTransfers(a.AdmitIdle(now), now)
	}
}

// kickMember is a member's kick: its rank admits its waiting handoffs if it is idle, and the
// group's leader steps the group at now if a step is due (queued in the leader's own queue, as
// the host loop arms the group's step in rank 0's queue), or else wakes when one is.
func (d *EngineDriver) kickMember(now int64) {
	d.admitIdle(now)
	d.rearmWake(now)
	if g, ok := d.leader.eng.(groupEngine); ok && g.StepDue(now) {
		d.leader.stepAfterArrivals(now)
		return
	}
	d.leader.rearmWake(now)
}

// driverLeftEvent is a finished request leaving its instance at its step's end: it does nothing
// but advance the instance's clock there, as the compatibility engine's RequestLeftEvent does.
type driverLeftEvent struct {
	at int64
}

func (e *driverLeftEvent) Timestamp() int64  { return e.at }
func (e *driverLeftEvent) Priority() int     { return PriorityRequestLeft }
func (e *driverLeftEvent) run(*EngineDriver) {}
