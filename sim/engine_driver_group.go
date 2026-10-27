package sim

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/engine"
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

// ShareRenderStage makes the drivers of one endpoint's ranks render on one stage of apiServers
// servers of workers threads each: the replica behind vLLM's internal DP balancer, whose API
// servers render every request before the balancer picks a rank (INV-R1). A no-op when the
// drivers' front end is inert (each rendered alone, with the run's --renderer-num-workers).
func ShareRenderStage(drivers []*EngineDriver, apiServers, workers int) error {
	if len(drivers) == 0 || drivers[0].renderer == nil {
		return nil
	}
	r, err := NewRenderer(apiServers, workers)
	if err != nil {
		return err
	}
	for _, d := range drivers {
		if d.renderer == nil {
			return fmt.Errorf("ShareRenderStage: the ranks of one endpoint must all render")
		}
		d.renderer = r
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

// groupEngine is the leader's engine: a data-parallel group's rank 0 view (blis-vllm
// compat/contract.RankView), which steps the whole group.
type groupEngine interface {
	StepDue(now engine.Time) bool
	LockstepSteps() (steps, dummy int)
	LastActive() []bool
}

// kickGroup is the leader's kick: the group's lockstep step at now, when it is due, with every
// member's same-instant arrivals delivered first, and each driver's share of it applied by that
// driver at the step's start (the compatibility engine's host loop applies a step's results when
// the step starts). Every rank with work takes the step's instant as its clock; when rank 0 has
// none, the leader's clock stays at its last real event, as the host loop's group leaves it.
func (d *EngineDriver) kickGroup(now int64) {
	g, ok := d.eng.(groupEngine)
	if !ok {
		panic(fmt.Sprintf("engine driver: the engine %T leads a group but does not step one", d.eng))
	}
	if !g.StepDue(now) {
		// No step is due: the leader's rank, idle, admits its waiting handoffs at once.
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
	before, _ := g.LockstepSteps()
	b, err := d.eng.Schedule(now)
	if err != nil {
		panic(fmt.Sprintf("engine driver: the group's Schedule at %d: %v", now, err))
	}
	if after, _ := g.LockstepSteps(); after == before {
		d.rearmWake(now)
		return
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
		}
	}
	if !active[0] {
		d.Clock = prev
	}
	shares := d.splitGroupBatch(b)
	end := now
	var results map[*EngineDriver]engine.Result
	if !b.Empty() {
		end = now + max(d.eng.StepTime(b), 1)
		res, err := d.eng.Complete(b.ID, engine.Outcome{}, end)
		if err != nil {
			panic(fmt.Sprintf("engine driver: the group's Complete(%d) at %d: %v", b.ID, now, err))
		}
		results = d.splitGroupResult(res)
	}
	for _, o := range drivers {
		ob, ran := shares[o]
		if !ran {
			continue
		}
		o.notePreempted(ob)
		o.admit(ob, now, now)
	}
	for _, o := range drivers {
		r := results[o]
		o.apply(shares[o], r, end)
		for range r.Finished {
			// A finished request leaves its rank's instance when the step ends (the host loop's
			// RequestLeftEvent), which is the instance's last event if nothing follows it.
			o.schedule(&driverLeftEvent{at: end})
		}
	}
	d.rearmWake(now)
	for _, m := range d.members {
		m.rearmWake(now)
	}
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
