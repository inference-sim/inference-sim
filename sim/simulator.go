// sim/simulator.go: the host half of an instance (#2021 B3): the clock, the event queue, when
// to step and when to park. The engine it embeds is in engine.go.
package sim

import (
	"container/heap"

	"github.com/sirupsen/logrus"
)

// eventEntry wraps an Event with a sequence ID for deterministic ordering.
// The EventQueue orders by (Timestamp, Priority, seqID), matching the
// cluster event queue's scheme. seqID breaks ties within same-type same-timestamp events.
type eventEntry struct {
	event Event
	seqID int64
}

// EventQueue implements heap.Interface and orders events by (timestamp, priority, seqID).
// This ensures deterministic same-tick event ordering (INV-6 improvement).
type EventQueue []eventEntry

func (eq EventQueue) Len() int { return len(eq) }

func (eq EventQueue) Less(i, j int) bool {
	if eq[i].event.Timestamp() != eq[j].event.Timestamp() {
		return eq[i].event.Timestamp() < eq[j].event.Timestamp()
	}
	if eq[i].event.Priority() != eq[j].event.Priority() {
		return eq[i].event.Priority() < eq[j].event.Priority()
	}
	return eq[i].seqID < eq[j].seqID
}

func (eq EventQueue) Swap(i, j int) { eq[i], eq[j] = eq[j], eq[i] }

func (eq *EventQueue) Push(x any) {
	*eq = append(*eq, x.(eventEntry))
}

func (eq *EventQueue) Pop() any {
	old := *eq
	n := len(old)
	item := old[n-1]
	*eq = old[:n-1]
	return item
}

// Simulator is one instance: the host's event loop and clock (inference-sim's, #2021 B3) around
// the engine that makes every scheduling, KV-cache and metric decision (EngineCore). The host
// fields stay with the simulator when the engine moves to blis-vllm (#2027, addendum 4 item 2): the clock,
// the horizon, the event queue and its seqIDs, the pending step, the idle park, the progress
// hook and the callbacks. EngineCore is embedded so its fields and methods read as the
// simulator's own while the split is made one method at a time.
type Simulator struct {
	EngineCore

	Clock   int64
	Horizon int64
	// Metrics is the instance's report record (inference-sim's, #2021 B2). The engine writes it
	// only through its recorder (EngineCore.Rec).
	Metrics *Metrics
	// eventQueue has all the simulator events, like arrival and step events
	eventQueue EventQueue
	stepEvent  Event
	// busyUntil is when the last executed step ends (its start plus its duration). A step
	// that empties the running batch leaves stepEvent nil at that step's START, so an
	// arrival during the step must not start the next one before this (idleStepTime).
	busyUntil int64
	// Cumulative count of idle wakes superseded by scheduleStepBy — the one path whose
	// effect is invisible in metrics (a cancelled step executes nothing), so without a
	// counter a test cannot tell whether it exercised the path at all. Load-independent
	// diagnostic, in the DeferralsStarted/promotionsFired family (#1586).
	idleWakesSuperseded int64
	// lockstep is the data-parallel group this rank steps with (#1909); nil for an
	// independent engine.
	lockstep *LockstepGroup
	// preEventClock is Clock before the event being processed advanced it, so a lockstep
	// dummy step can leave the clock at the rank's last real work.
	preEventClock int64
	// park is the wait of an engine whose step scheduled nothing (parkIdle); nil otherwise.
	park *idlePark
	// idlePolling disables parking (SetIdlePolling); eventsProcessed counts executed events.
	idlePolling     bool
	eventsProcessed int64
	// curPriority is the priority of the event executing, -1 outside one.
	curPriority int
	seqCounter  int64 // monotonic counter for event queue seqID (deterministic ordering)
	// OnRequestDone is an optional callback invoked when a request reaches a terminal
	// state (completed, length-capped, or timed out). Returns follow-up requests to inject.
	// Set by the caller (cmd/root.go or ClusterSimulator). Nil = no callback.
	OnRequestDone func(req *Request, tick int64) []*Request
	// OnFirstToken is an optional callback invoked once per prefill completion, when a request's
	// first output token is produced. at is the simulated time the client receives that token
	// (ArrivalTime + FirstTokenTime). It must not mutate simulation state. Nil = no callback.
	OnFirstToken func(req *Request, at int64)

	progressHook               ProgressHook
	simClockProgressIntervalUs int64
	nextSnapshotClockUs        int64
}

func (sim *Simulator) ScheduleTimeout(req *Request, t int64) {
	sim.Schedule(&TimeoutEvent{time: t, Request: req})
}
func (sim *Simulator) ScheduleScheduled(req *Request, t int64) {
	sim.Schedule(&ScheduledEvent{time: t, Request: req})
}
func (sim *Simulator) ScheduleLeft(req *Request, t int64) {
	sim.Schedule(&RequestLeftEvent{time: t, Request: req})
}
func (sim *Simulator) ScheduleAdapterLoaded(adapter string, t int64) {
	sim.Schedule(&AdapterLoadCompletionEvent{time: t, Adapter: adapter})
}
func (sim *Simulator) WakeStep(t int64) { sim.scheduleStepBy(t) }
func (sim *Simulator) ArmStep(t int64)  { sim.scheduleStepAt(t) }
func (sim *Simulator) StepArmed() bool  { return sim.stepEvent != nil }
func (sim *Simulator) FirstTokenAt(req *Request, at int64) {
	if sim.OnFirstToken != nil {
		sim.OnFirstToken(req, at)
	}
}
func (sim *Simulator) RequestDone(req *Request, tick int64) {
	if sim.OnRequestDone != nil {
		for _, next := range sim.OnRequestDone(req, tick) {
			sim.InjectArrival(next)
		}
	}
}

// NewSimulator creates a Simulator from a SimConfig struct and pre-built dependencies.
// All workload generation now happens externally — callers inject requests via InjectArrival.
func NewSimulator(cfg SimConfig, kvStore KVStore, latencyModel LatencyModel) (*Simulator, error) {
	eng, err := NewEngineCore(cfg, kvStore, latencyModel)
	if err != nil {
		return nil, err
	}
	s := &Simulator{
		EngineCore: *eng,
		Clock:      0,
		Horizon:    cfg.Horizon,
		Metrics:    NewMetrics(),
		eventQueue: make(EventQueue, 0),
		stepEvent:  nil,
	}
	s.Host = s
	s.Rec = s.Metrics
	return s, nil
}

// Schedule pushes an event into the simulator's EventQueue with a monotonic seqID.
// Note, this has nothing to do with vLLM's scheduler.schedule().
func (sim *Simulator) Schedule(ev Event) {
	sim.seqCounter++
	heap.Push(&sim.eventQueue, eventEntry{event: ev, seqID: sim.seqCounter})
}

// HasPendingEvents returns true if the EventQueue is non-empty.
func (sim *Simulator) HasPendingEvents() bool {
	return len(sim.eventQueue) > 0
}

// PeekNextEventTime returns the timestamp of the earliest pending event.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
func (sim *Simulator) PeekNextEventTime() int64 {
	return sim.eventQueue[0].event.Timestamp()
}

// ProcessNextEvent pops the earliest event, advances Clock, executes it, and returns it.
// The returned Event lets callers react to what happened (e.g., detect QueuedEvent for
// pending-request tracking) without maintaining fragile before/after heuristics.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
// Does NOT check horizon — caller is responsible.
//
// Special case — lazy cancellation: events are never removed from the heap when they
// become obsolete; they are popped and skipped. A popped event that has been cancelled
// is returned immediately WITHOUT advancing Clock or calling Execute(), so Finalize()
// takes SimEndedTime from the last real-work event rather than from a no-op far in the
// future. Two kinds of event are cancelled this way, each recognized differently:
//
//   - a TimeoutEvent whose request already completed (State == StateCompleted) — the
//     real-world equivalent of a client cancelling its deadline timer the moment the
//     response arrives;
//   - a StepEvent marked cancelled — an idle wake (see scheduleNextStep/idleDeferralWake)
//     that scheduleStepBy superseded because work arrived before the wake tick.
//
// INV-3 (clock monotonicity) is stated over *processed* events and explicitly excludes
// this restore, so skipping does not break it. ClusterSimulator mirrors the skip by
// restoring prevClusterClock for both kinds (sim/cluster/cluster.go).
func (sim *Simulator) ProcessNextEvent() Event {
	entry := heap.Pop(&sim.eventQueue).(eventEntry)
	ev := entry.event

	// Lazy cancellation: a TimeoutEvent whose request already completed is an
	// orphan — the real-world equivalent of a client cancelling its deadline
	// timer when the response arrives. Skip it before advancing the clock so
	// Finalize() captures SimEndedTime from the last real-work event, not from
	// an orphaned no-op timeout 300s in the future.
	if te, ok := ev.(*TimeoutEvent); ok && te.Request.State == StateCompleted {
		return ev
	}
	// Same for a superseded idle wake (scheduleStepBy): it must not advance the clock
	// or end-of-run time.
	if se, ok := ev.(*StepEvent); ok && se.cancelled {
		return ev
	}

	sim.preEventClock = sim.Clock
	sim.Clock = ev.Timestamp()
	sim.eventsProcessed++
	sim.curPriority = ev.Priority()
	logrus.Debugf("[tick %07d] Executing %T", sim.Clock, ev)
	ev.Execute(sim)
	if sim.park != nil && !sim.HasWaiting() {
		sim.dropPark(sim.Clock)
	}
	sim.curPriority = -1
	return ev
}

// Finalize records end-of-run state and sets SimEndedTime.
// Call once after the event loop ends. Called by both sim.Run() (single-instance)
// and ClusterSimulator.Run() (cluster mode via inst.Finalize()).
func (sim *Simulator) Finalize() {
	sim.finalizePark()
	// Record conservation fields (BC-8, BC-9) — must happen in Finalize
	// because cluster mode drives events via ProcessNextEvent() directly
	// and never calls sim.Run().
	sim.RecordStillHeld()
	sim.Metrics.SimEndedTime = min(sim.Clock, sim.Horizon)
	logrus.Infof("[tick %07d] Simulation ended", sim.Clock)
}

// InjectArrival schedules an ArrivalEvent for req and registers it in Metrics.Requests.
func (sim *Simulator) InjectArrival(req *Request) {
	if req.ArrivalTime > sim.Horizon {
		logrus.Warnf("InjectArrival: request %s has ArrivalTime %d > Horizon %d; "+
			"ArrivalEvent will not fire (INV-1 conservation may be affected)",
			req.ID, req.ArrivalTime, sim.Horizon)
	}
	sim.Schedule(&ArrivalEvent{time: req.ArrivalTime, Request: req})
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
}

// InjectArrivalAt schedules an ArrivalEvent at eventTime (not req.ArrivalTime).
// Metrics.Requests uses req.ArrivalTime for ArrivedAt to preserve original arrival time.
// Used by cluster-mode online routing where event time differs from original arrival.
func (sim *Simulator) InjectArrivalAt(req *Request, eventTime int64) {
	sim.Schedule(&ArrivalEvent{time: eventTime, Request: req})
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
}

func (sim *Simulator) Run() {
	for sim.HasPendingEvents() {
		sim.ProcessNextEvent()
		if sim.Clock > sim.Horizon {
			break
		}
		sim.maybeDeliverProgressSnapshot(false)
	}
	sim.maybeDeliverProgressSnapshot(true)
	sim.Finalize()
}

// SetProgressHook registers an optional hook that receives periodic state
// snapshots during simulation execution. Must be called before Run().
// When hook is nil (default), there is zero behavioral or performance impact.
// simClockIntervalUs controls the minimum simulation-clock interval (microseconds)
// between periodic snapshots. If simClockIntervalUs <= 0, only the final snapshot
// is delivered.
func (sim *Simulator) SetProgressHook(hook ProgressHook, simClockIntervalUs int64) {
	sim.progressHook = hook
	if simClockIntervalUs > 0 {
		sim.simClockProgressIntervalUs = simClockIntervalUs
		sim.nextSnapshotClockUs = simClockIntervalUs
	}
}

func (sim *Simulator) maybeDeliverProgressSnapshot(isFinal bool) {
	if sim.progressHook == nil {
		return
	}
	if !isFinal && (sim.simClockProgressIntervalUs <= 0 || sim.Clock < sim.nextSnapshotClockUs) {
		return
	}
	clock := sim.Clock
	if isFinal {
		clock = min(sim.Clock, sim.Horizon)
	}
	snap := ProgressSnapshot{
		Clock:             clock,
		TotalCompleted:    sim.Metrics.CompletedRequests,
		TotalTimedOut:     sim.Metrics.TimedOutRequests,
		TotalDropped:      sim.Metrics.DroppedUnservable,
		TotalInputTokens:  sim.Metrics.TotalInputTokens,
		TotalOutputTokens: sim.Metrics.TotalOutputTokens,
		TotalPreemptions:  sim.Metrics.PreemptionCount,
		InstanceSnapshots: []InstanceSnapshot{sim.buildInstanceSnapshot()},
		TotalInstances:    1,
		ActiveInstances:   1,
		IsFinal:           isFinal,
	}
	sim.progressHook.OnProgress(snap)
	if !isFinal {
		sim.nextSnapshotClockUs += sim.simClockProgressIntervalUs
	}
}

func (sim *Simulator) buildInstanceSnapshot() InstanceSnapshot {
	return InstanceSnapshot{
		ID:                "instance-0",
		Model:             sim.ModelName,
		State:             InstanceStateActive,
		QueueDepth:        sim.QueueDepth(),
		BatchSize:         sim.BatchSize(),
		KVUtilization:     float64(sim.KVCache.UsedBlocks()) / float64(max(sim.KVCache.TotalCapacity(), 1)),
		KVFreeBlocks:      sim.KVCache.TotalCapacity() - sim.KVCache.UsedBlocks(),
		KVTotalBlocks:     sim.KVCache.TotalCapacity(),
		CacheHitRate:      sim.KVCache.CacheHitRate(),
		PreemptionCount:   sim.Metrics.PreemptionCount,
		CompletedRequests: sim.Metrics.CompletedRequests,
		TimedOutRequests:  sim.Metrics.TimedOutRequests,
	}
}

// CurrentClock returns the current simulation clock (in ticks).
func (sim *Simulator) CurrentClock() int64 { return sim.Clock }

// SimHorizon returns the simulation horizon (in ticks).
func (sim *Simulator) SimHorizon() int64 { return sim.Horizon }

// IdleWakesSuperseded returns how many times work arriving during an idle deferral wait
// superseded the pending wake (scheduleStepBy). A load-independent diagnostic, and the
// only way a test can assert it reached that path: the superseded step executes nothing,
// so it leaves no trace in any metric. 0 for a run that never slept on a deferral.
func (sim *Simulator) IdleWakesSuperseded() int64 { return sim.idleWakesSuperseded }

// scheduleStepBy makes sure a StepEvent fires no later than t. It arms one at t
// (scheduleStepAt) when none is pending, and supersedes a pending idle wake that is later
// than t, so work that arrives during an idle wait is served at once (INV-8, #1886). Any
// other pending step is left alone, exactly as the previous stepEvent == nil guards did; a
// lockstep rank's stepEvent is its group's step, which is never an idle wake.
func (sim *Simulator) scheduleStepBy(t int64) {
	if sim.stepEvent != nil {
		pending, ok := sim.stepEvent.(*StepEvent)
		if !ok || !pending.idleWake || pending.time <= t {
			return
		}
		pending.cancelled = true
		sim.idleWakesSuperseded++
		sim.stepEvent = nil
	}
	sim.scheduleStepAt(t)
}

// scheduleStepAt arms this instance's next step at time t, or at the end of the step still
// executing (idleStepTime, #1944). A data-parallel rank in a lockstep group (#1909) asks its
// group instead, which steps every rank together; a group step sets every rank's busyUntil,
// so neither path starts a step while one is running.
func (sim *Simulator) scheduleStepAt(t int64) {
	t = sim.idleStepTime(t)
	if sim.park != nil {
		t = sim.unpark(t)
	}
	if sim.lockstep != nil {
		sim.lockstep.wake(t)
		return
	}
	step := &StepEvent{time: t}
	sim.stepEvent = step
	sim.Schedule(step)
}

// idleStepTime is the earliest a step triggered at t by new work on an idle instance may
// start: t, or the end of the step still executing when the batch it ran emptied. The engine
// runs one forward pass at a time, so work arriving during a step waits for it to end, as it
// does whenever the batch is not empty (scheduleNextStep).
func (sim *Simulator) idleStepTime(t int64) int64 { return max(t, sim.busyUntil) }

// Step simulates a single vllm step(): batch scheduling, model execution, mirroring, and completion.
// Phases: (1) schedule batch, (2) execute prefill/decode, (2.5) mirror to CPU, (3) process completions, (4) schedule next step.
//
// Orphaned StepEvent guard: when a TimeoutEvent empties the RunningBatch it leaves
// sim.stepEvent pointing to the already-scheduled StepEvent (preventing the cascade
// described in #1096). If that StepEvent fires and finds nothing to do, clearing
// sim.stepEvent here prevents future QueuedEvent INV-8 guards from seeing a stale
// non-nil pointer and skipping their step-scheduling.
func (sim *Simulator) Step(now int64) {
	if sim.NothingToStep() {
		sim.stepEvent = nil
		sim.IdleStep(now)
		return
	}
	advance, remaining := sim.runStep(now)
	sim.scheduleNextStep(now, advance, remaining)
	sim.PublishStep(now + advance)
}

// runStep runs one engine step at now -- schedule, execute, mirror to the CPU tier, process
// completions -- and returns its duration and the requests still running. busyUntil is set
// before completions are processed, as it always was, so a completion callback that injects
// work sees the engine busy until the step ends (#1944).
func (sim *Simulator) runStep(now int64) (advance int64, remaining []*Request) {
	advance = sim.ExecuteStep(now)
	sim.busyUntil = now + advance
	return advance, sim.FinishStep(now, advance)
}

// scheduleNextStep handles Phase 4: schedules the next step event based on
// remaining requests, or starts a new batch if only WaitQ has pending work
// (work-conserving property, INV-8).
func (sim *Simulator) scheduleNextStep(now, currStepAdvance int64, remaining []*Request) {
	if len(remaining) > 0 {
		sim.KeepRunning(remaining)
		// estimate queue overhead from LR (sim.features)
		//
		sim.scheduleStepAt(now + currStepAdvance)
	} else {
		sim.ClearRunning()
		sim.stepEvent = nil
		// Work-conserving: if WaitQ has pending requests, immediately
		// schedule a new step to form the next batch. Without this,
		// queued requests are stranded until the next arrival event
		// triggers a QueuedEvent — violating the work-conserving
		// property that real vLLM maintains.
		//
		// Exception (cold-load gate, #1466): when a per-instance adapter load is in
		// flight, the wait-queue head is gated (held out of the batch) for the load
		// duration and nothing else can be admitted (blocking model) — including any
		// warm-adapter or base-model requests queued behind it (intentional
		// head-of-line blocking, DT-faithful; design §7/D1). The scheduled
		// work IS the load, so INV-8 is satisfied by the pending
		// AdapterLoadCompletionEvent — which re-forms a step on completion via
		// ScheduleStepIfIdle. Scheduling an empty step here instead would spin one
		// step per tick for the whole load. Inert when no LoRA (loadingAdapter == "").
		//
		// Idle wake: when every waiting request is deferred on a KV fetch, an empty step
		// costs 1 tick and admits nothing, so re-polling every tick crawls through a
		// millisecond transfer in thousands of steps. Sleep until the next tick a
		// deferred request can change state instead; arrivals pull the wake forward
		// (scheduleStepBy). Admission lands on the same tick either way.
		//
		// The wake is clamped to the horizon so a run truncated mid-deferral ends where
		// the 1-tick crawl ended. The cluster loop tests the horizon BEFORE executing an
		// instance event (sim/cluster/cluster.go), so a wake past the horizon would never
		// be processed, leaving the instance clock at the last real event and reporting a
		// SimEndedTime short of the horizon — a different throughput denominator than
		// before this optimization. Clamped, the step fires AT the horizon, admits
		// nothing, and the next step at horizon+1 ends the run with the clock at the
		// horizon, exactly as the crawl did. Inert at the default unlimited horizon.
		if sim.HasWaiting() && !sim.AdapterLoading() {
			if sim.canPark() {
				sim.parkIdle(now+currStepAdvance, currStepAdvance)
				return
			}
			// A deferral makes the KV store non-quiescent, so an engine that waits on one
			// never parks; it sleeps to the deferral's wake instead (#1886). A lockstep
			// rank keeps stepping with its group.
			next := now + currStepAdvance
			if wake, ok := sim.IdleDeferralWake(now); ok && wake > next && sim.lockstep == nil {
				if wake > sim.Horizon {
					wake = sim.Horizon
				}
				if wake > next {
					step := &StepEvent{time: sim.idleStepTime(wake), idleWake: true}
					sim.stepEvent = step
					sim.Schedule(step)
					return
				}
			}
			sim.scheduleStepAt(next)
		}
	}
}

// BusyUntil is when the engine's last executed step ends; its next schedule runs no earlier.
func (sim *Simulator) BusyUntil() int64 { return sim.busyUntil }

// ShareRendererOf makes the ranks of one endpoint share one render stage (ShareRenderer, over
// the ranks' engines).
func ShareRendererOf(ranks []*Simulator, apiServers int) error {
	engines := make([]*EngineCore, len(ranks))
	for i, r := range ranks {
		engines[i] = &r.EngineCore
	}
	return ShareRenderer(engines, apiServers)
}

// BuildServerSeriesOf assembles the server time series of the instances (BuildServerSeries,
// over their engines, each binned to its own clock).
func BuildServerSeriesOf(ids []string, sims []*Simulator) *ServerSeriesOutput {
	engines := make([]*EngineCore, len(sims))
	clocks := make([]int64, len(sims))
	for i, s := range sims {
		engines[i], clocks[i] = &s.EngineCore, s.Clock
	}
	return BuildServerSeries(ids, engines, clocks)
}
