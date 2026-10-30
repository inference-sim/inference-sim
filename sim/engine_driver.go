// sim/engine_driver.go: the host half of an instance whose engine is reached only through
// blis-schemas' engine.Engine contract (#2021 B3, GOAL.md). The host keeps the clock, the event
// queue, the timeouts and the report; the engine makes every scheduling, KV-cache and output
// decision and prices its own steps.
package sim

import (
	"container/heap"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"sort"
	"strings"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/blis-schemas/engine"
)

// EngineDriverConfig is the host side of an engine-driven instance.
type EngineDriverConfig struct {
	// Horizon ends the run: no event after it runs.
	Horizon int64
	// PostDecodeFixedOverhead is the host's fixed per-request work at completion, in µs (the
	// latency backend's PostDecodeFixedOverhead), which a completion's end-to-end latency
	// includes, as the compatibility engine's report does.
	PostDecodeFixedOverhead int64
	// OutputTokenProcessingTime is the host's per-emitted-token work in µs (detokenization and
	// streaming), which delays the client's view of a token.
	OutputTokenProcessingTime int64
	// QueueingTime is the host work before a request reaches the engine (tokenization and
	// prompt preprocessing), in µs; nil means none.
	QueueingTime func(req *Request) int64
	// Acceptance is the run's mean draft acceptance rate α in [0, 1], an input the simulator
	// owns (#2021 B3: spec-decode acceptance is an input). A request that verifies drafts in a
	// batch is told how many of them the target accepted (engine.Outcome.Accepted), by the
	// compatibility engine's deterministic carry: a mean of α x drafts per step, no RNG
	// (commitDecodeTokens). Zero accepts none.
	Acceptance float64
	// ResumeGapFromLastToken measures the gap before a preempted request's first token after it
	// resumes from its last token's emission (its arrival plus its TTFT and ITL so far), as the
	// compatibility engine's executeBatchStep does for a resumed prefill; off, from the end of
	// the last step that emitted for it. Only the compatibility engine's contract build sets it.
	ResumeGapFromLastToken bool
	// ScrapeEngineMetrics answers a scrape of the model server from the engine's own exported
	// metrics (num_requests_waiting, num_requests_running, kv_cache_usage_perc,
	// num_preemptions), as the engine published them, rather than from its live state. Only the
	// compatibility engine's contract build sets it.
	ScrapeEngineMetrics bool
	// ApplyAtStepStart applies a batch's results when its step starts, as the compatibility
	// engine's own host loop does (Simulator.Step executes a whole step at its start, so its
	// completions, first tokens and finishes reach the cluster then, stamped with the step's
	// end): the driver completes the batch at once, passing the step's end, as the contract allows
	// a simulator that applies results at a step's start. Only the compatibility engine's
	// contract build sets it.
	ApplyAtStepStart bool
}

// EngineDriver drives one engine.Engine from the host's event loop. It is the 0.32.0 engine's
// counterpart of Simulator: the same events reach it (arrivals, steps, timeouts), and it records
// the same report, but every engine decision is the engine's, through the contract only.
type EngineDriver struct {
	eng engine.Engine
	cfg EngineDriverConfig

	Clock   int64
	Horizon int64
	Metrics *Metrics

	eventQueue EventQueue
	seqCounter int64

	// reqs are the requests the engine holds, by ID.
	reqs map[string]*driverReq
	// inFlight are the batches scheduled and not yet completed, oldest first, each with when its
	// step ends.
	inFlight []driverBatch
	// gpuFreeAt is when the GPU finishes the last batch scheduled: the next one starts no earlier.
	gpuFreeAt int64
	// stageFreeAt is when each pipeline stage finishes the batch it holds, for an engine that
	// splits its steps by stage (engine.Pipelined); nil for one stage, where gpuFreeAt is the
	// one device's.
	stageFreeAt []int64
	// tierFreeAt is when the engine's offload-tier streams finish the transfers they hold, by
	// direction: vLLM's CPU offload worker runs stores and loads on separate handlers, each
	// chaining its own transfers (startTierTransfer).
	tierFreeAt map[engine.TransferKind]int64
	// recentStages are the last pipelined batches scheduled, newest last, at most one per stage:
	// each one's sampling requests, its preemptions and when it leaves the last stage (timeBatch's
	// T-pp dependency).
	recentStages []stagedBatch
	// pipelineDrainsAt is when the last batch scheduled leaves the last stage, under a
	// pipelined engine.
	pipelineDrainsAt int64
	// unserved are the requests the engine failed before they emitted anything, dropped as
	// unservable (finish), whose pending deadline no longer applies.
	unserved map[string]bool
	// pendingWake is the wake the engine last reported (0 = none).
	pendingWake *driverWakeEvent
	// renderer is the host's API-server render stage (INV-R1), nil when the front end is inert.
	renderer *Renderer
	// stepPending is set while a step is queued for after the current instant's arrivals.
	stepPending bool

	// OnRequestDone and OnFirstToken are the cluster's callbacks, as on Simulator.
	OnRequestDone func(req *Request, tick int64) []*Request
	OnFirstToken  func(req *Request, at int64)

	stepIdx int
	// preEventClock is the clock before the event running now (a group's dummy-only step restores
	// it, kickGroup).
	preEventClock int64
	// kvEventSink receives the engine's prefix-cache changes (SetKVEventSink); nil when none.
	kvEventSink func(at int64, events []KVEvent)

	// P/D handoffs (engine_driver_handoff.go): remote are the decode sub-requests the engine
	// holds before their pull lands, pulls the pulls in flight, and holds the KV the engine keeps
	// for a peer's pull, all by request ID. OnPullStarted is the host's pricing of a pull the
	// engine started, at the instant it started.
	remote        map[string]*Request
	pulls         map[string]engine.TransferID
	holds         map[string]engine.Handoff
	OnPullStarted func(req *Request, pulledTokens int64, at int64)

	// A data-parallel group with one instance per rank (engine_driver_group.go): leader is rank
	// 0's driver for a member, nil for the leader or an engine of its own; members are the
	// leader's other ranks' drivers, in rank order.
	leader  *EngineDriver
	members []*EngineDriver
}

type driverReq struct {
	req          *Request
	scheduledIdx int
	lastEnd      int64 // the end of the last step that emitted a token for the request
	emitted      int
	acceptCarry  float64 // the fraction of an accepted draft the request is owed
	resumed      bool    // preempted after its first token, not yet emitting again
	lookedUp     bool    // its first admission's prefix-cache lookup is recorded
	remote       bool    // a decode sub-request whose prompt a peer's prefill computed (LandRemote)
}

type driverBatch struct {
	b   engine.Batch
	end int64
}

// stagedBatch is a pipelined batch's sampling requests, the requests its schedule preempted, and
// when it left the last stage.
type stagedBatch struct {
	reqs, preempted map[string]bool
	end             int64
}

// NewEngineDriver wraps an engine for the host's event loop, with an inert front end: a
// request's QueueingTime is an uncontended delay.
func NewEngineDriver(eng engine.Engine, cfg EngineDriverConfig) *EngineDriver {
	return &EngineDriver{eng: eng, cfg: cfg, Horizon: cfg.Horizon, Metrics: NewMetrics(),
		reqs: map[string]*driverReq{}}
}

// NewEngineDriverWithFrontend wraps an engine behind the host's API-server front end, the
// compatibility path's render stage (FrontendConfig, Renderer; INV-R1): each request's
// QueueingTime runs on one of the endpoint's render threads, in arrival order, and the request
// reaches the engine when its render finishes. The front end is the host's, not the engine's.
func NewEngineDriverWithFrontend(eng engine.Engine, cfg EngineDriverConfig, f FrontendConfig) (*EngineDriver, error) {
	if err := f.Validate(); err != nil {
		return nil, fmt.Errorf("NewEngineDriverWithFrontend: %w", err)
	}
	d := NewEngineDriver(eng, cfg)
	if f.Enabled() {
		r, err := NewRenderer(max(f.APIServerCount, 1), f.RendererNumWorkers)
		if err != nil {
			return nil, err
		}
		d.renderer = r
	}
	return d, nil
}

// Engine returns the engine the driver holds.
func (d *EngineDriver) Engine() engine.Engine { return d.eng }

// Renderer is the host's API-server render stage requests reach the engine through (INV-R1), nil
// when the front end is inert.
func (d *EngineDriver) Renderer() *Renderer { return d.renderer }

// DriverEvent is an event of an engine-driven instance.
type DriverEvent interface {
	Timestamp() int64
	Priority() int
	run(d *EngineDriver)
}

// The driver's queue orders its events by EventQueue's (timestamp, priority, seqID), by
// wrapping each driver event in an Event shim, which the cluster loop handles as any instance
// event: a driver event runs on its driver.
type driverShim struct {
	ev DriverEvent
	d  *EngineDriver
}

func (s *driverShim) Timestamp() int64 { return s.ev.Timestamp() }
func (s *driverShim) Priority() int    { return s.ev.Priority() }
func (s *driverShim) Execute()         { s.ev.run(s.d) }

func (d *EngineDriver) schedule(ev DriverEvent) {
	d.seqCounter++
	heap.Push(&d.eventQueue, eventEntry{event: &driverShim{ev: ev, d: d}, seqID: d.seqCounter})
}

// inPipeline counts the requests whose arrival or render is still a queued event: accepted by
// the host and not yet handed to the engine, and not timed out (a timeout is its own bucket).
func (d *EngineDriver) inPipeline() int {
	n := 0
	for _, e := range d.eventQueue {
		var req *Request
		switch ev := e.event.(*driverShim).ev.(type) {
		case *driverArrivalEvent:
			req = ev.req
		case *driverQueuedEvent:
			req = ev.req
		}
		if req != nil && req.State != StateTimedOut {
			n++
		}
	}
	return n
}

// HasPendingEvents reports whether an event is queued.
func (d *EngineDriver) HasPendingEvents() bool { return len(d.eventQueue) > 0 }

// PeekNextEventTime is the earliest queued event's time. Callers check HasPendingEvents first.
func (d *EngineDriver) PeekNextEventTime() int64 { return d.eventQueue[0].event.Timestamp() }

// ProcessNextEvent pops the earliest event, advances the clock and runs it. A superseded wake, or
// the timeout of a request that already completed, is returned without advancing the clock, as
// Simulator drops a superseded idle wake and an orphaned timeout (lazy cancellation); Skipped
// reports it.
func (d *EngineDriver) ProcessNextEvent() Event {
	entry := heap.Pop(&d.eventQueue).(eventEntry)
	shim := entry.event.(*driverShim)
	if w, ok := shim.ev.(*driverWakeEvent); ok && w.cancelled {
		return entry.event
	}
	if te, ok := shim.ev.(*driverTimeoutEvent); ok && te.dead(d) {
		return entry.event
	}
	d.preEventClock = d.Clock
	d.Clock = shim.Timestamp()
	shim.ev.run(d)
	return entry.event
}

// Skipped reports whether ev, an event ProcessNextEvent returned, was dropped without running
// (lazy cancellation), so a caller that advanced its own clock to ev's time restores it, as it
// does for the compatibility engine's dropped events.
func (s *driverShim) Skipped() bool {
	switch e := s.ev.(type) {
	case *driverWakeEvent:
		return e.cancelled
	case *driverTimeoutEvent:
		return e.dead(s.d)
	}
	return false
}

// Run runs the instance alone to its horizon or until no event remains.
func (d *EngineDriver) Run() {
	for d.HasPendingEvents() {
		if d.PeekNextEventTime() > d.Horizon {
			break
		}
		d.ProcessNextEvent()
	}
	d.Finalize()
}

// Finalize records end-of-run state. A request the host accepted that the horizon caught in the
// pipeline before the engine -- its arrival or its render still an event -- is waiting too: it is
// in the host's records and in no other INV-1 bucket.
func (d *EngineDriver) Finalize() {
	waiting, running := 0, 0
	for _, r := range d.reqs {
		if r.req.State == StateRunning {
			running++
		} else {
			waiting++
		}
	}
	waiting += d.inPipeline()
	d.Metrics.StillQueued = waiting
	d.Metrics.StillRunning = running
	if q, ok := d.eng.(interface {
		QueueSnapshots() (waiting, running []int)
	}); ok {
		// The engine's per-step queue lengths, which the cluster's head-of-line blocking
		// detector averages (the compatibility engine records them at each step).
		d.Metrics.NumWaitQRequests, d.Metrics.NumRunningBatchRequests = q.QueueSnapshots()
	}
	if k, ok := d.eng.(interface {
		KVThrashing() (rate float64, events, opportunities int64, counted bool)
	}); ok {
		// The KV store's thrash counts, which the cluster aggregates (the compatibility engine's
		// offload stores count them).
		rate, events, opportunities, counted := k.KVThrashing()
		d.Metrics.KVThrashingRate = rate
		if counted {
			d.Metrics.KVThrashEvents, d.Metrics.KVThrashOpportunities = events, opportunities
		}
	}
	if a, ok := d.eng.(interface {
		AdapterEvents() (loads, evictions map[string]int64)
	}); ok {
		// The engine's cold LoRA adapter loads and evictions (INV-L3), which the report prints
		// per adapter.
		loads, evictions := a.AdapterEvents()
		maps.Copy(d.Metrics.AdapterLoadCounts, loads)
		maps.Copy(d.Metrics.AdapterEvictionCounts, evictions)
	}
	d.Metrics.SimEndedTime = min(d.Clock, d.Horizon)
	logrus.Infof("[tick %07d] Simulation ended", d.Clock)
}

// ResidentAdapterIDs are the LoRA adapters resident on the engine, for an engine that reports
// them; nil otherwise.
func (d *EngineDriver) ResidentAdapterIDs() []string {
	if a, ok := d.eng.(interface{ ResidentAdapterIDs() []string }); ok {
		return a.ResidentAdapterIDs()
	}
	return nil
}

// AdapterCapacity is the engine's resident LoRA adapter slots, for an engine that reports them;
// 0 otherwise.
func (d *EngineDriver) AdapterCapacity() int {
	if a, ok := d.eng.(interface{ AdapterCapacity() int }); ok {
		return a.AdapterCapacity()
	}
	return 0
}

// AdapterPins is how many of the engine's requests pin each LoRA adapter's resident slot, for an
// engine that reports it (an AdapterPins query); nil otherwise.
func (d *EngineDriver) AdapterPins() map[string]int {
	if a, ok := d.eng.(interface{ AdapterPins() map[string]int }); ok {
		return a.AdapterPins()
	}
	return nil
}

// InjectArrival schedules a request's arrival and registers it in the report.
func (d *EngineDriver) InjectArrival(req *Request) {
	d.InjectArrivalAt(req, req.ArrivalTime)
}

// InjectArrivalAt schedules a request's arrival at eventTime, keeping its own ArrivalTime.
func (d *EngineDriver) InjectArrivalAt(req *Request, eventTime int64) {
	d.schedule(&driverArrivalEvent{at: eventTime, req: req})
	d.Metrics.Requests[req.ID] = NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
}

// driverArrivalEvent delivers a request after the host's queueing work: its render, on the
// front end's render stage when there is one (Simulator.QueuedAt's rule).
type driverArrivalEvent struct {
	at  int64
	req *Request
}

func (e *driverArrivalEvent) Timestamp() int64 { return e.at }
func (e *driverArrivalEvent) Priority() int    { return PriorityArrival }
func (e *driverArrivalEvent) run(d *EngineDriver) {
	var render int64
	if d.cfg.QueueingTime != nil {
		render = d.cfg.QueueingTime(e.req)
	}
	at := e.at + render
	if d.renderer != nil {
		at = d.renderer.Admit(e.at, render)
	}
	d.schedule(&driverQueuedEvent{at: at, req: e.req})
}

// driverQueuedEvent hands a request to the engine.
type driverQueuedEvent struct {
	at  int64
	req *Request
}

func (e *driverQueuedEvent) Timestamp() int64 { return e.at }
func (e *driverQueuedEvent) Priority() int    { return PriorityQueued }
func (e *driverQueuedEvent) run(d *EngineDriver) {
	req := e.req
	if req.State == StateTimedOut {
		return
	}
	if req.Deadline > 0 && req.Deadline <= e.at {
		req.State = StateTimedOut
		d.Metrics.TimedOutRequests++
		d.Metrics.RecordTimeout(req)
		d.done(req, e.at)
		return
	}
	er := d.engineRequest(req)
	// A disaggregated prefill keeps its KV for the decode side's pull (Result.Handoffs).
	er.RemoteDecode = req.RetainKVOnFinish
	if err := d.eng.Add(er, e.at); err != nil {
		logrus.Warnf("dropping request %s: the engine refused it: %v", req.ID, err)
		d.Metrics.DroppedUnservable++
		d.Metrics.RecordUnserved(req, false)
		delete(d.Metrics.Requests, req.ID)
		d.done(req, e.at)
		return
	}
	d.Metrics.TotalInputTokens += len(req.InputTokens)
	d.Metrics.EngineAccepted++
	req.State = StateQueued
	d.reqs[req.ID] = &driverReq{req: req}
	if req.Deadline > 0 && req.Deadline <= d.Horizon {
		d.schedule(&driverTimeoutEvent{at: req.Deadline, req: req})
	}
	d.stepAfterArrivals(e.at)
}

// engineRequest is req as the engine is told it.
func (d *EngineDriver) engineRequest(req *Request) engine.Request {
	er := engine.Request{
		ID:              req.ID,
		Prompt:          int32s(req.InputTokens),
		Output:          int32s(req.OutputTokens),
		MaxOutputTokens: req.MaxOutputLen,
		ArrivalTime:     req.ArrivalTime,
		Adapter:         req.Adapter,
	}
	if req.SLOClass != "" {
		// The request's SLO class, an opaque fact for an engine whose priority policy reads it.
		er.Tags = map[string]string{"slo_class": req.SLOClass}
	}
	return er
}

// stepAfterArrivals answers an Add: it asks the engine when to wake, and, when no batch is in
// flight, queues a step at now that runs after every arrival of this instant (the contract's
// same-instant order, arrivals before steps; the compatibility engine's scheduleStepBy), so
// requests that reach the engine together share its next batch. With a batch in flight, vLLM's
// loop waits on that batch's output, and its Complete steps the engine.
func (d *EngineDriver) stepAfterArrivals(now int64) {
	if d.leader != nil {
		// A member's step is its group's, which its leader runs.
		d.kickMember(now)
		return
	}
	d.rearmWake(now)
	if len(d.inFlight) > 0 || d.stepPending {
		return
	}
	d.stepPending = true
	d.schedule(&driverArrivalStep{at: now})
}

// driverArrivalStep is the step stepAfterArrivals queued.
type driverArrivalStep struct {
	at int64
}

func (e *driverArrivalStep) Timestamp() int64 { return e.at }
func (e *driverArrivalStep) Priority() int    { return PriorityStep }
func (e *driverArrivalStep) run(d *EngineDriver) {
	d.stepPending = false
	d.kick(e.at)
}

// driverStepEnd completes the oldest in-flight batch at its step's end; for a batch already
// applied at its start (ApplyAtStepStart), it only steps the engine again.
type driverStepEnd struct {
	at      int64
	applied bool
}

func (e *driverStepEnd) Timestamp() int64 { return e.at }
func (e *driverStepEnd) Priority() int    { return PriorityStep }
func (e *driverStepEnd) run(d *EngineDriver) {
	if e.applied {
		d.kick(e.at)
		return
	}
	if len(d.inFlight) == 0 {
		panic(fmt.Sprintf("engine driver: a step ended at %d with no batch in flight", e.at))
	}
	fb := d.inFlight[0]
	d.inFlight = d.inFlight[1:]
	res, err := d.eng.Complete(fb.b.ID, d.outcome(fb.b), e.at)
	if err != nil {
		panic(fmt.Sprintf("engine driver: Complete(%d) at %d: %v", fb.b.ID, e.at, err))
	}
	d.apply(fb.b, res, e.at)
	d.kick(e.at)
}

// outcome is what executing b produced that the engine does not decide: for each request that
// verified drafts, how many the target accepted. The count is the compatibility engine's carry
// rule (commitDecodeTokens): the request accrues α x drafts each step it verifies, and is told
// the whole part, so the long-run mean is exactly α of the drafts, deterministically (INV-6).
func (d *EngineDriver) outcome(b engine.Batch) engine.Outcome {
	var out engine.Outcome
	if d.cfg.Acceptance <= 0 {
		return out
	}
	for _, s := range b.Reqs {
		r := d.reqs[s.ID]
		if r == nil || s.Drafts <= 0 {
			continue
		}
		r.acceptCarry += d.cfg.Acceptance * float64(s.Drafts)
		n := min(int(r.acceptCarry), s.Drafts)
		r.acceptCarry -= float64(n)
		if n > 0 {
			if out.Accepted == nil {
				out.Accepted = map[string]int{}
			}
			out.Accepted[s.ID] = n
		}
	}
	return out
}

// driverWakeEvent steps the engine at an instant it asked for.
type driverWakeEvent struct {
	at        int64
	cancelled bool
}

func (e *driverWakeEvent) Timestamp() int64 { return e.at }
func (e *driverWakeEvent) Priority() int    { return PriorityStep }
func (e *driverWakeEvent) run(d *EngineDriver) {
	if d.pendingWake == e {
		d.pendingWake = nil
	}
	d.kick(e.at)
}

// driverTimeoutEvent ends a request at its client deadline.
type driverTimeoutEvent struct {
	at  int64
	req *Request
}

func (e *driverTimeoutEvent) Timestamp() int64 { return e.at }
func (e *driverTimeoutEvent) Priority() int    { return PriorityTimeout }
func (e *driverTimeoutEvent) run(d *EngineDriver) {
	req := e.req
	if req.State == StateTimedOut || e.dead(d) {
		return
	}
	if err := d.eng.Finish(req.ID, engine.FinishTimeout, e.at); err != nil {
		panic(fmt.Sprintf("engine driver: Finish(%s): %v", req.ID, err))
	}
	req.State = StateTimedOut
	d.Metrics.TimedOutRequests++
	d.Metrics.RecordTimeout(req)
	delete(d.reqs, req.ID)
	d.done(req, e.at)
	d.kick(e.at)
}

// dead reports a timeout whose request already left the engine: completed, or failed by the
// engine before it emitted anything (finish).
func (e *driverTimeoutEvent) dead(d *EngineDriver) bool {
	return e.req.State == StateCompleted || d.unserved[e.req.ID]
}

// kick schedules as many batches as the engine admits at now, each starting when the GPU is
// free, then asks the engine when to wake. Batches are scheduled only when the GPU has a free
// in-flight slot: Schedule returns ErrNoCapacity at the engine's limit.
func (d *EngineDriver) kick(now int64) {
	if d.leader != nil {
		d.kickMember(now)
		return
	}
	if len(d.members) > 0 {
		d.kickGroup(now)
		return
	}
	for {
		b, err := d.eng.Schedule(now)
		if errors.Is(err, engine.ErrNoCapacity) {
			break
		}
		if err != nil {
			panic(fmt.Sprintf("engine driver: Schedule at %d: %v", now, err))
		}
		d.notePreempted(b)
		d.finishUnrun(b.Finished, now)
		if b.Empty() {
			d.startTransfers(b.TransfersStarted, now)
			break
		}
		start, end := d.timeBatch(b, now)
		// The worker submits a step's offload copies after its forward pass, and each copy's
		// stream waits on the compute stream (startTierTransfer): they start when the step ends.
		d.startTransfersAfter(b.TransfersStarted, now, end)
		d.stepIdx++
		d.admit(b, now, start)
		if d.cfg.ApplyAtStepStart {
			res, err := d.eng.Complete(b.ID, d.outcome(b), end)
			if err != nil {
				panic(fmt.Sprintf("engine driver: Complete(%d) at %d: %v", b.ID, now, err))
			}
			d.apply(b, res, end)
			d.schedule(&driverStepEnd{at: end, applied: true})
			continue
		}
		d.inFlight = append(d.inFlight, driverBatch{b: b, end: end})
		d.schedule(&driverStepEnd{at: end})
	}
	d.rearmWake(now)
}

// timeBatch is when a batch scheduled at now starts and ends. On one device it starts when the
// one before it ends. Through an engine that splits its steps by pipeline stage
// (engine.Pipelined, vLLM at pipeline_parallel_size > 1), each stage is its own device: stage s
// of the batch starts once the batch has left stage s-1 and stage s has finished the batch
// ahead of it, and the batch ends when its last stage does -- the overlap vLLM's batch queue
// keeps max_concurrent_batches batches in flight for (vllm/v1/engine/core.py:698-810@5df63a72).
// The contract is blis-schemas engine.Pipelined; its stages sum to StepTime.
func (d *EngineDriver) timeBatch(b engine.Batch, now int64) (start, end int64) {
	start = max(now, d.gpuFreeAt)
	total := max(d.eng.StepTime(b), 1)
	p, ok := d.eng.(engine.Pipelined)
	if !ok {
		d.gpuFreeAt = start + total
		return start, d.gpuFreeAt
	}
	stages := p.StageTimes(b)
	if len(stages) <= 1 && d.stageFreeAt == nil {
		d.gpuFreeAt = start + total
		return start, d.gpuFreeAt
	}
	if len(stages) <= 1 {
		// A batch the engine did not split, behind pipelined ones, holds every stage: it starts
		// when the first frees and ends no earlier than the pipeline drains, since batches
		// complete in the order they were scheduled (the step end completes inFlight[0]).
		start = max(now, d.stageFreeAt[0])
		end = max(start+total, d.pipelineDrainsAt)
		for i := range d.stageFreeAt {
			d.stageFreeAt[i] = end
		}
		d.gpuFreeAt, d.pipelineDrainsAt = end, end
		d.noteStaged(b, end, len(d.stageFreeAt))
		return start, end
	}
	if len(d.stageFreeAt) != len(stages) {
		d.stageFreeAt = make([]int64, len(stages))
	}
	start = max(now, d.stageFreeAt[0])
	// vLLM's V2 runner holds a non-last pipeline rank's step T until step T-pp's sampled tokens
	// are broadcast (update_pp_decode_requests before the forward, model_runner.py:1171-1179,
	// 1732; PPHandler.get_prev_sampled_outputs' wait_event on the slot pp steps back,
	// pp_utils.py:125-152@5df63a72). The wait is on the step, not on a shared request: it is
	// skipped only when that step sampled for no request still live, its requests all non-final
	// prefill chunks (compute_need_sampled_mask, pp_utils.py:37-53) or freed since (a preempted
	// request's slot is freed by finish_requests, model_runner.py:1152-1166). So the first stage of
	// this batch waits for the batch pp steps back to leave the last stage when it sampled for a
	// request not preempted since. The async scheduler's own stagger (a decode is eligible again
	// only pp steps later, async_scheduler.py:52-57) is the engine's.
	pp := len(stages)
	if n := len(d.recentStages); n >= pp {
		prev := d.recentStages[n-pp]
		for id := range prev.reqs {
			if !d.preemptedSince(id, n-pp) {
				start = max(start, prev.end)
				break
			}
		}
	}
	at := start
	for i, st := range stages {
		at = max(at, d.stageFreeAt[i]) + max(st, 1)
		d.stageFreeAt[i] = at
	}
	// The next batch may enter as soon as the first stage frees; BusyUntil reports when the
	// whole pipeline drains.
	d.gpuFreeAt = d.stageFreeAt[0]
	d.pipelineDrainsAt = at
	d.noteStaged(b, at, pp)
	return start, at
}

// noteStaged records a batch scheduled through the pipeline and when it leaves the last stage,
// keeping the last pp (timeBatch's T-pp dependency): the requests it sampled for (a decode or a
// prompt's final chunk; a non-final chunk samples nothing) and those its schedule preempted.
func (d *EngineDriver) noteStaged(b engine.Batch, end int64, pp int) {
	ids := map[string]bool{}
	for _, r := range b.Reqs {
		if r.Computed+r.Tokens >= r.PromptLen {
			ids[r.ID] = true
		}
	}
	var preempted map[string]bool
	if len(b.Preempted) > 0 {
		preempted = make(map[string]bool, len(b.Preempted))
		for _, id := range b.Preempted {
			preempted[id] = true
		}
	}
	d.recentStages = append(d.recentStages, stagedBatch{reqs: ids, preempted: preempted, end: end})
	if len(d.recentStages) > pp {
		d.recentStages = d.recentStages[len(d.recentStages)-pp:]
	}
}

// preemptedSince reports whether a schedule after recentStages[i] preempted request id, which
// frees its runner state, so recentStages[i]'s token for it no longer gates a step.
func (d *EngineDriver) preemptedSince(id string, i int) bool {
	for _, sb := range d.recentStages[i+1:] {
		if sb.preempted[id] {
			return true
		}
	}
	return false
}

// admit records a batch scheduled at now whose step starts at start: each request's first
// scheduling (its state, scheduling delay and admission prefix lookup), at the current step index.
func (d *EngineDriver) admit(b engine.Batch, now, start int64) {
	for _, s := range b.Reqs {
		if r := d.reqs[s.ID]; r != nil && r.req.State != StateRunning {
			if r.remote && r.emitted == 0 {
				// A remote-prefilled request's first token is a decode step's: its gap runs
				// from the step's start.
				r.lastEnd = start
			}
			r.req.State = StateRunning
			r.scheduledIdx = d.stepIdx
			d.Metrics.RequestSchedulingDelays[s.ID] = now - r.req.ArrivalTime
			if s.PrefixLookup && !r.lookedUp {
				// The first admission's prefix-cache lookup, as the engine reports it
				// (Scheduled.PrefixLookup): the prompt, and what the cache supplied of it, split
				// into the device cache's part and the KV offload tier's (an engine with an
				// offload tier says how much it supplied: OffloadCachedTokens). The report's
				// hit rate is vLLM's prefix-cache stat over first admissions: a preempted
				// request's lookup again is not counted (PrefixCacheStats.record,
				// vllm/v1/metrics/stats.py:138-155@5df63a72).
				r.lookedUp = true
				var offload int
				if o, ok := d.eng.(interface{ OffloadCachedTokens(id string) int }); ok {
					offload = o.OffloadCachedTokens(s.ID)
				}
				d.Metrics.recordPrefixLookup(r.req, PrefixLookup{QueryTokens: r.req.InputLen(),
					HitTokens: int64(s.CachedTokens - offload), CPUHitTokens: int64(offload)})
			}
		}
	}
}

func (d *EngineDriver) notePreempted(b engine.Batch) {
	for _, id := range b.Preempted {
		d.Metrics.PreemptionCount++
		if r := d.reqs[id]; r != nil {
			r.req.State = StateQueued
			r.resumed = r.emitted > 0
		}
	}
}

// rearmWake follows the contract's wake protocol: an earlier instant than the pending wake
// supersedes it, a later one or none replaces it.
func (d *EngineDriver) rearmWake(now int64) {
	at, ok := d.eng.NextWake(now)
	if d.pendingWake != nil {
		if ok && at == d.pendingWake.at {
			return
		}
		d.pendingWake.cancelled = true
		d.pendingWake = nil
	}
	if !ok || at > d.Horizon {
		return
	}
	w := &driverWakeEvent{at: at}
	d.pendingWake = w
	d.schedule(w)
}

// apply records what a completed batch made visible: first tokens, inter-token latencies and
// finishes, in the host's report terms.
func (d *EngineDriver) apply(b engine.Batch, res engine.Result, end int64) {
	d.noteHandoffs(res.Handoffs)
	if d.kvEventSink != nil && len(res.KVEvents) > 0 {
		events := make([]KVEvent, len(res.KVEvents))
		for i, e := range res.KVEvents {
			events[i] = KVEvent{Removed: e.Removed, Hash: e.Hash, Parent: e.Parent, Tokens: tokenIDs(e.Tokens), Tier: e.Tier}
		}
		d.kvEventSink(end, events)
	}
	tokenAt := end + d.cfg.OutputTokenProcessingTime
	firsts := map[string]bool{}
	for _, id := range res.FirstTokens {
		firsts[id] = true
	}
	ids := make([]string, 0, len(res.Emitted))
	for id := range res.Emitted {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		r := d.reqs[id]
		if r == nil {
			continue
		}
		n := res.Emitted[id]
		if firsts[id] && (r.emitted == 0 || r.remote && !r.req.TTFTSet) {
			// A remote-prefilled request's decode steps emit before any first token; preempted, it
			// prefills again here, and that prefill's token is its first, as the compatibility
			// engine stamps it.
			r.resumed = false
			r.req.FirstTokenTime = tokenAt - r.req.ArrivalTime
			r.req.TTFTSet = true
			d.Metrics.RequestTTFTs[id] = float64(r.req.FirstTokenTime)
			if d.OnFirstToken != nil {
				d.OnFirstToken(r.req, r.req.ArrivalTime+r.req.FirstTokenTime)
			}
			r.lastEnd = end
			n-- // the first token's latency is TTFT, not an ITL entry
		}
		if n > 0 {
			// A step emitting n tokens is one ITL entry, as the compatibility engine's
			// executeBatchStep records it: the step's advance since the request's last emitting
			// step plus n tokens of output processing (currStepAdvance + NumNewTokens x OTPT,
			// blis-vllm compat/engine/engine.go:996). The first token's own OTPT is TTFT's.
			gap := end - r.lastEnd + int64(n)*d.cfg.OutputTokenProcessingTime
			if d.cfg.ResumeGapFromLastToken && r.resumed {
				// A resumed prefill's sampled token: its gap runs from the last token emitted
				// before the preemption (compat/engine/engine.go:969-976).
				last := r.req.ArrivalTime + r.req.FirstTokenTime
				for _, v := range r.req.ITL {
					last += v
				}
				gap = max(end+d.cfg.OutputTokenProcessingTime-last, 0)
			}
			r.resumed = false
			r.req.ITL = append(r.req.ITL, gap)
			r.lastEnd = end
		}
		r.emitted += res.Emitted[id]
	}
	for _, f := range res.Finished {
		r := d.reqs[f.ID]
		if r == nil {
			continue
		}
		d.finish(r, f, end)
	}
}

// finishUnrun ends the requests a scheduling decision finished without running them
// (engine.Batch.Finished, blis-schemas 45f1d4e), at now: vLLM's update_from_output of a step that
// ran nothing finishes a failed KV load under kv_load_failure_policy "fail" as FINISHED_ERROR
// (vllm/v1/core/sched/scheduler.py:2306-2325@5df63a72). Such a request never appears in a Result,
// so without this it would stay on the driver's books, never done. An ID the driver does not hold
// is skipped, as apply skips one: a decode handoff whose pull the host failed (FailRemote) left
// the books when the cluster dropped it.
func (d *EngineDriver) finishUnrun(fs []engine.Finished, now int64) {
	for _, f := range fs {
		if r := d.reqs[f.ID]; r != nil {
			d.finish(r, f, now)
		}
	}
}

// finish ends a request the engine finished at end. One the engine failed before it emitted
// anything (engine.FinishError: vLLM answers it with an error, no token, vllm/v1/engine/
// __init__.py:56-57@5df63a72) is unservable, as a request the engine refuses at Add is: it is
// dropped, not completed, so it adds no latency sample. Any other finish is a completion.
func (d *EngineDriver) finish(r *driverReq, f engine.Finished, end int64) {
	if f.Reason != engine.FinishError || r.emitted > 0 {
		d.complete(r, f, end)
		return
	}
	req := r.req
	d.Metrics.DroppedUnservable++
	d.Metrics.RecordUnserved(req, false)
	delete(d.Metrics.Requests, req.ID)
	delete(d.reqs, req.ID)
	if d.unserved == nil {
		d.unserved = map[string]bool{}
	}
	// Its deadline, if one is pending, no longer applies (driverTimeoutEvent).
	d.unserved[req.ID] = true
	d.done(req, end)
}

// complete records a finished request as the compatibility engine's recordRequestCompletion
// does, with the engine's own output count.
func (d *EngineDriver) complete(r *driverReq, f engine.Finished, end int64) {
	req := r.req
	req.State = StateCompleted
	req.DepartureTime = end
	req.GeneratedTokens = f.OutputTokens
	// The computed-token count the host's session manager reads for the output a turn actually
	// produced (ProgressIndex - InputLen), as the compatibility engine leaves it: the prompt plus
	// every sampled token but the last, which is never fed back.
	req.ProgressIndex = req.InputLen() + int64(max(f.OutputTokens-1, 0))
	if f.Reason == engine.FinishLength {
		req.LengthCapped = true
		d.Metrics.RecordLengthCapped(req)
	}
	var itlSum int64
	for _, v := range req.ITL {
		itlSum += v
	}
	var overhead int64
	if len(req.OutputTokens) > 0 {
		overhead = d.cfg.PostDecodeFixedOverhead
	}
	e2e := req.FirstTokenTime + itlSum + overhead
	var itl float64
	if f.OutputTokens > 0 {
		itl = float64(itlSum) / float64(max(f.OutputTokens-1, 1))
	}
	// The completion is ordered by the clock it is processed at (Completion.Clock): the step's end,
	// or its start for a batch applied at its start, as the compatibility engine's host loop
	// processes it.
	clock := end
	if d.cfg.ApplyAtStepStart {
		clock = d.Clock
	}
	d.Metrics.RecordCompletion(Completion{
		Req: req, TTFT: req.FirstTokenTime, OutputTokens: f.OutputTokens, E2E: e2e, ITL: itl,
		Steps: d.stepIdx - r.scheduledIdx, Clock: clock, Overhead: overhead,
	})
	delete(d.reqs, req.ID)
	d.done(req, end)
}

func (d *EngineDriver) done(req *Request, at int64) {
	if d.OnRequestDone != nil {
		for _, next := range d.OnRequestDone(req, at) {
			d.InjectArrival(next)
		}
	}
}

// Evict ends a request the gateway evicts in flight, waiting or running, at now: the engine frees
// it (Finish with FinishAbort), and it leaves the instance's books without a completion, as a
// gateway-evicted request does on the compatibility path (counted by the cluster). It reports
// whether the instance held req.
func (d *EngineDriver) Evict(req *Request, now int64) bool {
	r := d.reqs[req.ID]
	if r == nil || req.State == StateCompleted || req.State == StateTimedOut {
		return false
	}
	if err := d.eng.Finish(req.ID, engine.FinishAbort, now); err != nil {
		panic(fmt.Sprintf("engine driver: Finish(%s, abort): %v", req.ID, err))
	}
	req.State = StateCompleted
	delete(d.reqs, req.ID)
	d.kick(now)
	return true
}

// DrainWaiting removes the requests waiting in the engine's queue (not yet scheduled, or preempted
// back to it), in queue order, for a drained instance whose cluster re-routes them: they leave the
// engine and the driver's books, holding no KV, and keep their own records (ArrivalTime, any
// first token already emitted). The engine must be able to drain (a DrainWaiting method);
// otherwise it is refused, naming it.
func (d *EngineDriver) DrainWaiting(now int64) []*Request {
	q, ok := d.eng.(interface{ DrainWaiting() []string })
	if !ok {
		panic(fmt.Sprintf("engine driver: the engine %T cannot drain its waiting queue", d.eng))
	}
	ids := q.DrainWaiting()
	out := make([]*Request, 0, len(ids))
	for _, id := range ids {
		r := d.reqs[id]
		if r == nil {
			panic(fmt.Sprintf("engine driver: the engine drained %s, which the driver does not hold", id))
		}
		out = append(out, r.req)
		delete(d.reqs, id)
	}
	d.rearmWake(now)
	return out
}

// SetKVEventSink makes the engine publish its prefix-cache changes to sink, each batch's at the
// instant its step ends, as the router's precise-prefix producer ingests them. The engine must be
// able to publish them (an EnableKVEvents method); otherwise it is refused, naming it.
func (d *EngineDriver) SetKVEventSink(sink func(at int64, events []KVEvent)) error {
	p, ok := d.eng.(interface{ EnableKVEvents() })
	if !ok {
		return fmt.Errorf("engine driver: the engine %T does not publish KV events", d.eng)
	}
	p.EnableKVEvents()
	d.kvEventSink = sink
	return nil
}

// KVEventsRefusal is why a router cannot index the engine's KV events, for an engine that says
// (a KVEventsRefusal method, blis-vllm's v0.32.0 engine for a full-attention group at a block size
// other than its hash block size); nil otherwise.
func (d *EngineDriver) KVEventsRefusal() error {
	if q, ok := d.eng.(interface{ KVEventsRefusal() error }); ok {
		return q.KVEventsRefusal()
	}
	return nil
}

func tokenIDs(in []int32) []TokenID {
	out := make([]TokenID, len(in))
	for i, v := range in {
		out[i] = TokenID(v)
	}
	return out
}

// ServerSeriesSpan is the bin width of the engine's server time series and the last instant it
// recorded anything (0, 0 for an engine that records none: one without a ServerSeriesSpan query).
func (d *EngineDriver) ServerSeriesSpan() (interval, end int64) {
	if q, ok := d.eng.(interface{ ServerSeriesSpan() (int64, int64) }); ok {
		return q.ServerSeriesSpan()
	}
	return 0, 0
}

// ServerSeriesBlock is the engine's server time series as id, binned to end, as a block holding
// this one engine; nil for an engine that records none.
func (d *EngineDriver) ServerSeriesBlock(id string, end int64) (*ServerSeriesOutput, error) {
	q, ok := d.eng.(interface {
		ServerSeriesBlock(id string, end int64) ([]byte, error)
	})
	if !ok {
		return nil, nil
	}
	b, err := q.ServerSeriesBlock(id, end)
	if err != nil || b == nil {
		return nil, err
	}
	var out ServerSeriesOutput
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, fmt.Errorf("engine driver: the engine's server series: %w", err)
	}
	return &out, nil
}

// CachedBlockHashes, CPUBlockHashes and TierBlockHashes are the hashes of the blocks the engine's
// prefix cache, its CPU KV tier and its secondary offload tiers publishing as medium name hold,
// sorted, for an engine that reports them; nil otherwise.
func (d *EngineDriver) CachedBlockHashes() []string {
	if q, ok := d.eng.(interface{ CachedBlockHashes() []string }); ok {
		return q.CachedBlockHashes()
	}
	return nil
}

func (d *EngineDriver) CPUBlockHashes() []string {
	if q, ok := d.eng.(interface{ CPUBlockHashes() []string }); ok {
		return q.CPUBlockHashes()
	}
	return nil
}

func (d *EngineDriver) TierBlockHashes(name string) []string {
	if q, ok := d.eng.(interface{ TierBlockHashes(string) []string }); ok {
		return q.TierBlockHashes(name)
	}
	return nil
}

// PostDecodeFixedOverhead is the host's fixed per-request work at completion.
func (d *EngineDriver) PostDecodeFixedOverhead() int64 { return d.cfg.PostDecodeFixedOverhead }

// Snapshot is the engine's state, for routing.
func (d *EngineDriver) Snapshot() engine.State { return d.eng.Snapshot() }

// QueueDepth and BatchSize are the engine's waiting and running counts.
func (d *EngineDriver) QueueDepth() int { return d.eng.Snapshot().Waiting }
func (d *EngineDriver) BatchSize() int  { return d.eng.Snapshot().Running }

// HasWork reports whether the engine holds a request.
func (d *EngineDriver) HasWork() bool { return len(d.reqs) > 0 }

// BusyUntil is when the GPU finishes the batches in flight.
func (d *EngineDriver) BusyUntil() int64 { return max(d.gpuFreeAt, d.pipelineDrainsAt) }

// EngineMetricsAt is what a scrape of the model server at t sees, from the engine's state.
func (d *EngineDriver) EngineMetricsAt(t int64) EngineMetrics {
	if d.cfg.ScrapeEngineMetrics {
		// The engine's own published view (the compatibility engine's, at its last step end).
		m := d.eng.Metrics(t)
		v := func(name string) float64 {
			x, _ := m.Value(name)
			return x
		}
		em := EngineMetrics{PublishedAt: m.At, Waiting: int(v("num_requests_waiting")),
			Running: int(v("num_requests_running")), KVUsage: v("kv_cache_usage_perc"),
			Preemptions: int64(v("num_preemptions"))}
		if f, ok := m.Family("lora_requests_info"); ok && len(f.Samples) == 1 {
			// The adapters with a running or waiting request, as the engine's
			// lora_requests_info labels list them.
			em.RunningAdapters = adapterList(f.Samples[0].Labels["running_lora_adapters"])
			em.WaitingAdapters = adapterList(f.Samples[0].Labels["waiting_lora_adapters"])
		}
		return em
	}
	s := d.eng.Snapshot()
	usage := 0.0
	if s.KVBlocksTotal > 0 {
		usage = float64(s.KVBlocksUsed) / float64(s.KVBlocksTotal)
	}
	return EngineMetrics{PublishedAt: t, Waiting: s.Waiting, Running: s.Running, KVUsage: usage,
		Preemptions: d.Metrics.PreemptionCount}
}

// adapterList splits a comma-joined adapter list; nil for none.
func adapterList(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, ",")
}

func int32s(ts []TokenID) []int32 {
	out := make([]int32, len(ts))
	for i, t := range ts {
		out[i] = int32(t)
	}
	return out
}
