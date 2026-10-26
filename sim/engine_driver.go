// sim/engine_driver.go: the host half of an instance whose engine is reached only through
// blis-schemas' engine.Engine contract (#2021 B3, GOAL.md). The host keeps the clock, the event
// queue, the timeouts and the report; the engine makes every scheduling, KV-cache and output
// decision and prices its own steps.
package sim

import (
	"container/heap"
	"errors"
	"fmt"
	"sort"

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
}

type driverReq struct {
	req          *Request
	scheduledIdx int
	lastEnd      int64 // the end of the last step that emitted a token for the request
	emitted      int
}

type driverBatch struct {
	b   engine.Batch
	end int64
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

// DriverEvent is an event of an engine-driven instance.
type DriverEvent interface {
	Timestamp() int64
	Priority() int
	run(d *EngineDriver)
}

// The driver's queue reuses EventQueue's ordering, (timestamp, priority, seqID), by wrapping
// each driver event in an Event shim, which the cluster loop handles as any instance event; the
// shim's Simulator argument is unused (a driver event runs on its driver).
type driverShim struct {
	ev DriverEvent
	d  *EngineDriver
}

func (s *driverShim) Timestamp() int64   { return s.ev.Timestamp() }
func (s *driverShim) Priority() int      { return s.ev.Priority() }
func (s *driverShim) Execute(*Simulator) { s.ev.run(s.d) }

func (d *EngineDriver) schedule(ev DriverEvent) {
	d.seqCounter++
	heap.Push(&d.eventQueue, eventEntry{event: &driverShim{ev: ev, d: d}, seqID: d.seqCounter})
}

// HasPendingEvents reports whether an event is queued.
func (d *EngineDriver) HasPendingEvents() bool { return len(d.eventQueue) > 0 }

// PeekNextEventTime is the earliest queued event's time. Callers check HasPendingEvents first.
func (d *EngineDriver) PeekNextEventTime() int64 { return d.eventQueue[0].event.Timestamp() }

// ProcessNextEvent pops the earliest event, advances the clock and runs it. A superseded wake is
// returned without advancing the clock, as Simulator drops a superseded idle wake.
func (d *EngineDriver) ProcessNextEvent() Event {
	entry := heap.Pop(&d.eventQueue).(eventEntry)
	shim := entry.event.(*driverShim)
	if w, ok := shim.ev.(*driverWakeEvent); ok && w.cancelled {
		return entry.event
	}
	if te, ok := shim.ev.(*driverTimeoutEvent); ok && te.req.State == StateCompleted {
		return entry.event
	}
	d.Clock = shim.Timestamp()
	shim.ev.run(d)
	return entry.event
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

// Finalize records end-of-run state.
func (d *EngineDriver) Finalize() {
	waiting, running := 0, 0
	for _, r := range d.reqs {
		if r.req.State == StateRunning {
			running++
		} else {
			waiting++
		}
	}
	d.Metrics.StillQueued = waiting
	d.Metrics.StillRunning = running
	d.Metrics.SimEndedTime = min(d.Clock, d.Horizon)
	logrus.Infof("[tick %07d] Simulation ended", d.Clock)
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
	er := engine.Request{
		ID:              req.ID,
		Prompt:          int32s(req.InputTokens),
		Output:          int32s(req.OutputTokens),
		MaxOutputTokens: req.MaxOutputLen,
		ArrivalTime:     req.ArrivalTime,
		Adapter:         req.Adapter,
	}
	if err := d.eng.Add(er, e.at); err != nil {
		logrus.Warnf("dropping request %s: the engine refused it: %v", req.ID, err)
		d.Metrics.DroppedUnservable++
		d.Metrics.RecordUnserved(req, false)
		delete(d.Metrics.Requests, req.ID)
		d.done(req, e.at)
		return
	}
	d.Metrics.TotalInputTokens += len(req.InputTokens)
	req.State = StateQueued
	d.reqs[req.ID] = &driverReq{req: req}
	if req.Deadline > 0 && req.Deadline <= d.Horizon {
		d.schedule(&driverTimeoutEvent{at: req.Deadline, req: req})
	}
	d.stepAfterArrivals(e.at)
}

// stepAfterArrivals answers an Add: it asks the engine when to wake, and, when no batch is in
// flight, queues a step at now that runs after every arrival of this instant (the contract's
// same-instant order, arrivals before steps; the compatibility engine's scheduleStepBy), so
// requests that reach the engine together share its next batch. With a batch in flight, vLLM's
// loop waits on that batch's output, and its Complete steps the engine.
func (d *EngineDriver) stepAfterArrivals(now int64) {
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

// driverStepEnd completes the oldest in-flight batch at its step's end.
type driverStepEnd struct {
	at int64
}

func (e *driverStepEnd) Timestamp() int64 { return e.at }
func (e *driverStepEnd) Priority() int    { return PriorityStep }
func (e *driverStepEnd) run(d *EngineDriver) {
	if len(d.inFlight) == 0 {
		panic(fmt.Sprintf("engine driver: a step ended at %d with no batch in flight", e.at))
	}
	fb := d.inFlight[0]
	d.inFlight = d.inFlight[1:]
	res, err := d.eng.Complete(fb.b.ID, engine.Outcome{}, e.at)
	if err != nil {
		panic(fmt.Sprintf("engine driver: Complete(%d) at %d: %v", fb.b.ID, e.at, err))
	}
	d.apply(fb.b, res, e.at)
	d.kick(e.at)
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
	if req.State == StateCompleted || req.State == StateTimedOut {
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

// kick schedules as many batches as the engine admits at now, each starting when the GPU is
// free, then asks the engine when to wake. Batches are scheduled only when the GPU has a free
// in-flight slot: Schedule returns ErrNoCapacity at the engine's limit.
func (d *EngineDriver) kick(now int64) {
	for {
		b, err := d.eng.Schedule(now)
		if errors.Is(err, engine.ErrNoCapacity) {
			break
		}
		if err != nil {
			panic(fmt.Sprintf("engine driver: Schedule at %d: %v", now, err))
		}
		d.notePreempted(b)
		if b.Empty() {
			break
		}
		start := max(now, d.gpuFreeAt)
		end := start + max(d.eng.StepTime(b), 1)
		d.gpuFreeAt = end
		d.stepIdx++
		for _, s := range b.Reqs {
			if r := d.reqs[s.ID]; r != nil && r.req.State != StateRunning {
				r.req.State = StateRunning
				r.scheduledIdx = d.stepIdx
				d.Metrics.RequestSchedulingDelays[s.ID] = now - r.req.ArrivalTime
			}
		}
		d.inFlight = append(d.inFlight, driverBatch{b: b, end: end})
		d.schedule(&driverStepEnd{at: end})
	}
	d.rearmWake(now)
}

func (d *EngineDriver) notePreempted(b engine.Batch) {
	for _, id := range b.Preempted {
		d.Metrics.PreemptionCount++
		if r := d.reqs[id]; r != nil {
			r.req.State = StateQueued
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
		if firsts[id] && r.emitted == 0 {
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
			r.req.ITL = append(r.req.ITL, end-r.lastEnd+int64(n)*d.cfg.OutputTokenProcessingTime)
			r.lastEnd = end
		}
		r.emitted += res.Emitted[id]
	}
	for _, f := range res.Finished {
		r := d.reqs[f.ID]
		if r == nil {
			continue
		}
		d.complete(r, f, end)
	}
}

// complete records a finished request as the compatibility engine's recordRequestCompletion
// does, with the engine's own output count.
func (d *EngineDriver) complete(r *driverReq, f engine.Finished, end int64) {
	req := r.req
	req.State = StateCompleted
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
	d.Metrics.RecordCompletion(Completion{
		Req: req, TTFT: req.FirstTokenTime, OutputTokens: f.OutputTokens, E2E: e2e, ITL: itl,
		Steps: d.stepIdx - r.scheduledIdx, Clock: end, Overhead: overhead,
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
func (d *EngineDriver) BusyUntil() int64 { return d.gpuFreeAt }

// EngineMetricsAt is what a scrape of the model server at t sees, from the engine's state.
func (d *EngineDriver) EngineMetricsAt(t int64) EngineMetrics {
	s := d.eng.Snapshot()
	usage := 0.0
	if s.KVBlocksTotal > 0 {
		usage = float64(s.KVBlocksUsed) / float64(s.KVBlocksTotal)
	}
	return EngineMetrics{PublishedAt: t, Waiting: s.Waiting, Running: s.Running, KVUsage: usage,
		Preemptions: d.Metrics.PreemptionCount}
}

func int32s(ts []TokenID) []int32 {
	out := make([]int32, len(ts))
	for i, t := range ts {
		out[i] = int32(t)
	}
	return out
}
