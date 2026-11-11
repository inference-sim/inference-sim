// Package enginetest holds checks of the engine.Engine contract that any implementation
// runs against itself, in the way package kerneltest checks a kernel.
//
// Its centre is Drive, a minimal discrete-event driver that follows the contract's wake
// protocol and same-instant order exactly. Driving an engine through it checks the laws
// that hold for every engine: no request is lost or finished twice, a batch runs only
// requests the engine holds, each once, and preempts only requests it holds and does not
// run, each once, batch IDs never repeat, batches complete in order,
// no more batches are in flight than the engine's limit, a wake is always after the
// instant it was asked at, a step always takes time, a batch's output names only its own
// requests and at most one token plus its accepted drafts each, a finish reports the
// output its batches emitted, an engine with work left always has a way to be stepped
// again, and the same inputs give the same run.
package enginetest

import (
	"container/heap"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// Arrival is one request and the instant it arrives.
type Arrival struct {
	At  engine.Time
	Req engine.Request
}

// Options are the inputs a simulator owns that Drive needs. Every field is optional.
type Options struct {
	// Accepted is how many drafts a request accepts in a batch; nil accepts none.
	Accepted func(b engine.Batch, s engine.Scheduled) int
	// TransferTime prices a transfer; nil takes one microsecond.
	TransferTime func(engine.Transfer) engine.Duration
	// TransferFails reports a transfer that fails; nil fails none.
	TransferFails func(engine.Transfer) bool
	// Boundary takes each transfer an engine starts, with the plan the engine states, across the
	// simulator's side of the boundary: in practice env.DemandFor against an Environment (see
	// envtest.Boundary). A non-nil error fails the run. It is separate from the checks below
	// only because this package imports no Environment; those run on every transfer whether or
	// not Boundary is set.
	Boundary func(p plan.PoolPlan, t engine.Transfer) error
	// Deadlines maps a request to the instant it times out; absent is never.
	Deadlines map[string]engine.Time
	// MaxSteps bounds the run; zero means 1,000,000.
	MaxSteps int
}

// Step is one executed batch.
type Step struct {
	Start  engine.Time
	End    engine.Time
	Batch  engine.Batch
	Result engine.Result
}

// Trace is what one run did.
type Trace struct {
	Steps []Step
	// Finished maps every request to how it finished.
	Finished map[string]engine.Finished
	// End is the instant of the last event.
	End engine.Time
}

// Event kinds in the contract's same-instant order: lower runs first.
const (
	kindArrival = iota
	kindTransfer
	kindComplete
	kindWake
	kindTimeout
)

type event struct {
	at   engine.Time
	kind int
	seq  int64
	// payload
	arrival  *Arrival
	batch    engine.BatchID
	transfer engine.Transfer
	request  string
	// a superseded wake is dropped without running
	dead bool
}

type queue []*event

func (q queue) Len() int { return len(q) }
func (q queue) Less(i, j int) bool {
	if q[i].at != q[j].at {
		return q[i].at < q[j].at
	}
	if q[i].kind != q[j].kind {
		return q[i].kind < q[j].kind
	}
	return q[i].seq < q[j].seq
}
func (q queue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *queue) Push(x any)   { *q = append(*q, x.(*event)) }
func (q *queue) Pop() any {
	old := *q
	e := old[len(old)-1]
	*q = old[:len(old)-1]
	return e
}

type driver struct {
	e        engine.Engine
	opt      Options
	q        queue
	seq      int64
	now      engine.Time
	wake     *event
	busyTill engine.Time
	inFlight []engine.BatchID
	// stageFree is when each pipeline stage finishes the batch it holds, for an engine that
	// is engine.Pipelined; nil otherwise, when busyTill is the one device's.
	stageFree []engine.Time
	started   map[engine.BatchID]engine.Time
	batches   map[engine.BatchID]engine.Batch
	lastID    engine.BatchID
	haveID    bool
	pending   map[string]bool
	reqs      map[string]engine.Request
	pool      *plan.PoolPlan
	held      map[string]bool
	emitted   map[string]int
	trace     Trace
	steps     int
}

func (d *driver) push(ev *event) {
	d.seq++
	ev.seq = d.seq
	heap.Push(&d.q, ev)
}

// rewake applies the wake protocol after a state-changing call. A wake at or before the
// instant it was asked at breaks the contract (NextWake is the earliest instant after now)
// and is reported, not moved: a driver that repaired it would hide the bug.
func (d *driver) rewake() error {
	at, ok := d.e.NextWake(d.now)
	if ok && at <= d.now {
		return fmt.Errorf("at %d: NextWake(%d) = %d, want an instant after now", d.now, d.now, at)
	}
	if d.wake != nil && (!ok || at != d.wake.at) {
		d.wake.dead = true
		d.wake = nil
	}
	if ok && d.wake == nil {
		d.wake = &event{at: at, kind: kindWake}
		d.push(d.wake)
	}
	return nil
}

// checkTransfer takes one started transfer across the boundary (design §6.2): it must agree with
// its own Move, the engine must state a plan (engine.Planned) that can make the move, a load
// from a peer must read from the peer the request named, and the simulator's side (Boundary)
// must turn it into a demand. A transfer the cost model could not price fails here, where the
// engine can be fixed, and not as a zero price in a run.
func (d *driver) checkTransfer(tr engine.Transfer) error {
	if err := tr.Validate(); err != nil {
		return fmt.Errorf("at %d: %w", d.now, err)
	}
	pl, ok := d.e.(engine.Planned)
	if !ok {
		return fmt.Errorf("at %d: transfer %d started, and the engine does not implement engine.Planned: the simulator has no plan to price it against", d.now, tr.ID)
	}
	p := pl.Plan()
	if err := p.Validate(); err != nil {
		return fmt.Errorf("at %d: Plan() is not a valid PoolPlan: %w", d.now, err)
	}
	if d.pool == nil {
		d.pool = &p
	} else if !reflect.DeepEqual(*d.pool, p) {
		return fmt.Errorf("at %d: Plan() returned a different plan from an earlier call", d.now)
	}
	if err := p.CheckTransfer(tr.Move); err != nil {
		return fmt.Errorf("at %d: transfer %d: %w", d.now, tr.ID, err)
	}
	if r, ok := d.reqs[tr.Request]; ok && tr.Move.Kind == plan.TransferPeer && tr.Kind == engine.TransferLoad {
		if r.Peer == nil || tr.Move.From != *r.Peer {
			return fmt.Errorf("at %d: transfer %d pulls request %s from %+v, and the request names peer %v", d.now, tr.ID, tr.Request, tr.Move.From, r.Peer)
		}
	}
	if d.opt.Boundary != nil {
		if err := d.opt.Boundary(p, tr); err != nil {
			return fmt.Errorf("at %d: transfer %d across the boundary: %w", d.now, tr.ID, err)
		}
	}
	return nil
}

// step schedules as many batches as the engine allows at d.now. A batch runs once the
// one before it ends, as on one device.
func (d *driver) step() error {
	for {
		if len(d.inFlight) >= d.e.InFlightLimit() {
			if _, err := d.e.Schedule(d.now); !errors.Is(err, engine.ErrNoCapacity) {
				return fmt.Errorf("at %d: Schedule with %d of %d batches in flight returned %v, want ErrNoCapacity",
					d.now, len(d.inFlight), d.e.InFlightLimit(), err)
			}
			return nil
		}
		b, err := d.e.Schedule(d.now)
		if err != nil {
			return fmt.Errorf("at %d: Schedule: %w", d.now, err)
		}
		if err := d.rewake(); err != nil {
			return err
		}
		// A batch that runs nothing may still start transfers (an asynchronous load
		// admitted with no compute this step), and those must be priced like any other.
		for _, tr := range b.TransfersStarted {
			if err := d.checkTransfer(tr); err != nil {
				return err
			}
			dt := engine.Duration(1)
			if d.opt.TransferTime != nil {
				dt = d.opt.TransferTime(tr)
			}
			if dt < 1 {
				return fmt.Errorf("at %d: transfer %d priced at %d, want at least 1", d.now, tr.ID, dt)
			}
			d.push(&event{at: d.now + dt, kind: kindTransfer, transfer: tr})
		}
		if err := d.checkMembers(b); err != nil {
			return err
		}
		for _, f := range b.Finished {
			if err := d.finished(f); err != nil {
				return err
			}
		}
		if b.Empty() {
			return nil
		}
		if d.haveID && b.ID <= d.lastID {
			return fmt.Errorf("at %d: batch ID %d does not follow %d", d.now, b.ID, d.lastID)
		}
		d.lastID, d.haveID = b.ID, true
		dur := d.e.StepTime(b)
		if dur < 1 {
			return fmt.Errorf("at %d: StepTime(%d) = %d, want at least 1", d.now, b.ID, dur)
		}
		start := max(d.now, d.busyTill)
		end := start + dur
		if p, ok := d.e.(engine.Pipelined); ok {
			if start, end, err = d.pipeline(p, b, dur); err != nil {
				return err
			}
		}
		d.busyTill = end
		d.inFlight = append(d.inFlight, b.ID)
		d.started[b.ID] = start
		d.batches[b.ID] = b
		d.push(&event{at: end, kind: kindComplete, batch: b.ID})
		if s := d.e.Snapshot(); s.InFlight != len(d.inFlight) {
			return fmt.Errorf("at %d: Snapshot().InFlight = %d with %d batches in flight", d.now, s.InFlight, len(d.inFlight))
		}
	}
}

// pipeline times a batch through an engine.Pipelined engine's stages: stage s starts once
// the batch has left stage s-1 and stage s has finished the batch ahead of it. It returns the
// batch's first-stage start and last-stage end, and checks the Pipelined law: one entry per
// stage, each at least one microsecond, summing to StepTime, and the same stage count every
// batch.
func (d *driver) pipeline(p engine.Pipelined, b engine.Batch, dur engine.Duration) (engine.Time, engine.Time, error) {
	stages := p.StageTimes(b)
	var sum engine.Duration
	for i, st := range stages {
		if st < 1 {
			return 0, 0, fmt.Errorf("at %d: StageTimes(%d)[%d] = %d, want at least 1", d.now, b.ID, i, st)
		}
		sum += st
	}
	switch {
	case len(stages) == 0:
		return 0, 0, fmt.Errorf("at %d: StageTimes(%d) is empty, want at least one stage", d.now, b.ID)
	case sum != dur:
		return 0, 0, fmt.Errorf("at %d: StageTimes(%d) sums to %d, StepTime is %d", d.now, b.ID, sum, dur)
	case d.stageFree == nil:
		d.stageFree = make([]engine.Time, len(stages))
	case len(d.stageFree) != len(stages):
		return 0, 0, fmt.Errorf("at %d: StageTimes(%d) has %d stages, an earlier batch had %d",
			d.now, b.ID, len(stages), len(d.stageFree))
	}
	start := max(d.now, d.stageFree[0])
	at := start
	for i, st := range stages {
		at = max(at, d.stageFree[i]) + st
		d.stageFree[i] = at
	}
	return start, at, nil
}

// checkMembers checks who a decision names. A batch runs only requests the engine holds
// (added and not finished), each once. It preempts only requests it holds and does not
// run, each once: the engine unschedules a request it preempts within the step. That holds for a
// decision that runs nothing too, as a step that preempts every running request does.
func (d *driver) checkMembers(b engine.Batch) error {
	inBatch := map[string]bool{}
	for _, s := range b.Reqs {
		switch {
		case !d.held[s.ID]:
			return fmt.Errorf("at %d: batch %d runs %s, which was never added or has finished", d.now, b.ID, s.ID)
		case inBatch[s.ID]:
			return fmt.Errorf("at %d: batch %d runs %s twice in batch", d.now, b.ID, s.ID)
		}
		inBatch[s.ID] = true
	}
	for _, f := range b.Finished {
		if inBatch[f.ID] {
			return fmt.Errorf("at %d: batch %d both runs and finishes %s", d.now, b.ID, f.ID)
		}
	}
	evicted := map[string]bool{}
	for _, id := range b.Preempted {
		switch {
		case !d.held[id]:
			return fmt.Errorf("at %d: batch %d preempts %s, which was never added or has finished", d.now, b.ID, id)
		case inBatch[id]:
			return fmt.Errorf("at %d: batch %d both runs and preempts %s", d.now, b.ID, id)
		case evicted[id]:
			return fmt.Errorf("at %d: batch %d preempts %s twice", d.now, b.ID, id)
		}
		evicted[id] = true
	}
	return nil
}

func (d *driver) complete(id engine.BatchID) error {
	b := d.batches[id]
	out := engine.Outcome{Accepted: map[string]int{}}
	if d.opt.Accepted != nil {
		for _, s := range b.Reqs {
			if s.Drafts > 0 {
				out.Accepted[s.ID] = min(d.opt.Accepted(b, s), s.Drafts)
			}
		}
	}
	res, err := d.e.Complete(id, out, d.now)
	if err != nil {
		return fmt.Errorf("at %d: Complete(%d): %w", d.now, id, err)
	}
	// A request emits at most one sampled token plus the drafts it had accepted, and only a
	// request this batch ran can emit at all.
	ran := map[string]bool{}
	for _, s := range b.Reqs {
		ran[s.ID] = true
	}
	for rid, n := range res.Emitted {
		if !ran[rid] {
			return fmt.Errorf("at %d: Complete(%d) emitted %d tokens for %s, which the batch did not run", d.now, id, n, rid)
		}
		if limit := 1 + out.Accepted[rid]; n < 0 || n > limit {
			return fmt.Errorf("at %d: Complete(%d) emitted %d tokens for %s, want 0 to %d (one sampled plus %d accepted drafts)",
				d.now, id, n, rid, limit, out.Accepted[rid])
		}
		d.emitted[rid] += n
	}
	d.inFlight = d.inFlight[1:]
	d.trace.Steps = append(d.trace.Steps, Step{Start: d.started[id], End: d.now, Batch: b, Result: res})
	delete(d.batches, id)
	for _, f := range res.Finished {
		if f.OutputTokens != d.emitted[f.ID] {
			return fmt.Errorf("at %d: Complete(%d) finished %s with %d output tokens, but its batches emitted %d",
				d.now, id, f.ID, f.OutputTokens, d.emitted[f.ID])
		}
		if err := d.finished(f); err != nil {
			return err
		}
	}
	return nil
}

func (d *driver) finished(f engine.Finished) error {
	if _, twice := d.trace.Finished[f.ID]; twice {
		return fmt.Errorf("at %d: request %s finished twice", d.now, f.ID)
	}
	if !d.pending[f.ID] {
		return fmt.Errorf("at %d: request %s finished but was never added", d.now, f.ID)
	}
	delete(d.pending, f.ID)
	delete(d.held, f.ID)
	d.trace.Finished[f.ID] = f
	return nil
}

// Drive runs arrivals through a fresh engine until every request has finished, following
// the wake protocol and the same-instant order the engine package states. It returns an
// error at the first broken law.
func Drive(newEngine func() (engine.Engine, error), arrivals []Arrival, opt Options) (Trace, error) {
	e, err := newEngine()
	if err != nil {
		return Trace{}, err
	}
	if e.InFlightLimit() < 1 {
		return Trace{}, fmt.Errorf("InFlightLimit() = %d, want at least 1", e.InFlightLimit())
	}
	if err := CheckFields(e); err != nil {
		return Trace{}, err
	}
	d := &driver{
		e: e, opt: opt,
		started: map[engine.BatchID]engine.Time{}, batches: map[engine.BatchID]engine.Batch{},
		pending: map[string]bool{}, reqs: map[string]engine.Request{}, held: map[string]bool{}, emitted: map[string]int{},
		trace: Trace{Finished: map[string]engine.Finished{}},
	}
	maxSteps := opt.MaxSteps
	if maxSteps == 0 {
		maxSteps = 1_000_000
	}
	for i := range arrivals {
		a := &arrivals[i]
		if d.pending[a.Req.ID] {
			return Trace{}, fmt.Errorf("request %s arrives twice", a.Req.ID)
		}
		d.pending[a.Req.ID] = true
		d.push(&event{at: a.At, kind: kindArrival, arrival: a})
		if at, ok := opt.Deadlines[a.Req.ID]; ok {
			d.push(&event{at: at, kind: kindTimeout, request: a.Req.ID})
		}
	}
	for d.q.Len() > 0 {
		ev := heap.Pop(&d.q).(*event)
		if ev.dead {
			continue
		}
		if ev.at < d.now {
			return Trace{}, fmt.Errorf("event at %d after %d", ev.at, d.now)
		}
		d.now = ev.at
		d.trace.End = d.now
		switch ev.kind {
		case kindArrival:
			if err := e.Add(ev.arrival.Req, d.now); err != nil {
				return Trace{}, fmt.Errorf("at %d: Add(%s): %w", d.now, ev.arrival.Req.ID, err)
			}
			d.held[ev.arrival.Req.ID] = true
			d.reqs[ev.arrival.Req.ID] = ev.arrival.Req
		case kindTransfer:
			ok := opt.TransferFails == nil || !opt.TransferFails(ev.transfer)
			if err := e.TransferDone(ev.transfer.ID, ok, d.now); err != nil {
				return Trace{}, fmt.Errorf("at %d: TransferDone(%d): %w", d.now, ev.transfer.ID, err)
			}
		case kindComplete:
			if err := d.complete(ev.batch); err != nil {
				return Trace{}, err
			}
		case kindWake:
			d.wake = nil
		case kindTimeout:
			if !d.pending[ev.request] {
				continue
			}
			if err := e.Finish(ev.request, engine.FinishTimeout, d.now); err != nil {
				return Trace{}, fmt.Errorf("at %d: Finish(%s): %w", d.now, ev.request, err)
			}
			if err := d.finished(engine.Finished{ID: ev.request, Reason: engine.FinishTimeout, OutputTokens: d.emitted[ev.request]}); err != nil {
				return Trace{}, err
			}
		}
		if err := d.rewake(); err != nil {
			return Trace{}, err
		}
		// Step once the instant's earlier kinds are processed, as a simulator does: after the
		// last arrival, transfer or completion at this instant, or at a wake.
		if next := d.q.peek(); next == nil || next.at > d.now || next.kind >= kindWake {
			if err := d.step(); err != nil {
				return Trace{}, err
			}
			d.steps++
			if d.steps > maxSteps {
				return Trace{}, fmt.Errorf("no progress after %d steps", maxSteps)
			}
		}
	}
	if len(d.pending) > 0 {
		ids := make([]string, 0, len(d.pending))
		for id := range d.pending {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		return Trace{}, fmt.Errorf("at %d: requests %v never finish: nothing is in flight, no transfer is outstanding and the engine reports no wake", d.now, ids)
	}
	return d.trace, nil
}

// peek returns the next live event without removing it, dropping superseded wakes.
func (q *queue) peek() *event {
	for q.Len() > 0 && (*q)[0].dead {
		heap.Pop(q)
	}
	if q.Len() == 0 {
		return nil
	}
	return (*q)[0]
}

// CheckFields checks the field list an engine reports (design VLM-3): it validates as a list,
// it names an engine kind and an upstream release, and a second call returns the same list.
// The substrate holds no engine's fields, so it cannot check that the list is complete; that
// is the engine repository's own completeness check.
func CheckFields(e engine.Engine) error {
	l := e.Fields()
	if err := l.Validate(); err != nil {
		return fmt.Errorf("Fields() is not a valid field list: %w", err)
	}
	if again := e.Fields(); !reflect.DeepEqual(l, again) {
		return fmt.Errorf("Fields() returned a different list on a second call")
	}
	return nil
}

// TestEngine drives arrivals through newEngine twice and fails on any broken law, or if the
// two runs differ. It fails on an empty arrival list, which would check nothing.
func TestEngine(t testing.TB, newEngine func() (engine.Engine, error), arrivals []Arrival, opt Options) Trace {
	t.Helper()
	if len(arrivals) == 0 {
		t.Fatal("enginetest.TestEngine: no arrivals, so nothing was checked")
	}
	first, err := Drive(newEngine, arrivals, opt)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Drive(newEngine, arrivals, opt)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Error("two runs of the same arrivals on fresh engines differ")
	}
	for _, a := range arrivals {
		f, ok := first.Finished[a.Req.ID]
		if !ok {
			t.Errorf("request %s did not finish", a.Req.ID)
			continue
		}
		if f.Reason == engine.FinishStop && f.OutputTokens != len(a.Req.Output) {
			t.Errorf("request %s stopped after %d output tokens, assigned %d", a.Req.ID, f.OutputTokens, len(a.Req.Output))
		}
		if f.OutputTokens > len(a.Req.Output) {
			t.Errorf("request %s emitted %d output tokens, more than the %d assigned", a.Req.ID, f.OutputTokens, len(a.Req.Output))
		}
	}
	return first
}

// CheckProtocolErrors checks the errors a simulator relies on to detect its own bugs:
// completing an unknown batch, completing out of order, and finishing an unknown request
// (which is not an error). It needs an engine that will schedule two batches at once for
// the out-of-order case, and skips that case for an engine whose limit is one.
func CheckProtocolErrors(newEngine func() (engine.Engine, error), a, b engine.Request) error {
	e, err := newEngine()
	if err != nil {
		return err
	}
	if _, err := e.Complete(12345, engine.Outcome{}, 0); !errors.Is(err, engine.ErrUnknownBatch) {
		return fmt.Errorf("Complete of an unknown batch returned %v, want ErrUnknownBatch", err)
	}
	if err := e.Finish("no-such-request", engine.FinishAbort, 0); err != nil {
		return fmt.Errorf("Finish of an unknown request returned %v, want nil", err)
	}
	if err := e.TransferDone(12345, true, 0); !errors.Is(err, engine.ErrUnknownTransfer) {
		return fmt.Errorf("TransferDone of an unknown transfer returned %v, want ErrUnknownTransfer", err)
	}
	if e.InFlightLimit() < 2 {
		return nil
	}
	for _, r := range []engine.Request{a, b} {
		if err := e.Add(r, 0); err != nil {
			return err
		}
	}
	first, err := e.Schedule(0)
	if err != nil || first.Empty() {
		return fmt.Errorf("first Schedule: batch %v, err %v", first, err)
	}
	second, err := e.Schedule(0)
	if err != nil || second.Empty() {
		return fmt.Errorf("second Schedule: batch %v, err %v", second, err)
	}
	if _, err := e.Complete(second.ID, engine.Outcome{}, 1); !errors.Is(err, engine.ErrOutOfOrder) {
		return fmt.Errorf("completing batch %d before %d returned %v, want ErrOutOfOrder", second.ID, first.ID, err)
	}
	return nil
}
