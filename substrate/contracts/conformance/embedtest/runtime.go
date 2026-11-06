package embedtest

import (
	"container/heap"
	"context"
	"fmt"
	"hash/fnv"
	"math/rand/v2"
	"net/http"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/embed"
)

// Epoch is the instant a Runtime's clock starts at.
var Epoch = time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)

// maxEvents bounds the events one Advance runs, so a loop that reschedules itself at the
// current instant fails instead of hanging.
const maxEvents = 1_000_000

// Runtime is a minimal deterministic embed.Runtime: a clock that moves only when Advance
// moves it, an event queue, and goroutines that run one at a time under a single baton, in an
// order fixed by the calls made on them. It also implements embed.RoundTripper, serving the
// router's outbound HTTP with Engines.
//
// A Runtime is driven from one goroutine: the test's. Router goroutines started with Go run
// only while that goroutine waits for them, inside a call into the router, Advance or Settle,
// so exactly one of them runs at a time. Every function a Runtime returns (a cancel function,
// a timer's Stop) is called under that discipline too.
type Runtime struct {
	// Engines serves the router's outbound HTTP requests. nil answers each with 404.
	Engines func(*http.Request) (*http.Response, error)

	seed    int64
	now     time.Duration // since Epoch
	events  eventQueue
	seq     uint64
	streams map[string]*stream
	locals  map[any]any

	current  *coroutine
	parked   []*coroutine
	nextID   int
	live     int
	settling bool

	randCalls    int  // Rand calls from any goroutine, to check that none happens during construction
	stopped      bool // set by the checks once Router.Stop has returned
	loopsStopped int  // Every callbacks run after stopped
}

var (
	_ embed.Runtime      = (*Runtime)(nil)
	_ embed.RoundTripper = (*Runtime)(nil)
)

// NewRuntime returns a Runtime at Epoch whose random streams derive from seed.
func NewRuntime(seed int64) *Runtime {
	return &Runtime{seed: seed, streams: map[string]*stream{}, locals: map[any]any{}}
}

// Now implements embed.Runtime.
func (r *Runtime) Now() time.Time { return Epoch.Add(r.now) }

// Advance moves the clock forward by d, running every event due on the way in time order
// (events at one instant in the order they were scheduled), and resumes parked goroutines
// after each.
func (r *Runtime) Advance(d time.Duration) {
	if d < 0 {
		panic(fmt.Sprintf("embedtest: Advance(%v): the clock does not run backwards", d))
	}
	end := r.now + d
	r.Settle()
	for n := 0; r.events.Len() > 0 && r.events[0].at <= end; n++ {
		if n == maxEvents {
			panic(fmt.Sprintf("embedtest: more than %d events within %v; a loop reschedules itself without advancing", maxEvents, d))
		}
		e := heap.Pop(&r.events).(*event)
		if e.cancelled {
			continue
		}
		r.now = e.at
		e.fn()
		r.Settle()
	}
	r.now = end
}

// Live returns the number of goroutines started with Go that have not finished.
func (r *Runtime) Live() int { return r.live }

// schedule runs fn d from now. The returned function cancels it.
func (r *Runtime) schedule(d time.Duration, fn func()) (cancel func()) {
	e := &event{at: r.now + max(0, d), seq: r.seq, fn: fn}
	r.seq++
	heap.Push(&r.events, e)
	return func() { e.cancelled = true }
}

// NewTimer implements embed.Runtime.
func (r *Runtime) NewTimer(d time.Duration) embed.Timer {
	t := &timer{rt: r, ch: make(chan time.Time, 1)}
	t.Reset(d)
	return t
}

type timer struct {
	rt     *Runtime
	ch     chan time.Time
	active bool
	cancel func()
}

func (t *timer) C() <-chan time.Time { return t.ch }

func (t *timer) Stop() bool {
	was := t.active
	t.active = false
	if t.cancel != nil {
		t.cancel()
	}
	return was
}

func (t *timer) Reset(d time.Duration) bool {
	was := t.Stop()
	t.active = true
	t.cancel = t.rt.schedule(d, func() {
		t.active = false
		select {
		case t.ch <- t.rt.Now():
		default:
		}
	})
	return was
}

// NewTicker implements embed.Runtime.
func (r *Runtime) NewTicker(d time.Duration) embed.Ticker {
	if d <= 0 {
		panic(fmt.Sprintf("embedtest: NewTicker(%v): the interval must be positive", d))
	}
	t := &ticker{ch: make(chan time.Time, 1)}
	var tick func()
	tick = func() {
		if t.stopped {
			return
		}
		select { // a slow reader drops ticks, as with time.Ticker
		case t.ch <- r.Now():
		default:
		}
		t.cancel = r.schedule(d, tick)
	}
	t.cancel = r.schedule(d, tick)
	return t
}

type ticker struct {
	ch      chan time.Time
	stopped bool
	cancel  func()
}

func (t *ticker) C() <-chan time.Time { return t.ch }

func (t *ticker) Stop() {
	t.stopped = true
	t.cancel()
}

// Every implements embed.Runtime.
func (r *Runtime) Every(ctx context.Context, name string, interval time.Duration, fn func(context.Context)) {
	if interval <= 0 {
		panic(fmt.Sprintf("embedtest: Every(%q, %v): the interval must be positive", name, interval))
	}
	var tick func()
	tick = func() {
		if ctx.Err() != nil {
			return
		}
		if r.stopped {
			r.loopsStopped++
		}
		fn(ctx)
		if ctx.Err() == nil {
			r.schedule(interval, tick)
		}
	}
	r.schedule(interval, tick)
}

// WithTimeout implements embed.Runtime.
func (r *Runtime) WithTimeout(parent context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	return r.WithDeadlineCause(parent, r.Now().Add(d), context.DeadlineExceeded)
}

// WithDeadlineCause implements embed.Runtime.
func (r *Runtime) WithDeadlineCause(parent context.Context, deadline time.Time, cause error) (context.Context, context.CancelFunc) {
	if cur, ok := parent.Deadline(); ok && cur.Before(deadline) {
		return context.WithCancel(parent)
	}
	// The context ends once, at one linearization point: decide, under mu. The first ending
	// to reach it (the deadline event, the parent ending, or the returned cancel) records its
	// error and cancels with its cause; later endings change nothing. A parent that has ended
	// always wins, even before its AfterFunc has run, because decide checks it first under the
	// lock. The context does not inherit the parent's cancellation directly, so the parent's
	// ending goes through decide too and never takes a second path around it.
	inner, cancelCause := context.WithCancelCause(context.WithoutCancel(parent))
	dc := &deadlineCtx{Context: inner, deadline: deadline}
	dc.decide = func(err, cause error) {
		dc.mu.Lock()
		defer dc.mu.Unlock()
		if dc.err != nil {
			return
		}
		if perr := parent.Err(); perr != nil {
			err, cause = perr, context.Cause(parent)
		}
		if err == nil {
			return // Err asked, and nothing has ended
		}
		dc.err = err
		cancelCause(cause)
	}
	stopParent := context.AfterFunc(parent, func() { dc.decide(parent.Err(), context.Cause(parent)) })
	stop := r.schedule(deadline.Sub(r.Now()), func() { dc.decide(context.DeadlineExceeded, cause) })
	return dc, func() {
		stop()
		stopParent()
		dc.decide(context.Canceled, context.Canceled)
	}
}

// deadlineCtx reports its deadline and, once it has ended, the error decide recorded. Err
// reads under the lock decide holds while it cancels, so it is nil until Done is closed and
// the same error from then on.
type deadlineCtx struct {
	context.Context
	deadline time.Time
	mu       sync.Mutex
	err      error
	decide   func(err, cause error)
}

func (c *deadlineCtx) Deadline() (time.Time, bool) { return c.deadline, true }

func (c *deadlineCtx) Err() error {
	c.decide(nil, nil) // a parent that has ended is decided now; nothing else is
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

// Rand implements embed.Runtime.
func (r *Runtime) Rand(name string) embed.Rand {
	r.randCalls++
	if s, ok := r.streams[name]; ok {
		return s
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(name))
	k := h.Sum64()
	s := &stream{r: rand.New(rand.NewPCG(uint64(r.seed), k))} //nolint:gosec // simulation randomness
	r.streams[name] = s
	return s
}

type stream struct{ r *rand.Rand }

func (s *stream) Float64() float64                   { return s.r.Float64() }
func (s *stream) IntN(n int) int                     { return s.r.IntN(n) }
func (s *stream) Shuffle(n int, swap func(i, j int)) { s.r.Shuffle(n, swap) }

// Local implements embed.Runtime.
func (r *Runtime) Local(key any, init func() any) any {
	if v, ok := r.locals[key]; ok {
		return v
	}
	v := init()
	r.locals[key] = v
	return v
}

// RoundTrip implements embed.RoundTripper.
func (r *Runtime) RoundTrip(req *http.Request) (*http.Response, error) {
	if r.Engines != nil {
		return r.Engines(req)
	}
	return Respond(req, http.StatusNotFound, "text/plain", "no engine serves this request\n"), nil
}

// Respond is a complete *http.Response for req, for an Engines function to return.
func Respond(req *http.Request, code int, contentType, body string) *http.Response {
	return &http.Response{
		Status:        fmt.Sprintf("%d %s", code, http.StatusText(code)),
		StatusCode:    code,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        http.Header{"Content-Type": []string{contentType}},
		Body:          readCloser{strings.NewReader(body)},
		ContentLength: int64(len(body)),
		Request:       req,
	}
}

type readCloser struct{ *strings.Reader }

func (readCloser) Close() error { return nil }

// --- goroutines ---

type coroutine struct {
	id      int
	resume  chan struct{}
	yielded chan struct{}
	ready   func() bool
	done    bool
	panicV  any
	stack   []byte
}

// Go implements embed.Runtime: fn runs at once, until it finishes or parks.
func (r *Runtime) Go(fn func()) {
	c := &coroutine{id: r.nextID, resume: make(chan struct{}), yielded: make(chan struct{})}
	r.nextID++
	r.live++
	go func() {
		<-c.resume
		defer func() {
			if v := recover(); v != nil {
				c.panicV, c.stack = v, debug.Stack()
			}
			c.done = true
			c.yielded <- struct{}{}
		}()
		fn()
	}()
	r.run(c)
}

// run hands the baton to c and takes it back when c finishes or parks.
func (r *Runtime) run(c *coroutine) {
	prev := r.current
	r.current = c
	c.resume <- struct{}{}
	// No wall-clock timeout: a goroutine that blocks on a wait the runtime cannot see (a
	// channel, lock or sleep not guarded by Park) hangs the test, and go test's -timeout
	// reports it with every goroutine's stack. Nothing here depends on how fast the host runs.
	<-c.yielded
	r.current = prev
	if c.done {
		r.live--
		if c.panicV != nil {
			panic(fmt.Sprintf("embedtest: router goroutine %d panicked: %v\n%s", c.id, c.panicV, c.stack))
		}
	}
}

// Park implements embed.Runtime.
func (r *Runtime) Park(ready func() bool) {
	if ready() {
		return
	}
	c := r.current
	if c == nil {
		panic("embedtest: the router parked on the host's goroutine; only goroutines started with Runtime.Go may wait")
	}
	c.ready = ready
	r.parked = append(r.parked, c)
	c.yielded <- struct{}{}
	<-c.resume
}

// Settle resumes parked goroutines whose ready holds, oldest first, until none is ready. Advance
// settles after every event; a test settles after a call into the router that may have made a
// parked goroutine ready (for example cancelling a request's context).
func (r *Runtime) Settle() {
	if r.current != nil || r.settling {
		return
	}
	r.settling = true
	defer func() { r.settling = false }()
	for {
		progressed := false
		for i := 0; i < len(r.parked); {
			c := r.parked[i]
			if !c.ready() {
				i++
				continue
			}
			r.parked = slices.Delete(r.parked, i, i+1)
			r.run(c)
			progressed = true
		}
		if !progressed {
			return
		}
		slices.SortFunc(r.parked, func(a, b *coroutine) int { return a.id - b.id })
	}
}

// --- event queue ---

type event struct {
	at        time.Duration
	seq       uint64
	fn        func()
	cancelled bool
}

type eventQueue []*event

func (q eventQueue) Len() int { return len(q) }
func (q eventQueue) Less(i, j int) bool {
	if q[i].at != q[j].at {
		return q[i].at < q[j].at
	}
	return q[i].seq < q[j].seq
}
func (q eventQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *eventQueue) Push(x any)   { *q = append(*q, x.(*event)) }
func (q *eventQueue) Pop() any {
	old := *q
	e := old[len(old)-1]
	*q = old[:len(old)-1]
	return e
}
