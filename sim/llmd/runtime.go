package llmd

import (
	"context"
	"fmt"
	"hash/fnv"
	"math/rand"
	"sync/atomic"
	"time"

	"github.com/llm-d/llm-d-router/pkg/common/hosting"
	"k8s.io/utils/clock"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// Epoch is the wall-clock instant that simulated time zero maps to. It is a fixed constant so
// that every timestamp the router computes is a function of simulated time only (INV-6).
var Epoch = time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)

// Host is the discrete-event engine the router runs on. ClusterSimulator implements it.
type Host interface {
	// Now returns the current simulated time in microseconds.
	Now() int64
	// Schedule runs fn at simulated time at (>= Now). Events at one timestamp run in priority
	// order, then in scheduling order. A daemon event does not keep the simulation alive: the
	// simulation ends when only daemon events remain and the gateway is idle.
	Schedule(at int64, priority int, daemon bool, fn func()) (cancel func())
}

// Event priorities of router work at one timestamp. The request path runs where the cluster's
// routing decision always ran; background work runs after every cluster event at the timestamp.
const (
	PriorityRequest     = cluster.PriorityGatewayRequest        // request-path hand-offs between BLIS and the router
	PriorityKVEvents    = cluster.PriorityGatewayBackground     // KV-event delivery
	PriorityScrape      = cluster.PriorityGatewayBackground + 1 // metrics collection
	PriorityTimer       = cluster.PriorityGatewayBackground + 2 // tickers, deadlines and other runtime timers
	PriorityHousekeeper = cluster.PriorityGatewayBackground + 3 // cleanup loops
)

// desRuntime is the hosting.Runtime of a router embedded in the DES. Every router action runs on
// the simulation goroutine or on a coroutine that holds the baton, so the router is a
// deterministic function of the configuration, the seed and the sequence of calls BLIS makes.
type desRuntime struct {
	host         Host
	seed         int64
	scope        string
	streams      map[string]*desRand
	locals       map[any]any
	co           *scheduler
	clk          *desClock
	server       *modelServers
	scrapeJitter time.Duration
	// warm makes collectors started now take their first sample immediately: endpoints that
	// exist when the simulation starts have been scraped by a router that was already running.
	warm bool
	// periodics are the router's repeating loops (Every and tickers), in creation order.
	periodics []*periodic
}

var _ hosting.Runtime = (*desRuntime)(nil)

func newDESRuntime(host Host, seed int64, scope string, scrapeJitter time.Duration) *desRuntime {
	rt := &desRuntime{
		host:         host,
		seed:         seed,
		scope:        scope,
		streams:      map[string]*desRand{},
		locals:       map[any]any{},
		scrapeJitter: scrapeJitter,
	}
	rt.co = newScheduler()
	rt.clk = &desClock{rt: rt}
	return rt
}

func (rt *desRuntime) Now() time.Time { return toTime(rt.host.Now()) }

func toTime(us int64) time.Time { return Epoch.Add(time.Duration(us) * time.Microsecond) }

func toMicros(d time.Duration) int64 {
	us := d.Microseconds()
	if d > 0 && us == 0 {
		return 1 // a positive duration never collapses onto the current instant
	}
	return us
}

func (rt *desRuntime) Clock() clock.WithTicker { return rt.clk }

func (rt *desRuntime) Go(fn func()) { rt.co.spawn(fn) }

func (rt *desRuntime) Park(ready func() bool) { rt.co.park(ready) }

func (rt *desRuntime) Hosted() bool { return true }

func (rt *desRuntime) Local(key any, init func() any) any {
	if v, ok := rt.locals[key]; ok {
		return v
	}
	v := init()
	rt.locals[key] = v
	return v
}

func (rt *desRuntime) Rand(stream string) hosting.Rand {
	if r, ok := rt.streams[stream]; ok {
		return r
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(rt.scope + "/" + stream))
	r := &desRand{r: rand.New(rand.NewSource(rt.seed ^ int64(h.Sum64())))} //nolint:gosec // simulation RNG
	rt.streams[stream] = r
	return r
}

// Every runs fn every interval on the simulation clock while ctx is live. The first call is one
// interval after Every (plus a seeded phase offset below the scrape jitter for collectors).
func (rt *desRuntime) Every(ctx context.Context, name string, interval time.Duration, fn func(context.Context)) {
	if interval <= 0 {
		panic(fmt.Sprintf("llmd: Every(%q) with non-positive interval %v", name, interval))
	}
	step := toMicros(interval)
	first := step
	if rt.warm && isCollector(name) {
		first = 0
	} else if rt.scrapeJitter > 0 && isCollector(name) {
		first += int64(rt.Rand("scrape-phase/"+name).Float64() * float64(toMicros(rt.scrapeJitter)))
	}
	prio := PriorityHousekeeper
	if isCollector(name) {
		prio = PriorityScrape
	}
	rt.startPeriodic(rt.host.Now()+first, step, prio, func() bool {
		if ctx.Err() != nil {
			return false
		}
		fn(ctx)
		rt.co.settle()
		return ctx.Err() == nil
	})
}

// periodic is a repeating daemon event: a router loop's ticks.
type periodic struct {
	due    int64 // time of the next tick
	step   int64
	prio   int
	fire   func() bool // runs one tick; false ends the loop
	cancel func()
	done   bool
}

func (rt *desRuntime) startPeriodic(first, step int64, prio int, fire func() bool) *periodic {
	p := &periodic{due: first, step: step, prio: prio, fire: fire}
	rt.periodics = append(rt.periodics, p)
	rt.armPeriodic(p)
	return p
}

func (rt *desRuntime) armPeriodic(p *periodic) {
	p.cancel = rt.host.Schedule(p.due, p.prio, true, func() {
		if p.done {
			return
		}
		if !p.fire() {
			p.stop()
			return
		}
		p.due += p.step
		rt.armPeriodic(p)
	})
}

func (p *periodic) stop() {
	p.done = true
	if p.cancel != nil {
		p.cancel()
	}
}

// fastForward moves every live periodic loop to its last two ticks before target, keeping each
// loop's phase. The caller guarantees the router is idle until target: no request is anywhere in
// the router or the engines, so the published metrics a scrape reads do not change and every
// background loop's effect over the skipped ticks is what its last ticks produce (a scrape of
// unchanged metrics, TTL reaping and garbage collection are idempotent over an idle period;
// two ticks cover loops that mark on one pass and sweep on the next).
func (rt *desRuntime) fastForward(target int64) {
	live := rt.periodics[:0]
	for _, p := range rt.periodics {
		if p.done {
			continue
		}
		live = append(live, p)
		if last := p.due + (target-1-p.due)/p.step*p.step; p.due < target && last-p.step > p.due {
			p.cancel()
			p.due = last - p.step
			rt.armPeriodic(p)
		}
	}
	rt.periodics = live
}

func isCollector(name string) bool {
	return len(name) > len("datalayer-collector/") && name[:len("datalayer-collector/")] == "datalayer-collector/"
}

// WithTimeout returns a context that never times out: router work takes no simulated time, so a
// timeout around it can never elapse. Cancellation still propagates.
func (rt *desRuntime) WithTimeout(parent context.Context, _ time.Duration) (context.Context, context.CancelFunc) {
	return context.WithCancel(parent)
}

// WithDeadlineCause returns a context whose deadline is a simulation event: it is cancelled with
// cause when the simulation clock reaches deadline.
func (rt *desRuntime) WithDeadlineCause(parent context.Context, deadline time.Time, cause error) (context.Context, context.CancelFunc) {
	if cur, ok := parent.Deadline(); ok && cur.Before(deadline) {
		return context.WithCancel(parent)
	}
	inner, cancelCause := context.WithCancelCause(parent)
	dc := &deadlineCtx{Context: inner, deadline: deadline}
	at := max(rt.host.Now(), deadline.Sub(Epoch).Microseconds())
	stopEvent := rt.host.Schedule(at, PriorityTimer, true, func() {
		if inner.Err() == nil {
			dc.expired.Store(true)
			cancelCause(cause)
			rt.co.settle()
		}
	})
	return dc, func() {
		stopEvent()
		cancelCause(context.Canceled)
	}
}

// deadlineCtx reports a deadline and, once that deadline has passed, DeadlineExceeded.
type deadlineCtx struct {
	context.Context
	deadline time.Time
	expired  atomic.Bool
}

func (c *deadlineCtx) Deadline() (time.Time, bool) { return c.deadline, true }

func (c *deadlineCtx) Err() error {
	if c.expired.Load() {
		return context.DeadlineExceeded
	}
	return c.Context.Err()
}

type desRand struct{ r *rand.Rand }

func (d *desRand) Float64() float64                   { return d.r.Float64() }
func (d *desRand) IntN(n int) int                     { return d.r.Intn(n) }
func (d *desRand) Shuffle(n int, swap func(i, j int)) { d.r.Shuffle(n, swap) }

// streamSeed derives the router seed from the run seed so that enabling a router stream never
// shifts another subsystem's draws.
func streamSeed(rng *sim.PartitionedRNG) int64 {
	return rng.ForSubsystem("epp").Int63()
}
