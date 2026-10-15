package llmd

import (
	"fmt"
	"time"

	"k8s.io/utils/clock"
)

// desClock is the router's clock.WithTicker on simulated time. Tickers and timers deliver on
// buffered channels from simulation events; a coroutine waiting on one parks until the value is
// there.
type desClock struct{ rt *desRuntime }

var _ clock.WithTicker = (*desClock)(nil)

func (c *desClock) Now() time.Time                  { return c.rt.Now() }
func (c *desClock) Since(t time.Time) time.Duration { return c.rt.Now().Sub(t) }

func (c *desClock) Sleep(d time.Duration) {
	panic(fmt.Sprintf("llmd: router called Sleep(%v): blocking sleeps are not supported in a hosted runtime", d))
}

func (c *desClock) After(d time.Duration) <-chan time.Time { return c.NewTimer(d).C() }

func (c *desClock) Tick(d time.Duration) <-chan time.Time { return c.NewTicker(d).C() }

func (c *desClock) NewTimer(d time.Duration) clock.Timer {
	t := &desTimer{rt: c.rt, ch: make(chan time.Time, 1)}
	t.Reset(d)
	return t
}

func (c *desClock) NewTicker(d time.Duration) clock.Ticker {
	if d <= 0 {
		panic(fmt.Sprintf("llmd: NewTicker with non-positive interval %v", d))
	}
	t := &desTicker{rt: c.rt, ch: make(chan time.Time, 1), step: toMicros(d)}
	t.arm()
	return t
}

type desTicker struct {
	rt      *desRuntime
	ch      chan time.Time
	step    int64
	stopped bool
	cancel  func()
}

func (t *desTicker) C() <-chan time.Time { return t.ch }

func (t *desTicker) arm() {
	t.cancel = t.rt.host.Schedule(t.rt.host.Now()+t.step, PriorityTimer, true, func() {
		if t.stopped {
			return
		}
		select { // like time.Ticker, a slow reader drops ticks
		case t.ch <- t.rt.Now():
		default:
		}
		t.arm()
		t.rt.co.settle()
	})
}

func (t *desTicker) Stop() {
	t.stopped = true
	if t.cancel != nil {
		t.cancel()
	}
}

type desTimer struct {
	rt     *desRuntime
	ch     chan time.Time
	active bool
	cancel func()
}

func (t *desTimer) C() <-chan time.Time { return t.ch }

func (t *desTimer) Stop() bool {
	was := t.active
	t.active = false
	if t.cancel != nil {
		t.cancel()
	}
	return was
}

func (t *desTimer) Reset(d time.Duration) bool {
	was := t.Stop()
	t.active = true
	t.cancel = t.rt.host.Schedule(t.rt.host.Now()+max(0, toMicros(d)), PriorityTimer, true, func() {
		if !t.active {
			return
		}
		t.active = false
		select {
		case t.ch <- t.rt.Now():
		default:
		}
		t.rt.co.settle()
	})
	return was
}
