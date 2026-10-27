package llmd

import (
	"fmt"
	"time"

	"github.com/inference-sim/blis-schemas/embed"
)

// The router's timers and tickers run on simulated time. They deliver on buffered channels from
// simulation events; a coroutine waiting on one parks until the value is there.

// NewTimer implements embed.Runtime.
func (rt *desRuntime) NewTimer(d time.Duration) embed.Timer {
	t := &desTimer{rt: rt, ch: make(chan time.Time, 1)}
	t.Reset(d)
	return t
}

// NewTicker implements embed.Runtime.
func (rt *desRuntime) NewTicker(d time.Duration) embed.Ticker {
	if d <= 0 {
		panic(fmt.Sprintf("llmd: NewTicker with non-positive interval %v", d))
	}
	t := &desTicker{ch: make(chan time.Time, 1)}
	step := toMicros(d)
	t.p = rt.startPeriodic(rt.host.Now()+step, step, PriorityTimer, func() bool {
		select { // like time.Ticker, a slow reader drops ticks
		case t.ch <- rt.Now():
		default:
		}
		rt.co.settle()
		return true
	})
	return t
}

type desTicker struct {
	ch chan time.Time
	p  *periodic
}

func (t *desTicker) C() <-chan time.Time { return t.ch }

func (t *desTicker) Stop() { t.p.stop() }

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
