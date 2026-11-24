package sim

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// deferredEngine is the FIFO engine with one engine rule on top: nothing is scheduled before
// until, which it reports as its next wake, the way an engine reports a deferred admission.
type deferredEngine struct {
	engine.Engine
	until engine.Time
}

func (e *deferredEngine) Schedule(now engine.Time) (engine.Batch, error) {
	if now < e.until {
		return engine.Batch{}, nil
	}
	return e.Engine.Schedule(now)
}

func (e *deferredEngine) NextWake(now engine.Time) (engine.Time, bool) {
	if now < e.until {
		return e.until, true
	}
	return e.Engine.NextWake(now)
}

// rearmAt calls the driver's rearmWake at its instant, ahead of the step events of that
// instant: what a host input at the instant a wake is due does (the cluster's AddRemote of a
// handoff to a decode engine that already holds work).
type rearmAt struct{ at int64 }

func (e *rearmAt) Timestamp() int64    { return e.at }
func (e *rearmAt) Priority() int       { return PriorityArrival }
func (e *rearmAt) run(d *EngineDriver) { d.rearmWake(e.at) }

// A wake that is due fires, whatever rearms the wake at its own instant before it does. NextWake
// names only instants after now, so at the wake's instant the engine cannot name it again, and
// the step it was for must not be lost (#2685: a decode request whose pull had landed was never
// stepped, and the run stalled, SIM-7).
func TestEngineDriver_ADueWakeSurvivesARearmAtItsInstant(t *testing.T) {
	for _, rearm := range []bool{false, true} {
		fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100, StepPerTokenUs: 1})
		if err != nil {
			t.Fatal(err)
		}
		d := NewEngineDriver(&deferredEngine{Engine: fifo, until: 500}, EngineDriverConfig{Horizon: 1 << 40})
		req := &Request{ID: "r", InputTokens: tokens(10), OutputTokens: tokens(2), ArrivalTime: 0}
		d.InjectArrival(req)
		if rearm {
			d.schedule(&rearmAt{at: 500})
		}
		d.Run()
		if d.Metrics.CompletedRequests != 1 {
			t.Fatalf("rearm at the wake's instant %v: completed %d, want 1 (the wake at 500 was lost)", rearm, d.Metrics.CompletedRequests)
		}
		// Admitted at the wake: prefill 100 + 10 us from 500.
		if req.FirstTokenTime != 610 {
			t.Errorf("rearm %v: first token at %d, want 610 (stepped at the wake's instant)", rearm, req.FirstTokenTime)
		}
	}
}
