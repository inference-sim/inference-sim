// Original to blis-vllm.

package contract

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
)

type fixedLatency struct{}

func (fixedLatency) StepTime([]*cengine.Request) int64   { return 1000 }
func (fixedLatency) QueueingTime(*cengine.Request) int64 { return 0 }
func (fixedLatency) OutputTokenProcessingTime() int64    { return 0 }
func (fixedLatency) PostDecodeFixedOverhead() int64      { return 0 }

func newEngine(t *testing.T) *Engine {
	t.Helper()
	cfg := cengine.SimConfig{
		Horizon:       1 << 40,
		KVCacheConfig: cengine.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
		BatchConfig:   cengine.NewBatchConfig(16, 2048, 0),
		LatencyModel:  fixedLatency{},
	}
	e, err := New("instance_0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func req(id string, prompt, output int) engine.Request {
	r := engine.Request{ID: id, Prompt: make([]int32, prompt), Output: make([]int32, output)}
	for i := range r.Prompt {
		r.Prompt[i] = int32(len(id)*1000 + i + 1)
	}
	return r
}

// The adapter keeps blis-schemas' contract laws (enginetest.TestEngine: wakes, ordering,
// liveness, conservation), and refuses the protocol's misuses as the contract states.
func TestTheAdapterKeepsTheContractsLaws(t *testing.T) {
	build := func() (engine.Engine, error) { return newEngine(t), nil }
	tr := enginetest.TestEngine(t, build, []enginetest.Arrival{
		{At: 0, Req: req("a", 40, 6)}, {At: 0, Req: req("bb", 200, 3)}, {At: 2500, Req: req("ccc", 64, 10)},
	}, enginetest.Options{})
	if len(tr.Finished) != 3 {
		t.Errorf("finished %d of 3 requests", len(tr.Finished))
	}
	if err := enginetest.CheckProtocolErrors(build, req("a", 40, 3), req("bb", 40, 3)); err != nil {
		t.Error(err)
	}
}

// An abort is a gateway eviction: the request's KV is freed at once, and it is not counted as
// timed out, unlike a timeout's cancellation (compat/engine Evict, CancelOnTimeout).
func TestAnAbortEvictsAndATimeoutCancels(t *testing.T) {
	for _, c := range []struct {
		why      engine.FinishReason
		timedOut bool
	}{{engine.FinishAbort, false}, {engine.FinishTimeout, true}} {
		e := newEngine(t)
		if err := e.Add(req("a", 64, 8), 0); err != nil {
			t.Fatal(err)
		}
		b, err := e.Schedule(0)
		if err != nil || b.Empty() {
			t.Fatalf("%s: the request did not run: %v", c.why, err)
		}
		if _, err := e.Complete(b.ID, engine.Outcome{}, 1000); err != nil {
			t.Fatal(err)
		}
		if used := e.Snapshot().KVBlocksUsed; used == 0 {
			t.Fatalf("%s: the running request holds no KV", c.why)
		}
		held := e.reqs["a"]
		if err := e.Finish("a", c.why, 1000); err != nil {
			t.Fatal(err)
		}
		if used := e.Snapshot().KVBlocksUsed; used != 0 {
			t.Errorf("%s: %d KV blocks still held after Finish", c.why, used)
		}
		if got := held.State == cengine.StateTimedOut; got != c.timedOut {
			t.Errorf("%s: timed out = %v, want %v", c.why, got, c.timedOut)
		}
	}
}
