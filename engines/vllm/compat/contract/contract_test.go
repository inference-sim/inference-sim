// Original to blis-vllm.

package contract

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	_ "github.com/inference-sim/blis-vllm/compat/lora" // registers the resident set and adapter cost
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

// A request's first admission reports its prefix-cache lookup (Scheduled.PrefixLookup), and
// what the cache supplied of it (CachedTokens): a miss for the first of two identical prompts,
// whole blocks for the second.
func TestTheFirstAdmissionReportsItsPrefixLookup(t *testing.T) {
	e := newEngine(t)
	var got []engine.Scheduled
	for i, id := range []string{"a", "b"} {
		r := req("p", 64, 2)
		r.ID = id
		at := engine.Time(i * 100000)
		if err := e.Add(r, at); err != nil {
			t.Fatal(err)
		}
		for range 4 {
			b, err := e.Schedule(at)
			if err != nil {
				t.Fatal(err)
			}
			if b.Empty() {
				break
			}
			for _, s := range b.Reqs {
				if s.ID == id && s.PrefixLookup {
					got = append(got, s)
				}
			}
			at += e.StepTime(b)
			if _, err := e.Complete(b.ID, engine.Outcome{}, at); err != nil {
				t.Fatal(err)
			}
		}
	}
	if len(got) != 2 {
		t.Fatalf("lookups reported: %d, want one per request's first admission", len(got))
	}
	if got[0].CachedTokens != 0 || got[1].CachedTokens != 48 {
		t.Errorf("cached tokens = %d, %d; want 0 (a miss) and 48 (three full blocks; the last is recomputed)",
			got[0].CachedTokens, got[1].CachedTokens)
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

// A cold adapter request waits for its adapter's load, which is a wake of its own: the engine
// reports the load's end through NextWake while nothing runs, completes the load at that wake
// before it schedules, and then runs the request (compat/engine maybeStartAdapterLoad,
// CompleteAdapterLoad; the host loop's AdapterLoadCompletionEvent). The resident adapters,
// the slots and the cold-load count are the engine's.
func TestAnAdapterLoadIsAWakeAndCompletesBeforeTheStep(t *testing.T) {
	slots, base, bw, fp := 1, 5000.0, 1.0e9, 1.0e3
	cfg := cengine.SimConfig{
		Horizon:       1 << 40,
		KVCacheConfig: cengine.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
		BatchConfig:   cengine.NewBatchConfig(16, 2048, 0),
		LatencyModel:  fixedLatency{},
	}
	cfg.LoRAConfig = cengine.LoRAConfig{AdapterCapacity: &slots, LoadBaseLatencyUs: &base,
		LoadBandwidthBytesUs: &bw, FootprintBytesPerRank: &fp, Adapters: []cengine.AdapterSpec{{ID: "a0", Rank: 8}}}
	e, err := New("instance_0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := e.AdapterCapacity(); got != slots {
		t.Fatalf("AdapterCapacity = %d, want %d", got, slots)
	}
	r := req("a", 40, 3)
	r.Adapter = "a0"
	if err := e.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	if b, err := e.Schedule(0); err != nil || !b.Empty() {
		t.Fatalf("a cold adapter's request ran before its load: %+v, %v", b, err)
	}
	at, ok := e.NextWake(0)
	if !ok || at < int64(base) {
		t.Fatalf("NextWake = %d, %v: want the load's end, at least %v", at, ok, base)
	}
	if ids := e.ResidentAdapterIDs(); len(ids) != 0 {
		t.Fatalf("resident before the load completes: %v", ids)
	}
	b, err := e.Schedule(at)
	if err != nil || b.Empty() {
		t.Fatalf("at the load's end the request did not run: %+v, %v", b, err)
	}
	if ids := e.ResidentAdapterIDs(); len(ids) != 1 || ids[0] != "a0" {
		t.Errorf("resident after the load: %v, want [a0]", ids)
	}
	if loads, _ := e.AdapterEvents(); loads["a0"] != 1 {
		t.Errorf("cold loads of a0 = %d, want 1", loads["a0"])
	}
}

// A drained instance's waiting requests leave the engine, in queue order, holding no KV: the
// engine forgets them, and the requests it is running stay (DrainWaiting, the compatibility
// engine's DrainWaitQueue).
func TestDrainWaitingReturnsTheQueueAndKeepsTheRunning(t *testing.T) {
	cfg := cengine.SimConfig{
		Horizon:       1 << 40,
		KVCacheConfig: cengine.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
		BatchConfig:   cengine.NewBatchConfig(1, 2048, 0), // one running request at a time
		LatencyModel:  fixedLatency{},
	}
	e, err := New("instance_0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"a", "b", "c"} {
		r := req("p", 32, 4)
		r.ID = id
		if err := e.Add(r, 0); err != nil {
			t.Fatal(err)
		}
	}
	b, err := e.Schedule(0)
	if err != nil || len(b.Reqs) != 1 || b.Reqs[0].ID != "a" {
		t.Fatalf("the first step must run a alone: %+v %v", b, err)
	}
	if _, err := e.Complete(b.ID, engine.Outcome{}, 1000); err != nil {
		t.Fatal(err)
	}
	got := e.DrainWaiting()
	if len(got) != 2 || got[0] != "b" || got[1] != "c" {
		t.Fatalf("DrainWaiting = %v, want [b c]", got)
	}
	if s := e.Snapshot(); s.Waiting != 0 || s.Running != 1 {
		t.Errorf("after the drain: %d waiting, %d running; want 0 and 1", s.Waiting, s.Running)
	}
	if again := e.DrainWaiting(); len(again) != 0 {
		t.Errorf("a second drain returned %v", again)
	}
	if err := e.Add(req("b", 32, 4), 2000); err != nil {
		t.Errorf("a drained request re-added (re-routed back) is refused: %v", err)
	}
}
