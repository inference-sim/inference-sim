// Original to blis-vllm.

package engine_test

import (
	"testing"

	"github.com/inference-sim/blis-vllm/compat/engine"
)

// newRemoteKVEngine is an engine with blocks KV blocks of 16 tokens and no speculation, so a
// remote-KV request whose prompt fills its last block needs one more block for its first
// decode step.
func newRemoteKVEngine(t *testing.T, blocks int64) *engine.EngineCore {
	t.Helper()
	cfg := engine.SimConfig{
		KVCacheConfig: engine.NewKVCacheConfig(blocks, 16, 0, 0, 0, 0),
		BatchConfig:   engine.NewBatchConfig(16, 2048, 0),
	}
	e, err := engine.NewEngineCore(cfg, engine.MustNewKVStoreFromConfig(cfg.KVCacheConfig), fixedLatency{})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

type fixedLatency struct{}

func (fixedLatency) StepTime([]*engine.Request) int64   { return 1000 }
func (fixedLatency) QueueingTime(*engine.Request) int64 { return 0 }
func (fixedLatency) OutputTokenProcessingTime() int64   { return 0 }
func (fixedLatency) PostDecodeFixedOverhead() int64     { return 0 }

// decodeReq is a P/D decode sub-request whose prompt is blocks whole 16-token blocks, unique
// to id so no two share a prefix block.
func decodeReq(id string, seed, blocks int) *engine.Request {
	in := make([]engine.TokenID, 16*blocks)
	for i := range in {
		in[i] = engine.TokenID(seed*100_003 + i + 1)
	}
	return &engine.Request{ID: id, InputTokens: in, OutputTokens: make([]engine.TokenID, 4),
		MaxOutputLen: 4, State: engine.StateWaitingForRemoteKVs, IsDecodeSubRequest: true}
}

// Pulls are admitted from the head for as long as each fits, leaving free the first-step
// blocks of the pulls already in flight; the first that does not fit stops the scan, so a
// smaller pull behind it waits too (vLLM's FCFS waiting queue). A landed pull gives its
// reservation back.
func TestAdmitRemoteKVReservesInFlightPullsAndStopsAtTheHead(t *testing.T) {
	e := newRemoteKVEngine(t, 20)
	// a takes 9 blocks and reserves 1 for its first step, leaving 11 free with 1 reserved.
	// b needs 10 + its own 1 = 11, but only 10 are free once a's block is reserved.
	a, b := decodeReq("a", 1, 9), decodeReq("b", 2, 10)
	got := e.AdmitRemoteKV([]*engine.Request{a, b})
	if len(got) != 1 || got[0].Req != a || got[0].PulledTokens != 16*9 {
		t.Fatalf("admitted %v, want only a with its 144 tokens pulled", got)
	}
	if e.InflightPullBlocks() != 1 {
		t.Errorf("in-flight first-step blocks %d, want a's 1", e.InflightPullBlocks())
	}
	// Head-of-line: b does not fit, so c, which would, waits behind it.
	c := decodeReq("c", 3, 1)
	if got := e.AdmitRemoteKV([]*engine.Request{b, c}); len(got) != 0 {
		t.Errorf("admitted %v behind a head that does not fit, want none", got)
	}
	// Alone, c fits: 11 free, minus a's reserved 1, leaves room for c's 1 + its own 1.
	if got := e.AdmitRemoteKV([]*engine.Request{c}); len(got) != 1 || got[0].Req != c {
		t.Errorf("c alone: admitted %v, want c", got)
	}
	// a lands: its reservation is released, and only c's remains. Landing twice is a no-op.
	e.RemoteKVLanded(a)
	e.RemoteKVLanded(a)
	if e.InflightPullBlocks() != 1 {
		t.Errorf("after a lands (twice): in-flight %d, want c's 1", e.InflightPullBlocks())
	}
}

// The in-flight reservation is what keeps a second pull from taking the blocks the first
// still needs to run: without it, b fits by blocks alone.
func TestAdmitRemoteKVLeavesTheFirstStepOfPullsInFlight(t *testing.T) {
	e := newRemoteKVEngine(t, 20)
	a := decodeReq("a", 1, 9)
	if got := e.AdmitRemoteKV([]*engine.Request{a}); len(got) != 1 {
		t.Fatalf("a must be admitted, got %v", got)
	}
	// 11 blocks free. b needs 9 + its own 1 = 10, which fits only if a's reserved block is
	// not counted: 11 - 1 = 10 is exactly enough. One more prompt block is not.
	if got := e.AdmitRemoteKV([]*engine.Request{decodeReq("b", 2, 10)}); len(got) != 0 {
		t.Errorf("b (10 blocks + 1) admitted into 11 free blocks with 1 reserved for a: %v", got)
	}
	if got := e.AdmitRemoteKV([]*engine.Request{decodeReq("b2", 3, 9)}); len(got) != 1 {
		t.Errorf("b2 (9 blocks + 1) must fit in 11 free with 1 reserved: %v", got)
	}
}

// A landed pull that waits to run (it holds its KV in the waiting queue) still needs its first
// decode step's block, so a new pull must leave it free as well as the pulls in flight.
func TestAdmitRemoteKVLeavesTheFirstStepOfLandedPullsWaitingToRun(t *testing.T) {
	e := newRemoteKVEngine(t, 20)
	a := decodeReq("a", 1, 9)
	if got := e.AdmitRemoteKV([]*engine.Request{a}); len(got) != 1 {
		t.Fatalf("a must be admitted, got %v", got)
	}
	// a's pull lands and a waits to run, holding its KV: no longer in flight, still reserved.
	e.RemoteKVLanded(a)
	a.State = engine.StateQueued
	e.WaitQ.Enqueue(a)
	if e.InflightPullBlocks() != 0 || e.WaitingHolderBlocks() != 1 {
		t.Fatalf("in flight %d, waiting holders %d; want 0 and a's 1", e.InflightPullBlocks(), e.WaitingHolderBlocks())
	}
	// 11 free; b needs 10 + 1 = 11, which fits only if a's waiting first step is not reserved.
	if got := e.AdmitRemoteKV([]*engine.Request{decodeReq("b", 2, 10)}); len(got) != 0 {
		t.Errorf("b admitted into the block a's waiting first step needs: %v", got)
	}
}

// The engine, not its caller, decides the order its waiting pulls are admitted in. The
// compatibility engine keeps the order the handoffs reached it under every scheduling policy
// (blis-vllm#30 tracks vLLM's priority order): a later, more urgent request does not overtake.
func TestAdmitRemoteKVKeepsTheArrivalOrderUnderEveryPolicy(t *testing.T) {
	for _, policy := range []string{"fcfs", "priority-fcfs"} {
		cfg := engine.SimConfig{
			KVCacheConfig: engine.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
			BatchConfig:   engine.NewBatchConfig(16, 2048, 0),
			PolicyConfig:  engine.NewPolicyConfig(policy, "fcfs"),
		}
		e, err := engine.NewEngineCore(cfg, engine.MustNewKVStoreFromConfig(cfg.KVCacheConfig), fixedLatency{})
		if err != nil {
			t.Fatal(err)
		}
		first, urgent := decodeReq("first", 1, 2), decodeReq("urgent", 2, 2)
		first.Priority, urgent.Priority = 5, 0 // lower is more urgent (vLLM convention)
		first.ArrivalTime, urgent.ArrivalTime = 0, 10
		got := e.AdmitRemoteKV([]*engine.Request{first, urgent})
		if len(got) != 2 || got[0].Req != first || got[1].Req != urgent {
			ids := []string{}
			for _, a := range got {
				ids = append(ids, a.Req.ID)
			}
			t.Errorf("%s: admitted %v, want the arrival order [first urgent]", policy, ids)
		}
	}
}
