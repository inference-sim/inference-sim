package sim

import (
	"container/heap"
	"fmt"
	"testing"
)

// testDefaultTimeoutUs mirrors workload.DefaultTimeoutUs (300s).
// Cannot import workload directly — that package imports sim (circular).
const testDefaultTimeoutUs = 300_000_000

// TestEventQueue_SameTimestamp_PriorityOrder verifies BC-12: at equal timestamps,
// events fire in priority order (lower priority number first).
// StepEvent (priority 2) must fire before TimeoutEvent (priority 5).
func TestEventQueue_SameTimestamp_PriorityOrder(t *testing.T) {
	eq := &EventQueue{}
	heap.Init(eq)

	// Push events at the same timestamp in reverse priority order
	tick := int64(1000)
	var seq int64

	// Push TimeoutEvent first (priority 5), then StepEvent (priority 2)
	seq++
	heap.Push(eq, eventEntry{event: &TimeoutEvent{time: tick, Request: &Request{ID: "r1"}}, seqID: seq})
	seq++
	heap.Push(eq, eventEntry{event: &StepEvent{time: tick}, seqID: seq})
	seq++
	heap.Push(eq, eventEntry{event: &ArrivalEvent{time: tick, Request: &Request{ID: "r2"}}, seqID: seq})

	// Pop and verify priority order: Arrival(0) < Step(2) < Timeout(5)
	e1 := heap.Pop(eq).(eventEntry)
	e2 := heap.Pop(eq).(eventEntry)
	e3 := heap.Pop(eq).(eventEntry)

	if e1.event.Priority() != PriorityArrival {
		t.Errorf("first event: got priority %d, want %d (Arrival)", e1.event.Priority(), PriorityArrival)
	}
	if e2.event.Priority() != PriorityStep {
		t.Errorf("second event: got priority %d, want %d (Step)", e2.event.Priority(), PriorityStep)
	}
	if e3.event.Priority() != PriorityTimeout {
		t.Errorf("third event: got priority %d, want %d (Timeout)", e3.event.Priority(), PriorityTimeout)
	}
}

// TestEventQueue_SeqID_BreaksTies verifies that seqID breaks ties within
// same-type same-timestamp events for INV-6 determinism.
func TestEventQueue_SeqID_BreaksTies(t *testing.T) {
	eq := &EventQueue{}
	heap.Init(eq)

	tick := int64(2000)
	// Two ArrivalEvents at same tick — seqID determines order
	heap.Push(eq, eventEntry{event: &ArrivalEvent{time: tick, Request: &Request{ID: "first"}}, seqID: 1})
	heap.Push(eq, eventEntry{event: &ArrivalEvent{time: tick, Request: &Request{ID: "second"}}, seqID: 2})

	e1 := heap.Pop(eq).(eventEntry)
	e2 := heap.Pop(eq).(eventEntry)

	if e1.seqID != 1 || e2.seqID != 2 {
		t.Errorf("seqID ordering: got %d then %d, want 1 then 2", e1.seqID, e2.seqID)
	}
}

// TestTimeout_QueuedRequest_TimesOut verifies BC-1: a queued request
// transitions to StateTimedOut when its deadline passes.
func TestTimeout_QueuedRequest_TimesOut(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(1, 2048, 0), // max 1 running request — forces queuing
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{0, 0, 0}), // zero alpha = no queueing delay
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test", "H100", 1, "roofline", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// Two requests: r1 arrives at 0 (will run), r2 arrives at 0 with short deadline (will timeout while queued)
	r1 := &Request{ID: "r1", ArrivalTime: 0, InputTokens: make([]int, 10), OutputTokens: make([]int, 100), State: StateQueued, MaxOutputLen: 100}
	r2 := &Request{ID: "r2", ArrivalTime: 0, InputTokens: make([]int, 10), OutputTokens: make([]int, 100), State: StateQueued, MaxOutputLen: 100, Deadline: 5000} // timeout at tick 5000

	sim.InjectArrival(r1)
	sim.InjectArrival(r2)
	sim.Run()

	// r2 should have timed out (batch size 1 means r1 runs, r2 waits, r2's deadline passes)
	if r2.State != StateTimedOut {
		t.Errorf("r2 state: got %s, want %s", r2.State, StateTimedOut)
	}
	if sim.Metrics.TimedOutRequests != 1 {
		t.Errorf("TimedOutRequests: got %d, want 1", sim.Metrics.TimedOutRequests)
	}
	// r1 should have completed
	if r1.State != StateCompleted {
		t.Errorf("r1 state: got %s, want %s", r1.State, StateCompleted)
	}
}

// TestTimeout_CompletedRequest_NoOp verifies BC-3: a TimeoutEvent for an
// already-completed request is a no-op.
func TestTimeout_CompletedRequest_NoOp(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test", "H100", 1, "roofline", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// One request with a deadline far in the future — will complete normally
	r1 := &Request{ID: "r1", ArrivalTime: 0, InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: StateQueued, MaxOutputLen: 5, Deadline: 999_999}

	sim.InjectArrival(r1)
	sim.Run()

	// Request should complete, timeout should be no-op
	if r1.State != StateCompleted {
		t.Errorf("r1 state: got %s, want %s", r1.State, StateCompleted)
	}
	if sim.Metrics.TimedOutRequests != 0 {
		t.Errorf("TimedOutRequests: got %d, want 0 (timeout should be no-op)", sim.Metrics.TimedOutRequests)
	}
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests: got %d, want 1", sim.Metrics.CompletedRequests)
	}
}

// TestTimeout_CompletionWinsAtEqualTimestamp verifies BC-12: when a StepEvent
// and TimeoutEvent fire at the same tick, the step event fires first (priority ordering).
func TestTimeout_CompletionWinsAtEqualTimestamp(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 0, 0}, []float64{0, 0, 0}), // step time = beta0 = 1000µs, no per-token cost
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test", "H100", 1, "roofline", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// Request with 1 input token, 1 output token. Step time = 1000µs.
	// Prefill step at t=0 completes at t=1000. Decode step at t=1000 completes at t=2000.
	// Set deadline = 2000 (same tick as completion).
	r1 := &Request{ID: "r1", ArrivalTime: 0, InputTokens: make([]int, 1), OutputTokens: make([]int, 1), State: StateQueued, MaxOutputLen: 1, Deadline: 2000}

	sim.InjectArrival(r1)
	sim.Run()

	// With priority ordering, StepEvent (priority 2) fires before TimeoutEvent (priority 5).
	// The request should complete, not time out.
	if r1.State != StateCompleted {
		t.Errorf("BC-12: r1 state: got %s, want %s (completion should win at equal tick)", r1.State, StateCompleted)
	}
	if sim.Metrics.TimedOutRequests != 0 {
		t.Errorf("BC-12: TimedOutRequests: got %d, want 0", sim.Metrics.TimedOutRequests)
	}
}

// TestWaitQueue_Remove verifies that Remove() correctly removes a request
// from the middle of the queue.
func TestWaitQueue_Remove(t *testing.T) {
	wq := &WaitQueue{}
	r1 := &Request{ID: "r1"}
	r2 := &Request{ID: "r2"}
	r3 := &Request{ID: "r3"}
	wq.Enqueue(r1)
	wq.Enqueue(r2)
	wq.Enqueue(r3)

	if wq.Len() != 3 {
		t.Fatalf("queue length: got %d, want 3", wq.Len())
	}

	// Remove middle element
	found := wq.Remove(r2)
	if !found {
		t.Error("Remove(r2): got false, want true")
	}
	if wq.Len() != 2 {
		t.Errorf("queue length after remove: got %d, want 2", wq.Len())
	}

	// Verify r1 and r3 remain in order
	if wq.Peek() != r1 {
		t.Errorf("peek after remove: got %s, want r1", wq.Peek().ID)
	}

	// Remove non-existent element
	found = wq.Remove(&Request{ID: "r4"})
	if found {
		t.Error("Remove(non-existent): got true, want false")
	}
}

// TestTimeout_RunningRequest_StateAndBatchCleanup verifies BC-2:
// a running request that times out is removed from RunningBatch,
// its state is StateTimedOut, and RunningBatch is nil'd when empty.
func TestTimeout_RunningRequest_StateAndBatchCleanup(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{5000, 10, 5}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test", "H100", 1, "roofline", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// Single request: will start running, then timeout before completing
	r1 := &Request{
		ID: "r1", ArrivalTime: 0,
		InputTokens: make([]int, 50), OutputTokens: make([]int, 100),
		MaxOutputLen: 100, State: StateQueued,
		Deadline: 15000, // times out at 15ms — only ~2 decode steps complete
	}
	sim.InjectArrival(r1)
	sim.Run()

	// BC-2: state must be StateTimedOut
	if r1.State != StateTimedOut {
		t.Errorf("BC-2: state = %s, want %s", r1.State, StateTimedOut)
	}
	// RunningBatch must be nil (the only running request timed out)
	if sim.RunningBatch != nil {
		t.Errorf("BC-2: RunningBatch should be nil after last running request timed out, got %d requests",
			len(sim.RunningBatch.Requests))
	}
	// Counter incremented
	if sim.Metrics.TimedOutRequests != 1 {
		t.Errorf("BC-2: TimedOutRequests = %d, want 1", sim.Metrics.TimedOutRequests)
	}
}

// TestTimeout_PreemptThenTimeout_SafeNoOp verifies BC-15:
// a request preempted (KV released, back in WaitQ) then timed out
// while queued should be safe — no double-free, no panic.
func TestTimeout_PreemptThenTimeout_SafeNoOp(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(5, 16, 0, 0, 0, 0), // tiny KV: 5 blocks = 80 tokens
		BatchConfig:         NewBatchConfig(2, 2048, 0),            // batch size 2
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test", "H100", 1, "roofline", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// r1: large request that will cause KV pressure and preempt r2
	r1 := &Request{
		ID: "r1", ArrivalTime: 0,
		InputTokens: make([]int, 60), OutputTokens: make([]int, 10),
		MaxOutputLen: 10, State: StateQueued,
	}
	// r2: small request with short deadline — will be preempted by r1's KV demand,
	// then timeout while back in queue
	r2 := &Request{
		ID: "r2", ArrivalTime: 0,
		InputTokens: make([]int, 30), OutputTokens: make([]int, 5),
		MaxOutputLen: 5, State: StateQueued,
		Deadline: 10000, // short deadline
	}

	sim.InjectArrival(r1)
	sim.InjectArrival(r2)
	sim.Run()

	// The test succeeds if no panic occurs (BC-15: no double-free).
	// Additionally verify conservation holds.
	completed := sim.Metrics.CompletedRequests
	queued := sim.WaitQ.Len()
	running := 0
	if sim.RunningBatch != nil {
		running = len(sim.RunningBatch.Requests)
	}
	dropped := sim.Metrics.DroppedUnservable
	timedOut := sim.Metrics.TimedOutRequests
	injected := 2 // we injected exactly 2 requests

	sum := completed + queued + running + dropped + timedOut
	if sum != injected {
		t.Errorf("BC-15 conservation: completed(%d) + queued(%d) + running(%d) + dropped(%d) + timedOut(%d) = %d, want %d",
			completed, queued, running, dropped, timedOut, sum, injected)
	}
}

// TestTimeout_OrphanedTimeout_DoesNotInflateSimEndedTime verifies that a
// completed request's orphaned TimeoutEvent does not advance the simulation
// clock. The real-world equivalent: a client cancels its deadline timer when
// the response arrives.
//
// Scenario: one request completes quickly, but its Deadline is 300s in the
// future. Without lazy cancellation, SimEndedTime would be ~300s (the orphaned
// timeout fires and advances the clock). With the fix, SimEndedTime reflects
// the actual completion time.
func TestTimeout_OrphanedTimeout_DoesNotInflateSimEndedTime(t *testing.T) {
	cfg := SimConfig{
		Horizon:             500_000_000, // 500s — well beyond the 300s deadline
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test", "H100", 1, "roofline", 0),
	}
	sim := mustNewSimulator(t, cfg)

	r1 := &Request{
		ID: "r1", ArrivalTime: 0,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
		State: StateQueued, MaxOutputLen: 5,
		Deadline: testDefaultTimeoutUs,
	}

	sim.InjectArrival(r1)
	sim.Run()

	if r1.State != StateCompleted {
		t.Fatalf("r1 state: got %s, want %s", r1.State, StateCompleted)
	}
	// INV-1: the skipped orphaned timeout must not corrupt conservation counters.
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests: got %d, want 1", sim.Metrics.CompletedRequests)
	}
	if sim.Metrics.TimedOutRequests != 0 {
		t.Errorf("TimedOutRequests: got %d, want 0 (orphaned timeout must not count as timed-out)", sim.Metrics.TimedOutRequests)
	}

	// SimEndedTime must reflect actual work completion, not the orphaned timeout.
	// With beta=[1000,10,5] (µs), beta0=base, beta1=cache-miss tokens (prefill only),
	// beta2=decode tokens. For 10 inputs + 5 outputs:
	//   prefill  = beta0 + beta1*10 = 1000 + 100 = 1100 µs
	//   decode×5 = 5 × (beta0 + beta2*1) = 5 × 1005 = 5025 µs
	//   total    ≈ 6125 µs
	// Lower bound (> 5_000): catches a regression where Clock is never advanced.
	// Upper bound (< 100_000): 3000× below testDefaultTimeoutUs, catches clock inflation.
	if sim.Metrics.SimEndedTime <= 5_000 {
		t.Errorf("SimEndedTime too low: got %d µs, want > 5_000 µs (Clock must advance for real work)",
			sim.Metrics.SimEndedTime)
	}
	const simEndedThreshold = 100_000 // 100 ms — 3000× above expected, 3000× below orphaned timeout
	if sim.Metrics.SimEndedTime > simEndedThreshold {
		t.Errorf("SimEndedTime inflated by orphaned timeout: got %d µs (%.1fs), want < %d µs",
			sim.Metrics.SimEndedTime, float64(sim.Metrics.SimEndedTime)/1e6, simEndedThreshold)
	}
}

// TestTimeout_OrphanedTimeout_MultipleOrphans_NoneInflateClock verifies that
// N > 1 completed requests each with an orphaned timeout do not cumulatively
// inflate SimEndedTime. Each orphaned pop must be skipped without advancing
// sim.Clock, so draining N orphaned timeouts leaves Clock at the last
// real-work timestamp rather than the last orphaned timestamp.
func TestTimeout_OrphanedTimeout_MultipleOrphans_NoneInflateClock(t *testing.T) {
	cfg := SimConfig{
		Horizon:             500_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test", "H100", 1, "roofline", 0),
	}
	sim := mustNewSimulator(t, cfg)

	const n = 3
	requests := make([]*Request, n)
	for i := 0; i < n; i++ {
		arrivalTime := int64(i * 50_000) // stagger by 50ms
		requests[i] = &Request{
			ID:           fmt.Sprintf("r%d", i),
			ArrivalTime:  arrivalTime,
			InputTokens:  make([]int, 10),
			OutputTokens: make([]int, 5),
			State:        StateQueued,
			MaxOutputLen: 5,
			Deadline:     arrivalTime + testDefaultTimeoutUs,
		}
		sim.InjectArrival(requests[i])
	}
	sim.Run()

	for i, r := range requests {
		if r.State != StateCompleted {
			t.Errorf("requests[%d] state: got %s, want %s", i, r.State, StateCompleted)
		}
	}
	// INV-1: n completed, 0 timed-out
	if sim.Metrics.CompletedRequests != n {
		t.Errorf("CompletedRequests: got %d, want %d", sim.Metrics.CompletedRequests, n)
	}
	if sim.Metrics.TimedOutRequests != 0 {
		t.Errorf("TimedOutRequests: got %d, want 0", sim.Metrics.TimedOutRequests)
	}
	// Last arrival at 100_000 µs; after processing all 3 requests SimEndedTime
	// should be > 100_000 (last arrival advanced the clock) and well under
	// testDefaultTimeoutUs (orphaned timeouts are all skipped).
	if sim.Metrics.SimEndedTime <= 100_000 {
		t.Errorf("SimEndedTime too low: got %d µs, want > 100_000 µs", sim.Metrics.SimEndedTime)
	}
	if sim.Metrics.SimEndedTime > 500_000 {
		t.Errorf("SimEndedTime inflated: got %d µs, want < 500_000 µs", sim.Metrics.SimEndedTime)
	}
}

// TestTimeout_CascadeDoesNotCreateOrphanedStepEvents verifies that simultaneous
// TimeoutEvents for running requests do not trigger the INV-8 guard and cascade
// into orphaned StepEvents (issue #1096).
//
// Without the fix, TimeoutEvent.Execute nil'd sim.stepEvent when emptying
// RunningBatch. The INV-8 guard then created a new StepEvent at the current tick,
// which (priority 2 < timeout priority 5) fired before remaining TimeoutEvents,
// pulling a queued seed that was immediately timed out. Each iteration left an
// orphaned StepEvent at the scheduled future tick. These orphans fired in lock-step
// with subsequent legitimate StepEvents, processing the batch N+1 times per tick
// and collapsing SimEndedTime by ~2x.
//
// GIVEN: 8 requests sharing the same arrival time; 6 with a short deadline (50ms),
//
//	2 with no deadline. KV cache: 4 blocks × 16 tokens, each request needs 2
//	blocks (InputLen=16 fills block0; first decode token at PI=16 needs block1).
//	This fits exactly 2 requests, leaving 6 to queue.
//
// WHEN: The simulation runs to completion.
//
// THEN: The 6 deadline requests time out, the 2 no-deadline requests complete
//
//	normally, and SimEndedTime reflects realistic per-step latency (~130ms),
//	not collapsed cascade timing (~70ms).
func TestTimeout_CascadeDoesNotCreateOrphanedStepEvents(t *testing.T) {
	const (
		deadlineTicks = int64(50_000) // 50ms — requests will not complete in time
		stepTimeTicks = int64(10_000) // 10ms per step (alpha[0]=10000)
		inputLen      = 16            // tokens per request — fills exactly 1 KV block (positions 0-15)
		outputLen     = 8             // needs 1 decode block (positions 16-23); total 2 blocks
		numDeadline   = 6             // requests that will time out
		numSurviving  = 2             // requests with no deadline (will complete)
		numRequests   = numDeadline + numSurviving
	)
	// KV cache: 4 blocks × 16 tokens.
	// Each request needs 2 blocks (input block + first decode block), so 2 fit simultaneously.
	// r0–r5 carry deadline=50ms; r6–r7 carry no deadline (Deadline==0 → no TimeoutEvent).
	cfg := SimConfig{
		Horizon:             500_000, // 500ms — well beyond the ~130ms expected completion
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(4, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 10_000, 16),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{float64(stepTimeTicks), 0, 0}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test", "H100", 1, "roofline", 0),
	}
	// Use a fixed-step-time latency model to get deterministic 10ms steps,
	// independent of the roofline model's FLOPs/bandwidth calculation.
	kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
	latencyModel := &fixedStepModel{stepTime: stepTimeTicks}
	sim, err := NewSimulator(cfg, kvStore, latencyModel)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	for i := 0; i < numRequests; i++ {
		deadline := deadlineTicks
		if i >= numDeadline {
			deadline = 0
		}
		sim.InjectArrival(&Request{
			ID:           fmt.Sprintf("r%d", i),
			ArrivalTime:  0,
			InputTokens:  make([]int, inputLen),
			OutputTokens: make([]int, outputLen),
			MaxOutputLen: outputLen,
			State:        StateQueued,
			Deadline:     deadline,
		})
	}
	sim.Run()

	// THEN: all deadline requests timed out, no-deadline requests completed.
	if sim.Metrics.TimedOutRequests != numDeadline {
		t.Errorf("TimedOutRequests: got %d, want %d", sim.Metrics.TimedOutRequests, numDeadline)
	}
	if sim.Metrics.CompletedRequests != numSurviving {
		t.Errorf("CompletedRequests: got %d, want %d", sim.Metrics.CompletedRequests, numSurviving)
	}

	// INV-1: conservation across all terminal states.
	total := sim.Metrics.CompletedRequests + sim.Metrics.TimedOutRequests +
		sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != numRequests {
		t.Errorf("INV-1 violated: total %d != injected %d", total, numRequests)
	}

	// THEN: SimEndedTime must reflect realistic per-step latency for the surviving
	// requests, not the collapsed timing caused by cascading orphaned StepEvents.
	//
	// With the fix: SimEndedTime ≈ 140ms (measured: 140,000 µs).
	//   Cascade at t=50ms: r0/r1 timeout, r2–r5 queue timeout, r6/r7 survive.
	//   Surviving requests admitted at ~t=60ms; 1 prefill + 7 decode = 8 steps
	//   at 10ms/step → complete at t=140ms.
	//
	// Without the fix (cascade): SimEndedTime ≈ 100ms (measured: 100,000 µs).
	//   Orphaned StepEvents fire alongside the legitimate ones, advancing the
	//   surviving requests faster than real step timing allows, collapsing
	//   SimEndedTime by ~30%.
	const (
		simEndedMin = int64(120_000) // 120ms: midpoint between fixed (140ms) and buggy (100ms)
		simEndedMax = int64(200_000) // 200ms: 1.43× expected 140ms, catches step-time regressions
	)
	if sim.Metrics.SimEndedTime < simEndedMin {
		t.Errorf("SimEndedTime collapsed by cascade: got %d µs, want >= %d µs (orphaned StepEvents suspected)",
			sim.Metrics.SimEndedTime, simEndedMin)
	}
	if sim.Metrics.SimEndedTime > simEndedMax {
		t.Errorf("SimEndedTime too high: got %d µs, want <= %d µs", sim.Metrics.SimEndedTime, simEndedMax)
	}
}
