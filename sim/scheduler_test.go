package sim

import (
	"testing"
)

func requestIDs(reqs []*Request) []string {
	ids := make([]string, len(reqs))
	for i, r := range reqs {
		ids[i] = r.ID
	}
	return ids
}

func sliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestFCFSScheduler_PreservesOrder(t *testing.T) {
	// FCFS is a no-op: order unchanged
	sched := &FCFSScheduler{}
	reqs := []*Request{
		{ID: "c", ArrivalTime: 300, Priority: 1.0},
		{ID: "a", ArrivalTime: 100, Priority: 3.0},
		{ID: "b", ArrivalTime: 200, Priority: 2.0},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"c", "a", "b"}
	if !sliceEqual(got, want) {
		t.Errorf("FCFSScheduler: got %v, want %v", got, want)
	}
}

func TestPriorityFCFSScheduler_SortsByPriorityDescending(t *testing.T) {
	// BC-3: higher priority first
	sched := &PriorityFCFSScheduler{}
	reqs := []*Request{
		{ID: "low", ArrivalTime: 100, Priority: 1.0},
		{ID: "high", ArrivalTime: 200, Priority: 3.0},
		{ID: "mid", ArrivalTime: 50, Priority: 2.0},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"high", "mid", "low"}
	if !sliceEqual(got, want) {
		t.Errorf("PriorityFCFS priority ordering: got %v, want %v", got, want)
	}
}

func TestPriorityFCFSScheduler_TieBreakByArrivalTime(t *testing.T) {
	// BC-3: same priority → earlier arrival first
	sched := &PriorityFCFSScheduler{}
	reqs := []*Request{
		{ID: "late", ArrivalTime: 300, Priority: 5.0},
		{ID: "early", ArrivalTime: 100, Priority: 5.0},
		{ID: "mid", ArrivalTime: 200, Priority: 5.0},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"early", "mid", "late"}
	if !sliceEqual(got, want) {
		t.Errorf("PriorityFCFS arrival tiebreak: got %v, want %v", got, want)
	}
}

func TestPriorityFCFSScheduler_TieBreakByID(t *testing.T) {
	// BC-8: same priority + same arrival → lexicographic ID
	sched := &PriorityFCFSScheduler{}
	reqs := []*Request{
		{ID: "charlie", ArrivalTime: 100, Priority: 5.0},
		{ID: "alpha", ArrivalTime: 100, Priority: 5.0},
		{ID: "bravo", ArrivalTime: 100, Priority: 5.0},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"alpha", "bravo", "charlie"}
	if !sliceEqual(got, want) {
		t.Errorf("PriorityFCFS ID tiebreak: got %v, want %v", got, want)
	}
}

func TestSJFScheduler_SortsByInputTokensAscending(t *testing.T) {
	// BC-4: shorter jobs first
	sched := &SJFScheduler{}
	reqs := []*Request{
		{ID: "long", ArrivalTime: 100, InputTokens: make([]int, 500)},
		{ID: "short", ArrivalTime: 200, InputTokens: make([]int, 50)},
		{ID: "medium", ArrivalTime: 50, InputTokens: make([]int, 200)},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"short", "medium", "long"}
	if !sliceEqual(got, want) {
		t.Errorf("SJF input token ordering: got %v, want %v", got, want)
	}
}

func TestSJFScheduler_TieBreakByArrivalTime(t *testing.T) {
	// BC-4: same length → earlier arrival first
	sched := &SJFScheduler{}
	reqs := []*Request{
		{ID: "late", ArrivalTime: 300, InputTokens: make([]int, 100)},
		{ID: "early", ArrivalTime: 100, InputTokens: make([]int, 100)},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"early", "late"}
	if !sliceEqual(got, want) {
		t.Errorf("SJF arrival tiebreak: got %v, want %v", got, want)
	}
}

func TestSJFScheduler_TieBreakByID(t *testing.T) {
	// BC-4 + BC-8: same length + same arrival → lexicographic ID
	sched := &SJFScheduler{}
	reqs := []*Request{
		{ID: "bravo", ArrivalTime: 100, InputTokens: make([]int, 100)},
		{ID: "alpha", ArrivalTime: 100, InputTokens: make([]int, 100)},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"alpha", "bravo"}
	if !sliceEqual(got, want) {
		t.Errorf("SJF ID tiebreak: got %v, want %v", got, want)
	}
}

func TestScheduler_AnyPolicy_PreservesAllRequests(t *testing.T) {
	// NC-2: sorting must not add/remove/duplicate requests
	schedulers := []struct {
		name  string
		sched InstanceScheduler
	}{
		{"fcfs", &FCFSScheduler{}},
		{"priority-fcfs", &PriorityFCFSScheduler{}},
		{"sjf", &SJFScheduler{}},
	}

	for _, tc := range schedulers {
		t.Run(tc.name, func(t *testing.T) {
			reqs := []*Request{
				{ID: "a", ArrivalTime: 100, Priority: 1.0, InputTokens: make([]int, 50)},
				{ID: "b", ArrivalTime: 200, Priority: 2.0, InputTokens: make([]int, 100)},
				{ID: "c", ArrivalTime: 300, Priority: 3.0, InputTokens: make([]int, 25)},
			}
			tc.sched.OrderQueue(reqs, 0)

			if len(reqs) != 3 {
				t.Fatalf("queue length changed: got %d, want 3", len(reqs))
			}
			seen := make(map[string]bool)
			for _, r := range reqs {
				if seen[r.ID] {
					t.Errorf("duplicate request %q", r.ID)
				}
				seen[r.ID] = true
			}
			for _, id := range []string{"a", "b", "c"} {
				if !seen[id] {
					t.Errorf("missing request %q", id)
				}
			}
		})
	}
}

func TestNewScheduler_ValidNames_ReturnsCorrectType(t *testing.T) {
	// EH-3: empty string returns FCFSScheduler
	s1 := NewScheduler("")
	if _, ok := s1.(*FCFSScheduler); !ok {
		t.Errorf("NewScheduler(\"\"): expected *FCFSScheduler, got %T", s1)
	}

	s2 := NewScheduler("fcfs")
	if _, ok := s2.(*FCFSScheduler); !ok {
		t.Errorf("NewScheduler(\"fcfs\"): expected *FCFSScheduler, got %T", s2)
	}

	s3 := NewScheduler("priority-fcfs")
	if _, ok := s3.(*PriorityFCFSScheduler); !ok {
		t.Errorf("NewScheduler(\"priority-fcfs\"): expected *PriorityFCFSScheduler, got %T", s3)
	}

	s4 := NewScheduler("sjf")
	if _, ok := s4.(*SJFScheduler); !ok {
		t.Errorf("NewScheduler(\"sjf\"): expected *SJFScheduler, got %T", s4)
	}
}

func TestNewScheduler_UnknownName_Panics(t *testing.T) {
	// EH-2: unknown name panics
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("NewScheduler(\"unknown\"): expected panic, got nil")
		}
	}()
	NewScheduler("unknown")
}

func TestSimulator_PriorityFCFS_SchedulesHighPriorityFirst(t *testing.T) {
	// BC-2 + BC-5: SLO-based priority assigns higher priority to older requests;
	// priority-fcfs scheduler should schedule the older (higher-priority) request first.
	// Uses MaxRunningReqs=1 to force sequential scheduling so step index proves ordering.
	cfg := SimConfig{
		Horizon:            10000000,
		Seed:               42,
		TotalKVBlocks:      1000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     1,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		PriorityPolicy:     "slo-based",
		Scheduler:          "priority-fcfs",
	}
	s := NewSimulator(cfg)

	// reqNewer arrives later (lower age → lower priority from SLO-based policy)
	// Inject it first so FCFS would schedule it first — but priority should override.
	reqNewer := &Request{
		ID:           "req_newer",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 5),
		ArrivalTime:  500000,
		State:        StateQueued,
	}
	// reqOlder arrives earlier (higher age → higher priority)
	reqOlder := &Request{
		ID:           "req_older",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 5),
		ArrivalTime:  0,
		State:        StateQueued,
	}

	// Inject newer first, then older — priority-fcfs should reorder
	s.InjectArrival(reqNewer)
	s.InjectArrival(reqOlder)

	s.Run()

	// Both should complete
	if s.Metrics.CompletedRequests != 2 {
		t.Fatalf("completed: got %d, want 2", s.Metrics.CompletedRequests)
	}

	// Older request (higher priority) should have been scheduled first
	if reqOlder.ScheduledStepIdx > reqNewer.ScheduledStepIdx {
		t.Errorf("priority inversion: older scheduled at step %d, newer at step %d",
			reqOlder.ScheduledStepIdx, reqNewer.ScheduledStepIdx)
	}
}

func TestSimulator_DefaultConfig_MatchesFCFS(t *testing.T) {
	// BC-1: default config (empty strings) uses ConstantPriority + FCFSScheduler
	cfg := SimConfig{
		Horizon:            1000000,
		Seed:               42,
		TotalKVBlocks:      1000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		// PriorityPolicy and Scheduler left empty (defaults)
	}
	s := NewSimulator(cfg)

	// Verify correct types were created
	if _, ok := s.priorityPolicy.(*ConstantPriority); !ok {
		t.Errorf("default priorityPolicy: got %T, want *ConstantPriority", s.priorityPolicy)
	}
	if _, ok := s.scheduler.(*FCFSScheduler); !ok {
		t.Errorf("default scheduler: got %T, want *FCFSScheduler", s.scheduler)
	}
}

func TestScheduler_EmptyQueue_NoOp(t *testing.T) {
	// Edge case: empty queue must not panic or modify slice
	schedulers := []struct {
		name  string
		sched InstanceScheduler
	}{
		{"fcfs", &FCFSScheduler{}},
		{"priority-fcfs", &PriorityFCFSScheduler{}},
		{"sjf", &SJFScheduler{}},
	}
	for _, tc := range schedulers {
		t.Run(tc.name, func(t *testing.T) {
			reqs := []*Request{}
			tc.sched.OrderQueue(reqs, 0)
			if len(reqs) != 0 {
				t.Errorf("empty queue modified: got len %d, want 0", len(reqs))
			}
		})
	}
}

func TestSimulator_SJF_SchedulesShortJobFirst(t *testing.T) {
	// BC-4 + BC-5: SJF should schedule shorter input request first.
	// Uses zero alpha delay so both requests queue simultaneously at tick 0,
	// and MaxRunningReqs=256 so both enter the batch in the same step.
	// SJF sorts short before long; short request gets lower E2E latency
	// because it has fewer prefill tokens to process.
	cfg := SimConfig{
		Horizon:            10000000,
		Seed:               42,
		TotalKVBlocks:      1000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{0, 0, 100}, // zero queueing delay so both queue at arrival time
		Scheduler:          "sjf",
	}
	s := NewSimulator(cfg)

	// Long request injected first — SJF should move it behind short
	reqLong := &Request{
		ID:           "req_long",
		InputTokens:  make([]int, 200),
		OutputTokens: make([]int, 2),
		ArrivalTime:  0,
		State:        StateQueued,
	}
	// Short request injected second — SJF should move it to front
	reqShort := &Request{
		ID:           "req_short",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 2),
		ArrivalTime:  0,
		State:        StateQueued,
	}

	s.InjectArrival(reqLong)
	s.InjectArrival(reqShort)

	s.Run()

	if s.Metrics.CompletedRequests != 2 {
		t.Fatalf("completed: got %d, want 2", s.Metrics.CompletedRequests)
	}

	// Both scheduled in same step (both queue before first step fires).
	// SJF puts short first in batch → short gets processed first → lower E2E.
	shortE2E := s.Metrics.RequestE2Es["req_short"]
	longE2E := s.Metrics.RequestE2Es["req_long"]
	if shortE2E > longE2E {
		t.Errorf("SJF violation: short E2E=%f > long E2E=%f (short should finish first)",
			shortE2E, longE2E)
	}
}

func TestSimulator_SLOBased_PriorityFCFS_OlderRequestFirst(t *testing.T) {
	// BC-7 + BC-5: SLO-based priority with priority-fcfs scheduler
	// Older requests should get higher priority and schedule first
	cfg := SimConfig{
		Horizon:            10000000,
		Seed:               42,
		TotalKVBlocks:      1000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     1, // only 1 slot: forces sequential scheduling
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		PriorityPolicy:     "slo-based",
		Scheduler:          "priority-fcfs",
	}
	s := NewSimulator(cfg)

	// Newer request injected first (arrives at t=500000)
	reqNew := &Request{
		ID:           "req_new",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 2),
		ArrivalTime:  500000,
		State:        StateQueued,
	}
	// Older request injected second (arrives at t=0)
	reqOld := &Request{
		ID:           "req_old",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 2),
		ArrivalTime:  0,
		State:        StateQueued,
	}

	s.InjectArrival(reqNew)
	s.InjectArrival(reqOld)

	s.Run()

	if s.Metrics.CompletedRequests != 2 {
		t.Fatalf("completed: got %d, want 2", s.Metrics.CompletedRequests)
	}

	// Older request should be scheduled first (higher priority from age)
	if reqOld.ScheduledStepIdx > reqNew.ScheduledStepIdx {
		t.Errorf("SLO-based priority violation: old scheduled at step %d, new at step %d",
			reqOld.ScheduledStepIdx, reqNew.ScheduledStepIdx)
	}
}

// TestReversePriority_LowestPriorityFirst verifies BC-7.
func TestReversePriority_LowestPriorityFirst(t *testing.T) {
	scheduler := NewScheduler("reverse-priority")
	reqs := []*Request{
		{ID: "high", Priority: 10.0, ArrivalTime: 100},
		{ID: "low", Priority: 1.0, ArrivalTime: 200},
		{ID: "mid", Priority: 5.0, ArrivalTime: 150},
	}

	scheduler.OrderQueue(reqs, 1_000_000)

	// THEN lowest priority should be first (reverse of PriorityFCFSScheduler)
	if reqs[0].ID != "low" {
		t.Errorf("expected 'low' first, got %q (priority=%f)", reqs[0].ID, reqs[0].Priority)
	}
	if reqs[1].ID != "mid" {
		t.Errorf("expected 'mid' second, got %q (priority=%f)", reqs[1].ID, reqs[1].Priority)
	}
	if reqs[2].ID != "high" {
		t.Errorf("expected 'high' last, got %q (priority=%f)", reqs[2].ID, reqs[2].Priority)
	}
}
