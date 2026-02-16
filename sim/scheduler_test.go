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
