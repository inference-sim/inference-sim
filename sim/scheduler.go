package sim

import (
	"fmt"
	"sort"
)

// InstanceScheduler reorders the wait queue before batch formation.
// Called each step to determine which requests should be considered first.
// Implementations sort the slice in-place using sort.SliceStable for determinism.
type InstanceScheduler interface {
	OrderQueue(requests []*Request, clock int64)
}

// FCFSScheduler preserves First-Come-First-Served order (no-op).
// This is the default scheduler matching existing BLIS behavior.
type FCFSScheduler struct{}

func (f *FCFSScheduler) OrderQueue(_ []*Request, _ int64) {
	// No-op: FIFO order preserved from enqueue order
}

// PriorityFCFSScheduler sorts requests by priority (descending),
// then by arrival time (ascending), then by ID (ascending) for determinism.
type PriorityFCFSScheduler struct{}

func (p *PriorityFCFSScheduler) OrderQueue(reqs []*Request, _ int64) {
	// Float != comparison is safe here: current policies produce exact arithmetic
	// (constant return or int-derived multiply+add). Revisit if policies use division/log.
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].Priority != reqs[j].Priority {
			return reqs[i].Priority > reqs[j].Priority
		}
		if reqs[i].ArrivalTime != reqs[j].ArrivalTime {
			return reqs[i].ArrivalTime < reqs[j].ArrivalTime
		}
		return reqs[i].ID < reqs[j].ID
	})
}

// SJFScheduler sorts requests by input token count (ascending, shortest first),
// then by arrival time (ascending), then by ID (ascending) for determinism.
// Warning: SJF can cause starvation for long requests under sustained load.
type SJFScheduler struct{}

func (s *SJFScheduler) OrderQueue(reqs []*Request, _ int64) {
	sort.SliceStable(reqs, func(i, j int) bool {
		li, lj := len(reqs[i].InputTokens), len(reqs[j].InputTokens)
		if li != lj {
			return li < lj
		}
		if reqs[i].ArrivalTime != reqs[j].ArrivalTime {
			return reqs[i].ArrivalTime < reqs[j].ArrivalTime
		}
		return reqs[i].ID < reqs[j].ID
	})
}

// NewScheduler creates an InstanceScheduler by name.
// Valid names: "fcfs" (default), "priority-fcfs", "sjf".
// Empty string defaults to FCFSScheduler (for CLI flag default compatibility).
// Panics on unrecognized names.
func NewScheduler(name string) InstanceScheduler {
	if !IsValidScheduler(name) {
		panic(fmt.Sprintf("unknown scheduler %q", name))
	}
	switch name {
	case "", "fcfs":
		return &FCFSScheduler{}
	case "priority-fcfs":
		return &PriorityFCFSScheduler{}
	case "sjf":
		return &SJFScheduler{}
	default:
		panic(fmt.Sprintf("unhandled scheduler %q", name))
	}
}
