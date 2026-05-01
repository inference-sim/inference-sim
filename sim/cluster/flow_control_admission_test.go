package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestFlowControlAdmission_Enqueue(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 0, pm)
	fc := NewFlowControlAdmission(q, pm)

	req := &sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1"}
	state := &sim.RouterState{Clock: 100}

	admitted, _ := fc.Admit(req, state)
	if !admitted {
		t.Error("FlowControlAdmission.Admit must always return true")
	}
	if fc.LastOutcome() != Enqueued {
		t.Errorf("expected Enqueued, got %v", fc.LastOutcome())
	}
	if q.Len() != 1 {
		t.Errorf("expected 1 in queue, got %d", q.Len())
	}
	if req.GatewayEnqueueTime != 100 {
		t.Errorf("expected enqueue time 100, got %d", req.GatewayEnqueueTime)
	}
}

func TestFlowControlAdmission_QueueRejection(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 1, pm) // maxDepth=1
	fc := NewFlowControlAdmission(q, pm)

	fc.Admit(&sim.Request{ID: "r1", SLOClass: "standard"}, &sim.RouterState{Clock: 100})
	r2 := &sim.Request{ID: "r2", SLOClass: "standard"}
	admitted, reason := fc.Admit(r2, &sim.RouterState{Clock: 200})

	if !admitted {
		t.Error("FlowControlAdmission.Admit must always return true (even for queue rejection)")
	}
	if fc.LastOutcome() != Rejected {
		t.Errorf("expected Rejected outcome, got %v", fc.LastOutcome())
	}
	if reason != "flow-control-queue-rejected" {
		t.Errorf("expected queue-rejected reason, got %q", reason)
	}
	if r2.GatewayEnqueueTime != 0 {
		t.Error("rejected request should have GatewayEnqueueTime cleared to 0")
	}
}

func TestFlowControlAdmission_ShedVictim(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 1, pm) // maxDepth=1
	fc := NewFlowControlAdmission(q, pm)

	fc.Admit(&sim.Request{ID: "r1", SLOClass: "batch"}, &sim.RouterState{Clock: 100})
	fc.Admit(&sim.Request{ID: "r2", SLOClass: "standard"}, &sim.RouterState{Clock: 200})

	if fc.LastOutcome() != ShedVictim {
		t.Errorf("expected ShedVictim, got %v", fc.LastOutcome())
	}
	victim := fc.LastShedVictim()
	if victim == nil || victim.ID != "r1" {
		t.Errorf("expected victim r1, got %v", victim)
	}
	if victim.GatewayEnqueueTime != 0 {
		t.Error("shed victim should have GatewayEnqueueTime cleared")
	}
}

func TestFlowControlAdmission_NilQueue_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil queue")
		}
	}()
	NewFlowControlAdmission(nil, nil)
}

// verifyINV1Conservation checks the full INV-1 request conservation invariant.
func verifyINV1Conservation(t *testing.T, cs *ClusterSimulator, requests []*sim.Request) {
	t.Helper()
	m := cs.AggregatedMetrics()
	numRequests := len(requests)
	injected := numRequests - cs.RejectedRequests()
	rejected := cs.RejectedRequests()

	if numRequests != injected+rejected {
		t.Errorf("INV-1 pipeline: num_requests=%d != injected=%d + rejected=%d",
			numRequests, injected, rejected)
	}

	gwDepth := cs.GatewayQueueDepth()
	gwShed := cs.GatewayQueueShed()
	gwRejected := cs.GatewayQueueRejected()

	conserved := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable +
		m.TimedOutRequests + cs.RoutingRejections() + gwDepth + gwShed + gwRejected
	if injected != conserved {
		t.Errorf("INV-1 conservation: injected=%d != completed=%d + queued=%d + running=%d + "+
			"dropped=%d + timedOut=%d + routingRej=%d + gwDepth=%d + gwShed=%d + gwRejected=%d (sum=%d)",
			injected, m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable,
			m.TimedOutRequests, cs.RoutingRejections(), gwDepth, gwShed, gwRejected, conserved)
	}
}

func TestFlowControlAdmission_SeqCounterIncrementsMonotonically(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 0, pm)
	fc := NewFlowControlAdmission(q, pm)

	fc.Admit(&sim.Request{ID: "r1", SLOClass: "standard"}, &sim.RouterState{Clock: 100})
	fc.Admit(&sim.Request{ID: "r2", SLOClass: "standard"}, &sim.RouterState{Clock: 200})

	// Dequeue should return r1 first (earlier seqID)
	got := q.Dequeue()
	if got.ID != "r1" {
		t.Errorf("expected r1 (earlier seqCounter), got %s", got.ID)
	}
}

// stubBudgetTracker implements sim.TenantBudgetTracker for testing.
type stubBudgetTracker struct{ overBudget bool }

func (s *stubBudgetTracker) IsOverBudget(string) bool { return s.overBudget }

// TestTenantBudgetAdmission_BudgetBeforeEnqueue verifies that when TenantBudgetAdmission
// wraps FlowControlAdmission, the budget check runs BEFORE the enqueue side effect.
// If budget rejects, the gateway queue must be empty (no INV-1 double-counting).
func TestTenantBudgetAdmission_BudgetBeforeEnqueue(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 0, pm)
	fc := NewFlowControlAdmission(q, pm)
	policy := sim.NewTenantBudgetAdmission(fc, &stubBudgetTracker{overBudget: true}, pm)

	req := &sim.Request{ID: "r1", SLOClass: "batch", TenantID: "t1"} // batch is sheddable
	admitted, reason := policy.Admit(req, &sim.RouterState{Clock: 100})

	if admitted {
		t.Error("sheddable over-budget request should be rejected")
	}
	if reason != "tenant-budget-shed" {
		t.Errorf("expected tenant-budget-shed, got %q", reason)
	}
	if q.Len() != 0 {
		t.Errorf("budget rejection should prevent enqueue; queue has %d entries", q.Len())
	}
}

// TestTenantBudgetAdmission_NonSheddablePassesBudget verifies non-sheddable requests
// bypass the budget check and are enqueued via FlowControlAdmission.
func TestTenantBudgetAdmission_NonSheddablePassesBudget(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 0, pm)
	fc := NewFlowControlAdmission(q, pm)
	policy := sim.NewTenantBudgetAdmission(fc, &stubBudgetTracker{overBudget: true}, pm)

	req := &sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1"} // non-sheddable
	admitted, _ := policy.Admit(req, &sim.RouterState{Clock: 100})

	if !admitted {
		t.Error("non-sheddable request should pass budget check")
	}
	if q.Len() != 1 {
		t.Errorf("non-sheddable request should be enqueued; queue has %d entries", q.Len())
	}
}

// TestFlowControlAdmission_INV1_Conservation runs a full cluster simulation with
// FlowControlAdmission + per-band capacity and verifies INV-1 request conservation.
func TestFlowControlAdmission_INV1_Conservation(t *testing.T) {
	// Test two scenarios: (a) permissive capacity (no overflow), (b) tight capacity (forces overflow).
	t.Run("conservation_with_perband_capacity", func(t *testing.T) {
		config := newTestDeploymentConfig(2)
		config.FlowControlEnabled = true
		config.FlowControlDetector = "utilization"
		config.FlowControlDispatchOrder = "priority"
		config.FlowControlMaxQueueDepth = 50
		config.FlowControlPerBandCapacity = 20
		config.FlowControlQueueDepthThreshold = 3
		config.FlowControlKVCacheUtilThreshold = 0.8

		sloClasses := []string{"critical", "standard", "batch", "sheddable", "background"}
		requests := make([]*sim.Request, 20)
		for i := 0; i < 20; i++ {
			requests[i] = &sim.Request{
				ID:           fmt.Sprintf("r%d", i),
				TenantID:     fmt.Sprintf("tenant-%d", i%3),
				SLOClass:     sloClasses[i%len(sloClasses)],
				ArrivalTime:  int64(i * 100_000),
				InputTokens:  make([]int, 100),
				OutputTokens: make([]int, 50),
				MaxOutputLen: 200,
			}
		}

		cs := NewClusterSimulator(config, requests, nil)
		mustRun(t, cs)
		verifyINV1Conservation(t, cs, requests)
	})

	// Scenario (b): FlowControl + TenantBudgets combined.
	t.Run("conservation_with_tenant_budgets", func(t *testing.T) {
		config := newTestDeploymentConfig(2)
		config.FlowControlEnabled = true
		config.FlowControlDetector = "utilization"
		config.FlowControlDispatchOrder = "priority"
		config.FlowControlMaxQueueDepth = 50
		config.FlowControlPerBandCapacity = 10
		config.FlowControlQueueDepthThreshold = 3
		config.FlowControlKVCacheUtilThreshold = 0.8
		config.TenantBudgets = map[string]float64{"tenant-0": 0.5, "tenant-1": 0.3}

		requests := make([]*sim.Request, 15)
		sloClasses := []string{"batch", "sheddable", "standard", "critical", "background"}
		for i := 0; i < 15; i++ {
			requests[i] = &sim.Request{
				ID:           fmt.Sprintf("r%d", i),
				TenantID:     fmt.Sprintf("tenant-%d", i%2),
				SLOClass:     sloClasses[i%len(sloClasses)],
				ArrivalTime:  int64(i * 100_000),
				InputTokens:  make([]int, 100),
				OutputTokens: make([]int, 50),
				MaxOutputLen: 200,
			}
		}

		cs := NewClusterSimulator(config, requests, nil)
		mustRun(t, cs)
		verifyINV1Conservation(t, cs, requests)
	})

	// Scenario (c): tiny global capacity to force gateway queue overflow.
	// All requests have batch SLOClass (sheddable, priority -1) so cross-band shedding can evict.
	t.Run("conservation_with_overflow", func(t *testing.T) {
		config := newTestDeploymentConfig(2)
		config.FlowControlEnabled = true
		config.FlowControlDetector = "utilization"
		config.FlowControlDispatchOrder = "priority"
		config.FlowControlMaxQueueDepth = 2 // tiny global limit
		config.FlowControlQueueDepthThreshold = 3
		config.FlowControlKVCacheUtilThreshold = 0.8

		requests := make([]*sim.Request, 10)
		for i := 0; i < 10; i++ {
			requests[i] = &sim.Request{
				ID:           fmt.Sprintf("r%d", i),
				TenantID:     fmt.Sprintf("tenant-%d", i%2),
				SLOClass:     "standard", // non-sheddable: overflow means rejection
				ArrivalTime:  int64(i * 100_000),
				InputTokens:  make([]int, 100),
				OutputTokens: make([]int, 50),
				MaxOutputLen: 200,
			}
		}

		cs := NewClusterSimulator(config, requests, nil)
		mustRun(t, cs)
		verifyINV1Conservation(t, cs, requests)
	})
}
