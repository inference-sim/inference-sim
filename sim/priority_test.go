package sim

import (
	"testing"
)

func TestConstantPriority_ReturnsFixedScore(t *testing.T) {
	// BC-6: ConstantPriority returns configured score regardless of request/clock
	policy := &ConstantPriority{Score: 5.0}
	req := &Request{ID: "r1", ArrivalTime: 100, InputTokens: make([]int, 50)}

	got := policy.Compute(req, 1000)
	if got != 5.0 {
		t.Errorf("ConstantPriority.Compute: got %f, want 5.0", got)
	}

	// Different request and clock — same result
	req2 := &Request{ID: "r2", ArrivalTime: 500, InputTokens: make([]int, 200)}
	got2 := policy.Compute(req2, 9999)
	if got2 != 5.0 {
		t.Errorf("ConstantPriority.Compute (different req): got %f, want 5.0", got2)
	}
}

func TestConstantPriority_DefaultZero(t *testing.T) {
	// BC-6: Zero-value ConstantPriority returns 0.0
	policy := &ConstantPriority{}
	req := &Request{ID: "r1", ArrivalTime: 0}
	got := policy.Compute(req, 0)
	if got != 0.0 {
		t.Errorf("ConstantPriority (zero): got %f, want 0.0", got)
	}
}

func TestSLOBasedPriority_OlderRequestGetsHigherPriority(t *testing.T) {
	// BC-7: With AgeWeight > 0, older requests get higher priority
	policy := &SLOBasedPriority{BaseScore: 0.0, AgeWeight: 1e-6}
	clock := int64(2000000) // 2 seconds in ticks

	older := &Request{ID: "old", ArrivalTime: 0}      // age = 2s
	newer := &Request{ID: "new", ArrivalTime: 1000000} // age = 1s

	pOlder := policy.Compute(older, clock)
	pNewer := policy.Compute(newer, clock)

	if pOlder <= pNewer {
		t.Errorf("SLOBasedPriority: older=%f should be > newer=%f", pOlder, pNewer)
	}
}

func TestSLOBasedPriority_FormulaCorrectness(t *testing.T) {
	// BC-7: priority = BaseScore + AgeWeight * float64(clock - ArrivalTime)
	policy := &SLOBasedPriority{BaseScore: 1.0, AgeWeight: 0.5}
	req := &Request{ID: "r1", ArrivalTime: 100}
	clock := int64(300)

	got := policy.Compute(req, clock)
	want := 1.0 + 0.5*float64(300-100) // 1.0 + 100.0 = 101.0
	if got != want {
		t.Errorf("SLOBasedPriority formula: got %f, want %f", got, want)
	}
}

func TestNewPriorityPolicy_ValidNames_ReturnsCorrectType(t *testing.T) {
	// EH-3: empty string returns default (ConstantPriority)
	p1 := NewPriorityPolicy("")
	if _, ok := p1.(*ConstantPriority); !ok {
		t.Errorf("NewPriorityPolicy(\"\"): expected *ConstantPriority, got %T", p1)
	}

	p2 := NewPriorityPolicy("constant")
	if _, ok := p2.(*ConstantPriority); !ok {
		t.Errorf("NewPriorityPolicy(\"constant\"): expected *ConstantPriority, got %T", p2)
	}

	p3 := NewPriorityPolicy("slo-based")
	if _, ok := p3.(*SLOBasedPriority); !ok {
		t.Errorf("NewPriorityPolicy(\"slo-based\"): expected *SLOBasedPriority, got %T", p3)
	}
}

func TestNewPriorityPolicy_UnknownName_Panics(t *testing.T) {
	// EH-1: unknown name panics
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("NewPriorityPolicy(\"unknown\"): expected panic, got nil")
		}
	}()
	NewPriorityPolicy("unknown")
}

func TestPriorityPolicy_Compute_NoSideEffects(t *testing.T) {
	// NC-3: Compute must not modify the request
	policies := []struct {
		name   string
		policy PriorityPolicy
	}{
		{"constant", &ConstantPriority{Score: 5.0}},
		{"slo-based", &SLOBasedPriority{BaseScore: 1.0, AgeWeight: 1e-6}},
	}
	for _, tc := range policies {
		t.Run(tc.name, func(t *testing.T) {
			req := &Request{
				ID: "r1", ArrivalTime: 100, InputTokens: make([]int, 50),
				Priority: 0.0, State: StateQueued,
			}
			tc.policy.Compute(req, 1000)
			if req.Priority != 0.0 {
				t.Errorf("Compute modified req.Priority: got %f, want 0.0", req.Priority)
			}
			if req.State != StateQueued {
				t.Errorf("Compute modified req.State: got %q, want %q", req.State, StateQueued)
			}
			if req.ID != "r1" {
				t.Errorf("Compute modified req.ID: got %q, want %q", req.ID, "r1")
			}
		})
	}
}

// TestInvertedSLO_OlderRequestsGetLowerPriority verifies BC-5.
func TestInvertedSLO_OlderRequestsGetLowerPriority(t *testing.T) {
	policy := NewPriorityPolicy("inverted-slo")

	oldReq := &Request{ID: "old", ArrivalTime: 0}
	newReq := &Request{ID: "new", ArrivalTime: 900_000}
	clock := int64(1_000_000)

	oldPriority := policy.Compute(oldReq, clock)
	newPriority := policy.Compute(newReq, clock)

	// THEN older request MUST have lower priority than newer request
	if oldPriority >= newPriority {
		t.Errorf("expected older request priority (%f) < newer request priority (%f)", oldPriority, newPriority)
	}
}
