package sim

import (
	"testing"
)

// TestAlwaysAdmit_AdmitsAll verifies AlwaysAdmit always returns (true, "").
func TestAlwaysAdmit_AdmitsAll(t *testing.T) {
	policy := &AlwaysAdmit{}

	tests := []struct {
		name  string
		req   *Request
		clock int64
	}{
		{name: "empty request", req: &Request{ID: "r0", InputTokens: []int{}}, clock: 0},
		{name: "small request", req: &Request{ID: "r1", InputTokens: make([]int, 10)}, clock: 1000},
		{name: "large request", req: &Request{ID: "r2", InputTokens: make([]int, 10000)}, clock: 5_000_000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := &RouterState{Clock: tt.clock}
			admitted, reason := policy.Admit(tt.req, state)
			if !admitted {
				t.Errorf("expected admitted=true, got false")
			}
			if reason != "" {
				t.Errorf("expected empty reason, got %q", reason)
			}
		})
	}
}

// TestTokenBucket_AdmitAndReject verifies TokenBucket admits/rejects based on token availability.
func TestTokenBucket_AdmitAndReject(t *testing.T) {
	t.Run("admits when tokens available", func(t *testing.T) {
		tb := NewTokenBucket(100, 10)
		req := &Request{ID: "r0", InputTokens: make([]int, 50)}
		admitted, reason := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("expected admission with sufficient tokens")
		}
		if reason != "" {
			t.Errorf("expected empty reason, got %q", reason)
		}
	})

	t.Run("rejects when tokens exhausted", func(t *testing.T) {
		// Use tiny refill rate so tokens don't replenish within the test window
		tb := NewTokenBucket(10, 0.001)
		req := &Request{ID: "r0", InputTokens: make([]int, 10)}

		admitted, _ := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("first request should be admitted")
		}

		admitted, reason := tb.Admit(req, &RouterState{Clock: 0})
		if admitted {
			t.Fatal("expected rejection with exhausted tokens")
		}
		if reason != "insufficient tokens" {
			t.Errorf("expected reason %q, got %q", "insufficient tokens", reason)
		}
	})

	t.Run("refill restores tokens over time", func(t *testing.T) {
		tb := NewTokenBucket(100, 1_000_000)
		req := &Request{ID: "r0", InputTokens: make([]int, 100)}

		admitted, _ := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("first request should be admitted")
		}

		// At t=50us, refill = 50 tokens. Cost=100, should reject.
		admitted, _ = tb.Admit(req, &RouterState{Clock: 50})
		if admitted {
			t.Fatal("expected rejection: only 50 tokens refilled, need 100")
		}

		// At t=150us, refill = 100 more tokens, capped at capacity=100. Should admit.
		admitted, reason := tb.Admit(req, &RouterState{Clock: 150})
		if !admitted {
			t.Fatalf("expected admission after refill, reason: %s", reason)
		}
	})

	t.Run("capacity caps refill", func(t *testing.T) {
		tb := NewTokenBucket(10, 1_000_000)
		req := &Request{ID: "r0", InputTokens: make([]int, 5)}

		admitted, _ := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("should admit")
		}

		// At t=1s, refill huge but capped at capacity=10
		admitted, _ = tb.Admit(req, &RouterState{Clock: 1_000_000})
		if !admitted {
			t.Fatal("should admit after refill")
		}

		admitted, _ = tb.Admit(req, &RouterState{Clock: 1_000_000})
		if !admitted {
			t.Fatal("should admit: 5 tokens remain")
		}

		admitted, _ = tb.Admit(req, &RouterState{Clock: 1_000_000})
		if admitted {
			t.Fatal("should reject: 0 tokens remain, no time elapsed for refill")
		}
	})

	t.Run("zero-cost request always admitted", func(t *testing.T) {
		// Even with minimal capacity, a zero-cost request (0 input tokens) is admitted
		tb := NewTokenBucket(1, 1)
		req := &Request{ID: "r0", InputTokens: []int{}}

		admitted, _ := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("zero-cost request should always be admitted")
		}
	})

	t.Run("panics on zero capacity", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for capacity=0, got none")
			}
		}()
		NewTokenBucket(0, 1)
	})

	t.Run("panics on zero refill rate", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for refillRate=0, got none")
			}
		}()
		NewTokenBucket(10, 0)
	})
}

// TestNewAdmissionPolicy_ValidNames verifies the factory produces correct behavioral policies.
func TestNewAdmissionPolicy_ValidNames(t *testing.T) {
	req := &Request{ID: "r0", InputTokens: make([]int, 10)}
	state := &RouterState{Clock: 0}

	t.Run("always-admit admits all", func(t *testing.T) {
		p := NewAdmissionPolicy("always-admit", 0, 0)
		admitted, _ := p.Admit(req, state)
		if !admitted {
			t.Error("always-admit policy should admit")
		}
	})

	t.Run("empty string defaults to always-admit", func(t *testing.T) {
		p := NewAdmissionPolicy("", 0, 0)
		admitted, _ := p.Admit(req, state)
		if !admitted {
			t.Error("default policy should admit")
		}
	})

	t.Run("token-bucket rate-limits", func(t *testing.T) {
		p := NewAdmissionPolicy("token-bucket", 5, 0.001) // capacity=5, negligible refill
		admitted, _ := p.Admit(req, state)             // cost=10 > capacity=5
		if admitted {
			t.Error("token-bucket should reject request exceeding capacity")
		}
	})
}

// TestNewAdmissionPolicy_InvalidName_Panics verifies unknown names cause a panic.
func TestNewAdmissionPolicy_InvalidName_Panics(t *testing.T) {
	tests := []struct {
		name       string
		policyName string
	}{
		{"unknown name", "round-robin"},
		{"typo", "always_admit"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Errorf("expected panic for policy name %q, got none", tt.policyName)
				}
			}()
			NewAdmissionPolicy(tt.policyName, 0, 0)
		})
	}
}

// TestRejectAll_RejectsAll verifies BC-4.
func TestRejectAll_RejectsAll(t *testing.T) {
	policy := NewAdmissionPolicy("reject-all", 0, 0)
	tests := []struct {
		name string
		req  *Request
	}{
		{name: "empty request", req: &Request{ID: "r0", InputTokens: []int{}}},
		{name: "normal request", req: &Request{ID: "r1", InputTokens: make([]int, 100)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			admitted, reason := policy.Admit(tt.req, &RouterState{Clock: 1000})
			if admitted {
				t.Error("expected reject-all to reject, but it admitted")
			}
			if reason != "reject-all" {
				t.Errorf("expected reason %q, got %q", "reject-all", reason)
			}
		})
	}
}

// stubTracker is a test double implementing TenantBudgetTracker.
type stubTracker struct{ overBudget bool }

func (s *stubTracker) IsOverBudget(string) bool { return s.overBudget }

func TestTenantBudgetAdmission_DelegatesToInnerPolicy(t *testing.T) {
	policy := NewTenantBudgetAdmission(&RejectAll{}, &stubTracker{overBudget: false}, DefaultSLOPriorityMap())
	admitted, reason := policy.Admit(&Request{ID: "r1", TenantID: "t1", SLOClass: "standard"}, &RouterState{})
	if admitted {
		t.Error("inner policy rejection should stand")
	}
	if reason == "" {
		t.Error("should have rejection reason from inner policy")
	}
}

func TestTenantBudgetAdmission_SheddableOverBudgetRejected(t *testing.T) {
	policy := NewTenantBudgetAdmission(&AlwaysAdmit{}, &stubTracker{overBudget: true}, DefaultSLOPriorityMap())
	admitted, reason := policy.Admit(&Request{ID: "r1", TenantID: "t1", SLOClass: "batch"}, &RouterState{})
	if admitted {
		t.Error("sheddable over-budget request should be rejected")
	}
	if reason != "tenant-budget-shed" {
		t.Errorf("expected reason 'tenant-budget-shed', got %q", reason)
	}
}

func TestTenantBudgetAdmission_NonSheddableOverBudgetAdmitted(t *testing.T) {
	policy := NewTenantBudgetAdmission(&AlwaysAdmit{}, &stubTracker{overBudget: true}, DefaultSLOPriorityMap())
	admitted, _ := policy.Admit(&Request{ID: "r1", TenantID: "t1", SLOClass: "standard"}, &RouterState{})
	if !admitted {
		t.Error("non-sheddable request should be admitted even when over budget")
	}
}

func TestTenantBudgetAdmission_SheddableUnderBudgetAdmitted(t *testing.T) {
	policy := NewTenantBudgetAdmission(&AlwaysAdmit{}, &stubTracker{overBudget: false}, DefaultSLOPriorityMap())
	admitted, _ := policy.Admit(&Request{ID: "r1", TenantID: "t1", SLOClass: "batch"}, &RouterState{})
	if !admitted {
		t.Error("sheddable under-budget request should be admitted")
	}
}

func TestNewTenantBudgetAdmission_NilInner_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil inner policy")
		}
	}()
	NewTenantBudgetAdmission(nil, &stubTracker{}, DefaultSLOPriorityMap())
}

func TestNewTenantBudgetAdmission_NilTracker_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil tracker")
		}
	}()
	NewTenantBudgetAdmission(&AlwaysAdmit{}, nil, DefaultSLOPriorityMap())
}

func TestSLOPriorityMap_InvertForVLLM_DefaultPriorities(t *testing.T) {
	m := DefaultSLOPriorityMap()
	tests := []struct {
		class    string
		expected int
	}{
		{"critical", 0},     // 4 - 4 = 0 (most urgent in vLLM)
		{"standard", 1},     // 4 - 3 = 1
		{"batch", 5},        // 4 - (-1) = 5
		{"sheddable", 6},    // 4 - (-2) = 6
		{"background", 7},   // 4 - (-3) = 7 (least urgent in vLLM)
		{"unknown", 1},      // 4 - 3 (defaultPri) = 1
		{"", 1},             // 4 - 3 (defaultPri) = 1
	}
	for _, tt := range tests {
		t.Run(tt.class, func(t *testing.T) {
			got := m.InvertForVLLM(tt.class)
			if got != tt.expected {
				t.Errorf("InvertForVLLM(%q) = %d, want %d", tt.class, got, tt.expected)
			}
		})
	}
}

func TestSLOPriorityMap_InvertForVLLM_CustomOverrides(t *testing.T) {
	// Override: batch=0 (non-sheddable), critical=10 (ultra-high)
	m := NewSLOPriorityMap(map[string]int{
		"batch":    0,
		"critical": 10,
	})
	tests := []struct {
		class    string
		expected int
	}{
		{"critical", 0},    // 10 - 10 = 0 (max is now 10)
		{"standard", 7},    // 10 - 3 = 7
		{"batch", 10},      // 10 - 0 = 10
		{"sheddable", 12},  // 10 - (-2) = 12
		{"background", 13}, // 10 - (-3) = 13
	}
	for _, tt := range tests {
		t.Run(tt.class, func(t *testing.T) {
			got := m.InvertForVLLM(tt.class)
			if got != tt.expected {
				t.Errorf("InvertForVLLM(%q) = %d, want %d", tt.class, got, tt.expected)
			}
		})
	}
}

func TestSLOPriorityMap_InvertForVLLM_PreservesUrgencyOrder(t *testing.T) {
	// GIVEN default priorities
	m := DefaultSLOPriorityMap()

	// WHEN inverting all classes
	critical := m.InvertForVLLM("critical")
	standard := m.InvertForVLLM("standard")
	batch := m.InvertForVLLM("batch")
	sheddable := m.InvertForVLLM("sheddable")
	background := m.InvertForVLLM("background")

	// THEN vLLM priorities preserve urgency order (lower = more urgent)
	if !(critical < standard && standard < batch && batch < sheddable && sheddable < background) {
		t.Errorf("InvertForVLLM broke urgency order: critical=%d, standard=%d, batch=%d, sheddable=%d, background=%d",
			critical, standard, batch, sheddable, background)
	}
}
