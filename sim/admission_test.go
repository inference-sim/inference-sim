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

// TestSLOGatedAdmission_ProtectedClassesAlwaysAdmitted verifies protected SLO classes
// are admitted regardless of queue depth.
func TestSLOGatedAdmission_ProtectedClassesAlwaysAdmitted(t *testing.T) {
	policy := NewSLOGatedAdmission([]string{"critical", "standard"}, 10)
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{QueueDepth: 100}}, // way over threshold
		Clock:     1000,
	}

	tests := []struct {
		name     string
		sloClass string
	}{
		{"critical always admitted", "critical"},
		{"standard always admitted", "standard"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{ID: "r0", SLOClass: tt.sloClass}
			admitted, _ := policy.Admit(req, state)
			if !admitted {
				t.Errorf("%s should always be admitted, even when queue depth exceeds threshold", tt.sloClass)
			}
		})
	}
}

// TestSLOGatedAdmission_SheddableRejectedOverThreshold verifies non-protected classes
// are rejected when total queue depth exceeds the threshold.
func TestSLOGatedAdmission_SheddableRejectedOverThreshold(t *testing.T) {
	policy := NewSLOGatedAdmission([]string{"critical", "standard"}, 50)
	overloaded := &RouterState{
		Snapshots: []RoutingSnapshot{
			{QueueDepth: 20}, {QueueDepth: 15}, {QueueDepth: 10}, {QueueDepth: 10},
		}, // total = 55 > 50
		Clock: 1000,
	}

	tests := []struct {
		name     string
		sloClass string
	}{
		{"sheddable rejected", "sheddable"},
		{"batch rejected", "batch"},
		{"background rejected", "background"},
		{"empty class rejected", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{ID: "r0", SLOClass: tt.sloClass}
			admitted, reason := policy.Admit(req, overloaded)
			if admitted {
				t.Errorf("%q should be rejected when queue exceeds threshold", tt.sloClass)
			}
			if reason == "" {
				t.Error("rejection should have a reason")
			}
		})
	}
}

// TestSLOGatedAdmission_SheddableAdmittedUnderThreshold verifies non-protected classes
// are admitted when total queue depth is under the threshold.
func TestSLOGatedAdmission_SheddableAdmittedUnderThreshold(t *testing.T) {
	policy := NewSLOGatedAdmission([]string{"critical", "standard"}, 50)
	light := &RouterState{
		Snapshots: []RoutingSnapshot{
			{QueueDepth: 5}, {QueueDepth: 5}, {QueueDepth: 5}, {QueueDepth: 5},
		}, // total = 20 < 50
		Clock: 1000,
	}

	req := &Request{ID: "r0", SLOClass: "sheddable"}
	admitted, _ := policy.Admit(req, light)
	if !admitted {
		t.Error("sheddable should be admitted when queue is under threshold")
	}
}

// TestSLOGatedAdmission_ExactThresholdAdmits verifies that requests at exactly
// the threshold boundary are admitted (rejection is strictly greater-than).
func TestSLOGatedAdmission_ExactThresholdAdmits(t *testing.T) {
	policy := NewSLOGatedAdmission([]string{"critical"}, 50)
	atThreshold := &RouterState{
		Snapshots: []RoutingSnapshot{
			{QueueDepth: 25}, {QueueDepth: 25},
		}, // total = 50 = threshold (not >)
		Clock: 1000,
	}

	req := &Request{ID: "r0", SLOClass: "sheddable"}
	admitted, _ := policy.Admit(req, atThreshold)
	if !admitted {
		t.Error("sheddable should be admitted when queue equals threshold (boundary is strictly >)")
	}
}

// TestSLOGatedAdmission_INV9_DoesNotReadOutputTokens verifies that the admission
// decision is independent of OutputTokens (INV-9: oracle knowledge boundary).
func TestSLOGatedAdmission_INV9_DoesNotReadOutputTokens(t *testing.T) {
	policy := NewSLOGatedAdmission([]string{"critical"}, 50)
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{QueueDepth: 100}},
		Clock:     1000,
	}

	req1 := &Request{SLOClass: "sheddable", OutputTokens: []int{1, 2, 3}}
	req2 := &Request{SLOClass: "sheddable", OutputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
	a1, _ := policy.Admit(req1, state)
	a2, _ := policy.Admit(req2, state)
	if a1 != a2 {
		t.Error("INV-9: admission should not depend on OutputTokens")
	}
}

// TestSLOGatedAdmission_PanicsOnZeroThreshold verifies R3 constructor validation.
func TestSLOGatedAdmission_PanicsOnZeroThreshold(t *testing.T) {
	tests := []struct {
		name      string
		threshold int
	}{
		{"zero", 0},
		{"negative", -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic for queueThreshold=%d, got none", tt.threshold)
				}
			}()
			NewSLOGatedAdmission([]string{"critical"}, tt.threshold)
		})
	}
}

// TestNewAdmissionPolicy_SLOGated verifies the factory creates a working slo-gated policy.
func TestNewAdmissionPolicy_SLOGated(t *testing.T) {
	// Factory uses capacity as queue threshold; protected classes = critical, standard
	p := NewAdmissionPolicy("slo-gated", 50, 0)

	overloaded := &RouterState{
		Snapshots: []RoutingSnapshot{{QueueDepth: 100}},
		Clock:     1000,
	}

	// Critical should be admitted even when overloaded
	critical := &Request{ID: "r0", SLOClass: "critical"}
	admitted, _ := p.Admit(critical, overloaded)
	if !admitted {
		t.Error("slo-gated factory: critical should be admitted")
	}

	// Sheddable should be rejected when overloaded
	sheddable := &Request{ID: "r1", SLOClass: "sheddable"}
	admitted, reason := p.Admit(sheddable, overloaded)
	if admitted {
		t.Error("slo-gated factory: sheddable should be rejected when overloaded")
	}
	if reason == "" {
		t.Error("slo-gated factory: rejection should have a reason")
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
