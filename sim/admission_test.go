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
		{
			name:  "empty request",
			req:   &Request{ID: "r0", InputTokens: []int{}},
			clock: 0,
		},
		{
			name:  "small request",
			req:   &Request{ID: "r1", InputTokens: make([]int, 10)},
			clock: 1000,
		},
		{
			name:  "large request",
			req:   &Request{ID: "r2", InputTokens: make([]int, 10000)},
			clock: 5_000_000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			admitted, reason := policy.Admit(tt.req, tt.clock)
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
		admitted, reason := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("expected admission with sufficient tokens")
		}
		if reason != "" {
			t.Errorf("expected empty reason, got %q", reason)
		}
	})

	t.Run("rejects when tokens exhausted", func(t *testing.T) {
		tb := NewTokenBucket(10, 0)
		req := &Request{ID: "r0", InputTokens: make([]int, 10)}

		admitted, _ := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("first request should be admitted")
		}

		admitted, reason := tb.Admit(req, 0)
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

		admitted, _ := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("first request should be admitted")
		}

		// At t=50us, refill = 50 tokens. Cost=100, should reject.
		admitted, _ = tb.Admit(req, 50)
		if admitted {
			t.Fatal("expected rejection: only 50 tokens refilled, need 100")
		}

		// At t=150us, refill = 100 more tokens, capped at capacity=100. Should admit.
		admitted, reason := tb.Admit(req, 150)
		if !admitted {
			t.Fatalf("expected admission after refill, reason: %s", reason)
		}
	})

	t.Run("capacity caps refill", func(t *testing.T) {
		tb := NewTokenBucket(10, 1_000_000)
		req := &Request{ID: "r0", InputTokens: make([]int, 5)}

		admitted, _ := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("should admit")
		}

		// At t=1s, refill huge but capped at capacity=10
		admitted, _ = tb.Admit(req, 1_000_000)
		if !admitted {
			t.Fatal("should admit after refill")
		}

		admitted, _ = tb.Admit(req, 1_000_000)
		if !admitted {
			t.Fatal("should admit: 5 tokens remain")
		}

		admitted, _ = tb.Admit(req, 1_000_000)
		if admitted {
			t.Fatal("should reject: 0 tokens remain, no time elapsed for refill")
		}
	})

	t.Run("zero-cost request always admitted", func(t *testing.T) {
		tb := NewTokenBucket(0, 0)
		req := &Request{ID: "r0", InputTokens: []int{}}

		admitted, _ := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("zero-cost request should always be admitted")
		}
	})
}

// TestNewAdmissionPolicy_ValidNames verifies the factory returns correct types.
func TestNewAdmissionPolicy_ValidNames(t *testing.T) {
	t.Run("always-admit", func(t *testing.T) {
		p := NewAdmissionPolicy("always-admit", 0, 0)
		if _, ok := p.(*AlwaysAdmit); !ok {
			t.Errorf("expected *AlwaysAdmit, got %T", p)
		}
	})

	t.Run("empty string returns AlwaysAdmit", func(t *testing.T) {
		p := NewAdmissionPolicy("", 0, 0)
		if _, ok := p.(*AlwaysAdmit); !ok {
			t.Errorf("expected *AlwaysAdmit for empty string, got %T", p)
		}
	})

	t.Run("token-bucket", func(t *testing.T) {
		p := NewAdmissionPolicy("token-bucket", 100, 10)
		if _, ok := p.(*TokenBucket); !ok {
			t.Errorf("expected *TokenBucket, got %T", p)
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
