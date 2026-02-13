package policy

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// BC-9: AlwaysAdmit always returns (true, "")
func TestAlwaysAdmit_AdmitsAll(t *testing.T) {
	policy := &AlwaysAdmit{}

	tests := []struct {
		name  string
		req   *sim.Request
		clock int64
	}{
		{
			name:  "empty request",
			req:   &sim.Request{ID: "r0", InputTokens: []int{}},
			clock: 0,
		},
		{
			name:  "small request",
			req:   &sim.Request{ID: "r1", InputTokens: make([]int, 10)},
			clock: 1000,
		},
		{
			name:  "large request",
			req:   &sim.Request{ID: "r2", InputTokens: make([]int, 10000)},
			clock: 5_000_000,
		},
		{
			name:  "any clock value",
			req:   &sim.Request{ID: "r3", InputTokens: make([]int, 5)},
			clock: 0,
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

// BC-10: TokenBucket admits and rejects based on token availability
func TestTokenBucket_AdmitAndReject(t *testing.T) {
	t.Run("admits when tokens available", func(t *testing.T) {
		tb := NewTokenBucket(100, 10) // capacity=100, refillRate=10/sec
		req := &sim.Request{ID: "r0", InputTokens: make([]int, 50)}
		admitted, reason := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("expected admission with sufficient tokens")
		}
		if reason != "" {
			t.Errorf("expected empty reason, got %q", reason)
		}
	})

	t.Run("rejects when tokens exhausted", func(t *testing.T) {
		tb := NewTokenBucket(10, 0) // capacity=10, no refill
		req := &sim.Request{ID: "r0", InputTokens: make([]int, 10)}

		// First request exhausts all tokens
		admitted, _ := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("first request should be admitted")
		}

		// Second request should be rejected (0 tokens left, cost=10)
		admitted, reason := tb.Admit(req, 0)
		if admitted {
			t.Fatal("expected rejection with exhausted tokens")
		}
		if reason != "insufficient tokens" {
			t.Errorf("expected reason %q, got %q", "insufficient tokens", reason)
		}
	})

	t.Run("refill restores tokens over time", func(t *testing.T) {
		tb := NewTokenBucket(100, 1_000_000) // capacity=100, refill=1M tokens/sec
		req := &sim.Request{ID: "r0", InputTokens: make([]int, 100)}

		// Exhaust all tokens at t=0
		admitted, _ := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("first request should be admitted")
		}

		// At t=50us, refill = 50us * 1M/1e6 = 50 tokens. Cost=100, should reject.
		admitted, _ = tb.Admit(req, 50)
		if admitted {
			t.Fatal("expected rejection: only 50 tokens refilled, need 100")
		}

		// At t=150us, refill = 100us more * 1M/1e6 = 100 more tokens. But capped at capacity=100.
		// Available = min(100, 50+100) = 100. Cost=100, should admit.
		admitted, reason := tb.Admit(req, 150)
		if !admitted {
			t.Fatalf("expected admission after refill, reason: %s", reason)
		}
	})

	t.Run("capacity caps refill", func(t *testing.T) {
		tb := NewTokenBucket(10, 1_000_000) // capacity=10, very fast refill
		req := &sim.Request{ID: "r0", InputTokens: make([]int, 5)}

		// Admit at t=0, tokens: 10-5=5
		admitted, _ := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("should admit")
		}

		// At t=1_000_000 (1 sec), refill would be huge but capped at capacity=10
		// Available = min(10, 5 + 1_000_000) = 10. Cost=5, admit.
		admitted, _ = tb.Admit(req, 1_000_000)
		if !admitted {
			t.Fatal("should admit after refill")
		}

		// Admit again immediately, tokens: 10-5=5, cost=5, should admit
		admitted, _ = tb.Admit(req, 1_000_000)
		if !admitted {
			t.Fatal("should admit: 5 tokens remain")
		}

		// Now tokens=0, should reject
		admitted, _ = tb.Admit(req, 1_000_000)
		if admitted {
			t.Fatal("should reject: 0 tokens remain, no time elapsed for refill")
		}
	})

	t.Run("zero-cost request always admitted", func(t *testing.T) {
		tb := NewTokenBucket(0, 0) // empty bucket, no refill
		req := &sim.Request{ID: "r0", InputTokens: []int{}}

		// Cost = len(InputTokens) = 0, so 0 >= 0 is true
		admitted, _ := tb.Admit(req, 0)
		if !admitted {
			t.Fatal("zero-cost request should always be admitted")
		}
	})
}

// EC-1: NewAdmissionPolicy panics on invalid policy name
func TestAdmissionPolicy_InvalidName_Panics(t *testing.T) {
	tests := []struct {
		name       string
		policyName string
	}{
		{"empty string", ""},
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

func TestNewAdmissionPolicy_ValidNames(t *testing.T) {
	t.Run("always-admit", func(t *testing.T) {
		p := NewAdmissionPolicy("always-admit", 0, 0)
		if _, ok := p.(*AlwaysAdmit); !ok {
			t.Errorf("expected *AlwaysAdmit, got %T", p)
		}
	})

	t.Run("token-bucket", func(t *testing.T) {
		p := NewAdmissionPolicy("token-bucket", 100, 10)
		if _, ok := p.(*TokenBucket); !ok {
			t.Errorf("expected *TokenBucket, got %T", p)
		}
	})
}
