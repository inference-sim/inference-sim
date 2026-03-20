package sim

import (
	"fmt"
	"testing"
)

// TestNeverDisaggregate_AlwaysReturnsFalse verifies BC-PD-5
func TestNeverDisaggregate_AlwaysReturnsFalse(t *testing.T) {
	decider := &NeverDisaggregate{}
	
	tests := []struct {
		name string
		req  *Request
	}{
		{
			name: "short request",
			req: &Request{
				InputTokens:  []int{1, 2, 3},
				MaxOutputLen: 10,
			},
		},
		{
			name: "long request",
			req: &Request{
				InputTokens:  make([]int, 1000),
				MaxOutputLen: 500,
			},
		},
		{
			name: "zero output",
			req: &Request{
				InputTokens:  []int{1},
				MaxOutputLen: 0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision := decider.Decide(tt.req)
			if decision.Disaggregate {
				t.Errorf("NeverDisaggregate.Decide() = true, want false")
			}
		})
	}
}

// TestAlwaysDisaggregate_AlwaysReturnsTrue verifies BC-PD-6
func TestAlwaysDisaggregate_AlwaysReturnsTrue(t *testing.T) {
	decider := &AlwaysDisaggregate{}
	
	tests := []struct {
		name string
		req  *Request
	}{
		{
			name: "short request",
			req: &Request{
				InputTokens:  []int{1, 2, 3},
				MaxOutputLen: 10,
			},
		},
		{
			name: "long request",
			req: &Request{
				InputTokens:  make([]int, 1000),
				MaxOutputLen: 500,
			},
		},
		{
			name: "zero output",
			req: &Request{
				InputTokens:  []int{1},
				MaxOutputLen: 0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision := decider.Decide(tt.req)
			if !decision.Disaggregate {
				t.Errorf("AlwaysDisaggregate.Decide() = false, want true")
			}
		})
	}
}

// TestNewDisaggregationDecider_Factory verifies BC-PD-7
func TestNewDisaggregationDecider_Factory(t *testing.T) {
	tests := []struct {
		name         string
		deciderName  string
		wantType     string
	}{
		{
			name:        "empty string defaults to never",
			deciderName: "",
			wantType:    "*sim.NeverDisaggregate",
		},
		{
			name:        "never",
			deciderName: "never",
			wantType:    "*sim.NeverDisaggregate",
		},
		{
			name:        "always",
			deciderName: "always",
			wantType:    "*sim.AlwaysDisaggregate",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider := NewDisaggregationDecider(tt.deciderName)
			gotType := fmt.Sprintf("%T", decider)
			if gotType != tt.wantType {
				t.Errorf("NewDisaggregationDecider(%q) type = %v, want %v", tt.deciderName, gotType, tt.wantType)
			}
		})
	}
}

// TestNewDisaggregationDecider_UnknownPanics verifies BC-PD-14
func TestNewDisaggregationDecider_UnknownPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewDisaggregationDecider with unknown name should panic")
		}
	}()
	
	NewDisaggregationDecider("unknown-decider")
}