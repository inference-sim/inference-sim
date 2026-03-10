package sim

import (
	"fmt"
	"testing"
)

// TestNeverDisaggregate_AlwaysReturnsFalse verifies that NeverDisaggregate
// always returns Disaggregate=false regardless of input.
func TestNeverDisaggregate_AlwaysReturnsFalse(t *testing.T) {
	decider := &NeverDisaggregate{}
	req := &Request{ID: "req-1", InputTokens: make([]int, 100)}

	decision := decider.Decide(req)

	if decision.Disaggregate {
		t.Error("NeverDisaggregate should return Disaggregate=false")
	}
}

// TestAlwaysDisaggregate_AlwaysReturnsTrue verifies that AlwaysDisaggregate
// always returns Disaggregate=true regardless of input.
func TestAlwaysDisaggregate_AlwaysReturnsTrue(t *testing.T) {
	decider := &AlwaysDisaggregate{}
	req := &Request{ID: "req-1", InputTokens: make([]int, 100)}

	decision := decider.Decide(req)

	if !decision.Disaggregate {
		t.Error("AlwaysDisaggregate should return Disaggregate=true")
	}
}

// TestDisaggregationDecider_Interface verifies both implementations satisfy the interface.
func TestDisaggregationDecider_Interface(t *testing.T) {
	var _ DisaggregationDecider = &NeverDisaggregate{}
	var _ DisaggregationDecider = &AlwaysDisaggregate{}
}

// TestNewDisaggregationDecider_Factory verifies factory dispatches correctly.
func TestNewDisaggregationDecider_Factory(t *testing.T) {
	tests := []struct {
		name     string
		wantType string
	}{
		{"", "*sim.NeverDisaggregate"},
		{"never", "*sim.NeverDisaggregate"},
		{"always", "*sim.AlwaysDisaggregate"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			decider := NewDisaggregationDecider(tc.name)
			if decider == nil {
				t.Fatal("NewDisaggregationDecider returned nil")
			}
			gotType := fmt.Sprintf("%T", decider)
			if gotType != tc.wantType {
				t.Errorf("NewDisaggregationDecider(%q) type = %s, want %s", tc.name, gotType, tc.wantType)
			}
		})
	}
}

// TestNewDisaggregationDecider_UnknownPanics verifies factory panics on unknown name.
func TestNewDisaggregationDecider_UnknownPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unknown decider name")
		}
	}()
	NewDisaggregationDecider("unknown-decider")
}

// TestIsValidDisaggregationDecider verifies the validation function.
func TestIsValidDisaggregationDecider(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"", true},
		{"never", true},
		{"always", true},
		{"unknown", false},
		{"NEVER", false}, // case-sensitive
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsValidDisaggregationDecider(tc.name); got != tc.want {
				t.Errorf("IsValidDisaggregationDecider(%q) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

// TestValidDisaggregationDeciderNames verifies the names list.
func TestValidDisaggregationDeciderNames(t *testing.T) {
	names := ValidDisaggregationDeciderNames()
	if len(names) < 2 {
		t.Errorf("expected at least 2 decider names, got %d", len(names))
	}
	// Verify sorted order and no empty string
	for i, n := range names {
		if n == "" {
			t.Errorf("names[%d] is empty", i)
		}
		if i > 0 && names[i-1] >= n {
			t.Errorf("names not sorted: %q >= %q", names[i-1], n)
		}
	}
}
