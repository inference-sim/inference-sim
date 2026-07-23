package creation

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// B-6 (#1494) contract tests for the pre-placement creation policy. They pin the
// two seam methods (Initial returns the assigned subset copy-safe; OnResidentMiss
// always admits) and the registry resolution (New/ValidNames), observed only
// through the public sim.CreationPolicy surface — never the internal type shape.

// TestPrePlacement_InitialReturnsAssigned pins C-B6-1: Initial returns exactly the
// assigned subset, order-preserving, for nil / empty / singleton / multi inputs.
func TestPrePlacement_InitialReturnsAssigned(t *testing.T) {
	p, err := New("pre-placement")
	if err != nil {
		t.Fatalf("New(pre-placement): %v", err)
	}
	tests := []struct {
		name     string
		assigned []string
		want     []string
	}{
		{"nil", nil, nil},
		{"empty", []string{}, nil},
		{"singleton", []string{"A"}, []string{"A"}},
		{"multi_order_preserved", []string{"A", "B", "C"}, []string{"A", "B", "C"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := p.Initial(sim.CreationContext{Assigned: tc.assigned})
			// nil and empty both seed nothing: an empty-length result is correct
			// regardless of nil-vs-empty (the seeding loop treats them identically).
			if len(tc.want) == 0 {
				if len(got) != 0 {
					t.Errorf("Initial(%v) = %v, want empty", tc.assigned, got)
				}
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Initial(%v) = %v, want %v", tc.assigned, got, tc.want)
			}
		})
	}
}

// TestPrePlacement_InitialCopySafe pins C-B6-1's copy-safety clause: mutating the
// returned slice must not corrupt the caller's assigned slice. A policy that
// returned the caller's slice by reference would fail this.
func TestPrePlacement_InitialCopySafe(t *testing.T) {
	p, err := New("pre-placement")
	if err != nil {
		t.Fatalf("New(pre-placement): %v", err)
	}
	assigned := []string{"A", "B"}
	got := p.Initial(sim.CreationContext{Assigned: assigned})
	if len(got) < 1 {
		t.Fatalf("Initial returned empty, want %v", assigned)
	}
	got[0] = "MUTATED"
	if assigned[0] != "A" {
		t.Errorf("mutating the returned slice corrupted the caller's assigned slice: assigned[0] = %q, want %q", assigned[0], "A")
	}
}

// TestPrePlacement_OnResidentMissAdmits pins C-B6-2/INV-8: OnResidentMiss returns
// true for any missed adapter (seeded or not), so no enqueued request is starved.
func TestPrePlacement_OnResidentMissAdmits(t *testing.T) {
	p, err := New("pre-placement")
	if err != nil {
		t.Fatalf("New(pre-placement): %v", err)
	}
	for _, id := range []string{"", "A", "unseeded"} {
		if !p.OnResidentMiss(sim.CreationContext{MissedAdapter: id}) {
			t.Errorf("OnResidentMiss(%q) = false, want true (pre-placement always admits)", id)
		}
	}
}

// TestValidNames_ContainsPrePlacement pins C-B6-5: ValidNames is sorted and lists
// both shipped policies, so the CLI help / fatal message names pre-placement.
func TestValidNames_ContainsPrePlacement(t *testing.T) {
	names := ValidNames()
	if len(names) < 2 {
		t.Fatalf("ValidNames = %v, want at least on-demand + pre-placement", names)
	}
	// Sorted invariant (INV-6 deterministic error text).
	for i := 1; i < len(names); i++ {
		if names[i-1] > names[i] {
			t.Errorf("ValidNames not sorted: %v", names)
		}
	}
	found := map[string]bool{}
	for _, n := range names {
		found[n] = true
	}
	for _, want := range []string{"on-demand", "pre-placement"} {
		if !found[want] {
			t.Errorf("ValidNames = %v, missing %q", names, want)
		}
	}
}
