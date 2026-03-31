package sim

import (
	"fmt"
	"testing"
)

// TestSLOLowestPriorityVictim_EvictsLowestSLO verifies that the SLO-priority
// victim selector picks the request with the lowest SLO tier priority,
// not simply the last element (LIFO).
// BC-P1: [sheddable, critical] → SLO evicts sheddable (idx 0); LIFO would evict idx 1.
func TestSLOLowestPriorityVictim_EvictsLowestSLO(t *testing.T) {
	reqs := []*Request{
		{ID: "shed", SLOClass: "sheddable"},
		{ID: "crit", SLOClass: "critical"},
	}
	idx := SLOLowestPriorityVictim(reqs)
	if reqs[idx].SLOClass != "sheddable" {
		t.Errorf("expected sheddable to be selected as victim (idx 0), got idx=%d class=%s", idx, reqs[idx].SLOClass)
	}
}

// TestSLOLowestPriorityVictim_TiesResolveLIFO verifies that when all requests
// share the same SLO class, the last element is selected (LIFO tie-breaking).
// BC-P2: [standard, standard, standard] → idx 2 (last).
func TestSLOLowestPriorityVictim_TiesResolveLIFO(t *testing.T) {
	reqs := []*Request{
		{ID: "std-0", SLOClass: "standard"},
		{ID: "std-1", SLOClass: "standard"},
		{ID: "std-2", SLOClass: "standard"},
	}
	idx := SLOLowestPriorityVictim(reqs)
	if idx != 2 {
		t.Errorf("expected last element (idx 2) for LIFO tie-breaking, got idx=%d", idx)
	}
}

// TestSLOLowestPriorityVictim_AllTierOrdering verifies correct victim selection
// across all SLO tier combinations.
// BC-P3: table-driven covering background(0) < batch(1) < sheddable(2) < standard(3) < critical(4).
func TestSLOLowestPriorityVictim_AllTierOrdering(t *testing.T) {
	tests := []struct {
		name      string
		classes   []string
		wantClass string
	}{
		{"critical+standard+sheddable", []string{"critical", "standard", "sheddable"}, "sheddable"},
		{"critical+standard", []string{"critical", "standard"}, "standard"},
		{"all critical", []string{"critical", "critical"}, "critical"},
		{"background+critical", []string{"background", "critical"}, "background"},
		{"batch+sheddable", []string{"batch", "sheddable"}, "batch"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reqs := make([]*Request, len(tc.classes))
			for i, c := range tc.classes {
				reqs[i] = &Request{ID: fmt.Sprintf("req-%d", i), SLOClass: c}
			}
			idx := SLOLowestPriorityVictim(reqs)
			if reqs[idx].SLOClass != tc.wantClass {
				t.Errorf("want victim class %q, got %q (idx=%d)", tc.wantClass, reqs[idx].SLOClass, idx)
			}
		})
	}
}
