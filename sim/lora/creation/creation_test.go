package creation

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestOnDemand_InitialReturnsEmpty pins C-4/INV-L1: on-demand seeds nothing at t=0,
// regardless of the assigned subset it is handed.
func TestOnDemand_InitialReturnsEmpty(t *testing.T) {
	p, err := New("on-demand")
	if err != nil {
		t.Fatalf("New(on-demand): %v", err)
	}
	for _, assigned := range [][]string{nil, {}, {"a"}, {"a", "b", "c"}} {
		if got := p.Initial(sim.CreationContext{Assigned: assigned}); len(got) != 0 {
			t.Errorf("on-demand Initial(assigned=%v) = %v, want empty", assigned, got)
		}
	}
}

// TestOnDemand_OnResidentMissAlwaysAdmits pins C-4: on-demand always admits, so the
// cold-load gate behaves exactly as pre-B-5 (INV-L1).
func TestOnDemand_OnResidentMissAlwaysAdmits(t *testing.T) {
	p, err := New("on-demand")
	if err != nil {
		t.Fatalf("New(on-demand): %v", err)
	}
	for _, missed := range []string{"", "x", "adapter-42"} {
		if !p.OnResidentMiss(sim.CreationContext{MissedAdapter: missed}) {
			t.Errorf("on-demand OnResidentMiss(missed=%q) = false, want true", missed)
		}
	}
}

// TestNew_EmptyDefaultsToOnDemand pins C-2/R20: the empty name resolves to on-demand,
// so LoRAConfig's zero value never surfaces an "unknown policy" error.
func TestNew_EmptyDefaultsToOnDemand(t *testing.T) {
	pEmpty, err := New("")
	if err != nil {
		t.Fatalf(`New(""): %v`, err)
	}
	pNamed, err := New("on-demand")
	if err != nil {
		t.Fatalf("New(on-demand): %v", err)
	}
	if reflect.TypeOf(pEmpty) != reflect.TypeOf(pNamed) {
		t.Errorf(`New("") type %T != New("on-demand") type %T`, pEmpty, pNamed)
	}
}

// TestNew_UnknownErrorsWithValidNames pins C-2: an unknown name errors and the message
// lists the valid options (deterministic, sorted).
func TestNew_UnknownErrorsWithValidNames(t *testing.T) {
	_, err := New("bogus")
	if err == nil {
		t.Fatal("New(bogus) = nil error, want error")
	}
	if !strings.Contains(err.Error(), "on-demand") {
		t.Errorf("error %q does not list valid name on-demand", err.Error())
	}
}

// TestValidNames_SortedContainsOnDemand pins C-2: ValidNames is sorted and includes the
// default.
func TestValidNames_SortedContainsOnDemand(t *testing.T) {
	names := ValidNames()
	if len(names) == 0 {
		t.Fatal("ValidNames() empty")
	}
	for i := 1; i < len(names); i++ {
		if names[i-1] > names[i] {
			t.Errorf("ValidNames() not sorted: %v", names)
		}
	}
	var found bool
	for _, n := range names {
		if n == "on-demand" {
			found = true
		}
	}
	if !found {
		t.Errorf("ValidNames() = %v, missing on-demand", names)
	}
}
