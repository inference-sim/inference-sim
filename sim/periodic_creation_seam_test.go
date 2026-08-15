package sim

import (
	"reflect"
	"strings"
	"testing"
)

// TestPeriodicCreationContext_FieldAudit pins the context's shape for the same reasons
// TestCreationContext_FieldAudit (creation_seam_test.go:32) pins CreationContext's, but
// mirrors that test's STRONGER structure rather than the weaker reflect.Kind comparison:
// reflect.Kind alone cannot distinguish "[]string" from "[]*Request" (both are
// reflect.Slice), which would let exactly the leak this test exists to catch — a
// *Request field exposing Request.OutputTokens (INV-9/INV-L6) — slip through disguised
// as a slice. Comparing exact type strings closes that hole. A map-typed collection
// would independently break INV-6 (seed-dependent range order), so that check stays too.
// The NumMethod checks pin the doc comment's promise that this type "exposes no
// mutators" (Principle I) through either the value or the pointer receiver.
func TestPeriodicCreationContext_FieldAudit(t *testing.T) {
	ct := reflect.TypeOf(PeriodicCreationContext{})

	if ct.Kind() != reflect.Struct {
		t.Fatalf("PeriodicCreationContext kind = %v, want struct", ct.Kind())
	}
	if ct.NumMethod() != 0 {
		t.Errorf("PeriodicCreationContext value type exposes %d methods, want 0 (no mutators)", ct.NumMethod())
	}
	pt := reflect.TypeOf(&PeriodicCreationContext{})
	if pt.NumMethod() != 0 {
		t.Errorf("*PeriodicCreationContext exposes %d methods, want 0 (no mutators)", pt.NumMethod())
	}

	want := map[string]string{
		"Now":       "int64",
		"Interval":  "int64",
		"Instances": "[]sim.InstanceResidency",
		"Demand":    "sim.DemandWindow",
		"Registry":  "sim.AdapterRegistry",
	}
	got := map[string]string{}
	for i := 0; i < ct.NumField(); i++ {
		f := ct.Field(i)
		got[f.Name] = f.Type.String()
		if f.Type.Kind() == reflect.Map {
			t.Errorf("PeriodicCreationContext.%s is a map: ranging it would break INV-6; expose a sorted slice or an accessor", f.Name)
		}
		if strings.Contains(f.Type.String(), "Request") {
			t.Errorf("PeriodicCreationContext.%s leaks a Request (INV-9/INV-L6): the tick must never see OutputTokens", f.Name)
		}
		if f.Name == "OutputTokens" {
			t.Errorf("PeriodicCreationContext exposes OutputTokens — oracle-knowledge violation (INV-9/INV-L6)")
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("PeriodicCreationContext fields = %v, want exactly %v", got, want)
	}
}

// TestInstanceResidency_FieldAudit pins the per-instance view, with the same
// strengthening as TestPeriodicCreationContext_FieldAudit above (exact type strings,
// not reflect.Kind; explicit no-mutator and no-Request/OutputTokens/map checks) —
// InstanceResidency is exactly the shape a leaked *Request would try to hide in, since
// Resident/Unpinned are both already slices. Capacity is TOTAL slots, not free ones:
// free = Capacity - len(Resident). The two adapter-id slices must be slices (not maps)
// so the policy's iteration order is the cluster's, not Go's map order (INV-6).
func TestInstanceResidency_FieldAudit(t *testing.T) {
	rt := reflect.TypeOf(InstanceResidency{})

	if rt.Kind() != reflect.Struct {
		t.Fatalf("InstanceResidency kind = %v, want struct", rt.Kind())
	}
	if rt.NumMethod() != 0 {
		t.Errorf("InstanceResidency value type exposes %d methods, want 0 (no mutators)", rt.NumMethod())
	}
	pt := reflect.TypeOf(&InstanceResidency{})
	if pt.NumMethod() != 0 {
		t.Errorf("*InstanceResidency exposes %d methods, want 0 (no mutators)", pt.NumMethod())
	}

	want := map[string]string{
		"ID":          "string",
		"Resident":    "[]string",
		"Unpinned":    "[]string",
		"Capacity":    "int",
		"Loading":     "string",
		"GateBlocked": "bool",
	}
	got := map[string]string{}
	for i := 0; i < rt.NumField(); i++ {
		f := rt.Field(i)
		got[f.Name] = f.Type.String()
		if f.Type.Kind() == reflect.Map {
			t.Errorf("InstanceResidency.%s is a map: ranging it would break INV-6; expose a sorted slice or an accessor", f.Name)
		}
		if strings.Contains(f.Type.String(), "Request") {
			t.Errorf("InstanceResidency.%s leaks a Request (INV-9/INV-L6): the tick must never see OutputTokens", f.Name)
		}
		if f.Name == "OutputTokens" {
			t.Errorf("InstanceResidency exposes OutputTokens — oracle-knowledge violation (INV-9/INV-L6)")
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("InstanceResidency fields = %v, want exactly %v", got, want)
	}
}

// stubPeriodic is the minimal implementer, proving the interface is satisfiable by a
// value type with no dependency on package internals.
type stubPeriodic struct{ out []PrefetchDecision }

func (s stubPeriodic) OnTick(PeriodicCreationContext) []PrefetchDecision { return s.out }

// TestPeriodicCreationPolicy_Satisfiable is a compile-time-ish assertion that the
// interface is implementable and that CreationPolicy is NOT widened — a policy may
// implement one, the other, or both. The onDemandForTest stub below is NOT the real
// on-demand policy (that lives in sim/lora/creation, a different package); it only
// proves the two interfaces are SEPARABLE — a type can implement CreationPolicy
// without being forced to implement PeriodicCreationPolicy too. Real per-policy
// coverage (including the shipped on-demand and keep-warm) arrives in Task 4's
// contract test.
func TestPeriodicCreationPolicy_Satisfiable(t *testing.T) {
	var p PeriodicCreationPolicy = stubPeriodic{out: []PrefetchDecision{{Instance: "i0", Adapter: "a1"}}}
	got := p.OnTick(PeriodicCreationContext{Now: 5, Interval: 5})
	if len(got) != 1 || got[0].Instance != "i0" || got[0].Adapter != "a1" {
		t.Errorf("OnTick round-trip failed, got %v", got)
	}
	// on-demand implements CreationPolicy only; it must NOT be forced to implement the
	// optional interface (that is the whole point of the sibling-interface shape).
	var cp CreationPolicy = onDemandForTest{}
	if _, ok := cp.(PeriodicCreationPolicy); ok {
		t.Error("a CreationPolicy must not be required to implement PeriodicCreationPolicy")
	}
}

type onDemandForTest struct{}

func (onDemandForTest) Initial(CreationContext) []string    { return nil }
func (onDemandForTest) OnResidentMiss(CreationContext) bool { return true }
