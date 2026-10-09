package sim

import (
	"reflect"
	"testing"
)

// These are the B-5 (#1493) integration contract tests for the creation seam,
// observed through resident state, the per-adapter load counts, and request
// completion — never through the seam's internal shape. LoRA is wired into
// package sim's tests via the blank import in lora_import_test.go, which
// registers the resident-set, registry, cost, eviction- and creation-policy
// construction funcs.
//
// gateTestConfig / mustNewSimulator / newTestRequest / newTestSimConfig are
// shared helpers from cold_load_gate_test.go and simulator_test.go.

// stubCreation is a test-only CreationPolicy that returns a fixed seed set and a
// fixed admit decision, letting the tests exercise the seeding accounting law
// (C-7) and the OnResidentMiss=false branch (C-9) that no shipped policy selects.
type stubCreation struct {
	initIDs []string
	admit   bool
}

func (s stubCreation) Initial(CreationContext) []string  { return s.initIDs }
func (s stubCreation) OnResidentMiss(CreationContext) bool { return s.admit }

// TestCreationContext_FieldAudit pins C-1 (INV-9/INV-L6): CreationContext is a
// read-only value struct carrying exactly Assigned/MissedAdapter/Registry, no
// *Request and no output-token field, and exposes no mutator methods.
func TestCreationContext_FieldAudit(t *testing.T) {
	ct := reflect.TypeOf(CreationContext{})

	if ct.Kind() != reflect.Struct {
		t.Fatalf("CreationContext kind = %v, want struct", ct.Kind())
	}
	if ct.NumMethod() != 0 {
		t.Errorf("CreationContext value type exposes %d methods, want 0 (no mutators)", ct.NumMethod())
	}
	pt := reflect.TypeOf(&CreationContext{})
	if pt.NumMethod() != 0 {
		t.Errorf("*CreationContext exposes %d methods, want 0 (no mutators)", pt.NumMethod())
	}

	want := map[string]string{
		"Assigned":      "[]string",
		"MissedAdapter": "string",
		"Registry":      "sim.AdapterRegistry",
	}
	got := map[string]string{}
	for i := 0; i < ct.NumField(); i++ {
		f := ct.Field(i)
		got[f.Name] = f.Type.String()
		if f.Type.String() == "*sim.Request" {
			t.Errorf("CreationContext.%s is *Request — the seam must not carry a request (INV-9)", f.Name)
		}
		if f.Name == "OutputTokens" {
			t.Errorf("CreationContext exposes OutputTokens — oracle-knowledge violation (INV-9/INV-L6)")
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("CreationContext fields = %v, want exactly %v", got, want)
	}
}

// TestApplyInitialCreation_StubSeedsResidentsUncharged pins C-7/INV-L3: when a
// policy's Initial returns ids, ApplyInitialCreation makes them resident with NO
// load-count increment (t=0 seeding is not a charged cold load).
func TestApplyInitialCreation_StubSeedsResidentsUncharged(t *testing.T) {
	cfg := gateTestConfig(2, AdapterSpec{ID: "A", Rank: 8}, AdapterSpec{ID: "B", Rank: 8})
	s := mustNewSimulator(t, cfg)
	s.creationPolicy = stubCreation{initIDs: []string{"A", "B"}, admit: true}

	s.ApplyInitialCreation([]string{"A", "B"})

	for _, id := range []string{"A", "B"} {
		if !s.residentAdapters.IsResident(id) {
			t.Errorf("adapter %s should be resident after Initial seeding", id)
		}
		if got := s.Metrics.AdapterLoadCounts[id]; got != 0 {
			t.Errorf("AdapterLoadCounts[%s] = %d after seeding, want 0 (INV-L3: seeding is uncharged)", id, got)
		}
	}
}

// TestApplyInitialCreation_OnDemandSeedsNothing pins C-4/INV-L1: the shipped
// on-demand default seeds nothing even when handed a non-empty assignment, so a
// LoRA-active instance starts with an empty resident set exactly as pre-B-5.
func TestApplyInitialCreation_OnDemandSeedsNothing(t *testing.T) {
	cfg := gateTestConfig(2, AdapterSpec{ID: "A", Rank: 8})
	s := mustNewSimulator(t, cfg) // creationPolicy defaults to on-demand

	s.ApplyInitialCreation([]string{"A"})

	if s.residentAdapters.IsResident("A") {
		t.Error("on-demand ApplyInitialCreation seeded adapter A; want empty resident set (C-4)")
	}
}

// TestApplyInitialCreation_InertSubsystemNoOp pins C-4: with the LoRA subsystem
// off (residentAdapters/creationPolicy nil), ApplyInitialCreation is a no-op and
// never panics.
func TestApplyInitialCreation_InertSubsystemNoOp(t *testing.T) {
	s := mustNewSimulator(t, newTestSimConfig()) // no adapters => subsystem inert

	if s.residentAdapters != nil {
		t.Fatalf("expected inert subsystem (residentAdapters == nil), got non-nil")
	}
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("ApplyInitialCreation panicked on inert subsystem: %v", r)
		}
	}()
	s.ApplyInitialCreation([]string{"A"}) // must be a safe no-op
}

// TestPrePlacementSelected_SeedsResidentsUncharged pins C-B6-1/INV-L3 end-to-end
// through the REAL pre-placement policy selected via cfg.CreationPolicy (not a
// stub): a "pre-placement"-configured Simulator seeds its assigned subset resident
// at t=0 with NO load-count increment. This exercises the T026 config-driven
// selection (NewSimulator reads cfg.CreationPolicy) plus the shipped policy.
func TestPrePlacementSelected_SeedsResidentsUncharged(t *testing.T) {
	cfg := gateTestConfig(2, AdapterSpec{ID: "A", Rank: 8}, AdapterSpec{ID: "B", Rank: 8})
	cfg.CreationPolicy = "pre-placement"
	s := mustNewSimulator(t, cfg)

	s.ApplyInitialCreation([]string{"A", "B"})

	for _, id := range []string{"A", "B"} {
		if !s.residentAdapters.IsResident(id) {
			t.Errorf("adapter %s should be resident after pre-placement seeding", id)
		}
		if got := s.Metrics.AdapterLoadCounts[id]; got != 0 {
			t.Errorf("AdapterLoadCounts[%s] = %d after seeding, want 0 (INV-L3: seeding is uncharged)", id, got)
		}
	}
}

// TestOnDemandSelected_SeedsNothing pins the regression guard for the T026
// selection change: an explicit CreationPolicy "on-demand" (like the empty
// default) still seeds nothing, so selecting the shipped default byte-identically
// preserves pre-B-6 behavior (INV-L1).
func TestOnDemandSelected_SeedsNothing(t *testing.T) {
	for _, name := range []string{"", "on-demand"} {
		cfg := gateTestConfig(2, AdapterSpec{ID: "A", Rank: 8})
		cfg.CreationPolicy = name
		s := mustNewSimulator(t, cfg)

		s.ApplyInitialCreation([]string{"A"})

		if s.residentAdapters.IsResident("A") {
			t.Errorf("CreationPolicy %q seeded adapter A; want empty resident set (INV-L1)", name)
		}
	}
}

// TestOnResidentMissFalse_HoldsRequestInert pins C-9/INV-8: a policy returning
// OnResidentMiss=false starts no cold load, leaves the adapter non-resident, and
// does not busy-loop — the request is simply held (not completed) without the
// simulator idling on runnable work.
func TestOnResidentMissFalse_HoldsRequestInert(t *testing.T) {
	cfg := gateTestConfig(2, AdapterSpec{ID: "A", Rank: 8})
	cfg.Horizon = 10_000_000 // finite bound: a held request must not hang Run()
	s := mustNewSimulator(t, cfg)
	s.creationPolicy = stubCreation{admit: false}

	req := newTestRequest("held", 0, 8, 4)
	req.Adapter = "A"
	s.InjectArrival(req)
	s.Run()

	if s.residentAdapters.IsResident("A") {
		t.Error("adapter A became resident despite OnResidentMiss=false")
	}
	if got := s.Metrics.AdapterLoadCounts["A"]; got != 0 {
		t.Errorf("AdapterLoadCounts[A] = %d, want 0 (no load started when admit=false)", got)
	}
	if s.Metrics.CompletedRequests != 0 {
		t.Errorf("CompletedRequests = %d, want 0 (request held at the cold-load gate)", s.Metrics.CompletedRequests)
	}
}

// TestCreationPolicyMethods_Pure pins C-9: invoking Initial/OnResidentMiss on the
// shipped on-demand policy mutates no simulator state (the caller performs any
// Store) — resident set stays empty and no request completes.
func TestCreationPolicyMethods_Pure(t *testing.T) {
	cfg := gateTestConfig(2, AdapterSpec{ID: "A", Rank: 8})
	s := mustNewSimulator(t, cfg)

	_ = s.creationPolicy.Initial(CreationContext{Assigned: []string{"A"}, Registry: s.adapterRegistry})
	_ = s.creationPolicy.OnResidentMiss(CreationContext{MissedAdapter: "A", Registry: s.adapterRegistry})

	if s.residentAdapters.Len() != 0 {
		t.Errorf("resident set size = %d after pure method calls, want 0", s.residentAdapters.Len())
	}
	if s.Metrics.CompletedRequests != 0 {
		t.Errorf("CompletedRequests = %d after pure method calls, want 0", s.Metrics.CompletedRequests)
	}
}
