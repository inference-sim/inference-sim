package workload

import (
	"strconv"
	"strings"
	"testing"
)

func phasedSpec() *WorkloadSpec {
	client := func(id, phase string) ClientSpec {
		return ClientSpec{ID: id, Phase: phase, TenantID: "t", SLOClass: "standard", RateFraction: 1,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 10}}}
	}
	return &WorkloadSpec{Version: "2", Seed: 3, Category: "language", AggregateRate: 10,
		Phases:  []PhaseSpec{{Name: "warmup", NumRequests: 4}, {Name: "measured", NumRequests: 9}},
		Clients: []ClientSpec{client("w", "warmup"), client("m", "measured")}}
}

// A phased workload yields each phase's own count, labelled, in phase order, with identifiers
// running on, and the same spec yields the same requests.
func TestGeneratePhased_CountsLabelsAndDeterminism(t *testing.T) {
	a, err := GeneratePhased(phasedSpec())
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 13 {
		t.Fatalf("%d requests, want 4 + 9", len(a))
	}
	for i, r := range a {
		want := "warmup"
		if i >= 4 {
			want = "measured"
		}
		if string(r.Phase) != want || r.ID != "request_"+itoa(i) {
			t.Errorf("request %d is %s in phase %q, want request_%d in %q", i, r.ID, r.Phase, i, want)
		}
	}
	b, err := GeneratePhased(phasedSpec())
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i].ArrivalTime != b[i].ArrivalTime || len(a[i].InputTokens) != len(b[i].InputTokens) ||
			(len(a[i].InputTokens) > 0 && a[i].InputTokens[0] != b[i].InputTokens[0]) {
			t.Fatalf("request %d differs between two generations of one spec", i)
		}
	}
	if a[0].InputTokens[0] == a[4].InputTokens[0] && a[0].InputTokens[1] == a[4].InputTokens[1] {
		t.Errorf("the warm-up and measured clients drew the same tokens: their streams are not distinct")
	}
}

func itoa(i int) string { return strconv.Itoa(i) }

// The spec refuses what the drain gate cannot run, naming the client or phase.
func TestPhasedSpec_Refusals(t *testing.T) {
	cases := []struct {
		name   string
		edit   func(*WorkloadSpec)
		expect string
	}{
		{"client with no phase", func(s *WorkloadSpec) { s.Clients[1].Phase = "" }, "client \"m\""},
		{"client in an unlisted phase", func(s *WorkloadSpec) { s.Clients[1].Phase = "soak" }, "soak"},
		{"phase with no client", func(s *WorkloadSpec) { s.Clients[1].Phase = "warmup" }, "no client"},
		{"phase with no count", func(s *WorkloadSpec) { s.Phases[1].NumRequests = 0 }, "num_requests"},
		{"count disagreeing with the phases", func(s *WorkloadSpec) { s.NumRequests = 5 }, "sum"},
		{"duplicate phase name", func(s *WorkloadSpec) { s.Phases[1].Name = "warmup" }, "two phases"},
		{"session client", func(s *WorkloadSpec) {
			s.Clients[1].Reasoning = &ReasoningSpec{MultiTurn: &MultiTurnSpec{MaxRounds: 2}}
		}, "closed-loop"},
		{"phase name on an unphased workload", func(s *WorkloadSpec) { s.Phases = nil }, "lists no phases"},
	}
	for _, c := range cases {
		s := phasedSpec()
		c.edit(s)
		err := s.Validate()
		if err == nil || !strings.Contains(err.Error(), c.expect) {
			t.Errorf("%s: err = %v, want one containing %q", c.name, err, c.expect)
		}
	}
	if err := phasedSpec().Validate(); err != nil {
		t.Errorf("the valid phased spec is refused: %v", err)
	}
}
