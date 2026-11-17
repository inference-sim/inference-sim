package sim

import "testing"

// A request that emitted one output token had no gap between tokens, so it has no ITL: it
// is absent from the per-request ITLs, and request-averaged consumers cannot count it as a
// zero sample (#1894). A request with two or more keeps its ITL.
func TestOneTokenCompletionHasNoITL(t *testing.T) {
	m := NewMetrics()
	one := &Request{ID: "one"}
	three := &Request{ID: "three"}
	m.RecordCompletion(Completion{Req: one, OutputTokens: 1, E2E: 1000, ITL: 0})
	m.RecordCompletion(Completion{Req: three, OutputTokens: 3, E2E: 3000, ITL: 400})
	if _, ok := m.RequestITLs["one"]; ok {
		t.Errorf("a one-token request has an ITL of %v; it has none", m.RequestITLs["one"])
	}
	if got, ok := m.RequestITLs["three"]; !ok || got != 400 {
		t.Errorf("a three-token request has ITL %v (present %v), want 400", got, ok)
	}
	if m.CompletedRequests != 2 {
		t.Errorf("completed %d, want 2: the one-token request still completed", m.CompletedRequests)
	}
}
