package sim

import (
	"os"
	"strings"
	"testing"
)

// inv1InstanceTerms lists the five terms of INV-1's single-instance specialisation,
// in the order docs/contributing/standards/invariants.md states them.
// TestINV1_InstanceHelperMatchesRegistry asserts this equals the terms parsed out of
// the registry, so the helper cannot drift from the statement it enforces.
//
// This list is duplicated in sim/cluster/inv1_conservation_test.go because a
// _test.go identifier is invisible across packages. The registry-diff test in each
// package is what keeps the two copies in step.
var inv1InstanceTerms = []string{
	"completed_requests",
	"still_queued",
	"still_running",
	"dropped_unservable",
	"timed_out",
}

func inv1Accounted(m *Metrics) int {
	return m.CompletedRequests + m.StillQueued + m.StillRunning +
		m.DroppedUnservable + m.TimedOutRequests
}

// assertINV1Conservation asserts INV-1's five-term single-instance specialisation:
// every injected request ends in exactly one of completed, still-queued,
// still-running, dropped-unservable or timed-out.
//
// Valid only for a single-instance simulation, which has no cluster router, gateway
// queue or encode pool — the seven cluster-only buckets are identically zero there.
// Cluster-level assertions must use assertClusterINV1Conservation in sim/cluster.
//
// injected must come from a source independent of the metrics under test — normally
// the number of requests the test enqueued. Passing a value derived from the same
// Metrics (for example MetricsOutput.InjectedRequests, which is itself defined as
// this sum) makes the assertion a tautology.
func assertINV1Conservation(t *testing.T, m *Metrics, injected int, label string) {
	t.Helper()
	if got := inv1Accounted(m); got != injected {
		t.Errorf("INV-1 conservation violated (%s): injected=%d != accounted=%d "+
			"(completed=%d still_queued=%d still_running=%d dropped_unservable=%d timed_out=%d)",
			label, injected, got, m.CompletedRequests, m.StillQueued, m.StillRunning,
			m.DroppedUnservable, m.TimedOutRequests)
	}
}

// TestINV1_InstanceHelperMatchesRegistry asserts this package's five-term list is
// the one the registry states as the single-instance specialisation. The cluster
// package runs the twelve-term half of the same check; between them the two copies
// of the five-term list cannot drift from the statement or from each other.
//
// Anchored on `injected_requests ==` because the registry also carries the
// full-pipeline clause `num_requests == injected_requests + rejected_requests`,
// which a looser match would pick up.
func TestINV1_InstanceHelperMatchesRegistry(t *testing.T) {
	const registryPath = "../docs/contributing/standards/invariants.md"
	data, err := os.ReadFile(registryPath)
	if err != nil {
		t.Fatalf("cannot read the invariant registry at %s: %v", registryPath, err)
	}
	var equations [][]string
	for _, line := range strings.Split(string(data), "\n") {
		expr := strings.TrimSpace(line)
		if !strings.HasPrefix(expr, "`injected_requests ==") || !strings.HasSuffix(expr, "`") {
			continue
		}
		rhs := strings.TrimSuffix(strings.SplitN(expr, "==", 2)[1], "`")
		terms := make([]string, 0, 12)
		for _, term := range strings.Split(rhs, "+") {
			terms = append(terms, strings.TrimSpace(term))
		}
		equations = append(equations, terms)
	}
	if len(equations) != 2 {
		t.Fatalf("expected exactly 2 `injected_requests == ...` equations in %s, found %d — the registry was restructured, so this test and the helper need re-checking against it",
			registryPath, len(equations))
	}

	registry := equations[1] // the five-term single-instance specialisation
	inHelper := make(map[string]bool, len(inv1InstanceTerms))
	for _, term := range inv1InstanceTerms {
		inHelper[term] = true
	}
	inRegistry := make(map[string]bool, len(registry))
	for _, term := range registry {
		inRegistry[term] = true
	}
	for _, term := range registry {
		if !inHelper[term] {
			t.Errorf("registry names %q in the single-instance equation but assertINV1Conservation does not sum it", term)
		}
	}
	for _, term := range inv1InstanceTerms {
		if !inRegistry[term] {
			t.Errorf("assertINV1Conservation sums %q but the registry's single-instance equation does not name it", term)
		}
	}
}

// TestINV1Accounted verifies every term contributes to the sum. Distinct powers of
// two make a dropped term identifiable from the total alone.
func TestINV1Accounted(t *testing.T) {
	m := &Metrics{
		CompletedRequests: 1,
		StillQueued:       2,
		StillRunning:      4,
		DroppedUnservable: 8,
		TimedOutRequests:  16,
	}
	if got, want := inv1Accounted(m), 31; got != want {
		t.Errorf("inv1Accounted() = %d, want %d (got bits: %05b)", got, want, got)
	}
	if len(inv1InstanceTerms) != 5 {
		t.Errorf("inv1InstanceTerms names %d terms, want 5", len(inv1InstanceTerms))
	}
}
