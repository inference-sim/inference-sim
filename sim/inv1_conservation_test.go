package sim

import (
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/internal/invariantscan"
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

// TestINV1_InstanceHelperMatchesRegistry asserts this package's five-term list is
// the one the registry states as the single-instance specialisation. The cluster
// package runs the eleven-term half of the same check; between them the two copies
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
		seen := map[string]bool{}
		for _, term := range strings.Split(rhs, "+") {
			term = strings.TrimSpace(term)
			// A duplicated term would survive the set comparison below, while making
			// the stated equation wrong.
			if seen[term] {
				t.Errorf("%s states %q twice in one equation", registryPath, term)
			}
			seen[term] = true
			terms = append(terms, term)
		}
		equations = append(equations, terms)
	}
	if len(equations) != 2 {
		t.Fatalf("expected exactly 2 `injected_requests == ...` equations in %s, found %d — the registry was restructured, so this test and the helper need re-checking against it",
			registryPath, len(equations))
	}

	// Pick the shorter of the two by term count rather than by document order, so
	// reordering the registry paragraphs cannot silently compare the wrong equation.
	registry := equations[0]
	if len(equations[1]) < len(registry) {
		registry = equations[1]
	}
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

// TestINV1_NoInlineConservationSums is the sim/ counterpart of the guard in
// sim/cluster. Both call the same scanner: a line-based check here would miss any
// gofmt-wrapped sum, which is the shape the cluster guard exists to catch.
func TestINV1_NoInlineConservationSums(t *testing.T) {
	// Files allowed to hand-roll the sum, with the reason.
	exempt := map[string]string{
		// Asserts over MetricsOutput (the JSON serialisation), not *Metrics, so the
		// helper's signature does not apply. It checks that ToOutput's field mapping is
		// complete, which is a different property from conservation.
		"metrics_test.go": "asserts over MetricsOutput fields, not *Metrics",
	}

	findings, scanned, stale, err := invariantscan.ScanTestDir(".", "inv1_conservation_test.go", exempt, invariantscan.DefaultThreshold)
	if err != nil {
		t.Fatalf("scan failed: %v", err)
	}
	for _, f := range findings {
		t.Errorf("%s: inline INV-1 conservation sum — use assertINV1Conservation instead, so a bucket added to the invariant is picked up here automatically", f)
	}
	if scanned == 0 {
		t.Fatal("scanned no test files — the directory walk or the filter is broken, so this test proves nothing")
	}
	for _, name := range stale {
		t.Errorf("exemption for %s (%s) names a file that does not exist — remove the stale entry", name, exempt[name])
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
