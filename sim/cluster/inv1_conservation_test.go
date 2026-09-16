package cluster

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// clusterLedgerTerms lists INV-1's twelve cluster-level bucket names in the order
// the registry states them. TestINV1_HelperMatchesRegistry asserts this equals the
// terms parsed out of docs/contributing/standards/invariants.md, so the helper
// cannot drift from the statement it claims to enforce.
var clusterLedgerTerms = []string{
	"completed_requests",
	"still_queued",
	"still_running",
	"dropped_unservable",
	"timed_out",
	"routing_rejections",
	"gateway_queue_depth",
	"gateway_queue_shed",
	"gateway_queue_rejected",
	"gateway_evicted",
	"gateway_expired",
	"encode_routing_rejections",
}

// clusterLedger holds the twelve terminal buckets of INV-1's canonical
// cluster-level equation. It is a plain struct so the equation can be unit-tested
// and diffed against the registry without running a simulation.
type clusterLedger struct {
	completedRequests       int
	stillQueued             int
	stillRunning            int
	droppedUnservable       int
	timedOut                int
	routingRejections       int
	gatewayQueueDepth       int
	gatewayQueueShed        int
	gatewayQueueRejected    int
	gatewayEvicted          int
	gatewayExpired          int
	encodeRoutingRejections int
}

// terms returns the bucket values in clusterLedgerTerms order. accounted() and
// String() both read this slice rather than listing fields again, so a bucket
// added to the struct and to clusterLedgerTerms but forgotten here is caught by
// TestClusterLedger_TermsMatchNames instead of silently dropping out of the sum.
func (l clusterLedger) terms() []int {
	return []int{
		l.completedRequests,
		l.stillQueued,
		l.stillRunning,
		l.droppedUnservable,
		l.timedOut,
		l.routingRejections,
		l.gatewayQueueDepth,
		l.gatewayQueueShed,
		l.gatewayQueueRejected,
		l.gatewayEvicted,
		l.gatewayExpired,
		l.encodeRoutingRejections,
	}
}

func (l clusterLedger) accounted() int {
	total := 0
	for _, v := range l.terms() {
		total += v
	}
	return total
}

func (l clusterLedger) String() string {
	parts := make([]string, 0, len(clusterLedgerTerms))
	for i, v := range l.terms() {
		parts = append(parts, fmt.Sprintf("%s=%d", clusterLedgerTerms[i], v))
	}
	return strings.Join(parts, " ")
}

func newClusterLedger(cs *ClusterSimulator) clusterLedger {
	m := cs.AggregatedMetrics()
	return clusterLedger{
		completedRequests:       m.CompletedRequests,
		stillQueued:             m.StillQueued,
		stillRunning:            m.StillRunning,
		droppedUnservable:       m.DroppedUnservable,
		timedOut:                m.TimedOutRequests,
		routingRejections:       cs.RoutingRejections(),
		gatewayQueueDepth:       cs.GatewayQueueDepth(),
		gatewayQueueShed:        cs.GatewayQueueShed(),
		gatewayQueueRejected:    cs.GatewayQueueRejected(),
		gatewayEvicted:          cs.GatewayEvicted(),
		gatewayExpired:          cs.GatewayExpired(),
		encodeRoutingRejections: cs.EncodeRoutingRejections(),
	}
}

// assertClusterINV1Conservation asserts both clauses of INV-1 at cluster level:
//
//	total    == injected + rejected   (the full-pipeline clause)
//	injected == <the twelve buckets>  (the canonical equation)
//
// total and rejected are taken separately, rather than a pre-subtracted injected,
// so the pipeline clause is asserted at every call site instead of assumed there.
//
// total must come from a source independent of the metrics under test — normally
// len(requests). That identity holds only when (a) the request source is an eager
// slice, (b) every ArrivalTime is inside the horizon, since a beyond-horizon
// ClusterArrivalEvent never executes and its request enters no bucket, and (c) no
// session follow-ups were injected, since those were never in the slice. A caller
// that breaks any of the three must compute total itself.
//
// This does not assert INV-1's exclusivity clause ("a request lands in exactly one
// bucket"): a double-count in one bucket cancels a loss from another, so no
// aggregate sum can detect it. Tracked separately.
func assertClusterINV1Conservation(t *testing.T, cs *ClusterSimulator, total, rejected int, label string) {
	t.Helper()
	l := newClusterLedger(cs)
	injected := total - rejected
	if got := l.accounted(); got != injected {
		t.Errorf("INV-1 conservation violated (%s): injected=%d (total=%d - rejected=%d) != accounted=%d [%s]",
			label, injected, total, rejected, got, l)
	}
}

// instanceLedgerTerms lists the five terms of INV-1's single-instance
// specialisation, in registry order.
var instanceLedgerTerms = []string{
	"completed_requests",
	"still_queued",
	"still_running",
	"dropped_unservable",
	"timed_out",
}

func instanceAccounted(m *sim.Metrics) int {
	return m.CompletedRequests + m.StillQueued + m.StillRunning +
		m.DroppedUnservable + m.TimedOutRequests
}

// assertInstanceINV1Conservation asserts the five-term single-instance
// specialisation of INV-1. Valid only against per-instance metrics: a cluster has
// a router, a gateway queue and possibly an encode pool, whose seven buckets this
// equation omits. Cluster-level callers must use assertClusterINV1Conservation,
// and TestINV1_NoInlineClusterConservationSums flags calls made against a cluster
// aggregate.
func assertInstanceINV1Conservation(t *testing.T, m *sim.Metrics, injected int, label string) {
	t.Helper()
	if got := instanceAccounted(m); got != injected {
		t.Errorf("INV-1 conservation violated (%s): injected=%d != accounted=%d "+
			"(completed=%d still_queued=%d still_running=%d dropped_unservable=%d timed_out=%d)",
			label, injected, got, m.CompletedRequests, m.StillQueued, m.StillRunning,
			m.DroppedUnservable, m.TimedOutRequests)
	}
}

// invariantsRegistryPath is the INV-1 statement's source of truth, relative to this
// package directory.
const invariantsRegistryPath = "../../docs/contributing/standards/invariants.md"

// parseINV1RegistryEquations returns the term lists of INV-1's two stated forms as
// written in the registry: the canonical twelve-term cluster equation first, then
// the five-term single-instance specialisation.
//
// Both are whole-line backticked expressions whose left-hand side is
// `injected_requests ==`. That anchor matters: the registry carries a third
// backticked equation, the full-pipeline clause `num_requests == injected_requests +
// rejected_requests`, and a looser match would latch onto it and compare the wrong
// term set.
//
// Duplicated deliberately in sim/inv1_conservation_test.go — a _test.go identifier
// is invisible across packages, and each package must be able to check its own
// helper. #1735 may later add a shared registry locator; reuse it if so.
func parseINV1RegistryEquations(t *testing.T, path string) [][]string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("cannot read the invariant registry at %s: %v", path, err)
	}
	var found [][]string
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
		found = append(found, terms)
	}
	if len(found) != 2 {
		t.Fatalf("expected exactly 2 `injected_requests == ...` equations in %s (the twelve-term cluster form and the five-term single-instance form), found %d — the registry was restructured, so this test and the helpers need re-checking against it",
			path, len(found))
	}
	return found
}

// TestINV1_HelperMatchesRegistry asserts the helper's term list is the registry's
// term list. The registry says a shared conservation helper "should be generated
// from the twelve-term list above"; nothing enforced that, so a bucket added to the
// statement could sit there for months with no test asserting it — which is how
// issue #1720's findings arose in the first place.
func TestINV1_HelperMatchesRegistry(t *testing.T) {
	eqs := parseINV1RegistryEquations(t, invariantsRegistryPath)

	for _, tc := range []struct {
		name     string
		registry []string
		helper   []string
	}{
		{"twelve-term cluster equation", eqs[0], clusterLedgerTerms},
		{"five-term single-instance specialisation", eqs[1], instanceLedgerTerms},
	} {
		inRegistry := make(map[string]bool, len(tc.registry))
		for _, term := range tc.registry {
			inRegistry[term] = true
		}
		inHelper := make(map[string]bool, len(tc.helper))
		for _, term := range tc.helper {
			inHelper[term] = true
		}
		for _, term := range tc.registry {
			if !inHelper[term] {
				t.Errorf("%s: registry names %q but the helper does not sum it — INV-1 declares a bucket no test checks",
					tc.name, term)
			}
		}
		for _, term := range tc.helper {
			if !inRegistry[term] {
				t.Errorf("%s: the helper sums %q but the registry does not name it — the helper and the statement disagree",
					tc.name, term)
			}
		}
	}
}

// TestClusterLedger_TermsMatchNames pins the two lists the rest of the file reads
// in lockstep: a bucket added to clusterLedgerTerms without a matching entry in
// terms() would otherwise be excluded from accounted() with no test failing.
func TestClusterLedger_TermsMatchNames(t *testing.T) {
	if got, want := len(clusterLedger{}.terms()), len(clusterLedgerTerms); got != want {
		t.Fatalf("terms() returns %d values but clusterLedgerTerms names %d — a bucket is missing from one of them", got, want)
	}
}

// TestClusterLedger_Accounted verifies every bucket contributes to accounted().
// Distinct powers of two make a dropped term identifiable from the sum alone.
func TestClusterLedger_Accounted(t *testing.T) {
	l := clusterLedger{
		completedRequests:       1,
		stillQueued:             2,
		stillRunning:            4,
		droppedUnservable:       8,
		timedOut:                16,
		routingRejections:       32,
		gatewayQueueDepth:       64,
		gatewayQueueShed:        128,
		gatewayQueueRejected:    256,
		gatewayEvicted:          512,
		gatewayExpired:          1024,
		encodeRoutingRejections: 2048,
	}
	if got, want := l.accounted(), 4095; got != want {
		t.Errorf("accounted() = %d, want %d — a bucket is missing from the sum (got bits: %012b)", got, want, got)
	}
	for i, v := range l.terms() {
		if v == 0 {
			t.Errorf("terms()[%d] (%s) is zero — the fixture or the field mapping is wrong", i, clusterLedgerTerms[i])
		}
	}
}

// TestClusterLedger_StringNamesEveryTerm verifies the failure message carries all
// twelve buckets, so a violation is diagnosable without re-running under a
// debugger.
func TestClusterLedger_StringNamesEveryTerm(t *testing.T) {
	s := clusterLedger{gatewayExpired: 7}.String()
	for _, name := range clusterLedgerTerms {
		if !strings.Contains(s, name+"=") {
			t.Errorf("String() omits %q: %s", name, s)
		}
	}
	if !strings.Contains(s, "gateway_expired=7") {
		t.Errorf("String() lost a value: %s", s)
	}
}

// TestInstanceAccounted verifies the five-term specialisation sums every bucket.
func TestInstanceAccounted(t *testing.T) {
	m := &sim.Metrics{
		CompletedRequests: 1,
		StillQueued:       2,
		StillRunning:      4,
		DroppedUnservable: 8,
		TimedOutRequests:  16,
	}
	if got, want := instanceAccounted(m), 31; got != want {
		t.Errorf("instanceAccounted() = %d, want %d (got bits: %05b)", got, want, got)
	}
}
