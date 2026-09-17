package cluster

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/internal/invariantscan"
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

// noRejections is the rejected argument for a fixture that must not reject
// anything. Passing it rather than cs.RejectedRequests() means a spurious rejection
// breaks the equation instead of being silently subtracted out — which is what the
// pre-#1720 sums did implicitly by comparing against a bare fixture count.
const noRejections = 0

// assertClusterINV1Conservation asserts INV-1's canonical twelve-term equation at
// cluster level:
//
//	total - rejected == <the twelve buckets>
//
// rejected is taken separately from total rather than pre-subtracted by the caller,
// so a fixture that must not reject anything can pass noRejections and have a
// spurious rejection fail the assertion. Note what this is NOT: folding the
// full-pipeline clause (total == injected + rejected) into the same equality means a
// failure cannot be attributed to one clause or the other. Where the pipeline clause
// needs checking against an independent observation of injected — the length of the
// Metrics.Requests map — the call site does that separately; the map is not a
// general-purpose baseline: the drop guards, drain redirect and PD parent collapse
// delete from it, while the timeout paths do not delete at all, so it can under- or
// overcount depending on the fixture.
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
// aggregate sum can detect it. Tracked in #1745.
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

// sortedStringKeys returns a map's keys in sorted order, so a test that reports several
// failures reports them in a stable order (R2). Named to avoid colliding with
// sortedKeys in metrics.go, which is float64-specific.
func sortedStringKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
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
		seen := map[string]bool{}
		for _, term := range strings.Split(rhs, "+") {
			term = strings.TrimSpace(term)
			// A duplicated term would survive the set comparison below, while making
			// the stated equation wrong.
			if seen[term] {
				t.Errorf("%s states %q twice in one equation", path, term)
			}
			seen[term] = true
			terms = append(terms, term)
		}
		found = append(found, terms)
	}
	if len(found) != 2 {
		t.Fatalf("expected exactly 2 `injected_requests == ...` equations in %s (the twelve-term cluster form and the five-term single-instance form), found %d — the registry was restructured, so this test and the helpers need re-checking against it",
			path, len(found))
	}
	// Select by term count rather than document order, so reordering the two
	// paragraphs in the registry does not silently compare the wrong pair.
	sort.Slice(found, func(i, j int) bool { return len(found[i]) > len(found[j]) })
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

// inlineSumExemptions lists files allowed to hand-roll a conservation sum, with the
// reason. Every entry must name a file that exists, so an exemption left behind after
// a rewrite fails rather than rots. Empty: invariantscan distinguishes a conservation
// ledger from the other multi-bucket expressions in this package, so no file needs
// excusing.
var inlineSumExemptions = map[string]string{}

// TestINV1_NoInlineClusterConservationSums forbids new hand-rolled cluster
// conservation sums. Before issue #1720 there were 29 of them, disagreeing about
// how many terms conservation has, and three omitted a bucket outright. The shared
// helper only stays authoritative if writing sum number 30 fails.
//
// Detection is AST-based rather than a line grep: the sums this replaced were
// gofmt-wrapped across lines, accumulated through intermediate locals, and varied
// their operand order, all of which a textual match misses or can be reformatted
// around.
func TestINV1_NoInlineClusterConservationSums(t *testing.T) {
	findings, scanned, stale, err := invariantscan.ScanTestDir(".", "inv1_conservation_test.go", inlineSumExemptions, invariantscan.DefaultThreshold)
	if err != nil {
		t.Fatalf("scan failed: %v", err)
	}
	for _, f := range findings {
		t.Errorf("%s: inline INV-1 conservation sum — use assertClusterINV1Conservation instead, so a bucket added to the invariant is picked up here automatically", f)
	}
	// Non-vacuity: a broken walk or filter must not read as a clean pass.
	if scanned == 0 {
		t.Fatal("scanned no test files — the directory walk or the filter is broken, so this test proves nothing")
	}
	for _, name := range stale {
		t.Errorf("exemption for %s (%s) names a file that does not exist — remove the stale entry", name, inlineSumExemptions[name])
	}
}

// collectAggregateLocals returns the local names assigned from AggregatedMetrics(),
// covering both `m := cs.AggregatedMetrics()` and `var m = cs.AggregatedMetrics()`.
// The var form matters because invariantscan handles it, and a guard that silently
// covered one spelling but not the other would be the same half-coverage this PR exists
// to remove.
func collectAggregateLocals(file *ast.File) map[string]bool {
	locals := map[string]bool{}
	record := func(lhs, rhs []ast.Expr) {
		for i, l := range lhs {
			id, ok := l.(*ast.Ident)
			if !ok || i >= len(rhs) {
				continue
			}
			if isAggregateMetricsCall(rhs[i]) {
				locals[id.Name] = true
			}
		}
	}
	ast.Inspect(file, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			record(x.Lhs, x.Rhs)
		case *ast.ValueSpec:
			lhs := make([]ast.Expr, 0, len(x.Names))
			for _, name := range x.Names {
				lhs = append(lhs, name)
			}
			record(lhs, x.Values)
		}
		return true
	})
	return locals
}

// isAggregateMetricsCall reports whether e is a call to AggregatedMetrics(), the
// cluster-level metrics accessor.
func isAggregateMetricsCall(e ast.Expr) bool {
	call, ok := e.(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	return ok && sel.Sel.Name == "AggregatedMetrics"
}

// TestINV1_NoInstanceHelperOnClusterMetrics stops the five-term specialisation from
// becoming a sanctioned weak cluster assertion. The AST guard above only sees
// inline sums, so a cluster test that called assertInstanceINV1Conservation with
// cs.AggregatedMetrics() would assert five of twelve terms with nothing
// complaining — the same gap in a new shape.
func TestINV1_NoInstanceHelperOnClusterMetrics(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("cannot list the package directory: %v", err)
	}
	fset := token.NewFileSet()
	scanned := 0
	callsFound := 0
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, "_test.go") || name == "inv1_conservation_test.go" {
			continue
		}
		file, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("cannot parse %s: %v", name, err)
		}
		scanned++
		// Locals assigned from AggregatedMetrics(). `m := cs.AggregatedMetrics()`
		// followed by `assertInstanceINV1Conservation(t, m, ...)` is the idiomatic way
		// to write this violation, so matching only the inline call would leave the
		// guard checking the one spelling nobody uses.
		aggregateLocals := collectAggregateLocals(file)
		ast.Inspect(file, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			id, ok := call.Fun.(*ast.Ident)
			if !ok || id.Name != "assertInstanceINV1Conservation" {
				return true
			}
			callsFound++
			for _, arg := range call.Args {
				clusterAggregate := isAggregateMetricsCall(arg)
				if ident, ok := arg.(*ast.Ident); ok && aggregateLocals[ident.Name] {
					clusterAggregate = true
				}
				if clusterAggregate {
					t.Errorf("%s:%d: assertInstanceINV1Conservation applied to a cluster aggregate — use assertClusterINV1Conservation, which checks all twelve buckets",
						name, fset.Position(call.Pos()).Line)
				}
			}
			return true
		})
	}
	if scanned == 0 {
		t.Fatal("scanned no test files — the directory walk is broken, so this test proves nothing")
	}
	// Non-vacuity: with no call sites at all the inner pass iterates nothing, and a
	// broken matcher is indistinguishable from a clean tree. instance_test.go calls the
	// helper, so at least one call must be seen.
	if callsFound == 0 {
		t.Fatal("found no assertInstanceINV1Conservation call anywhere — either the helper is unused (then delete it) or the matcher is broken, and either way this guard proves nothing")
	}
}

// TestINV1_AggregateLocalDetection proves the alias resolution in
// TestINV1_NoInstanceHelperOnClusterMetrics works, since that guard has nothing to
// find in a clean tree and would otherwise pass whether or not it functioned.
func TestINV1_AggregateLocalDetection(t *testing.T) {
	for _, tc := range []struct {
		name string
		src  string
	}{
		{
			name: "short variable declaration",
			src: `package p
func f() {
	m := cs.AggregatedMetrics()
	assertInstanceINV1Conservation(t, m, 5, "should be flagged")
}`,
		},
		{
			name: "var declaration",
			src: `package p
func f() {
	var m = cs.AggregatedMetrics()
	assertInstanceINV1Conservation(t, m, 5, "should be flagged")
}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, "fixture.go", tc.src, 0)
			if err != nil {
				t.Fatalf("cannot parse fixture: %v", err)
			}
			if !collectAggregateLocals(file)["m"] {
				t.Error("alias resolution missed the aggregate local, so the guard would not flag this violation")
			}
		})
	}
}

// ledgerFieldSources is the mapping newClusterLedger must implement: every ledger
// field and the exact metric or accessor it reads. Kept separate from the constructor
// so TestNewClusterLedger_FieldSources can compare the two — a swap between fields is
// invisible to any assertion over totals, because two buckets that are zero together
// in every fixture would still balance.
var ledgerFieldSources = map[string]string{
	"completedRequests":       "CompletedRequests",
	"stillQueued":             "StillQueued",
	"stillRunning":            "StillRunning",
	"droppedUnservable":       "DroppedUnservable",
	"timedOut":                "TimedOutRequests",
	"routingRejections":       "RoutingRejections",
	"gatewayQueueDepth":       "GatewayQueueDepth",
	"gatewayQueueShed":        "GatewayQueueShed",
	"gatewayQueueRejected":    "GatewayQueueRejected",
	"gatewayEvicted":          "GatewayEvicted",
	"gatewayExpired":          "GatewayExpired",
	"encodeRoutingRejections": "EncodeRoutingRejections",
}

// TestNewClusterLedger_FieldSources pins each ledger field to the metric it reads.
// Nothing else can: swap the sources of two buckets that are zero together in every
// fixture — gateway_evicted and gateway_expired, say — and every conservation
// assertion in the tree still passes. This compares the constructor's composite
// literal, key by key, against the table above.
func TestNewClusterLedger_FieldSources(t *testing.T) {
	const file = "inv1_conservation_test.go"
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, file, nil, 0)
	if err != nil {
		t.Fatalf("cannot parse %s: %v", file, err)
	}

	var body *ast.BlockStmt
	ast.Inspect(parsed, func(n ast.Node) bool {
		fn, ok := n.(*ast.FuncDecl)
		if ok && fn.Name.Name == "newClusterLedger" {
			body = fn.Body
			return false
		}
		return true
	})
	if body == nil {
		t.Fatal("newClusterLedger not found — this test needs updating alongside the rename")
	}

	got := map[string]string{}
	ast.Inspect(body, func(n ast.Node) bool {
		kv, ok := n.(*ast.KeyValueExpr)
		if !ok {
			return true
		}
		key, ok := kv.Key.(*ast.Ident)
		if !ok {
			return true
		}
		switch v := kv.Value.(type) {
		case *ast.SelectorExpr:
			got[key.Name] = v.Sel.Name
		case *ast.CallExpr:
			if sel, ok := v.Fun.(*ast.SelectorExpr); ok {
				got[key.Name] = sel.Sel.Name
			}
		}
		return true
	})

	for _, field := range sortedStringKeys(ledgerFieldSources) {
		if want := ledgerFieldSources[field]; got[field] != want {
			t.Errorf("newClusterLedger sets %s from %q, want %q", field, got[field], want)
		}
	}
	for _, field := range sortedStringKeys(got) {
		if _, expected := ledgerFieldSources[field]; !expected {
			t.Errorf("newClusterLedger sets unexpected field %s from %q — add it to ledgerFieldSources and to clusterLedgerTerms", field, got[field])
		}
	}
	if len(ledgerFieldSources) != len(clusterLedgerTerms) {
		t.Errorf("ledgerFieldSources has %d entries but clusterLedgerTerms names %d buckets", len(ledgerFieldSources), len(clusterLedgerTerms))
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
