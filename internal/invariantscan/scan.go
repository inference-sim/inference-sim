// Package invariantscan finds hand-rolled INV-1 conservation sums in Go source.
//
// It exists because the same guard is needed from both sim and sim/cluster, and a
// helper defined in a _test.go file is invisible across package boundaries. It is test
// tooling: nothing in the simulator calls it.
//
// It sits at the repository root rather than under sim/internal so that cmd could import
// it if a use arose. There is none today, and cmd is not a consumer: its only
// conservation check parses stdout JSON field names, which no AST walk over Go
// expressions would see.
//
// Why a guard at all: before issue #1720 there were 29 hand-rolled conservation sums
// in the tree, disagreeing about how many terms INV-1 has, and three omitted a bucket
// outright. Those are now shared helpers, and this keeps sum number 30 from being
// written — adding a bucket to INV-1 should be one edit, not a manual search.
package invariantscan

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// instanceBuckets are the five terminal buckets of INV-1's single-instance
// specialisation, as Go field names.
var instanceBuckets = []string{
	"CompletedRequests",
	"StillQueued",
	"StillRunning",
	"DroppedUnservable",
	"TimedOutRequests",
}

// clusterOnlyBuckets are the buckets that exist only above a single instance, as Go
// field or accessor names. RejectedRequests is included: it is not one of INV-1's
// twelve terminal buckets but appears on the right-hand side of the full-pipeline
// clause, and a sum combining it with the others is still a hand-rolled ledger.
var clusterOnlyBuckets = []string{
	"RoutingRejections",
	"GatewayQueueDepth",
	"GatewayQueueShed",
	"GatewayQueueRejected",
	"GatewayEvicted",
	"GatewayExpired",
	"EncodeRoutingRejections",
	"RejectedRequests",
}

// InstanceBuckets returns the five single-instance bucket names. A copy, because a
// package-level exported slice is mutable global state (R8) and a caller that
// reordered or truncated it would silently change what every guard detects.
func InstanceBuckets() []string { return append([]string(nil), instanceBuckets...) }

// ClusterOnlyBuckets returns the cluster-only bucket names, as a copy. See
// InstanceBuckets.
func ClusterOnlyBuckets() []string { return append([]string(nil), clusterOnlyBuckets...) }

// Finding is one hand-rolled sum.
type Finding struct {
	File    string
	Line    int
	Buckets []string
}

func (f Finding) String() string {
	return fmt.Sprintf("%s:%d (%d buckets: %v)", f.File, f.Line, len(f.Buckets), f.Buckets)
}

func bucketSet() map[string]bool {
	set := make(map[string]bool, len(instanceBuckets)+len(clusterOnlyBuckets))
	for _, name := range instanceBuckets {
		set[name] = true
	}
	for _, name := range clusterOnlyBuckets {
		set[name] = true
	}
	return set
}

// bucketNameOf reports the bucket that e reads, if any: a field selector
// (m.StillQueued) or an accessor call (cs.GatewayExpired()).
func bucketNameOf(e ast.Expr, buckets map[string]bool) (string, bool) {
	switch x := e.(type) {
	case *ast.SelectorExpr:
		if buckets[x.Sel.Name] {
			return x.Sel.Name, true
		}
	case *ast.CallExpr:
		if sel, ok := x.Fun.(*ast.SelectorExpr); ok && buckets[sel.Sel.Name] {
			return sel.Sel.Name, true
		}
	}
	return "", false
}

// exprKey renders an assignment target as a stable string, so an accumulator that is a
// struct field (`agg.total += m.StillQueued`) is tracked like a plain local. Anything
// more complex than an identifier or a chain of selectors is untrackable and returns
// false.
func exprKey(e ast.Expr) (string, bool) {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name, true
	case *ast.SelectorExpr:
		prefix, ok := exprKey(x.X)
		if !ok {
			return "", false
		}
		return prefix + "." + x.Sel.Name, true
	}
	return "", false
}

// bucketsIn returns the distinct buckets appearing as operands of the
// arithmetic/comparison tree rooted at e, resolving locals through aliases.
//
// Subtraction and the comparison operators are walked as well as addition: moving
// buckets to the other side of a `!=` is the natural rewrite once a plain sum is
// rejected, and it is the same equation.
//
// Known gap: a composite literal (`for _, v := range []int{m.CompletedRequests, ...}`)
// is not walked. Adding that case would also flag legitimate struct literals that
// enumerate metric fields, so it is left out deliberately; a contributor would have to
// go well out of their way to launder a ledger through one.
func bucketsIn(e ast.Expr, aliases map[string]map[string]bool, buckets map[string]bool) map[string]bool {
	seen := map[string]bool{}
	var walk func(ast.Expr)
	walk = func(e ast.Expr) {
		if name, ok := bucketNameOf(e, buckets); ok {
			seen[name] = true
			return
		}
		switch x := e.(type) {
		case *ast.BinaryExpr:
			switch x.Op {
			case token.ADD, token.SUB, token.EQL, token.NEQ:
				walk(x.X)
				walk(x.Y)
			}
		case *ast.ParenExpr:
			walk(x.X)
		case *ast.Ident, *ast.SelectorExpr:
			if key, ok := exprKey(x); ok {
				for bucket := range aliases[key] {
					seen[bucket] = true
				}
			}
		}
	}
	walk(e)
	return seen
}

// isLedger reports whether a set of buckets constitutes a conservation ledger rather
// than some other multi-bucket expression.
//
// The bucket count alone is not enough. Two things in the tree combine several
// cluster-only counters without being conservation sums: INV-5's non-vacuity floors
// subtract gateway counters from a request count, and progress_hook_test.go
// reconciles the per-tier shed breakdown against the shedding counters. Flagging
// those would earn this guard an exemption list, and an exemption list is how a guard
// stops meaning anything.
//
// What separates a real ledger is that it accounts for where requests *ended*, so it
// always reads the instance-side buckets. Requiring CompletedRequests alone would be
// too narrow — a short-horizon fixture where nothing completes can express a genuine
// ledger without it — so the rule is CompletedRequests, or two or more instance-side
// buckets. Neither non-ledger expression above reads any.
func isLedger(seen map[string]bool, threshold int) bool {
	if len(seen) < threshold {
		return false
	}
	if seen["CompletedRequests"] {
		return true
	}
	instanceSide := 0
	for _, name := range instanceBuckets {
		if seen[name] {
			instanceSide++
		}
	}
	return instanceSide >= 2
}

// DefaultThreshold is the number of distinct buckets an expression must combine before
// it counts as a conservation ledger. Exported so every guard provably shares one value:
// if one package bumped its own copy to 4 it would silently stop detecting the
// three-bucket sums this package exists to catch.
const DefaultThreshold = 3

// FindConservationSums returns every hand-rolled INV-1 ledger in src.
//
// Detection is flow-ordered: aliases are built and findings recorded in a single pass in
// source order, so a finding is recorded at the moment an accumulator crosses the
// threshold. That ordering is load-bearing. A two-pass design — build all aliases, then
// judge every expression against the finished map — lets a later statement retract
// earlier evidence: `total = 0` after three `+=` lines deletes the alias entry, and the
// completed map then shows an empty set for every preceding line, so a real ledger
// reports nothing. Reusing an accumulator name for a second sum is ordinary code, not
// deliberate evasion, which is what made that gap worth closing.
func FindConservationSums(filename, src string, threshold int) ([]Finding, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, src, 0)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", filename, err)
	}

	buckets := bucketSet()

	var findings []Finding
	report := func(pos token.Pos, seen map[string]bool) {
		names := make([]string, 0, len(seen))
		for _, name := range append(InstanceBuckets(), clusterOnlyBuckets...) {
			if seen[name] {
				names = append(names, name)
			}
		}
		findings = append(findings, Finding{File: filename, Line: fset.Position(pos).Line, Buckets: names})
	}

	// One alias scope per function body: a file-wide scope would let a `:=` in one test
	// wipe the set a different test had built for the same name, and `m`, `total` and
	// `agg` recur constantly in these files. Declarations outside any function get their
	// own scope.
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Body != nil {
			scanScope(fn.Body, buckets, threshold, report)
			continue
		}
		scanScope(decl, buckets, threshold, report)
	}
	return findings, nil
}

// scanScope walks one alias scope in source order, maintaining aliases as it goes and
// reporting a ledger the moment it is complete.
func scanScope(
	scope ast.Node,
	buckets map[string]bool,
	threshold int,
	reportRaw func(token.Pos, map[string]bool),
) {
	aliases := map[string]map[string]bool{}
	// One finding per accumulator: an accumulator crosses the threshold once but is
	// written by several `+=` lines, and the ones after the crossing add nothing.
	reported := map[string]bool{}
	// One finding per distinct bucket set in this scope. A hand-rolled ledger is
	// typically detected twice — once when the accumulator completes, once at the
	// comparison that uses it — and reporting both just doubles the noise for a single
	// thing to fix.
	reportedSets := map[string]bool{}

	report := func(pos token.Pos, seen map[string]bool) {
		names := make([]string, 0, len(seen))
		for name := range seen {
			names = append(names, name)
		}
		sort.Strings(names)
		key := strings.Join(names, "+")
		if reportedSets[key] {
			return
		}
		reportedSets[key] = true
		reportRaw(pos, seen)
	}

	// assign updates the alias set for one target. accumulate distinguishes `+=`, which
	// merges into whatever the target already carried, from `=` and `:=`, which replace it.
	assign := func(target ast.Expr, value ast.Expr, accumulate bool) (string, bool) {
		key, ok := exprKey(target)
		if !ok {
			return "", false
		}
		carried := bucketsIn(value, aliases, buckets)
		if !accumulate {
			delete(aliases, key)
			delete(reported, key)
		}
		if len(carried) == 0 && !accumulate {
			return key, false
		}
		if aliases[key] == nil {
			aliases[key] = map[string]bool{}
		}
		for bucket := range carried {
			aliases[key][bucket] = true
		}
		return key, true
	}

	ast.Inspect(scope, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			accumulate := x.Tok == token.ADD_ASSIGN || x.Tok == token.SUB_ASSIGN
			for i, lhs := range x.Lhs {
				if i >= len(x.Rhs) {
					break
				}
				key, tracked := assign(lhs, x.Rhs[i], accumulate)
				if !tracked || reported[key] {
					continue
				}
				if isLedger(aliases[key], threshold) {
					reported[key] = true
					report(x.Pos(), aliases[key])
				}
			}
			// Still descend: the right-hand side may itself be a complete ledger
			// expression, which the BinaryExpr case below reports.
			return true

		case *ast.ValueSpec:
			for i, name := range x.Names {
				if i >= len(x.Values) {
					break
				}
				key, tracked := assign(name, x.Values[i], false)
				if !tracked || reported[key] {
					continue
				}
				if isLedger(aliases[key], threshold) {
					reported[key] = true
					report(x.Pos(), aliases[key])
				}
			}
			return true

		case *ast.BinaryExpr:
			switch x.Op {
			case token.ADD, token.SUB, token.EQL, token.NEQ:
			default:
				return true
			}
			seen := bucketsIn(x, aliases, buckets)
			if isLedger(seen, threshold) {
				report(x.Pos(), seen)
				// Sub-expressions of a matched ledger would match too.
				return false
			}
		}
		return true
	})
}

// ScanTestDir scans every *_test.go file in dir for hand-rolled ledgers, skipping
// selfFile and any file named in exempt. It returns the findings, the number of files
// actually scanned, and any stale exemption entries (names that no longer exist).
//
// The walk lives here rather than in each package's guard because the two guards were
// copy-pasted and had already diverged — one sorted with a generic helper and the other
// hand-rolled the same sort, one held its exemption map at package level and the other
// function-local. That is the R23 parallel-path shape, in the very code meant to prevent
// a parallel path. Each caller still owns its own non-vacuity assertions, which is why
// the scanned count and the stale list come back rather than being asserted here.
func ScanTestDir(dir, selfFile string, exempt map[string]string, threshold int) (findings []Finding, scanned int, stale []string, err error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, 0, nil, fmt.Errorf("read dir %s: %w", dir, err)
	}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, "_test.go") || name == selfFile {
			continue
		}
		if _, ok := exempt[name]; ok {
			continue
		}
		src, readErr := os.ReadFile(filepath.Join(dir, name))
		if readErr != nil {
			return nil, 0, nil, fmt.Errorf("read %s: %w", name, readErr)
		}
		scanned++
		found, scanErr := FindConservationSums(name, string(src), threshold)
		if scanErr != nil {
			return nil, 0, nil, scanErr
		}
		findings = append(findings, found...)
	}

	names := make([]string, 0, len(exempt))
	for name := range exempt {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if _, statErr := os.Stat(filepath.Join(dir, name)); statErr != nil {
			stale = append(stale, name)
		}
	}
	return findings, scanned, stale, nil
}
