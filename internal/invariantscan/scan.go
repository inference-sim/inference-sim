// Package invariantscan finds hand-rolled INV-1 conservation sums in Go source.
//
// It exists because the same guard is needed from three packages — sim, sim/cluster
// and cmd — and a helper defined in a _test.go file is invisible across package
// boundaries. It lives at the repository root rather than under sim/internal so cmd
// can import it too. It is test tooling: nothing in the simulator calls it.
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

// aliasesOf maps local variable names to the set of buckets they carry, within a single
// function body.
//
// Two forms matter, and both were used by the sums this guard replaced:
// `completed := agg.CompletedRequests`, which launders one bucket into a local, and
// `total += m.StillQueued` in a loop or sequence, which accumulates several into one.
// Without the second, appending `+=` lines is a thirty-second way around the guard.
//
// Scoped per function rather than per file. A file-wide pass would let a `:=` in one
// test wipe the alias set a different test had built for the same variable name — `m`,
// `total` and `agg` recur constantly in these files — which judges an expression against
// aliases established elsewhere. That direction only produces false positives, but a
// false positive earns an exemption, and an exemption is what actually hides the next
// real violation.
func aliasesOf(scope ast.Node, buckets map[string]bool) map[string]map[string]bool {
	aliases := map[string]map[string]bool{}
	add := func(name, bucket string) {
		if aliases[name] == nil {
			aliases[name] = map[string]bool{}
		}
		aliases[name][bucket] = true
	}
	merge := func(name string, from map[string]bool) {
		for bucket := range from {
			add(name, bucket)
		}
	}

	record := func(lhs, rhs []ast.Expr, accumulate bool) {
		for i, l := range lhs {
			if i >= len(rhs) {
				return
			}
			id, ok := l.(*ast.Ident)
			if !ok {
				continue
			}
			if !accumulate {
				delete(aliases, id.Name)
			}
			merge(id.Name, bucketsIn(rhs[i], aliases, buckets))
		}
	}

	ast.Inspect(scope, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			record(x.Lhs, x.Rhs, x.Tok == token.ADD_ASSIGN || x.Tok == token.SUB_ASSIGN)
		case *ast.ValueSpec:
			lhs := make([]ast.Expr, 0, len(x.Names))
			for _, name := range x.Names {
				lhs = append(lhs, name)
			}
			record(lhs, x.Values, false)
		}
		return true
	})
	return aliases
}

// bucketsIn returns the distinct buckets appearing as operands of the
// arithmetic/comparison tree rooted at e, resolving locals through aliases.
//
// Subtraction and the comparison operators are walked as well as addition: moving
// buckets to the other side of a `!=` is the natural rewrite once a plain sum is
// rejected, and it is the same equation.
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
		case *ast.Ident:
			for bucket := range aliases[x.Name] {
				seen[bucket] = true
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

// FindConservationSums returns every hand-rolled INV-1 ledger in src. threshold is
// the number of distinct buckets an expression must combine to count; 3 is the value
// the guards use.
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

	// Walk each function body with its own alias scope. Declarations outside any
	// function (package-level vars) get a file-level scope of their own.
	scopes := []ast.Node{}
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Body != nil {
			scopes = append(scopes, fn.Body)
			continue
		}
		scopes = append(scopes, decl)
	}

	for _, scope := range scopes {
		aliases := aliasesOf(scope, buckets)
		inspectScope(scope, aliases, buckets, threshold, report)
	}
	return findings, nil
}

// inspectScope reports the ledgers inside one alias scope.
func inspectScope(
	scope ast.Node,
	aliases map[string]map[string]bool,
	buckets map[string]bool,
	threshold int,
	report func(token.Pos, map[string]bool),
) {
	// An accumulator crosses the threshold once but is written by several `+=` lines,
	// each of which would otherwise be reported. One finding per accumulator.
	reported := map[string]bool{}

	ast.Inspect(scope, func(n ast.Node) bool {
		switch x := n.(type) {
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
		case *ast.AssignStmt:
			// An accumulator built by `+=` holds no single expression with three
			// buckets, so the BinaryExpr case above never sees it. Flag the
			// assignment that pushes its alias set over the threshold.
			if x.Tok != token.ADD_ASSIGN && x.Tok != token.SUB_ASSIGN {
				return true
			}
			for _, lhs := range x.Lhs {
				id, ok := lhs.(*ast.Ident)
				if !ok {
					continue
				}
				if reported[id.Name] {
					continue
				}
				if isLedger(aliases[id.Name], threshold) {
					reported[id.Name] = true
					report(x.Pos(), aliases[id.Name])
					return false
				}
			}
		}
		return true
	})
}
