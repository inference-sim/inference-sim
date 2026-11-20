package result

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/metrics"
)

// A figure of a Result states its scope, member, aggregate and scope ranged over (design 8.7,
// DEV-8); known-bad figures, one refusal each.
func TestAResultFigureIsRefusedWithoutAValidBasis(t *testing.T) {
	for _, c := range []struct {
		name string
		set  func(*Figure)
		msg  string
	}{
		{"no scope", func(f *Figure) { f.Scope = "" }, "states no scope"},
		{"an unknown scope", func(f *Figure) { f.Scope = "rack" }, `scope "rack" is not one of`},
		{"a pool figure with no member", func(f *Figure) { f.Scope = metrics.ScopePool }, "names no member"},
		{"an instance figure with no member", func(f *Figure) { f.Scope = metrics.ScopeInstance }, "names no member"},
		{"a cluster figure with a member", func(f *Figure) { f.Member = "decode" }, "only a pool or an instance has one"},
		{"an aggregate with no scope to range over", func(f *Figure) { f.Aggregate = metrics.AggregateMax }, "ranges over no scope"},
		{"a scope to range over with no aggregate", func(f *Figure) { f.Over = metrics.ScopeInstance }, "names no aggregate"},
		{"an aggregate the metric does not admit", func(f *Figure) { f.Aggregate, f.Over = "mean", metrics.ScopeInstance }, "is not one of max, min"},
		{"a range over a scope that is not narrower", func(f *Figure) {
			f.Scope, f.Member, f.Aggregate, f.Over = metrics.ScopeInstance, "i0", metrics.AggregateMax, metrics.ScopePool
		}, "not narrower"},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := valid()
			f := r.Metrics["ttft_p99_ms"]
			c.set(&f)
			r.Metrics["ttft_p99_ms"] = f
			p := r.Validate()
			for _, it := range p.Errors() {
				if it.Path == "metrics.ttft_p99_ms" && strings.Contains(it.Message, c.msg) {
					return
				}
			}
			t.Errorf("expected a problem at metrics.ttft_p99_ms containing %q, got:\n%s", c.msg, p.Error())
		})
	}
}

// The design's own example is accepted: the largest p99 time to first token over instances is a
// figure at the cluster scope, aggregate max, over instance; and one pool's figure names the pool.
func TestAResultFigureMayAggregateOverANarrowerScope(t *testing.T) {
	for name, f := range map[string]Figure{
		"max over instances": {Value: 431.5, Samples: 2, Scope: metrics.ScopeCluster, Aggregate: metrics.AggregateMax, Over: metrics.ScopeInstance},
		"one pool":           {Value: 431.5, Samples: 2, Scope: metrics.ScopePool, Member: "decode"},
		"max over instances of one pool": {Value: 431.5, Samples: 2, Scope: metrics.ScopePool, Member: "decode",
			Aggregate: metrics.AggregateMax, Over: metrics.ScopeInstance},
	} {
		r := valid()
		r.Metrics["ttft_p99_ms"] = f
		if p := r.Validate(); !p.OK() {
			t.Errorf("%s was refused:\n%s", name, p.Error())
		}
	}
}

// Two figures compare only when the metric, scope, member, aggregate and scope ranged over are
// all the same; a basis that does not check is never comparable, even with an equal one.
func TestTwoBasesAreComparableOnlyWhenEveryFieldIsTheSame(t *testing.T) {
	d, _ := metrics.Lookup("ttft_p99_ms")
	cluster := Basis{Scope: metrics.ScopeCluster}
	worst := Basis{Scope: metrics.ScopeCluster, Aggregate: metrics.AggregateMax, Over: metrics.ScopeInstance}
	best := Basis{Scope: metrics.ScopeCluster, Aggregate: metrics.AggregateMin, Over: metrics.ScopeInstance}
	pool := func(m string) Basis { return Basis{Scope: metrics.ScopePool, Member: m} }
	if err := cluster.ComparableTo(cluster, d); err != nil {
		t.Errorf("equal bases refused: %v", err)
	}
	if err := pool("decode").ComparableTo(pool("decode"), d); err != nil {
		t.Errorf("the same pool refused: %v", err)
	}
	for _, c := range []struct {
		name string
		a, b Basis
		want string
	}{
		{"scope", cluster, pool("decode"), "scope is"},
		{"member", pool("prefill"), pool("decode"), "member is"},
		{"aggregate", worst, best, "aggregate is"},
		{"a direct figure and an aggregate", cluster, worst, "aggregate is"},
		{"over", worst, Basis{Scope: metrics.ScopeCluster, Aggregate: metrics.AggregateMax, Over: metrics.ScopePool}, "over is"},
		{"no scope on either side", Basis{}, Basis{}, "states no scope"},
	} {
		err := c.a.ComparableTo(c.b, d)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want an error containing %q", c.name, err, c.want)
		}
	}
}
