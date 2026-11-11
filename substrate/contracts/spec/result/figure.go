package result

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/metrics"
)

// Basis is what a figure states besides its metric: the scope it is at, the member of that
// scope, and, for a figure that combines the figures of a narrower scope, the aggregate and the
// scope it ranges over (design §8.7, DEV-8). Two figures are compared only when their metric
// and their basis are the same, and a basis is never inferred from a field's name: a figure
// that states no scope has none.
//
// It is the document form of the in-memory metrics.Figure's Scope, Member, Aggregate and
// Over, and is checked by that type's rules.
type Basis struct {
	// Scope is the scope the figure is at: cluster, pool or instance.
	Scope metrics.Scope `yaml:"scope"`
	// Member names the pool or instance a figure at the pool or instance scope is of, by the
	// name the deployment gives it. A figure at the cluster scope has none. For a figure that
	// aggregates, it is the member of Scope, not of the scope ranged over.
	Member string `yaml:"member,omitempty"`
	// Aggregate and Over are stated together when the figure combines the figures of a
	// narrower scope: the aggregate (max or min), and the scope it ranges over. Both are
	// absent for a figure computed directly from the scope's own population.
	Aggregate metrics.Aggregate `yaml:"aggregate,omitempty"`
	Over      metrics.Scope     `yaml:"over,omitempty"`
}

// Basis returns the figure's basis.
func (f Figure) Basis() Basis {
	return Basis{Scope: f.Scope, Member: f.Member, Aggregate: f.Aggregate, Over: f.Over}
}

// figure returns the in-memory figure of metric d at b, which carries the rules of DEV-8.
func (b Basis) figure(d metrics.Definition) metrics.Figure {
	return metrics.Figure{Definition: d, Scope: b.Scope, Member: b.Member, Aggregate: b.Aggregate, Over: b.Over}
}

// Check returns nil when b is a basis a figure of metric d may have, and otherwise an error
// naming each way it is not: no scope, a scope the metric is not defined at, a member at the
// cluster scope or none at another, an aggregate the metric does not admit, and an aggregate
// with no scope to range over or one that is not narrower than the figure's own.
func (b Basis) Check(d metrics.Definition) error {
	return b.figure(d).Validate()
}

// ComparableTo returns nil when a figure of metric d at b may be compared with one of the same
// metric at o, and otherwise an error naming the first field in which they differ: scope,
// member, aggregate or over. A basis that does not check is never comparable, even with an
// equal one (DEV-8).
func (b Basis) ComparableTo(o Basis, d metrics.Definition) error {
	return metrics.ComparableFigures(b.figure(d), o.figure(d))
}

// Validate records on p, at the path at, each way b is not a basis of a figure of metric d.
func (b Basis) Validate(p *validate.Problems, at string, d metrics.Definition) {
	if err := b.Check(d); err != nil {
		p.Field(at, "%v", err)
	}
}

// String renders b for a message: the scope and member, and the aggregate over a scope when
// it has one.
func (b Basis) String() string {
	s := string(b.Scope)
	if b.Member != "" {
		s += " " + b.Member
	}
	if b.Aggregate != "" {
		s += fmt.Sprintf(", %s over %s", b.Aggregate, b.Over)
	}
	return s
}
