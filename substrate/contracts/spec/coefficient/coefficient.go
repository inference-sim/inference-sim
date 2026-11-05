// Package coefficient defines a coefficient set: the learned numbers a cost
// estimate depends on, each recording where it came from and where it holds.
//
// The schema mirrors blis-registry, which owns the data. What this package adds is
// a typed form of the same contract, so a Go consumer validates against the same
// vocabulary the registry's own validator enforces rather than a second copy of it.
//
// Sets are standalone. There is no inheritance between them, because a coefficient
// is valid only for the functional form it was fitted against: a set fitted for a
// machine-utilization model has no bearing on a model that decomposes work by
// resource, however similar the names look.
package coefficient

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// Set is one immutable collection of coefficients feeding one cost model.
//
// The wire form nests each entry under its own name — a list of single-key maps —
// which is how blis-registry writes them:
//
//	coefficients:
//	  - mfu_prefill:
//	      value: 0.45
//	      units: dimensionless
//
// In Go an Entry carries its Name as a field instead, because a map with one key per
// element is awkward to range over and impossible to validate positionally. The
// UnmarshalYAML below converts between the two, so the committed files parse as
// written and callers get a flat list.
type Set struct {
	Kind         string  `yaml:"kind"` // "CoefficientSet"
	Name         string  `yaml:"name"`
	Coefficients []Entry `yaml:"coefficients"`
}

// UnmarshalYAML reads the registry's nested entry form into a flat list.
func (s *Set) UnmarshalYAML(node *yaml.Node) error {
	// Unknown top-level keys are rejected here: a custom unmarshal does not inherit the
	// decoder's KnownFields, so `backend:` or `extends:` (or a typo) would otherwise decode
	// to nothing silently. This matches blis-registry's strict top-level parse.
	if err := validate.RejectUnknownKeys(node, Set{}, nil); err != nil {
		return err
	}
	// An alias type without this method, so decoding it does not recurse.
	type setShape struct {
		Kind         string      `yaml:"kind"`
		Name         string      `yaml:"name"`
		Coefficients []yaml.Node `yaml:"coefficients"`
	}
	var raw setShape
	if err := node.Decode(&raw); err != nil {
		return err
	}
	s.Kind, s.Name = raw.Kind, raw.Name
	s.Coefficients = nil
	for i, item := range raw.Coefficients {
		if item.Kind != yaml.MappingNode || len(item.Content) != 2 {
			return fmt.Errorf("coefficients[%d]: each entry is a single-key map naming the coefficient", i)
		}
		var e Entry
		if err := validate.RejectUnknownKeys(item.Content[1], Entry{}, nil); err != nil {
			return fmt.Errorf("coefficients[%d] (%s): %w", i, item.Content[0].Value, err)
		}
		// fitted's Go zero value (false) is itself valid, so a plain decode cannot tell an
		// absent `fitted:` from an intentional `fitted: false`. The registry requires the
		// field present — it records a real provenance distinction (a datasheet figure is
		// not fitted; a curve fit is) — so enforce presence at the node here, keeping
		// Entry.Fitted a plain bool for every consumer that reads it.
		if !nodeHasKey(item.Content[1], "fitted") {
			return fmt.Errorf("coefficients[%d] (%s): missing required field %q",
				i, item.Content[0].Value, "fitted")
		}
		if err := item.Content[1].Decode(&e); err != nil {
			return fmt.Errorf("coefficients[%d] (%s): %w", i, item.Content[0].Value, err)
		}
		// A present `sources:` must be a non-empty list, matching the registry's validator.
		// Presence is read from the node because an absent key and `sources: []` both decode
		// to an empty slice, and only the latter is an error.
		if nodeHasKey(item.Content[1], "sources") && len(e.Sources) == 0 {
			return fmt.Errorf("coefficients[%d] (%s): sources, when present, must be a non-empty list",
				i, item.Content[0].Value)
		}
		e.Name = item.Content[0].Value
		s.Coefficients = append(s.Coefficients, e)
	}
	return nil
}

// nodeHasKey reports whether a mapping node declares key. It is how a required entry field
// whose decoded zero value is itself valid (fitted: false) is told apart from absent.
func nodeHasKey(node *yaml.Node, key string) bool {
	if node.Kind != yaml.MappingNode {
		return false
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value == key {
			return true
		}
	}
	return false
}

// Entry is one coefficient. Value, Units, Method, Fitted and Scope are required:
// a value without a unit is unusable, and a value without a method cannot be told
// apart from a measurement.
type Entry struct {
	Name   string       `yaml:"name"`
	Value  float64      `yaml:"value"`
	Units  vocab.Unit   `yaml:"units"`
	Method vocab.Method `yaml:"method"`
	// Fitted distinguishes a number obtained by fitting a curve from one read off a
	// datasheet or asserted. It is independent of Method: a vendor spec is not
	// fitted, and an assumed value may be a fitted curve's extrapolation.
	Fitted bool  `yaml:"fitted"`
	Scope  Scope `yaml:"scope"`

	// CI95 is a confidence interval, where the fit produced one.
	CI95 *Interval `yaml:"ci95,omitempty"`
	// Sources cite the evidence. A Method that claims evidence should carry at
	// least one; a rules pack, not this schema, decides how strictly to require it.
	Sources []Source `yaml:"sources,omitempty"`
	// Rationale explains a judgement. It is where the reasoning behind an assumed
	// value belongs, and it is what lets a reader weigh an estimate.
	Rationale string `yaml:"rationale,omitempty"`
	// CopiedFrom names the scope a copied value came from.
	CopiedFrom string `yaml:"copied_from,omitempty"`
	// Supersedes names an entry this one replaces.
	Supersedes string `yaml:"supersedes,omitempty"`
	// Validated and Unsupported record asymmetric evidence: a term may be checked
	// against one metric and explicitly not against another.
	//
	// Both accept a single metric name or a list. The registry writes the common case
	// as a scalar, and requiring a one-element list there would reject every
	// committed entry for no gain.
	Validated   MetricList `yaml:"validated,omitempty"`
	Unsupported MetricList `yaml:"unsupported,omitempty"`
}

// Interval is a closed numeric range. The wire form is a two-element sequence —
// [lower, upper] — which is what blis-registry validates and writes.
type Interval struct {
	Low  float64
	High float64
}

// UnmarshalYAML reads the two-element sequence form.
func (iv *Interval) UnmarshalYAML(node *yaml.Node) error {
	var pair []float64
	if err := node.Decode(&pair); err != nil {
		return fmt.Errorf("an interval is a two-element sequence [lower, upper]: %w", err)
	}
	if len(pair) != 2 {
		return fmt.Errorf("an interval has two elements, got %d", len(pair))
	}
	iv.Low, iv.High = pair[0], pair[1]
	return nil
}

// MarshalYAML writes the two-element sequence form, so a round trip is stable.
func (iv Interval) MarshalYAML() (any, error) {
	return []float64{iv.Low, iv.High}, nil
}

// Source is one citation.
type Source struct {
	Kind vocab.SourceKind `yaml:"kind"`
	Cite string           `yaml:"cite"`
	Role vocab.SourceRole `yaml:"role"`
}

// UnmarshalYAML rejects unknown keys in a source object before decoding, so a stray field
// (a typo, or a key the registry's own validator forbids) is an error rather than silently
// dropped — the custom unmarshal that gives the entry its name would otherwise lose the
// decoder's KnownFields for the nested source nodes too.
func (s *Source) UnmarshalYAML(node *yaml.Node) error {
	if err := validate.RejectUnknownKeys(node, Source{}, nil); err != nil {
		return err
	}
	type sourceShape Source // no UnmarshalYAML, so decoding does not recurse
	var raw sourceShape
	if err := node.Decode(&raw); err != nil {
		return err
	}
	*s = Source(raw)
	return nil
}

// Scope bounds where an entry holds. An empty scope is not "everywhere" but "not
// stated", and validation rejects it: a coefficient whose applicability is unknown
// cannot be applied safely.
//
// Dtype and backend are absent by design. Both vary a coefficient, and both ride
// in the entry name, following the convention blis-registry established for its
// per-backend dials.
type Scope struct {
	Hardware     []string `yaml:"hardware,omitempty"`
	Model        []string `yaml:"model,omitempty"`
	TP           []int    `yaml:"tp,omitempty"`
	EP           []int    `yaml:"ep,omitempty"`
	NodesSpanned []int    `yaml:"nodes_spanned,omitempty"`
	// Fabric lists the inter-node interconnect classes the value holds over, by catalog
	// identity (the stem of networks/<name>.yaml). Matched without regard to case, like
	// hardware and model.
	Fabric []string `yaml:"fabric,omitempty"`
}

// UnmarshalYAML rejects unknown scope keys before decoding. The typed fields alone would
// silently drop a key the resolver does not know — a custom unmarshal does not inherit the
// decoder's KnownFields — so an unrecognized dimension would read as "no such scoping"
// rather than an error. This restores the strict check, matching blis-registry's own
// validator.
func (s *Scope) UnmarshalYAML(node *yaml.Node) error {
	if err := validate.RejectUnknownKeys(node, Scope{}, nil); err != nil {
		return err
	}
	type scopeShape Scope // no UnmarshalYAML, so decoding does not recurse
	var raw scopeShape
	if err := node.Decode(&raw); err != nil {
		return err
	}
	*s = Scope(raw)
	return nil
}

// Empty reports whether no dimension is stated.
func (s Scope) Empty() bool {
	return len(s.Hardware) == 0 && len(s.Model) == 0 && len(s.TP) == 0 &&
		len(s.EP) == 0 && len(s.NodesSpanned) == 0 && len(s.Fabric) == 0
}

// Key returns a canonical string identifying this scope, for use as a map key.
//
// Order-independent within each dimension: hardware [h100, h200] and [h200, h100] are
// the same scope, so two entries differing only in that ordering are duplicates rather
// than alternatives. The separators cannot appear in a scope value, so two different
// scopes cannot collide on one key.
func (s Scope) Key() string {
	var b strings.Builder
	writeStrings := func(label string, vs []string) {
		if len(vs) == 0 {
			return
		}
		sorted := append([]string(nil), vs...)
		sort.Strings(sorted)
		b.WriteString(label)
		b.WriteByte('=')
		b.WriteString(strings.Join(sorted, ","))
		b.WriteByte(';')
	}
	writeInts := func(label string, vs []int) {
		if len(vs) == 0 {
			return
		}
		sorted := append([]int(nil), vs...)
		sort.Ints(sorted)
		b.WriteString(label)
		b.WriteByte('=')
		for i, v := range sorted {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(strconv.Itoa(v))
		}
		b.WriteByte(';')
	}
	writeStrings("hardware", s.Hardware)
	writeStrings("model", s.Model)
	writeInts("tp", s.TP)
	writeInts("ep", s.EP)
	writeInts("nodes_spanned", s.NodesSpanned)
	writeStrings("fabric", s.Fabric)
	return b.String()
}

// Describe renders a scope for an error message.
func (s Scope) Describe() string {
	if s.Empty() {
		return "unscoped"
	}
	return strings.TrimSuffix(s.Key(), ";")
}

// MetricList is one or more metric names. YAML may give either a scalar or a
// sequence; both decode to the same slice so a caller never branches on the form.
type MetricList []string

// UnmarshalYAML accepts a scalar or a sequence of scalars.
func (m *MetricList) UnmarshalYAML(node *yaml.Node) error {
	switch node.Kind {
	case yaml.ScalarNode:
		var one string
		if err := node.Decode(&one); err != nil {
			return err
		}
		*m = MetricList{one}
		return nil
	case yaml.SequenceNode:
		var many []string
		if err := node.Decode(&many); err != nil {
			return err
		}
		*m = many
		return nil
	default:
		return fmt.Errorf("expected a metric name or a list of them")
	}
}
