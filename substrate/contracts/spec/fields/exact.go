package fields

import "strings"

// Facts are what a consumer knows about a recorded value when it asks Exact. A fact the
// consumer cannot supply is left at its zero value, and a clause that needs it does not hold.
type Facts struct {
	// Value is the recorded value. Exact trims it and folds it to lower case.
	Value string
	// TPGroupNodes is the number of nodes the pool's tensor-parallel group spans, from the
	// pool's tensor-parallel size and the GPUs per node. Nil when unknown.
	TPGroupNodes *int
	// ComputeCapability is the GPU part's compute capability, "major.minor", from the
	// catalog. Empty when unknown.
	ComputeCapability string
}

// Exact reports whether the estimation rule's stand-in is exact for the recorded value: the
// upstream resolves the field, left unset or set to the stand-in, as it resolves the value.
//
// A rule that is not leave_unset or substitute has no stand-in, so it is never exact. A rule
// with no valid_when applies to every value. Otherwise the stand-in is exact when at least
// one clause holds. A value for which none does must not be stood in for.
func Exact(e Estimation, f Facts) bool {
	if e.Rule != RuleLeaveUnset && e.Rule != RuleSubstitute {
		return false
	}
	if e.ValidWhen == nil {
		return true
	}
	v := strings.ToLower(strings.TrimSpace(f.Value))
	for _, c := range e.ValidWhen {
		if c.holds(v, f) {
			return true
		}
	}
	return false
}

func (c Clause) holds(v string, f Facts) bool {
	var items map[string]bool
	if c.ValueListHasAny != nil || c.ValueListHasNone != nil {
		items = map[string]bool{}
		for _, it := range strings.Split(v, ",") {
			if it = strings.TrimSpace(it); it != "" {
				items[it] = true
			}
		}
	}
	if c.ValueIn != nil && !oneOf(v, c.ValueIn) {
		return false
	}
	if c.ValueListHasAny != nil {
		found := false
		for _, s := range c.ValueListHasAny {
			found = found || items[fold(s)]
		}
		if !found {
			return false
		}
	}
	if c.ValueListHasNone != nil {
		for _, s := range c.ValueListHasNone {
			if items[fold(s)] {
				return false
			}
		}
	}
	for _, p := range c.ValueNotPrefixed {
		if strings.HasPrefix(v, fold(p)) {
			return false
		}
	}
	if c.TPGroupNodes != nil && (f.TPGroupNodes == nil || !c.TPGroupNodes.holds(*f.TPGroupNodes)) {
		return false
	}
	if c.ComputeCapability != nil {
		have, ok := parseCapability(f.ComputeCapability)
		if !ok || !c.ComputeCapability.holds(have) {
			return false
		}
	}
	return true
}

func fold(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

func oneOf(v string, set []string) bool {
	for _, s := range set {
		if v == fold(s) {
			return true
		}
	}
	return false
}

func (n NodeComparison) holds(have int) bool {
	return (n.Eq == nil || have == *n.Eq) && (n.Gt == nil || have > *n.Gt) &&
		(n.Gte == nil || have >= *n.Gte) && (n.Lt == nil || have < *n.Lt) &&
		(n.Lte == nil || have <= *n.Lte)
}

func (c CapabilityComparison) holds(have capability) bool {
	test := func(s *string, ok func(int) bool) bool {
		if s == nil {
			return true
		}
		want, valid := parseCapability(*s)
		return valid && ok(have.cmp(want))
	}
	return test(c.Eq, func(r int) bool { return r == 0 }) && test(c.Gt, func(r int) bool { return r > 0 }) &&
		test(c.Gte, func(r int) bool { return r >= 0 }) && test(c.Lt, func(r int) bool { return r < 0 }) &&
		test(c.Lte, func(r int) bool { return r <= 0 })
}
