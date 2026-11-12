package fields

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// issueRef is "<owner>/<repo>#<n>", n a positive integer.
var issueRef = regexp.MustCompile(`^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*$`)

// Validate checks the list on its own and returns every problem it finds, as a
// *validate.Problems, or nil when there is none.
func (l *FieldList) Validate() error {
	p := &validate.Problems{}
	if l.Kind != Kind {
		p.Field("kind", "must be %q, got %q", Kind, l.Kind)
	}
	if l.Engine.Kind == "" {
		p.Field("engine.kind", "required")
	}
	if l.Engine.Version == "" {
		p.Field("engine.version", "required")
	}
	for i := range l.Fields {
		f := &l.Fields[i]
		// Each field's findings are collected apart and merged under its path. (Problems.At
		// does not share storage with its parent, whatever its comment says, so findings
		// recorded through it would be lost.)
		at := &validate.Problems{}
		f.validate(at)
		if i > 0 {
			prev := l.Fields[i-1]
			if !less(prev, *f) {
				at.Errorf("out of order: %q (%s) must sort after %q (%s), by name and then source, "+
					"and appear once", f.Name, f.Source, prev.Name, prev.Source)
			}
		}
		p.Merge(fieldPath(i, f.Name), at)
	}
	if !p.OK() {
		return p
	}
	return nil
}

// fieldPath locates a field by its index and, when it has one, its name.
func fieldPath(i int, name string) string {
	if name == "" {
		return fmt.Sprintf("fields[%d]", i)
	}
	return fmt.Sprintf("fields[%d](%s)", i, name)
}

// less is the list's order: by name, then by source, in byte order.
func less(a, b Field) bool {
	if a.Name != b.Name {
		return a.Name < b.Name
	}
	return a.Source < b.Source
}

func (f *Field) validate(p *validate.Problems) {
	if f.Name == "" {
		p.Field("name", "required")
	}
	switch f.Source {
	case SourceConfig, SourceEnv:
	default:
		p.Field("source", "must be %q or %q, got %q", SourceConfig, SourceEnv, f.Source)
	}
	f.validateEvidence(p)
	f.ValueClasses.validate(p, f.Status)
	est := &validate.Problems{}
	f.Estimation.validate(est)
	p.Merge("estimation", est)
}

func (f *Field) validateEvidence(p *validate.Problems) {
	e := f.Evidence
	set := map[string]bool{
		"traces": e.Traces != nil,
		"proof":  e.Proof != "",
		"issue":  e.Issue != "",
		"note":   e.Note != "",
	}
	var want string
	switch f.Status {
	case StatusModelled:
		want = "traces"
	case StatusProvenInert:
		want = "proof"
	case StatusNotModelled:
		want = "issue"
	case StatusInertUnproven:
		want = "note"
	default:
		p.Field("status", "must be one of %q, %q, %q, %q, got %q", StatusModelled,
			StatusProvenInert, StatusNotModelled, StatusInertUnproven, f.Status)
	}
	ev := &validate.Problems{}
	defer p.Merge("evidence", ev)
	for _, k := range []string{"traces", "proof", "issue", "note"} {
		switch {
		case k == want && !set[k]:
			ev.Field(k, "required for status %q", f.Status)
		case k != want && set[k] && want != "":
			ev.Field(k, "not allowed for status %q, which takes evidence.%s", f.Status, want)
		}
	}
	if want == "traces" && e.Traces != nil {
		if len(e.Traces) == 0 {
			ev.Field("traces", "must name at least one trace")
		}
		seen := map[string]bool{}
		for i, t := range e.Traces {
			switch {
			case t == "":
				ev.Field(fmt.Sprintf("traces[%d]", i), "must be a path, got an empty string")
			case seen[t]:
				ev.Field(fmt.Sprintf("traces[%d]", i), "names %q twice", t)
			}
			seen[t] = true
		}
	}
	if want == "issue" && e.Issue != "" && !issueRef.MatchString(e.Issue) {
		ev.Field("issue", "must be \"<owner>/<repo>#<n>\", got %q", e.Issue)
	}
}

func (e Estimation) validate(p *validate.Problems) {
	switch e.Rule {
	case RuleSubstitute:
		if len(e.Setting) == 0 {
			p.Field("setting", "required for rule %q: the settings that stand in for the field", RuleSubstitute)
		}
		for k := range e.Setting {
			if k == "" {
				p.Field("setting", "a setting needs a field name, got an empty key")
			}
		}
		if e.Reason == "" {
			p.Field("reason", "required for rule %q: why the setting is a fair stand-in", RuleSubstitute)
		}
	case RuleLeaveUnset, RuleNone:
		if e.Setting != nil {
			p.Field("setting", "only with rule %q, got rule %q", RuleSubstitute, e.Rule)
		}
		if e.Reason != "" {
			p.Field("reason", "only with rule %q, got rule %q", RuleSubstitute, e.Rule)
		}
	default:
		p.Field("rule", "must be one of %q, %q, %q, got %q", RuleLeaveUnset, RuleSubstitute,
			RuleNone, e.Rule)
	}
	e.validateExactness(p)
}

// validateExactness checks preserves and valid_when: only a rule with a stand-in states when
// it is exact, and a clause that sets nothing would hold for every value.
func (e Estimation) validateExactness(p *validate.Problems) {
	if e.Rule == RuleNone {
		if e.Preserves != "" {
			p.Field("preserves", "only with rule %q or %q, got rule %q", RuleLeaveUnset, RuleSubstitute, e.Rule)
		}
		if e.ValidWhen != nil {
			p.Field("valid_when", "only with rule %q or %q, got rule %q", RuleLeaveUnset, RuleSubstitute, e.Rule)
		}
		return
	}
	if e.ValidWhen == nil {
		return
	}
	if e.Preserves == "" {
		p.Field("preserves", "required with valid_when: the sentence naming what the upstream resolves identically")
	}
	if len(e.ValidWhen) == 0 {
		p.Field("valid_when", "must list at least one clause; omit it to apply the rule to every value")
	}
	for i, c := range e.ValidWhen {
		cp := &validate.Problems{}
		c.validate(cp)
		p.Merge(fmt.Sprintf("valid_when[%d]", i), cp)
	}
}

func (c Clause) validate(p *validate.Problems) {
	lists := []struct {
		key  string
		vals []string
	}{{"value_in", c.ValueIn}, {"value_list_has_any", c.ValueListHasAny},
		{"value_list_has_none", c.ValueListHasNone}, {"value_not_prefixed", c.ValueNotPrefixed}}
	set := 0
	for _, l := range lists {
		if l.vals == nil {
			continue
		}
		set++
		if len(l.vals) == 0 {
			p.Field(l.key, "must list at least one string")
		}
		for i, v := range l.vals {
			if strings.TrimSpace(v) == "" {
				p.Field(fmt.Sprintf("%s[%d]", l.key, i), "must not be empty")
			}
		}
	}
	if c.TPGroupNodes != nil {
		set++
		n := c.TPGroupNodes
		if n.Eq == nil && n.Gt == nil && n.Gte == nil && n.Lt == nil && n.Lte == nil {
			p.Field("tp_group_nodes", "must set at least one of eq, gt, gte, lt, lte")
		}
	}
	if c.ComputeCapability != nil {
		set++
		k := c.ComputeCapability
		if k.Eq == nil && k.Gt == nil && k.Gte == nil && k.Lt == nil && k.Lte == nil {
			p.Field("compute_capability", "must set at least one of eq, gt, gte, lt, lte")
		}
		for _, o := range []struct {
			op string
			v  *string
		}{{"eq", k.Eq}, {"gt", k.Gt}, {"gte", k.Gte}, {"lt", k.Lt}, {"lte", k.Lte}} {
			if o.v == nil {
				continue
			}
			if _, ok := parseCapability(*o.v); !ok {
				p.Field("compute_capability."+o.op, "must be \"<major>.<minor>\", such as \"10.0\", got %q", *o.v)
			}
		}
	}
	if set == 0 {
		p.Errorf("a clause must set at least one key, or it would hold for every value")
	}
}
