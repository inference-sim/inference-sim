// Package validate holds the reporting types every validator shares: a located,
// accumulating problem list rather than a first-error return.
//
// Accumulation is the point. A reviewer fixing a scenario wants every problem in
// one pass, and a CI log that names one field at a time turns a five-minute fix
// into five runs.
package validate

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Severity separates a problem that makes a document unusable from one that makes
// it questionable. Both are reported; only Error fails a check.
type Severity string

const (
	// SeverityError: the document cannot be used as it stands.
	SeverityError Severity = "error"
	// SeverityWarning: usable, but a reader should look. A coefficient asserted
	// without a rationale is the archetype — valid, and worth questioning.
	SeverityWarning Severity = "warning"
)

// Problem is one finding, located precisely enough to fix without searching.
type Problem struct {
	// Path is a dotted location: "pools[1].engine.cache_dtype".
	Path     string
	Message  string
	Severity Severity
	// Rule names the rule that fired, empty for field-level checks. It lets a
	// reader tell a structural violation from a version-specific one, and lets a
	// CI job waive one rule without waiving field validation.
	Rule string
}

func (p Problem) String() string {
	prefix := string(p.Severity)
	if p.Rule != "" {
		prefix += " [" + p.Rule + "]"
	}
	return fmt.Sprintf("%s: %s: %s", prefix, p.Path, p.Message)
}

// Problems accumulates findings under a path prefix.
type Problems struct {
	items  []Problem
	prefix []string
}

// At returns a Problems that records under an extended path. The returned value
// shares storage with the receiver, so findings recorded through either appear in
// the parent's list.
func (p *Problems) At(segment string) *Problems {
	return &Problems{items: nil, prefix: append(append([]string{}, p.prefix...), segment)}
}

// Errorf records an error at the current path.
func (p *Problems) Errorf(format string, args ...any) {
	p.add(SeverityError, "", fmt.Sprintf(format, args...))
}

// Warnf records a warning at the current path.
func (p *Problems) Warnf(format string, args ...any) {
	p.add(SeverityWarning, "", fmt.Sprintf(format, args...))
}

// RuleErrorf records an error attributed to a named rule.
func (p *Problems) RuleErrorf(rule, format string, args ...any) {
	p.add(SeverityError, rule, fmt.Sprintf(format, args...))
}

// RuleWarnf records a warning attributed to a named rule.
func (p *Problems) RuleWarnf(rule, format string, args ...any) {
	p.add(SeverityWarning, rule, fmt.Sprintf(format, args...))
}

// Field records a problem at a named field below the current path, which is the
// common case and avoids a temporary At for one call.
func (p *Problems) Field(name, format string, args ...any) {
	p.items = append(p.items, Problem{
		Path:     joinPath(append(append([]string{}, p.prefix...), name)),
		Message:  fmt.Sprintf(format, args...),
		Severity: SeverityError,
	})
}

// FiniteField records a problem at a named field if x is NaN or ±Inf. Every numeric
// schema field needs this before its magnitude check, because the magnitude checks
// alone do not catch a non-finite value: IEEE-754 makes every ordered comparison with
// NaN false, so a `x <= 0` / `x < 0` guard returns false for NaN *and* for +Inf and
// lets both through (only -Inf is incidentally caught). It belongs here as a shared
// primitive so every validator reading a float — hardware datasheet figures, registry
// coefficients — rejects a non-finite value identically rather than re-deriving the
// check, or forgetting it. A non-finite number that reaches a cost model poisons every
// arithmetic it touches.
//
// It returns whether x is finite, so a caller can gate the field's magnitude and
// relational checks on it: once a field is flagged non-finite, re-running those checks
// is at best a duplicate error (a -Inf also trips "must be positive") and at worst a
// misattribution (a +Inf BF16 makes an FP8-vs-BF16 check complain about the FP8 field).
// Reporting the fault once, at the field that is actually wrong, keeps the diagnostic
// honest.
func (p *Problems) FiniteField(name string, x float64) bool {
	// Name which non-finite value it is: NaN points at a missing or corrupted input,
	// an infinity at an unbounded or divide-by-zero scale, and the two call for
	// different fixes in the raw catalog data.
	switch {
	case math.IsNaN(x):
		p.Field(name, "must be a finite number, got NaN")
	case math.IsInf(x, 1):
		p.Field(name, "must be a finite number, got +Inf")
	case math.IsInf(x, -1):
		p.Field(name, "must be a finite number, got -Inf")
	default:
		return true
	}
	return false
}

func (p *Problems) add(sev Severity, rule, msg string) {
	p.items = append(p.items, Problem{
		Path: joinPath(p.prefix), Message: msg, Severity: sev, Rule: rule,
	})
}

// Merge folds another Problems' findings into this one, prefixing each path with
// the given segment.
func (p *Problems) Merge(segment string, other *Problems) {
	for _, it := range other.items {
		path := segment
		if it.Path != "" {
			path = segment + "." + it.Path
		}
		it.Path = path
		p.items = append(p.items, it)
	}
}

// All returns every finding, sorted by path then message so a CI diff is stable
// across runs.
func (p *Problems) All() []Problem {
	out := append([]Problem{}, p.items...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].Message < out[j].Message
	})
	return out
}

// Errors returns only the findings that fail a check.
func (p *Problems) Errors() []Problem {
	var out []Problem
	for _, it := range p.All() {
		if it.Severity == SeverityError {
			out = append(out, it)
		}
	}
	return out
}

// Rebase returns the problems with prefix removed from the front of every path, so a
// check written against a placeholder location can be merged under another one.
func (p *Problems) Rebase(prefix string) *Problems {
	out := &Problems{prefix: append([]string{}, p.prefix...)}
	for _, it := range p.items {
		switch {
		case it.Path == prefix:
			it.Path = ""
		case strings.HasPrefix(it.Path, prefix+"."):
			it.Path = strings.TrimPrefix(it.Path, prefix+".")
		}
		out.items = append(out.items, it)
	}
	return out
}

// OK reports whether nothing failed. Warnings do not make a document invalid.
func (p *Problems) OK() bool { return len(p.Errors()) == 0 }

// Error renders every finding, one per line, for a CI log.
func (p *Problems) Error() string {
	items := p.All()
	lines := make([]string, 0, len(items))
	for _, it := range items {
		lines = append(lines, it.String())
	}
	return strings.Join(lines, "\n")
}

func joinPath(segs []string) string {
	var out []string
	for _, s := range segs {
		if s != "" {
			out = append(out, s)
		}
	}
	return strings.Join(out, ".")
}

// IsSHA256 reports whether s is a canonical sha256 digest: 64 lowercase hex characters.
// Lowercase only, so that a digest compares and caches byte for byte.
func IsSHA256(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// SHA256Field records a problem at a named field unless v is a canonical sha256 digest.
// An empty v is reported as missing, so callers with an optional digest check presence
// first.
func (p *Problems) SHA256Field(name, v string) bool {
	switch {
	case v == "":
		p.Field(name, "required: a sha256 digest, 64 lowercase hex characters")
	case !IsSHA256(v):
		p.Field(name, "must be a sha256 digest, 64 lowercase hex characters, got %q", v)
	default:
		return true
	}
	return false
}
