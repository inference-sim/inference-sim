package routertable

import (
	"fmt"
	"math"
	"sort"
	"strings"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/metrics"
)

// validateMapping checks what an entry that maps to a policy must add to the mapping: the
// differences and the bound. An entry whose policy is none states neither, because there is no
// stand-in to differ from or to bound; its reason says why none is near.
func (e *Entry) validateMapping(p *validate.Problems) {
	if e.Policy == PolicyNone {
		if len(e.Difference) > 0 {
			p.Field("difference", "an entry whose policy is %q states no differences: there is no stand-in policy to differ from; say why in the reason", PolicyNone)
		}
		if e.Bound != nil {
			p.Field("bound", "an entry whose policy is %q states no bound: there is no stand-in policy to bound; say why in the reason", PolicyNone)
		}
		return
	}
	if e.Policy == "" || !e.Policy.Offered() {
		return // the policy field already reports it
	}
	if len(e.Difference) == 0 {
		p.Field("difference", "required: each way the router's mode and the policy %q differ, one statement per aspect, each citing the code that shows it", e.Policy)
	}
	e.validateDifferences(p)
	if e.Bound == nil {
		p.Field("bound", "required: how far the choice of policy moves each figure, measured, with its regime and evidence")
		return
	}
	p.Merge("bound", e.Bound.Validate(e.Policy))
}

func (e *Entry) validateDifferences(p *validate.Problems) {
	statements := e.Statements()
	CheckStatements(p, "difference", statements)
	for i, d := range e.Difference {
		at := fmt.Sprintf("difference[%d]", i)
		if len(d.Evidence) == 0 {
			p.Field(at+".evidence", "required: the positions in the entry's evidence list of the citations that show the difference")
		}
		seen := map[int]bool{}
		for _, ix := range d.Evidence {
			switch {
			case ix < 0 || ix >= len(e.Evidence):
				p.Field(at+".evidence", "position %d is not in the evidence list of %d citations", ix, len(e.Evidence))
			case seen[ix]:
				p.Field(at+".evidence", "position %d is named twice", ix)
			}
			seen[ix] = true
		}
	}
}

// CheckStatements checks the differences an entry states or a point record carries: each names
// its aspect once and says something. Findings are recorded under at, for example "difference".
func CheckStatements(p *validate.Problems, at string, ss []Statement) {
	aspects := map[string]bool{}
	for i, d := range ss {
		loc := fmt.Sprintf("%s[%d]", at, i)
		switch {
		case d.Aspect == "":
			p.Field(loc+".aspect", "required: what differs, as a lower-case slug")
		case !token.MatchString(d.Aspect):
			p.Field(loc+".aspect", "must be lower case letters, digits, '.', '_' or '-', got %q", d.Aspect)
		case aspects[d.Aspect]:
			p.Field(loc+".aspect", "%q is stated twice; state each aspect once", d.Aspect)
		}
		aspects[d.Aspect] = true
		if strings.TrimSpace(d.Text) == "" {
			p.Field(loc+".text", "required: how the router's mode and the policy differ in this aspect")
		}
	}
}

// Validate checks a bound against the policy it bounds: complete, numeric, naming registered
// metrics and a regime, and saying what it does not cover. Paths are relative to the bound.
// Whether the ratios are the ones the evidence file gives is a check against the file, which
// the catalog's tests make; this package reads no file.
func (b *Bound) Validate(policy Policy) *validate.Problems {
	p := &validate.Problems{}
	b.Evidence.validate(p)
	if len(b.Ratios) == 0 {
		p.Field("ratios", "required: at least one ratio, with its metric, alternatives, regime and value")
	}
	seen := map[string]bool{}
	for i, r := range b.Ratios {
		at := fmt.Sprintf("ratios[%d]", i)
		r.validate(p, at, policy)
		key := r.Metric + "|" + strings.Join(r.Against, ",") + "|" + r.Regime.key()
		if seen[key] {
			p.Field(at, "repeats the metric, alternatives and regime of an earlier ratio")
		}
		seen[key] = true
	}
	if len(b.NotMeasured) == 0 {
		p.Field("not_measured", "required: what the bound does not cover; a bound that claims to cover everything is not a measured bound")
	}
	for i, s := range b.NotMeasured {
		if strings.TrimSpace(s) == "" {
			p.Field(fmt.Sprintf("not_measured[%d]", i), "must not be empty")
		}
	}
	return p
}

func (v BoundEvidence) validate(p *validate.Problems) {
	switch {
	case v.File == "":
		p.Field("evidence.file", "required: the CSV of runs the ratios are recomputed from, from the repository's root")
	case strings.HasPrefix(v.File, "/") || strings.Contains(v.File, ".."):
		p.Field("evidence.file", "is a path from the repository's root, got %q", v.File)
	}
	if !sha256Hex.MatchString(v.SHA256) {
		p.Field("evidence.sha256", "must be 64 lower-case hex digits, the SHA-256 of the file, got %q", v.SHA256)
	}
	if !repository.MatchString(v.Producer) {
		p.Field("evidence.producer", "must be <owner>/<name>, the repository BLIS was built from, got %q", v.Producer)
	}
	if !commit.MatchString(v.Commit) {
		p.Field("evidence.commit", "must be a full 40-character lowercase commit, got %q", v.Commit)
	}
	if strings.TrimSpace(v.Command) == "" {
		p.Field("evidence.command", "required: the command that reproduces the file")
	}
}

func (r Ratio) validate(p *validate.Problems, at string, policy Policy) {
	if r.Metric == "" {
		p.Field(at+".metric", "required: a metric-registry identifier")
	} else if _, ok := metrics.Lookup(r.Metric); !ok {
		p.Field(at+".metric", "%q is not in the metric registry (contracts/metrics): a bound on a figure without a definition cannot be compared", r.Metric)
	}
	if len(r.Against) == 0 {
		p.Field(at+".against", "required: the alternatives the policy was compared with")
	}
	seen := map[string]bool{}
	for _, a := range r.Against {
		switch {
		case !token.MatchString(a):
			p.Field(at+".against", "%q must be lower case letters, digits, '.', '_' or '-'", a)
		case a == string(policy):
			p.Field(at+".against", "%q is the entry's own policy; a ratio is against the alternatives", a)
		case seen[a]:
			p.Field(at+".against", "%q is named twice", a)
		}
		seen[a] = true
	}
	if !sort.StringsAreSorted(r.Against) {
		p.Field(at+".against", "alternatives are sorted, so one set has one spelling")
	}
	switch {
	case !p.FiniteField(at+".max_ratio", r.MaxRatio):
		// reported once, at the field
	case r.MaxRatio < 1:
		p.Field(at+".max_ratio", "must be a number of at least 1: it is the larger figure over the smaller, got %v", r.MaxRatio)
	case math.Abs(r.MaxRatio*100-math.Round(r.MaxRatio*100)) > 1e-6:
		p.Field(at+".max_ratio", "is stated to two decimals, rounded up, got %v", r.MaxRatio)
	}
	r.Regime.validate(p, at+".regime")
}

func (g Regime) validate(p *validate.Problems, at string) {
	if !g.Workload.valid() {
		p.Field(at+".workload", "must be one of %v, got %q", WorkloadKinds(), g.Workload)
	}
	switch {
	case g.ConcurrencyMin < 1:
		p.Field(at+".concurrency_min", "must be at least 1, got %d", g.ConcurrencyMin)
	case g.ConcurrencyMax < g.ConcurrencyMin:
		p.Field(at+".concurrency_max", "must be at least concurrency_min (%d), got %d", g.ConcurrencyMin, g.ConcurrencyMax)
	}
	if !token.MatchString(g.Model) {
		p.Field(at+".model", "required: the catalog model the runs used, got %q", g.Model)
	}
	if !token.MatchString(g.Hardware) {
		p.Field(at+".hardware", "required: the catalog hardware the runs used, got %q", g.Hardware)
	}
	if len(g.Shapes) == 0 {
		p.Field(at+".shapes", "required: the pool shapes the cells cover, each <n>P<m>D")
	}
	for i, s := range g.Shapes {
		if !shape.MatchString(s) {
			p.Field(at+".shapes", "%q must be <n>P<m>D: n prefill and m decode instances", s)
		}
		if i > 0 && s <= g.Shapes[i-1] {
			p.Field(at+".shapes", "shapes are sorted, each once")
		}
	}
}

func (g Regime) key() string {
	return fmt.Sprintf("%s/%d-%d/%s/%s/%s", g.Workload, g.ConcurrencyMin, g.ConcurrencyMax, g.Model, g.Hardware, strings.Join(g.Shapes, ","))
}
