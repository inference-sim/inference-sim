package measurement

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"testing"
)

// These tests state the verdict rules as properties over generated runs, read through
// the public Assess and Validate. Random inputs come from fixed seeds, so a failure
// reproduces exactly.

var rowVerdictsInOrder = []RowVerdict{RowMatch, RowMismatch, RowUnstated, RowUnverifiable}
var diffKindsInOrder = []DiffKind{DiffNotExpressible, DiffNotModelled, DiffConflict,
	DiffUnresolved, DiffApproximated}

func row(rng *rand.Rand, i int, v RowVerdict) ProvenanceRow {
	r := ProvenanceRow{Setting: fmt.Sprintf("setting_%d", i), Verdict: v, Material: MaterialUnknown}
	switch v {
	case RowMatch, RowMismatch:
		r.Report = &Observed{Value: "4", Ref: "report:1"}
		r.BLIS = &Observed{Value: "4"}
	case RowUnstated:
		r.BLIS = &Observed{Value: "16"}
		if rng.Intn(2) == 0 {
			r.ResolvedBy = "vllm/config/cache.py default"
		}
	case RowUnverifiable:
		r.Report = &Observed{Value: "4", Ref: "report:1"}
		r.Script = &Observed{Value: "8", Ref: "script:1"}
	}
	return r
}

func diff(i int, k DiffKind) DiffEntry {
	return DiffEntry{Path: fmt.Sprintf("deployment.x%d", i), Kind: k, Material: MaterialYes}
}

// generate returns a structurally valid run whose verdict-relevant facts are random.
func generate(rng *rand.Rand) (*MeasuredRun, bool) {
	r := exactRun()
	r.Provenance = nil
	for i, n := 0, rng.Intn(6); i < n; i++ {
		r.Provenance = append(r.Provenance, row(rng, i, rowVerdictsInOrder[rng.Intn(4)]))
	}
	if rng.Intn(3) == 0 {
		for i, n := 0, 1+rng.Intn(3); i < n; i++ {
			r.Diff = append(r.Diff, diff(i, diffKindsInOrder[rng.Intn(5)]))
		}
	}
	switch rng.Intn(12) {
	case 0:
		r.Model = ""
	case 1:
		r.Deployment = nil
	case 2:
		r.Workload = nil
	case 3:
		r.Cluster = nil
	case 4:
		r.EngineVersion = ""
	case 5:
		setDef(r, "itl_p99", func(d *MetricDef) { d.Statistic, d.PercentileMethod = "", "" })
	case 6:
		setDef(r, "output_tps", func(d *MetricDef) { d.Denominator = "" })
	case 7:
		r.Client.Harness = ""
	}
	return r, rng.Intn(5) != 0
}

// oracle is the verdict rule as the package documents it, written from the text rather
// than from Assess.
func oracle(r *MeasuredRun, permitted bool) (Verdict, Resolution) {
	if r.Model == "" || r.EngineVersion == "" || r.Cluster == nil || r.Deployment == nil ||
		r.Workload == nil {
		return VerdictNotReproducible, ""
	}
	for _, d := range r.Diff {
		if d.Kind == DiffNotExpressible {
			return VerdictNotReproducible, ""
		}
	}
	exact := len(r.Diff) == 0 && permitted && r.Client.Harness != ""
	resolved := false
	for _, row := range r.Provenance {
		switch {
		case row.Verdict == RowMismatch || row.Verdict == RowUnverifiable:
			exact = false
		case row.Verdict == RowUnstated && row.ResolvedBy == "":
			exact = false
		case row.Verdict == RowUnstated:
			resolved = true
		}
	}
	for _, d := range r.MetricDefs {
		if d.Formula == "" || d.Population == "" || d.Clock == "" ||
			(d.Formula.Throughput() && d.Denominator == "") ||
			(!d.Formula.Throughput() && d.Statistic == "") {
			exact = false
		}
	}
	switch {
	case !exact:
		return VerdictPartial, ""
	case resolved:
		return VerdictExact, ResolutionResolved
	default:
		return VerdictExact, ResolutionStrict
	}
}

func eachRun(t *testing.T, n int, f func(t *testing.T, rng *rand.Rand, r *MeasuredRun, permitted bool)) {
	rng := rand.New(rand.NewSource(20261015))
	for i := 0; i < n; i++ {
		r, permitted := generate(rng)
		if p := r.Validate(); !p.OK() {
			t.Fatalf("run %d: the generator produced an invalid run:\n%s", i, p.Error())
		}
		f(t, rng, r, permitted)
	}
}

func TestAssessMatchesTheRule(t *testing.T) {
	seen := map[string]int{}
	eachRun(t, 600, func(t *testing.T, _ *rand.Rand, r *MeasuredRun, permitted bool) {
		a := Assess(r, permitted)
		v, res := oracle(r, permitted)
		if a.Verdict != v || a.Resolution != res {
			t.Fatalf("Assess = %s/%s, rule = %s/%s for %+v", a.Verdict, a.Resolution, v, res, r.Provenance)
		}
		if (a.Verdict == VerdictExact) != (len(a.Reasons) == 0) {
			t.Fatalf("an EXACT run has no reasons and every other run has one: %s %v", a.Verdict, a.Reasons)
		}
		seen[string(a.Verdict)+"/"+string(a.Resolution)]++
	})
	for _, want := range []string{"EXACT/STRICT", "EXACT/RESOLVED", "PARTIAL/", "NOT_REPRODUCIBLE/"} {
		if seen[want] == 0 {
			t.Errorf("the generator never produced %s; the property is untested there (%v)", want, seen)
		}
	}
}

// Adding a MISMATCH or UNVERIFIABLE row anywhere can never make a run EXACT.
func TestDisputedRowNeverExact(t *testing.T) {
	eachRun(t, 400, func(t *testing.T, rng *rand.Rand, r *MeasuredRun, permitted bool) {
		for _, v := range []RowVerdict{RowMismatch, RowUnverifiable} {
			at := rng.Intn(len(r.Provenance) + 1)
			bad := row(rng, 99, v)
			rows := append(append(append([]ProvenanceRow{}, r.Provenance[:at]...), bad), r.Provenance[at:]...)
			c := *r
			c.Provenance = rows
			if a := Assess(&c, permitted); a.Verdict == VerdictExact {
				t.Fatalf("a %s row at %d left the run EXACT", v, at)
			}
			if a := Assess(&c, true); a.Verdict == VerdictExact {
				t.Fatalf("a %s row at %d left the run EXACT under a permitting licence", v, at)
			}
		}
	})
}

// Adding a difference can only keep or lower a verdict.
func TestDiffIsMonotone(t *testing.T) {
	rank := map[Verdict]int{VerdictExact: 2, VerdictPartial: 1, VerdictNotReproducible: 0}
	eachRun(t, 400, func(t *testing.T, rng *rand.Rand, r *MeasuredRun, permitted bool) {
		before := Assess(r, permitted).Verdict
		c := *r
		c.Diff = append(append([]DiffEntry{}, r.Diff...), diff(99, diffKindsInOrder[rng.Intn(5)]))
		if after := Assess(&c, permitted).Verdict; rank[after] > rank[before] || after == VerdictExact {
			t.Fatalf("adding a difference moved %s to %s", before, after)
		}
	})
}

// Reordering provenance rows and differences changes neither the verdict nor the set of
// reasons.
func TestPermutationInvariant(t *testing.T) {
	eachRun(t, 400, func(t *testing.T, rng *rand.Rand, r *MeasuredRun, permitted bool) {
		a := Assess(r, permitted)
		c := *r
		c.Provenance = append([]ProvenanceRow{}, r.Provenance...)
		c.Diff = append([]DiffEntry{}, r.Diff...)
		rng.Shuffle(len(c.Provenance), func(i, j int) { c.Provenance[i], c.Provenance[j] = c.Provenance[j], c.Provenance[i] })
		rng.Shuffle(len(c.Diff), func(i, j int) { c.Diff[i], c.Diff[j] = c.Diff[j], c.Diff[i] })
		b := Assess(&c, permitted)
		if a.Verdict != b.Verdict || a.Resolution != b.Resolution {
			t.Fatalf("permuting moved %s/%s to %s/%s", a.Verdict, a.Resolution, b.Verdict, b.Resolution)
		}
		if sorted(a.Reasons) != sorted(b.Reasons) {
			t.Fatalf("permuting changed the reasons:\n%v\n%v", a.Reasons, b.Reasons)
		}
	})
}

func sorted(s []string) string {
	c := append([]string{}, s...)
	sort.Strings(c)
	return strings.Join(c, "|")
}

var statedForms = []struct {
	v Verdict
	r Resolution
}{
	{VerdictExact, ResolutionStrict}, {VerdictExact, ResolutionResolved},
	{VerdictPartial, ""}, {VerdictNotReproducible, ""},
	{VerdictPartial, ResolutionStrict}, {VerdictNotReproducible, ResolutionResolved},
}

// A stated verdict is accepted against the registry exactly when it is the computed one.
func TestStatedVerdictMustBeComputed(t *testing.T) {
	eachRun(t, 250, func(t *testing.T, _ *rand.Rand, r *MeasuredRun, permitted bool) {
		reg := permitting()
		if !permitted {
			r.Source.LicenseRef = "hud"
		}
		a := Assess(r, permitted)
		for _, s := range statedForms {
			c := *r
			c.Verdict, c.Resolution = s.v, s.r
			ok := c.ValidateWithSources(reg).OK()
			if want := s.v == a.Verdict && s.r == a.Resolution; ok != want {
				t.Fatalf("stating %s/%s when the run computes to %s/%s: accepted = %v (%v)",
					s.v, s.r, a.Verdict, a.Resolution, ok, a.Reasons)
			}
		}
	})
}

// Without the registry, Validate cannot tell EXACT from a licence-denied PARTIAL, and
// accepts either for a run whose own facts are EXACT; it still rejects every other
// contradiction.
func TestStatedVerdictWithoutRegistry(t *testing.T) {
	eachRun(t, 250, func(t *testing.T, _ *rand.Rand, r *MeasuredRun, _ bool) {
		own := Assess(r, true)
		for _, s := range statedForms {
			c := *r
			c.Verdict, c.Resolution = s.v, s.r
			want := s.v == own.Verdict && s.r == own.Resolution ||
				own.Verdict == VerdictExact && s.v == VerdictPartial && s.r == ""
			if ok := c.Validate().OK(); ok != want {
				t.Fatalf("stating %s/%s when the run's own facts give %s/%s: accepted = %v",
					s.v, s.r, own.Verdict, own.Resolution, ok)
			}
		}
	})
}
