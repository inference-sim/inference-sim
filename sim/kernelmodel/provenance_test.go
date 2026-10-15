package kernelmodel

import (
	"slices"
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/vocab"
)

// A pool's provenance restates the kernel's own records without losing or inventing one
// (#1942): the method counts add up to the kernel's Provenance, Evidenced and Total are its
// Evidence, every kernel-made assumption and every non-evidenced registry entry is listed
// once in its own list, and the skipped rules are the release pack's minus the ones a run is
// refused on.
func TestProvenanceRestatesTheKernelsEvidence(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	m, err := OpenPool(dspark, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	p := m.Provenance()
	origins := m.Kernel().Provenance()

	sum := 0
	for _, n := range p.MethodCounts {
		sum += n
	}
	if sum != len(origins) || p.Total != len(origins) {
		t.Errorf("method counts sum to %d, total %d, the kernel recorded %d", sum, p.Total, len(origins))
	}
	measured, total, assumed := m.Kernel().(*latencykernel.Kernel).Evidence()
	if p.Evidenced != measured || p.Total != total {
		t.Errorf("evidenced %d of %d, the kernel's Evidence says %d of %d", p.Evidenced, p.Total, measured, total)
	}
	if len(p.Assumptions) == 0 {
		t.Fatal("V4.1 on h100 records kernel assumptions; none were listed")
	}
	for _, a := range p.Assumptions {
		if a.Set != latencykernel.KernelAssumptionSet {
			t.Errorf("assumption %s comes from set %q, not the kernel's", a.Name, a.Set)
		}
	}
	for _, o := range p.NotEvidenced {
		if vocab.Method(o.Method).Evidenced() || o.Set == latencykernel.KernelAssumptionSet {
			t.Errorf("%s (%s, %s) listed as a fallback or assumption", o.Name, o.Set, o.Method)
		}
	}
	assumedRegistry := 0
	for _, o := range origins {
		if vocab.Method(o.Method) == vocab.MethodAssumed && o.Set != latencykernel.KernelAssumptionSet {
			assumedRegistry++
		}
	}
	if got := len(assumed) - len(p.Assumptions); got != assumedRegistry {
		t.Errorf("the kernel's assumed list has %d registry entries beyond its own assumptions, "+
			"the origins %d", got, assumedRegistry)
	}

	pack := rules.Lookup("0.31.0")
	if pack == nil {
		t.Fatal("no 0.31.0 rules pack registered")
	}
	if want := len(pack.Names()) - len(enforcedRules); len(p.SkippedRules) != want {
		t.Errorf("%d skipped rules, want the pack's %d less the %d enforced", len(p.SkippedRules),
			len(pack.Names()), len(enforcedRules))
	}
	for name := range enforcedRules {
		if slices.Contains(p.SkippedRules, name) {
			t.Errorf("enforced rule %s reported as skipped", name)
		}
	}
	if !slices.Contains(p.SkippedRules, "quantization-known") {
		t.Error("quantization-known is applied by no one, yet not reported as skipped")
	}
}
