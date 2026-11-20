package latencykernel

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
	"github.com/inference-sim/inference-sim/substrate/load"
)

const (
	testHostCode   = "0123456789abcdef"
	testGraphsCode = "fedcba9876543210"
	otherCode      = "1111111111111111"
)

// keyedByCode returns the inputs with every engine-path coefficient listing the CodeID of its
// path, so the registry's engine-path values are valid for that code only.
func keyedByCode(in Inputs) Inputs {
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			switch coefficient.CodePathOf(c.Coefficients[j].Name) {
			case coefficient.CodePathHost:
				c.Coefficients[j].CodeIDs = []string{testHostCode}
			case coefficient.CodePathGraphs:
				c.Coefficients[j].CodeIDs = []string{testGraphsCode}
			}
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

// The CodeID the engine states for the code that does the work decides which engine-path
// coefficient prices it: the same plan with a different CodeID takes none of the values keyed to
// the other, and prices those costs at zero, recorded as assumptions rather than guessed from a
// sibling CodeID.
func TestTheEnginePathCodeIDDecidesWhichHostCoefficientPrices(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	at := func(host, graphs string) (Inputs, *Kernel) {
		in := keyedByCode(planCase(t, d, "llama-3.1-70b-instruct", "h100", 1, "tp: 4, pp: 1, dp: 1",
			func(l *plan.Lowered) { l.Plan.Host.CodeID, l.Plan.Graphs.CodeID = host, graphs }))
		return in, mustKernel(t, in)
	}

	t.Run("host path", func(t *testing.T) {
		_, same := at(testHostCode, testGraphsCode)
		_, diff := at(otherCode, testGraphsCode)
		if same.AdmissionOverhead(4096) == 0 || same.OutputTokenOverhead() == 0 {
			t.Fatal("the registry's host costs are not priced for the CodeID they list")
		}
		if got := diff.AdmissionOverhead(4096); got != 0 {
			t.Errorf("admission priced at %v under a different host CodeID; a lookup must fail on it", got)
		}
		if got := diff.OutputTokenOverhead(); got != 0 {
			t.Errorf("output-token cost %v under a different host CodeID", got)
		}
		if got := diff.CompletionOverhead(); got != 0 {
			t.Errorf("completion cost %v under a different host CodeID", got)
		}
	})

	t.Run("graph path", func(t *testing.T) {
		in, _ := at(testHostCode, testGraphsCode)
		inDiff, _ := at(testHostCode, otherCode)
		if prices(t, in) == prices(t, inDiff) {
			t.Error("a different graphs CodeID moves no price, so launch costs are not keyed by it")
		}
	})

	t.Run("no CodeID stated", func(t *testing.T) {
		_, none := at("", "")
		if got := none.AdmissionOverhead(4096); got != 0 {
			t.Errorf("admission priced at %v by a plan that states no host CodeID against keyed entries", got)
		}
	})

	t.Run("provenance", func(t *testing.T) {
		_, same := at(testHostCode, testGraphsCode)
		_, diff := at(otherCode, testGraphsCode)
		origin := func(k *Kernel, name string) (kernel.CoefficientOrigin, bool) {
			for _, o := range k.Provenance() {
				if o.Name == name && o.Set != KernelAssumptionSet {
					return o, true
				}
			}
			return kernel.CoefficientOrigin{}, false
		}
		o, ok := origin(same, "host_output_token")
		if !ok || o.CodeMatch != string(coefficient.CodeExact) || o.CodeID != testHostCode {
			t.Errorf("host_output_token provenance = %+v, %v; want an exact match on %s", o, ok, testHostCode)
		}
		if o, ok := origin(diff, "host_output_token"); ok {
			t.Errorf("host_output_token is reported as priced under a different CodeID: %+v", o)
		}
		if o, ok := origin(same, "gemm_eps_max_bf16"); ok && o.CodeMatch != "" {
			t.Errorf("a hardware rate reports a code match: %+v", o)
		}
	})
}

// registryCodeIDs is the host and graph CodeIDs the committed engine-path coefficients are keyed
// to: a fixture plan that states them is priced by the registry's values, as an engine whose code
// is the code those values were carried to would be. The registry lists one CodeID per engine; a
// fixture is no engine, so it states the first.
func registryCodeIDs() (host, graphs string) {
	s, err := contracts.LoadCoefficientSet("../registry/coefficients/cost-model-host-overheads.yaml")
	if err != nil {
		panic(err)
	}
	for _, e := range s.Coefficients {
		if len(e.CodeIDs) == 0 {
			continue
		}
		switch coefficient.CodePathOf(e.Name) {
		case coefficient.CodePathHost:
			host = e.CodeIDs[0]
		case coefficient.CodePathGraphs:
			graphs = e.CodeIDs[0]
		}
	}
	if host == "" || graphs == "" {
		panic("the registry's engine-path coefficients list no CodeID")
	}
	return host, graphs
}
