// Moved from blis-latency-kernel/provenance_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"

	"github.com/inference-sim/blis-schemas/spec/coefficient"
)

// dropCoefficient returns the sets without the named entry.
func dropCoefficient(sets []*coefficient.Set, name string) []*coefficient.Set {
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			if e.Name != name {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		out[i] = &c
	}
	return out
}

// A MISSING OPTIONAL COEFFICIENT IS DISCLOSED WHERE IT MATTERS. The kernel prices without
// an optional coefficient, so dropping one the deployment needs must leave a step that
// differs from the full one AND a Provenance entry naming what is missing; dropping one it
// does not need -- a recurrent mixer's terms on a model with none -- must change neither.
// Before this, a registry missing kimi-k3's KDA rate priced its decode a tenth cheaper with
// nothing reported.
func TestAMissingOptionalCoefficientIsDisclosedWhereItMatters(t *testing.T) {
	decode := decodeBatch(32, 1, 4096)
	for _, c := range []struct {
		fixture, name string
		needed        bool
	}{
		{"kimi-k3-h100-nospec.yaml", "recurrent_decode_rate_kda", true},
		{dcpMLAFixture, "host_launch_per_kernel", true},
		{dcpMLAFixture, "attention_decode_rate_mla", true},
		{dcpMLAFixture, "recurrent_decode_rate_kda", false},
	} {
		full := fixture(t, c.fixture)
		in := fixtureInputs(t, c.fixture)
		in.Coefficients = dropCoefficient(in.Coefficients, c.name)
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("%s without %s: %v", c.fixture, c.name, err)
		}
		if hasAssumption(full, c.name) {
			t.Errorf("%s with every coefficient discloses %s as missing", c.fixture, c.name)
		}
		moved := k.StepTime(decode).NoOverlap != full.StepTime(decode).NoOverlap
		if got := hasAssumption(k, c.name); got != c.needed {
			t.Errorf("%s without %s: disclosed %v, want %v", c.fixture, c.name, got, c.needed)
		}
		if moved != c.needed {
			t.Errorf("%s without %s: step moved %v, want %v", c.fixture, c.name, moved,
				c.needed)
		}
	}
}

// THE ENGINE RELEASE PRICED IS DISCLOSED WHEN IT IS NOT THE ONE DECLARED. The fixtures declare
// 0.29.0, a release the vLLM rules packs cover, and the kernel encodes 0.31.0's behaviour; that
// must show in Provenance, and must not when the scenario declares 0.31.0 itself.
func TestTheEngineReleasePricedIsDisclosed(t *testing.T) {
	in := fixtureInputs(t, dcpMLAFixture)
	declared := func(v string) bool {
		t.Helper()
		sc := *in.Scenario
		sc.EngineVersion = v
		in2 := in
		in2.Scenario = &sc
		k, err := newKernel(in2)
		if err != nil {
			t.Fatal(err)
		}
		return hasAssumption(k, "engine_version")
	}
	if !declared("0.29.0") {
		t.Error("a 0.29.0 scenario priced at 0.31.0's behaviour discloses nothing")
	}
	if declared(EngineBehaviourVersion) {
		t.Errorf("a %s scenario discloses a version mismatch", EngineBehaviourVersion)
	}
}
