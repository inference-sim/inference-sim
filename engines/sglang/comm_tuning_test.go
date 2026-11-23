// Original to blis-sglang.

package blissglang

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// Behaviour contract (the communication-tuning fields).
//
// The cost model prices every collective from the NCCL sweep at NCCL's defaults and does not
// read enable_symm_mem or the NCCL variables. The field list says, per stated value, whether
// that pricing is what the run used (an exact stand-in), a different configuration with a named
// difference (an approximation), or not known (neither, so the field stays refused).
//
//   - NCCL_MNNVL_ENABLE 1 and 2 are exact: 2 is NCCL's default, and 1 differs from it only in
//     failing initialization, which a completed run did not do. 0 is neither.
//   - NCCL_CUMEM_ENABLE 1 and NCCL_NVLS_ENABLE 1 and 2 are approximated, never exact: SGLang
//     writes 0 for either variable when it is absent, so leaving it unset is not the stated value.
//     0 is neither.
//   - enable_symm_mem true on compute capability 9.x and 10.x is approximated: the symmetric-memory
//     all-reduce can run NVLS kernels the sweep did not. false is neither, and so is a part
//     outside the range.
//   - Every approximation prices the engine's own resolution of the unset field (a null mode),
//     and none of these fields becomes modelled.
func TestCommTuningFieldsAreClassifiedPerValue(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]fields.Field{}
	for _, f := range l.Fields {
		byName[f.Name] = f
	}
	const (
		none   = "neither"
		approx = "approximated"
		exact  = "exact"
	)
	for _, c := range []struct {
		field, value, cc, want string
	}{
		{"NCCL_MNNVL_ENABLE", "1", "10.0", exact},
		{"NCCL_MNNVL_ENABLE", "2", "9.0", exact},
		{"NCCL_MNNVL_ENABLE", "0", "10.0", none},

		{"NCCL_CUMEM_ENABLE", "1", "9.0", approx},
		{"NCCL_CUMEM_ENABLE", "1", "", approx},
		{"NCCL_CUMEM_ENABLE", "0", "9.0", none},

		{"NCCL_NVLS_ENABLE", "1", "9.0", approx},
		{"NCCL_NVLS_ENABLE", "2", "10.0", approx},
		{"NCCL_NVLS_ENABLE", "0", "9.0", none},

		{"enable_symm_mem", "true", "9.0", approx},
		{"enable_symm_mem", "True", "10.3", approx},
		{"enable_symm_mem", "false", "9.0", none},
		{"enable_symm_mem", "true", "8.0", none},
		{"enable_symm_mem", "true", "12.0", none},
		{"enable_symm_mem", "true", "", none},
	} {
		f, ok := byName[c.field]
		if !ok {
			t.Fatalf("%s is not in the field list", c.field)
		}
		facts := fields.Facts{Value: c.value, ComputeCapability: c.cc}
		got := none
		if _, ok := fields.Approximate(f.Estimation, facts); ok {
			got = approx
		}
		if fields.Exact(f.Estimation, facts) {
			if got == approx {
				t.Errorf("%s=%q on %q is both exact and approximated", c.field, c.value, c.cc)
			}
			got = exact
		}
		if got != c.want {
			t.Errorf("%s=%q on compute capability %q: %s, want %s", c.field, c.value, c.cc, got, c.want)
		}
	}

	for _, c := range []struct{ name, mode string }{
		{"NCCL_CUMEM_ENABLE", "env.NCCL_CUMEM_ENABLE"},
		{"NCCL_NVLS_ENABLE", "env.NCCL_NVLS_ENABLE"},
		{"enable_symm_mem", "enable_symm_mem"},
	} {
		f := byName[c.name]
		if f.Status != fields.StatusNotModelled {
			t.Errorf("%s: status %q, want not_modelled", c.name, f.Status)
		}
		a := f.Estimation.Approximates
		if a == nil {
			t.Errorf("%s: no approximation", c.name)
			continue
		}
		if v, ok := a.Mode[c.mode]; !ok || v != nil {
			t.Errorf("%s: the mode is %v, want %s null: the engine's own resolution of the unset field is what is priced", c.name, a.Mode, c.mode)
		}
	}
}
