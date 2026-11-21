// Original to blis-sglang.

package blissglang

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// Behaviour contract (the kernel-backend fields).
//
// SGLang picks the MoE, FP8 GEMM, FP4 GEMM and sampling kernels by four fields. The cost model
// prices what the engine's own resolution of an unset field runs on the part, from tables
// measured on that kernel. The field list says, for each value a run states, whether that
// pricing is the engine's kernel (an exact stand-in), a different kernel with a named
// difference (an approximation), or not known (neither, so the field stays refused).
//
//   - moe_runner_backend flashinfer_trtllm on compute capability 10.x is approximated, never
//     exact: the engine resolves auto to it only for some architectures and checkpoints, which
//     the field list cannot see. flashinfer_cutlass, flashinfer_mxfp4 and marlin are neither.
//   - fp8_gemm_runner_backend flashinfer_trtllm on 10.x is approximated: the registry ramp was
//     measured on the FlashInfer (M<32) / DeepGEMM (M>=32) dispatch. Other values are neither.
//   - fp4_gemm_runner_backend flashinfer_cutlass on 10.x is approximated: the registry ramp was
//     measured on the CuTe DSL kernel, which auto resolves to on SM100. Other values are neither.
//   - sampling_backend flashinfer is an exact stand-in on an NVIDIA part, since it is what the
//     engine resolves for an unset field. pytorch is neither, and nor is any value on a part
//     with no compute capability (AMD), where the engine's default is pytorch.
//   - A value never matches on a part whose compute capability is outside the clause.
//   - Every approximation prices the engine's own resolution of the unset field (a null mode),
//     and none of these fields becomes modelled: the stated value is still refused by the
//     config until the kernel is lowered.
func TestKernelBackendFieldsAreClassifiedPerValue(t *testing.T) {
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
		{"moe_runner_backend", "flashinfer_trtllm", "10.0", approx},
		{"moe_runner_backend", "FlashInfer_TRTLLM", "10.3", approx},
		{"moe_runner_backend", "flashinfer_trtllm", "9.0", none},
		{"moe_runner_backend", "flashinfer_trtllm", "12.0", none},
		{"moe_runner_backend", "flashinfer_trtllm", "", none},
		{"moe_runner_backend", "flashinfer_cutlass", "10.0", none},
		{"moe_runner_backend", "flashinfer_mxfp4", "10.0", none},
		{"moe_runner_backend", "marlin", "10.0", none},
		{"moe_runner_backend", "auto", "10.0", none},

		{"fp8_gemm_runner_backend", "flashinfer_trtllm", "10.0", approx},
		{"fp8_gemm_runner_backend", "flashinfer_trtllm", "10.3", approx},
		{"fp8_gemm_runner_backend", "flashinfer_trtllm", "9.0", none},
		{"fp8_gemm_runner_backend", "flashinfer_trtllm", "12.0", none},
		{"fp8_gemm_runner_backend", "deep_gemm", "10.0", none},
		{"fp8_gemm_runner_backend", "triton", "10.0", none},

		{"fp4_gemm_runner_backend", "flashinfer_cutlass", "10.0", approx},
		{"fp4_gemm_runner_backend", "flashinfer_cutlass", "10.3", approx},
		{"fp4_gemm_runner_backend", "flashinfer_cutlass", "9.0", none},
		{"fp4_gemm_runner_backend", "flashinfer_cutlass", "12.0", none},
		{"fp4_gemm_runner_backend", "marlin", "10.0", none},
		{"fp4_gemm_runner_backend", "flashinfer_cutedsl", "10.0", none},

		{"sampling_backend", "flashinfer", "9.0", exact},
		{"sampling_backend", "flashinfer", "10.0", exact},
		{"sampling_backend", "FlashInfer", "8.0", exact},
		{"sampling_backend", "flashinfer", "", none},
		{"sampling_backend", "pytorch", "9.0", none},
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

	for _, name := range []string{"moe_runner_backend", "fp8_gemm_runner_backend", "fp4_gemm_runner_backend", "sampling_backend"} {
		f := byName[name]
		if f.Status != fields.StatusNotModelled {
			t.Errorf("%s: status %q, want not_modelled until the kernel is lowered", name, f.Status)
		}
		a := f.Estimation.Approximates
		if a == nil {
			continue
		}
		if v, ok := a.Mode[name]; !ok || v != nil {
			t.Errorf("%s: the mode is %v, want %s null: the engine's own resolution of the unset field is what is priced", name, a.Mode, name)
		}
		if strings.TrimSpace(a.Evidence) == "" || strings.TrimSpace(a.Differs) == "" {
			t.Errorf("%s: an approximation names no evidence or no difference", name)
		}
	}
}
