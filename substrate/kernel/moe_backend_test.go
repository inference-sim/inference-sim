package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// decodeStep is the step time of n one-token decodes at a 2,048-token context.
func decodeStep(t *testing.T, in Inputs, n int) int64 {
	t.Helper()
	k, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	reqs := make([]kernel.ReqShape, n)
	for i := range reqs {
		reqs[i] = kernel.ReqShape{Scheduled: 1, Computed: 2048, PromptLen: 1024}
	}
	return int64(k.StepTime(kernel.Batch{Reqs: reqs, DecodeThreshold: 1}).NoOverlap)
}

func experts(impl plan.KernelImpl) func(*plan.Lowered) {
	return func(l *plan.Lowered) { l.Plan.MoE.Experts = impl }
}

// pinnedBuild is the engine pin's build of an experts kernel on a part, as the registry keys its
// transferred entry: what an engine that identifies the build states (plan.MoEPlan.Experts).
func pinnedBuild(t *testing.T, d *load.Data, name, chip string) plan.KernelImpl {
	t.Helper()
	set, err := d.CoefficientSet("cost-model-primitives")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range set.Coefficients {
		if e.Name == name && string(e.Method) == "transferred" && len(e.Scope.Hardware) == 1 && e.Scope.Hardware[0] == chip &&
			len(e.KernelImpls) == 1 {
			k := e.KernelImpls[0]
			return plan.KernelImpl{Library: k.Library, Variant: k.Variant, CodeHash: k.CodeHash}
		}
	}
	t.Fatalf("the registry carries no transferred %s on %s", name, chip)
	return plan.KernelImpl{}
}

func provenanceNames(t *testing.T, in Inputs) string {
	t.Helper()
	k, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var b strings.Builder
	for _, o := range k.Provenance() {
		b.WriteString(o.Name + "=" + o.Scope + "\n")
	}
	return b.String()
}

// Block-FP8 routed experts on Hopper are priced on the ramp and floor measured for the grouped
// kernel the plan says the engine runs (blis-substrate#150): Triton's at expert-parallel width
// 1, FlashInfer's CUTLASS experts above it. A few-token decode then pays each layer's measured
// floor, which the dense FP8 ramp the routed term was priced on before does not charge, so the
// step is dearer than with the kernel unstated; and the price reads the backend's own entries, which
// are bound to the build the plan states (VAL-8).
func TestHopperFP8ExpertsArePricedOnTheirBackendsRamp(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name, parallel string
		impl           plan.KernelImpl
		entry          string
	}{
		{"triton tp4", "tp: 4, pp: 1, dp: 1", plan.ExpertsTriton, "gemm_floor_fp8_grouped_triton"},
		{"flashinfer_cutlass ep4", "tp: 1, pp: 1, dp: 4, enable_expert_parallel: true", plan.ExpertsFlashInferCutlass,
			"gemm_floor_fp8_grouped_flashinfer_cutlass"},
	} {
		t.Run(c.name, func(t *testing.T) {
			base := planCase(t, d, "minimax-m2.5", "h200", 1, c.parallel, nil)
			build := pinnedBuild(t, d, c.entry, "h200")
			if build.Variant != c.impl.Variant {
				t.Fatalf("%s is keyed to %+v, not the %s kernel", c.entry, build, c.impl.Variant)
			}
			stated := planCase(t, d, "minimax-m2.5", "h200", 1, c.parallel, experts(build))
			// The kernel named without its build reads no keyed entry (VAL-8) and is priced as unstated
			// with the fallback recorded.
			unbuilt := planCase(t, d, "minimax-m2.5", "h200", 1, c.parallel, experts(c.impl))
			if a, b := decodeStep(t, unbuilt, 4), decodeStep(t, base, 4); a != b {
				t.Errorf("%v with no build prices 4 decodes at %d ns, unstated at %d: no keyed entry may be read", c.impl, a, b)
			}
			for _, n := range []int{1, 4} {
				if a, b := decodeStep(t, stated, n), decodeStep(t, base, n); a <= b {
					t.Errorf("%d decodes: %v experts price the step at %d ns, unstated at %d: the measured floor must cost more", n, c.impl, a, b)
				}
			}
			if p := provenanceNames(t, stated); !strings.Contains(p, c.entry+"=") {
				t.Errorf("the price does not read %s:\n%s", c.entry, p)
			}
		})
	}
}

// An experts kernel the registry has no ramp for is priced by its weight format, as an unstated
// one is, and the price says so.
func TestAnExpertsKernelWithNoRampIsPricedByFormatAndRecorded(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	at := "tp: 4, pp: 1, dp: 1"
	base := planCase(t, d, "minimax-m2.5", "h200", 1, at, nil)
	other := planCase(t, d, "minimax-m2.5", "h200", 1, at, experts(plan.KernelImpl{Library: "deep_gemm", Variant: "grouped_experts"}))
	if a, b := decodeStep(t, other, 4), decodeStep(t, base, 4); a != b {
		t.Errorf("a deep_gemm experts kernel prices the step at %d ns, unstated at %d: with no ramp it is the weight format's price", a, b)
	}
	if p := provenanceNames(t, other); !strings.Contains(p, "moe_experts_kernel=deep_gemm grouped_experts") {
		t.Errorf("no assumption records the experts kernel priced by format:\n%s", p)
	}
}
