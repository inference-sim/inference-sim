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
// step is dearer than with the kernel unstated; and the price reads the backend's own entries.
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
			stated := planCase(t, d, "minimax-m2.5", "h200", 1, c.parallel, experts(c.impl))
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
