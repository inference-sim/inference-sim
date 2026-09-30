package kernelmodel

import (
	"strings"
	"testing"
)

// A KV budget must be derived from the kernel's own memory answers, and it must be
// internally consistent: the blocks it reports, times the bytes one block costs, must fit
// in what is left of the device after fixed occupancy. A budget that did not satisfy this
// would admit requests the modelled device cannot hold.
func TestKVBudgetFitsTheDeviceItWasDerivedFrom(t *testing.T) {
	m := open(t, "gpt-oss-120b-h200-fp4-vllm-tp8.yaml")
	b, err := m.KVBudget()
	if err != nil {
		t.Fatalf("KVBudget: %v", err)
	}
	if b.TotalBlocks <= 0 {
		t.Fatalf("budget has %d blocks", b.TotalBlocks)
	}
	if used := b.TotalBlocks * b.PerBlockBytes; used > b.AllocatableBytes {
		t.Errorf("%d blocks x %d bytes = %d exceeds the %d allocatable",
			b.TotalBlocks, b.PerBlockBytes, used, b.AllocatableBytes)
	}
	if b.FixedBytes+b.AllocatableBytes != b.BudgetBytes {
		t.Errorf("fixed %d + allocatable %d != budget %d",
			b.FixedBytes, b.AllocatableBytes, b.BudgetBytes)
	}
	if b.BudgetBytes >= b.DeviceBytes {
		t.Errorf("budget %d is not below device %d; utilization was not applied",
			b.BudgetBytes, b.DeviceBytes)
	}
}

// A narrower tensor-parallel width must yield FEWER KV blocks per rank, because each rank
// holds a larger KV share. This is the behaviour a resident-batch bound depends on: if the
// budget did not move with the layout, every deployment would admit the same number of
// requests regardless of how it was sharded.
func TestKVBudgetGrowsWithTensorParallelWidth(t *testing.T) {
	widths := []struct {
		tp       int
		scenario string
	}{
		{1, "gpt-oss-120b-h200-fp4-vllm-tp1.yaml"},
		{2, "gpt-oss-120b-h200-fp4-vllm-tp2.yaml"},
		{4, "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"},
		{8, "gpt-oss-120b-h200-fp4-vllm-tp8.yaml"},
	}
	var prev int64
	for i, w := range widths {
		b, err := open(t, w.scenario).KVBudget()
		if err != nil {
			t.Fatalf("tp=%d: %v", w.tp, err)
		}
		if i > 0 && b.TotalBlocks <= prev {
			t.Errorf("tp=%d gives %d blocks, not above the %d at the narrower width",
				w.tp, b.TotalBlocks, prev)
		}
		prev = b.TotalBlocks
	}
}

// The refusal must fire, with a diagnostic, when the kernel reports a per-rank occupancy
// above the device size. Compensating silently would produce a plausible capacity built on
// a known-wrong weight figure, and the resident batch is the quantity this whole experiment
// measures.
//
// Behavioural: it asserts that no budget is returned and that the error names the defect,
// not that any particular source line exists.
func TestAnImpossiblePerRankOccupancyIsRefusedNotCompensated(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	b, err := m.KVBudget()
	if err == nil {
		t.Fatalf("a %.1f GiB per-rank occupancy on a 141 GiB device produced a budget of "+
			"%d blocks instead of an error",
			float64(m.Kernel().FixedBytes().Total())/(1<<30), b.TotalBlocks)
	}
	if b.TotalBlocks != 0 {
		t.Errorf("a refused budget still reported %d blocks", b.TotalBlocks)
	}
	for _, want := range []string{"per-rank", "UpstreamExpertWeightDefect"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not mention %q, so a reader cannot tell why "+
				"capacity is unavailable: %v", want, err)
		}
	}
}

// The defect's signature, asserted as behaviour: with expert parallelism OFF, the kernel's
// reported weights are consistent with undivided expert bytes. This test documents the bug
// so that fixing it upstream makes this test fail loudly rather than leaving a stale
// workaround in place.
func TestExpertWeightsAreUndividedWithExpertParallelismOff(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	if w := m.Kernel().Resolved().ExpertParallelWidth; w > 1 {
		t.Skipf("this scenario resolved expert-parallel width %d; the defect needs it off", w)
	}
	// GLM-5: 75 MoE layers, 256 experts, 3 matrices, n=2048, k=6144, fp8 at 1 byte.
	const wholeModelExpertBytes = 75.0 * 256 * 3 * 2048 * 6144
	got := float64(m.Kernel().FixedBytes().Weights)
	// Undivided, the expert term alone is 675 GiB; the reported figure adds dense layers,
	// embeddings and the head, so it sits just above. Correctly divided by tp=8 it would be
	// near 84 GiB. A 20% band distinguishes those two cases unambiguously.
	if got < wholeModelExpertBytes*0.95 {
		t.Errorf("reported weights %.1f GiB are below the undivided expert term "+
			"%.1f GiB: the upstream defect may have been fixed, in which case the "+
			"refusal in KVBudget and its documentation should be removed",
			got/(1<<30), wholeModelExpertBytes/(1<<30))
	}
}
