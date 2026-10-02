package kernelmodel

import (
	"os"
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

// A per-rank occupancy must be physically possible: it cannot exceed the part it sits on.
// The guard in KVBudget exists because an impossible figure divided into a budget yields a
// plausible-looking capacity built on a wrong weight term, and the resident batch is the
// quantity this experiment measures.
//
// This asserts the guard does not fire on any evaluation deployment -- i.e. that every one
// of them reports a possible occupancy and a derivable budget. It replaced a test that
// asserted the opposite: before the upstream /expertTensorShards fix, 17 of 39 deployments
// reported more per-rank memory than their device had.
func TestEveryEvaluationDeploymentHasADerivableBudget(t *testing.T) {
	for _, scenario := range evaluationScenarios(t) {
		m, err := Open(scenario, repos())
		if err != nil {
			t.Errorf("%s: Open: %v", scenario, err)
			continue
		}
		b, err := m.KVBudget()
		if err != nil {
			t.Errorf("%s: %v", scenario, err)
			continue
		}
		if b.TotalBlocks <= 0 {
			t.Errorf("%s: %d blocks", scenario, b.TotalBlocks)
		}
		if b.FixedBytes >= b.DeviceBytes {
			t.Errorf("%s: per-rank occupancy %.1f GiB does not fit a %.1f GiB device",
				scenario, float64(b.FixedBytes)/(1<<30),
				float64(b.DeviceBytes)/(1<<30))
		}
	}
}

// Expert weights must be TENSOR-SLICED when expert parallelism is off: a rank holds every
// expert, each divided across the tensor-parallel group. The signature is arithmetic rather
// than a recorded byte count.
//
// GLM-5 served fp8 has 75 MoE layers of 256 experts, three matrices each at 2048x6144, so
// 675 GiB of expert parameters for the whole model and 84.4 GiB for one rank of eight. A
// rank reporting the whole model's figure is the defect this test was written against; it
// now pins the fix.
func TestExpertWeightsAreTensorSlicedWhenExpertParallelismIsOff(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	if w := m.Kernel().Resolved().ExpertParallelWidth; w > 1 {
		t.Skipf("this scenario resolved expert-parallel width %d; the test needs it off", w)
	}
	const (
		wholeModelExpertBytes = 75.0 * 256 * 3 * 2048 * 6144
		tp                    = 8.0
	)
	perRank := wholeModelExpertBytes / tp
	got := float64(m.Kernel().FixedBytes().Weights)

	// Above the expert slice (dense layers, embeddings and the head add to it) and well
	// below twice it. The undivided figure is 8x, so this band excludes it decisively.
	if got < perRank || got > perRank*1.5 {
		t.Errorf("a tp=8 rank reports %.1f GiB of weights; the expert slice is %.1f GiB "+
			"and the dense remainder is small, so anything outside [%.1f, %.1f] GiB means "+
			"the expert term is sharded wrongly",
			got/(1<<30), perRank/(1<<30), perRank/(1<<30), perRank*1.5/(1<<30))
	}
}

// evaluationScenarios lists the scenario files the comparison runs over. Reading the
// directory rather than hardcoding a list means a scenario added upstream is covered
// automatically instead of silently skipped.
func evaluationScenarios(t *testing.T) []string {
	t.Helper()
	ents, err := os.ReadDir(repos().Scenarios)
	if err != nil {
		t.Fatalf("reading scenarios: %v", err)
	}
	var out []string
	for _, e := range ents {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".yaml") {
			out = append(out, e.Name())
		}
	}
	if len(out) == 0 {
		t.Fatal("no scenario files found, so this test would pass vacuously")
	}
	return out
}
