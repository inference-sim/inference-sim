package latencykernel

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-substrate/load"
)

// The memory a rank holds inside the KV budget besides its weights, as the kernel plans it, against
// what vLLM's own start-up decomposition logged for the same deployment (inference-sim/blis-substrate
// #76). vLLM subtracts the non-torch growth and the peak activation, and its logged peak activation
// already holds the CUDA-graph estimate (vllm/v1/worker/gpu_worker.py:720-756@v0.33.0), so the logged
// figure is peak activation + (consumed memory - Model loading took). The units are fit-role
// InferenceX start-ups of Kimi-K2.5-NVFP4 on B300 at TP4, vLLM 0.26.1rc1, whose derived figures the
// registry vendors (registry/validator/testdata/anchors/memory_startup_fit_units.json); the log's
// SHA-256 names each.

// startupCaptureSizes is vLLM's default capture list for a ceiling (vllm/config/vllm.py), as the
// memory writer computes it: 1, 2, 4, then every 8 below 256 and every 16 from 256.
func startupCaptureSizes(ceiling int) []int {
	var s []int
	for _, x := range []int{1, 2, 4} {
		if x <= ceiling {
			s = append(s, x)
		}
	}
	for x := 8; x <= ceiling && x < 256; x += 8 {
		s = append(s, x)
	}
	for x := 256; x <= ceiling; x += 16 {
		s = append(s, x)
	}
	return s
}

func TestPlannedNonKVMemoryIsWhatTheStartupLogged(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	const gib = 1 << 30
	for _, u := range []struct {
		log                         string
		ceiling, maxNumSeqs         int
		peakActivation, nonTorchGiB float64 // logged: peak activation; consumed - Model loading took
	}{
		{"bc8525f97df4bfd12f1fb6e656371844767afae072fbf8ca90e6ae54467cc930", 2, 1, 2.72, 140.76 - 139.81},
		{"a6bdd871412489bfc56ff1b2d9281eb4a794eaadf5d4e107ef541d1b4ecfafe7", 64, 32, 2.89, 140.87 - 139.81},
		{"dfae9668a6fd5d702801032e488caefe5e54855fa6ec1f267fa79d525c96e830", 512, 256, 3.83, 140.87 - 139.81},
	} {
		t.Run(fmt.Sprintf("ceiling%d", u.ceiling), func(t *testing.T) {
			in := planCase(t, d, "kimi-k2.5-nvfp4", "b300", 1, "tp: 4, pp: 1, dp: 1", func(l *plan.Lowered) {
				l.Plan.WeightFormat = string(model.DTypeNVFP4)
				l.Plan.Batch = plan.BatchLimits{Requests: u.maxNumSeqs, Tokens: 8192}
				l.Plan.Graphs.Buckets = startupCaptureSizes(u.ceiling)
				l.Plan.Graphs.Ceiling = units.Tokens(u.ceiling)
			})
			f := mustKernel(t, in).FixedBytes()
			planned := float64(f.ActivationPeak+f.CUDAGraph+f.NonTorchInBudget) / gib
			logged := u.peakActivation + u.nonTorchGiB
			// The fit's own spread on these units plus the log's 0.01 GiB rounding of each figure.
			if math.Abs(planned-logged) > 0.25 {
				t.Errorf("log %s: planned %.3f GiB inside the budget besides the weights (activation %.3f, "+
					"graphs %.3f, non-torch %.3f); vLLM logged %.3f", u.log[:12], planned,
					float64(f.ActivationPeak)/gib, float64(f.CUDAGraph)/gib, float64(f.NonTorchInBudget)/gib, logged)
			}
		})
	}
}

// Where no fit-unit group measured the non-torch growth, nothing is charged for it and the point
// says so: a stated gap, never a silent zero.
func TestAnUnmeasuredNonTorchTermIsAStatedGap(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	k := mustKernel(t, planCase(t, d, "kimi-k2.5-nvfp4", "b200", 1, "tp: 4, pp: 1, dp: 1", nil))
	if k.FixedBytes().NonTorchInBudget != 0 {
		t.Fatalf("charged %d B with no measurement in scope", k.FixedBytes().NonTorchInBudget)
	}
	_, _, assumed := k.Evidence()
	if !slices.ContainsFunc(assumed, func(s string) bool { return s == "nontorch_in_budget_bytes_4rank" }) {
		t.Errorf("no stated gap for the unmeasured non-torch term among %v", assumed)
	}
}
