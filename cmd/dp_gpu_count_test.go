package cmd

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/metrics"
)

// A data-parallel deployment is one instance whose engine is the whole DP group (blis-vllm's
// v0.33.0 engine), so the GPUs that instance occupies are its pool's
// tp x pp x pcp x dp, every rank of the group (#2254). The Result's per-GPU figures divide by
// that count: output_tok_s over output_tok_s_per_gpu is the run's GPUs, and over the decode form
// the decode pool's. The law holds whatever the step pricing does, so the test pins the
// count rather than a throughput.
func TestRunGPUs_CountDataParallelRanks(t *testing.T) {
	pd, err := filepath.Abs(pdScenarios)
	if err != nil {
		t.Fatal(err)
	}
	_, haveDecode := metrics.Lookup(outputTokensPerDecodeGPU)
	cases := []struct {
		name       string
		args       []string
		gpus       int // tp x pp x pcp x dp per instance, summed over the instances
		decodeGPUs int // the decode pool's share, 0 for a colocated run
	}{
		{
			name: "colocated tp2 dp2, one instance",
			args: dpRunArgs(1),
			gpus: 2 * 2,
		},
		{
			name: "colocated tp2 dp2, two instances",
			args: dpRunArgs(2),
			gpus: 2 * (2 * 2),
		},
		{
			name: "P/D tp4 dp2 in both pools, 1P + 1D",
			args: append([]string{"run", "--scenarios", pd, "--scenario", pdDPScenario,
				"--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1",
				"--seed", "7", "--horizon", "900000000"}, burst(8, 8, 512, 32)...),
			gpus:       2 * (4 * 2),
			decodeGPUs: 4 * 2,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stdout, stderr, err := runKernelCLI(t, tc.args...)
			if err != nil {
				t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
			}
			figs, _ := resultDoc(t, stdout)["metrics"].(map[string]any)
			value := func(id string) (float64, bool) {
				f, ok := figs[id].(map[string]any)
				v, _ := f["value"].(float64)
				return v, ok
			}
			out, ok := value("output_tok_s")
			if !ok || out <= 0 {
				t.Fatalf("non-vacuity: the run states no output_tok_s: %v", figs["output_tok_s"])
			}
			perGPU, ok := value("output_tok_s_per_gpu")
			if !ok || perGPU <= 0 {
				t.Fatalf("the run states no output_tok_s_per_gpu: %v", figs["output_tok_s_per_gpu"])
			}
			if got := out / perGPU; math.Abs(got-float64(tc.gpus)) > 1e-9*float64(tc.gpus) {
				t.Errorf("output_tok_s / output_tok_s_per_gpu = %v GPUs, want %d (tp x pp x pcp x dp per instance)", got, tc.gpus)
			}
			if tc.decodeGPUs == 0 {
				return
			}
			if !haveDecode {
				t.Logf("the registry at this pin defines no %s: the decode pool's count is not checked", outputTokensPerDecodeGPU)
				return
			}
			perDecode, ok := value(outputTokensPerDecodeGPU)
			if !ok || perDecode <= 0 {
				t.Fatalf("the registry defines %s and the disaggregated run states none", outputTokensPerDecodeGPU)
			}
			if got := out / perDecode; math.Abs(got-float64(tc.decodeGPUs)) > 1e-9*float64(tc.decodeGPUs) {
				t.Errorf("output_tok_s / %s = %v GPUs, want the decode pool's %d", outputTokensPerDecodeGPU, got, tc.decodeGPUs)
			}
		})
	}
}
