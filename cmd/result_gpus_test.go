package cmd

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// output_tok_s_per_gpu is output_tok_s over every GPU the deployment occupies: each instance
// spans tp x pp x pcp x dp ranks (the engine builds one instance for a whole data-parallel
// group), and the pools of a disaggregated deployment add up (inference-sim#2254). The law is
// checked against GPU counts stated by hand from each scenario's layout, so a count that drops
// a factor shows as a figure off by exactly that factor.
func TestResult_OutputPerGPUCountsEveryParallelRank(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	dpBody := readFile(t, filepath.Join(scenarios, dpScenario))
	tp1dp4 := t.TempDir()
	edited := strings.Replace(strings.Replace(dpBody, "      tp: 2\n", "      tp: 1\n", 1), "      dp: 2\n", "      dp: 4\n", 1)
	if edited == dpBody {
		t.Fatal("the dp fixture no longer states tp 2 and dp 2")
	}
	if err := os.WriteFile(filepath.Join(tp1dp4, dpScenario), []byte(edited), 0o644); err != nil {
		t.Fatal(err)
	}
	pdDir, pdName := dpPoolScenarios(t, 4, 4)

	for _, c := range []struct {
		name       string
		args       []string
		gpus       int
		decodeGPUs int
	}{
		{"tp4 dp1 colocated", []string{"--scenario", kernelTestScenario, "--num-instances", "1"}, 4, 0},
		{"tp2 x dp2 colocated", []string{"--scenario", dpScenario, "--num-instances", "1"}, 4, 0},
		{"tp2 x dp2 colocated, two groups", []string{"--scenario", dpScenario, "--num-instances", "2"}, 8, 0},
		{"tp1 x dp4 colocated", []string{"--scenarios", tp1dp4, "--scenario", dpScenario, "--num-instances", "1"}, 4, 0},
		// prefill: 4 groups of tp2 x dp2 (4 GPUs); decode: 2 groups of tp4 x dp2 (8 GPUs).
		{"P/D pools at dp2", []string{"--scenarios", pdDir, "--scenario", pdName, "--num-instances", "6",
			"--prefill-instances", "4", "--decode-instances", "2"}, 32, 16},
	} {
		t.Run(c.name, func(t *testing.T) {
			args := append([]string{"run", "--rate", "10", "--num-requests", strconv.Itoa(dpFixtureNumRequests), "--seed", "42"}, c.args...)
			out, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
			}
			total, perGPU := metric(t, out, "output_tok_s"), metric(t, out, "output_tok_s_per_gpu")
			if total <= 0 {
				t.Fatalf("non-vacuity: output_tok_s = %v", total)
			}
			if got := total / perGPU; math.Abs(got-float64(c.gpus)) > 1e-6*float64(c.gpus) {
				t.Errorf("output_tok_s %v over output_tok_s_per_gpu %v implies %v GPUs, want %d", total, perGPU, got, c.gpus)
			}
			if c.decodeGPUs > 0 {
				if dec, ok := clusterFigures(t, out)["metric:output_tokens_per_second_per_decode_gpu"]; ok {
					if got := total / dec; math.Abs(got-float64(c.decodeGPUs)) > 1e-6*float64(c.decodeGPUs) {
						t.Errorf("the decode form implies %v decode GPUs, want %d", got, c.decodeGPUs)
					}
				}
			}
		})
	}
}
