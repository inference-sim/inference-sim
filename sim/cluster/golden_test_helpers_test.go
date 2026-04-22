package cluster

import (
	"math"
	"path/filepath"
	"runtime"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// Shared golden test helpers used by both roofline_golden_test.go and
// trained_physics_golden_test.go to avoid duplication.

// goldenExpected holds expected metrics for a golden dataset experiment.
type goldenExpected struct {
	CompletedRequests int     `json:"completed_requests"`
	TotalInputTokens  int     `json:"total_input_tokens"`
	TotalOutputTokens int     `json:"total_output_tokens"`
	TTFTSumUs         int64   `json:"ttft_sum_us"`
	TTFTMeanMs        float64 `json:"ttft_mean_ms"`
	TTFTP90Ms         float64 `json:"ttft_p90_ms"`
	TTFTP99Ms         float64 `json:"ttft_p99_ms"`
	E2EMeanMs         float64 `json:"e2e_mean_ms"`
	E2EP90Ms          float64 `json:"e2e_p90_ms"`
	E2EP99Ms          float64 `json:"e2e_p99_ms"`
	ITLMeanMs         float64 `json:"itl_mean_ms"`
}

// goldenWorkloadSpec mirrors the top-level keys written by the Python runner's
// write_workload_spec (v2 inference_perf format). Fields carry both json and
// yaml struct tags so the struct can be decoded from either format.
type goldenWorkloadSpec struct {
	Version       string                 `json:"version"       yaml:"version"`
	Seed          int64                  `json:"seed"          yaml:"seed"`
	NumRequests   int                    `json:"num_requests"  yaml:"num_requests"`
	InferencePerf *goldenInferencePerfWS `json:"inference_perf" yaml:"inference_perf"`
}

type goldenInferencePerfWS struct {
	Stages       []goldenStage              `json:"stages"        yaml:"stages"`
	SharedPrefix *workload.SharedPrefixSpec `json:"shared_prefix" yaml:"shared_prefix"`
}

type goldenStage struct {
	Rate     float64 `json:"rate"     yaml:"rate"`
	Duration int64   `json:"duration" yaml:"duration"`
}

// goldenRepoRoot returns the absolute path to the repository root, resolved
// relative to the calling source file (sim/cluster/ → ../.. → repo root).
func goldenRepoRoot() string {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		panic("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(thisFile), "..", "..")
}

// goldenSortedValues returns the values of a map[string]float64 sorted in
// ascending order. The key iteration is sorted first (R2: deterministic map
// traversal), then the extracted values are sorted by value for use in
// percentile and mean calculations.
func goldenSortedValues(m map[string]float64) []float64 {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys) // R2: deterministic iteration order
	vals := make([]float64, len(keys))
	for i, k := range keys {
		vals[i] = m[k]
	}
	sort.Float64s(vals) // sort by value for percentile/mean computation
	return vals
}

// goldenAssertApprox asserts that got ≈ want within the given relative tolerance.
func goldenAssertApprox(t *testing.T, name string, want, got, relTol float64) {
	t.Helper()
	if want == 0 && got == 0 {
		return
	}
	diff := math.Abs(want - got)
	maxVal := math.Max(math.Abs(want), math.Abs(got))
	if maxVal > 0 && diff/maxVal > relTol {
		t.Errorf("%s: got %.10f, want %.10f (relDiff=%.2e, tolerance=%.2e)",
			name, got, want, diff/maxVal, relTol)
	}
}
