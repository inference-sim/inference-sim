//go:build ignore

package latency

// H27 Mixed-Batch Max Combination Experiment
//
// This test compares the current weighted-average mixed-batch combination
// (roofline_step.go lines 407-430) against max(prefillTime, decodeTime)
// for a range of prefill/decode token ratios.
//
// Method:
//   For each test case with both prefill and decode requests:
//   1. Run rooflineStepTime with the mixed StepConfig -> "weighted_avg" time
//   2. Run rooflineStepTime with prefill-only StepConfig -> prefillTime
//   3. Run rooflineStepTime with decode-only StepConfig -> decodeTime
//   4. Compute max(prefillTime, decodeTime) -> "max" time
//   5. Compare: if weighted_avg < max, the weighted-average underpredicts

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
)

// h27OutputDir returns the output directory for H27 results.
// Uses runtime.Caller to find the source file location (in sim/),
// then navigates to the hypothesis output directory.
// Falls back to a relative path if runtime.Caller fails.
func h27OutputDir() string {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return filepath.Join("hypotheses", "h27-mixed-batch-max", "output")
	}
	// filename is <repo>/sim/h27_mixed_batch_max_test.go when copied to sim/
	repoRoot := filepath.Dir(filepath.Dir(filename))
	return filepath.Join(repoRoot, "hypotheses", "h27-mixed-batch-max", "output")
}

// h27TestCase defines a single mixed-batch test configuration.
type h27TestCase struct {
	Name            string
	PrefillRequests []PrefillRequestConfig
	DecodeRequests  []DecodeRequestConfig
	TP              int
}

// buildH27TestCases generates a comprehensive set of mixed-batch scenarios
// varying the prefill/decode token ratio.
func buildH27TestCases() []h27TestCase {
	cases := []h27TestCase{}

	// Scenario family 1: Fixed decode batch (8 requests at various KV lengths),
	// varying prefill token count
	decodeReqs8 := []DecodeRequestConfig{
		{ProgressIndex: 256},
		{ProgressIndex: 512},
		{ProgressIndex: 1024},
		{ProgressIndex: 1536},
		{ProgressIndex: 2048},
		{ProgressIndex: 3072},
		{ProgressIndex: 4096},
		{ProgressIndex: 6144},
	}

	for _, prefillTokens := range []int{16, 64, 128, 256, 512, 1024, 2048, 4096} {
		cases = append(cases, h27TestCase{
			Name: fmt.Sprintf("8decode_1prefill_%dtok", prefillTokens),
			PrefillRequests: []PrefillRequestConfig{
				{ProgressIndex: 0, NumNewPrefillTokens: prefillTokens},
			},
			DecodeRequests: decodeReqs8,
			TP:             1,
		})
	}

	// Scenario family 2: Fixed prefill (512 tokens), varying decode batch size
	for _, numDecode := range []int{1, 2, 4, 8, 16, 32, 64} {
		decodeReqs := make([]DecodeRequestConfig, numDecode)
		for i := range decodeReqs {
			// Spread KV lengths from 256 to 4096
			kvLen := 256 + int64(i)*(4096-256)/int64(max(numDecode-1, 1))
			decodeReqs[i] = DecodeRequestConfig{ProgressIndex: kvLen}
		}
		cases = append(cases, h27TestCase{
			Name: fmt.Sprintf("1prefill512_%ddecode", numDecode),
			PrefillRequests: []PrefillRequestConfig{
				{ProgressIndex: 0, NumNewPrefillTokens: 512},
			},
			DecodeRequests: decodeReqs,
			TP:             1,
		})
	}

	// Scenario family 3: Multiple prefill requests + decode requests
	// (simulates chunked-prefill with concurrent decodes)
	for _, numPrefill := range []int{1, 2, 4} {
		for _, numDecode := range []int{4, 8, 16} {
			prefillReqs := make([]PrefillRequestConfig, numPrefill)
			for i := range prefillReqs {
				prefillReqs[i] = PrefillRequestConfig{
					ProgressIndex:       0,
					NumNewPrefillTokens: 256,
				}
			}
			decodeReqs := make([]DecodeRequestConfig, numDecode)
			for i := range decodeReqs {
				kvLen := 512 + int64(i)*256
				decodeReqs[i] = DecodeRequestConfig{ProgressIndex: kvLen}
			}
			cases = append(cases, h27TestCase{
				Name: fmt.Sprintf("%dprefill256_%ddecode", numPrefill, numDecode),
				PrefillRequests: prefillReqs,
				DecodeRequests:  decodeReqs,
				TP:              1,
			})
		}
	}

	// Scenario family 4: TP=2 and TP=4 variants of representative mixed batches
	for _, tp := range []int{2, 4} {
		cases = append(cases, h27TestCase{
			Name: fmt.Sprintf("tp%d_1prefill512_8decode", tp),
			PrefillRequests: []PrefillRequestConfig{
				{ProgressIndex: 0, NumNewPrefillTokens: 512},
			},
			DecodeRequests: decodeReqs8,
			TP:             tp,
		})
		cases = append(cases, h27TestCase{
			Name: fmt.Sprintf("tp%d_1prefill2048_8decode", tp),
			PrefillRequests: []PrefillRequestConfig{
				{ProgressIndex: 0, NumNewPrefillTokens: 2048},
			},
			DecodeRequests: decodeReqs8,
			TP:             tp,
		})
	}

	// Scenario family 5: Extreme ratios (prefill-dominated and decode-dominated)
	// These trigger the branching thresholds in the current code
	cases = append(cases, h27TestCase{
		Name: "extreme_prefill_dominated_4096tok_2decode",
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 4096},
		},
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 1024},
			{ProgressIndex: 2048},
		},
		TP: 1,
	})
	cases = append(cases, h27TestCase{
		Name: "extreme_decode_dominated_32tok_32decode",
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 32},
		},
		DecodeRequests: func() []DecodeRequestConfig {
			reqs := make([]DecodeRequestConfig, 32)
			for i := range reqs {
				reqs[i] = DecodeRequestConfig{ProgressIndex: int64(256 + i*128)}
			}
			return reqs
		}(),
		TP: 1,
	})

	return cases
}

func TestH27_MixedBatchMaxComparison(t *testing.T) {
	mc := testModelConfig()
	hc := testHardwareCalib()
	mfuDB := loadTestMFUDatabase(t)

	cases := buildH27TestCases()

	// Prepare CSV output
	outputDir := h27OutputDir()
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("failed to create output dir: %v", err)
	}

	csvPath := filepath.Join(outputDir, "h27_results.csv")
	f, err := os.Create(csvPath)
	if err != nil {
		t.Fatalf("failed to create CSV: %v", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	// Write header
	header := []string{
		"case_name",
		"tp",
		"num_prefill_requests",
		"total_prefill_tokens",
		"num_decode_requests",
		"prefill_only_us",
		"decode_only_us",
		"max_combination_us",
		"weighted_avg_us",
		"ratio_wa_to_max",
		"delta_us",
		"regime",
	}
	if err := w.Write(header); err != nil {
		t.Fatalf("failed to write CSV header: %v", err)
	}

	var totalCases, underpredictCases int
	var sumRatio float64

	for _, tc := range cases {
		// 1. Mixed step (current weighted-average combination)
		mixedStep := StepConfig{
			PrefillRequests: tc.PrefillRequests,
			DecodeRequests:  tc.DecodeRequests,
		}
		weightedAvgUS := rooflineStepTime(mc, hc, mixedStep, tc.TP, mfuDB)

		// 2. Prefill-only step
		prefillOnlyStep := StepConfig{
			PrefillRequests: tc.PrefillRequests,
		}
		prefillOnlyUS := rooflineStepTime(mc, hc, prefillOnlyStep, tc.TP, mfuDB)

		// 3. Decode-only step
		decodeOnlyStep := StepConfig{
			DecodeRequests: tc.DecodeRequests,
		}
		decodeOnlyUS := rooflineStepTime(mc, hc, decodeOnlyStep, tc.TP, mfuDB)

		// 4. Max combination
		maxCombinationUS := int64(math.Max(float64(prefillOnlyUS), float64(decodeOnlyUS)))

		// 5. Compute ratio and delta
		var ratio float64
		if maxCombinationUS > 0 {
			ratio = float64(weightedAvgUS) / float64(maxCombinationUS)
		}
		deltaUS := weightedAvgUS - maxCombinationUS

		// Classify token ratio regime
		totalPrefillTokens := 0
		for _, req := range tc.PrefillRequests {
			totalPrefillTokens += req.NumNewPrefillTokens
		}
		numDecodeTokens := len(tc.DecodeRequests)
		regime := classifyRegime(totalPrefillTokens, numDecodeTokens)

		// Write row
		row := []string{
			tc.Name,
			strconv.Itoa(tc.TP),
			strconv.Itoa(len(tc.PrefillRequests)),
			strconv.Itoa(totalPrefillTokens),
			strconv.Itoa(numDecodeTokens),
			strconv.FormatInt(prefillOnlyUS, 10),
			strconv.FormatInt(decodeOnlyUS, 10),
			strconv.FormatInt(maxCombinationUS, 10),
			strconv.FormatInt(weightedAvgUS, 10),
			fmt.Sprintf("%.4f", ratio),
			strconv.FormatInt(deltaUS, 10),
			regime,
		}
		if err := w.Write(row); err != nil {
			t.Fatalf("failed to write CSV row: %v", err)
		}

		// Track statistics
		totalCases++
		sumRatio += ratio
		if weightedAvgUS < maxCombinationUS {
			underpredictCases++
		}

		t.Logf("%-50s tp=%d  WA=%8d  Max=%8d  ratio=%.4f  delta=%+8d  regime=%s",
			tc.Name, tc.TP, weightedAvgUS, maxCombinationUS, ratio, deltaUS, regime)
	}

	// Summary statistics
	avgRatio := sumRatio / float64(totalCases)
	underpredictPct := float64(underpredictCases) / float64(totalCases) * 100

	t.Logf("")
	t.Logf("=== H27 Summary ===")
	t.Logf("Total cases:        %d", totalCases)
	t.Logf("Underpredict cases: %d (%.1f%%)", underpredictCases, underpredictPct)
	t.Logf("Avg WA/Max ratio:   %.4f", avgRatio)
	t.Logf("Results written to: %s", csvPath)

	// Write summary to separate file for analyze.py
	summaryPath := filepath.Join(outputDir, "h27_summary.txt")
	sf, err := os.Create(summaryPath)
	if err != nil {
		t.Fatalf("failed to create summary: %v", err)
	}
	defer sf.Close()
	fmt.Fprintf(sf, "total_cases=%d\n", totalCases)
	fmt.Fprintf(sf, "underpredict_cases=%d\n", underpredictCases)
	fmt.Fprintf(sf, "underpredict_pct=%.2f\n", underpredictPct)
	fmt.Fprintf(sf, "avg_wa_max_ratio=%.6f\n", avgRatio)
}

// classifyRegime determines the token ratio regime for a mixed batch.
// Matches the branching thresholds in roofline_step.go lines 419-430.
func classifyRegime(prefillTokens, decodeTokens int) string {
	pf := float64(prefillTokens)
	df := float64(decodeTokens)

	if pf > df*4 && pf > 100 {
		return "prefill-dominated"
	} else if df > pf*2 && df > 50 {
		return "decode-dominated"
	}
	return "balanced"
}

