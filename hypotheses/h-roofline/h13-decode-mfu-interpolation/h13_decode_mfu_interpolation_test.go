//go:build ignore

package latency

import (
	"fmt"
	"math"
	"sort"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// =============================================================================
// H13: Decode MFU Interpolation Accuracy
//
// Hypothesis: The decode attention MFU values obtained via bilinear interpolation
// from the bench_data grid systematically underestimate actual FlashAttention
// decode kernel efficiency, because the MFU grid was measured at coarse
// (batchSize, kvLen) anchor points that miss the higher efficiency achieved at
// intermediate operating points.
//
// Since we cannot run actual kernel benchmarks in this test, we verify
// structural properties of the interpolation:
//   (a) On-grid accuracy: interpolation returns exact grid values
//   (b) Boundedness: off-grid values lie within the convex hull of neighbors
//   (c) Monotonicity along each axis (where the grid data is monotone)
//   (d) MFU range sanity at representative eval operating points
//   (e) Interpolation vs nearest-neighbor delta magnitude
// =============================================================================

// h13LoadMFUDatabase loads the real bench_data for H13 tests.
// Skips the test if bench_data is not available.
func h13LoadMFUDatabase(t *testing.T) *sim.MFUDatabase {
	t.Helper()
	mc := testModelConfig() // Llama-3.1-8B: 32-8-128
	benchDataPath := "../bench_data"
	db, err := sim.NewMFUDatabase(mc, benchDataPath, "h100")
	if err != nil {
		t.Skipf("bench_data not available, skipping H13 test: %v", err)
	}
	return db
}

// h13ExtractDecodeGrid extracts the (batchSize, kvLen) -> MFU grid for decode
// attention from the MFU database for the given TP config.
func h13ExtractDecodeGrid(t *testing.T, db *sim.MFUDatabase, tp int) (
	bsVals []int, kvVals []int, grid map[[2]int]float64,
) {
	t.Helper()
	configKey := fmt.Sprintf("%s-tp%d", db.attentionConfig, tp)
	rows := db.decodeData[configKey]
	if len(rows) == 0 {
		t.Fatalf("no decode data for config %s", configKey)
	}

	grid = make(map[[2]int]float64)
	bsSet := make(map[int]bool)
	kvSet := make(map[int]bool)

	for _, r := range rows {
		grid[[2]int{r.BatchSize, r.KVLen}] = r.MFU
		bsSet[r.BatchSize] = true
		kvSet[r.KVLen] = true
	}

	bsVals = make([]int, 0, len(bsSet))
	for v := range bsSet {
		bsVals = append(bsVals, v)
	}
	sort.Ints(bsVals)

	kvVals = make([]int, 0, len(kvSet))
	for v := range kvSet {
		kvVals = append(kvVals, v)
	}
	sort.Ints(kvVals)

	return bsVals, kvVals, grid
}

// --- Test (a): On-grid accuracy ---
// Interpolation at exact grid points must return the exact grid MFU value.
func TestH13_OnGridAccuracy(t *testing.T) {
	db := h13LoadMFUDatabase(t)
	tp := 1
	bsVals, kvVals, grid := h13ExtractDecodeGrid(t, db, tp)

	t.Logf("Grid dimensions: %d batch sizes x %d kv lengths = %d points",
		len(bsVals), len(kvVals), len(grid))

	var maxErr float64
	var errCount int
	var zeroMFUFallbackCount int

	for _, bs := range bsVals {
		for _, kv := range kvVals {
			expected, exists := grid[[2]int{bs, kv}]
			if !exists {
				continue // sparse grid — not all (bs, kv) combinations present
			}

			got := db.GetAttnDecodeMFU(bs, kv, tp)

			// Allow tiny floating point tolerance
			diff := math.Abs(got - expected)
			if diff > 1e-9 {
				// Distinguish zero-MFU protection (expected behavior) from
				// genuine interpolation errors.
				if expected < 0.0001 {
					// Zero-MFU fallback is a known mechanism in GetAttnDecodeMFU.
					// The grid value is ~0 but the function returns a nearby non-zero
					// value to prevent division-by-zero. Log as INFO, not error.
					zeroMFUFallbackCount++
					t.Logf("ON-GRID zero-MFU fallback at (bs=%d, kv=%d): grid=%.6f, returned=%.6f (expected behavior)",
						bs, kv, expected, got)
				} else {
					t.Errorf("ON-GRID mismatch at (bs=%d, kv=%d): grid=%.6f, interpolated=%.6f, diff=%.2e",
						bs, kv, expected, got, diff)
					errCount++
				}
			}
			if diff > maxErr {
				maxErr = diff
			}
		}
	}

	t.Logf("RESULT: on-grid max error = %.2e, mismatches = %d / %d (plus %d zero-MFU fallbacks)",
		maxErr, errCount, len(grid), zeroMFUFallbackCount)
	fmt.Printf("H13_ON_GRID_MAX_ERROR=%.6e\n", maxErr)
	fmt.Printf("H13_ON_GRID_MISMATCHES=%d\n", errCount)
	fmt.Printf("H13_ON_GRID_ZERO_MFU_FALLBACKS=%d\n", zeroMFUFallbackCount)
	fmt.Printf("H13_ON_GRID_TOTAL=%d\n", len(grid))
}

// --- Test (b): Off-grid boundedness ---
// Interpolated values at midpoints between grid points must lie within the
// range [min(neighbors), max(neighbors)].
func TestH13_OffGridBoundedness(t *testing.T) {
	db := h13LoadMFUDatabase(t)
	tp := 1
	bsVals, kvVals, grid := h13ExtractDecodeGrid(t, db, tp)

	var totalTests, violations int
	var maxOvershoot float64

	// Test midpoints between consecutive batch sizes at each kv_len
	for ki := 0; ki < len(kvVals); ki++ {
		kv := kvVals[ki]
		for bi := 0; bi < len(bsVals)-1; bi++ {
			bs0 := bsVals[bi]
			bs1 := bsVals[bi+1]

			mfu0, ok0 := grid[[2]int{bs0, kv}]
			mfu1, ok1 := grid[[2]int{bs1, kv}]
			if !ok0 || !ok1 {
				continue
			}

			midBS := (bs0 + bs1) / 2
			got := db.GetAttnDecodeMFU(midBS, kv, tp)

			lo := math.Min(mfu0, mfu1)
			hi := math.Max(mfu0, mfu1)

			totalTests++
			if got < lo-1e-9 || got > hi+1e-9 {
				overshoot := 0.0
				if got < lo {
					overshoot = lo - got
				} else {
					overshoot = got - hi
				}
				if overshoot > maxOvershoot {
					maxOvershoot = overshoot
				}
				violations++
				t.Logf("BOUND violation at (bs=%d [%d..%d], kv=%d): got=%.6f, range=[%.6f, %.6f]",
					midBS, bs0, bs1, kv, got, lo, hi)
			}
		}
	}

	// Test midpoints between consecutive kv_lens at each batch_size
	for bi := 0; bi < len(bsVals); bi++ {
		bs := bsVals[bi]
		for ki := 0; ki < len(kvVals)-1; ki++ {
			kv0 := kvVals[ki]
			kv1 := kvVals[ki+1]

			mfu0, ok0 := grid[[2]int{bs, kv0}]
			mfu1, ok1 := grid[[2]int{bs, kv1}]
			if !ok0 || !ok1 {
				continue
			}

			midKV := (kv0 + kv1) / 2
			got := db.GetAttnDecodeMFU(bs, midKV, tp)

			lo := math.Min(mfu0, mfu1)
			hi := math.Max(mfu0, mfu1)

			totalTests++
			if got < lo-1e-9 || got > hi+1e-9 {
				overshoot := 0.0
				if got < lo {
					overshoot = lo - got
				} else {
					overshoot = got - hi
				}
				if overshoot > maxOvershoot {
					maxOvershoot = overshoot
				}
				violations++
				t.Logf("BOUND violation at (bs=%d, kv=%d [%d..%d]): got=%.6f, range=[%.6f, %.6f]",
					bs, midKV, kv0, kv1, got, lo, hi)
			}
		}
	}

	t.Logf("RESULT: boundedness violations = %d / %d, max overshoot = %.6f",
		violations, totalTests, maxOvershoot)
	fmt.Printf("H13_BOUND_VIOLATIONS=%d\n", violations)
	fmt.Printf("H13_BOUND_TOTAL=%d\n", totalTests)
	fmt.Printf("H13_BOUND_MAX_OVERSHOOT=%.6e\n", maxOvershoot)
}

// --- Test (c): KV-axis monotonicity ---
// For a fixed batch size, MFU should be non-decreasing as kv_len increases
// (larger KV = more arithmetic intensity = higher utilization). Check both
// grid data and interpolated off-grid points.
func TestH13_KVAxisMonotonicity(t *testing.T) {
	db := h13LoadMFUDatabase(t)
	tp := 1
	bsVals, kvVals, _ := h13ExtractDecodeGrid(t, db, tp)

	var totalTests, violations int

	// Check monotonicity on-grid
	for _, bs := range bsVals {
		var prevMFU float64
		prevKV := 0
		for _, kv := range kvVals {
			mfu := db.GetAttnDecodeMFU(bs, kv, tp)
			if prevKV > 0 && mfu < prevMFU-1e-9 {
				violations++
				t.Logf("KV-MONOTONICITY violation at bs=%d: kv=%d (mfu=%.6f) < kv=%d (mfu=%.6f)",
					bs, kv, mfu, prevKV, prevMFU)
			}
			if prevKV > 0 {
				totalTests++
			}
			prevMFU = mfu
			prevKV = kv
		}
	}

	// Check monotonicity at off-grid kv points for representative batch sizes
	offGridBatchSizes := []int{1, 16, 64, 128, 256}
	for _, bs := range offGridBatchSizes {
		// Sample at regular intervals between min and max kv
		minKV := kvVals[0]
		maxKV := kvVals[len(kvVals)-1]
		step := (maxKV - minKV) / 20
		if step < 1 {
			step = 1
		}

		prevMFU := 0.0
		prevKV := 0
		for kv := minKV; kv <= maxKV; kv += step {
			mfu := db.GetAttnDecodeMFU(bs, kv, tp)
			if prevKV > 0 && mfu < prevMFU-1e-9 {
				violations++
				t.Logf("KV-MONOTONICITY (off-grid) violation at bs=%d: kv=%d (mfu=%.6f) < kv=%d (mfu=%.6f)",
					bs, kv, mfu, prevKV, prevMFU)
			}
			if prevKV > 0 {
				totalTests++
			}
			prevMFU = mfu
			prevKV = kv
		}
	}

	t.Logf("RESULT: kv-axis monotonicity violations = %d / %d", violations, totalTests)
	fmt.Printf("H13_KV_MONOTONICITY_VIOLATIONS=%d\n", violations)
	fmt.Printf("H13_KV_MONOTONICITY_TOTAL=%d\n", totalTests)
}

// --- Test (d): MFU range sanity at eval operating points ---
// The 13 evaluation experiments use operating points roughly in the range
// batchSize=[1..256], kvLen=[512..8192]. Check that MFU values at these
// points are reasonable (not suspiciously low or high).
func TestH13_EvalOperatingPointMFURanges(t *testing.T) {
	db := h13LoadMFUDatabase(t)
	tp := 1

	// Representative operating points from eval experiments
	// (batchSize, maxKVLen) pairs that roughly match the 13 evaluation configs
	evalPoints := []struct {
		name string
		bs   int
		kv   int
	}{
		{"chat-light-bs8", 8, 2048},
		{"chat-light-bs16", 16, 2048},
		{"chat-medium-bs32", 32, 4096},
		{"chat-heavy-bs64", 64, 4096},
		{"code-light-bs8", 8, 4096},
		{"code-medium-bs32", 32, 8192},
		{"code-heavy-bs64", 64, 8192},
		{"mixed-light-bs16", 16, 2048},
		{"mixed-heavy-bs128", 128, 8192},
		{"batch-bs256", 256, 4096},
		{"single-req", 1, 1024},
		{"small-batch", 4, 2048},
		{"large-batch", 512, 8192},
	}

	var suspiciouslyLow int

	fmt.Printf("H13_EVAL_OPERATING_POINTS:\n")
	fmt.Printf("%-25s %6s %6s %10s %10s\n", "Name", "BS", "KVLen", "MFU", "Status")
	for _, ep := range evalPoints {
		mfu := db.GetAttnDecodeMFU(ep.bs, ep.kv, tp)

		status := "OK"
		if mfu < 0.001 {
			status = "VERY_LOW"
			suspiciouslyLow++
		} else if mfu < 0.005 {
			status = "LOW"
		} else if mfu > 0.05 {
			status = "HIGH"
		}

		fmt.Printf("%-25s %6d %6d %10.6f %10s\n", ep.name, ep.bs, ep.kv, mfu, status)

		if mfu <= 0 {
			t.Errorf("zero MFU at eval point %s (bs=%d, kv=%d)", ep.name, ep.bs, ep.kv)
		}
	}

	fmt.Printf("H13_EVAL_SUSPICIOUS_LOW=%d\n", suspiciouslyLow)
	t.Logf("RESULT: %d / %d eval points with suspiciously low MFU (<0.001)",
		suspiciouslyLow, len(evalPoints))
}

// --- Test (e): Interpolation vs nearest-neighbor delta ---
// Measure how much bilinear interpolation changes the MFU value compared to
// what nearest-neighbor lookup would return. Large deltas indicate the grid
// is too coarse for accurate interpolation at intermediate points.
func TestH13_InterpolationVsNearestNeighborDelta(t *testing.T) {
	db := h13LoadMFUDatabase(t)
	tp := 1
	bsVals, kvVals, grid := h13ExtractDecodeGrid(t, db, tp)

	// For each off-grid point, compute both the interpolated and nearest-neighbor MFU
	type deltaResult struct {
		bs, kv           int
		interpolatedMFU  float64
		nearestMFU       float64
		absDelta         float64
		relDelta         float64 // relative to nearest-neighbor
		nearestBS        int
		nearestKV        int
	}

	var results []deltaResult

	// Test midpoints in both dimensions
	for bi := 0; bi < len(bsVals)-1; bi++ {
		for ki := 0; ki < len(kvVals)-1; ki++ {
			bs0, bs1 := bsVals[bi], bsVals[bi+1]
			kv0, kv1 := kvVals[ki], kvVals[ki+1]

			// Check all four corners exist
			_, ok00 := grid[[2]int{bs0, kv0}]
			_, ok10 := grid[[2]int{bs1, kv0}]
			_, ok01 := grid[[2]int{bs0, kv1}]
			_, ok11 := grid[[2]int{bs1, kv1}]
			if !ok00 || !ok10 || !ok01 || !ok11 {
				continue
			}

			// Test the center of the cell
			midBS := (bs0 + bs1) / 2
			midKV := (kv0 + kv1) / 2

			interpMFU := db.GetAttnDecodeMFU(midBS, midKV, tp)

			// Find nearest grid point (by Euclidean distance on (bs, kv))
			minDist := math.MaxFloat64
			nearestBS, nearestKV := bs0, kv0
			nearestMFU := 0.0
			for _, bsC := range []int{bs0, bs1} {
				for _, kvC := range []int{kv0, kv1} {
					mfu, ok := grid[[2]int{bsC, kvC}]
					if !ok {
						continue
					}
					dbs := float64(bsC - midBS)
					dkv := float64(kvC - midKV)
					dist := math.Sqrt(dbs*dbs + dkv*dkv)
					if dist < minDist {
						minDist = dist
						nearestBS = bsC
						nearestKV = kvC
						nearestMFU = mfu
					}
				}
			}

			absDelta := math.Abs(interpMFU - nearestMFU)
			relDelta := 0.0
			if nearestMFU > 1e-6 {
				relDelta = absDelta / nearestMFU
			}

			results = append(results, deltaResult{
				bs: midBS, kv: midKV,
				interpolatedMFU: interpMFU,
				nearestMFU:      nearestMFU,
				absDelta:        absDelta,
				relDelta:        relDelta,
				nearestBS:       nearestBS,
				nearestKV:       nearestKV,
			})
		}
	}

	// Sort by relative delta descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].relDelta > results[j].relDelta
	})

	// Summarize
	var sumAbsDelta, sumRelDelta, maxRelDelta, maxAbsDelta float64
	var count5pct int // count of points where relative delta > 5%
	for _, r := range results {
		sumAbsDelta += r.absDelta
		sumRelDelta += r.relDelta
		if r.relDelta > maxRelDelta {
			maxRelDelta = r.relDelta
		}
		if r.absDelta > maxAbsDelta {
			maxAbsDelta = r.absDelta
		}
		if r.relDelta > 0.05 {
			count5pct++
		}
	}

	meanAbsDelta := sumAbsDelta / float64(len(results))
	meanRelDelta := sumRelDelta / float64(len(results))

	fmt.Printf("H13_INTERP_VS_NN_TOTAL=%d\n", len(results))
	fmt.Printf("H13_INTERP_VS_NN_MEAN_ABS_DELTA=%.6e\n", meanAbsDelta)
	fmt.Printf("H13_INTERP_VS_NN_MAX_ABS_DELTA=%.6e\n", maxAbsDelta)
	fmt.Printf("H13_INTERP_VS_NN_MEAN_REL_DELTA=%.4f\n", meanRelDelta)
	fmt.Printf("H13_INTERP_VS_NN_MAX_REL_DELTA=%.4f\n", maxRelDelta)
	fmt.Printf("H13_INTERP_VS_NN_COUNT_GT_5PCT=%d\n", count5pct)

	t.Logf("RESULT: %d off-grid midpoints tested", len(results))
	t.Logf("  Mean abs delta: %.6e", meanAbsDelta)
	t.Logf("  Max abs delta:  %.6e", maxAbsDelta)
	t.Logf("  Mean rel delta: %.2f%%", meanRelDelta*100)
	t.Logf("  Max rel delta:  %.2f%%", maxRelDelta*100)
	t.Logf("  Points with >5%% rel delta: %d / %d", count5pct, len(results))

	// Print top-10 largest deltas
	fmt.Printf("H13_TOP_DELTAS:\n")
	fmt.Printf("%-8s %-8s %-12s %-12s %-10s %-10s %-8s %-8s\n",
		"BS", "KVLen", "Interp_MFU", "NN_MFU", "AbsDelta", "RelDelta", "NN_BS", "NN_KV")
	for i, r := range results {
		if i >= 10 {
			break
		}
		fmt.Printf("%-8d %-8d %-12.6f %-12.6f %-10.6f %-10.4f %-8d %-8d\n",
			r.bs, r.kv, r.interpolatedMFU, r.nearestMFU, r.absDelta, r.relDelta,
			r.nearestBS, r.nearestKV)
	}
}

// --- Test (f): Full grid dump for analysis ---
// Dumps the complete decode MFU grid and a set of interpolated points
// for external analysis by analyze.py.
func TestH13_GridDump(t *testing.T) {
	db := h13LoadMFUDatabase(t)
	tp := 1
	bsVals, kvVals, grid := h13ExtractDecodeGrid(t, db, tp)

	// Print grid header
	fmt.Printf("H13_GRID_DUMP_START\n")
	fmt.Printf("batch_sizes=%v\n", bsVals)
	fmt.Printf("kv_lengths=%v\n", kvVals)

	// Print full grid (on-grid values)
	fmt.Printf("H13_GRID_DATA:\n")
	fmt.Printf("%-8s %-8s %-12s\n", "BS", "KVLen", "MFU")
	for _, bs := range bsVals {
		for _, kv := range kvVals {
			mfu, ok := grid[[2]int{bs, kv}]
			if ok {
				fmt.Printf("%-8d %-8d %-12.6f\n", bs, kv, mfu)
			}
		}
	}

	// Print interpolated values at quarter-points between grid cells
	fmt.Printf("H13_INTERPOLATED_DATA:\n")
	fmt.Printf("%-8s %-8s %-12s\n", "BS", "KVLen", "MFU")
	for bi := 0; bi < len(bsVals)-1; bi++ {
		for ki := 0; ki < len(kvVals)-1; ki++ {
			bs0, bs1 := bsVals[bi], bsVals[bi+1]
			kv0, kv1 := kvVals[ki], kvVals[ki+1]

			// Quarter, half, three-quarter points
			for _, bsFrac := range []float64{0.25, 0.5, 0.75} {
				for _, kvFrac := range []float64{0.25, 0.5, 0.75} {
					bs := bs0 + int(bsFrac*float64(bs1-bs0))
					kv := kv0 + int(kvFrac*float64(kv1-kv0))
					mfu := db.GetAttnDecodeMFU(bs, kv, tp)
					fmt.Printf("%-8d %-8d %-12.6f\n", bs, kv, mfu)
				}
			}
		}
	}
	fmt.Printf("H13_GRID_DUMP_END\n")
}

// --- Test (g): TP scaling consistency ---
// MFU values should be consistent across TP configurations. The hypothesis
// posits systematic underestimation; check if TP=2 shows similar patterns to TP=1.
func TestH13_TPScalingConsistency(t *testing.T) {
	db := h13LoadMFUDatabase(t)

	// Check which TP configs are available
	tpConfigs := []int{1, 2, 4}
	availableTP := make([]int, 0)
	for _, tp := range tpConfigs {
		configKey := fmt.Sprintf("%s-tp%d", db.attentionConfig, tp)
		if rows := db.decodeData[configKey]; len(rows) > 0 {
			availableTP = append(availableTP, tp)
		}
	}

	if len(availableTP) < 2 {
		t.Skipf("need at least 2 TP configs for comparison, only have %v", availableTP)
	}

	t.Logf("Available TP configs: %v", availableTP)

	// Compare MFU at common operating points across TP configs
	testPoints := [][2]int{
		{16, 4096}, {32, 4096}, {64, 8192}, {128, 8192},
		{32, 16384}, {64, 16384}, {256, 4096},
	}

	fmt.Printf("H13_TP_COMPARISON:\n")
	fmt.Printf("%-8s %-8s", "BS", "KVLen")
	for _, tp := range availableTP {
		fmt.Printf(" TP%-6d", tp)
	}
	fmt.Printf("\n")

	for _, pt := range testPoints {
		bs, kv := pt[0], pt[1]
		fmt.Printf("%-8d %-8d", bs, kv)
		for _, tp := range availableTP {
			mfu := db.GetAttnDecodeMFU(bs, kv, tp)
			fmt.Printf(" %-8.6f", mfu)
		}
		fmt.Printf("\n")
	}
}
