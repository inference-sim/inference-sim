// hw_fp8_ratio_test.go — the dense FP8:BF16 peak-FLOPs ratio guard (#1829).
//
// `TFlopsPeak` and `TFlopsFP8` must be read out of the SAME column of a GPU datasheet.
// NVIDIA quotes each tensor-core rate twice — a dense figure and a 2x-larger
// "with sparsity" figure — and the bundled L40S entry mixed the two: `TFlopsPeak`
// 362.05 was dense BF16 while `TFlopsFP8` 1466.0 was FP8 *with sparsity*, so BLIS
// shipped an L40S whose FP8 compute ceiling was double the real one. Nothing caught it,
// because each number is individually plausible and the roofline model consumes them
// independently (sim/latency/roofline.go picks TFlopsFP8 whenever the weights are
// 1 byte/param). blis-catalog fixed its copy in blis-catalog#2 / #1738; this repository's
// table was never propagated back.
//
// What makes the mix detectable is the RATIO. FP8 tensor-core throughput is exactly
// double BF16 on every part BLIS models — Hopper (H100/H200 1979/989.5 = 2.0x), Ada
// (L40S 733/362.05 = 2.02x, the small excess being datasheet clock rounding), and
// Blackwell — and sparsity is itself exactly a 2x multiplier. So a convention mix cannot
// land in band: taking FP8 from the sparse column reads ~4x, and taking BF16 from the
// sparse column (the mirror-image slip) reads ~1x. Both directions are rejected here,
// naming the GPU, which is the part an operator needs in a file of a dozen entries.
package latency_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// The plausible band for TFlopsFP8 / TFlopsPeak. The physical value is 2.0 on every
// architecture BLIS models; the band is widened to ±~25% so datasheet rounding and a
// future part that quotes a slightly different boost clock for the two rates still pass,
// while leaving a 1.6x margin to the nearest thing the guard must catch (a 2x sparsity
// multiplier applied to one of the two fields).
const (
	minDenseFP8Ratio = 1.8
	maxDenseFP8Ratio = 2.5
)

// denseFP8RatioProblem returns a diagnostic naming gpu when hc's peak-FLOPs pair is not
// a self-consistent DENSE pair, or "" when the entry is in band.
//
// An entry with TFlopsFP8 == 0 declares no native FP8 path (an A100 has none) and is
// exempt: the roofline model never divides by it, it falls back to TFlopsPeak.
func denseFP8RatioProblem(gpu string, hc sim.HardwareCalib) string {
	if hc.TFlopsFP8 == 0 {
		return ""
	}
	if !(hc.TFlopsPeak > 0) || math.IsInf(hc.TFlopsPeak, 0) {
		return fmt.Sprintf("GPU %q declares TFlopsFP8 = %v but TFlopsPeak = %v: the FP8 rate "+
			"cannot be checked against a missing or non-finite BF16 rate", gpu, hc.TFlopsFP8, hc.TFlopsPeak)
	}
	ratio := hc.TFlopsFP8 / hc.TFlopsPeak
	if !(ratio >= minDenseFP8Ratio) || ratio > maxDenseFP8Ratio {
		return fmt.Sprintf("GPU %q has a dense FP8:BF16 peak-FLOPs ratio of %.3gx "+
			"(TFlopsFP8 = %v, TFlopsPeak = %v), outside the plausible band [%.2g, %.2g]. "+
			"FP8 tensor-core throughput is 2x BF16 on every architecture BLIS models, so a "+
			"ratio near 4x means TFlopsFP8 was taken from the datasheet's WITH-SPARSITY "+
			"column while TFlopsPeak is dense (and a ratio near 1x means the mirror-image "+
			"slip). Read both numbers out of the same column — see #1829",
			gpu, ratio, hc.TFlopsFP8, hc.TFlopsPeak, minDenseFP8Ratio, maxDenseFP8Ratio)
	}
	return ""
}

// committedHWConfigPath is the bundled hardware table, as seen from this package.
func committedHWConfigPath() string { return filepath.Join("..", "..", "hardware_config.json") }

// committedHWGPUNames enumerates the GPU keys of the bundled table, sorted. Enumerating
// the file rather than a hardcoded list is what makes the ratio guard cover entries added
// after this test was written.
func committedHWGPUNames(t *testing.T) []string {
	t.Helper()
	raw, err := os.ReadFile(committedHWConfigPath())
	require.NoError(t, err)
	var entries map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &entries))
	require.NotEmpty(t, entries, "the bundled hardware table must not be empty")
	names := make([]string, 0, len(entries))
	for gpu := range entries {
		names = append(names, gpu)
	}
	sort.Strings(names)
	return names
}

// TestHWFP8Ratio_CommittedTableIsDenseThroughout covers BC-2: every entry of the bundled
// hardware table that declares an FP8 rate declares a DENSE one, consistent with its
// dense BF16 rate. This is the guard the issue asks for — it reads each entry through the
// production parse path (latency.GetHWConfig), so it fences exactly what a run sees, and
// it enumerates the file, so adding a GPU with a sparsity-mixed pair fails here rather
// than silently halving that GPU's predicted step time.
func TestHWFP8Ratio_CommittedTableIsDenseThroughout(t *testing.T) {
	path := committedHWConfigPath()
	checked := 0
	for _, gpu := range committedHWGPUNames(t) {
		hc, err := latency.GetHWConfig(path, gpu)
		require.NoError(t, err, "bundled entry %q must load", gpu)
		assert.Empty(t, denseFP8RatioProblem(gpu, hc),
			"bundled entry %q mixes datasheet conventions", gpu)
		if hc.TFlopsFP8 != 0 {
			checked++
		}
	}
	// Non-vacuity: if every entry declared TFlopsFP8 = 0 the loop above would pass
	// without comparing anything.
	assert.GreaterOrEqual(t, checked, 1, "no bundled entry declares an FP8 rate, so the ratio guard checked nothing")
}

// TestHWFP8Ratio_L40SIsTheDenseFigure covers BC-1, the value fix itself: the bundled L40S
// entry carries the dense FP8 rate of the NVIDIA L40S datasheet (733 TFLOPS), which is
// what blis-catalog/hardware/l40s.yaml carries. Before #1829 it carried 1466.0, the
// with-sparsity figure from the same datasheet row.
//
// The expected value is not "whatever the code produced" (R7/R12): 733 is the datasheet's
// dense FP8 number, and it is independently corroborated by the ratio law asserted below —
// an Ada part's FP8 rate is 2x its BF16 rate, and 2 x 362.05 = 724.1, which 733 matches to
// within datasheet clock rounding, while 1466 does not.
func TestHWFP8Ratio_L40SIsTheDenseFigure(t *testing.T) {
	hc, err := latency.GetHWConfig(committedHWConfigPath(), "L40S")
	require.NoError(t, err)

	assert.Equal(t, 733.0, hc.TFlopsFP8,
		"bundled L40S TFlopsFP8 must be the DENSE figure 733.0 (blis-catalog/hardware/l40s.yaml), not the with-sparsity 1466.0")
	assert.Equal(t, 362.05, hc.TFlopsPeak, "the BF16 rate it is checked against must stay the dense figure")
	assert.InDelta(t, 2.0, hc.TFlopsFP8/hc.TFlopsPeak, 0.1,
		"an Ada part's dense FP8 rate is 2x its dense BF16 rate")
}

// TestHWFP8Ratio_PeakFLOPsPairsAreFenced is the byte-identity carve-out fence. #1829
// deliberately changes output — it roughly halves L40S's FP8 compute ceiling — and the
// acceptance criterion is "one GPU, one field, one direction": every OTHER GPU stays
// byte-identical. A ratio band alone cannot express that, because 1466/362.05 and 733/362.05
// are not both in band but 1979/989.5 and (say) 2400/1200 both are. So the two peak-FLOPs
// fields of every bundled entry are pinned to their datasheet values here.
//
// Adding a GPU therefore means adding a row to wantPeakFLOPs. That is deliberate: a change
// to a peak-FLOPs number is never incidental, and this test is where it gets stated.
func TestHWFP8Ratio_PeakFLOPsPairsAreFenced(t *testing.T) {
	// GPU -> {dense BF16, dense FP8}. 0 FP8 = no native FP8 path.
	wantPeakFLOPs := map[string]struct{ bf16, fp8 float64 }{
		"H100":     {989.5, 1979.0}, // Hopper GH100, NVIDIA H100 datasheet (dense)
		"H200":     {989.5, 1979.0}, // same GH100 compute die as H100 => identical peak FLOPs
		"A100-SXM": {312, 0},        // Ampere GA100: no FP8 tensor cores
		"A100-80":  {312, 0},        // alias of A100-SXM
		"L40S":     {362.05, 733.0}, // Ada AD102, NVIDIA L40S datasheet (dense); 1466.0 is the with-sparsity figure (#1829)
	}

	gpus := committedHWGPUNames(t)
	wantNames := make([]string, 0, len(wantPeakFLOPs))
	for gpu := range wantPeakFLOPs {
		wantNames = append(wantNames, gpu)
	}
	sort.Strings(wantNames)
	require.Equal(t, wantNames, gpus,
		"the bundled hardware table's entry set changed; add or remove the corresponding row in wantPeakFLOPs "+
			"(and state the datasheet column the numbers came from)")

	for _, gpu := range gpus {
		hc, err := latency.GetHWConfig(committedHWConfigPath(), gpu)
		require.NoError(t, err)
		want := wantPeakFLOPs[gpu]
		assert.Equal(t, want.bf16, hc.TFlopsPeak, "GPU %q: TFlopsPeak changed", gpu)
		assert.Equal(t, want.fp8, hc.TFlopsFP8, "GPU %q: TFlopsFP8 changed", gpu)
	}
}

// TestHWFP8Ratio_OutOfBandPairIsRejectedNamingTheGPU covers BC-3: the guard actually
// rejects a sparsity mix, and its diagnostic names the offending GPU. Without this the
// band could be vacuous (or inverted) and the committed-table test above would still pass.
//
// Each row is a hardware entry as GetHWConfig hands it back, so the rows double as a
// statement of which pairs are legitimate.
func TestHWFP8Ratio_OutOfBandPairIsRejectedNamingTheGPU(t *testing.T) {
	tests := []struct {
		name       string
		gpu        string
		bf16, fp8  float64
		wantReject bool
	}{
		{
			name: "the L40S regression: FP8 from the with-sparsity column, BF16 dense",
			gpu:  "L40S", bf16: 362.05, fp8: 1466.0, wantReject: true,
		},
		{
			name: "the mirror-image slip: BF16 from the with-sparsity column, FP8 dense",
			gpu:  "L40S-mirrored", bf16: 724.1, fp8: 733.0, wantReject: true,
		},
		{
			name: "an H100-shaped sparsity mix is rejected too, not just the L40S numbers",
			gpu:  "H100", bf16: 989.5, fp8: 3958.0, wantReject: true,
		},
		{
			name: "the corrected L40S pair is accepted",
			gpu:  "L40S", bf16: 362.05, fp8: 733.0,
		},
		{
			name: "an exactly-2.0x pair is accepted",
			gpu:  "H100", bf16: 989.5, fp8: 1979.0,
		},
		{
			name: "no native FP8 path is exempt, not treated as a 0x ratio",
			gpu:  "A100-SXM", bf16: 312, fp8: 0,
		},
		{
			name: "an FP8 rate with no BF16 rate to check it against is rejected, not divided by zero",
			gpu:  "Nonsense", bf16: 0, fp8: 733.0, wantReject: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Round-trip through the production parse path, so a row cannot pass by
			// describing a config GetHWConfig would never produce.
			fields := baseHWFields()
			fields["TFlopsPeak"] = fmt.Sprintf("%v", tt.bf16)
			fields["TFlopsFP8"] = fmt.Sprintf("%v", tt.fp8)
			hc, err := latency.GetHWConfig(writeHWConfig(t, tt.gpu, fields), tt.gpu)
			if err != nil {
				// A row whose numbers ValidateRooflineConfig itself refuses (a 0 BF16
				// rate) never reaches the ratio guard — the refusal is the rejection,
				// and it must still name the field.
				require.True(t, tt.wantReject, "a legitimate pair must load: %v", err)
				assert.Contains(t, err.Error(), "TFlops", "the refusal must name the peak-FLOPs field")
				return
			}

			problem := denseFP8RatioProblem(tt.gpu, hc)
			if !tt.wantReject {
				assert.Empty(t, problem, "a legitimate dense pair must not be flagged")
				return
			}
			require.NotEmpty(t, problem, "an out-of-band peak-FLOPs pair must be rejected")
			assert.Contains(t, problem, tt.gpu, "the diagnostic must name the offending GPU")
		})
	}
}
