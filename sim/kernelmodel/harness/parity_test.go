package harness

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel/internal/artifacts"
)

// This file is the apples-to-apples guarantee, expressed as checks that fail when the
// comparison stops being one. Three things must hold, and each has been violated once:
//
//  1. The ERROR DEFINITION must be AISimulate's. Verified by reproduction: their definition
//     applied to their own published per-point data must return the figure they publish.
//  2. The WORKLOAD must be AISimulate's. Verified against the constants in its source.
//  3. Both sides must be normalised to their OWN anchor, or a shape is compared to a
//     shape-plus-level.
//
// A test here that merely restated the intended values would be structural and would pass
// while the harness disagreed. Each of these instead recomputes a published quantity, or
// compares two things the harness actually produces.

// AISimulate publishes tpot_shape_error_pct over its whole snapshot. Recomputing it from its
// own per-point relatives, under its own definition, must return that number. This is the
// arbiter for every choice in the error computation: which points are included, what the
// anchor is, and whether the mean is plain or weighted.
//
// It caught a real defect. The harness averaged over every point INCLUDING the anchor, whose
// error is zero by construction, which diluted the figure from 11.55% to 9.41% on the scored
// subset and could not reproduce the published number at all.
func TestAISimulatesPublishedShapeErrorIsReproducible(t *testing.T) {
	// The FULL snapshot, not the scored subset. 10.05% is published over all 1137 points, and
	// comparing a subset figure against it proves nothing -- a first version of this test did
	// exactly that and "passed" the wrong definition, because on the 447-point subset the
	// anchor-INCLUSIVE value (9.41%) happens to sit nearer 10.05% than the correct
	// anchor-exclusive one (11.55%). A coincidence of subsetting, not evidence.
	snap, published, err := loadSnapshot(snapshotPath(t))
	if err != nil {
		t.Fatalf("full snapshot: %v", err)
	}
	excl := snapshotShapeError(snap, true)
	incl := snapshotShapeError(snap, false)
	if excl.n == 0 {
		t.Fatal("no comparisons in the snapshot")
	}
	// Exact reproduction, to the precision the figure is published at.
	if math.Abs(excl.mean-published) > 0.005 {
		t.Errorf("recomputing AISimulate's shape error under its own anchor-exclusive "+
			"definition gives %.4f%% on %d comparisons, against a published %.2f%%. The "+
			"harness is not computing the quantity AISimulate publishes",
			excl.mean, excl.n, published)
	}
	// And the anchor-inclusive variant must NOT reproduce it, or the two definitions are
	// indistinguishable and this test cannot discriminate.
	if math.Abs(incl.mean-published) <= 0.005 {
		t.Errorf("the anchor-inclusive variant also reproduces %.2f%%; the test cannot "+
			"tell the definitions apart", published)
	}
}

type errStat struct {
	mean float64
	n    int
}

// snapshotPath is the full published snapshot, beside the extracted corpus.
func snapshotPath(t testing.TB) string { return artifacts.Measurement(t, "aisimulate_summary.json") }

// loadSnapshot reads the nested published summary and its whole-snapshot shape error.
func loadSnapshot(path string) (*snapshot, float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	var s snapshot
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, 0, err
	}
	return &s, s.Totals.AISimulate.TPOTShapeErrorPct, nil
}

// snapshot mirrors only the fields this check needs from the published summary.
type snapshot struct {
	Models []struct {
		Workloads []struct {
			GPUs []struct {
				Topologies []struct {
					Points []struct {
						Concurrency int    `json:"concurrency"`
						Status      string `json:"status"`
						Measured    struct {
							TPOTRelative float64 `json:"tpot_relative"`
						} `json:"measured"`
						AISimulate struct {
							TPOTRelative float64 `json:"tpot_relative"`
						} `json:"aisimulate"`
					} `json:"points"`
				} `json:"topologies"`
			} `json:"gpus"`
		} `json:"workloads"`
	} `json:"models"`
	Totals struct {
		AISimulate struct {
			TPOTShapeErrorPct float64 `json:"tpot_shape_error_pct"`
		} `json:"aisimulate"`
	} `json:"totals"`
}

// snapshotShapeError recomputes the published shape error from the snapshot's own per-point
// relatives, each side re-anchored to its own lowest concurrency. excludeAnchor selects
// between the two candidate definitions.
func snapshotShapeError(s *snapshot, excludeAnchor bool) errStat {
	var sum float64
	var n int
	for _, m := range s.Models {
		for _, w := range m.Workloads {
			for _, g := range w.GPUs {
				for _, topo := range g.Topologies {
					pts := topo.Points[:0:0]
					for _, p := range topo.Points {
						if p.Status == "success" {
							pts = append(pts, p)
						}
					}
					sort.Slice(pts, func(i, j int) bool {
						return pts[i].Concurrency < pts[j].Concurrency
					})
					if len(pts) < 2 {
						continue
					}
					am, ap := pts[0].Measured.TPOTRelative, pts[0].AISimulate.TPOTRelative
					if am == 0 || ap == 0 {
						continue
					}
					start := 0
					if excludeAnchor {
						start = 1
					}
					for _, p := range pts[start:] {
						ms := p.Measured.TPOTRelative / am
						ps := p.AISimulate.TPOTRelative / ap
						if ms == 0 {
							continue
						}
						sum += math.Abs((ps-ms)/ms) * 100
						n++
					}
				}
			}
		}
	}
	if n == 0 {
		return errStat{}
	}
	return errStat{sum / float64(n), n}
}

// The workload this harness drives must be the one AISimulate's replay drives. The constants
// are read from its source (scripts/run_e2e_accuracy.py and
// python/aisimulate/src/aisimulate/runner.py) and asserted here so a drift in either is a
// test failure rather than a silent divergence.
//
// AISimulate samples ISL and OSL independently and uniformly on [int(len*0.8), len]
// INCLUSIVE, and runs concurrency*10 requests. The label is an upper bound, not a value --
// which is the defect this check exists for: the harness previously used a constant at the
// label, a mean context about 10% above what the baseline simulated.
func TestTheWorkloadMatchesAISimulatesReplaySpec(t *testing.T) {
	for _, tc := range []struct{ isl, osl, concurrency int }{
		{1024, 1024, 4}, {8192, 1024, 32}, {1024, 8192, 8},
	} {
		w := AISimulateWorkload(tc.isl, tc.osl, tc.concurrency)

		// Literal 0.8, for the same reason: the ratio is AISimulate's published constant,
		// restated here rather than read from the code under test.
		if got, want := w.ISLLow, int(float64(tc.isl)*0.8); got != want {
			t.Errorf("isl=%d: low bound %d, AISimulate uses int(upper*0.8)=%d",
				tc.isl, got, want)
		}
		if w.ISLHigh != tc.isl {
			t.Errorf("isl=%d: high bound %d must be the label itself", tc.isl, w.ISLHigh)
		}
		if got, want := w.OSLLow, int(float64(tc.osl)*0.8); got != want {
			t.Errorf("osl=%d: low bound %d, expected %d", tc.osl, got, want)
		}
		if w.OSLHigh != tc.osl {
			t.Errorf("osl=%d: high bound %d must be the label itself", tc.osl, w.OSLHigh)
		}
		// Literals, not the package constants. Asserting against the same constant the code
		// uses is circular: changing the constant would change both sides and the check would
		// still pass, which is exactly what happened to a first version of this test.
		//
		// Both numbers are restated from the REAL harness rather than from AISimulate's
		// replay, because that is the protocol the measurements were taken under:
		// InferenceX's srt_fixed_sequence.sh passes --num-prompts $((CONC * 10)) and
		// --num-warmups $((2 * CONC)). AISimulate's replay independently uses the same
		// request count, so matching the harness matches it too.
		if got, want := w.RequestCount, tc.concurrency*10; got != want {
			t.Errorf("concurrency=%d: measured request count %d, the harness uses "+
				"concurrency*10=%d", tc.concurrency, got, want)
		}
		if got, want := w.WarmupCount, tc.concurrency*2; got != want {
			t.Errorf("concurrency=%d: warm-up count %d, the harness uses concurrency*2=%d",
				tc.concurrency, got, want)
		}
		// The mean must sit BELOW the label, which is the whole point.
		mean := float64(w.ISLLow+w.ISLHigh) / 2
		if mean >= float64(tc.isl) {
			t.Errorf("isl=%d: mean sampled length %.0f is not below the label; the "+
				"interval is not one-sided", tc.isl, mean)
		}
	}
}

// The PDF handed to BLIS must be a proper discrete uniform over the stated interval: every
// length in it equally likely, nothing outside it, and the weights summing to one. A PDF that
// was subtly wrong -- an off-by-one on the inclusive upper bound, say -- would shift the mean
// context and every latency with it.
func TestTheLengthPDFIsUniformOverAISimulatesInterval(t *testing.T) {
	w := AISimulateWorkload(1024, 1024, 4)
	pdf := uniformPDF(w.ISLLow, w.ISLHigh)

	if got, want := len(pdf), w.ISLHigh-w.ISLLow+1; got != want {
		t.Errorf("the PDF has %d bins for the inclusive interval [%d, %d], expected %d",
			got, w.ISLLow, w.ISLHigh, want)
	}
	var total float64
	var first float64
	for length, weight := range pdf {
		if length < w.ISLLow || length > w.ISLHigh {
			t.Errorf("the PDF carries length %d, outside [%d, %d]",
				length, w.ISLLow, w.ISLHigh)
		}
		if first == 0 {
			first = weight
		} else if math.Abs(weight-first) > 1e-12 {
			t.Errorf("weights differ (%g vs %g); the distribution is not uniform",
				weight, first)
		}
		total += weight
	}
	if math.Abs(total-1) > 1e-9 {
		t.Errorf("the PDF sums to %.12f, not 1", total)
	}
}
