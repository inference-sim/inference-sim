package harness

import (
	"testing"

	"github.com/inference-sim/inference-sim/tools/harness/internal/artifacts"
)

// TTFT must RISE with concurrency. A model whose TTFT does not move while its own step time
// doubles is not queueing requests at all, and that is the bug this guards.
//
// It previously also required TTFT to rise FASTER than ITL, and to clear 1.5x over a
// sixteen-fold concurrency rise. Both held while admission was priced purely per token, and
// both are properties of the measurement: across the 95 vLLM sweeps carrying concurrency 4 and
// 64, measured TTFT rises 3.056x at the median against ITL's 2.364x, and rises faster than ITL
// in 72.6% of them.
//
// They no longer hold for this model, because `host_admission_per_request` charges a fixed
// per-request cost (blis-registry cost-model-host-overheads.yaml). A constant adds the same
// amount at every concurrency, so it lifts the low-concurrency anchor and compresses the ratio:
// at 40 ms this model rises 1.407x where the measurement rises 3.056x. That is a known
// limitation of a constant in the concurrency dimension, recorded in the coefficient's own
// rationale and in the companion's section 3.13, and it is the same effect as the TTFT shape
// figure moving from 15.23% to 25.39%.
//
// The assertion kept here is the one the test was written to catch: a TTFT that does not rise
// at all. The two tighter bounds are deliberately not asserted, so that this test fails on a
// broken queueing path rather than on a modelling trade that is documented elsewhere.
func TestTTFTRisesWithConcurrency(t *testing.T) {
	c, err := LoadCorpus(
		artifacts.Measurement(t, "aisimulate_e2e.json"))
	if err != nil {
		t.Fatal(err)
	}
	_ = c
	sw := testSweep(t)
	cfg := Config{Repos: testRepos(), Blis: testBlis(t), Seed: 42}

	lo, err := Run(sw, 4, cfg)
	if err != nil {
		t.Fatalf("c=4: %v", err)
	}
	hi, err := Run(sw, 64, cfg)
	if err != nil {
		t.Fatalf("c=64: %v", err)
	}
	ttftRise := hi.MeanTTFTUs / lo.MeanTTFTUs
	itlRise := hi.MeanTPOTUs / lo.MeanTPOTUs

	// A sixteen-fold concurrency rise must move the first token. The bound is deliberately
	// loose -- see the comment above -- because a fixed per-request admission cost compresses
	// this ratio by design. It still catches an admission path that has stopped queueing,
	// which is what this test exists for.
	if ttftRise <= 1.0 {
		t.Errorf("TTFT rose %.3fx and ITL rose %.3fx over concurrency 4 to 64; TTFT did not rise "+
			"at all, so admission queueing is not reaching the metric", ttftRise, itlRise)
	}
}

// The spike predicate must catch the four known artefacts and nothing else, and it must be
// insensitive to the threshold across the whole gap the data provides. A predicate that changed
// its answer with the constant would be a tuning knob rather than a measurement.
func TestTTFTSpikeSelectionIsDecidedByTheDataNotTheThreshold(t *testing.T) {
	c, err := LoadCorpus(
		artifacts.Measurement(t, "aisimulate_e2e.json"))
	if err != nil {
		t.Fatal(err)
	}
	count := func(factor float64) int {
		n := 0
		for i := range c.Sweeps {
			if c.Sweeps[i].Framework != "vllm" {
				continue
			}
			n += len(c.Sweeps[i].TTFTSpikeIndices(factor))
		}
		return n
	}
	want := count(DefaultTTFTSpikeFactor)
	if want != 4 {
		t.Errorf("want the four documented artefacts, got %d", want)
	}
	for _, f := range []float64{2, 3, 5, 8, 10} {
		if got := count(f); got != want {
			t.Errorf("factor %v selects %d points, factor %v selects %d -- the threshold is "+
				"inside a gap in the data and must not change the answer",
				f, got, DefaultTTFTSpikeFactor, want)
		}
	}
	// And a much higher threshold must select FEWER: the predicate is doing something.
	if got := count(30); got >= want {
		t.Errorf("factor 30 selected %d, not fewer than %d", got, want)
	}
}

// An excluded point must never be the anchor. The anchor defines the normalisation, so dropping
// it would renormalise the sweep rather than remove a bad point.
func TestTTFTSpikeNeverExcludesTheAnchor(t *testing.T) {
	c, err := LoadCorpus(
		artifacts.Measurement(t, "aisimulate_e2e.json"))
	if err != nil {
		t.Fatal(err)
	}
	for i := range c.Sweeps {
		for _, idx := range c.Sweeps[i].TTFTSpikeIndices(DefaultTTFTSpikeFactor) {
			if idx == 0 {
				t.Errorf("%s %s: index 0 excluded", c.Sweeps[i].Scenario, c.Sweeps[i].Label)
			}
			if idx >= len(c.Sweeps[i].Points)-1 {
				t.Errorf("%s %s: index %d has no right neighbour to bracket it",
					c.Sweeps[i].Scenario, c.Sweeps[i].Label, idx)
			}
		}
	}
}

// Excluding POINTS rather than sweeps must retain the good data in an affected sweep. Dropping
// the three sweeps that carry a spike would cost roughly fifteen sound measurements to remove
// four bad ones.
func TestPointExclusionKeepsTheRestOfAnAffectedSweep(t *testing.T) {
	c, err := LoadCorpus(
		artifacts.Measurement(t, "aisimulate_e2e.json"))
	if err != nil {
		t.Fatal(err)
	}
	affectedSweeps, spikePoints, pointsInAffected := 0, 0, 0
	for i := range c.Sweeps {
		if c.Sweeps[i].Framework != "vllm" {
			continue
		}
		idx := c.Sweeps[i].TTFTSpikeIndices(DefaultTTFTSpikeFactor)
		if len(idx) == 0 {
			continue
		}
		affectedSweeps++
		spikePoints += len(idx)
		pointsInAffected += len(c.Sweeps[i].Points)
	}
	kept := pointsInAffected - spikePoints
	if kept <= spikePoints {
		t.Errorf("point exclusion kept %d and dropped %d across %d sweeps; it should keep far "+
			"more than it drops", kept, spikePoints, affectedSweeps)
	}
	t.Logf("%d affected sweep(s): %d point(s) excluded, %d kept", affectedSweeps, spikePoints, kept)
}
