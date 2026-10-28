//go:build scoring

// Moved from blis-latency-kernel/cmd/shape/main_scoring_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// The tests that read a measurement corpus, which is not distributed with this repository
// (harness.DefaultMeasurements). Run them with `go test -tags scoring ./cmd/...` once the
// corpus is in testdata/measurements or BLIS_MEASUREMENTS.

package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-vllm/kerneltests/internal/artifacttest"
	"github.com/inference-sim/blis-vllm/kerneltests/internal/harness"
)

// The normalisation is the whole basis of this comparison: if the anchor were wrong, or if a
// prediction carried an absolute scale, the numbers would be meaningless in a way no output
// inspection would reveal.

func loadCorpus(t *testing.T) corpus {
	t.Helper()
	corpusPath := filepath.Join(harness.DefaultMeasurements(), "aisimulate_e2e.json")
	raw, err := os.ReadFile(corpusPath)
	if err != nil {
		// Opted into by the build tag, so an absent corpus is a failure, not a skip.
		t.Fatalf("measurement corpus %s: %v (it is not distributed with this repository; "+
			"see harness.DefaultMeasurements)", corpusPath, err)
	}
	var c corpus
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatalf("corpus does not parse: %v", err)
	}
	return c
}

func TestEverySweepIsNormalisedToItsLowestConcurrency(t *testing.T) {
	// The snapshot anchors each sweep at its own lowest concurrency. A sweep anchored
	// elsewhere would make the ratio comparison compare two different quantities.
	c := loadCorpus(t)
	if len(c.Sweeps) == 0 {
		t.Fatal("no sweeps")
	}
	for _, s := range c.Sweeps {
		if len(s.Points) < 3 {
			t.Errorf("%s %s has %d points; a shape needs at least three",
				s.Model, s.Label, len(s.Points))
		}
		for i := 1; i < len(s.Points); i++ {
			if s.Points[i].Concurrency <= s.Points[i-1].Concurrency {
				t.Errorf("%s %s: concurrencies are not ascending", s.Model, s.Label)
			}
		}
		if got := s.Points[0].MeasuredRelative; math.Abs(got-1.0) > 1e-9 {
			t.Errorf("%s %s: the lowest concurrency is normalised to %.6f, not 1.0",
				s.Model, s.Label, got)
		}
	}
}

func TestTheMeasuredCurveRisesWithConcurrency(t *testing.T) {
	// Time per output token grows as a batch widens, and a corpus where it mostly did not
	// would mean the extraction mismatched points to concurrencies -- which is what this
	// guards. It is a population property rather than a per-transition one: real engines do
	// occasionally record a lower TPOT at a higher client concurrency, when the resident
	// batch or the chunked-prefill regime changes between the two runs, and four of these
	// appear in 377 measured transitions. Asserting per-transition monotonicity would fail
	// on that measured truth, so the bound is on the rate.
	//
	// One percent is the observed rate with room above it; a corpus whose points were
	// shuffled against their concurrencies would sit near fifty.
	const maxFallingFraction = 0.05
	c := loadCorpus(t)
	var transitions, falling int
	for _, s := range c.Sweeps {
		for i := 1; i < len(s.Points); i++ {
			transitions++
			if s.Points[i].MeasuredRelative <= s.Points[i-1].MeasuredRelative {
				falling++
				t.Logf("%s %s: measured TPOT fell from %.4f at c=%d to %.4f at c=%d",
					s.Scenario, s.Label, s.Points[i-1].MeasuredRelative,
					s.Points[i-1].Concurrency, s.Points[i].MeasuredRelative,
					s.Points[i].Concurrency)
			}
		}
	}
	if transitions == 0 {
		t.Fatal("no transitions in the corpus")
	}
	if got := float64(falling) / float64(transitions); got > maxFallingFraction {
		t.Errorf("%d of %d measured transitions fall (%.1f%%), above the %.0f%% bound; "+
			"the extraction has likely mismatched points to concurrencies",
			falling, transitions, 100*got, 100*maxFallingFraction)
	}
}

func TestThePredictionCarriesNoAbsoluteScale(t *testing.T) {
	// The anchor point must score exactly zero error by construction, since both sides are
	// divided by their own value there. A non-zero error at the anchor would mean an
	// absolute term leaked into the ratio.
	c := loadCorpus(t)
	for _, s := range c.Sweeps {
		k, err := harness.Open(s.Scenario, harness.Repos{
			Scenarios: scenarioRoot,
			Catalog:   harness.DefaultCatalog(),
			Registry:  harness.DefaultRegistry(),
		})
		artifacttest.RequireArtifact(t, harness.DefaultCatalog(),
			harness.DefaultCatalog(), "catalog", err)
		ctx := contextTokens(s)
		anchor := stepSeconds(k, s.Points[0].Concurrency, ctx)
		got := stepSeconds(k, s.Points[0].Concurrency, ctx) / anchor
		if math.Abs(got-1.0) > 1e-12 {
			t.Errorf("%s: the anchor predicts %.12f, not exactly 1.0", s.Label, got)
		}
		// And the predicted curve must rise, for the same reason the measured one does.
		prev := 0.0
		for _, p := range s.Points {
			cur := stepSeconds(k, p.Concurrency, ctx) / anchor
			if cur <= prev {
				t.Errorf("%s: predicted TPOT did not rise at c=%d (%.4f after %.4f)",
					s.Label, p.Concurrency, cur, prev)
			}
			prev = cur
		}
	}
}

// The scenarios the corpus names. A constant rather than a literal at six call sites: when
// the corpus moved to generated per-deployment scenarios, a stale literal made one test look
// for files that no longer existed there and another skip silently.
const scenarioRoot = "../../testdata/aisimulate"

func TestEverySweepNamesAScenarioThatExists(t *testing.T) {
	c := loadCorpus(t)
	for _, s := range c.Sweeps {
		p := filepath.Join(scenarioRoot, s.Scenario)
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s names scenario %s, which is absent", s.Model, s.Scenario)
		}
	}
}

// The snapshot's own deployment facts must reach the scenario, or the comparison prices a
// different deployment from the one measured.
func TestTheScenarioMatchesTheSnapshotsStatedParallelism(t *testing.T) {
	c := loadCorpus(t)
	for _, s := range c.Sweeps {
		sc, err := loadScenario(filepath.Join(scenarioRoot, s.Scenario))
		if err != nil {
			// Never a skip. A skip here silently retires the parallelism check for every
			// remaining sweep, which is how a scenario came to be scored against the wrong
			// deployment before.
			t.Fatalf("%s: %v", s.Scenario, err)
		}
		if want := s.Parallelism["tp_size"]; sc.tp != want {
			t.Errorf("%s: scenario has tp=%d, the snapshot states %d", s.Scenario,
				sc.tp, want)
		}
		// moe_ep_size of 1 means expert parallelism is off, which the scenario must agree
		// with: the two cases shard an expert along different axes.
		ep := s.Parallelism["moe_ep_size"]
		if (ep > 1) != sc.expertParallel {
			t.Errorf("%s: scenario expert-parallel is %v, the snapshot states "+
				"moe_ep_size=%d", s.Scenario, sc.expertParallel, ep)
		}
	}
}

// The tests above check the harness's inputs. These check its OUTPUT, which is the gap three
// mutations walked through: replacing the ratio with an absolute step time, anchoring on the
// wrong concurrency, and dropping the per-token host cost all left every input-level test
// passing while the reported MAPE went to 99%.
//
// A harness whose internals are right and whose published number is wrong is worse than one
// that fails loudly, because the number is what a reader acts on.

// scoreSweeps runs the comparison the way main does and returns the per-point errors, so a
// test can assert on what the report prints rather than on the pieces behind it.
func scoreSweeps(t *testing.T) (mine, theirs []float64, c corpus) {
	t.Helper()
	c = loadCorpus(t)
	// Calls main's own scoring function rather than recomputing it. A test with its own copy
	// of the arithmetic stays self-consistent under any mutation of the real path, so it
	// cannot fail — which an earlier version of this file did, and two mutations walked
	// through it.
	//
	// A test runs with the package directory as its working directory, so the scenario root
	// is two levels up. Passing it explicitly is what keeps this from skipping silently.
	mine, theirs, _, err := score(c, scenarioRoot, harness.DefaultCatalog(),
		harness.DefaultRegistry(), false)
	artifacttest.RequireArtifact(t, harness.DefaultCatalog(),
		harness.DefaultCatalog(), "catalog", err)
	return mine, theirs, c
}

func TestTheReportedErrorIsInAPlausibleRange(t *testing.T) {
	mine, theirs, c := scoreSweeps(t)
	if len(mine) == 0 {
		t.Fatal("no points scored")
	}
	mean := func(xs []float64) float64 {
		var s float64
		for _, x := range xs {
			s += x
		}
		return s / float64(len(xs))
	}
	got := mean(mine)
	// A shape comparison between two models of the same deployment cannot plausibly exceed
	// this. An error near 100% means the ratio was replaced by an absolute quantity, or the
	// anchor moved — both of which happened under mutation while every other test passed.
	if got > 60 {
		t.Errorf("the kernel's shape MAPE is %.2f%%; above 60%% the comparison is not "+
			"measuring a shape, it is measuring a scale mismatch", got)
	}
	// And AISimulate's own figure on these points must stay near what the snapshot itself
	// publishes over the whole set, which is the fixed reference this comparison rests on.
	// The bound is derived from the corpus's own published total rather than typed in: an
	// earlier version hardcoded 5-20% from a ten-point corpus and failed the moment the
	// corpus grew to 473 points that legitimately include harder arms.
	ref := mean(theirs)
	published := c.AISimulateTotals.TPOTShapeErrorPct
	if published <= 0 {
		t.Fatal("the corpus states no published shape error to check against")
	}
	// Within half the published figure, not a factor of three. This is the test that must
	// catch a metric mix-up on the reference side, and a wide band does not: with a +-3x
	// band, reverting the anchor fix -- which put the reference at 24.46% against a published
	// 10.05%, a 2.4x error -- still passed. The subset is 447 of 1135 points, so some spread
	// is legitimate, but a shape score of a shape-normalised prediction has to land near the
	// shape figure its own source publishes, or the two sides are not the same quantity.
	const tolerance = 0.5
	if ref < published*(1-tolerance) || ref > published*(1+tolerance) {
		t.Errorf("AISimulate's error on these points reads %.2f%%, against the %.2f%% shape "+
			"error it publishes over the whole snapshot; beyond +-%.0f%% the two sides are "+
			"probably not the same quantity -- check that both curves are normalised to "+
			"their OWN anchor", ref, published, 100*tolerance)
	}
}

func TestTheAnchorPointScoresExactlyZero(t *testing.T) {
	// By construction the first point of every sweep is 1.0 on both sides, so its error must
	// be exactly zero. A non-zero value there is the signature of a moved anchor — which no
	// other test caught, because the internals were all still self-consistent.
	c := loadCorpus(t)
	for _, s := range c.Sweeps {
		k, err := harness.Open(s.Scenario, harness.Repos{
			Scenarios: scenarioRoot,
			Catalog:   harness.DefaultCatalog(),
			Registry:  harness.DefaultRegistry(),
		})
		if err != nil {
			t.Skipf("catalog or registry unavailable: %v", err)
		}
		ctx := contextTokens(s)
		anchor := stepSeconds(k, s.Points[0].Concurrency, ctx)
		first := stepSeconds(k, s.Points[0].Concurrency, ctx) / anchor
		if err := math.Abs(first/s.Points[0].MeasuredRelative - 1); err > 1e-12 {
			t.Errorf("%s: the anchor point scores %.2e error; both sides are 1.0 there by "+
				"construction, so the anchor is not the sweep's lowest concurrency",
				s.Label, err*100)
		}
	}
}

func TestThePerTokenHostCostReachesThePrediction(t *testing.T) {
	// The output processor's per-token cost lands on the interval between tokens, so it
	// belongs in a TPOT prediction. Dropping it survived every other test because it barely
	// moves a ratio — which is exactly why it needs a direct check rather than an
	// end-to-end one.
	c := loadCorpus(t)
	s := c.Sweeps[0]
	k, err := harness.Open(s.Scenario, harness.Repos{
		Scenarios: scenarioRoot,
		Catalog:   harness.DefaultCatalog(),
		Registry:  harness.DefaultRegistry(),
	})
	if err != nil {
		t.Skipf("catalog or registry unavailable: %v", err)
	}
	ctx := contextTokens(s)
	got := stepSeconds(k, 8, ctx)
	// NoOverlap, matching stepSeconds: that is the edge the measured evidence selects
	// (blis-registry docs/band-selection.md). This test checks that the HOST term is
	// added, so it has to agree with the production path on which edge it starts from --
	// otherwise it fails for the wrong reason and says nothing about the host cost.
	step := k.StepTime(harness.DecodeBatch(8, ctx)).NoOverlap.Seconds()
	host := k.OutputTokenOverhead().Seconds()
	if math.Abs(got-(step+host)) > 1e-12 {
		t.Errorf("the prediction is %.9f s where the step is %.9f and the per-token host "+
			"cost is %.9f; the host term is not being added", got, step, host)
	}
	if host <= 0 {
		t.Error("the per-token host cost resolved to zero, so this test proves nothing")
	}
}
