//go:build scoring

// Moved from blis-latency-kernel/cmd/score/main_scoring_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// The test that reads the measurement corpus, which is not distributed with this repository
// (harness.DefaultMeasurements). Run it with `go test -tags scoring ./cmd/...` once the
// corpus is in testdata/measurements or BLIS_MEASUREMENTS.

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/kerneltests/internal/harness"
)

// --- The measurement corpus ------------------------------------------------------

func TestTheCommittedCorpusIsWellFormed(t *testing.T) {
	corpusPath := filepath.Join(harness.DefaultMeasurements(), "scoreable.json")
	raw, err := os.ReadFile(corpusPath)
	if err != nil {
		// Opted into by the build tag, so an absent corpus is a failure, not a skip.
		t.Fatalf("measurement corpus %s: %v (it is not distributed with this repository; "+
			"see harness.DefaultMeasurements)", corpusPath, err)
	}
	var points []point
	if err := json.Unmarshal(raw, &points); err != nil {
		t.Fatalf("corpus does not parse: %v", err)
	}
	// The corpus held 35 points until 27 were dropped: they named a scenario whose
	// model is not in blis-catalog, so the scenario could not be loaded and cmd/score
	// failed outright rather than scoring what it could. The floor is the count that
	// remains, so a further silent shrink still fails.
	if len(points) < 8 {
		t.Errorf("%d points; the corpus holds 8", len(points))
	}
	seen := map[string]bool{}
	for _, p := range points {
		if p.ITLms <= 0 {
			t.Errorf("%s/%s at c=%d has no measured ITL", p.Report, p.Series,
				p.Concurrency)
		}
		if p.Concurrency <= 0 {
			t.Errorf("%s/%s has no concurrency", p.Report, p.Series)
		}
		if p.Scenario == "" {
			t.Errorf("%s/%s names no scenario", p.Report, p.Series)
		}
		key := p.Scenario + "|" + p.Series + "|" + string(rune(p.Concurrency))
		if seen[key] {
			t.Errorf("duplicate point: %s %s c=%d", p.Scenario, p.Series, p.Concurrency)
		}
		seen[key] = true
	}
	// Every scenario named must exist, or the harness would skip silently.
	for _, p := range points {
		if _, err := os.Stat(filepath.Join("../../testdata", p.Scenario)); err != nil {
			t.Errorf("%s names scenario %s, which is absent", p.Report, p.Scenario)
		}
	}
	// And the corpus must span more than one model, since one model is a weak test.
	scenarios := map[string]bool{}
	for _, p := range points {
		scenarios[p.Scenario] = true
	}
	// Two scenarios, two models: Kimi-K3 and Nemotron-3-Ultra. Two is thin but it is
	// more than one, which is the property that matters:
	// a single model would let a model-specific error look like a general one.
	if len(scenarios) < 2 {
		t.Errorf("the corpus covers %d scenarios; scoring one model proves little",
			len(scenarios))
	}
	models := map[string]bool{}
	for _, p := range points {
		models[strings.SplitN(p.Scenario, "-", 2)[0]] = true
	}
	if len(models) < 2 {
		t.Errorf("the corpus covers %d model families; one is a weak test", len(models))
	}
}
