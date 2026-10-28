// absolutes.go loads SemiAnalysis InferenceX's absolute measured latencies, which is what makes a
// mean absolute percentage error computable for a model whose predictions are in real units.
//
// NVIDIA's accuracy artifact normalises every latency to its sweep's lowest concurrency, so the
// only quantity it withholds is the measured latency at that anchor. InferenceX publishes it: the
// artifact's own snapshot block names the source and pins the release. blis-latency-kernel's
// scripts/extract_inferencex_absolutes.py reads that dump and writes the file this loads, after
// validating each sweep's run against the artifact's own relatives -- all 83 sweeps reproduce them
// to 0.000%, which is what establishes that these are the same rows AISimulate was scored on.
//
// Without this file a BLIS mape would require inventing the anchor. With it, mape and shape are
// both computable from the same measurement, and they answer different questions: shape divides
// the level out and asks only whether latency responds to concurrency correctly, while mape asks
// whether the latency itself is right.
package harness

import (
	"encoding/json"
	"fmt"
	"os"
)

// Absolutes is the measured absolute latency curve for one sweep, in microseconds.
type Absolutes struct {
	Scenario  string             `json:"scenario"`
	Label     string             `json:"label"`
	Framework string             `json:"framework"`
	GPU       string             `json:"gpu"`
	TPOTUs    map[string]float64 `json:"absolute_tpot_us"`
	TTFTUs    map[string]float64 `json:"absolute_ttft_us"`
	// RunDate and ConfigID identify the InferenceX run these came from, so a figure can be
	// traced to a row rather than to a file.
	RunDate  string `json:"source_run_date"`
	ConfigID int    `json:"source_config_id"`
	// Agreement is how closely this run's relatives reproduce the artifact's. Zero means exact.
	Agreement float64 `json:"relative_agreement"`
}

// AbsoluteSet indexes the measured curves by sweep.
type AbsoluteSet struct {
	Source     string      `json:"source"`
	SourceURL  string      `json:"source_url"`
	ReleaseTag string      `json:"release_tag"`
	Metrics    string      `json:"metrics"`
	Validation string      `json:"validation"`
	Anchors    []Absolutes `json:"anchors"`

	byKey map[string]*Absolutes
}

// LoadAbsolutes reads the extracted measured curves. A missing file is reported rather than
// silently producing a run with no mape column: a caller that asked for absolute scoring and got
// shape scoring instead would not know.
func LoadAbsolutes(path string) (*AbsoluteSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("measured absolutes: %w", err)
	}
	var set AbsoluteSet
	if err := json.Unmarshal(data, &set); err != nil {
		return nil, fmt.Errorf("measured absolutes: %w", err)
	}
	if len(set.Anchors) == 0 {
		return nil, fmt.Errorf("measured absolutes at %s carry no sweeps", path)
	}
	set.byKey = set.index()
	return &set, nil
}

// index builds the lookup. Split out so the key format is stated once.
func (s *AbsoluteSet) index() map[string]*Absolutes {
	m := make(map[string]*Absolutes, len(s.Anchors))
	for i := range s.Anchors {
		m[absKey(s.Anchors[i].Scenario, s.Anchors[i].Label)] = &s.Anchors[i]
	}
	return m
}

func absKey(scenario, label string) string { return scenario + "\x00" + label }

// For returns the measured curve for a sweep, or nil when the extraction omitted it.
func (s *AbsoluteSet) For(sw Sweep) *Absolutes {
	if s == nil {
		return nil
	}
	if s.byKey == nil {
		s.byKey = s.index()
	}
	return s.byKey[absKey(sw.Scenario, sw.Label)]
}

// At returns the measured absolute latency in microseconds for one metric at one concurrency.
func (a *Absolutes) At(m Metric, concurrency int) (float64, bool) {
	if a == nil {
		return 0, false
	}
	table := a.TPOTUs
	if m == MetricTTFT {
		table = a.TTFTUs
	}
	v, ok := table[fmt.Sprintf("%d", concurrency)]
	if !ok || v <= 0 {
		return 0, false
	}
	return v, true
}
