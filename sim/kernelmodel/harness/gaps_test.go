package harness

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// gapsFor assesses a one-sweep corpus and returns its gaps.
func gapsFor(t *testing.T, sw Sweep, cfg Config) []Gap {
	t.Helper()
	return AssessCoverage(&Corpus{Sweeps: []Sweep{sw}}, cfg).Gaps
}

func hasGap(gaps []Gap, owner, causePrefix string) bool {
	for _, g := range gaps {
		if g.Owner == owner && strings.HasPrefix(g.Cause, causePrefix) {
			return true
		}
	}
	return false
}

func committedSweep(framework string) Sweep {
	return Sweep{Scenario: "gpt-oss-120b-h200-fp4-vllm-tp4.yaml", Label: "1k1k", Workload: "1024:1024",
		Framework: framework, Serving: "aggregated", SpecMethod: "none",
		Points: []Point{{Concurrency: 4}, {Concurrency: 8}}}
}

// A deployment whose model the catalog lacks is reported against the catalog, from the
// scenario the sweep names -- not from a list of known-missing models.
func TestAModelMissingFromTheCatalogIsAGapOwnedByTheCatalog(t *testing.T) {
	const name = "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"
	raw, err := os.ReadFile(filepath.Join(kernelmodel.DefaultScenarios(), name))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	edited := strings.Replace(string(raw), "model: gpt-oss-120b", "model: no-such-model", 1)
	if err := os.WriteFile(filepath.Join(dir, name), []byte(edited), 0o644); err != nil {
		t.Fatal(err)
	}
	r := testRepos()
	r.Scenarios = dir
	gaps := gapsFor(t, committedSweep("vllm"), Config{Repos: r})
	if !hasGap(gaps, OwnerCatalog, "model not in blis-catalog") {
		t.Errorf("gaps %+v lack the missing model", gaps)
	}
	for _, g := range gaps {
		if g.Points != 2 {
			t.Errorf("sweep-level gap %+v covers %d points, want the sweep's 2", g, g.Points)
		}
	}
}

// A non-vLLM measurement is a gap, and the deployment checks still run on it.
func TestANonVLLMSweepIsAGap(t *testing.T) {
	gaps := gapsFor(t, committedSweep("sglang"), Config{Repos: testRepos()})
	if !hasGap(gaps, OwnerSim, "engine framework is not vLLM") {
		t.Errorf("gaps %+v lack the framework", gaps)
	}
}

// With no engine-settings record, every admission setting of every point is reported as not
// measured; a setting the run passed is checked against the schema and the scenario.
func TestSettingsProvenanceAndPassedSettingsAreGaps(t *testing.T) {
	sw := committedSweep("vllm")
	gaps := gapsFor(t, sw, Config{Repos: testRepos()})
	n := 0
	for _, g := range gaps {
		if g.Owner == OwnerMeasurements {
			n++
		}
	}
	if n != 4*len(sw.Points) {
		t.Errorf("%d not-measured gaps, want 4 settings x %d points", n, len(sw.Points))
	}

	var ps PointSettings
	if err := json.Unmarshal([]byte(`{"passed":{"max_num_seqs":8,"max_num_batched_tokens":8192,
		"block_size":16,"enable_prefix_caching":false,"max_model_len":2304,
		"gpu_memory_utilization":0.9,"max_cudagraph_capture_size":2048},"resolved":{}}`), &ps); err != nil {
		t.Fatal(err)
	}
	set := &EngineSettingSet{Settings: []SweepSettings{{Scenario: sw.Scenario, Label: sw.Label,
		ByConcurrency: map[string]*PointSettings{"4": &ps}}}}
	gaps = gapsFor(t, sw, Config{Repos: testRepos(), EngineSettings: set})
	// max_model_len is a schema field the scenario states differently: not carried.
	if !hasGap(gaps, OwnerSim, "run setting not carried into the simulation: max_model_len") {
		t.Errorf("gaps lack max_model_len: %+v", gaps)
	}
	// The schema has no field for the capture size.
	if !hasGap(gaps, OwnerSchemas, "run setting blis-schemas cannot express: max_cudagraph_capture_size") {
		t.Errorf("gaps lack max_cudagraph_capture_size: %+v", gaps)
	}
	// gpu_memory_utilization matches the scenario's 0.9, and the four applied settings are
	// carried, so none of them is a gap.
	for _, k := range []string{"gpu_memory_utilization", "max_num_seqs)", ": max_num_seqs"} {
		for _, g := range gaps {
			if g.Concurrency == 4 && strings.Contains(g.Cause, k) {
				t.Errorf("unexpected gap %+v", g)
			}
		}
	}
}

// The report ends with a per-cause summary, largest first.
func TestTheReportSummarisesPointsPerCause(t *testing.T) {
	cv := &Coverage{Gaps: []Gap{
		{Scenario: "a", Points: 1, Cause: "small", Owner: OwnerSim},
		{Scenario: "b", Points: 5, Cause: "big", Owner: OwnerCatalog},
		{Scenario: "c", Points: 1, Cause: "small", Owner: OwnerSim},
	}}
	var buf bytes.Buffer
	if err := cv.Write(&buf); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	summary := out[strings.Index(out, "summary"):]
	if i, j := strings.Index(summary, "big"), strings.Index(summary, "small"); i < 0 || j < 0 || i > j {
		t.Errorf("summary not ordered largest first:\n%s", summary)
	}
	if !strings.Contains(summary, "      2  "+OwnerSim) {
		t.Errorf("summary does not sum points per cause:\n%s", summary)
	}
}
