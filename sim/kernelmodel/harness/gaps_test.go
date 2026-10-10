package harness

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"pgregory.net/rapid"
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

// Gap conservation: every corpus point is either scored or covered by a gap. A scorer takes a
// sweep whole or not at all, so a sweep it could not score is covered by one sweep-level gap
// counting all its points, whichever point failed and however (a failed run or an unusable
// measurement); a point-level gap counts exactly one point of its own sweep.
func TestEveryPointIsScoredOrCoveredByAGap(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		cv := &Coverage{}
		var corpus []Sweep
		scored := map[int]bool{}
		for s := 0; s < rapid.IntRange(1, 6).Draw(rt, "sweeps"); s++ {
			sw := Sweep{Scenario: fmt.Sprintf("s%d.yaml", s), Label: "1k1k"}
			for c := 0; c < rapid.IntRange(1, 8).Draw(rt, "points"); c++ {
				sw.Points = append(sw.Points, Point{Concurrency: 1 << c})
			}
			corpus = append(corpus, sw)
			fail := rapid.IntRange(-1, len(sw.Points)-1).Draw(rt, "failingPoint")
			switch {
			case fail < 0:
				scored[s] = true
			case rapid.Bool().Draw(rt, "badMeasurement"):
				cv.BadMeasurement(sw, sw.Points[fail].Concurrency, "relative <= 0")
			default:
				cv.Dropped(sw, sw.Points[fail].Concurrency, "blis run: exit status 1\nlast line")
			}
		}
		covered := map[string]int{}
		for _, g := range cv.Gaps {
			covered[g.Scenario] += g.Points
		}
		total, accounted := 0, 0
		for s, sw := range corpus {
			total += len(sw.Points)
			switch {
			case scored[s] && covered[sw.Scenario] > 0:
				rt.Fatalf("%s was scored and also covered by %d gap points", sw.Scenario, covered[sw.Scenario])
			case scored[s]:
				accounted += len(sw.Points)
			case covered[sw.Scenario] != len(sw.Points):
				rt.Fatalf("%s: a sweep that could not be scored is covered for %d of its %d points",
					sw.Scenario, covered[sw.Scenario], len(sw.Points))
			default:
				accounted += covered[sw.Scenario]
			}
		}
		if accounted != total {
			rt.Fatalf("%d of %d points scored or covered", accounted, total)
		}
	})
	// The same law on the gaps AssessCoverage finds itself: a sweep-level gap counts the
	// sweep's points and a point-level gap one point of that sweep.
	sw := committedSweep("sglang")
	gaps := gapsFor(t, sw, Config{Repos: testRepos()})
	if len(gaps) == 0 {
		t.Fatal("a non-vLLM sweep produced no gap; the check below would be vacuous")
	}
	for _, g := range gaps {
		if g.Concurrency == 0 && g.Points != len(sw.Points) {
			t.Errorf("sweep-level gap %+v counts %d points, the sweep has %d", g, g.Points, len(sw.Points))
		}
		if g.Concurrency > 0 && (g.Points != 1 || !slices.ContainsFunc(sw.Points, func(p Point) bool { return p.Concurrency == g.Concurrency })) {
			t.Errorf("point-level gap %+v is not one point of its sweep", g)
		}
	}
}

// The gap report never reaches stdout, which carries the score tables: with no path it goes to
// stderr, with one to the file.
func TestWriteGapsNeverWritesStdout(t *testing.T) {
	cv := &Coverage{Gaps: []Gap{{Scenario: "a.yaml", Points: 2, Cause: "a cause", Owner: OwnerSim}}}
	capture := func(f **os.File, write func()) string {
		r, w, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		orig := *f
		*f = w
		write()
		*f = orig
		_ = w.Close()
		var b bytes.Buffer
		_, _ = b.ReadFrom(r)
		return b.String()
	}
	path := filepath.Join(t.TempDir(), "gaps.txt")
	var stderr string
	stdout := capture(&os.Stdout, func() {
		stderr = capture(&os.Stderr, func() {
			if err := cv.WriteGaps(""); err != nil {
				t.Error(err)
			}
			if err := cv.WriteGaps(path); err != nil {
				t.Error(err)
			}
		})
	})
	if stdout != "" {
		t.Errorf("the gap report reached stdout:\n%s", stdout)
	}
	if !strings.Contains(stderr, "a cause") {
		t.Errorf("with no path the report did not go to stderr: %q", stderr)
	}
	if raw, err := os.ReadFile(path); err != nil || !strings.Contains(string(raw), "a cause") {
		t.Errorf("with a path the report was not written there: %v %q", err, raw)
	}
}
