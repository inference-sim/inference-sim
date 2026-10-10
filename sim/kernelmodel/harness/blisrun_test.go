package harness

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// The variant scenario a point runs carries the point's resolved admission settings and
// nothing else changed: the kernel and `blis run` both read the engine from it, so a setting
// lost on the way would be simulated at the committed value while reported as measured.
func TestScenarioVariantCarriesTheResolvedSettings(t *testing.T) {
	const name = "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"
	src := filepath.Join(kernelmodel.DefaultScenarios(), name)
	for _, pc := range []PointConfig{
		{MaxNumSeqs: 37, MaxNumBatchedTokens: 4097, BlockSize: 32, PrefixCachingDisabled: true},
		{MaxNumSeqs: 1024, MaxNumBatchedTokens: 16384, BlockSize: 16, PrefixCachingDisabled: false},
	} {
		dst := filepath.Join(t.TempDir(), name)
		if err := writeScenarioVariant(src, dst, pc); err != nil {
			t.Fatalf("writeScenarioVariant: %v", err)
		}
		wantSc, wantDep, err := latencykernel.LoadBundle(src)
		if err != nil {
			t.Fatal(err)
		}
		gotSc, gotDep, err := latencykernel.LoadBundle(dst)
		if err != nil {
			t.Fatalf("the variant does not load as a scenario: %v", err)
		}
		e := gotDep.Pools[0].Engine
		if e.MaxNumSeqs != pc.MaxNumSeqs || e.MaxNumBatchedTokens != pc.MaxNumBatchedTokens ||
			e.BlockSize != pc.BlockSize || e.EnablePrefixCaching == nil ||
			*e.EnablePrefixCaching == pc.PrefixCachingDisabled {
			t.Errorf("variant engine %+v does not carry %+v", e, pc)
		}
		// Everything else is the committed file's.
		applyPointConfig(&wantDep.Pools[0].Engine, pc)
		if !reflect.DeepEqual(gotSc, wantSc) || !reflect.DeepEqual(gotDep, wantDep) {
			t.Errorf("the variant changed more than the admission settings:\n got %+v %+v\nwant %+v %+v",
				gotSc, gotDep, wantSc, wantDep)
		}
		// And the kernel reads the same settings back, which is what `blis run` sizes from.
		r := testRepos()
		r.Scenarios = filepath.Dir(dst)
		m, err := kernelmodel.Open(name, r)
		if err != nil {
			t.Fatalf("kernel refuses the variant: %v", err)
		}
		s, err := m.Settings()
		if err != nil {
			t.Fatal(err)
		}
		if s.MaxNumSeqs != pc.MaxNumSeqs || s.MaxNumBatchedTokens != pc.MaxNumBatchedTokens ||
			s.BlockSize != pc.BlockSize || s.PrefixCachingDisabled != pc.PrefixCachingDisabled {
			t.Errorf("kernel settings %+v, want %+v", s, pc)
		}
	}
}

// The command line is the one a user types: it names no data-parallel width (blis run
// places dp replicas from the scenario) and no engine setting (the scenario states them).
func TestTheCommandLineIsAUsersBlisRun(t *testing.T) {
	cfg := Config{Repos: testRepos(), Seed: 7}
	args := blisArgs(cfg, "s.yaml", "/d", "/d/w.yaml", "/d/m.json")
	if args[0] != "run" {
		t.Fatalf("first argument %q, want run", args[0])
	}
	line := strings.Join(args, " ")
	for _, want := range []string{"--scenario s.yaml", "--scenarios /d", "--workload-spec /d/w.yaml",
		"--metrics-path /d/m.json", "--seed 7", "--catalog " + cfg.Repos.Catalog, "--registry " + cfg.Repos.Registry} {
		if !strings.Contains(line, want) {
			t.Errorf("%q lacks %q", line, want)
		}
	}
	for _, banned := range []string{"--dp", "--num-instances", "--max-num-running-reqs", "--max-num-scheduled-tokens", "--total-kv-blocks"} {
		if strings.Contains(line, banned) {
			t.Errorf("%q passes %s; the scenario is the only source of the deployment", line, banned)
		}
	}
}

// A missing or unusable blis binary is refused with a message saying how to provide one,
// before any point runs.
func TestAMissingBlisBinaryIsRefused(t *testing.T) {
	dir := t.TempDir()
	notExec := filepath.Join(dir, "blis")
	if err := os.WriteFile(notExec, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"", filepath.Join(dir, "absent"), dir, notExec} {
		err := RequireBlis(path)
		if err == nil {
			t.Errorf("RequireBlis(%q) accepted it", path)
			continue
		}
		if !strings.Contains(err.Error(), "blis") {
			t.Errorf("RequireBlis(%q): %v does not name the blis binary", path, err)
		}
	}
	if _, err := Run(Sweep{Scenario: "x.yaml", Workload: "1:1"}, 1, Config{}); err == nil ||
		!strings.Contains(err.Error(), "-blis") {
		t.Errorf("Run with no binary: %v, want the refusal naming -blis", err)
	}
}

// summarize reads completion order from completion_index, not from the order the metrics file
// lists requests in (it lists them by arrival), and averages only completed requests.
func TestSummarizeCutsTheWarmupByCompletionIndex(t *testing.T) {
	// Listed out of completion order; "never" did not complete and must not count.
	reqs := []sim.RequestMetrics{
		{ID: "late", CompletionIndex: 3, ITL: 3, TTFT: 30},
		{ID: "never", ITL: 1000, TTFT: 1000},
		{ID: "first", CompletionIndex: 1, ITL: 1, TTFT: 10},
		{ID: "mid", CompletionIndex: 2, ITL: 2, TTFT: 20},
	}
	obs, err := summarize(reqs, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if obs.Completed != 3 || obs.WarmupDiscarded != 1 || obs.Measured != 2 {
		t.Fatalf("completed %d, discarded %d, measured %d; want 3, 1, 2",
			obs.Completed, obs.WarmupDiscarded, obs.Measured)
	}
	// ms in the file, µs out: the post-warm-up ITL mean is (2+3)/2 ms.
	if obs.MeanITLUs != 2500 {
		t.Errorf("mean ITL %v us, want 2500 (the cut must drop the FIRST completion)", obs.MeanITLUs)
	}
	// TTFT keeps every completion.
	if obs.MeanTTFTUs != 20000 || obs.MeasuredTTFT != 3 {
		t.Errorf("mean TTFT %v us over %d, want 20000 over 3", obs.MeanTTFTUs, obs.MeasuredTTFT)
	}
}
