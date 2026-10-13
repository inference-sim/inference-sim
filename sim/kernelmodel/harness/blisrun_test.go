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

// The variant scenario a point runs carries the point's resolved run settings and nothing
// else changed: the kernel and `blis replay` both read the engine from it, so a setting lost
// on the way would be simulated at the committed value while reported as measured.
func TestScenarioVariantCarriesTheResolvedSettings(t *testing.T) {
	const name = "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"
	src := filepath.Join(kernelmodel.DefaultScenarios(), name)
	for _, pc := range []PointConfig{
		{MaxNumSeqs: 37, MaxNumBatchedTokens: 4097, BlockSize: 32, PrefixCachingDisabled: true,
			MaxModelLen: 2304, GPUMemoryUtilization: 0.95, EnableExpertParallel: true},
		{MaxNumSeqs: 1024, MaxNumBatchedTokens: 16384, BlockSize: 16, PrefixCachingDisabled: false,
			MaxModelLen: 9472, GPUMemoryUtilization: 0.9, EnableExpertParallel: false},
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
			*e.EnablePrefixCaching == pc.PrefixCachingDisabled || e.MaxModelLen != pc.MaxModelLen ||
			e.GPUMemoryUtilization != pc.GPUMemoryUtilization ||
			gotDep.Pools[0].Parallel.EnableExpertParallel != pc.EnableExpertParallel {
			t.Errorf("variant pool %+v does not carry %+v", gotDep.Pools[0], pc)
		}
		// Everything else is the committed file's.
		applyPointConfig(&wantDep.Pools[0], pc)
		if !reflect.DeepEqual(gotSc, wantSc) || !reflect.DeepEqual(gotDep, wantDep) {
			t.Errorf("the variant changed more than the run settings:\n got %+v %+v\nwant %+v %+v",
				gotSc, gotDep, wantSc, wantDep)
		}
		// And the kernel reads the same settings back, which is what `blis replay` sizes from.
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
			s.BlockSize != pc.BlockSize || s.PrefixCachingDisabled != pc.PrefixCachingDisabled ||
			s.MaxModelLen != pc.MaxModelLen {
			t.Errorf("kernel settings %+v, want %+v", s, pc)
		}
		if d := m.Deployment(); d.GPUMemUtil != pc.GPUMemoryUtilization || d.ExpertParallel != pc.EnableExpertParallel {
			t.Errorf("kernel deployment memory utilisation %v and expert parallelism %v, want %v and %v",
				d.GPUMemUtil, d.ExpertParallel, pc.GPUMemoryUtilization, pc.EnableExpertParallel)
		}
	}
}

// The command line is a user's `blis replay` in pool mode: CONC concurrent one-round sessions,
// every trace record replayed once and in order. It names no data-parallel width (the command
// places dp replicas from the scenario) and no engine setting (the scenario states them).
func TestTheCommandLineIsAUsersBlisReplay(t *testing.T) {
	cfg := Config{Repos: testRepos(), Seed: 7}
	args := blisArgs(cfg, "s.yaml", "/d", "/d/t.yaml", "/d/t.csv", "/d/m.json", 40, 4)
	if args[0] != "replay" {
		t.Fatalf("first argument %q, want replay", args[0])
	}
	line := strings.Join(args, " ")
	for _, want := range []string{"--scenario s.yaml", "--scenarios /d", "--trace-header /d/t.yaml",
		"--trace-data /d/t.csv", "--session-mode closed-loop", "--concurrent-sessions 4",
		"--total-sessions 40", "--metrics-path /d/m.json", "--seed 7",
		"--catalog " + cfg.Repos.Catalog, "--registry " + cfg.Repos.Registry} {
		if !strings.Contains(line, want) {
			t.Errorf("%q lacks %q", line, want)
		}
	}
	for _, banned := range []string{"--dp", "--num-instances", "--max-num-running-reqs", "--max-num-scheduled-tokens",
		"--total-kv-blocks", "--shuffle-corpus", "--think-time", "--horizon"} {
		if strings.Contains(line, banned) {
			t.Errorf("%q passes %s; the scenario and the trace are the only sources of the point", line, banned)
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

// summarize computes the client's two means over exactly the point's requests: TTFT over all of
// them, TPOT from each request's own inter-token mean (never e2e, which carries the completion
// overhead after the last token), and it refuses a run that is missing a request, ran an extra
// one, or ran a request at lengths other than the fixture's.
func TestSummarizeIsTheClientsMetricOverTheMainPhase(t *testing.T) {
	pl := PointLengths{ISL: 4, OSL: 4, Concurrency: 1, InputLens: []int{4, 3}, OutputLens: []int{4, 1}}
	reqs := []sim.RequestMetrics{
		// Listed out of order; e2e deliberately inconsistent with ITL so a TPOT taken from
		// e2e would differ.
		{ID: "request_1", CompletionIndex: 2, NumPrefillTokens: 3, NumDecodeTokens: 1, TTFT: 30, E2E: 30},
		{ID: "request_0", CompletionIndex: 1, NumPrefillTokens: 4, NumDecodeTokens: 4, TTFT: 10, ITL: 2, E2E: 99},
	}
	obs, err := summarize(reqs, pl, nil)
	if err != nil {
		t.Fatal(err)
	}
	// ms in the file, us out. TTFT over both requests; TPOT over the one with output > 1.
	if obs.Completed != 2 || obs.MeanTTFTUs != 20000 || obs.TPOTRequests != 1 || obs.MeanTPOTUs != 2000 {
		t.Errorf("got %+v; want 2 completed, TTFT 20000 us, TPOT 2000 us over 1 request", obs)
	}

	for name, bad := range map[string][]sim.RequestMetrics{
		"a missing request": reqs[1:],
		"an unknown request": append([]sim.RequestMetrics{{ID: "request_7", CompletionIndex: 3,
			NumPrefillTokens: 4, NumDecodeTokens: 4}}, reqs...),
		"a request at other lengths": {reqs[0], {ID: "request_0", CompletionIndex: 1, NumPrefillTokens: 5,
			NumDecodeTokens: 4}},
		"an uncompleted request": {reqs[0], {ID: "request_0", NumPrefillTokens: 4, NumDecodeTokens: 4}},
	} {
		if _, err := summarize(bad, pl, nil); err == nil {
			t.Errorf("summarize accepted %s", name)
		}
	}
}
