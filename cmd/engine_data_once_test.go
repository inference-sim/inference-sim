package cmd

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/load"

	coreengine "github.com/inference-sim/blis-vllm/vllmengine"
)

// A run opens its catalog and registry once and builds every engine over that Data
// (blis-vllm#90, design §3.3, VAL-6). Three properties, each through a real multi-pool run:
//
//	(a) the Result's one selected-entry report holds the lookups of every pool's engine;
//	(b) the run opens its Data once through the host's opener, and every engine build the
//	    host makes is handed that same Data;
//	(c) a configuration that names a root beside the Data is refused with the host-facing
//	    message, and the closest valid one (the Data alone) builds (SIM-9).

const engineDataLegEnv = "BLIS_ENGINE_DATA_LEG"

// engineDataLegReport is the line the leg writes to stderr for the parent to read.
var engineDataLegReport = regexp.MustCompile(`ENGINE-DATA opens=(\d+) builds=(\d+) distinct=(\d+) coefficients=(\d+)`)

// TestEngineDataLeg is the re-exec target: it runs a 3P1D disaggregated run (four engine
// builds over two pools), counting the opens of the run's Data and the Data each build is
// handed, and reports them on stderr.
func TestEngineDataLeg(t *testing.T) {
	if os.Getenv(engineDataLegEnv) == "" {
		t.Skip("re-exec target only; driven by TestRun_OpensItsDataOnceForEveryPoolsEngine")
	}
	opens, builds, coefficients := 0, 0, 0
	seen := map[*load.Data]bool{}
	openWas, buildWas := openEngineData, newContractEngine
	openEngineData = func(o load.Options) (*load.Data, error) { opens++; return openWas(o) }
	newContractEngine = func(spec engine.Spec, data *load.Data) (engine.Engine, error) {
		builds++
		seen[data] = true
		eng, err := buildWas(spec, data)
		coefficients = len(data.Report().Result().Coefficients)
		return eng, err
	}
	scenarios, catalog, registry := kernelRepos(t)
	_ = scenarios
	args := append([]string{"run"}, pdTopology...)
	args = append(args, "--num-requests", "20", "--rate", "50", "--catalog", catalog, "--registry", registry,
		"--defaults-filepath", "../defaults.yaml")
	rootCmd.SetArgs(args)
	err := rootCmd.Execute()
	fmt.Fprintf(os.Stderr, "ENGINE-DATA opens=%d builds=%d distinct=%d coefficients=%d\n", opens, builds, len(seen), coefficients)
	if err != nil {
		os.Exit(1)
	}
	os.Exit(0)
}

// What this proves and what it does not: it counts the host's own opener and the Data handed
// to the host's engine-build seam, so it fails if the host opens its Data twice or hands two
// builds different Data. It cannot see blis-vllm opening data inside a build (that opener is
// unexported there), so it does not by itself fail on the old roots-based path; the report
// test below and the root-refusal test do.
func TestRun_OpensItsDataOnceForEveryPoolsEngine(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestEngineDataLeg$")
	cmd.Env = append(os.Environ(), engineDataLegEnv+"=1")
	var out, errBuf strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &errBuf
	if err := cmd.Run(); err != nil {
		t.Fatalf("the 3P1D run failed: %v\n%s", err, lastLines(errBuf.String(), 5))
	}
	m := engineDataLegReport.FindStringSubmatch(errBuf.String())
	if m == nil {
		t.Fatalf("the leg reported no counts:\n%s", lastLines(errBuf.String(), 5))
	}
	opens, _ := strconv.Atoi(m[1])
	builds, _ := strconv.Atoi(m[2])
	distinct, _ := strconv.Atoi(m[3])
	coefficients, _ := strconv.Atoi(m[4])

	// (b) One open for the run, however many engines it builds; every build over the same Data.
	if builds < 2 {
		t.Fatalf("non-vacuity: %d engine builds; a multi-pool run builds several", builds)
	}
	if opens != 1 {
		t.Errorf("the run opened its data %d times, want 1 (builds: %d)", opens, builds)
	}
	if distinct != 1 {
		t.Errorf("%d engine builds were handed %d different Data, want one shared Data", builds, distinct)
	}

	// (a) The Result's single report is that Data's report: it holds what the engines looked
	// up, all of it, and not a share of it.
	r := parseResult(t, out.String())
	if coefficients == 0 {
		t.Fatalf("non-vacuity: the engines looked up no coefficient in the run's Data")
	}
	if got := len(r.SelectedEntries.Coefficients); got != coefficients {
		t.Errorf("selected_entries holds %d coefficients, the run's Data holds %d after the engines were built: "+
			"the report is split", got, coefficients)
	}
}

// (a), at the factory: the engines of a multi-pool deployment, built one after another over one
// fresh Data, put their lookups in that Data's report, and each pool's build adds to the same
// report rather than to a report of its own. A host whose engines opened their own data would
// leave this report empty. The prefill and decode pools of the 3P1D scenario are pools 0 and 1.
func TestContractEngine_EveryPoolsLookupsLandInTheRunsOneReport(t *testing.T) {
	registerContractEngines()
	scenarioPath, err := filepath.Abs(filepath.Join(pdScenarios, "glm-5-h200-3p1d-ib.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	data, err := load.Open(load.Options{Catalog: testCatalogDir(), Registry: testRegistryDir()})
	if err != nil {
		t.Fatal(err)
	}
	if n := len(data.Report().Result().Coefficients); n != 0 {
		t.Fatalf("non-vacuity: a fresh Data already reports %d coefficients", n)
	}
	var after []int
	for pool := 0; pool < 2; pool++ {
		spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: contractRoots(scenarioPath), Pool: pool, Seed: 1}
		if _, err := newContractEngine(spec, data); err != nil {
			t.Fatalf("pool %d: %v", pool, err)
		}
		after = append(after, len(data.Report().Result().Coefficients))
	}
	if after[0] == 0 {
		t.Errorf("the first pool's engine put no lookup in the run's report (%v)", after)
	}
	if after[1] < after[0] {
		t.Errorf("the report shrank between pools: %v", after)
	}
	// One report: a second Data opened for the same run would hold only its own pool's share.
	other, err := load.Open(load.Options{Catalog: testCatalogDir(), Registry: testRegistryDir()})
	if err != nil {
		t.Fatal(err)
	}
	if n := len(other.Report().Result().Coefficients); n != 0 {
		t.Errorf("a Data no engine was handed reports %d coefficients", n)
	}
}

// (c) The SIM-9 pair, at the engine factory the run builds through: a root beside the Data is
// refused by the engine with a message that tells the host what to do, and the closest valid
// configuration, the Data alone with only the scenario root, builds.
func TestContractEngine_RootBesideDataIsRefusedAndDataAloneBuilds(t *testing.T) {
	registerContractEngines()
	scenarioPath, err := filepath.Abs(filepath.Join(testScenarios, kernelTestScenario))
	if err != nil {
		t.Fatal(err)
	}
	data, err := load.Open(load.Options{Catalog: testCatalogDir(), Registry: testRegistryDir()})
	if err != nil {
		t.Fatal(err)
	}
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Pool: 0, Seed: 1}

	roots := contractRoots(scenarioPath)
	for k := range roots {
		if k != coreengine.RootScenario {
			t.Errorf("contractRoots names %q beside the Data; the run's Data carries it", k)
		}
	}
	for _, root := range []string{coreengine.RootCatalog, coreengine.RootRegistry,
		coreengine.RootCatalogOverlays, coreengine.RootRegistryOverlays} {
		bad := spec
		bad.Roots = map[string]string{coreengine.RootScenario: scenarioPath, root: testCatalogDir()}
		_, err := newContractEngine(bad, data)
		if err == nil || !strings.Contains(err.Error(), "Input.Data") || !strings.Contains(err.Error(), `Roots["`+root+`"]`) {
			t.Errorf("Roots[%q] beside Data: err = %v, want the refusal that names the root and Input.Data", root, err)
		}
	}
	ok := spec
	ok.Roots = roots
	if _, err := newContractEngine(ok, data); err != nil {
		t.Fatalf("the Data alone, with the scenario root, must build: %v", err)
	}
}

// A scenario whose engine the factory refuses at startup (a combination the engine does not
// build, such as KV transfer with data parallelism) is a refusal the user can fix, so the CLI
// reports it as one message and exits 1, and does not dump a goroutine trace (exit 2).
func TestCLI_AnEngineRefusedAtStartupIsAFatalMessageNotAPanic(t *testing.T) {
	scenarios, err := filepath.Abs(pdScenarios)
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"run", "--scenarios", scenarios, "--scenario", "minimax-m2.5-gb200-1p1d-dp2.yaml",
		"--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1", "--num-requests", "4"}
	if _, statErr := os.Stat(filepath.Join(scenarios, "minimax-m2.5-gb200-1p1d-dp2.yaml")); statErr != nil {
		t.Fatalf("the fixture this test runs is missing from %s: %v", pdScenarios, statErr)
	}
	_, stderr, runErr := runKernelCLI(t, args...)
	if runErr == nil {
		t.Fatalf("the engine refuses this scenario at startup, and the run succeeded")
	}
	if strings.Contains(stderr, "goroutine ") || strings.Contains(stderr, "panic:") {
		t.Errorf("a startup refusal dumped a panic:\n%s", lastLines(stderr, 6))
	}
	if !strings.Contains(stderr, "level=fatal") || !strings.Contains(stderr, "not built yet") {
		t.Errorf("the refusal is not reported as a fatal message naming why:\n%s", lastLines(stderr, 4))
	}
	var ee *exec.ExitError
	if errors.As(runErr, &ee) && ee.ExitCode() != 1 {
		t.Errorf("exit status %d, want 1", ee.ExitCode())
	}
}
