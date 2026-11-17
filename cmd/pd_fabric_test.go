package cmd

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// A disaggregated scenario that omits cost-model-fabric is not refused by the engine: it prices
// every handoff at the fabric's nominal rate (efficiency 1), about 23% fast on ib-400g, and says
// so in the Result as an assumed fabric_bandwidth_efficiency. That label is the honest outcome
// the engine offers, so the scenarios this repository ships state the set, and these tests hold
// both halves: a shipped P/D scenario carries no such assumption, and one without the set does.

// shippedPDScenarios are the disaggregated scenario files a reader or a test runs, by directory.
// testdata/bytegate/scenarios is left out on purpose: it is the byte gate's fixed input, and
// changing it would move the gate's pd golden, which exists to show what a pin move does.
var shippedPDScenarios = []string{"../examplescenarios", "../examples/llm-d/scenarios", "../testdata/scenarios", "../testdata/scenarios/muse-spark"}

var pdName = regexp.MustCompile(`(\d+)p(\d+)d`)

// pdScenarioFiles returns every file of shippedPDScenarios with a prefill pool.
func pdScenarioFiles(t *testing.T) []string {
	t.Helper()
	var out []string
	for _, dir := range shippedPDScenarios {
		files, err := filepath.Glob(filepath.Join(dir, "*.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range files {
			raw, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(raw), "role: prefill") {
				out = append(out, f)
			}
		}
	}
	return out
}

// pdFlags are the instance counts of a disaggregated file named like 3p1d (1p1d otherwise).
func pdFlags(file string) []string {
	p, d := 1, 1
	if m := pdName.FindStringSubmatch(filepath.Base(file)); m != nil {
		p, _ = strconv.Atoi(m[1])
		d, _ = strconv.Atoi(m[2])
	}
	return []string{"--num-instances", strconv.Itoa(p + d), "--prefill-instances", strconv.Itoa(p), "--decode-instances", strconv.Itoa(d)}
}

// fabricAssumed reports whether the Result's kernel reports carry an assumed
// fabric_bandwidth_efficiency in any pool.
func fabricAssumed(t *testing.T, stdout string) bool {
	t.Helper()
	kernel, _ := resultDoc(t, stdout)["reports"].(map[string]any)["kernel"].(map[string]any)
	for _, pool := range kernel {
		rep, _ := pool.(map[string]any)
		as, _ := rep["assumptions"].([]any)
		for _, a := range as {
			if m, _ := a.(map[string]any); m["Name"] == "fabric_bandwidth_efficiency" {
				return true
			}
		}
	}
	return false
}

// engineRefuses lists the shipped disaggregated scenarios the pinned engine refuses, each with
// the reason it states. The fabric set of these is checked in the file; they are not run.
var engineRefuses = map[string]string{
	// (deepseek-v4.1-flash-h100-1p1d-dspark5-cpu1t.yaml was listed here until blis-vllm#242: the engine
	// offloads only the prefix-cacheable KV cache groups, so it runs.)
	// A prefill data-parallel group that spans nodes: the handoff names no endpoint (blis-substrate#48).
	"minimax-m2.5-gb200-1p1d-dp2.yaml": "spans 2 nodes",
}

func TestShippedPDScenariosStateTheFabricSet(t *testing.T) {
	files := pdScenarioFiles(t)
	if len(files) < 6 {
		t.Fatalf("non-vacuity: found %d disaggregated scenarios, the repository ships at least 6", len(files))
	}
	for _, f := range files {
		t.Run(strings.TrimPrefix(f, "../"), func(t *testing.T) {
			if !strings.Contains(readFile(t, f), "cost-model-fabric") {
				t.Fatalf("%s lists no cost-model-fabric: its handoff would be priced at the nominal rate", f)
			}
			args := append([]string{"run", "--scenarios", filepath.Dir(f), "--scenario", filepath.Base(f),
				"--rate", "5", "--num-requests", "6", "--seed", "3",
				"--speculative-thinking", "on", "--speculative-category", "coding"}, pdFlags(f)...)
			stdout, stderr, err := runKernelCLI(t, args...)
			want, listed := engineRefuses[filepath.Base(f)]
			if listed {
				// The scenario is expected to be refused: it must be, with the stated reason, so a
				// scenario that starts to run (or is refused for another reason) fails here and the
				// entry is removed or corrected.
				if err == nil {
					t.Fatalf("%s is listed as refused by the engine but ran: remove it from engineRefuses", f)
				}
				if !strings.Contains(stderr, want) {
					t.Fatalf("%s is refused, but not for %q:\n%s", f, want, lastLines(stderr, 3))
				}
				return
			}
			if err != nil {
				t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
			}
			if fabricAssumed(t, stdout) {
				t.Errorf("%s runs with an assumed fabric_bandwidth_efficiency: no coefficient of its sets covers its fabric", f)
			}
		})
	}
}

func TestPDScenarioWithoutTheFabricSetIsLabelled(t *testing.T) {
	src := "../examplescenarios/glm-5-h200-3p1d-ib.yaml"
	body := readFile(t, src)
	bare := strings.Replace(body, ", cost-model-fabric]", "]", 1)
	if bare == body {
		t.Fatal("the shipped scenario lists no cost-model-fabric to remove")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "bare.yaml"), []byte(bare), 0o644); err != nil {
		t.Fatal(err)
	}
	// The scenario names its router config by a relative path.
	if err := os.CopyFS(filepath.Join(dir, "router"), os.DirFS("../examplescenarios/router")); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := runKernelCLI(t, append([]string{"run", "--scenarios", dir, "--scenario", "bare.yaml",
		"--rate", "5", "--num-requests", "6", "--seed", "3"}, pdFlags(src)...)...)
	if err != nil {
		t.Fatalf("the engine refused a scenario without cost-model-fabric, so the test's premise (it labels) is wrong: %v\n%s", err, lastLines(stderr, 5))
	}
	if !fabricAssumed(t, stdout) {
		t.Error("a disaggregated scenario without cost-model-fabric ran with no assumed fabric_bandwidth_efficiency in its Result")
	}
}
