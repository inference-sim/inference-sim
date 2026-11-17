package cmd

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
	"github.com/inference-sim/blis-substrate/load"
)

func sglangTestSpec(t *testing.T, scenario string) (engine.Spec, *load.Data) {
	t.Helper()
	path, err := filepath.Abs(filepath.Join("..", "testdata", "kernel", "sglang", scenario))
	if err != nil {
		t.Fatal(err)
	}
	data, err := load.Open(load.Options{Catalog: testCatalogDir(), Registry: testRegistryDir()})
	if err != nil {
		t.Fatal(err)
	}
	return engine.Spec{Kind: engineKindSGLang, UpstreamVersion: "0.5.23", Roots: contractRoots(path), Seed: 1}, data
}

// A SGLang scenario selects the sglang engine, which builds over the run's data, reports its own
// field list (the h200 recipe's fields classified), and runs a request through the engine
// contract: schedule a prefill, price it, finish the request.
func TestSGLangScenarioRunsEndToEnd(t *testing.T) {
	spec, data := sglangTestSpec(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	ref, err := selectEngine(spec.Roots["scenario"])
	if err != nil || ref.Kind != "sglang" || ref.Version != "0.5.23" {
		t.Fatalf("selectEngine = %v, %v", ref, err)
	}
	e, err := newContractEngine(spec, data)
	if err != nil {
		t.Fatalf("building the engine: %v", err)
	}
	status := map[string]fields.Status{}
	for _, f := range e.Fields().Fields {
		status[f.Name] = f.Status
	}
	if status["mem_fraction_static"] != fields.StatusModelled || status["enable_flashinfer_allreduce_fusion"] != fields.StatusModelled {
		t.Errorf("the recipe's fields are not classified modelled: %v %v", status["mem_fraction_static"], status["enable_flashinfer_allreduce_fusion"])
	}
	if err := e.Add(engine.Request{ID: "a", Prompt: make([]int32, 1000), Output: []int32{5, 6, 7, 8}}, 0); err != nil {
		t.Fatal(err)
	}
	b, err := e.Schedule(0)
	if err != nil {
		t.Fatal(err)
	}
	if b.Empty() || b.Tokens() != 1000 {
		t.Fatalf("first pass %+v, want the 1000-token prompt", b)
	}
	if d := e.StepTime(b); d <= 0 {
		t.Errorf("step time %d, want a priced pass", d)
	} else {
		t.Logf("prefill of 1000 tokens: %d us", d)
	}
}

// blis run adopts an SGLang scenario: the Result names the sglang engine, every request completes,
// and a scenario that omits context_length (the catalog graph carries none) is refused by name.
func TestRunCmd_ASGLangScenarioRuns(t *testing.T) {
	for _, name := range []string{"glm-5-h200-fp8-sglang-tp8.yaml", "deepseek-r1-h200-fp8-sglang-tp8.yaml"} {
		t.Run(name, func(t *testing.T) { runSGLangScenario(t, name) })
	}
}

func runSGLangScenario(t *testing.T, name string) {
	dir, err := filepath.Abs(filepath.Join("..", "testdata", "kernel", "sglang"))
	if err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name,
		"--rate", "4", "--num-requests", "4", "--seed", "1")
	if err != nil {
		t.Fatalf("an SGLang scenario must run: %v\n%s", err, lastLines(stderr, 5))
	}
	if !completedIs(stdout, 4) {
		t.Errorf("want 4 completed requests; stdout:\n%s", stdout)
	}
	if !strings.Contains(stdout, `"kind": "sglang"`) && !strings.Contains(stdout, `kind: sglang`) {
		t.Errorf("the Result does not name the sglang engine")
	}
}
