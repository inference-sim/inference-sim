package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
	"github.com/inference-sim/inference-sim/substrate/load"
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

// sglangScenarioCopy copies the SGLang scenario name into a fresh directory with edit applied to
// its text, and returns the directory.
func sglangScenarioCopy(t *testing.T, name string, edit func(string) string) string {
	t.Helper()
	src, err := os.ReadFile(filepath.Join("..", "testdata", "kernel", "sglang", name))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(edit(string(src))), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func sglangResult(t *testing.T, stdout string) map[string]any {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal([]byte(stdout), &doc); err != nil {
		t.Fatalf("the Result is not JSON: %v", err)
	}
	return doc
}

// A run of an SGLang scenario, and the replay of its exported trace, agree (INV-13), and the
// Result names the SGLang engine release, its module, SGLang's own kernel evidence, and the one
// gap that remains between the run and its scenario as a named substitution.
func TestSGLangRunAndReplayAreOneResult(t *testing.T) {
	const name = "deepseek-r1-h200-fp8-sglang-tp8.yaml"
	dir, err := filepath.Abs(filepath.Join("..", "testdata", "kernel", "sglang"))
	if err != nil {
		t.Fatal(err)
	}
	common := []string{"--scenarios", dir, "--scenario", name, "--seed", "1", "--horizon", "600000000"}
	prefix := filepath.Join(t.TempDir(), "trace")
	runOut, stderr, err := runKernelCLI(t, append([]string{"run", "--trace-output", prefix, "--rate", "4", "--num-requests", "6"}, common...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	repOut, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	if a, b := inv13View(t, runOut), inv13View(t, repOut); a != b {
		t.Errorf("INV-13: the replay's Result differs from the run's")
	}
	for leg, out := range map[string]string{"run": runOut, "replay": repOut} {
		doc := sglangResult(t, out)
		eng, _ := doc["engine"].(map[string]any)
		if eng["kind"] != "sglang" || eng["version"] != "0.5.23" {
			t.Errorf("%s: Result engine = %v, want sglang 0.5.23", leg, eng)
		}
		reports, _ := doc["reports"].(map[string]any)
		kern, _ := reports["kernel"].(map[string]any)
		pool, _ := kern["colocated"].(map[string]any)
		subs, _ := pool["substitutions"].([]any)
		var named []string
		for _, s := range subs {
			named = append(named, fmt.Sprint(s.(map[string]any)["setting"]))
		}
		if !slices.Contains(named, "host_side_facts") {
			t.Errorf("%s: the vLLM-form twin is not named a substitution; substitutions: %v", leg, named)
		}
		raw, _ := json.Marshal(pool["assumptions"])
		if strings.Contains(string(raw), "vLLM v0.31.0's default (vllm/config") {
			t.Errorf("%s: the kernel evidence is the vLLM twin's, not SGLang's", leg)
		}
	}
}

// An SGLang pool's engine setting that SGLang's rules do not model is refused by name, and the
// run does not start.
func TestSGLangUnsupportedFlagIsRefusedByName(t *testing.T) {
	const name = "glm-5-h200-fp8-sglang-tp8.yaml"
	dir := sglangScenarioCopy(t, name, func(s string) string {
		return s + "      enable_hierarchical_cache_v9: true\n"
	})
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name, "--rate", "4", "--num-requests", "2", "--seed", "1")
	if err == nil {
		t.Fatal("a run with an unknown SGLang engine setting must be refused")
	}
	if !strings.Contains(stderr, "enable_hierarchical_cache_v9") {
		t.Errorf("the refusal does not name the setting:\n%s", lastLines(stderr, 5))
	}
}

// stream_interval is read by the host (the tokens the API server holds before streaming them), so
// an SGLang pool's value must reach the run: a pool that states 10 and one that states nothing
// deliver tokens in different chunks.
func TestSGLangStreamIntervalReachesTheRun(t *testing.T) {
	const name = "deepseek-r1-h200-fp8-sglang-tp8.yaml"
	icl := func(edit func(string) string) float64 {
		dir := sglangScenarioCopy(t, name, edit)
		out, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name, "--rate", "4", "--num-requests", "6", "--seed", "1")
		if err != nil {
			t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
		}
		m, _ := sglangResult(t, out)["metrics"].(map[string]any)
		v, _ := m["icl_mean_ms"].(map[string]any)
		f, _ := v["value"].(float64)
		return f
	}
	stated := icl(func(s string) string { return s })
	if !strings.Contains(sglangScenarioText(t, name), "stream_interval: 10") {
		t.Fatal("the fixture no longer states stream_interval: 10")
	}
	unstated := icl(func(s string) string { return strings.ReplaceAll(s, "stream_interval: 10", "") })
	if stated == unstated {
		t.Errorf("mean inter-chunk latency is %v with stream_interval 10 and without it: the pool's value never reached the host", stated)
	}
}

func sglangScenarioText(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "kernel", "sglang", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// A Result's substrate and engine_module revisions are the simulator's own build for either
// engine: both engines are packages of its module; the router fork is the only other revision.
func TestResultRevisionsAreOneModuleAndTheRouterFork(t *testing.T) {
	saved := scenarioEngine
	defer func() { scenarioEngine = saved }()
	info := &debug.BuildInfo{
		Main: debug.Module{Version: "v0.19.0"},
		Deps: []*debug.Module{{Path: "github.com/llm-d/llm-d-router", Version: "v1.0.0",
			Replace: &debug.Module{Path: "github.com/inference-sim/llm-d-router", Version: "v1.0.0-blis.5"}}},
	}
	for _, kind := range []string{"vllm", "sglang"} {
		scenarioEngine = engineRef{Kind: kind}
		got := revisionsFrom(info)
		if got.Simulator != "v0.19.0" || got.Substrate != "v0.19.0" || got.EngineModule != "v0.19.0" || got.Router != "v1.0.0-blis.5" {
			t.Errorf("%s run revisions = %+v", kind, got)
		}
	}
}

// A file the scenario names relative to its own directory (here the router's endpoint-picker
// configuration, as the accuracy harness writes it beside every point's scenario) is found by an
// SGLang run, although the host adopts the scenario through a twin in another directory.
func TestSGLangScenarioFindsItsCompanionFiles(t *testing.T) {
	const name = "glm-5-h200-fp8-sglang-tp8.yaml"
	dir := sglangScenarioCopy(t, name, func(s string) string {
		return s + "\nrouter:\n  endpoint_picker_config: lb.epp.yaml\n"
	})
	// The router block belongs to the Deployment document, which is the last one.
	epp := `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: inflight-load-producer
- type: active-request-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: active-request-scorer
    weight: 1
  - pluginRef: max-score-picker
`
	if err := os.WriteFile(filepath.Join(dir, "lb.epp.yaml"), []byte(epp), 0o644); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name, "--rate", "4", "--num-requests", "4", "--seed", "1")
	if err != nil {
		t.Fatalf("an SGLang scenario with a router file beside it must run: %v\n%s", err, lastLines(stderr, 3))
	}
	if !completedIs(stdout, 4) {
		t.Errorf("want 4 completed requests")
	}
}

// The registered SGLang engine prices a batch of decode requests as decode (the kernel's
// DecodeThreshold 1), not as the prefill path. 16 requests of 4096 computed tokens, one token each,
// on GLM-5 H200 TP8 cost exactly what kernel.StepTime charges the same batch with
// DecodeThreshold 1 (25145 us in this scenario at the first decode step); priced as the prefill path the same batch costs 25503 us. A regression pin
// for blis-sglang#96's DecodeThreshold fix: a pin without it fails this test.
func TestSGLangEnginePricesDecodeAsDecode(t *testing.T) {
	spec, data := sglangTestSpec(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	sglangKernels.reset()
	e, err := newContractEngine(spec, data)
	if err != nil {
		t.Fatal(err)
	}
	const n, prompt = 16, 4096
	for i := 0; i < n; i++ {
		out := make([]int32, 64)
		for j := range out {
			out[j] = int32(5 + j)
		}
		if err := e.Add(engine.Request{ID: fmt.Sprintf("r%d", i), Prompt: make([]int32, prompt), Output: out}, 0); err != nil {
			t.Fatal(err)
		}
	}
	var decode engine.Batch
	now := int64(0)
	for step := 0; step < 200 && decode.Empty(); step++ {
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			t.Fatalf("step %d: nothing scheduled before all %d requests decoded", step, n)
		}
		now += e.StepTime(b)
		if _, err := e.Complete(b.ID, engine.Outcome{}, now); err != nil {
			t.Fatal(err)
		}
		if len(b.Reqs) == n {
			all := true
			for _, r := range b.Reqs {
				all = all && r.Tokens == 1 && r.Computed >= prompt
			}
			if all {
				decode = b
			}
		}
	}
	if decode.Empty() {
		t.Fatalf("no batch of %d single-token decodes within 200 steps", n)
	}
	k := sglangKernels.get(0)
	if k == nil {
		t.Fatal("the engine opened no kernel")
	}
	shape := make([]kernel.ReqShape, len(decode.Reqs))
	for i, r := range decode.Reqs {
		shape[i] = kernel.ReqShape{Scheduled: r.Tokens, Computed: r.Computed, PromptLen: r.PromptLen}
	}
	want := k.StepTime(kernel.Batch{Reqs: shape, DecodeThreshold: 1}).NoOverlap.Microseconds()
	asPrefill := k.StepTime(kernel.Batch{Reqs: shape}).NoOverlap.Microseconds()
	if want == asPrefill {
		t.Fatalf("the kernel prices decode and prefill alike (%d us): the test would pass without the fix", want)
	}
	got := e.StepTime(decode)
	if got != want {
		t.Errorf("the engine charges a %d-request decode batch %d us; the kernel charges %d us as decode (%d us as prefill)",
			n, got, want, asPrefill)
	}
	t.Logf("decode batch: engine %d us, kernel decode %d us, kernel prefill path %d us", got, want, asPrefill)
}

// The DSA attention backends the lowering resolved (flashmla_sparse for prefill and fa3 for decode on an H200
// with a bf16 cache) are priced as the substrate's part-wide sparse-MLA attention pair, not as those kernels.
// The Result names each as a machine-readable substitution on the point, where a grade reads it, with the
// backend as the stated value; a DeepSeek-R1 pool, which has no DSA layer, states none.
func TestSGLangResultNamesTheDSAAttentionBackendsAsSubstitutions(t *testing.T) {
	subs := func(scenario string) map[string]map[string]any {
		dir, err := filepath.Abs(filepath.Join("..", "testdata", "kernel", "sglang"))
		if err != nil {
			t.Fatal(err)
		}
		out, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", scenario, "--rate", "4", "--num-requests", "4", "--seed", "1")
		if err != nil {
			t.Fatalf("%s: %v\n%s", scenario, err, lastLines(stderr, 5))
		}
		doc := sglangResult(t, out)
		reports, _ := doc["reports"].(map[string]any)
		kern, _ := reports["kernel"].(map[string]any)
		pool, _ := kern["colocated"].(map[string]any)
		list, _ := pool["substitutions"].([]any)
		by := map[string]map[string]any{}
		for _, s := range list {
			m := s.(map[string]any)
			by[fmt.Sprint(m["setting"])] = m
		}
		return by
	}
	glm := subs("glm-5-h200-fp8-sglang-tp8.yaml")
	dec, pre := glm["dsa_decode_backend"], glm["dsa_prefill_backend"]
	if dec == nil || dec["stated"] != "fa3" || !strings.Contains(fmt.Sprint(dec["priced"]), "sparse-MLA attention pair") {
		t.Errorf("the decode backend fa3 is not a substitution on the Result: %v", dec)
	}
	if pre == nil || pre["stated"] != "flashmla_sparse" {
		t.Errorf("the prefill backend flashmla_sparse is not a substitution on the Result: %v", pre)
	}
	if glm["host_side_facts"] == nil {
		t.Error("the twin's substitution was lost")
	}
	r1 := subs("deepseek-r1-h200-fp8-sglang-tp8.yaml")
	if r1["dsa_decode_backend"] != nil || r1["dsa_prefill_backend"] != nil {
		t.Errorf("a pool with no DSA layer states DSA backends: %v", r1)
	}
}
