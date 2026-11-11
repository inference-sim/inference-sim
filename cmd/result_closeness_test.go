package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// The Result carries what the dashboard needs for its G1 conditions (design 11.3, COV-4): per
// selected coefficient, the kernel implementations, the fit range and the extrapolated and
// unidentified flags; and the engine's resolution of each setting. These tests run the real
// command on the committed 4-node scenario, whose h200 fp8 GEMM envelope (gemm_eps_max_fp8,
// gemm_m_half_fp8) prices every step.

const closenessScenario = "llama-3.1-70b-instruct-h200-tp4-4node.yaml"

func closenessArgs(extra ...string) []string {
	return runArgs(append([]string{"--scenarios", pdScenarios, "--scenario", closenessScenario,
		"--num-instances", "1", "--seed", "3"}, burst(20, 10, 512, 64)...), extra...)
}

// primitivesOverlay is a registry overlay that replaces two h200 entries of the committed
// cost-model-primitives set: gemm_eps_max_fp8 gains a fit range of one token (every priced step
// runs more) and a kernel implementation, and gemm_m_half_fp8 gains a 95% interval far wider
// than its value, so it is unidentified.
func primitivesOverlay(t *testing.T) string {
	t.Helper()
	return primitivesOverlayWithFit(t, "      fit_range: [{dim: tokens, low: 1, high: 1}]\n      kernel_impls: [{library: vllm, variant: linear_fp8, code_hash: 0123456789abcdef}]\n")
}

// primitivesOverlayWithFit is primitivesOverlay with the lines added to gemm_eps_max_fp8 given.
func primitivesOverlayWithFit(t *testing.T, epsLines string) string {
	t.Helper()
	src, err := os.ReadFile(filepath.Join(testRegistryDir(), "coefficients", "cost-model-primitives.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	edit := func(old, add string) {
		t.Helper()
		if strings.Count(body, old) != 1 {
			t.Fatalf("the committed set has %d copies of %q; the overlay needs exactly one", strings.Count(body, old), old)
		}
		body = strings.Replace(body, old, old+add, 1)
	}
	edit("      value: 0.736\n      units: dimensionless\n      method: measured\n      fitted: true\n      scope: {hardware: [h200]}\n",
		epsLines)
	edit("      value: 113\n      units: tokens\n      method: measured\n      fitted: true\n      scope: {hardware: [h200]}\n",
		"      ci95: [0, 1000]\n")
	dir := filepath.Join(t.TempDir(), "overlay")
	if err := os.MkdirAll(filepath.Join(dir, "coefficients"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "coefficients", "cost-model-primitives.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// selectedByName reads the Result's selected coefficients from a command's stdout, by name.
func selectedByName(t *testing.T, stdout string) map[string]map[string]any {
	t.Helper()
	doc := resultDoc(t, stdout)
	entries, _ := doc["selected_entries"].(map[string]any)["coefficients"].([]any)
	out := map[string]map[string]any{}
	for _, e := range entries {
		m := e.(map[string]any)
		if m["scope"] != nil && !strings.Contains(strings.ReplaceAll(toJSON(m["scope"]), " ", ""), "h200") {
			continue
		}
		out[m["name"].(string)] = m
	}
	return out
}

// An entry its data do not determine is reported unidentified, and an entry that states them
// carries its kernel implementation and fit range; entries that state none report none, and omit
// the flags. (Extrapolation is a fact about priced steps: the instance and pool tests below
// show that it holds across instances and pools.)
func TestResult_ReportsExtrapolatedUnidentifiedKernelImplAndFitRange(t *testing.T) {
	over := primitivesOverlay(t)
	stdout, stderr, err := runKernelCLI(t, closenessArgs("--registry-overlay", over)...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	got := selectedByName(t, stdout)

	eps, ok := got["gemm_eps_max_fp8"]
	if !ok {
		t.Fatalf("the run selected no gemm_eps_max_fp8 for h200: %d entries", len(got))
	}
	if eps["extrapolated"] != true {
		t.Errorf("gemm_eps_max_fp8 priced steps above its 1-token fit range and is not reported extrapolated: %v", eps)
	}
	if half := got["gemm_m_half_fp8"]; half["extrapolated"] != nil {
		t.Errorf("gemm_m_half_fp8 states no fit range and is reported extrapolated")
	}
	if !reflect.DeepEqual(eps["fit_range"], []any{map[string]any{"dim": "tokens", "low": 1.0, "high": 1.0}}) {
		t.Errorf("fit_range = %v", eps["fit_range"])
	}
	if !reflect.DeepEqual(eps["kernel_impls"], []any{map[string]any{"library": "vllm", "variant": "linear_fp8", "code_hash": "0123456789abcdef"}}) {
		t.Errorf("kernel_impls = %v", eps["kernel_impls"])
	}
	if eps["unidentified"] != nil {
		t.Errorf("gemm_eps_max_fp8 has no interval and is reported unidentified")
	}

	half, ok := got["gemm_m_half_fp8"]
	if !ok || half["unidentified"] != true {
		t.Errorf("gemm_m_half_fp8 has a 95%% interval of [0, 1000] around 113 and is not reported unidentified: %v", half)
	}
	if half["extrapolated"] != nil {
		t.Errorf("gemm_m_half_fp8 states no fit range and is reported extrapolated")
	}

	if got, _ := resultDoc(t, stdout)["selected_entries"].(map[string]any)["closeness_version"].(float64); got != 1 {
		t.Errorf("closeness_version = %v, want 1: every coefficient was recorded whole", got)
	}

	// Every other selected coefficient states none of the four and reports none.
	for name, e := range got {
		if name == "gemm_eps_max_fp8" || name == "gemm_m_half_fp8" {
			continue
		}
		for _, k := range []string{"kernel_impls", "fit_range", "extrapolated", "unidentified"} {
			if e[k] != nil {
				t.Errorf("%s reports %s = %v though its entry states none", name, k, e[k])
			}
		}
	}
}

// The same run on the committed registry reports no coefficient extrapolated or unidentified:
// the flags come from the entries, not from every run.
func TestResult_CommittedRegistryReportsNoExtrapolation(t *testing.T) {
	stdout, stderr, err := runKernelCLI(t, closenessArgs()...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	got := selectedByName(t, stdout)
	if len(got) < 10 {
		t.Fatalf("non-vacuity: %d selected coefficients", len(got))
	}
	if v, _ := resultDoc(t, stdout)["selected_entries"].(map[string]any)["closeness_version"].(float64); v != 1 {
		t.Errorf("closeness_version = %v, want 1: coefficients were recorded and every one whole", v)
	}
	for name, e := range got {
		if e["extrapolated"] != nil || e["unidentified"] != nil {
			t.Errorf("%s reported %v / %v on the committed registry", name, e["extrapolated"], e["unidentified"])
		}
	}
}

// The Result's resolutions are the engine's: those the substrate derives from the kernel's own
// Resolved(), for the scenario's tp4 pool, labelled with the pool, and the document passes the
// validator (which checks each pool against the deployment's).
func TestResult_ResolutionsAreTheEngines(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	saved := [4]string{kernelScenario, kernelScenarioDir, catalogPath, registryPath}
	savedOpened, savedFor := openedData, openedFor
	t.Cleanup(func() {
		kernelScenario, kernelScenarioDir, catalogPath, registryPath = saved[0], saved[1], saved[2], saved[3]
		openedData, openedFor = savedOpened, savedFor
	})
	catalogPath, registryPath, kernelScenario, kernelScenarioDir = catalog, registry, closenessScenario, pdScenarios
	m, err := openModel()
	if err != nil {
		t.Fatal(err)
	}
	want := result.ResolutionsOfPool(m.Kernel().Resolved(), "colocated")
	if len(want) < 5 {
		t.Fatalf("non-vacuity: the engine resolves %d settings", len(want))
	}

	stdout, stderr, err := runKernelCLI(t, closenessArgs()...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	rows, _ := resultDoc(t, stdout)["resolutions"].([]any)
	if len(rows) != len(want) {
		t.Fatalf("the Result has %d resolutions, the engine resolves %d", len(rows), len(want))
	}
	for i, w := range want {
		g := rows[i].(map[string]any)
		if g["setting"] != w.Setting || g["engine"] != w.Engine || g["pool"] != "colocated" {
			t.Errorf("resolution %d is %v, want %s = %s in pool colocated", i, g, w.Setting, w.Engine)
		}
	}
	for _, w := range want {
		if w.Setting == "tensor_parallel_width" && w.Engine != "4" {
			t.Errorf("tensor_parallel_width resolves to %q for a tp4 scenario", w.Engine)
		}
	}
	validResult(t, stdout)
}

// validResult fails unless the substrate's validator accepts the Result document.
func validResult(t *testing.T, stdout string) {
	t.Helper()
	file := filepath.Join(t.TempDir(), "result.json")
	if err := os.WriteFile(file, []byte(stdout), 0o644); err != nil {
		t.Fatal(err)
	}
	if findings, err := validators["result"](file); err != nil || len(findings) != 0 {
		t.Errorf("the substrate validator refuses the Result: %v %v", err, findings)
	}
}

// A disaggregated run states each pool's resolutions under its role, sorted by setting, then
// pool, as the validator requires.
func TestResult_DisaggregatedResolutionsCarryEachPool(t *testing.T) {
	args := runArgs(append([]string{"--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--seed", "11",
		"--horizon", "900000000"}, burst(12, 8, 1500, 64)...))
	stdout, stderr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	rows, _ := resultDoc(t, stdout)["resolutions"].([]any)
	pools := map[string]int{}
	var prev [2]string
	for _, r := range rows {
		m := r.(map[string]any)
		k := [2]string{m["setting"].(string), m["pool"].(string)}
		if k[0] < prev[0] || (k[0] == prev[0] && k[1] <= prev[1]) {
			t.Errorf("%v follows %v: resolutions are sorted by setting, then pool", k, prev)
		}
		prev = k
		pools[k[1]]++
	}
	if pools["prefill"] == 0 || pools["decode"] == 0 || len(pools) != 2 {
		t.Errorf("rows per pool = %v, want rows for prefill and decode only", pools)
	}
	validResult(t, stdout)
}

// Recording is cumulative across the engines of a run: with three instances each pricing its own
// steps, the flag any raised stays raised and the entry is reported once.
func TestResult_ExtrapolationHoldsAcrossInstances(t *testing.T) {
	over := primitivesOverlay(t)
	args := append(closenessArgs("--registry-overlay", over), "--num-instances", "3")
	stdout, stderr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	got := selectedByName(t, stdout)
	if got["gemm_eps_max_fp8"]["extrapolated"] != true {
		t.Errorf("three instances priced steps above the 1-token fit range and the entry is not extrapolated: %v", got["gemm_eps_max_fp8"])
	}
	validResult(t, stdout)
}

func toJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// Recording is cumulative across the pools of a disaggregated run: each pool's engine registers
// its own kernel with the run's data when it opens, so a step priced only by the decode pool
// still raises the flag. The fit range starts above any decode batch (at most 12 requests, one
// token each) and below every prefill step (1,500-token prompts), so only the decode pool's
// steps are outside it. A run in which the decode pool's kernel is not registered reports the
// entry as inside its range; that run fails this test.
func TestResult_ExtrapolationHoldsAcrossPools(t *testing.T) {
	over := primitivesOverlayWithFit(t, "      fit_range: [{dim: tokens, low: 13, high: 100000000}]\n")
	pd := func(extra ...string) []string {
		return runArgs(append([]string{"--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml",
			"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--seed", "11",
			"--horizon", "900000000"}, burst(12, 8, 1500, 64)...), extra...)
	}
	stdout, stderr, err := runKernelCLI(t, pd("--registry-overlay", over)...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	eps, ok := selectedByName(t, stdout)["gemm_eps_max_fp8"]
	if !ok {
		t.Fatal("the disaggregated run selected no gemm_eps_max_fp8 for h200")
	}
	if eps["extrapolated"] != true {
		t.Errorf("the decode pool priced steps below the fit range's 13 tokens and the entry is not reported extrapolated: %v", eps)
	}
	if v, _ := resultDoc(t, stdout)["selected_entries"].(map[string]any)["closeness_version"].(float64); v != 1 {
		t.Errorf("closeness_version = %v, want 1", v)
	}
	validResult(t, stdout)
}
