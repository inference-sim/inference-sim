package cmd

import (
	"encoding/json"
	"fmt"
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

// primitivesOverlay is a registry overlay for the closeness scenario's model: two h200 entries of
// the committed cost-model-primitives set, copied with a scope narrowed to that model (so they
// outrank the committed ones, which the loader would otherwise find equal in rank). The first,
// gemm_eps_max_fp8, gains a fit range of one token (every priced step runs more); the second,
// gemm_m_half_fp8, a 95% interval far wider than its value, so it is unidentified. Neither names a
// kernel implementation: an entry that lists one is used only when the implementations that run
// are among them (closeness_version 2), which the tests below state on their own.
func primitivesOverlay(t *testing.T) string {
	t.Helper()
	return primitivesOverlayFor(t, "llama-3.1-70b-instruct", "      fit_range: [{dim: tokens, low: 1, high: 1}]\n")
}

// primitivesOverlayFor is primitivesOverlay for a model, with the lines added to
// gemm_eps_max_fp8 given.
func primitivesOverlayFor(t *testing.T, model, epsLines string) string {
	t.Helper()
	src, err := os.ReadFile(filepath.Join(testRegistryDir(), "coefficients", "cost-model-primitives.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	const base = "      scope: {hardware: [h200]}\n"
	narrow := "      scope: {hardware: [h200], model: [" + model + "]}\n"
	edit := func(old, add string) {
		t.Helper()
		if !strings.HasSuffix(old, base) {
			t.Fatalf("anchor %q does not end in the h200 scope", old)
		}
		if strings.Count(body, old) != 1 {
			t.Fatalf("the committed set has %d copies of %q; the overlay needs exactly one", strings.Count(body, old), old)
		}
		body = strings.Replace(body, old, strings.TrimSuffix(old, base)+narrow+add, 1)
	}
	edit("      value: 0.736\n      units: dimensionless\n      method: measured\n      fitted: true\n"+base, epsLines)
	edit("      value: 113\n      units: tokens\n      method: measured\n      fitted: true\n"+base, "      ci95: [0, 1000]\n")
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
	if eps["kernel_impls"] != nil || eps["kernel_match"] != "unkeyed" || eps["primitive"] != "gemm" {
		t.Errorf("gemm_eps_max_fp8 lists no implementation: kernel_impls = %v, kernel_match = %v, primitive = %v; want none, unkeyed and gemm",
			eps["kernel_impls"], eps["kernel_match"], eps["primitive"])
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

	if got, _ := resultDoc(t, stdout)["selected_entries"].(map[string]any)["closeness_version"].(float64); got != 2 {
		t.Errorf("closeness_version = %v, want 2: the layout that states the implementation each coefficient priced", got)
	}

	// Every other selected coefficient states none of the four and reports none.
	for name, e := range got {
		if name == "gemm_eps_max_fp8" || name == "gemm_m_half_fp8" {
			continue
		}
		for _, k := range []string{"kernel_impls", "fit_range", "extrapolated", "unidentified", "priced"} {
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
	if v, _ := resultDoc(t, stdout)["selected_entries"].(map[string]any)["closeness_version"].(float64); v != 2 {
		t.Errorf("closeness_version = %v, want 2: coefficients were recorded and every one whole", v)
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
	over := primitivesOverlayFor(t, "glm-5", "      fit_range: [{dim: tokens, low: 13, high: 100000000}]\n")
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
	if v, _ := resultDoc(t, stdout)["selected_entries"].(map[string]any)["closeness_version"].(float64); v != 2 {
		t.Errorf("closeness_version = %v, want 2", v)
	}
	validResult(t, stdout)
}

// implKey names a kernel implementation of the Result by its three identifying members.
func implKey(m map[string]any) string {
	return fmt.Sprintf("%v/%v/%v", m["library"], m["variant"], m["code_hash"])
}

func implSet(v any) map[string]bool {
	out := map[string]bool{}
	l, _ := v.([]any)
	for _, x := range l {
		m, _ := x.(map[string]any)
		out[implKey(m)] = true
	}
	return out
}

// pricedProblems checks the selected-entry report against closeness version 2's rules: what a
// coefficient priced is a subset of the implementations its primitive ran in the run
// (selected_entries.implementations) and, for an exact entry, of the implementations it lists;
// an exact entry lists some, and an unkeyed one lists none, so an unkeyed lookup is never
// reported as matched. A sibling build the entry is also valid for, which no operation reading
// the entry ran, is never in priced.
func pricedProblems(se map[string]any) []string {
	ran, _ := se["implementations"].(map[string]any)
	var problems []string
	coeffs, _ := se["coefficients"].([]any)
	for _, c := range coeffs {
		e, _ := c.(map[string]any)
		name, _ := e["name"].(string)
		priced, listed := implSet(e["priced"]), implSet(e["kernel_impls"])
		match, _ := e["kernel_match"].(string)
		prim, _ := e["primitive"].(string)
		switch match {
		case "exact":
			if len(listed) == 0 {
				problems = append(problems, name+": exact, but it lists no implementation")
			}
			for k := range priced {
				if !listed[k] {
					problems = append(problems, name+": priced "+k+", which the entry does not list")
				}
			}
		case "unkeyed":
			if len(listed) != 0 {
				problems = append(problems, name+": unkeyed, but it lists implementations")
			}
		default:
			problems = append(problems, fmt.Sprintf("%s: kernel_match = %q, want exact or unkeyed", name, match))
		}
		onPrimitive := implSet(ran[prim])
		for k := range priced {
			if !onPrimitive[k] {
				problems = append(problems, name+": priced "+k+", which the run did not run for primitive "+prim)
			}
		}
	}
	return problems
}

// The checker finds each way a report breaks the rules, so the end-to-end test below cannot pass
// for want of it (X18): a priced sibling that the entry does not list, an exact entry that lists
// nothing, an unkeyed entry that lists some, a priced kernel its primitive never ran.
func TestPricedProblems_FindsEachBrokenRule(t *testing.T) {
	a := map[string]any{"library": "L", "variant": "a", "code_hash": "aa"}
	b := map[string]any{"library": "L", "variant": "b", "code_hash": "bb"}
	report := func(c map[string]any, ran ...any) map[string]any {
		c["name"], c["primitive"] = "x", "gemm"
		return map[string]any{"coefficients": []any{c}, "implementations": map[string]any{"gemm": ran}}
	}
	good := report(map[string]any{"kernel_match": "exact", "kernel_impls": []any{a, b}, "priced": []any{b}}, a, b)
	if p := pricedProblems(good); len(p) != 0 {
		t.Fatalf("a report that prices only B of A and B was refused: %v", p)
	}
	for name, se := range map[string]map[string]any{
		"a priced sibling the entry does not list": report(map[string]any{"kernel_match": "exact", "kernel_impls": []any{b}, "priced": []any{a}}, a, b),
		"an exact entry that lists nothing":        report(map[string]any{"kernel_match": "exact", "priced": []any{b}}, a, b),
		"an unkeyed entry that lists some":         report(map[string]any{"kernel_match": "unkeyed", "kernel_impls": []any{a}}, a, b),
		"a priced kernel the primitive never ran":  report(map[string]any{"kernel_match": "exact", "kernel_impls": []any{a, b}, "priced": []any{b}}, a),
		"no kernel_match":                          report(map[string]any{"priced": []any{b}}, a, b),
	} {
		if p := pricedProblems(se); len(p) == 0 {
			t.Errorf("%s was accepted", name)
		}
	}
}

// Closeness version 2 on a real run: every coefficient of the Result obeys the rules above (priced
// within the builds the primitive ran and within a listed exact entry; an unlisted entry unkeyed).
// The engine at the pin reports no lowered builds, so priced is empty here, which is what the rules
// require; the counterexample that exercises them is TestResult_PricedIsTheBuildOfTheOperationsThatReadTheEntry.
func TestResult_ARealRunObeysClosenessVersion2(t *testing.T) {
	stdout, stderr, err := runKernelCLI(t, closenessArgs()...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	se, _ := resultDoc(t, stdout)["selected_entries"].(map[string]any)
	if n := len(selectedByName(t, stdout)); n < 10 {
		t.Fatalf("non-vacuity: %d selected coefficients", n)
	}
	if p := pricedProblems(se); len(p) != 0 {
		t.Fatalf("the report breaks closeness version 2's rules: %v", p)
	}
}
