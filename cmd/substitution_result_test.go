package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A model whose catalog graph holds the text decoder only is priced in the mode language_model_only:
// true whatever the deployment states for a multimodal checkpoint (vLLM keeps the vision tower
// resident otherwise). The engine says so as a machine-readable substitution in its provenance, and
// the Result carries it, under reports.kernel.<pool>.substitutions, so a grade reads it and not the
// stderr text: stating language_model_only: true removes it.
func TestResult_CarriesTheSubstitutionTheEngineRecords(t *testing.T) {
	const dir, scenario = "../testdata/scenarios/muse-spark", "muse-h200-ib-tep8-dspark.yaml"
	run := func(scenarios string) map[string]any {
		t.Helper()
		stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", scenarios, "--scenario", scenario, "--num-requests", "4",
			"--rate", "2", "--speculative-thinking", "on", "--speculative-category", "coding")
		if err != nil {
			t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
		}
		kernel, _ := resultDoc(t, stdout)["reports"].(map[string]any)["kernel"].(map[string]any)
		if len(kernel) == 0 {
			t.Fatal("non-vacuity: the Result reports no kernel provenance")
		}
		return kernel
	}
	for pool, v := range run(dir) {
		subs, _ := v.(map[string]any)["substitutions"].([]any)
		if len(subs) != 1 {
			t.Fatalf("%s: substitutions = %v, want the one language_model_only entry", pool, subs)
		}
		s, _ := subs[0].(map[string]any)
		if s["setting"] != "language_model_only" || s["stated"] != "unset" || s["priced"] != "true" || s["reason"] == "" {
			t.Errorf("%s: substitution = %v, want language_model_only stated unset, priced true, with a reason", pool, s)
		}
	}

	body, err := os.ReadFile(filepath.Join(dir, scenario))
	if err != nil {
		t.Fatal(err)
	}
	stated := strings.Replace(string(body), "    engine:\n", "    engine:\n      language_model_only: true\n", 1)
	if stated == string(body) {
		t.Fatal("the scenario has no engine block to state language_model_only in")
	}
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, scenario), []byte(stated), 0o644); err != nil {
		t.Fatal(err)
	}
	for pool, v := range run(tmp) {
		if subs, ok := v.(map[string]any)["substitutions"]; ok {
			t.Errorf("%s: a deployment that states language_model_only: true still reports substitutions %v", pool, subs)
		}
	}
}
