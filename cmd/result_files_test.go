package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
)

// A Result holds the small files its run read (input.files): the router file the deployment
// names and the workload spec an option names. So --from-result reproduces the run with neither
// passed again: the spec file is deleted and the router file is in no directory the
// reproduction is given, and the new Result equals the original outside runtime and rerun_of.
// A trace or a corpus stays a sha256 only (Principle 6 of blis-substrate#154), so no held file
// is bulk data.
func TestFromResult_NeedsNoInputFilePassedAgain(t *testing.T) {
	dir := t.TempDir()
	spec := filepath.Join(dir, "workload.yaml")
	if err := os.WriteFile(spec, []byte("version: \"2\"\nseed: 7\ncategory: language\naggregate_rate: 4\nnum_requests: 12\n"+
		"clients:\n  - id: c1\n    tenant_id: t1\n    slo_class: standard\n    rate_fraction: 1.0\n"+
		"    arrival: {process: poisson}\n    input_distribution: {type: constant, params: {value: 64}}\n"+
		"    output_distribution: {type: constant, params: {value: 16}}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1",
		"--workload-spec", spec, "--seed", "2")
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	orig := parseResult(t, stdout)
	held := map[string]bool{}
	for _, f := range orig.Input.Files {
		held[f.Path] = true
	}
	if !held[pdAlwaysEPPConfig] || !held["workload-spec"] {
		t.Fatalf("input.files holds %v; want the router file %s and the workload spec", held, pdAlwaysEPPConfig)
	}
	if err := os.Remove(spec); err != nil {
		t.Fatal(err)
	}
	saved := filepath.Join(dir, "result.json")
	if err := os.WriteFile(saved, []byte(stdout), 0o644); err != nil {
		t.Fatal(err)
	}
	// runKernelCLI names its default scenario directory, which holds no router file.
	again, stderr, err := runKernelCLI(t, "run", "--from-result", saved)
	if err != nil {
		t.Fatalf("--from-result with no input file passed again: %v\n%s", err, lastLines(stderr, 3))
	}
	got := parseResult(t, again)
	if got.RerunOf == nil || got.RerunOf.Name != orig.Name {
		t.Fatalf("rerun_of = %+v, want the original %q", got.RerunOf, orig.Name)
	}
	got.RerunOf = nil
	if a, b := withoutRuntime(t, string(mustMarshal(t, orig))), withoutRuntime(t, string(mustMarshal(t, got))); a != b {
		t.Errorf("the reproduction differs from the original:\n%s", firstDiff(a, b))
	}
}

func mustMarshal(t *testing.T, r *result.Result) []byte {
	t.Helper()
	b, err := result.Marshal(r, result.FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
