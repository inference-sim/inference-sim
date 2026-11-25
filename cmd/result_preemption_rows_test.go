package cmd

// A Result row states when the engine preempted its request: preemptions and preempted_at_us.

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// preemptingRun runs the fixture scenario with its KV pool shrunk by lowering
// gpu_memory_utilization, under a burst of long requests that cannot all be held at once, so
// the engine preempts.
func preemptingRun(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(testScenariosDir(t), kernelTestScenario))
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`(?m)^(\s*max_model_len: \d+)$`)
	if !re.Match(raw) {
		t.Fatalf("%s states no max_model_len to put the pool size after", kernelTestScenario)
	}
	body := re.ReplaceAllString(string(raw), "${1}\n      gpu_memory_utilization: 0.17")
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, kernelTestScenario), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	spec := filepath.Join(dir, "spec.yaml")
	if err := os.WriteFile(spec, []byte(`version: "2"
seed: 5
category: language
aggregate_rate: 100
num_requests: 60
clients:
  - {id: a, rate_fraction: 1.0, arrival: {process: poisson},
     input_distribution: {type: constant, params: {value: 8000}},
     output_distribution: {type: constant, params: {value: 8000}}}
`), 0o644); err != nil {
		t.Fatal(err)
	}
	out, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", kernelTestScenario, "--workload-spec", spec)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	return out
}

func TestResult_RowStatesWhenItsRequestWasPreempted(t *testing.T) {
	// GIVEN a run in which the engine preempts requests
	out := preemptingRun(t)
	r := parseResult(t, out)
	total := int(clusterFigures(t, out)["preemption_count"])
	if total == 0 {
		t.Fatal("non-vacuity: the run preempted nothing")
	}
	// THEN the rows' counts sum to the cluster's preemption total (INV-1 style conservation)
	sum := 0
	for _, row := range r.RequestRows {
		sum += row.Preemptions
		// AND each row lists one time per preemption, in order, within the request's life on
		// the instance that served it
		if len(row.PreemptedAtUs) != row.Preemptions {
			t.Errorf("%s: %d preemptions but %d times", row.ID, row.Preemptions, len(row.PreemptedAtUs))
		}
		if row.Preemptions > 0 && row.Instance == "" {
			t.Errorf("%s: preempted but names no instance", row.ID)
		}
		for i, at := range row.PreemptedAtUs {
			if at < row.ArrivalUs || (i > 0 && at < row.PreemptedAtUs[i-1]) {
				t.Errorf("%s: preemption %d at %d (arrival %d, previous %v)", row.ID, i, at, row.ArrivalUs, row.PreemptedAtUs[:i])
			}
			if row.E2EUs != nil && at > row.ArrivalUs+*row.E2EUs {
				t.Errorf("%s: preempted at %d, after it completed at %d", row.ID, at, row.ArrivalUs+*row.E2EUs)
			}
		}
	}
	if sum != total {
		t.Errorf("rows count %d preemptions, the cluster %d", sum, total)
	}
}

func TestResult_RowOfARunWithoutPreemptionStatesNone(t *testing.T) {
	// GIVEN a small run on the fixture scenario, which preempts nothing
	out, stderr, err := runKernelCLI(t, resultRunArgs()...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	if n := clusterFigures(t, out)["preemption_count"]; n != 0 {
		t.Fatalf("non-vacuity: the run preempted %v", n)
	}
	// THEN no row carries the fields, so the document is as before
	for _, row := range parseResult(t, out).RequestRows {
		if row.Preemptions != 0 || row.PreemptedAtUs != nil {
			t.Errorf("%s: states %d preemptions %v", row.ID, row.Preemptions, row.PreemptedAtUs)
		}
	}
}
