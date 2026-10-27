package cmd

import (
	"path/filepath"
	"testing"
)

// The formerly refused guide configuration preserves content-sensitive routing
// when workload and router inputs are exported and replayed by the CLI.
func TestINV13_RunReplayParity_DefaultAffinity(t *testing.T) {
	dir := colocatedRouterVariant(t, "router:\n  endpoint_picker_config: affinity.yaml\n", "", map[string]string{"affinity.yaml": "../sim/llmd/testdata/llm-d-guides/optimized-baseline.yaml"})
	prefix := filepath.Join(t.TempDir(), "trace")
	const horizon = "9223372036854775807"
	run, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "4", "--workload-spec", "../examples/prefix-affinity-demo.yaml", "--seed", "1", "--horizon", horizon, "--trace-output", prefix)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 10))
	}
	replay, stderr, err := runKernelCLI(t, "replay", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "4", "--trace-header", prefix+".yaml", "--trace-data", prefix+".csv", "--horizon", horizon)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 10))
	}
	if run != replay {
		t.Fatal("default affinity: run/replay output differs")
	}
}
