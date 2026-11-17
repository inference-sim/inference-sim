package cmd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// withOffloadTiers lists the registry set that carries the storage tiers' rates, which a
// scenario must name before a weight source can be priced.
var withOffloadTiers = []string{"cost-model-memory]", "cost-model-memory, cost-model-offload-tiers]"}

// runLifecycle runs two H200 and two H100 instances under node pools with the given
// instance_lifecycle block and returns the run's simulated duration, which ends when the last
// instance has loaded (no instance serves before then, so no request completes).
func runLifecycle(t *testing.T, edits []string, lifecycle string) (duration float64, stderr string, err error) {
	t.Helper()
	dir := writeColocatedVariant(t, append(twoChipEdits("h200"), edits...)...)
	bundle := writePoolsBundle(t, [4]string{"fast", "h200", "139.80072021", "32"}, [4]string{"older", "h100", "79.1895752", "20"})
	raw, rerr := os.ReadFile(bundle)
	if rerr != nil {
		t.Fatal(rerr)
	}
	path := filepath.Join(t.TempDir(), "lifecycle.yaml")
	if werr := os.WriteFile(path, append(raw, []byte("instance_lifecycle:\n"+lifecycle)...), 0o644); werr != nil {
		t.Fatal(werr)
	}
	out, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-requests", "8",
		"--rate", "1", "--num-instances", "4", "--policy-config", path)
	if err != nil {
		return 0, stderr, err
	}
	var r struct {
		Window struct {
			DurationS float64 `json:"duration_s"`
		} `json:"window"`
	}
	if jerr := json.Unmarshal([]byte(out), &r); jerr != nil {
		t.Fatal(jerr)
	}
	return r.Window.DurationS, stderr, nil
}

// TestRunCmd_LoadingDelayFollowsTheWeightSource is #1215: with a weight source an instance
// loads in loading_delay plus the time to read its weights from that tier, so a slower tier
// loads longer and the stated overhead adds exactly. Before the change instance_lifecycle had
// no weight_source and every instance loaded in loading_delay alone, whatever its model.
func TestRunCmd_LoadingDelayFollowsTheWeightSource(t *testing.T) {
	fast, stderr, err := runLifecycle(t, withOffloadTiers, "  weight_source: nvme_gen4\n")
	if err != nil {
		t.Fatalf("nvme_gen4: %v\n%s", err, lastLines(stderr, 3))
	}
	slow, stderr, err := runLifecycle(t, withOffloadTiers, "  weight_source: s3\n")
	if err != nil {
		t.Fatalf("s3: %v\n%s", err, lastLines(stderr, 3))
	}
	withOverhead, stderr, err := runLifecycle(t, withOffloadTiers, "  weight_source: nvme_gen4\n  loading_delay: {mean: 5, stddev: 0}\n")
	if err != nil {
		t.Fatalf("overhead: %v\n%s", err, lastLines(stderr, 3))
	}
	if fast <= 1 {
		t.Errorf("loading a 70B model's weights from NVMe took %.3f s; want seconds", fast)
	}
	if slow <= 2*fast {
		t.Errorf("object storage loaded in %.3f s, NVMe in %.3f s; the slower tier must load much longer", slow, fast)
	}
	if math.Abs(withOverhead-(fast+5)) > 1e-3 {
		t.Errorf("with loading_delay 5 s the load took %.4f s, want the weights' %.4f s plus 5", withOverhead, fast)
	}
}

// A weight source the scenario's coefficient sets cannot price for a chip is refused before
// the run, naming the registry entries it needs.
func TestRunCmd_WeightSourceWithoutCoefficientsIsRefused(t *testing.T) {
	_, stderr, err := runLifecycle(t, nil, "  weight_source: nvme_gen4\n")
	if err == nil || !strings.Contains(stderr, "offload_tier_nvme_gen4_read_rate") {
		t.Errorf("want a refusal naming offload_tier_nvme_gen4_read_rate, got err=%v\n%s", err, lastLines(stderr, 3))
	}
}
