package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// P/D handoff fidelity on the kernel backend (#1922, #1934, #1940): per-link contention, the
// prefill KV hold, and offload tiers under P/D, each through the CLI as a user runs them and
// each reproduced by replay (INV-13).

// pdBurst is a 3P1D GLM-5 burst whose prefills finish in the same steps, so their handoffs
// overlap on the prefill nodes' NIC egress and the decode node's NIC ingress.
var pdTopology = []string{"--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml",
	"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--seed", "1",
	"--horizon", "600000000"}

var pdBurst = append(append([]string(nil), pdTopology...), "--num-requests", "40", "--rate", "500", "--prompt-tokens", "2000", "--prompt-tokens-stdev", "0",
	"--prompt-tokens-min", "2000", "--prompt-tokens-max", "2000", "--output-tokens", "20")

// pdFigure is a value of the Result's breakdowns.pd, and whether the run reports it.
func pdFigure(t *testing.T, stdout, name string) (float64, bool) {
	t.Helper()
	b := parseResult(t, stdout).Breakdowns
	if b == nil {
		return 0, false
	}
	v, ok := b.PD[name]
	return v, ok
}

// With contention on, overlapping handoffs queue on the shared NICs: the run reports a
// positive queueing delay and per-link peaks, the mean handoff lengthens over the uncontended
// run, and a replay with the same flags is byte-identical (INV-13).
func TestRunCmd_KernelBackend_PDTransferContention(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	on, stderr, err := runKernelCLI(t, append([]string{"run", "--pd-transfer-contention", "--trace-output", prefix}, pdBurst...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	off, stderr, err := runKernelCLI(t, append([]string{"run"}, pdBurst...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	for name := range parseResult(t, off).Breakdowns.PD {
		if strings.HasPrefix(name, "kv_transfer_queue_delay") || strings.HasPrefix(name, "link_peak_concurrent_transfers") {
			t.Errorf("without contention the P/D breakdown must not report link sharing: %s", name)
		}
	}
	if delay, ok := pdFigure(t, on, "kv_transfer_queue_delay_mean_ms"); !ok || delay <= 0 {
		t.Errorf("overlapping handoffs report no queueing delay: %v (reported %t)", delay, ok)
	}
	if peak, ok := pdFigure(t, on, "link_peak_concurrent_transfers_nic-ingress-node-3"); !ok || peak <= 1 {
		t.Errorf("the decode node's ingress must carry overlapping handoffs: peak %v (reported %t)", peak, ok)
	}
	onMean, _ := pdFigure(t, on, "kv_transfer_mean_ms")
	offMean, _ := pdFigure(t, off, "kv_transfer_mean_ms")
	if onMean == offMean {
		t.Errorf("contention left every handoff duration unchanged: mean %v ms", onMean)
	}
	clusterConservationHolds(t, on, 40)

	rep, stderr, err := runKernelCLI(t, append([]string{"replay", "--pd-transfer-contention",
		"--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, pdTopology...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
	}
	if inv13View(t, rep) != inv13View(t, on) {
		t.Errorf("the contended run's replay differs\n--- run\n%s\n--- replay\n%s", on, rep)
	}
}

// Every NIXL handoff holds the prefill KV until the pull completes: a kernel P/D run reports
// the hold's peak, and its breakdowns.pd block carries the keys the campaign adapter reads.
func TestRunCmd_KernelBackend_PDBreakdown(t *testing.T) {
	out, stderr, err := runKernelCLI(t, append([]string{"run", "--pd-transfer-contention"}, pdBurst...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	if _, ok := pdFigure(t, out, "prefill_kv_hold_peak_tokens"); !ok {
		t.Errorf("a NIXL P/D run must report its prefill KV hold:\n%s", out)
	}
	for _, k := range []string{"kv_transfers", "kv_transfer_p50_ms", "kv_transfer_p90_ms",
		"kv_transfer_queue_delay_p90_ms", "prefill_kv_hold_peak_tokens", "cpu_tier_hit_rate_prefill"} {
		if _, ok := pdFigure(t, out, k); !ok {
			t.Logf("breakdowns.pd lacks %q (known keys vary with the topology)", k)
		}
	}
	if v, _ := pdFigure(t, out, "kv_transfer_queue_delay_p90_ms"); v <= 0 {
		t.Errorf("kv_transfer_queue_delay_p90_ms = %v, want > 0 under contention", v)
	}
	if v, _ := pdFigure(t, out, "prefill_kv_hold_peak_tokens"); v <= 0 {
		t.Errorf("prefill_kv_hold_peak_tokens = %v, want > 0", v)
	}
}

// Refusals: contention needs a disaggregated run, and a handoff connector other than NIXL's
// is not simulated as NIXL's.
func TestRunCmd_KernelBackend_PDHandoffRefusals(t *testing.T) {
	if _, stderr, err := runKernelCLI(t, "run", "--scenario", "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml",
		"--num-requests", "2", "--rate", "1", "--pd-transfer-contention"); err == nil ||
		!strings.Contains(stderr, "--pd-transfer-contention needs a disaggregated run") {
		t.Errorf("contention on a colocated run must be refused; err=%v\n%s", err, lastLines(stderr, 3))
	}
	dir := t.TempDir()
	body, err := os.ReadFile(filepath.Join(pdScenarios, "glm-5-h200-3p1d-ib.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	mooncake := strings.Replace(string(body), "connector: NixlConnector", "connector: MooncakeConnector", 1)
	if mooncake == string(body) {
		t.Fatal("fixture: the scenario names no NixlConnector to replace")
	}
	if err := os.WriteFile(filepath.Join(dir, "mooncake.yaml"), []byte(mooncake), 0o644); err != nil {
		t.Fatal(err)
	}
	withPDRouter(t, dir)
	if _, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "mooncake.yaml",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--num-requests", "2", "--rate", "1"); err == nil ||
		!strings.Contains(stderr, "BLIS models NIXL's handoff") {
		t.Errorf("a MooncakeConnector handoff must be refused; err=%v\n%s", err, lastLines(stderr, 3))
	}
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
