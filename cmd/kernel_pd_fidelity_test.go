package cmd

import (
	"encoding/json"
	"fmt"
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
	"--pd-decider", "always", "--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--seed", "1",
	"--horizon", "600000000"}

var pdBurst = append(append([]string(nil), pdTopology...), "--num-requests", "40", "--rate", "500", "--prompt-tokens", "2000", "--prompt-tokens-stdev", "0",
	"--prompt-tokens-min", "2000", "--prompt-tokens-max", "2000", "--output-tokens", "20")

// pdLine returns stdout's line beginning with prefix, or "".
func pdLine(stdout, prefix string) string {
	for _, l := range strings.Split(stdout, "\n") {
		if strings.HasPrefix(strings.TrimSpace(l), prefix) {
			return strings.TrimSpace(l)
		}
	}
	return ""
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
	if strings.Contains(off, "Queueing Delay") || strings.Contains(off, "Peak Concurrent") {
		t.Errorf("without contention the PD section must not report link sharing:\n%s", off)
	}
	delay := pdLine(on, "KV Transfer Queueing Delay")
	if delay == "" || strings.Contains(delay, "mean=0.0 ") {
		t.Errorf("overlapping handoffs report no queueing delay: %q", delay)
	}
	if !strings.Contains(on, "Peak Concurrent Transfers (nic/ingress/node/3): ") ||
		strings.Contains(on, "Peak Concurrent Transfers (nic/ingress/node/3): 1\n") {
		t.Errorf("the decode node's ingress must carry overlapping handoffs:\n%s", on)
	}
	if pdLine(on, "KV Transfer Duration") == pdLine(off, "KV Transfer Duration") {
		t.Errorf("contention left every handoff duration unchanged: %q", pdLine(on, "KV Transfer Duration"))
	}
	clusterConservationHolds(t, on, 40)

	rep, stderr, err := runKernelCLI(t, append([]string{"replay", "--pd-transfer-contention",
		"--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, pdTopology...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
	}
	if rep != on {
		t.Errorf("the contended run's replay differs\n--- run\n%s\n--- replay\n%s", on, rep)
	}
}

// Every NIXL handoff holds the prefill KV until the pull completes: a kernel P/D run reports
// the hold's peak, and the results file carries the pd block with the keys the campaign
// adapter reads.
func TestRunCmd_KernelBackend_PDResultsFileBlock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.json")
	out, stderr, err := runKernelCLI(t, append([]string{"run", "--pd-transfer-contention", "--metrics-path", path}, pdBurst...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	if pdLine(out, "Prefill KV Hold Peak (tokens):") == "" {
		t.Errorf("a NIXL P/D run must report its prefill KV hold:\n%s", out)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		PD map[string]json.RawMessage `json:"pd"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"transfers", "transfer_time_p50_ms", "transfer_time_p90_ms",
		"transfer_queue_delay_p90_ms", "prefill_kv_hold_peak_tokens", "cpu_tier_hit_rate",
		"link_peak_concurrent_transfers"} {
		if _, ok := file.PD[k]; !ok {
			t.Errorf("results file pd block lacks %q: %s", k, raw[:min(len(raw), 400)])
		}
	}
	var got struct {
		Transfers int     `json:"transfers"`
		P50       float64 `json:"transfer_time_p50_ms"`
		DelayP90  float64 `json:"transfer_queue_delay_p90_ms"`
		Hold      int64   `json:"prefill_kv_hold_peak_tokens"`
	}
	if err := json.Unmarshal(mustMarshal(t, file.PD), &got); err != nil {
		t.Fatal(err)
	}
	if got.Transfers != 40 {
		t.Errorf("pd.transfers = %d, want 40", got.Transfers)
	}
	// The file's values are stdout's PD section in milliseconds.
	if want := fmt.Sprintf("p50=%.1f", got.P50*1e3); !strings.Contains(pdLine(out, "KV Transfer Duration"), want) {
		t.Errorf("pd.transfer_time_p50_ms %v disagrees with stdout %q", got.P50, pdLine(out, "KV Transfer Duration"))
	}
	if want := fmt.Sprintf("p90=%.1f", got.DelayP90*1e3); got.DelayP90 <= 0 || !strings.Contains(pdLine(out, "KV Transfer Queueing Delay"), want) {
		t.Errorf("pd.transfer_queue_delay_p90_ms %v: want > 0 and equal to stdout %q", got.DelayP90, pdLine(out, "KV Transfer Queueing Delay"))
	}
	if want := fmt.Sprintf("Prefill KV Hold Peak (tokens): %d", got.Hold); got.Hold <= 0 || pdLine(out, "Prefill KV Hold Peak") != want {
		t.Errorf("pd.prefill_kv_hold_peak_tokens %d: want > 0 and equal to stdout %q", got.Hold, pdLine(out, "Prefill KV Hold Peak"))
	}
	if strings.Contains(out, `"pd"`) {
		t.Errorf("the pd block is file-only; stdout must not carry it")
	}
}

// A CPU tier per pool (MultiConnector of NIXL and the OffloadingConnector) runs under P/D,
// with and without contention, conserves every request and replays byte-identically. A
// 0-byte tier is refused, as vLLM refuses it (vllm/v1/kv_offload/cpu/spec.py:97-100 at
// v0.31.0). (Which tier serves what is pinned in sim/cluster's pd_decode_load_test.go and
// pd_offload_test.go; this is the CLI path.)
func TestRunCmd_KernelBackend_PDWithOffloadTiers(t *testing.T) {
	tier := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 1099511627776\n")
	workload := []string{"--num-requests", "30", "--rate", "8", "--prefix-tokens", "1024"}
	for _, extra := range [][]string{nil, {"--pd-transfer-contention"}} {
		args := append(append([]string{"--kv-offload-config", tier}, pdTopology...), extra...)
		prefix := filepath.Join(t.TempDir(), "trace")
		out, stderr, err := runKernelCLI(t, append(append([]string{"run", "--trace-output", prefix}, workload...), args...)...)
		if err != nil {
			t.Fatalf("run %v: %v\n%s", extra, err, lastLines(stderr, 3))
		}
		clusterConservationHolds(t, out, 30)
		if !strings.Contains(out, "Disaggregated Requests: 30") {
			t.Errorf("run %v: not every request was disaggregated:\n%s", extra, out)
		}
		rep, stderr, err := runKernelCLI(t, append([]string{"replay",
			"--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, args...)...)
		if err != nil {
			t.Fatalf("replay %v: %v\n%s", extra, err, lastLines(stderr, 3))
		}
		if rep != out {
			t.Errorf("the offloaded P/D run's replay differs (%v)\n--- run\n%s\n--- replay\n%s", extra, out, rep)
		}
	}
	zero := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 0\n")
	if _, stderr, err := runKernelCLI(t, append(append([]string{"run", "--kv-offload-config", zero}, workload...), pdTopology...)...); err == nil ||
		!strings.Contains(stderr, "cpu_bytes_to_use must be > 0") {
		t.Errorf("a 0-byte CPU tier must be refused, as vLLM refuses it; err=%v\n%s", err, lastLines(stderr, 3))
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
	if _, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "mooncake.yaml", "--pd-decider", "always",
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
