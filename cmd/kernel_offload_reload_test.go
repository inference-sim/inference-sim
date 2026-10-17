package cmd

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// catalogWithCPUDRAMRead copies the vendored catalog and rewrites cpu_dram's read bandwidth
// (the CPU->GPU direction of a reload), so a run can be priced against a slower host tier.
func catalogWithCPUDRAMRead(t *testing.T, mbps string) string {
	t.Helper()
	_, catalog, _ := kernelRepos(t)
	dst := filepath.Join(t.TempDir(), "catalog")
	if err := os.CopyFS(dst, os.DirFS(catalog)); err != nil {
		t.Fatal(err)
	}
	path := catalogStorageDevicesPath(dst)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	const old = "  read_bandwidth_mb_s: 27058.0\n  write_bandwidth_mb_s: 46089.0\n  base_latency_us: 19.3\n"
	body := string(raw)
	if !strings.Contains(body, old) {
		t.Fatalf("the vendored catalog's cpu_dram entry changed; update this fixture:\n%s", body)
	}
	body = strings.Replace(body, old, strings.Replace(old, "27058.0", mbps, 1), 1)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dst
}

// runKernelCLIOnCatalog is runKernelCLI against the catalog at catalog.
func runKernelCLIOnCatalog(t *testing.T, catalog string, args ...string) (stdout, stderr string, err error) {
	t.Helper()
	scenarios, _, registry := kernelRepos(t)
	full := append(args, "--catalog", catalog, "--registry", registry,
		"--defaults-filepath", "../defaults.yaml", "--scenarios", scenarios)
	cmd := exec.Command(os.Args[0], "-test.run=^TestKernelCLILeg$")
	cmd.Env = append(os.Environ(), kernelCLIArgsEnv+"="+strings.Join(full, "\x1f"))
	var out, errBuf strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &errBuf
	err = cmd.Run()
	return out.String(), errBuf.String(), err
}

// INV-K7 at the CLI: a CPU-tier hit is loaded over the host link the catalog's cpu_dram
// prices, so halving that bandwidth strictly lengthens mean TTFT at a non-zero CPU-tier hit
// rate. Before the fix the reload was a free same-step hit and the two runs were identical
// (validation gap re-check: cpu_dram / 1e4 left c512 byte-identical at 38.6% CPU hits).
func TestRunCmd_KernelBackend_CPUTierReloadIsPricedByTheHostTier(t *testing.T) {
	offload := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 274877906944\n")
	args := []string{"run", "--scenario", "glm-5-h200-fp8-sglang-tp8.yaml",
		"--workload-spec", writePrefixPressureSpec(t, 48, 192), "--seed", "3", "--kv-offload-config", offload}
	ttft := map[string]float64{}
	for _, bw := range []string{"27058.0", "13529.0"} {
		path := filepath.Join(t.TempDir(), "metrics.json")
		_, stderr, err := runKernelCLIOnCatalog(t, catalogWithCPUDRAMRead(t, bw), append(args, "--metrics-path", path)...)
		if err != nil {
			t.Fatalf("cpu_dram %s: %v\n%s", bw, err, lastLines(stderr, 3))
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(readFile(t, path)), &m); err != nil {
			t.Fatal(err)
		}
		if cpu, _ := m["cpu_tier_hit_rate"].(float64); cpu <= 0 {
			t.Fatalf("cpu_dram %s: no CPU-tier hit (cpu_tier_hit_rate %v); the comparison is vacuous", bw, m["cpu_tier_hit_rate"])
		}
		ttft[bw] = m["ttft_mean_ms"].(float64)
	}
	if ttft["13529.0"] <= ttft["27058.0"] {
		t.Errorf("halving cpu_dram's read bandwidth left mean TTFT at %.3f ms against %.3f; a CPU-tier "+
			"reload must cost its transfer", ttft["13529.0"], ttft["27058.0"])
	}
}
