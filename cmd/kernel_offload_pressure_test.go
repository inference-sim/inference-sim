package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// writePrefixPressureSpec writes a workload of groups prefix groups, each a 20000-token shared
// prefix, under Poisson arrivals: groups x 20000 tokens is more KV than one glm-5 TP8 H200
// rank holds, so the GPU prefix cache evicts prefixes a later request of the group needs.
func writePrefixPressureSpec(t *testing.T, groups, requests int) string {
	t.Helper()
	var b strings.Builder
	fmt.Fprintf(&b, "version: \"2\"\ncategory: language\naggregate_rate: 4.0\nnum_requests: %d\nclients:\n", requests)
	for g := 0; g < groups; g++ {
		fmt.Fprintf(&b, "  - id: c%d\n    tenant_id: t\n    slo_class: batch\n    rate_fraction: 1.0\n"+
			"    prefix_group: g%d\n    prefix_length: 20000\n    arrival:\n      process: poisson\n"+
			"    input_distribution:\n      type: constant\n      params: { value: 64 }\n"+
			"    output_distribution:\n      type: constant\n      params: { value: 4 }\n", g, g)
	}
	path := filepath.Join(t.TempDir(), "pressure.yaml")
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// cacheHitRate reads the run's "Cache Hit Rate:" line from stdout.
func cacheHitRate(t *testing.T, stdout string) float64 {
	t.Helper()
	const key = "Cache Hit Rate: "
	i := strings.Index(stdout, key)
	if i < 0 {
		t.Fatalf("no %q in stdout:\n%s", key, stdout)
	}
	rest := stdout[i+len(key):]
	v, err := strconv.ParseFloat(strings.TrimSpace(rest[:strings.IndexByte(rest, '\n')]), 64)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// Under prefix pressure an offload tier changes the outcome: prefixes the GPU cache evicted are
// found again on the CPU tier, so the cache hit rate is strictly higher with offload than
// without, and the recovered prefill work makes mean E2E strictly shorter -- colocated and
// disaggregated. The disaggregated leg routes prefill round-robin over twice the groups, so
// every prefill instance (not only the one decode instance) holds more prefixes than its KV
// fits. Without this, an offload wiring that stored or found nothing would pass every
// completion and parity test.
func TestRunCmd_KernelBackend_KVOffloadUnderPrefixPressure(t *testing.T) {
	offload := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 274877906944\n")
	for _, tt := range []struct {
		name             string
		groups, requests int
		args             []string
	}{
		{"colocated", 48, 192, []string{"run", "--scenario", "glm-5-h200-fp8-sglang-tp8.yaml"}},
		{"disaggregated", 96, 384, []string{"run", "--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml",
			"--pd-decider", "always", "--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1",
			"--routing-policy", "round-robin"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			requests := tt.requests
			base := append(tt.args, "--workload-spec", writePrefixPressureSpec(t, tt.groups, requests), "--seed", "3")
			var rates, e2e []float64
			for _, extra := range [][]string{nil, {"--kv-offload-config", offload}} {
				out, stderr, err := runKernelCLI(t, append(base, extra...)...)
				if err != nil {
					t.Fatalf("%v: %v\n%s", extra, err, lastLines(stderr, 3))
				}
				if got := clusterMetricInt(t, out, "completed_requests"); got != requests {
					t.Fatalf("%v: completed %d of %d requests", extra, got, requests)
				}
				rates = append(rates, cacheHitRate(t, out))
				e2e = append(e2e, e2eMean(t, out))
			}
			if rates[1] <= rates[0] {
				t.Errorf("offload left the cache hit rate at %.4f against %.4f without it; under prefix "+
					"pressure the CPU tier must recover evicted prefixes", rates[1], rates[0])
			}
			if e2e[1] >= e2e[0] {
				t.Errorf("offload left mean E2E at %.1f ms against %.1f without it; recovered prefixes "+
					"must save prefill work", e2e[1], e2e[0])
			}
		})
	}
}
