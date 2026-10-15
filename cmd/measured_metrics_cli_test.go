package cmd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"
)

// dsparkScenario drafts K=5 DSpark tokens on DeepSeek-V4.1-Flash.
const dsparkScenario = "deepseek-v4.1-flash-h100-tp8-dspark5.yaml"

// contractKeys are the cluster-aggregate keys docs/BLIS_CLI_CONTRACT.md promises at the top
// level of the --metrics-path file (those this metrics work owns).
var contractKeys = []string{
	"ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
	"request_itl_mean_ms", "request_itl_p50_ms", "request_itl_p90_ms", "request_itl_p99_ms",
	"interactivity_p50", "interactivity_p90", "output_tok_per_user_mean",
	"e2e_mean_ms", "e2e_p50_ms", "e2e_p90_ms", "e2e_p99_ms",
	"step_interval_mean_ms", "step_interval_p90_ms",
	"output_throughput_tok_s", "measured_window_s",
	"measured_completed_requests", "cancelled_requests", "context_overflow_requests", "failed_requests",
	"cache_hit_rate", "cpu_tier_hit_rate", "preemption_count",
	"gateway_queue_delay_mean_ms", "gateway_queue_delay_p90_ms",
	"in_flight_requests_mean", "in_flight_requests_max", "kv_capacity_tokens_per_replica",
	"kernel_provenance", "blis_version", "kernel_version", "schemas_version",
	"percentile_method",
}

// Under DSpark K=5 a verify step advances AL = 1 + alpha*K tokens on average, so aiperf's
// per-token request ITL is the per-step interval divided by AL (#1938 acceptance). The run is
// light enough that step times barely vary, which is what lets the two p90s be compared.
// The same run pins that the file carries every contract key at top level and stdout none of
// the new ones (INV-6).
func TestMetricsPath_DSparkRequestITLIsTheStepIntervalOverTheAcceptedLength(t *testing.T) {
	const alpha, k = 0.502, 5
	acceptedLength := 1 + alpha*k
	path := filepath.Join(t.TempDir(), "metrics.json")
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios, "--scenario", dsparkScenario,
		"--num-requests", "40", "--rate", "0.5", "--seed", "42",
		"--speculative-acceptance-rate", "0.502", "--metrics-path", path)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var file map[string]any
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatal(err)
	}
	for _, key := range contractKeys {
		if _, ok := file[key]; !ok {
			t.Errorf("--metrics-path file lacks contract key %q", key)
		}
	}
	for _, key := range []string{"request_itl_p90_ms", "step_interval_p90_ms", "kernel_provenance", "blis_version"} {
		if strings.Contains(stdout, `"`+key+`"`) {
			t.Errorf("stdout carries the file-only key %q (INV-6)", key)
		}
	}
	if got := file["measured_completed_requests"].(float64); got != 40 {
		t.Fatalf("%v of 40 requests completed in the measured window; the comparison would be vacuous", got)
	}

	itl, step := file["request_itl_p90_ms"].(float64), file["step_interval_p90_ms"].(float64)
	if want := step / acceptedLength; math.Abs(itl-want)/want > 0.10 {
		t.Errorf("request_itl_p90_ms = %.4f, want step_interval_p90_ms/AL = %.4f/%.3f = %.4f within 10%%",
			itl, step, acceptedLength, want)
	}
	if got, want := file["interactivity_p90"].(float64), 1000/itl; math.Abs(got-want) > 1e-9*want {
		t.Errorf("interactivity_p90 = %v, want 1000/request_itl_p90_ms = %v", got, want)
	}
	if legacy := file["itl_p90_ms"].(float64); legacy != step {
		t.Errorf("with no phase set, step_interval_p90_ms (%v) must equal the legacy itl_p90_ms (%v)", step, legacy)
	}

	prov, ok := file["kernel_provenance"].(map[string]any)["colocated"].(map[string]any)
	if !ok {
		t.Fatalf("kernel_provenance has no colocated pool: %v", file["kernel_provenance"])
	}
	if total := prov["total"].(float64); total <= 0 || prov["evidenced"].(float64) > total {
		t.Errorf("kernel_provenance.colocated: evidenced %v of total %v", prov["evidenced"], total)
	}
	if len(prov["assumptions"].([]any)) == 0 {
		t.Error("the V4.1 kernel records assumptions on h100, but none reached the file")
	}
	if len(prov["skipped_rules"].([]any)) == 0 {
		t.Error("no skipped release rules reported, though the kernel applies only the field layer")
	}
}

// The build pins come from the binary's build information: the main module's version with
// its VCS revision, and the kernel and schemas dependency versions (a replaced module reports
// its replacement). Nothing is guessed when the information is absent.
func TestVersionsFromBuildInfo(t *testing.T) {
	info := &debug.BuildInfo{
		Main: debug.Module{Path: "github.com/inference-sim/inference-sim", Version: "(devel)"},
		Deps: []*debug.Module{
			{Path: "github.com/inference-sim/blis-latency-kernel", Version: "v0.2.0"},
			{Path: "github.com/inference-sim/blis-schemas", Version: "v0.3.0",
				Replace: &debug.Module{Path: "github.com/inference-sim/blis-schemas", Version: "v0.3.1"}},
		},
		Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: "abc123"}, {Key: "vcs.modified", Value: "true"}},
	}
	got := versionsFrom(info)
	if got.BLIS != "(devel)+abc123-dirty" || got.Kernel != "v0.2.0" || got.Schemas != "v0.3.1" {
		t.Errorf("versionsFrom = %+v", *got)
	}
	if none := versionsFrom(nil); none.BLIS != "unknown" || none.Kernel != "unknown" || none.Schemas != "unknown" {
		t.Errorf("versionsFrom(nil) = %+v, want every pin unknown", *none)
	}
}
