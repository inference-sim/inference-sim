package cmd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim"
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
	local := versionsFrom(&debug.BuildInfo{Main: debug.Module{Version: "v1.2.3"}, Deps: []*debug.Module{
		{Path: "github.com/inference-sim/blis-latency-kernel", Version: "v0.2.0",
			Replace: &debug.Module{Path: "../blis-latency-kernel"}},
	}})
	if local.BLIS != "v1.2.3" || local.Kernel != "replace:../blis-latency-kernel" || local.Schemas != "unknown" {
		t.Errorf("a filesystem replace must be reported as such, got %+v", *local)
	}
	// A worktree build carries no VCS settings; the -ldflags revision names the commit, and a
	// stamped revision wins over it.
	saved := buildRevision
	t.Cleanup(func() { buildRevision = saved })
	buildRevision = "def456-dirty"
	if w := versionsFrom(&debug.BuildInfo{Main: debug.Module{Version: "(devel)"}}); w.BLIS != "(devel)+def456-dirty" {
		t.Errorf("an injected revision must be reported, got %q", w.BLIS)
	}
	if s := versionsFrom(info); s.BLIS != "(devel)+abc123-dirty" {
		t.Errorf("a stamped revision must win over an injected one, got %q", s.BLIS)
	}
	buildRevision = ""
	if none := versionsFrom(nil); none.BLIS != "unknown" || none.Kernel != "unknown" || none.Schemas != "unknown" {
		t.Errorf("versionsFrom(nil) = %+v, want every pin unknown", *none)
	}
}

// calibrate compares whole-run TTFT with whole-run throughput, so it reads whole_run's TTFT
// from a file that has it (the top-level ttft_mean_ms is the measured window's) and the
// top-level value from an older file without it.
func TestLoadBLISAggregate_ReadsTheWholeRunTTFT(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	now := loadBLISAggregate(write("new.json",
		`{"ttft_mean_ms": 50, "vllm_estimated_duration_s": 2, "total_output_tokens": 10, "whole_run": {"ttft_mean_ms": 70}}`))
	if now.TTFTMs != 70 {
		t.Errorf("TTFT %v, want whole_run's 70", now.TTFTMs)
	}
	old := loadBLISAggregate(write("old.json", `{"ttft_mean_ms": 50, "vllm_estimated_duration_s": 2, "total_output_tokens": 10}`))
	if old.TTFTMs != 50 {
		t.Errorf("TTFT %v, want the top-level 50 of a file without whole_run", old.TTFTMs)
	}
}

// A run no kernel priced still states its provenance block, empty, so the file's shape does
// not depend on the backend.
func TestKernelProvenance_IsEmptyNotAbsentWithoutAKernel(t *testing.T) {
	got := kernelProvenance(nil, nil)
	if got == nil || len(got) != 0 {
		t.Fatalf("kernelProvenance(nil, nil) = %v, want an empty map", got)
	}
	m := sim.NewMetrics()
	path := filepath.Join(t.TempDir(), "m.json")
	if err := m.EmitOutput(m.BuildOutput("cluster"), path, sim.WithKernelProvenance(got)); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"kernel_provenance": {}`) {
		t.Errorf("the file must carry an empty kernel_provenance block:\n%s", data)
	}
}

// cache_hit_rate is the GPU prefix cache's rate and cpu_tier_hit_rate the CPU tier's share
// of the same queried tokens, so calibrate compares their sum against a tiered observation
// (blocks either tier served) and the GPU rate alone against a GPU-cache observation.
func TestLoadSimCacheHitRate_AddsTheCPUTierForATieredObservation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.json")
	if err := os.WriteFile(path, []byte(`{"cache_hit_rate": 0.5, "cpu_tier_hit_rate": 0.2, "percentile_method": "numpy_linear"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	for source, want := range map[string]float64{"tiered": 0.7, "gpu-prefix-cache-fallback": 0.5} {
		got, err := loadSimCacheHitRate(path, source)
		if err != nil || got == nil || math.Abs(*got-want) > 1e-12 {
			t.Errorf("source %s: %v (err %v), want %v", source, got, err, want)
		}
	}
}

// The results file's prefix_cache block is each pool's prefix-hit resolution as the engine model
// gives it (build.PrefixHitOf), --swa-bounded-replay included: DeepSeek-V4.1's bounded replay on
// by vLLM v0.31.0's default (a replay window), and off by the flag (the window group instead).
func TestMetricsPath_PrefixCacheIsTheModelsResolution(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	m, err := kernelmodel.Open(dsparkScenario, kernelmodel.Repos{Scenarios: pdScenarios, Catalog: catalog, Registry: registry})
	if err != nil {
		t.Fatal(err)
	}
	off := false
	for _, c := range []struct {
		flags    []string
		override *bool
		from     string
	}{
		{nil, nil, "v0.31.0 default"},
		{[]string{"--swa-bounded-replay", "false"}, &off, "--swa-bounded-replay"},
	} {
		path := filepath.Join(t.TempDir(), "metrics.json")
		args := append([]string{"run", "--scenarios", pdScenarios, "--scenario", dsparkScenario,
			"--num-requests", "5", "--rate", "2", "--seed", "4", "--speculative-acceptance-length", "3.2",
			"--metrics-path", path}, c.flags...)
		if _, stderr, err := runKernelCLI(t, args...); err != nil {
			t.Fatalf("run %v: %v\n%s", c.flags, err, lastLines(stderr, 5))
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var file struct {
			PrefixCache map[string]sim.PrefixCacheRecord `json:"prefix_cache"`
		}
		if err := json.Unmarshal(data, &file); err != nil {
			t.Fatal(err)
		}
		got, ok := file.PrefixCache["colocated"]
		if !ok {
			t.Fatalf("flags %v: no colocated prefix_cache record in %s", c.flags, data)
		}
		want := m.PrefixHit(c.override)
		if got.Layout != want.Hit || got.ModelRunner != want.Runner || got.SWABoundedReplayFrom != c.from ||
			got.SWABoundedReplay == nil || want.BoundedReplay == nil || *got.SWABoundedReplay != *want.BoundedReplay {
			t.Errorf("flags %v: recorded %+v, want the model's %+v from %q", c.flags, got, want, c.from)
		}
	}
	if def, offR := m.PrefixHit(nil), m.PrefixHit(&off); def.Hit == offR.Hit {
		t.Fatal("non-vacuity: the bounded replay must change DeepSeek-V4.1's layout")
	}
}
