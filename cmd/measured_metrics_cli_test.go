package cmd

import (
	"encoding/json"
	"math"
	"runtime/debug"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/metrics"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim"
)

// dsparkScenario drafts K=5 DSpark tokens on DeepSeek-V4.1-Flash.
const dsparkScenario = "deepseek-v4.1-flash-h100-tp8-dspark5.yaml"

// Under DSpark K=5 a verify step advances AL = 1 + alpha*K tokens on average, so aiperf's
// per-token request ITL is the per-step interval divided by AL (#1938 acceptance). The run is
// light enough that step times barely vary, which is what lets the two p90s be compared. The
// 10% band holds for 39 of 40 seeds checked (101-120 on the named random streams and on the
// sequential ones before them); seed 2 is one of them.
// The same run pins that the Result names the two series as blis-schemas does (#1871): its
// tpot_p90_ms is the per-token request ITL, never the per-step interval, which it reports as
// the engine's step_interval_p90_ms; and that the kernel's evidence reaches the report.
func TestResult_DSparkRequestITLIsTheStepIntervalOverTheAcceptedLength(t *testing.T) {
	const alpha, k = 0.502, 5
	acceptedLength := 1 + alpha*k
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios, "--scenario", dsparkScenario,
		"--num-requests", "40", "--rate", "0.5", "--seed", "2", "--speculative-acceptance-rate", "0.502")
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	if def, ok := metrics.Lookup("tpot_p90_ms"); !ok || def.Formula != metrics.FormulaTPOT {
		t.Errorf("tpot_p90_ms is defined as %+v, want the registry's per-request tpot", def)
	}
	if f, ok := parseResult(t, stdout).Metrics["itl_p90_ms"]; !ok || f.Samples <= 0 {
		t.Errorf("the Result reports no itl_p90_ms with samples: %+v", f)
	}
	if got := parseResult(t, stdout).Requests.Completed; got != 40 {
		t.Fatalf("%d of 40 requests completed; the comparison would be vacuous", got)
	}

	itl := metric(t, stdout, "tpot_p90_ms")
	step := resultNum(t, stdout, "engine.cluster.step_interval_p90_ms")
	if want := step / acceptedLength; math.Abs(itl-want)/want > 0.10 {
		t.Errorf("tpot_p90_ms = %.4f, want step_interval_p90_ms/AL = %.4f/%.3f = %.4f within 10%%",
			itl, step, acceptedLength, want)
	}
	if got, want := metric(t, stdout, "interactivity_p90"), 1000/itl; math.Abs(got-want) > 1e-9*want {
		t.Errorf("interactivity_p90 = %v, want 1000/tpot_p90_ms = %v", got, want)
	}

	prov, ok := resultDoc(t, stdout)["reports"].(map[string]any)["kernel"].(map[string]any)["colocated"].(map[string]any)
	if !ok {
		t.Fatalf("reports.kernel has no colocated pool")
	}
	if total := prov["total"].(float64); total <= 0 || prov["evidenced"].(float64) > total {
		t.Errorf("reports.kernel.colocated: evidenced %v of total %v", prov["evidenced"], total)
	}
	if len(prov["assumptions"].([]any)) == 0 {
		t.Error("the V4.1 kernel records assumptions on h100, but none reached the report")
	}
	if len(prov["skipped_rules"].([]any)) == 0 {
		t.Error("no skipped release rules reported, though the kernel applies only the field layer")
	}
}

// The build pins come from the binary's build information: the main module's version with
// its VCS revision, and the blis-substrate version, which is both the kernel's and the schemas'
// (a replaced module reports its replacement). Nothing is guessed when the information is absent.
func TestVersionsFromBuildInfo(t *testing.T) {
	info := &debug.BuildInfo{
		Main: debug.Module{Path: "github.com/inference-sim/inference-sim", Version: "(devel)"},
		Deps: []*debug.Module{
			{Path: "github.com/inference-sim/blis-substrate", Version: "v0.11.0",
				Replace: &debug.Module{Path: "github.com/inference-sim/blis-substrate", Version: "v0.11.1"}},
		},
		Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: "abc123"}, {Key: "vcs.modified", Value: "true"}},
	}
	got := versionsFrom(info)
	if got.BLIS != "(devel)+abc123-dirty" || got.Kernel != "v0.11.1" || got.Schemas != "v0.11.1" {
		t.Errorf("versionsFrom = %+v", *got)
	}
	local := versionsFrom(&debug.BuildInfo{Main: debug.Module{Version: "v1.2.3"}, Deps: []*debug.Module{
		{Path: "github.com/inference-sim/blis-substrate", Version: "v0.11.0",
			Replace: &debug.Module{Path: "../blis-substrate"}},
	}})
	if local.BLIS != "v1.2.3" || local.Kernel != "replace:../blis-substrate" || local.Schemas != "replace:../blis-substrate" {
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

// A run no kernel priced still states its provenance as an empty map, so the report's shape
// does not depend on the backend.
func TestKernelProvenance_IsEmptyNotAbsentWithoutAKernel(t *testing.T) {
	got := kernelProvenance(nil, nil)
	if got == nil || len(got) != 0 {
		t.Fatalf("kernelProvenance(nil, nil) = %v, want an empty map", got)
	}
}

// The Result's engine report carries, per pool, a prefix_cache block: each pool's prefix-hit resolution as the engine model
// gives it (build.PrefixHitOf), --swa-bounded-replay included: DeepSeek-V4.1's bounded replay on
// by vLLM v0.31.0's default (a replay window), and off by the flag (the window group instead).
func TestResult_PrefixCacheIsTheModelsResolution(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	m, err := kernelmodel.Open(dsparkScenario, reposOf(pdScenarios, catalog, registry))
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
		args := append([]string{"run", "--scenarios", pdScenarios, "--scenario", dsparkScenario,
			"--num-requests", "5", "--rate", "2", "--seed", "4", "--speculative-acceptance-length", "3.2"}, c.flags...)
		stdout, stderr, err := runKernelCLI(t, args...)
		if err != nil {
			t.Fatalf("run %v: %v\n%s", c.flags, err, lastLines(stderr, 5))
		}
		eng, _ := resultDoc(t, stdout)["reports"].(map[string]any)["engine"].(map[string]any)
		pool, _ := eng["colocated"].(map[string]any)
		raw, ok := pool["prefix_cache"]
		if !ok {
			t.Fatalf("flags %v: no colocated prefix_cache record in the engine report", c.flags)
		}
		enc, err := json.Marshal(raw)
		if err != nil {
			t.Fatal(err)
		}
		var got sim.PrefixCacheRecord
		if err := json.Unmarshal(enc, &got); err != nil {
			t.Fatalf("flags %v: the prefix_cache record: %v", c.flags, err)
		}
		want := m.PrefixHit(c.override)
		if got.Layout != sim.KVPrefixHit(want.Hit) || got.ModelRunner != want.Runner || got.SWABoundedReplayFrom != c.from ||
			got.SWABoundedReplay == nil || want.BoundedReplay == nil || *got.SWABoundedReplay != *want.BoundedReplay {
			t.Errorf("flags %v: recorded %+v, want the model's %+v from %q", c.flags, got, want, c.from)
		}
	}
	if def, offR := m.PrefixHit(nil), m.PrefixHit(&off); def.Hit == offR.Hit {
		t.Fatal("non-vacuity: the bounded replay must change DeepSeek-V4.1's layout")
	}
}
