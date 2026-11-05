package cmd

import (
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// DEV-7: the stop rule names what ended the run, from the request cap and the horizon, and the
// drain rule follows from it: a horizon leaves requests unfinished, exhaustion runs every
// request to its end.
func TestResultFrame_StopRuleFromCapAndHorizon(t *testing.T) {
	r := &result.Result{MetricDefs: map[string]measurement.MetricDef{"ttft_mean_ms": {}, "e2e_mean_ms": {}},
		Window:   result.Window{Start: windowStartFirstArrival, Stop: windowStopLastComplete},
		Requests: result.Requests{StillQueued: 2, StillRunning: 1, GatewayQueueDepth: 4}}
	for _, tc := range []struct {
		cap     int
		horizon int64
		rule    string
		drain   string
	}{
		{30, 600_000_000, frameStopBoth, frameDrainNone},
		{30, math.MaxInt64, frameStopRequestCount, frameDrainToEnd},
		{0, 600_000_000, frameStopHorizon, frameDrainNone},
		{0, math.MaxInt64, frameStopExhaustion, frameDrainToEnd},
		{0, 0, frameStopExhaustion, frameDrainToEnd},
	} {
		f := buildFrame(frameGenerator{Kind: generatorDistribution, Seed: 7}, r, tc.cap, tc.horizon)
		if f.StopRule.Rule != tc.rule || f.Phases.Drain != tc.drain {
			t.Errorf("cap %d, horizon %d: stop rule %q, drain %q; want %q, %q", tc.cap, tc.horizon,
				f.StopRule.Rule, f.Phases.Drain, tc.rule, tc.drain)
		}
		want := tc.horizon
		if want == math.MaxInt64 {
			want = 0 // an unbounded run records no horizon
		}
		if f.StopRule.HorizonUs != want {
			t.Errorf("horizon %d recorded as %d, want %d", tc.horizon, f.StopRule.HorizonUs, want)
		}
	}
	f := buildFrame(frameGenerator{Kind: generatorDistribution}, r, 0, 0)
	if f.Acceptance.UnfinishedAtStop != 7 {
		t.Errorf("unfinished at stop %d, want 7 (queued + running + held by the router)", f.Acceptance.UnfinishedAtStop)
	}
	if strings.Join(f.Metrics, ",") != "e2e_mean_ms,ttft_mean_ms" {
		t.Errorf("metrics %v, want the metric_defs keys sorted", f.Metrics)
	}
	if f.Phases.Warmup != frameNoWarmup || f.InitialState.Caches != frameCachesEmpty || f.Generator.Random == "" {
		t.Errorf("frame %+v: want no warm-up, empty caches and a named random generator", f)
	}
}

// DEV-7 through the CLI: every Result records its frame under engine.frame. A run names its
// generator and seed, a replay of its trace names the trace by sha256 and its session mode,
// and the two agree on everything else in the frame. A frame that is not written, or whose
// metric identifiers drift from metric_defs, fails.
func TestResultFrame_EveryResultRecordsItsFrame(t *testing.T) {
	prefix := t.TempDir() + "/trace"
	runOut, stderr, err := runKernelCLI(t, resultRunArgs("--trace-output", prefix)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	repOut, stderr, err := runKernelCLI(t, "replay", "--scenario", kernelTestScenario, "--seed", "7",
		"--horizon", "600000000", "--trace-header", prefix+".yaml", "--trace-data", prefix+".csv")
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
	}
	frame := func(stdout string) map[string]any {
		f, ok := resultValue(resultDoc(t, stdout), "engine.frame")
		if !ok {
			t.Fatalf("the Result records no engine.frame")
		}
		return f.(map[string]any)
	}
	rf, pf := frame(runOut), frame(repOut)
	rg, pg := rf["generator"].(map[string]any), pf["generator"].(map[string]any)
	if rg["kind"] != generatorDistribution || rg["seed"] != float64(7) {
		t.Errorf("run generator %v, want distribution at seed 7", rg)
	}
	if pg["kind"] != generatorTrace || pg["session_mode"] != "fixed" || !strings.HasPrefix(pg["source"].(string), "sha256:") {
		t.Errorf("replay generator %v, want the trace by sha256 in fixed mode", pg)
	}
	if rs := rf["stop_rule"].(map[string]any); rs["rule"] != frameStopBoth || rs["num_requests"] != float64(30) {
		t.Errorf("run stop rule %v, want both: 30 requests and the horizon", rs)
	}
	defs := parseResult(t, runOut).MetricDefs
	ids := rf["metrics"].([]any)
	if len(ids) != len(defs) || len(ids) == 0 {
		t.Errorf("frame names %d metrics, metric_defs %d", len(ids), len(defs))
	}
	for _, id := range ids {
		if _, ok := defs[id.(string)]; !ok {
			t.Errorf("frame metric %q is not in metric_defs", id)
		}
	}
	for _, k := range []string{"phases", "initial_state", "acceptance", "metrics"} {
		if !jsonEqual(t, rf[k], pf[k]) {
			t.Errorf("frame.%s: run %v, replay %v", k, rf[k], pf[k])
		}
	}
}
