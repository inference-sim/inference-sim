package cmd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/metrics"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
)

// DEV-7: the stop rule states the request cap and the horizon that ended the run, and the drain
// rule follows from them: a horizon leaves requests unfinished, exhaustion runs every request to
// its end.
func TestResultFrame_StopRuleFromCapAndHorizon(t *testing.T) {
	r := &result.Result{Metrics: map[string]result.Figure{"ttft_mean_ms": {Scope: metrics.ScopeCluster}, "e2e_mean_ms": {Scope: metrics.ScopeCluster}},
		Requests: result.Requests{Generated: 20, Completed: 13, QueuedAtHorizon: 6, InFlightAtHorizon: 1}}
	rules := windowRules{result.StartFirstArrivalAfterWarmup, result.StopLastCompletion}
	for _, tc := range []struct {
		cap     int
		horizon int64
		drain   result.DrainRule
	}{
		{30, 600_000_000, result.DrainNone},
		{30, math.MaxInt64, result.DrainUntilComplete},
		{0, 600_000_000, result.DrainNone},
		{0, math.MaxInt64, result.DrainUntilComplete},
	} {
		buildFrame(frameGenerator{Kind: generatorDistribution, Seed: 7}, r, rules, tc.cap, tc.horizon)
		f := r.Frame
		if f.Stop.Requests != tc.cap || f.Phases.Drain.Rule != tc.drain {
			t.Errorf("cap %d, horizon %d: stop %+v, drain %q; want requests %d, drain %q", tc.cap, tc.horizon,
				f.Stop, f.Phases.Drain.Rule, tc.cap, tc.drain)
		}
		bounded := tc.horizon > 0 && tc.horizon < math.MaxInt64
		if (f.Stop.HorizonS != nil) != bounded || (bounded && *f.Stop.HorizonS != float64(tc.horizon)/1e6) {
			t.Errorf("horizon %d recorded as %v", tc.horizon, f.Stop.HorizonS)
		}
	}
	// A run with neither a cap nor a horizon has no stop rule to state: the command refuses it
	// before the frame is built (a run that generated nothing and has no horizon), so the frame
	// is built here for a run that stopped on its cap.
	buildFrame(frameGenerator{Kind: generatorDistribution}, r, rules, 30, 0)
	f := r.Frame
	if len(f.Acceptance) != 1 || f.Acceptance[0].Condition != result.ConditionUnfinishedAtMost ||
		f.Acceptance[0].Observed != 7.0/20 || !f.Acceptance[0].Met {
		t.Errorf("acceptance %+v, want the unfinished fraction 7/20 observed and met", f.Acceptance)
	}
	if strings.Join(f.Metrics, ",") != "e2e_mean_ms,ttft_mean_ms" {
		t.Errorf("metrics %v, want the figures' identifiers sorted", f.Metrics)
	}
	if f.Kind != result.FrameTerminating || f.Phases.Warmup.Rule != result.WarmupNone ||
		f.InitialState.Caches != result.CachesEmpty || f.Generator.RNG == "" ||
		f.Phases.Measured.Start != rules.start {
		t.Errorf("frame %+v: want terminating, no warm-up, empty caches, a named RNG and the window's rules", f)
	}
	if v := r.Validate(); v != nil {
		for _, p := range v.All() {
			// The Result here carries no workload or offered load: those are the run's, set
			// by the command (TestResultFrame_EveryResultRecordsItsFrame validates a whole one).
			if strings.HasPrefix(p.Path, "frame") && p.Path != "frame.generator" {
				t.Errorf("the contracts refuse the frame: %s", p)
			}
		}
	}
}

// DEV-7 through the CLI: every Result records its frame. A run names its workload and seed, a
// replay of its trace names the trace by sha256 in its workload binding, and the two agree on
// everything else in the frame. A frame that is not written, or whose metric identifiers drift
// from the figures, fails.
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
	run, rep := parseResult(t, runOut), parseResult(t, repOut)
	if g := run.Frame.Generator; g.Seed != 7 || g.Load.RequestRate <= 0 {
		t.Errorf("run generator %+v, want seed 7 and the offered rate", g)
	}
	if w := rep.Frame.Generator.Workload; w == nil || w.Trace == nil || len(w.Trace.SHA256) != 64 {
		t.Errorf("replay generator %+v, want the trace by sha256", rep.Frame.Generator)
	}
	if s := run.Frame.Stop; s.Requests != 30 || s.HorizonS == nil {
		t.Errorf("run stop rule %+v, want 30 requests and the horizon", s)
	}
	if len(run.Frame.Metrics) != len(run.Metrics) || len(run.Metrics) == 0 {
		t.Errorf("frame names %d metrics, the Result reports %d", len(run.Frame.Metrics), len(run.Metrics))
	}
	for _, id := range run.Frame.Metrics {
		if _, ok := run.Metrics[id]; !ok {
			t.Errorf("frame metric %q is not reported", id)
		}
	}
	rf, pf := resultDoc(t, runOut)["frame"].(map[string]any), resultDoc(t, repOut)["frame"].(map[string]any)
	for _, k := range []string{"phases", "initial_state", "acceptance", "metrics", "kind"} {
		if !jsonEqual(t, rf[k], pf[k]) {
			t.Errorf("frame.%s: run %v, replay %v", k, rf[k], pf[k])
		}
	}
}

// DEV-7 (#2107): a replay reproduced from a Result whose recorded frame has been taken out or
// altered would run without the frame it was measured under. It is refused, and the refusal
// names the frame; the unaltered Result reproduces.
func TestReplayFromResult_RefusesAResultWhoseFrameWasDropped(t *testing.T) {
	dir := t.TempDir()
	prefix := dir + "/trace"
	if _, stderr, err := runKernelCLI(t, resultRunArgs("--trace-output", prefix)...); err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	traceArgs := []string{"--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}
	repOut, stderr, err := runKernelCLI(t, append([]string{"replay", "--scenario", kernelTestScenario, "--seed", "7",
		"--horizon", "600000000"}, traceArgs...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
	}
	reproduce := func(name string, edit func(doc map[string]any)) (string, error) {
		doc := resultDoc(t, repOut)
		edit(doc)
		raw, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		file := filepath.Join(dir, name+".json")
		if err := os.WriteFile(file, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		_, stderr, err := runKernelCLI(t, append([]string{"replay", "--from-result", file}, traceArgs...)...)
		return stderr, err
	}
	if stderr, err := reproduce("intact", func(map[string]any) {}); err != nil {
		t.Fatalf("the unaltered Result did not reproduce: %v\n%s", err, lastLines(stderr, 5))
	}
	for name, edit := range map[string]func(map[string]any){
		"dropped": func(d map[string]any) { delete(d, "frame") },
		"no-stop": func(d map[string]any) { delete(d["frame"].(map[string]any), "stop") },
	} {
		stderr, err := reproduce(name, edit)
		if err == nil {
			t.Errorf("%s: a replay of a Result with its frame altered ran", name)
			continue
		}
		if !strings.Contains(stderr, "frame") {
			t.Errorf("%s: refused, but not for the frame:\n%s", name, lastLines(stderr, 3))
		}
	}
}
