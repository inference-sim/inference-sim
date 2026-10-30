package cmd

// CLI-level tests for `blis replay --agentic-scenario inferencex-agentx-mvp` (#1937).
// NOTE: these tests mutate package-level CLI vars and MUST NOT use t.Parallel().

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

const agentXSlice = "../sim/workload/testdata/agentx/weka_v7_slice.jsonl"

func agentXSliceSHA(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(agentXSlice)
	if err != nil {
		t.Fatal(err)
	}
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// runAgentXReplay runs replayCmd over the committed corpus slice and returns stdout and
// the --metrics-path file.
func runAgentXReplay(t *testing.T, args ...string) (stdout, file []byte) {
	t.Helper()
	restore := captureCmdLevelVars()
	origSession, origMetrics := replaySessionMode, replayMetricsPath
	defer func() {
		restore.restore()
		replaySessionMode, replayMetricsPath = origSession, origMetrics
		agxScenarioFlag, agxWorkloadSpec, agxCorpusPath, agxCorpusSHA256 = "", "", "", ""
		agxConcurrency, agxBenchmarkDur, agxGracePeriod, agxMaxContextLength, agxCorpusCache = 0, 0, "", "", ""
		workloadAcceptance = nil
	}()
	catalogDir := setupKernelTestFixtures(t)
	metrics := filepath.Join(t.TempDir(), "m.json")
	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	registerAgentXReplayFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	testCmd.Flags().StringVar(&replaySessionMode, "session-mode", "fixed", "")
	testCmd.Flags().StringVar(&replayMetricsPath, "metrics-path", "", "")
	base := []string{
		// The vendored DeepSeek-V4.1-Flash scenario: its auto max_model_len (1 Mi tokens)
		// holds the corpus's long agentic prompts, as the campaign's deployment does.
		"--catalog", catalogDir, "--scenario", "deepseek-v4.1-flash-h100-tp8-dspark5.yaml", "--scenarios", "../testdata/scenarios",
		"--registry", kernelRegistryDir(t), "--speculative-acceptance-rate", "0.502",
		"--defaults-filepath", "../defaults.yaml",
		"--agentic-scenario", "inferencex-agentx-mvp", "--corpus", agentXSlice, "--corpus-sha256", agentXSliceSHA(t),
		"--benchmark-duration", "900s", "--metrics-path", metrics,
	}
	// noDefaultAcceptance drops the default CLI acceptance, so a lower layer can supply it.
	if len(args) > 0 && args[0] == noDefaultAcceptance {
		args = args[1:]
		base = slices.DeleteFunc(base, func(s string) bool { return s == "--speculative-acceptance-rate" || s == "0.502" })
	}
	if err := testCmd.ParseFlags(append(base, args...)); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	done := make(chan []byte)
	go func() { b, _ := io.ReadAll(r); done <- b }()
	replayCmd.Run(testCmd, nil)
	_ = w.Close()
	os.Stdout = origStdout
	stdout = <-done
	file, err := os.ReadFile(metrics)
	if err != nil {
		t.Fatal(err)
	}
	return stdout, file
}

type agentXFile struct {
	InjectedRequests  int `json:"injected_requests"`
	CompletedRequests int `json:"completed_requests"`
	StillRunning      int `json:"still_running"`
	StillQueued       int `json:"still_queued"`
	Agentic           struct {
		Lanes             int    `json:"lanes"`
		MainRequests      int    `json:"main_requests"`
		SubagentRequests  int    `json:"subagent_requests"`
		WarmupRequests    int    `json:"warmup_requests"`
		CancelledRequests int    `json:"cancelled_requests"`
		Filtered          int    `json:"traces_filtered_by_max_context"`
		MaxContextLength  int    `json:"max_context_length"`
		ProfileStartUs    int64  `json:"profile_start_us"`
		ProfileCompleteUs int64  `json:"profile_complete_us"`
		TraceIdleCapJumps int    `json:"trace_idle_cap_jumps"`
		CorpusSHA256      string `json:"corpus_sha256"`
	} `json:"agentic"`
	Requests []struct {
		Phase   string `json:"phase"`
		Agentic *struct {
			Lane    int    `json:"lane"`
			Stream  string `json:"stream"`
			Outcome string `json:"outcome"`
		} `json:"agentic"`
	} `json:"requests"`
}

// TestReplayAgentX_DeterministicConservingAndTagged: INV-6 (two runs' Results are
// byte-identical outside runtime, and their results files byte-identical) and INV-AX1 through the real cluster (every request the
// replay sent was injected, and ended completed or cancelled at the grace end), with every
// request tagged with its phase and lane/tree/stream.
func TestReplayAgentX_DeterministicConservingAndTagged(t *testing.T) {
	a, fa := runAgentXReplay(t, "--concurrency", "3", "--max-context-length", "max_model_len")
	b, fb := runAgentXReplay(t, "--concurrency", "3", "--max-context-length", "max_model_len")
	same := withoutRuntimeOrArtifactPaths(t, string(a)) == withoutRuntimeOrArtifactPaths(t, string(b))
	if len(a) == 0 || !same || !bytes.Equal(fa, fb) {
		t.Fatalf("agentic replay is not byte-identical across runs (Result equal outside runtime: %v, file equal: %v)", same, bytes.Equal(fa, fb))
	}
	var m agentXFile
	if err := json.Unmarshal(fa, &m); err != nil {
		t.Fatal(err)
	}
	ag := m.Agentic
	sent := ag.MainRequests + ag.SubagentRequests + ag.WarmupRequests
	if ag.Lanes != 3 || sent == 0 || sent != m.InjectedRequests {
		t.Fatalf("lanes %d, sent %d, injected %d", ag.Lanes, sent, m.InjectedRequests)
	}
	if m.InjectedRequests != m.CompletedRequests+ag.CancelledRequests || ag.CancelledRequests != m.StillRunning+m.StillQueued {
		t.Errorf("INV-AX1: injected %d != completed %d + cancelled %d (still running %d, queued %d)",
			m.InjectedRequests, m.CompletedRequests, ag.CancelledRequests, m.StillRunning, m.StillQueued)
	}
	if ag.ProfileStartUs <= 0 || ag.ProfileCompleteUs < ag.ProfileStartUs+900_000_000 || ag.CorpusSHA256 != agentXSliceSHA(t) {
		t.Errorf("window [%d, %d], sha %s", ag.ProfileStartUs, ag.ProfileCompleteUs, ag.CorpusSHA256)
	}
	warm := 0
	for _, r := range m.Requests {
		if r.Agentic == nil || r.Agentic.Stream == "" || (r.Phase != "warmup" && r.Phase != "profile") {
			t.Fatalf("untagged request row %+v", r)
		}
		if r.Phase == "warmup" {
			warm++
		}
	}
	if warm != ag.WarmupRequests || len(m.Requests) != m.InjectedRequests {
		t.Errorf("%d rows (%d warm-up), want %d (%d)", len(m.Requests), warm, m.InjectedRequests, ag.WarmupRequests)
	}
}

// TestReplayAgentX_MaxContextFiltersAgainstTheDeployment: max_model_len resolves to the
// scenario's resolved max_model_len, and a smaller explicit limit drops more traces.
func TestReplayAgentX_MaxContextFiltersAgainstTheDeployment(t *testing.T) {
	_, f := runAgentXReplay(t, "--concurrency", "1", "--max-context-length", "max_model_len")
	var all agentXFile
	if err := json.Unmarshal(f, &all); err != nil {
		t.Fatal(err)
	}
	if all.Agentic.MaxContextLength <= 0 {
		t.Fatalf("max_model_len resolved to %d", all.Agentic.MaxContextLength)
	}
	_, f = runAgentXReplay(t, "--concurrency", "1", "--max-context-length", "60000")
	var some agentXFile
	if err := json.Unmarshal(f, &some); err != nil {
		t.Fatal(err)
	}
	if some.Agentic.MaxContextLength != 60000 || some.Agentic.Filtered <= all.Agentic.Filtered {
		t.Errorf("limit 60000 filtered %d traces, max_model_len (%d) filtered %d", some.Agentic.Filtered, all.Agentic.MaxContextLength, all.Agentic.Filtered)
	}
}

// TestReplayAgentX_Refusals: a wrong corpus digest, a window under the scenario's 900 s
// floor, and a TraceV2 flag are each refused at startup.
func TestReplayAgentX_Refusals(t *testing.T) {
	cases := map[string][]string{
		"digest":   {"--corpus-sha256", strings.Repeat("0", 64)},
		"duration": {"--benchmark-duration", "600s"},
		"tracev2":  {"--session-mode", "closed-loop"},
	}
	if name := os.Getenv("BLIS_AGENTX_REFUSAL"); name != "" {
		runAgentXReplay(t, append([]string{"--concurrency", "1"}, cases[name]...)...)
		os.Exit(0)
	}
	for name, want := range map[string]string{"digest": "does not match", "duration": "900", "tracev2": "does not apply"} {
		cmd := exec.Command(os.Args[0], "-test.run=TestReplayAgentX_Refusals")
		cmd.Env = append(os.Environ(), "BLIS_AGENTX_REFUSAL="+name)
		out, err := cmd.CombinedOutput()
		if err == nil || !strings.Contains(string(out), want) {
			t.Errorf("%s: want a fatal refusal mentioning %q, got err=%v\n%s", name, want, err, out)
		}
	}
}

// TestReplayAgentX_WorkloadSpecWithFlagOverrides: a blis-schemas replay descriptor states
// the workload; a flag overrides the field it names.
func TestReplayAgentX_WorkloadSpecWithFlagOverrides(t *testing.T) {
	spec := filepath.Join(t.TempDir(), "replay.yaml")
	doc := "workload:\n  replay:\n    scenario: inferencex-agentx-mvp\n    harness_version: 0.12.0\n" +
		"    corpus:\n      loader: semianalysis_cc_traces_weka_062126_256k\n      format: weka-trace-v7\n      encoding: jsonl\n" +
		"      source:\n        huggingface: semianalysisai/cc-traces-weka-062126-256k\n        revision: test\n" +
		"      sha256: " + agentXSliceSHA(t) + "\n" +
		"    concurrency: 2\n    benchmark_duration_s: 900\n    max_context_length: max_model_len\n"
	if err := os.WriteFile(spec, []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		args  []string
		lanes int
	}{
		{[]string{"--workload-spec", spec}, 2},
		{[]string{"--workload-spec", spec, "--concurrency", "1"}, 1},
	} {
		_, f := runAgentXReplay(t, tc.args...)
		var m agentXFile
		if err := json.Unmarshal(f, &m); err != nil {
			t.Fatal(err)
		}
		if m.Agentic.Lanes != tc.lanes || m.Agentic.MaxContextLength <= 0 {
			t.Errorf("%v: %d lanes (want %d), max_context_length %d", tc.args, m.Agentic.Lanes, tc.lanes, m.Agentic.MaxContextLength)
		}
	}
}

// TestReplayAgentX_ForkExtensions: a workload spec that states the per-trace idle cap and a
// request-bounded cache-pressure warmup (the InferenceX fork run's knobs, #1971) flows
// end-to-end — the warmup extras raise the warmup count, the per-trace cap fires at
// concurrency >= 2, and omitting both reproduces the baseline summary.
func TestReplayAgentX_ForkExtensions(t *testing.T) {
	head := "workload:\n  replay:\n    scenario: inferencex-agentx-mvp\n    harness_version: 0.12.0\n" +
		"    corpus:\n      loader: semianalysis_cc_traces_weka_062126_256k\n      format: weka-trace-v7\n      encoding: jsonl\n" +
		"      source:\n        huggingface: semianalysisai/cc-traces-weka-062126-256k\n        revision: test\n" +
		"      sha256: " + agentXSliceSHA(t) + "\n" +
		"    concurrency: 4\n    benchmark_duration_s: 900\n    max_context_length: max_model_len\n"
	write := func(extra string) string {
		p := filepath.Join(t.TempDir(), "replay.yaml")
		if err := os.WriteFile(p, []byte(head+extra), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	base := write("")
	fork := write("    trace_idle_gap_cap_s: 300\n    warmup:\n      mode: cache_pressure\n      requests_per_lane: 5\n")

	var mb, mf agentXFile
	_, fb := runAgentXReplay(t, "--workload-spec", base)
	_, ff := runAgentXReplay(t, "--workload-spec", fork)
	if err := json.Unmarshal(fb, &mb); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(ff, &mf); err != nil {
		t.Fatal(err)
	}
	// The per-lane extras add to the warmup count (5 per lane that had a primer).
	if mf.Agentic.WarmupRequests <= mb.Agentic.WarmupRequests {
		t.Errorf("fork warmup requests %d not above baseline %d", mf.Agentic.WarmupRequests, mb.Agentic.WarmupRequests)
	}
	// The per-trace cap fires at concurrency 4; the baseline (no cap) never does.
	if mb.Agentic.TraceIdleCapJumps != 0 {
		t.Errorf("baseline must not fire the per-trace cap: %d jumps", mb.Agentic.TraceIdleCapJumps)
	}
	if mf.Agentic.TraceIdleCapJumps == 0 {
		t.Errorf("the per-trace cap should have fired at concurrency 4")
	}
}

// noDefaultAcceptance, as runAgentXReplay's first argument, runs without the default
// --speculative-acceptance-rate.
const noDefaultAcceptance = "<no-default-acceptance>"

// The agentic descriptor is the simulated workload, so the acceptance it states is the
// workload layer (#1939): with no CLI acceptance the run speculates at it and records it.
func TestReplayAgentX_DescriptorAcceptanceIsTheWorkloadLayer(t *testing.T) {
	spec := filepath.Join(t.TempDir(), "replay.yaml")
	doc := "workload:\n  replay:\n    scenario: inferencex-agentx-mvp\n    harness_version: 0.12.0\n" +
		"    corpus:\n      loader: semianalysis_cc_traces_weka_062126_256k\n      format: weka-trace-v7\n      encoding: jsonl\n" +
		"      source:\n        huggingface: semianalysisai/cc-traces-weka-062126-256k\n        revision: test\n" +
		"      sha256: " + agentXSliceSHA(t) + "\n" +
		"    concurrency: 2\n    benchmark_duration_s: 900\n    max_context_length: max_model_len\n" +
		"    acceptance:\n      num_spec_tokens: 5\n      acceptance_length: 3.1\n      source: test descriptor\n"
	if err := os.WriteFile(spec, []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}
	_, f := runAgentXReplay(t, noDefaultAcceptance, "--workload-spec", spec)
	var m struct {
		Acceptance *struct {
			Length      float64 `json:"length"`
			SourceLayer string  `json:"source_layer"`
		} `json:"acceptance"`
	}
	if err := json.Unmarshal(f, &m); err != nil {
		t.Fatal(err)
	}
	if m.Acceptance == nil || m.Acceptance.SourceLayer != "workload" || m.Acceptance.Length != 3.1 {
		t.Errorf("acceptance = %+v, want length 3.1 from the workload layer", m.Acceptance)
	}
}

// TestReplayAgentX_CorpusCacheIsInvisible: a run that writes the corpus snapshot and a run
// that reads it are byte-identical (the Result outside runtime, and the results file) to a run
// without the cache (INV-6), and a damaged snapshot is ignored and rewritten.
func TestReplayAgentX_CorpusCacheIsInvisible(t *testing.T) {
	dir := t.TempDir()
	base, fbase := runAgentXReplay(t, "--concurrency", "2")
	miss, fmiss := runAgentXReplay(t, "--concurrency", "2", "--corpus-cache", dir)
	snaps, _ := filepath.Glob(filepath.Join(dir, "agentx-corpus-*.bin"))
	if len(snaps) != 1 {
		t.Fatalf("snapshots written: %v", snaps)
	}
	hit, fhit := runAgentXReplay(t, "--concurrency", "2", "--corpus-cache", dir)
	if err := os.WriteFile(snaps[0], []byte("damaged"), 0o644); err != nil {
		t.Fatal(err)
	}
	dmg, fdmg := runAgentXReplay(t, "--concurrency", "2", "--corpus-cache", dir)
	for name, out := range map[string][2][]byte{"miss": {miss, fmiss}, "hit": {hit, fhit}, "damaged": {dmg, fdmg}} {
		if withoutRuntimeOrArtifactPaths(t, string(out[0])) != withoutRuntimeOrArtifactPaths(t, string(base)) || !bytes.Equal(out[1], fbase) {
			t.Errorf("%s: output differs from an uncached run", name)
		}
	}
	if b, _ := os.ReadFile(snaps[0]); bytes.Equal(b, []byte("damaged")) {
		t.Error("the damaged snapshot was not rewritten")
	}
}

// The Result of an agentic replay measures over the replay's profiling window, the same window
// the results file's measured aggregates are over (#1871): window.duration_s is the file's
// measured_window_s and the window is the explicit one.
func TestReplayAgentX_ResultMeasuresTheProfilingWindow(t *testing.T) {
	stdout, file := runAgentXReplay(t, "--concurrency", "2")
	var f map[string]any
	if err := json.Unmarshal(file, &f); err != nil {
		t.Fatal(err)
	}
	r := parseResult(t, string(stdout))
	if r.Window.Start != windowStartExplicit {
		t.Errorf("window.start = %q, want the explicit profiling window %q", r.Window.Start, windowStartExplicit)
	}
	if want := f["measured_window_s"].(float64); want <= 0 || r.Window.DurationS != want {
		t.Errorf("window.duration_s = %v, want the file's measured_window_s %v", r.Window.DurationS, want)
	}
	if r.Engine["cluster"].(map[string]any)["agentic"] == nil {
		t.Error("the agentic replay's counts are not in engine.cluster.agentic")
	}
}
