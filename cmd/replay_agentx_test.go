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
		agxConcurrency, agxBenchmarkDur, agxGracePeriod, agxMaxContextLength = 0, 0, "", ""
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

// TestReplayAgentX_DeterministicConservingAndTagged: INV-6 (two runs are byte-identical on
// stdout and in the results file) and INV-AX1 through the real cluster (every request the
// replay sent was injected, and ended completed or cancelled at the grace end), with every
// request tagged with its phase and lane/tree/stream.
func TestReplayAgentX_DeterministicConservingAndTagged(t *testing.T) {
	a, fa := runAgentXReplay(t, "--concurrency", "3", "--max-context-length", "max_model_len")
	b, fb := runAgentXReplay(t, "--concurrency", "3", "--max-context-length", "max_model_len")
	if len(a) == 0 || !bytes.Equal(a, b) || !bytes.Equal(fa, fb) {
		t.Fatalf("agentic replay is not byte-identical across runs (stdout equal: %v, file equal: %v)", bytes.Equal(a, b), bytes.Equal(fa, fb))
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
