package cmd

// CLI-level tests for `blis replay --session-mode fixed-accumulate` (#1692).
//
// fixed-accumulate injects every session round at its recorded arrival time
// (open-loop, like --session-mode fixed) while reconstructing the growing
// accumulate-delta input (like --session-mode closed-loop). These tests pin the
// mode's guards (BC-5), end-to-end reconstruction (BC-1), and determinism (BC-4).
//
// NOTE: these tests mutate package-level CLI vars and MUST NOT use t.Parallel().

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// accumulateCorpusCSV is a 2-session accumulate corpus (deltas, with a
// session_context_growth=accumulate header) written to a temp dir. Session s1
// grows 100→180→245; session s2 grows 60→95. Arrivals overlap across sessions
// (the cross-session contention fixed-accumulate exists to preserve).
func writeAccumulateCorpus(t *testing.T) (headerPath, dataPath string) {
	t.Helper()
	dir := t.TempDir()
	headerPath = filepath.Join(dir, "corpus.yaml")
	dataPath = filepath.Join(dir, "corpus.csv")
	header := "trace_version: 3\ntime_unit: microseconds\nmode: generated\nsession_context_growth: accumulate\nwarm_up_requests: 0\n"
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}
	// deltas: s1 r0=100; s1 r1 = 180-100-50 = 30; s1 r2 = 245-180-40 = 25; s2 r0=60; s2 r1 = 95-60-15 = 20.
	csv := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,false,100,50,100,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n" +
		"1,c1,t1,standard,s1,1,,0,false,30,40,30,0,0,0,0.0,,0,0,1000000,1000000,0,0,0,ok,,\n" +
		"2,c1,t1,standard,s1,2,,0,false,25,20,25,0,0,0,0.0,,0,0,2000000,2000000,0,0,0,ok,,\n" +
		"3,c2,t1,standard,s2,0,,0,false,60,15,60,0,0,0,0.0,,0,0,500000,500000,0,0,0,ok,,\n" +
		"4,c2,t1,standard,s2,1,,0,false,20,10,20,0,0,0,0.0,,0,0,1500000,1500000,0,0,0,ok,,\n"
	if err := os.WriteFile(dataPath, []byte(csv), 0644); err != nil {
		t.Fatal(err)
	}
	return headerPath, dataPath
}

// runFixedAccumulateReplay runs replayCmd.Run with --session-mode fixed-accumulate over
// the given corpus and captures stdout. Returns the captured stdout bytes. When
// traceOutPrefix != "", it also sets --trace-output to re-export the replay.
func runFixedAccumulateReplay(t *testing.T, headerPath, dataPath string, seedVal int64) []byte {
	return runFixedAccumulateReplayWithOutput(t, headerPath, dataPath, seedVal, "")
}

func runFixedAccumulateReplayWithOutput(t *testing.T, headerPath, dataPath string, seedVal int64, traceOutPrefix string) []byte {
	t.Helper()
	restore := captureCmdLevelVars()
	origSession := replaySessionMode
	origHeader := traceHeaderPath
	origData := traceDataPath
	origTraceOut := replayTraceOutput // captureCmdLevelVars covers run's traceOutput, not replay's
	defer func() {
		restore.restore()
		replaySessionMode = origSession
		traceHeaderPath = origHeader
		traceDataPath = origData
		replayTraceOutput = origTraceOut
	}()

	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
	model = "test-model"
	latencyModelBackend = "trained-physics"
	totalKVBlocks = 100000
	blockSizeTokens = 16
	maxNumSeqs = 64
	maxNumBatchedTokens = 4096
	numInstances = 1
	seed = seedVal
	resultsPath = ""
	longPrefillTokenThreshold = 0
	kvCPUBlocks = 0
	kvOffloadThreshold = 0.9
	kvTransferBandwidth = 100.0
	kvTransferBaseLatency = 0
	snapshotRefreshInterval = 0
	admissionPolicy = "always-admit"
	routingPolicy = "round-robin"
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 1000000
	traceLevel = "none"
	counterfactualK = 0
	traceHeaderPath = headerPath
	traceDataPath = dataPath
	modelConfigFolder = mcFolder
	hwConfigPath = hwPath
	gpu = "H100"
	tensorParallelism = 1
	defaultsFilePath = "../defaults.yaml"
	replaySessionMode = "fixed-accumulate"
	replayTraceOutput = traceOutPrefix

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	testCmd.Flags().StringVar(&replaySessionMode, "session-mode", "fixed", "")
	testCmd.Flags().IntVar(&replayConcurrentSessions, "concurrent-sessions", 0, "")
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--total-kv-blocks", "100000", "--hardware", "H100", "--tp", "1",
		"--max-model-len", "1000000",
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--trace-header", headerPath, "--trace-data", dataPath,
		"--defaults-filepath", "../defaults.yaml",
		"--session-mode", "fixed-accumulate",
	}); err != nil {
		t.Fatalf("ParseFlags failed: %v", err)
	}

	// Capture stdout.
	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	replayCmd.Run(testCmd, nil)
	_ = w.Close()
	os.Stdout = origStdout
	out, _ := io.ReadAll(r)
	return out
}

// TestReplayFixedAccumulate_EndToEnd (BC-1): a fixed-accumulate replay of an accumulate
// corpus completes every round and reports non-vacuous completions (conservation companion).
func TestReplayFixedAccumulate_EndToEnd(t *testing.T) {
	headerPath, dataPath := writeAccumulateCorpus(t)
	out := runFixedAccumulateReplay(t, headerPath, dataPath, 42)
	s := string(out)
	if len(s) == 0 {
		t.Fatal("expected non-empty stdout")
	}
	// The aggregate metrics JSON reports completed requests; all 5 rounds should complete.
	if !strings.Contains(s, "\"completed_requests\": 5") && !strings.Contains(s, "\"completed_requests\":5") {
		t.Errorf("expected 5 completed_requests in output (all rounds), got:\n%s", s)
	}
	// INV-1 conservation companion: all 5 rounds injected, and all 5 completed
	// (nothing queued/dropped) — the corpus arrivals span 2s, well within the horizon.
	if !strings.Contains(s, "\"injected_requests\": 5") && !strings.Contains(s, "\"injected_requests\":5") {
		t.Errorf("expected 5 injected_requests (INV-1 conservation), got:\n%s", s)
	}
}

// TestReplayFixedAccumulate_Deterministic (BC-4, INV-6): two fixed-accumulate replays at
// the same seed produce byte-identical stdout.
func TestReplayFixedAccumulate_Deterministic(t *testing.T) {
	headerPath, dataPath := writeAccumulateCorpus(t)
	a := runFixedAccumulateReplay(t, headerPath, dataPath, 777)
	b := runFixedAccumulateReplay(t, headerPath, dataPath, 777)
	if !bytes.Equal(a, b) {
		t.Fatalf("fixed-accumulate stdout non-deterministic across runs\nRUN A:\n%s\nRUN B:\n%s", a, b)
	}
	if len(a) == 0 {
		t.Fatal("stdout is empty; determinism test must produce output to be meaningful")
	}
}

// TestReplayFixedAccumulate_TraceOutput_AbsoluteCorpus documents and pins the
// re-export behavior flagged in PR review: fixed-accumulate reconstructs the growing
// context in memory, so --trace-output writes the ALREADY-RECONSTRUCTED ABSOLUTE
// per-round inputs. The exported header carries NO session_context_growth and the CSV
// records absolute input_tokens (not deltas), so the export is a faithful absolute-mode
// corpus: re-replay it with --session-mode fixed, NOT fixed-accumulate.
func TestReplayFixedAccumulate_TraceOutput_AbsoluteCorpus(t *testing.T) {
	headerPath, dataPath := writeAccumulateCorpus(t)
	outPrefix := filepath.Join(t.TempDir(), "reexport")
	_ = runFixedAccumulateReplayWithOutput(t, headerPath, dataPath, 42, outPrefix)

	// Load the re-exported trace and assert it is an absolute (non-accumulate) corpus.
	trace, err := workload.LoadTraceV2(outPrefix+".yaml", outPrefix+".csv")
	if err != nil {
		t.Fatalf("re-exported trace failed to load: %v", err)
	}
	if trace.Header.SessionContextGrowth != "" {
		t.Errorf("re-export header session_context_growth = %q, want empty (absolute corpus)", trace.Header.SessionContextGrowth)
	}
	// The reconstructed absolute inputs for session s1 are 100, 180, 245 (deltas
	// 100/30/25 + prev outputs 50/40). The re-export must record those absolutes, not deltas.
	byRound := map[int]int{}
	for _, rec := range trace.Records {
		if rec.SessionID == "s1" {
			byRound[rec.RoundIndex] = rec.InputTokens
		}
		if rec.InputTokensReset != nil {
			t.Errorf("re-export should carry no input_tokens_reset markers (absolute corpus), round %d has one", rec.RoundIndex)
		}
	}
	want := map[int]int{0: 100, 1: 180, 2: 245}
	for r, w := range want {
		if byRound[r] != w {
			t.Errorf("re-export s1 round %d input_tokens = %d, want absolute %d", r, byRound[r], w)
		}
	}
	// The re-export must be arrival-ordered: the run above completed without the
	// arrival-hook monotonicity panic (the writeAccumulateCorpus corpus interleaves
	// s1 and s2 in time), and the exported records are non-decreasing in arrival.
	for i := 1; i < len(trace.Records); i++ {
		if trace.Records[i].ArrivalTimeUs < trace.Records[i-1].ArrivalTimeUs {
			t.Errorf("re-export record %d arrival %d < record %d arrival %d — not arrival-ordered",
				i, trace.Records[i].ArrivalTimeUs, i-1, trace.Records[i-1].ArrivalTimeUs)
		}
	}
}

// TestReplayFixedAccumulate_RejectsConcurrentSessions (BC-5): --concurrent-sessions is
// incompatible with fixed-accumulate; replay must Fatalf.
func TestReplayFixedAccumulate_RejectsConcurrentSessions(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		headerPath, dataPath := writeAccumulateCorpus(t)
		restore := captureCmdLevelVars()
		defer restore.restore()
		mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
		model = "test-model"
		latencyModelBackend = "trained-physics"
		totalKVBlocks = 100000
		blockSizeTokens = 16
		maxNumSeqs = 64
		maxNumBatchedTokens = 4096
		numInstances = 1
		seed = 1
		admissionPolicy = "always-admit"
		routingPolicy = "round-robin"
		scheduler = "fcfs"
		maxModelLen = 1000000
		traceLevel = "none"
		traceHeaderPath = headerPath
		traceDataPath = dataPath
		modelConfigFolder = mcFolder
		hwConfigPath = hwPath
		gpu = "H100"
		tensorParallelism = 1
		defaultsFilePath = "../defaults.yaml"
		replaySessionMode = "fixed-accumulate"
		replayConcurrentSessions = 4

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
		testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
		testCmd.Flags().StringVar(&replaySessionMode, "session-mode", "fixed", "")
		testCmd.Flags().IntVar(&replayConcurrentSessions, "concurrent-sessions", 0, "")
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--total-kv-blocks", "100000", "--hardware", "H100", "--tp", "1",
			"--max-model-len", "1000000",
			"--model-config-folder", mcFolder, "--hardware-config", hwPath,
			"--trace-header", headerPath, "--trace-data", dataPath,
			"--defaults-filepath", "../defaults.yaml",
			"--session-mode", "fixed-accumulate", "--concurrent-sessions", "4",
		}); err != nil {
			fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
			os.Exit(2)
		}
		replayCmd.Run(testCmd, nil) // must Fatalf before here
		os.Exit(0)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestReplayFixedAccumulate_RejectsConcurrentSessions", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("expected non-zero exit for --concurrent-sessions + fixed-accumulate, got exit 0")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("unexpected error type: %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "concurrent-sessions") {
		t.Errorf("fatal message should mention 'concurrent-sessions', got:\n%s", out)
	}
}

// TestReplayFixedAccumulate_RejectsNonAccumulateTrace (BC-5): fixed-accumulate on a
// non-accumulate trace must Fatalf (fixed mode is correct there).
func TestReplayFixedAccumulate_RejectsNonAccumulateTrace(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "trace.yaml")
		dataPath := filepath.Join(dir, "trace.csv")
		// No session_context_growth → non-accumulate.
		_ = os.WriteFile(headerPath, []byte("trace_version: 3\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0644)
		_ = os.WriteFile(dataPath, []byte("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n"), 0644)
		restore := captureCmdLevelVars()
		defer restore.restore()
		mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
		model = "test-model"
		latencyModelBackend = "trained-physics"
		totalKVBlocks = 1000
		blockSizeTokens = 16
		maxNumSeqs = 64
		maxNumBatchedTokens = 2048
		numInstances = 1
		seed = 1
		admissionPolicy = "always-admit"
		routingPolicy = "round-robin"
		scheduler = "fcfs"
		maxModelLen = 0
		traceLevel = "none"
		traceHeaderPath = headerPath
		traceDataPath = dataPath
		modelConfigFolder = mcFolder
		hwConfigPath = hwPath
		gpu = "H100"
		tensorParallelism = 1
		defaultsFilePath = "../defaults.yaml"
		replaySessionMode = "fixed-accumulate"

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
		testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
		testCmd.Flags().StringVar(&replaySessionMode, "session-mode", "fixed", "")
		testCmd.Flags().IntVar(&replayConcurrentSessions, "concurrent-sessions", 0, "")
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
			"--model-config-folder", mcFolder, "--hardware-config", hwPath,
			"--trace-header", headerPath, "--trace-data", dataPath,
			"--defaults-filepath", "../defaults.yaml",
			"--session-mode", "fixed-accumulate",
		}); err != nil {
			fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
			os.Exit(2)
		}
		replayCmd.Run(testCmd, nil) // must Fatalf before here
		os.Exit(0)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestReplayFixedAccumulate_RejectsNonAccumulateTrace", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("expected non-zero exit for fixed-accumulate on non-accumulate trace, got exit 0")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("unexpected error type: %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "fixed-accumulate requires an accumulate corpus") {
		t.Errorf("fatal message should explain the accumulate requirement, got:\n%s", out)
	}
}

// TestReplayFixedAccumulate_RejectsThinkTime (BC-5): --think-time-ms requires closed-loop;
// fixed-accumulate uses recorded arrivals (not regenerated), so it must be rejected.
func TestReplayFixedAccumulate_RejectsThinkTime(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		headerPath, dataPath := writeAccumulateCorpus(t)
		restore := captureCmdLevelVars()
		origThink := replayThinkTimeMs
		defer func() { restore.restore(); replayThinkTimeMs = origThink }()
		mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
		model = "test-model"
		latencyModelBackend = "trained-physics"
		totalKVBlocks = 100000
		blockSizeTokens = 16
		maxNumSeqs = 64
		maxNumBatchedTokens = 4096
		numInstances = 1
		seed = 1
		admissionPolicy = "always-admit"
		routingPolicy = "round-robin"
		scheduler = "fcfs"
		maxModelLen = 1000000
		traceLevel = "none"
		traceHeaderPath = headerPath
		traceDataPath = dataPath
		modelConfigFolder = mcFolder
		hwConfigPath = hwPath
		gpu = "H100"
		tensorParallelism = 1
		defaultsFilePath = "../defaults.yaml"
		replaySessionMode = "fixed-accumulate"
		replayThinkTimeMs = 500

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
		testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
		testCmd.Flags().StringVar(&replaySessionMode, "session-mode", "fixed", "")
		testCmd.Flags().IntVar(&replayConcurrentSessions, "concurrent-sessions", 0, "")
		testCmd.Flags().IntVar(&replayThinkTimeMs, "think-time-ms", 0, "")
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--total-kv-blocks", "100000", "--hardware", "H100", "--tp", "1",
			"--max-model-len", "1000000",
			"--model-config-folder", mcFolder, "--hardware-config", hwPath,
			"--trace-header", headerPath, "--trace-data", dataPath,
			"--defaults-filepath", "../defaults.yaml",
			"--session-mode", "fixed-accumulate", "--think-time-ms", "500",
		}); err != nil {
			fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
			os.Exit(2)
		}
		replayCmd.Run(testCmd, nil) // must Fatalf before here
		os.Exit(0)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestReplayFixedAccumulate_RejectsThinkTime", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("expected non-zero exit for --think-time-ms + fixed-accumulate, got exit 0")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("unexpected error type: %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "think-time-ms requires --session-mode closed-loop") {
		t.Errorf("fatal message should explain think-time requires closed-loop, got:\n%s", out)
	}
}
