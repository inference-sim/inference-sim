package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// TestPlanDPPlacement is the pure-function contract for DP-as-real-placement
// (#1531). It verifies the decision (BC-1) and the unsupported-combo guards
// (BC-7) without touching any package state, so it survives a rewrite of the
// runCmd wiring that applies the plan.
func TestPlanDPPlacement(t *testing.T) {
	tests := []struct {
		name             string
		isMoE            bool
		dp               int
		epOn             bool
		pdActive         bool
		autoscalerActive bool
		wantActive       bool
		wantReplicas     int
		wantPerRankDP    int
		wantErrContains  string // non-empty ⇒ expect an error containing this substring
	}{
		{
			name:          "default dp=1 MoE is a no-op",
			isMoE:         true,
			dp:            1,
			wantActive:    false,
			wantReplicas:  1,
			wantPerRankDP: 1,
		},
		{
			name:          "dense dp>1 is a no-op here (rejected upstream)",
			isMoE:         false,
			dp:            4,
			wantActive:    false,
			wantReplicas:  1,
			wantPerRankDP: 4,
		},
		{
			name:          "MoE dp>1 EP-off no-PD no-autoscaler expands to dp replicas at DP=1",
			isMoE:         true,
			dp:            4,
			wantActive:    true,
			wantReplicas:  4,
			wantPerRankDP: 1,
		},
		{
			name:            "MoE dp>1 with expert parallel is guarded (→#1548)",
			isMoE:           true,
			dp:              2,
			epOn:            true,
			wantErrContains: "1548",
		},
		{
			name:            "MoE dp>1 with PD disaggregation is guarded",
			isMoE:           true,
			dp:              2,
			pdActive:        true,
			wantErrContains: "disaggregation",
		},
		{
			name:             "MoE dp>1 with autoscaler is guarded",
			isMoE:            true,
			dp:               2,
			autoscalerActive: true,
			wantErrContains:  "autoscaler",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := planDPPlacement(tc.isMoE, tc.dp, tc.epOn, tc.pdActive, tc.autoscalerActive)
			if tc.wantErrContains != "" {
				if err == nil {
					t.Fatalf("expected an error containing %q, got nil (plan=%+v)", tc.wantErrContains, plan)
				}
				if !strings.Contains(err.Error(), tc.wantErrContains) {
					t.Errorf("error should mention %q, got: %v", tc.wantErrContains, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if plan.Active != tc.wantActive {
				t.Errorf("Active: got %v, want %v", plan.Active, tc.wantActive)
			}
			if plan.Replicas != tc.wantReplicas {
				t.Errorf("Replicas: got %d, want %d", plan.Replicas, tc.wantReplicas)
			}
			if plan.PerRankDP != tc.wantPerRankDP {
				t.Errorf("PerRankDP: got %d, want %d", plan.PerRankDP, tc.wantPerRankDP)
			}
		})
	}
}

// replayDPSubprocess drives the real replayCmd with an MoE model + --dp 2 against
// a minimal generated trace, so the #1531 run-only guard fires (logrus.Fatalf →
// exit 1). Reached only inside the re-exec subprocess.
func replayDPSubprocess() {
	dir, err := os.MkdirTemp("", "replaydp")
	if err != nil {
		os.Exit(2)
	}
	mcDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0755); err != nil {
		os.Exit(2)
	}
	// Minimal MoE config (num_local_experts > 1 ⇒ IsMoE); --total-kv-blocks is set
	// explicitly below so the auto-capacity path (needing vocab_size etc.) is skipped.
	moeConfig := `{
  "architectures": ["MixtralForCausalLM"],
  "num_attention_heads": 4,
  "num_hidden_layers": 2,
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_key_value_heads": 4,
  "num_local_experts": 8,
  "num_experts_per_tok": 2,
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`
	if err := os.WriteFile(filepath.Join(mcDir, "config.json"), []byte(moeConfig), 0644); err != nil {
		os.Exit(2)
	}
	hwPath := filepath.Join(dir, "hw.json")
	if err := os.WriteFile(hwPath, []byte(`{"H100": {"MemoryGiB": 80.0, "TFlopsPeak": 989.5, "BwPeakTBs": 3.35}}`), 0644); err != nil {
		os.Exit(2)
	}
	headerPath := filepath.Join(dir, "trace.yaml")
	if err := os.WriteFile(headerPath, []byte("trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0644); err != nil {
		os.Exit(2)
	}
	dataPath := filepath.Join(dir, "trace.csv")
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n" +
		"1,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,100000,100000,0,0,0,ok,,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		os.Exit(2)
	}

	model = "test-model"
	latencyModelBackend = "trained-physics"
	gpu = "H100"
	tensorParallelism = 1
	dataParallelism = 2
	enableExpertParallel = false
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxModelLen = 0
	maxRunningReqs = 256
	maxScheduledTokens = 2048
	longPrefillTokenThreshold = 0
	gpuMemoryUtilization = 0.9
	numInstances = 1
	modelConfigFolder = mcDir
	hwConfigPath = hwPath
	traceHeaderPath = headerPath
	traceDataPath = dataPath
	defaultsFilePath = "../defaults.yaml"
	simulationHorizon = math.MaxInt64

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	if perr := testCmd.ParseFlags([]string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1", "--dp", "2",
		"--model-config-folder", mcDir, "--hardware-config", hwPath,
		"--trace-header", headerPath, "--trace-data", dataPath,
		"--defaults-filepath", "../defaults.yaml",
	}); perr != nil {
		os.Exit(2)
	}
	replayCmd.Run(testCmd, nil)
	os.Exit(0) // reached only if the run-only guard did NOT fire
}

// TestReplayCmd_MoEDPPlacement_Rejected verifies BC-3 / INV-13: `blis replay`
// fails fast (logrus.Fatalf, exit 1) on an MoE model with --dp > 1, because
// DP-as-placement is a run-only feature. Driven in a re-exec subprocess because
// the guard exits the process.
func TestReplayCmd_MoEDPPlacement_Rejected(t *testing.T) {
	if os.Getenv("BLIS_REPLAY_DP_SUBPROCESS") == "1" {
		replayDPSubprocess()
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestReplayCmd_MoEDPPlacement_Rejected")
	cmd.Env = append(os.Environ(), "BLIS_REPLAY_DP_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected non-zero exit (Fatalf) for MoE --dp>1 replay, got exit 0; output:\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("unexpected error type: %v; output:\n%s", err, out)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "blis run") || !strings.Contains(string(out), "run-only") {
		t.Errorf("fatal message should say the feature is run-only and point to blis run; got:\n%s", out)
	}
}

// dpRunArgs builds the offline-safe `blis run` args for an MoE (deepseek-v2-lite)
// DP-as-placement integration run. Paths are relative to the cmd/ test cwd.
func dpRunArgs(numInstances, dp int) []string {
	return []string{
		"run",
		"--model", "deepseek-ai/deepseek-v2-lite",
		"--model-config-folder", "../model_configs/deepseek-v2-lite",
		"--hardware", "H100",
		"--hardware-config", "../hardware_config.json",
		"--tp", "1",
		"--dp", strconv.Itoa(dp),
		"--num-instances", strconv.Itoa(numInstances),
		"--rate", "10",
		"--num-requests", "40",
		"--total-kv-blocks", "20000",
		"--seed", "42",
		"--defaults-filepath", "../defaults.yaml",
	}
}

// runBlisRunSubprocess re-execs this test binary in a subprocess that runs the
// real `blis run` command (rootCmd) with the given args and returns its stdout.
// The subprocess pattern is required because the run path may logrus.Fatalf, and
// os.Exit(0) suppresses the test framework's own stdout for a clean capture.
func runBlisRunSubprocess(t *testing.T, testName string, numInstances, dp int) string {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^"+testName+"$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_DP_SUBPROCESS=1",
		"BLIS_RUN_DP_NUMINST="+strconv.Itoa(numInstances), "BLIS_RUN_DP_DP="+strconv.Itoa(dp))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("subprocess `blis run` failed: %v\nstderr:\n%s", err, stderr.String())
	}
	return stdout.String()
}

// extractJSONObjects returns the substrings of s that are top-level, balanced
// {...} objects, skipping the non-JSON "=== Simulation Metrics ===" preambles
// blis interleaves between per-instance and aggregate metric dumps. String
// contents (which may contain braces) are respected.
func extractJSONObjects(s string) []string {
	var objs []string
	depth, start := 0, -1
	inStr, esc := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			depth--
			if depth == 0 && start >= 0 {
				objs = append(objs, s[start:i+1])
				start = -1
			}
		}
	}
	return objs
}

// clusterConservationHolds parses the aggregate ("cluster") metrics object from
// blis run stdout and checks INV-1 for a default-admission run (no flow control,
// no routing rejections): injected == completed + queued + running + dropped +
// timed_out, with injected > 0.
func clusterConservationHolds(t *testing.T, stdout string) {
	t.Helper()
	found := false
	for _, raw := range extractJSONObjects(stdout) {
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(raw), &obj); err != nil {
			continue
		}
		if obj["instance_id"] != "cluster" {
			continue
		}
		found = true
		num := func(k string) int {
			v, ok := obj[k].(float64)
			if !ok {
				t.Fatalf("cluster metrics missing numeric field %q", k)
			}
			return int(v)
		}
		injected := num("injected_requests")
		sum := num("completed_requests") + num("still_queued") + num("still_running") +
			num("dropped_unservable") + num("timed_out_requests")
		if injected <= 0 {
			t.Errorf("INV-1: expected injected_requests > 0, got %d", injected)
		}
		if injected != sum {
			t.Errorf("INV-1 violated: injected=%d != completed+queued+running+dropped+timedout=%d", injected, sum)
		}
	}
	if !found {
		t.Fatalf("no cluster aggregate metrics object found in stdout:\n%s", stdout)
	}
}

// TestRunCmd_MoEDPPlacement_SpawnsReplicas verifies BC-1/BC-4/BC-5/BC-8: MoE
// `--dp N` on `blis run` spawns numInstances × N real engine replicas, request
// conservation holds across them (INV-1), and stdout is deterministic (INV-6).
// Uses the git-tracked deepseek-v2-lite MoE fixture (offline).
func TestRunCmd_MoEDPPlacement_SpawnsReplicas(t *testing.T) {
	if os.Getenv("BLIS_RUN_DP_SUBPROCESS") == "1" {
		ni, _ := strconv.Atoi(os.Getenv("BLIS_RUN_DP_NUMINST"))
		dp, _ := strconv.Atoi(os.Getenv("BLIS_RUN_DP_DP"))
		rootCmd.SetArgs(dpRunArgs(ni, dp))
		_ = rootCmd.Execute()
		os.Exit(0)
	}

	// Case A: --num-instances 1 --dp 2 → exactly 2 replicas (instance_0, instance_1).
	outA := runBlisRunSubprocess(t, "TestRunCmd_MoEDPPlacement_SpawnsReplicas", 1, 2)
	if !strings.Contains(outA, `"instance_id": "instance_1"`) {
		t.Errorf("BC-1: expected a second replica instance_1 with --num-instances 1 --dp 2; stdout:\n%s", outA)
	}
	if strings.Contains(outA, `"instance_id": "instance_2"`) {
		t.Errorf("BC-1: expected exactly 2 replicas (dp=2), but found instance_2")
	}
	clusterConservationHolds(t, outA) // BC-4

	// Case B: --num-instances 2 --dp 2 → 4 replicas (M×N), confirming the M>1 multiply.
	outB := runBlisRunSubprocess(t, "TestRunCmd_MoEDPPlacement_SpawnsReplicas", 2, 2)
	if !strings.Contains(outB, `"instance_id": "instance_3"`) {
		t.Errorf("BC-8: expected instance_3 with --num-instances 2 --dp 2 (M×N=4); stdout:\n%s", outB)
	}
	if strings.Contains(outB, `"instance_id": "instance_4"`) {
		t.Errorf("BC-8: expected exactly 4 replicas (2×2), but found instance_4")
	}

	// BC-5 (INV-6): a repeat of Case A is byte-identical.
	outA2 := runBlisRunSubprocess(t, "TestRunCmd_MoEDPPlacement_SpawnsReplicas", 1, 2)
	if outA != outA2 {
		t.Errorf("INV-6: two identical DP-placement runs produced different stdout")
	}
}

// TestDPPlacement_PerRankKV_NoDoubleCount is the BC-2 law: with DP-as-placement,
// each replica is sized with the per-rank (dp=1) KV budget and the aggregate over
// the dp replicas equals the lumped single-instance dp-multiplied total — never
// dp²·perRank. It resolves the same MoE fixture auto-KV at dp=1 and dp=2 (the
// auto-capacity path scales the total by dp; #1420 / kv_capacity.go Step 6), then
// applies the run-body per-rank division and checks the two laws directly.
// writeCompleteMoEFixture writes a complete MoE config.json (with vocab_size and
// realistic dims so the KV auto-capacity path yields a positive block count on an
// 80 GiB GPU) plus a hardware config, returning their paths.
func writeCompleteMoEFixture(t *testing.T) (mcDir, hwPath string) {
	t.Helper()
	dir := t.TempDir()
	mcDir = filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	configJSON := `{
  "architectures": ["MixtralForCausalLM"],
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_key_value_heads": 8,
  "num_local_experts": 8,
  "num_experts_per_tok": 2,
  "vocab_size": 32000,
  "hidden_act": "silu",
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`
	if err := os.WriteFile(filepath.Join(mcDir, "config.json"), []byte(configJSON), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	hwPath = filepath.Join(dir, "hw.json")
	if err := os.WriteFile(hwPath, []byte(`{"H100": {"MemoryGiB": 80.0, "TFlopsPeak": 989.5, "BwPeakTBs": 3.35}}`), 0644); err != nil {
		t.Fatalf("write hw: %v", err)
	}
	return mcDir, hwPath
}

func TestDPPlacement_PerRankKV_NoDoubleCount(t *testing.T) {
	mcDir, hwPath := writeCompleteMoEFixture(t)

	resolveAutoKV := func(dp int) int64 {
		model = "test-model"
		latencyModelBackend = "trained-physics"
		gpu = "H100"
		tensorParallelism = 2 // TP=2 so the 8x7B MoE weights fit in 80 GiB per GPU
		dataParallelism = dp
		enableExpertParallel = false
		moeCommBackend = ""
		totalKVBlocks = 0 // auto-derive
		blockSizeTokens = 16
		maxModelLen = 0
		gpuMemoryUtilization = 0.9
		modelConfigFolder = mcDir
		hwConfigPath = hwPath
		defaultsFilePath = "../defaults.yaml"

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		// No --total-kv-blocks ⇒ the auto-capacity path (CalculateKVBlocks with dp) runs.
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--hardware", "H100", "--tp", "2", "--dp", strconv.Itoa(dp),
			"--model-config-folder", mcDir, "--hardware-config", hwPath,
			"--defaults-filepath", "../defaults.yaml",
		}); err != nil {
			t.Fatalf("dp=%d ParseFlags: %v", dp, err)
		}
		resolveLatencyConfig(testCmd)
		return totalKVBlocks
	}

	perRank := resolveAutoKV(1) // a single dp=1 rank's budget
	lumped := resolveAutoKV(2)  // today's dp-multiplied single-instance total
	if perRank <= 0 {
		t.Fatalf("dp=1 auto KV capacity must be positive, got %d", perRank)
	}

	// The run body divides the dp-scaled auto total back to one rank when spawning
	// dp replicas (the "capacity calc receives dp=1" outcome).
	perReplica := lumped / 2

	// Law 1: each replica is sized exactly like a dp=1 rank (no residue).
	if perReplica != perRank {
		t.Errorf("BC-2: per-replica KV (%d) must equal the dp=1 per-rank budget (%d)", perReplica, perRank)
	}
	// Law 2: aggregate over the dp replicas equals the lumped total — no dp² double-count.
	if perReplica*2 != lumped {
		t.Errorf("BC-2: aggregate KV (perReplica×dp = %d) must equal the lumped dp-multiplied total (%d); "+
			"a dp² double-count would give %d", perReplica*2, lumped, lumped*2)
	}
}

// TestDPPlacement_ExplicitKV_SkipsPerRankDivision covers the BC-2 explicit-KV
// branch: the run-body per-rank division is gated on `lr.KVParamsOK && MemoryGiB>0`
// (auto-calc succeeded ⇒ the total was dp-scaled), so an explicit --total-kv-blocks
// (which sets KVParamsOK=false) is NOT divided — each replica keeps the operator's
// per-instance value (aggregate dp×value). This asserts the gating signal directly:
// explicit ⇒ KVParamsOK false (no division); auto ⇒ KVParamsOK true (division applies).
func TestDPPlacement_ExplicitKV_SkipsPerRankDivision(t *testing.T) {
	mcDir, hwPath := writeCompleteMoEFixture(t)

	resolve := func(explicitKV bool) latencyResolution {
		model = "test-model"
		latencyModelBackend = "trained-physics"
		gpu = "H100"
		tensorParallelism = 2
		dataParallelism = 2
		enableExpertParallel = false
		moeCommBackend = ""
		totalKVBlocks = 0
		blockSizeTokens = 16
		maxModelLen = 0
		gpuMemoryUtilization = 0.9
		modelConfigFolder = mcDir
		hwConfigPath = hwPath
		defaultsFilePath = "../defaults.yaml"

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		args := []string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--hardware", "H100", "--tp", "2", "--dp", "2",
			"--model-config-folder", mcDir, "--hardware-config", hwPath,
			"--defaults-filepath", "../defaults.yaml",
		}
		if explicitKV {
			args = append(args, "--total-kv-blocks", "12345")
		}
		if err := testCmd.ParseFlags(args); err != nil {
			t.Fatalf("ParseFlags: %v", err)
		}
		return resolveLatencyConfig(testCmd)
	}

	// Explicit --total-kv-blocks ⇒ auto-calc skipped ⇒ KVParamsOK false ⇒ the run body
	// does NOT divide (each replica keeps the operator's value).
	if lrExplicit := resolve(true); lrExplicit.KVParamsOK {
		t.Errorf("explicit --total-kv-blocks must yield KVParamsOK=false (division skipped), got true")
	}
	if totalKVBlocks != 12345 {
		t.Errorf("explicit --total-kv-blocks must be preserved unchanged, got %d", totalKVBlocks)
	}
	// Auto-calc (no --total-kv-blocks) with a valid GPU ⇒ KVParamsOK true ⇒ the run
	// body divides by dp to yield the per-rank budget.
	if lrAuto := resolve(false); !lrAuto.KVParamsOK {
		t.Errorf("auto KV path must yield KVParamsOK=true (division applies), got false")
	}
}
