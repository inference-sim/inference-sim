package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/spf13/cobra"
)

// TestPlanDPPlacement is the pure-function contract for DP-as-real-placement
// (#1531). It verifies the decision (BC-1) and the unsupported-combo guards
// (BC-7) without touching any package state, so it survives a rewrite of the
// command wiring (resolveDPPlacement) that applies the plan.
func TestPlanDPPlacement(t *testing.T) {
	tests := []struct {
		name             string
		isMoE            bool
		dp               int
		epOn             bool
		pdActive         bool
		autoscalerActive bool
		nodePoolsActive  bool
		wantActive       bool
		wantReplicas     int
		wantPerRankDP    int
		wantEPGroupDP    int    // logical EP-group DP width the plan must carry (#1548); 0 = none
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
			// #1548 lifted this rejection: expert parallelism reserves no GPUs beyond the
			// N×TP the DP placement already takes, so the PLAN is identical to EP-off. What
			// EP changes is how experts map onto that group, which travels separately as the
			// logical EP-group DP width (epGroupDPForPlacement), not in the plan.
			name:          "MoE dp>1 with expert parallel is allowed and plans identically (#1548)",
			isMoE:         true,
			dp:            2,
			epOn:          true,
			wantActive:    true,
			wantReplicas:  2,
			wantPerRankDP: 1,
			wantEPGroupDP: 2, // the one thing EP adds: the logical group width to carry
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
		{
			name:            "MoE dp>1 with node pools is guarded",
			isMoE:           true,
			dp:              2,
			nodePoolsActive: true,
			wantErrContains: "node pools",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := planDPPlacement(tc.isMoE, tc.dp, tc.epOn, tc.pdActive, tc.autoscalerActive, tc.nodePoolsActive)
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
			// #1548: the logical EP-group width must survive PerRankDP's erasure of DP —
			// and must be absent (0 ⇒ no option) whenever expert parallelism is off, which
			// is what keeps every pre-#1548 config byte-identical.
			if plan.EPGroupDP != tc.wantEPGroupDP {
				t.Errorf("EPGroupDP: got %d, want %d", plan.EPGroupDP, tc.wantEPGroupDP)
			}
			if opts := plan.EPGroupOptions(); (len(opts) > 0) != (tc.wantEPGroupDP > 1) {
				t.Errorf("EPGroupOptions() returned %d options for EPGroupDP=%d; a width of 0 or 1 "+
					"must yield none (INV-6)", len(opts), plan.EPGroupDP)
			}
		})
	}
}

// TestApplyDPPlacement is the BC-2 formula contract for the production per-rank KV
// division, instance expansion, and per-rank max-model-len re-cap. It exercises
// applyDPPlacement directly (the exact statement resolveDPPlacement calls), so
// deleting, inverting, or mis-gating any of the three fails here — including the dp²
// double-count BC-2 exists to prevent. Pure, so it survives a rewrite of the command
// wiring that applies it.
func TestApplyDPPlacement(t *testing.T) {
	const blockSize int64 = 16
	active4 := dpPlacementPlan{Active: true, Replicas: 4, PerRankDP: 1}
	inactive := dpPlacementPlan{Active: false, Replicas: 1, PerRankDP: 1}

	tests := []struct {
		name            string
		plan            dpPlacementPlan
		dp              int
		in              dpPlacementDeployment
		autoScaledKV    bool
		wantErrContains string
		want            dpPlacementDeployment
	}{
		{
			// The no-op law that makes the feature byte-identical when unused (INV-6):
			// every quantity survives untouched.
			name:         "inactive plan is the identity on all three quantities",
			plan:         inactive,
			dp:           1,
			in:           dpPlacementDeployment{NumInstances: 3, TotalKVBlocks: 5000, MaxModelLen: 1_000_000},
			autoScaledKV: true,
			want:         dpPlacementDeployment{NumInstances: 3, TotalKVBlocks: 5000, MaxModelLen: 1_000_000},
		},
		{
			// Auto-KV: the incoming total is the dp-multiplied aggregate, so it divides
			// back to one rank; max-model-len is re-capped to that smaller budget.
			name:         "auto-KV divides to per-rank, expands the count, re-caps max-model-len",
			plan:         active4,
			dp:           4,
			in:           dpPlacementDeployment{NumInstances: 2, TotalKVBlocks: 40000, MaxModelLen: 1_000_000},
			autoScaledKV: true,
			// 8 replicas (2×4), 10000 blocks each (40000/4), max-model-len 10000×16.
			want: dpPlacementDeployment{NumInstances: 8, TotalKVBlocks: 10000, MaxModelLen: 160000},
		},
		{
			// A max-model-len that already fits the per-rank budget must NOT be raised.
			name:         "auto-KV leaves a feasible max-model-len alone",
			plan:         active4,
			dp:           4,
			in:           dpPlacementDeployment{NumInstances: 1, TotalKVBlocks: 40000, MaxModelLen: 4096},
			autoScaledKV: true,
			want:         dpPlacementDeployment{NumInstances: 4, TotalKVBlocks: 10000, MaxModelLen: 4096},
		},
		{
			// An explicit --total-kv-blocks is already per-instance: no division, and no
			// re-cap either (no aggregate was ever used as the cap).
			name:         "explicit KV keeps the operator value and never re-caps",
			plan:         active4,
			dp:           4,
			in:           dpPlacementDeployment{NumInstances: 1, TotalKVBlocks: 12345, MaxModelLen: 1_000_000},
			autoScaledKV: false,
			want:         dpPlacementDeployment{NumInstances: 4, TotalKVBlocks: 12345, MaxModelLen: 1_000_000},
		},
		{
			// The division floors: a --dp bigger than the auto-derived block count would
			// leave 0 blocks per replica (NewSimulator panics) and a kvFeasibleMax of 0
			// would silently mean "unlimited" — the inverse of a cap. Must error, and
			// must leave the deployment untouched so a caller that ignored the error
			// cannot run a half-applied plan.
			name:            "auto-KV division to zero blocks errors instead of panicking downstream",
			plan:            dpPlacementPlan{Active: true, Replicas: 8, PerRankDP: 1},
			dp:              8,
			in:              dpPlacementDeployment{NumInstances: 1, TotalKVBlocks: 5, MaxModelLen: 4096},
			autoScaledKV:    true,
			wantErrContains: "exceeds the auto-derived KV capacity",
			want:            dpPlacementDeployment{NumInstances: 1, TotalKVBlocks: 5, MaxModelLen: 4096},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := applyDPPlacement(tc.plan, tc.dp, tc.in, tc.autoScaledKV, blockSize)
			if tc.wantErrContains != "" {
				if err == nil {
					t.Fatalf("expected an error containing %q, got nil", tc.wantErrContains)
				}
				if !strings.Contains(err.Error(), tc.wantErrContains) {
					t.Errorf("error %q must mention %q", err.Error(), tc.wantErrContains)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("deployment: got %+v, want %+v", got, tc.want)
			}
		})
	}

	// The conservation law, stated independently of the table: the aggregate KV over all
	// spawned replicas equals the pre-#1531 lumped total (logical instances × the
	// dp-multiplied total). A dp² double-count would inflate it by dp.
	const inNumInst, inTotalKV = 2, int64(40000)
	got, err := applyDPPlacement(active4, 4, dpPlacementDeployment{NumInstances: inNumInst, TotalKVBlocks: inTotalKV}, true, blockSize)
	if err != nil {
		t.Fatalf("conservation case: unexpected error: %v", err)
	}
	if int64(got.NumInstances)*got.TotalKVBlocks != int64(inNumInst)*inTotalKV {
		t.Errorf("aggregate KV (%d×%d=%d) must equal the lumped total (%d×%d=%d); a dp² double-count would give %d",
			got.NumInstances, got.TotalKVBlocks, int64(got.NumInstances)*got.TotalKVBlocks,
			inNumInst, inTotalKV, int64(inNumInst)*inTotalKV, int64(inNumInst)*inTotalKV*4)
	}
}

// TestDPPlacement_PerRankDP_ConfiguresConstructor is the behavioral companion to
// the source-level wiring guard: it proves the plan's PerRankDP, threaded through
// the canonical NewModelHardwareConfig, yields a config that reports DP=1 and
// moeGroup=TP (experts replicated per rank — EP-off physics) for an active MoE
// plan, and leaves DP=1 for the dp=1 no-op. Refactor-safe (asserts observable
// config behavior, not source text).
func TestDPPlacement_PerRankDP_ConfiguresConstructor(t *testing.T) {
	moe := sim.ModelConfig{NumLocalExperts: 8} // >= MoEMinExperts ⇒ IsMoE
	hw := sim.HardwareCalib{}
	const tp = 2

	// Active plan (MoE, dp=4): PerRankDP=1 ⇒ each replica's config is DP=1, moeGroup=TP.
	planActive, err := planDPPlacement(true, 4, false, false, false, false)
	if err != nil {
		t.Fatalf("planDPPlacement(active): %v", err)
	}
	mhcActive := sim.NewModelHardwareConfig(moe, hw, "m", "H100", tp, planActive.PerRankDP, false, "", "trained-physics", 0)
	if mhcActive.EffectiveDP() != 1 {
		t.Errorf("active plan: EffectiveDP got %d, want 1 (per-rank)", mhcActive.EffectiveDP())
	}
	if mhcActive.EffectiveMoEGroupSize() != tp {
		t.Errorf("active plan: EffectiveMoEGroupSize got %d, want %d (TP; experts replicated per DP rank)",
			mhcActive.EffectiveMoEGroupSize(), tp)
	}

	// dp=1 no-op: PerRankDP=1 ⇒ unchanged DP=1 behavior.
	planNoop, err := planDPPlacement(true, 1, false, false, false, false)
	if err != nil {
		t.Fatalf("planDPPlacement(noop): %v", err)
	}
	mhcNoop := sim.NewModelHardwareConfig(moe, hw, "m", "H100", tp, planNoop.PerRankDP, false, "", "trained-physics", 0)
	if mhcNoop.EffectiveDP() != 1 {
		t.Errorf("dp=1 no-op: EffectiveDP got %d, want 1", mhcNoop.EffectiveDP())
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

// dpRunBaseArgs returns the offline-safe `blis run` args for the deepseek-v2-lite
// MoE fixture (paths relative to the cmd/ test cwd), minus the DP/KV/topology
// flags each test appends.
func dpRunBaseArgs() []string {
	return []string{
		"run",
		"--model", "deepseek-ai/deepseek-v2-lite",
		"--model-config-folder", "../model_configs/deepseek-v2-lite",
		"--hardware", "H100",
		"--hardware-config", "../hardware_config.json",
		"--tp", "1",
		"--rate", "10",
		"--num-requests", "40",
		"--seed", "42",
		"--defaults-filepath", "../defaults.yaml",
	}
}

// TestRunCmd_MoEDPPlacement_AutoKV_NoPanic exercises the auto-KV path
// (KVParamsOK=true) end-to-end — the production per-rank division in resolveDPPlacement
// actually fires (Issue #1531 review Finding 2) — and confirms the max-model-len
// re-cap prevents the per-replica "KV cache too small for MaxModelLen" panic
// (Finding 1). A huge --max-model-len is capped to the aggregate by
// resolveLatencyConfig, then must be re-capped to the per-rank budget after the
// division; without the re-cap each replica's NewSimulator panics (non-zero exit).
func TestRunCmd_MoEDPPlacement_AutoKV_NoPanic(t *testing.T) {
	if os.Getenv("BLIS_RUN_DP_AUTOKV") == "1" {
		args := append(dpRunBaseArgs(),
			"--dp", "2", "--num-instances", "1",
			"--max-model-len", "10000000", // forces the per-rank re-cap after auto-KV division
		)
		rootCmd.SetArgs(args)
		_ = rootCmd.Execute()
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_MoEDPPlacement_AutoKV_NoPanic$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_DP_AUTOKV=1")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("auto-KV DP-placement run must not panic/fatal (per-rank max-model-len re-cap); err=%v\nstderr:\n%s",
			err, stderr.String())
	}
	out := stdout.String()
	// instance_1 present ⇒ the auto path reached applyDPPlacement (expansion + division).
	if !strings.Contains(out, `"instance_id": "instance_1"`) {
		t.Errorf("auto-KV: expected 2 replicas (instance_1 present); stdout:\n%s", out)
	}
	clusterConservationHolds(t, out)
}

// TestRunCmd_MoEDP1_ByteIdentical is the BC-6 (INV-6 no-op) system guard: an MoE
// run with --dp 1 (the default, planDPPlacement inactive) is deterministic across
// runs. Catches a future regression that adds nondeterministic code to the DP path.
func TestRunCmd_MoEDP1_ByteIdentical(t *testing.T) {
	if os.Getenv("BLIS_RUN_DP1") == "1" {
		args := append(dpRunBaseArgs(), "--dp", "1", "--num-instances", "1", "--total-kv-blocks", "20000")
		rootCmd.SetArgs(args)
		_ = rootCmd.Execute()
		os.Exit(0)
	}
	run := func() string {
		cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_MoEDP1_ByteIdentical$")
		cmd.Env = append(os.Environ(), "BLIS_RUN_DP1=1")
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			t.Fatalf("--dp 1 run failed: %v\nstderr:\n%s", err, stderr.String())
		}
		return stdout.String()
	}
	first, second := run(), run()
	if first != second {
		t.Errorf("BC-6/INV-6: two --dp 1 MoE runs produced different stdout")
	}
}

// TestRunCmd_MoEDPPlacement_GuardedCombo_Rejected is the BC-7 system guard: a
// planDPPlacement error for a still-unsupported combo (PD disaggregation + MoE --dp>1,
// #1553) is actually converted to a logrus.Fatalf by runCmd (exit 1), not merely
// returned. Complements the pure-function TestPlanDPPlacement guard cases.
//
// It used to exercise --enable-expert-parallel; #1548 made that combination SUPPORTED
// (see TestRunCmd_MoEDPPlacement_EPOn_Runs), so the system-level "the error really does
// terminate" coverage moved to a combo that is still guarded.
func TestRunCmd_MoEDPPlacement_GuardedCombo_Rejected(t *testing.T) {
	if os.Getenv("BLIS_RUN_DP_EPGUARD") == "1" {
		args := append(dpRunBaseArgs(),
			// --num-instances must cover the pools, or the PD topology check fatals first
			// on its own (unrelated) message before planDPPlacement is reached.
			"--dp", "2", "--num-instances", "2", "--total-kv-blocks", "20000",
			"--prefill-instances", "1", "--decode-instances", "1",
		)
		rootCmd.SetArgs(args)
		_ = rootCmd.Execute()
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_MoEDPPlacement_GuardedCombo_Rejected$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_DP_EPGUARD=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected non-zero exit (Fatalf) for PD disaggregation + MoE --dp>1, got exit 0; output:\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1 (logrus.Fatalf), got %v; output:\n%s", err, out)
	}
	if !strings.Contains(string(out), "#1553") {
		t.Errorf("PD guard message should reference #1553 (hashed, as production writes it); got:\n%s", out)
	}
}

// TestRunCmd_MoEDPPlacement_EPOn_Runs is BC-1 at the system level: MoE --dp N with
// --enable-expert-parallel — rejected before #1548 — now completes, spawns exactly the
// same num_instances × N replicas as the EP-off run (expert parallelism reserves NO extra
// GPUs; the EP group IS those replicas' GPUs), and conserves requests (INV-1).
func TestRunCmd_MoEDPPlacement_EPOn_Runs(t *testing.T) {
	if os.Getenv("BLIS_RUN_EP_PLACEMENT") == "1" {
		args := append(dpRunBaseArgs(),
			"--dp", "2", "--num-instances", "2", "--total-kv-blocks", "20000",
			"--enable-expert-parallel", "--latency-model", "trained-physics",
		)
		rootCmd.SetArgs(args)
		if err := rootCmd.Execute(); err != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_MoEDPPlacement_EPOn_Runs$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_EP_PLACEMENT=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("MoE --dp 2 --enable-expert-parallel must now run (#1548), got %v; output:\n%s", err, out)
	}
	got := string(out)
	// 2 logical × dp 2 = 4 replicas — the same count the EP-off plan produces, which is
	// AC-1: expert parallelism reserves no GPUs beyond the ones DP placement already took.
	if !strings.Contains(got, `"instance_id": "instance_3"`) {
		t.Errorf("BC-1: expected 4 engine replicas (2 logical × --dp 2), same as EP-off; output:\n%s", got)
	}
	if strings.Contains(got, `"instance_id": "instance_4"`) {
		t.Errorf("BC-1: expected exactly 4 replicas, but instance_4 is present")
	}
	// The unpriced inter-replica fabric must be disclosed, not silently optimistic (R1).
	if !strings.Contains(got, "inter-replica fabric cost is NOT priced") {
		t.Errorf("expected the unpriced inter-replica fabric disclosure; output:\n%s", got)
	}
	clusterConservationHolds(t, got) // INV-1
}

// TestDPPlacement_PerRankKV_NoDoubleCount is the BC-2 law: with DP-as-placement,
// each replica is sized with the per-rank (dp=1) KV budget and the aggregate over
// the dp replicas equals the lumped single-instance dp-multiplied total — never
// dp²·perRank. It resolves the same MoE fixture auto-KV at dp=1 and dp=2 (the
// auto-capacity path scales the total by dp; #1420 / kv_capacity.go Step 6), then
// applies the shared resolver's per-rank division and checks the two laws directly.
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

	// resolveDPPlacement divides the dp-scaled auto total back to one rank when spawning
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
// branch: the shared resolver's per-rank division is gated on `lr.KVParamsOK && MemoryGiB>0`
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

	// Explicit --total-kv-blocks ⇒ auto-calc skipped ⇒ KVParamsOK false ⇒ the resolver
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

// dpResolveVars snapshots the DP-relevant package-level CLI vars that
// captureCmdLevelVars does not cover, so a resolveDPPlacement test can set them
// freely and restore them afterwards.
type dpResolveVars struct {
	dp, prefill, decode, prefillDecode, encode int
	epOn                                       bool
	commBackend                                string
}

func captureDPResolveVars() dpResolveVars {
	return dpResolveVars{
		dp: dataParallelism, prefill: prefillInstances, decode: decodeInstances,
		prefillDecode: prefillDecodeInstances, encode: encodeInstances,
		epOn: enableExpertParallel, commBackend: moeCommBackend,
	}
}

func (o dpResolveVars) restore() {
	dataParallelism = o.dp
	prefillInstances = o.prefill
	decodeInstances = o.decode
	prefillDecodeInstances = o.prefillDecode
	encodeInstances = o.encode
	enableExpertParallel = o.epOn
	moeCommBackend = o.commBackend
}

// dpMoELatencyResolution builds a minimal latencyResolution describing an MoE model on
// an 80 GiB GPU with the auto-KV path having succeeded (KVParamsOK). That combination
// is the gate resolveDPPlacement uses to decide the incoming --total-kv-blocks is the
// dp-multiplied aggregate and must be divided back to one rank.
func dpMoELatencyResolution(autoKV bool) latencyResolution {
	return latencyResolution{
		ModelConfig: sim.ModelConfig{NumLocalExperts: 8},
		HWConfig:    sim.HardwareCalib{MemoryGiB: 80.0},
		KVParamsOK:  autoKV,
	}
}

// TestResolveDPPlacement_MutatesDeploymentVars is the contract for the shared resolver
// both `blis run` and `blis replay` call (#1556). resolveDPPlacement deliberately reads
// and writes the cmd/ package flag vars itself (like resolveLatencyConfig and
// resolvePolicies) so that neither command body carries wiring that could drift — this
// test therefore states the parity law at the only place it now lives: what the shared
// resolver does to numInstances / totalKVBlocks / maxModelLen, and what it refuses.
//
// NOTE: mutates package-level vars — must NOT use t.Parallel().
func TestResolveDPPlacement_MutatesDeploymentVars(t *testing.T) {
	tests := []struct {
		name            string
		dp              int
		epOn            bool
		prefill         int
		autoKV          bool
		autoscaler      bool
		nodePools       bool
		inNumInstances  int
		inTotalKV       int64
		inMaxModelLen   int64
		wantErrContains string
		// wantPerRankDP is only checked on the success rows: when wantErrContains is set
		// the returned plan is the zero value and is not asserted, so those rows leave
		// this field at Go's 0 rather than stating an expectation.
		wantPerRankDP   int
		wantNumInst     int
		wantTotalKV     int64
		wantMaxModelLen int64
	}{
		{
			// Active: 2 logical × dp 4 = 8 replicas, KV divided back to one rank, and
			// max-model-len re-capped to the per-rank budget (10000 blocks × 16 tokens).
			name: "active plan expands the count, divides KV, re-caps max-model-len",
			dp:   4, autoKV: true,
			inNumInstances: 2, inTotalKV: 40000, inMaxModelLen: 1_000_000,
			wantPerRankDP: 1, wantNumInst: 8, wantTotalKV: 10000, wantMaxModelLen: 160000,
		},
		{
			// The INV-6 no-op: --dp 1 must leave every var byte-for-byte as it was.
			name: "dp=1 mutates nothing",
			dp:   1, autoKV: true,
			inNumInstances: 2, inTotalKV: 40000, inMaxModelLen: 1_000_000,
			wantPerRankDP: 1, wantNumInst: 2, wantTotalKV: 40000, wantMaxModelLen: 1_000_000,
		},
		{
			// #1548: EP-on is no longer a guard. It mutates the deployment vars EXACTLY as
			// the EP-off active plan does (same row values as "active plan expands the
			// count..." above), because expert parallelism reserves no extra GPUs.
			name: "EP-on is allowed and mutates identically to EP-off",
			dp:   4, epOn: true, autoKV: true,
			inNumInstances: 2, inTotalKV: 40000, inMaxModelLen: 1_000_000,
			wantPerRankDP: 1, wantNumInst: 8, wantTotalKV: 10000, wantMaxModelLen: 160000,
		},
		{
			name: "PD guard errors and mutates nothing",
			dp:   4, prefill: 1, autoKV: true,
			inNumInstances: 2, inTotalKV: 40000, inMaxModelLen: 1_000_000,
			wantErrContains: "#1553",
			wantNumInst:     2, wantTotalKV: 40000, wantMaxModelLen: 1_000_000,
		},
		{
			name: "autoscaler guard errors and mutates nothing",
			dp:   4, autoscaler: true, autoKV: true,
			inNumInstances: 2, inTotalKV: 40000, inMaxModelLen: 1_000_000,
			wantErrContains: "#1553",
			wantNumInst:     2, wantTotalKV: 40000, wantMaxModelLen: 1_000_000,
		},
		{
			name: "node-pool guard errors and mutates nothing",
			dp:   4, nodePools: true, autoKV: true,
			inNumInstances: 2, inTotalKV: 40000, inMaxModelLen: 1_000_000,
			wantErrContains: "#1553",
			wantNumInst:     2, wantTotalKV: 40000, wantMaxModelLen: 1_000_000,
		},
		{
			// Explicit --total-kv-blocks (KVParamsOK false): per-instance already, so the
			// value is preserved on every replica and max-model-len is not re-capped.
			name: "explicit KV is preserved and max-model-len is not re-capped",
			dp:   4, autoKV: false,
			inNumInstances: 1, inTotalKV: 12345, inMaxModelLen: 1_000_000,
			wantPerRankDP: 1, wantNumInst: 4, wantTotalKV: 12345, wantMaxModelLen: 1_000_000,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			origCmd := captureCmdLevelVars()
			defer origCmd.restore()
			origDP := captureDPResolveVars()
			defer origDP.restore()

			dataParallelism = tc.dp
			enableExpertParallel = tc.epOn
			prefillInstances = tc.prefill
			decodeInstances, prefillDecodeInstances, encodeInstances = 0, 0, 0
			moeCommBackend = ""
			numInstances = tc.inNumInstances
			totalKVBlocks = tc.inTotalKV
			maxModelLen = tc.inMaxModelLen
			blockSizeTokens = 16

			plan, err := resolveDPPlacement(dpMoELatencyResolution(tc.autoKV), tc.autoscaler, tc.nodePools)
			if tc.wantErrContains != "" {
				if err == nil {
					t.Fatalf("expected an error containing %q, got nil", tc.wantErrContains)
				}
				if !strings.Contains(err.Error(), tc.wantErrContains) {
					t.Errorf("error %q must reference %q", err.Error(), tc.wantErrContains)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if plan.PerRankDP != tc.wantPerRankDP {
					t.Errorf("PerRankDP: got %d, want %d", plan.PerRankDP, tc.wantPerRankDP)
				}
			}
			if numInstances != tc.wantNumInst {
				t.Errorf("numInstances: got %d, want %d", numInstances, tc.wantNumInst)
			}
			if totalKVBlocks != tc.wantTotalKV {
				t.Errorf("totalKVBlocks: got %d, want %d", totalKVBlocks, tc.wantTotalKV)
			}
			if maxModelLen != tc.wantMaxModelLen {
				t.Errorf("maxModelLen: got %d, want %d", maxModelLen, tc.wantMaxModelLen)
			}
		})
	}
}

// TestResolveDPPlacement_DenseModelIsInert pins the dense no-op: a non-MoE model must
// leave every deployment var alone even at --dp > 1 (dense dp>1 is rejected earlier, in
// resolveLatencyConfig, so here it must simply not expand).
func TestResolveDPPlacement_DenseModelIsInert(t *testing.T) {
	origCmd := captureCmdLevelVars()
	defer origCmd.restore()
	origDP := captureDPResolveVars()
	defer origDP.restore()

	dataParallelism = 4
	enableExpertParallel = false
	prefillInstances, decodeInstances, prefillDecodeInstances, encodeInstances = 0, 0, 0, 0
	moeCommBackend = ""
	numInstances, totalKVBlocks, maxModelLen, blockSizeTokens = 2, 40000, 1_000_000, 16

	dense := latencyResolution{
		ModelConfig: sim.ModelConfig{NumLocalExperts: 0}, // dense
		HWConfig:    sim.HardwareCalib{MemoryGiB: 80.0},
		KVParamsOK:  true,
	}
	plan, err := resolveDPPlacement(dense, false, false)
	if err != nil {
		t.Fatalf("dense model must not error: %v", err)
	}
	if plan.Active {
		t.Errorf("dense model must not activate DP-as-placement")
	}
	if plan.PerRankDP != 4 {
		t.Errorf("dense PerRankDP must stay the CLI --dp (4), got %d", plan.PerRankDP)
	}
	if numInstances != 2 || totalKVBlocks != 40000 || maxModelLen != 1_000_000 {
		t.Errorf("dense model must mutate nothing, got (%d, %d, %d)", numInstances, totalKVBlocks, maxModelLen)
	}
}
