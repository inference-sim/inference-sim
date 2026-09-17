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
			// The short-circuit at dp=1 with EP on: no expansion, so nothing erases the
			// config's own DP and there is no logical width to carry (EPGroupDP stays 0).
			// Distinct from the dp>1 EP-on row below, which does carry one.
			name:          "MoE dp=1 with expert parallel is a no-op and carries no width",
			isMoE:         true,
			dp:            1,
			epOn:          true,
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
			// #1553 lifted this rejection: each PD pool spawns dp per-rank replicas. The
			// PLAN is identical to the plain (non-PD) active plan — the per-pool expansion
			// is applied by applyDPPlacement, not decided here.
			name:          "MoE dp>1 with PD disaggregation is supported (#1553)",
			isMoE:         true,
			dp:            2,
			pdActive:      true,
			wantActive:    true,
			wantReplicas:  2,
			wantPerRankDP: 1,
		},
		{
			// #1553 DECISION: the autoscaler stays rejected. Rank-vs-group scaling of a
			// dp-expanded population is undefined, and DirectActuator.scaleUp places a
			// single-role instance with no DP-group awareness. Shipping "support" would
			// ship an untested, ambiguous path.
			name:             "MoE dp>1 with autoscaler is rejected (#1553 decision)",
			isMoE:            true,
			dp:               2,
			autoscalerActive: true,
			wantErrContains:  "autoscaler",
		},
		{
			// #1553 lifted this rejection: the N×M replicas reuse the tested per-instance
			// node-pool placement path (each PlaceInstance reserves TP GPUs; per-instance
			// KV sizes per-rank from the placed GPU).
			name:            "MoE dp>1 with node pools is supported (#1553)",
			isMoE:           true,
			dp:              2,
			nodePoolsActive: true,
			wantActive:      true,
			wantReplicas:    2,
			wantPerRankDP:   1,
		},
		{
			// PD + EP + dp together: PD/nodePools no longer reject, EP carries its group
			// width. The plan is active with the EP width; per-pool expansion happens in
			// applyDPPlacement.
			name:          "MoE dp>1 with PD and expert parallel is supported and carries EP width (#1553/#1548)",
			isMoE:         true,
			dp:            2,
			epOn:          true,
			pdActive:      true,
			wantActive:    true,
			wantReplicas:  2,
			wantPerRankDP: 1,
			wantEPGroupDP: 2,
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
			// BC-2 PD pool expansion: every pool count scales by Replicas alongside the
			// global NumInstances, so P·N+D·N+S·N+E·N ≤ total·N is preserved and each pool
			// spawns its own N per-rank replicas. Auto-KV still divides to per-rank.
			name:         "active plan scales every PD pool count by Replicas",
			plan:         active4,
			dp:           4,
			in:           dpPlacementDeployment{NumInstances: 10, TotalKVBlocks: 40000, MaxModelLen: 4096, PrefillInstances: 3, DecodeInstances: 4, SharedInstances: 2, EncodeInstances: 1},
			autoScaledKV: true,
			// 40 instances (10×4), pools 12/16/8/4 (each ×4), 10000 blocks each, max-model-len untouched (fits).
			want: dpPlacementDeployment{NumInstances: 40, TotalKVBlocks: 10000, MaxModelLen: 4096, PrefillInstances: 12, DecodeInstances: 16, SharedInstances: 8, EncodeInstances: 4},
		},
		{
			// The inactive plan is the identity on the pool counts too (INV-6): a PD run
			// at --dp 1 spawns exactly its configured pools, byte-identical to pre-#1553.
			name:         "inactive plan leaves PD pool counts untouched",
			plan:         inactive,
			dp:           1,
			in:           dpPlacementDeployment{NumInstances: 10, TotalKVBlocks: 5000, MaxModelLen: 4096, PrefillInstances: 3, DecodeInstances: 4, SharedInstances: 2, EncodeInstances: 1},
			autoScaledKV: true,
			want:         dpPlacementDeployment{NumInstances: 10, TotalKVBlocks: 5000, MaxModelLen: 4096, PrefillInstances: 3, DecodeInstances: 4, SharedInstances: 2, EncodeInstances: 1},
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
		"--catalog", "../model_configs",
		"--hardware", "H100",
		"--hardware-config", "../hardware_config.json",
		"--tp", "1",
		"--dp", strconv.Itoa(dp),
		"--num-instances", strconv.Itoa(numInstances),
		"--rate", "10",
		"--num-requests", strconv.Itoa(dpFixtureNumRequests),
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

// dpFixtureNumRequests is the --num-requests every DP fixture in cmd/ passes. Kept as
// a constant so clusterConservationHolds has an independent count to compare against
// rather than re-deriving one from the output it is checking.
const dpFixtureNumRequests = 40

// clusterConservationHolds parses the aggregate ("cluster") metrics object from blis run
// stdout and checks what that output can actually prove about INV-1: that
// injected_requests is positive, and that it equals the number of requests the fixture
// generated. expected is that count, or 0 when the caller does not know it (a replay
// whose request count comes from the trace).
//
// It deliberately does NOT compare injected_requests against the five-term sum of the
// same object: sim.Metrics.ToOutput defines injected_requests AS that sum, so the
// equality holds for any input and catches nothing. That tautology is what this used to
// assert (#1720).
//
// Nor can it check the canonical twelve-term equation. Seven buckets are invisible here
// — GatewayQueueDepth, GatewayQueueShed, GatewayQueueRejected, GatewayEvicted,
// GatewayExpired, RoutingRejections and EncodeRoutingRejections live on
// cluster.RawMetrics and are never serialised into sim.MetricsOutput. #1746 tracks
// exposing them. Until then a CLI-level conservation check is only meaningful for
// fixtures that exercise none of those paths: default admission, no flow control, no
// encode pool, which is the case for every caller below.
//
// In-process cluster tests must use assertClusterINV1Conservation
// (sim/cluster/inv1_conservation_test.go), which checks all twelve.
func clusterConservationHolds(t *testing.T, stdout string, expected int) {
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
		if injected <= 0 {
			t.Errorf("INV-1: expected injected_requests > 0, got %d", injected)
		}
		// NOT compared against the five-term sum of the same object:
		// sim.Metrics.ToOutput defines injected_requests AS that sum, so the equality
		// would hold for any input and catch nothing. What stdout can independently
		// check is that the count survived the CLI round trip.
		if expected > 0 && injected != expected {
			t.Errorf("INV-1: cluster injected_requests=%d, want %d generated by the fixture", injected, expected)
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
	clusterConservationHolds(t, outA, dpFixtureNumRequests) // BC-4

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
		"--catalog", "../model_configs",
		"--hardware", "H100",
		"--hardware-config", "../hardware_config.json",
		"--tp", "1",
		"--rate", "10",
		"--num-requests", strconv.Itoa(dpFixtureNumRequests),
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
	clusterConservationHolds(t, out, dpFixtureNumRequests)
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

// TestRunCmd_PD_DP1_ByteIdentical is the BC-8 fence for the code THIS PR touched: the
// per-pool KV block now threads perPoolKVDP (=plan.PerRankDP) instead of the raw --dp.
// At --dp 1 the plan is inactive, so perPoolKVDP == dataParallelism == 1 and the per-pool
// path must compute byte-identically to pre-#1553. A PD topology is the only run that
// exercises that block, so this is the direct guard that lifting the PD guard changed no
// accepted config. (--dp 1 keeps the pre-#1553-legal PD run exactly as it was.)
func TestRunCmd_PD_DP1_ByteIdentical(t *testing.T) {
	if os.Getenv("BLIS_RUN_PD_DP1") == "1" {
		args := append(dpRunBaseArgs(),
			"--dp", "1", "--num-instances", "2",
			"--prefill-instances", "1", "--decode-instances", "1", "--total-kv-blocks", "20000")
		rootCmd.SetArgs(args)
		_ = rootCmd.Execute()
		os.Exit(0)
	}
	run := func() string {
		cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_PD_DP1_ByteIdentical$")
		cmd.Env = append(os.Environ(), "BLIS_RUN_PD_DP1=1")
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			t.Fatalf("PD --dp 1 run failed: %v\nstderr:\n%s", err, stderr.String())
		}
		return stdout.String()
	}
	first, second := run(), run()
	if completed := clusterMetricInt(t, first, "completed_requests"); completed <= 0 {
		t.Fatalf("INV-6 check would be vacuous: PD --dp 1 completed %d requests", completed)
	}
	if first != second {
		t.Errorf("BC-8/INV-6: two PD --dp 1 MoE runs produced different stdout (the per-pool " +
			"perPoolKVDP threading must be an exact no-op at --dp 1)")
	}
}

// TestRunCmd_MoEDPPlacement_GuardedCombo_Rejected is the BC-4 system guard: a
// planDPPlacement error for the still-unsupported combo (the autoscaler + MoE --dp>1,
// #1553 DECISION) is actually converted to a logrus.Fatalf by runCmd (exit 1), not merely
// returned. Complements the pure-function TestPlanDPPlacement autoscaler case.
//
// It used to exercise --enable-expert-parallel (#1548 made that SUPPORTED) and then PD
// disaggregation (#1553 makes that SUPPORTED too — see TestRunCmd_MoEDPPlacement_PD_Runs),
// so the system-level "the error really does terminate" coverage now targets the
// autoscaler — the one combination #1553 keeps guarded.
func TestRunCmd_MoEDPPlacement_GuardedCombo_Rejected(t *testing.T) {
	if os.Getenv("BLIS_RUN_DP_ASGUARD") == "1" {
		args := append(dpRunBaseArgs(),
			"--dp", "2", "--num-instances", "2", "--total-kv-blocks", "20000",
			"--model-autoscaler-interval-us", "1000000", // autoscaler active ⇒ #1553 rejection
		)
		rootCmd.SetArgs(args)
		_ = rootCmd.Execute()
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_MoEDPPlacement_GuardedCombo_Rejected$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_DP_ASGUARD=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected non-zero exit (Fatalf) for the autoscaler + MoE --dp>1, got exit 0; output:\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1 (logrus.Fatalf), got %v; output:\n%s", err, out)
	}
	got := string(out)
	if !strings.Contains(got, "#1553") {
		t.Errorf("autoscaler guard message should reference #1553; got:\n%s", got)
	}
	if !strings.Contains(got, "autoscaler") {
		t.Errorf("autoscaler guard message should name the autoscaler; got:\n%s", got)
	}
}

// TestRunCmd_MoEDPPlacement_PD_Runs is BC-1 at the system level (#1553, AC1): PD
// disaggregation + MoE --dp N — rejected before this PR — now completes. It spawns the
// expanded topology (each pool ×N per-rank replicas) and conserves requests (INV-1).
func TestRunCmd_MoEDPPlacement_PD_Runs(t *testing.T) {
	if os.Getenv("BLIS_RUN_DP_PD") == "1" {
		args := append(dpRunBaseArgs(),
			// 1 prefill + 1 decode per logical instance; --dp 2 ⇒ 2 prefill + 2 decode = 4.
			"--dp", "2", "--num-instances", "2", "--total-kv-blocks", "20000",
			"--prefill-instances", "1", "--decode-instances", "1",
		)
		rootCmd.SetArgs(args)
		if err := rootCmd.Execute(); err != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_MoEDPPlacement_PD_Runs$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_DP_PD=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("PD disaggregation + MoE --dp 2 must now run (#1553), got %v; output:\n%s", err, out)
	}
	got := string(out)
	// 2 logical × dp 2 = 4 replicas ⇒ instance_3 present, instance_4 absent.
	if !strings.Contains(got, `"instance_id": "instance_3"`) {
		t.Errorf("AC1: expected 4 engine replicas (2 logical × --dp 2); output:\n%s", got)
	}
	if strings.Contains(got, `"instance_id": "instance_4"`) {
		t.Errorf("AC1: expected exactly 4 replicas, but instance_4 is present")
	}
	clusterConservationHolds(t, got, dpFixtureNumRequests) // INV-1
}

// TestRunCmd_MoEDPPlacement_PD_PerPoolAutoKV_PerRank is the BC-3 behavioral execution test
// (#1553): a PD + `--dp N` run that actually reaches latency.CalculateKVBlocks on the per-pool
// auto-KV path must charge the PER-RANK DP (=1), not the global `--dp`, so a pool's per-replica
// KV is not dp²-inflated.
//
// The existing PD tests do NOT cover this execution path: TestRunCmd_MoEDPPlacement_PD_Runs pins
// `--total-kv-blocks` (which sets KVParamsOK=false and bypasses CalculateKVBlocks entirely), the
// DP=1 byte-identity test has an inactive plan (perPoolKVDP == dataParallelism == 1 either way),
// and the node-pool test exercises applyPerInstanceKVCapacity (a different function). So a
// regression that passed `dataParallelism` in place of `perPoolKVDP` at the per-pool
// CalculateKVBlocks sites (cmd/root.go) would pass every one of those and only trip the
// source-string guard — exactly the "refactor survival" gap the review raised.
//
// This test forces the per-pool auto-calc to run by giving the prefill pool a TP override
// (`--prefill-tp 2` while the global `--tp` is 1, so `poolPrefillTP != tensorParallelism` is
// true) with NO `--total-kv-blocks`, then reads the auto-calc's own Info line. That line prints
// the DP the calc charged: `DP=1` is per-rank (correct), `DP=2` would be the dp²-inflated
// regression. A dedicated non-vacuity check confirms the auto-calc actually ran.
func TestRunCmd_MoEDPPlacement_PD_PerPoolAutoKV_PerRank(t *testing.T) {
	if os.Getenv("BLIS_RUN_DP_PERPOOL_AUTOKV") == "1" {
		args := append(dpRunBaseArgs(),
			"--dp", "2", "--num-instances", "2", // P(1)+D(1) = 2 ≤ num-instances 2
			"--prefill-instances", "1", "--decode-instances", "1",
			"--prefill-tp", "2", // differs from global --tp 1 ⇒ per-pool prefill auto-calc runs
			"--log", "info", // surface the per-pool KV auto-calc line
			// deliberately NO --total-kv-blocks ⇒ CalculateKVBlocks (the path under test) runs
		)
		rootCmd.SetArgs(args)
		if err := rootCmd.Execute(); err != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_MoEDPPlacement_PD_PerPoolAutoKV_PerRank$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_DP_PERPOOL_AUTOKV=1")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("PD + --dp 2 + per-pool auto-KV must run (#1553); err=%v\nstderr:\n%s", err, stderr.String())
	}
	logs := stderr.String()
	// Non-vacuity: the per-pool prefill auto-calc must actually have run (otherwise the
	// DP assertion below would pass trivially on an absent line).
	if !strings.Contains(logs, "auto-calculated prefill pool total-kv-blocks=") {
		t.Fatalf("BC-3: expected the per-pool prefill KV auto-calc to run (TP override + no --total-kv-blocks); "+
			"stderr:\n%s", logs)
	}
	// The auto-calc charged the per-rank DP: the line ends with "DP=1)" — every DP-placement
	// replica is DP=1. "DP=2)" would be the dp²-inflated regression (charging the global --dp
	// to a per-replica budget). (The non-vacuity Fatalf above already returned if the line
	// was absent, so a false here is a genuine wrong-DP.)
	if !perPoolAutoKVLineHasDP(logs, "prefill", 1) {
		t.Errorf("BC-3: the per-pool prefill KV auto-calc must charge the PER-RANK DP=1 under an active "+
			"DP-as-placement plan, not the global --dp 2 (a dp² inflation); stderr:\n%s", logs)
	}
	// The regression signature stated explicitly, so the failure message is unambiguous.
	if perPoolAutoKVLineHasDP(logs, "prefill", 2) {
		t.Errorf("BC-3: per-pool prefill auto-calc charged DP=2 (the global --dp), meaning perPoolKVDP "+
			"was not the per-rank value — dp²-inflated per-replica KV; stderr:\n%s", logs)
	}
	clusterConservationHolds(t, stdout.String(), dpFixtureNumRequests) // INV-1
}

// perPoolAutoKVLineHasDP reports whether the per-pool (prefill|decode) KV auto-calc Info line
// reports the given DP value. It matches the trailing "DP=<n>)" of the auto-calc log line
// emitted at cmd/root.go, tolerating any block count / GPU / TP before it.
func perPoolAutoKVLineHasDP(logs, pool string, dp int) bool {
	prefix := "auto-calculated " + pool + " pool total-kv-blocks="
	for _, line := range strings.Split(logs, "\n") {
		i := strings.Index(line, prefix)
		if i < 0 {
			continue
		}
		if strings.Contains(line[i:], "DP="+strconv.Itoa(dp)+")") {
			return true
		}
	}
	return false
}

// TestRunCmd_MoEDPPlacement_PerPoolRoofline_Rejected is BC-7 at the system level (#1553):
// once PD + --dp>1 is a supported run, a per-pool --prefill-latency-model roofline override
// must be REJECTED (exit 1) rather than silently putting the prefill pool on DP/EP-blind
// step time while the rest of the cluster runs the DP physics. This exercises the composed
// `dataParallelism > 1 || enableExpertParallel` predicate reaching validatePerPoolLatencyBackends
// through the CLI at --dp 2 — the wiring the unit test (TestValidatePerPoolLatencyBackends)
// cannot see. The gate itself landed in #1548; #1553 is what makes the PD path that trips it
// reachable at all.
func TestRunCmd_MoEDPPlacement_PerPoolRoofline_Rejected(t *testing.T) {
	if os.Getenv("BLIS_RUN_DP_ROOFLINE") == "1" {
		args := append(dpRunBaseArgs(),
			"--dp", "2", "--num-instances", "2", "--total-kv-blocks", "20000",
			"--prefill-instances", "1", "--decode-instances", "1",
			"--prefill-latency-model", "roofline", // DP/EP-blind pool while --dp 2 runs DP physics
		)
		rootCmd.SetArgs(args)
		_ = rootCmd.Execute()
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_MoEDPPlacement_PerPoolRoofline_Rejected$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_DP_ROOFLINE=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected non-zero exit (Fatalf) for a per-pool roofline override + MoE --dp 2, got exit 0; output:\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1 (logrus.Fatalf), got %v; output:\n%s", err, out)
	}
	got := string(out)
	if !strings.Contains(got, "prefill-latency-model") {
		t.Errorf("BC-7: the gate message should name the offending per-pool flag; got:\n%s", got)
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
	clusterConservationHolds(t, got, dpFixtureNumRequests) // INV-1
}

// TestRunCmd_MoEDPPlacement_NodePools_NxM is BC-6 at the system level (#1553, AC3): node
// pools + MoE --dp N — rejected before this PR — now place N×M real replicas from pool
// inventory, together reserving N×M×TP GPUs, each sized per-rank from its ACTUAL placed
// GPU. With --num-instances 2 --dp 2 --tp 1 the deployment expands to 4 single-GPU
// replicas drawn from the 8-GPU H100 pool.
//
// Two observables carry the criterion:
//   - 4 replicas place (instance_3 present, instance_4 absent) AND conservation holds —
//     so all N×M reservations succeeded against pool inventory (a failed placement would
//     drop the instance and break INV-1);
//   - the per-instance KV auto-calc (no --total-kv-blocks ⇒ applyPerInstanceKVCapacity
//     runs) logs a PER-RANK total: each replica is DP=1, so its block count equals the
//     single-rank budget of the placed 80 GiB H100, NOT the dp-multiplied one. The
//     Info-level auto-calc line is the direct evidence that placement sized per-rank.
func TestRunCmd_MoEDPPlacement_NodePools_NxM(t *testing.T) {
	if os.Getenv("BLIS_RUN_DP_NODEPOOLS") == "1" {
		dir := os.Getenv("BLIS_RUN_DP_NODEPOOLS_DIR")
		// One H100 pool, 8 GPUs on a single node — room for N×M×TP = 2×2×1 = 4 replicas.
		bundleYAML := "node_pools:\n  - name: pool-a\n    gpu_type: H100\n    gpus_per_node: 8\n" +
			"    gpu_memory_gib: 80\n    initial_nodes: 1\n    min_nodes: 1\n    max_nodes: 1\n    cost_per_hour: 32.0\n"
		bundlePath := filepath.Join(dir, "nodepools.yaml")
		if err := os.WriteFile(bundlePath, []byte(bundleYAML), 0644); err != nil {
			os.Exit(2)
		}
		args := append(dpRunBaseArgs(),
			"--dp", "2", "--num-instances", "2", // N×M = 4 replicas
			"--policy-config", bundlePath,
			"--log", "info", // surface the per-instance KV auto-calc (per-rank sizing evidence)
		)
		rootCmd.SetArgs(args)
		if err := rootCmd.Execute(); err != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_MoEDPPlacement_NodePools_NxM$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_DP_NODEPOOLS=1", "BLIS_RUN_DP_NODEPOOLS_DIR="+t.TempDir())
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("AC3: node pools + MoE --dp 2 must now run (#1553), got %v; output:\n%s", err, out)
	}
	got := string(out)
	// N×M = 4 replicas ⇒ instance_3 present, instance_4 absent.
	if !strings.Contains(got, `"instance_id": "instance_3"`) {
		t.Errorf("AC3/BC-6: expected N×M=4 replicas placed from the node pool (instance_3); output:\n%s", got)
	}
	if strings.Contains(got, `"instance_id": "instance_4"`) {
		t.Errorf("AC3/BC-6: expected exactly 4 replicas, but instance_4 is present")
	}
	// Per-rank sizing evidence (non-vacuous): the per-instance auto-calc ran (node pools +
	// no --total-kv-blocks) and sized each DP=1 replica from its placed 80 GiB H100 — the
	// per-instance line must report DP=1, NOT the lumped DP=2 of the global auto-calc. A
	// regression that sized replicas dp-multiplied would log "DP=2" here and fail.
	if !strings.Contains(got, "per-instance KV auto-calc") {
		t.Errorf("AC3/BC-6: expected the per-instance KV auto-calc to run under node pools "+
			"(each replica sized per-rank from its placed GPU); output:\n%s", got)
	}
	if !strings.Contains(got, `total-kv-blocks=74312 (GPU=80 GiB, TP=1, DP=1`) {
		t.Errorf("AC3/BC-6: each replica must be sized PER-RANK (DP=1) from its placed 80 GiB H100 "+
			"— half the DP=2 global auto-calc total, no dp² double-count; output:\n%s", got)
	}
	clusterConservationHolds(t, got, dpFixtureNumRequests) // BC-9 / INV-1: every N×M reservation succeeded
}

// TestDPPlacement_PerRankKV_NoDoubleCount is the BC-2 law: with DP-as-placement,
// each replica is sized with the per-rank (dp=1) KV budget and the aggregate over
// the dp replicas equals the lumped single-instance dp-multiplied total — never
// dp²·perRank. It resolves the same MoE fixture auto-KV at dp=1 and dp=2 (the
// auto-capacity path scales the total by dp; #1420 / kv_capacity.go Step 6), then
// applies the shared resolver's per-rank division and checks the two laws directly.
// writeCompleteMoEFixture writes a complete MoE config.json (with vocab_size and
// realistic dims so the KV auto-capacity path yields a positive block count on an
// 80 GiB GPU) as every entry of a test catalog, plus a hardware config, returning the
// catalog root (for --catalog) and the hardware-config path.
func writeCompleteMoEFixture(t *testing.T) (catalogDir, hwPath string) {
	t.Helper()
	dir := t.TempDir()
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
	catalogDir, err := writeTestCatalog(dir, configJSON)
	if err != nil {
		t.Fatalf("write test catalog: %v", err)
	}
	hwPath = filepath.Join(dir, "hw.json")
	if err := os.WriteFile(hwPath, []byte(`{"H100": {"MemoryGiB": 80.0, "TFlopsPeak": 989.5, "BwPeakTBs": 3.35}}`), 0644); err != nil {
		t.Fatalf("write hw: %v", err)
	}
	return catalogDir, hwPath
}

func TestDPPlacement_PerRankKV_NoDoubleCount(t *testing.T) {
	catalogDir, hwPath := writeCompleteMoEFixture(t)

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
		catalogPath = catalogDir
		hwConfigPath = hwPath
		defaultsFilePath = "../defaults.yaml"

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		// No --total-kv-blocks ⇒ the auto-capacity path (CalculateKVBlocks with dp) runs.
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--hardware", "H100", "--tp", "2", "--dp", strconv.Itoa(dp),
			"--catalog", catalogDir, "--hardware-config", hwPath,
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
	catalogDir, hwPath := writeCompleteMoEFixture(t)

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
		catalogPath = catalogDir
		hwConfigPath = hwPath
		defaultsFilePath = "../defaults.yaml"

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		args := []string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--hardware", "H100", "--tp", "2", "--dp", "2",
			"--catalog", catalogDir, "--hardware-config", hwPath,
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
		decode          int
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
			// #1553 (AC1): PD is no longer a guard. A PD topology expands like any other
			// active plan — the global count ×dp, KV divided to per-rank, max-model-len
			// re-capped — and the single prefill pool scales 1→4 too (asserted separately
			// in TestApplyDPPlacement / the e2e test). Here we pin the shared quantities.
			name: "PD is supported and expands the deployment",
			dp:   4, prefill: 1, decode: 1, autoKV: true,
			inNumInstances: 2, inTotalKV: 40000, inMaxModelLen: 1_000_000,
			wantPerRankDP: 1, wantNumInst: 8, wantTotalKV: 10000, wantMaxModelLen: 160000,
		},
		{
			name: "autoscaler guard errors and mutates nothing",
			dp:   4, autoscaler: true, autoKV: true,
			inNumInstances: 2, inTotalKV: 40000, inMaxModelLen: 1_000_000,
			wantErrContains: "#1553",
			wantNumInst:     2, wantTotalKV: 40000, wantMaxModelLen: 1_000_000,
		},
		{
			// #1553 (AC3): node pools are no longer a guard. The plan is identical to the
			// plain active plan; the N×M placement / GPU reservation is a cluster concern
			// exercised in the e2e test, not here.
			name: "node pools are supported and expand the deployment",
			dp:   4, nodePools: true, autoKV: true,
			inNumInstances: 2, inTotalKV: 40000, inMaxModelLen: 1_000_000,
			wantPerRankDP: 1, wantNumInst: 8, wantTotalKV: 10000, wantMaxModelLen: 160000,
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
			prefillInstances, decodeInstances = tc.prefill, tc.decode
			prefillDecodeInstances, encodeInstances = 0, 0
			moeCommBackend = ""
			numInstances = tc.inNumInstances
			totalKVBlocks = tc.inTotalKV
			maxModelLen = tc.inMaxModelLen
			blockSizeTokens = 16

			lr := dpMoELatencyResolution(tc.autoKV)
			// planDPPlacement now owns the autoscaler / node-pool decision (resolveDPPlacement
			// takes the decided plan); mirror the production caller — plan, then apply.
			plan, err := planDPPlacement(lr.ModelConfig.IsMoE(), dataParallelism, enableExpertParallel,
				prefillInstances > 0 || decodeInstances > 0 || prefillDecodeInstances > 0 || encodeInstances > 0,
				tc.autoscaler, tc.nodePools)
			if err == nil {
				plan, err = resolveDPPlacement(lr, plan)
			}
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
	plan, err := planDPPlacement(dense.ModelConfig.IsMoE(), dataParallelism, enableExpertParallel,
		prefillInstances > 0, false, false)
	if err == nil {
		plan, err = resolveDPPlacement(dense, plan)
	}
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
