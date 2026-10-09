package cmd

import (
	"encoding/json"
	"errors"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"pgregory.net/rapid"
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
	mhcActive := sim.NewModelHardwareConfig(moe, hw, "m", "H100", tp, planActive.PerRankDP, false, "", sim.LatencyBackendKernel, 0)
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
	mhcNoop := sim.NewModelHardwareConfig(moe, hw, "m", "H100", tp, planNoop.PerRankDP, false, "", sim.LatencyBackendKernel, 0)
	if mhcNoop.EffectiveDP() != 1 {
		t.Errorf("dp=1 no-op: EffectiveDP got %d, want 1", mhcNoop.EffectiveDP())
	}
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

// dpScenario is the vendored MoE scenario stating dp=2 with expert parallelism: the kernel
// sets dataParallelism=2, and DP-as-placement runs one engine replica per rank.
const dpScenario = "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml"

// dpRunArgs is a `blis run` of the dp=2 MoE scenario over numInstances logical instances.
func dpRunArgs(numInstances int, extra ...string) []string {
	return append([]string{"run", "--scenario", dpScenario,
		"--num-instances", strconv.Itoa(numInstances),
		"--rate", "10", "--num-requests", strconv.Itoa(dpFixtureNumRequests), "--seed", "42",
	}, extra...)
}

// instanceIDs returns the per-instance ids a run reported, in output order.
func instanceIDs(t *testing.T, stdout string) []string {
	t.Helper()
	var ids []string
	for _, raw := range extractJSONObjects(stdout) {
		var obj map[string]any
		if err := json.Unmarshal([]byte(raw), &obj); err != nil {
			continue
		}
		if id, ok := obj["instance_id"].(string); ok && id != "cluster" {
			ids = append(ids, id)
		}
	}
	return ids
}

// TestRunCmd_MoEDPPlacement_SpawnsReplicas: a scenario stating dp=N runs numInstances x N real
// engine replicas -- the replica count is a law of the two inputs, not of the kernel's prices
// -- requests are conserved across them (INV-1), the unpriced inter-replica fabric of the
// expert-parallel group is disclosed, and a repeat run is byte-identical (INV-6).
func TestRunCmd_MoEDPPlacement_SpawnsReplicas(t *testing.T) {
	out, stderr, err := runKernelCLI(t, dpRunArgs(1)...)
	if err != nil {
		t.Fatalf("dp=2 scenario run: %v\n%s", err, lastLines(stderr, 3))
	}
	if ids := instanceIDs(t, out); len(ids) != 2 {
		t.Errorf("1 logical instance x dp 2 must run 2 replicas, got %d: %v", len(ids), ids)
	}
	clusterConservationHolds(t, out, dpFixtureNumRequests)
	if !strings.Contains(stderr, "inter-replica fabric cost is NOT priced") {
		t.Errorf("the expert-parallel group spans replicas; its unpriced fabric must be disclosed:\n%s",
			lastLines(stderr, 10))
	}
	again, _, err := runKernelCLI(t, dpRunArgs(1)...)
	if err != nil {
		t.Fatal(err)
	}
	if again != out {
		t.Error("INV-6: two identical DP-placement runs produced different stdout")
	}
}

// TestRunCmd_MoEDP1_ByteIdentical: on a dp=1 MoE scenario the placement plan is inactive --
// one logical instance is one engine -- and the run is deterministic (INV-6).
func TestRunCmd_MoEDP1_ByteIdentical(t *testing.T) {
	args := []string{"run", "--scenario", kernelTestScenario, "--num-instances", "1",
		"--rate", "10", "--num-requests", strconv.Itoa(dpFixtureNumRequests), "--seed", "42"}
	first, stderr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("dp=1 run: %v\n%s", err, lastLines(stderr, 3))
	}
	if ids := instanceIDs(t, first); len(ids) > 1 {
		t.Errorf("dp=1 must not expand the instance count, got %v", ids)
	}
	clusterConservationHolds(t, first, dpFixtureNumRequests)
	second, _, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Error("INV-6: two dp=1 MoE runs produced different stdout")
	}
}

// TestRunCmd_MoEDPPlacement_GuardedCombo_Rejected: the autoscaler over a dp-expanded population
// is the one combination still refused (#1553 decision), and the refusal terminates the run
// naming the autoscaler rather than being returned and ignored.
func TestRunCmd_MoEDPPlacement_GuardedCombo_Rejected(t *testing.T) {
	_, stderr, err := runKernelCLI(t, dpRunArgs(1, "--model-autoscaler-interval-us", "1000000")...)
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit 1 (logrus.Fatalf) for the autoscaler + dp>1, got %v\n%s", err, lastLines(stderr, 3))
	}
	if !strings.Contains(stderr, "autoscaler") || !strings.Contains(stderr, "#1553") {
		t.Errorf("the refusal must name the autoscaler and the #1553 decision:\n%s", lastLines(stderr, 3))
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

// TestResolveDPPlacement_IsAPerRankExpansion is the law of the shared resolver both `blis run`
// and `blis replay` call (#1556): the kernel sizes each data-parallel rank, so an active plan
// multiplies the instance count and every P/D pool count by dp while each replica keeps the
// kernel's per-rank KV budget and max-model-len unchanged -- the aggregate is replicas x
// per-rank, never a per-rank budget divided again. A dense model or dp=1 mutates nothing
// (INV-6), and the autoscaler refusal mutates nothing either.
//
// NOTE: mutates package-level vars — must NOT use t.Parallel().
func TestResolveDPPlacement_IsAPerRankExpansion(t *testing.T) {
	origCmd := captureCmdLevelVars()
	defer origCmd.restore()
	origDP := captureDPResolveVars()
	defer origDP.restore()

	rapid.Check(t, func(rt *rapid.T) {
		moe := rapid.Bool().Draw(rt, "moe")
		dp := rapid.IntRange(1, 8).Draw(rt, "dp")
		epOn := moe && rapid.Bool().Draw(rt, "ep")
		autoscaler := rapid.Bool().Draw(rt, "autoscaler")
		inInst := rapid.IntRange(1, 8).Draw(rt, "instances")
		pd := rapid.Bool().Draw(rt, "pd")
		inPrefill, inDecode := 0, 0
		if pd && inInst >= 2 {
			inPrefill = rapid.IntRange(1, inInst-1).Draw(rt, "prefill")
			inDecode = rapid.IntRange(1, inInst-inPrefill).Draw(rt, "decode")
		}
		inKV := rapid.Int64Range(1, 1<<20).Draw(rt, "kv")
		inLen := rapid.Int64Range(0, 1<<22).Draw(rt, "maxModelLen")

		dataParallelism, enableExpertParallel, moeCommBackend = dp, epOn, ""
		prefillInstances, decodeInstances, prefillDecodeInstances, encodeInstances = inPrefill, inDecode, 0, 0
		numInstances, totalKVBlocks, maxModelLen, blockSizeTokens = inInst, inKV, inLen, 16

		experts := 0
		if moe {
			experts = 8
		}
		lr := latencyResolution{ModelConfig: sim.ModelConfig{NumLocalExperts: experts}}
		plan, err := planDPPlacement(lr.ModelConfig.IsMoE(), dp, epOn, inPrefill > 0, autoscaler, false)
		if err == nil {
			plan, err = resolveDPPlacement(lr, plan)
		}
		active := moe && dp > 1
		if active && autoscaler {
			if err == nil || !strings.Contains(err.Error(), "autoscaler") {
				rt.Fatalf("autoscaler + MoE dp %d must be refused naming the autoscaler, got %v", dp, err)
			}
		} else if err != nil {
			rt.Fatalf("unexpected refusal: %v", err)
		}
		factor := 1
		if active && !autoscaler {
			factor = dp
		}
		if numInstances != inInst*factor || prefillInstances != inPrefill*factor || decodeInstances != inDecode*factor {
			rt.Fatalf("instances %d->%d, prefill %d->%d, decode %d->%d; want each x%d",
				inInst, numInstances, inPrefill, prefillInstances, inDecode, decodeInstances, factor)
		}
		if totalKVBlocks != inKV || maxModelLen != inLen {
			rt.Fatalf("per-rank KV %d->%d / max-model-len %d->%d moved; the kernel already sized one rank",
				inKV, totalKVBlocks, inLen, maxModelLen)
		}
		if err == nil && active && (plan.PerRankDP != 1 || (epOn && plan.EPGroupDP != dp)) {
			rt.Fatalf("active plan %+v: each replica must run DP=1 and carry the EP group width %d", plan, dp)
		}
	})
}
