package cmd

// CLI-level DP-as-placement parity for `blis replay` (issue #1556, follow-up to
// #1531). #1531 shipped DP-as-real-placement as a `blis run`-only feature and made
// `blis replay` logrus.Fatalf on MoE --dp>1, because replay kept the lumped
// single-instance DP-math model and the two paths would have diverged for identical
// flags (INV-13). #1556 lifts that guard: both commands now resolve the placement
// through the single shared resolveDPPlacement, so the deferred parity holds.
//
// These are the system-level laws (the pure-function contracts live in
// dp_placement_test.go):
//
//   - #1556 BC-1: replay expands MoE --dp N into --num-instances × N engine replicas.
//   - #1556 BC-2 (INV-13): a trace exported by `blis run --dp N` replayed with the same
//     flags produces byte-identical stdout — the parity the issue asks for.
//   - #1556 BC-3: the auto-KV path (no --total-kv-blocks) divides to the per-rank budget
//     and re-caps --max-model-len, so no replica's NewSimulator panics.
//   - #1556 BC-4: the #1548 / #1553 guarded combos still fail fast on replay.
//   - #1556 BC-5 (INV-6): --dp 1 on replay stays byte-identical run to run.
//   - #1556 BC-6 (INV-1): request conservation holds across the expanded replicas.
//
// The BC numbers are #1556's own; dp_placement_test.go in this same package carries
// #1531's BC-1..BC-8, which mean different things — always cite the issue.
//
// Every leg re-execs this test binary as a subprocess: the CLI paths logrus.Fatalf
// (which would kill the test process) and os.Exit(0) gives a clean stdout capture.
// The run and replay legs are separate subprocesses sharing a trace prefix under the
// parent's t.TempDir(), because both mutate the same package-level cobra flag vars.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// Environment keys the subprocess legs read. Kept in one place so the parent and
// child agree.
const (
	dpLegEnv      = "BLIS_DP_LEG"     // "run" | "replay" — which blis command to drive
	dpTraceEnv    = "BLIS_DP_TRACE"   // trace prefix (<prefix>.yaml / <prefix>.csv)
	dpDPEnv       = "BLIS_DP_DP"      // --dp
	dpNumInstEnv  = "BLIS_DP_NUMINST" // --num-instances
	dpExtraEnv    = "BLIS_DP_EXTRA"   // extra args, unit-separator delimited
	dpExtraSepStr = "\x1f"            // unit separator (never appears in a flag)
)

// dpLegCobraErrorExit is the code a leg exits with when cobra rejects the arg list —
// distinct from 0 (success) and from 1 (logrus.Fatalf), so the harness can tell its own
// bugs apart from the simulator's.
const dpLegCobraErrorExit = 3

// dpParityHorizon is passed explicitly to BOTH legs so the two commands agree on the
// simulation horizon. `blis run` defaults to math.MaxInt64 while `blis replay`
// defaults to computeReplayHorizon(requests) (a drain-time estimate) — a pre-existing
// difference orthogonal to DP that would otherwise truncate the replay leg. The value
// passed to run is run's own default, so it changes no simulation behavior here; note
// it does flip cmd.Flags().Changed("horizon"), which suppresses a workload spec's
// `horizon` field (irrelevant for these --rate/--num-requests legs, which use no spec).
// math.MaxInt64 mirrors `blis run`'s own --horizon default (registerSimConfigFlags in
// cmd/root.go). The coupling is silent: if that default ever changes, this stays
// correct as a *shared* value for both legs (which is all the parity law needs) but
// stops being "run's default", so the "no-op for run" reasoning below would need
// re-checking.
var dpParityHorizon = strconv.FormatInt(math.MaxInt64, 10)

// dpMoEFixtureArgs are the model/hardware flags shared by both legs: the git-tracked
// deepseek-v2-lite MoE fixture (offline; paths relative to the cmd/ test cwd).
func dpMoEFixtureArgs() []string {
	return []string{
		"--model", "deepseek-ai/deepseek-v2-lite",
		"--model-config-folder", "../model_configs/deepseek-v2-lite",
		"--hardware", "H100",
		"--hardware-config", "../hardware_config.json",
		"--tp", "1",
		"--seed", "42",
		"--horizon", dpParityHorizon,
		"--defaults-filepath", "../defaults.yaml",
	}
}

// dpLegSubprocess drives one real blis command inside the re-exec'd subprocess,
// using the leg/trace/dp/num-instances/extra args passed via the environment. It
// never returns: os.Exit(0) on a clean finish (a logrus.Fatalf exits 1 itself).
func dpLegSubprocess() {
	dp := os.Getenv(dpDPEnv)
	numInst := os.Getenv(dpNumInstEnv)
	tracePrefix := os.Getenv(dpTraceEnv)
	var args []string
	switch os.Getenv(dpLegEnv) {
	case "run":
		args = append([]string{"run"}, dpMoEFixtureArgs()...)
		args = append(args, "--rate", "10", "--num-requests", "40",
			"--trace-output", tracePrefix)
	case "replay":
		args = append([]string{"replay"}, dpMoEFixtureArgs()...)
		args = append(args, "--trace-header", tracePrefix+".yaml",
			"--trace-data", tracePrefix+".csv")
	default:
		os.Exit(2)
	}
	args = append(args, "--dp", dp, "--num-instances", numInst)
	if extra := os.Getenv(dpExtraEnv); extra != "" {
		args = append(args, strings.Split(extra, dpExtraSepStr)...)
	}
	rootCmd.SetArgs(args)
	// dpLegCobraErrorExit on a cobra-level error — a flag typo in this harness would
	// otherwise surface as an empty stdout and a confusing assertion failure instead of a
	// loud one. Cobra already printed the error and usage to stderr, which dpLeg captures.
	if err := rootCmd.Execute(); err != nil {
		os.Exit(dpLegCobraErrorExit)
	}
	os.Exit(0)
}

// dpLeg re-execs testName as a subprocess driving one blis leg and returns its
// stdout, stderr, and error. The caller decides whether a non-zero exit is expected.
func dpLeg(t *testing.T, testName, leg, tracePrefix string, dp, numInst int, extra ...string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^"+testName+"$")
	cmd.Env = append(os.Environ(),
		dpLegEnv+"="+leg,
		dpTraceEnv+"="+tracePrefix,
		dpDPEnv+"="+strconv.Itoa(dp),
		dpNumInstEnv+"="+strconv.Itoa(numInst),
		dpExtraEnv+"="+strings.Join(extra, dpExtraSepStr),
	)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	err = cmd.Run()
	return out.String(), errBuf.String(), err
}

// dpLegOK is dpLeg with a required clean exit. It separates the harness's own failure
// mode (exit 3 = a cobra flag error, i.e. THIS FILE built a bad arg list) from a genuine
// simulator failure (exit 1 = logrus.Fatalf), so a future test-author's flag typo says so
// instead of surfacing as a confusing downstream assertion failure.
func dpLegOK(t *testing.T, testName, leg, tracePrefix string, dp, numInst int, extra ...string) string {
	t.Helper()
	stdout, stderr, err := dpLeg(t, testName, leg, tracePrefix, dp, numInst, extra...)
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && exitErr.ExitCode() == dpLegCobraErrorExit {
			t.Fatalf("test harness bug, not a simulator failure: `blis %s` rejected the arg list this file "+
				"built (cobra flag error, exit %d) — check dpMoEFixtureArgs and the extra args %v\nstderr:\n%s",
				leg, dpLegCobraErrorExit, extra, stderr)
		}
		t.Fatalf("subprocess `blis %s` (dp=%d, num-instances=%d) failed: %v\nstderr:\n%s",
			leg, dp, numInst, err, stderr)
	}
	return stdout
}

// clusterMetricInt returns one integer field of the "cluster" aggregate metrics
// object in blis stdout. Used for non-vacuity checks (a parity assertion over two
// empty runs passes trivially). Reuses extractJSONObjects from dp_placement_test.go.
func clusterMetricInt(t *testing.T, stdout, field string) int {
	t.Helper()
	for _, raw := range extractJSONObjects(stdout) {
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(raw), &obj); err != nil {
			continue
		}
		if obj["instance_id"] != "cluster" {
			continue
		}
		v, ok := obj[field].(float64)
		if !ok {
			t.Fatalf("cluster metrics missing numeric field %q", field)
		}
		return int(v)
	}
	t.Fatalf("no cluster aggregate metrics object found in stdout:\n%s", stdout)
	return 0
}

// TestINV13_RunReplayParity_MoEDPPlacement is #1556 BC-2, the law the issue asks for:
// a trace exported by `blis run --dp N` on an MoE model, replayed with the same flags,
// yields byte-identical stdout. Byte-identity over the whole metrics dump (per-replica
// AND cluster aggregate) is the strongest available statement of INV-13 at the CLI
// boundary — it covers per-request latency distributions, cache behavior, and the
// conservation counters at once. `clusterConservationHolds` is its invariant companion
// (R7): byte-identity alone would be satisfied by two identically-wrong runs.
//
// Before #1556 the replay leg exited 1 with the run-only guard, so this test is also
// the regression fence against re-introducing it.
//
// The TP=2 case is not redundant: every other DP test runs at TP=1, where a regression
// that hardcoded the TP argument to 1 at either NewModelHardwareConfig call site would
// be behaviorally invisible.
func TestINV13_RunReplayParity_MoEDPPlacement(t *testing.T) {
	if os.Getenv(dpLegEnv) != "" {
		dpLegSubprocess()
		return
	}
	name := t.Name()

	cases := []struct {
		label string
		extra []string
	}{
		{label: "tp1", extra: []string{"--total-kv-blocks", "20000"}},
		// A trailing --tp overrides the fixture's --tp 1 (cobra keeps the last value).
		{label: "tp2", extra: []string{"--total-kv-blocks", "20000", "--tp", "2"}},
	}
	for _, tc := range cases {
		t.Run(tc.label, func(t *testing.T) {
			prefix := filepath.Join(t.TempDir(), "trace")
			runOut := dpLegOK(t, name, "run", prefix, 2, 1, tc.extra...)
			// dpLeg (not dpLegOK) for the replay leg so a divergence report can include the
			// leg's stderr: the capacity/re-cap diagnostics are warn-level, so they are
			// present without raising the log level, and they are the first thing to look at
			// when the two stdouts differ.
			replayOut, replayErr, err := dpLeg(t, name, "replay", prefix, 2, 1, tc.extra...)
			if err != nil {
				t.Fatalf("replay leg failed: %v\nstderr:\n%s", err, replayErr)
			}

			// Non-vacuity: a parity assertion over two empty runs would pass trivially.
			if completed := clusterMetricInt(t, runOut, "completed_requests"); completed <= 0 {
				t.Fatalf("INV-13 parity would be vacuous: run leg completed %d requests; stdout:\n%s", completed, runOut)
			}
			clusterConservationHolds(t, runOut) // INV-1 companion to the byte-identity law
			if runOut != replayOut {
				t.Errorf("#1556 BC-2 (INV-13): `blis run --dp 2` and the replay of its trace must produce "+
					"identical stdout\nRUN:\n%s\nREPLAY:\n%s\nREPLAY stderr:\n%s", runOut, replayOut, replayErr)
			}
		})
	}
}

// TestReplayCmd_MoEDPPlacement_SpawnsReplicas is BC-1 + BC-6: MoE --dp N on replay
// expands into --num-instances × N real engine replicas (not one lumped instance),
// and request conservation holds across them (INV-1). Replaces the #1531-era
// TestReplayCmd_MoEDPPlacement_Rejected, whose guard #1556 lifted.
func TestReplayCmd_MoEDPPlacement_SpawnsReplicas(t *testing.T) {
	if os.Getenv(dpLegEnv) != "" {
		dpLegSubprocess()
		return
	}
	name := t.Name()
	prefix := filepath.Join(t.TempDir(), "trace")

	// Export once at --dp 1: the trace is the workload, so it is DP-independent —
	// which also proves replay does not need a DP-aware trace to place replicas.
	dpLegOK(t, name, "run", prefix, 1, 1, "--total-kv-blocks", "20000")

	// Case A: --num-instances 1 --dp 2 → exactly 2 replicas.
	outA := dpLegOK(t, name, "replay", prefix, 2, 1, "--total-kv-blocks", "20000")
	if !strings.Contains(outA, `"instance_id": "instance_1"`) {
		t.Errorf("BC-1: replay with --num-instances 1 --dp 2 must spawn a second replica (instance_1); stdout:\n%s", outA)
	}
	if strings.Contains(outA, `"instance_id": "instance_2"`) {
		t.Errorf("BC-1: replay with --dp 2 must spawn exactly 2 replicas, but instance_2 is present")
	}
	clusterConservationHolds(t, outA) // BC-6 / INV-1

	// Case B: --num-instances 2 --dp 2 → 4 replicas, confirming the M×N multiply.
	outB := dpLegOK(t, name, "replay", prefix, 2, 2, "--total-kv-blocks", "20000")
	if !strings.Contains(outB, `"instance_id": "instance_3"`) {
		t.Errorf("BC-1: replay with --num-instances 2 --dp 2 must spawn M×N=4 replicas (instance_3); stdout:\n%s", outB)
	}
	if strings.Contains(outB, `"instance_id": "instance_4"`) {
		t.Errorf("BC-1: replay with --num-instances 2 --dp 2 must spawn exactly 4 replicas, but instance_4 is present")
	}
	clusterConservationHolds(t, outB)

	// INV-6: a repeat of Case A is byte-identical.
	outA2 := dpLegOK(t, name, "replay", prefix, 2, 1, "--total-kv-blocks", "20000")
	if outA != outA2 {
		t.Errorf("INV-6: two identical DP-placement replays produced different stdout")
	}
}

// The auto-KV path emits two capacity warnings whose numbers state the whole per-rank
// law, so the test can read them instead of needing a KV-bound workload:
//
//	--latency-model: max-model-len N exceeds KV capacity (<AGG> blocks × <BS> tokens); …
//	--max-model-len N exceeds per-rank KV capacity (<RANK> blocks × <BS> tokens) under
//	  DP-as-placement; capping to <CAP> tokens
//
// <AGG> is the dp-multiplied global total resolveLatencyConfig derived; <RANK> is what
// applyDPPlacement divided it down to; <CAP> is what the re-cap wrote back to
// --max-model-len. Both are logrus warnings, so they appear at the default log level.
var (
	dpAggregateBlocksRE = regexp.MustCompile(`exceeds KV capacity \((\d+) blocks × (\d+) tokens\)`)
	dpPerRankBlocksRE   = regexp.MustCompile(`exceeds per-rank KV capacity \((\d+) blocks × (\d+) tokens\) under DP-as-placement; capping to (\d+) tokens`)
)

// dpAutoKVNumbers pulls (aggregate blocks, per-rank blocks, block size, capped
// max-model-len) out of one leg's stderr, failing the test if either warning is absent
// — a missing per-rank line means the division/re-cap never ran at all.
func dpAutoKVNumbers(t *testing.T, stderr, leg string) (aggregate, perRank, blockSize, capped int64) {
	t.Helper()
	parse := func(sub string) int64 {
		v, err := strconv.ParseInt(sub, 10, 64)
		if err != nil {
			t.Fatalf("%s leg: unparseable number %q: %v", leg, sub, err)
		}
		return v
	}
	agg := dpAggregateBlocksRE.FindStringSubmatch(stderr)
	if agg == nil {
		t.Fatalf("%s leg: no aggregate KV-capacity warning in stderr (auto-KV path did not run?):\n%s", leg, stderr)
	}
	rank := dpPerRankBlocksRE.FindStringSubmatch(stderr)
	if rank == nil {
		t.Fatalf("%s leg: no per-rank DP-as-placement re-cap warning in stderr — the per-rank division "+
			"and/or the max-model-len re-cap did not run:\n%s", leg, stderr)
	}
	return parse(agg[1]), parse(rank[1]), parse(rank[2]), parse(rank[3])
}

// TestReplayCmd_MoEDPPlacement_AutoKV_Parity is #1556 BC-3: on the auto-KV path (no
// --total-kv-blocks), replay must divide the dp-multiplied global total back to the
// per-rank budget and re-cap --max-model-len to it. A missing re-cap makes each
// replica's NewSimulator panic ("KV cache too small for MaxModelLen"), so a clean exit
// is one observable — but this fixture is nowhere near KV-bound (~1.19M tokens of KV
// per rank vs 40 requests of a few hundred tokens), so a *wrong* per-rank budget or a
// skipped division changes no stdout metric and stdout parity alone would be vacuous.
// The test therefore also asserts the numbers the resolver reports:
//
//  1. per-rank × dp == the aggregate the auto-calc derived — the division happened and
//     the aggregate over the replicas is conserved (a dp² double-count fails this);
//  2. the re-cap wrote per-rank blocks × block size, not the aggregate;
//  3. run and replay report the SAME per-rank budget.
//
// This is also the path where a run/replay divergence would be silent rather than
// fatal: it is the one place each command derives capacity from the model instead of
// reading a flag.
func TestReplayCmd_MoEDPPlacement_AutoKV_Parity(t *testing.T) {
	if os.Getenv(dpLegEnv) != "" {
		dpLegSubprocess()
		return
	}
	name := t.Name()
	prefix := filepath.Join(t.TempDir(), "trace")
	// A huge --max-model-len forces resolveLatencyConfig to cap against the AGGREGATE
	// total, after which the per-rank re-cap must fire.
	const hugeMaxLen = "10000000"
	const dp = 2

	runOut, runErr, err := dpLeg(t, name, "run", prefix, dp, 1, "--max-model-len", hugeMaxLen)
	if err != nil {
		t.Fatalf("auto-KV run leg failed: %v\nstderr:\n%s", err, runErr)
	}
	replayOut, replayErr, err := dpLeg(t, name, "replay", prefix, dp, 1, "--max-model-len", hugeMaxLen)
	if err != nil {
		t.Fatalf("auto-KV replay leg failed (a missing per-rank max-model-len re-cap panics "+
			"per-replica NewSimulator): %v\nstderr:\n%s", err, replayErr)
	}

	if !strings.Contains(replayOut, `"instance_id": "instance_1"`) {
		t.Errorf("#1556 BC-3: auto-KV replay must spawn 2 replicas (instance_1); stdout:\n%s", replayOut)
	}
	clusterConservationHolds(t, replayOut)
	if runOut != replayOut {
		t.Errorf("#1556 BC-3/INV-13: auto-KV run and replay must agree\nRUN:\n%s\nREPLAY:\n%s", runOut, replayOut)
	}

	runAgg, runRank, runBS, runCap := dpAutoKVNumbers(t, runErr, "run")
	repAgg, repRank, repBS, repCap := dpAutoKVNumbers(t, replayErr, "replay")

	for _, leg := range []struct {
		label                        string
		agg, rank, blockSize, capped int64
	}{
		{"run", runAgg, runRank, runBS, runCap},
		{"replay", repAgg, repRank, repBS, repCap},
	} {
		// (1) The division happened and conserved the aggregate: perRank × dp == aggregate.
		if leg.rank*dp != leg.agg {
			t.Errorf("#1556 BC-3: %s leg per-rank KV (%d) × dp (%d) must equal the auto-derived aggregate (%d) — "+
				"a skipped per-rank division or a dp² double-count breaks this", leg.label, leg.rank, dp, leg.agg)
		}
		// (2) The re-cap used the PER-RANK budget, not the aggregate.
		if want := leg.rank * leg.blockSize; leg.capped != want {
			t.Errorf("#1556 BC-3: %s leg re-capped --max-model-len to %d, want %d (per-rank %d blocks × %d tokens); "+
				"capping to the aggregate instead panics per-replica NewSimulator",
				leg.label, leg.capped, want, leg.rank, leg.blockSize)
		}
	}
	// (3) Both commands derived the same per-replica budget (INV-13 on the derivation).
	if runRank != repRank {
		t.Errorf("#1556 BC-3/INV-13: per-replica KV budget must match: run=%d replay=%d", runRank, repRank)
	}
}

// TestReplayCmd_MoEDPPlacement_GuardedCombos_Rejected is BC-4: #1556 lifted the
// run-only guard but NOT the physics guards. An unsupported combination must still
// exit 1 naming its tracking issue, on replay exactly as on run — never a silently
// mis-modeled replay.
func TestReplayCmd_MoEDPPlacement_GuardedCombos_Rejected(t *testing.T) {
	if os.Getenv(dpLegEnv) != "" {
		dpLegSubprocess()
		return
	}
	name := t.Name() // captured before t.Run so subtests address the parent leg
	prefix := filepath.Join(t.TempDir(), "trace")
	dpLegOK(t, name, "run", prefix, 1, 1, "--total-kv-blocks", "20000")

	// Only EP and PD are listed: the autoscaler and node pools are rejected by
	// blis replay unconditionally (before DP is considered), so their #1553 DP guard is
	// unreachable here — TestReplayCmd_AutoscalerBundleFatal /
	// TestReplayCmd_NodePoolsBundleFatal cover those, and
	// TestResolveDPPlacement_MutatesDeploymentVars covers their DP guard directly.
	cases := []struct {
		label   string
		numInst int
		extra   []string
		wantRef string // tracking issue the message must name
	}{
		{
			label:   "expert parallelism on",
			numInst: 1,
			extra:   []string{"--total-kv-blocks", "20000", "--enable-expert-parallel"},
			wantRef: "1548",
		},
		{
			label: "PD disaggregation",
			// --num-instances 2 (vs 1 for the EP case): ValidatePoolTopology runs BEFORE
			// the DP guard and requires the pool sizes to fit the instance count, so
			// --prefill-instances 1 --decode-instances 1 needs 2. Without it the run would
			// exit on the topology error and never reach the #1553 guard under test.
			numInst: 2,
			extra: []string{"--total-kv-blocks", "20000",
				"--prefill-instances", "1", "--decode-instances", "1", "--pd-decider", "always"},
			wantRef: "1553",
		},
	}
	for _, tc := range cases {
		t.Run(tc.label, func(t *testing.T) {
			stdout, stderr, err := dpLeg(t, name, "replay", prefix, 2, tc.numInst, tc.extra...)
			if err == nil {
				t.Fatalf("expected a non-zero exit for MoE --dp 2 + %s; stdout:\n%s\nstderr:\n%s",
					tc.label, stdout, stderr)
			}
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
				t.Fatalf("expected exit code 1 (logrus.Fatalf), got %v; stderr:\n%s", err, stderr)
			}
			if !strings.Contains(stderr, tc.wantRef) {
				t.Errorf("guard message must reference #%s so the user can find the tracking issue; stderr:\n%s",
					tc.wantRef, stderr)
			}
		})
	}
}

// TestReplayCmd_MoEDP1_ByteIdentical is the INV-6 no-op fence on the newly-supported
// path. #1556 made `blis replay` traverse resolveDPPlacement for EVERY MoE replay, not
// just --dp>1 ones — so the "--dp 1 is byte-identical" claim, previously fenced only on
// the run side (TestRunCmd_MoEDP1_ByteIdentical), now needs a replay fence too. Catches
// a future regression that makes the shared resolver perturb the inactive path.
//
// Both KV paths are fenced, because they reach the resolver with different state and
// only one of them is covered elsewhere:
//
//   - explicit --total-kv-blocks: KVParamsOK is false, so autoScaledKV is false;
//   - auto-KV (no --total-kv-blocks): KVParamsOK is true and totalKVBlocks is the
//     dp-multiplied auto total, i.e. the state that DOES trigger the division and the
//     max-model-len re-cap at --dp>1. At --dp 1 the inactive-plan early return must skip
//     both. TestReplayCmd_MoEDPPlacement_AutoKV_Parity exercises auto-KV only at --dp 2,
//     so without this row the auto-KV × --dp 1 × replay corner — precisely the INV-6
//     byte-identity path that must never regress — would be unfenced.
func TestReplayCmd_MoEDP1_ByteIdentical(t *testing.T) {
	if os.Getenv(dpLegEnv) != "" {
		dpLegSubprocess()
		return
	}
	name := t.Name()

	for _, tc := range []struct {
		label string
		extra []string
	}{
		// --log info makes resolveDPPlacement's activation line visible: at --dp 1 the
		// division, the re-cap and the expansion are all ×1 or no-ops, so an inactive-plan
		// early return that regressed into an ACTIVE plan would be byte-identical on
		// stdout. The diagnostic is the only observable that separates "looks the same"
		// from "took the early return", and it is a logrus.Infof — invisible at the
		// default `warn`.
		{label: "explicit-kv", extra: []string{"--total-kv-blocks", "20000", "--log", "info"}},
		// No --total-kv-blocks ⇒ the auto-capacity path runs, so KVParamsOK is true and
		// totalKVBlocks is the dp-multiplied auto total: the exact state that triggers the
		// division and the re-cap at --dp>1, which --dp 1 must skip.
		{label: "auto-kv", extra: []string{"--max-model-len", "10000000", "--log", "info"}},
	} {
		t.Run(tc.label, func(t *testing.T) {
			prefix := filepath.Join(t.TempDir(), "trace")
			dpLegOK(t, name, "run", prefix, 1, 1, tc.extra...)

			first, firstErr, err := dpLeg(t, name, "replay", prefix, 1, 1, tc.extra...)
			if err != nil {
				t.Fatalf("--dp 1 replay failed: %v\nstderr:\n%s", err, firstErr)
			}
			second := dpLegOK(t, name, "replay", prefix, 1, 1, tc.extra...)
			if completed := clusterMetricInt(t, first, "completed_requests"); completed <= 0 {
				t.Fatalf("INV-6 check would be vacuous: --dp 1 replay completed %d requests; stdout:\n%s", completed, first)
			}
			if first != second {
				t.Errorf("INV-6: two identical MoE --dp 1 replays produced different stdout")
			}
			// The inactive plan must mutate nothing: no DP-as-placement diagnostic may be
			// emitted at all. This is what distinguishes "--dp 1 happens to look the same"
			// from "--dp 1 took the early return" — and on the auto-KV row it is the direct
			// assertion that neither the per-rank division nor the re-cap ran.
			if strings.Contains(firstErr, "DP-as-placement") {
				t.Errorf("INV-6: --dp 1 must not activate DP-as-placement (no division, no re-cap); stderr:\n%s", firstErr)
			}
		})
	}
}

// dpForeignTraceEnv marks the subprocess leg for
// TestReplayCmd_MoEDPPlacement_ForeignTrace_Mixtral; its value is the temp dir holding
// the fixture the parent wrote.
const dpForeignTraceEnv = "BLIS_DP_FOREIGN_DIR"

// writeMixtralForeignFixture writes, into dir: a synthetic NON-MLA Mixtral-style MoE
// config.json, a hardware config, and a hand-authored TraceV2 header + CSV. Both halves
// restore coverage the #1531-era TestReplayCmd_MoEDPPlacement_Rejected carried and the
// deepseek-v2-lite-based tests do not:
//
//   - a second MoE architecture family on the replay DP path (deepseek-v2-lite is MLA
//     with `n_routed_experts`; this is plain `num_local_experts`), so MoE detection —
//     the predicate that gates the whole feature — is exercised on more than one shape;
//   - a trace this binary did not generate, which is the realistic operator flow (an
//     `observe`-origin or converted corpus) and takes a different load path from a
//     `blis run --trace-output` file.
func writeMixtralForeignFixture(t *testing.T, dir string) {
	t.Helper()
	mcDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// num_local_experts > 1 ⇒ IsMoE. --total-kv-blocks is passed explicitly by the leg,
	// so the auto-capacity path (which would need vocab_size etc.) is not exercised here.
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
		t.Fatalf("write config.json: %v", err)
	}
	hw := `{"H100": {"MemoryGiB": 80.0, "TFlopsPeak": 989.5, "BwPeakTBs": 3.35}}`
	if err := os.WriteFile(filepath.Join(dir, "hw.json"), []byte(hw), 0644); err != nil {
		t.Fatalf("write hw.json: %v", err)
	}
	header := "trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"
	if err := os.WriteFile(filepath.Join(dir, "trace.yaml"), []byte(header), 0644); err != nil {
		t.Fatalf("write trace.yaml: %v", err)
	}
	csv := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length," +
		"streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio," +
		"model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us," +
		"last_chunk_time_us,num_chunks,status,error_message,finish_reason\n"
	for i := 0; i < 6; i++ {
		arrival := i * 50000
		csv += fmt.Sprintf("%d,c1,t1,standard,,0,,0,false,64,16,64,0,0,0,0.0,,0,0,%d,%d,0,0,0,ok,,\n", i, arrival, arrival)
	}
	if err := os.WriteFile(filepath.Join(dir, "trace.csv"), []byte(csv), 0644); err != nil {
		t.Fatalf("write trace.csv: %v", err)
	}
}

// TestReplayCmd_MoEDPPlacement_ForeignTrace_Mixtral is #1556 BC-1 on a second MoE
// architecture and on a trace this binary did not produce: `blis replay --dp 2` must
// expand into 2 engine replicas and conserve requests (INV-1), where the #1531-era
// behavior was a hard exit 1.
func TestReplayCmd_MoEDPPlacement_ForeignTrace_Mixtral(t *testing.T) {
	if dir := os.Getenv(dpForeignTraceEnv); dir != "" {
		rootCmd.SetArgs([]string{
			"replay",
			"--model", "mistralai/mixtral-8x7b",
			"--model-config-folder", filepath.Join(dir, "config"),
			"--hardware", "H100",
			"--hardware-config", filepath.Join(dir, "hw.json"),
			"--trace-header", filepath.Join(dir, "trace.yaml"),
			"--trace-data", filepath.Join(dir, "trace.csv"),
			"--tp", "1", "--dp", "2", "--num-instances", "1",
			"--total-kv-blocks", "1000",
			"--seed", "42",
			"--horizon", dpParityHorizon,
			"--defaults-filepath", "../defaults.yaml",
		})
		if err := rootCmd.Execute(); err != nil {
			os.Exit(3)
		}
		os.Exit(0)
	}

	dir := t.TempDir()
	writeMixtralForeignFixture(t, dir)

	cmd := exec.Command(os.Args[0], "-test.run=^"+t.Name()+"$")
	cmd.Env = append(os.Environ(), dpForeignTraceEnv+"="+dir)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("#1556 BC-1: replaying a hand-authored trace for a non-MLA MoE model with --dp 2 must "+
			"succeed (it exited 1 with the #1531 run-only guard): %v\nstderr:\n%s", err, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `"instance_id": "instance_1"`) {
		t.Errorf("#1556 BC-1: --dp 2 must spawn a second replica (instance_1); stdout:\n%s", out)
	}
	if strings.Contains(out, `"instance_id": "instance_2"`) {
		t.Errorf("#1556 BC-1: --dp 2 must spawn exactly 2 replicas, but instance_2 is present")
	}
	clusterConservationHolds(t, out) // INV-1 across the expanded replicas
}
