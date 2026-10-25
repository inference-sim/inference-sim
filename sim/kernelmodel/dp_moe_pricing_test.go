package kernelmodel

import (
	"testing"

	sim "github.com/inference-sim/inference-sim/sim/engine"
)

// dp_moe_pricing_test.go is committed, independently reproducible acceptance evidence for #1902
// ("the accuracy harness over-prices data-parallel MoE steps: an aggregate batch is priced as one
// rank's"), supplied for #1929.
//
// #1851 did not implement #1902's proposed in-process "max over ranks" rank-splitter. It deleted
// the in-process scorer entirely and routes every scored point through normal `blis replay`, which
// prices a dp>1 MoE deployment as dp per-rank replicas (DP-as-placement) -- the path #1902 itself
// notes was never affected. Each replica hands the kernel ONLY its own rank's requests; the kernel
// scales the routed-expert term by the DP width (the experts see dp x one rank's tokens). The
// over-pricing #1902 described was the old harness handing the kernel the AGGREGATE of every rank's
// requests as if it were one rank's batch.
//
// #1902 acceptance -> where each criterion is pinned (reproducible):
//   1. "a dp=1 step's price is unchanged / dp=1 points byte-identical" -- the "unchanged across the
//      fix" half is superseded with criterion 4 (there is no pre-#1851 scorer to diff against; see
//      point 4). What is committed is that DP placement is a NO-OP at dp=1 (one logical instance =
//      one engine, no expansion -- cmd.TestPlanDPPlacement) and that a dp=1 MoE run is byte-identical
//      run-to-run (cmd.TestRunCmd_MoEDP1_ByteIdentical, INV-6 determinism -- NOT a before/after
//      comparison), so dp=1 points are unaffected by the DP-placement machinery.
//   2. "a step over 2N identical requests costs what one rank's step over N costs (metamorphic)" --
//      this is the COMPOSITION of two already-committed facts, not a single test:
//        - the replay path sizes and spawns each rank independently (replica count, per-rank KV
//          with explicit dp^2 double-count rejection, request conservation), and that dp=2
//          placement is byte-reproducible run->replay (INV-13):
//          cmd.TestRunCmd_MoEDPPlacement_SpawnsReplicas, cmd.TestApplyDPPlacement,
//          cmd.dp_kv_per_rank_test.go (TestRunCmd_MoEDPPlacement_EachReplicaIsSizedPerRank),
//          cmd.TestINV13_RunReplayParity_MoEDPPlacement;
//        - the kernel scales the routed-expert term by the DP width (the source, moeDPFunnel,
//          returns float64(k.layout.DP)), exercised directionally -- with a near-cancellation
//          against the falling local-expert share -- by blis-latency-kernel methods_test.go
//          TestRoutedComputeScalesWithTheLocalExpertShare.
//      TestDPMoEAggregateStepCostsMoreThanOneRank below adds the one piece not previously
//      co-located with #1902: on the exact dp=2 MoE scenario, one rank's price (N) is strictly
//      below the aggregate's (2N), so pricing the aggregate as one rank (the old bug) strictly
//      over-prices. (That assertion is batch-size monotonicity -- the funnel cancels from its
//      direction -- so it is a scenario-scoped smoke + direction check, NOT the funnel proof.)
//   3. "a rank's price never exceeds the aggregate's; the aggregate's never falls below any rank's"
//      -- TestDPMoEAggregateStepCostsMoreThanOneRank below.
//   4. "before/after scoring tables, change confined to the dp>1 points" -- SUPERSEDED. The
//      before/after is a comparison against the pre-#1851 in-process scorer, which #1851 deleted;
//      the DP fix shipped inside a wholesale latency-backend replacement, so there is no commit at
//      which only the DP change was applied in isolation. The replacement acceptance is criteria
//      1-3 above, now committed and reproducible. See #1902 and the #1929 PR body.

// dpPricingScenario is the vendored dp=2 MoE deployment #1902 was about (tp2, ep4, dp2). The three
// dp=2 corpus points (minimax-m2.5-b200/b300, minimax-m3-b300) share this parallel shape; it
// resolves through the adapter to DP=2, MoE, expert-parallel (the class the test guards for).
const dpPricingScenario = "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml"

// identicalDecodeBatch returns n byte-identical decode requests (one scheduled token each, the same
// prompt length, each at the end of its prompt so the kernel classifies it as a decode). Identical
// so a comparison between n and 2n isolates batch SIZE -- one rank's share vs the aggregate -- and
// nothing about request shape. Each decode schedules one token, so n and 2n schedule n and 2n
// tokens, far below the kernel's capture-ceiling anomaly (blis-latency-kernel#42, pinned by
// TestTheKernelsCaptureCeilingDropIsStillPresent on GLM-5); the per-request prompt length does not
// count toward that ceiling.
func identicalDecodeBatch(n int, promptLen int64) []*sim.Request {
	batch := make([]*sim.Request, n)
	for i := range batch {
		batch[i] = decodeReq(promptLen, promptLen)
	}
	return batch
}

// identicalPrefillBatch returns n byte-identical chunked-prefill requests: each starts at the
// beginning of a long prompt (ProgressIndex 0) and schedules `sched` tokens of it. A batch
// schedules n*sched tokens, so callers keep n*sched (and 2*n*sched) well below the capture ceiling.
// Covering prefill as well as decode guards the over-pricing direction in the TTFT (prefill) regime
// too, not only the TPOT (decode) regime.
func identicalPrefillBatch(n int, promptLen int64, sched int) []*sim.Request {
	batch := make([]*sim.Request, n)
	for i := range batch {
		batch[i] = &sim.Request{
			InputTokens:   make([]sim.TokenID, promptLen),
			ProgressIndex: 0,
			NumNewTokens:  sched,
		}
	}
	return batch
}

// TestDPMoEAggregateStepCostsMoreThanOneRank pins #1902 criterion 3 and the over-pricing DIRECTION
// on the exact dp=2 MoE deployment #1902 was about, through the adapter StepTime the scorer calls.
// The pre-#1851 harness handed the kernel every rank's requests at once (the AGGREGATE, 2N) as one
// rank's batch; the merged DP-as-placement path hands each rank only its own N. This asserts the
// aggregate (2N) costs strictly more than one rank (N) -- so pricing the aggregate as one rank
// over-prices -- and that a rank never exceeds the aggregate.
//
// Honest scope: for identical batches the MoE funnel is a common factor on both the N and the 2N
// price, so it cancels from the DIRECTION of this inequality -- the assertion is strict batch-size
// monotonicity and would hold on a dense model too. It is therefore a scenario-scoped smoke +
// over-pricing-direction check, NOT a proof of the funnel or of per-rank feeding. Those are the
// composition named in the file doc comment above (the cmd placement/per-rank-KV tests and the
// upstream funnel test); the adapter prices whatever batch it is handed and cannot show feeding by
// itself. The guard below only confirms this is the right deployment class.
func TestDPMoEAggregateStepCostsMoreThanOneRank(t *testing.T) {
	m := open(t, dpPricingScenario)

	// Scenario-relevance guard: confirm this is the dp=2 MoE expert-parallel class #1902 was about.
	// (It is NOT what makes the assertion below meaningful -- see the doc comment -- it only keeps
	// the test pointed at the right deployment.)
	dep := m.Deployment()
	if m.DataParallelWidth() != 2 || !dep.MoE || !dep.ExpertParallel {
		t.Fatalf("%s must resolve to a dp=2 MoE expert-parallel deployment; got DP=%d MoE=%v EP=%v",
			dpPricingScenario, m.DataParallelWidth(), dep.MoE, dep.ExpertParallel)
	}

	// Both regimes the scorer measures: decode (TPOT) and chunked prefill (TTFT). All scheduled-token
	// totals (n and 2n) stay far below the capture ceiling: decodes schedule 1 token each (<=32);
	// the prefill cases schedule n*sched tokens (<=256).
	for _, tc := range []struct {
		name  string
		n     int
		batch func(n int) []*sim.Request
	}{
		{"decode L=512", 4, func(n int) []*sim.Request { return identicalDecodeBatch(n, 512) }},
		{"decode L=1024", 8, func(n int) []*sim.Request { return identicalDecodeBatch(n, 1024) }},
		{"decode L=2048", 16, func(n int) []*sim.Request { return identicalDecodeBatch(n, 2048) }},
		{"prefill 64-tok chunks", 2, func(n int) []*sim.Request { return identicalPrefillBatch(n, 4096, 64) }},
		{"prefill 32-tok chunks", 4, func(n int) []*sim.Request { return identicalPrefillBatch(n, 2048, 32) }},
	} {
		perRank := m.StepTime(tc.batch(tc.n))
		aggregate := m.StepTime(tc.batch(2 * tc.n))

		// The strict inequality establishes both #1902 properties at once: a rank's price never
		// exceeds the aggregate's (criterion 3), and the aggregate is strictly dearer than one rank
		// (the over-pricing direction). Pricing the aggregate (2N) as one rank -- the #1902 bug --
		// would carry no penalty only if these were equal; the merged per-rank path pays the
		// smaller `perRank`.
		if perRank >= aggregate {
			t.Errorf("%s: one rank (n=%d, %d ticks) was not strictly cheaper than the aggregate "+
				"(n=%d, %d ticks); the aggregate must cost strictly more than one rank -- so a rank "+
				"never exceeds it (criterion 3) and pricing the aggregate as one rank (the #1902 bug) "+
				"carries a penalty", tc.name, tc.n, perRank, 2*tc.n, aggregate)
		}
	}
}
