package latency_test

import (
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// --- EP-aware weight-footprint sizing (#1656) ---
//
// CalculateKVBlocks charges the model's weight footprint against the memory budget of
// one DP rank's TP GPUs. Before #1656 every weight — including the routed experts of a
// MoE model — was charged as tensor-sharded across TP only, so an expert-parallel
// deployment (vLLM: ep_size = tp·dp, each rank holding num_experts/ep_size WHOLE
// experts) was over-counted by a factor of DP on its dominant term. For Kimi-K3-class
// models that over-count is the difference between "sizes on its real topology" and
// "auto-calculation fails outright".
//
// These tests pin the arithmetic BASIS (per-GPU, not per-total) rather than merely
// "it fits now", because the plausible-but-wrong implementation — dividing the returned
// TOTAL by ep instead of scaling it by tp/ep — also makes the model fit, while
// under-counting expert weights by a further factor of TP.

// routedExpertWeightBytes recomputes the routed-expert weight bytes from the model
// shape, independently of the production estimator, so the laws below are validated
// from first principles rather than against whatever the code happens to produce (R7).
func routedExpertWeightBytes(mc sim.ModelConfig, params latency.KVCapacityParams) float64 {
	if !params.IsMoE || params.NumLocalExperts < sim.MoEMinExperts {
		return 0
	}
	expertFFNDim := int64(mc.IntermediateDim)
	if params.MoEExpertFFNDim > 0 {
		expertFFNDim = int64(params.MoEExpertFFNDim)
	}
	numLayers := int64(mc.NumLayers)
	numDense := int64(mc.FirstKDenseReplace)
	if numDense < 0 {
		numDense = 0
	}
	if numDense > numLayers {
		numDense = numLayers
	}
	numMoELayers := numLayers - numDense
	params64 := numMoELayers * 3 * int64(mc.HiddenDim) * expertFFNDim * int64(params.NumLocalExperts)
	return float64(params64) * mc.EffectiveWeightBytesPerParam()
}

// TestCalculateKVBlocks_EP_OffIsByteIdentical is BC-1 (INV-6): every spelling of
// "expert parallelism is off" — the option absent, an explicit 0 (unset), an explicit 1
// (sim.EffectiveEPSize's off value), and an explicit group equal to TP (a DP=1
// deployment, where the EP group IS the TP group) — must produce exactly the
// pre-#1656 block count.
func TestCalculateKVBlocks_EP_OffIsByteIdentical(t *testing.T) {
	mc, hc, params := validMoEModelConfig(), validHWConfig(), validMoEKVParams()
	const tp, dp, blockSize, util = 2, 2, int64(16), 0.9

	base, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params)
	if err != nil {
		t.Fatalf("no EP option: %v", err)
	}
	for _, ep := range []int{0, 1, tp} {
		got, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params,
			latency.WithExpertParallelSize(ep))
		if err != nil {
			t.Fatalf("WithExpertParallelSize(%d): %v", ep, err)
		}
		if got != base {
			t.Errorf("BC-1: WithExpertParallelSize(%d) = %d blocks, want %d (EP off must be byte-identical)",
				ep, got, base)
		}
	}
}

// TestCalculateKVBlocks_EP_ShardsRoutedExpertsAcrossTheEPGroup is BC-2, the arithmetic
// basis. The memory freed by expert parallelism must be exactly the routed-expert bytes
// that move off the rank: R·(1 − tp/ep) per rank, so per-GPU expert bytes become R/ep
// (not R/(tp·ep)).
//
// The test is discriminating by construction: it also computes what the delta would be
// under the "divide the total by ep" misreading (R·(1 − 1/ep)) and asserts the observed
// delta is NOT that value, so the tolerance can never be loose enough to accept both.
func TestCalculateKVBlocks_EP_ShardsRoutedExpertsAcrossTheEPGroup(t *testing.T) {
	mc, hc, params := validMoEModelConfig(), validHWConfig(), validMoEKVParams()
	const tp, dp, blockSize, util = 2, 2, int64(16), 0.9
	const ep = tp * dp // vLLM: ep_size = tp·dp under --enable-expert-parallel

	off, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params)
	if err != nil {
		t.Fatalf("EP off: %v", err)
	}
	on, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params,
		latency.WithExpertParallelSize(ep))
	if err != nil {
		t.Fatalf("EP on: %v", err)
	}

	routedBytes := routedExpertWeightBytes(mc, params)
	if routedBytes <= 0 {
		t.Fatalf("test fixture has no routed-expert weights; the law below would be vacuous")
	}
	perBlock := perBlockBytesFor(t, mc, tp, blockSize)

	// Per-rank freed bytes → per-rank blocks. Step 6 scales the per-rank total by dp for
	// MoE, so dividing the observed delta by dp recovers the per-rank gain exactly. The
	// slack is two blocks: one from the float→int64 truncation of the allocatable byte
	// budget, one from the floor division into blocks. It is ~4 orders of magnitude
	// tighter than the gap to the misreading checked below.
	const truncationSlackBlocks = 2.0
	wantPerRank := routedBytes * (1.0 - float64(tp)/float64(ep)) / float64(perBlock)
	gotPerRank := float64(on-off) / float64(dp)
	if math.Abs(gotPerRank-wantPerRank) > truncationSlackBlocks {
		t.Errorf("BC-2: EP freed %.1f blocks/rank, want %.1f (= R·(1 − tp/ep)/perBlock, R=%.0f bytes, tp=%d, ep=%d)",
			gotPerRank, wantPerRank, routedBytes, tp, ep)
	}

	// Anti-assertion: the "divide the TOTAL by ep" misreading would free this much
	// instead, under-counting per-GPU expert weights by a further factor of TP.
	wrongPerRank := routedBytes * (1.0 - 1.0/float64(ep)) / float64(perBlock)
	if math.Abs(gotPerRank-wrongPerRank) <= truncationSlackBlocks {
		t.Errorf("BC-2: freed blocks/rank (%.1f) match the per-TOTAL misreading (%.1f); "+
			"the routed term must be scaled by tp/ep, not 1/ep", gotPerRank, wrongPerRank)
	}
}

// TestCalculateKVBlocks_EP_MonotoneInGroupSize is a shape law: a larger EP group moves
// more expert weight off each rank, so usable KV capacity is non-decreasing in the group
// size and strictly increasing once the group exceeds TP.
func TestCalculateKVBlocks_EP_MonotoneInGroupSize(t *testing.T) {
	mc, hc, params := validMoEModelConfig(), validHWConfig(), validMoEKVParams()
	// dp=4 ⇒ the largest group (tp·dp = 8) equals the fixture's routed-expert count, the
	// widest EP a deployment of this model can have.
	const tp, dp, blockSize, util = 2, 4, int64(16), 0.9

	prev := int64(-1)
	for _, ep := range []int{tp, tp * 2, tp * dp} {
		got, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params,
			latency.WithExpertParallelSize(ep))
		if err != nil {
			t.Fatalf("ep=%d: %v", ep, err)
		}
		if prev >= 0 && got <= prev {
			t.Errorf("ep=%d: %d blocks, want strictly more than the smaller group's %d", ep, got, prev)
		}
		prev = got
	}
}

// TestCalculateKVBlocks_EP_NoRoutedExpertsIsInert is BC-4: expert parallelism can only
// move routed-expert weights. A dense model has none, and a sub-threshold expert count
// takes the dense weight path, so both must be unchanged even when a large EP group is
// supplied — never a phantom capacity gain.
func TestCalculateKVBlocks_EP_NoRoutedExpertsIsInert(t *testing.T) {
	hc := validHWConfig()
	const tp, dp, blockSize, util = 2, 4, int64(16), 0.9

	cases := []struct {
		name   string
		mc     sim.ModelConfig
		params latency.KVCapacityParams
	}{
		{"dense", validDenseModelConfig(), validDenseKVParams()},
		{
			// IsMoE asserted by the caller but only one expert: the weight estimator takes
			// the dense path (MoEMinExperts guard), so there is nothing to shard.
			name:   "moe_sub_threshold_expert_count",
			mc:     validDenseModelConfig(),
			params: latency.NewKVCapacityParams(true, 1, false, "silu", 14336, 0),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			off, err := latency.CalculateKVBlocks(tc.mc, hc, tp, dp, blockSize, util, tc.params)
			if err != nil {
				t.Fatalf("EP off: %v", err)
			}
			on, err := latency.CalculateKVBlocks(tc.mc, hc, tp, dp, blockSize, util, tc.params,
				latency.WithExpertParallelSize(tp*dp))
			if err != nil {
				t.Fatalf("EP on: %v", err)
			}
			if on != off {
				t.Errorf("BC-4: %s must be EP-inert: EP on gave %d blocks, EP off gave %d", tc.name, on, off)
			}
		})
	}
}

// TestCalculateKVBlocks_EP_RejectsInvalidGroupSize is BC-6 (R1/R3): a group size that
// cannot describe a real deployment is an error naming the offending value — never a
// silent clamp (which would hide a mis-plumbed call site) and never an inflated
// footprint.
func TestCalculateKVBlocks_EP_RejectsInvalidGroupSize(t *testing.T) {
	mc, hc, params := validMoEModelConfig(), validHWConfig(), validMoEKVParams()
	const tp, dp, blockSize, util = 4, 2, int64(16), 0.9

	cases := []struct {
		name     string
		ep       int
		wantText string
	}{
		{"negative", -1, "must be >= 0"},
		{"above_tp_times_dp", tp*dp + 1, "must be <= TP·DP"},
		// MaxInt is still the upper-bound check (not the overflow probe, which needs an
		// absurd dp — see TestCalculateKVBlocks_EP_RejectsUncomputableBound).
		{"far_above_tp_times_dp", math.MaxInt, "must be <= TP·DP"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params,
				latency.WithExpertParallelSize(tc.ep))
			if err == nil {
				t.Fatalf("ep=%d must be rejected, got nil error", tc.ep)
			}
			if !strings.Contains(err.Error(), tc.wantText) {
				t.Errorf("ep=%d error %q must explain the bound (%q)", tc.ep, err.Error(), tc.wantText)
			}
		})
	}
}

// TestCalculateKVBlocks_EP_ErrorMessageNamesTheEPGroup verifies the overhead-exceeded
// diagnostic stays honest under expert parallelism: the minimum-GPU estimate holds the
// EP group size fixed (growing TP would grow the group too), so the message says so. The
// EP-off message must NOT carry the clause (byte-identical diagnostics, INV-6).
func TestCalculateKVBlocks_EP_ErrorMessageNamesTheEPGroup(t *testing.T) {
	mc, params := glmFixtureConfig(t), glmFixtureParams(t)
	hc := validHWConfig() // 80 GiB
	const blockSize, util = int64(16), 0.9

	// TP=2, DP=2: too small to hold this model either way, so both branches error.
	_, offErr := latency.CalculateKVBlocks(mc, hc, 2, 2, blockSize, util, params)
	if offErr == nil {
		t.Fatal("EP off at TP=2 must exceed 80 GiB GPUs")
	}
	if strings.Contains(offErr.Error(), "expert-parallel group") {
		t.Errorf("EP-off message must not mention the EP group: %q", offErr.Error())
	}

	_, onErr := latency.CalculateKVBlocks(mc, hc, 2, 2, blockSize, util, params,
		latency.WithExpertParallelSize(4))
	if onErr == nil {
		t.Fatal("EP on at TP=2 must still exceed 80 GiB GPUs for this model")
	}
	if !strings.Contains(onErr.Error(), "expert-parallel group (size 4)") {
		t.Errorf("EP-on message must name the EP group size: %q", onErr.Error())
	}
}

// glmFixtureConfig / glmFixtureParams load the committed GLM-5.2-FP8 fixture — a real
// MLA MoE checkpoint (78 layers, 256 routed experts at moe_intermediate_size=2048, FP8
// weights, first_k_dense_replace=3) whose routed experts are ~97% of its weight bytes.
func glmFixtureConfig(t *testing.T) sim.ModelConfig {
	t.Helper()
	mc, err := latency.GetModelConfig(modelConfigPath("glm-5.2-fp8"))
	if err != nil {
		t.Fatalf("GetModelConfig(glm-5.2-fp8): %v", err)
	}
	return *mc
}

func glmFixtureParams(t *testing.T) latency.KVCapacityParams {
	t.Helper()
	params, err := latency.ExtractKVCapacityParamsFromFile(modelConfigPath("glm-5.2-fp8"))
	if err != nil {
		t.Fatalf("ExtractKVCapacityParamsFromFile(glm-5.2-fp8): %v", err)
	}
	if !params.IsMoE {
		t.Fatalf("fixture must be MoE for this test to mean anything: %+v", params)
	}
	return params
}

// TestCalculateKVBlocks_EP_RealMoEFixtureSizesOnItsEPTopology is BC-8 — the issue's
// blocker, reproduced on a committed fixture. A large MoE deployed at TP=8/DP=2 with
// expert parallelism (EP=16) really does fit on 80 GiB GPUs, because the 675 GiB of
// routed experts are spread over all 16 GPUs; charging them to each rank's 8 GPUs
// instead makes auto-sizing fail. Before #1656 only the failing answer was reachable.
func TestCalculateKVBlocks_EP_RealMoEFixtureSizesOnItsEPTopology(t *testing.T) {
	mc, params := glmFixtureConfig(t), glmFixtureParams(t)
	hc := validHWConfig() // 80 GiB, H100-like
	const tp, dp, blockSize, util = 8, 2, int64(16), 0.9

	if _, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params); err == nil {
		t.Fatal("EP off: expected the weight footprint to exceed the TP-group budget (the #1656 symptom)")
	} else if !strings.Contains(err.Error(), "exceeds available GPU memory") {
		t.Fatalf("EP off: expected the overhead-exceeded error, got %v", err)
	}

	blocks, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params,
		latency.WithExpertParallelSize(tp*dp))
	if err != nil {
		t.Fatalf("EP on (ep=%d): expected the real EP topology to size successfully, got %v", tp*dp, err)
	}
	if blocks <= 0 {
		t.Errorf("EP on: expected a positive block count, got %d", blocks)
	}
}

// TestCalculateKVBlocks_EP_ClampsGroupToRoutedExpertCount pins the wide-EP law: an EP
// group with more GPUs than the model has routed experts is CLAMPED to the expert count,
// not rejected and not charged the sub-one-expert average.
//
// Why clamp: a group wider than the expert count is a real planning input (DeepSeek-class
// EP320 over 256 experts; anything using --enable-eplb / --num-redundant-experts). The
// ranks that hold an expert still hold one WHOLE expert, so num_experts is the widest
// divisor the footprint supports — charging num_experts/ep would hand back capacity for
// memory that does not exist. Rejecting instead would fail deployments whose block count
// the reduction does not even change, and would mask the CLI's own topology diagnostics.
func TestCalculateKVBlocks_EP_ClampsGroupToRoutedExpertCount(t *testing.T) {
	const numExperts, tp, dp = 4, 2, 4
	mc := validMoEModelConfig()
	mc.NumLocalExperts = numExperts
	params := latency.NewKVCapacityParams(true, numExperts, false, "silu", 14336, 0)
	hc := validHWConfig()

	// The widest supportable group: one whole expert per GPU. This one genuinely shards
	// (ep=4 > tp=2), so it is not a vacuous "did not error" check.
	atCount, err := latency.CalculateKVBlocks(mc, hc, tp, dp, 16, 0.9, params,
		latency.WithExpertParallelSize(numExperts))
	if err != nil {
		t.Fatalf("ep=%d (one expert per GPU) must size, got %v", numExperts, err)
	}
	epOff, err := latency.CalculateKVBlocks(mc, hc, tp, dp, 16, 0.9, params)
	if err != nil {
		t.Fatalf("EP off: %v", err)
	}
	if atCount <= epOff {
		t.Fatalf("ep=%d must free routed-expert memory (got %d blocks, EP off gave %d)",
			numExperts, atCount, epOff)
	}

	// tp·dp = 8 is legal for the topology but wider than the 4 experts: it must resolve to
	// the SAME capacity as ep=num_experts, never more.
	wider, err := latency.CalculateKVBlocks(mc, hc, tp, dp, 16, 0.9, params,
		latency.WithExpertParallelSize(tp*dp))
	if err != nil {
		t.Fatalf("ep=%d wider than %d experts must clamp, not error: %v", tp*dp, numExperts, err)
	}
	if wider != atCount {
		t.Errorf("ep=%d must clamp to the routed-expert count (%d blocks), got %d — a wider group "+
			"cannot put less than one whole expert on a loaded rank", tp*dp, atCount, wider)
	}
}

// TestCalculateKVBlocks_EP_GroupAtOrBelowTPNeverFails is the regression guard for the
// clamp's predecessor: an EP group at or below TP is a strict no-op on the block count
// (the reduction needs ep > tp), so validation must never fail there — including when the
// model has FEWER routed experts than TP, which a bound on the expert count would have
// rejected. `--tp 16 --enable-expert-parallel` on an 8-expert Mixtral is exactly that
// shape, and it sizes fine on any build without this feature.
func TestCalculateKVBlocks_EP_GroupAtOrBelowTPNeverFails(t *testing.T) {
	mc := validMoEModelConfig() // 8 routed experts
	params := validMoEKVParams()
	hc := validHWConfig()
	const dp, blockSize, util = 1, int64(16), 0.9

	for _, tp := range []int{2, 8, 16} {
		base, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params)
		if err != nil {
			t.Fatalf("tp=%d EP off: %v", tp, err)
		}
		// ep == tp is what every DP=1 EP-on deployment produces.
		got, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params,
			latency.WithExpertParallelSize(tp))
		if err != nil {
			t.Fatalf("tp=%d, ep=tp (%d experts) must not fail validation: %v", tp, params.NumLocalExperts, err)
		}
		if got != base {
			t.Errorf("tp=%d: ep=tp gave %d blocks, want the EP-off %d (a group equal to TP shards nothing)",
				tp, got, base)
		}
	}
}

// TestCalculateKVBlocks_EP_RejectsUncomputableBound covers the degenerate-topology arm of
// BC-6: Go's int64 multiply wraps silently, so a TP·DP product that overflows must be
// reported rather than trusted as an upper bound (a wrapped product would accept any group
// size and hand back capacity for memory that does not exist).
func TestCalculateKVBlocks_EP_RejectsUncomputableBound(t *testing.T) {
	mc, hc, params := validMoEModelConfig(), validHWConfig(), validMoEKVParams()
	// ep must exceed tp to reach the bound at all (ep <= tp resolves to "no sharding").
	_, err := latency.CalculateKVBlocks(mc, hc, 4, math.MaxInt, 16, 0.9, params,
		latency.WithExpertParallelSize(8))
	if err == nil {
		t.Fatal("an overflowing TP·DP bound must be rejected, got nil error")
	}
	if !strings.Contains(err.Error(), "overflows") {
		t.Errorf("error must name the overflow, got %q", err.Error())
	}
}

// TestCalculateKVBlocks_EP_ExactLawOnDensePrefixSharedExpertModel runs BC-2's exact-bytes
// law on the committed glm-5.2-fp8 fixture, which — unlike the Mixtral-shaped fixture —
// has BOTH a dense prefix (first_k_dense_replace=3 of 78 layers) and a shared expert
// (n_shared_experts=1). Those are the two terms that must NOT move with the EP group:
// an implementation that scaled all layers instead of the MoE layers, or that folded the
// shared expert into the routed subtotal, produces a different freed-bytes figure and
// fails here while passing every pass/fail test in the file.
func TestCalculateKVBlocks_EP_ExactLawOnDensePrefixSharedExpertModel(t *testing.T) {
	mc, params := glmFixtureConfig(t), glmFixtureParams(t)
	hc := validHWConfig() // 80 GiB
	// tp=16 fits the model both ways (EP off is ~697 GiB against a 1152 GiB budget), so the
	// delta is measurable rather than one side erroring.
	const tp, dp, blockSize, util = 16, 2, int64(16), 0.9
	const ep = tp * dp

	if params.SharedExpertFFNDim <= 0 || mc.FirstKDenseReplace <= 0 {
		t.Fatalf("fixture must have a shared expert and a dense prefix for this law to bite: shared=%d firstKDense=%d",
			params.SharedExpertFFNDim, mc.FirstKDenseReplace)
	}

	off, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params)
	if err != nil {
		t.Fatalf("EP off: %v", err)
	}
	on, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params,
		latency.WithExpertParallelSize(ep))
	if err != nil {
		t.Fatalf("EP on: %v", err)
	}

	routedBytes := routedExpertWeightBytes(mc, params)
	perBlock := perBlockBytesFor(t, mc, tp, blockSize)
	wantPerRank := routedBytes * (1.0 - float64(tp)/float64(ep)) / float64(perBlock)
	gotPerRank := float64(on-off) / float64(dp)
	if math.Abs(gotPerRank-wantPerRank) > 2.0 {
		t.Errorf("BC-2 on glm-5.2-fp8: EP freed %.1f blocks/rank, want %.1f — only the routed-expert "+
			"term over the %d MoE layers may move (not the %d dense-prefix layers, not the shared expert)",
			gotPerRank, wantPerRank, mc.NumLayers-mc.FirstKDenseReplace, mc.FirstKDenseReplace)
	}
}

// TestCalculateKVBlocks_EP_RoutedWeightConservation is the conservation companion to the
// delta law, and the shortest statement of what expert parallelism means: summed over the
// deployment's dp ranks, the routed-expert weight charged at ep = tp·dp is exactly R —
// the experts are stored ONCE for the whole deployment — whereas with EP off it is dp·R
// (a full copy per independent replica, BLIS's #1531 DP model). Asserted in bytes via the
// allocatable-memory difference, so it holds under any refactor of how the divisor is
// plumbed.
func TestCalculateKVBlocks_EP_RoutedWeightConservation(t *testing.T) {
	mc, hc, params := validMoEModelConfig(), validHWConfig(), validMoEKVParams()
	const tp, dp, blockSize, util = 2, 4, int64(16), 0.9
	const ep = tp * dp

	off, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params)
	if err != nil {
		t.Fatalf("EP off: %v", err)
	}
	on, err := latency.CalculateKVBlocks(mc, hc, tp, dp, blockSize, util, params,
		latency.WithExpertParallelSize(ep))
	if err != nil {
		t.Fatalf("EP on: %v", err)
	}

	routedBytes := routedExpertWeightBytes(mc, params)
	perBlock := perBlockBytesFor(t, mc, tp, blockSize)

	// Bytes freed across the whole deployment = dp·R − R = (dp−1)·R.
	wantFreedTotal := routedBytes * float64(dp-1)
	gotFreedTotal := float64(on-off) * float64(perBlock)
	// Slack: two blocks of truncation per rank, converted back to bytes.
	slack := 2.0 * float64(dp) * float64(perBlock)
	if math.Abs(gotFreedTotal-wantFreedTotal) > slack {
		t.Errorf("routed-weight conservation: EP freed %.0f bytes deployment-wide, want %.0f = (dp−1)·R "+
			"(R=%.0f, dp=%d) — at ep=tp·dp the experts are stored exactly once",
			gotFreedTotal, wantFreedTotal, routedBytes, dp)
	}
}
