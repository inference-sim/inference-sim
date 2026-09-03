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
		{"below_tp", tp - 1, "must be >= TP"},
		{"above_tp_times_dp", tp*dp + 1, "must be <= TP·DP"},
		{"overflow_guard", math.MaxInt, "must be <= TP·DP"},
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

// TestCalculateKVBlocks_EP_RejectsGroupWiderThanExpertCount is the other half of BC-6: an
// EP group with more GPUs than the model has routed experts would leave ranks holding no
// expert, while the per-rank average this model charges (num_experts/ep whole experts)
// would drop below one expert on the ranks that do hold them. vLLM does not build that
// deployment, so it is rejected rather than sized optimistically.
func TestCalculateKVBlocks_EP_RejectsGroupWiderThanExpertCount(t *testing.T) {
	const numExperts, tp, dp = 4, 4, 2
	mc := validMoEModelConfig()
	mc.NumLocalExperts = numExperts
	params := latency.NewKVCapacityParams(true, numExperts, false, "silu", 14336, 0)
	hc := validHWConfig()

	// tp·dp = 8 is a legal group size for the topology but wider than the 4 experts.
	_, err := latency.CalculateKVBlocks(mc, hc, tp, dp, 16, 0.9, params,
		latency.WithExpertParallelSize(tp*dp))
	if err == nil {
		t.Fatalf("ep=%d with only %d routed experts must be rejected", tp*dp, numExperts)
	}
	if !strings.Contains(err.Error(), "must be <= the routed-expert count") {
		t.Errorf("error must explain the expert-count bound, got %q", err.Error())
	}

	// The widest legal group — one expert per GPU — still sizes.
	if _, err := latency.CalculateKVBlocks(mc, hc, tp, dp, 16, 0.9, params,
		latency.WithExpertParallelSize(numExperts)); err != nil {
		t.Errorf("ep=%d (one expert per GPU) must be accepted, got %v", numExperts, err)
	}
}

// TestCalculateKVBlocks_EP_RejectsUncomputableBound covers the degenerate-topology arm of
// BC-6: Go's int64 multiply wraps silently, so a TP·DP product that overflows must be
// reported rather than trusted as an upper bound (a wrapped product would accept any group
// size and hand back capacity for memory that does not exist).
func TestCalculateKVBlocks_EP_RejectsUncomputableBound(t *testing.T) {
	mc, hc, params := validMoEModelConfig(), validHWConfig(), validMoEKVParams()
	_, err := latency.CalculateKVBlocks(mc, hc, 4, math.MaxInt, 16, 0.9, params,
		latency.WithExpertParallelSize(4))
	if err == nil {
		t.Fatal("an overflowing TP·DP bound must be rejected, got nil error")
	}
	if !strings.Contains(err.Error(), "overflows") {
		t.Errorf("error must name the overflow, got %q", err.Error())
	}
}
