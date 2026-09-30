package latency_test

// Tests for the TP BASIS of CalculateKVBlocks' block-count division (#1846).
//
// The bug this file guards against: the memory budget is a TOTAL across the rank's tp
// GPUs, while KVBytesPerToken returns a PER-GPU cost. Dividing the first by the second
// inflated the auto-calculated pool by ~tp — 8.7x on a granite-5.0-230b-sft / H200 / TP8
// deployment whose real vLLM pool was 480,473 blocks, enough to make the KV-exhaustion
// knee of a capacity study disappear entirely.
//
// Why a whole file of its own: the bug was invisible for months because EVERY
// absolute-value CalculateKVBlocks test ran at TP=1, where the factor is exactly 1 (a
// genuine no-op), and every TP>1 test asserted only relative laws (ratios, monotonicity,
// EP/DP deltas) that the inflated count satisfies just as well as the correct one. The
// four laws below are each chosen to be violated by a factor-of-tp error:
//
//  1. TP=8 / TP=16 known-answer golden values (absolute, exact ==).
//  2. The per-TP slope law that validates those goldens from first principles, with no
//     reference to the implementation's own weight/overhead arithmetic.
//  3. Physical realizability: the pool must fit on the GPUs it claims to live on.
//  4. TP=1 byte-identity (INV-6) — the correction must be inert on one GPU.

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

const (
	// bytesPerGiB mirrors the gibToBytes conversion CalculateKVBlocks uses.
	bytesPerGiB = float64(1 << 30)
	// nonTorchPerGPUMultiGiB / nonTorchPerGPUTP1GiB restate the documented per-GPU
	// non-PyTorch overhead constants (NCCL buffers, CUDA context) from
	// kv_capacity.go. They are restated rather than exported because the slope law
	// below needs the ONE overhead term that genuinely scales with tp; if these
	// constants are ever retuned, this file's expectations move with them and that
	// is the intended coupling.
	nonTorchPerGPUMultiGiB = 0.6
	nonTorchPerGPUTP1GiB   = 0.15
)

// llama70bModelConfig is the Llama-3.1-70B shape (80 layers, 8192 hidden, 64 heads,
// 8 GQA KV heads, bf16). Used for a second, independently-shaped high-TP fixture so the
// goldens below cannot all be satisfied by one accidental coincidence.
func llama70bModelConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:       80,
		HiddenDim:       8192,
		NumHeads:        64,
		NumKVHeads:      8,
		VocabSize:       128256,
		BytesPerParam:   2,
		IntermediateDim: 28672,
	}
}

// h200HWConfig is the catalog H200 entry's memory (141 GiB HBM3e); only MemoryGiB is
// read by CalculateKVBlocks.
func h200HWConfig() sim.HardwareCalib {
	hc := validHWConfig()
	hc.MemoryGiB = 141.0
	return hc
}

// tpBasisCase is one (model, GPU, TP) point with its pinned block count.
type tpBasisCase struct {
	name       string
	mc         sim.ModelConfig
	hc         sim.HardwareCalib
	params     latency.KVCapacityParams
	tp         int
	wantBlocks int64
	// wantPreFixBlocks is the count the pre-#1846 aggregate-over-per-GPU division
	// produced for the same inputs, recorded so the anti-assertion below is a real
	// measurement of the regression rather than a restatement of wantBlocks*tp.
	wantPreFixBlocks int64
}

func tpBasisCases() []tpBasisCase {
	return []tpBasisCase{
		// Llama-3.1-8B on an 80 GiB H100, bf16, block size 16, util 0.9.
		// First principles: per-token KV over the whole model is
		// 32 layers x 2 (K+V) x 128 headDim x 8 KV heads x 2 B = 131,072 B, so one
		// 16-token block costs 2,097,152 B across the TP group at any tp, and
		// 1 GiB of freed budget buys exactly 512 blocks. The budget is
		// 80 x 0.9 x tp GiB less 14.96 GiB of weights (8.03B params x 2 B),
		// 5.5 GiB of activation (charged once) and 0.6 GiB/GPU of non-torch, i.e.
		// (71.4 x tp - 20.46) GiB, so blocks = floor((71.4 x tp - 20.46) x 512).
		// tp=8:  (571.2 - 20.46) x 512 = 281,979.x  -> 281,980
		// tp=16: (1142.4 - 20.46) x 512 = 574,433.x -> 574,434
		{
			name: "llama-3.1-8b/H100/tp8", mc: validDenseModelConfig(), hc: validHWConfig(),
			params: validDenseKVParams(), tp: 8, wantBlocks: 281980, wantPreFixBlocks: 2255841,
		},
		{
			name: "llama-3.1-8b/H100/tp16", mc: validDenseModelConfig(), hc: validHWConfig(),
			params: validDenseKVParams(), tp: 16, wantBlocks: 574434, wantPreFixBlocks: 9190952,
		},
		// Llama-3.1-70B on a 141 GiB H200. Per-token KV over the whole model is
		// 80 x 2 x 128 x 8 x 2 = 327,680 B, so one block costs 5,242,880 B across the
		// group and 1 GiB buys 204.8 blocks. A 70B model does not fit on one H200, so
		// this fixture exercises the high-TP regime the bug was reported in.
		{
			name: "llama-3.1-70b/H200/tp8", mc: llama70bModelConfig(), hc: h200HWConfig(),
			params: validDenseKVParams(), tp: 8, wantBlocks: 178889, wantPreFixBlocks: 1431115,
		},
		{
			name: "llama-3.1-70b/H200/tp16", mc: llama70bModelConfig(), hc: h200HWConfig(),
			params: validDenseKVParams(), tp: 16, wantBlocks: 385819, wantPreFixBlocks: 6173109,
		},
		// Mixtral-8x7B-shaped MoE on an 80 GiB H100 (same attention shape as the 8B
		// fixture, 8 routed experts of 14336). Covers the MoE branch: the higher 8.0 GiB
		// activation constant and a much larger weight term, neither of which may change
		// the tp basis of the division.
		{
			name: "mixtral-8x7b/H100/tp8", mc: validMoEModelConfig(), hc: validHWConfig(),
			params: validMoEKVParams(), tp: 8, wantBlocks: 243067, wantPreFixBlocks: 1944537,
		},
		{
			name: "mixtral-8x7b/H100/tp16", mc: validMoEModelConfig(), hc: validHWConfig(),
			params: validMoEKVParams(), tp: 16, wantBlocks: 535521, wantPreFixBlocks: 8568344,
		},
	}
}

// TestCalculateKVBlocks_KnownAnswer_HighTP pins the auto-calculated block count to an
// exact value at TP=8 and TP=16 (#1846). This is the assertion whose absence let a
// factor-of-tp error live in the auto-calc: it is not a ratio, not a monotonic
// direction, and not "robust to the overhead/truncation constants" — it is the number.
//
// The goldens are derived in the comments on each case, and validated independently of
// the implementation by TestCalculateKVBlocks_PerTPSlopeLaw (which cancels the weight
// and activation terms out entirely) and by
// TestCalculateKVBlocks_PoolIsPhysicallyRealizable. So they are not golden-only values
// that could perpetuate the behavior they were captured from (R7 / test epistemology).
//
// Each case also asserts the count is NOT the pre-#1846 value, so no tolerance or
// refactor can leave both readings acceptable.
func TestCalculateKVBlocks_KnownAnswer_HighTP(t *testing.T) {
	for _, tc := range tpBasisCases() {
		t.Run(tc.name, func(t *testing.T) {
			got, err := latency.CalculateKVBlocks(tc.mc, tc.hc, tc.tp, 1, 16, 0.9, tc.params)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.wantBlocks {
				t.Errorf("blocks = %d, want %d (TP=%d, %.0f GiB GPU)", got, tc.wantBlocks, tc.tp, tc.hc.MemoryGiB)
			}
			if got == tc.wantPreFixBlocks {
				t.Errorf("blocks = %d is the pre-#1846 count (aggregate budget divided by the PER-GPU "+
					"per-block cost, ~%dx too large); the divisor must be the per-GPU cost on all %d GPUs",
					got, tc.tp, tc.tp)
			}
		})
	}
}

// TestCalculateKVBlocks_PerTPSlopeLaw validates the goldens above from first principles,
// without duplicating any of the implementation's weight or activation arithmetic.
//
// The law: for a fixed model on a fixed GPU, the only tp-dependent quantities in the
// budget are the aggregate available memory (gpu_mem x util x tp) and the non-torch
// overhead (0.6 GiB x tp). Model weights and activation are tp-independent per rank, so
// they CANCEL between any two tp values >= 2:
//
//	blocks(b) - blocks(a) == (gpu_mem x util - 0.6) x (b - a) x 2^30 / blockCostBytes
//
// where blockCostBytes is what one block of the pool costs across the whole TP group
// (per-GPU bytes x tp) — a tp-invariant equal to the whole model's per-token KV bytes x
// block size on the MHA/GQA path. Both sides are computed here from published hardware
// numbers and the KV shape alone.
//
// This law is exactly what the bug broke. Pre-#1846 the block count carried a factor of
// tp, so the left-hand side grew super-linearly in tp and missed the right-hand side by
// hundreds of thousands of blocks. tp >= 2 throughout, because tp=1 uses the smaller
// 0.15 GiB/GPU non-torch constant and so is not on this line.
//
// It ALSO pins the activation-memory basis (#1846 acceptance criterion 3): activation is
// subtracted ONCE per rank, not x tp. Were it x tp, the slope would be
// (gpu_mem x util - 0.6 - activationGiB) per tp and every step would be off by
// thousands of blocks. The anti-assertion below states that explicitly, so the
// asymmetry is a tested decision rather than an accident — the residual it leaves (the
// reason a corrected estimate lands ~1.09x a measured pool rather than ~1.0x) is
// documented at the subtraction site and tracked separately.
func TestCalculateKVBlocks_PerTPSlopeLaw(t *testing.T) {
	cases := []struct {
		name          string
		mc            sim.ModelConfig
		hc            sim.HardwareCalib
		params        latency.KVCapacityParams
		activationGiB float64 // the constant the MoE/dense branch subtracts, for the anti-assertion
		tps           []int
	}{
		{"llama-3.1-8b/H100", validDenseModelConfig(), validHWConfig(), validDenseKVParams(), 5.5, []int{2, 4, 8, 16}},
		{"llama-3.1-70b/H200", llama70bModelConfig(), h200HWConfig(), validDenseKVParams(), 5.5, []int{2, 4, 8, 16}},
		{"mixtral-8x7b/H100", validMoEModelConfig(), validHWConfig(), validMoEKVParams(), 8.0, []int{2, 4, 8, 16}},
	}

	const blockSize = int64(16)
	const util = 0.9
	// Slack of two blocks: one from the float->int64 truncation of the allocatable byte
	// budget, one from the floor division into blocks. Four orders of magnitude tighter
	// than the gap to either misreading checked below.
	const slackBlocks = 2.0

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			blockCost := blockBudgetCostFor(t, tc.mc, tc.tps[0], blockSize)
			// Per-GiB-of-budget block yield, from the KV shape alone.
			blocksPerGiB := bytesPerGiB / float64(blockCost)
			// Budget gained per additional GPU: its whole allotment less its non-torch share.
			gainPerTP := tc.hc.MemoryGiB*util - nonTorchPerGPUMultiGiB

			counts := make(map[int]int64, len(tc.tps))
			for _, tp := range tc.tps {
				// blockCost must be tp-invariant for the law to be a law; assert it rather
				// than assume it, so an MLA-style (TP-replicated) KV shape cannot silently
				// be measured against a formula that does not apply to it.
				if c := blockBudgetCostFor(t, tc.mc, tp, blockSize); c != blockCost {
					t.Fatalf("per-group block cost is not TP-invariant for this fixture: %d at TP=%d vs %d at TP=%d",
						c, tp, blockCost, tc.tps[0])
				}
				got, err := latency.CalculateKVBlocks(tc.mc, tc.hc, tp, 1, blockSize, util, tc.params)
				if err != nil {
					t.Fatalf("TP=%d: %v", tp, err)
				}
				counts[tp] = got
			}

			for i := 1; i < len(tc.tps); i++ {
				a, b := tc.tps[i-1], tc.tps[i]
				got := float64(counts[b] - counts[a])
				want := gainPerTP * float64(b-a) * blocksPerGiB
				if math.Abs(got-want) > slackBlocks {
					t.Errorf("blocks(TP=%d) - blocks(TP=%d) = %.0f, want %.1f "+
						"(= (%.1f GiB x %.2f util - %.2f GiB non-torch) x %d GPUs x %.4f blocks/GiB)",
						b, a, got, want, tc.hc.MemoryGiB, util, nonTorchPerGPUMultiGiB, b-a, blocksPerGiB)
				}

				// Anti-assertion 1: the pre-#1846 per-GPU divisor. Its slope is not even
				// affine in tp, so no single value is "the" wrong answer — but the
				// observed step must at minimum not match the per-GPU-divisor step.
				wrongPerGPUDivisor := gainPerTP * float64(b-a) * blocksPerGiB * float64(b)
				if math.Abs(got-wrongPerGPUDivisor) <= slackBlocks {
					t.Errorf("blocks(TP=%d) - blocks(TP=%d) = %.0f matches the pre-#1846 per-GPU-divisor "+
						"step (%.1f): the aggregate budget must be divided by the per-block cost on ALL %d GPUs",
						b, a, got, wrongPerGPUDivisor, b)
				}

				// Anti-assertion 2: activation charged x tp instead of once (#1846 AC-3).
				wrongActivationBasis := (gainPerTP - tc.activationGiB) * float64(b-a) * blocksPerGiB
				if math.Abs(got-wrongActivationBasis) <= slackBlocks {
					t.Errorf("blocks(TP=%d) - blocks(TP=%d) = %.0f matches an activation term charged "+
						"x TP (%.1f); activation is a per-rank constant subtracted ONCE",
						b, a, got, wrongActivationBasis)
				}
			}
		})
	}
}

// TestCalculateKVBlocks_PoolIsPhysicallyRealizable asserts the invariant the bug report
// named: the auto-calculated pool must fit on the hardware it is sized for. A KV block
// is a GLOBAL quantity (a request is charged ceil(InputLen/BlockSize) blocks once
// against the pool, and the pool is reported as TotalBlocks x BlockSizeTokens tokens), so
// each of the tp ranks must hold its own shard of every block:
//
//	blocks x blockSize x perGPUBytesPerToken(tp) + (weights + activation)/tp + nonTorch
//	    <= gpu_mem x util          [all per GPU]
//
// The tp-independent (weights + activation) total is recovered from the TP=1 call, whose
// arithmetic this PR does not touch and which is separately anchored to real deployments
// (Llama-3.1-8B on one 80 GiB H100: ~26.3K blocks = ~421K tokens, matching vLLM's
// reported pool). So the check is not circular — nothing on the left is derived from the
// TP>1 result being checked.
//
// Pre-#1846 this failed at every tp >= 2: at TP=8 the 8B fixture claimed 2,255,841
// blocks, which is ~551 GiB of KV per GPU on an 80 GiB card.
func TestCalculateKVBlocks_PoolIsPhysicallyRealizable(t *testing.T) {
	mc, hc, params := validDenseModelConfig(), validHWConfig(), validDenseKVParams()
	const blockSize = int64(16)
	const util = 0.9
	perGPUBudgetGiB := hc.MemoryGiB * util

	// Recover the tp-independent weights+activation total from the single-GPU budget:
	//   blocks(1) x blockCost(1) = (gpu_mem x util - (W+A) - 0.15) x 2^30
	blocks1, err := latency.CalculateKVBlocks(mc, hc, 1, 1, blockSize, util, params)
	if err != nil {
		t.Fatalf("TP=1: %v", err)
	}
	kvGiB1 := float64(blocks1*blockBudgetCostFor(t, mc, 1, blockSize)) / bytesPerGiB
	weightsPlusActivationGiB := perGPUBudgetGiB - nonTorchPerGPUTP1GiB - kvGiB1
	if weightsPlusActivationGiB <= 0 {
		t.Fatalf("recovered weights+activation total is %.2f GiB; fixture cannot anchor the check",
			weightsPlusActivationGiB)
	}

	for _, tp := range []int{1, 2, 4, 8, 16} {
		blocks, err := latency.CalculateKVBlocks(mc, hc, tp, 1, blockSize, util, params)
		if err != nil {
			t.Fatalf("TP=%d: %v", tp, err)
		}
		nonTorch := nonTorchPerGPUMultiGiB
		if tp == 1 {
			nonTorch = nonTorchPerGPUTP1GiB
		}
		kvPerGPUGiB := float64(blocks) * float64(blockSize) * perGPUKVBytesPerToken(t, mc, tp) / bytesPerGiB
		usedPerGPUGiB := kvPerGPUGiB + weightsPlusActivationGiB/float64(tp) + nonTorch

		// Tolerance of one block's per-GPU bytes absorbs the two truncations.
		oneBlockGiB := float64(blockSize) * perGPUKVBytesPerToken(t, mc, tp) / bytesPerGiB
		if usedPerGPUGiB > perGPUBudgetGiB+oneBlockGiB {
			t.Errorf("TP=%d: auto-calculated pool of %d blocks needs %.2f GiB per GPU "+
				"(%.2f KV + %.2f weights+activation share + %.2f non-torch) but only %.2f GiB is available "+
				"per GPU — the pool is not physically realizable",
				tp, blocks, usedPerGPUGiB, kvPerGPUGiB, weightsPlusActivationGiB/float64(tp), nonTorch,
				perGPUBudgetGiB)
		}
	}
}

// perGPUKVBytesPerToken is KVBytesPerToken's per-GPU value, named at the call site so
// the realizability arithmetic above cannot be misread as using group totals.
func perGPUKVBytesPerToken(t *testing.T, mc sim.ModelConfig, tp int) float64 {
	t.Helper()
	v, err := latency.KVBytesPerToken(mc, tp)
	if err != nil {
		t.Fatalf("KVBytesPerToken(TP=%d): %v", tp, err)
	}
	return v
}

// TestCalculateKVBlocks_TP1IsByteIdenticalAcrossTheFix pins the TP=1 block counts the
// #1846 correction must NOT move (INV-6). At tp=1 the aggregate budget and the per-GPU
// budget are the same thing, so the new divisor (per-GPU cost x 1) is the old one and
// every single-GPU result — including every pre-existing golden in this package — is
// unchanged. These literals were captured from a pre-#1846 build.
func TestCalculateKVBlocks_TP1IsByteIdenticalAcrossTheFix(t *testing.T) {
	cases := []struct {
		name   string
		mc     sim.ModelConfig
		hc     sim.HardwareCalib
		params latency.KVCapacityParams
		want   int64
	}{
		{"llama-3.1-8b/H100", validDenseModelConfig(), validHWConfig(), validDenseKVParams(), 26312},
		{"llama-3.1-8b/H200", validDenseModelConfig(), h200HWConfig(), validDenseKVParams(), 54421},
		{"mixtral-8x7b/H200", validMoEModelConfig(), h200HWConfig(), validMoEKVParams(), 15508},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := latency.CalculateKVBlocks(tc.mc, tc.hc, 1, 1, 16, 0.9, tc.params)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("TP=1 blocks = %d, want %d — the #1846 correction must be inert on a single GPU (INV-6)",
					got, tc.want)
			}
		})
	}
}

// TestCalculateKVBlocks_DenseDP1Unaffected re-checks INV-BC-DP1 across the correction:
// for a dense model the block count is dp-independent (only an MoE model's aggregate
// scales with dp), at every tp — the fix changes the divisor, never the dp gate.
func TestCalculateKVBlocks_DenseDP1Unaffected(t *testing.T) {
	mc, hc, params := validDenseModelConfig(), validHWConfig(), validDenseKVParams()
	for _, tp := range []int{1, 2, 8, 16} {
		base, err := latency.CalculateKVBlocks(mc, hc, tp, 1, 16, 0.9, params)
		if err != nil {
			t.Fatalf("TP=%d dp=1: %v", tp, err)
		}
		for _, dp := range []int{2, 4} {
			got, err := latency.CalculateKVBlocks(mc, hc, tp, dp, 16, 0.9, params)
			if err != nil {
				t.Fatalf("TP=%d dp=%d: %v", tp, dp, err)
			}
			if got != base {
				t.Errorf("INV-BC-DP1: dense TP=%d dp=%d blocks = %d, want %d (dense capacity is dp-independent)",
					tp, dp, got, base)
			}
		}
	}
}
