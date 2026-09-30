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
//  4. Agreement with the two REAL MEASURED pools the bug was reported against, within a
//     stated tolerance (the last sentence of #1846 acceptance criterion 2).
//  5. TP=1 byte-identity (INV-6) — the correction must be inert on one GPU.
//
// One cross-cutting caveat, which every test below states at its own site: KVBytesPerToken
// divides the model's KV bytes by tp even when numKVHeads < tp, where vLLM would instead
// replicate KV heads per rank (a documented optimistic approximation that predates #1846
// and is out of its scope). A TP=16 assertion on an 8-KV-head fixture therefore pins the
// aggregate UNITS basis — the thing this file guards — without being a physically
// realizable per-GPU derivation. The llama-2-7b (32 KV heads) and llama-3.1-405b (16 KV
// heads) fixtures have numKVHeads divisible by 16, so they carry the first-principles
// claim at TP=16 that #1846 acceptance criterion 2 asks for.

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

// --- Fixtures whose KV-head count is a multiple of TP=16 (no head replication) ---
//
// The three GQA fixtures above and below all carry 8 KV heads, which is FEWER than 16.
// KVBytesPerToken documents that regime as a known optimistic approximation
// (kv_capacity.go, "when numKVHeads < TP ... vLLM replicates KV heads per GPU; dividing
// by TP underestimates per-GPU KV bytes"): the code divides the whole model's KV bytes by
// tp regardless, so at TP=16 an 8-KV-head fixture's implied per-GPU cost is half what vLLM
// would actually allocate. That approximation predates #1846 and is out of its scope — but
// it means a TP=16 golden on an 8-KV-head fixture pins the UNITS BASIS of the division
// without being a physically realizable per-GPU derivation.
//
// The two fixtures below have numKVHeads % 16 == 0, so at TP=16 the heads divide evenly,
// no replication applies, and their goldens are first-principles all the way down — which
// is what #1846 acceptance criterion 2 asks for at TP=16.

// llama2MHAModelConfig is the Llama-2-7B shape (32 layers, 4096 hidden, 32 heads, 32 KV
// heads — MHA, no GQA grouping, so numKVHeads is divisible by every tp in 1..16, bf16,
// 11008 FFN, 32000 vocab). It is the only fixture in this file that is on the exact
// per-GPU basis at EVERY tp tested AND small enough to fit on one 80 GiB H100, which is
// what lets the realizability check anchor it from its own TP=1 call.
func llama2MHAModelConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      32,
		VocabSize:       32000,
		BytesPerParam:   2,
		IntermediateDim: 11008,
	}
}

// llama405bModelConfig is the Llama-3.1-405B shape (126 layers, 16384 hidden, 128 heads,
// 16 GQA KV heads, 53248 FFN, bf16). 16 KV heads divides both 8 and 16, so this fixture is
// exact at both high-TP points — and unlike the synthetic option of bumping an 8B model's
// head count, it is a real deployment that genuinely REQUIRES TP>=8 (its weights alone are
// ~764 GiB), i.e. the regime #1846 was reported in. It cannot be exercised below TP=7 at
// all: CalculateKVBlocks refuses TP=2 and TP=4 on a 141 GiB H200 because the weight term
// exceeds the aggregate budget.
func llama405bModelConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:       126,
		HiddenDim:       16384,
		NumHeads:        128,
		NumKVHeads:      16,
		VocabSize:       128256,
		BytesPerParam:   2,
		IntermediateDim: 53248,
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
	// headReplicationRegime marks a case where numKVHeads < tp, i.e. where vLLM would
	// replicate KV heads across ranks but KVBytesPerToken divides by tp anyway (the
	// documented optimistic approximation, pre-existing and out of #1846's scope). Such a
	// case still pins the aggregate UNITS basis this file exists to guard — the divisor's
	// tp factor is what it measures — but its per-GPU cost is not a physically realizable
	// number, so it is not first-principles coverage in the sense acceptance criterion 2
	// asks for. Recorded per case rather than left implicit so the distinction is visible
	// in the table instead of buried in prose.
	headReplicationRegime bool
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
			headReplicationRegime: true, // 8 KV heads < tp=16
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
			headReplicationRegime: true, // 8 KV heads < tp=16
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
			headReplicationRegime: true, // 8 KV heads < tp=16
		},
		// --- Exact-at-TP=16 fixtures (numKVHeads % 16 == 0, no head replication) ---
		//
		// Llama-2-7B on an 80 GiB H100, bf16, block size 16, util 0.9. MHA: 32 KV heads, so
		// every tp in 1..16 divides them evenly and the per-GPU cost below is the physical
		// one, not the approximation. Per-token KV over the whole model is
		// 32 layers x 2 (K+V) x 128 headDim x 32 KV heads x 2 B = 524,288 B, so one 16-token
		// block costs exactly 8 MiB across the TP group at any tp and 1 GiB of freed budget
		// buys exactly 128 blocks. Weights are 6.738B params x 2 B = 12.5513 GiB (the
		// standard transformer count for this shape: 2 x 32000 x 4096 embed+lm_head,
		// 32 x (4096 x (4096 + 2 x 4096) + 4096 x 4096 + 2 x 4096) attention+norms, and
		// 32 x 3 x 4096 x 11008 SwiGLU MLP). Budget is (71.4 x tp - 18.0513) GiB, so
		// blocks = floor((71.4 x tp - 18.0513) x 128).
		// tp=8:  (571.2 - 18.0513) x 128  = 70,803.x  -> 70,803
		// tp=16: (1142.4 - 18.0513) x 128 = 143,916.x -> 143,916
		{
			name: "llama-2-7b/H100/tp8", mc: llama2MHAModelConfig(), hc: validHWConfig(),
			params: validDenseKVParams(), tp: 8, wantBlocks: 70803, wantPreFixBlocks: 566424,
		},
		{
			name: "llama-2-7b/H100/tp16", mc: llama2MHAModelConfig(), hc: validHWConfig(),
			params: validDenseKVParams(), tp: 16, wantBlocks: 143916, wantPreFixBlocks: 2302666,
		},
		// Llama-3.1-405B on a 141 GiB H200. 16 GQA KV heads divides both 8 and 16 evenly, so
		// both points are exact. Per-token KV over the whole model is
		// 126 x 2 x 128 x 16 x 2 = 1,032,192 B, so one block costs 16,515,072 B across the
		// group and 1 GiB buys 65.0159 blocks. Weights are 410.081B params x 2 B =
		// 763.8358 GiB (same standard count, at 16384 hidden / 53248 FFN / 128256 vocab).
		// Budget is (126.3 x tp - 769.3358) GiB, so blocks = floor(budget x 65.0159).
		// tp=8:  241.0642 x 65.0159  = 15,672.x -> 15,672
		// tp=16: 1251.4642 x 65.0159 = 81,365.x -> 81,365
		{
			name: "llama-3.1-405b/H200/tp8", mc: llama405bModelConfig(), hc: h200HWConfig(),
			params: validDenseKVParams(), tp: 8, wantBlocks: 15672, wantPreFixBlocks: 125383,
		},
		{
			name: "llama-3.1-405b/H200/tp16", mc: llama405bModelConfig(), hc: h200HWConfig(),
			params: validDenseKVParams(), tp: 16, wantBlocks: 81365, wantPreFixBlocks: 1301840,
		},
	}
}

// TestCalculateKVBlocks_KnownAnswerFixtures_CoverExactTP16 guards the PROPERTY that makes
// the goldens above first-principles rather than self-consistent: at least one TP=16 case
// must be outside the head-replication approximation, and every case claiming to be
// outside it must really have numKVHeads % tp == 0.
//
// Without this, the table could drift back to all-8-KV-head fixtures at TP=16 — the exact
// gap @namasl's review caught — and every other test in this file would still pass, because
// the aggregate per-block cost is tp-invariant whether or not the per-GPU value it divides
// into is physically realizable.
func TestCalculateKVBlocks_KnownAnswerFixtures_CoverExactTP16(t *testing.T) {
	exactTP16 := 0
	for _, tc := range tpBasisCases() {
		kvHeads := tc.mc.NumKVHeads
		if kvHeads == 0 {
			kvHeads = tc.mc.NumHeads
		}
		replicates := kvHeads < tc.tp
		if replicates != tc.headReplicationRegime {
			t.Errorf("%s: headReplicationRegime = %v but numKVHeads=%d vs tp=%d says %v",
				tc.name, tc.headReplicationRegime, kvHeads, tc.tp, replicates)
		}
		if tc.tp == 16 && !replicates {
			if kvHeads%tc.tp != 0 {
				t.Errorf("%s: %d KV heads is not a multiple of tp=16, so the per-GPU cost is not exact",
					tc.name, kvHeads)
			}
			exactTP16++
		}
	}
	if exactTP16 == 0 {
		t.Errorf("no TP=16 case has numKVHeads >= 16: every TP=16 golden would then rest on the "+
			"documented head-replication approximation (numKVHeads < tp), which is not the "+
			"first-principles per-GPU basis #1846 acceptance criterion 2 asks for; got %d exact cases",
			exactTP16)
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
// ONE LIMIT ON "first principles", and it is the reason the table carries a
// headReplicationRegime flag: at TP=16 the three 8-KV-head GQA fixtures sit in the
// approximation KVBytesPerToken documents (numKVHeads < tp ⇒ vLLM replicates heads per
// rank, BLIS divides by tp anyway). For those three the derivation above reproduces what
// BLIS computes and pins the tp factor in the divisor — which is the regression this file
// exists to catch — but the per-GPU cost it implies is optimistic by 16/8, so it is not a
// physically realizable per-GPU derivation. The llama-2-7b (32 KV heads) and
// llama-3.1-405b (16 KV heads) cases are exact at TP=16 and carry the first-principles
// claim there; TestCalculateKVBlocks_KnownAnswerFixtures_CoverExactTP16 keeps at least one
// such case in the table. The 16/8 optimism itself predates #1846 and is out of its scope.
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
// WHAT THE LAW DOES AND DOES NOT CERTIFY. blockCostBytes is tp-invariant whether or not
// numKVHeads >= tp, because KVBytesPerToken divides by tp unconditionally and this law
// multiplies that back. So on the three 8-KV-head fixtures the tp=16 step is a statement
// about the division's units basis, not about a physically realizable per-GPU cost — at
// 8 KV heads and tp=16 vLLM would replicate heads and the real per-GPU cost is 2x what
// the invariance guard below measures (kv_capacity.go's documented optimistic
// approximation, out of #1846's scope). The llama-2-7b (32 KV heads, exact at every tp
// here) and llama-3.1-405b (16 KV heads, exact at 8 and 16) fixtures carry the law on the
// exact per-GPU basis, so the slope is validated in both regimes rather than only the
// approximate one.
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
// documented at the subtraction site and tracked as #1848.
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
		// 32 KV heads: exact at every tp below, so the law is checked off the approximation.
		{"llama-2-7b/H100", llama2MHAModelConfig(), validHWConfig(), validDenseKVParams(), 5.5, []int{2, 4, 8, 16}},
		// 16 KV heads: exact at both points. Only 8 and 16 — a 764 GiB weight term does not
		// fit in the aggregate budget of 2 or 4 H200s, so CalculateKVBlocks refuses those.
		{"llama-3.1-405b/H200", llama405bModelConfig(), h200HWConfig(), validDenseKVParams(), 5.5, []int{8, 16}},
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
//
// TWO FIXTURES, because perGPUBytesPerToken on the left-hand side is KVBytesPerToken's own
// value and that value is the documented optimistic approximation once numKVHeads < tp:
//
//   - llama-3.1-8b (8 KV heads) is exact up to tp=8 and approximate at tp=16, where the
//     left-hand side under-charges KV by 16/8 — so at that one point the check is
//     correspondingly lenient, and passing it is not a claim about vLLM's real footprint.
//   - llama-2-7b (32 KV heads, MHA) is exact at EVERY tp here, so its tp=16 row is a real
//     physical-realizability statement. It is also small enough to fit on one 80 GiB H100,
//     which is what lets the same TP=1 anchoring work for it.
//
// Both are anchored the same way and neither anchor is derived from a TP>1 result, so the
// check stays non-circular for both.
func TestCalculateKVBlocks_PoolIsPhysicallyRealizable(t *testing.T) {
	fixtures := []struct {
		name   string
		mc     sim.ModelConfig
		params latency.KVCapacityParams
		// exactAtEveryTP records whether numKVHeads >= 16, i.e. whether the tp=16 row is on
		// the exact per-GPU basis or inside the head-replication approximation.
		exactAtEveryTP bool
	}{
		{"llama-3.1-8b/H100", validDenseModelConfig(), validDenseKVParams(), false},
		{"llama-2-7b/H100", llama2MHAModelConfig(), validDenseKVParams(), true},
	}

	const blockSize = int64(16)
	const util = 0.9
	hc := validHWConfig()
	perGPUBudgetGiB := hc.MemoryGiB * util

	for _, f := range fixtures {
		t.Run(f.name, func(t *testing.T) {
			// Recover the tp-independent weights+activation total from the single-GPU budget:
			//   blocks(1) x blockCost(1) = (gpu_mem x util - (W+A) - 0.15) x 2^30
			blocks1, err := latency.CalculateKVBlocks(f.mc, hc, 1, 1, blockSize, util, f.params)
			if err != nil {
				t.Fatalf("TP=1: %v", err)
			}
			kvGiB1 := float64(blocks1*blockBudgetCostFor(t, f.mc, 1, blockSize)) / bytesPerGiB
			weightsPlusActivationGiB := perGPUBudgetGiB - nonTorchPerGPUTP1GiB - kvGiB1
			if weightsPlusActivationGiB <= 0 {
				t.Fatalf("recovered weights+activation total is %.2f GiB; fixture cannot anchor the check",
					weightsPlusActivationGiB)
			}

			for _, tp := range []int{1, 2, 4, 8, 16} {
				blocks, err := latency.CalculateKVBlocks(f.mc, hc, tp, 1, blockSize, util, f.params)
				if err != nil {
					t.Fatalf("TP=%d: %v", tp, err)
				}
				nonTorch := nonTorchPerGPUMultiGiB
				if tp == 1 {
					nonTorch = nonTorchPerGPUTP1GiB
				}
				kvPerGPUGiB := float64(blocks) * float64(blockSize) * perGPUKVBytesPerToken(t, f.mc, tp) / bytesPerGiB
				usedPerGPUGiB := kvPerGPUGiB + weightsPlusActivationGiB/float64(tp) + nonTorch

				// Tolerance of one block's per-GPU bytes absorbs the two truncations.
				oneBlockGiB := float64(blockSize) * perGPUKVBytesPerToken(t, f.mc, tp) / bytesPerGiB
				if usedPerGPUGiB > perGPUBudgetGiB+oneBlockGiB {
					t.Errorf("TP=%d: auto-calculated pool of %d blocks needs %.2f GiB per GPU "+
						"(%.2f KV + %.2f weights+activation share + %.2f non-torch) but only %.2f GiB is available "+
						"per GPU — the pool is not physically realizable",
						tp, blocks, usedPerGPUGiB, kvPerGPUGiB, weightsPlusActivationGiB/float64(tp), nonTorch,
						perGPUBudgetGiB)
				}
			}

			// Guard the fixture property the leniency note above depends on, so a later
			// head-count edit cannot silently turn the exact tp=16 row into an approximate one.
			kvHeads := f.mc.NumKVHeads
			if kvHeads == 0 {
				kvHeads = f.mc.NumHeads
			}
			if exact := kvHeads >= 16 && kvHeads%16 == 0; exact != f.exactAtEveryTP {
				t.Errorf("exactAtEveryTP = %v but %d KV heads vs tp=16 says %v",
					f.exactAtEveryTP, kvHeads, exact)
			}
		})
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

// --- The measured-pool comparison (#1846 acceptance criterion 2, last sentence) ---

// The granite-5.0-230b-sft / TP8 / fp8-KV deployment #1846 was reported from, as four
// measured numbers. The two pool sizes come from OUTSIDE BLIS — each engine's own startup
// "GPU KV cache size" divided by the block size, recorded in the curvebender-tools
// granite-230b-h200 and granite-230b-h100 serving-capacity reports. The two pre-fix counts
// are what BLIS's own auto-calc printed for those same two deployments before this fix.
const (
	graniteMeasuredPoolH200   = 480473  // 7,687,568 tokens / 16
	graniteMeasuredPoolH100   = 89552   // 1,432,832 tokens / 16
	graniteReportedPreFixH200 = 4172133 // BLIS auto-calc, H200, pre-#1846 (8.7x measured)
	graniteReportedPreFixH100 = 973976  // BLIS auto-calc, H100, pre-#1846 (10.9x measured)
)

// granite230bMoEConfig is a MEMORY-EQUIVALENT STAND-IN for that deployment, not a copy of
// its HuggingFace config: granite-5.0-230b-sft has no entry in blis-catalog at the pinned
// 0.1.1 tag, so there is no authentic config.json to build a fixture from.
//
// A stand-in is sound here because CalculateKVBlocks reads the model through exactly two
// scalars — the per-GPU KV bytes per token, and the total weight bytes — and both are
// pinned by the two pre-fix counts above rather than guessed:
//
//	perBlockBytes = (1015.2 - 576.0) GiB / (4,172,133 - 973,976) blocks = 147,456 B/GPU
//	  (the 12.8 GiB of MoE activation + non-torch overhead and the weight total cancel
//	   between the two GPUs, leaving only the memory difference and the per-GPU block cost
//	   that the pre-fix division used as its divisor)
//	weights       = 576.0 - 12.8 - 973,976 x 147,456 B = 429.44 GiB  (~230.6B params, bf16)
//
// The shape below realizes those two scalars: 36 x 2 x 128 headDim x 8 KV heads x 1 B
// (fp8 KV) = 73,728 B/token over the group = 9,216 B/token/GPU at TP=8, so one 16-token
// block costs exactly 147,456 B/GPU; and MoEExpertFFNDim is SOLVED (not observed) to put
// the weight total as close to 429.44 GiB as an integer FFN dim allows — 1521 lands on
// 429.39 GiB, 0.05 GiB low, which is the whole reason the fit below is a stated tolerance
// rather than an equality. Everything else is plausible GraniteMoE-shaped filler that the
// block count cannot see. TestCalculateKVBlocks_MeasuredPool_Granite230B checks the
// equivalence rather than asserting it, by reconstructing both pre-fix counts.
func granite230bMoEConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:       36,
		HiddenDim:       6144,
		NumHeads:        48,
		NumKVHeads:      8,
		VocabSize:       100352,
		BytesPerParam:   2, // bf16 weights
		KVBytesPerParam: 1, // --kv-cache-dtype fp8
		IntermediateDim: 6144,
		// 224 routed experts (the reported GraniteMoE expert count). The per-expert FFN
		// dim is the solved knob described above; at 3 x hidden x dim x 224 x 36 layers
		// each unit of it moves the weight total by ~0.28 GiB, so 1521 is the closest an
		// integer gets to the 429.44 GiB target.
		NumLocalExperts:  224,
		NumExpertsPerTok: 8,
		MoEExpertFFNDim:  1521,
	}
}

func granite230bKVParams() latency.KVCapacityParams {
	return latency.NewKVCapacityParams(true, 224, false, "silu", 1521, 0)
}

// TestCalculateKVBlocks_MeasuredPool_Granite230B compares the auto-calculated pool
// against the two REAL measured pools of the deployment #1846 was reported from, within a
// stated tolerance — the assertion acceptance criterion 2's last sentence asks for, and
// the one that ties this whole file to something outside BLIS.
//
// Each case asserts two independent things:
//
//  1. THE MEASURED-POOL TOLERANCE. The corrected count must sit inside a band around the
//     engine's own pool. The band is not symmetric and not the same on both GPUs, because
//     two residuals #1846 explicitly scopes out push the estimate high:
//     - activation is charged once per replica rather than per GPU, under-subtracting
//     (tp-1) x 8 GiB here (#1848); and
//     - the fixed overhead constants are a larger fraction of the budget when weights
//     nearly fill the card, which the issue's notes call out as the reason the over-count
//     was 8.7x on a 141 GiB H200 but 10.9x on an 80 GiB H100 ("out of scope for this
//     issue").
//     So the ceiling is 1.10x on H200 (the issue's own ~1.09x prediction) and 1.40x on the
//     cramped H100. The floor is 0.90x on both: nothing in the fix may push the pool BELOW
//     a measured one, and in particular a correction applied twice would land at ~0.14x.
//
//  2. FIXTURE FIDELITY. Undoing the fix (got x tp, which reconstructs the pre-#1846
//     numerator/per-GPU-cost division to within tp blocks) must land on the pre-fix count
//     BLIS actually printed for that deployment. This is what makes the stand-in config
//     admissible: it is not "a 230B model someone made up", it is a config pinned by two
//     independent real measurements, either of which it would miss if the shape were wrong.
//
// Pre-fix both parts fail on both GPUs: the pool comes out at 8.68x and 10.88x measured,
// reproducing the 8.7x/10.9x the issue reported.
func TestCalculateKVBlocks_MeasuredPool_Granite230B(t *testing.T) {
	cases := []struct {
		name string
		hc   sim.HardwareCalib
		// measured is the engine's reported pool; preFix is what BLIS printed before
		// the fix; maxRatio/minRatio bound got/measured (see the doc comment).
		measured, preFix int64
		minRatio         float64
		maxRatio         float64
	}{
		{"H200/tp8/fp8-kv", h200HWConfig(), graniteMeasuredPoolH200, graniteReportedPreFixH200, 0.90, 1.10},
		{"H100/tp8/fp8-kv", validHWConfig(), graniteMeasuredPoolH100, graniteReportedPreFixH100, 0.90, 1.40},
	}

	const tp = 8
	// The two anchors are reproduced to within 0.05%, which is four orders of magnitude
	// tighter than the ~8-11x error being guarded against, and is the residual of the
	// solved expert dim (~0.28 GiB of weights per unit) rather than slack in the law.
	const anchorTolerance = 0.0005

	mc, params := granite230bMoEConfig(), granite230bKVParams()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := latency.CalculateKVBlocks(mc, tc.hc, tp, 1, 16, 0.9, params)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// 1. The measured-pool tolerance.
			ratio := float64(got) / float64(tc.measured)
			if ratio < tc.minRatio || ratio > tc.maxRatio {
				t.Errorf("auto-calculated pool = %d blocks, %.3fx the engine's measured pool of %d "+
					"(allowed %.2fx-%.2fx on a %.0f GiB GPU at TP=%d); pre-#1846 this was %d blocks = %.1fx",
					got, ratio, tc.measured, tc.minRatio, tc.maxRatio, tc.hc.MemoryGiB, tp,
					tc.preFix, float64(tc.preFix)/float64(tc.measured))
			}

			// 2. Fixture fidelity against the measured pre-fix count. Undoing the fix
			// (x tp) must land back on what BLIS printed for this deployment. A mismatch
			// means one of two things, and both invalidate the comparison above: the
			// stand-in shape is not memory-equivalent to the real deployment, or the block
			// count is no longer on the aggregate basis that makes x tp the inverse.
			reconstructedPreFix := got * tp
			if delta := math.Abs(float64(reconstructedPreFix-tc.preFix)) / float64(tc.preFix); delta > anchorTolerance {
				t.Errorf("undoing the fix gives %d blocks, but BLIS reported %d pre-#1846 for this "+
					"deployment (off by %.4f%%, tolerance %.2f%%): either this stand-in config is not "+
					"memory-equivalent to it, or the count is not on the aggregate-over-%d-GPUs basis",
					reconstructedPreFix, tc.preFix, delta*100, anchorTolerance*100, tp)
			}
		})
	}
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
