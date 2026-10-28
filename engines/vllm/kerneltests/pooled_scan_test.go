// Moved from blis-latency-kernel/pooled_scan_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"
)

const pooledFixture = "aisimulate/deepseek-v4-pro-b300-fp4-vllm-tp4.yaml"

// pooledAt builds the pooled-scan fixture with every block-index scorer pooling ratio
// tokens into one state: the scorer's compress_ratio, the index cache it states, and the
// store that writes that cache, all moved together, as a graph pooling at that ratio would
// state them. Zero leaves the cache's tokens_per_entry and the store's per_tokens unset,
// which mean one; the scorer itself must state a bound, so it states one.
func pooledAt(t *testing.T, ratio int) *Kernel {
	t.Helper()
	in := fixtureInputs(t, pooledFixture)
	scorers := 0
	for i := range in.Model.LayerKinds {
		for j := range in.Model.LayerKinds[i].Nodes {
			n := &in.Model.LayerKinds[i].Nodes[j]
			if n.Role == "block_index_scores" {
				// A sparse scorer must state some bound, so "absent" is said on the cache.
				n.CompressRatio = max(ratio, 1)
				scorers++
			}
			for c := range n.Caches {
				if n.Caches[c].Kind == model.CacheIndex {
					n.Caches[c].TokensPerEntry = ratio
				}
			}
			if n.WritesCache != "" && cacheKindIn(in.Model.LayerKinds[i], n.WritesCache) ==
				model.CacheIndex {
				n.PerTokens = ratio
			}
		}
	}
	if scorers == 0 {
		t.Fatalf("the fixture's graph launches no block-index scorer, so the pooled-scan " +
			"term cannot be exercised; pick a fixture whose model has one")
	}
	return build(t, in)
}

// cacheKindIn is the kind of the cache a layer kind names name.
func cacheKindIn(lk model.LayerKind, name string) model.CacheKind {
	for _, n := range lk.Nodes {
		for _, c := range n.Caches {
			if c.Name == name {
				return c.Kind
			}
		}
	}
	return ""
}

// A pooled indexer caches one state per index_kpool tokens and scores over those states,
// so the ratio bounds how many candidates its scan covers. The bound was dropped in
// planning, which priced every pooled scorer over the full token context.
//
// Asserted on STEP TIME, not on the plan: a plan that merely carries the ratio while the
// step ignores it is the failure mode a plan-level check cannot see -- the same trap
// TestSlidingWindowAndFullAttentionPriceDifferently documents for the per-kind rate. And
// varied on the GRAPH, every field that states the ratio together: a scorer whose graph
// states its index cache reads that cache's entries, so a ratio moved on the node alone
// would describe a graph that contradicts itself.
//
// No committed catalog graph declares a pooled SCORER yet. deepseek-v4-pro carries
// compress_ratio on its PRIMARY attention (the compressed-tail read that
// selectedKVTokens prices), not on its scorer, and GLM-5.3-Flash is the first with a
// pooled one. So these tests vary the bound on a real fixture's own scorer rather than
// skipping: what is under test is the kernel's pricing, and a skip would leave the term
// unexercised in CI -- the silent-skip hole 7c556d7 closed elsewhere in this suite.
func TestAPooledScorersRatioChangesTheStepTime(t *testing.T) {
	b := longDecodeBatch()

	// Unpooled: every token is its own candidate, which is what the kernel charged
	// before this term existed.
	unpooled := pooledAt(t, 1).StepTime(b).Overlap

	// Pooled 4:1, GLM-5.3-Flash's index_kpool. Everything else held fixed.
	pooled := pooledAt(t, 4).StepTime(b).Overlap

	if pooled == unpooled {
		t.Fatalf("step time is %v whether or not the scorer is pooled; the plan carries "+
			"the ratio but the step does not price with it", pooled)
	}
	if pooled >= unpooled {
		t.Fatalf("the pooled step (%v) is not below the unpooled one (%v); pooling "+
			"divides the candidate count, so a pooled scan must read LESS",
			pooled, unpooled)
	}
}

// METAMORPHIC: a larger pool must never cost more. The scan covers ceil(context/ratio)
// states, so priced time is monotonically non-increasing in the ratio. A kernel that
// multiplied instead of divided, or that clamped, passes a single-value check and fails
// this one.
func TestPooledScanTimeIsMonotonicInThePoolSize(t *testing.T) {
	b := longDecodeBatch()

	var prev float64
	seen := map[float64]bool{}
	for _, ratio := range []int{1, 2, 4, 8, 16, 64} {
		got := pooledAt(t, ratio).StepTime(b).Overlap.Seconds()
		if prev != 0 && got > prev {
			t.Fatalf("ratio %d priced %.9fs, above the previous ratio's %.9fs; a larger "+
				"pool holds FEWER states per span, so the scan cannot cost more",
				ratio, got, prev)
		}
		prev = got
		seen[got] = true
	}
	// STRICT sensitivity: each DOUBLING of the pool must move the price, not merely
	// fail to raise it. Monotonicity alone cannot catch a kernel that divides by a
	// CONSTANT -- that is trivially non-increasing, and because the `> 1` guard still
	// makes ratio 1 differ, it even produces two distinct values. Requiring a distinct
	// time per ratio is what closes that: a constant divisor collapses 2, 4, 8, 16 and
	// 64 onto one time, and an off-by-one divisor (ratio+1) shifts every value but
	// still yields six distinct ones, so it is caught by the exact-arithmetic test
	// instead.
	if len(seen) != 6 {
		t.Fatalf("six pool sizes from 1 to 64 produced only %d distinct step times; a "+
			"scan that read its ratio gives a different time for each, so the field is "+
			"being ignored or collapsed to a constant", len(seen))
	}
}

// The scan bound must be the ratio EXACTLY, not merely something that decreases with it.
// Asserted against the arithmetic rather than against a recorded number: the scorer term
// is linear in the candidate count, so halving the pool size must halve the scorer's
// contribution to the step. Measured as a difference of differences, which cancels every
// other term in the step and leaves the scan.
func TestPooledScanBoundIsTheRatioExactly(t *testing.T) {
	b := longDecodeBatch()
	at := func(ratio int) float64 { return pooledAt(t, ratio).StepTime(b).Overlap.Seconds() }

	// The scorer contributes S/r at ratio r, so t(r) = base + S/r. Then
	//   t(1) - t(2) = S/2  and  t(2) - t(4) = S/4,
	// whose ratio is exactly 2 for any S and any base. A divisor that is off by one, or
	// a constant, breaks that identity while still being monotone.
	d12 := at(1) - at(2)
	d24 := at(2) - at(4)
	if d24 <= 0 {
		t.Fatalf("the scan did not shrink between ratio 2 and 4 (delta %.9fs)", d24)
	}
	if got := d12 / d24; got < 1.9 || got > 2.1 {
		t.Fatalf("t(1)-t(2) over t(2)-t(4) is %.4f, want 2.0: the scan term is S/ratio, "+
			"so the two deltas are S/2 and S/4. A value away from 2 means the divisor is "+
			"not the ratio itself", got)
	}
}

// An unset pool size and a pool of one mean no pooling, and must price exactly alike: the
// bound is absent either way. This is the no-regression guarantee for every model that declares
// no index_kpool, which is all of them but one.
func TestAnAbsentPoolRatioPricesAsOne(t *testing.T) {
	b := longDecodeBatch()
	zero := pooledAt(t, 0).StepTime(b).Overlap
	one := pooledAt(t, 1).StepTime(b).Overlap

	if zero != one {
		t.Fatalf("an unset ratio priced %v and a ratio of 1 priced %v; both mean one "+
			"state per token, so a model declaring no pooling must be unaffected",
			zero, one)
	}
}

// longDecodeBatch is a decode batch at a long context, because the scan is the
// context-proportional term: at a short context the bound changes little and a test
// would not discriminate.
func longDecodeBatch() kernel.Batch {
	b := kernel.Batch{DecodeThreshold: 8, SMBudget: 132}
	for i := 0; i < 32; i++ {
		b.Reqs = append(b.Reqs, kernel.ReqShape{
			Scheduled: 1, Computed: 65536, PromptLen: 65536})
	}
	return b
}
