package kv

import (
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
)

// BenchmarkPollDeferred documents property P: the per-step re-poll cost is
// O(deferred), NOT O(waitq). PollDeferred only ever touches OffloadCache.deferred —
// it has no reference to the wait queue — so the wait-queue length is irrelevant by
// construction; this benchmark guards against a regression that reintroduces a
// full-queue rescan, and confirms cost scales with the number of live deferrals.
func BenchmarkPollDeferred(b *testing.B) {
	for _, n := range []int{10, 100, 1000} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			gpu := NewKVCacheState(64, 2)
			// CPU large enough to hold one pending (-1) block per deferral.
			oc := NewOffloadCache(gpu, enabledOffloadCfg(int64(n+8)*4096, 4096, 1))
			// n requests each parked in deferPromoting, awaiting a HIT_PENDING block —
			// awaitedState stays cpuHitPending each poll, so all n remain deferred.
			for i := 0; i < n; i++ {
				key := kvkey.BlockKey("defer-bench-key-" + strconv.Itoa(i))
				oc.cpu.prepareStore([]kvkey.BlockKey{key}) // creates a -1 (HIT_PENDING) block
				oc.deferred[strconv.Itoa(i)] = &deferralState{
					phase: deferPromoting,
					keys:  []kvkey.BlockKey{key},
					tier:  0,
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				still := oc.PollDeferred(int64(i + 1))
				if len(still) != n {
					b.Fatalf("all %d deferrals must remain pending, got %d", n, len(still))
				}
			}
		})
	}
}

// Sanity: the benchmark's setup keeps every deferral pending (so the O(deferred)
// claim is measured over a stable set, not a shrinking one).
func TestPollDeferred_StablePendingSet(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(64*4096, 4096, 1))
	var reqIDs []string
	for i := 0; i < 5; i++ {
		key := kvkey.BlockKey("k" + strconv.Itoa(i))
		oc.cpu.prepareStore([]kvkey.BlockKey{key})
		id := "r" + strconv.Itoa(i)
		reqIDs = append(reqIDs, id)
		oc.deferred[id] = &deferralState{phase: deferPromoting, keys: []kvkey.BlockKey{key}, tier: 0}
	}
	for step := int64(1); step <= 3; step++ {
		if got := len(oc.PollDeferred(step * 1000)); got != len(reqIDs) {
			t.Fatalf("pending set must be stable while blocks stay HIT_PENDING, step %d got %d want %d", step, got, len(reqIDs))
		}
	}
}
