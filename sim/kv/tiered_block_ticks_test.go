package kv

import "testing"

// A per-block reload charge given whole is what reloads cost: n reloaded blocks accumulate
// exactly n times the charge, whatever the block size; a charge below one tick is refused.
func TestTieredKVCache_AGivenPerBlockChargeIsWhatReloadsCost(t *testing.T) {
	for _, ticks := range []int64{1, 2, 37, 1_000_000} {
		for _, bs := range []int64{16, 64} {
			c := NewTieredKVCacheWithBlockTicks(NewKVCacheState(100, bs), 10, 0, ticks)
			for n := int64(1); n <= 5; n++ {
				c.accumulateTransferLatency()
			}
			if got := c.ConsumePendingTransferLatency(); got != 5*ticks {
				t.Errorf("block size %d: 5 reloads cost %d ticks, want 5 x %d", bs, got, ticks)
			}
		}
	}
	defer func() {
		if recover() == nil {
			t.Error("a zero-tick reload charge was accepted")
		}
	}()
	NewTieredKVCacheWithBlockTicks(NewKVCacheState(100, 16), 10, 0, 0)
}
