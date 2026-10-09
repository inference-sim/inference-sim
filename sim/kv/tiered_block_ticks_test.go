package kv

import "testing"

// A per-block reload charge given whole is the charge: the legacy tier adds exactly it per
// reloaded block, whatever the block size.
func TestTieredKVCache_AGivenPerBlockChargeIsTheCharge(t *testing.T) {
	for _, ticks := range []int64{1, 2, 37, 1_000_000} {
		for _, bs := range []int64{16, 64} {
			c := NewTieredKVCacheWithBlockTicks(NewKVCacheState(100, bs), 10, 0, ticks)
			if c.transferLatencyPerBlock != ticks {
				t.Errorf("block size %d: charge %d per block, given %d", bs, c.transferLatencyPerBlock, ticks)
			}
		}
	}
}
