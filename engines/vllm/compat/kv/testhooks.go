// Original to blis-vllm.

package kv

// Test hooks for inference-sim's simulator-level tests, which drive these stores through a
// whole simulation (event loop and engine together) and check the store's own invariants after
// every event. Nothing outside tests calls them.

// VerifyBlockConservationForTest reports whether the store's GPU pool is conserved: every
// block is free, held, cached or blank exactly once (verifyBlockConservation). nil when it is.
func (kvc *KVCacheState) VerifyBlockConservationForTest() error { return kvc.verifyBlockConservation() }

// GPUTierForTest returns the GPU tier of a store NewKVStore built (the store itself when
// single-tier), nil for any other store.
func GPUTierForTest(s any) *KVCacheState {
	switch v := s.(type) {
	case *KVCacheState:
		return v
	case *TieredKVCache:
		return v.gpu
	case *OffloadCache:
		return v.gpu
	}
	return nil
}
