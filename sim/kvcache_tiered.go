package sim

import "fmt"

// TieredKVCache will be fully implemented in Task 4. Stub for compilation.
type TieredKVCache struct{ gpu *KVCacheState }

// NewTieredKVCache creates a TieredKVCache. Stub — panics until Task 4.
func NewTieredKVCache(gpu *KVCacheState, cpuBlocks int64, threshold, bandwidth float64, baseLat int64) *TieredKVCache {
	panic(fmt.Sprintf("TieredKVCache not yet implemented (cpuBlocks=%d)", cpuBlocks))
}

func (t *TieredKVCache) AllocateKVBlocks(req *Request, s, e int64, c []int64) bool { panic("stub") }
func (t *TieredKVCache) GetCachedBlocks(tokens []int) []int64                      { panic("stub") }
func (t *TieredKVCache) ReleaseKVBlocks(req *Request)                              { panic("stub") }
func (t *TieredKVCache) BlockSize() int64                                          { return t.gpu.BlockSize() }
func (t *TieredKVCache) UsedBlocks() int64                                         { return t.gpu.UsedBlocks() }
func (t *TieredKVCache) TotalCapacity() int64                                      { return t.gpu.TotalCapacity() }
func (t *TieredKVCache) CacheHitRate() float64                                     { return 0 }
func (t *TieredKVCache) PendingTransferLatency() int64                             { return 0 }
func (t *TieredKVCache) KVThrashingRate() float64                                  { return 0 }
