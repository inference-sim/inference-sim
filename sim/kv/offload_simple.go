package kv

import (
	"github.com/inference-sim/blis-vllm/compat/kvkey"
	sim "github.com/inference-sim/inference-sim/sim/engine"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// vLLM v0.31.0's SimpleCPUOffloadConnector (vllm/v1/simple_kv_offload/manager.py, M below), as
// it differs from the OffloadingConnector the rest of OffloadCache models. Both keep a CPU LRU
// keyed by the GPU block hash and load a CPU-resident prefix asynchronously into blocks taken
// at load start (M:337-505, the same WAITING_FOR_REMOTE_KVS path), so the load side is shared.
// The store side differs:
//
//   - Block granular (M:207-219: a second KVCacheCoordinator over the GPU's own block size).
//   - Eager (the default): every confirmed full block of a scheduled request is stored, decode
//     blocks included -- there is no prompt-only clamp (M:641-803, 817-951).
//   - Lazy (lazy_offload): only blocks near the head of the GPU free queue are stored, the next
//     to be evicted, up to a window of 2 x ceil(max_num_batched_tokens / block_size)
//     (M:312-330, 585-639); a running request's blocks are never stored.
//   - A store is an asynchronous copy: the CPU block is allocated at once, evicting the LRU
//     entry, and is readable only when the copy lands (M:630-635, 770-786, 1008-1165). Copies
//     run one after another on the connector's store stream (worker.py:164-167, 273-295).
//   - Already cached, already in flight, unhashed: skipped (M:953-967). A CPU tier with no
//     unpinned block drops the store (BlockPool.get_num_free_blocks counts cached-unpinned
//     blocks as free).

// simpleStore is one in-flight GPU->CPU copy of a block.
type simpleStore struct {
	key     kvkey.BlockKey
	readyAt int64
}

// mirrorSimple is MirrorToCPU for the Simple connector.
func (o *OffloadCache) mirrorSimple(batch []*sim.Request) {
	o.landSimpleStores()
	if o.lazyOffload {
		o.storeLazy()
		return
	}
	bs := o.gpu.BlockSize()
	for _, req := range batch {
		for _, id := range o.gpu.RequestMap[req.ID] {
			blk := o.gpu.Blocks[id]
			if blk.Hash == "" || util.Len64(blk.Tokens) < bs {
				continue
			}
			o.storeSimple(kvkey.BlockKey(blk.Hash))
		}
	}
}

// storeLazy offers the blocks at the head of the GPU free queue: cached, unreferenced blocks
// a later allocation evicts first.
func (o *OffloadCache) storeLazy() {
	bs := o.gpu.BlockSize()
	n := int64(0)
	for blk := o.gpu.FreeHead; blk != nil && n < o.lazyTargetFree; blk = blk.NextFree {
		n++
		if blk.Hash == "" || util.Len64(blk.Tokens) < bs {
			continue
		}
		o.storeSimple(kvkey.BlockKey(blk.Hash))
	}
}

// storeSimple starts one block's copy unless the CPU tier has it or is copying it.
func (o *OffloadCache) storeSimple(key kvkey.BlockKey) {
	if o.cpu.lookup(key) != cpuMiss {
		o.cpu.touchKey(key)
		return
	}
	if o.cpu.prepareStore([]kvkey.BlockKey{key}) == 0 {
		o.mirrorSkipped++ // no unpinned CPU block: the store is dropped
		return
	}
	ready := o.clock
	if o.cpuStoreTime != nil {
		o.storeBusyUntil = max(o.clock, o.storeBusyUntil) + o.cpuStoreTime(o.perBlockBytes)
		ready = o.storeBusyUntil
	}
	o.simpleStores = append(o.simpleStores, simpleStore{key: key, readyAt: ready})
	o.landSimpleStores()
}

// landSimpleStores makes every copy that has landed by now readable, in submission order
// (one stream: they land in order).
func (o *OffloadCache) landSimpleStores() {
	i := 0
	for ; i < len(o.simpleStores) && o.simpleStores[i].readyAt <= o.clock; i++ {
		o.cpu.completeStore(o.simpleStores[i].key, true)
	}
	o.simpleStores = o.simpleStores[i:]
}
