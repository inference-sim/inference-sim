package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// The offload tiers hold vLLM's offload CHUNKS, blocks_per_chunk GPU blocks each, not GPU
// blocks (vLLM v0.31.0, distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py):
//
//   - A chunk's key is the hash of its LAST block (update_offload_keys, :418-436:
//     block_hashes[hashes_per_chunk*(i+1)-1]), so the chunk keyspace is a subset of the
//     GPU block hashes and blocks_per_chunk 1 is block-granular offload exactly (BC-K1).
//   - Lookup starts at the chunk holding the GPU-cached boundary
//     (start_chunk_idx = num_computed_tokens // tokens_per_chunk, :841) and counts whole
//     chunks; only complete chunks have keys (:418-436), so a partial trailing chunk is
//     never offloaded or loaded.
//   - A load shorter than one chunk is no load (:818-821, :888-891: "We can only load less
//     than a chunk, so skip").
//   - A store offers whole chunks: storable_chunks = offloadable tokens // tokens_per_chunk,
//     capped by allocated blocks // blocks_per_chunk (:448-479).
//   - The CPU tier holds cpu_bytes_to_use // (kv_bytes_per_block * blocks_per_chunk) chunks
//     (kv_offload/cpu/spec.py:108-121), each one LRU entry, and every transfer moves whole
//     chunks.

// offloadTail is a prompt's uncached tail cut into offload chunks: the chunks
// [firstChunk, endChunk) covering blocks [startBlock, endChunk*bpc).
type offloadTail struct {
	startBlock           int64
	bpc                  int64
	firstChunk, endChunk int64
	blockKeys            []kvkey.BlockKey // blockKeys[i-startBlock] is block i's hash
}

// tailOf keys the tail of tokens past startBlock, seeded by prevHash (the hash of block
// startBlock-1, "" at 0), so the cached prefix is never re-hashed. ok is false when no
// complete chunk reaches past startBlock.
func (o *OffloadCache) tailOf(tokens []sim.TokenID, startBlock int64, prevHash string) (offloadTail, bool) {
	bs := o.gpu.BlockSize()
	t := offloadTail{startBlock: startBlock, bpc: o.blocksPerChunk}
	t.firstChunk = startBlock / t.bpc
	t.endChunk = util.Len64(tokens) / bs / t.bpc
	if t.firstChunk >= t.endChunk {
		return t, false
	}
	t.blockKeys = kvkey.DeriveChunkKeys(kvkey.BlockKey(prevHash), tokens[startBlock*bs:t.endChunk*t.bpc*bs], int(bs))
	return t, true
}

// key is chunk c's offload key: its last block's hash.
func (t offloadTail) key(c int64) kvkey.BlockKey { return t.blockKeys[(c+1)*t.bpc-1-t.startBlock] }

// blocks is the range [lo, hi) of chunk c's blocks at or past startBlock.
func (t offloadTail) blocks(c int64) (lo, hi int64) { return max(c*t.bpc, t.startBlock), (c + 1) * t.bpc }

func (t offloadTail) blockKey(i int64) kvkey.BlockKey { return t.blockKeys[i-t.startBlock] }

// onGPU reports whether every block of chunk c past startBlock is GPU-resident.
func (o *OffloadCache) onGPU(t offloadTail, c int64) bool {
	lo, hi := t.blocks(c)
	for i := lo; i < hi; i++ {
		if _, ok := o.gpu.HashToBlock[string(t.blockKey(i))]; !ok {
			return false
		}
	}
	return true
}

// cpuRun counts the leading chunks of t available this step -- each GPU-resident or
// CPU-readable -- and returns the CPU state of the chunk that ended the run (cpuHit when it
// reached the end). A pure query.
func (o *OffloadCache) cpuRun(t offloadTail) (chunks int64, stop cpuLookupResult) {
	for c := t.firstChunk; c < t.endChunk; c++ {
		if o.onGPU(t, c) {
			chunks++
			continue
		}
		if s := o.cpu.lookup(t.key(c)); s != cpuHit {
			return chunks, s
		}
		chunks++
	}
	return chunks, cpuHit
}

// loadableBlocks is the blocks past startBlock a run of chunks supplies, 0 when that is less
// than one chunk (vLLM loads nothing then).
func (t offloadTail) loadableBlocks(chunks int64) int64 {
	n := (t.firstChunk+chunks)*t.bpc - t.startBlock
	if n < t.bpc {
		return 0
	}
	return n
}

// chunkBytes is the bytes one offload transfer of a chunk moves.
func (o *OffloadCache) chunkBytes() int64 { return o.perBlockBytes * o.blocksPerChunk }
