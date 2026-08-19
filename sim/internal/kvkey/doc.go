// Package kvkey is the single gated home for KV cache block keying: deriving
// hierarchical content keys at a chosen stride, and interning those keys to
// compact integer handles. It fills hole H4 (kv_block_key) of the multi-tier
// KV-offload epic (#1585).
//
// # Two keyspaces, one hash source
//
// BLIS has exactly one hash function (sim/internal/hash). Everything that keys
// KV content — the GPU tier, the CPU tier, every secondary/offload tier, and
// the router's prefix-affinity index — must derive identical bytes for
// identical tokens (BC-K1). This package delegates ALL hashing to
// sim/internal/hash and never imports crypto/hash/fnv directly; a static test
// (static_test.go) enforces the same rule for sim/kv, promoting the "one hash
// source" convention into a gated contract.
//
// There are two derivation strides:
//
//   - Block stride — the granularity at which GPU/CPU tiers and the router index
//     key blocks. Produced by hash.ComputeBlockHashes / hash.HashBlock.
//   - Chunk stride — the granularity at which transfer/offload tiers key chunks
//     (a group of blocks_per_chunk blocks). Produced by DeriveChunkKeys with
//     tokensPerChunk = tokensPerBlock * blocks_per_chunk. This mirrors vLLM,
//     which derives offload keys per chunk, NOT per block (BC-K4).
//
// # Contracts
//
//   - BC-K1: one hash source; identical bytes for identical tokens. At
//     tokensPerChunk == blockSize and prevKey == "", DeriveChunkKeys is
//     byte-identical to hash.ComputeBlockHashes.
//   - BC-K2: chunk keys chain hierarchically — a match at chunk i implies a
//     match at every chunk j < i.
//   - BC-K3: interning is injective (distinct keys -> distinct ids, collisions
//     impossible), idempotent, dense, and deterministic given a fixed call order.
//   - BC-K4: offload keys are derived at chunk stride, not block stride.
//
// # Chunk keyspace is disjoint from the block keyspace
//
// DeriveChunkKeys single-shot chain-hashes each whole chunk, so a chunk key
// matches no block hash by value. This is functionally sufficient for the
// offload model (a stable per-chunk content key) and loses no capability: a
// consumer with the request tokens derives both keyspaces from this one hash
// source. vLLM instead anchors each chunk key to the chunk's trailing
// block-hash (a subsequence of the block chain); the frozen surface here
// (chunk stride only, no block size) specifies chunk-granular hashing.
//
// The Interner and DeriveChunkKeys are not safe for concurrent use — the
// simulator runs one goroutine per run.
package kvkey
