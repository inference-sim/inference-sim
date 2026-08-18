package kvkey

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/inference-sim/inference-sim/sim/internal/tokenid"
)

// BlockKey is a content-identity key for a KV cache block or transfer chunk:
// the 64-hex-character SHA256 digest produced by sim/internal/hash, wrapped in a
// defined type so a raw string can't be mixed with a key by accident (the same
// compile-time discipline tokenid.TokenID applies to token IDs). Identical
// tokens always produce identical BlockKeys across every tier and the router
// (BC-K1).
type BlockKey string

// KeyID is a compact, run-local integer handle for a BlockKey, assigned by an
// Interner. IDs are dense (0,1,2,…) and stable for the Interner's lifetime.
// A KeyID is meaningful only relative to the Interner that produced it; it is
// NOT comparable across Interners or across runs.
type KeyID int64

// DeriveChunkKeys returns hierarchical content keys for tokens at the given
// chunk stride. Each chunk key chains the previous chunk's key, so a match at
// chunk i implies a match at every chunk j < i (BC-K2); the chain is seeded with
// prevKey, letting a caller continue past an already-derived prefix. Only
// complete chunks produce a key — a partial trailing chunk is ignored, mirroring
// hash.ComputeBlockHashes at block stride.
//
// This is the CHUNK-stride keyspace (BC-K4): with
// tokensPerChunk = tokensPerBlock * blocksPerChunk it yields one key per group
// of blocksPerChunk blocks — matching vLLM's per-chunk offload keys, NOT one key
// per block. At tokensPerChunk == blockSize and prevKey == "" it is
// byte-identical to hash.ComputeBlockHashes (BC-K1).
//
// Panics if tokensPerChunk <= 0 (a caller bug, mirroring hash.ComputeBlockHashes).
func DeriveChunkKeys(prevKey BlockKey, tokens []tokenid.TokenID, tokensPerChunk int) []BlockKey {
	if tokensPerChunk <= 0 {
		panic(fmt.Sprintf("DeriveChunkKeys: tokensPerChunk must be > 0, got %d", tokensPerChunk))
	}
	numChunks := len(tokens) / tokensPerChunk
	if numChunks == 0 {
		return nil
	}
	keys := make([]BlockKey, numChunks)
	prev := string(prevKey)
	for i := 0; i < numChunks; i++ {
		start := i * tokensPerChunk
		end := start + tokensPerChunk
		h := hash.HashBlock(prev, tokens[start:end])
		keys[i] = BlockKey(h)
		prev = h
	}
	return keys
}

// Interner assigns each distinct BlockKey a dense KeyID (0,1,2,…), stable for
// the Interner's lifetime. Interning is injective (distinct keys -> distinct IDs,
// collisions impossible) and idempotent (the same key always returns the same
// ID). Given a fixed sequence of Intern calls the ID assignment is fully
// deterministic (INV-6). Not safe for concurrent use — the simulator runs one
// goroutine per run.
type Interner struct {
	ids  map[BlockKey]KeyID
	keys []BlockKey // reverse table: keys[id] == the key interned as id
}

// NewInterner returns an empty Interner.
func NewInterner() *Interner {
	return &Interner{ids: make(map[BlockKey]KeyID)}
}

// Intern returns the KeyID for k, assigning a new dense ID on first sight.
func (in *Interner) Intern(k BlockKey) KeyID {
	if id, ok := in.ids[k]; ok {
		return id
	}
	id := KeyID(len(in.keys))
	in.ids[k] = id
	in.keys = append(in.keys, k)
	return id
}

// Key returns the BlockKey an ID maps to and whether that ID has been assigned.
// Use it at boundaries where content identity must cross as a string (KV events,
// traces, the router index).
func (in *Interner) Key(id KeyID) (BlockKey, bool) {
	if id < 0 || int(id) >= len(in.keys) {
		return "", false
	}
	return in.keys[id], true
}

// Len returns the number of distinct keys interned so far.
func (in *Interner) Len() int { return len(in.keys) }
