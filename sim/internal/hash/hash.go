// Package hash provides SHA256 hashing utilities for KV cache prefix matching
// and routing prefix affinity. These functions are shared between sim/ (routing)
// and sim/kv/ (cache) to ensure hash consistency (BC-3).
package hash

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
)

// HashBlock computes a SHA256 hash of a token block chained with the previous block's hash.
// Format: prevHash bytes, then for each token: "tokenN" + "|" (pipe AFTER each token).
// This creates hierarchical block hashes for prefix caching.
// Also inlined in ComputeBlockHashes for hasher reuse.
// TestComputeBlockHashes_MatchesManualChaining guards consistency between the two paths.
func HashBlock(prevHash string, tokens []int) string {
	h := sha256.New()
	h.Write([]byte(prevHash))
	var buf [20]byte // stack buffer: max int64 (19 digits) + pipe
	for _, t := range tokens {
		b := strconv.AppendInt(buf[:0], int64(t), 10)
		b = append(b, '|')
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ComputeBlockHashes returns hierarchical block hashes for a token sequence.
// Each hash chains with the previous block's hash, enabling prefix matching.
// Tokens that don't fill a complete block are ignored.
// Produces the same output as calling HashBlock sequentially, but reuses a
// single SHA256 hasher instance across blocks to reduce allocations.
// Output equivalence is enforced by TestComputeBlockHashes_MatchesManualChaining.
func ComputeBlockHashes(blockSize int, tokens []int) []string {
	if blockSize <= 0 {
		panic(fmt.Sprintf("ComputeBlockHashes: blockSize must be > 0, got %d", blockSize))
	}
	numBlocks := len(tokens) / blockSize
	if numBlocks == 0 {
		return nil
	}
	hashes := make([]string, numBlocks)
	h := sha256.New()
	prevHash := ""
	var buf [20]byte // stack buffer: reused across all tokens in all blocks
	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		h.Reset()
		// Inlines HashBlock logic for hasher reuse — keep in sync with HashBlock above.
		h.Write([]byte(prevHash))
		for _, t := range tokens[start:end] {
			b := strconv.AppendInt(buf[:0], int64(t), 10)
			b = append(b, '|')
			h.Write(b)
		}
		hashes[i] = hex.EncodeToString(h.Sum(nil))
		prevHash = hashes[i]
	}
	return hashes
}
