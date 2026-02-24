// Package hash provides SHA256 hashing utilities for KV cache prefix matching
// and routing prefix affinity. These functions are shared between sim/ (routing)
// and sim/kv/ (cache) to ensure hash consistency (BC-3).
package hash

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
)

// HashTokens computes a SHA256 hash of a token sequence.
// Format: "token1|token2|token3" (pipe delimiter between tokens, not trailing).
// Used for KV cache prefix matching and routing prefix affinity.
func HashTokens(tokens []int) string {
	h := sha256.New()

	// string version of token ids to be joined
	var tokenStrings strings.Builder

	for i, token := range tokens {
		if i > 0 {
			// Add a | delimiter before all tokens except the first
			tokenStrings.WriteString("|")
		}
		tokenStrings.WriteString(strconv.Itoa(token))
	}

	h.Write([]byte(tokenStrings.String()))
	return hex.EncodeToString(h.Sum(nil))
}

// HashBlock computes a SHA256 hash of a token block chained with the previous block's hash.
// Format: prevHash bytes, then for each token: "tokenN" + "|" (pipe AFTER each token).
// This creates hierarchical block hashes for prefix caching.
func HashBlock(prevHash string, tokens []int) string {
	h := sha256.New()
	h.Write([]byte(prevHash))
	for _, t := range tokens {
		h.Write([]byte(strconv.Itoa(t)))
		h.Write([]byte("|"))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ComputeBlockHashes returns hierarchical block hashes for a token sequence.
// Each hash chains with the previous block's hash, enabling prefix matching.
// Tokens that don't fill a complete block are ignored.
func ComputeBlockHashes(blockSize int, tokens []int) []string {
	numBlocks := len(tokens) / blockSize
	if numBlocks == 0 {
		return nil
	}
	hashes := make([]string, numBlocks)
	prevHash := ""
	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		hashes[i] = HashBlock(prevHash, tokens[start:end])
		prevHash = hashes[i]
	}
	return hashes
}
