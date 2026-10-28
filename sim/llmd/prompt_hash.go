// Moved from blis-vllm/compat/prefix/hash.go at 635b7bf (HashBlock), byte for byte: the router's
// prompt-chain hash, which must key a block exactly as the engines' prefix caches do, so a KV
// event's block hash names the chain the router was shown (canonicalBlock). It is pinned to the
// engine's by TestHashBlockIsTheEnginesPrefixHash (phase 2c).

package llmd

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"

	"github.com/inference-sim/inference-sim/sim"
)

// hashBlock computes a SHA256 hash of a token block chained with the previous block's hash.
// Format: prevHash bytes, then for each token: "tokenN" + "|" (pipe AFTER each token).
func hashBlock(prevHash string, tokens []sim.TokenID) string {
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
