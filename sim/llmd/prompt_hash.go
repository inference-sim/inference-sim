// Moved from blis-vllm/compat/prefix/hash.go at 635b7bf (HashBlock), byte for byte: the router's
// prompt-chain hash, which must key a block exactly as the engines' prefix caches do, so a KV
// event's block hash names the chain the router was shown (canonicalBlock). It is pinned to the
// engine's by TestHashBlockIsTheEnginesPrefixHash (phase 2c).

package llmd

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"

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

// identityRoot is the start of the prefix chain of blocks an engine keys with an adapter and extra
// keys besides their tokens: "" for none, which leaves the chain as it was for a base-model request.
// adapter is the stored block's LoRAName (the request's adapter), keys its ExtraKeys. The adapter
// enters the root once, whichever way the engine writes it into the keys (vLLM v0.32.0's
// ("lora", name, path), or the bare name), so the router-side prompt, which knows only the adapter,
// and the engine's events name one chain; the rest of the keys (a cache salt, multimodal
// identifiers) enter as they are. Equal identities give equal roots and different ones different
// roots, so an adapter's blocks and the base model's, or two salts', never share a chain.
func identityRoot(adapter string, keys []any) string {
	var rest []string
	for _, k := range keys {
		switch v := k.(type) {
		case string:
			if v != adapter || adapter == "" {
				rest = append(rest, "s:"+v)
			}
		case []string:
			if len(v) == 0 || v[0] != "lora" {
				rest = append(rest, "t:"+strings.Join(v, "\x1f"))
			}
		case []any:
			if len(v) > 0 && v[0] == "lora" {
				continue
			}
			parts := make([]string, len(v))
			for i, e := range v {
				parts[i] = fmt.Sprint(e)
			}
			rest = append(rest, "t:"+strings.Join(parts, "\x1f"))
		default:
			rest = append(rest, fmt.Sprintf("v:%v", v))
		}
	}
	if adapter == "" && len(rest) == 0 {
		return ""
	}
	return "extra\x1e" + adapter + "\x1e" + strings.Join(rest, "\x1e")
}
