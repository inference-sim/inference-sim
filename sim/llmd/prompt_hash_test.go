package llmd

import (
	"testing"

	"github.com/inference-sim/blis-vllm/compat/prefix"
	"github.com/inference-sim/blis-vllm/compat/tokenid"

	"github.com/inference-sim/inference-sim/sim"
)

// The router's prompt-chain hash is the engines' prefix-cache block hash, chain by chain: a KV
// event an engine publishes names the chain the router hashed (blis-vllm compat/prefix).
func TestHashBlockIsTheEnginesPrefixHash(t *testing.T) {
	blocks := [][]int{{}, {0}, {1, 2, 3}, {127999, 5, 0, 42}, {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}}
	ours, theirs := "", ""
	for i, b := range blocks {
		host, eng := make([]sim.TokenID, len(b)), make([]tokenid.TokenID, len(b))
		for j, v := range b {
			host[j], eng[j] = sim.TokenID(v), tokenid.TokenID(v)
		}
		ours, theirs = hashBlock(ours, host), prefix.HashBlock(theirs, eng)
		if ours != theirs {
			t.Fatalf("block %d: the router hashes %s, the engine %s", i, ours, theirs)
		}
	}
}
