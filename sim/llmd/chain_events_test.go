package llmd

import (
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// A vLLM v0.32.0 engine keys its blocks by its own 64-bit hashes, in decimal. chainEvents maps
// each to the prefix chain the canonicalizer keyed the same block of the prompt by, so a block the
// engine cached from a prompt names the chain the router was shown (canonicalBlock). Removals
// resolve to the chain of the store they undo, and the mapping goes with the last store.
func TestEngineKeyedEventsNameThePromptsChains(t *testing.T) {
	const bs = 4
	prompt := make([]sim.TokenID, 3*bs)
	for i := range prompt {
		prompt[i] = sim.TokenID(100 + i)
	}
	var want []string
	chain := ""
	for s := 0; s < len(prompt); s += bs {
		chain = hashBlock(chain, prompt[s:s+bs])
		want = append(want, chain)
	}
	key := func(i int) string { return strconv.Itoa(7000 + i) }
	var stores []sim.KVEvent
	for i := 0; i < 3; i++ {
		e := sim.KVEvent{Hash: key(i), Tokens: prompt[i*bs : (i+1)*bs]}
		if i > 0 {
			e.Parent = key(i - 1)
		}
		stores = append(stores, e)
	}
	chains := map[string]*engineChain{}
	got := chainEvents(chains, stores)
	for i, e := range got {
		if e.Hash != want[i] {
			t.Fatalf("block %d names chain %s, want the prompt's %s", i, e.Hash, want[i])
		}
		if i > 0 && e.Parent != want[i-1] {
			t.Fatalf("block %d's parent is %s, want %s", i, e.Parent, want[i-1])
		}
	}
	// The same block stored twice (a recompute), then removed once: it stays mapped until its
	// second removal.
	chainEvents(chains, stores[2:3])
	rm := chainEvents(chains, []sim.KVEvent{{Removed: true, Hash: key(2)}})
	if rm[0].Hash != want[2] || chains[key(2)] == nil {
		t.Fatalf("first removal: %+v, mapped %v", rm[0], chains[key(2)] != nil)
	}
	chainEvents(chains, []sim.KVEvent{{Removed: true, Hash: key(2)}})
	if chains[key(2)] != nil {
		t.Fatal("the block is still mapped after its last store was removed")
	}
	// A compatibility engine's events are BLIS chain hashes already: passed through.
	compat := []sim.KVEvent{{Hash: want[0], Tokens: prompt[:bs]}}
	if out := chainEvents(map[string]*engineChain{}, compat); out[0].Hash != want[0] {
		t.Fatalf("a chain-keyed event was rewritten to %s", out[0].Hash)
	}
}
