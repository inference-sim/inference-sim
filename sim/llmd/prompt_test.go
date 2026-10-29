package llmd

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestPromptCanon_EqualExactlyWhereBlocksAreShared: canonical prompts agree on every whole block
// two prompts share from the start, and nowhere after the first differing block.
func TestPromptCanon_EqualExactlyWhereBlocksAreShared(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	prefix := sim.GenerateRandomTokenIDs(rng, 40) // 2 whole 16-token blocks and 8 more tokens
	a := append(append([]sim.TokenID{}, prefix...), sim.GenerateRandomTokenIDs(rng, 30)...)
	b := append(append([]sim.TokenID{}, prefix...), sim.GenerateRandomTokenIDs(rng, 30)...)
	c := newPromptCanon(7, 16, 1000)
	ca, cb := c.canonical(a), c.canonical(b)
	if !slices.Equal(ca[:32], cb[:32]) {
		t.Errorf("the two shared whole blocks differ")
	}
	if slices.Equal(ca[32:48], cb[32:48]) {
		t.Errorf("the block where the prompts diverge is equal")
	}
	if again := c.canonical(a); !slices.Equal(again, ca) {
		t.Errorf("a repeated prompt is shown differently")
	}
}

// TestPromptCanon_IndependentOfPlaceholderValues: two workloads with the same lengths and the same
// prefix sharing, but different placeholder token values (a run and its replay), are shown the
// same prompts in the same order.
func TestPromptCanon_IndependentOfPlaceholderValues(t *testing.T) {
	gen := func(seed int64) [][]sim.TokenID {
		rng := rand.New(rand.NewSource(seed))
		shared := sim.GenerateRandomTokenIDs(rng, 256)
		var out [][]sim.TokenID
		for i := 0; i < 20; i++ {
			p := sim.GenerateRandomTokenIDs(rng, 50+i)
			if i%3 != 0 {
				p = append(append([]sim.TokenID{}, shared...), p...)
			}
			out = append(out, p)
		}
		return out
	}
	run, replay := gen(1), gen(2)
	c1, c2 := newPromptCanon(9, 16, 1<<16), newPromptCanon(9, 16, 1<<16)
	for i := range run {
		if !slices.Equal(c1.canonical(run[i]), c2.canonical(replay[i])) {
			t.Fatalf("prompt %d is shown differently in the run and its replay", i)
		}
	}
}

// TestPromptCanon_BoundedAndDeterministic: the chain set never exceeds its capacity, and evictions
// depend only on the prompt sequence.
func TestPromptCanon_BoundedAndDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	var prompts [][]sim.TokenID
	for i := 0; i < 200; i++ {
		prompts = append(prompts, sim.GenerateRandomTokenIDs(rng, 100))
	}
	c1, c2 := newPromptCanon(1, 16, 50), newPromptCanon(1, 16, 50)
	for _, p := range prompts {
		if !slices.Equal(c1.canonical(p), c2.canonical(p)) {
			t.Fatal("same sequence, different canonical prompts")
		}
		if c1.lru.Len() > 50 || len(c1.chains) != c1.lru.Len() {
			t.Fatalf("chain set %d / %d exceeds capacity 50", c1.lru.Len(), len(c1.chains))
		}
	}
}

// TestPromptCanon_PinnedChainsSurviveChurn (review 6205960463): a chain an engine reports cached
// keeps its canonical tokens however many unrelated prompts churn the LRU, and is forgotten only
// after it is unpinned.
func TestPromptCanon_PinnedChainsSurviveChurn(t *testing.T) {
	c := newPromptCanon(7, 2, 2)
	p := []sim.TokenID{1, 2}
	before := c.canonical(p)
	c.pin(hashBlock("", p)) // the engine stored the block
	for i := 0; i < 10; i++ {
		_ = c.canonical([]sim.TokenID{sim.TokenID(100 + 2*i), sim.TokenID(101 + 2*i)})
	}
	if got := c.canonical(p); !slices.Equal(got, before) {
		t.Fatalf("a cached block's canonical tokens changed under churn: %v -> %v", before, got)
	}
	c.unpin(hashBlock("", p)) // the engine evicted it
	for i := 0; i < 10; i++ {
		_ = c.canonical([]sim.TokenID{sim.TokenID(300 + 2*i), sim.TokenID(301 + 2*i)})
	}
	if c.lru.Len() > 2 {
		t.Fatalf("%d unpinned chains kept, capacity 2", c.lru.Len())
	}
}
