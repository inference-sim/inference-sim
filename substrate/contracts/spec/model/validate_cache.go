package model

import (
	"fmt"
	"maps"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// validateNodeCaches checks the caches one attention node states, in isolation: each is
// well formed, and the node's read shape agrees with what it reads. Name uniqueness is
// layer-wide, so validateLayerCacheNames checks it.
func (n Node) validateNodeCaches(p *validate.Problems, at string) {
	if len(n.Caches) == 0 {
		return
	}
	if n.Op != OpAttention {
		p.Field(at+".caches", "only an attention node reads a cache")
		return
	}
	for i, c := range n.Caches {
		ca := fmt.Sprintf("%s.caches[%d]", at, i)
		if c.Name == "" {
			p.Field(ca+".name", "required")
		}
		if !c.Kind.Valid() {
			p.Field(ca+".kind", "%q is not a recognized cache kind", c.Kind)
		}
		requirePositive(p, ca, "bytes_per_entry", c.BytesPerEntry)
		for field, v := range map[string]int{
			"tokens_per_entry": c.TokensPerEntry, "window": c.Window, "entries": c.Entries,
			"block_size": c.BlockSize, "page_alignment": c.PageAlignment,
		} {
			if v < 0 {
				p.Field(ca+"."+field, "must not be negative")
			}
		}
		// A window bounds a window cache and nothing else, and a fixed entry count sizes
		// a state cache and nothing else: either on another kind would be a second,
		// contradictory statement of how the cache grows.
		switch c.Kind {
		case CacheWindow:
			if c.Window < 1 {
				p.Field(ca+".window", "a window cache requires a positive window")
			}
		case CacheState:
			if c.Entries < 1 {
				p.Field(ca+".entries", "a state cache requires a positive entry count")
			}
		}
		// Fixed per-request state is the writing layer's own: every compressor owns
		// its ring (models/deepseek_v41/compressor.py), and a reader layer has no
		// compressor to read one with. A state cache marked as read would silently drop
		// the ring's residency, so it is refused.
		if c.Kind == CacheState && !c.Owned {
			p.Field(ca+".owned", "a state cache is its layer's own; it cannot be read from another layer")
		}
		if c.Window != 0 && c.Kind != CacheWindow {
			p.Field(ca+".window", "only a window cache is bounded by a window")
		}
		if c.Entries != 0 && c.Kind != CacheState {
			p.Field(ca+".entries", "only a state cache has a fixed entry count")
		}
		if c.Kind == CacheWindow && c.TokensPerEntry > 1 {
			p.Field(ca+".tokens_per_entry", "a window cache holds one entry per token")
		}
		for _, dt := range sortedKeys(c.BytesByCacheDType) {
			if dt == "" {
				p.Field(ca+".bytes_by_cache_dtype", "a cache dtype name is empty")
			} else if v := c.BytesByCacheDType[dt]; v < 1 {
				p.Field(ca+".bytes_by_cache_dtype."+dt, "must be positive, got %d", v)
			}
		}
		// A cache another layer owns is paged by that layer, so its page size and
		// alignment are the owner's to state; a reader restating them could disagree with
		// nothing to say which one holds. What a reader reads -- the entry, under each
		// cache dtype, and how many tokens and entries it spans -- it states, and
		// validateCacheSharing requires it to equal the owner's.
		if !c.Owned {
			for field, set := range map[string]bool{
				"block_size": c.BlockSize != 0, "page_alignment": c.PageAlignment != 0,
			} {
				if set {
					p.Field(ca+"."+field, "describes how the owning layer pages the cache; remove it from a cache this layer does not own")
				}
			}
		}
	}
	// The read shape and the cache it reads must describe one stream. A compressed
	// sparse layer reads kv and index caches holding one entry per compress_ratio tokens
	// -- V4.1's indexer keys are pooled at the layer's ratio like its latent
	// (MLAAttentionSpec tokens_per_state=self.compress_ratio for both) -- and a node that
	// does not compress reads one entry per token. A cache at any other granularity would
	// size one stream and price the read of another.
	ratio := n.CompressRatio
	if ratio < 1 {
		ratio = 1
	}
	for i, c := range n.Caches {
		at := fmt.Sprintf("%s.caches[%d]", at, i)
		switch c.Kind {
		case CacheKV, CacheIndex:
			if c.EntryTokens() != ratio {
				p.Field(at+".tokens_per_entry",
					"is %d but the node reads at compress_ratio %d", c.EntryTokens(), ratio)
			}
		case CacheWindow:
			// A sliding-window cache is read by a sliding-window read at least as wide; a
			// wider one is allowed only where validateWindowReads finds the engine reads
			// past the window.
			if n.Window < 1 || c.Window > n.Window {
				p.Field(at+".window", "is %d but the node reads a window of %d", c.Window, n.Window)
			}
		}
	}
}

func sortedKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// validateCacheSharing resolves every cache a layer reads but does not own against the
// expanded stack: the nearest preceding layer whose same-named cache is owned must exist
// and describe the same stream. It runs per stack, because a speculator's draft layers
// are a stack of their own and do not read the target's caches by this rule.
func (g *Graph) validateCacheSharing(p *validate.Problems, field string, s Stack) {
	kinds := map[string]int{}
	for i, lk := range g.LayerKinds {
		kinds[lk.ID] = i
	}
	// A layer kind repeated across the stack would otherwise report one fault once per
	// position it occupies; the first position is enough to find it.
	reported := map[string]bool{}
	owner := map[string]Cache{}
	ownerAt := map[string]int{}
	for i, id := range s.Expand() {
		li, ok := kinds[id]
		if !ok {
			continue // reported by validateStack
		}
		lk := g.LayerKinds[li]
		// Owners are recorded after the layer's readers are resolved, so a layer cannot
		// read a cache it owns itself under the same name: that is one cache, owned.
		var owned []Cache
		for j, n := range lk.Nodes {
			for k, c := range n.Caches {
				if c.Owned {
					owned = append(owned, c)
					continue
				}
				src, found := owner[c.Name]
				at := fmt.Sprintf("layer_kinds[%d].nodes[%d].caches[%d]", li, j, k)
				if reported[at] {
					continue
				}
				switch {
				case !found:
					p.Field(at, "%s layer %d reads cache %q, but no earlier layer owns one", field, i, c.Name)
				case !sameRead(src, c):
					p.Field(at, "%s layer %d reads cache %q as %s, but layer %d owns it as %s",
						field, i, c.Name, readShape(c), ownerAt[c.Name], readShape(src))
				default:
					continue
				}
				reported[at] = true
			}
		}
		for _, c := range owned {
			owner[c.Name] = c
			ownerAt[c.Name] = i
		}
	}
}

// validateLayerCacheNames requires cache names to be unique across a layer's nodes, not
// only within one node: a reader finds its writer by name, so two same-named caches in
// one layer would leave which one it reads undefined.
func validateLayerCacheNames(p *validate.Problems, at string, lk LayerKind) {
	seen := map[string]bool{}
	for j, n := range lk.Nodes {
		for k, c := range n.Caches {
			if c.Name == "" {
				continue
			}
			if seen[c.Name] {
				p.Field(fmt.Sprintf("%s.nodes[%d].caches[%d].name", at, j, k),
					"duplicate cache name %q in one layer", c.Name)
			}
			seen[c.Name] = true
		}
	}
}

// sameRead reports whether a reader states exactly what its owner stores: the same kind,
// entry, per-dtype records and span. The records must be equal as stated, not only as
// looked up, because an arch override restates bytes_per_entry and the records separately:
// an owner listing {fp8_ds_mla: 584} beside bytes_per_entry 584 and a reader listing
// nothing agree until an override moves bytes_per_entry, and then they would not.
func sameRead(owner, reader Cache) bool {
	return owner.Kind == reader.Kind && owner.EntryTokens() == reader.EntryTokens() &&
		owner.Window == reader.Window && owner.Entries == reader.Entries &&
		owner.BytesPerEntry == reader.BytesPerEntry &&
		maps.Equal(owner.BytesByCacheDType, reader.BytesByCacheDType)
}

func readShape(c Cache) string {
	s := fmt.Sprintf("%s, %d B per %d tokens", c.Kind, c.BytesPerEntry, c.EntryTokens())
	if len(c.BytesByCacheDType) > 0 {
		s += fmt.Sprintf(" (%v by cache dtype)", c.BytesByCacheDType)
	}
	if c.Window > 0 {
		s += fmt.Sprintf(", window %d", c.Window)
	}
	if c.Entries > 0 {
		s += fmt.Sprintf(", %d entries", c.Entries)
	}
	return s
}

// validateWindowReads requires a sliding-window read to be exactly as wide as the window
// cache it reads, with one exception the engine makes: a parallel-block drafter's query
// reads the trailing window plus every position of its block, non-causally, so its read is
// window + num_spec_tokens (v1/attention/backends/mla/sparse_swa.py;
// compressor_utils.py get_dspark_swa_index_width). That holds only for layer kinds the
// draft stack alone runs; any other excess describes no read the engine makes.
func (g *Graph) validateWindowReads(p *validate.Problems) {
	draftOnly := map[string]bool{}
	if s := g.Speculator; s != nil && s.Drafting == DraftingParallelBlock {
		for _, id := range s.Stack.Expand() {
			draftOnly[id] = true
		}
		other := g.Stack.Expand()
		if s.ContextStack != nil {
			other = append(other, s.ContextStack.Expand()...)
		}
		for _, id := range other {
			delete(draftOnly, id)
		}
	}
	for i, lk := range g.LayerKinds {
		for j, n := range lk.Nodes {
			for k, c := range n.Caches {
				if c.Kind != CacheWindow || c.Window < 1 || n.Window <= c.Window {
					continue
				}
				extra := n.Window - c.Window
				if draftOnly[lk.ID] && extra == g.Speculator.NumSpec {
					continue
				}
				at := fmt.Sprintf("layer_kinds[%d].nodes[%d].caches[%d].window", i, j, k)
				if draftOnly[lk.ID] {
					p.Field(at, "is %d and the read %d, but a parallel-block draft reads its window plus its %d block positions, %d",
						c.Window, n.Window, g.Speculator.NumSpec, c.Window+g.Speculator.NumSpec)
				} else {
					p.Field(at, "is %d but the node reads a window of %d; only a parallel-block draft layer reads past its window", c.Window, n.Window)
				}
			}
		}
	}
}
