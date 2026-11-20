package rules

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// KVCacheLayer is one engine KV cache layer: one owned cache of one layer of the graph.
// the engine registers each such cache as its own layer with its own spec.
type KVCacheLayer struct {
	Name  string
	Cache model.Cache
}

// OwnedKVCacheLayers lists every owned cache of a graph, in the order the engine meets
// them: the target's stack, layer by layer, then the speculator's, each layer's caches in
// the order its nodes state them. They are named layers.<i>.<cache> and
// speculator.layers.<i>.<cache>. A cache a layer reads but does not own is not a layer.
func OwnedKVCacheLayers(g *model.Graph) []KVCacheLayer {
	kinds := map[string]*model.LayerKind{}
	for i := range g.LayerKinds {
		kinds[g.LayerKinds[i].ID] = &g.LayerKinds[i]
	}
	var out []KVCacheLayer
	walk := func(prefix string, s model.Stack) {
		for i, id := range s.Expand() {
			lk := kinds[id]
			if lk == nil {
				continue
			}
			for _, n := range lk.Nodes {
				for _, c := range n.Caches {
					if c.Owned {
						out = append(out, KVCacheLayer{Name: fmt.Sprintf("%slayers.%d.%s", prefix, i, c.Name), Cache: c})
					}
				}
			}
		}
	}
	walk("", g.Stack)
	if g.Speculator != nil {
		walk("speculator.", g.Speculator.Stack)
	}
	return out
}

// KVCacheGroupsInput is what the grouping reads besides the caches.
type KVCacheGroupsInput struct {
	// Layers are the owned caches in engine order (OwnedKVCacheLayers). The order decides
	// which layers share a group, not the bytes per block.
	Layers []KVCacheLayer
	// EngineBlockSize is the deployment's block_size, for a cache that states none.
	EngineBlockSize int
	// CacheDType is the cache dtype that runs, which picks each entry's record.
	CacheDType string
	// BlockStrideAlignment is the least common multiple of the block_stride_alignment the
	// engine's backends state for these layers, which the graph does not carry: on
	// DeepSeek-V4.1 the indexer's lcm(512, page alignment) when indexer_sparse_logits is
	// on (models/deepseek_v41/attention.py). Zero means none.
	BlockStrideAlignment int
	// NumSpecTokens is the deployment's speculative K, which sizes a state cache's ring
	// where the release derives it from K (KVCachePacking.StateRingFromSpecTokens). Nil
	// takes the ring the graph states (Cache.Entries), which is right only at the K the
	// graph's speculator was derived for.
	NumSpecTokens *int
}

// KVCacheGroup is one KV cache group: layers sharing a block table.
type KVCacheGroup struct {
	Layers []string
	// BlockSize is the group's tokens per block.
	BlockSize int
	// PageBytes is the bytes its layers occupy in one block, the sum of their pages.
	PageBytes int
}

// KVCacheGrouping is a model's KV cache groups and the bytes each pool block costs.
type KVCacheGrouping struct {
	// Packed is whether the mixed-page-size packer formed the groups; otherwise every
	// layer shares one group.
	Packed bool
	Groups []KVCacheGroup
	// BytesPerBlock is what one block of the shared pool costs: the widest group's page
	// bytes, rounded to the block-stride alignment. Every group's block costs this much,
	// whatever its own layers occupy.
	BytesPerBlock int
	// PageBytes is each layer's own page size, by layer name.
	PageBytes map[string]int
}

// GroupOf returns the index of the group holding the named layer, or -1.
func (g KVCacheGrouping) GroupOf(layer string) int {
	for i, gr := range g.Groups {
		for _, n := range gr.Layers {
			if n == layer {
				return i
			}
		}
	}
	return -1
}
