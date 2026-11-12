// Original to blis-vllm.

package kernelmodel

import (
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// lowerIndexerKV is the catalog graph as the deployment's attention_config.indexer_kv_dtype
// runs it: MiniMax-M3's indexer cache, which the graph states at its default record, takes the
// record of the dtype stated.
//
// The graph's bytes_per_entry is the record under the dtype the graph's base description
// assumes (model.Cache.BytesPerEntry), and which dtype runs is the consumer's to resolve. The
// indexer's key cache has its own dtype, not kv_cache_dtype, so bytes_by_cache_dtype cannot
// state it. "auto" is bf16 (vllm/models/minimax_m3/nvidia/model.py:505-509@5df63a72) and the
// graph's 256 bytes are 128 bf16 values. fp8 and fp8_e4m3 store the same 128 values as
// torch.float8_e4m3fn with no scale (MiniMaxM3IndexerCache, common/indexer.py:118-121 and
// :145-154@5df63a72, a key-only MLAAttentionSpec of head_size 128 and that dtype; the score
// kernels take the e4m3 keys as they are, common/ops/index_topk.py:129-133 and :385-403), 128
// bytes. The kernel prices the cache at the graph's entry, so it is handed a graph whose
// entry is 128.
//
// Only a graph that states the default record for the key (2 bytes a head_dim element) is
// lowered; any other entry is left as stated and m3IndexerElemBytes refuses the dtype by name.
// The receiver graph is not modified.
func lowerIndexerKV(g *model.Graph, dtype string) *model.Graph {
	if g == nil || (dtype != "fp8" && dtype != "fp8_e4m3") {
		return g
	}
	var out *model.Graph
	for ki := range g.LayerKinds {
		for ni, n := range g.LayerKinds[ki].Nodes {
			if n.Op != model.OpAttention || n.AttentionKind != model.AttentionGQA {
				continue
			}
			for ci, c := range n.Caches {
				if c.Kind != model.CacheIndex || !c.Owned || n.HeadDim <= 0 || c.BytesPerEntry != 2*n.HeadDim {
					continue
				}
				if out == nil {
					cp := *g
					cp.LayerKinds = slices.Clone(g.LayerKinds)
					out = &cp
				}
				k := &out.LayerKinds[ki]
				if &k.Nodes[0] == &g.LayerKinds[ki].Nodes[0] {
					k.Nodes = slices.Clone(k.Nodes)
				}
				nn := &k.Nodes[ni]
				if &nn.Caches[0] == &g.LayerKinds[ki].Nodes[ni].Caches[0] {
					nn.Caches = slices.Clone(nn.Caches)
				}
				nn.Caches[ci].BytesPerEntry = n.HeadDim
			}
		}
	}
	if out == nil {
		return g
	}
	return out
}
