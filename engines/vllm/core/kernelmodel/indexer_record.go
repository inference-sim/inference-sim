// Original to blis-vllm.

package kernelmodel

import (
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/core/rulepack/pack"
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
//
// DeepSeek V4's and V4.1's indexer cache is stated at its FP8 record, head_dim fp8 values and one
// fp32 scale per 128 (132 bytes at head_dim 128). Under indexer_kv_dtype mxfp4 vLLM stores two
// values to the byte and one UE8M0 scale byte per 32 values, 68 bytes
// (vllm/models/deepseek_v4/attention.py:1020-1029@4ba787ec, pack.IndexerRecordBytes), so the
// kernel is handed a graph whose entry is that record, in the layer that owns each index cache and in
// every layer that reads it. Whether mxfp4 runs on the serving part is
// the release's rule to resolve (pack.Pack.ResolveIndexerKVDType), which deepseekKVLayers applies.
//
// The receiver graph is not modified.
func lowerIndexerKV(g *model.Graph, dtype string) *model.Graph {
	if g == nil {
		return g
	}
	if dtype == "mxfp4" && deepseekIndexer(g.Global.Architecture) {
		// The record each owned index cache takes, by name: an owner states its head_dim, and a
		// layer that reads the cache takes the owner's record.
		records := map[string]int{}
		for _, k := range g.LayerKinds {
			for _, n := range k.Nodes {
				for _, c := range n.Caches {
					if c.Kind == model.CacheIndex && c.Owned && n.Op == model.OpAttention && n.HeadDim > 0 &&
						c.BytesPerEntry == pack.IndexerRecordBytes("fp8", n.HeadDim) {
						records[c.Name] = pack.IndexerRecordBytes("mxfp4", n.HeadDim)
					}
				}
			}
		}
		return withIndexRecord(g, func(_ model.Node, c model.Cache) (int, bool) {
			r, ok := records[c.Name]
			return r, ok
		})
	}
	if dtype != "fp8" && dtype != "fp8_e4m3" {
		return g
	}
	return withIndexRecord(g, func(n model.Node, c model.Cache) (int, bool) {
		return n.HeadDim, c.Owned && n.Op == model.OpAttention && n.AttentionKind == model.AttentionGQA &&
			n.HeadDim > 0 && c.BytesPerEntry == 2*n.HeadDim
	})
}

// deepseekIndexer reports whether an architecture builds its indexer's key cache through
// dsa_indexer_uses_fp4 (DeepSeek V4 and V4.1).
func deepseekIndexer(architecture string) bool {
	return architecture == "DeepseekV4ForCausalLM" || architecture == "DeepseekV41ForCausalLM"
}

// withIndexRecord is g with each index cache that record accepts given the entry record returns,
// copying only what changes; g itself when none is. A DeepSeek-V4.1 layer that reads a cache another
// layer owns states its record too, and the two must agree, so the reader is lowered with the owner.
func withIndexRecord(g *model.Graph, record func(model.Node, model.Cache) (int, bool)) *model.Graph {
	var out *model.Graph
	for ki := range g.LayerKinds {
		for ni, n := range g.LayerKinds[ki].Nodes {
			for ci, c := range n.Caches {
				if c.Kind != model.CacheIndex {
					continue
				}
				entry, ok := record(n, c)
				if !ok {
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
				nn.Caches[ci].BytesPerEntry = entry
			}
		}
	}
	if out == nil {
		return g
	}
	return out
}
