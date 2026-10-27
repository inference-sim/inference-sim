// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/model_executor/models/gpt_oss.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// The KV cache spec each attention layer builds: gpt_oss.py:176 (per_layer_sliding_window),
// vllm/models/minimax_m3/nvidia/model.py:562-571 (MiniMaxM3SparseAttention.get_kv_cache_spec)
// and vllm/models/minimax_m3/common/indexer.py:145-154 (the indexer's); DeepSeek V4's and V4.1's
// caches as their modules register them: vllm/models/deepseek_v4/attention.py:286-372 and :856-925,
// vllm/models/deepseek_v4/compressor.py:139-193, vllm/models/deepseek_v41/attention.py:420-600 and
// :1138-1240, vllm/models/deepseek_v41/compressor.py:136-167 and
// vllm/v1/attention/backends/mla/sparse_swa.py:74-140.

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// KVLayerRun is a run of consecutive attention layers of one KV cache spec, in core/vllm's
// kv_layers terms: Spec is "full", "sliding_window" or "indexer", or one of DeepSeek V4's "mla",
// "sliding_window_mla" and "circular_buffer", and NumKVHeads is per tensor-parallel rank. A
// DeepSeek V4 spec also states its own block size, the tokens a state covers, a packed state's
// bytes, the dtype's bytes, its page alignment, its block stride alignment and its window's
// bounded replay.
type KVLayerRun struct {
	Count           int
	Spec            string
	SlidingWindow   int
	NumKVHeads      int
	HeadSize        int
	BlockSize       int
	TokensPerState  int
	StateBytes      int
	ElemBytes       int
	Alignment       int
	StrideAlignment int
	BoundedReplay   bool
}

// KVLayers is the pool's attention layers by the KV cache spec vLLM v0.32.0 builds for each, in
// model order:
//
//   - a sliding-window attention node is SlidingWindowSpec (gpt-oss's even layers,
//     gpt_oss.py:176), except in a layer whose indexer selects positions from the whole
//     context: that layer keeps full-attention KV, and the window bounds only what its sparse
//     kernel reads (MiniMax-M3's sparse layers, minimax_m3/nvidia/model.py:562-571);
//   - a grouped-query node owning an index cache is the indexer's key-only spec, one head
//     (indexer.py:145-154);
//   - any other attention node is full attention.
//
// DeepSeek V4 and V4.1 are mapped by deepseekKVLayers, every cache they register in order.
//
// It is nil when every layer is full attention, which is one group. What it does not map is an
// error naming it: recurrent layers, window and state caches, another kind's indexer, shared KV
// beside these, and a drafter's layers beside them.
func (m *Model) KVLayers() ([]KVLayerRun, error) {
	runner, _ := m.ModelRunner()
	return kvLayersOf(m.graph, m.k.Deployment(), m.engine,
		deepseekServing{runner: runner, family: m.id.capabilityMajor, cacheDType: m.cacheDType})
}

func kvLayersOf(g *model.Graph, pool deployment.Pool, engine engineconfig.Engine, ds deepseekServing) ([]KVLayerRun, error) {
	if g == nil {
		return nil, nil
	}
	if a := g.Global.Architecture; a == rules.ArchDeepseekV4 || a == rules.ArchDeepseekV41 {
		return deepseekKVLayers(g, pool, engine, ds)
	}
	kinds := map[string]model.LayerKind{}
	for _, k := range g.LayerKinds {
		kinds[k.ID] = k
	}
	tp := max(pool.Parallel.TP, 1)
	var runs []KVLayerRun
	plain, shared := true, false
	for _, id := range g.Stack.Expand() {
		indexed := false
		for _, n := range kinds[id].Nodes {
			for _, c := range n.Caches {
				indexed = indexed || c.Kind == model.CacheIndex && c.Owned
			}
		}
		for _, n := range kinds[id].Nodes {
			if n.Op == model.OpRecurrentUpdate {
				return nil, fmt.Errorf("layer kind %s: recurrent state is not mapped to a KV cache spec", id)
			}
			if n.Op != model.OpAttention {
				continue
			}
			r := KVLayerRun{Count: 1, Spec: "full", NumKVHeads: max(n.NumKVHeads/tp, 1), HeadSize: n.HeadDim}
			for _, c := range n.Caches {
				switch {
				case c.Kind == model.CacheKV:
					shared = shared || !c.Owned
				case c.Kind == model.CacheIndex && c.Owned && n.AttentionKind == model.AttentionGQA:
					r.Spec, r.NumKVHeads = "indexer", 1
				default:
					return nil, fmt.Errorf("layer kind %s: its %s cache is not mapped to a KV cache spec", id, c.Kind)
				}
			}
			if n.AttentionKind == model.AttentionSWA && !indexed {
				r.Spec, r.SlidingWindow = "sliding_window", n.Window
			}
			plain = plain && r.Spec == "full"
			if l := len(runs) - 1; l >= 0 && runs[l].Spec == r.Spec && runs[l].SlidingWindow == r.SlidingWindow &&
				runs[l].NumKVHeads == r.NumKVHeads && runs[l].HeadSize == r.HeadSize {
				runs[l].Count++
			} else {
				runs = append(runs, r)
			}
		}
	}
	switch sp := engine.Speculative; {
	case plain:
		return nil, nil
	case shared:
		return nil, fmt.Errorf("a layer reading another layer's KV beside these KV cache specs is not mapped")
	case sp != nil && sp.NumSpecTokens > 0:
		return nil, fmt.Errorf("a %s drafter's KV layers beside these KV cache specs are not mapped", sp.Method)
	}
	return runs, nil
}

// deepseekServing is what DeepSeek V4's caches turn on beside the graph and the pool: the model
// runner (V4.1's bounded replay needs V2) and the chip's compute-capability family, which picks
// V4.1's KV record (_use_v41_mxfp8_kv_record, vllm/models/deepseek_v41/attention.py:128-129@5df63a72)
// and the graph's architecture overlay.
type deepseekServing struct {
	runner string
	family int
	// cacheDType is the cache dtype that runs, as the release's rules resolve the requested one
	// for the architecture, family and TP (rules.EngineRules.ResolveCacheDType), which picks a
	// cache's record (V4.1's compressed cache is NVFP4 under nvfp4_ds_mla).
	cacheDType string
}

// deepseekKVLayers is DeepSeek V4's and V4.1's KV cache layers as vLLM v0.32.0 registers them, one
// per cache, in model order: per layer of the target's stack, an indexer's key cache (V4's ratio-4
// layers, attention.py:298-317; V4.1's kv sources, :432-455), the sliding window
// (DeepseekV4SWACache, sparse_swa.py:116-140), then the compressed MLA cache (a V4 layer of ratio
// above 1, attention.py:856-880; a V4.1 kv source, :1138-1165) and its compressor state, a
// SlidingWindowMLASpec on V4 (compressor.py:179-193) and a ring on V4.1 (compressor.py:159-167);
// then the drafter's layers, which are pure sliding windows (MTP's, and DSpark's past the target's
// layers, attention.py:310-315).
//
// The graph's caches give each spec's record: a window cache is the sliding window, a kv or index
// cache the compressed MLA cache at its tokens_per_entry, and a state cache the compressor's state,
// of 2 x coff x head_dim float32 rows. The graph is resolved for the chip's family first, as the
// kernel resolves it, so V4.1's SM100 record and alignment are the ones that run.
func deepseekKVLayers(g *model.Graph, pool deployment.Pool, engine engineconfig.Engine, ds deepseekServing) ([]KVLayerRun, error) {
	g, err := g.ForCapability(ds.family)
	if err != nil {
		return nil, fmt.Errorf("deepseek KV layers: %w", err)
	}
	v41 := g.Global.Architecture == rules.ArchDeepseekV41
	e := engine
	k := 0
	if sp := e.Speculative; sp != nil {
		k = sp.NumSpecTokens
	}
	// swa_bounded_replay defaults to true (vllm/config/cache.py:250@5df63a72) and V4.1 turns it off
	// on the V1 model runner (attention.py:529-536); prefill context parallelism turns it off too
	// (:537-547). V4 states no bounded replay.
	replay := v41 && ds.runner == "v2" && max(pool.Parallel.PCP, 1) == 1
	if e.SWABoundedReplay != nil && !*e.SWABoundedReplay {
		replay = false
	}
	kinds := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		kinds[lk.ID] = lk
	}
	var runs []KVLayerRun
	add := func(r KVLayerRun) {
		r.Count = 1
		if l := len(runs) - 1; l >= 0 && sameRun(runs[l], r) {
			runs[l].Count++
			return
		}
		runs = append(runs, r)
	}
	layer := func(id string, draft bool) error {
		lk, ok := kinds[id]
		if !ok {
			return fmt.Errorf("deepseek KV layers: layer kind %s is not in the graph", id)
		}
		// The indexer's node (role block_index_scores) owns its key cache and, on V4, its own
		// compressor's state; the attention node owns the window, the compressed cache and the
		// main compressor's state. A cache another layer owns is that layer's.
		var index, indexState, swa, compressed, state *model.Cache
		indexRatio, ratio := 1, 1
		for _, n := range lk.Nodes {
			if n.Op != model.OpAttention {
				continue
			}
			scorer := n.Role == "block_index_scores"
			for i := range n.Caches {
				c := &n.Caches[i]
				if !c.Owned {
					continue
				}
				switch {
				case scorer && c.Kind == model.CacheIndex:
					index, indexRatio = c, max(n.CompressRatio, 1)
				case scorer && c.Kind == model.CacheState:
					indexState = c
				case scorer:
					return fmt.Errorf("deepseek KV layers: layer kind %s: its indexer's %s cache is not mapped", id, c.Kind)
				case c.Kind == model.CacheWindow:
					swa = c
				case c.Kind == model.CacheKV:
					compressed, ratio = c, max(n.CompressRatio, 1)
				case c.Kind == model.CacheState:
					state = c
				default:
					return fmt.Errorf("deepseek KV layers: layer kind %s: its %s cache is not mapped", id, c.Kind)
				}
			}
		}
		if swa == nil {
			return fmt.Errorf("deepseek KV layers: layer kind %s owns no sliding-window cache", id)
		}
		if index != nil {
			// DeepseekV4IndexerCache: a key-only MLAAttentionSpec at the engine's block size, one
			// state per compress_ratio tokens (deepseek_v4/attention.py:906-925,
			// deepseek_v41/attention.py:1206-1240).
			add(KVLayerRun{Spec: "mla", NumKVHeads: 1, HeadSize: index.BytesPerEntry,
				BlockSize: e.BlockSize, TokensPerState: index.EntryTokens(), Alignment: index.PageAlignment})
		}
		if indexState != nil {
			if v41 {
				return fmt.Errorf("deepseek KV layers: layer kind %s: a V4.1 indexer states no compressor state", id)
			}
			add(v4CompressorState(indexState, indexRatio, swa.PageAlignment))
		}
		add(KVLayerRun{Spec: "sliding_window_mla", SlidingWindow: swa.Window, NumKVHeads: 1, HeadSize: 512,
			BlockSize: swa.BlockSize, StateBytes: swa.EntryBytes(ds.cacheDType), Alignment: swa.PageAlignment,
			BoundedReplay: replay})
		if draft || compressed == nil {
			return nil
		}
		add(KVLayerRun{Spec: "mla", NumKVHeads: 1, HeadSize: 512, BlockSize: e.BlockSize,
			TokensPerState: ratio, StateBytes: compressed.EntryBytes(ds.cacheDType), Alignment: compressed.PageAlignment})
		switch {
		case state == nil:
		case v41 && ratio > 1:
			// The compressor ring: drafts + bonus token + the previous row, as a power of two of at
			// least 8 (deepseek_v41/compressor.py:150-153), float32 rows of 2 x head_dim.
			add(KVLayerRun{Spec: "circular_buffer", NumKVHeads: 1, HeadSize: state.BytesPerEntry / 4,
				BlockSize: rules.CompressorRingRows(k), ElemBytes: 4})
		case v41:
		default:
			add(v4CompressorState(state, ratio, swa.PageAlignment))
		}
		return nil
	}
	for _, id := range g.Stack.Expand() {
		if err := layer(id, false); err != nil {
			return nil, err
		}
	}
	if sp := g.Speculator; sp != nil && e.Speculative != nil && e.Speculative.NumSpecTokens > 0 {
		for _, id := range sp.Stack.Expand() {
			if err := layer(id, true); err != nil {
				return nil, err
			}
		}
	}
	return runs, nil
}

// sameRun is whether two layers build one spec, so a run counts both.
func sameRun(a, b KVLayerRun) bool {
	a.Count, b.Count = 0, 0
	return a == b
}

// v4CompressorState is DeepSeek V4's CompressorStateCache spec (deepseek_v4/compressor.py:139-193
// @5df63a72): a SlidingWindowMLASpec of coff x ratio tokens (coff 2 at ratio 4, else 1), at block
// 4 for ratio 4 and 8 for 128, of float32 rows of 2 x coff x head_dim, which the graph states as
// its bytes per entry, aligned as the layer's fp8_ds_mla pages are.
func v4CompressorState(c *model.Cache, ratio, alignment int) KVLayerRun {
	coff, block := 1, 8
	if ratio == 4 {
		coff, block = 2, 4
	}
	return KVLayerRun{Spec: "sliding_window_mla", SlidingWindow: coff * ratio, NumKVHeads: 1,
		HeadSize: c.BytesPerEntry / 4, BlockSize: block, ElemBytes: 4, Alignment: alignment}
}
