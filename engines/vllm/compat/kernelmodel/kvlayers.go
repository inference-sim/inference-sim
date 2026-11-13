// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/model_executor/models/gpt_oss.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// The KV cache spec each attention layer builds: gpt_oss.py:176 (per_layer_sliding_window),
// vllm/models/minimax_m3/nvidia/model.py:562-571 (MiniMaxM3SparseAttention.get_kv_cache_spec)
// and vllm/models/minimax_m3/common/indexer.py:145-154 (the indexer's), whose dtype is
// indexer.py:118-127 and nvidia/model.py:505-509; DeepSeek V4's and V4.1's
// caches as their modules register them: vllm/models/deepseek_v4/attention.py:286-372 and :856-925,
// vllm/models/deepseek_v4/compressor.py:139-193, vllm/models/deepseek_v41/attention.py:420-600 and
// :1138-1240, vllm/models/deepseek_v41/compressor.py:136-167 and
// vllm/v1/attention/backends/mla/sparse_swa.py:74-140; a recurrent layer's MambaSpec is
// mamba_state.go's.

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
	"github.com/inference-sim/blis-vllm/rules/pack"
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
	// A "mamba" run is MambaSpec (core/vllm KVLayerRun): StateBytes per rank, BlockSize the
	// mamba_block_size, and the rest of the spec.
	PageSizePadded             int
	MambaCacheMode             string
	NumSpeculativeBlocks       int
	NumPrefillCheckpointBlocks int
	PrefillCheckpointAlignment int
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
//   - any other attention node is full attention;
//   - a recurrent node (Mamba-2, KDA, GDN) is MambaSpec, a "mamba" run (planMamba), beside the
//     attention layers' runs in model order.
//
// DeepSeek V4 and V4.1 are mapped by deepseekKVLayers, every cache they register in order.
//
// It is nil when every layer is full attention, which is one group. What it does not map is an
// error naming it: a recurrent layer whose state the graph or the engine block does not state,
// window and state caches, another kind's indexer, shared KV beside these, and a drafter's layers
// beside them.
func (m *Model) KVLayers() ([]KVLayerRun, error) {
	runner, _ := m.ModelRunner()
	return kvLayersOf(m.graph, m.k.Deployment(), m.engine, m.serving(runner))
}

// serving is what the layers' caches turn on beside the graph and the pool.
func (m *Model) serving(runner string) deepseekServing {
	return deepseekServing{runner: runner, family: m.id.capabilityMajor, cacheDType: m.cacheDType,
		indexer: m.indexerRules, mamba: m.resolved.mamba, mambaErr: m.resolved.mambaErr}
}

// PipelineStages is the pool's pipeline_parallel_size, at least one.
func (m *Model) PipelineStages() int { return max(m.k.Deployment().Parallel.PP, 1) }

// KVLayersOfStage is KVLayers for one pipeline stage's rank: only the layers vLLM places on
// stage pp of the pool's pipeline_parallel_size (get_pp_indices,
// vllm/distributed/utils.py:126-169@5df63a72), and on the last stage the drafter's, which
// registers after the target's (dspark.py, start_layer_id = num_hidden_layers,
// vllm/models/muse_spark/common/dspark.py@0f6b2619). Each worker sizes its pool from its own
// layers and free memory, and the engine keeps the smallest (get_kv_cache_configs,
// vllm/v1/core/kv_cache_utils.py:2530-2660@5df63a72). At pp 1 it is KVLayers.
func (m *Model) KVLayersOfStage(stage int) ([]KVLayerRun, error) {
	pool := m.k.Deployment()
	pp := max(pool.Parallel.PP, 1)
	if pp == 1 {
		return m.KVLayers()
	}
	if stage < 0 || stage >= pp {
		return nil, fmt.Errorf("pipeline stage %d of %d", stage, pp)
	}
	g := *m.graph
	seq := g.Stack.Expand()
	if len(seq) < pp {
		return nil, fmt.Errorf("%d layers over %d pipeline stages", len(seq), pp)
	}
	lo, hi := ppIndices(len(seq), pp, stage)
	g.Stack = model.Stack{Pattern: seq[lo:hi], Repeat: 1}
	runner, _ := m.ModelRunner()
	e := m.engine
	// The drafter runs on the last stage only: earlier stages register none of its layers,
	// which kvLayersOf reads from the engine's draft length.
	if stage != pp-1 && e.Speculative != nil {
		cp := *e.Speculative
		cp.NumSpecTokens = 0
		e.Speculative = &cp
	}
	return kvLayersOf(&g, pool, e, m.serving(runner))
}

// ppIndices is get_pp_indices (vllm/distributed/utils.py:126-169@5df63a72) without
// VLLM_PP_LAYER_PARTITION: stage's [start, end) of layers over pp stages.
func ppIndices(layers, pp, stage int) (int, int) {
	parts := make([]int, pp)
	for i := range parts {
		parts[i] = layers / pp
	}
	if rem := layers % pp; rem > 0 {
		for i := 2; i < rem+2; i++ {
			parts[pp-i]++
		}
	}
	start := 0
	for i := range stage {
		start += parts[i]
	}
	return start, start + parts[stage]
}

func kvLayersOf(g *model.Graph, pool deployment.Pool, engine engineconfig.Engine, ds deepseekServing) ([]KVLayerRun, error) {
	runs, err := kvLayersRead(g, pool, engine, ds)
	if err != nil {
		return nil, err
	}
	// attention_config.indexer_kv_dtype is read only for MiniMax-M3's indexer cache
	// (m3IndexerElemBytes). Stated for a model with no such cache nothing reads it, so it would be
	// dropped (blis-vllm#84); "auto" is every model's own default and is not a deviation.
	// DeepSeek V4 and V4.1 build their indexer's cache as an MLA spec, and deepseekKVLayers resolves the
	// dtype for it.
	if d := engine.IndexerKVDType; g != nil && d != "" && d != "auto" && !isDeepseekV4(g) {
		indexer := false
		for _, r := range runs {
			indexer = indexer || r.Spec == "indexer"
		}
		if !indexer {
			return nil, fmt.Errorf("indexer_kv_dtype=%q is read only for MiniMax-M3's indexer cache, and this model has none, so a "+
				"stated value would be dropped (https://github.com/inference-sim/blis-vllm/issues/84)", d)
		}
	}
	return runs, nil
}

// isDeepseekV4 is whether g is DeepSeek V4 or V4.1.
func isDeepseekV4(g *model.Graph) bool {
	a := g.Global.Architecture
	return a == rules.ArchDeepseekV4 || a == rules.ArchDeepseekV41
}

// recurrentDrafterRefusal refuses an enabled speculator whose stack holds a recurrent layer, whatever
// the target is. vLLM registers a drafter's layers in the same KV cache config as the target's, so
// a recurrent drafter layer adds a Mamba group and changes the cache bytes; that state is not
// mapped. planMamba refuses it too for a recurrent target, but a full-attention target never reaches
// planMamba, and the attention-only drafter mapping below would skip the layer without a word.
func recurrentDrafterRefusal(g *model.Graph, e engineconfig.Engine) error {
	if g.Speculator == nil || e.Speculative == nil || e.Speculative.NumSpecTokens <= 0 {
		return nil
	}
	var err error
	stackNodes(g, g.Speculator.Stack, func(id string, d model.Node) {
		if d.Op == model.OpRecurrentUpdate && err == nil {
			err = fmt.Errorf("recurrent layers: drafter layer kind %s holds a recurrent layer, whose state is not mapped", id)
		}
	})
	return err
}

func kvLayersRead(g *model.Graph, pool deployment.Pool, engine engineconfig.Engine, ds deepseekServing) ([]KVLayerRun, error) {
	if g == nil {
		return nil, nil
	}
	if err := recurrentDrafterRefusal(g, engine); err != nil {
		return nil, err
	}
	if isDeepseekV4(g) {
		return deepseekKVLayers(g, pool, engine, ds)
	}
	kinds := map[string]model.LayerKind{}
	for _, k := range g.LayerKinds {
		kinds[k.ID] = k
	}
	tp := max(pool.Parallel.TP, 1)
	var runs []KVLayerRun
	plain, shared := true, false
	// Beside recurrent layers a model is a hybrid, whose groups are sized by page bytes, and an
	// MLA layer's page is one latent a token, not keys and values (MLAAttentionSpec,
	// vllm/v1/kv_cache_interface.py:686-760@5df63a72). An MLA-only model is one full-attention
	// group the kernel sizes, so its layers stay "full".
	hybridMLA := false
	for _, id := range g.Stack.Expand() {
		for _, n := range kinds[id].Nodes {
			hybridMLA = hybridMLA || n.Op == model.OpRecurrentUpdate
		}
	}
	for _, id := range g.Stack.Expand() {
		indexed := false
		for _, n := range kinds[id].Nodes {
			for _, c := range n.Caches {
				indexed = indexed || c.Kind == model.CacheIndex && c.Owned
			}
		}
		for _, n := range kinds[id].Nodes {
			if n.Op == model.OpRecurrentUpdate {
				if ds.mambaErr != nil {
					return nil, ds.mambaErr
				}
				r, err := ds.mamba.mambaRun(id, n)
				if err != nil {
					return nil, err
				}
				plain = false
				if l := len(runs) - 1; l >= 0 && sameRun(runs[l], r) {
					runs[l].Count++
				} else {
					runs = append(runs, r)
				}
				continue
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
					b, err := m3IndexerElemBytes(engine.IndexerKVDType, c, n.HeadDim)
					if err != nil {
						return nil, fmt.Errorf("layer kind %s: %w", id, err)
					}
					r.Spec, r.NumKVHeads, r.ElemBytes = "indexer", 1, b
				default:
					return nil, fmt.Errorf("layer kind %s: its %s cache is not mapped to a KV cache spec", id, c.Kind)
				}
			}
			if n.AttentionKind == model.AttentionSWA && !indexed {
				r.Spec, r.SlidingWindow = "sliding_window", n.Window
			}
			if hybridMLA && r.Spec == "full" && n.AttentionKind == model.AttentionMLA {
				r.Spec, r.TokensPerState = "mla", 1
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
	// A drafter's layers register after the target's, each attention its own spec: a speculator
	// the graph states (its stack, which a deployment's speculative.draft_model grafts in) is
	// mapped as the target's layers are, appended. A sliding-window-only drafter beside a hybrid
	// target is Muse Spark's DSpark (five 2048-token windows, global_attn_cfg,
	// vllm/models/muse_spark/nvidia/model.py:1101-1108@0f6b2619).
	sp := engine.Speculative
	drafting := sp != nil && sp.NumSpecTokens > 0
	if drafting && g.Speculator != nil {
		for _, id := range g.Speculator.Stack.Expand() {
			for _, n := range kinds[id].Nodes {
				if n.Op != model.OpAttention {
					continue
				}
				if len(n.Caches) > 0 {
					return nil, fmt.Errorf("drafter layer kind %s states caches, which are not mapped to a KV cache spec", id)
				}
				r := KVLayerRun{Count: 1, Spec: "full", NumKVHeads: max(n.NumKVHeads/tp, 1), HeadSize: n.HeadDim}
				if n.AttentionKind == model.AttentionSWA {
					r.Spec, r.SlidingWindow = "sliding_window", n.Window
				}
				plain = plain && r.Spec == "full"
				if l := len(runs) - 1; l >= 0 && sameRun(runs[l], r) {
					runs[l].Count++
				} else {
					runs = append(runs, r)
				}
			}
		}
	}
	switch {
	case plain:
		return nil, nil
	case shared:
		return nil, fmt.Errorf("a layer reading another layer's KV beside these KV cache specs is not mapped")
	case drafting && g.Speculator == nil:
		return nil, fmt.Errorf("a %s drafter's KV layers beside these KV cache specs are not mapped: the graph "+
			"states no speculator to read them from", sp.Method)
	}
	return runs, nil
}

// m3IndexerElemBytes is the element size of MiniMax-M3's indexer key cache, which has its own
// dtype, attention_config.indexer_kv_dtype, not the cache dtype: "auto" (or unstated) is the
// model's default, bf16 (vllm/models/minimax_m3/nvidia/model.py:505-509@5df63a72), and the
// indexer refuses every dtype but bf16 and fp8/fp8_e4m3 (vllm/models/minimax_m3/common/indexer.py:
// 118-127@5df63a72). The latency kernel prices the cache as the catalog graph states it, the
// model's default entry (c.BytesPerEntry, one key of head_dim elements), so a stated dtype whose
// entry differs from it is refused by name rather than sized one way and priced another.
func m3IndexerElemBytes(dtype string, c model.Cache, headDim int) (int, error) {
	elem := 0
	switch dtype {
	case "", "auto", "bf16":
		elem = 2
	case "fp8", "fp8_e4m3":
		elem = 1
	default:
		return 0, fmt.Errorf("indexer_kv_dtype=%q is not supported by the MiniMax M3 indexer cache (only 'bf16' or "+
			"'fp8'/'fp8_e4m3'; https://github.com/inference-sim/blis-vllm/issues/84)", dtype)
	}
	if headDim <= 0 || elem*headDim != c.BytesPerEntry {
		return 0, fmt.Errorf("indexer_kv_dtype %q stores %d bytes a key of %d elements, and the catalog graph's %s "+
			"cache states %d; the latency kernel prices the graph's entry, so this indexer dtype is not modelled "+
			"(https://github.com/inference-sim/blis-vllm/issues/84)",
			dtype, elem*headDim, headDim, c.Name, c.BytesPerEntry)
	}
	return elem, nil
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
	// indexer resolves attention_config.indexer_kv_dtype for the DeepSeek indexer; nil when the
	// release's rules carry no such rule.
	indexer indexerKVDTypeRules
	// mamba is the spec the model's recurrent layers build (planMamba), nil for a model with none,
	// and mambaErr why they are not mapped.
	mamba    *mambaPlan
	mambaErr error
}

// indexerKVDTypeRules is the rules pack's resolution of the DeepSeek indexer's cache dtype
// (pack.Pack.ResolveIndexerKVDType); a release without the rule does not implement it.
type indexerKVDTypeRules interface {
	ResolveIndexerKVDType(architecture string, family int, requested string) (resolved string, known bool, err error)
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
	indexerDType, err := resolveIndexerDType(g.Global.Architecture, ds, engine.IndexerKVDType)
	if err != nil {
		return nil, err
	}
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
		indexRatio, ratio, indexHeadDim := 1, 1, 0
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
					index, indexRatio, indexHeadDim = c, max(n.CompressRatio, 1), n.HeadDim
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
			if indexerDType != "" {
				if w := pack.IndexerRecordBytes(indexerDType, indexHeadDim); w != index.BytesPerEntry {
					return fmt.Errorf("deepseek KV layers: layer kind %s: indexer_kv_dtype %q runs as %s, a key record of %d bytes "+
						"for an index head of %d, and the catalog graph's %s cache states %d; the latency kernel keys the index "+
						"record by one cache dtype and prices the graph's, so this indexer dtype is not modelled "+
						"(https://github.com/inference-sim/blis-vllm/issues/84)",
						id, engine.IndexerKVDType, indexerDType, w, indexHeadDim, index.Name, index.BytesPerEntry)
				}
			}
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
				BlockSize: pack.CompressorRingRows(k), ElemBytes: 4})
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

// resolveIndexerDType is the dtype that DeepSeek V4's indexer cache runs in for the requested
// attention_config.indexer_kv_dtype, on the serving chip (pack.Pack.ResolveIndexerKVDType), or
// vLLM's refusal of it. It is "" when the release's rules carry no such rule, where only an unstated
// or "auto" value is accepted, as before.
func resolveIndexerDType(arch string, ds deepseekServing, requested string) (string, error) {
	if ds.indexer != nil {
		resolved, known, err := ds.indexer.ResolveIndexerKVDType(arch, ds.family, requested)
		if err != nil {
			return "", fmt.Errorf("deepseek KV layers: indexer_kv_dtype=%q: vLLM refuses it for the DeepSeek sparse indexer: %w "+
				"(vllm/v1/attention/backends/mla/indexer.py:59-84@5df63a72; https://github.com/inference-sim/blis-vllm/issues/84)",
				requested, err)
		}
		if known {
			return resolved, nil
		}
	}
	if requested != "" && requested != "auto" {
		return "", fmt.Errorf("deepseek KV layers: indexer_kv_dtype=%q: this release's rules do not model the DeepSeek indexer's "+
			"cache dtype, so a stated value would be dropped (https://github.com/inference-sim/blis-vllm/issues/84)", requested)
	}
	return "", nil
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
