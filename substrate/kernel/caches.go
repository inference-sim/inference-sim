package latencykernel

import (
	"fmt"
	"math"

	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
)

// Per-layer KV caches, sized from what each layer states it owns.
//
// A graph that states an attention node's caches (blis-schemas v0.3.0 model.Cache) says
// which layer allocates each cache, how large one stored record is, how many context
// tokens a record covers, and whether a window or a fixed entry count bounds it. That is
// what residency, P/D transfer volume and offload volume follow, and none of it is
// recoverable from a read shape. For DeepSeek-V4.1-Flash on vLLM v0.31.0 the difference is
// an order of magnitude: one per-model figure times every attention layer charged 20,480
// B a context token, where only four layers own a compressed cache, a compressed record
// covers two tokens on three of them, and every layer's 128-token window is a bounded
// ring rather than a cache that grows with context -- about 1,790 B a token plus a fixed
// ring term per sequence.
//
// A layer that states no caches keeps the inference it always had (Kernel.kvBytesPerToken),
// so a graph opts in per node and no other graph moves.

// layerKV is one planned layer's KV, as the pricer reads it.
//
// The opt-in is per NODE, as blis-schemas defines it ("a node that states caches is sized
// from them alone"): a layer's primary attention may state its caches while its scorer
// does not, or the reverse, and each is sized its own way.
type layerKV struct {
	// stated is a layer whose PRIMARY attention states its caches; perToken is otherwise
	// that primary's inferred cache, in bytes per context token, per rank.
	stated   bool
	perToken float64
	// window is a sliding-window primary's window where its cache is inferred: the cache is
	// a ring of that many tokens, not one that grows with context (SlidingWindowSpec,
	// vllm/v1/kv_cache_interface.py:840-907@v0.32.0), so perToken prices its read and its
	// ring's pages, and it adds nothing to the growing per-token figure (ringBytes).
	window int
	// inferred is the primary's inferred width whether or not it states its caches: the
	// width a scorer that states none is scaled from.
	inferred float64
}

// ringed reports whether a layer whose cache is inferred holds a sliding-window ring: its
// primary attention is a sliding window with a window stated, and the layer owns no index
// cache. A sparse or compressed layer's window bounds its read, not its cache, so only the
// swa kind is a ring; and a sliding-window layer that owns an indexer's cache is a sparse
// attention whose main cache the engine registers as full attention -- MiniMax-M3's
// (get_kv_cache_spec returns FullAttentionSpec, vllm/models/minimax_m3/nvidia/model.py:
// 562-571@v0.32.0), the rule blis-vllm's KV layer mapping applies too.
func ringed(l price.PlannedLayer) bool {
	if l.AttnKind != model.AttentionSWA || l.AttnWindow <= 0 {
		return false
	}
	for _, c := range layerCaches(l) {
		if c.Kind == model.CacheIndex && c.Owned {
			return false
		}
	}
	return true
}

// cacheTerm is one cache some layers own, with its paging resolved.
type cacheTerm struct {
	cache model.Cache
	// entryBytes is the record under the cache dtype that runs, and pageBytes one page of
	// it: block tokens hold block / tokens_per_entry records, rounded up to the page
	// alignment.
	entryBytes int
	block      int
	pageBytes  float64
	// count is how many layers own it.
	count int
	// draft is a speculator layer's cache, resident only while the deployment speculates.
	draft bool
}

// layerCaches returns every cache a layer's attention kernels state, owned or read.
func layerCaches(l price.PlannedLayer) []price.PlannedCache {
	var out []price.PlannedCache
	for _, a := range l.Attentions {
		out = append(out, a.Caches...)
	}
	return out
}

// cacheTerms resolves the caches a layer owns, count times over.
func (k *Kernel) cacheTerms(stated []price.PlannedCache, count int) []cacheTerm {
	var out []cacheTerm
	for _, c := range stated {
		if !c.Owned {
			// A reader adds read traffic and nothing resident: the cache is its owner's.
			continue
		}
		block := c.BlockSize
		if block < 1 {
			block = max(k.blockSize, 1)
		}
		entries := (block + c.EntryTokens() - 1) / c.EntryTokens()
		page := float64(entries * c.EntryBytes)
		if a := c.PageAlignment; a > 0 {
			page = math.Ceil(page/float64(a)) * float64(a)
		}
		out = append(out, cacheTerm{cache: c.Cache, entryBytes: c.EntryBytes, block: block,
			pageBytes: page, count: count})
	}
	return out
}

// checkCaches refuses a window ring the engine rules size at no pages, and records the two
// facts a stated cache is sized on that the deployment does not settle: which record runs,
// where the graph lists none for the cache dtype, and that a window ring stays bounded,
// which holds only under the hybrid KV cache manager.
func (k *Kernel) checkCaches() error {
	window, unlisted := false, false
	for _, t := range k.caches {
		if t.cache.Kind == model.CacheWindow {
			window = true
			if k.rules.WindowPagesPerRequest(t.cache, k.blockSize, 1, 0, k.maxModelLen) <= 0 {
				return fmt.Errorf("the window cache %q sizes at no pages under these engine "+
					"rules, so its ring would be free", t.cache.Name)
			}
		}
		if _, ok := t.cache.BytesByCacheDType[k.cacheDType]; !ok {
			unlisted = true
		}
	}
	// A record not listed under the running dtype is the base one, which the base
	// description assumes for the engine's own default (blis-schemas model.Cache): so it is
	// an assumption only where the deployment runs a dtype other than the default the
	// release resolves for this architecture.
	base := k.rules.ResolveCacheDType(k.architecture, k.family, k.layout.TP, "")
	if unlisted && len(k.caches) > 0 && k.cacheDType != base {
		dtype := k.cacheDType
		if dtype == "" {
			dtype = "auto"
		}
		k.assume("cache_record", dtype,
			"the graph lists no record under this cache dtype for some cache it states, so "+
				"that cache is sized at its bytes_per_entry, the record under the cache dtype "+
				"the graph's base description assumes")
	}
	if window && k.kvConnector {
		k.assume("hybrid_kv_cache_manager", "on",
			"window rings are sized as bounded, which holds only under the hybrid KV cache "+
				"manager; a KV connector that does not support it turns them into "+
				"full-length caches (vllm/config/vllm.py), which the kernel cannot see")
	}
	return k.resolveKVPool()
}

// kvPool is the engine's shared KV block pool, as the stated caches are drawn from it.
//
// vLLM v0.31.0 draws every KV cache group from one pool whose block costs the widest group's
// page bytes, every group's block alike ("Groups alias from byte 0",
// get_kv_cache_config_from_groups, vllm/v1/core/kv_cache_utils.py:1768-1785;
// _get_kv_cache_bytes_per_block, :1606-1632), and packs layers of mixed page sizes into
// groups (_get_packed_kv_cache_groups, :2001-2120). A request holds whole blocks in each
// group, and the sum over groups is what it draws from the pool
// (get_max_concurrency_for_kv_cache_config, :1090-1122). Sizing each cache at its own page
// understates every group narrower than the widest: DeepSeek-V4.1-Flash on H200 holds 49
// blocks of 116,928 B in steady decode, 5.73 MB, where its caches' own pages sum to 5.00 MB.
type kvPool struct {
	bytesPerBlock float64
	groups        []poolGroup
}

// poolGroup is one KV cache group: its layers' cache, as the engine sizes a request in it.
// Every layer of a group is one uniform type, so one member's cache sizes the group.
type poolGroup struct {
	cache model.Cache
	block int
	draft bool
}

// resolveKVPool groups the stated caches through the rules pack (rules.Pack.KVCacheGroups,
// blis-schemas#66), from the records and pages they are sized at here. Where the pack does
// not model the grouping, the caches stay sized at their own pages, and that is recorded.
func (k *Kernel) resolveKVPool() error {
	var layers []rules.KVCacheLayer
	term := map[string]int{}
	for i, t := range k.caches {
		if t.draft && !k.speculating {
			continue
		}
		c := t.cache
		c.BytesPerEntry, c.BytesByCacheDType, c.BlockSize = t.entryBytes, nil, t.block
		for j := 0; j < t.count; j++ {
			name := fmt.Sprintf("%d.%s.%d", i, c.Name, j)
			layers = append(layers, rules.KVCacheLayer{Name: name, Cache: c})
			term[name] = i
		}
	}
	if len(layers) == 0 {
		return nil
	}
	// No block-stride alignment: the one a V4.1 backend states is the indexer's under
	// indexer_sparse_logits, off by default (vllm/config/attention.py:89;
	// vllm/models/deepseek_v41/attention.py:1186-1191), and the deployment cannot state it.
	// The deployment's K sizes a compressor ring as the engine does, max(8, the power of
	// two above K + 1) rows (vllm/models/deepseek_v41/compressor.py:147-149), which the
	// graph's fixed entry count states only for K below 7.
	spec := k.numSpecTokens()
	g, err := k.rules.KVCacheGroups(rules.KVCacheGroupsInput{
		Layers: layers, EngineBlockSize: k.blockSize, CacheDType: k.cacheDType,
		NumSpecTokens: &spec,
	})
	if err != nil {
		k.assume("kv_cache_grouping", "per cache",
			"the stated caches are sized at their own pages rather than in the engine's "+
				"shared block pool, because the rules pack cannot group them: "+err.Error())
		return nil
	}
	pool := &kvPool{bytesPerBlock: float64(g.BytesPerBlock)}
	for _, gr := range g.Groups {
		if len(gr.Layers) == 0 {
			continue
		}
		t := k.caches[term[gr.Layers[0]]]
		pool.groups = append(pool.groups, poolGroup{cache: t.cache, block: gr.BlockSize,
			draft: t.draft})
	}
	// The pack pages each cache from the same record the kernel sizes it at, so the two must
	// agree. A state cache is the exception: its ring's rows follow the deployment's K in
	// the pack, and the pool charges it one block whatever its page.
	for name, page := range g.PageBytes {
		t := k.caches[term[name]]
		if t.cache.Kind == model.CacheState {
			continue
		}
		if want := t.pageBytes; float64(page) != want {
			return fmt.Errorf("the rules pack pages the cache %q at %d B and the kernel at "+
				"%.0f B", t.cache.Name, page, want)
		}
	}
	k.kvPool = pool
	return nil
}

// pages is how many pages a cache holds for tokens context tokens.
func (t cacheTerm) pages(tokens int) int {
	if tokens <= 0 {
		return 0
	}
	return (tokens + t.block - 1) / t.block
}

// growing reports whether the cache grows with context: a kv or index cache.
func (t cacheTerm) growing() bool {
	return t.cache.Kind == model.CacheKV || t.cache.Kind == model.CacheIndex
}

// cacheVariableBytes is what the context-growing caches hold for a sequence of tokens, after
// decode-context parallelism's shard of the token count.
func (k *Kernel) cacheVariableBytes(tokens int) float64 {
	// An entry is not divided across tensor-parallel ranks, and decode-context parallelism
	// gives each rank 1/dcp of the entries (blis-schemas model.Cache,
	// vllm/v1/kv_cache_interface.py max_num_blocks_per_req).
	local := tokens
	if k.layout.DCP > 1 {
		local = (tokens + k.layout.DCP - 1) / k.layout.DCP
	}
	if p := k.kvPool; p != nil && tokens > 0 {
		var blocks int
		for _, g := range p.groups {
			if g.cache.Kind == model.CacheKV || g.cache.Kind == model.CacheIndex {
				blocks += (local + g.block - 1) / g.block
			}
		}
		return float64(blocks) * p.bytesPerBlock
	}
	var total float64
	for _, t := range k.caches {
		if !t.growing() || (t.draft && !k.speculating) {
			continue
		}
		total += float64(t.pages(local)) * t.pageBytes * float64(t.count)
	}
	return total
}

// cacheFixedBytes is what the bounded caches hold per sequence: each window ring at the
// pages the engine lets a request hold, and each state cache's fixed entries.
//
// vLLM caps a request at cdiv(min(window - 1 + extra_retained + max_in_flight_tokens,
// max_model_len), block) + 1
// pages (max_admission_blocks_per_request, vllm/v1/kv_cache_interface.py:840-866 at
// v0.31.0), its in-flight term being the startup sizing's concurrent batches x
// max_num_batched_tokens. The kernel takes a request's steady-state decode hold instead:
// 1 + num_spec_tokens tokens in flight and no extra retention, through the rules pack's
// WindowPagesPerRequest. So DSpark's five draft tokens make a 128-token ring of 32-token
// pages six pages rather than five. The admission cap is cacheAdmissionBytes, which
// SequenceAdmissionBytes reports.
func (k *Kernel) cacheFixedBytes() float64 {
	return k.cacheBoundedBytes(1+k.numSpecTokens(), 0)
}

// cacheAdmissionBytes is what the bounded caches reserve for a request at the engine's
// admission cap: each window ring at max_admission_blocks_per_request for the startup
// sizing's max_in_flight_tokens and extra_retained_tokens (vllm/v1/kv_cache_interface.py:
// 840-866), which the runtime admission gate clamps a request's reservation to
// (vllm/v1/core/single_type_kv_cache_manager.py:216-229), and which the engine's own
// capacity figure divides by (get_kv_cache_capacity, vllm/v1/core/kv_cache_utils.py:
// 2444-2471).
func (k *Kernel) cacheAdmissionBytes() float64 {
	return k.cacheBoundedBytes(k.maxInFlightTokens(), k.extraRetainedTokens())
}

// cacheBoundedBytes is the bounded caches' bytes for a request with inFlight tokens in flight:
// each window ring at the pages the rules pack gives it, and each state cache's entries --
// one pool block a group where the caches are drawn from the shared pool, since a
// circular-buffer group holds one block a request (CircularBufferSpec.max_memory_usage_bytes,
// vllm/v1/kv_cache_interface.py:904-907).
func (k *Kernel) cacheBoundedBytes(inFlight, extraRetained int) float64 {
	return k.cacheBoundedBytesUpTo(inFlight, extraRetained, k.maxModelLen)
}

// cacheBoundedBytesUpTo is cacheBoundedBytes for a request whose window has seen only
// tokens context tokens: WindowPagesPerRequest's own min() then caps each ring at those. A
// negative count is a request that has seen none, whose rings hold no page.
func (k *Kernel) cacheBoundedBytesUpTo(inFlight, extraRetained, tokens int) float64 {
	return k.statedBoundedBytesUpTo(inFlight, extraRetained, tokens) +
		k.ringBytes(inFlight, extraRetained, tokens)
}

// ringBytes is what the rank's inferred window rings hold for a request with inFlight tokens
// in flight that has seen tokens context tokens: each ring at the pages the engine lets a
// request hold, as a stated window cache is sized (WindowPagesPerRequest), times the layer's
// per-token width. The drafter's rings count while the deployment speculates and the rank
// holds the drafter.
func (k *Kernel) ringBytes(inFlight, extraRetained, tokens int) float64 {
	if tokens < 0 {
		return 0
	}
	block := max(k.blockSize, 1)
	ring := func(layers []price.PlannedLayer, kv []layerKV) float64 {
		var total float64
		for i, l := range layers {
			if i >= len(kv) || kv[i].window <= 0 {
				continue
			}
			pages := k.rules.WindowPagesPerRequest(model.Cache{Kind: model.CacheWindow,
				Window: kv[i].window}, block, inFlight, extraRetained, tokens)
			total += float64(pages*block) * kv[i].perToken * float64(l.Count)
		}
		return total
	}
	m := k.rankMemory()
	total := ring(m.layers, m.kv)
	if m.drafter && k.speculating && k.plan.Draft != nil {
		total += ring(k.plan.Draft.Layers, k.draftLayerKV)
	}
	return total
}

// ringTransferBytes is what a P/D handoff moves of the target's inferred window rings for a
// sequence of tokens: each ring clipped to cdiv(window, block) + 1 pages, as NIXL clips a
// sliding-window group (cacheTransferBytes).
func (k *Kernel) ringTransferBytes(tokens int) float64 {
	block := max(k.blockSize, 1)
	var total float64
	for i, l := range k.plan.Layers {
		if w := k.layerKV[i].window; w > 0 {
			pages := min((tokens+block-1)/block, (w+block-1)/block+1)
			total += float64(pages*block) * k.layerKV[i].perToken * float64(l.Count)
		}
	}
	return total
}

// statedBoundedBytesUpTo is cacheBoundedBytesUpTo's share for the caches a graph states.
func (k *Kernel) statedBoundedBytesUpTo(inFlight, extraRetained, tokens int) float64 {
	if p := k.kvPool; p != nil {
		var blocks int
		for _, g := range p.groups {
			switch g.cache.Kind {
			case model.CacheWindow:
				if tokens >= 0 {
					blocks += k.rules.WindowPagesPerRequest(g.cache, k.blockSize, inFlight,
						extraRetained, tokens)
				}
			case model.CacheState:
				blocks++
			}
		}
		return float64(blocks) * p.bytesPerBlock
	}
	var total float64
	for _, t := range k.caches {
		if t.draft && !k.speculating {
			continue
		}
		switch t.cache.Kind {
		case model.CacheWindow:
			if tokens < 0 {
				continue
			}
			p := k.rules.WindowPagesPerRequest(t.cache, k.blockSize, inFlight, extraRetained,
				tokens)
			total += float64(p) * t.pageBytes * float64(t.count)
		case model.CacheState:
			total += float64(t.cache.Entries*t.entryBytes) * float64(t.count)
		}
	}
	return total
}

// maxInFlightTokens is vLLM's max_in_flight_tokens: max_concurrent_batches x
// max_num_batched_tokens, the batches being pp + 1 under asynchronous scheduling on the model
// runner V2 that CUDA defaults to, and pp without it (vllm/config/vllm.py:592-613).
//
// COVERAGE LIMIT: the V1 model runner holds pp batches under asynchronous scheduling with
// pipeline parallelism, one fewer than this; the kernel cannot see which runner the engine
// picks, and at pp 1 both hold two.
func (k *Kernel) maxInFlightTokens() int {
	batches := max(k.layout.PP, 1)
	if k.asyncScheduling {
		batches++
	}
	return batches * k.batchedTokens
}

// extraRetainedTokens is extra_retained_tokens, max(0, num_prefill_lookahead_tokens - 1)
// (vllm/v1/core/kv_cache_utils.py:2738-2744): num_speculative_tokens - 1 for multi-module MTP,
// zero for every other drafter, which reads one token ahead or none
// (vllm/config/vllm.py:657-677; use_multi_module_mtp, vllm/config/speculative.py:1973-1979).
func (k *Kernel) extraRetainedTokens() int {
	s := k.engine.Speculative
	if s == nil || s.Method != "mtp" || !k.speculating || k.plan.Draft == nil {
		return 0
	}
	modules := 0
	for _, l := range k.plan.Draft.Layers {
		modules += l.Count
	}
	if min(modules, s.NumSpecTokens) > 1 {
		return s.NumSpecTokens - 1
	}
	return 0
}

// cacheTransferBytes is what a P/D handoff moves of the stated caches for a sequence of
// tokens: the target's owned caches only, as NIXL moves them -- the growing caches in full,
// each window cache clipped to cdiv(window, block) + 1 pages
// (vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:133-146 at v0.31.0,
// which clips the SlidingWindowSpec groups), and the state caches whole, as the kernel
// takes them: they are a request's bounded state and cross with it.
// A reader's cache is its owner's and crosses once; a speculator's caches cross with the
// target's while the deployment speculates -- they share its KV cache groups, and every
// group is transferable unless marked otherwise (KVCacheGroupSpec.enable_kv_transfer,
// vllm/v1/kv_cache_interface.py:1461-1464, 1500-1507).
func (k *Kernel) cacheTransferBytes(tokens int) float64 {
	var total float64
	for _, t := range k.caches {
		if t.draft && !k.speculating {
			continue
		}
		var pages int
		switch t.cache.Kind {
		case model.CacheKV, model.CacheIndex:
			pages = t.pages(tokens)
		case model.CacheWindow:
			pages = min(t.pages(tokens), (t.cache.Window+t.block-1)/t.block+1)
		case model.CacheState:
			total += float64(t.cache.Entries*t.entryBytes) * float64(t.count)
			continue
		}
		total += float64(pages) * t.pageBytes * float64(t.count)
	}
	return total
}

// statedReadBytes is what one decode request at context ctx reads from a layer's stated
// caches through attention a: each window cache's most recent min(ctx, window) records,
// and each growing cache's records -- ceil(ctx / tokens_per_entry) of them, capped at the
// top-k the layer selects where it selects one. Only the caches the kernel reads count: the
// indexer reads the index cache, attention the others.
//
// The shard of decode-context parallelism applies to the growing part only, the part
// stored by token position -- the same caches cacheVariableBytes shards.
//
// window is the node's own read window where it states one: a parallel-block draft reads its
// ring plus its block positions (DSpark's 133 over a 128-token ring), which the ring holds
// while they are in flight.
func (k *Kernel) statedReadBytes(caches []price.PlannedCache, topk, window, ctx int) float64 {
	var total float64
	for _, c := range caches {
		switch c.Kind {
		case model.CacheWindow:
			w := ctx
			if bound := max(c.Window, window); bound > 0 {
				w = min(ctx, bound)
			}
			total += float64(w) * float64(c.EntryBytes)
		case model.CacheKV, model.CacheIndex:
			entries := (ctx + c.EntryTokens() - 1) / c.EntryTokens()
			if topk > 0 {
				entries = min(entries, topk)
			}
			if k.layout.DCP > 1 {
				entries = int(dcpLocalTokens([]int{entries}, k.layout.DCP,
					k.decodeContext.Interleave))
			}
			total += float64(entries) * float64(c.EntryBytes)
		}
	}
	return total
}

// numSpecTokens is the draft tokens per request per step the deployment runs, zero when it
// does not speculate.
func (k *Kernel) numSpecTokens() int {
	if s := k.engine.Speculative; s != nil && s.NumSpecTokens > 0 {
		return s.NumSpecTokens
	}
	return 0
}
