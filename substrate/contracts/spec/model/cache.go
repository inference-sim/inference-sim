package model

import (
	"fmt"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Cache is one per-request cache an attention layer reads, with enough of its layout to
// size it: who writes it, how many bytes one stored entry takes, how many tokens one
// entry covers, and whether it is bounded by a window or a fixed entry count.
//
// It exists because an attention node's read shape (heads, window, compress_ratio,
// index_topk) does not say what is RESIDENT. A consumer that had only the read shape had
// to treat every attention layer as owning one full-length cache of the read width. For
// DeepSeek-V4.1-Flash on the engine that is 40 x 584 = 23,360 B per context token,
// against a real 1,790 B: only layers 2, 8, 14 and 20 own a compressed cache, a
// compressed cache holds one entry per compress_ratio tokens, and the 40 sliding-window
// caches are bounded rings rather than one entry per context token.
//
// A node with no caches keeps the meaning it always had: the consumer infers one cache
// from the read shape. A node that states caches is sized from them alone, so a graph
// opts in per node and no committed graph changes meaning.
type Cache struct {
	// Name identifies the cache within its layer, and pairs a reader with its writer: a
	// cache this layer does not own is the same-named cache of another layer.
	Name string `yaml:"name"`

	// Kind is what the cache holds. See CacheKind.
	Kind CacheKind `yaml:"kind"`

	// Owned says whether this layer allocates the cache. A layer that does not own one
	// reads the same-named cache of the NEAREST PRECEDING layer of its stack that owns
	// it. That is DeepSeek-V4.1's own rule for its compressed cache, whose readers bind
	// max(s for s in kv_source_layer_ids if s <= layer_id) (the engine's
	// models/deepseek_v41/attention.py). The engine binds its indexer keys to that same
	// kv source and refuses a source owning none, where this rule would look further
	// back; on V4.1's configuration the two coincide. A reference by rule rather than by
	// layer index is what lets one layer kind repeat across several source spans: a kind
	// used at layers 3-7 and again at 9-13 reads layer 2's cache in the first span and
	// layer 8's in the second, which no single stated index could say.
	//
	// Only an owned cache adds residency, PD-transfer volume and offload volume; a read
	// one adds none. It is required in a document (see UnmarshalYAML), because an omitted
	// bool would read as a reader.
	Owned bool `yaml:"owned"`

	// BytesPerEntry is what one stored entry occupies, as stored: scales and padding
	// within the record included. It is stated rather than computed from d_h and a dtype
	// because records are packed formats -- V4.1's 584 B record on SM90 is 448 FP8 NoPE
	// values, 64 BF16 RoPE values, 7 UE8M0 scales and a pad byte (attention.py, the
	// comment on _use_v41_mxfp8_kv_record). On a cache the layer does not own it is the
	// entry the layer reads, which must equal its owner's, as must every field but the
	// owner's paging (BlockSize, PageAlignment), which a reader does not state.
	// It is the whole layer's entry: an entry is not divided across tensor-parallel ranks
	// (a latent cache has one head, and the engine's window spec states max_tp_shards=1), and
	// where the engine shards a cache by decode-context parallelism each rank holds 1/dcp
	// of the entries (v1/kv_cache_interface.py max_num_blocks_per_req). A page of
	// block_size tokens holds block_size / tokens_per_entry entries.
	//
	// It is the record under the cache dtype the graph's base description assumes, which
	// for V4.1 on SM90 is fp8_ds_mla. Which dtype actually runs -- including an attention
	// backend's own resolution of auto, such as SM100 mega attention resolving it to
	// nvfp4_ds_mla wherever that backend is available at the layout -- is the consumer's
	// to resolve, and EntryBytes gives the record for the one it resolves.
	BytesPerEntry int `yaml:"bytes_per_entry"`
	// BytesByCacheDType is the entry size under cache dtypes whose record differs from
	// BytesPerEntry, keyed by the engine's cache-dtype name. V4.1's compressed cache is a
	// 288 B NVFP4 record under nvfp4_ds_mla while its window cache keeps the MXFP8 one
	// (attention.py: "nvfp4_ds_mla keeps the MXFP8 sliding-window record and stores the
	// compressed cache as NVFP4"). Which cache dtype runs is the consumer's to resolve from
	// the deployment and the engine; a dtype not listed takes BytesPerEntry.
	BytesByCacheDType map[string]int `yaml:"bytes_by_cache_dtype,omitempty"`

	// TokensPerEntry is how many context tokens one entry covers: a compressed stream at
	// ratio r holds one entry per r tokens (MLAAttentionSpec tokens_per_state). Zero means
	// one.
	TokensPerEntry int `yaml:"tokens_per_entry,omitempty"`

	// Window bounds a kind: window cache to the most recent Window tokens. The engine caps a
	// request at cdiv(min(Window - 1 + extra retained + tokens in flight, max_model_len),
	// block) + 1 pages of it (the engine's v1/kv_cache_interface.py
	// max_admission_blocks_per_request), and frees pages that fall out of the window, so
	// its residency does not grow with context. The engine reserves for tokens in flight
	// at its max_in_flight_tokens (concurrent batches x max_num_batched_tokens); a request
	// holds its own in-flight tokens, 1 + num_spec_tokens per decode step. Pack method
	// WindowPagesPerRequest applies the formula.
	//
	// The window is bounded only under the hybrid KV cache manager. With it off -- which
	// the engine does on its own for a KV connector that does not support it
	// (the engine's top-level configuration) -- unify_hybrid_kv_cache_specs turns every sliding-window spec
	// into a full-attention one (v1/core/kv_cache_utils.py), and the cache grows with
	// context like a kv cache.
	Window int `yaml:"window,omitempty"`

	// Entries is the fixed entry count of a kind: state cache, per request for the
	// request's whole lifetime. V4.1's compressor ring is one page of 8 FP32 rows of
	// 4,096 B on each ratio-2 source layer. The count follows the engine's own rule at
	// the speculator's num_spec_tokens (compressor.py: the smallest power of two of at
	// least max(8, num_spec_tokens + 2)), so a deployment drafting more tokens than the
	// graph's speculator can need a larger ring.
	Entries int `yaml:"entries,omitempty"`

	// BlockSize is the cache's page size in tokens where the model fixes it rather than
	// taking the engine's block_size -- V4.1's sliding-window cache is always 32-token
	// pages (attention.py:547-558). Zero means the engine's block_size.
	BlockSize int `yaml:"block_size,omitempty"`

	// PageAlignment rounds each page's byte size up to a multiple of itself, where the
	// layout requires it: V4.1's records pad each page to the decode kernel's 576 B TMA
	// stride on SM90 (attention.py:512), so a 32-token page of 584 B entries occupies
	// 19,008 B rather than 18,688. Zero means no padding.
	PageAlignment int `yaml:"page_alignment,omitempty"`
}

// CacheKind is what a cache holds, which decides how it grows.
type CacheKind string

const (
	// CacheKV is a cache of the layer's keys and values (or latent), one entry per
	// TokensPerEntry context tokens, growing with context.
	CacheKV CacheKind = "kv"
	// CacheIndex is a sparse-attention indexer's key cache, growing with context like
	// CacheKV but read by the index scorer rather than by attention itself.
	CacheIndex CacheKind = "index"
	// CacheWindow is a sliding-window cache: entries for the most recent Window tokens
	// only, bounded per request.
	CacheWindow CacheKind = "window"
	// CacheState is fixed-size per-request state that is neither keys nor values, such
	// as a compressor's ring of not-yet-pooled rows: Entries entries for the request's
	// lifetime, independent of context.
	CacheState CacheKind = "state"
)

var cacheKinds = map[CacheKind]bool{
	CacheKV: true, CacheIndex: true, CacheWindow: true, CacheState: true,
}

// Valid reports whether k is a recognized cache kind.
func (k CacheKind) Valid() bool { return cacheKinds[k] }

// EntryTokens returns TokensPerEntry with its default of one applied.
func (c Cache) EntryTokens() int {
	if c.TokensPerEntry < 1 {
		return 1
	}
	return c.TokensPerEntry
}

// BytesPerContextToken is what an owned cache adds per context token under the named
// cache dtype, before paging: the entry size over the tokens an entry covers. It is zero
// for a cache the layer reads but does not own, and for a window or state cache, whose
// size does not grow with context.
func (c Cache) BytesPerContextToken(cacheDType string) float64 {
	if !c.Owned || (c.Kind != CacheKV && c.Kind != CacheIndex) {
		return 0
	}
	return float64(c.EntryBytes(cacheDType)) / float64(c.EntryTokens())
}

// EntryBytes returns the entry size under the named cache dtype.
func (c Cache) EntryBytes(cacheDType string) int {
	if b, ok := c.BytesByCacheDType[cacheDType]; ok {
		return b
	}
	return c.BytesPerEntry
}

// UnmarshalYAML requires the owned key. An omitted bool reads as false, which would make
// a forgotten key turn an owner into a reader: the cache would validate and its residency
// would silently vanish. Every cache must therefore say which it is.
func (c *Cache) UnmarshalYAML(node *yaml.Node) error {
	if err := validate.RejectUnknownKeys(node, Cache{}, nil); err != nil {
		return fmt.Errorf("line %d: cache: %w", node.Line, err)
	}
	if node.Kind == yaml.MappingNode {
		stated := false
		for i := 0; i+1 < len(node.Content); i += 2 {
			if node.Content[i].Value == "owned" {
				stated = true
			}
		}
		if !stated {
			return fmt.Errorf("line %d: a cache must state owned: true or owned: false", node.Line)
		}
	}
	type shape Cache
	var raw shape
	if err := node.Decode(&raw); err != nil {
		return err
	}
	*c = Cache(raw)
	return nil
}
