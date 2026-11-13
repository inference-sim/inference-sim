// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// _approximate_gcd (:1825-1857) and _get_packed_kv_cache_groups (:1860-2014), with the page
// sizes of vllm/v1/kv_cache_interface.py's AttentionSpec (:511-571), MLAAttentionSpec
// (:686-760), SlidingWindowMLASpec (:949-1033) and CircularBufferSpec (:909-946).

package kv

import "errors"

// pageBytes is a layer's page_size_bytes: heads x states per block x bytes per state, padded up
// to the spec's alignment (AttentionSpec.page_size_bytes, kv_cache_interface.py:546-563;
// _apply_alignment_padding, :670-676@5df63a72). A state covers TokensPerState tokens
// (get_num_kernel_states, :231-234). StateBytes is the state's content (state_content_bytes),
// or keys and values of HeadSize elements of ElemBytes each, keys only for MLA, an indexer and
// a ring (head_size_v 0, :700, :1003).
func pageBytes(l Layer) int {
	if l.Spec == LayerMamba {
		// MambaSpec.page_size_bytes: the states' bytes, or page_size_padded when set
		// (kv_cache_interface.py:1067-1075@5df63a72).
		return max(l.StateBytes, l.PageSizePadded)
	}
	states := l.BlockSize / max(l.TokensPerState, 1)
	content := l.StateBytes
	if content == 0 {
		v := l.HeadSize
		if l.Spec == LayerIndexer || l.Spec == LayerMLA || l.Spec == LayerSlidingWindowMLA || l.Spec == LayerCircularBuffer {
			v = 0
		}
		content = (l.HeadSize + v) * max(l.ElemBytes, 1)
	}
	p := max(l.KVHeads, 1) * states * content
	if l.Alignment > 0 {
		p = (p + l.Alignment - 1) / l.Alignment * l.Alignment
	}
	return p
}

// specType is the uniform type a layer's spec belongs to: the bucket key of
// UniformTypeKVCacheSpecs.is_uniform_type (kv_cache_interface.py:1282-1295@5df63a72) with the
// registry's base specs (single_type_kv_cache_manager.py:2633-2710@5df63a72). Full attention
// takes MLA and the indexer (MLAAttentionSpec registers with FullAttentionSpec as its base); a
// sliding window and a sliding-window MLA are each their own base, split by window size and
// replay; every circular buffer is one type. Block size and alignment are part of the key, as
// is_uniform_type refuses layers of two block sizes.
type specType struct {
	kind      GroupKind
	window    int
	replay    bool
	blockSize int
	// mamba is a mamba layer's uniform-type key (MambaSpec.is_uniform_with_collection).
	mamba mambaKey
}

type mambaKey struct {
	mode                     string
	speculative, ckpt, align int
	page                     int
}

// mambaMode is a Mamba layer's mamba_cache_mode: "none" when it states none (the CacheConfig
// default, vllm/config/cache.py).
func mambaMode(mode string) string {
	if mode == "" {
		return "none"
	}
	return mode
}

// pageAt is pageBytes at the engine's block size, for a layer that states none.
func pageAt(l Layer, engine int) int {
	if l.BlockSize == 0 {
		l.BlockSize = engine
	}
	return pageBytes(l)
}

func typeOf(l Layer) specType {
	switch l.Spec {
	case LayerSlidingWindow:
		return specType{kind: SlidingWindow, window: l.Window, blockSize: l.BlockSize}
	case LayerSlidingWindowMLA:
		return specType{kind: SlidingWindowMLA, window: l.Window, replay: l.BoundedReplay, blockSize: l.BlockSize}
	case LayerCircularBuffer:
		return specType{kind: CircularBuffer, blockSize: l.BlockSize}
	case LayerMamba:
		return specType{kind: Mamba, blockSize: l.BlockSize, mamba: mambaKey{mambaMode(l.MambaCacheMode),
			l.NumSpeculativeBlocks, l.NumPrefillCheckpointBlocks, l.PrefillCheckpointAlignment, pageBytes(l)}}
	}
	return specType{kind: FullAttention, blockSize: l.BlockSize}
}

// group is the KV cache group a spec type makes, at its block size.
func (t specType) group() Group {
	return Group{Kind: t.kind, Window: t.window, BlockSize: t.blockSize, BoundedReplay: t.replay,
		MambaCacheMode: t.mamba.mode, MambaSpeculativeBlocks: t.mamba.speculative,
		MambaCheckpointBlocks: t.mamba.ckpt, MambaCheckpointAlignment: t.mamba.align}
}

// approximateGCD is _approximate_gcd (kv_cache_utils.py:1825-1857@5df63a72): the d in
// [lowerBound, max(values)] that pads the values up to multiples of d the least, the larger d
// on a tie.
func approximateGCD(values []int, lowerBound int) int {
	minD, maxD := max(1, lowerBound), 0
	for _, x := range values {
		maxD = max(maxD, x)
	}
	if minD > maxD {
		return minD
	}
	bestD, bestPad := minD, -1
	for d := minD; d <= maxD; d++ {
		pad := 0
		for _, x := range values {
			pad += (d - x%d) % d
		}
		if bestPad < 0 || pad < bestPad || pad == bestPad && d > bestD {
			bestD, bestPad = d, pad
		}
	}
	return bestD
}

// packedBucket is one uniform-type bucket of _get_packed_kv_cache_groups: its layers by page
// size, each page size's layers in model order, page sizes in order of first appearance.
type packedBucket struct {
	typ      specType
	n        int
	pages    []int
	layers   map[int][]int
	balanced bool
}

func (b packedBucket) maxLayersPerPage() int {
	most := 0
	for _, p := range b.pages {
		most = max(most, len(b.layers[p]))
	}
	return most
}

// isState is is_state_bucket (kv_cache_utils.py:1941-1946@5df63a72): a Mamba layer or a circular
// buffer always, a sliding window when nothing repeats.
func (b packedBucket) isState(repeating bool) bool {
	if b.typ.kind == CircularBuffer || b.typ.kind == Mamba {
		return true
	}
	return !repeating && (b.typ.kind == SlidingWindow || b.typ.kind == SlidingWindowMLA)
}

// widestGroupBytes is widest_group_bytes (:1931-1936): the page bytes of the largest of the n
// groups a bucket splits into.
func (b packedBucket) widestGroupBytes(n int) int {
	total := 0
	for _, p := range b.pages {
		total += (len(b.layers[p]) + n - 1) / n * p
	}
	return total
}

// packedGroups is _get_packed_kv_cache_groups (kv_cache_utils.py:1860-2014@5df63a72) for the
// block-outermost layout, which the DeepSeek V4 indexer backends declare
// (DeepseekV4IndexerBackend.supported_kv_cache_layouts, vllm/v1/attention/backends/mla/
// indexer.py:347-351@5df63a72): the layers are bucketed by uniform type in order of first
// appearance; a bucket with one layer of each of its page sizes per repeat is balanced, and the
// balanced buckets mixing page sizes set a floor on the repeats per group, which
// _approximate_gcd picks over every balanced bucket; each balanced bucket then splits into
// cdiv(its repeats, that) groups, and a state bucket is split until a group fits the block the
// others need. It returns nil when every layer has one page size, which vLLM groups another
// way. Each group is one spec type at its own block size; members are each group's layers.
//
// A Mamba bucket is a state bucket like a circular buffer's: it may split down to one layer a
// group, so its widest group is sized at one layer, and it is then split no finer than the
// states the anchor block already holds (is_state_bucket :1941-1946, the anchor :1948-1960, the cap :1965-1970). engine is the engine's block
// size, which a layer that states none has; an attention layer that states none with engine 0
// has no page to pack, and is refused.
func packedGroups(layers []Layer, engine int) (groups []Group, members [][]int, err error) {
	pageSet := map[int]bool{}
	for _, l := range layers {
		if l.Spec != LayerMamba && l.BlockSize == 0 && engine <= 0 {
			return nil, nil, errors.New("kv: packing KV cache groups of unequal page sizes needs the engine's block size " +
				"for a layer that states none (GroupOptions.BlockSize)")
		}
		pageSet[pageAt(l, engine)] = true
	}
	if len(pageSet) <= 1 {
		return nil, nil, nil
	}
	var buckets []packedBucket
	index := map[specType]int{}
	for i, l := range layers {
		t := typeOf(l)
		bi, ok := index[t]
		if !ok {
			bi = len(buckets)
			index[t] = bi
			buckets = append(buckets, packedBucket{typ: t, layers: map[int][]int{}})
		}
		b := &buckets[bi]
		p := pageAt(l, engine)
		if _, seen := b.layers[p]; !seen {
			b.pages = append(b.pages, p)
		}
		b.layers[p] = append(b.layers[p], i)
		b.n++
	}
	minRepeats := 0
	var balancedCounts []int
	for i := range buckets {
		b := &buckets[i]
		counts := map[int]bool{}
		for _, p := range b.pages {
			counts[len(b.layers[p])] = true
		}
		b.balanced = len(counts) == 1
		if b.balanced {
			balancedCounts = append(balancedCounts, b.maxLayersPerPage())
			if len(b.pages) > 1 {
				minRepeats = max(minRepeats, b.maxLayersPerPage())
			}
		}
	}
	repeats := 0 // repeats_per_group; 0 is None
	if minRepeats > 0 {
		repeats = approximateGCD(balancedCounts, minRepeats)
	}
	numGroupsFor := func(b packedBucket) int {
		if b.balanced && repeats > 0 {
			return (b.maxLayersPerPage() + repeats - 1) / repeats
		}
		return 1
	}
	anchor := 0
	for _, b := range buckets {
		n := numGroupsFor(b)
		if b.isState(repeats > 0) {
			n = b.n
		}
		anchor = max(anchor, b.widestGroupBytes(n))
	}
	for _, b := range buckets {
		n := numGroupsFor(b)
		if anchor > 0 && b.isState(repeats > 0) {
			perBlock := max(anchor/b.pages[0], 1)
			n = max(n, (b.n+perBlock-1)/perBlock)
		}
		if n == 1 {
			var all []int
			for i, l := range layers {
				if typeOf(l) == b.typ {
					all = append(all, i)
				}
			}
			groups, members = append(groups, b.typ.group()), append(members, all)
			continue
		}
		// pattern_repeats: one layer of each page size per repeat (zip over the page sizes),
		// and group i takes repeats i, i+n, ... (:1975-1983).
		repeatsOf := len(b.layers[b.pages[0]])
		for _, p := range b.pages {
			repeatsOf = min(repeatsOf, len(b.layers[p]))
		}
		for i := range n {
			var m []int
			for r := i; r < repeatsOf; r += n {
				for _, p := range b.pages {
					m = append(m, b.layers[p][r])
				}
			}
			groups, members = append(groups, b.typ.group()), append(members, m)
		}
	}
	return groups, members, nil
}
