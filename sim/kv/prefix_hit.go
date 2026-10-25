package kv

import (
	hash "github.com/inference-sim/blis-vllm/compat/prefix"
	sim "github.com/inference-sim/inference-sim/sim/engine"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// This file reconciles a full-attention prefix hit with the model's other KV cache groups,
// as vLLM v0.31.0's HybridKVCacheCoordinator.find_longest_cache_hit does
// (vllm/v1/core/kv_cache_coordinator.py:842-977), for the layouts sim.KVPrefixHit states.
//
// A prefix-cacheable sliding-window group can resume a hit only at a token boundary whose
// window tail it still holds. Under prefix_cache_retention_interval 0, the default
// (vllm/config/cache.py:160), a request caches window pages only in the tails ending at its
// replay boundaries and at a shared-prefix junction (SlidingWindowManager.reachable_block_mask,
// vllm/v1/core/single_type_kv_cache_manager.py:1072-1151). So the store keeps, per retained
// boundary, a key naming the token prefix the tail covers: the full-attention chain up to the
// boundary plus, for an EAGLE window group, the window page past it that the lookup peeks at
// and drops (single_type_kv_cache_manager.py:960-1069). A tail lives as long as the
// full-attention block ending at its boundary keeps its hash; vLLM evicts window pages on
// their own LRU position, which this ties to that block's.

// pendingTails are the boundaries a request retains window tails at, registered once its
// prefill has computed through each tail.
type pendingTails struct {
	from       int64 // the request's own hit: a tail below it lies in null pages
	boundaries []int64
}

// ReconcilePrefixHit implements sim.PrefixHitReconciler: it cuts cached, the request's
// full-attention prefix hit already capped below its last prompt token, to the hit every
// KV cache group agrees on, and records the window tails the request will retain.
func (kvc *KVCacheState) ReconcilePrefixHit(req *sim.Request, cached []int64) []int64 {
	p := kvc.PrefixHit
	bs := kvc.BlockSizeTokens
	if p.FullEagle && len(cached) > 0 {
		cached = cached[:len(cached)-1]
	}
	if p.Window > 0 {
		full := util.Len64(cached)
		keep := kvc.longestRetainedBoundary(req, cached)
		cached = cached[:keep]
		var junction []int64
		// The longest single-group hit is pinned as a shared-prefix junction
		// (kv_cache_manager.py:312-318; kv_cache_coordinator.py:960-966).
		if full > keep {
			junction = []int64{full * bs}
		}
		kvc.recordPendingTails(req, util.Len64(cached)*bs, junction)
	}
	if p.ReplayTokens > 0 && util.Len64(cached)*bs <= p.ReplayTokens {
		// A hit no longer than the replayed window saves nothing (scheduler.py:1016-1026).
		cached = nil
	}
	return cached
}

// PrefixReplayTokens implements sim.PrefixHitReconciler.
func (kvc *KVCacheState) PrefixReplayTokens() int64 { return kvc.PrefixHit.ReplayTokens }

// windowPeek is the window tokens past a boundary an EAGLE window lookup must match.
func (kvc *KVCacheState) windowPeek() int64 {
	if kvc.PrefixHit.WindowEagle {
		return kvc.PrefixHit.WindowBlock
	}
	return 0
}

// tailKey names the token prefix a window tail at a boundary covers: the full-attention
// chain hash up to the boundary and the peeked tokens past it.
func tailKey(chain string, peek []sim.TokenID) string {
	if len(peek) == 0 {
		return chain
	}
	return hash.HashBlock(chain, peek)
}

// longestRetainedBoundary is the most blocks of cached a window lookup accepts: the highest
// boundary k*block whose tail is retained, within max_length min(hit + peek, prompt - 1)
// (kv_cache_coordinator.py:905-916). A hit inside the first window without a retained tail
// is not modelled (at most one block, single_type_kv_cache_manager.py:1045-1055).
func (kvc *KVCacheState) longestRetainedBoundary(req *sim.Request, cached []int64) int64 {
	if len(kvc.tails) == 0 {
		return 0
	}
	bs, peek := kvc.BlockSizeTokens, kvc.windowPeek()
	tokens := req.FullInputTokens()
	maxLen := req.InputLen() - 1
	for k := util.Len64(cached); k >= 1; k-- {
		a := k * bs
		if a+peek > maxLen || len(kvc.tailsAt[cached[k-1]]) == 0 {
			continue
		}
		if _, ok := kvc.tails[tailKey(kvc.Blocks[cached[k-1]].Hash, tokens[a:a+peek])]; ok {
			return k
		}
	}
	return 0
}

// recordPendingTails sets the boundaries the request retains window tails at: its replay
// boundaries (kv_cache_coordinator.py:316-338) and any junction.
func (kvc *KVCacheState) recordPendingTails(req *sim.Request, hit int64, junction []int64) {
	bs, n := kvc.BlockSizeTokens, req.InputLen()
	var bounds []int64
	if kvc.PrefixHit.EagleBoundaries {
		bounds = []int64{max((n-1)/bs*bs-bs, 0), max(n/bs*bs-bs, 0)}
	} else {
		bounds = []int64{max(n-1, 0) / bs * bs}
	}
	bounds = append(bounds, junction...)
	pt := pendingTails{from: hit}
	for _, b := range bounds {
		if b > 0 && b >= hit && !containsInt64(pt.boundaries, b) {
			pt.boundaries = append(pt.boundaries, b)
		}
	}
	if kvc.pending == nil {
		kvc.pending = make(map[string]pendingTails)
	}
	kvc.pending[req.ID] = pt
}

// registerTails retains the window tails of the request's pending boundaries that its
// computed tokens [0, computed) now cover.
func (kvc *KVCacheState) registerTails(req *sim.Request, computed int64) {
	pt, ok := kvc.pending[req.ID]
	if !ok || kvc.remoteKV[req.ID] {
		return
	}
	bs, peek := kvc.BlockSizeTokens, kvc.windowPeek()
	ids := kvc.RequestMap[req.ID]
	tokens := req.FullInputTokens()
	left := pt.boundaries[:0]
	for _, a := range pt.boundaries {
		end := a + peek
		if end > computed || end > util.Len64(tokens) || a/bs > util.Len64(ids) {
			left = append(left, a)
			continue
		}
		anchor := kvc.Blocks[ids[a/bs-1]]
		if anchor.Hash == "" {
			continue
		}
		key := tailKey(anchor.Hash, tokens[a:end])
		if kvc.tails == nil {
			kvc.tails = make(map[string]int64)
			kvc.tailsAt = make(map[int64][]string)
		}
		if _, dup := kvc.tails[key]; !dup {
			kvc.tails[key] = anchor.ID
			kvc.tailsAt[anchor.ID] = append(kvc.tailsAt[anchor.ID], key)
		}
	}
	if len(left) == 0 {
		delete(kvc.pending, req.ID)
		return
	}
	pt.boundaries = left
	kvc.pending[req.ID] = pt
}

// dropTails forgets the window tails anchored at a block whose hash is being dropped.
func (kvc *KVCacheState) dropTails(blockID int64) {
	for _, key := range kvc.tailsAt[blockID] {
		if kvc.tails[key] == blockID {
			delete(kvc.tails, key)
		}
	}
	delete(kvc.tailsAt, blockID)
}

func containsInt64(xs []int64, x int64) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}
