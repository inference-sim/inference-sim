// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/unified_cache/components/mamba.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/mem_cache/base_prefix_cache.py (InsertParams, InsertResult, EvictParams).

package cache

import (
	"errors"
	"fmt"
)

// MambaInsert is the Mamba fields of InsertParams (base_prefix_cache.py:83-85@297e54fd): the slot
// the request donates to the tree, and the ping-pong entry prepare picked.
type MambaInsert struct {
	// Value is mamba_value: the one donated slot, or nil.
	Value []int64
	// KeepIdx is mamba_keep_idx: the ping-pong entry a finished request's donation came from.
	KeepIdx *int
}

// MambaInsertResult is the Mamba field of InsertResult (base_prefix_cache.py:124@297e54fd):
// whether the tree already held a state at the insert's end, so the donated one went unused.
type MambaInsertResult struct{ Exist bool }

// MambaEvictor is the part of the unified radix cache MambaComponent calls to make room in the
// mamba pool: evict_for_alloc with EvictParams(num_tokens=0, mamba_num=n) (mamba.py:545).
type MambaEvictor interface {
	EvictForAlloc(mambaNum int)
}

// ErrMambaAllocFailed is the assertion _alloc_mamba_slot raises when eviction freed no slot
// (mamba.py:546).
var ErrMambaAllocFailed = errors.New("can not alloc mamba cache")

// MambaLifecycle is the request-side half of MambaComponent (components/mamba.py:527-727@297e54fd):
// how a request's mamba state is chosen for the tree when its KV is cached, and what is freed
// afterwards. It works on the hybrid request pool and a stand-in for the cache's eviction; the
// tree-side half (nodes, locks, matching, eviction order) is a later stage (inference-sim/inference-sim#2442).
type MambaLifecycle struct {
	Pool        *HybridRequestPool
	ExtraBuffer bool
	Evictor     MambaEvictor
}

// MambaLifecycleParams are the settings a MambaLifecycle is built from. Int8Checkpoint is a pool
// with an int8 checkpoint pool (mamba_ckpt_pool: SGLang commits and frees the donated state
// through it) and ReplaySSM a mamba pool with a ReplaySSM write cursor (replayssm_write_pos: SGLang's
// no_buffer finished path caps the donated length by the cursor and resets it); neither is modelled.
type MambaLifecycleParams struct {
	Pool           *HybridRequestPool
	ExtraBuffer    bool
	Evictor        MambaEvictor
	Int8Checkpoint bool
	ReplaySSM      bool
}

// NewMambaLifecycle builds a MambaLifecycle, refusing by name a pool whose lifecycle it does not
// port.
func NewMambaLifecycle(p MambaLifecycleParams) (*MambaLifecycle, error) {
	switch {
	case p.Int8Checkpoint:
		return nil, fmt.Errorf("cache: the int8 mamba checkpoint pool is not modelled; refused until it is (inference-sim/inference-sim#2442)")
	case p.ReplaySSM:
		return nil, fmt.Errorf("cache: ReplaySSM's write cursor (replayssm_write_pos), which caps a finished request's donated length, is not modelled; refused until it is (inference-sim/inference-sim#2442)")
	case p.Pool == nil || p.Evictor == nil:
		return nil, fmt.Errorf("cache: a MambaLifecycle needs its hybrid request pool and an evictor")
	}
	return &MambaLifecycle{Pool: p.Pool, ExtraBuffer: p.ExtraBuffer, Evictor: p.Evictor}, nil
}

// allocMambaSlot is _alloc_mamba_slot (:527-554): one slot of the mamba allocator, evicting for it
// when the allocator is empty. The extra_buffer_lazy branch that first drops the request's own
// COW-source lock needs the tree's locks; a request holding one is refused.
func (m *MambaLifecycle) allocMambaSlot(req *Req) (int64, error) {
	if s, ok := m.Pool.Mamba.Alloc(1); ok {
		return s[0], nil
	}
	if req != nil && m.Pool.lazy && req.Lock != nil {
		return 0, fmt.Errorf("cache: dropping the COW-source lock before a mamba eviction (extra_buffer_lazy) is not modelled; inference-sim/inference-sim#2442")
	}
	m.Evictor.EvictForAlloc(1)
	s, ok := m.Pool.Mamba.Alloc(1)
	if !ok {
		return 0, ErrMambaAllocFailed
	}
	return s[0], nil
}

// FreeMambaValue is _free_mamba_value (:576-580) without the int8 pool: the donated slot goes back
// to the mamba allocator.
func (m *MambaLifecycle) FreeMambaValue(v []int64) { m.Pool.Mamba.Free(v) }

// selectFinishedCheckpoint is _select_finished_checkpoint (:582-602): the tracked state a finished
// request donates and the ping-pong entry it sits in, or none when no state fits the key.
func (m *MambaLifecycle) selectFinishedCheckpoint(mk *MambaKV, tokenIDsLen int) (cacheLen, keepIdx int, ok bool) {
	keepIdx = m.Pool.PingPongKeepIdx(mk)
	cacheLen = 0
	if mk.LastTrackSeqlen != nil {
		cacheLen = *mk.LastTrackSeqlen
	}
	if cacheLen <= tokenIDsLen {
		return cacheLen, keepIdx, true
	}
	// Overshoot: the latest state ran past the key; the other entry's seqlen says whether a key
	// can name it.
	if m.Pool.PingPongSize() != 2 || mk.PrevTrackSeqlen == nil || *mk.PrevTrackSeqlen > tokenIDsLen {
		return 0, 0, false
	}
	return *mk.PrevTrackSeqlen, m.Pool.PingPongOtherIdx(keepIdx), true
}

// PrepareForCachingReq is prepare_for_caching_req (:604-727) without the int8 pool and ReplaySSM:
// the length the component lets the insert cover, with ins.Value (and, finished under the extra
// buffer, ins.KeepIdx) set to the state the request donates.
func (m *MambaLifecycle) PrepareForCachingReq(req *Req, ins *MambaInsert, tokenIDsLen int, finished bool) (int, error) {
	mk := req.KV.Mamba
	if mk == nil || !mk.HoldsMamba() {
		return 0, fmt.Errorf("cache: PrepareForCachingReq on a request with no mamba state")
	}
	row := *req.KV.ReqPoolIdx
	var cacheLen *int
	if m.ExtraBuffer {
		cacheLen = mk.LastTrackSeqlen
	} else {
		n := tokenIDsLen
		cacheLen = &n
	}
	if finished {
		n := 0
		if cacheLen != nil {
			n = *cacheLen
		}
		var active int64
		if m.ExtraBuffer {
			l, keep, ok := m.selectFinishedCheckpoint(mk, tokenIDsLen)
			if !ok {
				return 0, nil
			}
			n = l
			ins.KeepIdx = &keep
			active = mk.PingPongBuffer[keep]
		} else {
			active = *mk.PoolIdx
		}
		ins.Value = []int64{active}
		return n, nil
	}
	if cacheLen == nil {
		return 0, nil
	}
	// Donate the mamba index to the radix cache instead of copying.
	var donated int64
	if m.ExtraBuffer {
		slot, err := m.allocMambaSlot(req)
		if err != nil {
			return 0, err
		}
		donated = m.Pool.DonatePingPongSlot(row, mk, slot)
	} else {
		slot, err := m.allocMambaSlot(req)
		if err != nil {
			return 0, err
		}
		// SGLang copies the request's state into the new slot; the state tensors are not modelled.
		donated = slot
	}
	ins.Value = []int64{donated}
	return *cacheLen, nil
}

// CleanupAfterCachingReq is cleanup_after_caching_req (:684-727) without the int8 pool: what the
// request frees once the tree took, or did not take, its donated state.
func (m *MambaLifecycle) CleanupAfterCachingReq(req *Req, finished bool, res *MambaInsertResult, ins *MambaInsert) error {
	mk := req.KV.Mamba
	if finished {
		inserted := res != nil && !res.Exist
		row := 0
		if req.KV.ReqPoolIdx != nil {
			row = *req.KV.ReqPoolIdx
		}
		if m.ExtraBuffer {
			// Keep the slot prepare picked, so cleanup cannot pick another.
			var keep *int
			if inserted && ins != nil {
				keep = ins.KeepIdx
			}
			return m.Pool.FreeMambaCache(row, mk, keep)
		}
		if !inserted {
			return m.Pool.FreeMambaCache(row, mk, nil)
		}
		return nil
	}
	if ins != nil && ins.Value != nil && (res == nil || res.Exist) {
		m.FreeMambaValue(ins.Value)
	}
	mk.LastTrackSeqlen, mk.PrevTrackSeqlen = nil, nil
	return nil
}
