// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/manager.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// CPUPrimaryTierOffloadingManager (:114-173): the CPU tier's OffloadingManager methods, and the
// read/write aliases the tiering manager calls them by.

package offloading

import "fmt"

// CPUPrimary is the CPU tier as a primary tier under TieringOffloadingManager
// (CPUPrimaryTierOffloadingManager, vllm/v1/kv_offload/tiering/manager.py:114-173@4ba787ec): the
// same CPUOffloadingManager and LRU policy this package's scheduler uses, exported for
// core/connector/tiering. Each method is the cpuManager method it names; nothing here decides.
type CPUPrimary struct {
	m    *cpuManager
	info map[string]any
}

// NewCPUPrimary is a CPU tier of numChunks chunks.
func NewCPUPrimary(numChunks int) *CPUPrimary { return &CPUPrimary{m: newCPUManager(numChunks)} }

// Access is one request's access state in the CPU tier (_RequestCacheAccess and the ReqContext's
// key positions); the caller keeps one per request.
type Access struct {
	a      *reqAccess
	id     string
	state  any
	params map[string]any
}

// NewAccess is a request's access state.
func NewAccess() *Access { return &Access{a: newReqAccess()} }

// SetPosition is ReqContext.set_offload_key_position.
func (a *Access) SetPosition(k Key, end int) { a.a.positions[k] = end }

// MakeKey is make_offload_key (vllm/v1/kv_offload/base.py:30-32).
func MakeKey(hash []byte, group int) Key { return makeKey(hash, group) }

// NumChunks is the tier's capacity.
func (p *CPUPrimary) NumChunks() int { return p.m.numChunks }

// Lookup is lookup: found, and whether the chunk is ready (HIT) or its store in flight
// (HIT_PENDING).
func (p *CPUPrimary) Lookup(k Key) (found, ready bool) {
	switch p.m.lookup(k) {
	case hit:
		return true, true
	case hitPending:
		return true, false
	}
	return false, false
}

// LoadSource is get_load_source.
func (p *CPUPrimary) LoadSource() string { return p.m.loadSource() }

// PrepareLoad is prepare_load.
func (p *CPUPrimary) PrepareLoad(keys []Key, a *Access) []int { return p.m.prepareLoad(keys, a.a) }

// PrepareRead is prepare_read (tiering/manager.py:150-158): _prepare_load with record_access
// False (cpu/manager.py:221-242), which pins the chunks as prepare_load does and records no access.
func (p *CPUPrimary) PrepareRead(keys []Key) []int {
	m := p.m
	ids := make([]int, len(keys))
	for i, k := range keys {
		c := m.lru.chunks[k]
		if c == nil || !c.ready() {
			panic(fmt.Sprintf("offloading: chunk %x is not ready for reading", []byte(k)))
		}
		if c.refCnt == 0 {
			m.lru.markNonEvictable(k)
			m.numEvictable--
		}
		c.refCnt++
		ids[i] = c.id
	}
	return ids
}

// CompleteLoad is complete_load, and complete_read.
func (p *CPUPrimary) CompleteLoad(keys []Key) { p.m.completeLoad(keys) }

// PrepareStore is prepare_store, and prepare_write.
func (p *CPUPrimary) PrepareStore(keys []Key, a *Access) (stored []Key, chunkIDs []int, evicted []Key, ok bool) {
	return p.m.prepareStore(keys, a.a)
}

// CompleteStore is complete_store, and complete_write.
func (p *CPUPrimary) CompleteStore(keys []Key, success bool) { p.m.completeStore(keys, success) }

// Touch is touch.
func (p *CPUPrimary) Touch(keys []Key) { p.m.touch(keys) }

// OnRequestFinished is on_request_finished.
func (p *CPUPrimary) OnRequestFinished(a *Access) { p.m.requestFinished(a.a) }

// ResetCache is reset_cache.
func (p *CPUPrimary) ResetCache() { p.m.resetCache() }

// ReqID is the request's id (ReqContext.req_id).
func (a *Access) ReqID() string { return a.id }

// State is the per-request value a composing manager keeps with the request (ReqContext's
// set_state/get_state scratch space, vllm/v1/kv_offload/base.py:98-112@4ba787ec), nil until set.
func (a *Access) State() any { return a.state }

// SetState sets State.
func (a *Access) SetState(v any) { a.state = v }

// Params is the request's kv_transfer_params (Request.kv_transfer_params), nil for none.
func (a *Access) Params() map[string]any { return a.params }

// newAccess is a request's access state, for the request id.
func newAccess(id string) *Access { return &Access{a: newReqAccess(), id: id} }

// CPUStats are the CPU tier's get_stats values (cpu/manager.py:486-519@4ba787ec): its cache usage
// and the shares of it in writes and reads, and the chunk counts of the prepare_store calls since the
// last Stats.
type CPUStats struct {
	Usage, Write, Read float64
	Allocations        []int
}

// Stats is get_stats; the allocation sizes start again.
func (p *CPUPrimary) Stats() CPUStats {
	st := p.m.stats()
	return CPUStats{Usage: st.usage, Write: st.write, Read: st.read, Allocations: st.allocations}
}

// SetConfigInfo states the tier's config_info series (ConfigInfo), which the spec's sizes give.
func (p *CPUPrimary) SetConfigInfo(info map[string]any) { p.info = info }

// ConfigInfo is config_info's one series: nil until SetConfigInfo.
func (p *CPUPrimary) ConfigInfo() map[string]any { return p.info }
