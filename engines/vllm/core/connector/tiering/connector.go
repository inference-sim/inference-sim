// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/manager.py and tiering/spec.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import "github.com/inference-sim/blis-vllm/core/connector/offloading"

// CPU is the connector's CPU tier as the tiering manager's primary tier
// (CPUPrimaryTierOffloadingManager, manager.py:114-173): each method is the CPUPrimary method it
// names. A request's CPU access state is the ReqContext's Primary, the connector's own Access
// when the connector made the context.
func CPU(p *offloading.CPUPrimary) Primary { return cpuPrimary{p} }

type cpuPrimary struct{ p *offloading.CPUPrimary }

func okeys(ks []Key) []offloading.Key {
	out := make([]offloading.Key, len(ks))
	for i, k := range ks {
		out[i] = offloading.Key(k)
	}
	return out
}

func tkeys(ks []offloading.Key) []Key {
	out := make([]Key, len(ks))
	for i, k := range ks {
		out[i] = Key(k)
	}
	return out
}

func access(c *ReqContext) *offloading.Access {
	if c.Primary == nil {
		c.Primary = offloading.NewAccess()
	}
	return c.Primary.(*offloading.Access)
}

func (c cpuPrimary) NumChunks() int { return c.p.NumChunks() }
func (c cpuPrimary) Lookup(k Key, _ *ReqContext) Lookup {
	found, ready := c.p.Lookup(offloading.Key(k))
	switch {
	case !found:
		return Miss
	case !ready:
		return HitPending
	}
	return Hit
}
func (c cpuPrimary) LoadSource(Key, *ReqContext) string { return c.p.LoadSource() }
func (c cpuPrimary) PrepareLoad(ks []Key, r *ReqContext) []int {
	return c.p.PrepareLoad(okeys(ks), access(r))
}
func (c cpuPrimary) PrepareRead(ks []Key, _ *ReqContext) []int { return c.p.PrepareRead(okeys(ks)) }
func (c cpuPrimary) CompleteLoad(ks []Key, _ *ReqContext)      { c.p.CompleteLoad(okeys(ks)) }
func (c cpuPrimary) PrepareStore(ks []Key, r *ReqContext) ([]Key, []int, []Key, bool) {
	s, ids, ev, ok := c.p.PrepareStore(okeys(ks), access(r))
	return tkeys(s), ids, tkeys(ev), ok
}
func (c cpuPrimary) CompleteStore(ks []Key, _ *ReqContext, ok bool) {
	c.p.CompleteStore(okeys(ks), ok)
}
func (c cpuPrimary) Touch(ks []Key, _ *ReqContext)   { c.p.Touch(okeys(ks)) }
func (c cpuPrimary) OnRequestFinished(r *ReqContext) { c.p.OnRequestFinished(access(r)) }
func (c cpuPrimary) ResetCache()                     { c.p.ResetCache() }
func (c cpuPrimary) TakeEvents() []Event             { return nil }

// Connector is the tiering manager as the OffloadingConnector's scheduler side calls it
// (offloading.Manager): TieringOffloadingSpec's manager (spec.py) over the connector's CPU tier.
// Each request's ReqContext is made at on_new_request and kept in the request's Access.
type Connector struct{ M *Manager }

// ctx is the request's ReqContext (_create_req_context, offloading/scheduler.py:540-552): its
// kv_transfer_params, its load-tier filter from their kv_load_tiers (every tier allowed without one),
// and the CPU tier's state the connector's Access.
func ctx(a *offloading.Access) *ReqContext {
	if c, ok := a.State().(*ReqContext); ok {
		return c
	}
	params := a.Params()
	filter := AllTiers
	if len(params) > 0 {
		if raw, stated := params[KVLoadTiersKey]; stated && raw != nil {
			filter = ParseTierFilter(raw)
		}
	}
	c := &ReqContext{ReqID: a.ReqID(), Filter: filter, Params: params, Primary: a}
	a.SetState(c)
	return c
}

func (c Connector) OnNewRequest(a *offloading.Access) offloading.Policy {
	if c.M.OnNewRequest(ctx(a)) == RequestLevel {
		return offloading.RequestLevel
	}
	return offloading.ChunkLevel
}

func (c Connector) Lookup(k offloading.Key, a *offloading.Access) offloading.LookupResult {
	switch c.M.Lookup(Key(k), ctx(a)) {
	case Hit:
		return offloading.Hit
	case HitPending:
		return offloading.HitPending
	case Retry:
		return offloading.Retry
	}
	return offloading.Miss
}

func (c Connector) LoadSource(k offloading.Key, a *offloading.Access) string {
	return c.M.LoadSource(Key(k), ctx(a))
}

func (c Connector) PrepareLoad(keys []offloading.Key, a *offloading.Access) []int {
	return c.M.PrepareLoad(tkeys(keys), ctx(a))
}

func (c Connector) CompleteLoad(keys []offloading.Key, a *offloading.Access) {
	c.M.CompleteLoad(tkeys(keys), ctx(a))
}

func (c Connector) PrepareStore(keys []offloading.Key, a *offloading.Access) ([]offloading.Key, []int, []offloading.Key, bool) {
	s, ids, ev, ok := c.M.PrepareStore(tkeys(keys), ctx(a))
	return okeys(s), ids, okeys(ev), ok
}

func (c Connector) CompleteStore(keys []offloading.Key, a *offloading.Access, success bool) {
	c.M.CompleteStore(tkeys(keys), ctx(a), success)
}

func (c Connector) OnRequestFinished(a *offloading.Access) { c.M.OnRequestFinished(ctx(a)) }

func (c Connector) OnScheduleEnd(e offloading.ScheduleEnd) {
	c.M.OnScheduleEnd(ScheduleEnd{NewReqIDs: e.NewReqIDs, PreemptedReqIDs: e.PreemptedReqIDs})
}

func (c Connector) HasPendingWork() bool { return c.M.HasPendingWork() }

// Stats is the CPU tier's get_stats (cpu/manager.py:486-519@4ba787ec): its usage, the allocation sizes
// since the last call, and the write and read shares of the usage.
func (c cpuPrimary) Stats() *Stats {
	st := c.p.Stats()
	s := NewStats()
	s.SetGauge(MetricCPUUsage, st.Usage)
	for _, a := range st.Allocations {
		s.ObserveHistogram(MetricCPUAllocation, float64(a))
	}
	s.SetGauge(MetricCPUWriteUsage, st.Write)
	s.SetGauge(MetricCPUReadUsage, st.Read)
	return s
}

// ConfigInfo is the CPU tier's config_info series.
func (c cpuPrimary) ConfigInfo() map[string]any { return c.p.ConfigInfo() }
