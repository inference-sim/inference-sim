// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).

// Package multi is the scheduler side of vLLM's MultiConnector
// (vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:135@5df63a72): an ordered list
// of child connectors behind the one Connector the scheduler calls. Each hook goes to every child
// in order; the one choice the wrapper makes is which child loads a request's KV: the first, in
// the configured order, that reports matched tokens (get_num_new_matched_tokens, :434-456). It
// saves to all of them. The worker side, which moves the bytes, is the transport.
//
// It composes the scheduler sides this repository models, NIXL (core/connector/nixl) and the
// OffloadingConnector (core/connector/offloading), and states a capability only when every child
// does.
package multi

import (
	"fmt"

	"github.com/inference-sim/blis-vllm/core/sched"
)

// Child is one connector of the list, with the name vLLM's config states for it.
type Child struct {
	Name string
	Conn sched.Connector
	// Producer is whether the child's own kv_role is a producer role (KVTransferConfig.is_kv_producer
	// of the child's config, multi_connector.py:174-190@5df63a72): the default of requires_kv_delivery
	// for a child that does not state it (base.py:196-206@5df63a72).
	Producer bool
}

// Meta is MultiKVConnectorMetadata (multi_connector.py:49-52): each child's build_connector_meta
// result, in child order. Step.KVConnectorMetadata carries it.
type Meta []any

// At is child i's metadata, nil for a child that returned none.
func (m Meta) At(i int) any {
	if i < 0 || i >= len(m) {
		return nil
	}
	return m[i]
}

// WorkerMeta is MultiKVConnectorWorkerMetadata (multi_connector.py:55-72): each child's worker
// metadata (kv_connector_worker_meta) in child order, nil for a child whose worker reported none.
// TransferOutcome.WorkerMeta carries it.
type WorkerMeta []any

// metaBuilder is a child whose build_connector_meta takes no part of the step's output (NIXL's
// scheduler clears its per-step sets, nixl.Scheduler.BuildMeta).
type metaBuilder interface{ BuildMeta() }

// Connector is MultiConnector for one engine.
type Connector struct {
	children []Child
	// chosen is _requests_to_connector (:203): the child chosen to load a request's KV.
	chosen map[string]int
	caps   sched.Capabilities
	hma    bool
}

// New builds the wrapper over the children, in order. A MultiConnector with no child is refused.
func New(children []Child) (*Connector, error) {
	if len(children) == 0 {
		return nil, fmt.Errorf("multi: MultiConnector needs at least one child connector " +
			"(kv_connector_extra_config connectors, multi_connector.py:226-249@5df63a72)")
	}
	c := &Connector{children: append([]Child(nil), children...), chosen: map[string]int{}, hma: true,
		caps: sched.Capabilities{HybridGroups: true, Drafters: true, ConcurrentBatches: true}}
	for _, ch := range children {
		if ch.Conn == nil {
			return nil, fmt.Errorf("multi: child %q is nil", ch.Name)
		}
		var caps sched.Capabilities
		if cc, ok := ch.Conn.(sched.ConnectorCapabilities); ok {
			caps = cc.ConnectorModels()
		}
		c.caps.HybridGroups = c.caps.HybridGroups && caps.HybridGroups
		c.caps.Drafters = c.caps.Drafters && caps.Drafters
		c.caps.ConcurrentBatches = c.caps.ConcurrentBatches && caps.ConcurrentBatches
		if _, ok := ch.Conn.(sched.SupportsHMA); !ok {
			c.hma = false
		}
	}
	// _all_support_hma (:159-172, 193-200): without it vLLM turns the hybrid manager off, so the
	// scheduler only ever sees one KV cache group, and the wrapper models none beyond that.
	c.caps.HybridGroups = c.caps.HybridGroups && c.hma
	return c, nil
}

// ConnectorModels implements sched.ConnectorCapabilities: what every child is diff-tested
// beside.
func (c *Connector) ConnectorModels() sched.Capabilities { return c.caps }

// Len is the number of children; Name is child i's name.
func (c *Connector) Len() int                    { return len(c.children) }
func (c *Connector) Name(i int) string           { return c.children[i].Name }
func (c *Connector) Child(i int) sched.Connector { return c.children[i].Conn }

// Chosen is the child chosen to load the request's KV, if any (_requests_to_connector).
func (c *Connector) Chosen(id string) (int, bool) {
	i, ok := c.chosen[id]
	return i, ok
}

// RequiresKVDelivery is requires_kv_delivery (:221-224): any child's.
func (c *Connector) RequiresKVDelivery() bool {
	for _, ch := range c.children {
		d, ok := ch.Conn.(sched.KVDelivery)
		if (ok && d.RequiresKVDelivery()) || (!ok && ch.Producer) {
			return true
		}
	}
	return false
}

// OnNewRequest is on_new_request (:479-481): every child.
func (c *Connector) OnNewRequest(r *sched.Request) {
	for _, ch := range c.children {
		ch.Conn.OnNewRequest(r)
	}
}

// MatchedTokens is get_num_new_matched_tokens without a deferral (sched.Connector).
func (c *Connector) MatchedTokens(r *sched.Request, local int) (int, bool) {
	n, async, _ := c.MatchedTokensOrDefer(r, local)
	return n, async
}

// MatchedTokensOrDefer is get_num_new_matched_tokens (:434-456). Every child is asked, in order,
// even after one has matched tokens: a child still looking up the hit (None) defers the whole
// request, and the first child with tokens > 0 is the one that loads it. A request no child
// matches keeps the child an earlier call chose, as vLLM's map is only written, never cleared,
// here (:451).
func (c *Connector) MatchedTokensOrDefer(r *sched.Request, local int) (tokens int, async, deferred bool) {
	for i, ch := range c.children {
		var n int
		var a, d bool
		if dc, ok := ch.Conn.(sched.Deferrer); ok {
			n, a, d = dc.MatchedTokensOrDefer(r, local)
		} else {
			n, a = ch.Conn.MatchedTokens(r, local)
		}
		if d {
			return 0, false, true
		}
		if tokens == 0 && n > 0 {
			c.chosen[r.ID] = i
			tokens, async = n, a
		}
	}
	return tokens, async, false
}

// AfterAlloc is update_state_after_alloc (:467-477): the chosen child sees the external tokens,
// every other child the request's real blocks and none.
func (c *Connector) AfterAlloc(r *sched.Request, blocks sched.AllocatedBlocks, external int) {
	chosen, ok := c.chosen[r.ID]
	if !ok {
		chosen = -1
	}
	for i, ch := range c.children {
		if i == chosen {
			ch.Conn.AfterAlloc(r, blocks, external)
		} else {
			ch.Conn.AfterAlloc(r, blocks, 0)
		}
	}
}

// Finished is request_finished (:573-593) for one KV cache group.
func (c *Connector) Finished(r *sched.Request, computedBlockIDs []int) bool {
	return c.finished(r, func(ch Child) bool { return ch.Conn.Finished(r, computedBlockIDs) })
}

// FinishedAllGroups is request_finished_all_groups (:595-616). With every child SupportsHMA
// each gets every group's ids; otherwise vLLM runs one group and calls request_finished.
func (c *Connector) FinishedAllGroups(r *sched.Request, blockIDs [][]int) bool {
	if !c.hma {
		if len(blockIDs) != 1 {
			panic("multi: HMA with several KV cache groups requires every child to support HMA (multi_connector.py:600-604@5df63a72)")
		}
		return c.Finished(r, blockIDs[0])
	}
	return c.finished(r, func(ch Child) bool { return ch.Conn.(sched.SupportsHMA).FinishedAllGroups(r, blockIDs) })
}

// finished is _aggregate_request_finished (:542-571): every child is called, the blocks are held
// if any child holds them, and the request's chosen child is forgotten. vLLM also counts the
// children that hold them (async_saves) and, above one, makes the worker report the request's
// finished_sending once per holder (_extra_async_saves). That is not modelled: it needs two
// children that both hold a finished request's blocks, and of the children modelled only NIXL does
// (the offloading connector never does, offloading/scheduler.py:2005-2044).
func (c *Connector) finished(r *sched.Request, one func(Child) bool) bool {
	hold := false
	for _, ch := range c.children {
		if one(ch) {
			hold = true
		}
	}
	delete(c.chosen, r.ID)
	return hold
}

// BuildMeta implements sched.MetaBuilder: build_connector_meta (:483-494), every child's, in
// order. The result is a Meta.
func (c *Connector) BuildMeta(step sched.StepOutput) any {
	m := make(Meta, len(c.children))
	for i, ch := range c.children {
		switch b := ch.Conn.(type) {
		case sched.MetaBuilder:
			m[i] = b.BuildMeta(step)
		case metaBuilder:
			b.BuildMeta()
		}
	}
	return m
}

// HasPendingPushWork is has_pending_push_work (:621-622): any child's.
func (c *Connector) HasPendingPushWork() bool {
	for _, ch := range c.children {
		if p, ok := ch.Conn.(sched.PushWork); ok && p.HasPendingPushWork() {
			return true
		}
	}
	return false
}

// AfterOutput is update_connector_output (:496-517): each child sees the step's output with its
// own slice of the worker metadata (a WorkerMeta), none when the worker reported none.
func (c *Connector) AfterOutput(o sched.TransferOutcome) {
	wm, _ := o.WorkerMeta.(WorkerMeta)
	for i, ch := range c.children {
		oc := o
		oc.WorkerMeta = nil
		if wm != nil && i < len(wm) {
			oc.WorkerMeta = wm[i]
		}
		ch.Conn.AfterOutput(oc)
	}
}

var (
	_ sched.Connector             = (*Connector)(nil)
	_ sched.SupportsHMA           = (*Connector)(nil)
	_ sched.MetaBuilder           = (*Connector)(nil)
	_ sched.PushWork              = (*Connector)(nil)
	_ sched.Deferrer              = (*Connector)(nil)
	_ sched.KVDelivery            = (*Connector)(nil)
	_ sched.ConnectorCapabilities = (*Connector)(nil)
)
