package llmd

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"net"
	"sort"
	"strconv"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache"
	"github.com/llm-d/llm-d-router/pkg/kvevents"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// KV events: the simulated engines publish their prefix caches' changes as vLLM does, and the
// router's precise-prefix-cache-producer indexes them, exactly as it would vLLM's events over ZMQ.
//
//   - An engine records a stored block when a block hash enters its prefix cache and a removed
//     block when one leaves (sim/kv), so the router's index holds the engine's own set of cached
//     blocks, full prefill and decode blocks alike.
//   - It publishes them at the end of each step, as vLLM does; the router receives them
//     Options.KVEventDelayUs later, in order, per engine.
//   - Each event carries the block's tokens as the router saw them (promptCanon), so the router's
//     keys, recomputed from those tokens, are its keys for the same prompts.

// kvIngesters are the precise-prefix producers of the configuration, in name order.
func kvIngesters(handle fwkplugin.Handle) []*preciseprefixcache.Producer {
	var names []string
	all := handle.GetAllPluginsWithNames()
	for name := range all {
		names = append(names, name)
	}
	sort.Strings(names)
	var out []*preciseprefixcache.Producer
	for _, name := range names {
		if p, ok := all[name].(*preciseprefixcache.Producer); ok {
			out = append(out, p)
		}
	}
	return out
}

// subscribeKVEvents makes inst publish its KV events to the router's precise-prefix producers.
func (g *Gateway) subscribeKVEvents(inst *cluster.InstanceSimulator) {
	if len(g.kvIngest) == 0 {
		return
	}
	inst.SetKVEventSink(func(at int64, events []sim.KVEvent) {
		g.host.Schedule(at+g.opts.KVEventDelayUs, PriorityKVEvents, true, func() { g.deliverKVEvents(inst, at, events) })
	})
}

// deliverKVEvents hands one engine step's events to the router.
func (g *Gateway) deliverKVEvents(inst *cluster.InstanceSimulator, at int64, events []sim.KVEvent) {
	ep, ok := g.endpoints[inst.ID()]
	if !ok {
		return
	}
	batch := g.kvEventBatch(at, events)
	pod := net.JoinHostPort(ep.ip, strconv.Itoa(ep.port))
	for _, p := range g.kvIngest {
		if err := p.HostedIngest(g.ctx, pod, g.servedModel(inst), batch); err != nil {
			g.host.Abort(fmt.Errorf("llm-d router: KV events of %s: %w", inst.ID(), err))
			return
		}
	}
	g.rt.co.settle()
}

// clearKVEvents tells the router that inst's prefix cache is gone (its pod was removed).
func (g *Gateway) clearKVEvents(ep *endpoint) {
	pod := net.JoinHostPort(ep.ip, strconv.Itoa(ep.port))
	for _, p := range g.kvIngest {
		if err := p.HostedClearPod(g.ctx, pod); err != nil {
			g.host.Abort(fmt.Errorf("llm-d router: clearing the KV events of %s: %w", ep.inst.ID(), err))
			return
		}
	}
	g.rt.co.settle()
}

// kvEventBatch converts one step's events into vLLM's event batch: runs of stored blocks that
// chain (each block's parent is the previous one) become one BlockStored, runs of removals one
// BlockRemoved, in order.
func (g *Gateway) kvEventBatch(at int64, events []sim.KVEvent) *kvevents.EventBatch {
	batch := &kvevents.EventBatch{Timestamp: float64(toTime(at).UnixNano()) / 1e9}
	var stored *kvevents.BlockStoredEvent
	var removed *kvevents.BlockRemovedEvent
	last := ""
	flush := func() {
		if stored != nil {
			batch.Events = append(batch.Events, stored)
			stored = nil
		}
		if removed != nil {
			batch.Events = append(batch.Events, removed)
			removed = nil
		}
	}
	for _, e := range events {
		if e.Removed {
			if stored != nil {
				flush()
			}
			if removed == nil {
				removed = &kvevents.BlockRemovedEvent{}
			}
			removed.BlockHashes = append(removed.BlockHashes, engineKey(e.Hash))
			continue
		}
		if removed != nil || (stored != nil && e.Parent != last) {
			flush()
		}
		if stored == nil {
			stored = &kvevents.BlockStoredEvent{BlockSize: g.canon.blockSize}
			if e.Parent != "" {
				stored.ParentHash = engineKey(e.Parent)
			}
		}
		stored.BlockHashes = append(stored.BlockHashes, engineKey(e.Hash))
		stored.Tokens = append(stored.Tokens, g.canon.canonicalBlock(e.Hash, len(e.Tokens))...)
		last = e.Hash
	}
	flush()
	return batch
}

// engineKey is the 64-bit block hash an engine would publish for a BLIS block hash (the first 8
// bytes of its SHA-256). Zero is vLLM's "no parent", so a hash that truncates to zero is moved off
// it; the router only needs engine keys to be consistent with each other.
func engineKey(h string) uint64 {
	b, err := hex.DecodeString(h)
	if err != nil || len(b) < 8 {
		panic(fmt.Sprintf("llmd: block hash %q is not a SHA-256 hex digest", h))
	}
	if k := binary.BigEndian.Uint64(b[:8]); k != 0 {
		return k
	}
	return 1
}
