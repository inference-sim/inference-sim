package llmd

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"sort"

	"github.com/inference-sim/blis-schemas/embed"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// KV events: the simulated engines publish their prefix caches' changes as vLLM does, and the
// router indexes them (embed.Router.IngestKVEvents), exactly as it would vLLM's events over ZMQ.
//
//   - An engine records a stored block when a block hash enters its prefix cache and a removed
//     block when one leaves (blis-vllm compat/kv), so the router's index holds the engine's own set of cached
//     blocks, full prefill and decode blocks alike.
//   - It publishes them at the end of each step, as vLLM does; the router receives them
//     Options.KVEventDelayUs later, in order, per engine.
//   - Each event carries the block's tokens as the router saw them (promptCanon), so the router's
//     keys, recomputed from those tokens, are its keys for the same prompts.

// subscribeKVEvents makes inst publish its KV events to the router, when its configuration
// consumes them.
func (g *Gateway) subscribeKVEvents(inst *cluster.InstanceSimulator) {
	if !g.router.WantsKVEvents() {
		return
	}
	g.kvHeld[inst.ID()] = map[string]int{}
	g.kvChains[inst.ID()] = map[string]*engineChain{}
	inst.SetKVEventSink(func(at int64, events []sim.KVEvent) {
		held := g.kvHeld[inst.ID()]
		if held == nil {
			return // the router has cleared the pod: nothing it publishes is delivered
		}
		events = chainEvents(g.kvChains[inst.ID()], events)
		// Pin each chain the engine stores, at publication: its canonical tokens must not change
		// while the router may still index the block (promptCanon). A removal unpins only when
		// the router receives it (deliverKVEvents).
		for _, e := range events {
			if !e.Removed {
				held[e.Hash]++
				g.canon.pin(e.Hash)
			}
		}
		// The router receives an engine's batches in the order the engine published them, as over
		// vLLM's ordered event stream: a batch is delivered no earlier than the one before it (a
		// step can be timed to end before the previous one's end when it starts on an engine that
		// looked idle), so a block's store never reaches the router ahead of its parent's.
		g.host.Schedule(g.kvDeliveryTime(inst.ID(), at), PriorityKVEvents, true, func() { g.deliverKVEvents(inst, at, events) })
	})
}

// engineChain is the prefix chain an engine-keyed block hash names, and how many of the engine's
// published stores hold it.
type engineChain struct {
	chain  string
	stores int
}

// chainEvents rewrites events an engine keys in its own form into BLIS's prefix chains. An engine
// whose prefix cache keys blocks as BLIS does (the compatibility engines: SHA-256 chain hashes in
// hex) is passed through. vLLM v0.32.0 keys a block by its own 64-bit hash, published in decimal;
// its stored block's chain is BLIS's hash of its parent's chain and its tokens, the same
// hashBlock the canonicalizer applies to the prompt (promptCanon.canonical), so a block the
// engine cached from a prompt names the chain the router was shown for it. Removals resolve
// through the chains of the stores they undo; the mapping is dropped once no store holds it.
func chainEvents(chains map[string]*engineChain, events []sim.KVEvent) []sim.KVEvent {
	if len(events) == 0 || isChainHash(events[0].Hash) {
		return events
	}
	out := make([]sim.KVEvent, len(events))
	for i, e := range events {
		parent := ""
		if c := chains[e.Parent]; c != nil {
			parent = c.chain
		} else if e.Parent != "" && !e.Removed {
			panic(fmt.Sprintf("llmd: the engine stored block %s under parent %s, which it has not published", e.Hash, e.Parent))
		}
		c := chains[e.Hash]
		if e.Removed {
			if c == nil {
				panic(fmt.Sprintf("llmd: the engine removed block %s, which it has not published", e.Hash))
			}
			if c.stores--; c.stores == 0 {
				delete(chains, e.Hash)
			}
			out[i] = sim.KVEvent{Removed: true, Hash: c.chain, Tier: e.Tier}
			continue
		}
		chain := hashBlock(parent, e.Tokens)
		if c == nil {
			c = &engineChain{chain: chain}
			chains[e.Hash] = c
		}
		c.stores++
		out[i] = sim.KVEvent{Hash: c.chain, Parent: parent, Tokens: e.Tokens, Tier: e.Tier}
	}
	return out
}

// isChainHash reports whether h is a BLIS prefix-chain hash: a SHA-256 digest in hex.
func isChainHash(h string) bool {
	if len(h) != 2*sha256.Size {
		return false
	}
	_, err := hex.DecodeString(h)
	return err == nil
}

// kvDeliveryTime is when the router receives what instance id publishes at time at: after the
// delivery delay, and no earlier than anything it published before (subscribeKVEvents).
func (g *Gateway) kvDeliveryTime(id cluster.InstanceID, at int64) int64 {
	t := max(at+g.opts.KVEventDelayUs, g.kvLastDelivery[id])
	g.kvLastDelivery[id] = t
	return t
}

// deliverKVEvents hands one engine step's events to the router.
func (g *Gateway) deliverKVEvents(inst *cluster.InstanceSimulator, at int64, events []sim.KVEvent) {
	ep, ok := g.endpoints[inst.ID()]
	if !ok || ep.kvCleared {
		return
	}
	if g.observe != nil {
		g.observe("kvdeliver", fmt.Sprintf("%s %d", inst.ID(), at))
		for _, e := range events {
			g.observe("kvevent", fmt.Sprintf("%s %q %s %v", inst.ID(), e.Tier, e.Hash, e.Removed))
		}
	}
	batch := g.kvEventBatch(at, events)
	held := g.kvHeld[inst.ID()]
	for _, e := range events {
		if e.Removed && held[e.Hash] > 0 {
			// A chain no store holds any more leaves the instance's holdings, so they stay the
			// engine's cached set rather than every chain it ever published.
			if held[e.Hash]--; held[e.Hash] == 0 {
				delete(held, e.Hash)
			}
			g.canon.unpin(e.Hash)
		}
	}
	if err := g.router.IngestKVEvents(g.ctx, ep.ep.Address(), g.servedModel(inst), batch); err != nil {
		g.host.Abort(fmt.Errorf("llm-d router: KV events of %s: %w", inst.ID(), err))
		return
	}
	g.rt.co.settle()
}

// clearKVEvents tells the router that inst's prefix cache is gone (its pod was removed), and
// releases every chain the instance still pins: removals delivered after the clear are dropped.
func (g *Gateway) clearKVEvents(ep *endpoint) {
	ep.kvCleared = true
	held := g.kvHeld[ep.inst.ID()]
	keys := make([]string, 0, len(held))
	for h := range held {
		keys = append(keys, h)
	}
	sort.Strings(keys)
	for _, h := range keys {
		for ; held[h] > 0; held[h]-- {
			g.canon.unpin(h)
		}
	}
	delete(g.kvHeld, ep.inst.ID())
	delete(g.kvChains, ep.inst.ID())
	if g.observe != nil {
		g.observe("kvclear", string(ep.inst.ID()))
	}
	if err := g.router.ClearKVEvents(g.ctx, ep.ep.Address()); err != nil {
		g.host.Abort(fmt.Errorf("llm-d router: clearing the KV events of %s: %w", ep.inst.ID(), err))
		return
	}
	g.rt.co.settle()
}

// kvEventBatch converts one step's events into vLLM's event batch, in order: a run of GPU stores
// that chain (each block's parent is the previous one) becomes one BlockStored with the blocks'
// canonical tokens; a run of CPU stores one location-only BlockStored (Tier "cpu", no tokens, as
// vLLM's offloading events); a run of removals on one tier one BlockRemoved.
func (g *Gateway) kvEventBatch(at int64, events []sim.KVEvent) embed.KVBatch {
	batch := embed.KVBatch{Timestamp: toTime(at)}
	var stored *embed.BlockStored
	var removed *embed.BlockRemoved
	last, runTier := "", ""
	flush := func() {
		if stored != nil {
			batch.Events = append(batch.Events, *stored)
			stored = nil
		}
		if removed != nil {
			batch.Events = append(batch.Events, *removed)
			removed = nil
		}
	}
	for _, e := range events {
		if e.Removed {
			if stored != nil || (removed != nil && runTier != e.Tier) {
				flush()
			}
			if removed == nil {
				removed = &embed.BlockRemoved{Tier: e.Tier}
				runTier = e.Tier
			}
			removed.Hashes = append(removed.Hashes, engineKey(e.Hash))
			continue
		}
		gpu := e.Tier == ""
		if removed != nil || (stored != nil && (runTier != e.Tier || (gpu && e.Parent != last))) {
			flush()
		}
		if stored == nil {
			stored = &embed.BlockStored{BlockSize: g.canon.blockSize, Tier: e.Tier}
			runTier = e.Tier
			if gpu && e.Parent != "" {
				stored.Parent = engineKey(e.Parent)
			}
		}
		stored.Hashes = append(stored.Hashes, engineKey(e.Hash))
		if gpu {
			stored.Tokens = append(stored.Tokens, g.canon.canonicalBlock(e.Hash, len(e.Tokens))...)
		}
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
