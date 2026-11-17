package llmd

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"slices"
	"strconv"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/embed"
	"github.com/inference-sim/blis-substrate/contracts/engine"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

func intp(v int) *int { return &v }

// TestChainEvents_AdapterBlocksDoNotCollapseIntoTheBaseModels: an engine that keys a block under the
// adapter's extra keys stores the same tokens as a different block, so the publisher names it a
// different chain. One chain for both would let the base model's removal unindex the adapter's block
// and show the router the same tokens for both.
func TestChainEvents_AdapterBlocksDoNotCollapseIntoTheBaseModels(t *testing.T) {
	tokens := []sim.TokenID{1, 2, 3, 4}
	lora := []any{[]any{"lora", "sql", "sql"}}
	evs := []sim.KVEvent{
		{Hash: "11", Tokens: tokens, LoRAName: "sql", ExtraKeys: lora},
		{Hash: "22", Tokens: tokens},
		{Hash: "33", Tokens: tokens, ExtraKeys: []any{"tenant-7"}},
		{Hash: "44", Tokens: tokens, ExtraKeys: []any{"tenant-8"}},
	}
	out := chainEvents(map[string]*engineChain{}, evs)
	seen := map[string]int{}
	for i, e := range out {
		if j, dup := seen[e.Hash]; dup {
			t.Errorf("events %d and %d name one chain %s though their blocks are keyed differently", j, i, e.Hash)
		}
		seen[e.Hash] = i
	}
	if out[1].Hash != hashBlock("", tokens) {
		t.Errorf("the base model's chain changed: %s, want %s", out[1].Hash, hashBlock("", tokens))
	}
	// A second block chains from its parent and inherits the adapter through it.
	next := chainEvents(map[string]*engineChain{}, []sim.KVEvent{
		{Hash: "11", Tokens: tokens, LoRAName: "sql", ExtraKeys: lora},
		{Hash: "12", Parent: "11", Tokens: tokens},
	})
	if next[1].Parent != next[0].Hash {
		t.Errorf("the second block's parent %s is not the first block's chain %s", next[1].Parent, next[0].Hash)
	}
	// The prompt the router is shown for an adapter request is keyed the same way.
	c := newPromptCanon(1, 4, 100)
	if slices.Equal(c.canonicalUnder(identityRoot("sql", nil), tokens), c.canonical(tokens)) {
		t.Error("an adapter's prompt and the base model's got the same canonical tokens")
	}
	if got, want := c.canonicalBlock(next[0].Hash, 4), c.canonicalUnder(identityRoot("sql", nil), tokens); !slices.Equal(got, want) {
		t.Errorf("the engine's adapter block shows tokens %v, the adapter's prompt %v", got, want)
	}
}

// TestKVEventBatch_CarriesTheStoredBlocksIdentity: the BlockStored the publisher builds names the
// adapter, extra keys, group and kind of the blocks the engine stored, in one run per identity, and
// the batch names its engine; an AllBlocksCleared is forwarded.
func TestKVEventBatch_CarriesTheStoredBlocksIdentity(t *testing.T) {
	g := &Gateway{canon: newPromptCanon(1, 4, 100)}
	h := func(c byte) string { return hashBlock(string(c), []sim.TokenID{1, 2, 3, 4}) }
	lora := []any{[]any{"lora", "sql", "sql"}}
	batch := g.kvEventBatch(5, []sim.KVEvent{
		{Hash: h('a'), Tokens: []sim.TokenID{1, 2, 3, 4}, LoRAID: intp(1), LoRAName: "sql", ExtraKeys: lora, GroupIdx: intp(0), SpecKind: "full_attention"},
		{Hash: h('b'), Parent: h('a'), Tokens: []sim.TokenID{1, 2, 3, 4}, LoRAID: intp(1), LoRAName: "sql", GroupIdx: intp(0), SpecKind: "full_attention"},
		{Hash: h('c'), Tokens: []sim.TokenID{1, 2, 3, 4}, GroupIdx: intp(0), SpecKind: "full_attention"},
		{Cleared: true},
	})
	if batch.Engine != embed.EngineVLLM {
		t.Errorf("batch engine %q, want %q", batch.Engine, embed.EngineVLLM)
	}
	if len(batch.Events) != 3 {
		t.Fatalf("got %d events, want an adapter run, a base run and a clear", len(batch.Events))
	}
	a, ok := batch.Events[0].(embed.BlockStored)
	if !ok || len(a.Hashes) != 2 || a.LoRAName != "sql" || a.LoRAID == nil || *a.LoRAID != 1 ||
		a.GroupIdx == nil || *a.GroupIdx != 0 || a.SpecKind != "full_attention" {
		t.Fatalf("adapter run = %+v", batch.Events[0])
	}
	if len(a.ExtraKeys) != 2 || fmt.Sprint(a.ExtraKeys[0]) != fmt.Sprint(lora) || a.ExtraKeys[1] != nil {
		t.Errorf("adapter run ExtraKeys = %v, want one entry per block: the adapter's key, then none", a.ExtraKeys)
	}
	b, ok := batch.Events[1].(embed.BlockStored)
	if !ok || b.LoRAName != "" || b.LoRAID != nil || b.ExtraKeys != nil {
		t.Errorf("base run carries an identity: %+v", batch.Events[1])
	}
	if _, ok := batch.Events[2].(embed.AllBlocksCleared); !ok {
		t.Errorf("last event %T, want AllBlocksCleared", batch.Events[2])
	}
}

// TestKVEventBatch_RemovalNamesTheGroupOfTheStore: the router counts a grouped store apart from an
// ungrouped one, so a removal that names no group leaves the block indexed after the engine has
// evicted it (the precise-prefix router then routes to caches that no longer hold the prefix: the
// Muse-Spark routing test measured it worse than load-only routing). Removals of different groups
// are separate runs.
func TestKVEventBatch_RemovalNamesTheGroupOfTheStore(t *testing.T) {
	g := &Gateway{canon: newPromptCanon(1, 4, 100)}
	h := func(c byte) string { return hashBlock(string(c), []sim.TokenID{1, 2, 3, 4}) }
	batch := g.kvEventBatch(5, []sim.KVEvent{
		{Removed: true, Hash: h('a'), GroupIdx: intp(0)},
		{Removed: true, Hash: h('b'), GroupIdx: intp(0)},
		{Removed: true, Hash: h('c'), GroupIdx: intp(1)},
		{Removed: true, Hash: h('d')},
	})
	if len(batch.Events) != 3 {
		t.Fatalf("got %d events, want one removal run per group (0, 1, none)", len(batch.Events))
	}
	want := []*int{intp(0), intp(1), nil}
	for i, w := range want {
		r, ok := batch.Events[i].(embed.BlockRemoved)
		if !ok || !intPtrEq(r.GroupIdx, w) {
			t.Errorf("event %d = %+v, want a removal of group %v", i, batch.Events[i], w)
		}
	}
}

// identityEngine is an engine whose stored blocks carry the adapter of the request that stored
// them, as blis-vllm's v0.33.0 engine publishes (requests run one at a time in the test, so the
// request a step serves is the last one added).
type identityEngine struct {
	engine.Engine
	enable  func()
	adapter string
	keys    map[string]string // compat chain hash -> the key published when it was first stored
}

func (e *identityEngine) AdapterCapacity() int {
	return e.Engine.(interface{ AdapterCapacity() int }).AdapterCapacity()
}

func (e *identityEngine) EnableKVEvents()             { e.enable() }
func (e *identityEngine) KVEventsCarryIdentity() bool { return true }

func (e *identityEngine) Add(r engine.Request, now engine.Time) error {
	e.adapter = r.Adapter
	return e.Engine.Add(r, now)
}

// vllmHash is a block's key as vLLM publishes it, a 64-bit integer in decimal, here derived from
// the compat engine's chain hash and the adapter, so an adapter's block and the base model's
// differ as they do in vLLM's prefix cache.
func vllmHash(adapter, chain string) string {
	if chain == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(adapter + "|" + chain))
	return strconv.FormatUint(binary.BigEndian.Uint64(sum[:8]), 10)
}

func (e *identityEngine) Complete(id engine.BatchID, o engine.Outcome, end engine.Time) (engine.Result, error) {
	res, err := e.Engine.Complete(id, o, end)
	for i := range res.KVEvents {
		ev := &res.KVEvents[i]
		if e.keys == nil {
			e.keys = map[string]string{}
		}
		if _, known := e.keys[ev.Hash]; !known {
			e.keys[ev.Hash] = vllmHash(e.adapter, ev.Hash)
		}
		ev.Hash = e.keys[ev.Hash]
		if ev.Parent != "" {
			ev.Parent = e.keys[ev.Parent]
		}
		if !ev.Removed && e.adapter != "" {
			ev.LoRAName, ev.LoRAID = e.adapter, intp(1)
			if ev.Parent == "" {
				ev.ExtraKeys = []any{e.adapter}
			}
		}
	}
	return res, err
}

// TestPrecisePrefix_RoutesAnAdapterToTheEndpointHoldingItsBlocks: over an engine whose KV events
// name the adapter, a precise-prefix router is not refused, and a request for an adapter goes where
// that adapter's blocks are cached, while the same prompt for the base model finds no hit there.
func TestPrecisePrefix_RoutesAnAdapterToTheEndpointHoldingItsBlocks(t *testing.T) {
	baseOnFirst := 0
	const seeds = 10
	for seed := int64(1); seed <= seeds; seed++ {
		reqs := testRequests(3, 5, 5_000_000, 320) // one prompt shape: three requests share a 320-token prefix
		shared := reqs[0].InputTokens
		for i, r := range reqs {
			r.InputTokens = shared
			r.Adapter = []string{"a0", "", "a0"}[i]
		}
		dep := testDeployment(2)
		capacity, base, bw, fp := 2, 1000.0, 2.0e6, 2.0e6
		editEngine(&dep, func(c *cengine.SimConfig) {
			c.LoRAConfig = cengine.LoRAConfig{AdapterCapacity: &capacity, LoadBaseLatencyUs: &base,
				LoadBandwidthBytesUs: &bw, FootprintBytesPerRank: &fp, Adapters: []cengine.AdapterSpec{{ID: "a0", Rank: 8}}}
		})
		inner := compatEngineFactory(0)
		dep.NewEngine = func(i int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
			d, err := inner(i, ic)
			if err != nil {
				return nil, err
			}
			ie := &identityEngine{Engine: d.Engine(), enable: d.Engine().(interface{ EnableKVEvents() }).EnableKVEvents}
			return sim.NewEngineDriver(ie, sim.EngineDriverConfig{Horizon: ic.Horizon}), nil
		}
		cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
		s := seed
		cs.SetGateway(New(Options{Model: "test-model", Adapters: []string{"a0"}, Seed: &s,
			EPPFlags: append(DefaultEPPFlags(), "--config-text="+preciseConfig(""))}))
		if err := cs.Run(); err != nil {
			t.Fatalf("seed %d: Run() = %v: an engine that names the adapter must not be refused", seed, err)
		}
		if reqs[2].AssignedInstance != reqs[0].AssignedInstance {
			t.Errorf("seed %d: the second adapter request went to %s, not %s where the adapter's blocks are",
				seed, reqs[2].AssignedInstance, reqs[0].AssignedInstance)
		}
		if reqs[1].AssignedInstance == reqs[0].AssignedInstance {
			baseOnFirst++
		}
	}
	// The base request has no hit anywhere, so it lands on the adapter's endpoint by chance only.
	if baseOnFirst == seeds {
		t.Errorf("the base-model request landed on the adapter's endpoint in all %d runs: it was scored a hit on the adapter's blocks", seeds)
	}
}
