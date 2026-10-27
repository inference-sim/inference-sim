package llmd

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// evictionConfig is flow control with in-flight eviction (enableEviction): when higher-priority
// requests are blocked by pool saturation, in-flight sheddable (priority < 0) requests are
// revoked to reclaim capacity.
const evictionConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: queue-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
  - pluginRef: queue-scorer
flowControl:
  enableEviction: true
`

// pdEvictionConfig is pdConfig with flow control and in-flight eviction.
var pdEvictionConfig = "featureGates:\n- flowControl\nflowControl:\n  enableEviction: true\n" + pdConfig

// pdPreciseEvictionConfig is pdEvictionConfig with the precise prefix cache (the engines' KV
// events) in place of the approximate one.
var pdPreciseEvictionConfig = strings.Replace(pdEvictionConfig, `- type: approx-prefix-cache-producer
  parameters:
    maxPrefixTokensToMatch: 16384
    lruCapacityPerServer: 31250
- type: prefix-cache-scorer
`, `- type: token-producer
  parameters:
    modelName: test-model
    vllm:
      url: http://render.blis.local:8000
- type: precise-prefix-cache-producer
  parameters:
    tokenProcessorConfig:
      blockSizeTokens: 16
    kvEventsConfig:
      zmqEndpoint: tcp://0.0.0.0:5557
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
`, 1)

// victim is what an evicted request held on its engine just before and just after its eviction.
type victim struct {
	running              bool  // it had computed tokens
	blocksBefore, after  int64 // the engine's used KV blocks
	pinnedBefore, pinned bool  // its adapter pin
	revoked              int
	// pending is set for a decode reservation still waiting for its KV transfer; held* report
	// whether its decode engine's cache and the router's index for that pod held the shared
	// prefix's first block before and after the eviction.
	pending                          bool
	heldEngine, heldRouter           bool
	stillHeldEngine, stillHeldRouter bool
}

type evictionSetup struct {
	config   string
	pd       bool   // 2 prefill + 2 decode pods
	adapter  string // the sheddable requests' LoRA adapter
	capacity int    // adapter slots per engine
	// kvBound runs every sheddable request at once on a small KV cache, so the pool saturates on
	// KV utilization and the victims are running.
	kvBound bool
	// sharedPrefix gives every sheddable request one prefix of that many tokens, so a revoked
	// transfer's reservation duplicates a prefix its decode engine already caches.
	sharedPrefix int
	// transferUs and criticalAt, when set, are the P/D KV transfer time and the critical burst's
	// start (µs).
	transferUs, criticalAt int64
	// sheddable and sheddableGapUs, when set, are the number of sheddable requests and their
	// spacing (default 40, 1 ms apart).
	sheddable      int
	sheddableGapUs int64
}

// runEviction runs long sheddable requests that saturate the pool, then a burst of critical ones.
func runEviction(t *testing.T, s evictionSetup) (runResult, map[string]*victim) {
	t.Helper()
	var reqs []*sim.Request
	n, gap := 40, int64(1000)
	if s.sheddable > 0 {
		n, gap = s.sheddable, s.sheddableGapUs
	}
	for i, r := range testRequests(n, 5, gap, s.sharedPrefix) {
		r.ID, r.SLOClass, r.Adapter = fmt.Sprintf("sheddable_%d", i), "sheddable", s.adapter
		r.OutputTokens = sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(int64(i))), 2000) // long-running
		r.MaxOutputLen = len(r.OutputTokens)
		reqs = append(reqs, r)
	}
	for i, r := range testRequests(40, 6, 500, 0) {
		r.ID, r.SLOClass = fmt.Sprintf("critical_%d", i), "critical"
		r.ArrivalTime += 200_000
		if s.criticalAt > 0 {
			r.ArrivalTime += s.criticalAt - 200_000
		}
		if s.kvBound {
			r.ArrivalTime += 600_000 // once the running requests' KV has grown
		}
		reqs = append(reqs, r)
	}
	dep := testDeployment(2)
	labels := func(cluster.InstanceID, cluster.PoolRole) map[string]string { return map[string]string{"app": "vllm"} }
	if s.pd {
		dep = testDeployment(4)
		dep.PrefillInstances, dep.DecodeInstances = 2, 2
		dep.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 + tokens/50 }
		if s.transferUs > 0 {
			dep.PDTransferTime = func(int64, cluster.InstanceID, cluster.InstanceID) int64 { return s.transferUs }
		}
		labels = func(id cluster.InstanceID, role cluster.PoolRole) map[string]string {
			l := map[string]string{"app": "vllm"}
			for k, v := range roleLabels(id, role) {
				l[k] = v
			}
			return l
		}
	}
	dep.BatchConfig = sim.NewBatchConfig(8, 2048, 0)
	if s.kvBound {
		dep.BatchConfig = sim.NewBatchConfig(64, 8192, 0)
		dep.KVCacheConfig = sim.NewKVCacheConfig(800, 16, 0, 0, 0, 0)
	}
	if s.adapter != "" {
		capacity, base, bw, fp := s.capacity, 1000.0, 2.0e6, 2.0e6
		dep.LoRAConfig = sim.LoRAConfig{AdapterCapacity: &capacity, LoadBaseLatencyUs: &base,
			LoadBandwidthBytesUs: &bw, FootprintBytesPerRank: &fp, Adapters: []sim.AdapterSpec{{ID: s.adapter, Rank: 8}}}
	}
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{
		Model:     "test-model",
		EPPFlags:  []string{"--pool-name=vllm", "--config-text=" + s.config},
		Manifests: []byte(poolManifests),
		PodLabels: labels,
	})
	byID := map[string]*sim.Request{}
	for _, r := range reqs {
		byID[r.ID] = r
	}
	// engine is the request (or, disaggregated, its active sub-request) on an engine, and that engine.
	engine := func(r *sim.Request) (*sim.Request, *cluster.InstanceSimulator) {
		at := r.AssignedInstance
		for _, p := range cs.ParentRequests() {
			if p.ID == r.ID {
				switch {
				case p.DecodeSubReq != nil:
					r, at = p.DecodeSubReq, string(p.DecodeInstanceID) // reserved there, or decoding
				case p.PrefillSubReq != nil:
					r, at = p.PrefillSubReq, p.PrefillSubReq.AssignedInstance
				}
			}
		}
		for _, inst := range cs.Instances() {
			if string(inst.ID()) == at {
				return r, inst
			}
		}
		return r, nil
	}
	// prefixHeld reports whether the decode engine of a pending reservation, and the router's index
	// for its pod, hold the shared prefix's first block.
	prefixHeld := func(r *sim.Request) (pending, engineHas, routerHas bool) {
		if s.sharedPrefix < 16 {
			return false, false, false
		}
		first := sim.HashBlock("", reqs[0].InputTokens[:16])
		for _, p := range cs.ParentRequests() {
			if p.ID != r.ID || p.DecodeSubReq == nil || p.DecodeSubReq.State != sim.StateWaitingForRemoteKVs {
				continue
			}
			for _, inst := range cs.Instances() {
				if inst.ID() != p.DecodeInstanceID {
					continue
				}
				for _, h := range inst.CachedBlockHashes() {
					engineHas = engineHas || h == first
				}
				got := kvBlocks(t, gw, inst.ID(), "")
				return true, engineHas, got[engineKey(first)]
			}
		}
		return false, false, false
	}
	pendingVictims := map[string]*cluster.ParentRequest{}
	victims := map[string]*victim{}
	gw.observe = func(kind, id string) {
		switch kind {
		case "evict":
			v := victims[id]
			if v == nil {
				v = &victim{}
				victims[id] = v
			}
			r, inst := engine(byID[id])
			v.running, v.pinnedBefore = r.ProgressIndex > 0, r.AdapterPinned()
			if inst != nil {
				v.blocksBefore = inst.UsedKVBlocks()
			}
			v.pending, v.heldEngine, v.heldRouter = prefixHeld(byID[id])
			if v.pending {
				for _, p := range cs.ParentRequests() {
					if p.ID == id {
						pendingVictims[id] = p
					}
				}
			}
		case "evicted":
			v := victims[id]
			v.revoked++
			r, inst := engine(byID[id])
			v.pinned = r.AdapterPinned()
			if inst != nil {
				v.after = inst.UsedKVBlocks()
			}
			if p := pendingVictims[id]; p != nil && s.sharedPrefix >= 16 {
				// Its reservation is released: look at its decode engine directly.
				first := sim.HashBlock("", reqs[0].InputTokens[:16])
				for _, inst := range cs.Instances() {
					if inst.ID() != p.DecodeInstanceID {
						continue
					}
					for _, h := range inst.CachedBlockHashes() {
						v.stillHeldEngine = v.stillHeldEngine || h == first
					}
					got := kvBlocks(t, gw, inst.ID(), "")
					v.stillHeldRouter = got[engineKey(first)]
				}
			}
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	for id, v := range victims {
		if v.revoked == 0 {
			delete(victims, id) // signalled, but it completed before it could be revoked
		}
	}
	return runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}, victims
}

// checkEviction: victims are dispatched sheddable requests, each revoked once and counted
// gateway-evicted (INV-1); a running victim's KV blocks and adapter pin are released at its
// eviction; the router's response lifecycle ends for every request; the run is deterministic.
func checkEviction(t *testing.T, s evictionSetup) map[string]*victim {
	t.Helper()
	res, victims := runEviction(t, s)
	assertConservation(t, res, len(res.reqs))
	if res.cs.GatewayEvicted() == 0 {
		t.Fatal("no request was evicted in flight; the test does not exercise the path")
	}
	if len(victims) != res.cs.GatewayEvicted() {
		t.Errorf("%d requests revoked, %d counted gateway-evicted", len(victims), res.cs.GatewayEvicted())
	}
	running := 0
	for _, r := range res.reqs {
		v := victims[r.ID]
		if v == nil {
			continue
		}
		if v.revoked != 1 || r.SLOClass != "sheddable" {
			t.Errorf("%s (class %s) revoked %d times; want each victim a sheddable request, once", r.ID, r.SLOClass, v.revoked)
		}
		if v.running {
			running++
			if v.after >= v.blocksBefore {
				t.Errorf("%s: running victim's engine holds %d KV blocks after its eviction, %d before", r.ID, v.after, v.blocksBefore)
			}
		}
		if v.pinned || (s.adapter != "" && v.running && !v.pinnedBefore) {
			t.Errorf("%s: adapter pin before eviction %v, after %v; want a running victim's pin released", r.ID, v.pinnedBefore, v.pinned)
		}
	}
	if running == 0 {
		t.Error("no running victim was evicted; the check of released resources would be vacuous")
	}
	for _, inst := range res.cs.Instances() {
		if u := inst.UsedKVBlocks(); u != 0 {
			t.Errorf("%s: %d KV blocks used at the end", inst.ID(), u)
		}
		// The engines published their state after the last eviction or completion (vLLM's empty
		// step after an abort): the router's scrapes see the requests gone.
		if m := inst.EngineMetricsAt(math.MaxInt64); m.Running != 0 || m.Waiting != 0 || m.KVUsage != 0 {
			t.Errorf("%s: last published running %d, waiting %d, KV usage %g", inst.ID(), m.Running, m.Waiting, m.KVUsage)
		}
		// With the precise prefix cache, the router indexes exactly what each engine caches: no
		// block of a cancelled transfer.
		if res.gw.router.WantsKVEvents() {
			got := kvBlocks(t, res.gw, inst.ID(), "")
			want := inst.CachedBlockHashes()
			if len(got) != len(want) {
				t.Errorf("%s: the router indexes %d blocks, the engine caches %d", inst.ID(), len(got), len(want))
			}
			for _, h := range want {
				if !got[engineKey(h)] {
					t.Errorf("%s: cached block %s is not in the router's index", inst.ID(), h[:12])
					break
				}
			}
		}
	}
	if res.gw.flightsDispatched == 0 || res.gw.flightsEnded != res.gw.flightsDispatched {
		t.Errorf("the router was handed %d flights and %d were ended; every dispatched request's response must end once",
			res.gw.flightsDispatched, res.gw.flightsEnded)
	}
	if res.gw.LiveCoroutines() != 0 || len(res.gw.flights) != 0 {
		t.Errorf("%d coroutines, %d flights left after the run", res.gw.LiveCoroutines(), len(res.gw.flights))
	}
	again, victims2 := runEviction(t, s)
	if len(victims2) != len(victims) || again.metrics.CompletedRequests != res.metrics.CompletedRequests {
		t.Fatalf("two identical runs differ: %d vs %d evicted", len(victims), len(victims2))
	}
	for i := range res.reqs {
		if res.reqs[i].AssignedInstance != again.reqs[i].AssignedInstance || res.reqs[i].GatewayDispatchTime != again.reqs[i].GatewayDispatchTime {
			t.Fatalf("%s: placed differently in two identical runs", res.reqs[i].ID)
		}
	}
	t.Logf("evicted %d sheddable requests in flight (%d running); completed %d", len(victims), running, res.metrics.CompletedRequests)
	return victims
}

// TestFlowControl_InFlightEviction (Discussion #1933 §5.7): with enableEviction the router revokes
// in-flight sheddable requests while critical ones wait on the saturated pool; running victims,
// pinning a LoRA adapter, release their KV blocks and pin when revoked.
func TestFlowControl_InFlightEviction(t *testing.T) {
	checkEviction(t, evictionSetup{config: evictionConfig, adapter: "a0", capacity: 2, kvBound: true})
}

// TestFlowControl_InFlightEvictionDisaggregated: a disaggregated victim is revoked at the stage
// it is in (its prefill, its KV transfer, or its decode), releasing what that stage holds, once.
func TestFlowControl_InFlightEvictionDisaggregated(t *testing.T) {
	checkEviction(t, evictionSetup{config: pdEvictionConfig, pd: true})
}

// TestFlowControl_InFlightEvictionDisaggregatedPrecise: as above, with the precise prefix cache:
// a transfer evicted before its KV arrived leaves no block in the decode engine's cache, and so
// none in the router's index.
func TestFlowControl_InFlightEvictionDisaggregatedPrecise(t *testing.T) {
	checkEviction(t, evictionSetup{config: pdPreciseEvictionConfig, pd: true})
}

// TestFlowControl_InFlightEvictionDisaggregatedSharedPrefix: with a prefix every sheddable request
// shares, revoked transfers reserve blocks for a prefix their decode engine already caches;
// revoking them keeps that prefix cached, in the engine and in the router's index (which
// checkEviction compares block for block).
func TestFlowControl_InFlightEvictionDisaggregatedSharedPrefix(t *testing.T) {
	// 120 sheddable requests 2 ms apart, 100 ms transfers, the critical burst at 200 ms: the
	// earliest transfers have completed (their decode engines cache the prefix), later ones are
	// still pending when they are revoked.
	victims := checkEviction(t, evictionSetup{config: pdPreciseEvictionConfig, pd: true, sharedPrefix: 256,
		transferUs: 100_000, criticalAt: 200_000, sheddable: 120, sheddableGapUs: 2000})
	duplicates := 0
	for id, v := range victims {
		if !v.pending || !v.heldEngine || !v.heldRouter {
			continue
		}
		duplicates++ // a pending duplicate of a prefix its decode engine (and the router) already held
		if !v.stillHeldEngine || !v.stillHeldRouter {
			t.Errorf("%s: after revoking its pending transfer, the shared prefix is cached on its decode engine %v, indexed by the router %v; want both",
				id, v.stillHeldEngine, v.stillHeldRouter)
		}
	}
	if duplicates == 0 {
		t.Fatal("no pending reservation duplicating a cached prefix was revoked; the test does not exercise the path")
	}
	t.Logf("%d pending duplicate transfers revoked", duplicates)
}
