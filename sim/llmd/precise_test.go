package llmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"sigs.k8s.io/yaml"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// preciseConfig is llm-d's deploy/config/sim-epp-kvcache-config.yaml with its two environment
// variables substituted (the model, and a render URL the simulated model servers answer), plus
// any extra producer parameters.
func preciseConfig(extra string) string {
	return `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: token-producer
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
    indexerConfig:
      kvBlockIndexConfig:
        enableMetrics: false
        metricsLoggingInterval: 60000000000
` + extra + `- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 10
`
}

type preciseRun struct {
	reqs []*sim.Request
	cs   *cluster.ClusterSimulator
	gw   *Gateway
}

func runPrecise(t *testing.T, dep cluster.DeploymentConfig, reqs []*sim.Request, config string, opts Options) preciseRun {
	t.Helper()
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
	opts.Model = "test-model"
	if opts.EPPFlags == nil {
		opts.EPPFlags = append(DefaultEPPFlags(), "--config-text="+config)
	}
	gw := New(opts)
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	return preciseRun{reqs: reqs, cs: cs, gw: gw}
}

// prefixGroups returns n requests per group, each group sharing one prefixLen-token prefix,
// groups interleaved, gapUs apart.
func prefixGroups(groups, n int, prefixLen int, gapUs int64) []*sim.Request {
	var reqs []*sim.Request
	for g := 0; g < groups; g++ {
		for i, r := range testRequests(n, int64(50+g), 0, prefixLen) {
			r.ID = fmt.Sprintf("g%d_%d", g, i)
			r.ArrivalTime = int64(i*groups+g) * gapUs
			r.PrefixGroup = fmt.Sprintf("g%d", g)
			reqs = append(reqs, r)
		}
	}
	return reqs
}

// TestPrecisePrefix_RoutesToTheInstanceHoldingTheBlocks: with the engines' KV events, requests
// sharing a prefix go to the instance whose cache holds it, group by group.
func TestPrecisePrefix_RoutesToTheInstanceHoldingTheBlocks(t *testing.T) {
	run := runPrecise(t, testDeployment(4), prefixGroups(3, 15, 256, 30000), preciseConfig(""), Options{})
	for g := 0; g < 3; g++ {
		where := map[string]int{}
		for _, r := range run.reqs {
			if r.PrefixGroup == fmt.Sprintf("g%d", g) {
				where[r.AssignedInstance]++
			}
		}
		if len(where) != 1 {
			t.Errorf("prefix group %d spread over %v; precise prefix routing keeps it where its blocks are", g, where)
		}
	}
	if len(run.gw.kvIngest) != 1 {
		t.Fatalf("%d precise-prefix producers receive KV events, want 1", len(run.gw.kvIngest))
	}
}

// TestPrecisePrefix_EventDelay: the router learns of cached blocks only when their events arrive.
// With no delay a group's later requests follow the first; with a delay longer than the run the
// router never sees a block and the prefix scorer cannot keep the group together.
func TestPrecisePrefix_EventDelay(t *testing.T) {
	spread := func(delay int64) int {
		run := runPrecise(t, testDeployment(4), prefixGroups(1, 20, 256, 30000), preciseConfig(""), Options{KVEventDelayUs: delay})
		where := map[string]bool{}
		for _, r := range run.reqs {
			where[r.AssignedInstance] = true
		}
		return len(where)
	}
	if prompt, late := spread(0), spread(3_600_000_000); prompt != 1 || late < 2 {
		t.Errorf("instances used: %d with no delay (want 1), %d with a one-hour delay (want several)", prompt, late)
	}
}

// TestPrecisePrefix_FollowsEvictions: when small caches evict prefixes, the precise index (fed by
// removed-block events) knows which instance still holds a prefix, where the approximate index
// keeps believing in blocks long evicted; precise routing then serves more prompts from cache.
func TestPrecisePrefix_FollowsEvictions(t *testing.T) {
	const approx = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
- type: prefix-cache-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 10
`
	ttft := func(config string) float64 {
		dep := testDeployment(3)
		dep.KVCacheConfig = sim.NewKVCacheConfig(120, 16, 0, 0, 0, 0) // a few prompts' worth per instance
		run := runPrecise(t, dep, prefixGroups(12, 6, 512, 20000), config, Options{})
		m := run.cs.AggregatedMetrics()
		var sum float64
		for _, v := range m.RequestTTFTs {
			sum += v
		}
		return sum / float64(len(m.RequestTTFTs))
	}
	precise, approximate := ttft(preciseConfig("")), ttft(approx)
	t.Logf("mean TTFT: precise %.0f us, approximate %.0f us", precise, approximate)
	if precise >= approximate {
		t.Errorf("precise prefix routing (%.0f us) did not beat the approximate index (%.0f us) under evictions", precise, approximate)
	}
}

// TestPrecisePrefix_IndexFollowsTheEnginesCache: every block an engine still caches at the end of a
// run with no event delay is in the router's index for that pod, and nothing else is.
func TestPrecisePrefix_IndexFollowsTheEnginesCache(t *testing.T) {
	dep := testDeployment(2)
	dep.KVCacheConfig = sim.NewKVCacheConfig(200, 16, 0, 0, 0, 0)
	run := runPrecise(t, dep, prefixGroups(6, 5, 128, 15000), preciseConfig(""), Options{})
	for _, inst := range run.cs.Instances() {
		ep := run.gw.endpoints[inst.ID()]
		got, err := run.gw.kvIngest[0].HostedPodBlocks(ep.ip + ":" + fmt.Sprint(ep.port))
		if err != nil {
			t.Fatal(err)
		}
		want := inst.CachedBlockHashes()
		if len(want) == 0 {
			t.Fatalf("%s caches no block; the check would be vacuous", inst.ID())
		}
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

// TestPrecisePrefix_SampleConfigs: llm-d's own precise-prefix sample configurations either run
// (environment variables substituted, as kind-dev-env.sh does) or are refused with the reason.
func TestPrecisePrefix_SampleConfigs(t *testing.T) {
	dir := forkConfigDir(t)
	for _, tc := range []struct {
		file       string
		blockSize  int64
		wantRefuse string
	}{
		{"epp-precise-prefix-cache-config.yaml", 64, ""},
		{"sim-epp-kvcache-config.yaml", 16, ""},
		{"sim-epp-tokenizer-vllm-http-config.yaml", 16, ""},
		{"sim-epp-no-hit-lru.yaml", 5, ""},
		{"sim-epp-no-hit-lru.yaml", 16, "cuts block keys every 5 tokens"}, // on engines with another block size
	} {
		t.Run(fmt.Sprintf("%s/%d-token-blocks", tc.file, tc.blockSize), func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(dir, tc.file))
			if err != nil {
				t.Fatal(err)
			}
			config := strings.NewReplacer("${MODEL_NAME}", "test-model", "${VLLM_RENDER_URL}", "http://render.blis.local:8000",
				"hf-repo/model-name", "test-model", "http://localhost:8000", "http://render.blis.local:8000").Replace(string(raw))
			dep := testDeployment(2)
			dep.KVCacheConfig = sim.NewKVCacheConfig(10000, tc.blockSize, 0, 0, 0, 0)
			cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(prefixGroups(2, 5, 128, 20000)), nil)
			cs.SetGateway(New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)}))
			err = cs.Run()
			switch {
			case tc.wantRefuse == "" && err != nil:
				t.Fatalf("want it to run, got %v", err)
			case tc.wantRefuse != "" && (err == nil || !strings.Contains(err.Error(), tc.wantRefuse)):
				t.Fatalf("want a refusal containing %q, got %v", tc.wantRefuse, err)
			}
		})
	}
}

// forkConfigDir is deploy/config of the router module BLIS builds against.
func forkConfigDir(t *testing.T) string {
	t.Helper()
	out, err := goListModuleDir("github.com/llm-d/llm-d-router")
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(out, "deploy", "config")
}

// TestPrecisePrefix_ClearIsOrderedWithDelayedBatches: when an instance terminates while its KV
// event batches are still in flight (a delivery delay), the clear arrives after every batch the
// instance published before terminating, nothing of it is delivered after the clear, and the
// router's index for its pod ends empty.
func TestPrecisePrefix_ClearIsOrderedWithDelayedBatches(t *testing.T) {
	reqs := prefixGroups(1, 10, 256, 30000) // one group: it stays on one instance
	late := testRequests(1, 99, 0, 0)[0]
	late.ID, late.ArrivalTime = "late", 2_000_000
	after := testRequests(1, 98, 0, 0)[0] // keeps the run going past the clear (deliveries are daemon events)
	after.ID, after.ArrivalTime = "after", 6_000_000
	reqs = append(reqs, late, after)
	cs := cluster.NewClusterSimulator(testDeployment(2), cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", KVEventDelayUs: 1_900_000, EPPFlags: append(DefaultEPPFlags(), "--config-text="+preciseConfig(""))})
	var victim cluster.InstanceID
	var events []string
	gw.observe = func(kind, detail string) {
		switch kind {
		case "request":
			if detail == "late" {
				victim = cluster.InstanceID(reqs[0].AssignedInstance)
				events = append(events, "terminate")
				for _, inst := range cs.Instances() {
					if inst.ID() == victim {
						inst.TransitionTo(sim.InstanceStateTerminated)
					}
				}
			}
		case "kvdeliver":
			if victim != "" && strings.HasPrefix(detail, string(victim)+" ") {
				events = append(events, "deliver")
			}
		case "kvclear":
			events = append(events, "clear "+detail)
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	seq := strings.Join(events, ",")
	if !strings.HasPrefix(seq, "terminate,deliver") || !strings.HasSuffix(seq, ",clear "+string(victim)) {
		t.Fatalf("after termination: %s; want the in-flight batches delivered, then the clear, then nothing", seq)
	}
	ep := gw.endpoints[victim]
	if got, err := gw.kvIngest[0].HostedPodBlocks(ep.ip + ":" + fmt.Sprint(ep.port)); err != nil || len(got) != 0 {
		t.Fatalf("the terminated pod still indexes %d blocks (%v)", len(got), err)
	}
}

// TestPrecisePrefix_PerfHelmConfig: the EndpointPickerConfig and EPP flags embedded in llm-d's
// test/perf/config/router-configs/precise-prefix.yaml (Helm values) run unchanged: endpoint
// notification source, pod discovery settings (inert: the simulated pods are given, not
// discovered), speculative indexing, and the active-request, KV-utilization, prefix and no-hit-LRU
// scorers.
func TestPrecisePrefix_PerfHelmConfig(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(forkConfigDir(t), "..", "..", "test", "perf", "config", "router-configs", "precise-prefix.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var values struct {
		Router struct {
			EPP struct {
				Flags               map[string]any    `json:"flags"`
				PluginsConfigFile   string            `json:"pluginsConfigFile"`
				PluginsCustomConfig map[string]string `json:"pluginsCustomConfig"`
			} `json:"epp"`
		} `json:"router"`
	}
	if err := yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	epp := values.Router.EPP
	config := epp.PluginsCustomConfig[epp.PluginsConfigFile]
	if !strings.Contains(config, "precise-prefix-cache-producer") || !strings.Contains(config, "speculativeIndexing: true") {
		t.Fatalf("the embedded config %q was not found or changed shape", epp.PluginsConfigFile)
	}
	flags := DefaultEPPFlags()
	for name, v := range epp.Flags {
		flags = append(flags, fmt.Sprintf("--%s=%v", name, v))
	}
	sort.Strings(flags[2:])
	flags = append(flags, "--config-text="+config)
	run := runPrecise(t, testDeployment(3), prefixGroups(3, 8, 256, 20000), "", Options{EPPFlags: flags})
	if m := run.cs.AggregatedMetrics(); m.CompletedRequests != len(run.reqs) {
		t.Fatalf("completed %d of %d", m.CompletedRequests, len(run.reqs))
	}
}

// TestPrecisePrefix_IndexFollowsEveryTier: with a CPU tier (the legacy tiered cache, or the
// offload chain), the router's index holds, per pod and per tier, exactly the blocks the engine
// holds there: its GPU prefix cache, and the blocks ready on its CPU tier.
func TestPrecisePrefix_IndexFollowsEveryTier(t *testing.T) {
	offload := sim.KVOffloadConfig{
		Enabled: true, CPUBytesToUse: 300 * 1000, PerBlockBytes: 1000,
		BlockSize: 16, BlocksPerChunk: 1, TokensPerHash: 16, EvictionPolicy: "lru", OffloadPromptOnly: true,
	}
	for name, kv := range map[string]sim.KVCacheConfig{
		"tiered":  sim.NewKVCacheConfig(120, 16, 300, 0.5, 1e6, 0),
		"offload": sim.NewKVCacheConfig(120, 16, 0, 0, 0, 0, sim.WithKVOffload(offload)),
	} {
		t.Run(name, func(t *testing.T) {
			dep := testDeployment(2)
			dep.KVCacheConfig = kv
			run := runPrecise(t, dep, prefixGroups(10, 4, 256, 15000), preciseConfig(""), Options{})
			cpuBlocks := 0
			for _, inst := range run.cs.Instances() {
				ep := run.gw.endpoints[inst.ID()]
				pod := ep.ip + ":" + fmt.Sprint(ep.port)
				for tier, want := range map[string][]string{"gpu": inst.CachedBlockHashes(), "cpu": inst.CPUBlockHashes()} {
					got, err := run.gw.kvIngest[0].HostedPodTierBlocks(pod, tier)
					if err != nil {
						t.Fatal(err)
					}
					if tier == "cpu" {
						cpuBlocks += len(want)
					}
					if len(got) != len(want) {
						t.Errorf("%s %s: the router indexes %d blocks, the engine holds %d", inst.ID(), tier, len(got), len(want))
						continue
					}
					for _, h := range want {
						if !got[engineKey(h)] {
							t.Errorf("%s %s: block %s is not in the router's index", inst.ID(), tier, h[:12])
							break
						}
					}
				}
			}
			if cpuBlocks == 0 {
				t.Fatal("no block reached a CPU tier; the check would be vacuous")
			}
		})
	}
}

// TestPrecisePrefix_CachedBlockSurvivesCanonicalizerChurn (review 6205960463): unique prompts
// churn the prompt canonicalizer far past its capacity while a shared prefix stays cached on an
// engine; a later request with that prefix still matches it in the router's index and goes there.
func TestPrecisePrefix_CachedBlockSurvivesCanonicalizerChurn(t *testing.T) {
	first := prefixGroups(1, 1, 256, 0)[0]
	var reqs []*sim.Request
	reqs = append(reqs, first)
	for i, r := range testRequests(40, 77, 20000, 0) { // ~6 unique blocks each: ~240 chains
		r.ID = fmt.Sprintf("unique_%d", i)
		r.ArrivalTime += 100000
		reqs = append(reqs, r)
	}
	again := prefixGroups(1, 2, 256, 0)[1] // same prefix as first, new suffix
	again.ID, again.ArrivalTime = "again", 2_000_000
	reqs = append(reqs, again)

	cs := cluster.NewClusterSimulator(testDeployment(4), cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+preciseConfig(""))})
	gw.canonCapacity = 40 // far below the chains the unique prompts introduce
	var score string
	gw.observe = func(kind, detail string) {
		if kind == "score" && strings.HasPrefix(detail, "again ") {
			score = detail
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if again.AssignedInstance != first.AssignedInstance || strings.HasSuffix(score, " 0") {
		t.Fatalf("the repeated prefix went to %s (score %q), not %s where it is cached", again.AssignedInstance, score, first.AssignedInstance)
	}
}

// TestPrecisePrefix_DelayedRemovalKeepsTheRouterKey (review 6206561059): an engine evicts a cached
// prefix while unique prompts churn the prompt canonicalizer far past its capacity, and the
// removal's delivery is delayed. Until it arrives the router still indexes the prefix, so a
// matching request still matches it; after it arrives, one no longer does. The probes are too
// long ever to fit the engine's cache, so they are routed and then dropped, caching nothing.
func TestPrecisePrefix_DelayedRemovalKeepsTheRouterKey(t *testing.T) {
	const delay = 1_000_000
	group := prefixGroups(1, 1, 256, 0)
	reqs := []*sim.Request{group[0]}
	for i, r := range testRequests(12, 77, 20000, 320) { // each needs most of the cache
		r.ID = fmt.Sprintf("unique_%d", i)
		r.ArrivalTime += 2_000_000 // after the prefix's store has reached the router
		reqs = append(reqs, r)
	}
	probe := func(id string, at int64) *sim.Request {
		r := *group[0]
		r.ID, r.ArrivalTime, r.State = id, at, sim.StateQueued
		r.InputTokens = append(append([]sim.TokenID{}, group[0].InputTokens[:256]...), make([]sim.TokenID, 700)...)
		return &r
	}
	reqs = append(reqs, probe("before", 2_500_000), probe("after", 4_500_000))

	dep := testDeployment(1)
	dep.KVCacheConfig = sim.NewKVCacheConfig(40, 16, 0, 0, 0, 0)
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", KVEventDelayUs: delay, EPPFlags: append(DefaultEPPFlags(), "--config-text="+preciseConfig(""))})
	gw.canonCapacity = 20
	var prefixBlocks []string
	scores := map[string]string{}
	gw.observe = func(kind, detail string) {
		switch kind {
		case "request":
			inst := cs.Instances()[0]
			switch detail {
			case "unique_0":
				prefixBlocks = inst.CachedBlockHashes() // the first request's blocks
			case "before":
				cached := map[string]bool{}
				for _, h := range inst.CachedBlockHashes() {
					cached[h] = true
				}
				for _, h := range prefixBlocks {
					if cached[h] {
						t.Errorf("the engine still caches the prefix when %q arrives; the check would be vacuous", detail)
						break
					}
				}
			}
		case "score":
			f := strings.Fields(detail)
			scores[f[0]] = f[2]
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(prefixBlocks) == 0 {
		t.Fatal("the first request cached nothing")
	}
	if scores["before"] == "0" || scores["after"] != "0" {
		t.Fatalf("prefix score before the removal arrives %s (want > 0), after %s (want 0)", scores["before"], scores["after"])
	}
	if m := cs.AggregatedMetrics(); m.DroppedUnservable != 2 {
		t.Fatalf("%d requests dropped as unservable, want the 2 probes", m.DroppedUnservable)
	}
}

// TestPrecisePrefix_IndexFollowsAFilesystemTier (review 6206521350): with a filesystem offload
// tier, prefixes cycle through GPU and CPU eviction, survive on the tier, and are promoted back
// to CPU. When the tier publishes KV events (vLLM's per-tier enable_kv_events; medium
// shared_storage, as llm-d's filesystem backend), the router's index holds, per pod and per tier,
// exactly the blocks the engine holds there, and a recurring prefix is routed to the pod whose
// tiers hold it; when it does not, the router indexes nothing on the tier and never a block the
// engine does not hold.
func TestPrecisePrefix_IndexFollowsAFilesystemTier(t *testing.T) {
	for _, events := range []bool{true, false} {
		t.Run(fmt.Sprintf("enable_kv_events=%v", events), func(t *testing.T) {
			off := sim.KVOffloadConfig{
				Enabled: true, CPUBytesToUse: 24 * 4096, PerBlockBytes: 4096,
				BlockSize: 16, BlocksPerChunk: 1, TokensPerHash: 16, EvictionPolicy: "lru", OffloadPromptOnly: true,
				Tiers: []sim.KVOffloadTier{{
					Type: "fs", RootDir: "/mnt", NReadThreads: 4, NWriteThreads: 4, EnableKVEvents: events,
					DirectIO: true, ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
				}},
			}
			dep := testDeployment(2)
			dep.KVCacheConfig = sim.NewKVCacheConfig(64, 16, 0, 0, 0, 0, sim.WithKVOffload(off)) // small GPU
			reqs := prefixGroups(16, 4, 48, 2000)                                                // 16 three-block prefixes, cycled 4 times
			cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
			gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+preciseConfig(""))})
			cpuGone := map[string]bool{} // instance+hash whose CPU copy the router saw removed
			promoted := 0
			scores := map[string]string{}
			gw.observe = func(kind, detail string) {
				if kind == "score" {
					f := strings.Fields(detail)
					scores[f[0]] = f[2]
				}
				if kind != "kvevent" {
					return
				}
				f := strings.Fields(detail)
				key := f[0] + " " + f[2]
				switch {
				case f[1] == `"cpu"` && f[3] == "true":
					cpuGone[key] = true
				case f[1] == `"cpu"` && cpuGone[key]:
					promoted++ // back on CPU after leaving it: read from the fs tier
					delete(cpuGone, key)
				}
			}
			cs.SetGateway(gw)
			if err := cs.Run(); err != nil {
				t.Fatalf("Run: %v", err)
			}
			fsBlocks := 0
			for _, inst := range cs.Instances() {
				ep := gw.endpoints[inst.ID()]
				pod := ep.ip + ":" + fmt.Sprint(ep.port)
				want := map[string][]string{"gpu": inst.CachedBlockHashes(), "cpu": inst.CPUBlockHashes(), "shared_storage": inst.TierBlockHashes("shared_storage")}
				fsBlocks += len(want["shared_storage"])
				if !events {
					want["shared_storage"] = nil
				}
				for _, tier := range []string{"gpu", "cpu", "shared_storage"} {
					got, err := gw.kvIngest[0].HostedPodTierBlocks(pod, tier)
					if err != nil {
						t.Fatal(err)
					}
					if !events {
						// The router loses a block whose only copy is on the silent tier: its
						// promotion back to CPU is a location-only store of a key the router no
						// longer maps (llm-d's handleDeviceTierUpdate skips it). It never indexes a
						// block the engine does not hold.
						engine := map[uint64]bool{}
						for _, h := range want[tier] {
							engine[engineKey(h)] = true
						}
						for k := range got {
							if !engine[k] {
								t.Errorf("%s %s: the router indexes a block the engine does not hold", inst.ID(), tier)
								break
							}
						}
						t.Logf("%s %s: the router indexes %d of the engine's %d blocks", inst.ID(), tier, len(got), len(want[tier]))
						continue
					}
					if len(got) != len(want[tier]) {
						t.Errorf("%s %s: the router indexes %d blocks, want %d", inst.ID(), tier, len(got), len(want[tier]))
						continue
					}
					for _, h := range want[tier] {
						if !got[engineKey(h)] {
							t.Errorf("%s %s: block %s is not in the router's index", inst.ID(), tier, h[:12])
							break
						}
					}
				}
			}
			if fsBlocks == 0 {
				t.Fatal("no block reached the fs tier; the check would be vacuous")
			}
			if !events {
				return
			}
			if promoted == 0 {
				t.Fatal("no block was promoted back to CPU; the check would be vacuous")
			}
			// By the last cycle every prefix has run before, and the fs tier keeps what it is given:
			// the router finds each one on some tier of the pod that holds it.
			for _, r := range reqs {
				if strings.HasSuffix(r.ID, "_3") && scores[r.ID] == "0" {
					t.Errorf("%s: no pod's index holds its prefix (score 0)", r.ID)
				}
			}
		})
	}
}

// TestPrecisePrefix_NoEventsWithPrefixCachingDisabled: an engine launched without prefix caching
// caches no block for reuse and, as vLLM, publishes no KV event; the router's index stays empty
// and no request is scored for a hit the engine would not take.
func TestPrecisePrefix_NoEventsWithPrefixCachingDisabled(t *testing.T) {
	dep := testDeployment(2)
	dep.BatchConfig = sim.NewBatchConfig(256, 2048, 0, sim.WithPrefixCachingDisabled(true))
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(prefixGroups(2, 8, 256, 20000)), nil)
	gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+preciseConfig(""))})
	delivered := 0
	gw.observe = func(kind, detail string) {
		switch kind {
		case "kvdeliver":
			delivered++
		case "score":
			if !strings.HasSuffix(detail, " 0") {
				t.Errorf("scored for a prefix hit with prefix caching disabled: %s", detail)
			}
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if delivered != 0 {
		t.Errorf("%d KV event batches delivered with prefix caching disabled", delivered)
	}
	for _, inst := range cs.Instances() {
		ep := gw.endpoints[inst.ID()]
		if got, err := gw.kvIngest[0].HostedPodBlocks(ep.ip + ":" + fmt.Sprint(ep.port)); err != nil || len(got) != 0 {
			t.Errorf("%s: the router indexes %d blocks (%v)", inst.ID(), len(got), err)
		}
	}
}
