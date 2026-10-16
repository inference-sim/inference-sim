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
