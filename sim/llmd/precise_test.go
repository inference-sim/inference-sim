package llmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
		{"sim-epp-no-hit-lru.yaml", 16, "cuts block keys every 5 tokens"},
	} {
		t.Run(tc.file, func(t *testing.T) {
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
