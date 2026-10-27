package llmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	_ "github.com/inference-sim/inference-sim/sim/lora"
)

// TestStartupRefusalMessages pins the full startup error a user sees for each of a fixed set of
// inputs (testdata/refusals.golden). The messages were recorded on the code before the router
// contract, so a diff here is a change of a user-visible refusal: a translation bug unless the
// golden marks it as a known change. UPDATE_REFUSALS=1 rewrites the golden, keeping its comments.
func TestStartupRefusalMessages(t *testing.T) {
	var lines []string
	add := func(name string, err error) {
		lines = append(lines, fmt.Sprintf("%s\t%v", name, err))
	}
	pd := func() cluster.DeploymentConfig {
		dep := testDeployment(4)
		dep.PrefillInstances, dep.DecodeInstances = 2, 2
		dep.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 }
		return dep
	}
	run := func(dep cluster.DeploymentConfig, o Options) error {
		if o.Model == "" {
			o.Model = "test-model"
		}
		cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(testRequests(2, 1, 1000, 0)), nil)
		cs.SetGateway(New(o))
		return cs.Run()
	}
	cfg := func(c string, extra ...string) []string {
		return append(append(DefaultEPPFlags(), extra...), "--config-text="+c)
	}
	cases := map[string]func() error{
		"flags-bad": func() error { return run(testDeployment(2), Options{EPPFlags: []string{"--no-such-flag"}}) },
		"flags-extra-args": func() error {
			return run(testDeployment(2), Options{EPPFlags: []string{"--endpoint-selector=a=b", "stray"}})
		},
		"flags-config-file": func() error {
			return run(testDeployment(2), Options{EPPFlags: append(DefaultEPPFlags(), "--config-file=/nonexistent/x.yaml")})
		},
		"flags-legacy-metric": func() error {
			return run(testDeployment(2), Options{EPPFlags: append(DefaultEPPFlags(), "--total-queued-requests-metric=sglang:num_queue_reqs")})
		},
		"target-ports": func() error {
			return run(testDeployment(2), Options{EPPFlags: []string{"--endpoint-selector=llm-d.ai/inference-serving=true", "--endpoint-target-ports=8000,8001"}})
		},
		"pool-missing": func() error {
			return run(testDeployment(2), Options{EPPFlags: []string{"--pool-name=nope"}, Manifests: []byte(poolManifests)})
		},
		"pool-two": func() error {
			two := poolManifests + "\n---\n" + strings.Replace(poolManifests, "name: vllm", "name: vllm2", 1)
			return run(testDeployment(2), Options{EPPFlags: []string{"--pool-name=vllm"}, Manifests: []byte(two)})
		},
		"pool-with-selector": func() error { return run(testDeployment(2), Options{Manifests: []byte(poolManifests)}) },
		"objective-with-selector": func() error {
			return run(testDeployment(2), Options{Manifests: []byte("apiVersion: llm-d.ai/v1alpha2\nkind: InferenceObjective\nmetadata:\n  name: x\nspec:\n  priority: 1\n  poolRef:\n    name: vllm\n")})
		},
		"manifest-unserved": func() error {
			return run(testDeployment(2), Options{Manifests: []byte("apiVersion: llm-d.ai/v9\nkind: InferenceObjective\nmetadata:\n  name: x\n")})
		},
		"manifest-pod": func() error {
			return run(testDeployment(2), Options{Manifests: []byte("apiVersion: v1\nkind: Pod\nmetadata:\n  name: p\n")})
		},
		"manifest-strict": func() error {
			bad := strings.Replace(poolManifests, "spec:", "spec:\n  priorty: 3", 1)
			return run(testDeployment(2), Options{EPPFlags: []string{"--pool-name=vllm"}, Manifests: []byte(bad)})
		},
		"manifest-yaml": func() error { return run(testDeployment(2), Options{Manifests: []byte("a: [\n")}) },
		"plugin-unaudited": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: no-such-type-xyz\n- type: queue-scorer\n- type: max-score-picker\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: queue-scorer\n  - pluginRef: max-score-picker\n")})
		},
		"plugin-session": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: session-affinity-scorer\n- type: max-score-picker\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: session-affinity-scorer\n  - pluginRef: max-score-picker\n")})
		},
		"plugin-redis": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: redis-syncer\n- type: queue-scorer\n- type: max-score-picker\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: queue-scorer\n  - pluginRef: max-score-picker\n")})
		},
		"config-malformed": func() error { return run(testDeployment(2), Options{EPPFlags: cfg("plugins: [\n")}) },
		"token-discovery": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: token-producer\n  parameters:\n    modelName: m\n    vllm:\n      endpointDiscovery:\n        discoverModelLimits: true\n")})
		},
		"metrics-path": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: metrics-data-source\n  parameters:\n    path: /stats\n")})
		},
		"extractor-sglang": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: core-metrics-extractor\n  parameters:\n    defaultEngine: sglang\n")})
		},
		"extractor-custom": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: core-metrics-extractor\n  parameters:\n    engineConfigs:\n    - name: vllm\n      queuedRequestsSpec: vllm:nope\n      customMetrics:\n      - {attributeKey: x, metricSpec: vllm:gpu_cache_usage_perc}\n")})
		},
		"extractor-label": func() error {
			return run(testDeployment(2), Options{PodLabels: func(cluster.InstanceID, cluster.PoolRole) map[string]string {
				return map[string]string{"llm-d.ai/engine-type": "sglang"}
			}})
		},
		"precise-blocksize": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: precise-prefix-cache-producer\n  parameters:\n    tokenProcessorConfig:\n      blockSizeTokens: 64\n")})
		},
		"precise-redis": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: precise-prefix-cache-producer\n  parameters:\n    indexerConfig:\n      kvBlockIndexConfig:\n        redisConfig:\n          address: x\n")})
		},
		"precise-costaware": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: precise-prefix-cache-producer\n  parameters:\n    indexerConfig:\n      kvBlockIndexConfig:\n        costAwareMemoryConfig:\n          size: 1\n")})
		},
		"precise-engine": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: precise-prefix-cache-producer\n  parameters:\n    kvEventsConfig:\n      engineType: sglang\n")})
		},
		"topology-hostname": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: topology-extractor\n  parameters:\n    hostname: example.com/node\n", "--allow-experimental-plugins")})
		},
		"topology-peer": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: topology-extractor\n- type: topology-affinity-filter\n  parameters:\n    peerTopologyHeader: x-peer-topology\n", "--allow-experimental-plugins")})
		},
		"topology-noextractor": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: topology-affinity-filter\n", "--allow-experimental-plugins")})
		},
		"topology-stamp-header": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: topology-extractor\n- type: topology-stamp-handler\n  parameters:\n    headerName: x-other\n", "--allow-experimental-plugins")})
		},
		"topology-stamp-profile": func() error {
			return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: topology-extractor\n- type: topology-stamp-handler\n  parameters:\n    profileName: nope\n", "--allow-experimental-plugins")})
		},
		"topology-unplaced": func() error {
			return run(testDeployment(1), Options{EPPFlags: cfg(topologyExtractorOnly, "--allow-experimental-plugins")})
		},
		"pd-flat": func() error { return run(pd(), Options{PodLabels: roleLabels}) },
		"precise-lora": func() error {
			dep := testDeployment(2)
			capacity, base, bw, fp := 2, 1000.0, 2.0e6, 2.0e6
			dep.LoRAConfig = sim.LoRAConfig{AdapterCapacity: &capacity, LoadBaseLatencyUs: &base,
				LoadBandwidthBytesUs: &bw, FootprintBytesPerRank: &fp, Adapters: []sim.AdapterSpec{{ID: "a", Rank: 8}}}
			return run(dep, Options{EPPFlags: cfg("apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: precise-prefix-cache-producer\n- type: queue-scorer\n- type: max-score-picker\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: queue-scorer\n  - pluginRef: max-score-picker\n")})
		},
	}
	// Double faults. Expected to change (a BLIS check now runs, in Config.Check, before the
	// fork's config-level hosting refusal or pool resolution):
	cases["double-topology-peer-unplaced"] = func() error {
		return run(testDeployment(1), Options{EPPFlags: cfg("apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: topology-extractor\n- type: topology-affinity-filter\n  parameters:\n    peerTopologyHeader: x-peer-topology\n- type: max-score-picker\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: max-score-picker\n", "--allow-experimental-plugins")})
	}
	cases["double-discovery-before-metrics-path"] = func() error {
		return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: token-producer\n  parameters:\n    modelName: m\n    vllm:\n      endpointDiscovery:\n        discoverModelLimits: true\n- type: metrics-data-source\n  parameters:\n    path: /stats\n")})
	}
	cases["double-pool-missing-extractor"] = func() error {
		return run(testDeployment(2), Options{EPPFlags: append([]string{"--pool-name=nope"}, "--config-text=plugins:\n- type: core-metrics-extractor\n  parameters:\n    defaultEngine: sglang\n"), Manifests: []byte(poolManifests)})
	}
	cases["double-objective-selector-plugin"] = func() error {
		return run(testDeployment(2), Options{Manifests: []byte("apiVersion: llm-d.ai/v1alpha2\nkind: InferenceObjective\nmetadata:\n  name: x\nspec:\n  priority: 1\n  poolRef:\n    name: vllm\n"),
			EPPFlags: cfg("apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: session-affinity-scorer\n- type: max-score-picker\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: session-affinity-scorer\n  - pluginRef: max-score-picker\n")})
	}
	cases["guide-precise-on-16"] = func() error {
		b, _ := os.ReadFile(filepath.Join("testdata", "llm-d-guides", "precise-prefix-cache-routing.yaml"))
		return run(testDeployment(2), Options{EPPFlags: cfg(string(b))})
	}
	placed := func() cluster.DeploymentConfig {
		dep := testDeployment(1)
		dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{"instance_0": {Hostname: "blis-node-0", NodeCount: 1}})
		return dep
	}
	for name, plugins := range map[string]string{
		"placed-topology-hostname":      "- type: topology-extractor\n  parameters:\n    hostname: example.com/node\n",
		"placed-topology-peer":          "- type: topology-extractor\n- type: topology-affinity-filter\n  parameters:\n    peerTopologyHeader: x-peer-topology\n",
		"placed-topology-noextractor":   "- type: topology-affinity-filter\n",
		"placed-topology-wrongproducer": "- type: topology-extractor\n  name: topology-a\n- type: topology-affinity-filter\n  parameters:\n    topologyProducerName: topology-b\n",
		"placed-stamp-noextractor":      "- type: topology-stamp-handler\n",
		"placed-stamp-header":           "- type: topology-extractor\n- type: topology-stamp-handler\n  parameters:\n    headerName: x-other\n",
		"placed-stamp-profile":          "- type: topology-extractor\n- type: topology-stamp-handler\n  parameters:\n    profileName: nope\n",
	} {
		plugins := plugins
		cases[name] = func() error {
			return run(placed(), Options{EPPFlags: cfg("apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n"+plugins+"- type: max-score-picker\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: max-score-picker\n", "--allow-experimental-plugins")})
		}
	}
	cases["double-metrics-path-before-blocksize"] = func() error {
		return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: metrics-data-source\n  parameters:\n    path: /stats\n- type: precise-prefix-cache-producer\n  parameters:\n    tokenProcessorConfig:\n      blockSizeTokens: 64\n")})
	}
	cases["double-extractor-before-metrics-path"] = func() error {
		return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: core-metrics-extractor\n  parameters:\n    defaultEngine: sglang\n- type: metrics-data-source\n  parameters:\n    path: /stats\n")})
	}
	// Expected to keep its message: the BLIS fault comes first in either order.
	cases["double-metrics-path-before-discovery"] = func() error {
		return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: metrics-data-source\n  parameters:\n    path: /stats\n- type: token-producer\n  parameters:\n    modelName: m\n    vllm:\n      endpointDiscovery:\n        discoverModelLimits: true\n")})
	}
	cases["double-blocksize-redis"] = func() error {
		return run(testDeployment(2), Options{EPPFlags: cfg("plugins:\n- type: precise-prefix-cache-producer\n  parameters:\n    tokenProcessorConfig:\n      blockSizeTokens: 64\n    indexerConfig:\n      kvBlockIndexConfig:\n        redisConfig:\n          address: x\n")})
	}
	files, _ := filepath.Glob(filepath.Join("testdata", "llm-d-guides", "*.yaml"))
	for _, f := range files {
		f := f
		cases["guide-"+filepath.Base(f)] = func() error {
			b, _ := os.ReadFile(f)
			return run(testDeployment(2), Options{EPPFlags: cfg(string(b))})
		}
	}
	names := make([]string, 0, len(cases))
	for n := range cases {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		add(n, cases[n]())
	}
	golden := filepath.Join("testdata", "refusals.golden")
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	var comments, want []string
	for _, l := range strings.Split(strings.TrimRight(string(raw), "\n"), "\n") {
		if strings.HasPrefix(l, "#") {
			comments = append(comments, l)
			continue
		}
		want = append(want, l)
	}
	if os.Getenv("UPDATE_REFUSALS") != "" {
		var out []string
		for _, l := range comments {
			if !strings.HasPrefix(l, "# changed: ") {
				out = append(out, l)
			}
		}
		changed := map[string]string{}
		for _, l := range comments {
			if rest, ok := strings.CutPrefix(l, "# changed: "); ok {
				changed[strings.SplitN(rest, "\t", 2)[0]] = l
			}
		}
		for _, l := range lines {
			if c, ok := changed[strings.SplitN(l, "\t", 2)[0]]; ok {
				out = append(out, c)
			}
			out = append(out, l)
		}
		if err := os.WriteFile(golden, []byte(strings.Join(out, "\n")+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		return
	}
	if len(lines) != len(want) {
		t.Fatalf("%d cases, the golden has %d", len(lines), len(want))
	}
	for i := range lines {
		if lines[i] != want[i] {
			t.Errorf("refusal changed:\n got: %s\nwant: %s", lines[i], want[i])
		}
	}
}
