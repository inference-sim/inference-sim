package llmd

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/embed"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

func TestCheckMetricSpecs(t *testing.T) {
	cases := []struct {
		name, config, wantErr string
	}{
		{"no configuration", "", ""},
		{"default extractor", "plugins:\n- type: core-metrics-extractor\n", ""},
		{"vLLM override with published series", `plugins:
- type: core-metrics-extractor
  parameters:
    engineConfigs:
    - name: vllm
      queuedRequestsSpec: vllm:num_requests_waiting
      kvUsageSpec: vllm:kv_cache_usage_perc
      customMetrics:
      - {attributeKey: p, metricSpec: vllm:num_preemptions_total}
      - {attributeKey: x, metricSpec: vllm:not_published_total, optional: true}
`, ""},
		{"required custom series not published", `plugins:
- type: core-metrics-extractor
  parameters:
    engineConfigs:
    - name: vllm
      customMetrics:
      - {attributeKey: x, metricSpec: vllm:gpu_cache_usage_perc}
`, `custom metric "vllm:gpu_cache_usage_perc"`},
		{"non-vLLM default engine", "plugins:\n- type: core-metrics-extractor\n  parameters:\n    defaultEngine: sglang\n", `"sglang" engine`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			config := ""
			if tc.config != "" {
				config = "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\n" + tc.config
			}
			err := refusal(t, testDeployment(2), config, nil)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("refused: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("want an error containing %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestPrecisePrefixWithAnotherBlockSizeIsRefused(t *testing.T) {
	// llm-d's deploy/config/epp-precise-prefix-cache-config.yaml
	config := `# Sample EPP config: split precise-prefix-cache pipeline
#   token-producer -> precise-prefix-cache-producer -> prefix-cache-scorer
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: token-producer
    parameters:
      modelName: hf-repo/model-name           # set to the model used in the vLLM deployment
      vllm:
        url: http://localhost:8000
  - type: endpoint-notification-source
  - type: metrics-data-source
  - type: core-metrics-extractor
  - type: single-profile-handler
  - type: decode-filter
  - type: precise-prefix-cache-producer
    parameters:
      tokenProcessorConfig:
        blockSizeTokens: 64
      indexerConfig:
        kvBlockIndexConfig:
          enableMetrics: true       # enable kv-block index metrics (prometheus)
  - type: prefix-cache-scorer
    parameters:
      prefixMatchInfoProducerName: precise-prefix-cache-producer
  - type: kv-cache-utilization-scorer
  - type: queue-scorer
  - type: max-score-picker
dataLayer:
  sources:
    - pluginRef: metrics-data-source
      extractors:
        - pluginRef: core-metrics-extractor
    - pluginRef: endpoint-notification-source
      extractors:
        - pluginRef: precise-prefix-cache-producer
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: decode-filter
      - pluginRef: prefix-cache-scorer
        weight: 2.0
      - pluginRef: kv-cache-utilization-scorer
        weight: 1.0
      - pluginRef: queue-scorer
        weight: 1.0
      - pluginRef: max-score-picker
`
	cs := newCluster(testDeployment(2), cluster.NewSliceRequestSource(testRequests(4, 1, 1000, 0)), nil)
	cs.SetGateway(New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)}))
	// It cuts block keys every 64 tokens; the test engines' blocks are 16 tokens, so the KV events
	// they publish could never match a request's keys.
	err := cs.Run()
	if err == nil || !strings.Contains(err.Error(), "KV block size is 16") {
		t.Fatalf("want precise-prefix-cache-producer refused for its block size, got %v", err)
	}
}

// refusal starts a router over dep with config and returns the startup error.
func refusal(t *testing.T, dep cluster.DeploymentConfig, config string, labels func(cluster.InstanceID, cluster.PoolRole) map[string]string, extraFlags ...string) error {
	t.Helper()
	cs := newCluster(dep, cluster.NewSliceRequestSource(testRequests(4, 1, 1000, 0)), nil)
	flags := append(DefaultEPPFlags(), extraFlags...)
	if config != "" {
		flags = append(flags, "--config-text="+config)
	}
	cs.SetGateway(New(Options{Model: "test-model", PodLabels: labels, EPPFlags: flags}))
	return cs.Run()
}

// TestUnsupportedPluginReachedThroughFlowControlIsRefused: a plugin the configuration reaches only
// through a flow-control reference (llm-d's deploy/config/sim-program-aware-config.yaml names
// program-aware-fairness as the default band's fairness policy) is refused at startup like any
// other unsupported plugin, instead of running unconverted.
func TestUnsupportedPluginReachedThroughFlowControlIsRefused(t *testing.T) {
	config := `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: program-aware-fairness
- type: queue-scorer
- type: max-score-picker
- type: single-profile-handler

featureGates:
- flowControl

flowControl:
  defaultPriorityBand:
    fairnessPolicyRef: program-aware-fairness

schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
  - pluginRef: max-score-picker
`
	err := refusal(t, testDeployment(2), config, nil)
	if err == nil || !strings.Contains(err.Error(), "program-aware-fairness") || !strings.Contains(err.Error(), "cannot be simulated faithfully") {
		t.Fatalf("want program-aware-fairness refused at startup, got %v", err)
	}
	ref := strings.Replace(config, "- type: program-aware-fairness\n", "", 1) // reached through the ref alone
	if err := refusal(t, testDeployment(2), ref, nil); err == nil || !strings.Contains(err.Error(), "program-aware-fairness") {
		t.Fatalf("want program-aware-fairness refused when only the flow-control ref names it, got %v", err)
	}
}

// TestPDTopologyNeedsADisaggregatingConfig: a P/D deployment with a router configuration that
// cannot disaggregate is refused instead of silently serving every request from a flat pool; with
// a disaggregating configuration it runs.
func TestPDTopologyNeedsADisaggregatingConfig(t *testing.T) {
	dep := testDeployment(4)
	dep.PrefillInstances, dep.DecodeInstances = 2, 2
	dep.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 }
	if err := refusal(t, dep, "", roleLabels); err == nil || !strings.Contains(err.Error(), "disagg-profile-handler") {
		t.Fatalf("want the default configuration refused on a P/D deployment, got %v", err)
	}
	if err := refusal(t, dep, pdConfig, roleLabels); err != nil {
		t.Fatalf("a disaggregating configuration must run: %v", err)
	}
}

// TestCheckRawConfig_EngineLabelsAndEndpointDiscovery: a pod whose engine label selects an engine
// mapping the simulated engines do not publish is refused, with the default configuration too;
// and a token-producer that discovers render endpoints is refused before the router is built.
func TestCheckRawConfig_EngineLabelsAndEndpointDiscovery(t *testing.T) {
	engine := func(key, value string) func(cluster.InstanceID, cluster.PoolRole) map[string]string {
		return func(id cluster.InstanceID, _ cluster.PoolRole) map[string]string {
			if id == "instance_1" {
				return map[string]string{key: value}
			}
			return map[string]string{key: "vllm"}
		}
	}
	if err := refusal(t, testDeployment(2), "", engine("llm-d.ai/engine-type", "sglang")); err == nil || !strings.Contains(err.Error(), `"sglang"`) {
		t.Errorf("default configuration, an sglang pod: got %v, want a refusal naming sglang", err)
	}
	if err := refusal(t, testDeployment(2), "", engine("llm-d.ai/engine-type", "vllm")); err != nil {
		t.Errorf("vllm pods must run: %v", err)
	}
	head := "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\n"
	custom := head + "plugins:\n- type: core-metrics-extractor\n  parameters:\n    engineLabelKey: example.com/engine\n"
	if err := refusal(t, testDeployment(2), custom, engine("example.com/engine", "sglang")); err == nil {
		t.Errorf("a custom engineLabelKey selecting sglang must be refused")
	}
	discovery := head + "plugins:\n- type: token-producer\n  parameters:\n    modelName: m\n    vllm:\n      endpointDiscovery:\n        discoverModelLimits: true\n"
	if err := refusal(t, testDeployment(2), discovery, nil); err == nil || !strings.Contains(err.Error(), "endpointDiscovery") {
		t.Errorf("token-producer endpoint discovery: got %v, want a refusal", err)
	}
	if err := refusal(t, testDeployment(2), head+"plugins:\n- type: metrics-data-source\n  parameters:\n    path: /stats\n", nil); err == nil || !strings.Contains(err.Error(), "/stats") {
		t.Errorf("a metrics path other than /metrics must be refused, got %v", err)
	}
	for _, ok := range []string{
		head + "plugins:\n- type: token-producer\n  parameters:\n    modelName: m\n    vllm:\n      url: http://localhost:8000\n",
	} {
		if err := refusal(t, testDeployment(2), ok, nil); err != nil && strings.Contains(err.Error(), "endpointDiscovery") {
			t.Errorf("a token-producer with a fixed url must not be refused for discovery: %v", err)
		}
	}
}

// TestCheckEngineSignals: the facts the router reports are compared with what the simulated
// engines produce, the implicit extractor included.
func TestCheckEngineSignals(t *testing.T) {
	vllm := embed.MetricsExtractor{Name: "core-metrics-extractor", Implicit: true}
	for _, tc := range []struct {
		name string
		req  embed.Requirements
		pods []map[string]string
		want string
	}{
		{"implicit extractor, vllm pods", embed.Requirements{Extractors: []embed.MetricsExtractor{vllm}}, []map[string]string{{"llm-d.ai/engine-type": "vllm"}}, ""},
		{"implicit extractor, an sglang pod", embed.Requirements{Extractors: []embed.MetricsExtractor{vllm}}, []map[string]string{{"llm-d.ai/engine-type": "sglang"}}, `"sglang" engine`},
		{"block size", embed.Requirements{PrefixIndexes: []embed.PrefixIndex{{Name: "p", BlockSize: 64, EngineType: "vllm"}}}, nil, "KV block size is 16"},
		{"KV-event engine", embed.Requirements{PrefixIndexes: []embed.PrefixIndex{{Name: "p", BlockSize: 16, EngineType: "sglang"}}}, nil, "reads sglang KV events"},
		{"metrics path", embed.Requirements{MetricsSources: []embed.MetricsSource{{Name: "m", Path: "/stats"}}}, nil, `scrapes "/stats"`},
		{"optional custom series", embed.Requirements{Extractors: []embed.MetricsExtractor{{Name: "x", Engines: []embed.EngineMapping{{Name: "vllm",
			Custom: []embed.CustomMetric{{Series: "vllm:not_published", Optional: true}}}}}}}, nil, ""},
		{"required custom series", embed.Requirements{Extractors: []embed.MetricsExtractor{{Name: "x", Engines: []embed.EngineMapping{{Name: "vllm",
			Series: map[string]string{"kvUsageSpec": "vllm:kv_cache_usage_perc{a=\"b\"}"}, Custom: []embed.CustomMetric{{Series: "vllm:not_published"}}}}}}}, nil, `custom metric "vllm:not_published"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkEngineSignals(tc.req, tc.pods, 16)
			if (tc.want == "") != (err == nil) || (err != nil && !strings.Contains(err.Error(), tc.want)) {
				t.Fatalf("checkEngineSignals = %v, want %q", err, tc.want)
			}
		})
	}
}

// TestMultipleTargetPortsAreRefused: a pool whose pods each serve several data-parallel ranks on
// several ports is refused; BLIS simulates each rank as its own pod.
func TestMultipleTargetPortsAreRefused(t *testing.T) {
	cs := newCluster(testDeployment(2), cluster.NewSliceRequestSource(testRequests(2, 1, 1000, 0)), nil)
	cs.SetGateway(New(Options{Model: "test-model", EPPFlags: []string{"--endpoint-selector=llm-d.ai/inference-serving=true", "--endpoint-target-ports=8000,8001"}}))
	if err := cs.Run(); err == nil || !strings.Contains(err.Error(), "target ports") {
		t.Fatalf("want two target ports refused, got %v", err)
	}
}

// TestCheckMetricFlags: a legacy metric flag naming an unpublished series is refused.
func TestCheckMetricFlags(t *testing.T) {
	if err := checkMetricFlags(map[string]string{"total-queued-requests-metric": "sglang:num_queue_reqs"}); err == nil ||
		!strings.Contains(err.Error(), "--total-queued-requests-metric=sglang:num_queue_reqs") {
		t.Errorf("an unpublished legacy metric must be refused, got %v", err)
	}
	if err := checkMetricFlags(map[string]string{"total-queued-requests-metric": "vllm:num_requests_waiting"}); err != nil {
		t.Errorf("a published legacy metric must run: %v", err)
	}
}

// TestSignalPolicyRefusesThroughTheRouter: a type the fork can host but BLIS's signal policy
// refuses (session affinity: the simulated workload sends no session header) is refused at startup
// with BLIS's reason; that the router consults the policy before the plugin's constructor runs is
// the fork's guarantee (its pkg/embed tests and blis-schemas embedtest).
func TestSignalPolicyRefusesThroughTheRouter(t *testing.T) {
	config := "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: session-affinity-scorer\n- type: max-score-picker\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: session-affinity-scorer\n  - pluginRef: max-score-picker\n"
	err := refusal(t, testDeployment(2), config, nil)
	if err == nil || !strings.Contains(err.Error(), `plugin "session-affinity-scorer" (type session-affinity-scorer): the simulated workload sends no session header`) {
		t.Fatalf("want session-affinity-scorer refused by BLIS's signal policy, got %v", err)
	}
}

// TestAllow: BLIS's policy allows exactly its supported types, and gives each refused type its
// recorded reason or the generic one.
func TestAllow(t *testing.T) {
	for typ := range supported {
		if ok, reason := allow(typ); !ok || reason != "" {
			t.Errorf("allow(%s) = %v, %q; want allowed", typ, ok, reason)
		}
		if signalReasons[typ] != "" {
			t.Errorf("%s is both supported and refused", typ)
		}
	}
	if ok, reason := allow("dcgm-extractor"); ok || reason != "the simulated engines publish no DCGM GPU telemetry" {
		t.Errorf("allow(dcgm-extractor) = %v, %q", ok, reason)
	}
	if ok, reason := allow("blis-test-unknown-type"); ok || reason != notAudited {
		t.Errorf("allow(unknown) = %v, %q", ok, reason)
	}
}

// TestLatePodIsChecked: a pod that appears during the run (an autoscaled role, say) gets the
// startup checks: an engine label selecting an unpublished mapping is an error.
func TestLatePodIsChecked(t *testing.T) {
	engine := "vllm"
	cs := newCluster(testDeployment(2), cluster.NewSliceRequestSource(testRequests(2, 1, 1000, 0)), nil)
	gw := New(Options{Model: "test-model", PodLabels: func(cluster.InstanceID, cluster.PoolRole) map[string]string {
		return map[string]string{"llm-d.ai/engine-type": engine}
	}})
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	inst := cs.Instances()[0]
	if err := gw.checkLatePod(inst); err != nil {
		t.Fatalf("a vllm pod must pass: %v", err)
	}
	engine = "sglang" // the next pod of this role would carry the sglang label
	if err := gw.checkLatePod(inst); err == nil || !strings.Contains(err.Error(), "sglang") {
		t.Fatalf("want a late sglang pod refused, got %v", err)
	}
}
