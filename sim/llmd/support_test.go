package llmd

import (
	"encoding/json"
	"errors"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"strings"
	"testing"

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
			err := checkRawConfig([]byte(tc.config), nil, 16)
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
	cs := cluster.NewClusterSimulator(testDeployment(2), cluster.NewSliceRequestSource(testRequests(4, 1, 1000, 0)), nil)
	cs.SetGateway(New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)}))
	// It cuts block keys every 64 tokens; the test engines' blocks are 16 tokens, so the KV events
	// they publish could never match a request's keys.
	err := cs.Run()
	if err == nil || !strings.Contains(err.Error(), "KV block size is 16") {
		t.Fatalf("want precise-prefix-cache-producer refused for its block size, got %v", err)
	}
}

// refusal starts a router over dep with config and returns the startup error.
func refusal(t *testing.T, dep cluster.DeploymentConfig, config string, labels func(cluster.InstanceID, cluster.PoolRole) map[string]string) error {
	t.Helper()
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(testRequests(4, 1, 1000, 0)), nil)
	flags := DefaultEPPFlags()
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

// TestEncodePoolIsRefused: encode placement is not read from the router, so a deployment with an
// encode pool is refused.
func TestEncodePoolIsRefused(t *testing.T) {
	dep := testDeployment(3)
	dep.PrefillInstances, dep.DecodeInstances, dep.EncodeInstances = 1, 1, 1
	dep.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 }
	if err := refusal(t, dep, pdConfig, roleLabels); err == nil || !strings.Contains(err.Error(), "encode pool") {
		t.Fatalf("want an encode pool refused, got %v", err)
	}
}

// TestCheckRawConfig_EngineLabelsAndEndpointDiscovery: a pod whose engine label selects an engine
// mapping the simulated engines do not publish is refused, with the default configuration too;
// and a token-producer that discovers render endpoints is refused before the router is built.
func TestCheckRawConfig_EngineLabelsAndEndpointDiscovery(t *testing.T) {
	sglang := []map[string]string{{"llm-d.ai/engine-type": "vllm"}, {"llm-d.ai/engine-type": "sglang"}}
	if err := checkRawConfig(nil, sglang, 16); err == nil || !strings.Contains(err.Error(), `"sglang"`) {
		t.Errorf("default configuration, an sglang pod: got %v, want a refusal naming sglang", err)
	}
	if err := checkRawConfig(nil, []map[string]string{{"llm-d.ai/engine-type": "vllm"}}, 16); err != nil {
		t.Errorf("vllm pods must run: %v", err)
	}
	custom := "plugins:\n- type: core-metrics-extractor\n  parameters:\n    engineLabelKey: example.com/engine\n"
	if err := checkRawConfig([]byte(custom), []map[string]string{{"example.com/engine": "sglang"}}, 16); err == nil {
		t.Errorf("a custom engineLabelKey selecting sglang must be refused")
	}
	discovery := "plugins:\n- type: token-producer\n  parameters:\n    modelName: m\n    vllm:\n      endpointDiscovery:\n        discoverModelLimits: true\n"
	if err := checkRawConfig([]byte(discovery), nil, 16); err == nil || !strings.Contains(err.Error(), "endpointDiscovery") {
		t.Errorf("token-producer endpoint discovery: got %v, want a refusal", err)
	}
	if err := checkRawConfig([]byte("plugins:\n- type: metrics-data-source\n  parameters:\n    path: /stats\n"), nil, 16); err == nil {
		t.Errorf("a metrics path other than /metrics must be refused")
	}
	if err := checkRawConfig([]byte("plugins:\n- type: token-producer\n  parameters:\n    vllm:\n      url: http://x\n      endpointDiscovery: false\n"), nil, 16); err != nil {
		t.Errorf("endpointDiscovery: false must run: %v", err)
	}
	fixed := "plugins:\n- type: token-producer\n  parameters:\n    modelName: m\n    vllm:\n      url: http://localhost:8000\n"
	if err := checkRawConfig([]byte(fixed), nil, 16); err != nil {
		t.Errorf("token-producer with a fixed url must run: %v", err)
	}
}

// TestMultipleTargetPortsAreRefused: a pool whose pods each serve several data-parallel ranks on
// several ports is refused; BLIS simulates each rank as its own pod.
func TestMultipleTargetPortsAreRefused(t *testing.T) {
	cs := cluster.NewClusterSimulator(testDeployment(2), cluster.NewSliceRequestSource(testRequests(2, 1, 1000, 0)), nil)
	cs.SetGateway(New(Options{Model: "test-model", EPPFlags: []string{"--endpoint-selector=llm-d.ai/inference-serving=true", "--endpoint-target-ports=8000,8001"}}))
	if err := cs.Run(); err == nil || !strings.Contains(err.Error(), "target ports") {
		t.Fatalf("want two target ports refused, got %v", err)
	}
}

// TestCheckEPPFlags: a legacy metric flag naming an unpublished series is refused; flags that only
// change how a deployed router is served are reported as inert.
func TestCheckEPPFlags(t *testing.T) {
	if _, err := checkEPPFlags([]string{"--total-queued-requests-metric=sglang:num_queue_reqs"}); err == nil {
		t.Errorf("an unpublished legacy metric must be refused")
	}
	if _, err := checkEPPFlags([]string{"--total-queued-requests-metric=vllm:num_requests_waiting"}); err != nil {
		t.Errorf("a published legacy metric must run: %v", err)
	}
	inert, err := checkEPPFlags([]string{"--tracing=true", "--ha-enable-leader-election", "--refresh-metrics-interval=20ms"})
	if err != nil || strings.Join(inert, " ") != "--ha-enable-leader-election --tracing" {
		t.Errorf("inert = %v, %v; want --ha-enable-leader-election --tracing", inert, err)
	}
}

// TestUnsupportedPluginIsRefusedBeforeItsConstructorRuns: a plugin type the simulation does not
// support is refused at its factory, so its real constructor (which may start goroutines, timers
// or external work) never runs, whether the configuration names it under plugins: or reaches it
// only through a flow-control reference.
func TestUnsupportedPluginIsRefusedBeforeItsConstructorRuns(t *testing.T) {
	const typ = "blis-test-constructor-recorder"
	ran := false
	fwkplugin.Register(typ, fwkplugin.StabilityStable, func(string, *json.Decoder, fwkplugin.Handle) (fwkplugin.Plugin, error) {
		ran = true
		return nil, errors.New("the real constructor ran")
	})
	t.Cleanup(func() { delete(fwkplugin.Registry, typ); delete(fwkplugin.RegistryMetadata, typ) })
	config := "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: " + typ + "\n- type: queue-scorer\n- type: max-score-picker\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: queue-scorer\n  - pluginRef: max-score-picker\n"
	err := refusal(t, testDeployment(2), config, nil)
	if err == nil || !strings.Contains(err.Error(), typ) || !strings.Contains(err.Error(), "not been audited") {
		t.Fatalf("want %s refused as not audited, got %v", typ, err)
	}
	if ran {
		t.Fatal("the unsupported plugin's constructor ran before the refusal")
	}
}

// TestLatePodIsChecked: a pod that appears during the run (an autoscaled role, say) gets the
// startup checks: an engine label selecting an unpublished mapping is an error.
func TestLatePodIsChecked(t *testing.T) {
	engine := "vllm"
	cs := cluster.NewClusterSimulator(testDeployment(2), cluster.NewSliceRequestSource(testRequests(2, 1, 1000, 0)), nil)
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
