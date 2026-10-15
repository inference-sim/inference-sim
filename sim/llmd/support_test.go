package llmd

import (
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
			err := checkMetricSpecs([]byte(tc.config))
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

func TestUnsupportedPluginIsRefusedWithItsReason(t *testing.T) {
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
	cs.SetGateway(New(Options{Model: "test-model", EPPFlags: append(append([]string{}, DefaultEPPFlags...), "--config-text="+config)}))
	// The CLI exits on a refusal; the refused plugin's own constructor may have started
	// goroutines (its kv-events worker pool) that only process exit ends.
	err := cs.Run()
	if err == nil || !strings.Contains(err.Error(), "KV-cache events") {
		t.Fatalf("want precise-prefix-cache-producer refused for its KV-event dependency, got %v", err)
	}
}
