package llmd

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// TestSingleInstance_RouterDoesNotChangeLatency pins the single-instance accuracy contract: with
// one model server every request goes to it whatever the router decides, so routing a workload
// through the llm-d router must give exactly the latencies of injecting it into the engine
// directly. Any difference is a regression in the router seam, never an expected effect.
func TestSingleInstance_RouterDoesNotChangeLatency(t *testing.T) {
	direct := func() (map[string]float64, map[string]float64) {
		reqs := testRequests(200, 21, 700, 48)
		inst := cluster.NewInstanceSimulator("instance_0", testDeployment(1).SimConfig)
		for _, r := range reqs {
			inst.InjectRequest(r)
		}
		inst.Run()
		return inst.Metrics().RequestTTFTs, inst.Metrics().RequestE2Es
	}
	wantTTFT, wantE2E := direct()

	for name, cfg := range map[string]string{"default": "", "scorers": pdSingleProfileConfig, "random-picker": randomPickerConfig} {
		t.Run(name, func(t *testing.T) {
			reqs := testRequests(200, 21, 700, 48)
			cs := cluster.NewClusterSimulator(testDeployment(1), cluster.NewSliceRequestSource(reqs), nil)
			flags := append([]string{}, DefaultEPPFlags...)
			if cfg != "" {
				flags = append(flags, "--config-text="+cfg)
			}
			cs.SetGateway(New(Options{Model: "test-model", EPPFlags: flags}))
			if err := cs.Run(); err != nil {
				t.Fatalf("Run: %v", err)
			}
			m := cs.AggregatedMetrics()
			if !reflect.DeepEqual(m.RequestTTFTs, wantTTFT) {
				t.Errorf("TTFTs differ from direct injection")
			}
			if !reflect.DeepEqual(m.RequestE2Es, wantE2E) {
				t.Errorf("E2E latencies differ from direct injection")
			}
		})
	}
}

// pdSingleProfileConfig is a scorer-heavy single-profile configuration.
const pdSingleProfileConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
- type: prefix-cache-scorer
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: prefix-cache-scorer
    weight: 2
  - pluginRef: queue-scorer
    weight: 1
  - pluginRef: kv-cache-utilization-scorer
    weight: 1
  - pluginRef: max-score-picker
`

const randomPickerConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: random-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
`
