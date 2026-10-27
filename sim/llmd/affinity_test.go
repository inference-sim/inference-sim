package llmd

import (
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

func affinityGuide(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile("testdata/llm-d-guides/optimized-baseline.yaml")
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestAffinityConfiguration(t *testing.T) {
	for _, tc := range []struct{ name, parameters, want string }{
		{"default", "", ""},
		{"explicit safe mode", "ttftSource: prefillThroughput\n    explorationProbability: 0", ""},
		{"predictor", "ttftSource: latencyPredictor", "external latency predictor"},
		{"exploration", "explorationProbability: 0.1", "process-wide randomness"},
		{"reported deployment", "affinityThreshold: 0.5\n    maxTTFTPenaltyMs: 4500\n    peakPrefillThroughput: 15000\n    inFlightLoadProducerName: inflight-load-producer", ""},
		{"always explore", "explorationProbability: 1", "process-wide randomness"},
		{"case-insensitive predictor", "TTFTSOURCE: latencyPredictor", "external latency predictor"},
		{"case-insensitive exploration", "ExplorationProbability: 0.1", "process-wide randomness"},
		{"uppercase exploration", "EXPLORATIONPROBABILITY: 0.1", "process-wide randomness"},
		{"alias overrides canonical zero", "explorationProbability: 0\n    explorationprobability: 0.1", "process-wide randomness"},
		{"alias disables exploration", "explorationProbability: 0.1\n    explorationprobability: 0", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config := affinityGuide(t)
			if tc.parameters != "" {
				config = strings.Replace(config, "- type: prefix-cache-affinity-filter", "- type: prefix-cache-affinity-filter\n  parameters:\n    "+tc.parameters, 1)
			}
			err := refusal(t, testDeployment(2), config, nil)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("safe configuration refused: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("want %q, got %v", tc.want, err)
			}
		})
	}
}

func TestAffinityRoutesCachedRequests(t *testing.T) {
	run := func(enabled bool) []string {
		config := affinityGuide(t)
		config = strings.Replace(config, "plugins:\n", "plugins:\n- type: random-picker\n", 1)
		config += "  - pluginRef: random-picker\n"
		if !enabled {
			config = strings.Replace(config, "  - pluginRef: prefix-cache-affinity-filter\n", "", 1)
		}
		reqs := testRequests(8, 11, 1_000_000, 1024)
		for _, r := range reqs {
			r.Model = "test-model"
		}
		cs := cluster.NewClusterSimulator(testDeployment(3), cluster.NewSliceRequestSource(reqs), nil)
		gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)})
		cs.SetGateway(gw)
		if err := cs.Run(); err != nil {
			t.Fatal(err)
		}
		assertConservation(t, runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}, len(reqs))
		out := make([]string, len(reqs))
		for i, r := range reqs {
			out[i] = r.AssignedInstance
		}
		if cs.AggregatedMetrics().CompletedRequests != len(reqs) {
			t.Fatal("requests did not complete")
		}
		return out
	}
	a, b := run(true), run(true)
	if !slices.Equal(a, b) {
		t.Fatalf("routes changed: %v vs %v", a, b)
	}
	for _, where := range a {
		if where != a[0] {
			t.Fatalf("cached prefix lost affinity: %v", a)
		}
	}
	control := run(false)
	targets := map[string]bool{}
	for _, where := range control {
		targets[where] = true
	}
	if len(targets) < 2 {
		t.Fatalf("filter-free control never changed endpoint: %v", control)
	}
}

// A shared prefix does not force every request onto a loaded endpoint when the
// configured throughput gate permits a cheaper uncached endpoint.
func TestAffinityLoadGateCanReleaseStickiness(t *testing.T) {
	targets := func(penalty string) int {
		config := strings.Replace(affinityGuide(t), "- type: prefix-cache-affinity-filter", "- type: prefix-cache-affinity-filter\n  parameters:\n    peakPrefillThroughput: 1\n    maxTTFTPenaltyMs: "+penalty, 1)
		reqs := testRequests(30, 11, 0, 1024)
		for _, r := range reqs {
			r.Model = "test-model"
		}
		cs := cluster.NewClusterSimulator(testDeployment(3), cluster.NewSliceRequestSource(reqs), nil)
		gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)})
		cs.SetGateway(gw)
		if err := cs.Run(); err != nil {
			t.Fatal(err)
		}
		assertConservation(t, runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}, len(reqs))
		where := map[string]bool{}
		for _, r := range reqs {
			where[r.AssignedInstance] = true
		}
		return len(where)
	}
	if sticky, escaping := targets("0"), targets("0.001"); sticky != 1 || escaping < 2 {
		t.Fatalf("endpoints with gate disabled: %d (want 1); with load escape: %d (want several)", sticky, escaping)
	}
}

// JSON inputs retain field order. The guard must make the same last-alias-wins
// decision as the real factory, rather than sorting a re-marshaled map.
func TestAffinityJSONParameterOrder(t *testing.T) {
	for _, tc := range []struct{ name, params, want string }{
		{"later exploration alias", `{"explorationProbability":0,"ExplorationProbability":0.1}`, "process-wide randomness"},
		{"later canonical zero", `{"ExplorationProbability":0.1,"explorationProbability":0}`, ""},
		{"later predictor alias", `{"ttftSource":"prefillThroughput","TTFTSource":"latencyPredictor"}`, "external latency predictor"},
		{"later safe source", `{"TTFTSource":"latencyPredictor","ttftSource":"prefillThroughput"}`, ""},
		{"unknown field still refused", `{"notAFilterSetting":true}`, "unknown field"},
		{"invalid value still refused", `{"explorationProbability":-1}`, "invalid config"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config := `{"apiVersion":"llm-d.ai/v1","kind":"EndpointPickerConfig","plugins":[{"type":"approx-prefix-cache-producer"},{"type":"inflight-load-producer"},{"name":"affinity","type":"prefix-cache-affinity-filter","parameters":` + tc.params + `},{"type":"token-load-scorer"}],"schedulingProfiles":[{"name":"default","plugins":[{"pluginRef":"affinity"},{"pluginRef":"token-load-scorer"}]}]}`
			err := refusal(t, testDeployment(2), config, nil)
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("want %q, got %v", tc.want, err)
			}
		})
	}
}
