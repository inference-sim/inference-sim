package llmd

import (
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	_ "github.com/inference-sim/inference-sim/sim/lora" // links the LoRA subsystem, as the CLI does
)

// TestINV5_FullChainThroughTheRouter asserts INV-5's chain on requests the real router held in
// flow control, rejected at capacity and expired:
//
//	arrival <= gateway enqueue <= gateway dispatch <= schedule <= completion
//
// with schedule = arrival + the scheduling delay and completion absolute, all on the cluster clock.
func TestINV5_FullChainThroughTheRouter(t *testing.T) {
	var held, checked, rejected, expired int
	for _, config := range []string{boundedQueueConfig, globalStrictConfig} {
		reqs := testRequests(300, 5, 200, 0)
		dep := testDeployment(2)
		dep.BatchConfig = sim.NewBatchConfig(4, 2048, 0)
		cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
		gw := New(Options{Model: "test-model", GatewayLatencyUs: 150, EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)})
		cs.SetGateway(gw)
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		m := cs.AggregatedMetrics()
		assertConservation(t, runResult{cs: cs, gw: gw, reqs: reqs, metrics: m}, len(reqs))
		rejected += cs.GatewayQueueRejected()
		expired += cs.GatewayExpired()
		for _, r := range reqs {
			if r.GatewayEnqueueTime < r.ArrivalTime+150 {
				t.Errorf("%s: enqueued at %d, before its arrival %d plus the gateway latency", r.ID, r.GatewayEnqueueTime, r.ArrivalTime)
			}
			if r.AssignedInstance == "" {
				continue // rejected or expired: never dispatched
			}
			if r.GatewayDispatchTime < r.GatewayEnqueueTime {
				t.Errorf("%s: dispatched at %d, before it was enqueued at %d", r.ID, r.GatewayDispatchTime, r.GatewayEnqueueTime)
			}
			if r.GatewayDispatchTime > r.GatewayEnqueueTime {
				held++
			}
			delay, ok := m.RequestSchedulingDelays[r.ID]
			done, okDone := m.RequestCompletionTimes[r.ID]
			if !ok || !okDone {
				continue
			}
			checked++
			scheduled := float64(r.ArrivalTime) + float64(delay)
			if scheduled < float64(r.GatewayDispatchTime) {
				t.Errorf("%s: scheduled at %.0f, before its dispatch %d", r.ID, scheduled, r.GatewayDispatchTime)
			}
			if done < scheduled {
				t.Errorf("%s: completed at %.0f, before it was scheduled at %.0f", r.ID, done, scheduled)
			}
		}
	}
	// Non-vacuity: requests were held and then dispatched, rejected at capacity, and expired.
	if held == 0 || checked == 0 || rejected == 0 || expired == 0 {
		t.Fatalf("held %d, checked %d, rejected %d, expired %d: the chain was not fully exercised", held, checked, rejected, expired)
	}
}

const loraAffinityConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: lora-affinity-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: lora-affinity-scorer
  - pluginRef: max-score-picker
`

const loadOnlyConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
  - pluginRef: kv-cache-utilization-scorer
  - pluginRef: max-score-picker
`

// TestLoRAAffinity_FewerColdLoadsThanLoadOnly: under a skewed adapter popularity, llm-d's
// lora-affinity-scorer, reading the adapter sets the engines publish in vllm:lora_requests_info,
// keeps requests on instances that already hold their adapter, so the cluster loads adapters
// markedly fewer times than with a load-only configuration on the same workload.
func TestLoRAAffinity_FewerColdLoadsThanLoadOnly(t *testing.T) {
	adapters := []string{"a0", "a1", "a2", "a3", "a4", "a5"}
	loads := func(config string) int64 {
		reqs := testRequests(300, 13, 3000, 0)
		z := rand.NewZipf(rand.New(rand.NewSource(7)), 1.3, 1.0, uint64(len(adapters)-1))
		for _, r := range reqs {
			r.Adapter = adapters[z.Uint64()]
		}
		dep := testDeployment(4)
		capacity, base, bw, fp := 2, 1000.0, 2.0e6, 2.0e6
		specs := make([]sim.AdapterSpec, len(adapters))
		for i, id := range adapters {
			specs[i] = sim.AdapterSpec{ID: id, Rank: 8}
		}
		dep.LoRAConfig = sim.LoRAConfig{AdapterCapacity: &capacity, LoadBaseLatencyUs: &base,
			LoadBandwidthBytesUs: &bw, FootprintBytesPerRank: &fp, Adapters: specs}
		cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
		cs.SetGateway(New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)}))
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		m := cs.AggregatedMetrics()
		if m.CompletedRequests != len(reqs) {
			t.Fatalf("completed %d of %d", m.CompletedRequests, len(reqs))
		}
		var sum int64
		for _, c := range m.AdapterLoadCounts {
			sum += c
		}
		return sum
	}
	base, affinity := loads(loadOnlyConfig), loads(loraAffinityConfig)
	t.Logf("adapter loads: load-only %d, lora-affinity %d", base, affinity)
	if base == 0 || float64(affinity) > 0.7*float64(base) {
		t.Errorf("lora-affinity loaded adapters %d times, load-only %d: want at least 30%% fewer", affinity, base)
	}
}
