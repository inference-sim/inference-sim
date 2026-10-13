// network_topology_e2e_test.go — end-to-end tests for node-span accounting (#1530): every
// placement site must record how many nodes an instance occupies, so `blis run` can write the
// fleet's widest span into the trace header for replay to refuse. Instances price steps with
// the fake latency model; the span is a placement fact, independent of pricing.
package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
)

// netTestSimConfig is a fake-priced instance config at the given TP.
func netTestSimConfig(tp int) sim.SimConfig {
	return sim.SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 8192, 0),
		LatencyModel:        testFakeLatency(),
		ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "m", "H100", tp, 1, false, 0),
	}
}

// ─── All three placement sites (R23) ────────────────────────────────────────

// TestNodeSpan_RecordedAtAllThreePlacementSites verifies BC-6: an instance created at
// startup, through the deferred NodeReadyEvent path, or by autoscaler scale-up all have
// their node span recorded. The observable is MaxNodesSpanned, which only the recording
// step moves — so its change proves that step ran at that site.
func TestNodeSpan_RecordedAtAllThreePlacementSites(t *testing.T) {
	t.Run("startup", func(t *testing.T) {
		cfg := DeploymentConfig{
			SimConfig:    netTestSimConfig(16),
			NumInstances: 1,
			NodePools:    []NodePoolConfig{newTestPool("p", "H100", 8, 2)}, // tp=16 must span
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
		require.Len(t, cs.instances, 1, "startup must place the instance")
		assert.Equal(t, 2, cs.MaxNodesSpanned(), "the startup placement site must record the span")
	})

	t.Run("deferred_node_ready", func(t *testing.T) {
		cfg := DeploymentConfig{
			SimConfig:    netTestSimConfig(16),
			NumInstances: 1,
			// InitialNodes=0 → the instance is pending until a node becomes Ready.
			NodePools: []NodePoolConfig{{Name: "p", GPUType: "H100", GPUsPerNode: 8, GPUMemoryGiB: 80, InitialNodes: 0, MaxNodes: 4}},
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
		require.Empty(t, cs.instances, "precondition: no instance before a node is Ready")
		require.Equal(t, 0, cs.MaxNodesSpanned(), "precondition: nothing placed, nothing recorded")

		// Two nodes must be Ready before a tp=16 whole-node span can be satisfied.
		nodeA, _ := cs.placement.ProvisionNode("p", 0)
		nodeB, _ := cs.placement.ProvisionNode("p", 0)
		require.NotNil(t, nodeA)
		require.NotNil(t, nodeB)
		(&NodeReadyEvent{timestamp: 0, nodeID: nodeA.ID}).Execute(cs)
		(&NodeReadyEvent{timestamp: 0, nodeID: nodeB.ID}).Execute(cs)
		require.Len(t, cs.instances, 1, "precondition: the deferred instance must be placed once both nodes are Ready")
		assert.Equal(t, 2, cs.MaxNodesSpanned(), "the deferred NodeReadyEvent placement site must record the span")
	})

	t.Run("autoscaler_scale_up", func(t *testing.T) {
		cfg := DeploymentConfig{
			SimConfig:    netTestSimConfig(8),
			NumInstances: 1,
			// 3 nodes: the startup instance fits on one, the scaled-up tp=16 one spans two.
			NodePools: []NodePoolConfig{newTestPool("p", "H100", 8, 3)},
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
		require.Len(t, cs.instances, 1)
		require.Equal(t, 1, cs.MaxNodesSpanned(), "precondition: the startup instance is single-node")

		err := NewDirectActuator(cs).Apply([]ScaleDecision{
			{ModelID: "m", Variant: NewVariantSpec("H100", 16), Delta: 1},
		})
		require.NoError(t, err)
		require.Len(t, cs.instances, 2, "precondition: scale-up must add an instance")
		assert.Equal(t, 2, cs.MaxNodesSpanned(), "the autoscaler scale-up placement site must record the span")
	})
}

// ─── INV-6 and the trace-header signal ──────────────────────────────────────

// TestPlacementTopology_SpanningRunIsByteIdenticalAcrossRuns verifies INV-6 for a fleet that
// spans nodes: two identical SPANNING runs at the same seed must produce byte-identical
// output. The comparison is on the marshalled metrics payload — the same struct stdout is
// rendered from — so it is a genuine byte-level check rather than a field-by-field one.
func TestPlacementTopology_SpanningRunIsByteIdenticalAcrossRuns(t *testing.T) {
	makeReqs := func() []*sim.Request {
		reqs := make([]*sim.Request, 30)
		for i := range reqs {
			reqs[i] = &sim.Request{
				ID:           fmt.Sprintf("req_%d", i),
				Model:        "m",
				ArrivalTime:  int64(i) * 1500,
				InputTokens:  make([]sim.TokenID, 512),
				OutputTokens: make([]sim.TokenID, 24),
				State:        sim.StateQueued,
			}
		}
		return reqs
	}
	runOnce := func() string {
		cfg := DeploymentConfig{
			SimConfig:    netTestSimConfig(16),
			NumInstances: 1,
			NodePools:    []NodePoolConfig{newTestPool("p", "H100", 8, 2)}, // tp=16 must span
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(makeReqs()), nil)
		require.Equal(t, 2, cs.MaxNodesSpanned(), "precondition: the instance must span two nodes")
		mustRun(t, cs)
		payload, err := json.Marshal(cs.AggregatedMetrics().BuildOutput("cluster"))
		require.NoError(t, err)
		return string(payload)
	}

	assert.Equal(t, runOnce(), runOnce(),
		"two identical spanning runs at the same seed must produce byte-identical output (INV-6)")
}

// TestPlacementTopology_MaxNodesSpannedReportsWidestSpan verifies the signal `blis run`
// records in the trace header, and that replay uses to refuse a trace it cannot
// reproduce. It must report the widest span in the fleet, and must stay at 0 when there
// is no placement at all — the value that keeps the header byte-identical for every run
// without multi-node placement.
func TestPlacementTopology_MaxNodesSpannedReportsWidestSpan(t *testing.T) {
	newCluster := func(pools []NodePoolConfig, tp, instances int) *ClusterSimulator {
		cfg := DeploymentConfig{
			SimConfig:    netTestSimConfig(tp),
			NumInstances: instances,
			NodePools:    pools,
		}
		return NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	}

	t.Run("no node pools reports nothing", func(t *testing.T) {
		assert.Equal(t, 0, newCluster(nil, 16, 1).MaxNodesSpanned(),
			"without placement there is no span to record, and the trace header must stay unchanged")
	})
	t.Run("single-node fleet reports one", func(t *testing.T) {
		assert.Equal(t, 1, newCluster([]NodePoolConfig{newTestPool("p", "H100", 16, 2)}, 16, 1).MaxNodesSpanned())
	})
	t.Run("spanning fleet reports the span", func(t *testing.T) {
		assert.Equal(t, 2, newCluster([]NodePoolConfig{newTestPool("p", "H100", 8, 2)}, 16, 1).MaxNodesSpanned())
	})
	t.Run("wider span reported", func(t *testing.T) {
		assert.Equal(t, 4, newCluster([]NodePoolConfig{newTestPool("p", "H100", 4, 4)}, 16, 1).MaxNodesSpanned())
	})
}
