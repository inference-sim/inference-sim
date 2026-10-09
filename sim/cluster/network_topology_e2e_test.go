// network_topology_e2e_test.go — end-to-end tests for the inter-node network topology
// (#1530): a placement that spans a node boundary must be stamped onto the instance at
// every placement site, must agree with cost accounting, and must warn when the
// configured backend or calibration cannot price it. Instances price steps with the
// fake latency model; how a backend prices the cross-node term is tested with that
// backend, not here.
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

// netTestModelConfig is a mid-size dense model. NumHeads and NumKVHeads are divisible
// by 16 so the same config works across the TP matrix.
func netTestModelConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:       48,
		HiddenDim:       8192,
		NumHeads:        64,
		NumKVHeads:      16,
		VocabSize:       128256,
		BytesPerParam:   2,
		IntermediateDim: 28672,
	}
}

// netTestCalib returns an H100-like calibration. When ratio > 1 it declares an
// interconnect that is `ratio` times slower off-node than on-node; at ratio == 0 it
// declares none at all (the uncalibrated case).
func netTestCalib(ratio float64) sim.HardwareCalib {
	hc := sim.HardwareCalib{TFlopsPeak: 989.5, TFlopsFP8: 1979.0, BwPeakTBs: 3.35, MfuPrefill: 0.45, MfuDecode: 0.30, MemoryGiB: 80}
	if ratio > 0 {
		hc.IntraNodeBwGBps = 450
		hc.InterNodeBwGBps = 450 / ratio
	}
	return hc
}

// stampPlacementTopology places a tpDegree instance in a pool of gpusPerNode-sized
// nodes and runs the real placement + topology-stamping path (PlaceInstance,
// PlacedGPUsPerNode, applyPlacementTopology), which is where the cross-node
// diagnostics are emitted. Step pricing is not observed here: the instances price with
// the fake latency model, and how a backend prices a spanning collective is that
// backend's own test concern.
func stampPlacementTopology(t *testing.T, gpusPerNode, nodes, tpDegree int, calib sim.HardwareCalib, backend string) {
	t.Helper()
	cfg := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:              math.MaxInt64,
			Seed:                 42,
			KVCacheConfig:        sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:          sim.NewBatchConfig(256, 8192, 0),
			LatencyModelOverride: testFakeLatency(),
			ModelHardwareConfig:  sim.NewModelHardwareConfig(netTestModelConfig(), calib, "m", "H100", tpDegree, 1, false, "", backend, 0),
		},
		// NumInstances must be >= 1, and the cluster's own startup instance takes the
		// first `nodes` nodes through the real startup placement site (exercising
		// applyPlacementTopology there); the pool is sized for two identical instances
		// so the explicit placement below gets the same shape on the remaining nodes.
		NumInstances: 1,
		NodePools:    []NodePoolConfig{newTestPool("p", "H100", gpusPerNode, 2*nodes)},
	}
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	require.NotNil(t, cs.placement, "precondition: node pools must produce a PlacementManager")

	_, gpuIDs, _, err := cs.placement.PlaceInstance("inst-net", "m", "H100", tpDegree)
	require.NoError(t, err)

	simCfg := cfg.SimConfig
	cs.applyPlacementTopology(&simCfg, gpuIDs)
}

// TestPlacementTopology_SpanMatchesCostAccounting verifies BC-5 as an observable law
// between two independently-derived quantities: the node span the LATENCY model
// prices (derived from the hosting node size) must equal the node span the COST
// model bills (derived from the distinct node IDs). If they ever diverge, one of the
// two is describing a placement that did not happen.
func TestPlacementTopology_SpanMatchesCostAccounting(t *testing.T) {
	for _, shape := range []struct {
		name                   string
		gpusPerNode, nodes, tp int
	}{
		{"fits on one node", 8, 2, 4},
		{"exactly one node", 8, 2, 8},
		{"spans two nodes", 8, 2, 16},
		{"spans three nodes", 4, 3, 12},
		{"spans eight small nodes", 2, 8, 16},
	} {
		t.Run(shape.name, func(t *testing.T) {
			pm := newTestPM([]NodePoolConfig{newTestPool("p", "H100", shape.gpusPerNode, shape.nodes)})
			_, gpuIDs, _, err := pm.PlaceInstance("inst-0", "m", "H100", shape.tp)
			require.NoError(t, err)

			// Cost accounting bills one unit per distinct node occupied.
			billedNodes := pm.InstanceCostPerHour(gpuIDs, 1.0)
			// Latency pricing scores the collective over the hosting node size.
			topo := sim.NewNetworkTopology(pm.placedGPUsPerNode(gpuIDs))
			pricedNodes := float64(topo.NodesSpanned(shape.tp))

			assert.Equal(t, billedNodes, pricedNodes,
				"the node span used for latency pricing must equal the span used for cost accounting")
		})
	}
}

// ─── Diagnostics (R1: a cross-node cost that silently fails to apply is the ───
// ─── invisible optimism this feature exists to remove) ────────────────────────

// spanningWarnings places a spanning instance through the given site and returns the
// warnings emitted while the topology was stamped.
func placeSpanningAndCaptureWarnings(t *testing.T, calib sim.HardwareCalib, backend string) string {
	t.Helper()
	return captureLogWarn(t, func() {
		stampPlacementTopology(t, 8, 2, 16, calib, backend)
	})
}

// TestPlacementTopology_WarnsWhenUncalibrated verifies the operator gets told when a
// spanning placement will NOT be charged because the placed GPU declares no
// interconnect bandwidths — the case a policy bundle's hw_config_by_gpu override
// makes easy to hit, since it replaces the whole calibration.
func TestPlacementTopology_WarnsWhenUncalibrated(t *testing.T) {
	out := placeSpanningAndCaptureWarnings(t, netTestCalib(0), "trained-physics")
	assert.Contains(t, out, "declares no usable interconnect bandwidths")
	assert.Contains(t, out, "IntraNodeBwGBps")
}

// TestPlacementTopology_WarnsWhenBackendHasNoCommTerm verifies the warning is
// backend-aware: under roofline, which models no communication at all, a spanning
// placement is unpriced no matter how well the fabric is calibrated.
func TestPlacementTopology_WarnsWhenBackendHasNoCommTerm(t *testing.T) {
	out := placeSpanningAndCaptureWarnings(t, netTestCalib(9), "roofline")
	assert.Contains(t, out, "models no communication term")
	assert.Contains(t, out, "trained-physics")
}

// TestPlacementTopology_WarnsOnImplausibleFabric verifies a unit mistake surfaces:
// an inter-node fabric three orders of magnitude slower than the on-node link is a
// typo, and would otherwise silently dominate step time.
func TestPlacementTopology_WarnsOnImplausibleFabric(t *testing.T) {
	out := placeSpanningAndCaptureWarnings(t, netTestCalib(5000), "trained-physics")
	assert.Contains(t, out, "looks like a unit error")
	assert.Contains(t, out, "per-GPU GB/s")
}

// TestPlacementTopology_NoWarningWhenPriced verifies the quiet path: a properly
// calibrated spanning placement on the trained-physics backend raises no complaint
// about pricing (the #1529 span notice is separate and expected).
func TestPlacementTopology_NoWarningWhenPriced(t *testing.T) {
	out := placeSpanningAndCaptureWarnings(t, netTestCalib(9), "trained-physics")
	assert.NotContains(t, out, "declares no usable interconnect bandwidths")
	assert.NotContains(t, out, "models no communication term")
	assert.NotContains(t, out, "looks like a unit error")
	assert.NotContains(t, out, "could not be resolved from placement")
}

// TestPlacementTopology_NoWarningWhenContained verifies a single-node placement never
// triggers a cross-node diagnostic, even on uncalibrated hardware — there is nothing
// to price.
func TestPlacementTopology_NoWarningWhenContained(t *testing.T) {
	out := captureLogWarn(t, func() {
		stampPlacementTopology(t, 16, 1, 16, netTestCalib(0), "trained-physics")
	})
	assert.NotContains(t, out, "declares no usable interconnect bandwidths")
}

// ─── All three placement sites (R23) ────────────────────────────────────────

// TestPlacementTopology_AppliedAtAllThreePlacementSites verifies BC-6: an instance
// created at startup, through the deferred NodeReadyEvent path, or by autoscaler
// scale-up all get the placement-derived topology. The observable is the cross-node
// diagnostic, which only the topology-stamping step emits — so its presence proves
// that step ran at that site, and its absence would prove the site was missed.
func TestPlacementTopology_AppliedAtAllThreePlacementSites(t *testing.T) {
	// Uncalibrated fabric, so a spanning placement emits the "will not be priced"
	// warning at whichever site stamps the topology.
	baseSimCfg := func() sim.SimConfig {
		return sim.SimConfig{
			Horizon:              math.MaxInt64,
			Seed:                 42,
			KVCacheConfig:        sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:          sim.NewBatchConfig(256, 8192, 0),
			LatencyModelOverride: testFakeLatency(),
			ModelHardwareConfig:  sim.NewModelHardwareConfig(netTestModelConfig(), netTestCalib(0), "m", "H100", 16, 1, false, "", "trained-physics", 0),
		}
	}
	const wantWarning = "declares no usable interconnect bandwidths"

	t.Run("startup", func(t *testing.T) {
		cfg := DeploymentConfig{
			SimConfig:    baseSimCfg(),
			NumInstances: 1,
			NodePools:    []NodePoolConfig{newTestPool("p", "H100", 8, 2)}, // tp=16 must span
		}
		out := captureLogWarn(t, func() {
			cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
			require.Len(t, cs.instances, 1, "startup must place the instance")
		})
		assert.Contains(t, out, wantWarning, "the startup placement site must stamp the topology")
	})

	t.Run("deferred_node_ready", func(t *testing.T) {
		cfg := DeploymentConfig{
			SimConfig:    baseSimCfg(),
			NumInstances: 1,
			// InitialNodes=0 → the instance is pending until a node becomes Ready.
			NodePools: []NodePoolConfig{{Name: "p", GPUType: "H100", GPUsPerNode: 8, GPUMemoryGiB: 80, InitialNodes: 0, MaxNodes: 4}},
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
		require.Empty(t, cs.instances, "precondition: no instance before a node is Ready")

		// Two nodes must be Ready before a tp=16 whole-node span can be satisfied.
		nodeA, _ := cs.placement.ProvisionNode("p", 0)
		nodeB, _ := cs.placement.ProvisionNode("p", 0)
		require.NotNil(t, nodeA)
		require.NotNil(t, nodeB)
		(&NodeReadyEvent{timestamp: 0, nodeID: nodeA.ID}).Execute(cs)

		out := captureLogWarn(t, func() {
			(&NodeReadyEvent{timestamp: 0, nodeID: nodeB.ID}).Execute(cs)
		})
		require.Len(t, cs.instances, 1, "precondition: the deferred instance must be placed once both nodes are Ready")
		assert.Contains(t, out, wantWarning, "the deferred NodeReadyEvent placement site must stamp the topology")
	})

	t.Run("autoscaler_scale_up", func(t *testing.T) {
		cfg := DeploymentConfig{
			SimConfig:    baseSimCfg(),
			NumInstances: 1,
			// 4 nodes: the startup instance spans two, the scaled-up one spans the rest.
			NodePools: []NodePoolConfig{newTestPool("p", "H100", 8, 4)},
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
		require.Len(t, cs.instances, 1)
		// The startup site has already latched its warning; clear the latches so this
		// subtest observes the scale-up site's own diagnostic.
		cs.crossNodeBackendWarned = false
		cs.crossNodeUnresolvedWarned = false
		cs.crossNodeUncalibratedWarned = false
		cs.implausibleFabricWarned = false

		out := captureLogWarn(t, func() {
			err := NewDirectActuator(cs).Apply([]ScaleDecision{
				{ModelID: "m", Variant: NewVariantSpec("H100", 16), Delta: 1},
			})
			require.NoError(t, err)
		})
		require.Len(t, cs.instances, 2, "precondition: scale-up must add an instance")
		assert.Contains(t, out, wantWarning, "the autoscaler scale-up placement site must stamp the topology")
	})
}

// ─── INV-6 and the trace-header signal ──────────────────────────────────────

// TestPlacementTopology_SpanningRunIsByteIdenticalAcrossRuns verifies INV-6 for the case
// this feature actually changes: two identical SPANNING runs at the same seed must
// produce byte-identical output. The three inertness tests prove the feature does not
// perturb configs it should not touch; this proves the configs it DOES touch stay
// deterministic. The comparison is on the marshalled metrics payload — the same struct
// stdout is rendered from — so it is a genuine byte-level check rather than a
// field-by-field one.
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
			SimConfig: sim.SimConfig{
				Horizon:              math.MaxInt64,
				Seed:                 42,
				KVCacheConfig:        sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
				BatchConfig:          sim.NewBatchConfig(256, 8192, 0),
				LatencyModelOverride: testFakeLatency(),
				ModelHardwareConfig:  sim.NewModelHardwareConfig(netTestModelConfig(), netTestCalib(9), "m", "H100", 16, 1, false, "", "trained-physics", 0),
			},
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
			SimConfig: sim.SimConfig{
				Horizon:              math.MaxInt64,
				Seed:                 42,
				KVCacheConfig:        sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
				BatchConfig:          sim.NewBatchConfig(256, 8192, 0),
				LatencyModelOverride: testFakeLatency(),
				ModelHardwareConfig:  sim.NewModelHardwareConfig(netTestModelConfig(), netTestCalib(9), "m", "H100", tp, 1, false, "", "trained-physics", 0),
			},
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
