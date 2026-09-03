package cluster

import (
	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// maxPlausibleInterconnectRatio bounds how much slower an inter-node fabric can
// credibly be than an on-node link before the numbers look like a unit mistake.
// Real clusters land around 5–75× (NVLink 450 GB/s per GPU against anything from a
// 400 Gb/s NIC per GPU down to a single 100 GbE for a whole node). Three orders of
// magnitude is not a fabric, it is a typo — and because the penalty is linear in the
// ratio, such a value inflates step time until it clamps, destroying the run. Warn
// rather than reject: the model stays well-defined, and a user deliberately probing
// an extreme is not blocked.
const maxPlausibleInterconnectRatio = 1000.0

// latencyBackendTrainedPhysics is the only backend that models communication, and
// therefore the only one a cross-node penalty can apply to. Matches the name
// validated by sim.IsValidLatencyBackend.
const latencyBackendTrainedPhysics = "trained-physics"

// applyPlacementTopology stamps the placement-derived inter-node interconnect
// topology onto a per-instance SimConfig (#1530), so the latency model can price
// collective traffic that crosses a node boundary. Called from all three placement
// sites — startup, the deferred NodeReadyEvent path, and autoscaler scale-up — right
// after the per-instance GPU type (#893), KV capacity (#1522) and hourly cost (#1529)
// are resolved from the same placement (R23).
//
// A topology that cannot be resolved (no node pools, or an unresolvable GPU set)
// leaves the config inert, so step time is byte-identical to a pre-#1530 build
// (INV-6).
//
// It also raises the two diagnostics that a spanning placement needs (R1 — a
// cross-node cost that silently fails to apply is exactly the invisible optimism
// this feature exists to remove), each latched so a large fleet emits one line:
//
//   - the placement spans nodes but nothing will be charged, because the resolved
//     GPU calibration declares no usable interconnect bandwidths, or because the
//     configured latency backend models no communication at all. The first case is easy to hit
//     without noticing: a policy bundle's hw_config_by_gpu entry REPLACES the whole
//     HardwareCalib, so a per-pool calibration that omits the two fabric bandwidths
//     drops them — and hw_config_by_gpu is the documented way to calibrate exactly
//     the mixed-pool deployments this feature targets.
//   - the calibration is present but implausible (see maxPlausibleInterconnectRatio).
//
// Assumption worth knowing: the diagnostics score the span of simCfg.TP, the group the
// latency model will actually price, while the GPU set comes from the degree placement
// reserved. Those agree today at every site — the autoscaler's variant inventory is
// seeded from the cluster's own TP, and #1529 rejects a per-role TP override that
// differs from the global TP when node pools are configured. If they ever diverged, the
// priced group and the reserved group would describe different placements.
func (cs *ClusterSimulator) applyPlacementTopology(simCfg *sim.SimConfig, gpuIDs []string) {
	if cs.placement == nil {
		return // no node pools ⇒ no placement ⇒ topology stays unknown (inert)
	}
	topo := sim.NewNetworkTopology(cs.placement.PlacedGPUsPerNode(gpuIDs))
	simCfg.NetworkTopology = topo

	if topo.NodesSpanned(simCfg.TP) <= 1 {
		return // contained in one node — nothing to price, nothing to warn about
	}
	ratio := simCfg.HWConfig.InterconnectBwRatio()
	switch {
	case simCfg.Backend != latencyBackendTrainedPhysics:
		if !cs.crossNodeUnpricedWarned {
			cs.crossNodeUnpricedWarned = true
			logrus.Warnf("[cluster] an instance spans %d nodes for TP=%d, but the %q latency backend "+
				"models no communication term, so its cross-node collective traffic is unpriced and "+
				"latency/throughput for spanning instances are optimistic (#1530). Use "+
				"--latency-model trained-physics to price it",
				topo.NodesSpanned(simCfg.TP), simCfg.TP, backendDisplayName(simCfg.Backend))
		}
	case ratio == 1.0:
		if !cs.crossNodeUnpricedWarned {
			cs.crossNodeUnpricedWarned = true
			logrus.Warnf("[cluster] an instance spans %d nodes for TP=%d, but the hardware calibration "+
				"for GPU %q declares no usable interconnect bandwidths (IntraNodeBwGBps/InterNodeBwGBps), so "+
				"its cross-node collective traffic is priced at the on-node rate and latency/throughput for "+
				"spanning instances are optimistic (#1530). Add both bandwidths to the hardware config "+
				"(and to any hw_config_by_gpu override, which replaces the whole calibration)",
				topo.NodesSpanned(simCfg.TP), simCfg.TP, simCfg.GPU)
		}
	case ratio > maxPlausibleInterconnectRatio:
		if !cs.implausibleFabricWarned {
			cs.implausibleFabricWarned = true
			logrus.Warnf("[cluster] hardware calibration for GPU %q implies an inter-node fabric %.0f× "+
				"slower than its on-node link (IntraNodeBwGBps=%v, InterNodeBwGBps=%v) — that is far outside "+
				"the plausible 5–75× range and will dominate step time for spanning instances; check the "+
				"units (both are per-GPU GB/s)",
				simCfg.GPU, ratio, simCfg.HWConfig.IntraNodeBwGBps, simCfg.HWConfig.InterNodeBwGBps)
		}
	}
}

// backendDisplayName renders a latency backend for a diagnostic, spelling out the
// empty string (which resolves to roofline) rather than printing "".
func backendDisplayName(backend string) string {
	if backend == "" {
		return "roofline (default)"
	}
	return backend
}
