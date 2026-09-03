package cluster

import (
	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// maxPlausibleInterconnectRatio bounds how much slower an inter-node fabric can
// credibly be than an on-node link before the numbers look like a unit mistake.
// Real hardware lands roughly between 2× (a PCIe node, whose on-node link is itself
// slow — the committed L40S entry is 2.6×) and ~75× (NVLink against a single 100 GbE
// uplink shared by a whole node). The threshold is set far above that band, at three
// orders of magnitude, on purpose: it is a UNIT-ERROR detector, not a plausibility
// judgement. Anything near the band could be a real if unusual cluster, and warning
// about it would train operators to ignore the message; a 1000× ratio is a mistyped
// exponent or a bits-vs-bytes slip, and because the penalty is linear in the ratio it
// will dominate step time until it clamps. Warn rather than reject, so a user
// deliberately probing an extreme is not blocked.
const maxPlausibleInterconnectRatio = 1000.0

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
// The diagnostics live in warnIfCrossNodeUnpriced.
func (cs *ClusterSimulator) applyPlacementTopology(simCfg *sim.SimConfig, gpuIDs []string) {
	if cs.placement == nil {
		return // no node pools ⇒ no placement ⇒ topology stays unknown (inert)
	}
	topo := sim.NewNetworkTopology(cs.placement.placedGPUsPerNode(gpuIDs))
	simCfg.NetworkTopology = topo

	// Record the widest span in the fleet. `blis run --trace-output` writes it into the
	// trace header so replay can refuse a trace whose step times it cannot reproduce
	// (#1530). Derived from the distinct nodes actually occupied — NOT from
	// topo.NodesSpanned — so it is right even when the node size could not be resolved
	// and the topology itself came out inert.
	realSpan := len(cs.placement.distinctNodesForGPUs(gpuIDs))
	if realSpan > cs.maxNodesSpanned {
		cs.maxNodesSpanned = realSpan
	}
	cs.warnIfCrossNodeUnpriced(simCfg, topo, realSpan)
}

// warnIfCrossNodeUnpriced raises the diagnostics a spanning placement needs (#1530, R1
// — a cross-node cost that silently fails to apply is exactly the invisible optimism
// this feature exists to remove). Each is latched independently so a large fleet emits
// one line per distinct cause rather than one line total.
//
// Three causes, all of which leave a genuinely spanning instance priced as if it never
// left the node:
//
//   - the configured latency backend models no communication at all;
//   - the resolved GPU calibration declares no usable interconnect bandwidths and no
//     per-collective latency, so there is no fabric to price against. Note the
//     calibration a node-pool instance gets is the one --hardware-config resolved for
//     the --hardware GPU: a DeploymentConfig.HWConfigByGPU entry would override it per
//     placed pool, but that field has no policy-bundle key today (issue #893), so a
//     mixed-gpu_type fleet currently shares one calibration — including these fields;
//   - the node size could not be resolved at all (mixed-size span, or unresolvable GPU
//     ids), so the topology is inert even though the placement really does span. This
//     is why the span passed in is the REAL distinct-node count rather than
//     topo.NodesSpanned — scoring the diagnostic off the topology would make it go
//     quiet in exactly the case where it is most needed.
//
// A fourth, non-fatal cause warns separately: the calibration is present but its ratio
// looks like a unit error (see maxPlausibleInterconnectRatio).
//
// Assumption worth knowing: the diagnostics score the WIDEST collective group the latency
// model will price on this instance (widestCollectiveGroup) — the TP group, or the
// expert-owning group when expert parallelism makes that wider. Scoring only TP, as this
// did before #1548, would let a MoE-group-only span (TP fits inside a node, the expert
// group does not) be priced without being diagnosed. The GPU set still comes from the
// degree placement reserved: the autoscaler's variant inventory is seeded from the
// cluster's own TP, and #1529 rejects a per-role TP override that differs from the global
// TP when node pools are configured.
func (cs *ClusterSimulator) warnIfCrossNodeUnpriced(simCfg *sim.SimConfig, topo sim.NetworkTopology, realSpan int) {
	if realSpan <= 1 {
		return // contained in one node — nothing to price, nothing to warn about
	}
	group := widestCollectiveGroup(simCfg)
	switch {
	case simCfg.Backend != sim.LatencyBackendTrainedPhysics:
		if !cs.crossNodeBackendWarned {
			cs.crossNodeBackendWarned = true
			logrus.Warnf("[cluster] an instance spans %d nodes for a %d-GPU collective group, but the %q latency backend "+
				"models no communication term, so its cross-node collective traffic is unpriced and "+
				"latency/throughput for spanning instances are optimistic (#1530). Use "+
				"--latency-model trained-physics to price it",
				realSpan, group, backendDisplayName(simCfg.Backend))
		}
	case !topo.IsKnown():
		if !cs.crossNodeUnresolvedWarned {
			cs.crossNodeUnresolvedWarned = true
			logrus.Warnf("[cluster] an instance spans %d nodes for a %d-GPU collective group, but its node size could not be "+
				"resolved from placement, so cross-node collective traffic is priced at the on-node rate "+
				"and latency/throughput for spanning instances are optimistic (#1530). See the "+
				"PlacedGPUsPerNode errors above for the cause", realSpan, group)
		}
	case !simCfg.HWConfig.HasInterconnectCalibration():
		if !cs.crossNodeUncalibratedWarned {
			cs.crossNodeUncalibratedWarned = true
			logrus.Warnf("[cluster] an instance spans %d nodes for a %d-GPU collective group, but the hardware calibration "+
				"for GPU %q declares no usable interconnect bandwidths (IntraNodeBwGBps/InterNodeBwGBps) "+
				"and no per-collective latency (InterNodeLatencyUs), so its cross-node collective traffic "+
				"is priced at the on-node rate and latency/throughput for spanning instances are "+
				"optimistic (#1530). Add them to the entry for this GPU in --hardware-config",
				realSpan, group, simCfg.GPU)
		}
	case simCfg.HWConfig.InterconnectBwRatio() > maxPlausibleInterconnectRatio:
		if !cs.implausibleFabricWarned {
			cs.implausibleFabricWarned = true
			logrus.Warnf("[cluster] hardware calibration for GPU %q implies an inter-node fabric %.0f× "+
				"slower than its on-node link (IntraNodeBwGBps=%v, InterNodeBwGBps=%v) — real hardware "+
				"lands roughly between 2× and 75×, so this looks like a unit error and will dominate step "+
				"time for spanning instances; check that both values are per-GPU GB/s",
				simCfg.GPU, simCfg.HWConfig.InterconnectBwRatio(),
				simCfg.HWConfig.IntraNodeBwGBps, simCfg.HWConfig.InterNodeBwGBps)
		}
	}
}

// backendDisplayName renders a latency backend for a diagnostic, spelling out the
// empty string (which resolves to roofline) rather than printing "".
func backendDisplayName(backend string) string {
	if backend == "" {
		return sim.LatencyBackendRoofline + " (default)"
	}
	return backend
}

// widestCollectiveGroup is the largest participant group whose traffic the latency model
// prices for one instance: its TP group, or — when expert parallelism is in force on an
// MoE model — the wider expert-owning group the dispatch/combine all-to-all runs over
// (#1548). The diagnostics score it so a span that only the wider group crosses is still
// reported (R1: a cross-node cost that silently fails to apply is the invisible optimism
// #1530 exists to remove).
//
// Reachability, stated honestly: the two groups can only differ at --dp>1, which remains a
// fail-fast alongside node pools (#1553), and a topology is only ever resolved from node
// pools — so today this always returns simCfg.TP and the diagnostics are byte-identical to
// their pre-#1548 text for every reachable config. It is here so that whoever lifts #1553
// inherits a correct diagnostic rather than a silent gap.
func widestCollectiveGroup(simCfg *sim.SimConfig) int {
	if g := simCfg.EffectiveExpertShardGroupSize(); g > simCfg.TP {
		return g
	}
	return simCfg.TP
}
