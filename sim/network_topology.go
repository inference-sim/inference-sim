package sim

import "fmt"

// NetworkTopology is the 9th module sub-config (inter-node interconnect topology,
// #1530). It carries the ONE placement-derived signal the latency model needs to
// price cross-node collective traffic: how many GPUs sit on the node(s) an
// instance was actually placed on.
//
// Provenance — placement, never a flag. The value is produced only by
// PlacementManager (sim/cluster) from the nodes an instance's GPUs actually
// occupy, and stamped onto the per-instance SimConfig at the same three placement
// sites that already stamp the per-instance GPU type (#893), KV capacity (#1522)
// and hourly cost (#1529). There is deliberately NO CLI flag: a declared
// gpus-per-node knob can contradict the real node_pools placement, which is
// exactly the failure mode this signal exists to avoid.
//
// The zero value is inert. Without node_pools there is no placement, so
// PlacedGPUsPerNode stays 0, NodesSpanned always reports 1, and every latency
// term is byte-identical to a pre-#1530 build (INV-6, INV-BC-DP1). Because node
// pools are `blis run`-only (`blis replay`/`observe` reject them with a fatal
// error), a non-inert topology is unreachable from replay — so no TraceV2
// round-trip is needed for run/replay parity (INV-13 parity N/A, the same
// treatment #1522/#1529/#1531 took).
type NetworkTopology struct {
	// PlacedGPUsPerNode is the GPU count of the node(s) hosting this instance, as
	// resolved from real placement. 0 = no node-pool placement (topology unknown)
	// ⇒ inert. A negative value is treated as unknown by every accessor.
	PlacedGPUsPerNode int
}

// NewNetworkTopology creates a NetworkTopology with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// A non-positive placedGPUsPerNode yields the inert zero value rather than an
// error: "topology unknown" is a legitimate, common state (no node pools), and
// the accessors below all degrade to single-node behavior.
func NewNetworkTopology(placedGPUsPerNode int) NetworkTopology {
	if placedGPUsPerNode < 0 {
		placedGPUsPerNode = 0
	}
	return NetworkTopology{PlacedGPUsPerNode: placedGPUsPerNode}
}

// IsKnown reports whether a real node size was resolved from placement. When
// false every consumer must behave exactly as it did before #1530 (INV-6).
func (t NetworkTopology) IsKnown() bool {
	return t.PlacedGPUsPerNode > 0
}

// NodesSpanned returns how many physical nodes a collective over groupSize GPUs
// occupies: ceil(groupSize / PlacedGPUsPerNode). Returns 1 — single node, no
// cross-node traffic — when the topology is unknown, when groupSize is
// non-positive, or when the whole group fits inside one node.
//
// For a tensor-parallel group this is exactly the instance's real node span:
// PlaceInstance only ever spans nodes under whole-node occupancy (tpDegree >
// gpus_per_node AND tpDegree % gpus_per_node == 0, #1529), so the ceiling is
// exact rather than approximate there. For a larger conceptual group (the lumped
// MoE TP·DP group) the uniform-per-node assumption is an approximation, noted on
// the caller.
func (t NetworkTopology) NodesSpanned(groupSize int) int {
	if !t.IsKnown() || groupSize <= t.PlacedGPUsPerNode {
		return 1
	}
	return (groupSize + t.PlacedGPUsPerNode - 1) / t.PlacedGPUsPerNode
}

// MembersPerNode returns how many members of a collective over groupSize GPUs
// share a node: min(groupSize, PlacedGPUsPerNode), or groupSize when the topology
// is unknown (everything is treated as co-located). Returns 0 for a non-positive
// groupSize so callers see an empty group rather than a bogus count.
func (t NetworkTopology) MembersPerNode(groupSize int) int {
	if groupSize <= 0 {
		return 0
	}
	if !t.IsKnown() || groupSize < t.PlacedGPUsPerNode {
		return groupSize
	}
	return t.PlacedGPUsPerNode
}

// Validate checks the NetworkTopology for internal consistency. It is a pure
// query; the library boundary returns an error and the caller decides fatality
// (cmd/ → logrus.Fatalf, sim/ factory → error). The zero value is valid and
// inert.
//
// Note: this method is NOT usable through SimConfig's promoted selector —
// LoRAConfig already promotes a Validate() at the same embedding depth, which
// makes the promoted name ambiguous (the same situation SpeculativeConfig
// documents). Call cfg.NetworkTopology.Validate() explicitly if ever needed.
func (t NetworkTopology) Validate() error {
	if t.PlacedGPUsPerNode < 0 {
		return fmt.Errorf("NetworkTopology: PlacedGPUsPerNode must be >= 0, got %d", t.PlacedGPUsPerNode)
	}
	return nil
}
