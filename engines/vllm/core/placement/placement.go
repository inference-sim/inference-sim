// Original to blis-vllm.

// Package placement is where a pool's engine sits, in the terms the boundary types use: the catalog
// chip and network, the parallel widths and how the ranks fill nodes, and the node and rack of the
// engine itself. The engine states its transfers against it (engine.Transfer.Move) and builds the
// plan they are priced against from it (engine.Planned).
//
// It is the placement alone. The rest of a PoolPlan (the lowered layers, the KV groups, the captured
// graphs, the memory breakdown) is the cost model's resolution, which has not moved into this
// repository; see Plan.
package placement

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// CPUDRAM is the catalog device the engine's CPU KV tier stands for.
const CPUDRAM = "cpu_dram"

// Placement is the pool an engine is one of, and the engine's place in it.
type Placement struct {
	// Hardware is the catalog chip's name and Fabric the catalog network's, "" on one node.
	Hardware string `yaml:"hardware"`
	Fabric   string `yaml:"fabric,omitempty"`
	// The parallel widths of the pool (a width of 0 is read as 1), and the nodes that hold its ranks:
	// Nodes nodes of RanksPerNode ranks each.
	Tensor       int `yaml:"tensor,omitempty"`
	Pipeline     int `yaml:"pipeline,omitempty"`
	Data         int `yaml:"data,omitempty"`
	Expert       int `yaml:"expert,omitempty"`
	Context      int `yaml:"context,omitempty"`
	RanksPerNode int `yaml:"ranks_per_node,omitempty"`
	Nodes        int `yaml:"nodes,omitempty"`
	// NodeRacks is the rack of each node of the pool, in node order; empty is every node in rack 0.
	NodeRacks []int `yaml:"node_racks,omitempty"`
	// Node is the pool's node this engine's workers sit on, and the node that holds its CPU KV tier.
	Node int `yaml:"node,omitempty"`
}

// Layout is the pool's parallel widths and rank placement as the plan states them.
func (p Placement) Layout() plan.Layout {
	return plan.Layout{Tensor: p.Tensor, Pipeline: p.Pipeline, Data: p.Data, Expert: p.Expert, Context: p.Context,
		RanksPerNode: p.RanksPerNode, Nodes: p.Nodes}
}

// Validate refuses a placement the plan would refuse, or that puts the engine on a node the pool lacks.
func (p Placement) Validate() error {
	if strings.TrimSpace(p.Hardware) == "" {
		return fmt.Errorf("placement: no hardware; a plan names its catalog chip")
	}
	if p.Node < 0 || p.Node >= max(p.Nodes, 1) {
		return fmt.Errorf("placement: the engine is on node %d of a pool with %d node(s)", p.Node, max(p.Nodes, 1))
	}
	if n := len(p.NodeRacks); n != 0 && n != max(p.Nodes, 1) {
		return fmt.Errorf("placement: %d racks for %d node(s); there is one rack per node", n, max(p.Nodes, 1))
	}
	for i, r := range p.NodeRacks {
		if r < 0 {
			return fmt.Errorf("placement: node %d is in rack %d", i, r)
		}
	}
	return (plan.PoolPlan{Hardware: p.Hardware, Fabric: p.Fabric, Layout: p.Layout()}).Validate()
}

// Endpoint is where this engine's workers send and receive: its node and that node's rack.
func (p Placement) Endpoint() plan.Endpoint { return p.EndpointOf(p.Node) }

// EndpointOf is node n of the pool and its rack.
func (p Placement) EndpointOf(n int) plan.Endpoint {
	rack := 0
	if n >= 0 && n < len(p.NodeRacks) {
		rack = p.NodeRacks[n]
	}
	return plan.Endpoint{Node: n, Rack: rack}
}

// ForRank is the placement of data-parallel replica rank, whose workers are ranksPerReplica
// consecutive ranks of the pool, filling nodes in order (a replica's first rank decides its node).
func (p Placement) ForRank(rank, ranksPerReplica int) Placement {
	if p.RanksPerNode > 0 {
		p.Node = min(rank*max(ranksPerReplica, 1)/p.RanksPerNode, max(p.Nodes, 1)-1)
	}
	return p
}

// Plan is the PoolPlan this placement and its KV tiers describe. It states the pool's chip, network,
// layout and KV tiers, which are the engine's own; it does not state the lowered layers, the KV groups,
// the captured graphs or the memory breakdown, which are the cost model's resolution and are not moved
// here (blis-latency-kernel#84). A transfer is priced against the plan's tiers and layout alone
// (PoolPlan.CheckTransfer). The result shares no memory with the placement.
func (p Placement) Plan(tiers []plan.KVTier) plan.PoolPlan {
	return plan.PoolPlan{Hardware: p.Hardware, Fabric: p.Fabric, Layout: p.Layout(), Tiers: append([]plan.KVTier(nil), tiers...)}
}
