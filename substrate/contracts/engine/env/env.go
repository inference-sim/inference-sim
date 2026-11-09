// Package env is what an engine may ask of the substrate while it is built and run (design
// §6.3). inference-sim implements Environment with the cost model and the data loader; an
// engine imports only packages under contracts/, and its tests use the reference implementation
// in contracts/conformance/envtest.
//
// An engine is built in two phases. In the first it resolves its configuration, lowers the
// model into a partial PoolPlan whose KV budget is still empty, and asks Memory for the
// resident weights, the buffers and the activations of its profiling shape; it then applies
// its upstream's budget rule and fills in KVBudget. After that it forms a StepShape each step
// and asks Price for its time.
package env

import (
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// Environment is what an engine asks of the substrate.
type Environment interface {
	// Graph returns the catalog's model graph.
	Graph(modelName string) (*model.Graph, error)
	// Chip returns the catalog's hardware facts.
	Chip(name string) (*hardware.Chip, error)
	// Memory sums a partial plan's resident bytes and sizes what the cost model owns:
	// activations of the plan's profiling shape, and graph pools. Its result is per stage.
	Memory(partial plan.PoolPlan) (kernel.MemoryBreakdown, error)
	// Price returns the time of one step. The step is already padded; the cost model prices the
	// padded shape and chooses nothing.
	Price(p plan.PoolPlan, step plan.StepShape) (kernel.StepEstimate, error)
	// TransferDemand returns what a transfer asks of the fabric: its payload, its floor and the
	// links it crosses. Sharing a link among transfers that overlap is the simulator's work.
	TransferDemand(p plan.PoolPlan, t plan.Transfer) (kernel.TransferDemand, error)
	// Acceptance returns the acceptance profile for a model, its drafter and a workload.
	Acceptance(modelName, drafter, workload string) (acceptance.Profile, error)
}
