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
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine"
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
	// HostDemand returns what a request of promptTokens asks of the plan's admission queue
	// (plan.HostPlan.Admission): the time it holds a server and the time it only waits. Sharing
	// the servers among requests is the simulator's work.
	HostDemand(p plan.PoolPlan, promptTokens int) (kernel.HostDemand, error)
	// Acceptance returns the acceptance profile for a model, its drafter and a workload.
	Acceptance(modelName, drafter, workload string) (acceptance.Profile, error)
}

// DemandFor is the boundary a transfer crosses (design §6.2): a Transfer an engine returned in a
// Batch's TransfersStarted becomes the demand its Move makes of the fabric. A simulator calls
// it for each reported transfer, with the plan the engine states (engine.Planned) and the
// engine's own Move, and fills in nothing: the engine decided the tier, the endpoints and the
// direction, and the simulator only shares the links they cross.
//
// The demand is for the one move Move states. That is exact for a pool on one node and for a
// model with one KV group. For a pool that spans nodes it is the demand on the node Move names
// (inference-sim/blis-substrate#48), and for a hybrid model it is sized from Move.Tokens, not
// from the blocks each KV group moves (inference-sim/blis-substrate#49); both are deferred to M3.
//
// It refuses a transfer whose report and Move disagree (engine.Transfer.Validate) and a move
// the plan cannot make (plan.PoolPlan.CheckTransfer), so a transfer that was never connected
// to the cost model fails here and not as a zero price.
func DemandFor(e Environment, p plan.PoolPlan, t engine.Transfer) (kernel.TransferDemand, error) {
	if err := t.Validate(); err != nil {
		return kernel.TransferDemand{}, err
	}
	if err := p.CheckTransfer(t.Move); err != nil {
		return kernel.TransferDemand{}, fmt.Errorf("env: transfer %d of request %q: %w", t.ID, t.Request, err)
	}
	return e.TransferDemand(p, t.Move)
}
