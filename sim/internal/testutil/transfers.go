package testutil

import (
	"fmt"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// PullEnv is the Environment a host test gives an EngineDriver: it answers TransferDemand and
// refuses every other question by name, so a test that reaches for a step price learns it. A
// pull between pools is priced by Pull, for the whole blocks the tokens occupy; a move to or
// from a tier by Tier (nil refuses it).
type PullEnv struct {
	// BlockTokens is the tokens in one KV block: a pull moves whole blocks, so the tokens Pull
	// receives are the blocks' capacity, as the host's cost model prices them.
	BlockTokens int64
	// Pull is the demand of a pull of tokens (a multiple of BlockTokens, at least one block)
	// from the endpoint from to the endpoint to.
	Pull func(tokens int64, from, to plan.Endpoint) kernel.TransferDemand
	// Tier is the demand of an offload copy; nil refuses one.
	Tier func(t plan.Transfer) kernel.TransferDemand
}

var _ env.Environment = PullEnv{}

// FixedPull is a PullEnv whose pulls take exactly ticks(tokens, from, to) microseconds alone: a
// demand of no payload and a floor, which costs its floor and shares no link.
func FixedPull(blockTokens int64, ticks func(tokens int64, from, to plan.Endpoint) int64) PullEnv {
	return PullEnv{BlockTokens: blockTokens, Pull: func(tokens int64, from, to plan.Endpoint) kernel.TransferDemand {
		return kernel.TransferDemand{Floor: time.Duration(ticks(tokens, from, to)) * time.Microsecond}
	}}
}

// TransferDemand implements env.Environment.
func (e PullEnv) TransferDemand(p plan.PoolPlan, t plan.Transfer) (kernel.TransferDemand, error) {
	if err := p.CheckTransfer(t); err != nil {
		return kernel.TransferDemand{}, err
	}
	if t.Kind == plan.TransferPeer {
		if t.Tokens == 0 {
			return kernel.TransferDemand{}, nil
		}
		blocks := (int64(t.Tokens) + e.BlockTokens - 1) / e.BlockTokens
		return e.Pull(blocks*e.BlockTokens, t.From, t.To), nil
	}
	if e.Tier == nil {
		return kernel.TransferDemand{}, fmt.Errorf("testutil: this PullEnv prices no tier copy")
	}
	return e.Tier(t), nil
}

func (PullEnv) refuse(what string) error {
	return fmt.Errorf("testutil: PullEnv answers transfers only, not %s", what)
}

// Graph implements env.Environment.
func (e PullEnv) Graph(string) (*model.Graph, error) { return nil, e.refuse("a graph") }

// Chip implements env.Environment.
func (e PullEnv) Chip(string) (*hardware.Chip, error) { return nil, e.refuse("a chip") }

// Memory implements env.Environment.
func (e PullEnv) Memory(plan.PoolPlan) (kernel.MemoryBreakdown, error) {
	return kernel.MemoryBreakdown{}, e.refuse("memory")
}

// Price implements env.Environment.
func (e PullEnv) Price(plan.PoolPlan, plan.StepShape) (kernel.StepEstimate, error) {
	return kernel.StepEstimate{}, e.refuse("a step price")
}

// HostDemand implements env.Environment.
func (e PullEnv) HostDemand(plan.PoolPlan, int) (kernel.HostDemand, error) {
	return kernel.HostDemand{}, e.refuse("an admission demand")
}

// Acceptance implements env.Environment.
func (e PullEnv) Acceptance(string, string, string) (acceptance.Profile, error) {
	return acceptance.Profile{}, e.refuse("an acceptance profile")
}
