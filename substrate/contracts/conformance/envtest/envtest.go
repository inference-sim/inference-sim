// Package envtest is a small reference implementation of env.Environment that prices from
// fixed test values, and the checks every implementation of the Environment contract runs
// against itself (design §6.3). Engine tests use the reference; inference-sim's own
// implementation runs the checks.
//
// The reference is not a cost model. It prices a step as a base plus a rate per launched token,
// so a test can predict every figure it asserts, and it knows no engine.
package envtest

import (
	"fmt"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"
)

// Reference prices from fixed values. The zero value is not usable: set the maps and rates.
type Reference struct {
	Graphs   map[string]*model.Graph
	Chips    map[string]*hardware.Chip
	Profiles map[[3]string]acceptance.Profile
	// StepBase and PerToken price a step: StepBase plus PerToken for each launched token on the
	// busiest rank.
	StepBase time.Duration
	PerToken time.Duration
	// ActivationPerToken sizes activations per token of the plan's profiling shape, the cost
	// model's share of Memory.
	ActivationPerToken units.Bytes
	// BytesPerSecond is the one link every transfer crosses.
	BytesPerSecond float64
	// BytesPerToken is the KV bytes a transferred token carries.
	BytesPerToken units.Bytes
}

var _ env.Environment = (*Reference)(nil)

// Boundary returns the simulator's side of the transfer boundary for enginetest.Options: each
// transfer an engine starts goes through env.DemandFor against e, and the demand must carry a
// payload over at least one link (zero bytes for a move of no tokens), the same laws Check holds
// an Environment to. It is what a
// simulator does for every entry of Batch.TransfersStarted.
func Boundary(e env.Environment) func(plan.PoolPlan, engine.Transfer) error {
	return func(p plan.PoolPlan, t engine.Transfer) error {
		d, err := env.DemandFor(e, p, t)
		if err != nil {
			return err
		}
		if t.Move.Tokens == 0 {
			if d.Bytes != 0 {
				return fmt.Errorf("transfer %d moves no tokens and has demand %+v; it demands zero bytes", t.ID, d)
			}
			return nil
		}
		if d.Bytes <= 0 || len(d.Links) == 0 {
			return fmt.Errorf("transfer %d of %d tokens has demand %+v; it carries a payload over at least one link", t.ID, t.Tokens, d)
		}
		return nil
	}
}

// Graph implements env.Environment.
func (r *Reference) Graph(name string) (*model.Graph, error) {
	if g, ok := r.Graphs[name]; ok {
		return g, nil
	}
	return nil, fmt.Errorf("envtest: no model %q", name)
}

// Chip implements env.Environment.
func (r *Reference) Chip(name string) (*hardware.Chip, error) {
	if c, ok := r.Chips[name]; ok {
		return c, nil
	}
	return nil, fmt.Errorf("envtest: no chip %q", name)
}

// Memory implements env.Environment: the plan's GPU weights and buffers, plus activations for
// one launched token count. A partial plan has no KV budget yet and needs none.
func (r *Reference) Memory(partial plan.PoolPlan) (kernel.MemoryBreakdown, error) {
	if err := partial.Validate(); err != nil {
		return kernel.MemoryBreakdown{}, err
	}
	m, err := plan.MemoryOf(partial, max(1, partial.Layout.RanksPerNode))
	if err != nil {
		return kernel.MemoryBreakdown{}, err
	}
	return kernel.MemoryBreakdown{
		Weights:        int64(m.WeightsGPU.Value()),
		ActivationPeak: int64(r.ActivationPerToken) * int64(max(1, partial.Layout.Tensor)),
		CommBuffers:    int64(m.BuffersGPU.Value()),
	}, nil
}

// Price implements env.Environment.
func (r *Reference) Price(p plan.PoolPlan, step plan.StepShape) (kernel.StepEstimate, error) {
	if err := p.Validate(); err != nil {
		return kernel.StepEstimate{}, err
	}
	if err := step.Validate(max(1, p.Layout.Data), p.Layout.Stages()); err != nil {
		return kernel.StepEstimate{}, err
	}
	if step.Mode == plan.StepIdle || step.Mode == plan.StepPrebuilt {
		return kernel.StepEstimate{Bottleneck: kernel.ResourceHost}, nil
	}
	var busiest units.Tokens
	for i := range step.Ranks {
		n := step.Padded[i].Tokens
		busiest = max(busiest, n)
	}
	d := r.StepBase + time.Duration(busiest)*r.PerToken
	return kernel.StepEstimate{Overlap: d, NoOverlap: d, Bottleneck: kernel.ResourceSM}, nil
}

// TransferDemand implements env.Environment.
func (r *Reference) TransferDemand(p plan.PoolPlan, t plan.Transfer) (kernel.TransferDemand, error) {
	if err := p.CheckTransfer(t); err != nil {
		return kernel.TransferDemand{}, err
	}
	if t.Tokens == 0 {
		// A pull that finds its KV already here: no payload, no link, and no floor, because the
		// reference charges no handshake. The design prices none (section 6.2).
		return kernel.TransferDemand{}, nil
	}
	if r.BytesPerSecond <= 0 {
		return kernel.TransferDemand{}, fmt.Errorf("envtest: no link rate")
	}
	return kernel.TransferDemand{
		Bytes: int64(t.Tokens) * int64(r.BytesPerToken),
		Links: []kernel.LinkUse{{Link: "ref", BytesPerSecond: r.BytesPerSecond}},
	}, nil
}

// Acceptance implements env.Environment.
func (r *Reference) Acceptance(modelName, drafter, workload string) (acceptance.Profile, error) {
	if pr, ok := r.Profiles[[3]string{modelName, drafter, workload}]; ok {
		return pr, nil
	}
	return acceptance.Profile{}, fmt.Errorf("envtest: no acceptance profile for %s / %s / %s", modelName, drafter, workload)
}
