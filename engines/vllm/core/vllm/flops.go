package vllm

import "github.com/inference-sim/blis-substrate/contracts/engine"

// FLOPsPricer is a Pricer that also states a batch's model FLOPs and one GPU's peak, for a
// model-FLOPs-utilization figure (engine.Flopped).
type FLOPsPricer interface {
	StepFLOPs(b engine.Batch) float64
	PeakFLOPsPerGPU() float64
}

// StepFLOPs implements engine.Flopped; zero when the pricer states none.
func (e *Engine) StepFLOPs(b engine.Batch) float64 {
	if p, ok := e.pricer.(FLOPsPricer); ok {
		return p.StepFLOPs(b)
	}
	return 0
}

// PeakFLOPsPerGPU implements engine.Flopped; zero when the pricer states none.
func (e *Engine) PeakFLOPsPerGPU() float64 {
	if p, ok := e.pricer.(FLOPsPricer); ok {
		return p.PeakFLOPsPerGPU()
	}
	return 0
}

// StepFLOPs implements engine.Flopped for the group: the work is additive over requests, so the
// merged batch's count is the ranks' sum.
func (g *DPGroup) StepFLOPs(b engine.Batch) float64 {
	if p, ok := g.pricer.(FLOPsPricer); ok {
		return p.StepFLOPs(b)
	}
	return 0
}

// PeakFLOPsPerGPU implements engine.Flopped for the group.
func (g *DPGroup) PeakFLOPsPerGPU() float64 {
	if p, ok := g.pricer.(FLOPsPricer); ok {
		return p.PeakFLOPsPerGPU()
	}
	return 0
}
