// Original to blis-vllm.

package kernelmodel

import "github.com/inference-sim/blis-substrate/contracts/kernel"

// flopper is a kernel that states the model FLOPs of a step (blis-substrate kernel/flops.go).
type flopper interface {
	StepFLOPs(kernel.Batch) float64
	PeakFLOPsPerGPU() float64
}

// StepFLOPsOfShapes is the model FLOPs of a batch in the kernel's shapes, as the kernel counts
// them for a utilization figure; zero when the pool's kernel does not state them.
func (m *Model) StepFLOPsOfShapes(reqs []ReqShape) float64 {
	f, ok := m.k.(flopper)
	if !ok {
		return 0
	}
	return f.StepFLOPs(kernel.Batch{Reqs: reqs, DecodeThreshold: m.decodeThreshold, SMBudget: m.smBudget})
}

// PeakFLOPsPerGPU is one GPU's datasheet dense peak at the served format, in FLOP/s; zero when
// the kernel does not state it.
func (m *Model) PeakFLOPsPerGPU() float64 {
	if f, ok := m.k.(flopper); ok {
		return f.PeakFLOPsPerGPU()
	}
	return 0
}
