package vllmengine

import "github.com/inference-sim/blis-substrate/contracts/engine"

// StepFLOPs implements core/vllm.FLOPsPricer: the model FLOPs of the batch, from the same
// shapes StepTime priced.
func (k kernelPricer) StepFLOPs(b engine.Batch) float64 { return k.m.StepFLOPsOfShapes(shapes(b)) }

// PeakFLOPsPerGPU implements core/vllm.FLOPsPricer.
func (k kernelPricer) PeakFLOPsPerGPU() float64 { return k.m.PeakFLOPsPerGPU() }
