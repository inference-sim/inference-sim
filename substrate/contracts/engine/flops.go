package engine

// Flopped is implemented by an engine that can state the model FLOPs of a step, for a
// model-FLOPs-utilization figure (inference-sim#2081). A simulator that finds it records, for
// each step, StepFLOPs(b) against the step's time and divides by PeakFLOPsPerGPU times the GPUs
// the engine runs on. An engine that does not implement it reports no FLOPs, and the simulator
// states no utilization for it rather than a guess.
//
// StepFLOPs is the work of the whole model for the batch b, before any parallel split: the
// dense and expert GEMMs for every scheduled token, the head over the rows that sample, and the
// attention matmuls over the keys each query reads. It is a pure function of b, like StepTime.
// PeakFLOPsPerGPU is the datasheet dense peak of one GPU at the format the linear layers run
// in, in FLOP/s.
type Flopped interface {
	StepFLOPs(b Batch) float64
	PeakFLOPsPerGPU() float64
}
