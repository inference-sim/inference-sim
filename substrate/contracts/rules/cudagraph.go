package rules

import "fmt"

// CUDAGraphCapture is the CUDA graph capture a deployment's stated knobs resolve to.
type CUDAGraphCapture struct {
	// Sizes are the step sizes, in tokens, captured, ascending. A step larger than the
	// last runs eager. Empty means no step is captured.
	Sizes []int
	// Max is the largest captured size, the engine's final max_cudagraph_capture_size.
	Max int
	// Capped is whether max_num_batched_tokens lowered the stated maximum, Truncated
	// whether the generated sizes stop below it, and Dropped the stated sizes the engine
	// discarded as larger than max_num_batched_tokens.
	Capped, Truncated bool
	Dropped           []int
}

// ErrCUDAGraphCaptureInconsistent is returned when a deployment states both a capture
// maximum and capture sizes and the maximum is not the largest size the engine keeps,
// which the engine refuses at startup.
var ErrCUDAGraphCaptureInconsistent = fmt.Errorf("max_cudagraph_capture_size is not the largest of the cudagraph_capture_sizes the engine keeps")

// CUDAGraphCaptureInput is the stated knobs CUDAGraphCaptureSizes resolves the capture from,
// as a deployment's engine block states them: the graph mode, the stated maximum capture size
// (nil when unstated), the stated capture sizes (nil when unstated) and the per-step token
// budget.
type CUDAGraphCaptureInput struct {
	Mode                string
	MaxCaptureSize      *int
	CaptureSizes        []int
	MaxNumBatchedTokens int
}
