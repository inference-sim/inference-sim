package latencykernel

import "github.com/inference-sim/blis-substrate/contracts/rules"

// graphCaptureRequest is what the engine's rules resolve CUDA graph capture sizes from: the
// cudagraph mode the kernel resolved, the stated maximum or sizes, and the scheduler's token
// budget. The kernel fills it from its own engine settings, edited as the engine resolves them,
// so the rules see the mode and the budget the engine runs with.
type graphCaptureRequest = rules.CUDAGraphCaptureInput
