package sim_test

// Blank import triggers sim/latency's init(), which registers NewLatencyModelFunc.
// This allows package sim's internal test files to create latency models
// without directly importing sim/latency (which would create an import cycle).
import _ "github.com/inference-sim/inference-sim/sim/latency"
