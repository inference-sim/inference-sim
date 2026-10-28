package harness

// Estimator names the latency model a simulated arm scores with. blis-latency-kernel is the
// only one: the comparison arms against BLIS's former roofline and trained-physics backends
// were removed with those backends.
type Estimator string

// EstimatorKernel is blis-latency-kernel.
const EstimatorKernel Estimator = "kernel"
