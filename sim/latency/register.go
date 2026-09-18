// register.go wires sim/latency constructors into the sim package's registration
// variable (NewLatencyModelFunc). This init() runs when any package imports
// sim/latency, breaking the import cycle between sim/ (interface owner) and
// sim/latency/ (implementation). Production code imports sim/latency directly;
// test code in package sim uses latency_import_test.go for the blank import.
package latency

import "github.com/inference-sim/inference-sim/sim"

func init() {
	// NewLatencyModel is variadic (accepts Options); the seam type is not, so wrap
	// it. The seam is used only by MustNewLatencyModel (sim-package tests), which
	// never wire an adapter accessor — production adapter wiring passes
	// WithAdapterCost through latency.NewLatencyModel directly (sim/cluster).
	// NOTE for future maintainers: this wrapper intentionally discards all Options.
	// Any new Option that must take effect through the seam (e.g. for sim-package
	// tests) has to be threaded onto the seam type explicitly here — it will NOT
	// flow through this wrapper as written.
	sim.NewLatencyModelFunc = func(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (sim.LatencyModel, error) {
		return NewLatencyModel(coeffs, hw)
	}
}
