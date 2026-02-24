// register.go wires sim/latency constructors into the sim package's registration
// variable (NewLatencyModelFunc). This init() runs when any package imports
// sim/latency, breaking the import cycle between sim/ (interface owner) and
// sim/latency/ (implementation). Production code imports sim/latency directly;
// test code in package sim uses latency_import_test.go for the blank import.
package latency

import "github.com/inference-sim/inference-sim/sim"

func init() {
	sim.NewLatencyModelFunc = NewLatencyModel
}
