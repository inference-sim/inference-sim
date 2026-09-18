package sim_test

// Blank import triggers sim/lora's init(), which registers NewAdapterRegistryFunc
// and NewResidentAdapterSetFunc. This lets package sim's internal test files build
// a real resident-adapter set without importing sim/lora directly (which would
// create an import cycle, since sim/lora imports sim). Mirrors latency_import_test.go.
import _ "github.com/inference-sim/inference-sim/sim/lora"
