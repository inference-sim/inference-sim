package cluster_test

// Blank import triggers sim/lora's init(), registering NewResidentAdapterSetFunc so
// per-instance Simulators in cluster tests build a real resident-adapter set. sim/lora
// imports sim (not sim/cluster), so there is no import cycle. Mirrors the sim package's
// lora_import_test.go.
import _ "github.com/inference-sim/inference-sim/sim/lora"
