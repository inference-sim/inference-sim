package llmd

// The blank import registers blis-vllm compat/kv's KV-store constructor
// (sim.NewKVStoreFromConfig), which sim.NewEngine builds every engine's KV store with: these
// tests build clusters, and nothing else in their binary links compat/kv.
import _ "github.com/inference-sim/blis-vllm/compat/kv"
