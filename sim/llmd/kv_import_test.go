package llmd

// The blank import registers sim/kv's KV-store constructor (sim.NewKVStoreFromConfig), which
// sim.NewEngine builds every engine's KV store with: these tests build clusters, and nothing
// else in their binary imports sim/kv.
import _ "github.com/inference-sim/blis-vllm/compat/kv"
