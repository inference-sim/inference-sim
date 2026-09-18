package sim_test

// Blank import triggers sim/kv's init(), which registers NewKVCacheStateFunc.
// This allows package sim's internal test files to create KV caches
// without directly importing sim/kv (which would create an import cycle).
import _ "github.com/inference-sim/inference-sim/sim/kv"
