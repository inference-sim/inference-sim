package engine_test

// Blank import triggers sim/kv's init(), which registers the KV-store constructors
// (NewKVCacheStateFunc, NewKVStoreFromConfig). An external test file, so package engine's
// internal tests can build KV caches without importing sim/kv, which imports this package.
import _ "github.com/inference-sim/inference-sim/sim/kv"
