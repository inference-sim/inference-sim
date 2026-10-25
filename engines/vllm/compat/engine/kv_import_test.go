// Moved from inference-sim/sim/engine/kv_import_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package engine_test

// Blank import triggers sim/kv's init(), which registers the KV-store constructors
// (NewKVCacheStateFunc, NewKVStoreFromConfig). An external test file, so package engine's
// internal tests can build KV caches without importing sim/kv, which imports this package.
import _ "github.com/inference-sim/blis-vllm/compat/kv"
