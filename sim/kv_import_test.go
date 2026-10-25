package sim_test

// The blank import runs blis-vllm compat/kv's init(), which registers NewKVCacheStateFunc, so
// package sim's tests can build KV caches through the engine's registered constructor, as the
// binary does through its build site.
import _ "github.com/inference-sim/blis-vllm/compat/kv"
