package cluster

import (
	"reflect"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// reportedKVTokens is the KV cache size the engine itself reports, as vLLM's engine core logs it
// at start-up ("GPU KV cache size: N tokens"): the vLLM engine's Engine.KVCacheCapacity (and
// DPGroup.KVCacheCapacity), whose Tokens is the pool's blocks over the blocks one max_model_len
// request holds across every KV cache group, times max_model_len (vllm/v1/core/kv_cache_utils.py
// update_kv_cache_capacity). It is not blocks x block size, which for a hybrid model with several
// KV cache groups is many times larger (inference-sim#1969).
//
// Under data parallelism the figure is one rank's: every engine core of a DP group logs its own
// pool, and the vLLM engine's DPGroup.KVCacheCapacity returns rank 0's. A DP8 replica therefore reports
// the tokens one rank holds, not eight ranks' sum, and a request lives on one rank; a TP replica
// shards each request's KV over its GPUs and so reports a pool that many times larger.
//
// The method is not part of the engine contract, so it is looked up on the engine value: a method
// KVCacheCapacity() returning a struct with an integer field Tokens. The result type lives in
// the vLLM engine (core/kv.Capacity), and a lookup by shape reads it without this package importing it.
// An engine without the method, or one that reports no tokens, gives (0, false) and the caller
// keeps the pool's blocks x block size.
func reportedKVTokens(e engine.Engine) (int64, bool) {
	if e == nil {
		return 0, false
	}
	m := reflect.ValueOf(e).MethodByName("KVCacheCapacity")
	if !m.IsValid() || m.Type().NumIn() != 0 || m.Type().NumOut() != 1 {
		return 0, false
	}
	out := m.Call(nil)[0]
	for out.Kind() == reflect.Pointer && !out.IsNil() {
		out = out.Elem()
	}
	if out.Kind() != reflect.Struct {
		return 0, false
	}
	f := out.FieldByName("Tokens")
	if !f.IsValid() || !f.CanInt() || f.Int() <= 0 {
		return 0, false
	}
	return f.Int(), true
}
