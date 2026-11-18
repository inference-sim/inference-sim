package cluster

import (
	"reflect"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// reportedKVTokens is the KV cache size the engine itself reports, as vLLM's engine core logs it
// at start-up ("GPU KV cache size: N tokens"): blis-vllm's Engine.KVCacheCapacity (and
// DPGroup.KVCacheCapacity), whose Tokens is the pool's blocks over the blocks one max_model_len
// request holds across every KV cache group, times max_model_len (vllm/v1/core/kv_cache_utils.py
// update_kv_cache_capacity). It is not blocks x block size, which for a hybrid model with several
// KV cache groups is many times larger (inference-sim#1969).
//
// The method is not part of the engine contract, so it is looked up on the engine value: a method
// KVCacheCapacity() returning a struct with an integer field Tokens. The result type lives in
// blis-vllm (core/kv.Capacity), and a lookup by shape reads it without this package importing it.
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
