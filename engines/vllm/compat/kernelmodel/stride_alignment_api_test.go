// Original to blis-vllm.

package kernelmodel

import (
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/rules"
)

// What the substrate's KV grouping input can state about block-stride alignment today, and so
// what the kernel can size a pool with. vLLM v0.32.0 pads each layer's page to its own alignment
// when the layout is layer-outermost (_get_layer_block_stride, vllm/v1/core/kv_cache_utils.py:
// 1515-1518@5df63a72) and rounds the summed block once to the lcm of the alignments when it is
// block-outermost (:1540-1552). rules.KVCacheGroupsInput carries one integer, the lcm, which is
// the block-outermost rule and nothing else; the kernel passes none (kernel/caches.go
// resolveKVPool, pinned substrate 3053fa8).
//
// No setting a deployment states at the defaults needs more: core/kv's TestPoolBytesPerBlockIsVLLMsAtFP8
// finds rounding each layer and rounding once agree for gpt-oss-120b, minimax-m2.5, minimax-m3,
// kimi-k2.5, kimi-k3's MLA layers and deepseek-v4-pro, at every block size and kernel block size
// the attention backends declare. They differ only for DeepSeek-V4's indexer with
// indexer_sparse_logits on (a field the engine refuses today), whose layout is block-outermost.
//
// This is the record of that gap, written as a test that passes while the API is absent and fails
// the day it arrives. When a per-layer field (for instance a stride alignment on KVCacheLayer)
// exists, the kernel can price a layer-outermost pool with non-identity alignments: then add
// that case to core/kv's oracle sweep through the kernel, and delete this test.
func TestSubstrateStatesNoPerLayerStrideAlignment(t *testing.T) {
	var fields []string
	for _, typ := range []reflect.Type{reflect.TypeOf(rules.KVCacheGroupsInput{}), reflect.TypeOf(rules.KVCacheLayer{})} {
		for i := range typ.NumField() {
			if strings.Contains(strings.ToLower(typ.Field(i).Name), "align") {
				fields = append(fields, typ.Name()+"."+typ.Field(i).Name)
			}
		}
	}
	if !slices.Equal(fields, []string{"KVCacheGroupsInput.BlockStrideAlignment"}) {
		t.Fatalf("the substrate's KV grouping input now states %v, where the pinned one states only the block-outermost lcm: "+
			"the kernel can take a per-layer alignment, so add a layer-outermost non-identity case to the pool sweep and remove this test", fields)
	}
}
