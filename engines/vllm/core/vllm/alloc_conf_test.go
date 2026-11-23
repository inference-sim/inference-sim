// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"
)

// vLLM refuses PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True beside any KV connector unless the CuMem
// allocator is enabled (VllmConfig._verify_kv_transfer_compat, vllm/config/vllm.py:1337-1389@0fda86a3), and
// the engine refuses it too. Every other combination runs: the variable alone, another value beside a
// connector, and a connector with no variable.
func TestExpandableSegmentsBesideAKVConnectorIsRefusedAsVLLMRefusesIt(t *testing.T) {
	const expandable = "env: {PYTORCH_CUDA_ALLOC_CONF: \"expandable_segments:True\"}\n"
	for _, c := range []struct {
		what, extra string
		refused     bool
	}{
		{"variable alone", expandable, false},
		{"another value beside a transfer connector",
			"env: {PYTORCH_CUDA_ALLOC_CONF: \"max_split_size_mb:512\"}\n" + both, false},
		{"expandable_segments:False beside a transfer connector",
			"env: {PYTORCH_CUDA_ALLOC_CONF: \"expandable_segments:False\"}\n" + both, false},
		{"transfer connector, no variable", both, false},
		{"transfer connector", expandable + both, true},
		{"offload tier", expandable + "kv_offload: {connector: OffloadingConnector, cpu_bytes: 3276800, bytes_per_block: 4096, workers: 4}\n", true},
		{"stated beside other settings of the variable",
			"env: {PYTORCH_CUDA_ALLOC_CONF: \"garbage_collection_threshold:0.8,expandable_segments:True\"}\n" + both, true},
	} {
		t.Run(c.what, func(t *testing.T) {
			_, err := build(t, c.extra)()
			switch {
			case c.refused && (err == nil || !strings.Contains(err.Error(), "expandable_segments:True")):
				t.Errorf("err %v, want vLLM's refusal naming expandable_segments:True", err)
			case !c.refused && err != nil:
				t.Errorf("refused: %v", err)
			}
		})
	}
}
