// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"
)

// vLLM v0.33.0's OffloadingConnector takes the chunk size as blocks_per_chunk or as block_size, never
// both, and refuses a blocks_per_chunk of 0 or less
// (vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:121-135@4ba787ec). The key is
// read in kv_offload and in a MultiConnector's offloading child, and sizes the CPU tier the way
// the equal block_size does.
func TestBlocksPerChunkSizesTheCPUTierAsVLLMDoes(t *testing.T) {
	const tier = "kv_offload: {connector: OffloadingConnector, cpu_bytes: 3276800, bytes_per_block: 4096"
	cases := []struct {
		name, block string
		chunks      int
	}{
		{"kv_offload blocks_per_chunk 2", tier + ", blocks_per_chunk: 2}\n", 400},
		{"kv_offload the same as block_size 32", tier + ", block_size: 32}\n", 400},
		{"kv_offload blocks_per_chunk 1", tier + ", blocks_per_chunk: 1}\n", 800},
		{"child blocks_per_chunk 2", childWith(", blocks_per_chunk: 2"), 400},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			block := c.block
			if strings.Contains(block, "MultiConnector") {
				block = strings.Replace(block, "cpu_bytes_to_use: 819200", "cpu_bytes_to_use: 3276800", 1)
			}
			e := mustEngine(t, block)
			if _, free, _ := e.off.conn.CPU(); free != c.chunks {
				t.Errorf("%d CPU chunks, want %d", free, c.chunks)
			}
		})
	}
}

func TestBlocksPerChunkIsRefusedAsVLLMRefusesIt(t *testing.T) {
	const tier = "kv_offload: {connector: OffloadingConnector, cpu_bytes: 3276800, bytes_per_block: 4096"
	for _, c := range []struct{ name, block, want string }{
		{"beside block_size", tier + ", blocks_per_chunk: 2, block_size: 32}\n", "only one of them"},
		{"zero", tier + ", blocks_per_chunk: 0}\n", "must be greater than 0"},
		{"negative", tier + ", blocks_per_chunk: -1}\n", "must be greater than 0"},
		{"child zero", childWith(", blocks_per_chunk: 0"), "must be greater than 0"},
		{"child beside block_size", childWith(", blocks_per_chunk: 2, block_size: 32"), "only one of them"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := build(t, c.block)()
			if err == nil {
				t.Fatal("built, want refused")
			}
			if !strings.Contains(err.Error(), c.want) || !strings.Contains(err.Error(), "offloading/config.py:") {
				t.Errorf("refusal %q does not say %q and cite config.py", err, c.want)
			}
		})
	}
}
