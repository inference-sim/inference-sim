// Original to blis-vllm.

package vllm

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
)

// BlockSizeIssue is the open issue an unresolved block_size cites (design X28).
const BlockSizeIssue = "https://github.com/inference-sim/blis-vllm/issues/68"

// MaxBlockSizeTP is the widest tensor-parallel width the block-size oracle ran (its TP_WIDTHS are
// every width from 1 to 16). The per-rank head counts it feeds vLLM's selection shrink with the width, and
// a backend may accept a head count at one width and not at another, so a wider pool is refused.
const MaxBlockSizeTP = 16

// blockSizeTable is the output of the block-size oracle (oracle/blocksize/oracle.py). For each
// model, GPU compute capability and KV dtype it holds the block size vLLM v0.32.0 resolves an
// unstated block_size to. A combination is resolved only if that size is the same under every
// setting of the facts a CPU cannot discover: FlashAttention 3 and 4 in the wheel, FlashInfer's
// TRTLLM kernels, and FlashMLA in the wheel. Otherwise it is refused, naming the fact.
//
// The oracle runs each combination at every tensor-parallel width from 1 to 16 (TP_WIDTHS), and a
// combination resolves only if the size is the same at all of them. The table is therefore
// valid for a tensor-parallel width up to MaxBlockSizeTP, and ResolveBlockSize refuses a wider one.
// CI regenerates the table from vLLM's own code and fails if the committed file differs.
//
//go:embed blocksize_table.json
var blockSizeTable []byte

type blockSizeCombination struct {
	Model        string         `json:"model"`
	GPU          string         `json:"gpu"`
	Capability   string         `json:"capability"`
	KVCacheDType string         `json:"kv_cache_dtype"`
	Status       string         `json:"status"`
	BlockSize    int            `json:"block_size"`
	Backends     []string       `json:"backends"`
	ByFacts      map[string]int `json:"by_facts"`
	Note         string         `json:"note"`
}

type blockSizeFile struct {
	VLLM         string            `json:"vllm"`
	Facts        map[string]string `json:"facts"`
	Replacements []struct {
		What string `json:"what"`
		Why  string `json:"why"`
	} `json:"replacements"`
	NotResolved  map[string]string      `json:"not_resolved"`
	Combinations []blockSizeCombination `json:"combinations"`
}

var blockSizes = mustLoadBlockSizes()

func mustLoadBlockSizes() blockSizeFile {
	var f blockSizeFile
	dec := json.NewDecoder(bytes.NewReader(blockSizeTable))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		panic(fmt.Sprintf("vllm: blocksize_table.json: %v", err))
	}
	return f
}

// BlockSizeInput is what vLLM v0.32.0's block-size resolution depends on that BLIS knows.
type BlockSizeInput struct {
	// Model is the catalog model name, and Capability the chip's ComputeCapability ("9.0").
	Model, Capability string
	// KVCacheDType is the KV dtype that runs, "auto" when unstated.
	KVCacheDType string
	// Speculative is whether the pool speculates: a drafter adds attention layers whose
	// backends join the selection (platforms/interface.py:699-701, _find_non_ssm_backends).
	Speculative bool
	// ContextParallel is whether the pool shards a sequence (dcp or pcp above 1), which the
	// selector's validity filter reads (v1/attention/selector.py:149-155, use_pcp and use_dcp).
	ContextParallel bool
	// TP is the pool's tensor-parallel width; zero reads as one.
	TP int
	// Env are the pool's stated environment variables.
	Env map[string]string
	// UsesTRTLLMAttention is attention_config.use_trtllm_attention when stated.
	UsesTRTLLMAttention bool
	// MLAPrefillBackend is attention_config.mla_prefill_backend when stated.
	MLAPrefillBackend string
}

// BlockSize is a resolved block size and where it came from.
type BlockSize struct {
	Size   int
	Source string
}

// selectorEnv reports an environment variable the attention selector or a backend's validity
// check reads: VLLM_BATCH_INVARIANT (selector.py:140), VLLM_MLA_DISABLE (envs.py:1499),
// VLLM_KV_CACHE_LAYOUT (envs.py:1888), and the FlashAttention, FlashInfer, MLA and TRTLLM
// switches. The oracle ran with none set.
//
// The list is by name pattern and is not exhaustive. Four variables it does not match are read
// near attention, and none decides a block size: VLLM_USE_NCCL_SYMM_MEM (picks the all-reduce
// method, mla/indexer.py:259), VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (a logits budget,
// mla/indexer.py:1505), VLLM_TRITON_USE_TD (tensor-descriptor loads inside the Triton kernel,
// triton_attn.py:543) and VLLM_USE_OINK_OPS (the rms_norm op, platforms/cuda.py:724). None of them
// enters a backend's validity check or its supported sizes.
func selectorEnv(name string) bool {
	if !strings.HasPrefix(name, "VLLM_") {
		return false
	}
	for _, k := range []string{"ATTN", "ATTENTION", "FLASH", "MLA", "TRTLLM", "KV_CACHE", "BATCH_INVARIANT"} {
		if strings.Contains(name, k) {
			return true
		}
	}
	return false
}

// ResolveBlockSize returns the block size vLLM v0.32.0 resolves an unstated block_size to. It
// does so for the models, GPUs and KV dtypes the block-size oracle (oracle/blocksize) resolves, at
// a tensor-parallel width up to MaxBlockSizeTP. Otherwise it returns an error naming why, and
// https://github.com/inference-sim/blis-vllm/issues/68.
//
// vLLM picks the size in Platform.update_block_size_for_backend
// (vllm/platforms/interface.py:757-781@5df63a72). When the user did not state block_size, it takes
// the smallest size every attention backend accepts. It starts from CacheConfig.DEFAULT_BLOCK_SIZE
// 16 (vllm/config/cache.py:71) and each backend's get_preferred_block_size
// (v1/attention/backend.py:152-160).
//
// The backends are the ones the platform selects per layer (platforms/cuda.py:375-415, priority
// list :83-180). The answer therefore depends on the model's layers, the GPU's compute capability
// and the KV dtype. The oracle runs vLLM's own selection on a CPU for each combination under every
// setting of the facts it cannot discover. A combination resolves only if the size is the same
// under all of them.
func ResolveBlockSize(in BlockSizeInput) (BlockSize, error) {
	refuse := func(why string) (BlockSize, error) {
		return BlockSize{}, fmt.Errorf("block_size is not stated, and %s (%s)", why, BlockSizeIssue)
	}
	if why, ok := blockSizes.NotResolved[in.Model]; ok {
		return refuse(fmt.Sprintf("vLLM v0.32.0 resolves it from the attention backends of the model's layers, which are not resolved for %s: %s", in.Model, why))
	}
	switch {
	case in.TP > MaxBlockSizeTP:
		return refuse(fmt.Sprintf("tensor_parallel_size %d is wider than the %d the oracle ran, and the per-rank head counts it feeds vLLM's selection change with the width", in.TP, MaxBlockSizeTP))
	case in.Speculative:
		return refuse("a speculative drafter adds attention layers whose backends join the selection, which the oracle does not run")
	case in.ContextParallel:
		return refuse("decode or prefill context parallelism changes which backends are valid (v1/attention/selector.py:149-155), which the oracle does not run")
	case in.UsesTRTLLMAttention:
		return refuse("attention_config.use_trtllm_attention is stated, which the oracle does not run")
	case in.MLAPrefillBackend != "":
		return refuse("attention_config.mla_prefill_backend is stated, which the oracle does not run")
	}
	for k := range in.Env {
		if selectorEnv(k) {
			return refuse(fmt.Sprintf("%s is stated, which the attention selector reads and the oracle does not run", k))
		}
	}
	if in.Capability == "" {
		return refuse("the chip states no ComputeCapability")
	}
	for _, c := range blockSizes.Combinations {
		if c.Model != in.Model || c.Capability != in.Capability || c.KVCacheDType != in.KVCacheDType {
			continue
		}
		if c.Status != "resolved" {
			return refuse(fmt.Sprintf("%s on compute capability %s with KV dtype %s is not the same under every setting of what a CPU cannot discover: %s",
				c.Model, c.Capability, c.KVCacheDType, c.Note))
		}
		return BlockSize{Size: c.BlockSize, Source: fmt.Sprintf(
			"vLLM v0.32.0 resolves an unstated block_size from the attention backend it selects (%s) for %s on compute capability %s with KV dtype %s: "+
				"Platform.update_block_size_for_backend, vllm/platforms/interface.py:757-781@5df63a72, backend selection platforms/cuda.py:83-180, 375-415; "+
				"the oracle's table, oracle/blocksize",
			strings.Join(c.Backends, ", "), c.Model, c.Capability, c.KVCacheDType)}, nil
	}
	return refuse(fmt.Sprintf("%s on compute capability %s with KV dtype %q is not a combination the oracle resolves", in.Model, in.Capability, in.KVCacheDType))
}
