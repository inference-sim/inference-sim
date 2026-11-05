package vllm

import (
	"strings"
	"testing"
)

// The committed table is the block-size oracle's output (oracle/blocksize/oracle.py): a
// combination is resolved exactly when vLLM's size is the same under every setting of the facts
// a CPU cannot discover, and that size is the one the resolver returns.
func TestBlockSizeTableIsInternallyConsistent(t *testing.T) {
	if blockSizes.VLLM != "5df63a7203cdb9328d824a27b682e77c7b9928be" {
		t.Errorf("the table is for vLLM %s, not the pinned commit", blockSizes.VLLM)
	}
	if len(blockSizes.Facts) < 3 {
		t.Errorf("the table names %d facts, want the ones the oracle toggles", len(blockSizes.Facts))
	}
	resolved := 0
	for _, c := range blockSizes.Combinations {
		same := true
		for _, s := range c.ByFacts {
			if s != c.ByFacts[firstKey(c.ByFacts)] || s == 0 {
				same = false
			}
		}
		switch {
		case c.Status == "resolved" && (!same || c.BlockSize == 0 || len(c.Backends) == 0):
			t.Errorf("%s %s %s resolved to %d but the sizes under the facts are %v", c.Model, c.GPU, c.KVCacheDType, c.BlockSize, c.ByFacts)
		case c.Status == "refused" && same:
			t.Errorf("%s %s %s is refused although the size is the same under every fact: %v", c.Model, c.GPU, c.KVCacheDType, c.ByFacts)
		case c.Status == "resolved":
			resolved++
		}
	}
	if resolved == 0 {
		t.Error("no combination resolves")
	}
}

func firstKey(m map[string]int) string {
	for k := range m {
		return k
	}
	return ""
}

// One case per branch of the resolver: a resolved combination returns the oracle's size with a
// source that cites vLLM; everything else is refused naming issue 68.
func TestResolveBlockSize(t *testing.T) {
	base := BlockSizeInput{Model: "llama-3.1-70b-instruct", Capability: "9.0", KVCacheDType: "auto"}
	with := func(f func(*BlockSizeInput)) BlockSizeInput { in := base; f(&in); return in }
	for _, c := range []struct {
		name string
		in   BlockSizeInput
		size int
		why  string // a fragment of the refusal
	}{
		{"dense on Hopper", base, 16, ""},
		{"dense on Blackwell", with(func(i *BlockSizeInput) { i.Capability = "10.0" }), 16, ""},
		{"gpt-oss, which has sinks", BlockSizeInput{Model: "gpt-oss-120b", Capability: "10.3", KVCacheDType: "fp8"}, 16, ""},
		{"MLA on Blackwell is FlashInfer MLA's 32", BlockSizeInput{Model: "deepseek-v3", Capability: "10.0", KVCacheDType: "auto"}, 32, ""},
		{"MLA on Hopper depends on what the wheel ships", BlockSizeInput{Model: "deepseek-v3", Capability: "9.0", KVCacheDType: "fp8"}, 0, "flashmla_built"},
		{"a model the oracle does not resolve", BlockSizeInput{Model: "minimax-m3", Capability: "10.0", KVCacheDType: "fp8"}, 0, "MiniMaxM3SparseBackend"},
		{"a hybrid model", BlockSizeInput{Model: "qwen3.5-397b-a17b", Capability: "10.0", KVCacheDType: "fp8"}, 0, "hybrid"},
		{"a model the oracle has never seen", with(func(i *BlockSizeInput) { i.Model = "no-such-model" }), 0, "not a combination the oracle resolves"},
		{"a KV dtype the oracle has not run", with(func(i *BlockSizeInput) { i.KVCacheDType = "nvfp4" }), 0, "not a combination the oracle resolves"},
		{"a chip that states no compute capability", with(func(i *BlockSizeInput) { i.Capability = "" }), 0, "ComputeCapability"},
		{"a drafter adds layers", with(func(i *BlockSizeInput) { i.Speculative = true }), 0, "drafter"},
		{"context parallelism", with(func(i *BlockSizeInput) { i.ContextParallel = true }), 0, "context parallelism"},
		{"use_trtllm_attention stated", with(func(i *BlockSizeInput) { i.UsesTRTLLMAttention = true }), 0, "use_trtllm_attention"},
		{"mla_prefill_backend stated", with(func(i *BlockSizeInput) { i.MLAPrefillBackend = "FLASHINFER" }), 0, "mla_prefill_backend"},
		{"an attention environment variable", with(func(i *BlockSizeInput) { i.Env = map[string]string{"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"} }), 0, "VLLM_ATTENTION_BACKEND"},
		{"an unrelated environment variable is ignored", with(func(i *BlockSizeInput) { i.Env = map[string]string{"VLLM_LOGGING_LEVEL": "DEBUG"} }), 16, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := ResolveBlockSize(c.in)
			if c.why == "" {
				if err != nil || got.Size != c.size || !strings.Contains(got.Source, "vllm/platforms/interface.py:757-781@5df63a72") {
					t.Fatalf("got %+v, %v; want size %d with a cited source", got, err, c.size)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), c.why) || !strings.Contains(err.Error(), BlockSizeIssue) {
				t.Errorf("err %v, want a refusal containing %q and %s", err, c.why, BlockSizeIssue)
			}
		})
	}
}
