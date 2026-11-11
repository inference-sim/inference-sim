package vllm

import (
	"strings"
	"testing"
)

// The committed table is the block-size oracle's output (oracle/blocksize/oracle.py): a
// combination is resolved exactly when vLLM's size is the same under every setting of the facts a
// CPU cannot discover that agrees with what the oracle assumes for the capability
// (blockSizes.Assumptions), and that size is the one the resolver returns. A resolved combination
// carries the assumed facts exactly when its size is not the same under all 16 settings.
func TestBlockSizeTableIsInternallyConsistent(t *testing.T) {
	if blockSizes.VLLM != "5df63a7203cdb9328d824a27b682e77c7b9928be" {
		t.Errorf("the table is for vLLM %s, not the pinned commit", blockSizes.VLLM)
	}
	if len(blockSizes.Facts) < 3 {
		t.Errorf("the table names %d facts, want the ones the oracle toggles", len(blockSizes.Facts))
	}
	for cap, facts := range blockSizes.Assumptions {
		for f, a := range facts {
			if _, ok := blockSizes.Facts[f]; !ok || a.Evidence == "" {
				t.Errorf("capability %s assumes %q (%+v), which is not a fact of the table or has no evidence", cap, f, a)
			}
		}
	}
	resolved, assumed := 0, 0
	for _, c := range blockSizes.Combinations {
		// sameOver is whether every setting that passes keep has one nonzero size.
		sameOver := func(keep func(setting string) bool) bool {
			first, same := 0, true
			for setting, s := range c.ByFacts {
				if !keep(setting) {
					continue
				}
				if first == 0 {
					first = s
				}
				if s != first || s == 0 {
					same = false
				}
			}
			return same && first != 0
		}
		agrees := func(setting string) bool {
			for f, a := range blockSizes.Assumptions[c.Capability] {
				want := "0"
				if a.Value {
					want = "1"
				}
				if !strings.Contains(","+setting+",", ","+f+"="+want+",") {
					return false
				}
			}
			return true
		}
		same, sameEverywhere := sameOver(agrees), sameOver(func(string) bool { return true })
		switch {
		case c.Status == "resolved" && (!same || c.BlockSize == 0 || len(c.Backends) == 0):
			t.Errorf("%s %s %s resolved to %d but the sizes under the facts are %v", c.Model, c.GPU, c.KVCacheDType, c.BlockSize, c.ByFacts)
		case c.Status == "refused" && same:
			t.Errorf("%s %s %s is refused although the size is the same under every setting that agrees with the assumptions: %v", c.Model, c.GPU, c.KVCacheDType, c.ByFacts)
		case c.Status == "resolved" && sameEverywhere && len(c.AssumedFacts) != 0:
			t.Errorf("%s %s %s is the same under all 16 settings but states assumed facts %v", c.Model, c.GPU, c.KVCacheDType, c.AssumedFacts)
		case c.Status == "resolved" && !sameEverywhere && len(c.AssumedFacts) == 0:
			t.Errorf("%s %s %s rests on an assumption but states none: %v", c.Model, c.GPU, c.KVCacheDType, c.ByFacts)
		case c.Status == "refused" && len(c.AssumedFacts) != 0:
			t.Errorf("%s %s %s is refused but states assumed facts %v", c.Model, c.GPU, c.KVCacheDType, c.AssumedFacts)
		case c.Status == "resolved":
			resolved++
			if len(c.AssumedFacts) != 0 {
				assumed++
			}
		}
	}
	if resolved == 0 || assumed == 0 {
		t.Errorf("%d combinations resolve, %d of them by assumption", resolved, assumed)
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
		{"MLA on Hopper with fp8 KV still depends on FlashMLA", BlockSizeInput{Model: "deepseek-v3", Capability: "9.0", KVCacheDType: "fp8"}, 0, "flashmla_built"},
		{"Kimi-K2.5 on Hopper with fp8 KV still depends on FlashMLA", BlockSizeInput{Model: "kimi-k2.5", Capability: "9.0", KVCacheDType: "fp8"}, 0, "flashmla_built"},
		{"MLA on Hopper with auto KV is FlashAttention MLA's 16, by the FA3 assumption", BlockSizeInput{Model: "deepseek-v3", Capability: "9.0", KVCacheDType: "auto"}, 16, "fa3_available"},
		{"Kimi-K2.5 on Hopper with auto KV, by the FA3 assumption", BlockSizeInput{Model: "kimi-k2.5", Capability: "9.0", KVCacheDType: "auto"}, 16, "fa3_available"},
		{"Kimi-K2.5 on Hopper with bfloat16 KV", BlockSizeInput{Model: "kimi-k2.5", Capability: "9.0", KVCacheDType: "bfloat16"}, 16, "fa3_available"},
		{"Kimi-K2.5 on Blackwell needs no assumption", BlockSizeInput{Model: "kimi-k2.5", Capability: "10.0", KVCacheDType: "auto"}, 32, ""},
		{"a model the oracle does not resolve", BlockSizeInput{Model: "minimax-m3", Capability: "10.0", KVCacheDType: "fp8"}, 0, "MiniMaxM3SparseBackend"},
		{"a hybrid model", BlockSizeInput{Model: "qwen3.5-397b-a17b", Capability: "10.0", KVCacheDType: "fp8"}, 0, "hybrid"},
		{"a model the oracle has never seen", with(func(i *BlockSizeInput) { i.Model = "no-such-model" }), 0, "not a combination the oracle resolves"},
		{"a KV dtype the oracle has not run", with(func(i *BlockSizeInput) { i.KVCacheDType = "nvfp4" }), 0, "not a combination the oracle resolves"},
		{"a chip that states no compute capability", with(func(i *BlockSizeInput) { i.Capability = "" }), 0, "ComputeCapability"},
		{"tensor parallelism wider than the oracle ran", with(func(i *BlockSizeInput) { i.TP = 32 }), 0, "tensor_parallel_size 32"},
		{"the widest tensor parallelism the oracle ran", with(func(i *BlockSizeInput) { i.TP = MaxBlockSizeTP }), 16, ""},
		{"a drafter adds layers", with(func(i *BlockSizeInput) { i.Speculative = true }), 0, "drafter"},
		{"context parallelism", with(func(i *BlockSizeInput) { i.ContextParallel = true }), 0, "context parallelism"},
		{"use_trtllm_attention stated", with(func(i *BlockSizeInput) { i.UsesTRTLLMAttention = true }), 0, "use_trtllm_attention"},
		{"mla_prefill_backend stated", with(func(i *BlockSizeInput) { i.MLAPrefillBackend = "FLASHINFER" }), 0, "mla_prefill_backend"},
		{"an attention environment variable", with(func(i *BlockSizeInput) { i.Env = map[string]string{"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"} }), 0, "VLLM_ATTENTION_BACKEND"},
		{"an unrelated environment variable is ignored", with(func(i *BlockSizeInput) { i.Env = map[string]string{"VLLM_LOGGING_LEVEL": "DEBUG"} }), 16, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := ResolveBlockSize(c.in)
			if c.why == "" || c.size > 0 {
				if err != nil || got.Size != c.size || !strings.Contains(got.Source, "vllm/platforms/interface.py:757-781@5df63a72") {
					t.Fatalf("got %+v, %v; want size %d with a cited source", got, err, c.size)
				}
				// A size that rests on an assumed fact says so, with its evidence; one that does not, does not.
				if assumed := strings.Contains(got.Source, "assuming "); assumed != (c.why != "") {
					t.Errorf("source %q: assumes a fact: %t, want %t", got.Source, assumed, c.why != "")
				}
				if c.why != "" && !strings.Contains(got.Source, "assuming "+c.why+": ") {
					t.Errorf("source %q does not say it assumes %s", got.Source, c.why)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), c.why) || !strings.Contains(err.Error(), BlockSizeIssue) {
				t.Errorf("err %v, want a refusal containing %q and %s", err, c.why, BlockSizeIssue)
			}
		})
	}
}

// fp8_e4m3 is vLLM's name for fp8, and bfloat16 and float16 run as auto does where the model is
// served in bfloat16 (CacheDType, vllm/config/cache.py:39-59). The oracle runs every one of these
// names through vLLM's own selection, so the table, not an alias in Go, says they resolve alike:
// each name must give the same answer, resolved or refused, as the name it stands for.
func TestKVDTypeNamesResolveAsTheOraclesTwoNamesDo(t *testing.T) {
	alias := map[string]string{"fp8_e4m3": "fp8", "bfloat16": "auto", "float16": "auto"}
	seen := 0
	for _, c := range blockSizes.Combinations {
		base, ok := alias[c.KVCacheDType]
		if !ok {
			continue
		}
		seen++
		in := BlockSizeInput{Model: c.Model, Capability: c.Capability, KVCacheDType: c.KVCacheDType}
		want := BlockSizeInput{Model: c.Model, Capability: c.Capability, KVCacheDType: base}
		got, gerr := ResolveBlockSize(in)
		exp, eerr := ResolveBlockSize(want)
		if (gerr == nil) != (eerr == nil) || (gerr == nil && got.Size != exp.Size) {
			t.Errorf("%s %s: %s gives %+v, %v but %s gives %+v, %v", c.Model, c.Capability, c.KVCacheDType, got, gerr, base, exp, eerr)
		}
	}
	if seen == 0 {
		t.Error("the table holds none of fp8_e4m3, bfloat16 or float16")
	}
	// The case the review reproduced: llama-3.1-70b-instruct on compute capability 9.0.
	for _, kv := range []string{"fp8", "fp8_e4m3", "bfloat16", "float16", "auto"} {
		got, err := ResolveBlockSize(BlockSizeInput{Model: "llama-3.1-70b-instruct", Capability: "9.0", KVCacheDType: kv})
		if err != nil || got.Size != 16 {
			t.Errorf("llama-3.1-70b-instruct, 9.0, %s: %+v, %v; want 16", kv, got, err)
		}
	}
}
