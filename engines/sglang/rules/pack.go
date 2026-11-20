// Original to blis-sglang.

// Package rules is SGLang v0.5.23's engine rules: the answers a cost model and a validator ask of
// the engine release (blis-substrate contracts/rules.EngineRules), read from SGLang's code at
// 297e54fd. A question whose answer SGLang v0.5.23 does not make (vLLM's sequence-parallel MoE,
// dual-batch overlap, decode-context parallelism) is answered "no", and the engine refuses the
// settings that would ask it (config.Resolve, sglangengine.refuseUnmodelled), so no rule is a
// guess that a priced step depends on.
package rules

import (
	"fmt"
	"slices"
	"sort"

	"github.com/inference-sim/inference-sim/substrate/contracts/rules"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"

	blissglang "github.com/inference-sim/inference-sim/engines/sglang"
	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

// Kind is the engine kind these rules are for, as a scenario's engine block names it.
const Kind = "sglang"

// Pack is the rules of one SGLang release.
type Pack struct{ Version string }

// V0_5_23 is the rules of the release this build models.
var V0_5_23 = &Pack{Version: blissglang.Upstream.Version}

// Lookup returns the rules for a release, nil for a release this build has none for.
func Lookup(version string) *Pack {
	if version == V0_5_23.Version {
		return V0_5_23
	}
	return nil
}

// Versions lists the releases Lookup answers for.
func Versions() []string { return []string{V0_5_23.Version} }

var _ rules.EngineRules = (*Pack)(nil)

// EngineVersion implements rules.EngineRules.
func (p *Pack) EngineVersion() string { return p.Version }

// Check implements rules.EngineRules. SGLang's cross-document rules are the resolution
// config.Resolve runs on the pool's engine block, which needs the chip and the model and so runs
// when the pool is lowered (lower.Pool); no rule here reads the document set alone.
func (p *Pack) Check(rules.Input) *rules.Problems { return &rules.Problems{} }

// RuleNames implements rules.EngineRules: this release has no document-set rule.
func (p *Pack) RuleNames() []string { return nil }

// CheckEngineBlock implements rules.EngineRules: the block decodes as a mapping, and every key in
// it is a ServerArgs field of v0.5.23 by SGLang's own name (X1). A name that is not one is a
// finding that names it.
func (p *Pack) CheckEngineBlock(at string, b deployment.EngineBlock) *rules.Problems {
	out := &rules.Problems{}
	if b.IsZero() {
		return out
	}
	var args map[string]any
	if err := b.Decode(&args); err != nil {
		out.Field(at+".engine", "%v", err)
		return out
	}
	known := map[string]bool{}
	for _, n := range config.FieldNames() {
		known[n] = true
	}
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if !known[k] {
			out.Field(at+".engine."+k, "%q is not a ServerArgs field of SGLang %s; state SGLang's own field name", k, p.Version)
		}
	}
	return out
}

// CheckConnectorOptions implements rules.EngineRules: SGLang v0.5.23's offload is HiCache, which
// is configured by ServerArgs fields and has no connector options, so any are a finding.
func (p *Pack) CheckConnectorOptions(at string, b deployment.EngineBlock) *rules.Problems {
	out := &rules.Problems{}
	if !b.IsZero() {
		out.Field(at+".connector_options", "SGLang %s has no connector options: HiCache is configured by ServerArgs fields", p.Version)
	}
	return out
}

// DraftTokens implements rules.EngineRules: the draft tokens one speculative step verifies are
// speculative_num_draft_tokens (arg_groups/fields/spec.py@297e54fd); zero without a drafter.
func (p *Pack) DraftTokens(b deployment.EngineBlock) int {
	var args map[string]any
	if b.IsZero() || b.Decode(&args) != nil {
		return 0
	}
	if a, ok := args["speculative_algorithm"]; !ok || a == nil {
		return 0
	}
	switch n := args["speculative_num_draft_tokens"].(type) {
	case int:
		return n
	case int64:
		return int(n)
	case uint64:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}

// speculativeMethods are the builtin algorithms of --speculative-algorithm
// (arg_groups/fields/spec.py:33@297e54fd).
var speculativeMethods = []string{"DFLASH", "DSPARK", "EAGLE", "EAGLE3", "NEXTN", "NGRAM", "STANDALONE", "UNO"}

// SpeculativeMethodAccepted implements rules.EngineRules.
func (p *Pack) SpeculativeMethodAccepted(method string) bool {
	return slices.Contains(speculativeMethods, method)
}

// SpeculativeMethodNames implements rules.EngineRules.
func (p *Pack) SpeculativeMethodNames() []string { return slices.Clone(speculativeMethods) }

// AsyncCompatibleSpeculative implements rules.EngineRules: SGLang's overlap schedule is not
// switched off by a draft method in a list the release states, so it states none.
func (p *Pack) AsyncCompatibleSpeculative(string) (compatible, known bool) { return true, false }

// EngramOnHostByDefault implements rules.EngineRules: SGLang v0.5.23 has no Engram tables.
func (p *Pack) EngramOnHostByDefault() bool { return false }

// DCPCommBackendNames implements rules.EngineRules: decode context parallelism is refused
// (blis-sglang#26), so no backend list is stated.
func (p *Pack) DCPCommBackendNames() map[string]bool { return nil }

// MegaMoEBackend implements rules.EngineRules: the backend that runs the mega-MoE path
// (arg_groups/mega_moe_hook.py@297e54fd) is flashinfer_megamoe.
func (p *Pack) MegaMoEBackend(name string) bool { return name == "flashinfer_megamoe" }

// SequenceParallelMoE implements rules.EngineRules: SGLang v0.5.23 has no sequence-parallel MoE
// of vLLM's kind; its attention-DP layouts are refused until blis-sglang#70.
func (p *Pack) SequenceParallelMoE(string, bool, int, int) bool { return false }

// ModelSequenceParallel implements rules.EngineRules: SGLang v0.5.23 does not split the residual
// stream by token across tensor-parallel ranks (the layernorm sequence-parallel hook,
// arg_groups/layernorm_sp_hook.py, is a refused field).
func (p *Pack) ModelSequenceParallel(string, deployment.Parallelism, string) bool { return false }

// customAllReduceWorldSizes is CustomAllreduce._SUPPORTED_WORLD_SIZES
// (distributed/device_communicators/custom_all_reduce.py:41@297e54fd).
var customAllReduceWorldSizes = []int{2, 4, 6, 8}

// CustomAllReduceSupportsWidth implements rules.EngineRules.
func (p *Pack) CustomAllReduceSupportsWidth(tp int) bool {
	return slices.Contains(customAllReduceWorldSizes, tp)
}

// TritonFetchWithholdsSMs implements rules.EngineRules: SGLang's Triton kernels withhold no
// streaming multiprocessors from other kernels.
func (p *Pack) TritonFetchWithholdsSMs(int) int { return 0 }

// DBOEngages implements rules.EngineRules: two-batch overlap (enable_two_batch_overlap) is
// refused until blis-sglang#49 models it, so no step engages it.
func (p *Pack) DBOEngages(bool, int, bool) bool { return false }

// EffectiveAll2AllBackend implements rules.EngineRules: SGLang runs the all-to-all backend that
// was stated (moe_a2a_backend); "none" is none.
func (p *Pack) EffectiveAll2AllBackend(backend string) string { return backend }

// ResolveCacheDType implements rules.EngineRules: the catalog record a KV cache is stored under.
// SGLang's kv_cache_dtype resolution is config.Resolve's; the record is the checkpoint's own for
// the 16-bit words and "fp8" for the fp8 ones (kv_cache_hook.py:40-47@297e54fd).
func (p *Pack) ResolveCacheDType(_ string, _, _ int, requested string) string {
	switch requested {
	case "", "auto", "bfloat16", "float16", "half":
		return ""
	case "fp8_e4m3", "fp8_e5m2", "fp8":
		return "fp8"
	}
	return requested
}

// WindowPagesPerRequest implements rules.EngineRules: sliding-window caches are SWA pools, which
// are refused until blis-sglang#73; zero makes the kernel refuse a graph that has one.
func (p *Pack) WindowPagesPerRequest(model.Cache, int, int, int, int) int { return 0 }

// KVCacheGroups implements rules.EngineRules: SGLang v0.5.23 allocates every layer's KV from the
// same token index (the one TokenToKVPoolAllocator, mem_cache/allocator.py), so all owned caches
// of a graph without window or state caches form one group whose block is page_size tokens. A
// graph with a window or state cache is the hybrid pool (blis-sglang#73, #94) and is refused.
func (p *Pack) KVCacheGroups(in rules.KVCacheGroupsInput) (rules.KVCacheGrouping, error) {
	if len(in.Layers) == 0 {
		return rules.KVCacheGrouping{}, fmt.Errorf("no owned caches to group")
	}
	g := rules.KVCacheGroup{BlockSize: in.EngineBlockSize}
	pages := map[string]int{}
	for _, l := range in.Layers {
		c := l.Cache
		if c.Kind != model.CacheKV && c.Kind != model.CacheIndex {
			return rules.KVCacheGrouping{}, fmt.Errorf("layer %q has a %s cache: SGLang %s models a plain token pool only (hybrid pools: inference-sim/blis-sglang#94, #73)", l.Name, c.Kind, p.Version)
		}
		block := in.EngineBlockSize
		if block < 1 {
			return rules.KVCacheGrouping{}, fmt.Errorf("layer %q: page_size is unset", l.Name)
		}
		page := block / c.EntryTokens() * c.EntryBytes(in.CacheDType)
		if c.PageAlignment > 1 {
			page = (page + c.PageAlignment - 1) / c.PageAlignment * c.PageAlignment
		}
		if page < 1 {
			return rules.KVCacheGrouping{}, fmt.Errorf("layer %q has a page of %d bytes", l.Name, page)
		}
		g.Layers = append(g.Layers, l.Name)
		g.PageBytes += page
		pages[l.Name] = page
	}
	bytes := g.PageBytes
	if a := in.BlockStrideAlignment; a > 1 {
		bytes = (bytes + a - 1) / a * a
	}
	return rules.KVCacheGrouping{Groups: []rules.KVCacheGroup{g}, BytesPerBlock: bytes, PageBytes: pages}, nil
}

// CUDAGraphCaptureSizes implements rules.EngineRules. The question is vLLM's (a token-keyed
// capture list from max_cudagraph_capture_size); SGLang's decode capture sizes come from
// ServerArgs.cuda_graph_config (lower.CaptureBatchSizes), so the rule does not apply.
func (p *Pack) CUDAGraphCaptureSizes(rules.CUDAGraphCaptureInput) (rules.CUDAGraphCapture, bool, error) {
	return rules.CUDAGraphCapture{}, false, nil
}

// ResolveDPLoadBalancing implements rules.EngineRules: the pool's stated choice stands.
func (p *Pack) ResolveDPLoadBalancing(pl deployment.Parallelism) deployment.DPLoadBalancing {
	return pl.DataParallelLoadBalancing
}
