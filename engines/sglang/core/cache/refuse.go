// Original to blis-sglang.

package cache

import (
	"fmt"
	"sort"
	"strings"
)

// The allocators, KV pools and request pools of SGLang v0.5.23 (297e54fd) and which the port
// models (SGL-6). Every class below subclasses BaseTokenToKVPoolAllocator (allocator/base.py:41),
// KVCache (memory_pool.py:2005) or ReqToTokenPool (memory_pool.py:376), or stands in for one
// (DecodeReqToTokenPool, disaggregation/decode.py:171, borrows ReqToTokenPool's methods; MambaPool,
// memory_pool.py:505, is the request-indexed state pool). A configuration that selects any other
// is refused before the run starts, by the class's name, citing the open issue that tracks it
// (X28). The HiCache host pools are HiCache's (inference-sim/blis-sglang#38).

// RefusedPoolsIssue is the open issue every refused allocator and pool cites.
const RefusedPoolsIssue = "inference-sim/blis-sglang#44"

// PoolKind is which framework interface a class implements.
type PoolKind string

const (
	AllocatorKind PoolKind = "allocator"
	KVPoolKind    PoolKind = "kv_pool"
	ReqPoolKind   PoolKind = "req_to_token"
)

// The modelled classes, each with its Go counterpart.
var modelledPools = map[string]PoolKind{
	"TokenToKVPoolAllocator":      AllocatorKind, // TokenAllocator
	"PagedTokenToKVPoolAllocator": AllocatorKind, // PagedAllocator
	"MHATokenToKVPool":            KVPoolKind,    // MHAPool
	"MLATokenToKVPool":            KVPoolKind,    // MLAPool
	"DSATokenToKVPool":            KVPoolKind,    // DSAPool
	"ReqToTokenPool":              ReqPoolKind,   // RequestPool
}

// The refused classes, each with where SGLang defines it (python/sglang/srt/..., at 297e54fd).
var refusedPools = map[string]struct {
	Kind PoolKind
	Site string
}{
	"SWATokenToKVPoolAllocator":                {AllocatorKind, "mem_cache/allocator/swa.py:41"},
	"PureSWATokenToKVPoolAllocator":            {AllocatorKind, "mem_cache/allocator/swa.py:727"},
	"UnifiedSWAAllocatorBase":                  {AllocatorKind, "mem_cache/allocator/unified_hybrid_swa.py:55"},
	"UnifiedSWATokenToKVPoolAllocator":         {AllocatorKind, "mem_cache/allocator/unified_hybrid_swa.py:808"},
	"UnifiedMambaSWATokenToKVPoolAllocator":    {AllocatorKind, "mem_cache/allocator/unified_hybrid_swa.py:1340"},
	"UnifiedMambaTokenToKVPoolAllocator":       {AllocatorKind, "mem_cache/allocator/unified_mamba.py:42"},
	"HiSparseTokenToKVPoolAllocator":           {AllocatorKind, "mem_cache/allocator/hisparse.py:22"},
	"DeepSeekV4HiSparseTokenToKVPoolAllocator": {AllocatorKind, "mem_cache/allocator/hisparse.py:302"},
	"MultiEndedAllocator":                      {AllocatorKind, "mem_cache/allocator/unified_sub_pool.py:279"},
	"FloatMultiEndedAllocator":                 {AllocatorKind, "mem_cache/allocator/unified_sub_pool.py:2204"},
	"PageInterleavePoolAllocator":              {AllocatorKind, "mem_cache/allocator/page_interleave.py:58"},
	"NPUPagedTokenToKVPoolAllocator":           {AllocatorKind, "hardware_backend/npu/allocator_npu.py:15"},
	"DSV4NPUTokenToKVPoolAllocator":            {AllocatorKind, "hardware_backend/npu/dsv4/dsv4_allocator.py:104"},

	"BaseSWAKVPool":                  {KVPoolKind, "mem_cache/base_swa_memory_pool.py:14"},
	"SWAKVPool":                      {KVPoolKind, "mem_cache/swa_memory_pool.py:21"},
	"DeepSeekV4TokenToKVPool":        {KVPoolKind, "mem_cache/deepseek_v4_memory_pool.py:943"},
	"DeepSeekV4SingleKVPool":         {KVPoolKind, "mem_cache/deepseek_v4_memory_pool.py:237"},
	"DeepSeekV4UniformFP8KVPool":     {KVPoolKind, "mem_cache/deepseek_v4_memory_pool.py:383"},
	"DeepSeekV4IndexerPool":          {KVPoolKind, "mem_cache/deepseek_v4_memory_pool.py:545"},
	"HiSparseC4DevicePool":           {KVPoolKind, "mem_cache/deepseek_v4_memory_pool.py:437"},
	"HiSparseDSATokenToKVPool":       {KVPoolKind, "mem_cache/hisparse_memory_pool.py:37"},
	"HiSparseMHAMainPool":            {KVPoolKind, "mem_cache/hisparse_memory_pool.py:174"},
	"HybridLinearKVPool":             {KVPoolKind, "mem_cache/memory_pool.py:4228"},
	"QSATokenToKVPool":               {KVPoolKind, "mem_cache/qsa_kv_pool.py:48"},
	"LayerSplitDSATokenToKVPool":     {KVPoolKind, "mem_cache/dsa_cache_layer_split.py:194"},
	"MHATokenToKOnlyPool":            {KVPoolKind, "mem_cache/memory_pool.py:5732"},
	"MHATokenToKVPoolFP4":            {KVPoolKind, "mem_cache/memory_pool.py:3621"},
	"MHATokenToKVPoolMXFP8":          {KVPoolKind, "mem_cache/memory_pool.py:3847"},
	"MLATokenToKVPoolFP4":            {KVPoolKind, "mem_cache/memory_pool.py:5133"},
	"NoOpMHATokenToKVPool":           {KVPoolKind, "mem_cache/memory_pool.py:3510"},
	"PageMajorMHATokenToKVPool":      {KVPoolKind, "mem_cache/memory_pool.py:3775"},
	"MiniMaxSparseKVPool":            {KVPoolKind, "mem_cache/memory_pool.py:5847"},
	"MiniMaxM3Nvfp4KVPool":           {KVPoolKind, "mem_cache/minimax_m3_nvfp4_pool.py:603"},
	"MiniMaxM3Nvfp4MainKVPool":       {KVPoolKind, "mem_cache/minimax_m3_nvfp4_pool.py:347"},
	"MiniMaxM3Nvfp4IndexKPool":       {KVPoolKind, "mem_cache/minimax_m3_nvfp4_pool.py:123"},
	"PageInterleaveMHATokenToKVPool": {KVPoolKind, "mem_cache/page_interleave_pool.py:595"},
	"PageInterleaveMLATokenToKVPool": {KVPoolKind, "mem_cache/page_interleave_pool.py:694"},
	"PageInterleaveDSATokenToKVPool": {KVPoolKind, "mem_cache/page_interleave_pool.py:937"},
	"UnifiedDraftKVPool":             {KVPoolKind, "mem_cache/unified_draft_pool.py:37"},
	"UnifiedDraftSWAKVPool":          {KVPoolKind, "mem_cache/unified_draft_pool.py:128"},
	"UnifiedMHATokenToKVPool":        {KVPoolKind, "mem_cache/unified_memory_pool.py:809"},
	"UnifiedMLATokenToKVPool":        {KVPoolKind, "mem_cache/unified_memory_pool.py:952"},
	"UnifiedHybridLinearKVPool":      {KVPoolKind, "mem_cache/unified_memory_pool.py:1523"},
	"UnifiedSWAKVPool":               {KVPoolKind, "mem_cache/unified_memory_pool.py:1851"},
	"NPUMHATokenToKVPool":            {KVPoolKind, "hardware_backend/npu/memory_pool_npu.py:88"},
	"NPUMHATokenToKOnlyPool":         {KVPoolKind, "hardware_backend/npu/memory_pool_npu.py:496"},
	"NPUMiniMaxSparseKVPool":         {KVPoolKind, "hardware_backend/npu/memory_pool_npu.py:603"},
	"NPUMLATokenToKVPool":            {KVPoolKind, "hardware_backend/npu/memory_pool_npu.py:629"},
	"DSV4NPUTokenToKVPool":           {KVPoolKind, "hardware_backend/npu/dsv4/dsv4_memory_pool.py:258"},
	"NPUDeepSeekV4SingleKVPool":      {KVPoolKind, "hardware_backend/npu/dsv4/dsv4_memory_pool.py:40"},
	"NPUDeepSeekV4IndexerPool":       {KVPoolKind, "hardware_backend/npu/dsv4/dsv4_memory_pool.py:177"},
	"_DummyKVCache":                  {KVPoolKind, "hardware_backend/mlx/model_runner_stub.py:43"},

	"HybridReqToTokenPool":            {ReqPoolKind, "mem_cache/memory_pool.py:1355"},
	"HybridMambaDecodeReqToTokenPool": {ReqPoolKind, "disaggregation/decode.py:275"},
	"UnifiedHybridReqToTokenPool":     {ReqPoolKind, "mem_cache/unified_memory_pool.py:1357"},
	"DecodeReqToTokenPool":            {ReqPoolKind, "disaggregation/decode.py:171"},
	"MambaPool":                       {ReqPoolKind, "mem_cache/memory_pool.py:505"},
	"DSV4NPUReqToTokenPool":           {ReqPoolKind, "hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py:122"},
	"MlxAuxiliaryStateReqToTokenPool": {ReqPoolKind, "hardware_backend/mlx/kv_cache/auxiliary_state.py:218"},
}

// PoolRefusal is a start-up refusal of an allocator or pool class.
type PoolRefusal struct {
	Kind  PoolKind
	Class string
	// Site is where SGLang defines the class, or empty for a name that is no class of the release.
	Site  string
	Issue string
}

func (r *PoolRefusal) Error() string {
	if r.Site == "" {
		return fmt.Sprintf("%s %s is not an allocator or pool of SGLang v0.5.23", r.Kind, r.Class)
	}
	return fmt.Sprintf("%s %s (python/sglang/srt/%s@297e54fd) is not modelled; refused until it is (%s)",
		r.Kind, r.Class, r.Site, r.Issue)
}

// CheckPoolClass is nil when SGLang's class of the given kind is one the port models, and its
// refusal otherwise.
func CheckPoolClass(kind PoolKind, class string) error {
	if k, ok := modelledPools[class]; ok && k == kind {
		return nil
	}
	if r, ok := refusedPools[class]; ok && r.Kind == kind {
		return &PoolRefusal{Kind: kind, Class: class, Site: r.Site, Issue: RefusedPoolsIssue}
	}
	return &PoolRefusal{Kind: kind, Class: class}
}

// CheckPoolSelection checks the three classes a configuration selects, before the run starts,
// and refuses every one the port does not model, each by name.
func CheckPoolSelection(allocator, kvPool, reqPool string) error {
	var msgs []string
	for _, c := range []struct {
		kind  PoolKind
		class string
	}{{AllocatorKind, allocator}, {KVPoolKind, kvPool}, {ReqPoolKind, reqPool}} {
		if err := CheckPoolClass(c.kind, c.class); err != nil {
			msgs = append(msgs, err.Error())
		}
	}
	if len(msgs) == 0 {
		return nil
	}
	return fmt.Errorf("refused at start-up: %s", strings.Join(msgs, "; "))
}

// RefusedPoolClasses is every class refused by name, sorted.
func RefusedPoolClasses() []string {
	out := make([]string, 0, len(refusedPools))
	for c := range refusedPools {
		out = append(out, c)
	}
	sort.Strings(out)
	return out
}
