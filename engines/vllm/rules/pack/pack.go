// Moved from inference-sim/blis-schemas/rules/rules.go at 70246460a379f8b1898fbf6c36a3091364118950 (the Pack type and its version-scoped behaviour; the registry and Apply are gone).

// Package pack is the shape of a vLLM rules pack: one release's accepted values, its
// defaults and its version-scoped checks, as data (Pack), and the behaviour a consumer
// resolves against it. It implements blis-schemas' engine-neutral rules.EngineRules. The
// packs themselves (v0_29, v0_31, v0_32) declare their data with it.
//
// It moved from blis-schemas, which keeps only the engine-neutral contract (GOAL.md phase 2b).
package pack

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// Pack is the rule set for one engine version, plus the version-specific data those
// rules read. Holding the data here rather than in each rule's body is what lets a
// new release be a new Pack with different constants and the same rule bodies.
type Pack struct {
	// Version is the engine release this pack describes.
	Version string

	// All2AllBackends are the accepted MoE dispatch backend names, and
	// SPMoEBackends the subset for which the engine makes the MoE input
	// sequence-parallel.
	All2AllBackends map[string]bool
	SPMoEBackends   map[string]bool

	AllReduceBackends map[string]bool
	CUDAGraphModes    map[string]bool
	CacheDTypes       map[string]bool
	// MambaDTypes are the dtypes a recurrent layer's state may be cached in, the
	// engine's MambaDType literal. It is a separate list from CacheDTypes: float32, the
	// common choice for an SSM state, is a member here and not there.
	MambaDTypes     map[string]bool
	MambaCacheModes map[string]bool
	// MoEBackends are the expert-kernel backends --moe-backend accepts, the engine's
	// MoEBackend literal, and MegaMoEBackends the subset that runs a model's mega-MoE
	// path, which DeepSeek-V4.1's sequence-parallelism rule reads.
	MoEBackends     map[string]bool
	MegaMoEBackends map[string]bool
	// ModelSequenceParallelRules are the architectures whose own code decides sequence
	// parallelism in this release, each with its condition, which ModelSequenceParallel
	// evaluates.
	ModelSequenceParallelRules map[string]SequenceParallelCondition

	// DefaultAll2AllBackend is the backend an unstated all2all_backend selects, and
	// All2AllBackendRenames the accepted names the engine replaces with another before
	// using them (pplx and naive run as allgather_reducescatter in both registered
	// releases, vllm/config/parallel.py).
	DefaultAll2AllBackend string
	All2AllBackendRenames map[string]string

	// CacheDTypeResolutions say, per architecture, which cache dtype runs when the
	// deployment requests one the attention backend resolves for itself, by compute
	// capability and tensor-parallel width. See ResolveCacheDType.
	CacheDTypeResolutions map[string]CacheDTypeResolutions

	// EngramArchitectures are the architectures --engram-config may be given for, the
	// keys of the engine's _NGRAM_LAYER_FIELDS; EngramCPUOffloadDefault is
	// EngramConfig.cpu_offload's default, which decides where an Engram table lives when
	// the deployment does not say. Both are empty in a release without Engram.
	EngramArchitectures     map[string]bool
	EngramCPUOffloadDefault bool
	SchedulerPolicies       map[string]bool
	// SpeculativeMethods are accepted draft methods, and AsyncCompatibleSpec the
	// subset that does not disable async scheduling.
	SpeculativeMethods  map[string]bool
	AsyncCompatibleSpec map[string]bool
	// OffloadSpecs are the registered offloading spec names.
	OffloadSpecs map[string]bool
	// DCPCommBackends are the decode-context-parallel collective sets the engine accepts,
	// from its DCPCommBackend literal. A closed literal rather than a registry, so an
	// unknown name is an error.
	DCPCommBackends map[string]bool

	// Quantizations are the weight-format names the engine serves, from its
	// QuantizationMethods literal. A name outside the set is priced against the wrong
	// weight width, compute peak and GEMM efficiency at once. Connectors are the KV
	// connector names registered with the engine's KVConnectorFactory, naming the
	// offload/PD-transfer implementation; an unknown one leaves the transfer on no
	// resource. EvictionPolicies are the KV-offload cache-policy names registered with the
	// engine's CachePolicyFactory (lru, arc), naming how an offload tier evicts. All three
	// are version facts, verified against the engine source like the sets above (#19), and
	// all three are registries the engine extends out of tree — so an unknown value warns
	// (probable typo) rather than errors (it might be a valid out-of-tree name).
	Quantizations    map[string]bool
	Connectors       map[string]bool
	EvictionPolicies map[string]bool

	// CustomAllReduceWorldSizes are the rank counts the SM-consuming all-reduce
	// kernel supports. A width outside the set falls back whatever was requested.
	CustomAllReduceWorldSizes map[int]bool

	// DefaultDBODecodeThreshold and DefaultDBOPrefillThreshold are the token counts
	// above which dual-batch overlap engages, by batch uniformity.
	DefaultDBODecodeThreshold  int
	DefaultDBOPrefillThreshold int

	// TritonFetchPageBytes is the offload page size below which a CPU-to-GPU fetch
	// takes an SM-consuming kernel rather than a copy engine, and TritonFetchSMs is
	// how many SMs that path withholds.
	TritonFetchPageBytes int
	TritonFetchSMs       int

	// CascadeAttnOptIn records that cascade attention is disabled by default.
	CascadeAttnOptIn bool

	// LongPrefillExemptsLoneRequest is whether long_prefill_token_threshold is waived
	// for a request with no other running or waiting request beside it, which then takes
	// the whole token budget.
	LongPrefillExemptsLoneRequest bool

	// KVCachePacking is how the release groups stated caches into KV cache groups that
	// share one block pool. See KVCacheGroups.
	KVCachePacking KVCachePacking

	// Rules are the checks themselves.
	Rules []rules.Rule
}

// Names returns the pack's rule names, sorted. Tests use it to assert that a rule
// has not been dropped, which is the failure a passing suite would otherwise hide.
func (p *Pack) Names() []string {
	out := make([]string, 0, len(p.Rules))
	for _, r := range p.Rules {
		out = append(out, r.Name)
	}
	sort.Strings(out)
	return out
}

// --- Version-scoped behaviour a cost model resolves against --------------------
//
// The methods below answer questions about what the engine will DO, where the fields
// above state what it will ACCEPT. Both are version-scoped facts about one engine
// release, so they belong together: a cost model that compiled these in would churn
// whenever an engine shipped, and would silently misjudge a scenario pinned to an older
// release.
//
// They read only the pack's own fields, so a pack for a new version gets correct
// behaviour by declaring its constants and nothing else. ModelSequenceParallel is no
// exception: each architecture's condition is a SequenceParallelCondition, which is data.

// SequenceParallelMoE reports whether the engine makes the MoE input sequence-parallel.
//
// When it does, a layer's tensor-parallel all-reduce is replaced by a reduce-scatter and
// all-gather pair rather than joined by one, so the two are alternatives and a model that
// charges both over-prices the layer. All four conditions must hold: the backend must be
// one of the set that supports it, expert parallelism must be on, and both widths must
// exceed one.
//
// An empty backend is the engine's default, DefaultAll2AllBackend, which is a member in
// both registered releases, so an unstated backend is not a reason to answer false.
//
// A name the engine renames is looked up under the name it runs as.
func (p *Pack) SequenceParallelMoE(backend string, expertParallel bool, tp, dp int) bool {
	if !expertParallel || tp <= 1 || dp <= 1 {
		return false
	}
	return p.SPMoEBackends[p.EffectiveAll2AllBackend(backend)]
}

// EffectiveAll2AllBackend is the backend the engine runs for a stated one: the default
// for an unstated backend, and the replacement for a renamed one.
func (p *Pack) EffectiveAll2AllBackend(backend string) string {
	if backend == "" {
		return p.DefaultAll2AllBackend
	}
	if to, ok := p.All2AllBackendRenames[backend]; ok {
		return to
	}
	return backend
}

// NormalizeMoEBackend spells a --moe-backend value as the engine does before validating
// it: lower case, dashes as underscores (vllm/config/kernel.py _normalize_moe_backend),
// so deep-gemm-mega-moe is deep_gemm_mega_moe. Both registered releases normalize alike.
func NormalizeMoEBackend(s string) string {
	return strings.ReplaceAll(strings.ToLower(s), "-", "_")
}

// CacheDTypeResolution is one way an attention backend resolves a requested cache dtype:
// on CapabilityFamily, a request in Requested runs as ResolvesTo, provided the tensor-
// parallel width divides TPDivides (0: any width). Entries are tried in order and the
// first that applies wins; a request no entry covers runs as stated.
type CacheDTypeResolution struct {
	CapabilityFamily int
	Requested        []string
	ResolvesTo       string
	TPDivides        int
}

// CacheDTypeResolutions is one architecture's resolutions, in the order they are tried.
type CacheDTypeResolutions []CacheDTypeResolution

// String renders the resolutions for the generated documentation.
func (rs CacheDTypeResolutions) String() string {
	parts := make([]string, 0, len(rs))
	for _, r := range rs {
		when := ""
		if r.TPDivides > 0 {
			when = fmt.Sprintf(" where tp divides %d", r.TPDivides)
		}
		parts = append(parts, fmt.Sprintf("sm%d0: %s -> %s%s", r.CapabilityFamily, strings.Join(r.Requested, "/"), r.ResolvesTo, when))
	}
	return strings.Join(parts, "; ")
}

// ResolveCacheDType returns the cache dtype that runs for a requested one -- unstated
// is auto -- on an architecture, compute-capability family and tensor-parallel width. It
// is what a consumer looks a cache's record up by (Cache.EntryBytes).
//
// DeepSeek-V4.1 is why it exists: on its packed layout the backend rewrites auto and fp8
// to the record its kernel prefers (vllm/models/deepseek_v41/attention.py), which is
// fp8_ds_mla under FlashMLA and nvfp4_ds_mla under SM100's mega attention -- the SM100
// default wherever mega attention can serve the width, which needs tp to divide the
// model's 8 output groups (nvidia/flash_mla_mega_attn.py is_available_for). A deployment
// that selects another attention backend, which the schema does not record, can change
// the answer.
func (p *Pack) ResolveCacheDType(architecture string, family, tp int, requested string) string {
	if requested == "" {
		requested = "auto"
	}
	if tp < 1 {
		tp = 1
	}
	for _, r := range p.CacheDTypeResolutions[architecture] {
		if r.CapabilityFamily != family || (r.TPDivides > 0 && r.TPDivides%tp != 0) {
			continue
		}
		for _, q := range r.Requested {
			if q == requested {
				return r.ResolvesTo
			}
		}
	}
	return requested
}

// WindowPagesPerRequest is how many pages of a window cache vLLM lets one request hold:
// cdiv(min(window - 1 + extraRetained + inFlight, maxModelLen), block) + 1, the same in
// both registered releases (vllm/v1/kv_cache_interface.py
// max_admission_blocks_per_request). The block is the cache's own page size where it
// states one, else engineBlock. inFlight is the request's scheduled-but-unsettled tokens --
// 1 + num_spec_tokens in a decode step -- or, for the engine's startup sizing, its
// max_in_flight_tokens; extraRetained is max(0, num_prefill_lookahead_tokens - 1), zero
// for every drafter but multi-module MTP. A cache that is not a window cache, or a
// non-positive block, gives zero.
func (p *Pack) WindowPagesPerRequest(c model.Cache, engineBlock, inFlight, extraRetained, maxModelLen int) int {
	block := c.BlockSize
	if block < 1 {
		block = engineBlock
	}
	if c.Kind != model.CacheWindow || block < 1 {
		return 0
	}
	tokens := c.Window - 1 + extraRetained + inFlight
	if maxModelLen > 0 && tokens > maxModelLen {
		tokens = maxModelLen
	}
	return (tokens+block-1)/block + 1
}

// ModelSequenceParallel reports whether a model's own sequence parallelism is on, for the
// architecture its graph names, one pool's layout and its MoE backend, by the condition
// this release declares for that architecture in ModelSequenceParallelRules. The
// condition is the CUDA model path's: on ROCm the same architectures load a module whose
// sequence parallelism is always off (see SequenceParallelCondition), so a consumer
// pricing a ROCm pool takes this as false. It decides
// the sequence_parallel emit conditions. An architecture with no rule in this release
// answers false. An unstated width counts as one, and an unstated MoE backend is the
// engine's "auto", which is not a mega-MoE backend.
func (p *Pack) ModelSequenceParallel(architecture string, pl deployment.Parallelism, moeBackend string) bool {
	c, ok := p.ModelSequenceParallelRules[architecture]
	if !ok {
		return false
	}
	atLeastOne := func(w int) int {
		if w < 1 {
			return 1
		}
		return w
	}
	pp, tp, dp := atLeastOne(pl.PP), atLeastOne(pl.TP), atLeastOne(pl.DP)
	if (c.PipelineParallelOne && pp != 1) || (c.ExpertParallel && !pl.EnableExpertParallel) ||
		(c.TensorParallelAboveOne && tp <= 1) {
		return false
	}
	if !c.MegaMoEOrDataParallelAboveOne {
		return true
	}
	return p.MegaMoEBackends[NormalizeMoEBackend(moeBackend)] || dp > 1
}

// SequenceParallelCondition is one architecture's sequence-parallelism condition, as
// data: every requirement set must hold. It is a pack field rather than code so a release
// that changes an architecture's condition is a new pack with different data, like every
// other version fact here.
//
// It describes the CUDA model path. vLLM can load a different model module per platform:
// DeepSeek-V4.1 on ROCm loads vllm/models/deepseek_v41/amd/, whose _use_sequence_parallel
// always returns False ("The ROCm fused-MoE path currently uses ordinary tensor/expert
// parallel"). A consumer pricing a ROCm pool takes such a model's sequence parallelism as
// off.
type SequenceParallelCondition struct {
	PipelineParallelOne    bool // pipeline parallelism is exactly one
	ExpertParallel         bool // expert parallelism is enabled
	TensorParallelAboveOne bool // tensor parallelism exceeds one
	// MegaMoEOrDataParallelAboveOne requires a MoE backend in MegaMoEBackends or data
	// parallelism above one.
	MegaMoEOrDataParallelAboveOne bool
}

// String renders the condition for the generated documentation.
func (c SequenceParallelCondition) String() string {
	var parts []string
	if c.PipelineParallelOne {
		parts = append(parts, "pp 1")
	}
	if c.ExpertParallel {
		parts = append(parts, "expert parallelism on")
	}
	if c.TensorParallelAboveOne {
		parts = append(parts, "tp above 1")
	}
	if c.MegaMoEOrDataParallelAboveOne {
		parts = append(parts, "a mega-MoE backend or dp above 1")
	}
	if len(parts) == 0 {
		return "always"
	}
	return strings.Join(parts, ", ")
}

// CustomAllReduceSupportsWidth reports whether the SM-consuming reduction kernel handles
// a given rank count.
//
// The width matters because the answer moves the cost between resources rather than
// scaling it: the custom kernel spends SMs, and NCCL spends link bandwidth. A width the
// kernel does not implement falls back to NCCL, so charging SMs for it would put the cost
// on a resource that is not being consumed.
func (p *Pack) CustomAllReduceSupportsWidth(tp int) bool {
	return p.CustomAllReduceWorldSizes[tp]
}

// TritonFetchWithholdsSMs reports how many SMs a concurrent offload fetch takes from the
// forward pass, given the page size it moves.
//
// Zero when the page size is one the engine moves with a copy engine instead, which costs
// no SMs at all. That distinction is the whole reason this takes the page size: the same
// offload configuration is free on one page size and costs 9% of an H200's SMs on another.
func (p *Pack) TritonFetchWithholdsSMs(pageBytes int) int {
	if pageBytes != p.TritonFetchPageBytes {
		return 0
	}
	return p.TritonFetchSMs
}

// DBOEngages reports whether dual-batch overlap splits a batch.
//
// The thresholds differ by region: a uniform-decode batch is measured against the decode
// threshold and anything else against the prefill one. A model that used one threshold for
// both would engage overlap on batches the engine leaves whole.
func (p *Pack) DBOEngages(enabled bool, totalTokens int, uniformDecode bool) bool {
	if !enabled {
		return false
	}
	threshold := p.DefaultDBOPrefillThreshold
	if uniformDecode {
		threshold = p.DefaultDBODecodeThreshold
	}
	return totalTokens >= threshold
}
