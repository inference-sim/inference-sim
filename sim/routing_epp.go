package sim

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// This file is the request-scoped data layer of llm-d's EndpointPicker that the
// weighted-scoring pipeline uses for llm-d parity (#1907, #1908, #1912, #1930):
// prefix-match producers, the in-flight load producer, the filters
// (prefix-cache-affinity-filter, concurrency-detector), the data-bound scorers
// (prefix-cache, token-load, the producer-bound no-hit-lru) and the
// prefix-based P/D decider.
//
// Reference implementation (MANDATORY): llm-d-router
//   - main@981acb8a (the default EPPVersionMain), and
//   - v0.10.0 (EPPVersionV0_10_0, the EPP image curvebender deploys:
//     ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.10.0).
// Citations read "main:<path>:<line>" or "v0.10.0:<path>:<line>"; paths are
// relative to pkg/epp/framework/plugins/ unless they start with pkg/. Where the
// two releases differ for a plugin the behavior is selected by
// RoutingPipelineConfig.Version, never silently; docs/guide/routing.md lists
// every delta.

// EPPVersion selects which llm-d-router release's plugin semantics the EPP
// pipeline mirrors.
type EPPVersion string

const (
	// EPPVersionMain is llm-d-router main at 981acb8a (after v0.11.0).
	EPPVersionMain EPPVersion = "main@981acb8a"
	// EPPVersionV0_10_0 is the v0.10.0 release.
	EPPVersionV0_10_0 EPPVersion = "v0.10.0"
)

// ParseEPPVersion accepts "main", "main@981acb8a", "latest" and "v0.10.0".
func ParseEPPVersion(s string) (EPPVersion, error) {
	switch s {
	case "", "main", "latest", string(EPPVersionMain):
		return EPPVersionMain, nil
	case string(EPPVersionV0_10_0), "0.10.0":
		return EPPVersionV0_10_0, nil
	}
	return "", fmt.Errorf("unknown llm-d EPP version %q (valid: %s, %s)", s, EPPVersionMain, EPPVersionV0_10_0)
}

// llm-d plugin type names (registration names: main:cmd/epp/runner/runner.go).
const (
	LLMDApproxPrefixProducer  = "approx-prefix-cache-producer"
	LLMDPrecisePrefixProducer = "precise-prefix-cache-producer"
	LLMDInFlightLoadProducer  = "inflight-load-producer"
	LLMDPrefixAffinityFilter  = "prefix-cache-affinity-filter"
	LLMDConcurrencyDetector   = "concurrency-detector"
)

// Producer and scorer defaults.
const (
	// approximateprefix/types.go: defaultBlockSizeTokens is 64 on main
	// (main:…/types.go:120) and 16 on v0.10.0 (v0.10.0:…/types.go:100); the 64-token
	// floor (main:…/plugin.go:56, v0.10.0:…/plugin.go:52) makes both resolve to 64.
	approxDefaultBlockSizeTokensMain = 64
	approxDefaultBlockSizeTokensV010 = 16
	approxDefaultMaxPrefixBlocks     = 2048   // main:…/types.go:129
	approxDefaultMaxPrefixTokens     = 131072 // main:…/types.go:135
	approxDefaultLRUCapacity         = 31250  // main:…/types.go:152
	approxMinBlockSizeTokens         = 64
	// inflightload/token_estimator.go: v0.10.0 estimates round(input·1.5)
	// (v0.10.0:…/token_estimator.go:39); main estimates by output-length bucket and
	// uses UnknownOutputTokens without an outlen-bucket attribute
	// (main:…/token_estimator.go:46, :90-115). BLIS requests carry no bucket header.
	inflightDefaultOutputRatio    = 1.5
	inflightUnknownOutputEstimate = 1000
	tokenLoadDefaultThreshold     = 4194304   // scorer/tokenload/token_load.go:35
	prefixScorerDefaultScaleToken = 8192      // main:scheduling/scorer/prefix/plugin.go:68
	noHitLRUDefaultSize           = 1024      // main:scheduling/scorer/nohitlru/no_hit_lru.go:39
	preciseDefaultSpeculativeTTL  = 2_000_000 // µs; main:requestcontrol/dataproducer/preciseprefixcache/prerequest.go:43
)

// PrefixProducerConfig configures one llm-d prefix-match producer instance.
// Type is LLMDApproxPrefixProducer or LLMDPrecisePrefixProducer.
type PrefixProducerConfig struct {
	Name string
	Type string

	// approx-prefix-cache-producer parameters (main:…/approximateprefix/types.go:155-185).
	AutoTune               bool
	BlockSizeTokens        int
	MaxPrefixBlocksToMatch int
	MaxPrefixTokensToMatch int
	LRUCapacityPerServer   int
	// MainIndexOrder selects main's indexer: a prompt's blocks are inserted last
	// block first, so the LRU evicts a prompt's tail before its head
	// (main:…/approximateprefix/indexer.go:94-103), and the match is the
	// galloping run-length search (indexer.go:133-160). v0.10.0 inserts block 0
	// first and scans linearly (v0.10.0:…/indexer.go:55-86, plugin.go:316-332).
	MainIndexOrder bool

	// CPUTier marks the producer as the CPU-offload-tier view (BLIS, #1912). When the
	// instances have a CPU offload tier, the producer reports each instance's
	// contiguous CPU-tier-resident prefix instead of its router-side LRU guess;
	// without an offload tier it is an ordinary approximate producer.
	CPUTier bool

	// precise-prefix-cache-producer: per-device-tier weights (gpu, cpu) of the
	// LongestPrefix match and the weight of a tier absent from the map — 1.0 on
	// v0.10.0 (v0.10.0:pkg/kvcache/kvblock_scorer.go fillMaxWeights), 0.0 on main
	// (main:pkg/kvcache/prefix_match.go:44-45).
	TierWeights       map[string]float64
	UnknownTierWeight float64
	// Speculative seeds the routed target with the request's blocks for
	// SpeculativeTTLUs (main:…/preciseprefixcache/prerequest.go:203-294), weight 1.0
	// (prefix_match.go:40-42).
	Speculative      bool
	SpeculativeTTLUs int64
}

// DefaultApproxPrefixProducerConfig returns llm-d's approx producer defaults.
func DefaultApproxPrefixProducerConfig(name string, v EPPVersion) PrefixProducerConfig {
	bs := approxDefaultBlockSizeTokensMain
	if v == EPPVersionV0_10_0 {
		bs = approxDefaultBlockSizeTokensV010
	}
	return PrefixProducerConfig{
		Name:                   name,
		Type:                   LLMDApproxPrefixProducer,
		AutoTune:               true,
		BlockSizeTokens:        bs,
		MaxPrefixBlocksToMatch: approxDefaultMaxPrefixBlocks,
		MaxPrefixTokensToMatch: approxDefaultMaxPrefixTokens,
		LRUCapacityPerServer:   approxDefaultLRUCapacity,
		MainIndexOrder:         v != EPPVersionV0_10_0,
	}
}

// InFlightLoadConfig configures one llm-d inflight-load-producer instance
// (main:requestcontrol/dataproducer/inflightload/producer.go:55-80).
type InFlightLoadConfig struct {
	Name               string
	AddEstimatedOutput bool
	// OutputRatio > 0 estimates output as round(input·ratio) (v0.10.0);
	// FixedOutputEstimate > 0 estimates a fixed count (main, no outlen bucket).
	OutputRatio         float64
	FixedOutputEstimate int64
	MaxEstimatedOutput  int64 // < 0 = no operator cap
	// RoleAware charges a prefill-role endpoint prompt tokens only and, with
	// output estimation, a decode-role endpoint output tokens only
	// (main:…/inflightload/producer.go:508-545). v0.10.0 is role-blind.
	RoleAware bool
	// PrefixMatchInfoProducerName selects the prefix producer whose match discounts
	// cached prompt tokens; empty = LLMDApproxPrefixProducer.
	PrefixMatchInfoProducerName string
}

// DefaultInFlightLoadConfig returns llm-d's defaults for the version.
func DefaultInFlightLoadConfig(name string, v EPPVersion) InFlightLoadConfig {
	c := InFlightLoadConfig{Name: name, MaxEstimatedOutput: -1}
	if v == EPPVersionV0_10_0 {
		c.OutputRatio = inflightDefaultOutputRatio
	} else {
		c.FixedOutputEstimate = inflightUnknownOutputEstimate
		c.RoleAware = true
	}
	return c
}

// PrefixAffinityFilterConfig is llm-d's prefix-cache-affinity-filter Config
// (main:scheduling/filter/prefixcacheaffinity/plugin.go:67-107). Only ttftSource
// prefillThroughput is modeled.
type PrefixAffinityFilterConfig struct {
	AffinityThreshold           float64
	ExplorationProbability      float64
	MaxTTFTPenaltyMs            float64
	PeakPrefillThroughput       float64
	PrefixMatchInfoProducerName string
	InFlightLoadProducerName    string
	// MissingSignalSkipsGate: main skips the TTFT gate when either side lacks an
	// in-flight signal (main:…/plugin.go:240-250); v0.10.0 reads a missing signal
	// as zero load (v0.10.0:…/plugin.go:251-276).
	MissingSignalSkipsGate bool
}

// DefaultPrefixAffinityFilterConfig returns the filter defaults (plugin.go:99-107).
func DefaultPrefixAffinityFilterConfig(v EPPVersion) PrefixAffinityFilterConfig {
	return PrefixAffinityFilterConfig{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 18000, PeakPrefillThroughput: 15928,
		MissingSignalSkipsGate: v != EPPVersionV0_10_0}
}

// Validate mirrors validate() (main:…/prefixcacheaffinity/plugin.go:141-163; main
// also rejects a negative affinityThreshold, v0.10.0 accepts it).
func (c PrefixAffinityFilterConfig) Validate(v EPPVersion) error {
	for _, f := range []struct {
		name string
		v    float64
	}{{"affinityThreshold", c.AffinityThreshold}, {"explorationProbability", c.ExplorationProbability},
		{"maxTTFTPenaltyMs", c.MaxTTFTPenaltyMs}, {"peakPrefillThroughput", c.PeakPrefillThroughput}} {
		if math.IsNaN(f.v) || math.IsInf(f.v, 0) {
			return fmt.Errorf("%s must be finite, got %v", f.name, f.v)
		}
	}
	if c.AffinityThreshold > 1.0 || (v != EPPVersionV0_10_0 && c.AffinityThreshold < 0) {
		return fmt.Errorf("affinityThreshold must be in [0, 1], got %f", c.AffinityThreshold)
	}
	if c.ExplorationProbability < 0 || c.ExplorationProbability > 1.0 {
		return fmt.Errorf("explorationProbability must be in [0, 1], got %f", c.ExplorationProbability)
	}
	if c.MaxTTFTPenaltyMs < 0 {
		return fmt.Errorf("maxTTFTPenaltyMs must be >= 0, got %f", c.MaxTTFTPenaltyMs)
	}
	if c.PeakPrefillThroughput < 0 {
		return fmt.Errorf("peakPrefillThroughput must be >= 0, got %f", c.PeakPrefillThroughput)
	}
	if c.MaxTTFTPenaltyMs > 0 && c.PeakPrefillThroughput == 0 {
		return fmt.Errorf("peakPrefillThroughput must be > 0 when ttftSource is prefillThroughput")
	}
	return nil
}

// FilterConfig is one filter of a scheduling profile, in profile order.
// Type is LLMDPrefixAffinityFilter or LLMDConcurrencyDetector.
type FilterConfig struct {
	Type        string
	Name        string
	Affinity    PrefixAffinityFilterConfig
	Concurrency ConcurrencyDetectorConfig
}

// ScorerParams carries llm-d scorer parameters for the data-layer-bound scorers.
type ScorerParams struct {
	// prefix-cache-scorer (main:scheduling/scorer/prefix/plugin.go Config) and the
	// producer-bound no-hit-lru-scorer.
	PrefixMatchInfoProducerName string
	MatchLengthWeight           float64
	MatchLengthScaleTokens      int
	// token-load-scorer (scheduling/scorer/tokenload/token_load.go:38-44).
	QueueThresholdTokens     int64
	InFlightLoadProducerName string
	// ProducerBound selects llm-d's no-hit-lru-scorer reading a prefix producer
	// (main:scheduling/scorer/nohitlru/no_hit_lru.go) over BLIS's
	// cache-query no-hit-lru; LRUSize is its endpoint LRU size (default 1024).
	ProducerBound bool
	LRUSize       int
}

// RoutingPipelineConfig is llm-d's data layer and filter stage. Filters belong to
// the default (or decode) profile, PrefillFilters to the prefill profile of a
// disaggregated deployment. Nil on DeploymentConfig means today's scorer-only
// pipeline.
type RoutingPipelineConfig struct {
	Version        EPPVersion
	Producers      []PrefixProducerConfig
	InFlightLoads  []InFlightLoadConfig
	Filters        []FilterConfig
	PrefillFilters []FilterConfig
}

// RoutingEnv carries the per-instance facts the data layer reads.
type RoutingEnv struct {
	// EngineBlockSize is the engine's KV block size in tokens (the endpoint's
	// CacheBlockSize metric, read by approx autoTune: main:…/plugin.go:384-400).
	EngineBlockSize int
	// KVBlocks returns an instance's GPU KV block count (CacheNumBlocks, read by
	// approx autoTune: main:…/plugin.go:361-372); <= 0 means unknown.
	KVBlocks func(instanceID string) int64
	// CPUTierCacheFn reports each instance's contiguous CPU-offload-tier prefix in
	// engine blocks; nil when no instance has a CPU offload tier.
	CPUTierCacheFn map[string]func([]TokenID) int
	// Role returns an instance's P/D role ("prefill", "decode", or "" for both).
	Role func(instanceID string) string
	// FilterRNG drives the affinity filter's exploration draw (INV-6).
	FilterRNG *rand.Rand
}

// ResolveRoutingPipeline applies llm-d's default-producer rule: a consumer naming no
// producer reads the producer's default name (its type), and a missing default
// producer is created with default parameters (RegisterAsDefaultProducer,
// main:cmd/epp/runner/runner.go). Explicitly named producers must exist. The
// inflight-load-producer's prefix binding is optional (producer.go Consumes) and
// never auto-creates a prefix producer. Returns a new config; inputs are unchanged.
func ResolveRoutingPipeline(scorers []ScorerConfig, cfg *RoutingPipelineConfig) (*RoutingPipelineConfig, error) {
	out := &RoutingPipelineConfig{Version: EPPVersionMain}
	if cfg != nil {
		out.Version = cfg.Version
		if out.Version == "" {
			out.Version = EPPVersionMain
		}
		out.Producers = append(out.Producers, cfg.Producers...)
		out.InFlightLoads = append(out.InFlightLoads, cfg.InFlightLoads...)
		out.Filters = append(out.Filters, cfg.Filters...)
		out.PrefillFilters = append(out.PrefillFilters, cfg.PrefillFilters...)
	}
	hasProducer := func(name string) bool {
		for _, p := range out.Producers {
			if p.Name == name {
				return true
			}
		}
		return false
	}
	hasInFlight := func(name string) bool {
		for _, l := range out.InFlightLoads {
			if l.Name == name {
				return true
			}
		}
		return false
	}
	needProducer := func(name, consumer string) error {
		if name == "" || name == LLMDApproxPrefixProducer {
			if !hasProducer(LLMDApproxPrefixProducer) {
				out.Producers = append(out.Producers, DefaultApproxPrefixProducerConfig(LLMDApproxPrefixProducer, out.Version))
			}
			return nil
		}
		if !hasProducer(name) {
			return fmt.Errorf("%s: prefixMatchInfoProducerName %q names no configured prefix producer", consumer, name)
		}
		return nil
	}
	needInFlight := func(name, consumer string) error {
		if name == "" || name == LLMDInFlightLoadProducer {
			if !hasInFlight(LLMDInFlightLoadProducer) {
				out.InFlightLoads = append(out.InFlightLoads, DefaultInFlightLoadConfig(LLMDInFlightLoadProducer, out.Version))
			}
			return nil
		}
		if !hasInFlight(name) {
			return fmt.Errorf("%s: inFlightLoadProducerName %q names no configured inflight-load-producer", consumer, name)
		}
		return nil
	}
	for _, sc := range scorers {
		switch {
		case sc.Name == "prefix-cache" || (sc.Name == "no-hit-lru" && sc.Params.ProducerBound):
			if err := needProducer(sc.Params.PrefixMatchInfoProducerName, sc.Name+" scorer"); err != nil {
				return nil, err
			}
		case sc.Name == "token-load":
			if err := needInFlight(sc.Params.InFlightLoadProducerName, "token-load scorer"); err != nil {
				return nil, err
			}
		}
	}
	for _, f := range append(append([]FilterConfig{}, out.Filters...), out.PrefillFilters...) {
		switch f.Type {
		case LLMDPrefixAffinityFilter:
			if err := needProducer(f.Affinity.PrefixMatchInfoProducerName, f.Name); err != nil {
				return nil, err
			}
			// The in-flight dependency is required only when the TTFT gate is on
			// (main:…/prefixcacheaffinity/plugin.go Consumes).
			if f.Affinity.MaxTTFTPenaltyMs > 0 {
				if err := needInFlight(f.Affinity.InFlightLoadProducerName, f.Name); err != nil {
					return nil, err
				}
			}
		case LLMDConcurrencyDetector:
			if err := needInFlight(f.Concurrency.InFlightLoadProducerName, f.Name); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("unknown routing filter type %q", f.Type)
		}
	}
	for _, l := range out.InFlightLoads {
		if l.PrefixMatchInfoProducerName != "" && l.PrefixMatchInfoProducerName != LLMDApproxPrefixProducer && !hasProducer(l.PrefixMatchInfoProducerName) {
			return nil, fmt.Errorf("%s: prefixMatchInfoProducerName %q names no configured prefix producer", l.Name, l.PrefixMatchInfoProducerName)
		}
	}
	return out, nil
}

// InFlightProducerName resolves an empty consumer binding to the default name.
func InFlightProducerName(name string) string {
	if name == "" {
		return LLMDInFlightLoadProducer
	}
	return name
}

func prefixProducerName(name string) string {
	if name == "" {
		return LLMDApproxPrefixProducer
	}
	return name
}

// prefixMatchInfo mirrors attribute/prefix.PrefixCacheMatchInfo
// (datalayer/attribute/prefix/data_types.go): matched (tier-weighted) and total
// prompt blocks in the producer's block unit, and the unweighted contiguous
// cached-block count.
type prefixMatchInfo struct {
	match, total, blockSize, cached int
}

// prefixProducer is one prefix-match producer: produce runs before filtering for
// every candidate (Produce), record runs on the routed target (PreRequest).
type prefixProducer interface {
	produce(req *Request, snapshots []RoutingSnapshot, clock int64) map[string]prefixMatchInfo
	record(req *Request, target string, clock int64)
}

// approxPrefixProducer mirrors approx-prefix-cache-producer: a router-side LRU of
// chained block hashes per server, updated with the routed request's blocks.
// The chain is hash.HashBlock (SHA-256 of the previous hash and block tokens) in
// place of xxhash64 (main:…/prefixhash/hashing.go:107-130); prefix semantics are
// the same. As in llm-d a trailing partial block is hashed and counted.
type approxPrefixProducer struct {
	idx       *PrefixCacheIndex
	maxBlocks int
	mainOrder bool
	reqID     string
	hashes    []string
}

func resolveApproxBlockSize(cfg PrefixProducerConfig, engineBlockSize int) int {
	bs := cfg.BlockSizeTokens
	if cfg.AutoTune && engineBlockSize > 0 {
		bs = engineBlockSize
	}
	if bs < approxMinBlockSizeTokens {
		bs = approxMinBlockSizeTokens
	}
	return bs
}

// resolveMaxBlocks mirrors main:…/approximateprefix/plugin.go:410-419.
func resolveMaxBlocks(cfg PrefixProducerConfig, blockSize int) int {
	if cfg.MaxPrefixTokensToMatch > 0 && blockSize > 0 {
		return cfg.MaxPrefixTokensToMatch / blockSize
	}
	if cfg.MaxPrefixBlocksToMatch == 0 {
		return math.MaxInt
	}
	return cfg.MaxPrefixBlocksToMatch
}

func newApproxPrefixProducer(cfg PrefixProducerConfig, env RoutingEnv) *approxPrefixProducer {
	bs := resolveApproxBlockSize(cfg, env.EngineBlockSize)
	capacityFor := func(id string) int {
		// makeserver (main:…/plugin.go:361-372): CacheNumBlocks under autoTune,
		// else lruCapacityPerServer, else 31250.
		if cfg.AutoTune && env.KVBlocks != nil {
			if n := env.KVBlocks(id); n > 0 {
				return int(n)
			}
		}
		if cfg.LRUCapacityPerServer > 0 {
			return cfg.LRUCapacityPerServer
		}
		return approxDefaultLRUCapacity
	}
	return &approxPrefixProducer{
		idx:       NewPrefixCacheIndexWithCapacity(bs, capacityFor),
		maxBlocks: resolveMaxBlocks(cfg, bs),
		mainOrder: cfg.MainIndexOrder,
	}
}

// approxBlockHashes returns the request's chained block hashes, capped at maxBlocks,
// including a trailing partial block (main:…/prefixhash/hashing.go:107-130).
func approxBlockHashes(tokens []TokenID, blockSize, maxBlocks int) []string {
	out := []string{}
	prev := ""
	for start := 0; start < len(tokens) && len(out) < maxBlocks; start += blockSize {
		end := start + blockSize
		if end > len(tokens) {
			end = len(tokens)
		}
		prev = hash.HashBlock(prev, tokens[start:end])
		out = append(out, prev)
	}
	return out
}

func (p *approxPrefixProducer) hashesFor(req *Request) []string {
	if p.hashes == nil || p.reqID != req.ID {
		p.hashes = approxBlockHashes(req.FullInputTokens(), p.idx.BlockSize(), p.maxBlocks)
		p.reqID = req.ID
	}
	return p.hashes
}

func (p *approxPrefixProducer) produce(req *Request, snapshots []RoutingSnapshot, _ int64) map[string]prefixMatchInfo {
	hashes := p.hashesFor(req)
	out := make(map[string]prefixMatchInfo, len(snapshots))
	for _, snap := range snapshots {
		var m int
		if p.mainOrder {
			m = p.idx.MatchLengthGalloping(hashes, snap.ID)
		} else {
			m = p.idx.MatchLength(hashes, snap.ID)
		}
		out[snap.ID] = prefixMatchInfo{match: m, total: len(hashes), blockSize: p.idx.BlockSize(), cached: m}
	}
	return out
}

func (p *approxPrefixProducer) record(req *Request, target string, _ int64) {
	if p.mainOrder {
		p.idx.RecordBlocksTailFirst(p.hashesFor(req), target)
		return
	}
	p.idx.RecordBlocks(p.hashesFor(req), target)
}

// cpuTierPrefixProducer is the CPU-offload-tier view of an approx producer slot
// (#1912). TotalBlocks follows the approx producer's block unit and cap; the match
// is the instance's contiguous CPU-tier-resident prefix, converted to that unit.
type cpuTierPrefixProducer struct {
	cacheFn         map[string]func([]TokenID) int
	engineBlockSize int
	blockSize       int
	maxBlocks       int
}

func (p *cpuTierPrefixProducer) produce(req *Request, snapshots []RoutingSnapshot, _ int64) map[string]prefixMatchInfo {
	tokens := req.FullInputTokens()
	total := (len(tokens) + p.blockSize - 1) / p.blockSize
	if total > p.maxBlocks {
		total = p.maxBlocks
	}
	out := make(map[string]prefixMatchInfo, len(snapshots))
	for _, snap := range snapshots {
		m := 0
		if fn, ok := p.cacheFn[snap.ID]; ok && fn != nil {
			m = fn(tokens) * p.engineBlockSize / p.blockSize
		}
		if m > total {
			m = total
		}
		out[snap.ID] = prefixMatchInfo{match: m, total: total, blockSize: p.blockSize, cached: m}
	}
	return out
}

func (p *cpuTierPrefixProducer) record(*Request, string, int64) {}

// precisePrefixProducer maps precise-prefix-cache-producer onto BLIS's per-instance
// KV cache view: GPU blocks from cacheQueryFn (frozen for --cache-signal-delay,
// modeling KV-event propagation) and, when the instances have a CPU offload tier,
// CPU-tier blocks from the live CPU-tier view. The match is the longest prefix held
// in any tier, each block weighted by its best tier (gpu, cpu, speculative), and
// truncated to an int (main:…/preciseprefixcache/producer.go:441); TotalBlocks
// counts full blocks only (pkg/kvcache/kvblock/token_processor.go chunkTokens).
// Each tier's set is taken as prefix-closed: GPU and CPU contribute their
// contiguous prefixes.
type precisePrefixProducer struct {
	cfg       PrefixProducerConfig
	cacheFn   cacheQueryFn
	cpuFn     map[string]func([]TokenID) int
	blockSize int
	spec      map[string]*speculativeSet
	reqID     string
	hashes    []string
}

// speculativeSet holds an instance's speculative block entries (TTL-bounded).
type speculativeSet struct {
	expiry map[string]int64
	queue  []speculativeEntry // insertion order; TTL is constant so expiries are monotone
}

type speculativeEntry struct {
	hash   string
	expiry int64
}

func (p *precisePrefixProducer) tierWeight(tier string) float64 {
	if w, ok := p.cfg.TierWeights[tier]; ok {
		return w
	}
	return p.cfg.UnknownTierWeight
}

func (p *precisePrefixProducer) hashesFor(req *Request) []string {
	if p.hashes == nil || p.reqID != req.ID {
		p.hashes = hash.ComputeBlockHashes(p.blockSize, req.FullInputTokens())
		if p.hashes == nil {
			p.hashes = []string{}
		}
		p.reqID = req.ID
	}
	return p.hashes
}

func (p *precisePrefixProducer) produce(req *Request, snapshots []RoutingSnapshot, clock int64) map[string]prefixMatchInfo {
	tokens := req.FullInputTokens()
	total := len(tokens) / p.blockSize
	wGPU, wCPU := p.tierWeight("gpu"), p.tierWeight("cpu")
	out := make(map[string]prefixMatchInfo, len(snapshots))
	for _, snap := range snapshots {
		gpu, cpu := 0, 0
		if fn, ok := p.cacheFn[snap.ID]; ok && fn != nil {
			gpu = fn(tokens)
		}
		if fn, ok := p.cpuFn[snap.ID]; ok && fn != nil {
			cpu = fn(tokens)
		}
		var spec *speculativeSet
		if p.cfg.Speculative {
			if s := p.spec[snap.ID]; s != nil && len(s.expiry) > 0 {
				spec = s
			}
		}
		if spec == nil && cpu == 0 {
			out[snap.ID] = prefixMatchInfo{match: int(float64(gpu) * wGPU), total: total, blockSize: p.blockSize, cached: gpu}
			continue
		}
		var hashes []string
		if spec != nil {
			hashes = p.hashesFor(req)
		}
		score, cached := 0.0, 0
		for j := 0; j < total; j++ {
			w, held := 0.0, false
			if j < gpu {
				w, held = math.Max(w, wGPU), true
			}
			if j < cpu {
				w, held = math.Max(w, wCPU), true
			}
			if spec != nil {
				if exp, ok := spec.expiry[hashes[j]]; ok && exp > clock {
					w, held = math.Max(w, 1.0), true
				}
			}
			if !held {
				break
			}
			score += w
			cached++
		}
		out[snap.ID] = prefixMatchInfo{match: int(score), total: total, blockSize: p.blockSize, cached: cached}
	}
	return out
}

func (p *precisePrefixProducer) record(req *Request, target string, clock int64) {
	if !p.cfg.Speculative {
		return
	}
	s := p.spec[target]
	if s == nil {
		s = &speculativeSet{expiry: make(map[string]int64)}
		p.spec[target] = s
	}
	for len(s.queue) > 0 && s.queue[0].expiry <= clock {
		e := s.queue[0]
		if s.expiry[e.hash] == e.expiry {
			delete(s.expiry, e.hash)
		}
		s.queue = s.queue[1:]
	}
	exp := clock + p.cfg.SpeculativeTTLUs
	for _, h := range p.hashesFor(req) {
		s.expiry[h] = exp
		s.queue = append(s.queue, speculativeEntry{hash: h, expiry: exp})
	}
}

// InFlightLoadTracker mirrors inflight-load-producer's token accounting
// (main:…/inflightload/producer.go:404-506, :569-640). Routing charges the target the
// request's uncached prompt tokens (plus estimated output tokens when
// configured). A request's charges — one per profile target, e.g. decode and
// prefill — form a group released by the client-facing stream: without output
// estimation every token charge is freed at the first token (StartOfStream),
// with it only the prefill charge; the rest is freed at completion (EndOfStream).
// The request counter is RoutingSnapshot.InFlightRequests.
type InFlightLoadTracker struct {
	cfg      InFlightLoadConfig
	tokens   map[string]int64
	groups   map[string]*inflightGroup
	byStream map[string]string // stream request ID → group ID
	alias    map[string]string // prefill sub-request ID → group ID
	pending  map[string][]*inflightGroup
}

type inflightGroup struct {
	id         string
	stream     *Request
	streamInst string
	charges    []*inflightCharge
	firstToken bool
}

type inflightCharge struct {
	instance string
	tokens   int64
	prefill  bool
}

func newInFlightLoadTracker(cfg InFlightLoadConfig) *InFlightLoadTracker {
	return &InFlightLoadTracker{
		cfg:      cfg,
		tokens:   make(map[string]int64),
		groups:   make(map[string]*inflightGroup),
		byStream: make(map[string]string),
		alias:    make(map[string]string),
		pending:  make(map[string][]*inflightGroup),
	}
}

// Name returns the producer instance name.
func (t *InFlightLoadTracker) Name() string { return t.cfg.Name }

// Tokens returns the instance's in-flight token count.
func (t *InFlightLoadTracker) Tokens(instanceID string) int64 { return t.tokens[instanceID] }

// estimate mirrors estimateRequestTokens + uncachedInputTokens
// (main:…/inflightload/producer.go:508-545, :700-736).
func (t *InFlightLoadTracker) estimate(req *Request, info *prefixMatchInfo, role string) int64 {
	input := int64(len(req.FullInputTokens()))
	tokens := input
	if info != nil && info.blockSize > 0 {
		bs := int64(info.blockSize)
		uncachedIndexed := int64(info.total)*bs - int64(info.match)*bs
		if uncachedIndexed < 0 {
			uncachedIndexed = 0
		}
		tail := input - int64(info.total)*bs
		if tail < 0 {
			tail = 0
		}
		tokens = uncachedIndexed + tail
	}
	if t.cfg.RoleAware && role == "prefill" {
		return tokens
	}
	if t.cfg.AddEstimatedOutput {
		out := t.estimateOutput(input, req.MaxOutputLen)
		if t.cfg.RoleAware && role == "decode" {
			return out
		}
		tokens += out
	}
	return tokens
}

// estimateOutput mirrors the output estimators: round(input·ratio) on v0.10.0
// (v0.10.0:…/token_estimator.go:100-112), the unknown-bucket constant on main
// (main:…/token_estimator.go:90-128); both clamp to the client cap (MaxOutputLen,
// vLLM max_tokens, 0 = none) and the operator cap.
func (t *InFlightLoadTracker) estimateOutput(input int64, maxOutputLen int) int64 {
	var est int64
	if t.cfg.FixedOutputEstimate > 0 {
		est = t.cfg.FixedOutputEstimate
	} else {
		if input <= 0 {
			return 0
		}
		est = int64(math.Round(float64(input) * t.cfg.OutputRatio))
	}
	if maxOutputLen > 0 && int64(maxOutputLen) < est {
		est = int64(maxOutputLen)
	}
	if t.cfg.MaxEstimatedOutput >= 0 && t.cfg.MaxEstimatedOutput < est {
		est = t.cfg.MaxEstimatedOutput
	}
	return est
}

// AliasPrefill attributes a prefill sub-request's charge to its parent's group.
func (t *InFlightLoadTracker) AliasPrefill(subReqID, parentID string) { t.alias[subReqID] = parentID }

// SetStream makes req the request whose first token and completion release the
// group (the decode sub-request of a disaggregated parent).
func (t *InFlightLoadTracker) SetStream(groupID string, req *Request) {
	g, ok := t.groups[groupID]
	if !ok {
		return
	}
	delete(t.byStream, g.stream.ID)
	g.stream = req
	t.byStream[req.ID] = groupID
}

func (t *InFlightLoadTracker) charge(req *Request, target string, tokens int64) {
	gid, isPrefill := t.alias[req.ID]
	if !isPrefill {
		gid = req.ID
	}
	g, ok := t.groups[gid]
	if !ok {
		g = &inflightGroup{id: gid, stream: req, streamInst: target}
		t.groups[gid] = g
		t.byStream[req.ID] = gid
		t.pending[target] = append(t.pending[target], g)
	}
	g.charges = append(g.charges, &inflightCharge{instance: target, tokens: tokens, prefill: isPrefill})
	t.tokens[target] += tokens
}

// releaseCharge frees one charge, flooring the counter at zero (decrementClamped,
// main:…/inflightload/producer.go:856-870).
func (t *InFlightLoadTracker) releaseCharge(c *inflightCharge) {
	if c.tokens == 0 {
		return
	}
	t.tokens[c.instance] -= c.tokens
	if t.tokens[c.instance] < 0 {
		t.tokens[c.instance] = 0
	}
	c.tokens = 0
}

// ReleaseGroup frees every charge of the group (EndOfStream, or a failed request).
func (t *InFlightLoadTracker) ReleaseGroup(groupID string) {
	g, ok := t.groups[groupID]
	if !ok {
		return
	}
	for _, c := range g.charges {
		t.releaseCharge(c)
	}
	delete(t.groups, groupID)
	delete(t.byStream, g.stream.ID)
	for sub, gid := range t.alias {
		if gid == groupID {
			delete(t.alias, sub)
		}
	}
}

// OnRequestDone releases the group whose stream request finished. A prefill
// sub-request finishing releases nothing: llm-d frees the prefill charge at the
// client stream's first token.
func (t *InFlightLoadTracker) OnRequestDone(reqID string) {
	if gid, ok := t.byStream[reqID]; ok {
		t.ReleaseGroup(gid)
	}
}

// ObserveInstance applies StartOfStream to every pending group whose stream
// request runs on the instance and has produced its first token.
func (t *InFlightLoadTracker) ObserveInstance(instanceID string) {
	list := t.pending[instanceID]
	if len(list) == 0 {
		return
	}
	kept := list[:0]
	for _, g := range list {
		if cur, live := t.groups[g.id]; !live || cur != g {
			continue
		}
		if !g.stream.TTFTSet {
			kept = append(kept, g)
			continue
		}
		g.firstToken = true
		for _, c := range g.charges {
			if !t.cfg.AddEstimatedOutput || c.prefill {
				t.releaseCharge(c)
			}
		}
	}
	for i := len(kept); i < len(list); i++ {
		list[i] = nil
	}
	t.pending[instanceID] = kept
}

// RoutingPipeline is the shared llm-d data layer of one deployment: every
// scheduling profile (default, or decode and prefill) reads and records into the
// same producers and in-flight trackers, as llm-d's plugin instances are shared
// across profiles.
type RoutingPipeline struct {
	cfg           *RoutingPipelineConfig
	env           RoutingEnv
	producerNames []string
	producers     map[string]prefixProducer
	inflight      []*InFlightLoadTracker
	inflightByNm  map[string]*InFlightLoadTracker
	inflightBind  map[string]string // tracker name → prefix producer name

	cycle map[string]map[string]prefixMatchInfo // producer → instance → info, current request
}

// NewRoutingPipeline builds the data layer for a resolved config
// (ResolveRoutingPipeline). Panics on an invalid config (R3).
func NewRoutingPipeline(cfg *RoutingPipelineConfig, env RoutingEnv, cacheFn map[string]func([]TokenID) int) *RoutingPipeline {
	p := &RoutingPipeline{
		cfg:          cfg,
		env:          env,
		producers:    make(map[string]prefixProducer),
		inflightByNm: make(map[string]*InFlightLoadTracker),
		inflightBind: make(map[string]string),
	}
	for _, pc := range cfg.Producers {
		var prod prefixProducer
		switch pc.Type {
		case LLMDApproxPrefixProducer:
			if pc.CPUTier && env.CPUTierCacheFn != nil {
				bs := resolveApproxBlockSize(pc, env.EngineBlockSize)
				prod = &cpuTierPrefixProducer{cacheFn: env.CPUTierCacheFn, engineBlockSize: env.EngineBlockSize, blockSize: bs, maxBlocks: resolveMaxBlocks(pc, bs)}
			} else {
				prod = newApproxPrefixProducer(pc, env)
			}
		case LLMDPrecisePrefixProducer:
			if pc.BlockSizeTokens <= 0 {
				panic(fmt.Sprintf("precise prefix producer %q: blockSizeTokens must be > 0", pc.Name))
			}
			prod = &precisePrefixProducer{cfg: pc, cacheFn: cacheQueryFn(cacheFn), cpuFn: env.CPUTierCacheFn,
				blockSize: pc.BlockSizeTokens, spec: make(map[string]*speculativeSet)}
		default:
			panic(fmt.Sprintf("unknown prefix producer type %q", pc.Type))
		}
		if _, dup := p.producers[pc.Name]; dup {
			panic(fmt.Sprintf("duplicate prefix producer name %q", pc.Name))
		}
		p.producerNames = append(p.producerNames, pc.Name)
		p.producers[pc.Name] = prod
	}
	for _, lc := range cfg.InFlightLoads {
		t := newInFlightLoadTracker(lc)
		p.inflight = append(p.inflight, t)
		p.inflightByNm[lc.Name] = t
		p.inflightBind[lc.Name] = prefixProducerName(lc.PrefixMatchInfoProducerName)
	}
	return p
}

// InFlightLoadTrackers returns the pipeline's trackers in configuration order.
func (p *RoutingPipeline) InFlightLoadTrackers() []*InFlightLoadTracker { return p.inflight }

// InFlightLoadTracker returns the named tracker ("" = the default name), or nil.
func (p *RoutingPipeline) InFlightLoadTracker(name string) *InFlightLoadTracker {
	return p.inflightByNm[InFlightProducerName(name)]
}

// begin runs every producer for the request over all candidates (Produce runs
// before filtering: main:pkg/epp/requestcontrol/director.go).
func (p *RoutingPipeline) begin(req *Request, snapshots []RoutingSnapshot, clock int64) {
	p.cycle = make(map[string]map[string]prefixMatchInfo, len(p.producerNames))
	if req == nil {
		return
	}
	for _, name := range p.producerNames {
		p.cycle[name] = p.producers[name].produce(req, snapshots, clock)
	}
}

// matchInfo returns the current request's match info from the named producer;
// ok is false when the producer is absent (the attribute is missing).
func (p *RoutingPipeline) matchInfo(producer, instanceID string) (prefixMatchInfo, bool) {
	m, ok := p.cycle[prefixProducerName(producer)]
	if !ok {
		return prefixMatchInfo{}, false
	}
	info, ok := m[instanceID]
	return info, ok
}

func (p *RoutingPipeline) tracker(name string) *InFlightLoadTracker {
	return p.inflightByNm[InFlightProducerName(name)]
}

func (p *RoutingPipeline) role(instanceID string) string {
	if p.env.Role == nil {
		return ""
	}
	return p.env.Role(instanceID)
}

// uncachedTokens is UncachedRequestTokens for the request on an instance under
// the named inflight producer (main:…/inflightload/producer.go:384-402).
func (p *RoutingPipeline) uncachedTokens(t *InFlightLoadTracker, req *Request, instanceID string) int64 {
	if info, ok := p.matchInfo(p.inflightBind[t.cfg.Name], instanceID); ok {
		return t.estimate(req, &info, p.role(instanceID))
	}
	return t.estimate(req, nil, p.role(instanceID))
}

// recordRouted is the PreRequest stage: producers record the target
// (approx: main:…/approximateprefix/plugin.go:284-359; precise speculative:
// prerequest.go:209-294) and inflight producers charge it (producer.go:404-506).
func (p *RoutingPipeline) recordRouted(req *Request, target string, clock int64) {
	if req == nil {
		return
	}
	for _, name := range p.producerNames {
		p.producers[name].record(req, target, clock)
	}
	for _, t := range p.inflight {
		t.charge(req, target, p.uncachedTokens(t, req, target))
	}
}

// buildFilter constructs one profile filter on the pipeline.
func (p *RoutingPipeline) buildFilter(f FilterConfig) filterFunc {
	switch f.Type {
	case LLMDPrefixAffinityFilter:
		return newPrefixAffinityFilter(f.Affinity, p, p.env.FilterRNG)
	case LLMDConcurrencyDetector:
		return newConcurrencyFilter(f.Concurrency, p)
	}
	panic(fmt.Sprintf("unknown routing filter type %q", f.Type))
}

// newPrefixAffinityFilter mirrors prefix-cache-affinity-filter's Filter
// (main:scheduling/filter/prefixcacheaffinity/plugin.go:176-260).
func newPrefixAffinityFilter(cfg PrefixAffinityFilterConfig, p *RoutingPipeline, rng *rand.Rand) filterFunc {
	return func(req *Request, snapshots []RoutingSnapshot) []RoutingSnapshot {
		if len(snapshots) <= 1 || cfg.AffinityThreshold <= 0 {
			return snapshots
		}
		// Exploration draw: llm-d draws from the global math/rand source; BLIS draws
		// from a seeded partition so INV-6 holds.
		draw := 1.0
		if rng != nil {
			draw = rng.Float64()
		}
		if draw < cfg.ExplorationProbability {
			return snapshots
		}
		var sticky, nonSticky []RoutingSnapshot
		for _, snap := range snapshots {
			if prefixRatio(p, cfg.PrefixMatchInfoProducerName, snap.ID) >= cfg.AffinityThreshold {
				sticky = append(sticky, snap)
			} else {
				nonSticky = append(nonSticky, snap)
			}
		}
		if len(sticky) == 0 {
			return snapshots
		}
		if cfg.MaxTTFTPenaltyMs > 0 && len(nonSticky) > 0 {
			t := p.tracker(cfg.InFlightLoadProducerName)
			if t == nil && cfg.MissingSignalSkipsGate {
				return snapshots // main:…/plugin.go:240-250: missing signal skips the gate
			}
			bestTTFT := func(set []RoutingSnapshot) float64 {
				best := math.MaxFloat64
				for _, snap := range set {
					var inflight int64
					if t != nil {
						inflight = t.Tokens(snap.ID)
					}
					// TTFT_ms = inFlightTokens / peakPrefillThroughput · 1000 (plugin.go:307-322).
					if ttft := float64(inflight) / cfg.PeakPrefillThroughput * 1000; ttft < best {
						best = ttft
					}
				}
				return best
			}
			if bestTTFT(sticky)-bestTTFT(nonSticky) > cfg.MaxTTFTPenaltyMs {
				return snapshots
			}
		}
		return sticky
	}
}

// prefixRatio is matchBlocks/totalBlocks, 0 when the attribute is missing or
// totalBlocks is 0 (main:…/prefixcacheaffinity/plugin.go:277-288).
func prefixRatio(p *RoutingPipeline, producer, instanceID string) float64 {
	info, ok := p.matchInfo(producer, instanceID)
	if !ok || info.total <= 0 {
		return 0
	}
	return float64(info.match) / float64(info.total)
}

// newConcurrencyFilter runs ConcurrencyDetector.Filter on the in-flight tokens of
// the detector's inflight producer and, when configured, the request's own
// uncached tokens (main:…/concurrency/detector.go:249-283).
func newConcurrencyFilter(cfg ConcurrencyDetectorConfig, p *RoutingPipeline) filterFunc {
	det := NewConcurrencyDetectorFromConfig(cfg)
	return func(req *Request, snapshots []RoutingSnapshot) []RoutingSnapshot {
		t := p.tracker(cfg.InFlightLoadProducerName)
		var incoming func(id string) int64
		if t != nil && cfg.CountIncomingTokens && req != nil {
			incoming = func(id string) int64 { return p.uncachedTokens(t, req, id) }
		}
		view := make([]RoutingSnapshot, len(snapshots))
		copy(view, snapshots)
		if t != nil {
			for i := range view {
				view[i].InFlightTokens = t.Tokens(view[i].ID)
			}
		}
		kept := det.filterWithIncoming(view, incoming)
		keep := make(map[string]bool, len(kept))
		for _, s := range kept {
			keep[s.ID] = true
		}
		out := make([]RoutingSnapshot, 0, len(kept))
		for _, snap := range snapshots {
			if keep[snap.ID] {
				out = append(out, snap)
			}
		}
		return out
	}
}

// newPrefixCacheScorer mirrors prefix-cache-scorer's Score
// (main:scheduling/scorer/prefix/plugin.go:135-185):
// w·(min(1, matched·blockSize/scale))² + (1−w)·matched/total; 0 when the attribute
// is missing or totalBlocks is 0.
func newPrefixCacheScorer(params ScorerParams, p *RoutingPipeline) scorerFunc {
	scale := params.MatchLengthScaleTokens
	if scale <= 0 {
		scale = prefixScorerDefaultScaleToken
	}
	return func(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		for _, snap := range snapshots {
			info, ok := p.matchInfo(params.PrefixMatchInfoProducerName, snap.ID)
			if !ok || info.total == 0 {
				scores[snap.ID] = 0
				continue
			}
			ratio := float64(info.match) / float64(info.total)
			lengthScore := 0.0
			if params.MatchLengthWeight > 0 && info.blockSize > 0 {
				norm := math.Min(1.0, float64(info.match)*float64(info.blockSize)/float64(scale))
				lengthScore = norm * norm
			}
			scores[snap.ID] = params.MatchLengthWeight*lengthScore + (1.0-params.MatchLengthWeight)*ratio
		}
		return scores
	}
}

// newTokenLoadScorer mirrors token-load-scorer's Score (scorer/tokenload/token_load.go:94-137):
// load = inFlight + uncached incoming tokens; 1 for load <= 0, else
// 1 − min(load, threshold)/threshold.
func newTokenLoadScorer(params ScorerParams, p *RoutingPipeline) scorerFunc {
	threshold := params.QueueThresholdTokens
	if threshold <= 0 {
		threshold = tokenLoadDefaultThreshold // token_load.go:65-67
	}
	return func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		t := p.tracker(params.InFlightLoadProducerName)
		for _, snap := range snapshots {
			var load int64
			if t != nil {
				load = t.Tokens(snap.ID)
				if req != nil {
					load += p.uncachedTokens(t, req, snap.ID)
				}
			}
			if load <= 0 {
				scores[snap.ID] = 1.0
				continue
			}
			l := math.Min(float64(load), float64(threshold))
			scores[snap.ID] = 1.0 - l/float64(threshold)
		}
		return scores
	}
}

// newProducerNoHitLRUScorer mirrors llm-d's no-hit-lru-scorer bound to a prefix
// producer (main:scheduling/scorer/nohitlru/no_hit_lru.go:184-340): a request is
// cold when no candidate has MatchBlocks > 0; cold requests rank never-used
// endpoints first (in candidate order), then used endpoints from least to most
// recently used by their position among all LRU keys; warm requests score 0.5.
// The routed target of a cold request moves to the LRU front.
func newProducerNoHitLRUScorer(params ScorerParams, p *RoutingPipeline) (scorerFunc, func(*Request, string, int64)) {
	size := params.LRUSize
	if size <= 0 {
		size = noHitLRUDefaultSize
	}
	var keys []string // oldest first, as golang-lru Keys()
	coldReq := ""
	scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		cold := true
		for _, snap := range snapshots {
			if info, ok := p.matchInfo(params.PrefixMatchInfoProducerName, snap.ID); ok && info.match > 0 {
				cold = false
				break
			}
		}
		coldReq = ""
		if !cold {
			for _, snap := range snapshots {
				scores[snap.ID] = 0.5
			}
			return scores
		}
		if req != nil {
			coldReq = req.ID
		}
		total := len(snapshots)
		if total == 1 {
			scores[snapshots[0].ID] = 1.0
			return scores
		}
		pos := make(map[string]int, len(keys))
		for i, k := range keys {
			pos[k] = i
		}
		var never []string
		for _, snap := range snapshots {
			if _, used := pos[snap.ID]; !used {
				never = append(never, snap.ID)
			}
		}
		for i, id := range never {
			scores[id] = 1.0 - float64(i)/float64(total-1)
		}
		for _, snap := range snapshots {
			if lp, used := pos[snap.ID]; used {
				s := 1.0 - float64(len(never)+lp)/float64(total-1)
				if s < 0 {
					s = 0
				}
				scores[snap.ID] = s
			}
		}
		return scores
	}
	observer := func(req *Request, target string, _ int64) {
		if req == nil || coldReq == "" || coldReq != req.ID {
			return
		}
		for i, k := range keys {
			if k == target {
				keys = append(keys[:i], keys[i+1:]...)
				break
			}
		}
		keys = append(keys, target)
		if len(keys) > size {
			keys = keys[len(keys)-size:]
		}
	}
	return scorer, observer
}

// LLMDPrefixPDDeciderConfig is prefix-based-pd-decider's config
// (main:scheduling/profilehandler/disagg/prefix_based_pd_decider.go:43-60).
type LLMDPrefixPDDeciderConfig struct {
	NonCachedTokens             int
	PromptTokens                int
	PrefixMatchInfoProducerName string
}

// LLMDPrefixPDDecider mirrors prefix-based-pd-decider's disaggregate decision
// (main:…/prefix_based_pd_decider.go:263-301, identical on v0.10.0:…:105-163): with
// nonCachedTokens 0 disaggregation is off; inputs shorter than promptTokens (when
// set) or nonCachedTokens stay local; otherwise disaggregate when
// input − cachedBlocks·blockSize >= nonCachedTokens, reading the pre-selected
// decode endpoint's match info from the bound producer. A missing attribute
// keeps the request on decode (the error path).
type LLMDPrefixPDDecider struct {
	cfg      LLMDPrefixPDDeciderConfig
	pipeline *RoutingPipeline
}

// NewLLMDPrefixPDDecider builds the decider on the deployment's pipeline.
// Panics on negative thresholds (R3).
func NewLLMDPrefixPDDecider(cfg LLMDPrefixPDDeciderConfig, p *RoutingPipeline) *LLMDPrefixPDDecider {
	if cfg.NonCachedTokens < 0 || cfg.PromptTokens < 0 {
		panic(fmt.Sprintf("NewLLMDPrefixPDDecider: thresholds must be >= 0, got %+v", cfg))
	}
	if p == nil {
		panic("NewLLMDPrefixPDDecider: pipeline must not be nil")
	}
	return &LLMDPrefixPDDecider{cfg: cfg, pipeline: p}
}

// Decide implements DisaggregationDecider.
func (d *LLMDPrefixPDDecider) Decide(req *Request, state *RouterState) DisaggregationDecision {
	if d.cfg.NonCachedTokens == 0 || req == nil || state == nil || state.SelectedInstance == "" {
		return DisaggregationDecision{}
	}
	input := int(req.InputLen())
	if d.cfg.PromptTokens > 0 && input < d.cfg.PromptTokens {
		return DisaggregationDecision{}
	}
	if input < d.cfg.NonCachedTokens {
		return DisaggregationDecision{}
	}
	info, ok := d.pipeline.matchInfo(d.cfg.PrefixMatchInfoProducerName, state.SelectedInstance)
	if !ok {
		return DisaggregationDecision{}
	}
	nonCached := input - info.cached*info.blockSize
	return DisaggregationDecision{Disaggregate: nonCached >= d.cfg.NonCachedTokens}
}
