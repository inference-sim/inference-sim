package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// OffloadCache is the N-tier KV-offload chain (H1, #1590): the GPU tier
// (KVCacheState) plus a ref_cnt-managed CPU staging tier and ordered secondary
// "fs" tiers, with a bounded transfer station (#1588) driving promotion/cascade
// service. It implements sim.KVStore (still 12 entities) and the optional
// SnapshotCachedBlocksFn used by cluster routing.
//
// It is driven through the synchronous KVStore seam: SetClock advances the
// station and applies completions; AllocateKVBlocks consults the chain and kicks
// off async promotions; MirrorToCPU stores freshly-computed blocks and cascades
// them. There are NO sim.Event objects — the station is the queueing model, and
// timing flows through token counts (recompute under lock-saturation raises
// prefill tokens via the existing StepTime model). Request-level deferral and
// per-request transfer latency (the dominant TTFT effect) are H3 (#1591).
type OffloadCache struct {
	gpu       *KVCacheState
	cpu       *offloadCPUTier
	secondary []*secondaryTier
	station   *kvtransfer.TransferStation // nil when there are no secondary tiers (CPU-only offload)

	perBlockBytes     int64 // resolved per-rank KV bytes of one GPU block (transfer-job sizing)
	blockSizeTokens   int64
	offloadPromptOnly bool
	clock             int64

	// Metrics / diagnostics (all load-independent where it matters, #1586).
	offloadMissCount int64 // neither CPU nor secondary served AND GPU alloc failed
	promotionsFired  int64 // secondary→CPU promotions initiated (Read jobs submitted)
	promotionsFailed int64 // secondary hits refused by the BC-C5 evictable gate → recompute
}

// NewOffloadCache builds the tier chain from a resolved, enabled KVOffloadConfig.
// It panics on a config the mechanism cannot honor — the derived PerBlockBytes
// (required, > 0), the H1 restrictions (BlocksPerChunk == 1, eviction_policy
// "lru"), and a CPU budget too small for one block. Panic (not logrus.Fatalf) is
// the library convention here (R6); the CLI validates first so these are
// defense-in-depth invariants.
func NewOffloadCache(gpu *KVCacheState, cfg sim.KVOffloadConfig) *OffloadCache {
	if gpu == nil {
		panic("NewOffloadCache: gpu must not be nil")
	}
	if !cfg.IsEnabled() {
		panic("NewOffloadCache: called with an inert (disabled) offload config")
	}
	if cfg.PerBlockBytes <= 0 {
		panic(fmt.Sprintf("NewOffloadCache: PerBlockBytes must be > 0 (derived from the model KV size), got %d", cfg.PerBlockBytes))
	}
	if cfg.BlocksPerChunk != 1 {
		panic(fmt.Sprintf("NewOffloadCache: H1 supports blocks_per_chunk == 1 only (block-granular offload); blocks_per_chunk > 1 (chunk coalescing) is a follow-up, got %d", cfg.BlocksPerChunk))
	}
	if cfg.EvictionPolicy != "lru" {
		panic(fmt.Sprintf("NewOffloadCache: H1 supports eviction_policy \"lru\" only; %q (e.g. arc) is a follow-up", cfg.EvictionPolicy))
	}
	capacity := cfg.CPUBytesToUse / cfg.PerBlockBytes
	if capacity < 1 {
		panic(fmt.Sprintf("NewOffloadCache: cpu_bytes_to_use (%d) is smaller than one block (%d bytes); no CPU offload block fits", cfg.CPUBytesToUse, cfg.PerBlockBytes))
	}

	oc := &OffloadCache{
		gpu:               gpu,
		cpu:               newOffloadCPUTier(capacity),
		perBlockBytes:     cfg.PerBlockBytes,
		blockSizeTokens:   gpu.BlockSizeTokens,
		offloadPromptOnly: cfg.OffloadPromptOnly,
	}

	if len(cfg.Tiers) > 0 {
		stationCfg := kvtransfer.Config{Tiers: make([]kvtransfer.TierConfig, len(cfg.Tiers))}
		for i, t := range cfg.Tiers {
			oc.secondary = append(oc.secondary, newSecondaryTier())
			stationCfg.Tiers[i] = kvtransfer.TierConfig{
				NRead:             int(t.NReadThreads),
				NWrite:            int(t.NWriteThreads),
				ReadBaseTicks:     int64(math.Round(t.BaseLatency)),
				WriteBaseTicks:    int64(math.Round(t.BaseLatency)),
				ReadBytesPerTick:  t.ReadBandwidth,
				WriteBytesPerTick: t.WriteBandwidth,
				MaxQueueDepth:     0, // unbounded (vLLM deque default); keeps prepareStore→Submit leak-safe
			}
		}
		station, err := kvtransfer.New(stationCfg)
		if err != nil {
			panic(fmt.Sprintf("NewOffloadCache: transfer station config invalid: %v", err))
		}
		oc.station = station
	}
	return oc
}

// --- Trivial sim.KVStore methods delegating to the GPU tier ---

func (o *OffloadCache) GetCachedBlocks(tokens []sim.TokenID) []int64 { return o.gpu.GetCachedBlocks(tokens) }
func (o *OffloadCache) ReleaseKVBlocks(req *sim.Request)             { o.gpu.ReleaseKVBlocks(req) }
func (o *OffloadCache) BlockSize() int64                            { return o.gpu.BlockSize() }
func (o *OffloadCache) UsedBlocks() int64                           { return o.gpu.UsedBlocks() }
func (o *OffloadCache) TotalCapacity() int64                        { return o.gpu.TotalCapacity() }

// SnapshotCachedBlocksFn delegates to the GPU tier so cluster routing keeps its
// frozen-snapshot semantics (INV-7). Offload tiers are intentionally invisible to
// the router in H1 (a routing-fidelity boundary; a later hole may expose them).
func (o *OffloadCache) SnapshotCachedBlocksFn() func([]sim.TokenID) int {
	return o.gpu.SnapshotCachedBlocksFn()
}

// PendingTransferLatency / ConsumePendingTransferLatency return 0: H1 charges no
// explicit per-request transfer latency for the offload path. Timing impact flows
// through token counts (recompute under lock-saturation → more prefill tokens →
// longer step via the existing StepTime model); the dominant step-boundary
// deferral latency is H3 (#1591).
func (o *OffloadCache) PendingTransferLatency() int64        { return 0 }
func (o *OffloadCache) ConsumePendingTransferLatency() int64 { return 0 }

// CacheHitRate mirrors the legacy tiered semantics (#1586, load-independent):
// hits are GPU cache hits (CPU reloads land there on a subsequent request);
// misses add offloadMissCount, incremented only when neither tier could serve AND
// the GPU allocation failed.
func (o *OffloadCache) CacheHitRate() float64 {
	hits := o.gpu.CacheHits
	total := hits + o.gpu.CacheMisses + o.offloadMissCount
	if total == 0 {
		return 0
	}
	return float64(hits) / float64(total)
}

// KVThrashingRate reports the fraction of promotion attempts refused by the
// evictable gate (BC-C5) — the recompute-pressure signal. Zero when no promotion
// was attempted (R11).
func (o *OffloadCache) KVThrashingRate() float64 {
	attempts := o.promotionsFired + o.promotionsFailed
	if attempts == 0 {
		return 0
	}
	return float64(o.promotionsFailed) / float64(attempts)
}

// SetClock advances the offload clock and applies transfer completions (Task 7
// fleshes out the completion dispatch). It is the ONLY point completions are
// applied, and it runs at the top of each step before batch formation.
func (o *OffloadCache) SetClock(clock int64) {
	o.clock = clock
	// Task 7: poll the station and apply completions (Read→completeStore,
	// Write→secondary.store+unpin).
}

// AllocateKVBlocks consults the chain then allocates on GPU (Task 5 adds the CPU
// reload + secondary promotion). For now it delegates to the GPU tier so the
// skeleton is a working single-tier equivalent.
func (o *OffloadCache) AllocateKVBlocks(req *sim.Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	ok := o.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	if !ok {
		o.offloadMissCount++
	}
	return ok
}

// MirrorToCPU stores newly-completed full blocks into the CPU tier and cascades
// them to the secondary tiers (Task 6 implements it). No-op for now.
func (o *OffloadCache) MirrorToCPU(batch []*sim.Request) {
	// Task 6: store new full blocks (OffloadPromptOnly gate) + write-through cascade.
}
