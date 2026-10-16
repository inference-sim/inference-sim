package kv

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
	"github.com/inference-sim/inference-sim/sim/internal/util"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// jitterMinFactor is the floor for a sampled latency-jitter multiplier (#1581
// BC-D5): a transfer can be at most this much FASTER than its mean device service
// time, so a left-tail Gaussian draw can never drive service time to zero or
// negative. It is a numerical guard, not a physical parameter.
const jitterMinFactor = 0.05

// Compile-time capability assertions: OffloadCache is a full KVStore, can defer
// secondary-tier fetches (H3, #1591), and reports reloaded-prefix boundaries (#1699).
var (
	_ sim.KVStore                = (*OffloadCache)(nil)
	_ sim.DeferrableKVStore      = (*OffloadCache)(nil)
	_ sim.ReloadReportingKVStore = (*OffloadCache)(nil)
	_ sim.RequestBlockReporter   = (*OffloadCache)(nil)
	_ sim.RemoteKVReserver       = (*OffloadCache)(nil)
)

// OffloadOption configures an OffloadCache at construction.
type OffloadOption func(*OffloadCache)

// WithOffloadRNG supplies the seeded RNG partition used to draw the optional
// per-transfer latency jitter (#1581 BC-D5). NewKVStore derives it from the run
// seed; direct callers that do not exercise jitter may omit it (jitter then stays
// disabled). It is a construction error for a tier to set latency_jitter_stddev>0
// without an RNG (enforced in NewOffloadCache).
func WithOffloadRNG(rng *rand.Rand) OffloadOption {
	return func(o *OffloadCache) { o.rng = rng }
}

// jobRef records what an in-flight transfer-station job represents so SetClock can
// apply its completion to the right tier action. A Read job (secondary→CPU)
// promotes its keys (completeStore); a Write job (CPU→secondary) lands its keys in
// the tier and releases the CPU pins (secondary.store + unpin). A single Read job
// may cover a run of keys; a cascade Write job covers one.
type jobRef struct {
	keys []kvkey.BlockKey
	tier int
	dir  kvtransfer.Direction
}

// OffloadCache is the N-tier KV-offload chain (H1, #1590): the GPU tier
// (KVCacheState) plus a ref_cnt-managed CPU staging tier and ordered secondary
// "fs" tiers, with a bounded transfer station (#1588) driving promotion/cascade
// service. It implements sim.KVStore (still 12 entities) and the optional
// SnapshotCachedBlocksFn used by cluster routing.
//
// It is driven through the synchronous KVStore seam: SetClock advances the
// station and applies completions; AllocateKVBlocks consults the chain and either
// defers a new admission or kicks off async promotions; MirrorToCPU stores
// freshly-computed blocks and cascades them. There are NO sim.Event objects — the
// station is the queueing model, and timing flows through token counts (recompute
// under lock-saturation raises prefill tokens via the existing StepTime model).
// Request-level step-boundary deferral (the dominant TTFT effect, H3 #1591) is
// realized by leaving a new admission in the WaitQ and re-polling it each step
// (PollDeferred / offload_deferral.go) — the delay is a whole multiple of step
// time, not a per-request transfer latency (ConsumePendingTransferLatency stays 0).
type OffloadCache struct {
	gpu       *KVCacheState
	cpu       *offloadCPUTier
	secondary []*secondaryTier
	station   *kvtransfer.TransferStation // nil when there are no secondary tiers (CPU-only offload)

	inflight map[kvtransfer.JobID]jobRef // in-flight transfer jobs, applied on SetClock→Poll

	// H3 (#1591) step-boundary re-poll state. deferred tracks new prefill admissions
	// set aside waiting for a secondary→CPU fetch (keyed by Request.ID). existenceKnown
	// models vLLM's async-lookup existence cache: a key present ⇒ its secondary
	// existence is resolved (warm, skips the RETRY round); absent ⇒ cold (+1 round).
	deferred       map[string]*deferralState
	prefilling     []sim.InflightPrefill // SetInflightPrefills: running requests still prefilling
	existenceKnown map[kvkey.BlockKey]struct{}

	perBlockBytes     int64 // resolved per-rank KV bytes of one GPU block (transfer-job sizing)
	offloadPromptOnly bool
	// blocksPerChunk is vLLM blocks_per_chunk: the tiers hold, look up, evict and move whole
	// chunks of this many GPU blocks, keyed by their last block's hash (offload_chunks.go).
	blocksPerChunk int64
	clock          int64

	// cpuLoadTime prices one CPU->GPU load (INV-K7); nil keeps a CPU-tier hit a same-step
	// reload. loadBusyUntil is when the last load issued finishes: vLLM chains its loads on
	// one stream (kv_offload/cpu/gpu_worker.py:696-701 at v0.31.0).
	cpuLoadTime   func(bytes int64) int64
	loadBusyUntil int64

	// Device-model latency jitter (#1581 BC-D5). jitterStddev[t] is tier t's relative
	// stddev σ (0 == off). rng is the seeded kv-offload partition; nil disables jitter
	// regardless of σ. The RNG lives here (not in the deterministic transfer station,
	// which stays RNG-free — BC-S4): the station consumes a pre-drawn scalar factor.
	jitterStddev []float64
	rng          *rand.Rand

	// Metrics / diagnostics (all load-independent where it matters, #1586).
	offloadMissCount int64 // neither CPU nor secondary served AND GPU alloc failed
	promotionsFired  int64 // secondary→CPU promotions initiated (Read jobs submitted)
	promotionsFailed int64 // secondary hits refused by the BC-C5 evictable gate → recompute
	reloadCount      int64 // CPU→GPU reloads performed (diagnostic)
	mirrorSkipped    int64 // MirrorToCPU stores skipped because CPU was full and all-pinned
	deferralsStarted int64 // new admissions set aside for a secondary fetch (H3, #1591)
	cpuLoadsStarted  int64 // new admissions set aside for a priced CPU->GPU load (INV-K7)
}

// NewOffloadCache builds the tier chain from a resolved, enabled KVOffloadConfig.
// It panics on a config the mechanism cannot honor — the derived PerBlockBytes
// (required, > 0), a non-positive BlocksPerChunk, eviction_policy other than
// "lru", and a CPU budget too small for one chunk. Panic (not logrus.Fatalf) is
// the library convention here (R6); the CLI validates first so these are
// defense-in-depth invariants.
func NewOffloadCache(gpu *KVCacheState, cfg sim.KVOffloadConfig, opts ...OffloadOption) *OffloadCache {
	if gpu == nil {
		panic("NewOffloadCache: gpu must not be nil")
	}
	if !cfg.IsEnabled() {
		panic("NewOffloadCache: called with an inert (disabled) offload config")
	}
	if cfg.PerBlockBytes <= 0 {
		panic(fmt.Sprintf("NewOffloadCache: PerBlockBytes must be > 0 (derived from the model KV size), got %d", cfg.PerBlockBytes))
	}
	if cfg.BlocksPerChunk <= 0 {
		panic(fmt.Sprintf("NewOffloadCache: blocks_per_chunk must be > 0, got %d", cfg.BlocksPerChunk))
	}
	if cfg.EvictionPolicy != "lru" {
		panic(fmt.Sprintf("NewOffloadCache: H1 supports eviction_policy \"lru\" only; %q (e.g. arc) is a follow-up", cfg.EvictionPolicy))
	}
	capacity := cfg.CPUChunkCapacity()
	if capacity < 1 {
		panic(fmt.Sprintf("NewOffloadCache: cpu_bytes_to_use (%d) is smaller than one offload chunk (%d blocks of %d bytes); no CPU offload chunk fits",
			cfg.CPUBytesToUse, cfg.BlocksPerChunk, cfg.PerBlockBytes))
	}

	oc := &OffloadCache{
		gpu:               gpu,
		cpu:               newOffloadCPUTier(capacity),
		inflight:          make(map[kvtransfer.JobID]jobRef),
		deferred:          make(map[string]*deferralState),
		existenceKnown:    make(map[kvkey.BlockKey]struct{}),
		perBlockBytes:     cfg.PerBlockBytes,
		offloadPromptOnly: cfg.OffloadPromptOnly,
		blocksPerChunk:    cfg.BlocksPerChunk,
		cpuLoadTime:       cfg.CPULoadTime,
	}
	for _, opt := range opts {
		opt(oc)
	}

	if len(cfg.Tiers) > 0 {
		stationCfg := kvtransfer.Config{Tiers: make([]kvtransfer.TierConfig, len(cfg.Tiers))}
		oc.jitterStddev = make([]float64, len(cfg.Tiers))
		for i, t := range cfg.Tiers {
			oc.secondary = append(oc.secondary, newSecondaryTier())
			oc.jitterStddev[i] = t.LatencyJitterStddev
			if t.LatencyJitterStddev > 0 && oc.rng == nil {
				panic(fmt.Sprintf("NewOffloadCache: tier %d sets latency_jitter_stddev>0 but no RNG was supplied (WithOffloadRNG); jitter requires a seeded RNG for determinism (#1581 INV-6)", i))
			}
			stationCfg.Tiers[i] = kvtransfer.TierConfig{
				NRead:                  int(t.NReadThreads),
				NWrite:                 int(t.NWriteThreads),
				ReadBaseTicks:          int64(math.Round(t.BaseLatency)),
				WriteBaseTicks:         int64(math.Round(t.BaseLatency)),
				ReadBytesPerTick:       t.ReadBandwidth,
				WriteBytesPerTick:      t.WriteBandwidth,
				SaturationQueueDepth:   int(t.SaturationQueueDepth),
				SingleTransferFraction: t.SingleTransferFraction,
				MaxQueueDepth:          0, // unbounded (vLLM deque default); keeps prepareStore→Submit leak-safe
			}
			if price := t.ServiceTime; price != nil {
				stationCfg.Tiers[i].ServiceTime = func(dir kvtransfer.Direction, bytes int64, q int) int64 {
					return price(dir == kvtransfer.Write, bytes, q)
				}
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

// drawJitterFactor returns the multiplicative service-time factor for a transfer
// on the given tier (#1581 BC-D5). It returns 0 — the station's "no jitter"
// sentinel, drawing NOTHING from the RNG — when the tier has no jitter configured,
// so a σ=0 run is byte-identical and its RNG stream is untouched (INV-6). Otherwise
// it draws 1+N(0,σ) from the seeded partition and clamps the lower tail to
// jitterMinFactor so service time stays positive.
func (o *OffloadCache) drawJitterFactor(tier int) float64 {
	if tier < 0 || tier >= len(o.jitterStddev) || o.jitterStddev[tier] <= 0 || o.rng == nil {
		return 0
	}
	factor := 1 + o.rng.NormFloat64()*o.jitterStddev[tier]
	if factor < jitterMinFactor {
		factor = jitterMinFactor
	}
	return factor
}

// --- Trivial sim.KVStore methods delegating to the GPU tier ---

func (o *OffloadCache) GetCachedBlocks(tokens []sim.TokenID) []int64 {
	return o.gpu.GetCachedBlocks(tokens)
}

func (o *OffloadCache) ReleaseKVBlocks(req *sim.Request) { o.gpu.ReleaseKVBlocks(req) }

// RequestBlocks implements sim.RequestBlockReporter from the GPU tier.
func (o *OffloadCache) RequestBlocks(reqID string) []int64 { return o.gpu.RequestBlocks(reqID) }

// AllocateRemoteKV implements sim.RemoteKVReserver on the GPU tier alone: a remotely
// prefilled prompt is NIXL's to load, so no offload tier is consulted.
func (o *OffloadCache) AllocateRemoteKV(req *sim.Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	return o.gpu.AllocateRemoteKV(req, startIndex, endIndex, cachedBlocks)
}

// CacheRemoteKV implements sim.RemoteKVReserver from the GPU tier.
func (o *OffloadCache) CacheRemoteKV(reqID string) { o.gpu.CacheRemoteKV(reqID) }

func (o *OffloadCache) BlockSize() int64                 { return o.gpu.BlockSize() }
func (o *OffloadCache) UsedBlocks() int64                { return o.gpu.UsedBlocks() }
func (o *OffloadCache) TotalCapacity() int64             { return o.gpu.TotalCapacity() }

// ReloadablePrefixEnd implements sim.ReloadReportingKVStore (#1699/#1706): a PURE
// pre-allocation query (the analogue of vLLM's get_num_new_matched_tokens) reporting the
// token boundary to which a same-step CPU->GPU reload WOULD extend this request's
// GPU-cached prefix, WITHOUT committing or mutating any tier. Batch formation folds it
// into the computed baseline before the chunk/budget clamp, so the whole reloaded prefix
// is credited even under chunked prefill.
//
// It walks the uncached tail past startIndex exactly like consultAndReload but only
// COUNTS contiguous CPU-resident (reloadable) blocks — skipping already-GPU-resident
// blocks, stopping at the first cpuHitPending/cpuMiss/end. A secondary-tier-only run is
// NOT counted (it defers via H3). Gated to genuinely-new prefill admissions
// (`!running && !IsDecodeSubRequest`), matching the class batch formation credits;
// returns (startIndex, false) for a running continuation, a PD decode sub-request, or
// when no reload would extend the prefix.
func (o *OffloadCache) ReloadablePrefixEnd(req *sim.Request, startIndex int64) (int64, bool) {
	if _, running := o.gpu.RequestMap[req.ID]; running || req.IsDecodeSubRequest {
		return startIndex, false
	}
	bs := o.gpu.BlockSize()
	tokens := req.FullInputTokens()
	startBlock := startIndex / bs
	// Seed the hash chain just past the GPU-cached prefix so only the uncached tail is
	// keyed (hot-path parity with consultAndReload). startIndex is block-aligned (a GPU
	// prefix boundary), so startBlock*bs == startIndex.
	prevHash := ""
	if startBlock > 0 {
		if cached := o.gpu.GetCachedBlocks(tokens); int64(len(cached)) >= startBlock {
			prevHash = o.gpu.Blocks[cached[startBlock-1]].Hash
		}
	}
	tail, ok := o.tailOf(tokens, startBlock, prevHash)
	if !ok {
		return startIndex, false
	}
	// Whole chunks, each GPU-resident or CPU-readable, up to the first pending or missing
	// one (a secondary-only chunk defers via H3); nothing when that is under one chunk.
	chunks, _ := o.cpuRun(tail)
	reloadableEnd := startIndex + tail.loadableBlocks(chunks)*bs
	return reloadableEnd, reloadableEnd > startIndex
}

// SnapshotCachedBlocksFn delegates to the GPU tier so cluster routing keeps its
// frozen-snapshot semantics (INV-7). Offload tiers are intentionally invisible to
// the router in H1 (a routing-fidelity boundary; a later hole may expose them).
func (o *OffloadCache) SnapshotCachedBlocksFn() func([]sim.TokenID) int {
	return o.gpu.SnapshotCachedBlocksFn()
}

// PendingTransferLatency / ConsumePendingTransferLatency return 0: the offload path
// charges no explicit per-request transfer latency. Timing impact flows through
// token counts (recompute under lock-saturation → more prefill tokens → longer step
// via the existing StepTime model) and, for the dominant TTFT effect, through
// step-boundary deferral (H3 #1591) — a deferred admission is delayed by whole
// steps of WaitQ residency, not by a latency added to any single step (BC-T7).
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

// ThrashCounts implements sim.ThrashCounter: the counts behind KVThrashingRate.
func (o *OffloadCache) ThrashCounts() sim.ThrashCounts {
	return sim.ThrashCounts{Events: o.promotionsFailed, Opportunities: o.promotionsFired + o.promotionsFailed}
}

// SetClock advances the offload clock and applies transfer completions. It is the
// ONLY point completions are applied (never reading the station's internal
// pendingCompleted mid-step), and it runs at the top of each step before batch
// formation — so a promotion that finished since the last step is a HIT this step,
// not next (vLLM within-step ordering §3.4 rule 1). A backward clock is a safe
// no-op (the station's Poll never moves the clock back, INV-3).
func (o *OffloadCache) SetClock(clock int64) {
	o.clock = clock
	if o.station == nil {
		return
	}
	for _, id := range o.station.Poll(clock) {
		ref, ok := o.inflight[id]
		if !ok {
			continue // defensive: unknown/duplicate id
		}
		delete(o.inflight, id)
		switch ref.dir {
		case kvtransfer.Read:
			// Promotion (secondary→CPU) landed: -1 → 0, block becomes readable.
			for _, k := range ref.keys {
				o.cpu.completeStore(k, true)
			}
		case kvtransfer.Write:
			// Cascade replica (CPU→secondary) landed: record it in the tier and
			// release the CPU write-lock (re-evictable once its last writer finishes).
			for _, k := range ref.keys {
				o.secondary[ref.tier].store(k)
				o.cpu.unpin(k)
			}
		}
	}
}

// AllocateKVBlocks consults the tier chain and allocates on GPU. It adds the H3
// (#1591) step-boundary deferral in front of the H1 allocate path
// (allocateThroughChain), gated to NEW prefill admissions:
//
//   - A new request (not yet on the GPU) whose needed prefix requires a
//     secondary→CPU fetch is DEFERRED — it is left in the WaitQ and re-polled each
//     step (PollDeferred) until its promotion lands (then admitted) or the fetch is
//     refused/lost (then recomputed). This realizes vLLM's step-boundary re-poll:
//     the offload-attributable TTFT delay is a whole multiple of step time.
//   - A running-request continuation (Phase-1 chunked prefill, decode sub-request,
//     final-token alloc) never defers — it keeps the H1 background-promote path,
//     where a false return means GPU pressure (preempt), not "skip".
//   - A new request whose prefix is CPU-resident waits for its priced CPU->GPU load
//     (INV-K7, beginCPULoad) and is admitted on the first step after it lands, as vLLM
//     holds it in WAITING_FOR_REMOTE_KVS (scheduler.py:1258-1275 at v0.31.0).
//
// The secondary deferral is inert with no secondary tiers, and the load with no
// CPULoadTime, so stores without them are byte-identical (INV-6).
func (o *OffloadCache) AllocateKVBlocks(req *sim.Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	if st, tracked := o.deferred[req.ID]; tracked && st.holds {
		return o.admitLoaded(req, st, endIndex)
	}
	// A sequence that fails the admission gate, or whose state or slots do not fit, is refused before any tier
	// is consulted or a deferral registered, so the refusal has no side effect.
	stateStart := startIndex
	if o.gpu.SequenceState.Hold != nil {
		end, ok := o.ReloadablePrefixEnd(req, startIndex)
		stateStart = o.gpu.loadedStateStart(req, startIndex, endIndex, end, ok)
	}
	// A load is admitted on its own terms (beginCPULoad); every other path must fit its chunk.
	if !o.loadDue(req, cachedBlocks) && !o.gpu.sequenceFits(req, startIndex, stateStart, endIndex, cachedBlocks) {
		return false
	}
	// Defer only genuinely-new prefill admissions. A running-request continuation
	// already owns GPU blocks (in RequestMap). A PD decode sub-request is also NOT a
	// prefill admission: its KV is pre-reserved via ReserveTransferredKV
	// (AllocateKVBlocks(req, 0, inputLen, nil) with req.IsDecodeSubRequest already
	// set) and its prompt prefix may be secondary-resident on the decode instance —
	// deferring it would make the reservation return false and the request be
	// dropped (dropAtStart). Both keep the H1 allocate path (pre-#1591 behavior).
	if _, running := o.gpu.RequestMap[req.ID]; !running && !req.IsDecodeSubRequest {
		// Resume seed for the read-only fetch classification: past the caller's
		// GPU-matched prefix (a fresh request owns nothing yet, so RequestMap does
		// not extend it).
		reloadStartBlock := int64(len(cachedBlocks))
		reloadPrevHash := ""
		if reloadStartBlock > 0 {
			reloadPrevHash = o.gpu.Blocks[cachedBlocks[reloadStartBlock-1]].Hash
		}

		if st, tracked := o.deferred[req.ID]; tracked {
			if !st.resolved && !st.recompute {
				// Still pending. Batch formation skips these via PollDeferred, so this
				// is defensive: never re-fire a promotion, never admit mid-fetch.
				return false
			}
			// A secondary fetch that landed in CPU still has the CPU->GPU load to make.
			if started, refused := o.beginCPULoad(req, startIndex, cachedBlocks, reloadStartBlock, reloadPrevHash); started || refused {
				return false
			}
			// resolved (blocks landed) or recompute (fetch refused/lost): admit through
			// the H1 path (reloads now-ready CPU blocks; a residual miss recomputes —
			// it never re-enters defer). Clear the episode only once actually admitted;
			// on GPU pressure the entry persists so the next step retries without
			// resetting state (no re-defer, backstop preserved).
			ok := o.allocateThroughChain(req, startIndex, stateStart, endIndex, cachedBlocks)
			if ok {
				delete(o.deferred, req.ID)
			}
			return ok
		}

		if fc := o.classifyFetch(req.FullInputTokens(), reloadStartBlock, reloadPrevHash); fc.kind != fetchNone {
			o.registerDeferral(req.ID, fc)
			return false // defer: stays in WaitQ, re-polled next step
		}
		if started, refused := o.beginCPULoad(req, startIndex, cachedBlocks, reloadStartBlock, reloadPrevHash); started || refused {
			return false // waits for its CPU->GPU load holding its blocks, or does not fit (INV-K7)
		}

	}
	return o.allocateThroughChain(req, startIndex, stateStart, endIndex, cachedBlocks)
}

// loadDue reports whether a new request's admission would start a priced CPU->GPU load: a
// pure query, so the chunk pre-check is skipped only where vLLM admits by the load's own check.
func (o *OffloadCache) loadDue(req *sim.Request, cachedBlocks []int64) bool {
	if o.cpuLoadTime == nil || req.IsDecodeSubRequest {
		return false
	}
	if _, running := o.gpu.RequestMap[req.ID]; running {
		return false
	}
	if st, tracked := o.deferred[req.ID]; tracked && (st.phase == deferLoading || (!st.resolved && !st.recompute)) {
		return false
	}
	startBlock := int64(len(cachedBlocks))
	prevHash := ""
	if startBlock > 0 {
		prevHash = o.gpu.Blocks[cachedBlocks[startBlock-1]].Hash
	}
	_, blocks := o.cpuLoadPlan(req.FullInputTokens(), startBlock, prevHash)
	return blocks > 0
}

// admitLoaded admits a request whose load holds its blocks. Until the load lands it is not
// schedulable. Then, as vLLM schedules a WAITING request with num_computed_tokens > 0, it is
// scheduled from the prefix the load computed: the full-sequence gate asks for what it still
// lacks of its whole sequence (get_num_blocks_to_allocate's held-blocks path,
// single_type_kv_cache_manager.py:232-238 at v0.31.0), then the chunk is allocated as a
// running request's is.
func (o *OffloadCache) admitLoaded(req *sim.Request, st *deferralState, endIndex int64) bool {
	if !st.loaded {
		return false
	}
	if o.gpu.remainingBlocks(req, st.computed) > o.gpu.countFreeBlocks() {
		return false
	}
	// The tokens up to stored are already in its blocks; a whole-prompt load recomputes its
	// last token in place.
	from := min(st.stored, max(endIndex, st.computed))
	if !o.gpu.allocate(req, from, st.computed, max(endIndex, from), nil) {
		return false
	}
	delete(o.deferred, req.ID)
	return true
}

// allocateThroughChain is the H1 (#1590) allocate path: reload CPU-resident prefix
// blocks to the GPU free list synchronously (so a subsequent GetCachedBlocks hits
// them); a secondary-resident prefix run kicks off an async background promotion (a
// station Read job that populates CPU for a LATER lookup) but is treated as a miss
// for THIS call. The GPU commit/allocate structure mirrors the legacy TieredKVCache
// path (INV-4 GPU conservation, hash-chain correctness). H3 deferral wraps this in
// AllocateKVBlocks; for a resolved deferral this reloads the now-CPU-resident run.
func (o *OffloadCache) allocateThroughChain(req *sim.Request, startIndex, stateStart, endIndex int64, cachedBlocks []int64) bool {
	// Resume the consult past the prefix the caller already matched on GPU, and
	// past what a running request already owns (its partially-filled last block
	// must not be reloaded — same running-request trap as TieredKVCache). The
	// resume seed prevHash lets consultAndReload key only the uncached tail instead
	// of re-hashing the whole prefix (hot-path parity with the legacy path).
	reloadStartBlock := int64(len(cachedBlocks))
	reloadPrevHash := ""
	if reloadStartBlock > 0 {
		reloadPrevHash = o.gpu.Blocks[cachedBlocks[reloadStartBlock-1]].Hash
	}
	if owned, ok := o.gpu.RequestMap[req.ID]; ok && int64(len(owned)) > reloadStartBlock {
		reloadStartBlock = int64(len(owned))
		reloadPrevHash = o.gpu.Blocks[owned[len(owned)-1]].Hash
	}

	if o.consultAndReload(req.FullInputTokens(), reloadStartBlock, reloadPrevHash) {
		// A CPU->GPU reload happened: recompute the GPU-cached prefix (now longer).
		newCached := o.gpu.GetCachedBlocks(req.FullInputTokens())
		newStart := int64(len(newCached)) * o.gpu.BlockSize()
		bs := o.gpu.BlockSize()
		if newStart > startIndex {
			_, running := o.gpu.RequestMap[req.ID]
			if newStart >= endIndex {
				// Entire requested range is cached after reload; commit it.
				endBlock := min((endIndex+bs-1)/bs, int64(len(newCached)))
				if running {
					if startBlock := (startIndex + bs - 1) / bs; startBlock < endBlock {
						o.gpu.commitCachedBlocks(req.ID, newCached[startBlock:endBlock])
					}
				} else {
					o.gpu.commitCachedBlocks(req.ID, newCached[:endBlock])
				}
				// Its bounded state (#1941) and slots (#1627), on every allocation as vLLM
				// passes num_lookahead_tokens to every allocate_slots
				// (scheduler.py:745-751, 1214 at v0.31.0); room was checked before the reload.
				o.gpu.chargeSequence(req.ID, stateStart, endIndex)
				return true
			}
			// Partial improvement: commit the reloaded prefix, then allocate the tail.
			newStartBlock := newStart / bs
			if running {
				if startBlock := (startIndex + bs - 1) / bs; startBlock < newStartBlock {
					o.gpu.commitCachedBlocks(req.ID, newCached[startBlock:newStartBlock])
				}
			} else {
				o.gpu.commitCachedBlocks(req.ID, newCached[:newStartBlock])
			}
			if o.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached) {
				return true
			}
			// The tail did not fit. A request that stays queued must not keep the reloaded
			// prefix it was committed: it would hold blocks no running request can free, and
			// a pool that tight livelocks. Undo the commit; the reloaded blocks go back to the
			// free list with their hashes, still a hit for the retry.
			if !running {
				o.gpu.ReleaseKVBlocks(req)
			}
			return false
		}
		// Reload produced no prefix hit beyond startIndex; allocate as given.
		return o.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	}

	// No CPU-resident continuation. Allocate on GPU exactly as the single-tier path.
	ok := o.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	if !ok {
		o.offloadMissCount++
	}
	return ok
}

// consultAndReload walks the uncached tail from startBlock in offload chunks
// (offload_chunks.go). It reloads every CPU-readable chunk's missing blocks onto the GPU
// free list (returning true if any reload happened), stops at a HIT_PENDING chunk (a
// promotion already in flight), and -- on the first CPU-miss chunk that is
// secondary-resident -- initiates one async promotion of the contiguous same-tier run and
// stops (the run is not available to this request). A run shorter than one chunk past
// startBlock is not loaded, as vLLM loads nothing then. It is the offload analogue of
// TieredKVCache's reloadPrefixFromCPU, keyed through sim/internal/kvkey so the
// CPU/secondary keyspace is a subset of the GPU block hashes (BC-K1).
func (o *OffloadCache) consultAndReload(tokens []sim.TokenID, startBlock int64, prevHash string) bool {
	tail, ok := o.tailOf(tokens, startBlock, prevHash)
	if !ok {
		return false
	}
	bs := o.gpu.BlockSize()
	chunks, stop := o.cpuRun(tail)
	reloaded := false
	if tail.loadableBlocks(chunks) > 0 {
		maxReloads := o.gpu.countFreeBlocks() // never re-pop the same free block
		var reloadCount int64
		for c := tail.firstChunk; c < tail.firstChunk+chunks; c++ {
			lo, hi := tail.blocks(c)
			var missing []int64
			for i := lo; i < hi; i++ {
				if _, inGPU := o.gpu.HashToBlock[string(tail.blockKey(i))]; !inGPU {
					missing = append(missing, i)
				}
			}
			if len(missing) == 0 {
				continue // already on GPU
			}
			if reloadCount+int64(len(missing)) > maxReloads {
				return reloaded // a chunk loads whole or not at all
			}
			for _, i := range missing {
				gpuBlk := o.gpu.popFreeBlock()
				if gpuBlk == nil {
					return reloaded
				}
				// Lazy hash deletion (vLLM parity): clear a stale hash before refilling.
				o.gpu.dropHash(gpuBlk)
				h := string(tail.blockKey(i))
				start := i * bs
				gpuBlk.Tokens = append(gpuBlk.Tokens[:0], tokens[start:start+bs]...)
				gpuBlk.Hash = h
				gpuBlk.RefCount = 0
				gpuBlk.InUse = false
				o.gpu.HashToBlock[h] = gpuBlk.ID
				o.gpu.appendToFreeList(gpuBlk)
				o.reloadCount++
				reloadCount++
			}
			o.cpu.touchKey(tail.key(c)) // chunk is hot: refresh LRU recency
			reloaded = true
		}
	}
	if stop == cpuMiss {
		o.maybePromoteFromSecondary(tail, tail.firstChunk+chunks)
	}
	return reloaded
}

// maybePromoteFromSecondary initiates one async promotion (secondary→CPU) for the
// contiguous run of not-yet-CPU-resident chunks, starting at chunk c,
// that are all held by the SAME first-matching secondary tier. It allocates
// NOT-READY CPU slots (prepareStore, all-or-nothing under the BC-C5 evictable
// gate) and submits a single Read job for the run. If the gate refuses the run the
// promotion is a miss (recompute); the request has already been served whatever
// GPU/CPU prefix it had.
//
// This is the H1 background-promotion path, used for RUNNING-request continuations
// that reach a secondary-resident block (the triggering request does NOT wait —
// the promotion populates CPU for a later lookup). NEW prefill admissions instead
// DEFER (H3, #1591): see classifyFetch/registerDeferral, which reuse the same
// gatherSecondaryRun + firePromotionRun helpers.
func (o *OffloadCache) maybePromoteFromSecondary(tail offloadTail, c int64) {
	run, tier, ok := o.gatherSecondaryRun(tail, c)
	if !ok {
		return // no station/tiers, or a genuine miss (absent from every secondary tier)
	}
	o.firePromotionRun(run, tier)
}

// gatherSecondaryRun returns the keys of the contiguous run of not-yet-CPU-resident chunks
// starting at chunk c that are all held by the SAME first-matching secondary tier, plus
// that tier index. ok is false when there is no station/tier or chunk c is absent from
// every secondary tier (a genuine miss). It is read-only (no allocation, no submission) so
// both the H1 promote path and the H3 deferral classifier can use it.
func (o *OffloadCache) gatherSecondaryRun(tail offloadTail, c int64) ([]kvkey.BlockKey, int, bool) {
	if o.station == nil || len(o.secondary) == 0 || c >= tail.endChunk {
		return nil, 0, false
	}
	tier, ok := lookupSecondary(o.secondary, tail.key(c))
	if !ok {
		return nil, 0, false // genuine miss (absent from every secondary tier)
	}
	run := []kvkey.BlockKey{tail.key(c)}
	for j := c + 1; j < tail.endChunk; j++ {
		if o.onGPU(tail, j) {
			break
		}
		if o.cpu.lookup(tail.key(j)) != cpuMiss {
			break // already CPU-resident or a promotion in flight
		}
		if tj, okj := lookupSecondary(o.secondary, tail.key(j)); !okj || tj != tier {
			break
		}
		run = append(run, tail.key(j))
	}
	return run, tier, true
}

// firePromotionRun allocates NOT-READY CPU slots for the run (prepareStore,
// all-or-nothing under the BC-C5 evictable gate) and submits a single Read job.
// It returns the station JobID and true on success; (0, false) when the gate
// refuses the run (promotionsFailed++, recompute) or the (unbounded) station
// somehow rejects. Shared by the H1 background path and the H3 deferral path so
// the ref_cnt/station bookkeeping lives in exactly one place.
func (o *OffloadCache) firePromotionRun(run []kvkey.BlockKey, tier int) (kvtransfer.JobID, bool) {
	granted := o.cpu.prepareStore(run) // all-or-nothing over the (all-fresh) run
	if granted == 0 {
		o.promotionsFailed++ // BC-C5: evictable gate refused the run -> recompute
		return 0, false
	}
	id, accepted := o.station.Submit(kvtransfer.TransferJob{
		Tier:         tier,
		Direction:    kvtransfer.Read,
		Bytes:        int64(granted) * o.chunkBytes(),
		SubmitTick:   o.clock,
		JitterFactor: o.drawJitterFactor(tier),
	})
	if !accepted {
		// MaxQueueDepth==0 makes rejection impossible; roll back defensively so a
		// future bounded queue can never strand -1 slots with no in-flight Read.
		for _, k := range run {
			o.cpu.completeStore(k, false)
		}
		return 0, false
	}
	o.promotionsFired++
	o.inflight[id] = jobRef{keys: run, tier: tier, dir: kvtransfer.Read}
	return id, true
}

// MirrorToCPU stores each request's newly-completed full GPU blocks into the CPU
// tier and, for a block that newly lands there, cascades it to every secondary
// tier (write-through, BC-C7a). Each cascade Write pins the CPU block for the
// write duration (ref_cnt++), which is what starves the evictable pool and forces
// the BC-C5 recompute path under a burst of writes. When the CPU tier is full and
// every block is pinned, the store finds no evictable victim and the block is
// SKIPPED (counted) — never force-evicting a locked block (BC-C4) or dropping data
// silently (R1).
//
// What is offered for store is decided by a single offloadable-token clamp modeling
// vLLM's mechanism (NOT a per-block prompt/decode classification): the request's
// computed KV is its full owned blocks; when OffloadPromptOnly (vLLM default TRUE) that
// count is truncated to the prompt length (offloading/scheduler.py:588-597,
// _calc_num_offloadable_tokens); then it is floor-divided into whole chunks
// (storable_chunks, :401) — so a chunk holding any decode token is never formed. When
// !OffloadPromptOnly, full decode blocks fall inside the offloadable range and are
// offloaded here too; they need no hashing step of their own — the GPU tier's partial-fill
// path (cache.go) already hashes every completed block prefix-consistently (for
// block_size > 1), so a later request whose input contains those tokens reloads them.
// MirrorToCPU never writes gpu.HashToBlock (pure consumer), so switching the policy cannot
// perturb GPU-tier behavior. Uses the stored clock (SetClock ran earlier this step).
func (o *OffloadCache) MirrorToCPU(batch []*sim.Request) {
	bs := o.gpu.BlockSize()
	bpc := o.blocksPerChunk
	for _, req := range batch {
		blockIDs, ok := o.gpu.RequestMap[req.ID]
		if !ok {
			continue
		}
		// Offloadable-token clamp (single decision point). The chunk stride derives from
		// the actual gpu.BlockSize(), NOT cfg.BlockSize (they can differ in test fixtures).
		numOffloadableTokens := int64(len(blockIDs)) * bs
		if o.offloadPromptOnly {
			numOffloadableTokens = min(numOffloadableTokens, req.InputLen()) // vLLM prompt-only clamp
		}
		storableChunks := numOffloadableTokens / (bs * bpc) // storable_chunks, whole chunks only
	chunks:
		for c := int64(0); c < storableChunks; c++ {
			for i := c * bpc; i < (c+1)*bpc; i++ {
				if blk := o.gpu.Blocks[blockIDs[i]]; blk.Hash == "" || util.Len64(blk.Tokens) < bs {
					continue chunks // only full, hashed chunks are offloadable
				}
			}
			key := kvkey.BlockKey(o.gpu.Blocks[blockIDs[(c+1)*bpc-1]].Hash) // the last block's hash
			if o.cpu.lookup(key) != cpuMiss {
				o.cpu.touchKey(key) // already resident: refresh recency, no re-cascade
				continue
			}
			if !o.cpu.store(key) {
				o.mirrorSkipped++ // CPU full and all-pinned: skip (BC-C4, R1)
				continue
			}
			o.cascade(key)
		}
	}
}

// cascade submits a write-through replica of a freshly-stored CPU block to every
// secondary tier and pins the CPU block once per in-flight write, so it is
// non-evictable until every replica completes (BC-C3 pin, BC-C7a fan-out). The
// pins are released as the Write jobs complete (SetClock). No-op when there are no
// secondary tiers (CPU-only offload).
func (o *OffloadCache) cascade(key kvkey.BlockKey) {
	if o.station == nil {
		return
	}
	for tier := range o.secondary {
		o.cpu.pin(key) // lock for the write duration (ref_cnt++)
		id, accepted := o.station.Submit(kvtransfer.TransferJob{
			Tier:         tier,
			Direction:    kvtransfer.Write,
			Bytes:        o.chunkBytes(),
			SubmitTick:   o.clock,
			JitterFactor: o.drawJitterFactor(tier),
		})
		if !accepted {
			o.cpu.unpin(key) // MaxQueueDepth==0 => unreachable; defensive
			continue
		}
		o.inflight[id] = jobRef{keys: []kvkey.BlockKey{key}, tier: tier, dir: kvtransfer.Write}
	}
}

// EnableCPUTierHistory starts journaling CPU-tier readability so
// CPUTierPrefixBlockCountAt can answer for an earlier time. Blocks already readable
// count as readable since the start.
func (o *OffloadCache) EnableCPUTierHistory() {
	if o.cpu.journal != nil {
		return
	}
	j := &cpuReadabilityJournal{now: func() int64 { return o.clock }, readableAt: map[kvkey.BlockKey]int64{}, gone: map[kvkey.BlockKey][2]int64{}}
	for k, blk := range o.cpu.blocks {
		if blk.refCnt >= 0 {
			j.readableAt[k] = math.MinInt64
		}
	}
	o.cpu.journal = j
}

// CPUTierPrefixBlockCountAt is CPUTierPrefixBlockCount as of time at (<= the
// current clock), from the readability journal; it also forgets history before at.
// Panics unless EnableCPUTierHistory was called (R3).
func (o *OffloadCache) CPUTierPrefixBlockCountAt(tokens []sim.TokenID, at int64) int {
	j := o.cpu.journal
	if j == nil {
		panic("OffloadCache.CPUTierPrefixBlockCountAt: EnableCPUTierHistory not called")
	}
	j.prune(at)
	return o.cpuPrefixBlocks(tokens, func(k kvkey.BlockKey) bool { return j.readableAtTime(k, at) })
}

// CPUTierPrefixBlockCount returns the number of leading full blocks of tokens that
// are readable in the CPU offload tier (cpuHit), in whole chunks, stopping at the first
// chunk that is not. A pure query: no tier state or recency changes. It is the CPU-tier
// analogue of GetCachedBlocks for the router's CPU-tier prefix view (#1912).
func (o *OffloadCache) CPUTierPrefixBlockCount(tokens []sim.TokenID) int {
	return o.cpuPrefixBlocks(tokens, func(k kvkey.BlockKey) bool { return o.cpu.lookup(k) == cpuHit })
}

// cpuPrefixBlocks counts the leading blocks of tokens in whole chunks whose key readable
// accepts.
func (o *OffloadCache) cpuPrefixBlocks(tokens []sim.TokenID, readable func(kvkey.BlockKey) bool) int {
	bs := int(o.gpu.BlockSize())
	bpc := int(o.blocksPerChunk)
	n := len(tokens) / bs / bpc * bpc
	prev := ""
	count := 0
	for i := 0; i < n; i++ {
		prev = hash.HashBlock(prev, tokens[i*bs:(i+1)*bs])
		if (i+1)%bpc != 0 {
			continue
		}
		if !readable(kvkey.BlockKey(prev)) {
			break
		}
		count += bpc
	}
	return count
}
