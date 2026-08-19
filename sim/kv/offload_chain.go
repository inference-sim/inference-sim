package kv

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
	"github.com/inference-sim/inference-sim/sim/internal/util"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// jitterMinFactor is the floor for a sampled latency-jitter multiplier (#1581
// BC-D5): a transfer can be at most this much FASTER than its mean device service
// time, so a left-tail Gaussian draw can never drive service time to zero or
// negative. It is a numerical guard, not a physical parameter.
const jitterMinFactor = 0.05

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
	existenceKnown map[kvkey.BlockKey]struct{}

	perBlockBytes     int64 // resolved per-rank KV bytes of one GPU block (transfer-job sizing)
	offloadPromptOnly bool
	blocksPerChunk    int64 // vLLM blocks_per_chunk (== 1 at H1; NewOffloadCache panics otherwise).
	// Kept so the offloadable-token floor-divide reads as vLLM's tokens_per_chunk = bs*blocksPerChunk
	// and stays correct when a future PR lands blocks_per_chunk > 1.
	clock int64

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
}

// NewOffloadCache builds the tier chain from a resolved, enabled KVOffloadConfig.
// It panics on a config the mechanism cannot honor — the derived PerBlockBytes
// (required, > 0), the H1 restrictions (BlocksPerChunk == 1, eviction_policy
// "lru"), and a CPU budget too small for one block. Panic (not logrus.Fatalf) is
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
		inflight:          make(map[kvtransfer.JobID]jobRef),
		deferred:          make(map[string]*deferralState),
		existenceKnown:    make(map[kvkey.BlockKey]struct{}),
		perBlockBytes:     cfg.PerBlockBytes,
		offloadPromptOnly: cfg.OffloadPromptOnly,
		blocksPerChunk:    cfg.BlocksPerChunk,
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
func (o *OffloadCache) BlockSize() int64                 { return o.gpu.BlockSize() }
func (o *OffloadCache) UsedBlocks() int64                { return o.gpu.UsedBlocks() }
func (o *OffloadCache) TotalCapacity() int64             { return o.gpu.TotalCapacity() }

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
//
// The deferral machinery is inert when there are no secondary tiers, so CPU-only
// offload and all non-offload stores are byte-identical (INV-6).
func (o *OffloadCache) AllocateKVBlocks(req *sim.Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
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
			// resolved (blocks landed) or recompute (fetch refused/lost): admit through
			// the H1 path (reloads now-ready CPU blocks; a residual miss recomputes —
			// it never re-enters defer). Clear the episode only once actually admitted;
			// on GPU pressure the entry persists so the next step retries without
			// resetting state (no re-defer, backstop preserved).
			ok := o.allocateThroughChain(req, startIndex, endIndex, cachedBlocks)
			if ok {
				delete(o.deferred, req.ID)
			}
			return ok
		}

		if fc := o.classifyFetch(req.FullInputTokens(), reloadStartBlock, reloadPrevHash); fc.kind != fetchNone {
			o.registerDeferral(req.ID, fc)
			return false // defer: stays in WaitQ, re-polled next step
		}
	}
	return o.allocateThroughChain(req, startIndex, endIndex, cachedBlocks)
}

// allocateThroughChain is the H1 (#1590) allocate path: reload CPU-resident prefix
// blocks to the GPU free list synchronously (so a subsequent GetCachedBlocks hits
// them); a secondary-resident prefix run kicks off an async background promotion (a
// station Read job that populates CPU for a LATER lookup) but is treated as a miss
// for THIS call. The GPU commit/allocate structure mirrors the legacy TieredKVCache
// path (INV-4 GPU conservation, hash-chain correctness). H3 deferral wraps this in
// AllocateKVBlocks; for a resolved deferral this reloads the now-CPU-resident run.
func (o *OffloadCache) allocateThroughChain(req *sim.Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
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
			return o.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
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

// consultAndReload walks the prefix blocks from startBlock. It reloads every
// CPU-resident (ready) block onto the GPU free list (returning true if any
// reload happened), stops at a HIT_PENDING block (a promotion already in flight),
// and — on the first CPU-miss whose block is secondary-resident — initiates one
// async promotion of the contiguous same-tier run and stops (the run is not
// available to this request). It is the offload analogue of TieredKVCache's
// reloadPrefixFromCPU: block-granular (BlocksPerChunk==1), keyed through
// sim/internal/kvkey so the CPU/secondary keyspace is byte-identical to the GPU
// block hashes (BC-K1).
func (o *OffloadCache) consultAndReload(tokens []sim.TokenID, startBlock int64, prevHash string) bool {
	bs := o.gpu.BlockSize()
	n := util.Len64(tokens) / bs
	if startBlock >= n {
		return false
	}
	// Key ONLY the uncached tail [startBlock, n), seeded by prevHash, so the
	// already-cached prefix is never re-hashed (hot-path parity). tailKeys[j] is the
	// key for block startBlock+j, byte-identical to that GPU block hash (BC-K1).
	tailKeys := kvkey.DeriveChunkKeys(kvkey.BlockKey(prevHash), tokens[startBlock*bs:], int(bs))
	maxReloads := o.gpu.countFreeBlocks() // never re-pop the same free block
	reloaded := false
	var reloadCount int64
	for i := startBlock; i < n; i++ {
		key := tailKeys[i-startBlock]
		h := string(key)
		if _, inGPU := o.gpu.HashToBlock[h]; inGPU {
			continue // already on GPU
		}
		switch o.cpu.lookup(key) {
		case cpuHit:
			if reloadCount >= maxReloads {
				return reloaded
			}
			gpuBlk := o.gpu.popFreeBlock()
			if gpuBlk == nil {
				return reloaded
			}
			// Lazy hash deletion (vLLM parity): clear a stale hash before refilling.
			if gpuBlk.Hash != "" {
				delete(o.gpu.HashToBlock, gpuBlk.Hash)
				gpuBlk.Hash = ""
			}
			start := i * bs
			gpuBlk.Tokens = append(gpuBlk.Tokens[:0], tokens[start:start+bs]...)
			gpuBlk.Hash = h
			gpuBlk.RefCount = 0
			gpuBlk.InUse = false
			o.gpu.HashToBlock[h] = gpuBlk.ID
			o.gpu.appendToFreeList(gpuBlk)
			o.cpu.touchKey(key) // block is hot: refresh LRU recency
			o.reloadCount++
			reloaded = true
			reloadCount++
		case cpuHitPending:
			return reloaded // promotion in flight for this block; stop (hierarchical)
		case cpuMiss:
			o.maybePromoteFromSecondary(tailKeys, int(i-startBlock))
			return reloaded // stop after initiating (or refusing) one promotion
		}
	}
	return reloaded
}

// maybePromoteFromSecondary initiates one async promotion (secondary→CPU) for the
// contiguous run of not-yet-CPU-resident blocks, starting at tailKeys[localIdx],
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
func (o *OffloadCache) maybePromoteFromSecondary(tailKeys []kvkey.BlockKey, localIdx int) {
	run, tier, ok := o.gatherSecondaryRun(tailKeys, localIdx)
	if !ok {
		return // no station/tiers, or a genuine miss (absent from every secondary tier)
	}
	o.firePromotionRun(run, tier)
}

// gatherSecondaryRun returns the contiguous run of not-yet-CPU-resident blocks
// starting at tailKeys[localIdx] that are all held by the SAME first-matching
// secondary tier, plus that tier index. ok is false when there is no station/tier
// or the starting block is absent from every secondary tier (a genuine miss). It is
// read-only (no allocation, no submission) so both the H1 promote path and the H3
// deferral classifier can use it.
func (o *OffloadCache) gatherSecondaryRun(tailKeys []kvkey.BlockKey, localIdx int) ([]kvkey.BlockKey, int, bool) {
	if o.station == nil || len(o.secondary) == 0 {
		return nil, 0, false
	}
	tier, ok := lookupSecondary(o.secondary, tailKeys[localIdx])
	if !ok {
		return nil, 0, false // genuine miss (absent from every secondary tier)
	}
	run := []kvkey.BlockKey{tailKeys[localIdx]}
	for j := localIdx + 1; j < len(tailKeys); j++ {
		if _, inGPU := o.gpu.HashToBlock[string(tailKeys[j])]; inGPU {
			break
		}
		if o.cpu.lookup(tailKeys[j]) != cpuMiss {
			break // already CPU-resident or a promotion in flight
		}
		if tj, okj := lookupSecondary(o.secondary, tailKeys[j]); !okj || tj != tier {
			break
		}
		run = append(run, tailKeys[j])
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
		Bytes:        int64(granted) * o.perBlockBytes,
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
	for _, req := range batch {
		blockIDs, ok := o.gpu.RequestMap[req.ID]
		if !ok {
			continue
		}
		// Offloadable-token clamp (single decision point). The chunk stride derives from
		// the actual gpu.BlockSize(), NOT cfg.BlockSize (they can differ in test fixtures).
		tokensPerChunk := bs * o.blocksPerChunk
		numOffloadableTokens := int64(len(blockIDs)) * bs
		if o.offloadPromptOnly {
			numOffloadableTokens = min(numOffloadableTokens, req.InputLen()) // vLLM prompt-only clamp
		}
		maxOffloadBlocks := (numOffloadableTokens / tokensPerChunk) * o.blocksPerChunk
		for i, blockID := range blockIDs {
			if int64(i) >= maxOffloadBlocks {
				break // outside the offloadable chunk range (prompt-only tail / decode)
			}
			blk := o.gpu.Blocks[blockID]
			if blk.Hash == "" || util.Len64(blk.Tokens) < bs {
				continue // only full, hashed blocks are offloadable
			}
			key := kvkey.BlockKey(blk.Hash)
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
			Bytes:        o.perBlockBytes,
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
