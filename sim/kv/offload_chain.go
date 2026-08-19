package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
	"github.com/inference-sim/inference-sim/sim/internal/util"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

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

	inflight map[kvtransfer.JobID]jobRef // in-flight transfer jobs, applied on SetClock→Poll

	perBlockBytes     int64 // resolved per-rank KV bytes of one GPU block (transfer-job sizing)
	blockSizeTokens   int64
	offloadPromptOnly bool
	clock             int64

	// Metrics / diagnostics (all load-independent where it matters, #1586).
	offloadMissCount int64 // neither CPU nor secondary served AND GPU alloc failed
	promotionsFired  int64 // secondary→CPU promotions initiated (Read jobs submitted)
	promotionsFailed int64 // secondary hits refused by the BC-C5 evictable gate → recompute
	reloadCount      int64 // CPU→GPU reloads performed (diagnostic)
	mirrorSkipped    int64 // MirrorToCPU stores skipped because CPU was full and all-pinned
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
		inflight:          make(map[kvtransfer.JobID]jobRef),
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

// AllocateKVBlocks consults the tier chain, then allocates on GPU. CPU-resident
// prefix blocks are reloaded to the GPU free list synchronously (so a subsequent
// GetCachedBlocks hits them); a secondary-resident prefix run kicks off an async
// promotion (a station Read job that populates CPU for a LATER lookup) but is
// treated as a miss for THIS request (the triggering request never waits for its
// own promotion — that request-level deferral is H3, #1591). The GPU commit/
// allocate structure mirrors the legacy TieredKVCache path (INV-4 GPU
// conservation, hash-chain correctness).
func (o *OffloadCache) AllocateKVBlocks(req *sim.Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	// Resume the consult past the prefix the caller already matched on GPU, and
	// past what a running request already owns (its partially-filled last block
	// must not be reloaded — same running-request trap as TieredKVCache).
	reloadStartBlock := int64(len(cachedBlocks))
	if owned, ok := o.gpu.RequestMap[req.ID]; ok && int64(len(owned)) > reloadStartBlock {
		reloadStartBlock = int64(len(owned))
	}

	if o.consultAndReload(req.FullInputTokens(), reloadStartBlock) {
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
func (o *OffloadCache) consultAndReload(tokens []sim.TokenID, startBlock int64) bool {
	bs := o.gpu.BlockSize()
	n := util.Len64(tokens) / bs
	if startBlock >= n {
		return false
	}
	keys := kvkey.DeriveChunkKeys("", tokens, int(bs)) // keys[i] == GPU block hash for block i (BC-K1)
	maxReloads := o.gpu.countFreeBlocks()              // never re-pop the same free block
	reloaded := false
	var reloadCount int64
	for i := startBlock; i < n; i++ {
		h := string(keys[i])
		if _, inGPU := o.gpu.HashToBlock[h]; inGPU {
			continue // already on GPU
		}
		switch o.cpu.lookup(keys[i]) {
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
			o.cpu.touchKey(keys[i]) // block is hot: refresh LRU recency
			o.reloadCount++
			reloaded = true
			reloadCount++
		case cpuHitPending:
			return reloaded // promotion in flight for this block; stop (hierarchical)
		case cpuMiss:
			o.maybePromoteFromSecondary(keys, i, n)
			return reloaded // stop after initiating (or refusing) one promotion
		}
	}
	return reloaded
}

// maybePromoteFromSecondary initiates one async promotion (secondary→CPU) for the
// contiguous run of not-yet-CPU-resident blocks, starting at block i, that are all
// held by the SAME first-matching secondary tier. It allocates NOT-READY CPU slots
// (prepareStore, all-or-nothing under the BC-C5 evictable gate) and submits a
// single Read job for the run. If the gate refuses the run the promotion is a miss
// (recompute); the request has already been served whatever GPU/CPU prefix it had.
func (o *OffloadCache) maybePromoteFromSecondary(keys []kvkey.BlockKey, i, n int64) {
	if o.station == nil || len(o.secondary) == 0 {
		return
	}
	tier, ok := lookupSecondary(o.secondary, keys[i])
	if !ok {
		return // genuine miss (absent from every secondary tier)
	}
	run := []kvkey.BlockKey{keys[i]}
	for j := i + 1; j < n; j++ {
		if _, inGPU := o.gpu.HashToBlock[string(keys[j])]; inGPU {
			break
		}
		if o.cpu.lookup(keys[j]) != cpuMiss {
			break // already CPU-resident or a promotion in flight
		}
		if tj, okj := lookupSecondary(o.secondary, keys[j]); !okj || tj != tier {
			break
		}
		run = append(run, keys[j])
	}
	granted := o.cpu.prepareStore(run) // all-or-nothing over the (all-fresh) run
	if granted == 0 {
		o.promotionsFailed++ // BC-C5: evictable gate refused the run -> recompute
		return
	}
	id, accepted := o.station.Submit(kvtransfer.TransferJob{
		Tier:       tier,
		Direction:  kvtransfer.Read,
		Bytes:      int64(granted) * o.perBlockBytes,
		SubmitTick: o.clock,
	})
	if !accepted {
		// MaxQueueDepth==0 makes rejection impossible; roll back defensively so a
		// future bounded queue can never strand -1 slots with no in-flight Read.
		for _, k := range run {
			o.cpu.completeStore(k, false)
		}
		return
	}
	o.promotionsFired++
	o.inflight[id] = jobRef{keys: run, tier: tier, dir: kvtransfer.Read}
}

// MirrorToCPU stores each request's newly-completed full GPU blocks into the CPU
// tier and, for a block that newly lands there, cascades it to every secondary
// tier (write-through, BC-C7a). Each cascade Write pins the CPU block for the
// write duration (ref_cnt++), which is what starves the evictable pool and forces
// the BC-C5 recompute path under a burst of writes. When the CPU tier is full and
// every block is pinned, the store finds no evictable victim and the block is
// SKIPPED (counted) — never force-evicting a locked block (BC-C4) or dropping data
// silently (R1). With OffloadPromptOnly (vLLM default TRUE) only prompt blocks are
// offloaded; decode-generated blocks are not. Uses the stored clock (SetClock ran
// earlier this step).
func (o *OffloadCache) MirrorToCPU(batch []*sim.Request) {
	bs := o.gpu.BlockSize()
	for _, req := range batch {
		blockIDs, ok := o.gpu.RequestMap[req.ID]
		if !ok {
			continue
		}
		promptBlocks := int64(len(blockIDs))
		if o.offloadPromptOnly {
			promptBlocks = req.InputLen() / bs // full prompt blocks only
		}
		for i, blockID := range blockIDs {
			if int64(i) >= promptBlocks {
				break // OffloadPromptOnly: stop at the first decode block
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
			Tier:       tier,
			Direction:  kvtransfer.Write,
			Bytes:      o.perBlockBytes,
			SubmitTick: o.clock,
		})
		if !accepted {
			o.cpu.unpin(key) // MaxQueueDepth==0 => unreachable; defensive
			continue
		}
		o.inflight[id] = jobRef{keys: []kvkey.BlockKey{key}, tier: tier, dir: kvtransfer.Write}
	}
}
