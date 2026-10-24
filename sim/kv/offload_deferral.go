package kv

import (
	"sort"

	"github.com/inference-sim/blis-vllm/compat/kvkey"
	"github.com/inference-sim/inference-sim/sim"
)

// H3 kv_deferral (#1591): step-boundary re-poll for secondary-tier (disk/object
// store) KV fetches.
//
// vLLM never blocks on the disk. When a waiting request needs KV blocks resident
// only on a secondary tier, the scheduler sets it aside and re-examines it on the
// NEXT step (scheduler.py:835-841 `continue`). So the offload-attributable TTFT
// delay is a WHOLE MULTIPLE of step time, not a disk-bandwidth latency — and it is
// realized here by the request sitting in the WaitQ and being admitted a few steps
// later (not by charging per-step latency; ConsumePendingTransferLatency stays 0).
//
// Round count (verified against vLLM 63a9a5010a):
//   - COLD secondary hit: k >= 3 — RETRY (the tier's existence check is itself
//     async/step-batched), then a promote round (existence resolved, secondary→CPU
//     Read submitted, CPU slot ref_cnt=-1), then >= 1 in-flight round until the
//     Read lands, then HIT → schedulable.
//   - WARM hit (existence already resolved/cached): k >= 2 — skips RETRY.
// Both are modeled and distinguished; the observable invariant is
// cold_rounds - warm_rounds == 1.
//
// The whole mechanism is INERT unless the offload chain has secondary tiers, and
// deferral applies ONLY to new prefill admissions (not running continuations), so
// existing runs are byte-identical (INV-6).

// maxDeferTicks is a defensive livelock backstop (R19): a deferral that has waited
// this many ticks without resolving is force-recomputed rather than deferring
// forever. It is a large, load-independent constant — the PRIMARY bound is the
// transfer station's completion guarantee (every submitted Read job completes in
// finite time) plus a single fetch attempt per episode, so this cap only catches
// impossible states (e.g. an awaited block evicted before use that somehow never
// re-resolves). Ticks are microseconds; a legitimate single fetch completes many
// orders of magnitude below this.
const maxDeferTicks int64 = 1_000_000_000 // ~1000 s

// deferralPhase is where a deferred request sits in the k-round sequence.
type deferralPhase int

const (
	// deferRetry: a COLD secondary hit whose tier existence check is still
	// outstanding (round 1). No promotion submitted yet; PollDeferred resolves the
	// existence next step and transitions to deferPromoting.
	deferRetry deferralPhase = iota
	// deferPromoting: existence resolved; the secondary→CPU promotion is in flight
	// (either this request's own Read job, or a block another request is already
	// promoting). Resolves when the awaited keys become CPU-readable (cpuHit).
	deferPromoting
	// deferLoading: the CPU->GPU load of a CPU-resident prefix is in flight (INV-K7). Resolves
	// when the clock reaches readyAt; its chunks stay pinned until then, as vLLM holds a
	// loading chunk's ref_cnt (kv_offload/cpu/manager.py:197-240 at v0.31.0).
	deferLoading
)

// deferralState is the per-request record of an in-progress secondary-tier fetch.
// A request has at most one live episode; a single fetch attempt is made, then the
// request is admitted (blocks landed) or recomputed (residual miss) — never
// re-deferred, so the episode is bounded (BC-T3, R19).
type deferralState struct {
	phase     deferralPhase
	keys      []kvkey.BlockKey // the contiguous run being fetched / awaited
	tier      int              // secondary tier holding the run; -1 when awaiting another request's promotion
	hasJob    bool             // this request submitted its own Read job (owns the -1 CPU slots)
	startTick int64            // when the episode began (for the maxDeferTicks backstop)
	resolved  bool             // awaited keys are CPU-readable; admit on next examination
	recompute bool             // fetch refused/lost; admit as a miss (recompute) on next examination
	loadKeys  []kvkey.BlockKey // deferLoading: the pinned chunks being loaded
	readyAt   int64            // deferLoading: when the load lands
	// A load holds the request's GPU blocks from its start (allocateLoad) until the request is
	// admitted or leaves: req is the request, stored the tokens its blocks hold, computed the
	// prefix it counts as computed (a whole-prompt load recomputes its last token), gpuHit the
	// GPU prefix hit it started from, and loaded that the load has landed.
	req                      *sim.Request
	holds                    bool
	loaded                   bool
	stored, computed, gpuHit int64
}

// fetchKind classifies what a new request's needed prefix requires beyond its
// GPU/CPU-resident-and-reloadable head.
type fetchKind int

const (
	fetchNone      fetchKind = iota // fully cached/reloadable, or a genuine miss (recompute) — no defer
	fetchSecondary                  // a cpuMiss block is secondary-resident — defer to promote its run
	fetchPending                    // a cpuHitPending block — defer to await an in-flight promotion
	fetchProbe                      // a cpuMiss tail no secondary tier holds, not yet probed — one RETRY round
)

// fetchClass is the read-only classification of a new request's consult.
type fetchClass struct {
	kind fetchKind
	run  []kvkey.BlockKey // fetchSecondary: the run to promote; fetchPending: the awaited key(s)
	tier int              // fetchSecondary: the holding tier; else -1
	// probe is the CPU-miss tail the lookup asks the secondary tiers about: every chunk from
	// the one that ended the GPU/CPU run to the prompt's last (the connector keeps scanning
	// past a RETRY, offloading/scheduler.py:697-700 at v0.31.0).
	probe []kvkey.BlockKey
}

// classifyFetch walks the uncached tail in offload chunks exactly like
// consultAndReload but WITHOUT mutating anything, and reports whether a NEW request
// must defer. It skips GPU-resident and CPU-ready (reloadable) blocks — those
// extend the cached prefix synchronously at admission — and stops at the first:
//   - cpuHitPending  → fetchPending (await the in-flight promotion),
//   - cpuMiss that is secondary-resident → fetchSecondary (promote its run),
//   - cpuMiss absent everywhere and never probed → fetchProbe (one RETRY round),
//   - cpuMiss absent everywhere, already probed (genuine miss) → fetchNone (recompute),
//   - end of tail → fetchNone (whole prefix cached/reloadable).
//
// A CPU miss is a secondary lookup whatever the tiers hold: the fs tier answers RETRY for a
// chunk whose existence it has not checked (tiering/fs/manager.py:215-220,
// tiering/async_lookup.py:140-161 at v0.31.0), and a RETRY defers the request
// (offloading/scheduler.py:697-703; sched/scheduler.py:956-961), so a miss costs a round too.
func (o *OffloadCache) classifyFetch(tokens []sim.TokenID, startBlock int64, prevHash string) fetchClass {
	none := fetchClass{kind: fetchNone, tier: -1}
	if len(o.secondary) == 0 {
		return none // CPU-only offload never defers (no secondary tier to fetch from)
	}
	tail, ok := o.tailOf(tokens, startBlock, prevHash)
	if !ok {
		return none
	}
	// GPU-resident and CPU-ready (reloadable) chunks extend the cached prefix synchronously
	// at admission; the chunk that ends that run decides.
	chunks, stop := o.cpuRun(tail)
	c := tail.firstChunk + chunks
	switch stop {
	case cpuHitPending:
		return fetchClass{kind: fetchPending, run: []kvkey.BlockKey{tail.key(c)}, tier: -1}
	case cpuMiss:
		probe := make([]kvkey.BlockKey, 0, tail.endChunk-c)
		for j := c; j < tail.endChunk; j++ {
			probe = append(probe, tail.key(j))
		}
		if run, tier, ok := o.gatherSecondaryRun(tail, c); ok {
			return fetchClass{kind: fetchSecondary, run: run, tier: tier, probe: probe}
		}
		if !o.allKnown(probe) {
			return fetchClass{kind: fetchProbe, tier: -1, probe: probe}
		}
	}
	return none // whole prefix cached/reloadable, or a genuine miss (recompute from here)
}

// allKnown reports whether every key's secondary existence has been probed for a live
// request (warm). A single unprobed key makes the lookup cold (+1 RETRY round).
func (o *OffloadCache) allKnown(keys []kvkey.BlockKey) bool {
	for _, k := range keys {
		if _, ok := o.lookups[k]; !ok {
			return false
		}
	}
	return true
}

// probeKeys records that owner's lookup has asked the secondary tiers about keys. The state
// lasts while some live request owns it (tiering/async_lookup.py:140-161, 226-236 at v0.31.0).
func (o *OffloadCache) probeKeys(owner string, keys []kvkey.BlockKey) {
	for _, k := range keys {
		owners, ok := o.lookups[k]
		if !ok {
			owners = make(map[string]struct{}, 1)
			o.lookups[k] = owners
		}
		if _, mine := owners[owner]; !mine {
			owners[owner] = struct{}{}
			o.lookupOwners[owner] = append(o.lookupOwners[owner], k)
		}
	}
}

// markKnown records keys as probed with no request owning them, so they stay warm (tests).
func (o *OffloadCache) markKnown(keys []kvkey.BlockKey) { o.probeKeys("", keys) }

// dropLookups forgets owner's lookups: a key no live request references loses its verdict and
// is probed again by the next request to ask (async_lookup.py:226-236; fs/manager.py:338-339).
func (o *OffloadCache) dropLookups(owner string) {
	for _, k := range o.lookupOwners[owner] {
		owners := o.lookups[k]
		delete(owners, owner)
		if len(owners) == 0 {
			delete(o.lookups, k)
		}
	}
	delete(o.lookupOwners, owner)
}

// registerDeferral records a fresh deferral episode for a NEW request whose consult
// classified as fetchSecondary or fetchPending, and kicks off the appropriate first
// action. Returns nothing; the request is left in the WaitQ (the caller returns
// "defer" to batch formation).
func (o *OffloadCache) registerDeferral(reqID string, fc fetchClass) {
	o.deferralsStarted++
	st := &deferralState{keys: fc.run, tier: fc.tier, startTick: o.clock}
	cold := !o.allKnown(fc.probe)
	o.probeKeys(reqID, fc.probe)
	switch fc.kind {
	case fetchProbe:
		// Nothing to promote: the RETRY round for the existence check, then a miss.
		st.phase = deferRetry
	case fetchPending:
		// Awaiting another request's in-flight promotion: existence is known (the
		// block is physically in CPU as -1), so no RETRY round — wait for it to land.
		st.phase = deferPromoting
		st.tier = -1
	case fetchSecondary:
		if !cold {
			// WARM: existence cached → promote now (round 1 = promote).
			if o.firePromotionRun(fc.run, fc.tier) {
				st.phase = deferPromoting
				st.hasJob = true
			} else {
				// Evictable gate refused (mechanism c) or station rejected → recompute.
				st.recompute = true
			}
		} else {
			// COLD: the tier's existence check is itself async → RETRY round first.
			st.phase = deferRetry
		}
	}
	o.deferred[reqID] = st
}

// PollDeferred advances every tracked deferred request by one scheduler round and
// returns the ids of those STILL deferred this step (phase pending, not yet
// resolved/recompute). It is called once per step at the top of batch formation,
// AFTER SetClock has applied station completions (so a promotion that finished
// since last step is already a cpuHit here — completions-before-lookups, BC-T6).
//
// Cost is O(deferred) (benchmark P). It iterates the tracked ids in SORTED order
// for all side effects (promotion Submit assigns station JobIDs in call order, and
// prepareStore evicts CPU-LRU victims in call order) so two runs with the same seed
// are byte-identical (INV-6). The returned slice's order is irrelevant (consumed as
// a membership set).
func (o *OffloadCache) PollDeferred(now int64) []string {
	o.clock = now
	if len(o.deferred) == 0 {
		return nil
	}
	ids := make([]string, 0, len(o.deferred))
	for id := range o.deferred {
		ids = append(ids, id)
	}
	sort.Strings(ids) // deterministic side-effect order (INV-6)

	still := make([]string, 0, len(ids))
	for _, id := range ids {
		st := o.deferred[id]
		if st.resolved || st.recompute {
			continue // terminal: admitted-or-recomputed on next examination, not re-advanced
		}
		switch st.phase {
		case deferRetry:
			// RETRY resolved: existence now known → promote (round 2 for cold). If a
			// concurrent request already promoted the run since we registered (its
			// blocks are now CPU-readable or a promotion is in flight), ride that
			// promotion instead of firing a duplicate (convoy dedup, BC-C2) — firing
			// would find no fresh CPU slots and wrongly recompute. Only fire when the
			// run is still genuinely CPU-absent; a gate refusal there means recompute.
			if st.tier < 0 {
				st.resolved = true // a probe that found nothing: admit and recompute
				break
			}
			switch o.awaitedState(st.keys) {
			case cpuHit:
				st.resolved = true // already landed
			case cpuHitPending:
				st.phase = deferPromoting // wait for the in-flight promotion (no own job)
			default: // cpuMiss: genuinely absent → fire this request's promotion
				if o.firePromotionRun(st.keys, st.tier) {
					st.phase = deferPromoting
					st.hasJob = true
				} else {
					st.recompute = true // evictable gate refused (locked tier) → recompute
				}
			}
		case deferLoading:
			if now >= st.readyAt {
				o.endCPULoad(st)
				if st.holds {
					// The blocks are written: cache them (_update_waiting_for_remote_kv,
					// sched/scheduler.py:3082-3085 at v0.31.0).
					o.gpu.CacheRemoteKV(id)
					st.loaded = true
				}
				st.resolved = true
			}
		case deferPromoting:
			switch o.awaitedState(st.keys) {
			case cpuHit:
				st.resolved = true // all keys CPU-readable → admit
			case cpuMiss:
				st.recompute = true // awaited block was evicted/lost before landing → recompute
			}
			// cpuHitPending: still in flight — keep waiting.
		}
		if !st.resolved && !st.recompute {
			// A load lands at readyAt however long it takes, as vLLM waits for it; the
			// backstop guards only the secondary-tier episodes.
			if st.phase != deferLoading && now-st.startTick > maxDeferTicks {
				o.endCPULoad(st)
				st.recompute = true // R19 defensive backstop (never trips for legitimate configs)
			} else {
				still = append(still, id)
			}
		}
	}
	return still
}

// awaitedState collapses the per-key CPU state of an awaited run into a single
// verdict: cpuHit iff every key is readable, cpuMiss if any key is absent
// (evicted/lost), else cpuHitPending (some promotion still in flight).
func (o *OffloadCache) awaitedState(keys []kvkey.BlockKey) cpuLookupResult {
	allHit := true
	for _, k := range keys {
		switch o.cpu.lookup(k) {
		case cpuMiss:
			return cpuMiss
		case cpuHitPending:
			allHit = false
		}
	}
	if allHit {
		return cpuHit
	}
	return cpuHitPending
}

// NextDeferralWake returns the earliest tick at which a still-pending deferral can
// change state, and false when none is pending. A deferRetry needs one more
// scheduler round (now+1). A deferLoading request changes when its CPU->GPU load lands
// (readyAt), which no backstop shortens. A deferPromoting request can only change when a
// secondary tier's station transfer completes, since evictions happen only inside
// allocation, which an idle scheduler does not run. Failing that, it changes at its
// maxDeferTicks backstop. The scheduler uses this to sleep through an idle wait instead of
// stepping 1 tick at a time.
func (o *OffloadCache) NextDeferralWake(now int64) (int64, bool) {
	wake, ok := int64(0), false
	consider := func(t int64) {
		if !ok || t < wake {
			wake, ok = t, true
		}
	}
	for _, st := range o.deferred {
		if st.resolved || st.recompute {
			continue
		}
		switch st.phase {
		case deferRetry:
			return now + 1, true
		case deferLoading:
			consider(max(st.readyAt, now+1))
		default:
			consider(st.startTick + maxDeferTicks + 1)
		}
	}
	if !ok {
		return 0, false
	}
	// Each secondary tier has its own station, and a tier may be shared with the node's
	// other engines, so any of their completions can resolve a promotion this engine awaits.
	for _, t := range o.secondary {
		if t.station == nil {
			continue
		}
		if at, inFlight := t.station.NextCompletion(); inFlight {
			consider(at)
		}
	}
	return wake, true
}

// IsDeferred reports whether the request is currently set aside and still pending
// (not yet resolved/recompute). Batch formation calls it after a failed admission
// to tell a fresh deferral (skip) apart from GPU pressure (break); a resolved
// request that hits GPU pressure returns false here so it breaks like any
// GPU-pressured request rather than being skipped.
func (o *OffloadCache) IsDeferred(id string) bool {
	st, ok := o.deferred[id]
	return ok && !st.resolved && !st.recompute
}

// ClearDeferred forgets a request that left the WaitQ by a non-admit path (timeout,
// gateway eviction, drain-redirect) so its deferral state does not leak. Idempotent.
// Any -1 CPU slots from its own in-flight promotion are left to complete normally
// (they become an evictable CPU block that a later request can use) — dropping the
// map entry only stops re-polling a request that is gone.
func (o *OffloadCache) ClearDeferred(id string) {
	if st, ok := o.deferred[id]; ok {
		o.endCPULoad(st)
		if st.holds {
			o.gpu.ReleaseKVBlocks(st.req) // vLLM frees a WAITING_FOR_REMOTE_KVS request's blocks
		}
	}
	delete(o.deferred, id)
	o.dropLookups(id) // the request is finished (aborted) for its engine
}

// DeferralsStarted returns the cumulative count of new prefill admissions that were
// set aside for a secondary-tier fetch (H3, #1591). A load-independent diagnostic
// (like promotionsFired, #1586); 0 for a run where no request ever waited on a
// secondary tier.
func (o *OffloadCache) DeferralsStarted() int64 { return o.deferralsStarted }

// ReloadsPerformed returns the cumulative count of CPU→GPU block reloads. It exposes
// the existing reloadCount diagnostic (the #1586 load-independent counter family); the
// reloaded blocks are exactly those whose prefill this PR bills as a cache hit rather
// than a recompute (#1699). 0 for a run where no CPU-resident prefix was ever reloaded.
func (o *OffloadCache) ReloadsPerformed() int64 { return o.reloadCount }

// beginCPULoad starts the CPU->GPU load of a new admission's CPU-resident prefix past its GPU
// hit and sets the request aside until it lands (INV-K7). started reports that it did; refused
// that a load was due but its blocks do not fit, which stops the admission loop as vLLM's
// failed allocate_slots does.
//
// vLLM's OffloadingConnector reports every CPU hit as an async load
// (get_num_new_matched_tokens returns (hits, True), offloading/scheduler.py:1036-1094 at
// v0.31.0). The scheduler allocates the request's blocks for its local and external tokens at
// once -- no new tokens, no lookahead, the full-sequence gate, and reserved_blocks left free
// for the in-flight prefills and one speculative step (sched/scheduler.py:1177-1220,
// 3003-3036) -- and parks it in WAITING_FOR_REMOTE_KVS, drained ahead of the waiting queue
// (:887, 1258-1281, 1343). The worker copies the chunks on its own stream, each transfer
// chained after the previous one (kv_offload/cpu/gpu_worker.py:676-724); the request is
// scheduled on the first step after the copy is seen done (gpu_worker.py:737-752;
// scheduler.py:3052-3106). So the load holds its GPU blocks for the whole transfer, is
// priced at the bytes of the blocks not already on the GPU, and queues behind earlier loads.
//
// It is a no-op without a CPULoadTime, with nothing to load, and for an episode that has
// already loaded (one load per episode, so the wait is bounded, R19).
func (o *OffloadCache) beginCPULoad(req *sim.Request, startIndex int64, cachedBlocks []int64, startBlock int64, prevHash string) (started, refused bool) {
	if o.cpuLoadTime == nil {
		return false, false
	}
	st := o.deferred[req.ID]
	if st != nil && st.phase == deferLoading {
		return false, false
	}
	keys, blocks := o.cpuLoadPlan(req.FullInputTokens(), startBlock, prevHash)
	if blocks == 0 {
		return false, false
	}
	end, ok := o.ReloadablePrefixEnd(req, startIndex)
	computed := min(end, max(req.PrefillEnd()-1, 0)) // a whole-prompt load recomputes its last token (:3095-3096)
	if !ok || computed <= startIndex {
		return false, false
	}
	if !o.gpu.allocateLoad(req, startIndex, end, computed, o.loadReserve(), cachedBlocks) {
		return false, true
	}
	o.loadBusyUntil = max(o.clock, o.loadBusyUntil) + o.cpuLoadTime(blocks*o.perBlockBytes)
	for _, k := range keys {
		o.cpu.pin(k)
		o.cpu.touchKey(k)
	}
	o.reloadCount += blocks
	if st == nil {
		st = &deferralState{tier: -1, startTick: o.clock}
		o.deferred[req.ID] = st
	}
	st.phase, st.loadKeys, st.readyAt = deferLoading, keys, o.loadBusyUntil
	st.resolved, st.recompute = false, false
	st.req, st.holds, st.loaded, st.computed, st.stored, st.gpuHit = req, true, false, computed, end, startIndex
	o.cpuLoadsStarted++
	return true, false
}

// loadReserve is vLLM's reserved_blocks for an async load: what the in-flight prefills -- the
// running requests still prefilling (SetInflightPrefills) and the loads holding blocks -- still
// need to hold their whole sequence, one speculative step each, and the new load's own step
// (sched/scheduler.py:1194-1205, 3003-3036 at v0.31.0).
func (o *OffloadCache) loadReserve() int64 {
	step := o.gpu.specStepBlocks()
	n := step
	for _, p := range o.prefilling {
		n += o.gpu.remainingBlocks(p.Req, p.Computed) + step
	}
	for _, st := range o.deferred {
		if st.holds {
			n += o.gpu.remainingBlocks(st.req, st.computed) + step
		}
	}
	return n
}

// SetInflightPrefills implements sim.LoadHoldingKVStore: the running requests still
// prefilling after this step's schedule, which a load must leave room for.
func (o *OffloadCache) SetInflightPrefills(prefills []sim.InflightPrefill) { o.prefilling = prefills }

// HoldsKV implements sim.LoadHoldingKVStore: the request's load holds GPU blocks and it has
// not been admitted.
func (o *OffloadCache) HoldsKV(id string) bool {
	st, ok := o.deferred[id]
	return ok && st.holds
}

// LoadedPrefix implements sim.LoadHoldingKVStore: for a request whose load has landed, the
// prefix it computed and the GPU hit it started from.
func (o *OffloadCache) LoadedPrefix(id string) (computed, gpuHitTokens int64, ok bool) {
	st, tracked := o.deferred[id]
	if !tracked || !st.holds || !st.loaded {
		return 0, 0, false
	}
	return st.computed, st.gpuHit, true
}

// endCPULoad releases a load's pinned chunks. Idempotent; a no-op for any other episode.
func (o *OffloadCache) endCPULoad(st *deferralState) {
	for _, k := range st.loadKeys {
		o.cpu.unpin(k)
	}
	st.loadKeys = nil
}

// cpuLoadPlan is what a load from startBlock would move, as consultAndReload walks it: the
// keys of the CPU-resident chunks in the loadable run and how many of their blocks are not on
// the GPU. A pure query; (nil, 0) when less than one chunk is loadable.
func (o *OffloadCache) cpuLoadPlan(tokens []sim.TokenID, startBlock int64, prevHash string) ([]kvkey.BlockKey, int64) {
	tail, ok := o.tailOf(tokens, startBlock, prevHash)
	if !ok {
		return nil, 0
	}
	chunks, _ := o.cpuRun(tail)
	if tail.loadableBlocks(chunks) == 0 {
		return nil, 0
	}
	var keys []kvkey.BlockKey
	var blocks int64
	for c := tail.firstChunk; c < tail.firstChunk+chunks; c++ {
		lo, hi := tail.blocks(c)
		missing := int64(0)
		for i := lo; i < hi; i++ {
			if _, onGPU := o.gpu.HashToBlock[string(tail.blockKey(i))]; !onGPU {
				missing++
			}
		}
		if missing > 0 {
			keys = append(keys, tail.key(c))
			blocks += missing
		}
	}
	return keys, blocks
}

// CPULoadsStarted is the cumulative count of new admissions that waited for a priced CPU->GPU
// load (INV-K7); 0 without a CPULoadTime.
func (o *OffloadCache) CPULoadsStarted() int64 { return o.cpuLoadsStarted }
