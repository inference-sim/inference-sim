package kv

import (
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
	"github.com/inference-sim/inference-sim/sim/internal/util"
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
}

// fetchKind classifies what a new request's needed prefix requires beyond its
// GPU/CPU-resident-and-reloadable head.
type fetchKind int

const (
	fetchNone      fetchKind = iota // fully cached/reloadable, or a genuine miss (recompute) — no defer
	fetchSecondary                  // a cpuMiss block is secondary-resident — defer to promote its run
	fetchPending                    // a cpuHitPending block — defer to await an in-flight promotion
)

// fetchClass is the read-only classification of a new request's consult.
type fetchClass struct {
	kind fetchKind
	run  []kvkey.BlockKey // fetchSecondary: the run to promote; fetchPending: the awaited key(s)
	tier int              // fetchSecondary: the holding tier; else -1
}

// classifyFetch walks the uncached tail [startBlock, n) exactly like
// consultAndReload but WITHOUT mutating anything, and reports whether a NEW request
// must defer. It skips GPU-resident and CPU-ready (reloadable) blocks — those
// extend the cached prefix synchronously at admission — and stops at the first:
//   - cpuHitPending  → fetchPending (await the in-flight promotion),
//   - cpuMiss that is secondary-resident → fetchSecondary (promote its run),
//   - cpuMiss absent everywhere (genuine miss) → fetchNone (recompute the tail),
//   - end of tail → fetchNone (whole prefix cached/reloadable).
func (o *OffloadCache) classifyFetch(tokens []sim.TokenID, startBlock int64, prevHash string) fetchClass {
	none := fetchClass{kind: fetchNone, tier: -1}
	if o.station == nil || len(o.secondary) == 0 {
		return none // CPU-only offload never defers (no secondary tier to fetch from)
	}
	bs := o.gpu.BlockSize()
	n := util.Len64(tokens) / bs
	if startBlock >= n {
		return none
	}
	tailKeys := kvkey.DeriveChunkKeys(kvkey.BlockKey(prevHash), tokens[startBlock*bs:], int(bs))
	for i := startBlock; i < n; i++ {
		key := tailKeys[i-startBlock]
		if _, inGPU := o.gpu.HashToBlock[string(key)]; inGPU {
			continue // already on GPU
		}
		switch o.cpu.lookup(key) {
		case cpuHit:
			continue // CPU-ready: reloadable at admission, not a fetch
		case cpuHitPending:
			return fetchClass{kind: fetchPending, run: []kvkey.BlockKey{key}, tier: -1}
		case cpuMiss:
			if run, tier, ok := o.gatherSecondaryRun(tailKeys, int(i-startBlock)); ok {
				return fetchClass{kind: fetchSecondary, run: run, tier: tier}
			}
			return none // genuine miss: recompute from here (no benefit to defer)
		}
	}
	return none
}

// allKnown reports whether every key's secondary existence is already resolved
// (warm). A single unknown key makes the whole run cold (+1 RETRY round).
func (o *OffloadCache) allKnown(keys []kvkey.BlockKey) bool {
	for _, k := range keys {
		if _, ok := o.existenceKnown[k]; !ok {
			return false
		}
	}
	return true
}

// markKnown records that the run's secondary existence has been resolved (the
// RETRY round completed). Only the promoted run's keys are marked (a documented
// second-order simplification vs vLLM's whole-prefix existence batch).
func (o *OffloadCache) markKnown(keys []kvkey.BlockKey) {
	for _, k := range keys {
		o.existenceKnown[k] = struct{}{}
	}
}

// registerDeferral records a fresh deferral episode for a NEW request whose consult
// classified as fetchSecondary or fetchPending, and kicks off the appropriate first
// action. Returns nothing; the request is left in the WaitQ (the caller returns
// "defer" to batch formation).
func (o *OffloadCache) registerDeferral(reqID string, fc fetchClass) {
	o.deferralsStarted++
	st := &deferralState{keys: fc.run, tier: fc.tier, startTick: o.clock}
	switch fc.kind {
	case fetchPending:
		// Awaiting another request's in-flight promotion: existence is known (the
		// block is physically in CPU as -1), so no RETRY round — wait for it to land.
		st.phase = deferPromoting
		st.tier = -1
	case fetchSecondary:
		if o.allKnown(fc.run) {
			// WARM: existence cached → promote now (round 1 = promote).
			o.markKnown(fc.run)
			if id, ok := o.firePromotionRun(fc.run, fc.tier); ok {
				st.phase = deferPromoting
				st.hasJob = true
				_ = id
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
			// RETRY resolved: existence now known → promote (round 2 for cold).
			o.markKnown(st.keys)
			if _, ok := o.firePromotionRun(st.keys, st.tier); ok {
				st.phase = deferPromoting
				st.hasJob = true
			} else {
				st.recompute = true
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
			if now-st.startTick > maxDeferTicks {
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
	delete(o.deferred, id)
}

// DeferralsStarted returns the cumulative count of new prefill admissions that were
// set aside for a secondary-tier fetch (H3, #1591). A load-independent diagnostic
// (like promotionsFired, #1586); 0 for a run where no request ever waited on a
// secondary tier.
func (o *OffloadCache) DeferralsStarted() int64 { return o.deferralsStarted }
