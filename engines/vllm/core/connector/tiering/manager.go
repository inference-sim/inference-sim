// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/manager.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import (
	"fmt"
	"sort"
)

// Primary is the CPU primary tier as the tiering manager calls it: CPUPrimaryTierOffloadingManager
// (tiering/manager.py:114-173@4ba787ec), a CPUOffloadingManager with read/write aliases for the
// secondary tiers' side. The connector package supplies it (offloading.CPUPrimary).
type Primary interface {
	NumChunks() int
	Lookup(k Key, c *ReqContext) Lookup
	// LoadSource is get_load_source.
	LoadSource(k Key, c *ReqContext) string
	// PrepareLoad pins ready chunks for a GPU load and records the request's access.
	PrepareLoad(keys []Key, c *ReqContext) []int
	// PrepareRead pins ready chunks for a cascade, without recording an access
	// (_prepare_load(record_access=False), manager.py:150-158).
	PrepareRead(keys []Key, c *ReqContext) []int
	// CompleteLoad is complete_load, which is also complete_read.
	CompleteLoad(keys []Key, c *ReqContext)
	// PrepareStore is prepare_store, which is also prepare_write: the keys stored and their chunk
	// ids, or ok false when the tier cannot make room (None).
	PrepareStore(keys []Key, c *ReqContext) (stored []Key, chunkIDs []int, evicted []Key, ok bool)
	// CompleteStore is complete_store, which is also complete_write.
	CompleteStore(keys []Key, c *ReqContext, success bool)
	Touch(keys []Key, c *ReqContext)
	OnRequestFinished(c *ReqContext)
	ResetCache()
	TakeEvents() []Event
}

// pendingPromotion is PendingPromotion (manager.py:91-97).
type pendingPromotion struct {
	req      *ReqContext
	keys     []Key
	chunkIDs []int
}

// reqState is RequestState (manager.py:100-106).
type reqState struct {
	req                  *ReqContext
	pendingPrimaryStores int
	finished             bool
	requestLevelTiers    map[int]bool // nil: none
	pendingCascadeKeys   []Key
}

type jobMeta struct {
	job  *TransferJob
	tier int
}

// Manager is TieringOffloadingManager (manager.py:207-1305) without its tier-poll thread: a tier
// that sets tolerates_off_thread_polling is also served by PollTiers, which the host calls between
// steps at the times the thread would run (manager.py:347-428).
type Manager struct {
	primary Primary
	tiers   []Tier
	clock   func() float64
	metrics *tracker

	jobCounter int
	jobs       map[int]jobMeta
	// promotionsHeld is _promotions_held: set by ResetCache until OnResetFlushed.
	promotionsHeld bool
	// pendingLoads is _pending_load_submissions: tier → request id → promotion, flushed in
	// OnScheduleEnd. order keeps each map's insertion order, as Python's dicts do.
	pendingLoads     map[int]map[string]*pendingPromotion
	pendingTierOrder []int
	pendingReqOrder  map[int][]string
	processedThis    bool // _processed_jobs_this_step
	reqs             map[string]*reqState
	reqOrder         []string // _req_state's insertion order
	loadSources      map[string]map[Key]string
	parents          []*tierParent
}

// New is TieringOffloadingManager.__init__ (manager.py:222-301). clock is time.monotonic on the
// virtual clock; it stamps each job's submit time, which only back-pressure reads.
func New(primary Primary, tiers []Tier, clock func() float64) *Manager {
	if clock == nil {
		clock = func() float64 { return 0 }
	}
	m := &Manager{primary: primary, tiers: tiers, clock: clock, jobs: map[int]jobMeta{},
		pendingLoads: map[int]map[string]*pendingPromotion{}, pendingReqOrder: map[int][]string{},
		reqs: map[string]*reqState{}, loadSources: map[string]map[Key]string{}}
	for i := range tiers {
		m.parents = append(m.parents, &tierParent{m: m, origin: i})
	}
	types := make([]string, len(tiers))
	for i, t := range tiers {
		types[i] = t.Type()
	}
	chunkBytes := int64(0)
	if len(tiers) > 0 {
		chunkBytes = tiers[0].BlockBytes()
	}
	m.metrics = newTracker(types, primary.NumChunks(), chunkBytes, clock)
	return m
}

// Tiers are the secondary tiers, in configuration order.
func (m *Manager) Tiers() []Tier { return m.tiers }

// PollTiers is poll_tiers (manager.py:397-428): one round of the tier-poll thread, for the tiers
// that set tolerates_off_thread_polling.
func (m *Manager) PollTiers() {
	gate := m.processedThis
	m.processedThis = true
	var polled []int
	for i, t := range m.tiers {
		if t.OffThreadPolling() {
			polled = append(polled, i)
		}
	}
	m.processFinished(polled)
	for _, i := range polled {
		m.tiers[i].ServeExternal(m.parents[i])
	}
	m.processedThis = gate
}

func (m *Manager) nextJobID() int {
	id := m.jobCounter
	m.jobCounter++
	return id
}

// maybeProcessFinished is _maybe_process_finished_jobs (manager.py:448-458).
func (m *Manager) maybeProcessFinished() {
	if m.processedThis {
		return
	}
	m.processedThis = true
	m.processFinished(nil)
}

// completePromotion is _complete_promotion (manager.py:460-499).
func (m *Manager) completePromotion(jm jobMeta, r JobResult) {
	j := jm.job
	var ok, failed []Key
	switch {
	case r.Success:
		ok = j.Keys
	case len(r.SuccessfulKeys) > 0:
		in := map[Key]bool{}
		for _, k := range r.SuccessfulKeys {
			in[k] = true
		}
		all := map[Key]bool{}
		for _, k := range j.Keys {
			all[k] = true
		}
		for _, k := range r.SuccessfulKeys {
			if !all[k] {
				panic(fmt.Sprintf("tiering: finished promotion job %d reported unknown successful keys", r.ID))
			}
		}
		ok = r.SuccessfulKeys
		// failed_keys is set(keys) - successful_keys (manager.py:470-476), a set of bytes, which
		// complete_write walks in hash order. A failed key's chunk goes back on the free list in that
		// order, so with two or more failed keys vLLM's later chunk ids follow PYTHONHASHSEED. The port
		// walks them in the job's order; which keys are stored, hit and evicted does not depend on it.
		for _, k := range j.Keys {
			if !in[k] {
				failed = append(failed, k)
			}
		}
	default:
		failed = j.Keys
	}
	if src := m.loadSources[j.Req.ReqID]; src != nil {
		hs := m.tiers[jm.tier].HitSource()
		for _, k := range failed {
			if src[k] == hs {
				delete(src, k)
			}
		}
	}
	if len(ok) > 0 {
		m.primary.CompleteStore(ok, j.Req, true)
	}
	if len(failed) > 0 {
		m.primary.CompleteStore(failed, j.Req, false)
	}
}

// processFinished is _process_finished_jobs (manager.py:501-549); tiers nil is every tier.
func (m *Manager) processFinished(tiers []int) {
	if tiers == nil {
		tiers = make([]int, len(m.tiers))
		for i := range tiers {
			tiers[i] = i
		}
	}
	for _, i := range tiers {
		t := m.tiers[i]
		for _, r := range t.FinishedJobs() {
			jm, ok := m.jobs[r.ID]
			if !ok {
				panic(fmt.Sprintf("tiering: finished job %d from tier #%d (%s) not in jobs", r.ID, i, t.Type()))
			}
			if jm.tier != i {
				panic(fmt.Sprintf("tiering: finished job %d reported by tier #%d but belongs to tier #%d", r.ID, i, jm.tier))
			}
			delete(m.jobs, r.ID)
			m.metrics.onJobFinished(jm, r)
			if jm.job.IsPromotion {
				m.completePromotion(jm, r)
			} else {
				m.primary.CompleteLoad(jm.job.Keys, jm.job.Req)
				m.updateBackpressure(t, jm, r)
			}
		}
	}
}

// registerJob is _register_job (manager.py:440-443).
func (m *Manager) registerJob(j *TransferJob, tier int) {
	jm := jobMeta{job: j, tier: tier}
	m.jobs[j.ID] = jm
	m.metrics.onJobRegistered(jm)
}

// shouldStore is _should_store_to_tier (manager.py:551-557).
func shouldStore(t Tier, n int) bool {
	d := t.Backpressure()
	return d == nil || d.ShouldStore(n)
}

// updateBackpressure is _update_backpressure (manager.py:559-587), without its log line. The
// line's two is_under_pressure calls stay: each applies the idle decay and may clear the pressure.
func (m *Manager) updateBackpressure(t Tier, jm jobMeta, r JobResult) {
	d := t.Backpressure()
	if d == nil {
		return
	}
	d.IsUnderPressure()
	defer d.IsUnderPressure()
	if r.Success {
		n := r.TransferBytes
		if n == 0 {
			n = int64(len(jm.job.Keys)) * t.BlockBytes()
		}
		d.Update(jm.job.SubmitTime, n)
	} else {
		d.OnStoreFailed()
	}
}

// Lookup is lookup (manager.py:589-675).
func (m *Manager) Lookup(k Key, c *ReqContext) Lookup { return m.lookup(k, c, -1) }

func (m *Manager) lookup(k Key, c *ReqContext, exclude int) Lookup {
	m.maybeProcessFinished()
	primary := m.primary.Lookup(k, c)
	m.metrics.onLookup(c, k, PrimaryLabel, primary)
	switch primary {
	case Hit:
		return Hit
	case HitPending:
		return HitPending
	}
	if m.promotionsHeld {
		return Miss
	}
	retry := false
	for i, t := range m.tiers {
		if i == exclude || !c.Filter.allows(t.Medium(), t.Locality()) {
			continue
		}
		r := t.Lookup(k, c)
		if r == Hit {
			m.metrics.onLookup(c, k, m.metrics.label(i), r)
			if !m.initiatePromotion(i, k, c) {
				return Miss
			}
			return HitPending
		}
		if r == Retry {
			retry = true
		}
		m.metrics.onLookup(c, k, m.metrics.label(i), r)
	}
	if retry {
		return Retry
	}
	return Miss
}

// LoadSource is get_load_source (manager.py:677-684): the tier a promoted chunk came from, else the
// primary tier's.
func (m *Manager) LoadSource(k Key, c *ReqContext) string {
	if src, ok := m.loadSources[c.ReqID][k]; ok {
		return src
	}
	return m.primary.LoadSource(k, c)
}

// initiatePromotion is _initiate_promotion (manager.py:686-738).
func (m *Manager) initiatePromotion(tier int, k Key, c *ReqContext) bool {
	stored, ids, _, ok := m.primary.PrepareStore([]Key{k}, c)
	if !ok {
		m.metrics.onPromotionAllocationFailure()
		return false
	}
	src := m.loadSources[c.ReqID]
	if src == nil {
		src = map[Key]string{}
		m.loadSources[c.ReqID] = src
	}
	hs := m.tiers[tier].HitSource()
	for _, pk := range stored {
		src[pk] = hs
	}
	byReq := m.pendingLoads[tier]
	if byReq == nil {
		byReq = map[string]*pendingPromotion{}
		m.pendingLoads[tier] = byReq
		m.pendingTierOrder = append(m.pendingTierOrder, tier)
	}
	e := byReq[c.ReqID]
	if e == nil {
		e = &pendingPromotion{req: c}
		byReq[c.ReqID] = e
		m.pendingReqOrder[tier] = append(m.pendingReqOrder[tier], c.ReqID)
	}
	e.keys = append(e.keys, stored...)
	e.chunkIDs = append(e.chunkIDs, ids...)
	return true
}

// flushPendingPromotions is _flush_pending_promotions (manager.py:740-763).
func (m *Manager) flushPendingPromotions() {
	for _, tier := range m.pendingTierOrder {
		for _, id := range m.pendingReqOrder[tier] {
			e := m.pendingLoads[tier][id]
			j := &TransferJob{ID: m.nextJobID(), Keys: e.keys, ChunkIDs: e.chunkIDs, IsPromotion: true, Req: e.req,
				SubmitTime: m.clock()}
			m.registerJob(j, tier)
			m.tiers[tier].SubmitLoad(j)
		}
	}
	m.pendingLoads = map[int]map[string]*pendingPromotion{}
	m.pendingTierOrder = nil
	m.pendingReqOrder = map[int][]string{}
}

// PrepareLoad is prepare_load (manager.py:765-785).
func (m *Manager) PrepareLoad(keys []Key, c *ReqContext) []int { return m.primary.PrepareLoad(keys, c) }

// Touch is touch (manager.py:787-798).
func (m *Manager) Touch(keys []Key, c *ReqContext) {
	m.primary.Touch(keys, c)
	for _, t := range m.tiers {
		t.Touch(keys, c)
	}
}

// CompleteLoad is complete_load (manager.py:800-812).
func (m *Manager) CompleteLoad(keys []Key, c *ReqContext) { m.primary.CompleteLoad(keys, c) }

// PrepareStore is prepare_store (manager.py:814-878).
func (m *Manager) PrepareStore(keys []Key, c *ReqContext) (stored []Key, chunkIDs []int, evicted []Key, ok bool) {
	m.processedThis = true
	m.processFinished(nil)
	stored, chunkIDs, evicted, ok = m.primary.PrepareStore(keys, c)
	if !ok {
		return nil, nil, nil, false
	}
	st := m.reqs[c.ReqID]
	if len(stored) > 0 {
		st.pendingPrimaryStores++
	}
	if len(st.requestLevelTiers) > 0 {
		in := map[Key]bool{}
		for _, k := range stored {
			in[k] = true
		}
		var present []Key
		for _, k := range keys {
			if !in[k] {
				present = append(present, k)
			}
		}
		if len(present) > 0 {
			m.cascadeExisting(present, c, st.requestLevelTiers)
		}
	}
	return stored, chunkIDs, evicted, true
}

// sortedTiers is a set's iteration order. Python iterates a set of small ints in ascending order
// (their hashes are themselves, and the table is not resized below 5 entries in a way that
// reorders them), which is the order request_level_tiers is walked.
func sortedTiers(set map[int]bool) []int {
	out := make([]int, 0, len(set))
	for i := range set {
		out = append(out, i)
	}
	sort.Ints(out)
	return out
}

// cascadeExisting is _cascade_existing_chunks_to_request_level_tiers (manager.py:880-919).
func (m *Manager) cascadeExisting(keys []Key, c *ReqContext, tiers map[int]bool) {
	st := m.reqs[c.ReqID]
	var ready []Key
	for _, k := range keys {
		switch r := m.primary.Lookup(k, c); r {
		case Hit:
			ready = append(ready, k)
		case HitPending:
			st.pendingCascadeKeys = append(st.pendingCascadeKeys, k)
		case Miss:
		default:
			panic(fmt.Sprintf("tiering: primary tier returned %v for a cascade key", r))
		}
	}
	if len(ready) == 0 {
		return
	}
	for _, i := range sortedTiers(tiers) {
		t := m.tiers[i]
		if !shouldStore(t, len(ready)) {
			continue
		}
		t.SubmitStore(m.createStoreJob(ready, c, i))
	}
}

// flushPendingCascades is _flush_pending_cascades (manager.py:921-935).
func (m *Manager) flushPendingCascades() {
	for _, id := range append([]string(nil), m.reqOrder...) {
		st := m.reqs[id]
		if st == nil || len(st.pendingCascadeKeys) == 0 {
			continue
		}
		if len(st.requestLevelTiers) == 0 {
			panic("tiering: parked cascade keys without a request-level tier")
		}
		keys := st.pendingCascadeKeys
		st.pendingCascadeKeys = nil
		m.cascadeExisting(keys, st.req, st.requestLevelTiers)
		m.maybeFinalize(id, -1)
	}
}

// CompleteStore is complete_store (manager.py:937-983): the GPU → primary store ended, and on
// success every secondary tier gets its own cascade.
func (m *Manager) CompleteStore(keys []Key, c *ReqContext, success bool) {
	m.primary.CompleteStore(keys, c, success)
	if success {
		for i, t := range m.tiers {
			if !shouldStore(t, len(keys)) {
				continue
			}
			t.SubmitStore(m.createStoreJob(keys, c, i))
		}
	}
	st := m.reqs[c.ReqID]
	if st.pendingPrimaryStores <= 0 {
		panic("tiering: complete_store without a pending primary store")
	}
	st.pendingPrimaryStores--
	m.maybeFinalize(c.ReqID, -1)
}

// CreateStoreJob is create_store_job (manager.py:985-1011): pin the primary chunks and register a
// cascade job for tier.
func (m *Manager) CreateStoreJob(keys []Key, c *ReqContext, tier int) *TransferJob {
	return m.createStoreJob(keys, c, tier)
}

func (m *Manager) createStoreJob(keys []Key, c *ReqContext, tier int) *TransferJob {
	ids := m.primary.PrepareRead(keys, c)
	j := &TransferJob{ID: m.nextJobID(), Keys: append([]Key(nil), keys...), ChunkIDs: ids, Req: c, SubmitTime: m.clock()}
	m.registerJob(j, tier)
	return j
}

// OnNewRequest is on_new_request (manager.py:1013-1042), which also starts the primary tier's
// request state (the connector calls the manager once; TieringOffloadingManager does not forward it,
// so the primary's state starts on first use: _get_request_cache_access, cpu/manager.py:150-161).
func (m *Manager) OnNewRequest(c *ReqContext) Policy { return m.onNewRequest(c, -1) }

func (m *Manager) onNewRequest(c *ReqContext, exclude int) Policy {
	st := &reqState{req: c}
	m.metrics.onNewRequest(c)
	for i, t := range m.tiers {
		if i == exclude {
			continue
		}
		if t.OnNewRequest(c) == RequestLevel {
			if st.requestLevelTiers == nil {
				st.requestLevelTiers = map[int]bool{}
			}
			st.requestLevelTiers[i] = true
		}
	}
	if _, ok := m.reqs[c.ReqID]; !ok {
		m.reqOrder = append(m.reqOrder, c.ReqID)
	}
	m.reqs[c.ReqID] = st
	if len(st.requestLevelTiers) > 0 {
		return RequestLevel
	}
	return ChunkLevel
}

// OnRequestFinished is on_request_finished (manager.py:1044-1054).
func (m *Manager) OnRequestFinished(c *ReqContext) { m.onRequestFinished(c, -1) }

func (m *Manager) onRequestFinished(c *ReqContext, exclude int) {
	m.primary.OnRequestFinished(c)
	st := m.reqs[c.ReqID]
	if st == nil {
		panic(fmt.Sprintf("tiering: on_request_finished for request %q never seen", c.ReqID))
	}
	st.finished = true
	m.maybeFinalize(c.ReqID, exclude)
}

// maybeFinalize is _maybe_finalize_request (manager.py:1056-1080).
func (m *Manager) maybeFinalize(id string, exclude int) {
	st := m.reqs[id]
	if !st.finished || st.pendingPrimaryStores != 0 || len(st.pendingCascadeKeys) > 0 {
		return
	}
	for i, t := range m.tiers {
		if i != exclude {
			t.OnRequestFinished(st.req)
		}
	}
	m.metrics.onRequestFinished(st.req)
	delete(m.loadSources, id)
	m.dropReq(id)
}

func (m *Manager) dropReq(id string) {
	delete(m.reqs, id)
	for i, r := range m.reqOrder {
		if r == id {
			m.reqOrder = append(m.reqOrder[:i], m.reqOrder[i+1:]...)
			break
		}
	}
}

// OnScheduleEnd is on_schedule_end (manager.py:1082-1115), the last manager call of a step.
func (m *Manager) OnScheduleEnd(e ScheduleEnd) {
	m.maybeProcessFinished()
	for i, t := range m.tiers {
		t.ServeExternal(m.parents[i])
	}
	m.processedThis = false
	m.flushPendingPromotions()
	m.flushPendingCascades()
	for _, t := range m.tiers {
		t.OnScheduleEnd(e)
	}
	for _, id := range e.NewReqIDs {
		if st := m.reqs[id]; st != nil {
			m.metrics.onRequestAllocated(st.req)
		}
	}
}

// HasPendingWork is has_pending_work (manager.py:1117-1126).
func (m *Manager) HasPendingWork() bool {
	if len(m.jobs) > 0 {
		return true
	}
	for _, st := range m.reqs {
		if len(st.pendingCascadeKeys) > 0 {
			return true
		}
	}
	for _, t := range m.tiers {
		if t.HasPendingWork() {
			return true
		}
	}
	return false
}

// TakeEvents is take_events (manager.py:1128-1138).
func (m *Manager) TakeEvents() []Event {
	ev := m.primary.TakeEvents()
	for _, t := range m.tiers {
		ev = append(ev, t.TakeEvents()...)
	}
	return ev
}

// ResetCache is reset_cache (manager.py:1140-1198). Each tier's DrainJobs ends its in-flight jobs
// first, as the host's transport decides.
func (m *Manager) ResetCache() {
	for _, t := range m.tiers {
		t.DrainJobs()
	}
	m.processFinished(nil)
	if len(m.jobs) > 0 {
		panic(fmt.Sprintf("tiering: %d job(s) still in flight after the tiers drained", len(m.jobs)))
	}
	m.pendingLoads = map[int]map[string]*pendingPromotion{}
	m.pendingTierOrder = nil
	m.pendingReqOrder = map[int][]string{}
	m.loadSources = map[string]map[Key]string{}
	if !m.metrics.idle() {
		panic("tiering: transfers are still counted active after the tiers drained")
	}
	var done []string
	for _, id := range m.reqOrder {
		st := m.reqs[id]
		st.pendingPrimaryStores = 0
		st.pendingCascadeKeys = nil
		if !st.finished {
			continue
		}
		for _, t := range m.tiers {
			t.OnRequestFinished(st.req)
		}
		m.metrics.onRequestFinished(st.req)
		done = append(done, id)
	}
	m.primary.ResetCache()
	for _, id := range done {
		m.dropReq(id)
	}
	m.processedThis = false
	for _, t := range m.tiers {
		if d := t.Backpressure(); d != nil {
			d.Reset()
		}
	}
	m.promotionsHeld = len(m.tiers) > 0
}

// OnResetFlushed is on_reset_flushed (manager.py:1200-1202).
func (m *Manager) OnResetFlushed() { m.promotionsHeld = false }

// InFlight is the number of jobs between the primary and the secondary tiers.
func (m *Manager) InFlight() int { return len(m.jobs) }

// tierParent is _SecondaryTierFacingParent (manager.py:176-204).
type tierParent struct {
	m      *Manager
	origin int
}

func (p *tierParent) OnNewRequest(c *ReqContext) Policy  { return p.m.onNewRequest(c, p.origin) }
func (p *tierParent) Lookup(k Key, c *ReqContext) Lookup { return p.m.lookup(k, c, p.origin) }
func (p *tierParent) CreateStoreJob(keys []Key, c *ReqContext) *TransferJob {
	return p.m.createStoreJob(keys, c, p.origin)
}
func (p *tierParent) OnRequestFinished(c *ReqContext) { p.m.onRequestFinished(c, p.origin) }
