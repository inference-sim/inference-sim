// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/unified_radix_cache.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Also derived from, at the same release: python/sglang/srt/mem_cache/base_prefix_cache.py and
// python/sglang/srt/mem_cache/common.py (free_kv_row_segments, coalesce_ranges).

package cache

import (
	"fmt"
	"math"
)

// RadixConfig is the part of CacheInitParams (cache_init_params.py:20-80@297e54fd) the FULL
// radix cache reads.
type RadixConfig struct {
	PageSize int
	// Disable is disable_radix_cache: the cache holds nothing and every match misses.
	Disable bool
	// IsEagle keys the tree on bigrams (EAGLE's draft reads pairs of tokens).
	IsEagle bool
	// KVCacheEvents queues BlockStored and BlockRemoved events; DynamoFormat namespaces their
	// hashes by extra_key and cache_salt.
	KVCacheEvents bool
	DynamoFormat  bool
	// EvictionPolicy is --radix-eviction-policy: lru (the default), lfu, fifo, mru, filo,
	// priority or slru.
	EvictionPolicy string
	// SessionRadixCache is --enable-session-radix-cache, which is refused.
	SessionRadixCache bool
	// Mamba adds the recurrent-state slot of a hybrid model to the tree, or is nil.
	Mamba *MambaTreeConfig
}

// RadixCache is UnifiedRadixCache (unified_radix_cache.py:199-3973@297e54fd) with the Python
// tree core (SGL-1) and the FULL component alone, on the device: no host tier, no storage, no
// external linker, no streaming session.
type RadixCache struct {
	cfg  RadixConfig
	tree *treeCore
	// core is the tree core as the cache calls it, through UnifiedTreeCoreInterface: every call
	// the cache makes at the tree core goes through it, in SGLang's order.
	core  TreeCore
	alloc Allocator
	pool  ReqToTokenPool
	// host is the HiCache tier init_hicache attaches, or nil for a device-only cache.
	host HostTier
	// life is MambaComponent's request side, for a tree with a state component.
	life *MambaLifecycle
}

// stateEvictor lets the lifecycle ask the cache to evict for a state slot (evict_for_alloc with
// KVDemand(state_slots=n), mamba.py:868).
type stateEvictor struct{ c *RadixCache }

func (e stateEvictor) EvictForAlloc(n int) { e.c.EvictForAlloc(KVDemand{StateSlots: n}) }

var _ PrefixCache = (*RadixCache)(nil)

// HostTier is the HiCache half of UnifiedRadixCache (unified_radix_cache.py:455-612,
// 1433-1446, 1647-1820, 3472-3530@297e54fd) as the cache's own paths call it: the backup an
// insert or a write-back eviction runs, the blocking write check write-back runs, the host
// slots a tree step frees, and the pending backup a split extends. core/hicache implements it.
type HostTier interface {
	// ExecuteKVBackup is _execute_and_commit_kv_backup: it backs the nodes up in order, and
	// returns the host slots the last one took, 0 when one failed.
	ExecuteKVBackup(nodes []NodeID, writeBack bool) int
	// WritingCheck is writing_check(write_back=True): it submits the queued backups and waits
	// for every ack.
	WritingCheck(writeBack bool)
	// FreeHostValues is FullComponent.free_host_values (components/full.py:519-523).
	FreeHostValues(values [][]int64)
	// ReplacePendingWriteThroughNode is _replace_pending_write_through_node.
	ReplacePendingWriteThroughNode(ack, old NodeID, replacement []NodeID)
}

// AttachHostTier is the end of init_hicache (unified_radix_cache.py:518-565): the tree core is
// told HiCache is on (set_hicache_enabled), and takes the write policy's threshold and
// write-back flag; the cache's own paths call h from now on.
func (c *RadixCache) AttachHostTier(h HostTier, writeBack bool, writeThroughThreshold int) {
	c.host = h
	c.core.SetHicacheEnabled()
	c.tree.ConfigureWritePolicy(writeBack, writeThroughThreshold)
}

// Core is the tree core as the cache calls it.
func (c *RadixCache) Core() TreeCore { return c.core }

// WrapCore puts wrap(core) between the cache and its tree core, so a replay can see each call
// the cache makes there.
func (c *RadixCache) WrapCore(wrap func(TreeCore) TreeCore) { c.core = wrap(c.core) }

// Allocator is the cache's device allocator.
func (c *RadixCache) Allocator() Allocator { return c.alloc }

// IsWriteBack is the cache's is_write_back property (unified_radix_cache.py:3925-3931).
func (c *RadixCache) IsWriteBack() bool { return c.tree.host.writeBack }

// ApplyActions is _apply_cache_actions and _apply_cache_action (:1374-1415).
func (c *RadixCache) ApplyActions(actions []CacheAction) {
	for _, a := range actions {
		switch a.Kind {
		case FreeDeviceKVAction:
			for _, s := range a.Indices {
				c.alloc.FreeSegment(s, 0)
			}
		case BackupKVAction:
			if c.host == nil {
				panic("cache: a backup action with no host tier")
			}
			c.host.ExecuteKVBackup(a.NodeIDs, false)
		case UpdatePendingBackupOnNodeSplitAction:
			// FullComponent.apply_component_action is a no-op for it (components/full.py:525-528).
			if a.AckID != nil && c.host != nil {
				c.host.ReplacePendingWriteThroughNode(*a.AckID, a.OldNode, []NodeID{a.NewNode, a.NewChild})
			}
		case MambaEvictExcessPathStatesAction:
			c.ApplyEvictionStep(c.core.EvictExcessPathStates(a.TailNode))
		default:
			panic(fmt.Sprintf("cache: unhandled cache action %s", a.Kind))
		}
	}
}

// ApplyEvictionStep is _apply_eviction_result (:853-858): device frees, host frees, actions.
func (c *RadixCache) ApplyEvictionStep(s EvictionStep) {
	for _, v := range s.DeviceFrees[FullComponentType] {
		// FreeComponentDeviceSlot, applied by FullComponent (components/full.py:537-552).
		c.alloc.FreeSegment(v, 0)
	}
	for _, v := range s.DeviceFrees[MambaComponentType] {
		// FreeComponentDeviceSlot of the state component: the slots go back to the state allocator.
		c.tree.mamba.cfg.Pool.Mamba.Free(v)
	}
	if hv := s.HostFrees[FullComponentType]; len(hv) > 0 {
		if c.host == nil {
			panic("cache: host slots freed with no host tier")
		}
		c.host.FreeHostValues(hv)
	}
	c.ApplyActions(s.Actions)
}

// NewRadixCache builds the cache over its allocator and request pool, as
// UnifiedRadixCache.__init__ does (:200-353), and resets it. It refuses, by name, what it does
// not model.
func NewRadixCache(cfg RadixConfig, alloc Allocator, pool ReqToTokenPool) (*RadixCache, error) {
	if cfg.SessionRadixCache {
		return nil, fmt.Errorf("cache: --enable-session-radix-cache is not modelled (%s)", RefusalIssue)
	}
	events := cfg.KVCacheEvents && !cfg.Disable
	policy := cfg.EvictionPolicy
	if cfg.Disable {
		// A disabled tree's eviction config is neither built nor checked (:242-253).
		policy = "lru"
	}
	prio, err := evictionPriority(policy)
	if err != nil {
		return nil, err
	}
	c := &RadixCache{cfg: cfg, alloc: alloc, pool: pool,
		tree: newTreeCore(cfg.PageSize, cfg.IsEagle, events, cfg.DynamoFormat)}
	c.tree.full.heap.priority = prio
	if cfg.Mamba != nil {
		if cfg.Mamba.Pool == nil {
			return nil, fmt.Errorf("cache: a tree with a state component needs the hybrid request pool")
		}
		c.tree.mamba = newMambaTree(c.tree, *cfg.Mamba)
		life, err := NewMambaLifecycle(MambaLifecycleParams{Pool: cfg.Mamba.Pool, ExtraBuffer: cfg.Mamba.Pool.extraBuffer,
			Evictor: stateEvictor{c}, CheckpointsLevel: cfg.Mamba.CheckpointsLevel})
		if err != nil {
			return nil, err
		}
		c.life = life
	}
	c.core = c.tree
	c.Reset()
	return c, nil
}

// Reset is reset and _reset_full (:416-453): a new tree, and an AllBlocksCleared event.
func (c *RadixCache) Reset() {
	c.core.Reset()
	if c.tree.events.enabled {
		// record_all_cleared (events.py:304-307) at the end of _reset_full (:451).
		c.tree.events.enqueue(Event{Type: "AllBlocksCleared"})
	}
}

// MatchPrefix is match_prefix and _match_tree (:613-642).
func (c *RadixCache) MatchPrefix(p MatchPrefixParams) MatchResult {
	if c.cfg.Disable {
		return c.tree.emptyMatch()
	}
	r := c.core.MatchPrefix(p)
	c.ApplyActions(r.Actions)
	r.Actions = nil
	if c.tree.mamba != nil {
		c.finalizeMatchInCache(p, r)
	}
	return r
}

// finalizeMatchInCache is MambaComponent.finalize_match_result_in_cache (components/mamba.py:220-265@c4af57ff):
// with copy on write, the matched state is staged into the request's own slot. The slot is taken
// now, evicting a state for it if the pool has none (the matched node pinned meanwhile), unless
// the row's binding takes it.
func (c *RadixCache) finalizeMatchInCache(p MatchPrefixParams, r MatchResult) {
	if !p.CowMamba {
		return
	}
	src := c.core.ComponentDeviceValue(MambaComponentType, r.BestMatchNode)
	if p.Req != nil && p.Req.KV.Mamba == nil {
		p.Req.KV.Mamba = &MambaKV{}
	}
	if src == nil {
		// A source from an earlier match of an unadmitted request may have been evicted and its slot
		// reused since; never carry it over.
		if p.Req != nil {
			p.Req.KV.Mamba.CowSrcIndex = nil
		}
		return
	}
	req := p.Req
	if req == nil {
		panic("cache: a copy-on-write match with no request")
	}
	m := req.KV.Mamba
	if !m.HoldsMamba() && !p.DeferCowSlot {
		slots := c.tree.mamba.cfg.Pool.Mamba
		dst, ok := slots.Alloc(1)
		if !ok {
			lock := c.IncLockRef(r.BestMatchNode)
			c.EvictForAlloc(KVDemand{StateSlots: 1})
			dst, ok = slots.Alloc(1)
			c.DecLockRef(r.BestMatchNode, lock)
			if !ok {
				panic("cache: can not alloc mamba cache")
			}
		}
		m.PoolIdx = &dst[0]
	}
	m.CowSrcIndex = &src[0]
	m.NeedsClear = false
}

// TouchPrefix is touch_prefix (:647-652): the tree walk alone.
func (c *RadixCache) TouchPrefix(k Key) {
	if c.cfg.Disable {
		return
	}
	c.ApplyActions(c.core.MatchPrefix(MatchPrefixParams{Key: k}).Actions)
}

func (c *RadixCache) SupportsFastMatchPrefix() bool { return c.core.SupportsFastMatchPrefix() }
func (c *RadixCache) SupportsPrefixSharing() bool   { return !c.cfg.Disable }
func (c *RadixCache) SupportsSWA() bool             { return false }
func (c *RadixCache) SupportsMamba() bool           { return c.tree.mamba != nil }

// Insert is insert (:661-688): the resumable walk, each step's actions applied at its barrier
// (the duplicate slots freed at the allocator as page-exact segments, a backup run), then
// whatever the walk still holds.
func (c *RadixCache) Insert(p InsertParams) InsertResult {
	if c.cfg.Disable {
		return InsertResult{}
	}
	if c.core.HasOngoingInsert() {
		panic("cache: re-entrant insert")
	}
	step := c.core.BeginInsert(p)
	for {
		c.ApplyActions(step.Actions)
		if step.Result != nil {
			break
		}
		step = c.core.ResumeInsert()
	}
	c.ApplyActions(c.core.EndInsert())
	return *step.Result
}

// Evict is evict and _evict (:690-839): the components' requests, each by its own walk.
func (c *RadixCache) Evict(p EvictParams) EvictResult {
	if c.cfg.Disable {
		return EvictResult{}
	}
	return c.evict(p, nil)
}

// evict is _evict (:1044-1090@c4af57ff): the walks, then under write-back the blocking write check.
// demand is the allocation the walk serves, per component, or nil.
func (c *RadixCache) evict(p EvictParams, demand map[ComponentType]int) EvictResult {
	tr := Tracker{FullComponentType: 0}
	request := map[ComponentType]int{FullComponentType: p.NumTokens}
	cts := []ComponentType{FullComponentType}
	if c.tree.mamba != nil {
		tr[MambaComponentType] = 0
		request[MambaComponentType] = p.MambaNum
		cts = append(cts, MambaComponentType)
	}
	c.evictComponents(cts, request, tr, demand)
	if c.host != nil && c.IsWriteBack() {
		c.host.WritingCheck(true)
	}
	return EvictResult{NumTokensEvicted: tr[FullComponentType], MambaNumEvicted: tr[MambaComponentType]}
}

// poolAvailable is _pool_available (:1029-1042): what the component's own pool can serve now.
func (c *RadixCache) poolAvailable(ct ComponentType) int {
	if ct == MambaComponentType {
		return c.tree.mamba.cfg.Pool.Mamba.AvailableSize()
	}
	return c.alloc.OwnerAvailableSize()
}

// EvictForAlloc is evict_for_alloc and evict_pool_shortfalls (unified_radix_cache.py:806-850@c4af57ff)
// for pools of their own: each component's shortfall against what d takes of it is what its walk is
// asked for, and a walk runs until its pool serves d.
func (c *RadixCache) EvictForAlloc(d KVDemand) EvictResult {
	if c.cfg.Disable {
		return EvictResult{}
	}
	demand := map[ComponentType]int{}
	if d.FullTokens > 0 {
		demand[FullComponentType] = d.FullTokens
	}
	if c.tree.mamba != nil && d.StateSlots > 0 {
		demand[MambaComponentType] = d.StateSlots
	}
	short := map[ComponentType]int{}
	for _, ct := range []ComponentType{FullComponentType, MambaComponentType} {
		n, ok := demand[ct]
		if !ok {
			continue
		}
		if s := n - c.poolAvailable(ct); s > 0 {
			short[ct] = s
		}
	}
	if len(short) == 0 {
		return EvictResult{}
	}
	served := map[ComponentType]int{}
	for ct := range short {
		served[ct] = demand[ct]
	}
	return c.evict(EvictParams{NumTokens: short[FullComponentType], MambaNum: short[MambaComponentType]}, served)
}

// WindowCheckpointInterval is window_checkpoint_interval (unified_radix_cache.py:4791-4799@c4af57ff):
// a tree with no state keeps no checkpoints; one with state keeps them on its grid.
func (c *RadixCache) WindowCheckpointInterval() int {
	if c.tree.mamba == nil {
		return 0
	}
	return c.tree.mamba.grid
}

// evictComponents is _evict_components (:1206-1257): each component in turn, FULL first. An
// allocation-driven FULL request names the allocator's shortfall, not what the walk must free, so
// its walk runs to the target; a walk skipped when its pool already serves the demand (an earlier
// component's cascade may have freed it).
func (c *RadixCache) evictComponents(cts []ComponentType, request map[ComponentType]int, tr Tracker, demand map[ComponentType]int) {
	reached := func(ct ComponentType) bool {
		d, ok := demand[ct]
		if demand == nil || !ok {
			return false
		}
		return c.poolAvailable(ct) >= d
	}
	for _, ct := range cts {
		req := request[ct]
		_, inDemand := demand[ct]
		toTarget := ct == FullComponentType && req > 0 && demand != nil && inDemand
		if reached(ct) || (!toTarget && tr[ct] >= req) {
			continue
		}
		quota := req
		if toTarget {
			quota = max(req, tr[ct]+c.core.FullEvictableSize())
		}
		c.evictWalk(ct, quota, tr, reached)
	}
}

// evictWalk is _evict_device_walk (:1276-1390) without the write-back batching: the component's
// walk, a leaf it selects evicted as a leaf, until the walk runs dry or the pool serves the demand.
func (c *RadixCache) evictWalk(ct ComponentType, quota int, tr Tracker, reached func(ComponentType) bool) {
	add := func(s EvictionStep) {
		for k, v := range s.Tracker {
			tr[k] += v
		}
	}
	c.core.EvictDeviceStart(ct, quota)
	defer c.core.EvictDeviceEnd(ct)
	for {
		step := c.core.EvictDeviceNextNode(ct, tr)
		c.ApplyEvictionStep(step)
		add(step)
		if step.Node == nil {
			if !step.MadeProgress {
				break
			}
		} else {
			leaf := c.core.EvictDeviceLeaf(*step.Node, c.IsWriteBack())
			c.ApplyEvictionStep(leaf)
			add(leaf)
			if leaf.BackupKV != nil {
				// Deferred demote (:1031-1052): back the leaf up, demote it only once written;
				// failing that, drop its subtree.
				if c.host.ExecuteKVBackup(leaf.BackupKV, true) > 0 {
					c.host.WritingCheck(true)
					d := c.core.Demote(*step.Node)
					c.ApplyEvictionStep(d)
					add(d)
				} else {
					d := c.core.DropSubtreeNoHost(*step.Node)
					c.ApplyEvictionStep(d)
					add(d)
				}
			}
		}
		if reached(ct) {
			break
		}
	}
}

// IncLockRef and DecLockRef are inc_lock_ref and dec_lock_ref (:1077-1092).
func (c *RadixCache) IncLockRef(n NodeID) LockReceipt {
	if c.cfg.Disable {
		return LockReceipt{}
	}
	return c.core.IncLockRef(n)
}

func (c *RadixCache) DecLockRef(n NodeID, r LockReceipt) {
	if c.cfg.Disable {
		return
	}
	c.core.DecLockRef(n, r)
}

// Lock and Unlock are lock and unlock (:1094-1106).
func (c *RadixCache) Lock(n NodeID) *TreeLock {
	// TreeLock(node, inc_lock_ref(node).to_dec_params()): the receipt keeps the anchor, and
	// DecLockRefParams carries no delta (base_prefix_cache.py:205-213@297e54fd).
	r := c.IncLockRef(n)
	return &TreeLock{Node: n, Receipt: LockReceipt{Node: r.Node}}
}

func (c *RadixCache) Unlock(l *TreeLock) {
	if l == nil {
		return
	}
	c.DecLockRef(l.Node, l.Receipt)
}

// Checkpoint is checkpoint and checkpoint_into_tree (:1632-1840@c4af57ff) for a request outside any
// streaming session. A tree with a state component first lets that component prepare the insert: a
// finished request donates its tracked state, a running one donates a copy of it (or holds the
// checkpoint back), and the length it lets the insert cover bounds the key.
func (c *RadixCache) Checkpoint(req *Req, upTo int) {
	finished := req.Finished
	if c.cfg.Disable {
		if finished && c.life != nil {
			c.cleanup(req, true, nil, nil)
		}
		return
	}
	tokens := req.FillIDs[:min(upTo, len(req.FillIDs))]
	p := InsertParams{PrevPrefixLen: req.KV.CacheProtectedLen, InsertedLen: req.KV.CacheInsertedLen,
		Priority: req.Priority, SessionID: req.SessionID}
	var ins MambaInsert
	effective := len(tokens)
	if c.life != nil {
		n, err := c.life.PrepareForCachingReq(req, &ins, len(tokens), finished)
		if err != nil {
			panic(err)
		}
		effective = min(effective, n)
	}
	key := Key{Tokens: tokens[:effective], ExtraKey: req.ExtraKey, CacheSalt: req.CacheSalt, Bigram: c.tree.isEagle}
	if effective <= 0 {
		c.cleanup(req, finished, nil, &ins)
		return
	}
	row := c.pool.Row(*req.KV.ReqPoolIdx)
	key = key.pageAligned(c.cfg.PageSize)
	aligned := key.Len()
	values := append([]int64{}, row[:aligned]...)
	p.Key, p.Value = key, values
	p.MambaValue, p.MambaKeepIdx, p.MambaCheckpointDeferred = ins.Value, ins.KeepIdx, ins.CheckpointDeferred
	res := c.Insert(p)
	req.KV.CacheInsertedLen = max(req.KV.CacheInsertedLen, aligned)

	// Split the leaf at the prompt boundary, so eviction can drop the output without the
	// prompt (:1750-1771): only a tree of FULL alone does.
	prompt := Key{Tokens: req.OriginInputIDs, ExtraKey: req.ExtraKey, CacheSalt: req.CacheSalt,
		Bigram: c.tree.isEagle}.pageAligned(c.cfg.PageSize)
	if pl := prompt.Len(); c.life == nil && pl > 0 && pl < aligned {
		prio := req.Priority
		if prio < math.MaxInt64 {
			prio++
		}
		c.Insert(InsertParams{Key: prompt, Value: values[:pl], PrevPrefixLen: pl, InsertedLen: pl,
			Priority: prio, SessionID: req.SessionID})
	}

	last := res.LastDeviceNode
	// path_device_indices (:3965-3966).
	indices := c.core.CollectFullDeviceIndices(last, c.core.RootNodeHandle(nil))
	if req.KV.CacheProtectedLen > len(indices)+c.cfg.PageSize-1 || res.PrefixLen > len(indices) || len(indices) != aligned {
		panic("cache: a checkpoint's insert does not cover the request's protected prefix")
	}
	c.pool.Write(*req.KV.ReqPoolIdx, req.KV.CacheProtectedLen, indices[req.KV.CacheProtectedLen:])
	c.Unlock(req.Lock)
	req.Lock = c.Lock(last)
	req.KV.CacheProtectedLen = len(indices)
	req.LastNode = &last
	c.cleanup(req, finished, &MambaInsertResult{Exist: res.MambaExist}, &ins)
}

// cleanup is each component's cleanup_after_caching_req: FULL has none; the state component frees
// what the request still holds once the tree took, or did not take, its donated state.
func (c *RadixCache) cleanup(req *Req, finished bool, res *MambaInsertResult, ins *MambaInsert) {
	if c.life == nil {
		return
	}
	if err := c.life.CleanupAfterCachingReq(req, finished, res, ins); err != nil {
		panic(err)
	}
}

// FreeKVRow is free_kv_row (base_prefix_cache.py:507-521@297e54fd) with no SWA: the row's
// slots in the coalesced ranges, freed as segments.
func (c *RadixCache) FreeKVRow(req *Req, ranges [][2]int) {
	// window_sub_pool_of reads the allocator's target window pool, whether or not it has one.
	c.alloc.TargetWindowPool()
	row := c.pool.Row(*req.KV.ReqPoolIdx)
	var segs []Segment
	for _, r := range coalesce(ranges) {
		segs = append(segs, Segment{Slots: append([]int64{}, row[r[0]:r[1]]...), StartPos: r[0]})
	}
	if len(segs) > 0 {
		c.alloc.FreeSegments(segs)
	}
}

// coalesce is coalesce_ranges (common.py:141-149@297e54fd).
func coalesce(ranges [][2]int) [][2]int {
	var out [][2]int
	for _, r := range ranges {
		if len(out) > 0 && r[0] == out[len(out)-1][1] {
			out[len(out)-1][1] = r[1]
		} else {
			out = append(out, r)
		}
	}
	return out
}

// PrefixDeviceIndices is prefix_device_indices (:3955-3963).
func (c *RadixCache) PrefixDeviceIndices(req *Req) []int64 {
	path := c.core.CollectFullDeviceIndices(*req.LastNode, c.core.RootNodeHandle(req.ExtraKey))
	return path[:min(len(path), req.PrefixLen)]
}

// The streaming-session entries: a request outside a session is never handed to one.
func (c *RadixCache) MaybeHandToSession(req *Req) {}
func (c *RadixCache) ClaimKVRow(req *Req) bool    { return false }

// OnRelease is on_release (:1620-1627@c4af57ff): a request that was not checkpointed still hands
// back what its components hold.
func (c *RadixCache) OnRelease(req *Req, checkpointed bool) {
	if checkpointed {
		return
	}
	c.cleanup(req, true, nil, nil)
}
func (c *RadixCache) SessionRecords() map[string]ReqKV       { return map[string]ReqKV{} }
func (c *RadixCache) WindowReprefillTailTokens(req *Req) int { return 0 }
func (c *RadixCache) RetainedWindowTokens() int              { return 0 }
func (c *RadixCache) RetainedWindowReqCount() int            { return 0 }
func (c *RadixCache) LeaseCheckpointStateSlots(reqs []*Req)  {}

// The host-tier entries, with no host tier.
func (c *RadixCache) FlushPendingBackups()                                {}
func (c *RadixCache) FinishStoragePrefetchAdmission(string, int, *string) {}
func (c *RadixCache) ReleaseHostResources()                               {}

func (c *RadixCache) EvictableSize() int     { return c.core.EvictableSize() }
func (c *RadixCache) ProtectedSize() int     { return c.core.FullProtectedSize() } // protected_size is full_protected_size (unified_radix_cache.py:4946@c4af57ff)
func (c *RadixCache) FullEvictableSize() int { return c.core.FullEvictableSize() }
func (c *RadixCache) FullProtectedSize() int { return c.core.FullProtectedSize() }
func (c *RadixCache) TotalSize() (int, int)  { return c.core.TotalSize() }
func (c *RadixCache) TakeEvents() []Event    { return c.core.TakeEvents() }

// DFSWeightOrder is dfs_weight_order (:3972-3973): the tree core's, over node handles.
func (c *RadixCache) DFSWeightOrder(handles []NodeID) []int { return c.core.DFSWeightOrder(handles) }
