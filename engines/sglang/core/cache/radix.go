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
}

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
	return r
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
func (c *RadixCache) SupportsMamba() bool           { return false }

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

// Evict is evict and _evict (:690-839) for the FULL component.
func (c *RadixCache) Evict(p EvictParams) EvictResult {
	if c.cfg.Disable {
		return EvictResult{}
	}
	return c.evict(p.NumTokens, -1)
}

// evict is _evict (:805-839): the walk, then under write-back the blocking write check.
func (c *RadixCache) evict(request, target int) EvictResult {
	n := c.evictFull(request, target)
	if c.host != nil && c.IsWriteBack() {
		c.host.WritingCheck(true)
	}
	return EvictResult{NumTokensEvicted: n}
}

// EvictForAlloc is evict_for_alloc and evict_pool_shortfalls (unified_radix_cache.py:806-850@c4af57ff)
// for the FULL component of a pool of its own: the token pool's shortfall against d is what the
// walk is asked for, and the walk runs until the pool serves d.
func (c *RadixCache) EvictForAlloc(d KVDemand) EvictResult {
	if c.cfg.Disable {
		return EvictResult{}
	}
	if d.FullTokens <= 0 {
		return EvictResult{}
	}
	short := d.FullTokens - c.alloc.OwnerAvailableSize()
	if short <= 0 {
		return EvictResult{}
	}
	return c.evict(short, d.FullTokens)
}

// WindowCheckpointInterval is window_checkpoint_interval (unified_radix_cache.py:4791-4799@c4af57ff):
// a tree with no state component keeps no checkpoints.
func (c *RadixCache) WindowCheckpointInterval() int { return 0 }

// evictFull is _evict_components (:952-1058) for the FULL component; target < 0 means no
// availability target, otherwise the owner pool's size the walk runs to.
func (c *RadixCache) evictFull(request, target int) int {
	tr := Tracker{FullComponentType: 0}
	// target_reached (:1217-1223@c4af57ff) reads the owner pool's free size once.
	reached := func() bool {
		if target < 0 {
			return false
		}
		return c.alloc.OwnerAvailableSize() >= target
	}
	toTarget := target >= 0 && request > 0
	if reached() || (!toTarget && tr[FullComponentType] >= request) {
		return 0
	}
	quota := request
	if toTarget {
		quota = max(request, tr[FullComponentType]+c.core.FullEvictableSize())
	}
	c.core.EvictDeviceStart(FullComponentType, quota)
	defer c.core.EvictDeviceEnd(FullComponentType)
	for {
		step := c.core.EvictDeviceNextNode(FullComponentType, tr)
		c.ApplyEvictionStep(step)
		tr[FullComponentType] += step.Tracker[FullComponentType]
		if step.Node == nil {
			if !step.MadeProgress {
				break
			}
		} else {
			leaf := c.core.EvictDeviceLeaf(*step.Node, c.IsWriteBack())
			c.ApplyEvictionStep(leaf)
			tr[FullComponentType] += leaf.Tracker[FullComponentType]
			if leaf.BackupKV != nil {
				// Deferred demote (:1031-1052): back the leaf up, demote it only once written;
				// failing that, drop its subtree.
				if c.host.ExecuteKVBackup(leaf.BackupKV, true) > 0 {
					c.host.WritingCheck(true)
					d := c.core.Demote(*step.Node)
					c.ApplyEvictionStep(d)
					tr[FullComponentType] += d.Tracker[FullComponentType]
				} else {
					d := c.core.DropSubtreeNoHost(*step.Node)
					c.ApplyEvictionStep(d)
					tr[FullComponentType] += d.Tracker[FullComponentType]
				}
			}
		}
		if reached() {
			break
		}
	}
	return tr[FullComponentType]
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

// Checkpoint is checkpoint and checkpoint_into_tree (:1159-1336) for the FULL component and a
// request outside any streaming session.
func (c *RadixCache) Checkpoint(req *Req, upTo int) {
	if c.cfg.Disable {
		return
	}
	tokens := req.FillIDs[:min(upTo, len(req.FillIDs))]
	row := c.pool.Row(*req.KV.ReqPoolIdx)
	if len(tokens) == 0 {
		return
	}
	key := Key{Tokens: tokens, ExtraKey: req.ExtraKey, CacheSalt: req.CacheSalt, Bigram: c.tree.isEagle}
	key = key.pageAligned(c.cfg.PageSize)
	aligned := key.Len()
	values := append([]int64{}, row[:aligned]...)
	p := InsertParams{Key: key, Value: values, PrevPrefixLen: req.KV.CacheProtectedLen,
		InsertedLen: req.KV.CacheInsertedLen, Priority: req.Priority, SessionID: req.SessionID}
	res := c.Insert(p)
	req.KV.CacheInsertedLen = max(req.KV.CacheInsertedLen, aligned)

	// Split the leaf at the prompt boundary, so eviction can drop the output without the
	// prompt (:1250-1271).
	prompt := Key{Tokens: req.OriginInputIDs, ExtraKey: req.ExtraKey, CacheSalt: req.CacheSalt,
		Bigram: c.tree.isEagle}.pageAligned(c.cfg.PageSize)
	if pl := prompt.Len(); pl > 0 && pl < aligned {
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
func (c *RadixCache) MaybeHandToSession(req *Req)            {}
func (c *RadixCache) ClaimKVRow(req *Req) bool               { return false }
func (c *RadixCache) OnRelease(req *Req, checkpointed bool)  {}
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
func (c *RadixCache) TotalSize() (int, int)  { return c.tree.totalSize(), 0 }
func (c *RadixCache) TakeEvents() []Event    { return c.core.TakeEvents() }

// DFSWeightOrder is dfs_weight_order (:3972-3973): the tree core's, over node handles.
func (c *RadixCache) DFSWeightOrder(handles []NodeID) []int { return c.core.DFSWeightOrder(handles) }
