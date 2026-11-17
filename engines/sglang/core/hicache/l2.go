// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/unified_radix_cache.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/mem_cache/unified_cache/components/full.py
// (free_host_values) and python/sglang/srt/arg_groups/hicache_hook.py (the layout and IO checks).

package hicache

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-sglang/core/cache"
)

// StorageRefusalIssue is the open issue HiCache's L3 storage is refused with until it is
// modelled (X28).
const StorageRefusalIssue = "https://github.com/inference-sim/blis-sglang/issues/40"

// L2RefusalIssue is the open issue the host tier's other unmodelled parts are refused with: the
// buffer_only host memory mode, the kernel_ascend IO backend and the NPU layout (X28).
const L2RefusalIssue = "https://github.com/inference-sim/blis-sglang/issues/85"

// IOBackends are the --hicache-io-backend values SGLang accepts (server_args.py@297e54fd); the
// port models direct and kernel.
var IOBackends = []string{"direct", "kernel", "kernel_ascend"}

// L2Config is what init_hicache (unified_radix_cache.py:455-592@297e54fd) reads of the server
// arguments on the KV-only path, after SGLang's resolution (ResolveLayoutIO).
type L2Config struct {
	WritePolicy string
	IOBackend   string
	Layout      string
	// HostMemoryMode is --hicache-host-memory-mode: "cache" (or empty, its default).
	HostMemoryMode string
	// StorageBackend is --hicache-storage-backend, empty for none.
	StorageBackend string
}

// Check refuses, by name, what the port does not model, and what SGLang rejects.
func (c L2Config) Check() error {
	if c.HostMemoryMode != "" && c.HostMemoryMode != "cache" {
		if c.HostMemoryMode == "buffer_only" {
			return fmt.Errorf("hicache: --hicache-host-memory-mode buffer_only is not modelled (%s)", L2RefusalIssue)
		}
		return fmt.Errorf("hicache_host_memory_mode must be 'cache' or 'buffer_only', got %q", c.HostMemoryMode)
	}
	if c.StorageBackend != "" {
		return fmt.Errorf("hicache: --hicache-storage-backend %s is not modelled until HiCache L3 storage (%s)",
			c.StorageBackend, StorageRefusalIssue)
	}
	if !slices.Contains(WritePolicies, c.WritePolicy) {
		return fmt.Errorf("hicache: invalid write policy: %s", c.WritePolicy)
	}
	switch c.IOBackend {
	case "direct", "kernel":
	case "kernel_ascend":
		return fmt.Errorf("hicache: --hicache-io-backend kernel_ascend is not modelled (%s)", L2RefusalIssue)
	default:
		return fmt.Errorf("hicache: unsupported io backend %q", c.IOBackend)
	}
	if c.Layout == "page_first_kv_split" {
		return fmt.Errorf("hicache: --hicache-mem-layout page_first_kv_split is not modelled (%s)", L2RefusalIssue)
	}
	if c.Layout == "page_first_direct" {
		// The oracle cannot trace its direct copies (transfer_kv_all_layer_direct_lf_pf is a
		// CUDA kernel no manifest entry replaces yet), so it is refused until it is traced (FID-5).
		return fmt.Errorf("hicache: --hicache-mem-layout page_first_direct is not modelled until it is traced (%s)", L2RefusalIssue)
	}
	if !slices.Contains(Layouts, c.Layout) {
		return fmt.Errorf("hicache: unsupported layout: %s", c.Layout)
	}
	// The direct backend copies layer_first and page_first_direct only
	// (cache_controller.py:944-984); ResolveLayoutIO has already turned page_first into
	// page_first_direct, which is refused above.
	if c.IOBackend == "direct" && c.Layout != "layer_first" {
		return fmt.Errorf("hicache: unsupported layout %q for io backend 'direct'", c.Layout)
	}
	return nil
}

// ongoingWriteThrough is _OngoingWriteThrough: the node a backup's ack is keyed on, the lock it
// holds while the copy is in flight (none under write-back), and the nodes the ack publishes.
type ongoingWriteThrough struct {
	node    cache.NodeID
	lock    *cache.LockReceipt
	publish []cache.NodeID
}

// ongoingLoadBack is _OngoingLoadBack: the anchor, its device lock and its host lock.
type ongoingLoadBack struct {
	node       cache.NodeID
	lock, host cache.LockReceipt
}

// L2 is the HiCache half of UnifiedRadixCache on the KV-only path, in its cache host-memory
// mode with no storage backend: the radix cache with a host tier attached. Insert, match, lock
// and evict are the radix cache's; L2 adds what init_hicache attaches (the write policy, the
// backups and their acks, the load-backs and theirs, host eviction) and the scheduler's HiCache
// entries (check_hicache_events, ready_to_load_host_cache, init_load_back).
type L2 struct {
	*cache.RadixCache
	cfg    L2Config
	ctrl   ControllerPort
	queues *Controller
	host   HostPoolPort

	ongoingWT map[cache.NodeID]*ongoingWriteThrough
	ongoingLB map[cache.NodeID]ongoingLoadBack
	// loadBackThreshold is load_back_threshold (unified_radix_cache.py:575): a load-back of
	// fewer tokens is not worth the copy.
	loadBackThreshold int
}

var _ cache.HostTier = (*L2)(nil)

// NewL2 is init_hicache (unified_radix_cache.py:455-592@297e54fd) on rc, with the controller
// attach_hybrid_pool_to_unified_cache built over host: the tree core is told HiCache is on, and
// the write policy sets the write-through threshold (1 for write_through, 2 otherwise) and the
// write-back flag. What the port does not model is refused by name.
func NewL2(rc *cache.RadixCache, cfg L2Config, ctrl *Controller, host HostPoolPort) (*L2, error) {
	if err := cfg.Check(); err != nil {
		return nil, err
	}
	if ctrl.cfg.WritePolicy != cfg.WritePolicy {
		return nil, fmt.Errorf("hicache: the controller's write policy %s is not the cache's %s", ctrl.cfg.WritePolicy, cfg.WritePolicy)
	}
	l := &L2{RadixCache: rc, cfg: cfg, ctrl: ctrl, queues: ctrl, host: host, loadBackThreshold: 10,
		ongoingWT: map[cache.NodeID]*ongoingWriteThrough{}, ongoingLB: map[cache.NodeID]ongoingLoadBack{}}
	threshold := 2
	if cfg.WritePolicy == "write_through" {
		threshold = 1
	}
	rc.AttachHostTier(l, cfg.WritePolicy == "write_back", threshold)
	return l, nil
}

// WrapController puts wrap(controller) between the cache and its controller, and WrapHost the
// same for the host pool, so a replay can see each call the cache makes there.
func (l *L2) WrapController(wrap func(ControllerPort) ControllerPort) { l.ctrl = wrap(l.ctrl) }

// WrapHost puts wrap(host pool) between the cache and its host pool.
func (l *L2) WrapHost(wrap func(HostPoolPort) HostPoolPort) { l.host = wrap(l.host) }

// Controller is the cache's controller.
func (l *L2) Controller() *Controller { return l.queues }

// Reset is reset and _reset_full (unified_radix_cache.py:416-453): a new tree, and the
// controller's queues and the host pool emptied.
func (l *L2) Reset() {
	l.RadixCache.Reset()
	l.ongoingWT = map[cache.NodeID]*ongoingWriteThrough{}
	l.ongoingLB = map[cache.NodeID]ongoingLoadBack{}
	q := l.queues
	q.WriteQueue, q.LoadQueue, q.AckWriteQueue, q.AckLoadQueue = nil, nil, nil, nil
	q.counter.Reset()
	l.host.Clear()
}

// receipt is IncLockRefResult.to_dec_params: the anchor and the host boundary, no delta.
func receipt(r cache.LockReceipt) cache.LockReceipt {
	return cache.LockReceipt{Node: r.Node, HostUUID: r.HostUUID}
}

// IncHostLockRef and DecHostLockRef are inc_host_lock_ref and dec_host_lock_ref (:1126-1136).
func (l *L2) IncHostLockRef(n cache.NodeID) cache.LockReceipt { return l.Core().IncHostLockRef(n) }

// DecHostLockRef releases a host lock.
func (l *L2) DecHostLockRef(n cache.NodeID, r cache.LockReceipt) { l.Core().DecHostLockRef(n, r) }

// ExecuteKVBackup is _execute_and_commit_kv_backup and _execute_kv_backup (:1681-1762): each
// node's device slots are queued for a copy to host slots the controller reserves (host leaves
// evicted first when the host pool is short), committed to the node, locked under write-through
// until the ack, and marked pending under the node's ack.
func (l *L2) ExecuteKVBackup(nodes []cache.NodeID, writeBack bool) int {
	core := l.Core()
	written := 0
	for _, id := range nodes {
		device := core.BuildBackupSpec(id)
		if len(device) == 0 {
			continue
		}
		if avail := l.host.AvailableSize(); avail < len(device) {
			need := len(device) - avail
			if l.EvictHost(need) < need {
				return 0
			}
		}
		core.RotationPhaseOf(id)
		host, ok := l.ctrl.Write(device, nil, int64(id), false)
		if !ok {
			return 0
		}
		core.CommitBackup(id, host)
		var lock *cache.LockReceipt
		if !writeBack {
			r := receipt(l.IncLockRef(id))
			lock = &r
		}
		publish := core.MarkWriteThroughPending([]cache.NodeID{id}, id)
		l.ongoingWT[id] = &ongoingWriteThrough{node: id, lock: lock, publish: publish}
		written = len(host)
	}
	return written
}

// ReplacePendingWriteThroughNode is _replace_pending_write_through_node (:1777-1801): a split
// node's ack publishes both halves.
func (l *L2) ReplacePendingWriteThroughNode(ack, old cache.NodeID, replacement []cache.NodeID) {
	p := l.ongoingWT[ack]
	if p == nil || !slices.Contains(p.publish, old) {
		return
	}
	var out []cache.NodeID
	for _, id := range p.publish {
		if id == old {
			out = append(out, replacement...)
		} else {
			out = append(out, id)
		}
	}
	p.publish = out
}

// finishWriteThroughAck is _finish_write_through_ack (:1803-1818).
func (l *L2) finishWriteThroughAck(ack cache.NodeID) {
	p := l.ongoingWT[ack]
	delete(l.ongoingWT, ack)
	l.Core().FinishWriteThrough(p.publish, ack)
	if p.lock != nil {
		l.DecLockRef(p.node, *p.lock)
	}
}

// FreeHostValues is FullComponent.free_host_values (components/full.py:519-523): each value
// back to the host pool.
func (l *L2) FreeHostValues(values [][]int64) {
	for _, v := range values {
		if _, err := l.host.Free(v); err != nil {
			panic(err)
		}
	}
}

// EvictHost is evict_host (:1433-1445): host leaves evicted until num host slots are free.
func (l *L2) EvictHost(num int) int {
	s := l.Core().DriveHostEviction(cache.FullComponentType, num)
	l.ApplyEvictionStep(s)
	return s.Tracker[cache.FullComponentType]
}

func readyAcks(q []*Ack) int {
	n := 0
	for _, a := range q {
		if !a.Finish.Query() {
			break
		}
		n++
	}
	return n
}

// WritingCheck is writing_check (:3472-3513). Under write-back it blocks: it submits what is
// queued and waits for every ack. Otherwise it reaps the acks whose copies are done, in order.
func (l *L2) WritingCheck(writeBack bool) {
	q := l.queues
	if writeBack {
		l.ctrl.StartWriting()
		for len(l.ongoingWT) > 0 {
			for _, ack := range q.AckWriteQueue {
				ack.Finish.Synchronize()
				for _, id := range ack.NodeIDs {
					if _, ok := l.ongoingWT[cache.NodeID(id)]; ok {
						l.finishWriteThroughAck(cache.NodeID(id))
					}
				}
			}
			q.AckWriteQueue = nil
			if len(l.ongoingWT) != 0 {
				panic("hicache: a write-back backup has no ack")
			}
		}
		return
	}
	l.writingCheck(readyAcks(q.AckWriteQueue))
}

func (l *L2) writingCheck(n int) {
	q := l.queues
	for ; n > 0; n-- {
		ack := q.AckWriteQueue[0]
		q.AckWriteQueue = q.AckWriteQueue[1:]
		ack.Finish.Synchronize()
		for _, id := range ack.NodeIDs {
			l.finishWriteThroughAck(cache.NodeID(id))
		}
	}
}

// LoadingCheck is loading_check (:3530-3584): the load-backs whose copies are done release
// their locks, and their nodes' host copies stay as duplicates.
func (l *L2) LoadingCheck() { l.loadingCheck(readyAcks(l.queues.AckLoadQueue)) }

func (l *L2) loadingCheck(n int) {
	q := l.queues
	for ; n > 0; n-- {
		ack := q.AckLoadQueue[0]
		q.AckLoadQueue = q.AckLoadQueue[1:]
		ack.Finish.Synchronize()
		for _, id := range ack.NodeIDs {
			lb := l.ongoingLB[cache.NodeID(id)]
			delete(l.ongoingLB, cache.NodeID(id))
			l.DecLockRef(lb.node, lb.lock)
			l.DecHostLockRef(lb.node, lb.host)
			l.Core().FinishLoadBack(lb.node)
		}
	}
}

// CheckHicacheEvents is check_hicache_events (:3651-3707), once per scheduler step: the pending
// backups are submitted, then the backups and load-backs whose copies are done are reaped. One
// rank is modelled, so the counts need no reduction.
func (l *L2) CheckHicacheEvents() {
	l.FlushPendingBackups()
	w, ld := readyAcks(l.queues.AckWriteQueue), readyAcks(l.queues.AckLoadQueue)
	l.writingCheck(w)
	l.loadingCheck(ld)
}

// FlushPendingBackups is flush_pending_backups (:3709-3713): the queued backups as one copy.
func (l *L2) FlushPendingBackups() { l.ctrl.StartWriting() }

// ReadyToLoadHostCache is ready_to_load_host_cache (:3715-3721): the queued load-backs start, on
// the layer counter's next event, whose index it returns (-1 when none is queued).
func (l *L2) ReadyToLoadHostCache() (int, error) { return l.ctrl.StartLoading() }

// LoadBackIsCacheOwned is load_back_is_cache_owned (:3626-3629): with no external linker, a
// load-back's slots belong to the tree.
func (l *L2) LoadBackIsCacheOwned() bool { return true }

// ReleaseHostResources is release_host_resources (:603-607): the host pool is destroyed.
func (l *L2) ReleaseHostResources() { l.host.Destroy() }

// LoadBack is load_back and _load_back_transfers (:1820-1937): the evicted path from node up is
// copied back to device slots the controller reserves, the device evicted first when it is
// short. The path stays locked on both tiers until loading_check reaps its ack. False when the
// path is too short to be worth it, does not fit memQuota, or finds no device room.
func (l *L2) LoadBack(node cache.NodeID, memQuota *int) bool {
	core := l.Core()
	hostLock := receipt(l.IncHostLockRef(node))
	res := l.IncLockRef(node)
	lock := receipt(res)
	fail := func() bool {
		l.DecLockRef(node, lock)
		l.DecHostLockRef(node, hostLock)
		return false
	}
	spec := core.BuildLoadBackSpec(node)
	kv := len(spec.Host)
	delta := 0
	if res.Delta != nil {
		delta = *res.Delta
	}
	if kv < max(1, l.loadBackThreshold) || (memQuota != nil && kv+delta > *memQuota) {
		return fail()
	}
	alloc := l.Allocator()
	if avail := alloc.AvailableSize(); avail < kv {
		l.EvictForAlloc(cache.EvictParams{NumTokens: kv - avail})
		if alloc.AvailableSize() < kv {
			return fail()
		}
	}
	device, ok := l.ctrl.Load(spec.Host, nil, int64(node))
	l.DecLockRef(node, lock)
	if !ok {
		l.DecHostLockRef(node, hostLock)
		return false
	}
	l.ApplyActions(core.CommitLoadBack(node, device, spec))
	l.ongoingLB[node] = ongoingLoadBack{node: node, lock: receipt(l.IncLockRef(node)), host: hostLock}
	return true
}

// InitLoadBackParams is InitLoadBackParams (base_prefix_cache.py@297e54fd).
type InitLoadBackParams struct {
	BestMatchNode cache.NodeID
	HostHitLength int
	MemQuota      *int
	Req           *cache.Req
}

// InitLoadBack is init_load_back (:3586-3624): when the best match is on host, it is loaded
// back, and the tokens loaded and the new device anchor are returned; otherwise none, and the
// request's own anchor.
func (l *L2) InitLoadBack(p InitLoadBackParams) (int, cache.NodeID) {
	last := *p.Req.LastNode
	core := l.Core()
	if core.IsFullDeviceEvicted(p.BestMatchNode) || p.HostHitLength > 0 {
		if l.LoadBack(p.BestMatchNode, p.MemQuota) {
			idx := core.CollectFullDeviceIndices(p.BestMatchNode, last)
			if len(idx) == 0 {
				return 0, last
			}
			return len(idx), p.BestMatchNode
		}
	}
	return 0, last
}
