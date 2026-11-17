// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/cache_controller.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py
// and python/sglang/srt/mem_cache/l2_transfer.py.

package hicache

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
)

// ErrNotModelled is a controller operation the port refuses by name: the storage prefetch's
// scheduler-side admission (HybridCacheController.submit_prefetch, allocate_storage_hit) and the
// prefetch thread bodies' reads, which HiCache's L3 stage ports (inference-sim/blis-sglang#40).
var ErrNotModelled = errors.New("hicache: not modelled until HiCache L3 storage (inference-sim/blis-sglang#40)")

// DeviceAllocator is what the controller calls on the device's KV allocator.
type DeviceAllocator interface {
	Alloc(needSize int) ([]int64, bool)
	// SetHicacheTransferDoneEvent hands the allocator the event a write or a load finishes on
	// (allocator/base.py:251-253@297e54fd); kind is "write" or "load".
	SetHicacheTransferDoneEvent(kind string, e *Event)
}

// ControllerConfig is the controller's construction (cache_controller.py:296-390@297e54fd), on
// the KV-only path: one host pool, no sidecar pools, one rank.
type ControllerConfig struct {
	WritePolicy string
	// IOBackend is "direct", "kernel" or "kernel_ascend"; Layout is the host pool's layout.
	IOBackend string
	Layout    string
	PageSize  int
	// TransferLayerIDMax is the device pool's layers a load-back copies, one at a time.
	TransferLayerIDMax int
}

// CacheOperation is CacheOperation (cache_controller.py:113-182@297e54fd): one queued copy.
type CacheOperation struct {
	Host, Device []int64
	NodeIDs      []int64
	ID, Priority int
}

// Ack is HiCacheAck (cache_controller.py:185-197@297e54fd): a submitted copy, done when its
// finish event is.
type Ack struct {
	Start, Finish *Event
	NodeIDs       []int64
	NumTokens     int
	NumBytes      int64
}

// StorageOperation is StorageOperation (cache_controller.py:223-258@297e54fd): one backup to
// storage.
type StorageOperation struct {
	ID              int
	Host            []int64
	TokenIDs        []int64
	HashValue       []string
	PrefixKeys      []string
	CompletedTokens int
}

// PrefetchOperation is PrefetchOperation (hybrid_cache_controller.py:145-197@297e54fd), as far
// as the port takes one: its counts and whether it was told to stop.
type PrefetchOperation struct {
	StorageOperation
	RequestID       string
	StorageHitCount int
	terminated      bool
}

// Terminated is is_terminated.
func (p *PrefetchOperation) Terminated() bool { return p.terminated }

// zeroCopyBackends are the backends whose pages move by host index (batch_set_v1, batch_get_v1)
// rather than as page data (cache_controller.py:681-699@297e54fd).
var zeroCopyBackends = []string{"hf3fs", "mooncake", "npu_memcache", "eic", "nixl", "simm", "mori", "tensorcast", "seaweedfs"}

// Controller is HybridCacheController on the KV-only path (hybrid_cache_controller.py:206@297e54fd,
// over HiCacheController, cache_controller.py:295): it queues copies between the device and host
// pools, submits them as one merged transfer, keeps their acks until the radix cache reaps them,
// and hands backups to a storage backend.
type Controller struct {
	cfg     ControllerConfig
	clock   *Clock
	host    HostPoolPort
	alloc   DeviceAllocator
	l2      L2TransferPort
	counter *LayerDoneCounter

	WriteQueue, LoadQueue       []*CacheOperation
	AckWriteQueue, AckLoadQueue []*Ack
	// opCounter is CacheOperation.counter: every operation, merged ones too, takes the next id.
	opCounter int

	storage        Storage
	storageType    string
	storageConfig  StorageConfig
	EnableStorage  bool
	zeroCopy       bool
	storageCounter int
	// PrefetchThreshold, PrefetchCapacityLimit and PrefetchTokensOccupied are the prefetch's rate
	// limit (cache_controller.py:640-654): the radix cache adds and removes what it locks.
	PrefetchThreshold      int
	PrefetchCapacityLimit  int
	PrefetchTokensOccupied int
	BackupQueue            []*StorageOperation
	AckBackupQueue         []*StorageOperation
	HostMemReleaseQueue    [][]int64
	hostSize               int
}

// NewController builds the controller (cache_controller.py:296-390): its layer counter on the
// device pool's layers, and its transfer engine. A write policy SGLang does not accept is refused
// with SGLang's reason.
func NewController(cfg ControllerConfig, clock *Clock, host HostPoolPort, hostSize int, alloc DeviceAllocator, l2 L2TransferPort) (*Controller, error) {
	if !slices.Contains(WritePolicies, cfg.WritePolicy) {
		return nil, fmt.Errorf("hicache: invalid write policy: %s", cfg.WritePolicy)
	}
	if l2 == nil {
		l2 = &L2Engine{Host: host, IOBackend: cfg.IOBackend, Clock: clock}
	}
	return &Controller{cfg: cfg, clock: clock, host: host, alloc: alloc, l2: l2, hostSize: hostSize,
		counter: NewLayerDoneCounter(clock, cfg.TransferLayerIDMax)}, nil
}

// Counter is the controller's layer counter.
func (c *Controller) Counter() *LayerDoneCounter { return c.counter }

// AttachStorage attaches a storage backend (cache_controller.py:533-726,
// hybrid_cache_controller.py:311-357@297e54fd): it registers the host pool, sets the prefetch
// limits (half the host pool, in cache mode) and selects the page functions.
func (c *Controller) AttachStorage(backendType string, s Storage, cfg StorageConfig, prefetchThreshold int) error {
	if c.EnableStorage {
		return errors.New("hicache: storage backend already attached")
	}
	c.storage, c.storageType, c.storageConfig = s, backendType, cfg
	s.RegisterMemPoolHost(c.host)
	c.EnableStorage = true
	c.PrefetchThreshold = max(prefetchThreshold, c.cfg.PageSize)
	c.PrefetchCapacityLimit = int(0.5 * float64(c.hostSize))
	c.PrefetchTokensOccupied = 0
	c.zeroCopy = slices.Contains(zeroCopyBackends, backendType)
	s.RegisterMemHostPoolV2(c.host, "KV")
	s.OnHostPoolsRegistered()
	return nil
}

func (c *Controller) newOp(host, device []int64, nodeID int64, priority *int) *CacheOperation {
	op := &CacheOperation{Host: host, Device: device, NodeIDs: []int64{nodeID}, ID: c.opCounter}
	c.opCounter++
	op.Priority = op.ID
	if priority != nil {
		op.Priority = *priority
	}
	return op
}

// mergeOps is CacheOperation.merge_ops (cache_controller.py:163-182): one operation stays as it
// is; several become one new operation, their slots in queue order, the lowest priority.
func (c *Controller) mergeOps(ops []*CacheOperation) *CacheOperation {
	if len(ops) == 1 {
		return ops[0]
	}
	var host, device []int64
	var nodes []int64
	prio := ops[0].Priority
	for _, op := range ops {
		host = append(host, op.Host...)
		device = append(device, op.Device...)
		nodes = append(nodes, op.NodeIDs...)
		prio = min(prio, op.Priority)
	}
	merged := c.newOp(host, device, -1, &prio)
	merged.NodeIDs = nodes
	return merged
}

// moveIndices is move_indices (cache_controller.py:944-984): the direct backend on the
// layer_first layout copies in host order, so the host slots are sorted and the device slots
// follow them; every other case keeps the order.
func (c *Controller) moveIndices(host, device []int64) ([]int64, []int64, error) {
	switch c.cfg.IOBackend {
	case "kernel", "kernel_ascend":
		return host, device, nil
	case "direct":
		switch c.cfg.Layout {
		case "layer_first":
			idx := make([]int, len(host))
			for i := range idx {
				idx[i] = i
			}
			sort.SliceStable(idx, func(a, b int) bool { return host[idx[a]] < host[idx[b]] })
			h, d := make([]int64, len(host)), make([]int64, len(device))
			for i, j := range idx {
				h[i], d[i] = host[j], device[j]
			}
			return h, d, nil
		case "page_first_direct":
			return host, device, nil
		}
		return nil, nil, fmt.Errorf("hicache: unsupported layout %q for io backend 'direct'", c.cfg.Layout)
	}
	return nil, nil, fmt.Errorf("hicache: unsupported io backend %q", c.cfg.IOBackend)
}

// Write reserves host slots for device's KV and queues the copy, submitting it at once when
// flush is set (hybrid_cache_controller.py:555-582); false when the host pool is too full.
func (c *Controller) Write(device []int64, priority *int, nodeID int64, flush bool) ([]int64, bool) {
	host, ok := c.host.Alloc(len(device))
	if !ok {
		return nil, false
	}
	c.WriteQueue = append(c.WriteQueue, c.newOp(host, device, nodeID, priority))
	if flush {
		c.StartWriting()
	}
	return host, true
}

// StartWriting submits every queued write as one device-to-host transfer and keeps its ack
// (cache_controller.py:882-921).
func (c *Controller) StartWriting() {
	if len(c.WriteQueue) == 0 {
		return
	}
	op := c.mergeOps(c.WriteQueue)
	host, device, err := c.moveIndices(op.Host, op.Device)
	if err != nil {
		panic(err)
	}
	c.WriteQueue = nil
	finish := c.l2.SubmitDeviceToHost(Transfer{Host: host, Device: device})
	c.alloc.SetHicacheTransferDoneEvent("write", finish)
	c.AckWriteQueue = append(c.AckWriteQueue, &Ack{Finish: finish, NodeIDs: op.NodeIDs, NumTokens: len(op.Device),
		NumBytes: int64(len(op.Device)) * c.sizePerToken()})
}

func (c *Controller) sizePerToken() int64 {
	if h, ok := c.host.(interface{ SizePerToken() int64 }); ok {
		return h.SizePerToken()
	}
	return 0
}

// Load reserves device slots for host's KV and queues the copy, which StartLoading submits
// (hybrid_cache_controller.py:1008-1068); false when the device pool is too full. A load of no
// host slots queues an empty copy.
func (c *Controller) Load(host []int64, priority *int, nodeID int64) ([]int64, bool) {
	device := []int64{}
	if len(host) > 0 {
		d, ok := c.alloc.Alloc(len(host))
		if !ok {
			return nil, false
		}
		device = d
	}
	c.LoadQueue = append(c.LoadQueue, c.newOp(host, device, nodeID, priority))
	return device, true
}

// StartLoading submits every queued load as one host-to-device transfer on the layer counter's
// next event, layer by layer, and keeps its ack; -1 when nothing is queued
// (cache_controller.py:1022-1069). When the next event's last load is still in flight it fails
// as SGLang's assertion does, with the counter advanced and the loads still queued.
func (c *Controller) StartLoading() (int, error) {
	if len(c.LoadQueue) == 0 {
		return -1, nil
	}
	producer, err := c.counter.UpdateProducer()
	if err != nil {
		return 0, err
	}
	op := c.mergeOps(c.LoadQueue)
	host, device, err := c.moveIndices(op.Host, op.Device)
	if err != nil {
		return 0, err
	}
	c.LoadQueue = nil
	ev := c.counter.Events[producer]
	ev.Start.Record()
	finish := c.l2.SubmitHostToDevice(Transfer{Host: host, Device: device}, c.cfg.TransferLayerIDMax, ev.Complete)
	c.alloc.SetHicacheTransferDoneEvent("load", finish)
	c.AckLoadQueue = append(c.AckLoadQueue, &Ack{Finish: finish, NodeIDs: op.NodeIDs, NumTokens: len(op.Device),
		NumBytes: int64(len(op.Device)) * c.sizePerToken()})
	return producer, nil
}

// EvictHost frees host slots (cache_controller.py:1071-1076); only backup-only eviction exists.
func (c *Controller) EvictHost(host []int64, backupOnly bool) (int, error) {
	if !backupOnly {
		return 0, errors.New("hicache: other eviction policies are not supported yet")
	}
	if _, err := c.host.Free(host); err != nil {
		return 0, err
	}
	return len(host), nil
}

// HasInflightDeviceTransfers is whether a copy is queued or not yet reaped
// (cache_controller.py:866-873).
func (c *Controller) HasInflightDeviceTransfers() bool {
	return len(c.WriteQueue) > 0 || len(c.LoadQueue) > 0 || len(c.AckWriteQueue) > 0 || len(c.AckLoadQueue) > 0
}

// WriteStorage queues a backup of host slots to storage under the pages' hashes, for the backup
// body to write at a later boundary (hybrid_cache_controller.py:1401-1417); it returns the
// operation's id.
func (c *Controller) WriteStorage(host, tokenIDs []int64, hashValue, prefixKeys []string) int {
	op := &StorageOperation{ID: c.storageCounter, Host: host, TokenIDs: tokenIDs, HashValue: hashValue, PrefixKeys: prefixKeys}
	c.storageCounter++
	c.BackupQueue = append(c.BackupQueue, op)
	return op.ID
}

// TerminatePrefetch tells a prefetch to stop, and returns what it has completed
// (cache_controller.py:1094-1105).
func (c *Controller) TerminatePrefetch(op *PrefetchOperation) (int, []string) {
	op.terminated = true
	return op.CompletedTokens, op.HashValue
}

// PrefetchRateLimited is whether the prefetches in flight lock as much of the host pool as the
// limit allows (cache_controller.py:1265-1285, cache mode).
func (c *Controller) PrefetchRateLimited() bool {
	return c.PrefetchTokensOccupied >= c.PrefetchCapacityLimit
}

// AppendHostMemRelease queues host slots, page by page, for the radix cache to free
// (hybrid_cache_controller.py:470-496); no sidecar pools are modelled.
func (c *Controller) AppendHostMemRelease(host []int64) {
	page := c.host.PageSize()
	for i := 0; i < len(host); i += page {
		c.HostMemReleaseQueue = append(c.HostMemReleaseQueue, host[i:min(i+page, len(host))])
	}
}

// GetPrefetchSubmission is nil: the PP ticket exists only with pipeline parallelism in buffer
// mode (hybrid_cache_controller.py:1090-1102), which is not modelled.
func (c *Controller) GetPrefetchSubmission(rid string) *PrefetchSubmission { return nil }

// ParseStorageBackendExtraConfig is HybridCacheController.parse_storage_backend_extra_config
// (hybrid_cache_controller.py:378-438@297e54fd) for an inline JSON object; a configuration file
// (@path) is refused.
func (c *Controller) ParseStorageBackendExtraConfig(extra string) (StorageExtraConfig, error) {
	out := StorageExtraConfig{Extra: map[string]any{}, PrefetchThreshold: 256, PrefetchTimeoutBase: 1, PrefetchTimeoutPerKiToken: 0.25}
	if extra == "" {
		return out, nil
	}
	if strings.HasPrefix(extra, "@") {
		return out, errors.New("hicache: a storage configuration file is not read; give the configuration inline")
	}
	dec := json.NewDecoder(strings.NewReader(extra))
	dec.UseNumber()
	if err := dec.Decode(&out.Extra); err != nil {
		return out, err
	}
	pop := func(k string) (any, bool) {
		v, ok := out.Extra[k]
		delete(out.Extra, k)
		return v, ok
	}
	if v, ok := pop("prefetch_threshold"); ok {
		n, isNum := v.(json.Number)
		i, err := n.Int64()
		if !isNum || err != nil {
			return out, fmt.Errorf("prefetch_threshold must be int, got %T", v)
		}
		out.PrefetchThreshold = int(i)
	}
	for _, f := range []struct {
		key string
		dst *float64
	}{{"prefetch_timeout_base", &out.PrefetchTimeoutBase}, {"prefetch_timeout_per_ki_token", &out.PrefetchTimeoutPerKiToken}} {
		if v, ok := pop(f.key); ok {
			n, isNum := v.(json.Number)
			x, err := n.Float64()
			if !isNum || err != nil {
				return out, fmt.Errorf("%s must be number, got %T", f.key, v)
			}
			*f.dst = x
		}
	}
	if v, ok := pop("hicache_storage_pass_prefix_keys"); ok {
		b, isBool := v.(bool)
		if !isBool {
			return out, fmt.Errorf("hicache_storage_pass_prefix_keys must be bool, got %T", v)
		}
		out.PassPrefixKeys = b
	}
	return out, nil
}

// Boundary is one step boundary: every copy submitted since the last one completes, and each
// storage thread body runs one iteration, in the order SGLang starts the threads
// (cache_controller.py:478-481). It returns the bodies run, in order.
func (c *Controller) Boundary() ([]string, error) {
	c.clock.boundaries++
	if !c.EnableStorage {
		return nil, nil
	}
	ran := []string{}
	for _, b := range []struct {
		name string
		run  func() error
	}{{"prefetch_thread_func", c.PrefetchThreadFunc}, {"prefetch_io_aux_func", c.PrefetchIoAuxFunc},
		{"prefetch_sync_thread_func", c.PrefetchSyncThreadFunc},
		{"backup_thread_func", func() error { c.BackupThreadFunc(); return nil }}} {
		if err := b.run(); err != nil {
			return ran, err
		}
		ran = append(ran, b.name)
	}
	return ran, nil
}

// Tick advances the clock by one boundary without running a thread body: what a replay does
// before it replays the bodies' calls one by one.
func (c *Controller) Tick() { c.clock.boundaries++ }

// PrefetchThreadFunc is one iteration of the prefetch thread (cache_controller.py:1333-1373). The
// port's prefetch queue is filled only by SubmitPrefetch, which it refuses, so an iteration finds
// it empty and does nothing, as SGLang's does on an empty queue.
func (c *Controller) PrefetchThreadFunc() error { return nil }

// PrefetchIoAuxFunc is one iteration of the prefetch IO thread (cache_controller.py:1247-1263),
// on an empty prefetch buffer: see PrefetchThreadFunc.
func (c *Controller) PrefetchIoAuxFunc() error { return nil }

// PrefetchSyncThreadFunc is one iteration of the prefetch sync thread
// (hybrid_cache_controller.py:1940-1976), on an empty sync queue: see PrefetchThreadFunc.
func (c *Controller) PrefetchSyncThreadFunc() error { return nil }

// SubmitPrefetch is refused (ErrNotModelled).
func (c *Controller) SubmitPrefetch() error { return ErrNotModelled }

// AllocateStorageHit is refused (ErrNotModelled).
func (c *Controller) AllocateStorageHit() error { return ErrNotModelled }

// backupSkip is backup_skip (cache_controller.py:757-759): a rank that is not the storage writer
// writes no KV pages.
func (c *Controller) backupSkip() bool { return !c.storageConfig.IsStorageWriter() }

// BackupThreadFunc is one iteration of the backup thread (hybrid_cache_controller.py:1761-1776):
// it takes the next backup, writes its pages to storage batch by batch, and queues its ack for the
// radix cache.
func (c *Controller) BackupThreadFunc() {
	if len(c.BackupQueue) == 0 {
		return
	}
	op := c.BackupQueue[0]
	c.BackupQueue = c.BackupQueue[1:]
	if !c.backupSkip() {
		c.pageBackup(op)
	} else {
		// No sidecar pool backs up, so nothing of the operation is complete
		// (hybrid_cache_controller.py:1715-1737).
		op.CompletedTokens = 0
	}
	c.AckBackupQueue = append(c.AckBackupQueue, op)
}

// pageBackup is _page_backup (cache_controller.py:1418-1437): batches of StorageBatchSize pages,
// stopping at the first batch the backend does not store.
func (c *Controller) pageBackup(op *StorageOperation) {
	page := c.cfg.PageSize
	prefix := op.PrefixKeys
	for i := 0; i < len(op.HashValue); i += StorageBatchSize {
		hashes := op.HashValue[i:min(i+StorageBatchSize, len(op.HashValue))]
		host := op.Host[i*page : (i+len(hashes))*page]
		extra := &StorageExtraInfo{PrefixKeys: prefix}
		if !c.pageSet(hashes, host, extra) {
			return
		}
		if prefix != nil {
			prefix = append(append([]string{}, prefix...), hashes...)
		}
		op.CompletedTokens += page * len(hashes)
	}
}

// pageSet is _generic_page_set or _page_set_zero_copy (cache_controller.py:1401-1414).
func (c *Controller) pageSet(hashes []string, host []int64, extra *StorageExtraInfo) bool {
	if c.zeroCopy {
		for _, ok := range c.storage.BatchSetV1(hashes, host, extra) {
			if !ok {
				return false
			}
		}
		return true
	}
	pages := make([]int64, len(hashes))
	for i := range hashes {
		pages[i] = c.host.GetDataPage(host[i*c.cfg.PageSize], true)
	}
	return c.storage.BatchSet(hashes, pages)
}

// L2Engine is L2TransferEngine (l2_transfer.py:50-203@297e54fd): it submits a transfer's copies
// to the host pool and returns the event the transfer finishes on.
type L2Engine struct {
	Host      HostPoolPort
	IOBackend string
	Clock     *Clock
}

// SubmitDeviceToHost copies device KV to host, every layer at once (l2_transfer.py:132-143).
func (e *L2Engine) SubmitDeviceToHost(t Transfer) *Event {
	host, device := e.Host.PrepareTransferIndices(t.Host, t.Device, e.IOBackend)
	e.Host.BackupFromDeviceAllLayerPhysical(host, device, e.IOBackend)
	finish := NewEvent(e.Clock)
	finish.Record()
	return finish
}

// SubmitHostToDevice copies host KV to the device layer by layer, reporting each layer as it is
// submitted (l2_transfer.py:145-183).
func (e *L2Engine) SubmitHostToDevice(t Transfer, transferLayerIDMax int, onLayerDone func(layer int)) *Event {
	host, device := e.Host.PrepareTransferIndices(t.Host, t.Device, e.IOBackend)
	for layer := range transferLayerIDMax {
		e.Host.LoadToDevicePerLayerPhysical(host, device, layer, e.IOBackend)
		if onLayerDone != nil {
			onLayerDone(layer)
		}
	}
	finish := NewEvent(e.Clock)
	finish.Record()
	return finish
}
