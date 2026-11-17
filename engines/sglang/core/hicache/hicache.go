// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/cache_controller.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/mem_cache/hicache_storage.py,
// python/sglang/srt/mem_cache/pool_host/base.py, python/sglang/srt/mem_cache/l2_transfer.py and
// python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py.

// Package hicache is the port of SGLang's HiCache framework (design §7.6): the controller that
// moves KV between the device pool, the host pool and a storage backend, the per-layer counter a
// load-back is consumed through, the host pool's base class, and the storage interface every
// backend implements. Each part is checked at its own interface against SGLang's own calls and
// answers (FID-5): the traces under testdata/golden-fid5, recorded while SGLang's scheduler runs a
// HiCache scenario (oracle/run_hicache.py recorded) or made by this repository's driver in
// orderings the scheduler does not make (oracle/run_hicache.py direct).
//
// SGLang does the controller's storage work in threads and its copies on device streams. The port
// has neither (X6): time advances in step boundaries, given to the controller by its caller
// (Controller.Boundary). At each boundary each thread body runs one iteration of its loop, and a
// copy submitted since the last boundary completes, which is the step-driven runner's rule
// (oracle/step_runner.py, design FID-1), so the port and the oracle order their decisions alike.
//
// The port holds no tensors: a host or device slot is an index, and a page of KV data is the host
// index it starts at.
//
// A Go method stands for the SGLang method whose snake_case name is its own name in CamelCase, as
// in core/cache (cache.GoName).
package hicache

// Interface names a HiCache interface as the traces name it.
type Interface string

// The HiCache interfaces, each with the SGLang class it ports.
const (
	// ControllerIface is HiCacheController (managers/cache_controller.py:295@297e54fd) with the
	// public methods HybridCacheController adds (hybrid_cache/hybrid_cache_controller.py:206).
	ControllerIface Interface = "controller"
	// LayerCounterIface is LayerDoneCounter (managers/cache_controller.py:83@297e54fd).
	LayerCounterIface Interface = "layer_counter"
	// L2TransferIface is L2TransferEngine (mem_cache/l2_transfer.py:50@297e54fd).
	L2TransferIface Interface = "l2_transfer"
	// HostPoolIface is HostKVCache (mem_cache/pool_host/base.py:157@297e54fd).
	HostPoolIface Interface = "host_pool"
	// StorageIface is HiCacheStorage (mem_cache/hicache_storage.py:293@297e54fd).
	StorageIface Interface = "storage"
)

// StorageBatchSize is STORAGE_BATCH_SIZE (hicache_storage.py:24@297e54fd): the pages one storage
// call reads or writes.
const StorageBatchSize = 128

// WritePolicies are the write policies HiCacheController accepts (cache_controller.py:371-378).
var WritePolicies = []string{"write_through", "write_through_selective", "write_back"}

// ControllerPort is HiCacheController's interface as the port answers it. The methods the port
// refuses (Refused) are listed so a trace calling them names the gap rather than fail unseen.
type ControllerPort interface {
	Write(device []int64, priority *int, nodeID int64, flush bool) ([]int64, bool)
	StartWriting()
	Load(host []int64, priority *int, nodeID int64) ([]int64, bool)
	StartLoading() (int, error)
	EvictHost(host []int64, backupOnly bool) (int, error)
	HasInflightDeviceTransfers() bool
	WriteStorage(host, tokenIDs []int64, hashValue, prefixKeys []string) int
	TerminatePrefetch(op *PrefetchOperation) (int, []string)
	PrefetchRateLimited() bool
	AppendHostMemRelease(host []int64)
	GetPrefetchSubmission(rid string) *PrefetchSubmission
	ParseStorageBackendExtraConfig(extra string) (StorageExtraConfig, error)
	PrefetchThreadFunc() error
	PrefetchIoAuxFunc() error
	PrefetchSyncThreadFunc() error
	BackupThreadFunc()
	SubmitPrefetch() error
	AllocateStorageHit() error
}

// LayerCounterPort is LayerDoneCounter's interface.
type LayerCounterPort interface {
	UpdateProducer() (int, error)
	SetConsumer(index int)
	WaitUntil(threshold int)
	Reset()
}

// L2TransferPort is L2TransferEngine's interface: the device<->host copies, which the simulator
// prices as moves (design §6.1).
type L2TransferPort interface {
	SubmitDeviceToHost(t Transfer) *Event
	SubmitHostToDevice(t Transfer, transferLayerIDMax int, onLayerDone func(layer int)) *Event
}

// HostPoolPort is HostKVCache's interface as the controller and the radix cache call it.
type HostPoolPort interface {
	Alloc(needSize int) ([]int64, bool)
	Free(indices []int64) (int, error)
	AvailableSize() int
	Clear()
	PageSize() int
	PrepareTransferIndices(host, device []int64, ioBackend string) ([]int64, []int64)
	BackupFromDeviceAllLayerPhysical(host, device []int64, ioBackend string)
	LoadToDevicePerLayerPhysical(host, device []int64, layerID int, ioBackend string)
	GetDataPage(index int64, flat bool) int64
	GetDummyFlatDataPage() int64
	SetFromFlatDataPage(index int64, page int64)
	Destroy()
}

// Storage is HiCacheStorage (hicache_storage.py:293-480@297e54fd): the key-value store of pages
// every L3 backend implements. A page's data is the host index it is read into or written from.
// Its backends are ported in core/hicache's later stages (S22, S23); the controller is checked
// against SGLang's backends' answers.
type Storage interface {
	RegisterMemPoolHost(host HostPoolPort)
	RegisterMemHostPoolV2(host HostPoolPort, name string)
	OnHostPoolsRegistered()
	BatchExistsV2(keys []string, extra *StorageExtraInfo) int
	BatchGetV2(keys []string, host []int64, extra *StorageExtraInfo) []bool
	BatchSetV2(keys []string, host []int64, extra *StorageExtraInfo) []bool
	BatchGetV1(keys []string, host []int64, extra *StorageExtraInfo) []bool
	BatchSetV1(keys []string, host []int64, extra *StorageExtraInfo) []bool
	Get(key string, page int64) bool
	BatchGet(keys []string, pages []int64) []bool
	Set(key string, page int64) bool
	BatchSet(keys []string, pages []int64) bool
	Exists(key string) bool
	BatchExists(keys []string, extra *StorageExtraInfo) int
	Clear()
	GetStats() map[string]float64
}

// StorageExtraInfo is HiCacheStorageExtraInfo (hicache_storage.py:78-82@297e54fd): the page
// hashes before a call's keys, from the sequence start.
type StorageExtraInfo struct {
	PrefixKeys []string
}

// StorageConfig is the part of HiCacheStorageConfig (hicache_storage.py:28-75@297e54fd) the
// controller reads.
type StorageConfig struct {
	TPRank, TPSize int
	IsMLAModel     bool
	DCPRank        int
}

// IsStorageWriter is HiCacheStorageConfig.is_storage_writer (hicache_storage.py:68-75): an MLA
// replica writes only from its DCP shard's first rank; every other model writes from every rank.
func (c StorageConfig) IsStorageWriter() bool {
	return !c.IsMLAModel || c.TPRank == c.DCPRank
}

// StorageExtraConfig is what HybridCacheController.parse_storage_backend_extra_config
// (hybrid_cache_controller.py:378-438@297e54fd) takes out of the backend's extra configuration.
type StorageExtraConfig struct {
	Extra                     map[string]any
	PrefetchThreshold         int
	PrefetchTimeoutBase       float64
	PrefetchTimeoutPerKiToken float64
	PassPrefixKeys            bool
}

// PrefetchSubmission is PrefetchSubmission (hybrid_cache_controller.py:200-203@297e54fd): the
// PP ticket state of a prefetch, which only pipeline-parallel buffer mode has.
type PrefetchSubmission struct {
	Decision *bool
}

// Transfer is one L2Transfer (l2_transfer.py:37-43@297e54fd) between the host pool and the
// device pool: host and device slots, in the order they are copied.
type Transfer struct {
	Host, Device []int64
}
