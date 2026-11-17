// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/base_prefix_cache.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/mem_cache/unified_cache/unified_tree_core_interface.py,
// python/sglang/srt/mem_cache/unified_cache/components/base.py, python/sglang/srt/mem_cache/allocator/base.py,
// python/sglang/srt/mem_cache/memory_pool.py (ReqToTokenPool, KVCache) and python/sglang/srt/disaggregation/kv_events.py.

// Package cache is the port of SGLang's cache framework (design §7.6): the tree-cache interface
// with the allocator and memory-pool base classes. Every tree cache, tree core, tree component,
// allocator and pool that blis-sglang models implements one of the six interfaces here, as its
// SGLang counterpart implements the abstract class of the same name, and is checked at that
// interface against SGLang's own calls and answers (FID-5): the traces under
// testdata/golden-fid5, recorded while SGLang's scheduler runs a scenario
// (oracle/run_cache_iface.py recorded) or made by this repository's driver in orderings the
// scheduler does not make (oracle/run_cache_iface.py direct).
//
// The port holds no tensors. A KV slot is an index into the token pool, a request's row is the
// slots its KV holds, and a tree node is a NodeID. Byte figures are the pools' sizing.
//
// A Go method stands for the SGLang method whose snake_case name is its own name in CamelCase
// (MatchPrefix is match_prefix). Each interface lists the SGLang methods it models, and the
// trace test fails on a recorded call to a method no Go interface models.
package cache

// Interface names a framework interface as the traces name it.
type Interface string

// The six framework interfaces, each with the SGLang abstract class it ports.
const (
	// PrefixCacheIface is BasePrefixCache (base_prefix_cache.py:390@297e54fd), with insert.
	PrefixCacheIface Interface = "prefix_cache"
	// TreeCoreIface is UnifiedTreeCoreInterface (unified_tree_core_interface.py:167@297e54fd).
	TreeCoreIface Interface = "tree_core"
	// ComponentIface is TreeComponent (components/base.py:173@297e54fd).
	ComponentIface Interface = "component"
	// AllocatorIface is BaseTokenToKVPoolAllocator (allocator/base.py:41@297e54fd).
	AllocatorIface Interface = "allocator"
	// ReqToTokenIface is ReqToTokenPool (memory_pool.py:376@297e54fd).
	ReqToTokenIface Interface = "req_to_token"
	// KVPoolIface is KVCache (memory_pool.py:2005@297e54fd).
	KVPoolIface Interface = "kv_pool"
)

// NodeID is a tree node's id (UnifiedTreeNode.id, unified_tree_core.py:144@297e54fd). The
// unified radix cache hands node ids, not nodes, to its callers.
type NodeID int64

// ComponentType is a tree component (unified_cache/component_type.py@297e54fd). Only FULL is
// modelled; SWA, MAMBA and C128 are later stages.
type ComponentType string

// FullComponentType is ComponentType.FULL, every tree's base component.
const FullComponentType ComponentType = "FULL"

// Key is a RadixKey (radix_cache.py:59-99@297e54fd): the tokens a tree is keyed on and the
// namespace that separates trees. Bigram is EAGLE's bigram view, whose length is one less than
// its tokens'.
type Key struct {
	Tokens    []int64 `json:"tokens"`
	ExtraKey  *string `json:"extra_key"`
	CacheSalt *string `json:"cache_salt"`
	Bigram    bool    `json:"bigram"`
}

// Len is len(RadixKey): the logical length.
func (k Key) Len() int {
	if k.Bigram {
		if len(k.Tokens) == 0 {
			return 0
		}
		return len(k.Tokens) - 1
	}
	return len(k.Tokens)
}

// MatchPrefixParams is MatchPrefixParams (base_prefix_cache.py:64-73@297e54fd) on the FULL path.
type MatchPrefixParams struct {
	Key Key
	// Req is the request the match is for, or nil for a match no request makes.
	Req *Req
}

// MatchResult is MatchResult (base_prefix_cache.py:269-313@297e54fd) on the FULL path.
type MatchResult struct {
	DevicePrefixLen int    `json:"device_prefix_len"`
	LastDeviceNode  NodeID `json:"last_device_node"`
	LastHostNode    NodeID `json:"last_host_node"`
	BestMatchNode   NodeID `json:"best_match_node"`
	HostHitLength   int    `json:"host_hit_length"`
	FullKVHitLength int    `json:"full_kv_hit_length"`
	// LastVisitedNode is the deepest non-root node the walk visited, or nil.
	LastVisitedNode *NodeID `json:"last_visited_node_id"`
	// CacheProtectedLen is set by a finalizer that protects less than the match, or nil.
	CacheProtectedLen *int `json:"cache_protected_len"`
	// FullDeviceMatchLen is the reusable prefix of FULL's device scan when a shorter prefix was
	// reported, or nil.
	FullDeviceMatchLen *int `json:"full_device_match_len"`
}

// InsertParams is InsertParams (base_prefix_cache.py:75-115@297e54fd) on the FULL path.
type InsertParams struct {
	Key Key
	// Value is the slots holding Key's KV, one per token.
	Value []int64
	// PrevPrefixLen is the prefix the inserting request already protects: its slots are not
	// freed as duplicates.
	PrevPrefixLen int
	// InsertedLen is what the request already inserted (req.kv.cache_inserted_len); only the
	// nodes past it count a hit.
	InsertedLen int
	Priority    int
	SessionID   *string
}

// InsertResult is InsertResult (base_prefix_cache.py:117-152@297e54fd) on the FULL path.
type InsertResult struct {
	PrefixLen      int    `json:"prefix_len"`
	TotalLen       int    `json:"total_len"`
	LastDeviceNode NodeID `json:"last_device_node"`
}

// EvictParams is EvictParams (base_prefix_cache.py:154-161@297e54fd) on the FULL path.
type EvictParams struct {
	NumTokens int
}

// EvictResult is EvictResult (base_prefix_cache.py:163-170@297e54fd) on the FULL path.
type EvictResult struct {
	NumTokensEvicted int `json:"num_tokens_evicted"`
}

// LockReceipt is the receipt a lock returns and its release replays: IncLockRefResult and
// DecLockRefParams (base_prefix_cache.py:172-240@297e54fd). Node is the anchor the lock was
// taken on; Delta is how many tokens the acquire moved from evictable to protected.
type LockReceipt struct {
	Delta *int   `json:"delta"`
	Node  NodeID `json:"node_id"`
}

// TreeLock is TreeLock (base_prefix_cache.py:242-250@297e54fd): one holder's lock on a node.
type TreeLock struct {
	Node    NodeID      `json:"node"`
	Receipt LockReceipt `json:"receipt"`
}

// ReqKV is the KV record of a request (ReqKvInfo, managers/schedule_batch.py:951-992@297e54fd)
// as the cache reads and changes it.
type ReqKV struct {
	// ReqPoolIdx is the request's row of the ReqToTokenPool, or nil when it holds none.
	ReqPoolIdx *int `json:"req_pool_idx"`
	// CacheProtectedLen: the tree cache owns [0, CacheProtectedLen) of the row.
	CacheProtectedLen int `json:"cache_protected_len"`
	CacheInsertedLen  int `json:"cache_inserted_len"`
	KVCommittedLen    int `json:"kv_committed_len"`
	KVAllocatedLen    int `json:"kv_allocated_len"`
	// Mamba is the recurrent state the request holds under a hybrid model, or nil.
	Mamba *MambaKV `json:"-"`
}

// Req is a request as the cache framework sees it: its tokens, its KV record, the node its
// prefix ends on and the lock it holds there.
type Req struct {
	ID string
	// OriginInputIDs and OutputIDs are the prompt and the tokens generated so far; FillIDs is
	// their concatenation as the cache reads it (full_untruncated_fill_ids).
	OriginInputIDs []int64
	OutputIDs      []int64
	FillIDs        []int64
	ExtraKey       *string
	CacheSalt      *string
	Priority       int
	SessionID      *string
	KV             ReqKV
	LastNode       *NodeID
	Lock           *TreeLock
	// PrefixLen is the matched prefix the request's extend starts after.
	PrefixLen int
	Finished  bool
}

// PrefixCache is BasePrefixCache (base_prefix_cache.py:390-741@297e54fd): what the scheduler
// calls on its tree cache. The lengths are tokens.
type PrefixCache interface {
	Reset()
	MatchPrefix(p MatchPrefixParams) MatchResult
	TouchPrefix(k Key)
	SupportsFastMatchPrefix() bool
	Insert(p InsertParams) InsertResult
	// Checkpoint inserts req's KV up to row position upTo, repoints its row onto the tree's
	// slots, moves its lock to the node the insert ended on and advances its protected length.
	Checkpoint(req *Req, upTo int)
	// FreeKVRow frees the slots of row positions in the half-open ranges of req's row.
	FreeKVRow(req *Req, ranges [][2]int)
	Evict(p EvictParams) EvictResult
	// EvictForAlloc evicts for an allocation short by p's counts: it stops once the allocator
	// can serve them, which may be before p's counts are evicted (unified_radix_cache.py:697-779
	// @297e54fd).
	EvictForAlloc(p EvictParams) EvictResult
	IncLockRef(n NodeID) LockReceipt
	DecLockRef(n NodeID, r LockReceipt)
	Lock(n NodeID) *TreeLock
	Unlock(l *TreeLock)
	// PrefixDeviceIndices is the slots of req's matched prefix.
	PrefixDeviceIndices(req *Req) []int64
	MaybeHandToSession(req *Req)
	ClaimKVRow(req *Req) bool
	OnRelease(req *Req, checkpointed bool)
	EvictableSize() int
	ProtectedSize() int
	FullEvictableSize() int
	FullProtectedSize() int
	// TotalSize is the FULL device slots the tree holds, and those of its other components.
	TotalSize() (int, int)
	TakeEvents() []Event
	// DFSWeightOrder is the indices of handles in the depth-first, heavier-subtree-first order
	// the scheduler's dfs-weight policy sorts its queue by (base_prefix_cache.py:347-398@297e54fd).
	DFSWeightOrder(handles []NodeID) []int
	SupportsSWA() bool
	SupportsMamba() bool
	SupportsPrefixSharing() bool
	SWAReprefillTailTokens(req *Req) int
	// SessionRecords is the KV records sessions own, by session id.
	SessionRecords() map[string]ReqKV
	FlushPendingBackups()
	FinishStoragePrefetchAdmission(rid string, fulfilledTokens int, reason *string)
	ReleaseHostResources()
}

// Tracker is the per-component evicted counts an eviction walk accumulates.
type Tracker map[ComponentType]int

// EvictionStep is BaseEvictionResult and its subclasses (unified_tree_core_interface.py:22-86
// @297e54fd): the device slots a tree step freed, for the cache to free at the allocator, and
// the step's evicted counts.
type EvictionStep struct {
	DeviceFrees map[ComponentType][][]int64
	Tracker     Tracker
	// Node is the leaf an evict_device_next_node step selects, or nil; MadeProgress also covers
	// an internal step that returned no leaf.
	Node         *NodeID
	MadeProgress bool
}

// InsertStep is InsertStepResult (unified_tree_core_interface.py:125-130@297e54fd): one step
// of a resumable insert. FreeDeviceKV is the slots the cache frees before it resumes; Result is
// set on the last step.
type InsertStep struct {
	FreeDeviceKV [][]int64
	Result       *InsertResult
}

// TreeCore is UnifiedTreeCoreInterface (unified_tree_core_interface.py:167-811@297e54fd): the
// tree's structure, its LRU lists and its sizes, which the cache drives.
type TreeCore interface {
	Reset()
	TakeEvents() []Event
	RootNodeHandle(extraKey *string) NodeID
	DFSWeightOrder(handles []NodeID) []int
	MatchPrefix(p MatchPrefixParams) MatchResult
	SupportsFastMatchPrefix() bool
	CollectFullDeviceIndices(from, until NodeID) []int64
	BeginInsert(p InsertParams) InsertStep
	ResumeInsert() InsertStep
	HasOngoingInsert() bool
	EndInsert() [][]int64
	IncLockRef(n NodeID) LockReceipt
	DecLockRef(n NodeID, r LockReceipt)
	EvictDeviceStart(c ComponentType, requestCnt int)
	EvictDeviceNextNode(c ComponentType, t Tracker) EvictionStep
	EvictDeviceLeaf(n NodeID, isWriteBack bool) EvictionStep
	EvictDeviceEnd(c ComponentType)
	EvictableSize() int
	ProtectedSize() int
	FullEvictableSize() int
	FullProtectedSize() int
}

// Component is TreeComponent (components/base.py:173-891@297e54fd): one kind of KV state the
// tree holds per node, with its own lock, LRU and eviction rules. FullComponent is the only
// one modelled.
type Component interface {
	Type() ComponentType
	ResetSessionState()
	ReclaimableValueLen(n NodeID) int
	FloorCacheLen(cacheLen int) int
	// PrepareForCachingReq returns the length the component lets a checkpoint insert, or nil
	// for no limit.
	PrepareForCachingReq(req *Req, tokenIDsLen int, isFinished bool) *int
	CleanupAfterCachingReq(req *Req, isFinished bool)
	FreeOutOfWindowSlots(req *Req, preLen int)
	CreateMatchValidator(matchDeviceOnly bool)
	FinalizeMatchResultInTreeCore(p MatchPrefixParams, r MatchResult, bestValueLen int) MatchResult
	FinalizeMatchResultInCache(p MatchPrefixParams, r MatchResult) MatchResult
	UpdateComponentOnInsertOverlap(n NodeID, prefixLen, totalPrefixLen int, value []int64) int
	CommitInsertComponentData(n NodeID, isNewLeaf bool)
	RedistributeOnNodeSplit(newParent, child NodeID)
	AcquireComponentLock(n NodeID, r LockReceipt) LockReceipt
	ReleaseComponentLock(n NodeID, r LockReceipt)
	EvictDeviceStart(requestCnt int)
	EvictDeviceNextNode(t Tracker) *NodeID
	EvictDeviceEnd()
	EvictComponent(n NodeID) EvictionStep
	DiscardDeletedSessionLeaf(n NodeID)
	// ApplyComponentAction runs an action the tree routes to the component: for FULL, freeing
	// device slots at the allocator (FreeComponentDeviceSlot, components/full.py:537-552@297e54fd).
	ApplyComponentAction(a ComponentAction)
}

// ComponentAction is a ComponentAction (unified_cache/cache_action.py@297e54fd) on the FULL
// path: device slots to free (FreeComponentDeviceSlot).
type ComponentAction struct {
	Component ComponentType
	FreeSlots [][]int64
}

// PrefillBudget is PrefillBudget (prefill_budget.py:62-168@297e54fd): what one prefill batch
// may still take from the pool, which the scheduler's admission draws down.
type PrefillBudget interface {
	RemainingTotal() int
	RemainingCurrent() int
	HasCapacity() bool
	// CanAllocatePrefill is whether pagedInput slots fit what remains.
	CanAllocatePrefill(pagedInput int) bool
	// Reserve takes a request's extend, rounded up to whole pages, one page, extraTokens and its
	// expected decode tokens from the budget (prefill_budget.py:156-168@297e54fd).
	Reserve(extendInputLen, maxNewTokens, extraTokens int)
}

// Allocator is BaseTokenToKVPoolAllocator (allocator/base.py:41-327@297e54fd): the free list
// of KV slots, in tokens or pages.
type Allocator interface {
	Size() int
	PageSize() int
	AvailableSize() int
	// Alloc takes needSize slots, or returns false when fewer are free.
	Alloc(needSize int) ([]int64, bool)
	// AllocExtend gives each request of an extend batch the slots for its new tokens, filling
	// the partly used page that ends at lastLoc first (allocator/paged.py:204-300@297e54fd); false
	// when too few pages are free.
	AllocExtend(prefixLens, seqLens []int, lastLoc []int64, extendNumTokens int) ([]int64, bool)
	// AllocDecode gives each request of a decode batch the slot for its next token (paged.py).
	AllocDecode(seqLens []int, lastLoc []int64) ([]int64, bool)
	Free(slots []int64)
	// FreeSegment frees slots that begin at row position startPos; FreeSegments frees several.
	FreeSegment(slots []int64, startPos int)
	FreeSegments(segments []Segment)
	FreeGroupBegin()
	FreeGroupEnd()
	Clear()
	// EvictToFreeTokens evicts from tree until numTokens slots are free or nothing evictable
	// remains.
	EvictToFreeTokens(tree PrefixCache, numTokens int)
	CheckDecodeCapacity(numTokens int, tree PrefixCache) bool
	RequestSlotReserve(hasReqPoolSlot bool, prefillTokens int) int
	// CreatePrefillBudget is the budget one prefill batch is admitted against.
	CreatePrefillBudget(tree PrefixCache, numMixedDecodeTokens int) PrefillBudget
	// MaxNewTokensForMemory clips generation to the empty pool's budget; false means the prompt
	// cannot fit.
	MaxNewTokensForMemory(inputTokens, maxNewTokens, tokenCapacity int, slidingWindowSize, chunkSize *int) (int, bool)
	VerifyByteAccounting() []string
	// MambaFullCacheDonor is the shared-pool donor (allocator/base.py:192-194@297e54fd), or nil
	// for an allocator with none, which every modelled allocator is.
	MambaFullCacheDonor() MambaFullCacheDonor
	// GetExtendAllocationDemand is the tokens an extend batch evicts for before a paged extend
	// (allocator/base.py:144-153@297e54fd).
	GetExtendAllocationDemand(prefixLens, seqLens []int, conservativeNumTokens, shardSize int) int
	GetKVCache() KVPool
}

// MambaFullCacheDonor is MambaFullCacheDonor (allocator/base.py:27-38@297e54fd): a shared pool
// whose FULL slots can fund Mamba state. No modelled allocator is one.
type MambaFullCacheDonor interface {
	FullTokensBeforeMambaRecheck(targetSize int) int
	PrepareMambaAllocation(targetSize int)
}

// Segment is one (slots, start position) pair of free_segments.
type Segment struct {
	Slots    []int64
	StartPos int
}

// ReqToTokenPool is ReqToTokenPool (memory_pool.py:376-492@297e54fd): one row of slot indices
// per request.
type ReqToTokenPool interface {
	Size() int
	MaxContextLen() int
	AvailableSize() int
	// Alloc gives each request without a row one, and returns every request's row, or false
	// when too few rows are free.
	Alloc(reqs []*Req) ([]int, bool)
	Free(req *Req)
	// Write sets row positions [start, start+len(values)) of row.
	Write(row, start int, values []int64)
	Row(row int) []int64
	SchedulableTokenCapacity(physicalCapacity int) int
	AllocAuxToLengths(reqPoolIndices, targetSeqLens []int)
}

// KVPool is KVCache (memory_pool.py:2005-2184@297e54fd) as sizing: what the pool holds and
// what each token of it costs. The buffers are the GPU's and are not modelled.
type KVPool interface {
	Size() int
	PageSize() int
	LayerNum() int
	// BytesPerToken is the bytes of K and V, over every layer, that one token takes.
	BytesPerToken() int64
}

// Event is a KV event as SGLang queues it (disaggregation/kv_events.py@297e54fd): a BlockStored,
// a BlockRemoved or an AllBlocksCleared.
type Event struct {
	Type            string  `json:"_t"`
	BlockHashes     []int64 `json:"block_hashes,omitempty"`
	ParentBlockHash *int64  `json:"parent_block_hash,omitempty"`
	TokenIDs        []int64 `json:"token_ids,omitempty"`
	BlockSize       int     `json:"block_size,omitempty"`
	LoraID          *string `json:"lora_id,omitempty"`
	LoraName        *string `json:"lora_name,omitempty"`
	CacheSalt       *string `json:"cache_salt,omitempty"`
	SessionID       *string `json:"session_id,omitempty"`
	Medium          string  `json:"medium,omitempty"`
}
