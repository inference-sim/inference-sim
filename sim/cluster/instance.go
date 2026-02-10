package cluster

import (
	"github.com/inference-sim/inference-sim/sim"
)

// InstanceSimulator wraps existing sim package components to represent a single vLLM instance
type InstanceSimulator struct {
	// Identity
	ID       InstanceID
	PoolType PoolType

	// Configuration
	Config *DeploymentConfig

	// State (wraps existing sim components)
	WaitQueue    *sim.WaitQueue
	RunningBatch *sim.Batch
	KVCache      *sim.KVCacheState

	// Simulation clock
	Clock int64

	// Metrics tracking
	CompletedRequests  int
	TotalInputTokens   int64
	TotalOutputTokens  int64
	PeakWaitQueueDepth int
	PeakBatchSize      int
}

// NewInstanceSimulator creates a new instance simulator
func NewInstanceSimulator(id InstanceID, poolType PoolType, config *DeploymentConfig, totalKVBlocks int64, blockSize int64) *InstanceSimulator {
	return &InstanceSimulator{
		ID:           id,
		PoolType:     poolType,
		Config:       config,
		WaitQueue:    &sim.WaitQueue{},
		RunningBatch: nil,
		KVCache:      sim.NewKVCacheState(totalKVBlocks, blockSize),
		Clock:        0,
	}
}

// EnqueueRequest adds a request to the wait queue
func (inst *InstanceSimulator) EnqueueRequest(req *Request) {
	// Convert cluster.Request to sim.Request for internal use
	simReq := &sim.Request{
		ID:           req.ID,
		InputTokens:  make([]int, req.PromptTokens),
		OutputTokens: make([]int, req.OutputTokens),
		State:        "queued",
		ArrivalTime:  req.ArrivalTime,
	}

	inst.WaitQueue.Enqueue(simReq)

	// Track peak queue depth
	queueDepth := inst.WaitQueueDepth()
	if queueDepth > inst.PeakWaitQueueDepth {
		inst.PeakWaitQueueDepth = queueDepth
	}
}

// WaitQueueDepth returns the current wait queue depth
// Uses the efficient Len() method instead of destructive queue traversal
func (inst *InstanceSimulator) WaitQueueDepth() int {
	if inst.WaitQueue == nil {
		return 0
	}
	return inst.WaitQueue.Len()
}

// RunningBatchSize returns the current running batch size
func (inst *InstanceSimulator) RunningBatchSize() int {
	if inst.RunningBatch == nil {
		return 0
	}
	return len(inst.RunningBatch.Requests)
}

// KVCacheUtilization returns the fraction of KV cache in use
func (inst *InstanceSimulator) KVCacheUtilization() float64 {
	if inst.KVCache == nil || inst.KVCache.TotalBlocks == 0 {
		return 0.0
	}
	return float64(inst.KVCache.UsedBlockCnt) / float64(inst.KVCache.TotalBlocks)
}

// TotalKVBlocks returns the total number of KV blocks
func (inst *InstanceSimulator) TotalKVBlocks() int {
	if inst.KVCache == nil {
		return 0
	}
	return int(inst.KVCache.TotalBlocks)
}

// FreeKVBlocks returns the number of free KV blocks
func (inst *InstanceSimulator) FreeKVBlocks() int {
	if inst.KVCache == nil {
		return 0
	}
	return int(inst.KVCache.TotalBlocks - inst.KVCache.UsedBlockCnt)
}

// Step performs one simulation step
// Returns step duration and completed requests
// This is a stub for Phase 1 - will be fully implemented when integrating with existing sim logic
func (inst *InstanceSimulator) Step(clock int64) (stepDuration int64, completedReqs []*Request) {
	inst.Clock = clock

	// Stub implementation - will be filled in when integrating with existing simulator
	return 0, nil
}
