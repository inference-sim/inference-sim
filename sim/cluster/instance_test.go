package cluster

import "testing"

// TestInstanceSimulator_Creation tests instance creation
func TestInstanceSimulator_Creation(t *testing.T) {
	config := &DeploymentConfig{
		ConfigID: "config1",
		EngineConfig: &VLLMEngineConfig{
			MaxNumSeqs: 128,
		},
	}

	inst := NewInstanceSimulator("inst1", PoolMonolithic, config, 1000, 16)

	if inst.ID != "inst1" {
		t.Errorf("ID = %s, want inst1", inst.ID)
	}
	if inst.PoolType != PoolMonolithic {
		t.Errorf("PoolType = %s, want %s", inst.PoolType, PoolMonolithic)
	}
	if inst.TotalKVBlocks() != 1000 {
		t.Errorf("TotalKVBlocks = %d, want 1000", inst.TotalKVBlocks())
	}
	if inst.WaitQueueDepth() != 0 {
		t.Errorf("Initial queue depth = %d, want 0", inst.WaitQueueDepth())
	}
	if inst.RunningBatchSize() != 0 {
		t.Errorf("Initial batch size = %d, want 0", inst.RunningBatchSize())
	}
}

// TestInstanceSimulator_EnqueueRequest tests request enqueueing
func TestInstanceSimulator_EnqueueRequest(t *testing.T) {
	config := &DeploymentConfig{
		ConfigID: "config1",
	}
	inst := NewInstanceSimulator("inst1", PoolMonolithic, config, 1000, 16)

	req1 := &Request{
		ID:           "req1",
		PromptTokens: 100,
		OutputTokens: 50,
		ArrivalTime:  1000,
	}

	inst.EnqueueRequest(req1)

	if inst.WaitQueueDepth() != 1 {
		t.Errorf("After enqueue, queue depth = %d, want 1", inst.WaitQueueDepth())
	}
	if inst.PeakWaitQueueDepth != 1 {
		t.Errorf("PeakWaitQueueDepth = %d, want 1", inst.PeakWaitQueueDepth)
	}

	req2 := &Request{
		ID:           "req2",
		PromptTokens: 200,
		OutputTokens: 100,
		ArrivalTime:  2000,
	}

	inst.EnqueueRequest(req2)

	if inst.WaitQueueDepth() != 2 {
		t.Errorf("After 2nd enqueue, queue depth = %d, want 2", inst.WaitQueueDepth())
	}
	if inst.PeakWaitQueueDepth != 2 {
		t.Errorf("PeakWaitQueueDepth = %d, want 2", inst.PeakWaitQueueDepth)
	}
}

// TestInstanceSimulator_KVCacheUtilization tests KV cache utilization calculation
func TestInstanceSimulator_KVCacheUtilization(t *testing.T) {
	config := &DeploymentConfig{
		ConfigID: "config1",
	}
	inst := NewInstanceSimulator("inst1", PoolMonolithic, config, 1000, 16)

	// Initially, utilization should be 0
	util := inst.KVCacheUtilization()
	if util != 0.0 {
		t.Errorf("Initial utilization = %.2f, want 0.00", util)
	}

	// Simulate allocating blocks by directly modifying UsedBlockCnt
	inst.KVCache.UsedBlockCnt = 500

	util = inst.KVCacheUtilization()
	expected := 0.5
	if util != expected {
		t.Errorf("Utilization = %.2f, want %.2f", util, expected)
	}

	// Full utilization
	inst.KVCache.UsedBlockCnt = 1000

	util = inst.KVCacheUtilization()
	if util != 1.0 {
		t.Errorf("Full utilization = %.2f, want 1.00", util)
	}
}

// TestInstanceSimulator_BC4_InstanceIsolation tests BC-4: instance isolation
func TestInstanceSimulator_BC4_InstanceIsolation(t *testing.T) {
	config := &DeploymentConfig{
		ConfigID: "config1",
	}

	// Create two instances
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, config, 1000, 16)
	inst2 := NewInstanceSimulator("inst2", PoolMonolithic, config, 1000, 16)

	// Enqueue requests to inst1
	req1 := &Request{
		ID:           "req1",
		PromptTokens: 100,
		OutputTokens: 50,
		ArrivalTime:  1000,
	}
	inst1.EnqueueRequest(req1)

	// Verify inst1 has the request, inst2 does not
	if inst1.WaitQueueDepth() != 1 {
		t.Errorf("inst1 queue depth = %d, want 1", inst1.WaitQueueDepth())
	}
	if inst2.WaitQueueDepth() != 0 {
		t.Errorf("inst2 queue depth = %d, want 0 (should be isolated)", inst2.WaitQueueDepth())
	}

	// Allocate KV blocks in inst1
	inst1.KVCache.UsedBlockCnt = 500

	// Verify inst2's KV cache is unaffected
	if inst2.KVCache.UsedBlockCnt != 0 {
		t.Errorf("inst2 KV used blocks = %d, want 0 (should be isolated)", inst2.KVCache.UsedBlockCnt)
	}

	// Update clock in inst1
	inst1.Clock = 5000

	// Verify inst2's clock is unaffected
	if inst2.Clock != 0 {
		t.Errorf("inst2 clock = %d, want 0 (should be isolated)", inst2.Clock)
	}
}

// TestInstanceSimulator_BC12_BatchSizeLimit tests BC-12: batch size limit
func TestInstanceSimulator_BC12_BatchSizeLimit(t *testing.T) {
	config := &DeploymentConfig{
		ConfigID: "config1",
		EngineConfig: &VLLMEngineConfig{
			MaxNumSeqs: 128,
		},
	}

	inst := NewInstanceSimulator("inst1", PoolMonolithic, config, 1000, 16)

	// Verify running batch respects max size
	// For now, this is a placeholder test since Step() is stubbed
	// When Step() is implemented, it should enforce MaxNumSeqs

	if inst.Config.EngineConfig.MaxNumSeqs != 128 {
		t.Errorf("MaxNumSeqs = %d, want 128", inst.Config.EngineConfig.MaxNumSeqs)
	}

	// Test that RunningBatchSize never exceeds MaxNumSeqs
	// This will be fully tested when Step() is implemented
	if inst.RunningBatchSize() > inst.Config.EngineConfig.MaxNumSeqs {
		t.Errorf("RunningBatchSize (%d) exceeds MaxNumSeqs (%d)", inst.RunningBatchSize(), inst.Config.EngineConfig.MaxNumSeqs)
	}
}

// TestInstanceSimulator_BC13_KVCacheConservation tests BC-13: KV cache conservation
func TestInstanceSimulator_BC13_KVCacheConservation(t *testing.T) {
	config := &DeploymentConfig{
		ConfigID: "config1",
	}

	totalBlocks := int64(1000)
	inst := NewInstanceSimulator("inst1", PoolMonolithic, config, totalBlocks, 16)

	// Invariant: allocated + free == total
	verifyConservation := func(label string) {
		used := inst.KVCache.UsedBlockCnt
		free := int64(inst.FreeKVBlocks())
		total := inst.KVCache.TotalBlocks

		if used+free != total {
			t.Errorf("%s: KV cache conservation violated: used(%d) + free(%d) = %d, want %d", label, used, free, used+free, total)
		}
	}

	// Initial state
	verifyConservation("Initial")

	// Simulate allocating blocks
	inst.KVCache.UsedBlockCnt = 300
	verifyConservation("After allocating 300 blocks")

	// Simulate allocating more blocks
	inst.KVCache.UsedBlockCnt = 700
	verifyConservation("After allocating 700 blocks")

	// Simulate freeing some blocks
	inst.KVCache.UsedBlockCnt = 400
	verifyConservation("After freeing to 400 blocks")

	// Simulate full allocation
	inst.KVCache.UsedBlockCnt = totalBlocks
	verifyConservation("After full allocation")

	// Simulate full deallocation
	inst.KVCache.UsedBlockCnt = 0
	verifyConservation("After full deallocation")
}

// TestInstanceSimulator_FreeKVBlocks tests free block calculation
func TestInstanceSimulator_FreeKVBlocks(t *testing.T) {
	config := &DeploymentConfig{
		ConfigID: "config1",
	}
	inst := NewInstanceSimulator("inst1", PoolMonolithic, config, 1000, 16)

	// Initially all blocks are free
	if inst.FreeKVBlocks() != 1000 {
		t.Errorf("Initial free blocks = %d, want 1000", inst.FreeKVBlocks())
	}

	// Allocate 300 blocks
	inst.KVCache.UsedBlockCnt = 300
	if inst.FreeKVBlocks() != 700 {
		t.Errorf("Free blocks after 300 used = %d, want 700", inst.FreeKVBlocks())
	}

	// Allocate all blocks
	inst.KVCache.UsedBlockCnt = 1000
	if inst.FreeKVBlocks() != 0 {
		t.Errorf("Free blocks when full = %d, want 0", inst.FreeKVBlocks())
	}
}

// TestInstanceSimulator_PeakTracking tests peak metric tracking
func TestInstanceSimulator_PeakTracking(t *testing.T) {
	config := &DeploymentConfig{
		ConfigID: "config1",
	}
	inst := NewInstanceSimulator("inst1", PoolMonolithic, config, 1000, 16)

	// Enqueue multiple requests and verify peak tracking
	for i := 0; i < 5; i++ {
		req := &Request{
			ID:           string(rune('a' + i)),
			PromptTokens: 100,
			OutputTokens: 50,
			ArrivalTime:  int64(1000 * (i + 1)),
		}
		inst.EnqueueRequest(req)
	}

	if inst.PeakWaitQueueDepth != 5 {
		t.Errorf("PeakWaitQueueDepth = %d, want 5", inst.PeakWaitQueueDepth)
	}

	// Dequeue some requests (simulated)
	// Peak should remain at 5 even if current depth decreases
	// This will be tested more fully when Step() is implemented
}
