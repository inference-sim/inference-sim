// capacity.go derives the KV-block budget from the kernel's own memory methods.
//
// # Why this exists
//
// BLIS sizes its KV cache with latency.CalculateKVBlocks, which is called from cmd/ and
// sits OUTSIDE the sim.LatencyModel seam. Registering this package's adapter at that seam
// therefore changes what a step costs but not how many requests fit, so the resident batch
// would still be decided by the legacy weight estimator.
//
// For this experiment that is not a cosmetic inconsistency. The resident batch is the
// quantity under test: blis-latency-kernel's 13.67% shape error against AISimulate assumes
// resident batch equals client concurrency, and the reason to run BLIS at all is that its
// scheduler decides the resident batch for real. If capacity came from the legacy path, the
// experiment would be measuring the legacy model's admission behaviour while attributing
// the result to the kernel.
//
// So capacity comes from the kernel too, and "BLIS relies exclusively on the kernel" is
// true of both terms.
//
// # The convention this reproduces
//
// latency.CalculateKVBlocks returns blocks PER DP RANK, computed as
//
//	allocatableBytes / perBlockBytes
//
// and multiplied by dp for an MoE model, because vLLM runs dp independent EngineCores each
// with a full KV budget and splits requests disjointly across them. This file keeps that
// convention so the block count means the same thing to BLIS's scheduler as before; only
// its provenance changes.
//
// The kernel supplies both inputs:
//
//   - FixedBytes() is per-rank occupancy independent of the request set: weights, the
//     engine's workspace, communicator buffers, graph capture. Subtracting it from the
//     budget is what leaves room for KV.
//   - SequenceVariableBytes(blockSize) is the per-sequence KV cost of one page, already
//     page-quantized and already divided by tensor-parallel width down to the engine's
//     one-KV-head floor.
package kernelmodel

import (
	"fmt"
)

// KVBudget is the KV-block budget derived from a kernel, with the terms that produced it.
// The terms are reported rather than just the total because a capacity number no one can
// decompose is not auditable: a run that admits fewer requests than expected is diagnosed
// from these fields.
type KVBudget struct {
	// TotalBlocks is what BLIS's KVCacheConfig wants: usable blocks, per DP rank, scaled
	// by DP for an MoE model.
	TotalBlocks int64

	// BlockSize is tokens per block, from the scenario's engine settings.
	BlockSize int

	// DeviceBytes is the chip's memory.
	DeviceBytes int64
	// BudgetBytes is DeviceBytes times the engine's gpu_memory_utilization.
	BudgetBytes int64
	// FixedBytes is the kernel's per-rank request-independent occupancy.
	FixedBytes int64
	// AllocatableBytes is BudgetBytes minus FixedBytes: what KV may use.
	AllocatableBytes int64
	// PerBlockBytes is the kernel's per-sequence KV cost of one page.
	PerBlockBytes int64
	// DPScaled records whether the MoE DP multiplication applied.
	DPScaled bool
}

// KVBudget derives the block budget for this model's deployment.
//
// It returns an error rather than a zero budget when the deployment does not fit: a
// simulation that silently ran with no KV would produce a number, and that number would be
// meaningless.
func (m *Model) KVBudget() (KVBudget, error) {
	// From the kernel, which resolved this pool. Reading the Deployment document instead
	// would mean holding it alongside and indexing back in -- a second answer to "which
	// pool is this", which can disagree with the one the pricing used.
	e := m.k.Engine()
	util := e.GPUMemoryUtilization
	if util <= 0 || util > 1 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: gpu_memory_utilization must be in (0, 1], got %v", util)
	}
	blockSize := e.BlockSize
	if blockSize <= 0 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: block_size must be > 0, got %d", blockSize)
	}

	deviceBytes := int64(m.k.Chip().MemoryGiB * float64(gibToBytes))
	budget := int64(float64(deviceBytes) * util)
	fixed := m.k.FixedBytes().Total()

	// A per-rank figure above the device's own memory cannot be per-rank, and dividing a
	// budget by it would produce a capacity number that looks plausible and is not. See
	// the guard note below for what it catches and why it refuses rather than
	// compensates.
	if fixed >= deviceBytes {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: the kernel reports %.1f GiB of per-rank fixed occupancy on a "+
				"%.1f GiB device (%s, tp=%d, expert-parallel width %d). A per-rank figure "+
				"cannot exceed the part, so a sharding term in the kernel's memory path "+
				"is wrong. No budget is guessed: one divided out of an impossible "+
				"occupancy would look plausible and set the wrong resident batch",
			float64(fixed)/float64(gibToBytes),
			float64(deviceBytes)/float64(gibToBytes),
			m.k.ModelName(), m.k.TensorParallelWidth(),
			m.k.Resolved().ExpertParallelWidth)
	}

	allocatable := budget - fixed
	if allocatable <= 0 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: no room for KV: a %.1f GiB device at utilization %.2f gives a "+
				"%.1f GiB budget against %.1f GiB of fixed occupancy per rank",
			float64(deviceBytes)/float64(gibToBytes), util,
			float64(budget)/float64(gibToBytes), float64(fixed)/float64(gibToBytes))
	}

	// One page of KV for one sequence, as the kernel prices it. Page-quantized already.
	perBlock := m.k.SequenceVariableBytes(blockSize)
	if perBlock <= 0 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: per-block KV resolved to %d bytes; a budget cannot be divided "+
				"by it", perBlock)
	}

	blocks := allocatable / perBlock
	if blocks <= 0 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: computed 0 KV blocks (allocatable %.2f GiB, per block %d bytes)",
			float64(allocatable)/float64(gibToBytes), perBlock)
	}

	// DP scaling, matching latency.CalculateKVBlocks: dp independent EngineCores each hold
	// a full budget and requests split disjointly across them. Gated on MoE for the same
	// reason it is there -- a dense model is never scaled.
	dp := m.k.DataParallelWidth()
	scaled := false
	if m.k.Experts() > 0 && dp > 1 {
		blocks *= int64(dp)
		scaled = true
	}

	return KVBudget{
		TotalBlocks:      blocks,
		BlockSize:        blockSize,
		DeviceBytes:      deviceBytes,
		BudgetBytes:      budget,
		FixedBytes:       fixed,
		AllocatableBytes: allocatable,
		PerBlockBytes:    perBlock,
		DPScaled:         scaled,
	}, nil
}

const gibToBytes = 1 << 30

// The guard below survives the upstream fix deliberately.
//
// blis-latency-kernel's FixedBytes once omitted the /expertTensorShards division that the
// step-time path applies, reporting a rank's MoE expert weights as the whole model's
// whenever expert parallelism was off -- 675 GiB for GLM-5 at tp=8, on a 141 GiB part. That
// is fixed upstream, and every evaluation deployment now yields a derivable budget.
//
// The check stays because it costs one comparison and catches the entire class: any future
// sharding error in a memory term shows up as a per-rank figure larger than the part. A
// capacity number divided out of an impossible occupancy looks plausible and silently sets
// the wrong resident batch, which is the quantity this experiment measures.
