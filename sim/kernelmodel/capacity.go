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
	pool := m.scenario.Pools[m.poolIndex]
	util := pool.Engine.GPUMemoryUtilization
	if util <= 0 || util > 1 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: gpu_memory_utilization must be in (0, 1], got %v", util)
	}
	blockSize := pool.Engine.BlockSize
	if blockSize <= 0 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: block_size must be > 0, got %d", blockSize)
	}

	deviceBytes := int64(m.chipMemoryGiB * float64(gibToBytes))
	budget := int64(float64(deviceBytes) * util)
	fixed := m.k.FixedBytes().Total()

	// A per-rank figure above the device's own memory cannot be per-rank, and dividing a
	// budget by it would produce a capacity number that looks plausible and is not. See
	// UpstreamExpertWeightDefect for what is wrong and why this refuses rather than
	// compensates.
	if fixed >= deviceBytes {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: the kernel reports %.1f GiB of per-rank fixed occupancy on a "+
				"%.1f GiB device (%s, tp=%d, expert-parallel width %d). A per-rank figure "+
				"cannot exceed the part. This is the upstream defect described in "+
				"UpstreamExpertWeightDefect; capacity is not derivable for this "+
				"deployment and no budget is guessed",
			float64(fixed)/float64(gibToBytes),
			float64(deviceBytes)/float64(gibToBytes),
			m.scenario.Model, pool.Parallel.TP, m.k.Resolved().ExpertParallelWidth)
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
	dp := pool.Parallel.DP
	scaled := false
	if m.isMoE && dp > 1 {
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

// UpstreamExpertWeightDefect documents a defect in blis-latency-kernel that this package
// detects and refuses to work around.
//
// # What is wrong
//
// `FixedBytes()` overstates MoE expert weights by the tensor-parallel width when expert
// parallelism is off. blis-latency-kernel/new.go:493 accumulates
//
//	weights += c * l.ExpertWeightBytesPerExpert * k.expertsPerRank
//
// with no division by `expertTensorShards`, while the step-time path at kernel.go:331 does
// divide by it. `expertTensorShards` is 1 when expert parallelism is on (each rank holds
// whole experts) and `tp` when it is off (experts are sliced across the tensor-parallel
// group) -- new.go:274-277, matching vLLM fused_moe/config.py:1225.
//
// # The arithmetic, checked without the kernel
//
// GLM-5 has 75 MoE layers, 256 experts, n=2048, k=6144, three matrices per expert, fp8 at
// one byte per parameter:
//
//	75 * 256 * 3 * 2048 * 6144 * 1 = 675.0 GiB   whole model
//	                        / tp=8 =  84.4 GiB   correct per rank
//
// The kernel reports 677.5 GiB of weights at tp=8 (the remainder being dense layers,
// embeddings and the head). 675.0 against 677.5 is the undivided expert term.
//
// # Why this package refuses rather than compensating
//
// Dividing by the shard count here would be easy and wrong for this experiment. This
// package exists to measure the kernel as committed; an adapter that silently corrected a
// kernel value would be measuring a patched kernel while reporting on the released one, and
// would hide a defect that affects every memory and capacity answer the kernel gives.
//
// # Scope
//
// The defect is confined to FixedBytes. It does not affect the 13.67% shape result reported
// in docs/perf-model/latency-kernel-implementation.html section 6.1, because that comparison
// scores step time and the step-time path divides correctly.
//
// Five of the 39 evaluation deployments trip the refusal above. The rest are MoE models
// whose expert weights are small enough, or on parts large enough, that the overstatement
// stays below the device size -- so their budgets would be wrong but positive, which is the
// more dangerous case and the reason the check is an explicit comparison against device
// memory rather than a non-negativity test.
const UpstreamExpertWeightDefect = "blis-latency-kernel/new.go:493 omits /expertTensorShards"
