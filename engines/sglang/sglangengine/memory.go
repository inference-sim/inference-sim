// Original to blis-sglang.

package sglangengine

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
)

// PoolMemory is one pool's memory as SGLang's own sizing leaves it, per rank (per GPU): the
// device, the memory the pool may use, the kernel's request-independent occupancy, and the KV
// pool the remainder was cut into.
//
// The terms are the sizing the engine runs, not a second derivation of it. BudgetBytes is the
// device's free memory before the weights load, which SGLang reads with
// get_available_gpu_memory; AllocatableBytes is _profile_available_bytes over the weights, which
// is what the token capacity was cut from. Fixed is the kernel's per-rank occupancy, which
// SGLang's formula subtracts only as the weights plus a flat memory-fraction slack, so
// AllocatableBytes is NOT BudgetBytes - Fixed.Total(): the part of the budget neither term claims
// (the slack beyond the kernel's other fixed terms, and CUDA's own reservation below the budget)
// stays visible between them instead of being charged to one.
type PoolMemory struct {
	// DeviceBytes is the device's total memory as SGLang reads it (nvidia-smi memory.total).
	DeviceBytes int64
	// BudgetBytes is the device's free memory before the weights load.
	BudgetBytes int64
	// Fixed is the kernel's request-independent occupancy on this rank.
	Fixed kernel.MemoryBreakdown
	// AllocatableBytes is the memory the KV pool may take: the budget less the weights and the
	// static memory fraction's slack.
	AllocatableBytes int64
	// BlockTokens is one page's tokens, BlockBytes the page's bytes, and PerTokenBytes the pool's
	// per-token cost. PerTokenBytes is ceil(BlockBytes / BlockTokens), so the three agree.
	BlockTokens     int
	BlockBytes      int64
	PerTokenBytes   int64
	BlocksPerRank   int64
	TokensPerRank   int64
	BytesPerRank    int64
	DataParallel    int
	AggregateBlocks int64
	AggregateTokens int64
	AggregateBytes  int64
}

// MemoryOf opens the pool for s as a run's factory does and reports its memory.
func MemoryOf(h Host, s engine.Spec) (PoolMemory, error) {
	pool, err := h.Open(s)
	if err != nil {
		return PoolMemory{}, err
	}
	return MemoryOfPool(pool, s)
}

// MemoryOfPool reports the memory of a pool the host has already opened (Pool), resolved and
// sized exactly as a run's factory would: the report describes the pool the engine would run,
// with the pool sizes that engine would use.
func MemoryOfPool(pool Pool, s engine.Spec) (PoolMemory, error) {
	r, err := resolveFromPool(pool, s)
	if err != nil {
		return PoolMemory{}, err
	}
	page := r.sa.PageSize
	if page < 1 {
		return PoolMemory{}, fmt.Errorf("sglangengine: the resolved page size is %d tokens", page)
	}
	tokens := int64(r.rt.MaxTotalNumTokens)
	if tokens <= 0 {
		return PoolMemory{}, fmt.Errorf("sglangengine: the KV budget leaves no pool (budget %d bytes, %d bytes per token)",
			r.rt.BudgetBytes, r.rt.CellBytes)
	}
	blocks := tokens / int64(page)
	blockBytes := r.rt.CellBytes * int64(page)
	// SGLang refuses data parallelism above one on both its attention and its rank axes, so a
	// pool is one rank's and its aggregate is its per-rank figure.
	const dp = 1
	return PoolMemory{
		DeviceBytes:      int64(math.Round(r.pool.Chip.MemoryMiB * float64(1<<20))),
		BudgetBytes:      int64(math.Round(r.pool.PreModelLoadGiB * float64(1<<30))),
		Fixed:            r.pool.Kernel.FixedBytes(),
		AllocatableBytes: r.rt.BudgetBytes,
		BlockTokens:      page,
		BlockBytes:       blockBytes,
		PerTokenBytes:    r.rt.CellBytes,
		BlocksPerRank:    blocks,
		TokensPerRank:    tokens,
		BytesPerRank:     blocks * blockBytes,
		DataParallel:     dp,
		AggregateBlocks:  blocks,
		AggregateTokens:  tokens,
		AggregateBytes:   blocks * blockBytes,
	}, nil
}
