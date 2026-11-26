package cmd

import (
	"fmt"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
)

// poolMemory is one pool's memory as the engine that sizes it leaves it, per rank (per GPU). It is
// the block a Result carries under reports.engine.<role>.memory.
//
// Every term is the engine's own: the device and the memory the pool may use, the static
// reservation taken out of it, the kernel's request-independent occupancy, and the KV pool the
// remainder was cut into. Nothing is re-derived here, so the report and the run's admission
// describe one pool. A caller reading the bar a UI draws gets its free term from
// max(0, device - fixed.total - reserved_bytes - kv.bytes_per_rank), which is what the engine
// neither claims (a memory fraction's slack, CUDA's own reservation).
type poolMemory struct {
	Role   deployment.Role `json:"role"`
	Engine string          `json:"engine"`
	// DeviceBytes is the device's total memory, BudgetBytes the part of it the pool may use (the
	// engine's utilization of it, or its free-before-load figure), and ReservedBytes what the
	// engine set aside before sizing the pool (a static LoRA adapter reservation; zero when none).
	DeviceBytes   int64 `json:"device_bytes"`
	BudgetBytes   int64 `json:"budget_bytes"`
	ReservedBytes int64 `json:"reserved_bytes"`
	// Fixed is the kernel's request-independent occupancy and AllocatableBytes what the engine's
	// sizing left for KV after it and the reservation.
	Fixed            fixedMemory  `json:"fixed"`
	AllocatableBytes int64        `json:"allocatable_bytes"`
	KV               kvPoolMemory `json:"kv"`
}

// fixedMemory is the kernel's request-independent occupancy, decomposed so a caller can see which
// term dominates (kernel.MemoryBreakdown). Total is their sum.
type fixedMemory struct {
	Total          int64 `json:"total"`
	Weights        int64 `json:"weights"`
	ActivationPeak int64 `json:"activation_peak"`
	CUDAGraph      int64 `json:"cuda_graph"`
	EPLBRedundant  int64 `json:"eplb_redundant"`
	CommBuffers    int64 `json:"comm_buffers"`
	NonTorch       int64 `json:"non_torch"`
	EngineBuffers  int64 `json:"engine_buffers"`
}

// fixedMemoryOf decomposes a kernel breakdown into its report form.
func fixedMemoryOf(b kernel.MemoryBreakdown) fixedMemory {
	return fixedMemory{
		Total:          b.Total(),
		Weights:        b.Weights,
		ActivationPeak: b.ActivationPeak,
		CUDAGraph:      b.CUDAGraph,
		EPLBRedundant:  b.EPLBRedundant,
		CommBuffers:    b.CommBuffers,
		NonTorch:       b.NonTorchInBudget,
		EngineBuffers:  b.EngineBuffers,
	}
}

// kvPoolMemory is the KV pool one rank holds. The three capacity figures are self-consistent:
// TokensPerRank is BlocksPerRank x BlockTokens, BytesPerRank is BlocksPerRank x BlockBytes, and
// PerTokenBytes is ceil(BlockBytes / BlockTokens), so a caller showing any two derives the third.
//
// TokensPerRank is the pool's slot count: the capacity the scheduler admits and evicts against. It
// is not the engine's start-up KV-capacity figure, which a Result carries as
// reports.engine.cluster.kv_capacity_tokens_per_replica. That figure is the engine's own
// definition -- for vLLM, the pool's requests at max_model_len times max_model_len, not blocks x
// block size (inference-sim#1969, #2431) -- and on a hybrid model it is a different number. Read
// TokensPerRank for the pool and the cluster figure for the engine's report; one is not a
// conversion of the other.
//
// The per-rank figures are what one GPU holds: one EngineCore's pool, or a TP rank's shard of it.
// The aggregate figures are what one instance holds, which differs from per rank x DataParallel
// only where the engine sizes its DP cores into one pool (a model with experts); a dense model's
// data parallelism is replicas, each an instance of its own.
type kvPoolMemory struct {
	BlockTokens     int   `json:"block_tokens"`
	BlockBytes      int64 `json:"block_bytes"`
	PerTokenBytes   int64 `json:"per_token_bytes"`
	BlocksPerRank   int64 `json:"blocks_per_rank"`
	TokensPerRank   int64 `json:"tokens_per_rank"`
	BytesPerRank    int64 `json:"bytes_per_rank"`
	DataParallel    int   `json:"data_parallel"`
	AggregateBlocks int64 `json:"aggregate_blocks"`
	AggregateTokens int64 `json:"aggregate_tokens"`
	AggregateBytes  int64 `json:"aggregate_bytes"`
}

// perTokenBytes is the pool's per-token cost, rounded up from one block, so that a page's bytes
// divided over its tokens agree with the block the engine allocates. A block of zero tokens is
// refused here rather than divided by: the engine's sizing requires a positive block, and SGLang
// refuses a page below one at its own sizing site, so this is the local check that the figure a
// Result publishes cannot be a division by zero.
func perTokenBytes(blockBytes int64, blockTokens int) (int64, error) {
	if blockTokens < 1 {
		return 0, fmt.Errorf("KV pool block has %d tokens; a per-token cost needs a positive block", blockTokens)
	}
	return (blockBytes + int64(blockTokens) - 1) / int64(blockTokens), nil
}

// poolMemoryRecords is each pool's memory as the engine that sizes it leaves it. It reads the
// pools a run's own build sites opened and opens none itself, so a Result states the pool the run
// priced.
func poolMemoryRecords() ([]poolMemory, error) {
	if scenarioEngine.Kind == engineKindSGLang {
		return sglangMemoryRecords()
	}
	return vllmMemoryRecords()
}

// poolMemorySection is each pool's memory block, keyed by the role the Result states it under.
func poolMemorySection() map[string]any {
	records, err := poolMemoryRecords()
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	out := make(map[string]any, len(records))
	for _, r := range records {
		out[string(r.Role)] = toGeneric(r)
	}
	return out
}
