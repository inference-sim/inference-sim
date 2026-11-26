package cmd

import (
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// poolMemory is the memory breakdown of one pool's GPU, as the engine sized its KV pool at
// start-up (vLLM's request_memory and determine_available_memory): the device's memory as the
// driver reports it, the share gpu_memory_utilization allows, and what that share is spent on.
// It is the Result's reports.kernel.<role>.memory. Every figure is in bytes; the figures in
// PerGPU are one GPU's (the kernel's rank, one tensor-parallel shard), and PerReplica holds the
// same figures summed over the GPUs of one replica.
//
// The parts add up to the budget:
//
//	weights + activation_peak + cuda_graph + eplb_redundant + comm_buffers + non_torch_in_budget
//	  + engine_buffers + lora_reserved + kv_pool == budget
//
// except that cuda_graph is left out of the sum when CUDAGraphInBudget is false: vLLM's
// profiler then does not estimate the graphs, and the engine does not take them from the
// budget.
type poolMemory struct {
	PerGPU memoryBytes `json:"per_gpu"`
	// PerReplica is PerGPU times GPUsPerReplica. Absent under pipeline parallelism, where the
	// stages hold different weights and the per-GPU figures are one stage's.
	PerReplica *memoryBytes `json:"per_replica,omitempty"`
	// CUDAGraphInBudget is whether the CUDA-graph estimate was taken from the budget.
	CUDAGraphInBudget bool `json:"cuda_graph_in_budget"`
	// GPUsPerReplica is the GPUs one replica (one engine core) occupies.
	GPUsPerReplica int `json:"gpus_per_replica"`
	// BlockSizeTokens is the tokens one KV block holds.
	BlockSizeTokens int64 `json:"block_size_tokens"`
	// KVBytesPerBlock is the bytes one KV block takes on one GPU, as the kernel prices it.
	// Absent when the kernel states none (a model whose every layer's cache is bounded).
	KVBytesPerBlock int64 `json:"kv_bytes_per_block,omitempty"`
	// NumGPUBlocks is the blocks the KV pool is sized to hold on one GPU, vLLM's num_gpu_blocks
	// (the count its start-up log states). For a model with one KV cache group,
	// NumGPUBlocks x BlockSizeTokens is kv_capacity_tokens_per_replica.
	NumGPUBlocks int64 `json:"num_gpu_blocks,omitempty"`
	// KVBlocks is the blocks of the pool a request can use on one data-parallel rank: the
	// instance's per-rank count, one less than NumGPUBlocks because the engine keeps block 0 as
	// vLLM's null block. For a pool of several instances it is the smallest. Absent when no
	// instance of the pool ran.
	KVBlocks int64 `json:"kv_blocks,omitempty"`
}

// memoryBytes is the split of a GPU's memory, or of a replica's.
type memoryBytes struct {
	// DeviceTotal is the memory the CUDA driver reports for the GPU.
	DeviceTotal int64 `json:"device_total_bytes"`
	// Budget is DeviceTotal times gpu_memory_utilization, rounded up to a byte, or Fixed plus
	// kv_cache_memory_bytes when kv_cache_memory_bytes is stated.
	Budget int64 `json:"budget_bytes"`
	// Weights is the model parameters resident on the GPU.
	Weights int64 `json:"weights_bytes"`
	// ActivationPeak is the transient scratch at the batched-token bound.
	ActivationPeak int64 `json:"activation_peak_bytes"`
	// CUDAGraph is the CUDA-graph capture estimate; zero when capture is off.
	CUDAGraph int64 `json:"cuda_graph_bytes"`
	// EPLBRedundant is the redundant expert replicas; zero when EPLB is off.
	EPLBRedundant int64 `json:"eplb_redundant_bytes"`
	// CommBuffers is the collective and all-to-all buffers.
	CommBuffers int64 `json:"comm_buffers_bytes"`
	// NonTorchInBudget is the non-torch memory the GPU grows inside the budget while the
	// engine loads and profiles.
	NonTorchInBudget int64 `json:"non_torch_in_budget_bytes"`
	// EngineBuffers is the GPU buffers and workspaces the engine allocates before the
	// budget's profiling.
	EngineBuffers int64 `json:"engine_buffers_bytes"`
	// LoRAReserved is the static LoRA adapter reservation the GPU gives up before the KV pool
	// is sized; zero without adapters or when kv_cache_memory_bytes is stated (the profiling
	// that nets it out is skipped).
	LoRAReserved int64 `json:"lora_reserved_bytes"`
	// KVPool is what is left for the KV cache: Budget minus every part above.
	KVPool int64 `json:"kv_pool_bytes"`
}

// scaled is b with every figure multiplied by n.
func (b memoryBytes) scaled(n int64) memoryBytes {
	return memoryBytes{b.DeviceTotal * n, b.Budget * n, b.Weights * n, b.ActivationPeak * n,
		b.CUDAGraph * n, b.EPLBRedundant * n, b.CommBuffers * n, b.NonTorchInBudget * n,
		b.EngineBuffers * n, b.LoRAReserved * n, b.KVPool * n}
}

// poolKVBlocks is the per-rank KV blocks each opened pool's instances were built with, by pool
// role: the smallest per-rank count among the instances serving only that role (every instance
// of a colocated run). A role with no such instance is absent.
func poolKVBlocks(cs *cluster.ClusterSimulator) map[string]int64 {
	out := map[string]int64{}
	membership := cs.PoolMembership()
	for _, inst := range cs.Instances() {
		role := "colocated"
		if len(membership) > 0 {
			switch membership[string(inst.ID())] {
			case cluster.PoolRolePrefill:
				role = "prefill"
			case cluster.PoolRoleDecode:
				role = "decode"
			default:
				continue
			}
		}
		if n := inst.PerRankKVBlocks(); n > 0 && (out[role] == 0 || n < out[role]) {
			out[role] = n
		}
	}
	return out
}

// memorySection is each opened pool's memory breakdown, keyed by pool role, for the Result's
// reports.kernel. Empty when no kernel was opened or when the run uses SGLang (whose
// kernelOpened is the host's vLLM twin, not SGLang's pool). A pool whose memory cannot be split
// is left out: the run has already sized its KV pool from the same figures, so this does not
// happen for a run that started.
func memorySection(models []*engineModel, cs *cluster.ClusterSimulator) map[string]poolMemory {
	if scenarioEngine.Kind == engineKindSGLang {
		return nil
	}
	out := map[string]poolMemory{}
	blocks := poolKVBlocks(cs)
	for _, m := range models {
		role := string(m.Role())
		pm, err := poolMemoryOf(m, loraReservedBytesForKV, blocks[role])
		if err != nil {
			continue
		}
		out[role] = pm
	}
	return out
}
