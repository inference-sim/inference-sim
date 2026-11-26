// Moved from inference-sim/sim/kernelmodel/capacity.go at ca532abca209af41425c007974a5f3ef336f5981.

// capacity.go derives the KV-block budget from the kernel's own memory methods.
//
// # Why this exists
//
// The KV budget decides how many requests are resident, and the resident batch is what a
// step is priced for. Taking the budget from the same kernel that prices the step keeps the
// two describing one engine; a budget from anywhere else would let admission behaviour
// disagree with the price.
//
// # The convention this reproduces
//
// The budget is blocks PER DP RANK, computed as
//
//	allocatableBytes / perBlockBytes
//
// and multiplied by dp for an MoE model, because vLLM runs dp independent EngineCores each
// with a full KV budget and splits requests disjointly across them.
//
// The kernel supplies the inputs:
//
//   - FixedBytes() is per-rank occupancy independent of the request set: weights, the
//     engine's workspace, communicator buffers, graph capture. Subtracting it from the
//     budget is what leaves room for KV.
//   - SequenceVariableBytes(blockSize) is the per-sequence KV cost of one page, already
//     page-quantized and already divided by tensor-parallel width down to the engine's
//     one-KV-head floor.
//   - SequenceBoundedBytesAt(computed, inFlight) is what a sequence's bounded state --
//     window rings, compressor and indexer state, bounded recurrent state -- holds with
//     computed tokens computed and inFlight scheduled, in whole blocks of the shared pool; it
//     settles at SequenceFixedBytes while decoding. SequenceAdmissionBytes is the window
//     rings at vLLM's admission cap, which vLLM applies only as an admission gate. The
//     simulator charges these per resident sequence step by step (Settings.SequenceState,
//     #1941) rather than out of the pool up front, because how many sequences are resident is
//     the scheduler's to decide.
package kernelmodel

import (
	"fmt"
	"math"
	"strings"

	latencykernel "github.com/inference-sim/inference-sim/substrate/kernel"
)

// KVBudget is the KV-block budget derived from a kernel, with the terms that produced it.
// The terms are reported rather than just the total because a capacity number no one can
// decompose is not auditable: a run that admits fewer requests than expected is diagnosed
// from these fields.
type KVBudget struct {
	// TotalBlocks is the aggregate budget a single instance modelling every DP rank wants:
	// PerRankBlocks scaled by DP for an MoE model.
	TotalBlocks int64
	// PerRankBlocks is one EngineCore's budget, which is what a simulator that runs each DP
	// rank as its own replica sizes each replica with.
	PerRankBlocks int64

	// BlockSize is tokens per block, from the scenario's engine settings.
	BlockSize int

	// DeviceBytes is the chip's memory.
	DeviceBytes int64
	// BudgetBytes is DeviceBytes times the engine's gpu_memory_utilization, rounded up to a byte.
	BudgetBytes int64
	// FixedBytes is the kernel's per-rank request-independent occupancy.
	FixedBytes int64
	// AllocatableBytes is BudgetBytes minus FixedBytes: what KV may use.
	AllocatableBytes int64
	// PerBlockBytes is the kernel's per-sequence KV cost of one page.
	PerBlockBytes int64
	// DPScaled records whether the MoE DP multiplication applied.
	DPScaled bool

	// SequenceAdmissionBytes is the kernel's admission-cap reservation for a sequence's
	// bounded state and SequenceAdmissionBlocks its whole blocks of PerBlockBytes: vLLM's
	// admission gate and its startup capacity figure divide by it. SequenceSteadyBytes is the
	// smaller hold a decoding sequence's bounded state settles at (the kernel's
	// SequenceFixedBytes), SequenceSteadyBlocks its blocks. All are zero for a model with no
	// such caches.
	SequenceAdmissionBytes  int64
	SequenceAdmissionBlocks int64
	SequenceSteadyBytes     int64
	SequenceSteadyBlocks    int64
}

// KVBudget derives the block budget for this model's deployment.
//
// It returns an error rather than a zero budget when the deployment does not fit: a
// simulation that silently ran with no KV would produce a number, and that number would be
// meaningless.
func (m *Model) KVBudget() (KVBudget, error) { return m.KVBudgetReserving(0) }

// KVBudgetReserving is KVBudget with reservedBytes of the rank's HBM set aside before KV is
// sized -- the static LoRA adapter reservation, which vLLM takes beside the weights before it
// profiles the KV pool. The reservation is a total across the rank's tensor-parallel GPUs,
// sharded like weights, so each GPU gives up reservedBytes / tp of the budget the kernel's
// per-GPU occupancy figures are measured against.
func (m *Model) KVBudgetReserving(reservedBytes int64) (KVBudget, error) {
	return m.kvBudget(reservedBytes, false)
}

// kvBudget is KVBudgetReserving. grouped is set for an engine that cuts its pool into the model's
// KV cache groups (SettingsCheckedByGroups): there a model whose every layer is bounded -- whose
// growing-context page is zero -- still has pool blocks, each one block of every KV layer
// (KVLayerPageBytes, window rings included), which is the page vLLM's pool of one such group is
// cut from (get_kv_cache_config_from_groups, vllm/v1/core/kv_cache_utils.py:1622-1628@5df63a72).
// The engine sizes the pool itself, over every group (vllmengine poolBlocks); this is the
// simulator's figure for it.
func (m *Model) kvBudget(reservedBytes int64, grouped bool) (KVBudget, error) {
	mem, err := m.kvMemory(reservedBytes)
	if err != nil {
		return KVBudget{}, err
	}
	blockSize, allocatable := m.engine.BlockSize, mem.AllocatableBytes

	// One page of KV for one sequence, as the kernel prices it. Page-quantized already.
	perBlock := m.k.SequenceVariableBytes(blockSize)
	if perBlock <= 0 && grouped {
		if s, ok := m.k.(stageMemorist); ok {
			perBlock = s.KVLayerPageBytes(blockSize)
		}
	}
	if perBlock <= 0 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: per-block KV resolved to %d bytes; a budget cannot be divided "+
				"by it", perBlock)
	}

	blocks := allocatable / perBlock
	if blocks <= 0 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: No available memory for the cache blocks. Try increasing "+
				"`gpu_memory_utilization` when initializing the engine (computed 0 KV blocks: "+
				"allocatable %.2f GiB, per block %d bytes)",
			float64(allocatable)/float64(gibToBytes), perBlock)
	}

	// DP scaling: dp independent EngineCores each hold a full budget and requests split
	// disjointly across them. Gated on MoE -- a dense model's data parallelism is
	// replicas, so it is never scaled.
	perRank := blocks
	dp := m.k.Resolved().DataParallel()
	scaled := false
	if m.id.experts > 0 && dp > 1 {
		blocks *= int64(dp)
		scaled = true
	}

	adm, ok := m.k.(admissionReserver)
	if !ok {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: the kernel (%T) does not report the per-sequence admission "+
				"reservation (SequenceAdmissionBytes), which the KV budget charges", m.k)
	}
	seqFixed := adm.SequenceAdmissionBytes()
	steady := m.k.SequenceFixedBytes()
	if seqFixed < 0 {
		return KVBudget{}, fmt.Errorf(
			"kernelmodel: per-sequence admission reservation resolved to %d bytes", seqFixed)
	}
	// A pool too small for one sequence is refused by Settings, as vLLM refuses it at start-up
	// (checkEnoughKVCacheMemory).
	seqFixedBlocks := (seqFixed + perBlock - 1) / perBlock

	return KVBudget{
		TotalBlocks:             blocks,
		PerRankBlocks:           perRank,
		BlockSize:               blockSize,
		DeviceBytes:             mem.DeviceBytes,
		BudgetBytes:             mem.BudgetBytes,
		FixedBytes:              mem.FixedBytes,
		AllocatableBytes:        allocatable,
		PerBlockBytes:           perBlock,
		DPScaled:                scaled,
		SequenceAdmissionBytes:  seqFixed,
		SequenceAdmissionBlocks: seqFixedBlocks,
		SequenceSteadyBytes:     steady,
		SequenceSteadyBlocks:    (steady + perBlock - 1) / perBlock,
	}, nil
}

// KVMemory is a rank's memory for its KV cache: the device, the engine's share of it, the
// kernel's request-independent occupancy and what is left. It is KVBudget's memory half, which
// needs no page size: an engine that cuts the pool into KV cache groups sizes its blocks from
// every group's pages (KVLayerPageBytes), window rings included, so a model whose every layer is
// bounded -- whose growing-context page (SequenceVariableBytes) is zero -- still has a pool.
type KVMemory struct {
	DeviceBytes, BudgetBytes, FixedBytes, AllocatableBytes int64
	// ReservedBytes is the static LoRA adapter reservation taken out of this rank before its pool
	// was sized (reservedBytes/tp), zero when none was. It is stated because AllocatableBytes on
	// its own cannot say what happened to the difference between the budget, the fixed occupancy
	// and the pool.
	ReservedBytes int64
}

// KVMemory is the pool's rank's KV memory at the engine's gpu_memory_utilization, refusing what
// KVBudget refuses before it divides by a page.
func (m *Model) KVMemory() (KVMemory, error) { return m.KVMemoryReserving(0) }

// KVMemoryReserving is KVMemory with reservedBytes of the rank's HBM set aside before KV is
// sized, as KVBudgetReserving takes it: the static LoRA adapter reservation.
func (m *Model) KVMemoryReserving(reservedBytes int64) (KVMemory, error) {
	return m.kvMemory(reservedBytes)
}

// kvMemory is KVMemoryReserving's body. AllocatableBytes is the budget less the occupancy the
// profiler charges -- FixedBytes, or FixedBytes less the CUDA graph capture where
// VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS is 0 (profiled, below) -- and less the reservation; a
// stated kv_cache_memory_bytes is the exception, because the worker reserves it whole and skips
// the profiling, and with it the reservation.
func (m *Model) kvMemory(reservedBytes int64) (KVMemory, error) {
	if reservedBytes < 0 {
		return KVMemory{}, fmt.Errorf("kernelmodel: reserved bytes must be >= 0, got %d", reservedBytes)
	}
	// From the kernel, which resolved this pool. Reading the Deployment document instead
	// would mean holding it alongside and indexing back in -- a second answer to "which
	// pool is this", which can disagree with the one the pricing used.
	e := m.engine
	util := e.GPUMemoryUtilization
	if util <= 0 || util > 1 {
		return KVMemory{}, fmt.Errorf(
			"kernelmodel: gpu_memory_utilization must be in (0, 1], got %v", util)
	}
	if e.BlockSize <= 0 {
		return KVMemory{}, fmt.Errorf(
			"kernelmodel: block_size must be > 0, got %d", e.BlockSize)
	}

	deviceBytes := int64(m.id.memoryGiB * float64(gibToBytes))
	// vLLM requests ceil(total memory x utilization) bytes (request_memory, vllm/v1/worker/utils.py:724-726@5df63a72).
	budget := int64(math.Ceil(float64(deviceBytes) * util))
	fixedBytes := m.k.FixedBytes()
	fixed := fixedBytes.Total()
	graphCapture := fixedBytes.CUDAGraph

	// A per-rank occupancy at or above the device's own memory leaves no KV at all, and no budget
	// can be divided out of it. Two things produce one, and the message says which it can tell:
	// a deployment too large for the part, whose weights alone do not fit (the figure is right),
	// and a sharding term wrong in the kernel's memory path (the figure is not). The weights
	// alone against the device tells them apart: weights that do not fit the part are the
	// deployment's, and an occupancy over the part with weights that fit is the kernel's other
	// terms, which this cannot attribute. Neither is guessed past.
	if fixed >= deviceBytes {
		fb := m.k.FixedBytes()
		tp, ep := m.k.Resolved().TensorParallel(), m.k.Resolved().ExpertParallel()
		if fb.Weights >= deviceBytes {
			return KVMemory{}, fmt.Errorf(
				"kernelmodel: %s does not fit %.1f GiB devices at tp=%d (expert-parallel width %d): "+
					"one rank's weights alone are %.1f GiB, and its fixed occupancy %.1f GiB; "+
					"widen tensor or expert parallelism or use a larger part. No KV budget exists",
				m.id.model, float64(deviceBytes)/float64(gibToBytes), tp, ep,
				float64(fb.Weights)/float64(gibToBytes), float64(fixed)/float64(gibToBytes))
		}
		return KVMemory{}, fmt.Errorf(
			"kernelmodel: the kernel reports %.1f GiB of per-rank fixed occupancy on a "+
				"%.1f GiB device (%s, tp=%d, expert-parallel width %d), of which the weights are "+
				"%.1f GiB. The weights fit, so a term other than the weights is larger than the part, "+
				"and a sharding term in the kernel's memory path may be wrong. No budget is guessed: "+
				"one divided out of an impossible occupancy would look plausible and set the wrong resident batch",
			float64(fixed)/float64(gibToBytes), float64(deviceBytes)/float64(gibToBytes),
			m.id.model, tp, ep, float64(fb.Weights)/float64(gibToBytes))
	}

	if stated := e.KVCacheMemoryBytes; stated != nil {
		// A stated kv_cache_memory_bytes is the KV cache's bytes on each GPU, taken whole: the worker
		// reserves exactly that many and skips memory profiling, so neither gpu_memory_utilization nor the
		// weights decide it (vllm/v1/worker/gpu_worker.py:656-681@4ba787ec; vllm/config/cache.py:374-382).
		// The static LoRA adapter reservation is not taken from it: the profiling that nets it out
		// is skipped (the multimodal IPC reservation, reserve_mm_ipc_gpu_memory, applies to a
		// multimodal model only). The rank's fixed occupancy and the stated cache must fit the
		// device, or the allocation fails.
		if fixed+*stated > deviceBytes {
			return KVMemory{}, fmt.Errorf(
				"kernelmodel: kv_cache_memory_bytes %d (%.1f GiB) and the rank's fixed occupancy %.1f GiB exceed "+
					"the %.1f GiB device, so the engine could not allocate the cache it was told to reserve",
				*stated, float64(*stated)/float64(gibToBytes), float64(fixed)/float64(gibToBytes),
				float64(deviceBytes)/float64(gibToBytes))
		}
		return KVMemory{DeviceBytes: deviceBytes, BudgetBytes: fixed + *stated, FixedBytes: fixed,
			AllocatableBytes: *stated}, nil
	}
	// Read only on the path vLLM reads it: a stated kv_cache_memory_bytes returned above, as the worker
	// returns before it profiles (gpu_worker.py:655-686@4ba787ec), so its value is never looked at.
	graphsEstimated, err := profilerEstimatesGraphs(e.Env)
	if err != nil {
		return KVMemory{}, err
	}
	profiled := fixed
	if !graphsEstimated {
		profiled -= graphCapture
	}
	// The reservation is a total across the rank's tensor-parallel GPUs, sharded as the weights
	// are (KVBudgetReserving), which is the share this rank gives up.
	reserved := reservedBytes / int64(m.k.Resolved().TensorParallel())
	allocatable := budget - profiled - reserved
	if allocatable <= 0 {
		// vLLM's start-up refusal when nothing is left for the cache
		// (_check_enough_kv_cache_memory, vllm/v1/core/kv_cache_utils.py:904-913 at v0.31.0).
		return KVMemory{}, fmt.Errorf(
			"kernelmodel: No available memory for the cache blocks. Try increasing "+
				"`gpu_memory_utilization` when initializing the engine (no room for KV: a %.1f GiB "+
				"device at utilization %.2f gives a %.1f GiB budget against %.1f GiB of fixed "+
				"occupancy per rank)",
			float64(deviceBytes)/float64(gibToBytes), util,
			float64(budget)/float64(gibToBytes), float64(fixed)/float64(gibToBytes))
	}
	return KVMemory{DeviceBytes: deviceBytes, BudgetBytes: budget, FixedBytes: fixed, AllocatableBytes: allocatable,
		ReservedBytes: reserved}, nil
}

// admissionReserver is the kernel's bounded per-sequence state, exported by
// blis-latency-kernel beside the published kernel.Kernel interface (https://github.com/inference-sim/blis-latency-kernel/issues/46,
// #2614).
type admissionReserver interface {
	SequenceAdmissionBytes() int64
	SequenceBoundedBytesAt(computedTokens, inFlightTokens int) int64
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

// StageBudget is one pipeline rank's KV budget: the memory its KV pool may take and one block
// of every KV layer it holds. vLLM sizes each worker's pool from its own free memory over its
// own layers and keeps the smallest (get_kv_cache_configs, vllm/v1/core/kv_cache_utils.py:
// 2531-2680@5df63a72), and the stages' weights and layers differ, so a pipeline pool is sized
// stage by stage from these.
type StageBudget struct {
	// AllocatableBytes is the stage rank's budget less its own fixed occupancy.
	AllocatableBytes int64
	// PageBytes is one block of every KV layer the stage holds whose cache the kernel infers,
	// window rings included: the bytes its share of the KV cache groups is cut from.
	PageBytes int64
}

// stageMemorist is the kernel's per-stage memory under pipeline parallelism, exported by
// blis-latency-kernel beside the published kernel.Kernel interface.
type stageMemorist interface {
	PipelineStageMemory() []latencykernel.StageMemory
	KVLayerPageBytes(blockTokens int) int64
}

// KVLayerPageBytes is one block of every KV layer the pool's rank holds whose cache the kernel
// infers, window rings included (blis-latency-kernel Kernel.KVLayerPageBytes): what an engine
// model cutting the pool into KV cache groups derives each layer's element bytes from. It is
// PerBlockBytes where every layer's cache grows with context.
func (m *Model) KVLayerPageBytes() (int64, error) {
	s, ok := m.k.(stageMemorist)
	if !ok {
		return 0, fmt.Errorf("kernelmodel: the kernel (%T) does not report its KV layers' page bytes", m.k)
	}
	return s.KVLayerPageBytes(m.engine.BlockSize), nil
}

// StageBudgets is every pipeline rank's KV budget, in stage order, at the engine's
// gpu_memory_utilization; nil at pp 1, where KVBudget is the one rank's.
func (m *Model) StageBudgets() ([]StageBudget, error) {
	s, ok := m.k.(stageMemorist)
	if !ok {
		return nil, fmt.Errorf("kernelmodel: the kernel (%T) does not report its pipeline stages' memory", m.k)
	}
	stages := s.PipelineStageMemory()
	if len(stages) == 0 {
		return nil, nil
	}
	// The memory half only: a stage's pool is cut from its groups' pages (PageBytes), so an
	// all-window model, whose growing-context page is zero, is still sized.
	b, err := m.KVMemory()
	if err != nil {
		return nil, err
	}
	out := make([]StageBudget, len(stages))
	for i, st := range stages {
		allocatable := b.BudgetBytes - st.Fixed.Total()
		if allocatable <= 0 {
			return nil, fmt.Errorf("kernelmodel: No available memory for the cache blocks. Try "+
				"increasing `gpu_memory_utilization` when initializing the engine (pipeline stage "+
				"%d holds %.1f GiB of fixed occupancy against a %.1f GiB budget)", i,
				float64(st.Fixed.Total())/float64(gibToBytes), float64(b.BudgetBytes)/float64(gibToBytes))
		}
		out[i] = StageBudget{AllocatableBytes: allocatable, PageBytes: st.LayerPageBytes}
	}
	return out, nil
}

// envProfilerEstimatesGraphs is the variable that decides whether vLLM's memory profiler counts the
// CUDA graphs it will capture against the KV budget.
const envProfilerEstimatesGraphs = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"

// profilerEstimatesGraphs reads VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS as vLLM v0.33.0 does,
// bool(int(value)) with a default of 1 (vllm/envs.py:2219-2221@4ba787ec). When it is 1 the profiler
// estimates the graph memory and the worker takes it out of the KV budget with the weights and the
// peak activation; when it is 0 the estimate is dropped (cudagraph_memory_estimate_applied = 0, and
// available_kv_cache_memory_bytes subtracts it, vllm/v1/worker/gpu_worker.py:720-724 and :753-757@4ba787ec)
// and the graphs, captured after the cache is sized, come out of the memory the utilization leaves free, so the KV pool is larger by the capture cost.
// A value Python's int() rejects stops vLLM at its first read of the variable, so it is refused. The
// forms int() accepts that a run can state are an optional sign and ASCII digits with single underscores
// between digits, around optional white space ("1", " 0 ", "+1", "1_0"); a form int() accepts that is
// outside those (non-ASCII digits) is refused by name rather than read.
func profilerEstimatesGraphs(env map[string]string) (bool, error) {
	v, ok := env[envProfilerEstimatesGraphs]
	if !ok {
		return true, nil
	}
	nonzero, ok := pythonIntNonzero(v)
	if !ok {
		return false, fmt.Errorf("kernelmodel: %s=%q is not a form this engine reads as vLLM's int() does "+
			"(an optional sign and ASCII digits, single underscores between digits; vllm/envs.py:2219-2221@4ba787ec)",
			envProfilerEstimatesGraphs, v)
	}
	return nonzero, nil
}

// pythonIntNonzero reports whether Python's int(v) is nonzero, and whether v is a form it reads: optional
// white space, an optional sign, then ASCII digits with single underscores only between digits.
func pythonIntNonzero(v string) (nonzero, ok bool) {
	t := strings.TrimSpace(v)
	if strings.HasPrefix(t, "+") || strings.HasPrefix(t, "-") {
		t = t[1:]
	}
	if t == "" || t[0] == '_' || t[len(t)-1] == '_' || strings.Contains(t, "__") {
		return false, false
	}
	for _, r := range t {
		switch {
		case r == '_':
		case r >= '0' && r <= '9':
			if r != '0' {
				nonzero = true
			}
		default:
			return false, false
		}
	}
	return nonzero, true
}
