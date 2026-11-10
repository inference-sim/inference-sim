// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/config/engram.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// EngramConfig._validate_shared_memory, resolve_dp_shared_memory, verify_parallel_config and
// get_parallel_size (:56-117).
// Also vllm/models/deepseek_v41/common/engram.py (ParallelEngramEmbedding, :942-1060, whose cpu_offload decides where the table lives, :1045-1052) and
// vllm/v1/worker/gpu/spec_decode/speculator.py (the draft-logits buffer, :178-190, 403-409).

package lower

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"
)

// EngramShared is EngramConfig's _validate_shared_memory on the stated values, then
// resolve_dp_shared_memory and verify_parallel_config: whether the host tables are shared between
// the node's data-parallel replicas. Unstated, they are shared whenever the tables are offloaded,
// transparent huge pages are off, there is more than one replica and the pool is not elastic. A
// setting that cannot hold is refused as vLLM refuses it (the oracle runs all three).
func EngramShared(stated *bool, cpuOffload, useTHP bool, dp int, elasticEP bool) (bool, error) {
	if stated != nil && *stated && !cpuOffload {
		return *stated, fmt.Errorf("lower: dp_shared_memory requires cpu_offload=True (vllm/config/engram.py:58-59)")
	}
	if useTHP && (!cpuOffload || (stated != nil && *stated)) {
		return stated != nil && *stated, fmt.Errorf("lower: use_thp requires cpu_offload=True and dp_shared_memory=False (vllm/config/engram.py:60-63)")
	}
	shared := false
	if stated != nil {
		shared = *stated
	} else {
		shared = cpuOffload && !useTHP && dp > 1 && !elasticEP
	}
	if shared {
		if dp <= 1 {
			return shared, fmt.Errorf("lower: dp_shared_memory requires data_parallel_size > 1 (vllm/config/engram.py:93)")
		}
		if elasticEP {
			return shared, fmt.Errorf("lower: dp_shared_memory is not supported with elastic EP (vllm/config/engram.py:95)")
		}
	}
	return shared, nil
}

// EngramParallelSize is EngramConfig.get_parallel_size: the embedding group is the tensor-parallel
// width, times the data-parallel width when the embedding is sharded across data parallelism.
func EngramParallelSize(tp, dp int, acrossDP bool) int {
	if acrossDP && dp > 1 {
		return tp * dp
	}
	return tp
}

// EngramShard is ParallelEngramEmbedding._get_shard_info: how many shards a table is cut into and
// which this rank holds. A shared table is cut over tensor parallelism alone (its data-parallel
// replicas read one copy), and an unshared one over tensor and data parallelism, tensor-major.
func EngramShard(tp, tpRank, engramDP, dpRank int, shared bool) (shards, rank int) {
	if shared || engramDP <= 1 {
		return tp, tpRank
	}
	return tp * engramDP, tpRank*engramDP + dpRank
}

// EngramTable is the tensor one rank's share of an Engram table makes in a PoolPlan: the rows of
// the hash heads this shard holds, each row `dim` fp8 values and one e8m0 scale per `block` of
// them. With cpuOffload it lives in pinned host memory; without, on the GPU (engram.py:1045-1052),
// and a table can be shared between replicas only in host memory. A shared table counts once for
// the node, an unshared one once for each rank (plan.MemoryOf). headSizes are the table's hash
// heads' row counts.
func EngramTable(name string, headSizes []int, dim, block, shards, rank int, shared, cpuOffload bool) (plan.ResidentTensor, error) {
	if shared && !cpuOffload {
		return plan.ResidentTensor{}, fmt.Errorf("lower: a shared Engram table lives in host memory, and cpu_offload is off")
	}
	if len(headSizes) == 0 || dim < 1 || block < 1 || dim%block != 0 || shards < 1 || rank < 0 || rank >= shards {
		return plan.ResidentTensor{}, fmt.Errorf("lower: an Engram table of %d heads, dim %d, block %d, shard %d of %d",
			len(headSizes), dim, block, rank, shards)
	}
	per := (len(headSizes) + shards - 1) / shards
	if (shards-1)*per >= len(headSizes) {
		return plan.ResidentTensor{}, fmt.Errorf("lower: Engram sharding leaves ranks without hash heads: %d heads over %d shards "+
			"(vllm/models/deepseek_v41/common/engram.py:993-996)", len(headSizes), shards)
	}
	start := rank * per
	end := min(start+per, len(headSizes))
	rows := 0
	for _, n := range headSizes[start:end] {
		rows += n
	}
	where := plan.ResidentGPU
	if cpuOffload {
		where = plan.ResidentHost
	}
	return plan.ResidentTensor{Name: name, Elements: int64(rows) * int64(dim),
		Format: units.QuantFormat{PayloadBits: 8, ScaleBits: 8, BlockElements: block},
		Where:  where, Shared: shared}, nil
}

// DraftLogits is the buffer a probabilistic draft sampler allocates when its speculator is built,
// before memory profiling, so inside the KV budget: (max_num_reqs, num_speculative_steps, vocab)
// in the head dtype (speculator.py:178-190). Greedy sampling allocates none. A speculator that
// writes only the candidate columns keeps the cache in float32 (dflash2, speculator.py:200-206),
// which the caller states with elemBytes.
func DraftLogits(probabilistic bool, maxNumSeqs, steps, vocab int, elemBytes units.Bytes) (plan.Buffer, bool, error) {
	if !probabilistic {
		return plan.Buffer{}, false, nil
	}
	if maxNumSeqs < 1 || steps < 1 || vocab < 1 || elemBytes < 1 {
		return plan.Buffer{}, false, fmt.Errorf("lower: draft logits of %d x %d x %d at %d bytes", maxNumSeqs, steps, vocab, elemBytes)
	}
	return plan.Buffer{Name: "draft_logits", Bytes: units.Bytes(maxNumSeqs) * units.Bytes(steps) * units.Bytes(vocab) * elemBytes,
		Basis: plan.BasisRank}, true, nil
}
