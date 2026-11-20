// Original to blis-vllm.

package lowering

import (
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/units"
)

// The bounds vLLM v0.33.0 compares a step's tokens with before it splits Muse Spark's MoE
// pooler and unpooler: _SHARDED_PROJECTION_MAX_TOKENS and _MULTI_NODE_SHARDED_UNPOOLER_MIN_TOKENS
// (vllm/models/muse_spark/nvidia/model.py:151, :154@4ba787ec9959).
const (
	museShardedProjectionMaxTokens  units.Tokens = 256
	museMultiNodeShardedUnpoolerMin units.Tokens = 32
)

// museSparkArchitectures are the architectures vLLM v0.33.0 builds with MuseSpark1xModel
// (MuseSpark1xForCausalLM and MuseSparkForConditionalGeneration, vllm/config/vllm.py:101-102).
var museSparkArchitectures = map[string]bool{
	"MuseSpark1xForCausalLM":            true,
	"MuseSparkForConditionalGeneration": true,
}

// decodeShards is the replicated GEMMs of the model whose output columns the engine splits over
// the tensor-parallel ranks and all-gathers, and the steps it does so for. A Muse Spark layer
// takes this path when it computes on replicated tokens (tep_default, model.py:505-523): a part
// of compute capability 10 or later, tensor parallelism above one, no data parallelism, no
// sequence parallelism and no mega-MoE. Its MoE pooler then splits for a step of at most 256
// tokens (shards_pooler, :635-636) and its unpooler for the same, from 32 tokens where the
// tensor-parallel group spans hosts (shards_unpooler, min_sharded_unpooler_tokens, :535-539,
// :638-644). Batch invariance, which would turn it off, has no deployment field. The host test is
// vLLM's tp_group_spans_nodes (vllm/models/muse_spark/nvidia/ops/mnnvl_fusions.py:21-23): a group
// over two trays of one NVLink rack spans hosts though its gather never leaves NVLink. No other
// architecture splits a replicated projection.
func decodeShards(arch string, family int, lay layout, fusedMoE bool) []plan.DecodeShard {
	if !museSparkArchitectures[arch] || family < 10 || lay.TP <= 1 || lay.DP > 1 ||
		lay.SequenceParallelMoE || lay.ModelSequenceParallel || fusedMoE {
		return nil
	}
	unpoolerFrom := units.Tokens(1)
	if lay.GPUsPerNode > 0 && lay.TP > lay.GPUsPerNode {
		unpoolerFrom = museMultiNodeShardedUnpoolerMin
	}
	return []plan.DecodeShard{
		{Role: "routed_expert_down_proj", FromTokens: 1, ToTokens: museShardedProjectionMaxTokens},
		{Role: "routed_expert_up_proj", FromTokens: unpoolerFrom, ToTokens: museShardedProjectionMaxTokens},
	}
}
