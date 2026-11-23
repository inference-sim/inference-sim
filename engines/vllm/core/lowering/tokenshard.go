// Original to blis-vllm.

package lowering

import (
	"slices"

	"github.com/inference-sim/inference-sim/substrate/contracts/enginefacts"
)

// tokenShardedByArchitecture is the roles of the nodes vLLM v0.33.0 runs on a rank's shard of the
// step's tokens when the model's own sequence parallelism is on, per architecture. The residual
// stream is cut by sp_shard (vllm/models/common/ops/sequence_parallel.py) and gathered back only
// around attention, so what runs outside attention -- the mHC, the router and hash gate, the
// routed and shared experts and, in V4.1, Engram's wkv and gate, the hyper-connection collapse
// and the final norm -- sees ceil(tokens / tp) rows (vllm/models/deepseek_v41/nvidia/model.py,
// vllm/models/deepseek_v41/common/engram.py; DeepSeek-V4's table is the same stream with the
// roles its graphs state, MTP and DSpark drafters included). The tables were carried over from
// the catalog graphs' former under_sequence_parallel markings (SUB-15). A table spans the variants
// of its architecture, so a role one variant does not have is not an error.
var tokenShardedByArchitecture = map[string][]string{
	enginefacts.ArchDeepseekV4: {
		"dspark_mhc_final_post", "experts", "hash_route", "hc_attn_fn", "hc_attn_fn_broadcast",
		"hc_ffn_fn", "mhc_attn_broadcast_pre", "mhc_attn_post_pre", "mhc_attn_pre",
		"mhc_ffn_post_pre", "mhc_final_post", "mtp_e_proj", "mtp_h_proj", "mtp_input_add",
		"mtp_mhc_post", "router", "shared_down", "shared_gate_up",
	},
	enginefacts.ArchDeepseekV41: {
		"engram_gate", "engram_wkv", "experts", "final_norm", "hc_attn_fn", "hc_attn_fn_broadcast",
		"hc_collapse", "hc_ffn_fn", "mhc_attn_broadcast_pre", "mhc_attn_post", "mhc_attn_post_pre",
		"mhc_attn_pre", "mhc_ffn_post_pre", "mhc_final_post", "router", "shared_down", "shared_gate_up",
	},
}

// tokenShardedRoles is the roles of the architecture's nodes that run on a token shard, a copy
// the caller may keep. Nothing for an architecture whose own sequence parallelism the engine
// does not model; the plan states them only where that parallelism is on (see the lowering).
func tokenShardedRoles(arch string) []string {
	return slices.Clone(tokenShardedByArchitecture[arch])
}
