// Original to blis-vllm.

package lowering

import (
	"slices"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/enginefacts"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"

	"github.com/inference-sim/inference-sim/engines/vllm/core/vllmsettings"
)

// DeepSeek-V4 and V4.1 shard their residual stream by token across the tensor-parallel ranks when
// their own sequence parallelism is on (sp_shard, vllm/models/common/ops/sequence_parallel.py),
// and run the mHC, router, experts and shared experts of a decoder layer, and V4.1's Engram, on a
// rank's shard. Which nodes those are is the engine's lowering of the model (SUB-15), so the plan
// names them and the graph does not.
var (
	v4proSP = where{"deepseek-v4-pro", "b200", "tp: 2, pp: 1, dp: 2, enable_expert_parallel: true"}
)

func TestSequenceParallelDeepSeekStatesTheTokenShardedRoles(t *testing.T) {
	for _, at := range []where{v41sp, v4proSP} {
		l := lowered(t, at, vllmsettings.Settings{})
		if !l.Plan.Collectives.SequenceParallelModel {
			t.Fatalf("%s: model sequence parallelism is off, so the case tests nothing", at.model)
		}
		for _, want := range []string{"router", "experts", "shared_gate_up", "shared_down", "hc_attn_fn", "mhc_final_post"} {
			if !slices.Contains(l.Plan.TokenShardedRoles, want) {
				t.Errorf("%s: TokenShardedRoles = %v, want %q among them", at.model, l.Plan.TokenShardedRoles, want)
			}
		}
	}
	if !slices.Contains(lowered(t, v41sp, vllmsettings.Settings{}).Plan.TokenShardedRoles, "engram_wkv") {
		t.Error("V4.1 shards Engram's wkv, and its roles do not say so")
	}
	if slices.Contains(lowered(t, v4proSP, vllmsettings.Settings{}).Plan.TokenShardedRoles, "engram_wkv") {
		t.Error("V4 has no Engram, yet its roles name Engram's wkv")
	}
}

func TestNoTokenShardedRolesWithoutModelSequenceParallelism(t *testing.T) {
	for _, at := range []where{v41, v41dp, v4pro, dense, moeOne, denseB} {
		l := lowered(t, at, vllmsettings.Settings{})
		if l.Plan.Collectives.SequenceParallelModel {
			t.Fatalf("%s on %s: model sequence parallelism is on, so the case tests nothing", at.model, at.parallel)
		}
		if len(l.Plan.TokenShardedRoles) != 0 {
			t.Errorf("%s on %s: TokenShardedRoles = %v, want none", at.model, at.parallel, l.Plan.TokenShardedRoles)
		}
	}
}

// An architecture the engine has a sequence-parallelism rule for but no token-shard table
// states none: the table is the engine's knowledge of that model's code.
func TestTokenShardedRolesAreKeyedByArchitecture(t *testing.T) {
	if r := tokenShardedRoles(enginefacts.ArchDeepseekV4); len(r) == 0 {
		t.Error("DeepSeek-V4 has no token-sharded roles")
	}
	if r := tokenShardedRoles(enginefacts.ArchDeepseekV41); len(r) == 0 {
		t.Error("DeepSeek-V4.1 has no token-sharded roles")
	}
	if r := tokenShardedRoles("LlamaForCausalLM"); len(r) != 0 {
		t.Errorf("another architecture: %v, want none", r)
	}
	r := tokenShardedRoles(enginefacts.ArchDeepseekV41)
	r[0] = "changed"
	if tokenShardedRoles(enginefacts.ArchDeepseekV41)[0] == "changed" {
		t.Error("a caller edits the engine's table through the slice it was handed")
	}
}

// A table that names a role no catalog graph of its architecture has has drifted from the
// graphs: the role would shard nothing.
func TestEveryTokenShardedRoleIsInACatalogGraphOfItsArchitecture(t *testing.T) {
	seen := map[string]map[string]bool{}
	for _, name := range data.Models() {
		g, err := data.Model(name)
		if err != nil {
			t.Fatal(err)
		}
		arch := g.Global.Architecture
		if _, ok := tokenShardedByArchitecture[arch]; !ok {
			continue
		}
		if seen[arch] == nil {
			seen[arch] = map[string]bool{}
		}
		g.EachNode(func(_ string, n model.Node) { seen[arch][n.Role] = true })
	}
	for arch, roles := range tokenShardedByArchitecture {
		if len(seen[arch]) == 0 {
			t.Errorf("%s: no catalog graph has this architecture", arch)
		}
		for _, r := range roles {
			if !seen[arch][r] {
				t.Errorf("%s: role %q is in no catalog graph of the architecture", arch, r)
			}
		}
	}
}
