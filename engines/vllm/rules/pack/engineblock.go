// Original to blis-vllm.

package pack

import (
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// CheckEngineBlock implements rules.EngineRules: it opens a pool's engine block as vLLM's
// engine configuration (engineconfig.Decode, strictly) and runs its field checks
// (engineconfig.Validate), at the block's paths below at + ".engine", with the messages
// blis-schemas reported before it held the block unopened. A block that does not decode is
// one finding at at + ".engine", naming the key at its line.
func (p *Pack) CheckEngineBlock(at string, b deployment.EngineBlock) *rules.Problems {
	out := &rules.Problems{}
	e, err := engineconfig.Decode(b)
	if err != nil {
		out.Field(at+".engine", "%v", err)
		return out
	}
	engineconfig.Validate(out, at, e)
	return out
}

// DraftTokens implements rules.EngineRules: the speculative tokens a pool's engine block
// drafts per step, zero when it states no drafter or does not decode.
func (p *Pack) DraftTokens(b deployment.EngineBlock) int {
	e, err := engineconfig.Decode(b)
	if err != nil || e.Speculative == nil {
		return 0
	}
	return e.Speculative.NumSpecTokens
}
