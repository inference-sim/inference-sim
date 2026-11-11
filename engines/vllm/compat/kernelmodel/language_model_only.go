// Original to blis-vllm.

package kernelmodel

import (
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// towerlessMultimodalConfigs are the sha256 of the vendor config.json of each catalog model that
// is a multimodal checkpoint (its config.json has a vision_config) and whose graph has no vision
// tower, projector or patch-merge node: the graph holds the text decoder alone. The graph states
// no architecture to key on (the architecture string for these models exists only in a later
// substrate layer), so the digest of the config it was derived from names the model, as
// glmMoeDsaConfigs does. TestTowerlessMultimodalConfigsNameTheCatalogModels checks each against
// the pinned catalog.
//
// Other multimodal checkpoints of the catalog are not listed; see the test for how the list was
// taken. A model not listed is treated as text-only here: any value is accepted and nothing is
// disclosed.
var towerlessMultimodalConfigs = map[string]string{
	"c9c97ce1e4eece60012d5a10ea87717458bfb1f19c2c7a615a3dbff83d090c6b": "minimax-m3",
	"a3698fff231a6cd92f632dcbe50f0b1dbda716994f29ccde9f0fa9d2a87698bf": "minimax-m3-mxfp8",
	"9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213": "kimi-k3",
}

// languageModelOnlyAssumptions is the rule for engine.language_model_only, which depends on the model.
//
// vLLM builds a MultiModalConfig only for a multimodal model (vllm/config/model.py:818-868@5df63a72,
// under supports_multimodal), so a text-only model is unaffected by the flag
// and every value is accepted. For a multimodal checkpoint, language_model_only true makes every
// modality's limit 0 (vllm/config/multimodal.py:696-700@5df63a72), and the model then wraps its
// vision tower and projector in no_init_weights with StageMissingLayer (_mark_tower_model,
// vllm/model_executor/models/interfaces.py:339-385@5df63a72; MiniMax-M3's vision_tower,
// vllm/models/minimax_m3/nvidia/model.py:1161@5df63a72; Kimi-K3's vision_tower and
// mm_projector, vllm/models/kimi_k3/nvidia/model.py:1783,1826@5df63a72): their weights are not loaded, and the
// scheduler gives the encoder no budget (vllm/v1/core/sched/scheduler.py:271-284@5df63a72).
//
// The catalog graph of each model listed in towerlessMultimodalConfigs holds the text decoder
// only, so the kernel's resident weights, KV blocks and step time are those of language_model_only
// true, with or without the key: true is accepted and changes no price. Unset or false is vLLM's
// default and keeps the tower resident, which the catalog does not price; those runs are accepted
// and the gap is returned as a disclosed assumption, because refusing them would refuse every
// committed deployment of these models that does not state the key.
func languageModelOnlyAssumptions(g *model.Graph, e engineconfig.Engine) []string {
	if g == nil {
		return nil
	}
	name, ok := towerlessMultimodalConfigs[g.DerivedFrom.SHA256]
	if !ok || (e.LanguageModelOnly != nil && *e.LanguageModelOnly) {
		return nil
	}
	stated := "unset"
	if e.LanguageModelOnly != nil {
		stated = "false"
	}
	return []string{"language_model_only = " + stated + ": " + name + " is a multimodal checkpoint, so vLLM keeps its vision " +
		"tower and projector resident and profiles the encoder, which the catalog graph does not price (it holds the text decoder " +
		"only); the weights, KV blocks and step time are those of language_model_only: true. State language_model_only: true " +
		"for the deployment the cost model prices"}
}
