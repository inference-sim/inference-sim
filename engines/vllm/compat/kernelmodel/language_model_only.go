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
// the pinned catalog and requires every catalog model whose config.json has a vision_config to be
// listed here or in pricedTowerConfigs, so a model cannot be left out unnoticed.
//
// What "no tower node" rests on. It is a reading of the catalog's graph: no node of any listed
// model is named vision, tower, projector, merger, patch_merge or vit. It has not been compared
// with the weights vLLM leaves resident: that needs vLLM's model classes built on a meta device
// with their tensor-parallel groups, which these CUDA-oriented classes do not allow on the CPU
// build the oracle runs on, and the catalog holds no checkpoint tensor index to count against.
// So the claim is that the catalog graph is the text decoder, not that its weight bytes equal
// vLLM's weights under language_model_only true; the test comments and fields.json say the same.
//
// A model not listed is treated as text-only here: any value is accepted and nothing is
// disclosed.
var towerlessMultimodalConfigs = map[string]string{
	"8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879": "deepseek-v4.1-flash",
	"bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f": "glm-5.3-flash",
	"58720f145bcecef9a7ab2b419ab346e7c634af8d2f3e7362e900d00f789ea46c": "inkling",
	"acd5bb01a16f64b309599cd6ed196be056f613c99d6bc9300692b82cd10882f6": "kimi-k2.5",
	"85825ca6e18cbe539eb83ee09eedfb3f4222265929f06e9f535a6d9364f55899": "kimi-k2.6",
	"9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213": "kimi-k3",
	"d4fbcc4bcefaf437121d74d355fe9bbe1d7bfd4262b15e1ed9ca75febafdd989": "llama-4-scout-17b-16e-instruct-fp8-dynamic",
	"c9c97ce1e4eece60012d5a10ea87717458bfb1f19c2c7a615a3dbff83d090c6b": "minimax-m3",
	"a3698fff231a6cd92f632dcbe50f0b1dbda716994f29ccde9f0fa9d2a87698bf": "minimax-m3-mxfp8",
	"440d22038ffa0bcb93fa01e8d2d555e3840d939167c08eace81b9cce386fb117": "minimax-m3-nvfp4",
	"4e91fd887cd1f61e59d717f1ec5dea5fa82f4c01f3dca8d7d96b5b29b7b25ea3": "muse-spark-open-816b-a42b",
	"439857095414fb030946020aea144b8d3545ce95b704893b341548db2e466567": "muse-spark-open-816b-a42b-nvfp4",
	"3ae7fa89c2f7d1354096418ddaf1331e9e0898a8ca11e804fb6dbaa087efb7da": "qwen3.5-397b-a17b",
	"9a142fc6c2d67703255ce6120ee04c4c720ec58dba6ae44bebb29c4a2d6e5db8": "qwen3.5-397b-a17b-fp8",
	"93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99": "qwen3.6-35b-a3b",
}

// pricedTowerConfigs is the partner list: catalog models whose config.json has a vision_config and
// whose graph names a vision tower, projector or patch merge, so the kernel prices a tower and
// language_model_only true would change the price. The rule does not cover them: it discloses
// nothing for them and does not claim true is free. It is empty at the pinned catalog; the test
// fails when a model with a vision_config is in neither list.
var pricedTowerConfigs = map[string]string{}

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
// only (a reading of the graph, not a comparison with vLLM's resident weights; see that list), so the kernel's resident weights, KV blocks and step time are those of language_model_only
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
