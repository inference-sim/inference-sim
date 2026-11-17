// Original to blis-vllm.

package kernelmodel

import (
	"strconv"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	sim "github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// towerlessMultimodalConfigs are the sha256 of the vendor config.json of each catalog model that is
// a multimodal checkpoint whose graph prices no vision tower: its encoder section lists the tower
// under omitted, with the open issue that tracks it. The graph holds the text decoder alone for
// these. The digest of the config the graph was derived from names the model, as glmMoeDsaConfigs
// does. TestPricedAndTowerlessConfigsFollowTheCatalogGraphs requires every catalog graph with an
// encoder section to be listed here or in pricedTowerConfigs, so a model cannot be left out unnoticed.
//
// A model not listed is treated as text-only here, or as one whose graph prices its tower: any value
// is accepted and nothing is disclosed.
var towerlessMultimodalConfigs = map[string]string{
	"8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879": "deepseek-v4.1-flash",
	"58720f145bcecef9a7ab2b419ab346e7c634af8d2f3e7362e900d00f789ea46c": "inkling",
	"889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b": "qwen3.8-flash-next",
	"c22eb0a053eed62e18f0184d3ca62d3798f208c3e00b9feee939fc3dbacdc8ca": "qwen3.8-flash-next-fp8",
	"deef67a61f3311faf051b23dc4192f442c7fee4f9cd2f38cbcbe4da55c763a80": "qwen3.8-flash-next-nvfp4",
	"e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624": "qwen3.8-flash-next-nvfp4-radixark",
}

// pricedTowerConfigs is the partner list: catalog models whose graph prices a vision tower. The
// engine reports the tower resident unless language_model_only is true (compat/lowering/encoder.go),
// the kernel holds its weights and prices an encode step as a named substitution, so the rule here
// discloses nothing for them.
var pricedTowerConfigs = map[string]string{
	"ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3": "gemma-4-26b-a4b-it",
	"9781d0d5918127cc95a0a9f76040ea11c6dfd356e18a2db8ab1fdc2e2941dd42": "gemma-4-26b-a4b-it-fp8-dynamic",
	"e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1": "gemma-4-31b-it",
	"bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f": "glm-5.3-flash",
	"41db2811023b40ba4c8f8bbba88bce7dff377af51ecd18b32469a2a07064ebaf": "glm-5.3-flash-nvfp4",
	"acd5bb01a16f64b309599cd6ed196be056f613c99d6bc9300692b82cd10882f6": "kimi-k2.5",
	"c375d7d566b6fa706ba46fec9ef938e829c5ba1accb20541d4f0c75dddefcd8f": "kimi-k2.5-nvfp4",
	"85825ca6e18cbe539eb83ee09eedfb3f4222265929f06e9f535a6d9364f55899": "kimi-k2.6",
	"af4c0fea7ec43d5438532eeef8daa9fcabbdf97ddedc9da2319aa675f68f51ff": "kimi-k2.6-nvfp4",
	"9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213": "kimi-k3",
	"d95037631e3f23bf527a9f476f25209b231b46853a84877b9c56d5672b8d1202": "kimi-k3.1",
	"d4fbcc4bcefaf437121d74d355fe9bbe1d7bfd4262b15e1ed9ca75febafdd989": "llama-4-scout-17b-16e-instruct-fp8-dynamic",
	"c9c97ce1e4eece60012d5a10ea87717458bfb1f19c2c7a615a3dbff83d090c6b": "minimax-m3",
	"a3698fff231a6cd92f632dcbe50f0b1dbda716994f29ccde9f0fa9d2a87698bf": "minimax-m3-mxfp8",
	"440d22038ffa0bcb93fa01e8d2d555e3840d939167c08eace81b9cce386fb117": "minimax-m3-nvfp4",
	"df6a3668e2de5b8dc0e883e3767495ad17af7e4d69fabe9683088be015fda9ab": "minimax-m3.1-flash",
	"4e91fd887cd1f61e59d717f1ec5dea5fa82f4c01f3dca8d7d96b5b29b7b25ea3": "muse-spark-open-816b-a42b",
	"439857095414fb030946020aea144b8d3545ce95b704893b341548db2e466567": "muse-spark-open-816b-a42b-nvfp4",
	"3ae7fa89c2f7d1354096418ddaf1331e9e0898a8ca11e804fb6dbaa087efb7da": "qwen3.5-397b-a17b",
	"9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63": "qwen3.5-397b-a17b-nvfp4-v2",
	"9a142fc6c2d67703255ce6120ee04c4c720ec58dba6ae44bebb29c4a2d6e5db8": "qwen3.5-397b-a17b-fp8",
	"8058591f5ffb5bdacf6cdd87a0d1db126a12cf19259257aec25a91ec22918779": "qwen3.5-397b-a17b-nvfp4",
	"93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99": "qwen3.6-35b-a3b",
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
// only (a reading of the graph, not a comparison with vLLM's resident weights; see that list), so the
// kernel's resident weights, KV blocks and step time are those of language_model_only true, with or
// without the key: true is accepted and changes no price. Unset or false is vLLM's default and keeps the
// tower resident, which the catalog does not price. Those runs are accepted, because refusing them would
// refuse every committed deployment of these models that does not state the key, and the substitution
// is recorded where the Result carries it: Provenance().Substitutions, a machine-readable entry naming the
// setting, what the deployment states and the mode that is priced, which a grade reads (the free text of
// EngineAssumptions reaches only the host's stderr). State language_model_only: true for the deployment
// the cost model prices.
func languageModelOnlySubstitution(g *model.Graph, e engineconfig.Engine, pp int) *sim.Substitution {
	if g == nil || (e.LanguageModelOnly != nil && *e.LanguageModelOnly) {
		return nil
	}
	stated := "unset"
	if e.LanguageModelOnly != nil {
		stated = "false"
	}
	if _, towered := g.ImageTower(); towered && pp > 1 {
		// vLLM builds the tower on the first pipeline stage only; the kernel prices a resident tower at
		// pipeline-parallel width 1 only (kernel/new.go refuses the other), so the lowering states none
		// (compat/lowering/encoder.go) and the tower is the gap.
		return &sim.Substitution{Setting: "language_model_only", Stated: stated, Priced: "true",
			Reason: "vLLM keeps the vision tower resident on the first of the " + strconv.Itoa(pp) + " pipeline stages, and the " +
				"cost model prices a resident tower at pipeline-parallel width 1 only; the weights, KV blocks and step time are " +
				"those of language_model_only: true"}
	}
	name, ok := towerlessMultimodalConfigs[g.DerivedFrom.SHA256]
	if g.PricesEncoder() || !ok {
		return nil
	}
	return &sim.Substitution{Setting: "language_model_only", Stated: stated, Priced: "true",
		Reason: name + " is a multimodal checkpoint, so vLLM keeps its vision tower and projector resident and profiles the " +
			"encoder, which the catalog graph does not price (it holds the text decoder only); the weights, KV blocks and " +
			"step time are those of language_model_only: true"}
}

// languageModelOnlyAssumptions is the substitution as the human-readable assumption the host reports.
func languageModelOnlyAssumptions(g *model.Graph, e engineconfig.Engine, pp int) []string {
	s := languageModelOnlySubstitution(g, e, pp)
	if s == nil {
		return nil
	}
	return []string{s.Setting + " = " + s.Stated + ": " + s.Reason + ". State language_model_only: true for the deployment the cost model prices"}
}
