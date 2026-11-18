// Original to blis-sglang.

package lower

import (
	"cmp"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-sglang/config"
)

// chipMemoryMiB is what nvidia-smi reports as memory.total for each chip the catalog names and the
// oracle's cases state (oracle/serverargs/cases.json "chips", each with its source). SGLang reads
// this figure, not the catalog's GiB, in get_device_memory_capacity (utils/common.py@297e54fd), so
// a chip with no entry is refused rather than converted by a guess.
var chipMemoryMiB = map[string]float64{
	"H100": 81559,
	"H200": 143771,
	"B200": 183359,
	"B300": 275040,
}

// ChipOf is the chip as SGLang's resolution reads it: the compute capability from the catalog chip
// and the nvidia-smi memory the oracle's cases state for it.
func ChipOf(c *hardware.Chip) (config.Chip, error) {
	major, minor, ok := capability(c.ComputeCapability)
	if !ok {
		return config.Chip{}, fmt.Errorf("chip %q states no ComputeCapability (%q), which SGLang's resolution reads", c.Name, c.ComputeCapability)
	}
	mib, ok := chipMemoryMiB[strings.ToUpper(c.Name)]
	if !ok {
		return config.Chip{}, &NotLowered{Name: "chip " + c.Name, Issue: issueLower}
	}
	return config.Chip{Capability: [2]int{major, minor}, MemoryMiB: mib}, nil
}

func capability(s string) (major, minor int, ok bool) {
	a, b, found := strings.Cut(s, ".")
	if !found {
		return 0, 0, false
	}
	major, err1 := strconv.Atoi(a)
	minor, err2 := strconv.Atoi(b)
	return major, minor, err1 == nil && err2 == nil && major > 0
}

// archFacts are the facts of a checkpoint's configuration that the catalog graph does not carry,
// by the architecture SGLang serves it with (the oracle records them for each model it resolves:
// model_facts in config/testdata/serverargs.json, compared in TestModelFactsMatchTheOracle).
type archFacts struct {
	modelType     string
	dsa           bool
	dsaIndexKpool *int
	asymmetricKV  bool
	// fp8Block is the block size of the architecture's fp8 checkpoint
	// (quantization_config.weight_block_size), which the graph's weight_dtype does not carry.
	fp8Block []int
	// multimodal is a model whose checkpoint carries a vision tower: textModelType is the language
	// model's own type, and the tower's depth and width are the vision_config's (the facts
	// adjust_mem_fraction_for_vlm reads).
	multimodal                 bool
	textModelType              string
	visionLayers, visionHidden int
}

func ptr[T any](v T) *T { return &v }

var archTable = map[string]archFacts{
	"DeepseekV3ForCausalLM": {modelType: "deepseek_v3", asymmetricKV: true, fp8Block: []int{128, 128}},
	"GlmMoeDsaForCausalLM":  {modelType: "glm_moe_dsa", dsa: true, dsaIndexKpool: ptr(1), fp8Block: []int{128, 128}},
	// Qwen3.5 (qwen35.go): a hybrid of Gated DeltaNet and attention layers with a vision tower.
	"Qwen3_5MoeForConditionalGeneration": {modelType: "qwen3_5_moe", fp8Block: []int{128, 128},
		multimodal: true, textModelType: "qwen3_5_moe_text", visionLayers: 24, visionHidden: 1152},
}

// ModelFactsOf is the model as SGLang's resolution reads it, from the catalog graph. The graph
// states the architecture, the hidden width, the served weight format and its block scale; the
// rest is the architecture's own (archTable). An architecture, or a weight format, outside what the
// oracle covers is refused by name.
func ModelFactsOf(g *model.Graph) (config.ModelFacts, error) {
	arch := g.Global.Architecture
	t, ok := archTable[arch]
	if !ok {
		return config.ModelFacts{}, &NotLowered{Name: "architecture " + arch, Issue: issueLower}
	}
	hidden := g.Global.HiddenSize
	f := config.ModelFacts{
		Architecture: arch, ModelType: t.modelType, TextModelType: cmp.Or(t.textModelType, t.modelType),
		AttentionArch: "MHA", IsDeepseekDSA: t.dsa, DSAIndexKpool: t.dsaIndexKpool,
		HasAsymmetricKV: t.asymmetricKV, HiddenSize: &hidden,
	}
	if t.multimodal {
		f.IsMultimodal, f.MultimodalBreakableGraphSupported, f.HasVisionConfig = true, true, true
		f.VisionNumHiddenLayers, f.VisionHiddenSize = ptr(t.visionLayers), ptr(t.visionHidden)
	}
	g.EachNode(func(_ string, n model.Node) {
		if n.Op == model.OpAttention && (n.AttentionKind == model.AttentionMLA || n.AttentionKind == model.AttentionSparseMLA) {
			f.AttentionArch = "MLA"
		}
	})
	// _deepseek_moe_quant_resolution reads the first expert tensor of a DeepseekV3ForCausalLM
	// checkpoint's safetensors header when the checkpoint declares no quantization. The lowering has
	// no header: this fact is read from the graph's weight dtype (an fp8 graph stands for a
	// float8_e4m3fn header). A graph whose dtype is bf16 for a checkpoint whose header is fp8 would
	// differ from SGLang here. The catalog graph is derived from the checkpoint's config.json, not
	// its safetensors header, so this holds only as far as the weight dtype follows the header;
	// TestBlackwellModelFactsMatchTheOracle holds the four catalog graphs of DeepSeek-R1 and GLM-5
	// to the oracle's reading of the header.
	f.HasFP8WeightsInCheckpoint = arch == "DeepseekV3ForCausalLM" && g.Global.WeightDType == model.DTypeFP8
	switch g.Global.WeightDType {
	case model.DTypeBF16:
	case model.DTypeFP8:
		// The fp8 checkpoint of these architectures is block-quantized (the oracle's model_facts
		// record weight_block_size); a graph that states a scale of another block is a different
		// quant method.
		if s := g.Global.WeightScale; t.fp8Block == nil || s != nil && s.Elements != t.fp8Block[0]*t.fp8Block[1] {
			return config.ModelFacts{}, &NotLowered{Name: "fp8 weights of another block scale", Issue: issueLower}
		}
		m := "fp8"
		f.QuantMethod, f.QuantWeightBlockSize = &m, slices.Clone(t.fp8Block)
	default:
		return config.ModelFacts{}, &NotLowered{Name: "weight format " + string(g.Global.WeightDType), Issue: issueLower}
	}
	return f, nil
}
