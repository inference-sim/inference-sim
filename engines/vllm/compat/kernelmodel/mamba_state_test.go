// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// kda is the Kimi-K3 catalog graph's KDA node, mamba2 the Nemotron-3-Ultra graph's and gdn the
// Qwen3.5-397B graph's (blis-catalog models/*/graph.yaml).
var (
	kda    = model.Node{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentKDA, NumHeads: 96, StateSize: 128, ConvKernel: 4, StateDType: model.DTypeFP32}
	mamba2 = model.Node{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2, NumHeads: 256, StateSize: 128,
		NumGroups: 8, ConvKernel: 4, IntermediateSize: 16384, StateDType: model.DTypeFP32}
	gdn = model.Node{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentGDN, NumHeads: 64, StateSize: 128, ConvKernel: 4,
		IntermediateSize: 12288, StateDType: model.DTypeFP32}
	mla = model.Node{Op: model.OpAttention, AttentionKind: model.AttentionMLA, NumQHeads: 96, NumKVHeads: 1, HeadDim: 576}
	gqa = model.Node{Op: model.OpAttention, AttentionKind: model.AttentionGQA, NumQHeads: 64, NumKVHeads: 2, HeadDim: 128}
)

// hybridGraph is a model of a recurrent layer and, when attn is not nil, an attention layer.
func hybridGraph(rec model.Node, attn *model.Node) *model.Graph {
	g := &model.Graph{
		Global:     model.GlobalShape{WeightDType: model.DTypeBF16, ContextWindow: 131072},
		LayerKinds: []model.LayerKind{{ID: "rec", Nodes: []model.Node{rec}}},
		Stack:      model.Stack{Pattern: []string{"rec"}, Repeat: 3},
	}
	if attn != nil {
		g.LayerKinds = append(g.LayerKinds, model.LayerKind{ID: "attn", Nodes: []model.Node{*attn}})
		g.Stack = model.Stack{Pattern: []string{"rec", "rec", "rec", "attn"}, Repeat: 2}
	}
	return g
}

func mustPlan(t *testing.T, g *model.Graph, e engineconfig.Engine, tp int) *mambaPlan {
	t.Helper()
	p, err := planMamba(g, e, mambaEnv{tp: tp, dcp: 1, pcp: 1, capability: 9, maxModelLen: 131072})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// The per-rank state bytes are the sum over the state tensors of their elements times their dtype,
// from vLLM's shape calculators (mamba_utils.py:195-327@5df63a72). Each case is worked by hand.
//
//	Kimi-K3, tp 8: conv (3 x 96 x 128 / 8 = 4608) x 3 = 13824 bf16 (27648 B), ssm 12 x 128 x 128 = 196608 fp32 (786432 B).
//	Nemotron-3-Ultra, tp 8: conv (16384 + 2 x 8 x 128) / 8 = 2304, x 3 = 6912 bf16 (13824 B), ssm 32 x 64 x 128 = 262144 fp32.
//	Qwen3.5-397B, tp 8: conv 12288 / 8 = 1536, x 3 = 4608 bf16 (9216 B), ssm 8 x 128 x 128 = 131072 fp32 (524288 B).
func TestRecurrentStateBytesAreTheCalculators(t *testing.T) {
	for _, c := range []struct {
		name string
		rec  model.Node
		e    engineconfig.Engine
		tp   int
		want int
	}{
		{"kimi-k3 tp8", kda, engineconfig.Engine{}, 8, 27648 + 786432},
		{"nemotron-3-ultra tp8", mamba2, engineconfig.Engine{}, 8, 13824 + 1048576},
		{"qwen3.5 tp8", gdn, engineconfig.Engine{}, 8, 9216 + 524288},
		// tp 1: the whole conv (36864 x 3 bf16) and the whole state (96 x 128 x 128 fp32).
		{"kimi-k3 tp1", kda, engineconfig.Engine{}, 1, 36864*3*2 + 96*128*128*4},
		// n_groups 1 at tp 4 is extended to 4 groups, one for each head shard
		// (extra_groups_for_head_shards): conv (16384 + 2 x 4 x 128) / 4 = 4352, x 3 bf16; ssm 64 x 64 x 128 fp32.
		{"mamba2 tp4 groups not divisible", withGroups(mamba2, 1), engineconfig.Engine{}, 4, 4352*3*2 + 64*64*128*4},
		// Speculative tokens widen the conv window: 3 + 3 = 6 rather than 3.
		{"nemotron tp8 3 spec tokens", mamba2, engineconfig.Engine{Speculative: &engineconfig.Speculative{Method: "mtp", NumSpecTokens: 3}}, 8,
			2304*6*2 + 1048576},
		// mamba_cache_dtype float32 widens the conv state; mamba_ssm_cache_dtype bfloat16 narrows KDA's.
		{"kimi-k3 conv fp32 ssm bf16", kda, engineconfig.Engine{MambaCacheDType: "float32", MambaSSMCacheDType: "bfloat16"}, 8,
			4608*3*4 + 196608*2},
		// A stated dtype equal to the resolved auto changes nothing.
		{"nemotron stated ssm fp32", mamba2, engineconfig.Engine{MambaSSMCacheDType: "float32"}, 8, 13824 + 1048576},
	} {
		p := mustPlan(t, hybridGraph(c.rec, nil), c.e, c.tp)
		if got := p.run.StateBytes; got != c.want {
			t.Errorf("%s: state bytes %d, want %d", c.name, got, c.want)
		}
	}
}

func withGroups(n model.Node, groups int) model.Node { n.NumGroups = groups; return n }

// The rest of the spec: Mamba-2 checkpoints a block at any position, KDA only on the FlashKDA
// prefill backend (which "auto" takes on compute capability 9, 10 and 12 for a bfloat16 model with
// a 128-wide head), and GDN not at all; the speculative blocks are the draft length.
func TestMambaSpecFieldsAsLayersBuildThem(t *testing.T) {
	spec := &engineconfig.Speculative{Method: "mtp", NumSpecTokens: 2}
	for _, c := range []struct {
		name      string
		rec       model.Node
		capab     int
		weights   model.DType
		ckpt, aln int
		spec      int
	}{
		{"mamba2", mamba2, 9, model.DTypeBF16, 1, 1, 2},
		{"kda on hopper", kda, 9, model.DTypeBF16, 1, 16, 2},
		{"kda on blackwell", kda, 10, model.DTypeBF16, 1, 16, 2},
		{"kda on ampere", kda, 8, model.DTypeBF16, 0, 0, 2},
		{"kda in fp16", kda, 9, model.DTypeFP16, 0, 0, 2},
		{"gdn", gdn, 9, model.DTypeBF16, 0, 0, 2},
	} {
		g := hybridGraph(c.rec, nil)
		g.Global.WeightDType = c.weights
		p, err := planMamba(g, engineconfig.Engine{Speculative: spec}, mambaEnv{tp: 8, dcp: 1, pcp: 1, capability: c.capab, maxModelLen: 4096})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		r := p.run
		if r.Spec != "mamba" || r.Count != 1 || r.MambaCacheMode != "align" || r.NumSpeculativeBlocks != c.spec ||
			r.NumPrefillCheckpointBlocks != c.ckpt || r.PrefillCheckpointAlignment != c.aln {
			t.Errorf("%s: %+v, want a mamba run in align mode with %d speculative blocks and checkpoint %d at %d",
				c.name, r, c.spec, c.ckpt, c.aln)
		}
		if got := len(p.assumptions) > 0; got != (c.ckpt > 0 && c.rec.RecurrentKind == model.RecurrentKDA) {
			t.Errorf("%s: assumptions %v", c.name, p.assumptions)
		}
	}
}

// With prefix caching on the hook runs "align" whatever mode is stated (a stated "none" cannot be told
// from the initializer); with it off the mode is "none" and a state covers max_model_len tokens
// (MambaModelConfig, vllm/model_executor/models/config.py:685-709@5df63a72). The engine's block size
// is then the aligned one in align mode only.
func TestMambaCacheModeAsTheHookRunsIt(t *testing.T) {
	off, on := false, true
	g := hybridGraph(mamba2, &gqa)
	for _, c := range []struct {
		name       string
		e          engineconfig.Engine
		mode       string
		mambaBlock int
	}{
		{"default", engineconfig.Engine{BlockSize: 16}, "align", 2080},
		{"stated none, prefix caching on", engineconfig.Engine{BlockSize: 16, MambaCacheMode: "none", EnablePrefixCaching: &on}, "align", 2080},
		{"stated align", engineconfig.Engine{BlockSize: 16, MambaCacheMode: "align"}, "align", 2080},
		{"prefix caching off", engineconfig.Engine{BlockSize: 16, EnablePrefixCaching: &off}, "none", 131072},
		{"stated align, prefix caching off", engineconfig.Engine{BlockSize: 16, MambaCacheMode: "align", EnablePrefixCaching: &off}, "none", 131072},
	} {
		p := mustPlan(t, g, c.e, 8)
		if p.run.MambaCacheMode != c.mode || p.run.BlockSize != c.mambaBlock || p.blockSize != 2080 {
			t.Errorf("%s: mode %q, mamba block %d, engine block %d; want %q, %d, 2080",
				c.name, p.run.MambaCacheMode, p.run.BlockSize, p.blockSize, c.mode, c.mambaBlock)
		}
	}
	// None mode with a window the engine fits to memory has no block size to state.
	_, err := planMamba(g, engineconfig.Engine{BlockSize: 16, EnablePrefixCaching: &off}, mambaEnv{tp: 8, dcp: 1, pcp: 1, capability: 9})
	if err == nil || !strings.Contains(err.Error(), "mamba_cache_mode none") {
		t.Errorf("none mode with no window: %v, want it refused by name", err)
	}
}

// The engine's block size is the aligned one for a hybrid model, from the attention page of the
// model's layers (Platform._align_hybrid_block_size): Kimi-K3's MLA layers are 1152 bytes a token at
// the model's dtype and 576 at fp8; Nemotron-3-Ultra's GQA layer on a rank is 2 x 1 x 128 x 2 = 512 bytes
// at bf16 and 256 at fp8. A Mamba page that is the attention page needs no padding, so the run states none.
func TestHybridEngineBlockSizeIsAligned(t *testing.T) {
	for _, c := range []struct {
		name          string
		rec           model.Node
		attn          model.Node
		e             engineconfig.Engine
		block, padded int
	}{
		{"kimi-k3 bf16, block unstated", kda, mla, engineconfig.Engine{}, 768, 884736},
		{"kimi-k3 bf16, block stated 16", kda, mla, engineconfig.Engine{BlockSize: 16}, 768, 884736},
		{"kimi-k3 fp8", kda, mla, engineconfig.Engine{CacheDType: "fp8"}, 1536, 884736},
		{"nemotron bf16, block stated 16", mamba2, gqa, engineconfig.Engine{BlockSize: 16}, 2080, 1064960},
		{"nemotron fp8", mamba2, gqa, engineconfig.Engine{BlockSize: 16, CacheDType: "fp8_e4m3"}, 4160, 4160 * 256},
		{"nemotron stated block above the aligned", mamba2, gqa, engineconfig.Engine{BlockSize: 4096}, 4096, 4096 * 512},
	} {
		attn := c.attn
		p := mustPlan(t, hybridGraph(c.rec, &attn), c.e, 8)
		if p.blockSize != c.block || p.run.PageSizePadded != c.padded || p.run.BlockSize != c.block {
			t.Errorf("%s: engine block %d, mamba block %d, padded %d; want %d, %d, %d", c.name,
				p.blockSize, p.run.BlockSize, p.run.PageSizePadded, c.block, c.block, c.padded)
		}
	}
}

// A model with no attention layer has no attention backend, so the platform aligns nothing
// (update_block_size_for_backend returns before it chooses a size, interface.py:763-766@5df63a72):
// the block size is the stated one or 16, the page is not padded, and in align mode the state block is it.
func TestAttentionFreeModelKeepsItsBlockSize(t *testing.T) {
	p := mustPlan(t, hybridGraph(mamba2, nil), engineconfig.Engine{}, 8)
	if p.blockSize != 16 || p.run.BlockSize != 16 || p.run.PageSizePadded != 0 {
		t.Errorf("block %d, mamba block %d, padded %d; want 16, 16, 0", p.blockSize, p.run.BlockSize, p.run.PageSizePadded)
	}
	p = mustPlan(t, hybridGraph(mamba2, nil), engineconfig.Engine{BlockSize: 64}, 8)
	if p.blockSize != 64 || p.run.BlockSize != 64 {
		t.Errorf("stated 64: block %d, mamba block %d", p.blockSize, p.run.BlockSize)
	}
}

// What the graph or the engine block cannot state, or vLLM v0.32.0 does not accept, is refused by
// name, and the text is stable.
func TestRecurrentLayersRefuseWhatTheyDoNotMap(t *testing.T) {
	noDType := mamba2
	noDType.StateDType = ""
	noConv := gdn
	noConv.ConvKernel = 0
	noInter := gdn
	noInter.IntermediateSize = 0
	otherGeometry := mamba2
	otherGeometry.NumHeads = 128
	swa := gqa
	swa.AttentionKind = model.AttentionSparseMLA
	wide := gqa
	wide.HeadDim = 256
	cached := gqa
	cached.Caches = []model.Cache{{Kind: model.CacheState}}
	mixed := hybridGraph(mamba2, &gqa)
	mixed.LayerKinds = append(mixed.LayerKinds, model.LayerKind{ID: "rec2", Nodes: []model.Node{otherGeometry}})
	mixed.Stack.Pattern = append(mixed.Stack.Pattern, "rec2")
	drafted := hybridGraph(mamba2, &gqa)
	drafted.Speculator = &model.Speculator{Method: "mtp", NumSpec: 1, Stack: model.Stack{Pattern: []string{"rec"}, Repeat: 1}}
	off := false

	for _, c := range []struct {
		name string
		g    *model.Graph
		e    engineconfig.Engine
		tp   int
		dcp  int
		want string
	}{
		{"mamba_cache_mode all", hybridGraph(kda, &mla), engineconfig.Engine{MambaCacheMode: "all"}, 8, 1, `mamba_cache_mode "all"`},
		{"state_dtype missing", hybridGraph(noDType, &gqa), engineconfig.Engine{BlockSize: 16}, 8, 1, "states no state_dtype"},
		{"conv_kernel missing", hybridGraph(noConv, nil), engineconfig.Engine{}, 8, 1, "states no conv_kernel"},
		{"intermediate_size missing", hybridGraph(noInter, nil), engineconfig.Engine{}, 8, 1, "states no intermediate_size"},
		{"kda ssm dtype fp16", hybridGraph(kda, &mla), engineconfig.Engine{MambaSSMCacheDType: "float16"}, 8, 1, "KDA recurrent state supports only"},
		{"unknown conv dtype", hybridGraph(kda, &mla), engineconfig.Engine{MambaCacheDType: "fp8"}, 8, 1, `mamba_cache_dtype "fp8"`},
		{"heads not divisible", hybridGraph(kda, &mla), engineconfig.Engine{}, 7, 1, "not divisible by tensor_parallel_size 7"},
		{"context parallel", hybridGraph(kda, &mla), engineconfig.Engine{}, 8, 2, "decode_context_parallel_size 2"},
		{"two geometries", mixed, engineconfig.Engine{BlockSize: 16}, 8, 1, "another geometry"},
		{"recurrent drafter", drafted, engineconfig.Engine{BlockSize: 16, Speculative: &engineconfig.Speculative{Method: "mtp", NumSpecTokens: 1}}, 8, 1,
			"drafter layer kind rec holds a recurrent layer"},
		{"sparse attention beside", hybridGraph(mamba2, &swa), engineconfig.Engine{BlockSize: 16}, 8, 1, "sparse_mla attention beside recurrent layers"},
		{"attention caches", hybridGraph(mamba2, &cached), engineconfig.Engine{BlockSize: 16}, 8, 1, "stating caches"},
		{"head size 256, block stated", hybridGraph(mamba2, &wide), engineconfig.Engine{BlockSize: 16}, 8, 1, "attention head size 256"},
		{"head size 256, block unstated", hybridGraph(mamba2, &wide), engineconfig.Engine{}, 8, 1, "blis-vllm/issues/68"},
		{"packed cache dtype", hybridGraph(mamba2, &gqa), engineconfig.Engine{BlockSize: 16, CacheDType: "nvfp4"}, 8, 1, `cache_dtype "nvfp4"`},
		{"mla block off the multiple", hybridGraph(kda, &mla), engineconfig.Engine{BlockSize: 200}, 8, 1, "block_size 200 beside MLA"},
		{"none mode, no window", hybridGraph(kda, &mla), engineconfig.Engine{EnablePrefixCaching: &off}, 8, 1, "mamba_cache_mode none"},
	} {
		env := mambaEnv{tp: c.tp, dcp: c.dcp, pcp: 1, capability: 9}
		if c.name != "none mode, no window" {
			env.maxModelLen = 4096
		}
		_, err := planMamba(c.g, c.e, env)
		if err == nil || !strings.HasPrefix(err.Error(), "recurrent layers: ") || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want a recurrent-layers refusal containing %q", c.name, err, c.want)
		}
	}
}

// A graph with no recurrent layer has no plan.
func TestAGraphWithoutRecurrentLayersHasNoPlan(t *testing.T) {
	g := &model.Graph{LayerKinds: []model.LayerKind{{ID: "attn", Nodes: []model.Node{gqa}}}, Stack: model.Stack{Pattern: []string{"attn"}, Repeat: 2}}
	if p, err := planMamba(g, engineconfig.Engine{}, mambaEnv{tp: 1}); p != nil || err != nil {
		t.Errorf("plan %v, %v for a graph without recurrent layers", p, err)
	}
}

// ownScenario writes the catalog scenario fixture with its engine settings edited, and opens it.
func ownScenario(t *testing.T, fixture string, edit func(string) string) *Model {
	t.Helper()
	src, err := os.ReadFile(filepath.Join(ownRoot(), "kerneltests", "testdata", fixture))
	if err != nil {
		t.Fatal(err)
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "s.yaml"), []byte(edit(string(src))), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open("s.yaml", r)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// dropLines removes the lines that contain any of the markers.
func dropLines(s string, markers ...string) string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		keep := true
		for _, m := range markers {
			keep = keep && !strings.Contains(l, m)
		}
		if keep {
			out = append(out, l)
		}
	}
	return strings.Join(out, "\n")
}

// Kimi-K3 on H200 at tp 8, every engine setting at its default (prefix caching on, so the mode is
// align; cache dtype auto, so bf16): 24 MLA layers, and 69 KDA layers in runs of three, then the
// first KDA layer's two dense and sparse ones before the first MLA layer. Its state is 814080 bytes a
// rank, its block the 768 tokens whose MLA page (768 x 1152 = 884736 bytes) holds it, and the KDA
// layers checkpoint a prefill at block 16 (FlashKDA on compute capability 9).
func TestKimiK3KVLayersAreVLLMs(t *testing.T) {
	m := ownScenario(t, "kimi-k3-h100-nospec.yaml", func(s string) string {
		s = strings.Replace(s, "hardware: h100", "hardware: h200", 1)
		return dropLines(s, "kv_cache_dtype:", "mamba_cache_mode:", "block_size:", "cudagraph_mode")
	})
	runs, err := m.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	want := KVLayerRun{Spec: "mamba", StateBytes: 814080, BlockSize: 768, PageSizePadded: 884736, MambaCacheMode: "align",
		NumPrefillCheckpointBlocks: 1, PrefillCheckpointAlignment: 16}
	mamba, mlaLayers := 0, 0
	for _, r := range runs {
		switch r.Spec {
		case "mamba":
			mamba += r.Count
			w := want
			w.Count = r.Count
			if r != w {
				t.Errorf("mamba run %+v, want %+v", r, w)
			}
		case "mla":
			// Beside recurrent layers a model is a hybrid, and an MLA layer's page is one latent a
			// token (MLAAttentionSpec), not keys and values.
			mlaLayers += r.Count
			if r.NumKVHeads != 1 || r.HeadSize != 576 || r.TokensPerState != 1 {
				t.Errorf("attention run %+v, want one latent of 576", r)
			}
		default:
			t.Errorf("run of spec %q", r.Spec)
		}
	}
	if mamba != 69 || mlaLayers != 24 {
		t.Errorf("%d mamba and %d attention layers, want 69 and 24", mamba, mlaLayers)
	}
	if runs[0].Spec != "mamba" || runs[0].Count != 3 || runs[1].Spec != "mla" || runs[1].Count != 1 {
		t.Errorf("first runs %+v %+v, want three KDA layers then one MLA layer", runs[0], runs[1])
	}
	e, err := m.Engine()
	if err != nil || e.BlockSize != 768 {
		t.Errorf("engine block size %d (%v), want the aligned 768", e.BlockSize, err)
	}
	if got := m.EngineSources()["block_size"]; !strings.Contains(got, "_align_hybrid_block_size") {
		t.Errorf("block_size source %q does not name the alignment", got)
	}
	named := false
	for _, a := range m.EngineAssumptions() {
		named = named || strings.Contains(a, "gate_lower_bound")
	}
	if !named {
		t.Errorf("assumptions %v do not name the KDA gate bound", m.EngineAssumptions())
	}
}

// Nemotron-3-Ultra on H200 at tp 8: its Mamba-2 layers' state is 1062400 bytes a rank (13824 conv bf16 + 1048576
// ssm fp32), and a block size of 16 stated by the deployment, or none, aligns to the 2080 tokens whose GQA page
// (2 x 1 x 128 x 2 = 512 bytes a token) holds it.
func TestNemotronUltraKVLayersAreVLLMs(t *testing.T) {
	for _, stated := range []bool{true, false} {
		m := ownScenario(t, "nemotron3-ultra-h100-agg.yaml", func(s string) string {
			s = strings.Replace(s, "hardware: h100", "hardware: h200", 1)
			s = dropLines(s, "kv_cache_dtype:", "mamba_cache_mode:", "speculative_config", "cudagraph_mode")
			if !stated {
				s = dropLines(s, "block_size:")
			}
			return s
		})
		runs, err := m.KVLayers()
		if err != nil {
			t.Fatalf("block_size stated %t: %v", stated, err)
		}
		mamba, attn := 0, 0
		for _, r := range runs {
			switch r.Spec {
			case "mamba":
				mamba += r.Count
				w := KVLayerRun{Count: r.Count, Spec: "mamba", StateBytes: 1062400, BlockSize: 2080, PageSizePadded: 1064960,
					MambaCacheMode: "align", NumPrefillCheckpointBlocks: 1, PrefillCheckpointAlignment: 1}
				if r != w {
					t.Errorf("block_size stated %t: mamba run %+v, want %+v", stated, r, w)
				}
			case "full":
				attn += r.Count
				if r.NumKVHeads != 1 || r.HeadSize != 128 {
					t.Errorf("attention run %+v, want one head of 128", r)
				}
			default:
				t.Errorf("run of spec %q", r.Spec)
			}
		}
		if mamba == 0 || attn == 0 {
			t.Errorf("block_size stated %t: %d mamba and %d attention layers", stated, mamba, attn)
		}
		if e, err := m.Engine(); err != nil || e.BlockSize != 2080 {
			t.Errorf("block_size stated %t: engine block size %d (%v), want 2080", stated, e.BlockSize, err)
		}
	}
}

// Settings vLLM v0.32.0 reads that change a recurrent layer's spec and that the engine block cannot
// state are refused where the block is read, naming the key; mamba_cache_mode all, which v0.32.0 does
// not accept (vllm/config/cache.py:63@5df63a72), is refused by KVLayers with its name.
func TestUnmatchedRecurrentSettingsStayRefusals(t *testing.T) {
	const fixture = "nemotron3-ultra-h100-agg.yaml"
	src, err := os.ReadFile(filepath.Join(ownRoot(), "kerneltests", "testdata", fixture))
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"use_replayssm", "use_kda_recoverssm", "enable_mamba_shared_prefix_checkpoint", "additional_config"} {
		r := repos()
		r.Scenarios = t.TempDir()
		body := strings.Replace(string(src), "      block_size: 16\n", "      block_size: 16\n      "+key+": true\n", 1)
		if err := os.WriteFile(filepath.Join(r.Scenarios, "s.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Open("s.yaml", r); err == nil || !strings.Contains(err.Error(), key) {
			t.Errorf("%s stated: %v, want the engine block refused naming it", key, err)
		}
	}
	m := ownScenario(t, fixture, func(s string) string { return strings.Replace(s, "mamba_cache_mode: none", "mamba_cache_mode: all", 1) })
	if _, err := m.KVLayers(); err == nil || !strings.HasPrefix(err.Error(), "recurrent layers: mamba_cache_mode \"all\"") {
		t.Errorf("mamba_cache_mode all: %v, want the refusal naming it", err)
	}
	if pieces := m.V032Pieces(); len(pieces) != 1 || !strings.Contains(pieces[0], `mamba_cache_mode "all"`) {
		t.Errorf("V032Pieces %v, want the one hybrid KV groups piece naming mamba_cache_mode", pieces)
	}
}
