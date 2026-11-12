// Original to blis-vllm.

package vllmengine

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/core/vllm"
)

// oracleRun and oracleCase are the fields of core/lower/testdata/vllm_lower.json's kv_pool cases
// this test reads (oracle/lower/extract.py kv_pool_cases): vLLM's own page_size_bytes and bytes
// per block for the layers an architecture builds at a cache stored as fp8.
type oracleRun struct {
	Count         int    `json:"count"`
	Spec          string `json:"spec"`
	SlidingWindow int    `json:"sliding_window"`
	NumKVHeads    int    `json:"num_kv_heads"`
	HeadSize      int    `json:"head_size"`
	PageSizeBytes int64  `json:"page_size_bytes"`
}

type oracleCase struct {
	Model         string           `json:"model"`
	TP            int              `json:"tp"`
	BlockSize     int              `json:"block_size"`
	Backend       string           `json:"backend"`
	Runs          []oracleRun      `json:"runs"`
	Layout        string           `json:"layout"`
	BytesPerBlock map[string]int64 `json:"bytes_per_block"`
}

func loadOracleKVPool(t *testing.T) []oracleCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "core", "lower", "testdata", "vllm_lower.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		KVPool []oracleCase `json:"kv_pool"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	return f.KVPool
}

// kernel's runs are matched to vLLM's by what each layer is, not by order.
func runKey(spec string, window, heads, head, count int) string {
	return fmt.Sprintf("%s window %d heads %d head %d x%d", spec, window, heads, head, count)
}

// A deployment that states kv_cache_dtype fp8 sizes its KV pool as vLLM v0.32.0 does, for each
// architecture the dashboard states it for. The engine block's key is cache_dtype, not
// kv_cache_dtype (the name of vLLM's flag and of fields.json's entry), until a later change renames
// it; the stated value here is cache_dtype fp8 in a committed fixture of each model, one per
// tensor-parallel width. The DeepSeek-V4 row assumes the packed fp8_ds_mla layout of the default
// FlashMLA backend (see core/kv's TestPoolBytesPerBlockIsVLLMsAtFP8 for the lines and what is not
// covered). For each, the KV
// layers the engine reads from the model graph (kernelmodel KVLayers), the page bytes the
// substrate kernel sizes (KVLayerPageBytes) and the bytes per block of the pool (stagePoolsOf)
// are compared with vLLM's own: the page_size_bytes of the specs it builds for those layers and
// _get_kv_cache_bytes_per_block over the groups it forms (core/lower/testdata/vllm_lower.json,
// printed by oracle/lower/extract.py). The oracle's layer counts and head sizes are the
// catalog's, so what is checked is the page and pool arithmetic, the grouping and the dtype
// bytes, not the catalog.
//
//	architecture       fixtures                                        engine layers           pool bytes per block
//	gpt-oss-120b       tp 2, 4, 8 (h100, h200)                         18 window + 18 full     two groups of 18
//	minimax-m2.5       tp 1, 2, 4, 8 (b200, h200)                      none (one full group)   62 pages
//	minimax-m3         tp 2, 4, 8 (b200)                               60 full + 57 indexer    one group of 117
//	kimi-k2.5          tp 4, 8 (b200)                                  none (one full group)   61 pages
//	deepseek-v4-pro    tp 8 (b200)                                     243 packed layers       packed, block-outermost
//	kimi-k3            tp 8 (h100)                                     not mapped (see below)  24 MLA pages
//
// Kimi-K3 is hybrid: 24 MLA layers and 41 KDA layers, whose recurrent state is a MambaSpec that
// vLLM pads in the hybrid-model config hook (HybridAttentionMambaModelConfig). The engine maps no
// recurrent layer to a KV cache spec and refuses the model's pool (kernelmodel KVLayers), so
// only the kernel's MLA page bytes are compared, with the oracle's 24 MLA layers; the day the
// engine maps the KDA layers the refusal assertion below fails and the pool comparison is added.
func TestKVPoolAtFP8IsVLLMsForEachArchitecture(t *testing.T) {
	cases := loadOracleKVPool(t)
	type fixture struct {
		file, model string
		tp          int
	}
	fixtures := []fixture{
		{"aisimulate/gpt-oss-120b-h100-fp4-vllm-tp2.yaml", "gpt-oss-120b", 2},
		{"aisimulate/gpt-oss-120b-h100-fp4-vllm-tp4.yaml", "gpt-oss-120b", 4},
		{"aisimulate/gpt-oss-120b-h200-fp4-vllm-tp8.yaml", "gpt-oss-120b", 8},
		{"aisimulate/minimax-m2.5-b200-fp4-vllm-tp1.yaml", "minimax-m2.5", 1},
		{"aisimulate/minimax-m2.5-b200-fp8-vllm-tp2.yaml", "minimax-m2.5", 2},
		{"aisimulate/minimax-m2.5-h200-fp8-vllm-tp4.yaml", "minimax-m2.5", 4},
		{"aisimulate/minimax-m2.5-h200-fp8-vllm-tp8.yaml", "minimax-m2.5", 8},
		{"aisimulate/minimax-m3-b200-fp4-vllm-tp2.yaml", "minimax-m3", 2},
		{"aisimulate/minimax-m3-b200-fp4-vllm-tp4.yaml", "minimax-m3", 4},
		{"aisimulate/minimax-m3-b200-fp4-vllm-tp8.yaml", "minimax-m3", 8},
		{"aisimulate/kimi-k2.5-b200-fp4-vllm-tp4.yaml", "kimi-k2.5", 4},
		{"aisimulate/kimi-k2.5-b200-fp4-vllm-tp8.yaml", "kimi-k2.5", 8},
		{"direct/deepseek-v4-pro-b200-fp4-vllm-tp8-cfg1.yaml", "deepseek-v4-pro", 8},
		{"kimi-k3-h100-nospec.yaml", "kimi-k3 (MLA layers only)", 8},
	}
	for _, f := range fixtures {
		t.Run(f.model+"/"+filepath.Base(f.file), func(t *testing.T) {
			t.Parallel()
			dir := filepath.Join("..", "kerneltests", "testdata", filepath.Dir(f.file))
			m, err := kernelmodel.OpenPool(filepath.Base(f.file), kernelmodel.Repos{Scenarios: dir, Data: embeddedData(t)}, 0)
			if err != nil {
				t.Fatal(err)
			}
			e, err := m.Engine()
			if err != nil {
				t.Fatal(err)
			}
			if e.CacheDType != "fp8" {
				t.Fatalf("the fixture's resolved cache dtype is %q, want the stated fp8", e.CacheDType)
			}
			if strings.HasPrefix(f.model, "kimi-k3") {
				// Kimi-K3 is a hybrid of 24 MLA layers and 69 KDA layers: its recurrent state is a
				// MambaSpec padded to the MLA page at the aligned block size (platforms/interface.py:
				// 1087-1126@5df63a72), and vLLM groups them at one page size
				// (_get_kv_cache_groups_uniform_page_size, kv_cache_utils.py:1362-1502): one group
				// of the 24 MLA layers and three of the 69 KDA layers (23 each), so one pool block is
				// the widest group's 24 pages. The vLLM comparison for this shape is the oracle
				// scenario mamba-kimi-k3-like (oracle/scenarios, core/sched/testdata/golden).
				runs, err := m.KVLayers()
				if err != nil {
					t.Fatal(err)
				}
				mla, kda, pad := 0, 0, 0
				for _, r := range runs {
					switch r.Spec {
					case "mamba":
						kda += r.Count
						pad = r.PageSizePadded
					case "mla":
						mla += r.Count
					}
				}
				if mla != 24 || kda != 69 || pad != e.BlockSize*576 {
					t.Fatalf("layers: %d MLA, %d KDA, mamba page padded to %d; want 24, 69 and the MLA page %d at block %d",
						mla, kda, pad, e.BlockSize*576, e.BlockSize)
				}
				page, err := m.KVLayerPageBytes()
				if err != nil {
					t.Fatal(err)
				}
				per, err := vllm.KVPoolBytesPerBlock(vllmRuns(runs), false, e.BlockSize, page)
				if err != nil {
					t.Fatal(err)
				}
				if want := int64(24 * pad); per != want {
					t.Errorf("pool bytes per block %d, vLLM's widest group of 24 pages %d", per, want)
				}
				if page != int64(24*pad) {
					t.Errorf("kernel page bytes of the 24 MLA layers = %d, want %d", page, 24*pad)
				}
				return
			}
			var oc *oracleCase
			for i := range cases {
				if c := &cases[i]; c.Model == f.model && c.TP == f.tp && c.BlockSize == e.BlockSize && strings.HasPrefix(c.Backend, "no kernel") {
					oc = c
				}
			}
			if oc == nil {
				t.Fatalf("no oracle case for %s tp%d at block size %d", f.model, f.tp, e.BlockSize)
			}
			var sum int64
			for _, r := range oc.Runs {
				sum += int64(r.Count) * r.PageSizeBytes
			}
			want := oc.BytesPerBlock[oc.Layout]
			page, err := m.KVLayerPageBytes()
			if err != nil {
				t.Fatal(err)
			}
			runs, err := m.KVLayers()

			switch {
			case err != nil:
				t.Fatal(err)
			case len(runs) == 0:
				// One full-attention group: the pool's block is every layer's page, which the
				// kernel sizes as one figure.
				if page != want || page != sum {
					t.Errorf("kernel page bytes = %d; vLLM's pool block %d, sum of its %d layers' pages %d", page, want, len(oc.Runs), sum)
				}
			default:
				got := map[string]int{}
				for _, r := range runs {
					got[runKey(r.Spec, r.SlidingWindow, r.NumKVHeads, r.HeadSize, 1)] += r.Count
				}
				exp := map[string]int{}
				for _, r := range oc.Runs {
					exp[runKey(r.Spec, r.SlidingWindow, r.NumKVHeads, r.HeadSize, 1)] += r.Count
				}
				if f.model == "deepseek-v4-pro" {
					// V4's runs carry their own block sizes and states; the oracle's are the
					// dashboard deployment's, so only the layer count by spec is compared.
					got, exp = map[string]int{}, map[string]int{}
					for _, r := range runs {
						got[r.Spec] += r.Count
					}
					for _, r := range oc.Runs {
						exp[r.Spec] += r.Count
					}
				}
				if fmt.Sprint(got) != fmt.Sprint(exp) {
					t.Errorf("engine layers %v, vLLM's built for the oracle case %v", got, exp)
				}
				pools, err := stagePoolsOf(m, runs)
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(pools.perBlock, []int64{want}) {
					t.Errorf("pool bytes per block %v, vLLM's %s %d", pools.perBlock, oc.Layout, want)
				}
				if f.model == "deepseek-v4-pro" {
					if page != want {
						t.Errorf("kernel pool bytes per block = %d, vLLM's %d", page, want)
					}
				} else if page != sum {
					t.Errorf("kernel page bytes = %d, sum of vLLM's %d layers' pages %d", page, len(oc.Runs), sum)
				}
			}
		})
	}
}
