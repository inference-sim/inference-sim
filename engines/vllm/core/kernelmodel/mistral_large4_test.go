package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// openMistral opens a Mistral-Large-4 scenario of testdata/scenarios/mistral-large-4 with the
// engine line extra (e.g. "kv_cache_dtype: bfloat16") added to its pool, if extra is not empty.
func openMistral(t *testing.T, scenario, extra string) (*Model, error) {
	t.Helper()
	dir := filepath.Join(ownRoot(), "..", "..", "testdata", "scenarios", "mistral-large-4")
	raw, err := os.ReadFile(filepath.Join(dir, scenario))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if extra != "" {
		const anchor = "      async_scheduling: true\n"
		if !strings.Contains(body, anchor) {
			t.Fatalf("%s has no %q line to extend", scenario, strings.TrimSpace(anchor))
		}
		body = strings.Replace(body, anchor, anchor+"      "+extra+"\n", 1)
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Open("v.yaml", r)
}

// Mistral-Large-4 registers, per layer in model order, the indexer's key cache when the layer
// runs its own indexer (44 of 72) and an MLA latent cache when it owns its KV (63 of 72: the 9
// layers that share a source's KV register none), vllm/models/mistral/nvidia/attention.py and
// indexer.py at 0fda86a3. A block of 64 tokens then holds 44 x 132 B + 63 x 656 B a token under
// the fp8_ds_mla record, which MistralLarge4ForCausalLM.verify_and_update_config
// (vllm/model_executor/models/config.py:147-178) makes of an unstated cache dtype, and 44 x 132 B
// + 63 x 1152 B once `bfloat16` opts out of it.
func TestMistralLarge4KVLayersAreTheEnginesAndTheCacheDTypeIsFP8DSMLA(t *testing.T) {
	for _, scenario := range []string{"mistral-nvfp4-gb200-tp8.yaml", "mistral-fp8-gb200-tp8.yaml"} {
		for _, c := range []struct {
			name, extra, dtype string
			mla                int
		}{
			{"unstated", "", "fp8_ds_mla", 656},
			{"auto", "kv_cache_dtype: auto", "fp8_ds_mla", 656},
			{"bfloat16 opts out", "kv_cache_dtype: bfloat16", "auto", 1152},
		} {
			t.Run(scenario+"/"+c.name, func(t *testing.T) {
				m, err := openMistral(t, scenario, c.extra)
				if err != nil {
					t.Fatal(err)
				}
				e, err := m.Engine()
				if err != nil {
					t.Fatal(err)
				}
				// The engine block reports what the run stated, or what the hook resolved an unstated
				// dtype to; a stated bfloat16 stays so there, and runs as "auto" (the lowering's record).
				wantStated := map[string]string{"unstated": "fp8_ds_mla", "auto": "fp8_ds_mla", "bfloat16 opts out": "bfloat16"}[c.name]
				if e.CacheDType != wantStated {
					t.Errorf("engine block cache dtype %q, want %q", e.CacheDType, wantStated)
				}
				runs, err := m.KVLayers()
				if err != nil {
					t.Fatal(err)
				}
				got := map[string]int{}
				for _, r := range runs {
					got[r.Spec] += r.Count
					want := map[string]int{"indexer": 132, "mla": c.mla}[r.Spec]
					if r.HeadSize != want || r.NumKVHeads != 1 {
						t.Errorf("%s run: %d head(s) of %d, want 1 of %d", r.Spec, r.NumKVHeads, r.HeadSize, want)
					}
				}
				if got["indexer"] != 44 || got["mla"] != 63 || len(got) != 2 {
					t.Errorf("layers by spec %v, want 44 indexer and 63 mla", got)
				}
				b, err := m.KVBudget()
				if err != nil {
					t.Fatal(err)
				}
				if want := int64(64 * (44*132 + 63*c.mla)); b.PerBlockBytes != want {
					t.Errorf("%d bytes a block, want %d", b.PerBlockBytes, want)
				}
			})
		}
	}
}

// A DSpark drafter's three layers register after the target's, each a SlidingWindowMLASpec of the
// window 2048 widened by the draft block (num_speculative_tokens), holding the draft's own bf16
// latent record (576 elements x 2 B) because the draft does not use fp8_ds_mla
// (v1/worker/gpu/spec_decode/dspark/config.py:79-86@0fda86a3). A stated draft cache dtype of
// fp8_ds_mla is refused where vLLM refuses it (mla_attention.py:524-530).
func TestMistralLarge4DSparkDraftLayersAreSlidingWindowMLA(t *testing.T) {
	const spec = "speculative_config: {method: dspark, num_speculative_tokens: 3%s}"
	for _, c := range []struct {
		name, kvDType string
		head          int
		refused       bool
	}{
		{"default", "", 1152, false},
		{"fp8 draft cache", ", kv_cache_dtype: fp8", 576, false},
		{"fp8_ds_mla draft cache", ", kv_cache_dtype: fp8_ds_mla", 0, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := openMistral(t, "mistral-nvfp4-gb200-tp8.yaml", strings.Replace(spec, "%s", c.kvDType, 1))
			if err != nil {
				t.Fatal(err)
			}
			runs, err := m.KVLayers()
			if c.refused {
				if err == nil || !strings.Contains(err.Error(), "mla_attention.py:524") {
					t.Fatalf("fp8_ds_mla draft cache: got %v, want vLLM's refusal named", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var draft []KVLayerRun
			target := 0
			for _, r := range runs {
				if r.Draft {
					draft = append(draft, r)
				} else {
					target += r.Count
				}
			}
			if target != 44+63 {
				t.Errorf("%d target layers, want 107 (44 indexer + 63 mla)", target)
			}
			if len(draft) != 1 || draft[0].Count != 3 || draft[0].Spec != "sliding_window_mla" ||
				draft[0].SlidingWindow != 2048+3 || draft[0].HeadSize != c.head || draft[0].NumKVHeads != 1 {
				t.Errorf("draft runs %+v, want 3 sliding_window_mla layers of window 2051 and head %d", draft, c.head)
			}
		})
	}
}
