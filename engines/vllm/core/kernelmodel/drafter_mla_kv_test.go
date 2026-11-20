// Original to inference-sim (inference-sim#2316).

package kernelmodel

import (
	"strings"
	"testing"
)

// kimiK3 opens Kimi-K3 on H200 at tp 8, every engine setting at its default, with the DSpark
// drafter the catalog names when drafter is set.
func kimiK3(t *testing.T, drafter bool) *Model {
	t.Helper()
	return ownScenario(t, "kimi-k3-h100-nospec.yaml", func(s string) string {
		s = strings.Replace(s, "hardware: h100", "hardware: h200", 1)
		s = dropLines(s, "kv_cache_dtype:", "mamba_cache_mode:", "block_size:", "cudagraph_mode")
		if drafter {
			s = strings.Replace(s, "      max_model_len:",
				"      speculative_config: {method: dspark, num_speculative_tokens: 3, model: kimi-k3-dspark}\n      max_model_len:", 1)
		}
		return s
	})
}

// elemsPerBlock is what the pool counts for a block of the runs: vLLM's page of an attention layer
// is K and V for a full-attention spec and one latent for an MLA spec (MLAAttentionSpec has no V,
// vllm/v1/kv_cache_interface.py:742-743@4ba787ec); a mamba run is not part of the kernel's page.
func elemsPerBlock(t *testing.T, runs []KVLayerRun, block int, draft bool) int64 {
	t.Helper()
	var n int64
	for _, r := range runs {
		if r.Draft != draft {
			continue
		}
		var perToken int64
		switch r.Spec {
		case "full":
			perToken = 2 * int64(r.NumKVHeads) * int64(r.HeadSize)
		case "mla":
			perToken = int64(r.NumKVHeads) * int64(r.HeadSize)
		case "mamba":
			continue
		default:
			t.Fatalf("a %s run is not part of this law", r.Spec)
		}
		n += int64(r.Count) * perToken * int64(block)
	}
	return n
}

// The memory model's bytes for a block of the KV layers equal the mapped layers' elements, at the
// cache dtype's bytes, for the target's layers and the drafter's alike (INV-1 sizes a pool from
// that page). Kimi-K3's 24 MLA layers are latents of 576 beside its KDA layers, and so are the
// five MLA layers of its DSpark drafter: vLLM builds both from MultiHeadLatentAttention.
func TestKimiK3DSparkDrafterMLALayersAreLatentsLikeTheTargets(t *testing.T) {
	const bf16 = 2
	pageOf := func(m *Model) (int64, int) {
		page, err := m.KVLayerPageBytes()
		if err != nil {
			t.Fatal(err)
		}
		e, err := m.Engine()
		if err != nil {
			t.Fatal(err)
		}
		return page, e.BlockSize
	}
	nospec := kimiK3(t, false)
	targetPage, block := pageOf(nospec)
	targetRuns, err := nospec.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	if got := elemsPerBlock(t, targetRuns, block, false) * bf16; got != targetPage {
		t.Errorf("target: the mapped layers hold %d bytes a block, the memory model %d", got, targetPage)
	}

	spec := kimiK3(t, true)
	page, block := pageOf(spec)
	runs, err := spec.KVLayers()
	if err != nil {
		t.Fatalf("Kimi-K3 with a dspark drafter: %v", err)
	}
	if got := elemsPerBlock(t, runs, block, false) * bf16; got != targetPage {
		t.Errorf("the target's layers beside a drafter hold %d bytes a block, want %d", got, targetPage)
	}
	draft := elemsPerBlock(t, runs, block, true) * bf16
	if draft != page-targetPage {
		t.Errorf("drafter: the mapped layers hold %d bytes a block, the memory model %d", draft, page-targetPage)
	}
	layers := 0
	for _, r := range runs {
		if !r.Draft {
			continue
		}
		layers += r.Count
		if r.Spec != "mla" || r.NumKVHeads != 1 || r.HeadSize != 576 || r.TokensPerState != 1 {
			t.Errorf("drafter run %+v, want latents of 576", r)
		}
	}
	if layers != 5 {
		t.Errorf("%d drafter layers, want 5", layers)
	}
}
