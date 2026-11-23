package kernelmodel

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Inkling 2 registers two caches a layer, the conv state then the attention
// (vllm/models/inkling2/nvidia/model.py@0fda86a3). At TP4 each rank holds 2 of the 8 KV heads, so
// the conv state's head is the next power of two over 2 x 128 + 2 x (4608 / 2) = 4864, which is
// 8192; the 40 local layers' attention is a sliding window of 1024, the 8 full layers' is full.
// Its page, 4 tokens x 2 heads x 8192 x 2 B, is above every attention page at block 16, so the
// attention blocks are scaled to it (unify_kv_cache_spec_page_size).
func TestInkling2KVLayersAreTheEnginesConvStateAndAttention(t *testing.T) {
	m, err := openScenarioDir(t, "inkling-2-small", "inkling-2-small-gb200-tp4.yaml", "")
	if err != nil {
		t.Fatal(err)
	}
	runs, err := m.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	conv, local, full, order := 0, 0, 0, 0
	for i, r := range runs {
		switch {
		case r.Spec == "sliding_window" && r.SlidingWindow == 4:
			conv += r.Count
			if r.BlockSize != 4 || r.NumKVHeads != 2 || r.HeadSize != 8192 || r.StateBytes != 8192*2 {
				t.Errorf("conv-state run %+v, want block 4, 2 heads of 8192 elements, 16384 B a state", r)
			}
			if i+1 >= len(runs) || runs[i+1].Spec == "sliding_window" && runs[i+1].SlidingWindow == 4 {
				t.Errorf("conv-state run %d is not followed by its layer's attention", i)
			}
			order++
		case r.Spec == "sliding_window" && r.SlidingWindow == 1024:
			local += r.Count
		case r.Spec == "full":
			full += r.Count
		default:
			t.Errorf("unexpected run %+v", r)
		}
		if r.Spec != "" && (r.NumKVHeads != 2 || r.ElemBytes != 2) {
			t.Errorf("run %+v: want 2 KV heads a rank and bf16 elements", r)
		}
	}
	if conv != 48 || local != 40 || full != 8 {
		t.Errorf("conv %d, local %d, full %d layers; want 48, 40, 8", conv, local, full)
	}
	// The conv state is the layer's own: the model prices it as no Mamba page.
	e, err := m.Engine()
	if err != nil {
		t.Fatal(err)
	}
	if e.BlockSize != 16 {
		t.Errorf("block size %d, want the stated 16", e.BlockSize)
	}
}

// A DSpark drafter's five attention layers register after the target's, as the target's own
// cache config (dspark.py:118-181): three windows of 1024 around one full layer.
func TestInkling2DSparkDraftLayersAreDraftAttention(t *testing.T) {
	m, err := openScenarioDir(t, "inkling-2-small", "inkling-2-small-gb200-tp4.yaml",
		"speculative_config: {method: dspark, num_speculative_tokens: 7}")
	if err != nil {
		t.Fatal(err)
	}
	runs, err := m.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	var window, full int
	for _, r := range runs {
		if !r.Draft {
			continue
		}
		switch r.Spec {
		case "sliding_window":
			if r.SlidingWindow != 1024 {
				t.Errorf("draft window %d, want 1024", r.SlidingWindow)
			}
			window += r.Count
		case "full":
			full += r.Count
		default:
			t.Errorf("draft run %+v", r)
		}
	}
	if window != 4 || full != 1 {
		t.Errorf("draft layers: %d windowed, %d full; want 4 and 1", window, full)
	}
	// The attention layer asserts a cache dtype of auto or bfloat16.
	m2, err := openScenarioDir(t, "inkling-2-small", "inkling-2-small-gb200-tp4.yaml", "kv_cache_dtype: fp8")
	if err == nil {
		_, err = m2.KVLayers()
	}
	if err == nil || !strings.Contains(err.Error(), "attention.py:215") {
		t.Errorf("fp8 cache: got %v, want the attention layer's assertion named", err)
	}
}

// openInkling2H200 opens the TP4 Inkling 2 fixture on one 8-GPU H200 node, the layout the vLLM
// recipe serves, with enable_expert_parallel as the test states it.
func openInkling2H200(t *testing.T, expertParallel bool, dp int, extra string) *Model {
	t.Helper()
	dir := filepath.Join(ownRoot(), "..", "..", "testdata", "scenarios", "inkling-2-small")
	raw, err := os.ReadFile(filepath.Join(dir, "inkling-2-small-gb200-tp4.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	for old, repl := range map[string]string{
		"hardware: gb200-nvl72":        "hardware: h200",
		"gpus_per_node: 4":             "gpus_per_node: 8",
		"  gpus_per_rack: 72\n":        "",
		"enable_expert_parallel: true": fmt.Sprintf("enable_expert_parallel: %t", expertParallel),
		"max_model_len: 262144":        "max_model_len: 131072",
		"dp: 1":                        fmt.Sprintf("dp: %d", dp),
	} {
		if !strings.Contains(body, old) {
			t.Fatalf("the fixture has no %q to rewrite", old)
		}
		body = strings.Replace(body, old, repl, 1)
	}
	if extra != "" {
		const anchor = "      async_scheduling: true\n"
		body = strings.Replace(body, anchor, anchor+"      "+extra+"\n", 1)
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open("v.yaml", r)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// Inkling 2 runs its routed experts expert-parallel across the tensor-parallel group whatever
// enable_expert_parallel says: Inkling2Experts takes ep_size and ep_rank from the tensor-parallel
// group and holds num_experts / ep_size whole experts (vllm/models/inkling2/nvidia/moe.py:93-96@
// 0fda86a3), and the recipe says TP4 means TP4 attention with EP4 experts. A scenario that leaves
// the flag off therefore serves the same layout as one that sets it: 96 whole experts a rank, not
// a 576-wide slice of every expert padded up to 640 for the Hopper repack, which is 11% more
// NVFP4 expert weight than the rank holds.
func TestInkling2ExpertsAreExpertParallelOverTPWhateverTheFlagSays(t *testing.T) {
	off := openInkling2H200(t, false, 1, "")
	on := openInkling2H200(t, true, 1, "")
	if got := off.k.Resolved().ExpertParallel(); got != 4 {
		t.Errorf("expert-parallel width %d with the flag off, want the tensor-parallel width 4", got)
	}
	if !off.Deployment().ExpertParallel {
		t.Error("the deployment reports expert parallelism off for a model that always runs it")
	}
	if a, b := off.k.FixedBytes(), on.k.FixedBytes(); a != b {
		t.Errorf("fixed bytes differ with the flag: off %+v, on %+v; the model runs one layout", a, b)
	}
	const gib = 1 << 30
	// 40 NVFP4 layers at 0.5625 B and 8 MXFP8 layers at 1.03125 B a parameter over 384 experts of
	// 3 x 2304 x 4608, a quarter to a rank, then the dense and embedding weights (about 4 GiB):
	// the checkpoint's own bytes, which the padded slice exceeded by 7 GiB.
	if w := float64(off.k.FixedBytes().Weights) / gib; w < 90 || w > 95 {
		t.Errorf("weights %.2f GiB a rank, want the checkpoint's 91 to 94", w)
	}
}

// The expert group is the tensor-parallel group alone, so with data parallelism the flag would
// widen it to tp x dp, which is not what the model runs: a pool that states dp 2 and no flag is
// priced as stated, not given a group the model never forms.
func TestInkling2WithDataParallelismKeepsTheFlagAsStated(t *testing.T) {
	m := openInkling2H200(t, false, 2, "")
	if got := m.k.Resolved().ExpertParallel(); got != 1 {
		t.Errorf("expert-parallel width %d (tp %d dp %d), want 1 as the pool states it", got, m.k.Resolved().TensorParallel(), m.k.Resolved().DataParallel())
	}
}
