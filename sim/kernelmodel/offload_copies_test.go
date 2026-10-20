package kernelmodel

import "testing"

// A latent (MLA) cache is the same on every TP rank, so vLLM's replicated layout keeps one copy
// in the CPU tier; a head-sharded cache keeps one per TP worker (offloading/config.py:158-171,
// kv_offload/cpu/spec.py:103-121 at v0.31.0).
func TestOffloadCopiesFollowTheKVLayout(t *testing.T) {
	if got := open(t, "deepseek-v3-h200-fp8-sglang-tp8.yaml").OffloadCopies(); got != 1 {
		t.Errorf("MLA at TP8: %d copies, want 1 (replicated layout)", got)
	}
	if got := open(t, "llama-3.1-70b-instruct-h100-fp8-vllm-tp4.yaml").OffloadCopies(); got != 4 {
		t.Errorf("GQA at TP4: %d copies, want 4", got)
	}
}

// GLM-5's DSA indexer keys are one head held alike on every rank (max_tp_shards 1,
// deepseek_v2.py:661 at v0.31.0), so its cache stays replicated.
func TestOffloadCopiesIgnoreTheIndexer(t *testing.T) {
	if got := open(t, "glm-5-h200-fp8-sglang-tp8.yaml").OffloadCopies(); got != 1 {
		t.Errorf("sparse MLA with an indexer at TP8: %d copies, want 1", got)
	}
}
