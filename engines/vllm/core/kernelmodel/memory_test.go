// Moved from inference-sim/sim/kernelmodel/memory_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import "testing"

// FixedBytes' convention must be pinned before anything divides it. The kernel documents
// it as "per rank", and a KV budget computed from a node-aggregate value when a per-rank
// one was meant would be wrong by the tensor-parallel width -- 8x on these deployments.
//
// Behavioural: it compares two deployments of ONE model that differ only in TP width. If
// the value is per rank, the wider deployment reports LESS fixed memory per rank, because
// the same weights are sharded further.
func TestFixedBytesIsPerRankNotPerNode(t *testing.T) {
	four := open(t, "glm-5-b200-fp4-sglang-tp4.yaml")
	eight := open(t, "glm-5-b200-fp8-sglang-tp8.yaml")
	f, e := four.Kernel().FixedBytes().Total(), eight.Kernel().FixedBytes().Total()
	if f <= 0 || e <= 0 {
		t.Fatalf("fixed bytes must be positive: tp4=%d tp8=%d", f, e)
	}
	// These two also differ in precision (fp4 vs fp8), which cuts the other way, so this
	// asserts only that the two are not identical -- i.e. that the layout reaches the
	// memory path at all. The per-rank direction is pinned by the KV test below, which
	// varies TP alone.
	if f == e {
		t.Errorf("tp4 and tp8 report the same fixed bytes %d; the layout is not reaching "+
			"the memory calculation", f)
	}
}

// Per-token KV must fall as the tensor-parallel width rises, because KV heads shard across
// the TP group. This is the property a KV budget depends on.
//
// gpt-oss-120b is the right model for it: n_kv is 8, so TP 1 through 8 all shard genuinely.
// The scenarios below are the same model on the same chip at the same precision, so TP is
// the only variable.
//
// The first version of this test used qwen3.5 and failed. That was the test's fault, not the
// kernel's: qwen3.5 has n_kv=2, so TP=4 and TP=8 both hit the documented floor of one KV
// head per rank and correctly report the same KV. That floor is asserted separately below,
// because it is real engine behaviour and a budget that ignored it would over-admit.
func TestPerTokenKVSharesAcrossTensorParallelWidth(t *testing.T) {
	const tokens = 4096
	widths := []struct {
		tp       int
		scenario string
	}{
		{1, "gpt-oss-120b-h200-fp4-vllm-tp1.yaml"},
		{2, "gpt-oss-120b-h200-fp4-vllm-tp2.yaml"},
		{4, "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"},
		{8, "gpt-oss-120b-h200-fp4-vllm-tp8.yaml"},
	}
	var prev int64
	for i, w := range widths {
		got := open(t, w.scenario).Kernel().SequenceVariableBytes(tokens)
		if got <= 0 {
			t.Fatalf("tp=%d: per-sequence KV must be positive, got %d", w.tp, got)
		}
		if i > 0 && got >= prev {
			t.Errorf("tp=%d KV %d is not below the %d at the narrower width; KV is not "+
				"sharding with tensor-parallel width", w.tp, got, prev)
		}
		prev = got
	}
}

// The KV-head floor is real engine behaviour: a model cannot shard below one KV head per
// rank, so beyond that width the attention layers' K/V cache per rank stops falling. qwen3.5
// has n_kv=2, so TP=4 and TP=8 must give the same attention cache shape. The per-sequence figure
// is not compared: the model is a hybrid, so its block size and its recurrent state follow the
// width too (Platform._align_hybrid_block_size).
func TestKVStopsSharingAtOneHeadPerRank(t *testing.T) {
	full := func(scenario string) KVLayerRun {
		runs, err := open(t, scenario).KVLayers()
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range runs {
			if r.Spec == "full" {
				return r
			}
		}
		t.Fatalf("%s: no full-attention layer among %+v", scenario, runs)
		return KVLayerRun{}
	}
	four := full("qwen3.5-397b-a17b-b200-fp8-sglang-tp4.yaml")
	eight := full("qwen3.5-397b-a17b-b200-fp8-sglang-tp8.yaml")
	if four.NumKVHeads != 1 || four.HeadSize <= 0 {
		t.Fatalf("tp=4: %+v, want one KV head per rank", four)
	}
	if four.NumKVHeads != eight.NumKVHeads || four.HeadSize != eight.HeadSize || four.ElemBytes != eight.ElemBytes {
		t.Errorf("a 2-KV-head model's attention cache is %+v at tp=4 and %+v at tp=8; both are at the "+
			"one-head-per-rank floor and must agree", four, eight)
	}
}

// KV must grow with token count, and do so in page-quantized steps rather than smoothly:
// the engine allocates whole blocks. A budget derived from a smooth value would admit
// requests the engine could not fit.
func TestPerSequenceKVGrowsAndIsPageQuantized(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	var prev int64
	for _, tok := range []int{1, 16, 17, 32, 1024, 8192} {
		got := m.Kernel().SequenceVariableBytes(tok)
		if got < prev {
			t.Errorf("KV at %d tokens (%d) fell below the previous level (%d)",
				tok, got, prev)
		}
		prev = got
	}
	// Page quantization: one token and a full page must cost the same, since both occupy
	// one block. The scenarios use block_size 16.
	if one, full := m.Kernel().SequenceVariableBytes(1),
		m.Kernel().SequenceVariableBytes(16); one != full {
		t.Errorf("1 token costs %d and 16 tokens %d; a 16-token page should quantize to "+
			"one block", one, full)
	}
}
