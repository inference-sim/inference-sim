package vllm

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// dsv41Offload is a DeepSeek V4.1-shaped engine (dsv41Layers: a bounded-replay window, the
// compressed cache and a ring) with a CPU tier and a GPU too small to keep one prompt past the next.
const dsv41Offload = "kv_offload: {connector: OffloadingConnector, cpu_bytes: 52428800, bytes_per_block: 4096}\n" +
	"block_size: 256\nnum_gpu_blocks: 20\nmax_model_len: 1024\nmax_num_batched_tokens: 256\n" + dsv41Layers

// vLLM offloads only the prefix-cacheable KV cache groups (get_offloading_group_ids), so beside
// DeepSeek V4.1's window, compressed cache and ring the connector takes the compressed cache alone,
// at its own 256-token block. A prompt stored once loads back after the GPU evicted it, and the load
// computes only what the CPU tier lacks. Before, the engine handed the connector all three groups and
// refused the 32-token window against the 256-token hash block.
func TestOffloadBesideDSV41GroupsLoadsOnlyTheCacheableGroup(t *testing.T) {
	e := mustEngine(t, dsv41Offload)
	a := req("a", 512, 2)
	if err := e.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	_, started, now := runOne(t, e, 5000)
	stored := 0
	for _, tr := range started {
		if tr.Kind == engine.TransferStore && tr.Tier == "cpu" {
			stored += tr.Tokens
		}
	}
	if stored == 0 {
		t.Fatal("a's prompt was not stored to the CPU tier")
	}
	// Other prompts (distinct tokens: req derives them from the id's length) push a out of the GPU's prefix cache.
	for i, id := range []string{"yy", "zzz", "wwww"} {
		if err := e.Add(req(id, 640, 2), now); err != nil {
			t.Fatalf("%d: %v", i, err)
		}
		_, _, now = runOne(t, e, 10)
	}
	b := req("b", 0, 2)
	b.Prompt = append(append([]int32(nil), a.Prompt...), 7, 8, 9, 10)
	if err := e.Add(b, now); err != nil {
		t.Fatal(err)
	}
	loaded := 0
	for steps := 0; steps < 1000 && (e.s.HasRequests() || loaded == 0); steps++ {
		batch, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		for _, tr := range batch.TransfersStarted {
			if tr.Kind == engine.TransferLoad {
				loaded += tr.Tokens
			}
			if err := e.TransferDone(tr.ID, true, now+1); err != nil {
				t.Fatal(err)
			}
		}
		now += 2
		if batch.Empty() {
			continue
		}
		if _, err := e.Complete(batch.ID, engine.Outcome{}, now); err != nil {
			t.Fatal(err)
		}
	}
	if loaded == 0 {
		t.Fatal("b's prompt did not load from the CPU tier")
	}
	if loaded%256 != 0 {
		t.Errorf("loaded %d tokens, want whole 256-token blocks of the cacheable group", loaded)
	}
}
