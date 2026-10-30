package vllm

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
)

// kvEventsOf runs requests a and then b (b sharing a's prompt) through an engine built from extra
// with its KV events turned on after construction, as a simulator does, and returns every event
// the results carried and the engine's scheduler.
func kvEventsOf(t *testing.T, extra string) ([]engine.KVEvent, *Engine) {
	t.Helper()
	eng, err := build(t, extra)()
	if err != nil {
		t.Fatal(err)
	}
	e := eng.(*Engine)
	e.EnableKVEvents()
	var out []engine.KVEvent
	now := engine.Time(0)
	run := func(r engine.Request) {
		if err := e.Add(r, now); err != nil {
			t.Fatal(err)
		}
		for e.s.HasRequests() {
			b, err := e.Schedule(now)
			if err != nil {
				t.Fatal(err)
			}
			if b.Empty() {
				now += 1000
				continue
			}
			now += e.StepTime(b)
			res, err := e.Complete(b.ID, engine.Outcome{}, now)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, res.KVEvents...)
		}
	}
	run(req("a", 80, 4))
	r := req("a", 80, 4)
	r.ID = "b"
	run(r)
	return out, e
}

// An engine whose KV events a simulator turns on publishes the blocks its prefix cache stores.
// On a hybrid model (a full-attention and a sliding-window group, Muse Spark's shape) it
// publishes only the full-attention group's: vLLM tags each event with its group's kind and
// llm-d's indexer drops a sliding window's, so the stream a router indexes is the same.
func TestKVEventsAreTheFullAttentionGroups(t *testing.T) {
	const hybrid = "kv_layers:\n" +
		"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n" +
		"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n"
	full, _ := kvEventsOf(t, "")
	mixed, e := kvEventsOf(t, hybrid)
	if e.s.KV.NumGroups() != 2 {
		t.Fatalf("the hybrid engine has %d KV cache groups, want 2", e.s.KV.NumGroups())
	}
	stored := func(evs []engine.KVEvent) []string {
		var hs []string
		for _, ev := range evs {
			if !ev.Removed {
				hs = append(hs, ev.Hash)
			}
		}
		return hs
	}
	// 80 prompt tokens fill five 16-token blocks, stored by request a; b's hit stops one token
	// short of its prompt, so it recomputes and stores the fifth again, as vLLM's stream does.
	if got := stored(full); len(got) != 6 || got[5] != got[4] {
		t.Fatalf("a full-attention engine published stored blocks %v, want a's five and b's last again", got)
	}
	got, want := stored(mixed), stored(full)
	if len(got) != len(want) {
		t.Fatalf("the hybrid engine published %d stored blocks, want its full-attention group's %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("stored block %d is %s, want %s", i, got[i], want[i])
		}
	}
	for _, ev := range mixed {
		if ev.Removed {
			t.Errorf("published a removal %s: the full-attention group evicted nothing, so it is the window's", ev.Hash)
		}
	}
}

// Events stay off until a simulator turns them on: a batch reports none.
func TestKVEventsAreOffUntilEnabled(t *testing.T) {
	eng, err := build(t, "")()
	if err != nil {
		t.Fatal(err)
	}
	if err := eng.Add(req("a", 40, 2), 0); err != nil {
		t.Fatal(err)
	}
	b, err := eng.Schedule(0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := eng.Complete(b.ID, engine.Outcome{}, eng.StepTime(b))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.KVEvents) != 0 {
		t.Fatalf("events off, yet the batch reported %v", res.KVEvents)
	}
}

// Under eviction pressure a hybrid engine's published stream stays one a router can index: every
// removal names a block the stream stored and has not removed since. A sliding window's block
// shares its full-attention block's hash, so a window's removal passed through would unindex a
// block still cached, and the full-attention group's own removal would later name a block the
// stream no longer holds.
func TestHybridKVEventsStayIndexableUnderEviction(t *testing.T) {
	const hybrid = "num_gpu_blocks: 24\nmax_model_len: 128\nkv_layers:\n" +
		"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n" +
		"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n"
	eng, err := build(t, hybrid)()
	if err != nil {
		t.Fatal(err)
	}
	e := eng.(*Engine)
	e.EnableKVEvents()
	held := map[string]int{}
	removals := 0
	now := engine.Time(0)
	for i := 0; i < 12; i++ {
		r := req(string(rune('a'+i)), 64, 3)
		if err := e.Add(r, now); err != nil {
			t.Fatal(err)
		}
		for e.s.HasRequests() {
			b, err := e.Schedule(now)
			if err != nil {
				t.Fatal(err)
			}
			now += 1000
			if b.Empty() {
				continue
			}
			res, err := e.Complete(b.ID, engine.Outcome{}, now)
			if err != nil {
				t.Fatal(err)
			}
			for _, ev := range res.KVEvents {
				if !ev.Removed {
					held[ev.Hash]++
					continue
				}
				removals++
				if held[ev.Hash] == 0 {
					t.Fatalf("request %d: removal of %s, which the published stream does not hold", i, ev.Hash)
				}
				held[ev.Hash] = 0
			}
		}
	}
	if removals == 0 {
		t.Fatal("no eviction: the pool is too large to exercise removals")
	}
}
