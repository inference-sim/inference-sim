package vllm

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// A request whose prompt loads back from the CPU tier is counted like any other: its prompt is in
// prompt_tokens_total when its first token comes, the tokens the tier supplied are
// prompt_tokens_by_source{external_kv_transfer}, and computed + local hit + external == total
// (PromptTokenStats, vllm/v1/metrics/stats.py:426-471@4ba787ec). The scheduler sets the request's
// PrefillStats at its first lookup, before the asynchronous load parks it
// (vllm/v1/core/sched/scheduler.py:1250-1257), so the load that lands later is no exception.
func TestOffloadLoadedPromptIsCountedBySource(t *testing.T) {
	e := mustEngine(t, cpuTier+tightGPU)
	a := req("a", 96, 2)
	if err := e.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	_, _, now := runOne(t, e, 5000)
	// A request of other tokens evicts a's prompt from the GPU's prefix cache.
	xx := req("xx", 140, 2)
	if err := e.Add(xx, now); err != nil {
		t.Fatal(err)
	}
	_, _, now = runOne(t, e, 10)
	b := req("b", 0, 3)
	b.Prompt = append(append([]int32(nil), a.Prompt...), 7, 8, 9, 10)
	if err := e.Add(b, now); err != nil {
		t.Fatal(err)
	}
	loaded := 0
	var lookup *engine.Scheduled // b's lookup, as its batch reports it
	var offloaded int
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
		for i, sr := range batch.Reqs {
			if sr.ID == "b" && sr.PrefixLookup {
				lookup, offloaded = &batch.Reqs[i], e.OffloadCachedTokens("b")
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
	s := e.metrics.Series()
	const src = "vllm:prompt_tokens_by_source_total{source="
	total := float64(len(a.Prompt) + len(xx.Prompt) + len(b.Prompt))
	if got := s["vllm:prompt_tokens_total"]; got != total {
		t.Errorf("prompt_tokens_total = %v, want %v: every request's prompt, the loaded one's too", got, total)
	}
	if got := s[src+"external_kv_transfer}"]; got != float64(loaded) {
		t.Errorf("external_kv_transfer = %v, want the %d tokens the CPU tier loaded", got, loaded)
	}
	sum := s[src+"local_compute}"] + s[src+"local_cache_hit}"] + s[src+"external_kv_transfer}"]
	if sum != total {
		t.Errorf("sources sum to %v, want prompt_tokens_total %v", sum, total)
	}
	if got, want := s["vllm:prompt_tokens_cached_total"], s[src+"local_cache_hit}"]+s[src+"external_kv_transfer}"]; got != want {
		t.Errorf("prompt_tokens_cached_total = %v, want local + external = %v", got, want)
	}
	// The connector's own lookup: it was asked about the prompt past the local hit, and hit
	// what it loaded (scheduler.py:1211-1214, 1486-1506).
	if got := s["vllm:external_prefix_cache_hits_total"]; got != float64(loaded) {
		t.Errorf("external_prefix_cache_hits = %v, want %d", got, loaded)
	}
	if s["vllm:external_prefix_cache_queries_total"] < float64(loaded) {
		t.Errorf("external_prefix_cache_queries = %v, want at least the %d hit", s["vllm:external_prefix_cache_queries_total"], loaded)
	}
	// The tokens the tier supplied are named by their source, and not counted as the device's.
	if got := s["vllm:prompt_tokens_cached_by_source_total{source=host}"]; got != float64(loaded) {
		t.Errorf("prompt_tokens_cached_by_source{host} = %v, want %d", got, loaded)
	}
	// The simulator reads the request's lookup from the batch that first runs it, after the load.
	if lookup == nil {
		t.Fatal("no batch reported b's prefix lookup")
	}
	if offloaded != loaded || lookup.CachedTokens < loaded {
		t.Errorf("b's batch reports %d cached tokens, %d of them from the offload tier; the tier loaded %d", lookup.CachedTokens, offloaded, loaded)
	}
}
