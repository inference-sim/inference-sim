package metrics

import (
	"testing"
)

// A request admitted with a prefix hit, then prefilled in two chunks, then decoding, then
// finished: the counters move only where vLLM's logger moves them.
func TestCountersFollowTheSteps(t *testing.T) {
	m := New([]Bucketed{{Family: "vllm:iteration_tokens_total", Bounds: []float64{1, 8, 16}}})
	// Step 0: r admitted with 10 prompt tokens, 4 of them a local prefix hit; 4 computed now.
	m.Step(Step{
		Running: 1, Waiting: 0, KVUsage: 0.25,
		Admitted:  []Admission{{ID: "r", PromptTokens: 10, LocalHit: 4, Preempted: false, CacheLookup: true}},
		Scheduled: map[string]int{"r": 4},
	})
	// Step 1: the rest of the prompt; the first token is sampled.
	m.Step(Step{Running: 1, KVUsage: 0.5, Scheduled: map[string]int{"r": 2},
		Emitted: map[string]int{"r": 1}, FirstTokens: []string{"r"}})
	// Step 2: one decode, and it stops.
	m.Step(Step{Running: 0, KVUsage: 0, Scheduled: map[string]int{"r": 1},
		Emitted: map[string]int{"r": 1}, Finished: []Finish{{ID: "r", Reason: "stop", PromptTokens: 10, GenerationTokens: 2}}})

	want := map[string]float64{
		"vllm:num_requests_running":                                  0,
		"vllm:num_requests_waiting":                                  0,
		"vllm:kv_cache_usage_perc":                                   0,
		"vllm:prefix_cache_queries_total":                            10,
		"vllm:prefix_cache_hits_total":                               4,
		"vllm:prompt_tokens_total":                                   10,
		"vllm:prompt_tokens_by_source_total{source=local_compute}":   6,
		"vllm:prompt_tokens_by_source_total{source=local_cache_hit}": 4,
		"vllm:prompt_tokens_cached_total":                            4,
		"vllm:prompt_tokens_cached_by_source_total{source=device}":   4,
		"vllm:generation_tokens_total":                               2,
		"vllm:num_preemptions_total":                                 0,
		"vllm:request_success_total{finished_reason=stop}":           1,
		"vllm:request_success_total{finished_reason=length}":         0,
		"vllm:iteration_tokens_total_count":                          2,
		"vllm:iteration_tokens_total_sum":                            8,
		"vllm:request_prompt_tokens_sum":                             10,
		"vllm:request_generation_tokens_sum":                         2,
	}
	got := m.Series()
	for k, w := range want {
		if g, ok := got[k]; !ok || g != w {
			t.Errorf("%s = %v (present %v), want %v", k, g, ok, w)
		}
	}
}

// vLLM counts a prompt's tokens and cache hits once, at its first admission: a request
// resuming after preemption is counted under preempted_* (not exported) and its prompt
// stats are not set again (vllm/v1/metrics/stats.py PrefixCacheStats.record;
// vllm/v1/core/sched/scheduler.py:1074-1080@5df63a72).
func TestAResumedRequestIsNotCountedAgain(t *testing.T) {
	m := New(nil)
	m.Step(Step{Admitted: []Admission{{ID: "r", PromptTokens: 8, LocalHit: 0, CacheLookup: true}}, Scheduled: map[string]int{"r": 8},
		Emitted: map[string]int{"r": 1}, FirstTokens: []string{"r"}})
	m.Step(Step{Preempted: []string{"r"}})
	m.Step(Step{Admitted: []Admission{{ID: "r", PromptTokens: 9, LocalHit: 4, Preempted: true, CacheLookup: true}}, Scheduled: map[string]int{"r": 5}})
	// The preemption is not counted yet: vLLM records it as an event on the request, which
	// reaches the frontend with the request's next output (scheduler.py:1609-1610,
	// 2200-2274@5df63a72), and IterationStats counts it there (stats.py:574-577).
	if got := m.Series()["vllm:num_preemptions_total"]; got != 0 {
		t.Errorf("num_preemptions_total = %v before the request emits again, want 0", got)
	}
	m.Step(Step{Scheduled: map[string]int{"r": 1}, Emitted: map[string]int{"r": 1}})
	got := m.Series()
	for k, w := range map[string]float64{
		"vllm:prefix_cache_queries_total": 8,
		"vllm:prefix_cache_hits_total":    0,
		"vllm:prompt_tokens_total":        8,
		"vllm:num_preemptions_total":      1,
	} {
		if got[k] != w {
			t.Errorf("%s = %v, want %v", k, got[k], w)
		}
	}
}

// With prefix caching off no lookup runs, so nothing is counted as a prefix-cache query
// or hit (prefix_cache_lookup_enabled, vllm/v1/core/kv_cache_manager.py:249-262@5df63a72),
// though the prompt still is.
func TestNoLookupNoPrefixCacheQueries(t *testing.T) {
	m := New(nil)
	m.Step(Step{Admitted: []Admission{{ID: "r", PromptTokens: 8, CacheLookup: false}}, Scheduled: map[string]int{"r": 8},
		Emitted: map[string]int{"r": 1}, FirstTokens: []string{"r"}})
	got := m.Series()
	if got["vllm:prefix_cache_queries_total"] != 0 || got["vllm:prompt_tokens_total"] != 8 {
		t.Errorf("queries %v, prompt tokens %v; want 0 and 8", got["vllm:prefix_cache_queries_total"], got["vllm:prompt_tokens_total"])
	}
}

// A request that never reaches its first token before the run ends has its prompt stats
// still pending: vLLM emits them with the first output, so they are not counted yet.
func TestPromptStatsArriveWithTheFirstToken(t *testing.T) {
	m := New(nil)
	m.Step(Step{Admitted: []Admission{{ID: "r", PromptTokens: 32, LocalHit: 0, CacheLookup: true}}, Scheduled: map[string]int{"r": 16}})
	if got := m.Series()["vllm:prompt_tokens_total"]; got != 0 {
		t.Errorf("prompt_tokens_total = %v before the first token, want 0", got)
	}
	if got := m.Series()["vllm:prefix_cache_queries_total"]; got != 32 {
		t.Errorf("prefix_cache_queries_total = %v, want 32: the lookup is counted at admission", got)
	}
}
