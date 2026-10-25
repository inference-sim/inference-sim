package sim

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// The gauges are the time-weighted mean of the step function set at each step's end, and
// _last is the value in effect at the bin's end.
func TestServerSeries_GaugeIsTimeWeightedMeanAndLastPoint(t *testing.T) {
	s := NewServerSeriesForTest(10)
	s.RecordGaugesForTest(0, 2, 0, 0)
	s.RecordGaugesForTest(5, 4, 0, 0)
	s.RecordGaugesForTest(15, 0, 0, 0)
	e := s.OutputForTest("e", 20, 2)
	if got := e.NumRequestsRunning; got[0] != 3 || got[1] != 2 {
		t.Fatalf("running means = %v, want [3 2]", got)
	}
	if got := e.NumRequestsRunningLast; got[0] != 4 || got[1] != 0 {
		t.Fatalf("running last = %v, want [4 0]", got)
	}
}

// vLLM's usage is 1 - free/(num_gpu_blocks - 1) (block_pool.py:880-891): the null block is
// out of the denominator, and a cached block no request holds is free.
func TestServerSeries_KVUsageExcludesNullBlockAndCountsCachedAsFree(t *testing.T) {
	kv := MustNewKVStoreFromConfig(NewKVCacheConfig(11, 16, 0, 0, 0, 0))
	tokens := make([]TokenID, 32)
	for i := range tokens {
		tokens[i] = TokenID(i + 1)
	}
	req := &Request{ID: "a", InputTokens: tokens}
	if !kv.AllocateKVBlocks(req, 0, 32, nil) {
		t.Fatal("allocate failed")
	}
	if got := KVCacheUsageForTest(kv); math.Abs(got-0.2) > 1e-12 {
		t.Fatalf("usage with 2 of 10 usable blocks held = %v, want 0.2", got)
	}
	kv.ReleaseKVBlocks(req)
	if len(kv.GetCachedBlocks(tokens)) == 0 {
		t.Fatal("released full blocks should stay cached")
	}
	if got := KVCacheUsageForTest(kv); got != 0 {
		t.Fatalf("usage with only cached-unreferenced blocks = %v, want 0", got)
	}
}

// A re-admission after preemption neither re-counts the lookup nor replaces the first
// scheduling's prompt split (scheduler.py:1034-1040).
func TestServerSeries_PromptSplitIsFirstSchedulingOnly(t *testing.T) {
	s := NewServerSeriesForTest(10)
	req := &Request{ID: "a", InputTokens: make([]TokenID, 100)}
	s.RecordAdmissionForTest(0, req, PrefixLookup{QueryTokens: 100, HitTokens: 48, CPUHitTokens: 16})
	req.SetPreemptionsForTest(1)
	s.RecordAdmissionForTest(12, req, PrefixLookup{}) // batch formation zeroes a re-admission's lookup
	s.RecordFirstTokenForTest(25, req)
	e := s.OutputForTest("e", 30, 3)
	if e.PrefixCacheQueries[0] != 100 || e.PrefixCacheQueries[1] != 0 {
		t.Fatalf("queries = %v, want [100 0 0]", e.PrefixCacheQueries)
	}
	if e.PromptTokens[2] != 100 || e.PromptTokensLocalCacheHit[2] != 48 ||
		e.PromptTokensExternalKVTransfer[2] != 16 || e.PromptTokensLocalCompute[2] != 36 {
		t.Fatalf("prompt by source at bin 2 = %d/%d/%d/%d, want 100/48/16/36", e.PromptTokens[2],
			e.PromptTokensLocalCacheHit[2], e.PromptTokensExternalKVTransfer[2], e.PromptTokensLocalCompute[2])
	}
}

// End to end: queries count at admission, prompt tokens when the first token is emitted
// (output_processor.py:697-707), and generation tokens at each step's end.
func TestServerSeries_PromptCountedAtFirstTokenNotAdmission(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.ServerSeriesIntervalUs = 1
	s := mustNewSimulator(t, cfg)
	req := &Request{ID: "r", ArrivalTime: 0, InputTokens: make([]TokenID, 64), OutputTokens: make([]TokenID, 3)}
	s.InjectArrival(req)
	s.Run()
	out := BuildServerSeriesOf([]string{"instance_0"}, []*Simulator{s})
	if out == nil || len(out.Engines) != 1 {
		t.Fatalf("series = %+v", out)
	}
	e := out.Engines[0]
	firstTokenAt := req.ArrivalTime + req.FirstTokenTime - s.LatencyModelForTest().OutputTokenProcessingTime()
	var admittedAt int64 = -1
	for k, q := range e.PrefixCacheQueries {
		if q > 0 {
			admittedAt = int64(k)
		}
	}
	if admittedAt < 0 || admittedAt >= firstTokenAt {
		t.Fatalf("admission bin %d must precede the first token's step end %d", admittedAt, firstTokenAt)
	}
	if e.PromptTokens[admittedAt] != 0 || e.PromptTokens[firstTokenAt] != 64 {
		t.Fatalf("prompt tokens at admission %d / first token %d = %d / %d, want 0 / 64",
			admittedAt, firstTokenAt, e.PromptTokens[admittedAt], e.PromptTokens[firstTokenAt])
	}
	var gen int64
	for _, g := range e.GenerationTokens {
		gen += g
	}
	if gen != 3 {
		t.Fatalf("generation tokens = %d, want 3", gen)
	}
}

// One entry per engine, each with its own values; off emits no block.
func TestServerSeries_PerEngineAndOff(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.ServerSeriesIntervalUs = 10
	a, b := mustNewSimulator(t, cfg), mustNewSimulator(t, cfg)
	a.ServerSeriesForTest().RecordGaugesForTest(0, 3, 1, 0)
	b.ServerSeriesForTest().RecordGaugesForTest(0, 0, 0, 0)
	a.ServerSeriesForTest().RecordGaugesForTest(10, 3, 1, 0)
	out := BuildServerSeriesOf([]string{"instance_0", "instance_1"}, []*Simulator{a, b})
	if len(out.Engines) != 2 || out.Engines[0].ID != "instance_0" || out.Engines[1].ID != "instance_1" {
		t.Fatalf("engines = %+v", out.Engines)
	}
	if out.Engines[0].NumRequestsRunning[0] != 3 || out.Engines[1].NumRequestsRunning[0] != 0 {
		t.Fatalf("per-engine running = %v / %v", out.Engines[0].NumRequestsRunning, out.Engines[1].NumRequestsRunning)
	}

	off := mustNewSimulator(t, newTestSimConfig())
	if BuildServerSeriesOf([]string{"instance_0"}, []*Simulator{off}) != nil {
		t.Fatal("off must build no series")
	}
	data, _ := json.Marshal(MetricsOutput{})
	if strings.Contains(string(data), "server_series") {
		t.Fatal("off must omit server_series")
	}
}
