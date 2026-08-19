package workload

import (
	"math"
	"testing"
)

func TestParsePromMetrics_SumsLabelSeriesAndFoldsTotal(t *testing.T) {
	text := `# HELP vllm:kv_offload_tiering_block_hits Number of block hits.
# TYPE vllm:kv_offload_tiering_block_hits counter
vllm:kv_offload_tiering_block_hits{model="x",tier="cpu"} 100
vllm:kv_offload_tiering_block_hits{model="x",tier="disk"} 40

vllm:kv_offload_tiering_block_queries{model="x",tier="cpu"} 120 1699999999999
vllm:kv_offload_tiering_block_queries{model="x",tier="disk"} 80
# a comment
vllm:gpu_prefix_cache_hits_total 55
irrelevant_metric{a="b"} 9`
	got := ParsePromMetrics(text)
	if got[PromTieringBlockHits] != 140 {
		t.Errorf("block_hits summed = %v, want 140", got[PromTieringBlockHits])
	}
	if got[PromTieringBlockQueries] != 200 {
		t.Errorf("block_queries summed = %v, want 200 (trailing timestamp ignored)", got[PromTieringBlockQueries])
	}
	// `_total` folds into the base family name.
	if got[PromGPUPrefixCacheHits] != 55 {
		t.Errorf("gpu hits (_total folded) = %v, want 55", got[PromGPUPrefixCacheHits])
	}
}

func TestParsePromMetrics_ToleratesMalformedValue(t *testing.T) {
	text := "vllm:kv_offload_tiering_block_hits notanumber\nvllm:kv_offload_tiering_block_queries 10\n"
	got := ParsePromMetrics(text)
	if _, ok := got[PromTieringBlockHits]; ok {
		t.Errorf("malformed value line must be skipped, got %v", got[PromTieringBlockHits])
	}
	if got[PromTieringBlockQueries] != 10 {
		t.Errorf("valid line after malformed one must still parse, got %v", got[PromTieringBlockQueries])
	}
}

func TestDeriveObservedHitRate_TieredDelta(t *testing.T) {
	start := map[string]float64{
		PromTieringBlockHits: 1000, PromTieringBlockQueries: 2000,
		PromTieringReadTime: 5, PromTieringWriteTime: 2,
	}
	end := map[string]float64{
		PromTieringBlockHits: 1734, PromTieringBlockQueries: 3000,
		PromTieringReadTime: 17.5, PromTieringWriteTime: 5.25,
	}
	got, err := DeriveObservedHitRate(start, end, "abc123")
	if err != nil {
		t.Fatal(err)
	}
	if got.Source != ObservedKVSourceTiered {
		t.Errorf("source = %q, want tiered", got.Source)
	}
	// hits delta 734 / queries delta 1000 = 0.734
	if math.Abs(got.HitRate-0.734) > 1e-9 {
		t.Errorf("hit_rate = %v, want 0.734", got.HitRate)
	}
	if got.BlockHits != 734 || got.BlockQueries != 1000 {
		t.Errorf("deltas = %d/%d, want 734/1000", got.BlockHits, got.BlockQueries)
	}
	if math.Abs(got.ReadTimeTotal-12.5) > 1e-9 || math.Abs(got.WriteTimeTotal-3.25) > 1e-9 {
		t.Errorf("read/write time deltas = %v/%v, want 12.5/3.25", got.ReadTimeTotal, got.WriteTimeTotal)
	}
	if got.VLLMCommit != "abc123" {
		t.Errorf("commit = %q", got.VLLMCommit)
	}
}

func TestDeriveObservedHitRate_FallbackToGPUCounters(t *testing.T) {
	start := map[string]float64{PromGPUPrefixCacheHits: 10, PromGPUPrefixCacheQueries: 100}
	end := map[string]float64{PromGPUPrefixCacheHits: 60, PromGPUPrefixCacheQueries: 200}
	got, err := DeriveObservedHitRate(start, end, "")
	if err != nil {
		t.Fatal(err)
	}
	if got.Source != ObservedKVSourceGPUCache {
		t.Errorf("source = %q, want gpu-prefix-cache-fallback", got.Source)
	}
	// hits delta 50 / queries delta 100 = 0.5
	if math.Abs(got.HitRate-0.5) > 1e-9 {
		t.Errorf("hit_rate = %v, want 0.5", got.HitRate)
	}
}

func TestDeriveObservedHitRate_FallbackToGaugeWhenOnlyRatePresent(t *testing.T) {
	end := map[string]float64{PromGPUPrefixCacheHitRate: 0.42}
	got, err := DeriveObservedHitRate(map[string]float64{}, end, "")
	if err != nil {
		t.Fatal(err)
	}
	if got.Source != ObservedKVSourceGPUCache || math.Abs(got.HitRate-0.42) > 1e-9 {
		t.Errorf("gauge fallback = %+v, want source gpu-prefix-cache-fallback hit_rate 0.42", got)
	}
}

func TestDeriveObservedHitRate_TieredPreferredOverGPU(t *testing.T) {
	start := map[string]float64{PromTieringBlockHits: 0, PromTieringBlockQueries: 0, PromGPUPrefixCacheHits: 0, PromGPUPrefixCacheQueries: 0}
	end := map[string]float64{PromTieringBlockHits: 8, PromTieringBlockQueries: 10, PromGPUPrefixCacheHits: 99, PromGPUPrefixCacheQueries: 100}
	got, err := DeriveObservedHitRate(start, end, "")
	if err != nil {
		t.Fatal(err)
	}
	if got.Source != ObservedKVSourceTiered || math.Abs(got.HitRate-0.8) > 1e-9 {
		t.Errorf("tiered must win when present: %+v", got)
	}
}

func TestDeriveObservedHitRate_NoRecognizedCounters(t *testing.T) {
	_, err := DeriveObservedHitRate(map[string]float64{}, map[string]float64{"vllm:something_else": 5}, "")
	if err == nil {
		t.Fatal("expected error when no recognized counter family is present")
	}
}

func TestDeriveObservedHitRate_ZeroQueriesGuard(t *testing.T) {
	start := map[string]float64{PromTieringBlockHits: 5, PromTieringBlockQueries: 100}
	end := map[string]float64{PromTieringBlockHits: 5, PromTieringBlockQueries: 100} // no activity
	if _, err := DeriveObservedHitRate(start, end, ""); err == nil {
		t.Fatal("expected zero-queries division guard to error (R11)")
	}
}

func TestDeriveObservedHitRate_CounterResetGuard(t *testing.T) {
	start := map[string]float64{PromTieringBlockHits: 500, PromTieringBlockQueries: 1000}
	end := map[string]float64{PromTieringBlockHits: 10, PromTieringBlockQueries: 20} // restarted
	if _, err := DeriveObservedHitRate(start, end, ""); err == nil {
		t.Fatal("expected counter-reset guard to error")
	}
}
