package cmd

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// TestScrapeKVMetrics_ParsesServedMetrics verifies the HTTP scrape + parse path
// against a server exposing tiering counters (#1583).
func TestScrapeKVMetrics_ParsesServedMetrics(t *testing.T) {
	body := "# TYPE vllm:kv_offload_tiering_block_hits counter\n" +
		"vllm:kv_offload_tiering_block_hits{tier=\"cpu\"} 30\n" +
		"vllm:kv_offload_tiering_block_hits{tier=\"disk\"} 12\n" +
		"vllm:kv_offload_tiering_block_queries 100\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	client := NewRealClient(srv.URL, "", "m", "vllm")
	got, err := client.ScrapeKVMetrics(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if got[workload.PromTieringBlockHits] != 42 {
		t.Errorf("block_hits = %v, want 42 (summed across tiers)", got[workload.PromTieringBlockHits])
	}
	if got[workload.PromTieringBlockQueries] != 100 {
		t.Errorf("block_queries = %v, want 100", got[workload.PromTieringBlockQueries])
	}
}

// TestResolveObservedKVMetrics_EndToEnd verifies the end-of-window scrape + derive
// wiring: given a start snapshot and a server serving the end snapshot, the derived
// header block reflects the windowed delta (BC-1). Values keep hits <= queries so the
// derived hit-rate stays in [0,1].
func TestResolveObservedKVMetrics_EndToEnd(t *testing.T) {
	// End-of-window server state: 500 hits / 1200 queries.
	end := "vllm:kv_offload_tiering_block_hits 500\nvllm:kv_offload_tiering_block_queries 1200\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(end))
	}))
	defer srv.Close()
	client := NewRealClient(srv.URL, "", "m", "vllm")

	// Start-of-window snapshot: 200 hits / 600 queries. Window delta = (500-200)/(1200-600)
	// = 300/600 = 0.5 (strictly interior, hits <= queries).
	start := map[string]float64{
		workload.PromTieringBlockHits:    200,
		workload.PromTieringBlockQueries: 600,
	}
	block := resolveObservedKVMetrics(context.Background(), client, "", start, "pinned-sha")
	if block == nil {
		t.Fatal("expected a derived observed-KV-metrics block")
	}
	// (500-200)/(1200-600) = 300/600 = 0.5.
	if block.Source != workload.ObservedKVSourceTiered {
		t.Errorf("source = %q, want tiered", block.Source)
	}
	if block.BlockHits != 300 || block.BlockQueries != 600 {
		t.Errorf("deltas = %d/%d, want 300/600", block.BlockHits, block.BlockQueries)
	}
	if block.HitRate != 0.5 {
		t.Errorf("hit_rate = %v, want 0.5", block.HitRate)
	}
	if block.VLLMCommit != "pinned-sha" {
		t.Errorf("commit = %q, want pinned-sha", block.VLLMCommit)
	}
}

// TestResolveObservedKVMetrics_DisabledReturnsNil verifies a nil start snapshot
// (scrape disabled or start-scrape failed) yields nil with no second scrape (BC-8).
func TestResolveObservedKVMetrics_DisabledReturnsNil(t *testing.T) {
	if got := resolveObservedKVMetrics(context.Background(), nil, "", nil, ""); got != nil {
		t.Errorf("nil start snapshot must yield nil, got %+v", got)
	}
}

// TestResolveObservedKVMetrics_ScrapeMissWarnsAndOmits verifies an unreachable /metrics
// (HTTP 404) at end-of-window warns and omits the block rather than aborting (BC-11).
func TestResolveObservedKVMetrics_ScrapeMissWarnsAndOmits(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()
	client := NewRealClient(srv.URL, "", "m", "vllm")
	start := map[string]float64{workload.PromTieringBlockQueries: 1, workload.PromTieringBlockHits: 1}
	if got := resolveObservedKVMetrics(context.Background(), client, "", start, ""); got != nil {
		t.Errorf("a 404 /metrics scrape must yield nil (graceful omit), got %+v", got)
	}
}

// TestResolveObservedKVMetrics_NoCountersWarnsAndOmits verifies that a reachable
// /metrics with no recognized KV counters warns and omits the block (BC-11).
func TestResolveObservedKVMetrics_NoCountersWarnsAndOmits(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		_, _ = fmt.Fprint(w, "vllm:num_requests_running 3\n")
	}))
	defer srv.Close()
	client := NewRealClient(srv.URL, "", "m", "vllm")
	start := map[string]float64{"vllm:num_requests_running": 2}
	if got := resolveObservedKVMetrics(context.Background(), client, "", start, ""); got != nil {
		t.Errorf("no recognized counters must yield nil, got %+v", got)
	}
}
