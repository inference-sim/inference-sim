package sim

import (
	"encoding/json"
	"strings"
	"testing"
)

// newAdapterTestMetrics constructs a Metrics with two adapter-attributed completed
// requests, one base-model completed request, and a positive runtime.
func newAdapterTestMetrics() *Metrics {
	m := NewMetrics()
	m.SimEndedTime = 1_000_000 // 1s in µs
	// adapter_0: two requests; adapter_1: one; base model: one.
	m.Requests["r0"] = RequestMetrics{ID: "r0", ArrivedAt: 1, NumDecodeTokens: 10, Adapter: "adapter_0"}
	m.Requests["r1"] = RequestMetrics{ID: "r1", ArrivedAt: 2, NumDecodeTokens: 20, Adapter: "adapter_0"}
	m.Requests["r2"] = RequestMetrics{ID: "r2", ArrivedAt: 3, NumDecodeTokens: 30, Adapter: "adapter_1"}
	m.Requests["rb"] = RequestMetrics{ID: "rb", ArrivedAt: 4, NumDecodeTokens: 40, Adapter: ""}
	for _, id := range []string{"r0", "r1", "r2", "rb"} {
		m.RequestTTFTs[id] = 8000.0 // µs (ticks)
		m.RequestE2Es[id] = 50000.0 // completed marker
	}
	m.CompletedRequests = 4
	m.TotalOutputTokens = 100
	return m
}

// TestBuildOutput_PerAdapterMetrics verifies the per-adapter aggregate block (US1):
// present with per-adapter TTFT/throughput when adapters are attributed; base-model
// requests attributed to no adapter (contracts/metrics.md).
func TestBuildOutput_PerAdapterMetrics(t *testing.T) {
	m := newAdapterTestMetrics()
	out := m.BuildOutput("cluster", nil)

	if out.Adapters == nil {
		t.Fatal("expected non-nil Adapters block when adapters are attributed")
	}
	if len(out.Adapters) != 2 {
		t.Fatalf("expected 2 adapters (adapter_0, adapter_1), got %d: %v", len(out.Adapters), out.Adapters)
	}
	if _, ok := out.Adapters[""]; ok {
		t.Error("base-model requests (empty adapter) must NOT form an adapter entry")
	}
	a0, ok := out.Adapters["adapter_0"]
	if !ok {
		t.Fatal("adapter_0 missing")
	}
	if a0.TTFTP50Us <= 0 {
		t.Errorf("adapter_0 ttft_p50_us should be > 0, got %v", a0.TTFTP50Us)
	}
	if a0.ThroughputTokPerS <= 0 {
		t.Errorf("adapter_0 throughput_tok_per_s should be > 0, got %v", a0.ThroughputTokPerS)
	}
	// This fixture records only request metrics — no resident-set events — so
	// load/eviction counts are 0 (they are driven by the scheduling hook, exercised
	// in the run-level tests). Counts surfacing from the maps is covered separately
	// by TestBuildOutput_AdapterEventCountsSurface.
	if a0.LoadCount != 0 || a0.EvictionCount != 0 {
		t.Errorf("no resident-set events in this fixture: counts should be 0, got %d/%d", a0.LoadCount, a0.EvictionCount)
	}
}

// TestBuildOutput_AdapterEventCountsSurface verifies buildAdapterMetrics carries
// per-adapter load/eviction counts into the output, including for an adapter that
// saw resident-set events but completed no request in-window (loaded-then-evicted).
func TestBuildOutput_AdapterEventCountsSurface(t *testing.T) {
	m := newAdapterTestMetrics()
	m.AdapterLoadCounts["adapter_0"] = 3
	m.AdapterEvictionCounts["adapter_0"] = 1
	// adapter_2 saw events but has no completed request attributed to it.
	m.AdapterLoadCounts["adapter_2"] = 2
	m.AdapterEvictionCounts["adapter_2"] = 2

	out := m.BuildOutput("cluster", nil)

	a0 := out.Adapters["adapter_0"]
	if a0.LoadCount != 3 || a0.EvictionCount != 1 {
		t.Errorf("adapter_0 counts = %d/%d, want 3/1", a0.LoadCount, a0.EvictionCount)
	}
	if a0.TTFTP50Us <= 0 {
		t.Errorf("adapter_0 should still carry TTFT from its completed requests, got %v", a0.TTFTP50Us)
	}
	a2, ok := out.Adapters["adapter_2"]
	if !ok {
		t.Fatal("adapter_2 (events but no completion) must still surface in the adapters block")
	}
	if a2.LoadCount != 2 || a2.EvictionCount != 2 {
		t.Errorf("adapter_2 counts = %d/%d, want 2/2", a2.LoadCount, a2.EvictionCount)
	}
	if a2.TTFTP50Us != 0 || a2.ThroughputTokPerS != 0 {
		t.Errorf("adapter_2 has no completed request: TTFT/throughput should be 0, got %v/%v", a2.TTFTP50Us, a2.ThroughputTokPerS)
	}
}

// TestBuildOutput_NoAdapters_OmitsBlock is the INV-6 companion: with no adapter-attributed
// requests, the adapters block is absent from the JSON entirely (no stdout change).
func TestBuildOutput_NoAdapters_OmitsBlock(t *testing.T) {
	m := NewMetrics()
	m.SimEndedTime = 1_000_000
	m.Requests["rb"] = RequestMetrics{ID: "rb", ArrivedAt: 1, NumDecodeTokens: 40} // no adapter
	m.RequestTTFTs["rb"] = 8000.0
	m.RequestE2Es["rb"] = 50000.0
	m.CompletedRequests = 1

	out := m.BuildOutput("cluster", nil)
	if out.Adapters != nil {
		t.Errorf("expected nil Adapters when none attributed, got %v", out.Adapters)
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(data), "\"adapters\"") {
		t.Errorf("INV-6: 'adapters' key must be absent from JSON when unconfigured:\n%s", data)
	}
}

// TestBuildOutput_AdapterKeysSorted verifies map keys serialize in sorted order (R2,
// determinism) regardless of insertion order.
func TestBuildOutput_AdapterKeysSorted(t *testing.T) {
	m := NewMetrics()
	m.SimEndedTime = 1_000_000
	for i, ad := range []string{"zeta", "alpha", "mu"} {
		id := string(rune('a' + i))
		m.Requests[id] = RequestMetrics{ID: id, ArrivedAt: float64(i), NumDecodeTokens: 10, Adapter: ad}
		m.RequestTTFTs[id] = 8000.0
		m.RequestE2Es[id] = 50000.0
	}
	m.CompletedRequests = 3

	out := m.BuildOutput("cluster", nil)
	data, _ := json.Marshal(out)
	s := string(data)
	ia := strings.Index(s, "\"alpha\"")
	im := strings.Index(s, "\"mu\"")
	iz := strings.Index(s, "\"zeta\"")
	if ia < 0 || ia >= im || im >= iz {
		t.Errorf("adapter keys must serialize sorted (alpha<mu<zeta); positions a=%d m=%d z=%d", ia, im, iz)
	}
}

// TestBuildOutput_AdapterPartition is the INV-1 companion (T018): per-adapter attribution
// partitions the completed requests — each completed request belongs to exactly one
// adapter group or the base-model group, never both, and the groups sum to the total.
func TestBuildOutput_AdapterPartition(t *testing.T) {
	m := newAdapterTestMetrics()

	adapterCount := 0
	baseCount := 0
	for id, rm := range m.Requests {
		if m.RequestE2Es[id] <= 0 {
			continue
		}
		if rm.Adapter != "" {
			adapterCount++
		} else {
			baseCount++
		}
	}
	if adapterCount+baseCount != m.CompletedRequests {
		t.Errorf("partition violated: adapter(%d) + base(%d) != completed(%d)",
			adapterCount, baseCount, m.CompletedRequests)
	}
	if adapterCount != 3 || baseCount != 1 {
		t.Errorf("expected 3 adapter-attributed + 1 base, got %d + %d", adapterCount, baseCount)
	}
}
