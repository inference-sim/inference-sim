package sim

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestAdapterPrefetchCounts_AllocatedAndEmpty pins that the new counter follows the
// AdapterLoadCounts idiom: always non-nil so callers may increment without a nil check,
// but empty on an adapter-blind run so no adapter output is produced (INV-6).
func TestAdapterPrefetchCounts_AllocatedAndEmpty(t *testing.T) {
	m := NewMetrics()
	if m.AdapterPrefetchCounts == nil {
		t.Fatal("AdapterPrefetchCounts must be allocated by NewMetrics (nil map breaks increment)")
	}
	if len(m.AdapterPrefetchCounts) != 0 {
		t.Errorf("AdapterPrefetchCounts must start empty, got %v", m.AdapterPrefetchCounts)
	}
}

// TestPrefetchCount_OmittedWhenZero is the inertness guard for this field: an adapter
// that saw loads but no prefetch must serialize WITHOUT a prefetch_count key, so every
// pre-Spec-3 artifact shape stays byte-identical.
func TestPrefetchCount_OmittedWhenZero(t *testing.T) {
	blob, err := json.Marshal(AdapterMetrics{LoadCount: 3, EvictionCount: 1})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(blob), "prefetch_count") {
		t.Errorf("zero PrefetchCount must be omitted (omitempty), got %s", blob)
	}
}

// TestPrefetchCount_PresentWhenNonZero is the other half: the field is not merely
// absent, it appears once a tick has charged a prefetch.
func TestPrefetchCount_PresentWhenNonZero(t *testing.T) {
	blob, err := json.Marshal(AdapterMetrics{LoadCount: 3, PrefetchCount: 2})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(blob), `"prefetch_count":2`) {
		t.Errorf("non-zero PrefetchCount must serialize, got %s", blob)
	}
}

// TestBuildAdapterMetrics_SurfacesPrefetchOnlyAdapter covers the case the idSet union
// exists for: an adapter that was prefetched and evicted before any request completed
// must still appear in the output.
func TestBuildAdapterMetrics_SurfacesPrefetchOnlyAdapter(t *testing.T) {
	m := NewMetrics()
	m.AdapterLoadCounts["a1"] = 1
	m.AdapterPrefetchCounts["a1"] = 1

	got := buildAdapterMetrics(m, 1.0)
	am, ok := got["a1"]
	if !ok {
		t.Fatalf("adapter a1 must surface from prefetch/load counts alone; got %v", got)
	}
	if am.PrefetchCount != 1 || am.LoadCount != 1 {
		t.Errorf("LoadCount/PrefetchCount = %d/%d, want 1/1", am.LoadCount, am.PrefetchCount)
	}
}
