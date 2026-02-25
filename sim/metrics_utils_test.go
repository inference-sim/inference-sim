package sim

import "testing"

// TestNewRequestMetrics_PropagatesAllFields is intentionally a construction-site guard test
// (CLAUDE.md antipattern #4). It verifies the canonical constructor propagates all fields,
// preventing silent field-zero bugs when new fields are added to RequestMetrics.
func TestNewRequestMetrics_PropagatesAllFields(t *testing.T) {
	// GIVEN a request with all metadata fields populated
	req := &Request{
		ID:              "test_req_1",
		ArrivalTime:     2000000, // 2 seconds in ticks
		InputTokens:     make([]int, 128),
		OutputTokens:    make([]int, 64),
		SLOClass:        "critical",
		TenantID:        "tenant_alpha",
		AssignedInstance: "instance_3",
		Model:           "llama-3.1-8b",
	}
	arrivedAt := float64(req.ArrivalTime) / 1e6

	// WHEN NewRequestMetrics is called
	rm := NewRequestMetrics(req, arrivedAt)

	// THEN all fields MUST be propagated
	if rm.ID != "test_req_1" {
		t.Errorf("ID: got %q, want %q", rm.ID, "test_req_1")
	}
	if rm.ArrivedAt != 2.0 {
		t.Errorf("ArrivedAt: got %f, want 2.0", rm.ArrivedAt)
	}
	if rm.NumPrefillTokens != 128 {
		t.Errorf("NumPrefillTokens: got %d, want 128", rm.NumPrefillTokens)
	}
	if rm.NumDecodeTokens != 64 {
		t.Errorf("NumDecodeTokens: got %d, want 64", rm.NumDecodeTokens)
	}
	if rm.SLOClass != "critical" {
		t.Errorf("SLOClass: got %q, want %q", rm.SLOClass, "critical")
	}
	if rm.TenantID != "tenant_alpha" {
		t.Errorf("TenantID: got %q, want %q", rm.TenantID, "tenant_alpha")
	}
	if rm.HandledBy != "instance_3" {
		t.Errorf("HandledBy: got %q, want %q", rm.HandledBy, "instance_3")
	}
	if rm.Model != "llama-3.1-8b" {
		t.Errorf("Model: got %q, want %q", rm.Model, "llama-3.1-8b")
	}
}

func TestNewRequestMetrics_ZeroValueFields_AreEmptyStrings(t *testing.T) {
	// GIVEN a request with empty metadata (typical CSV trace)
	req := &Request{
		ID:           "csv_req_1",
		ArrivalTime:  1000000,
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 5),
	}

	// WHEN NewRequestMetrics is called
	rm := NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)

	// THEN metadata fields MUST be empty strings (will be omitted in JSON via omitempty)
	if rm.SLOClass != "" {
		t.Errorf("SLOClass: got %q, want empty", rm.SLOClass)
	}
	if rm.TenantID != "" {
		t.Errorf("TenantID: got %q, want empty", rm.TenantID)
	}
	if rm.HandledBy != "" {
		t.Errorf("HandledBy: got %q, want empty", rm.HandledBy)
	}
	if rm.Model != "" {
		t.Errorf("Model: got %q, want empty", rm.Model)
	}
}

// TestCalculatePercentile_EmptyInput_ReturnsZero verifies BC-6, BC-8.
func TestCalculatePercentile_EmptyInput_ReturnsZero(t *testing.T) {
	// GIVEN empty float64 slice
	// WHEN CalculatePercentile is called
	result := CalculatePercentile([]float64{}, 99)
	// THEN it returns 0 (not panic)
	if result != 0.0 {
		t.Errorf("expected 0.0 for empty input, got %f", result)
	}

	// Also verify with int64 (generic constraint covers both)
	resultInt := CalculatePercentile([]int64{}, 50)
	if resultInt != 0.0 {
		t.Errorf("expected 0.0 for empty int64 input, got %f", resultInt)
	}
}

// TestCalculatePercentile_SingleElement_ReturnsScaled verifies BC-7.
func TestCalculatePercentile_SingleElement_ReturnsScaled(t *testing.T) {
	// GIVEN a single-element slice
	// WHEN CalculatePercentile is called
	result := CalculatePercentile([]float64{1000.0}, 99)
	// THEN it returns the element divided by 1000 (ms conversion)
	if result != 1.0 {
		t.Errorf("expected 1.0 for single element 1000.0, got %f", result)
	}
}
