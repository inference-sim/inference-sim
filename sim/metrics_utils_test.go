package sim

import "testing"

func TestNewRequestMetrics_PropagatesAllFields(t *testing.T) {
	// GIVEN a request with all metadata fields populated
	req := &Request{
		ID:              "test_req_1",
		ArrivalTime:     2000000, // 2 seconds in ticks
		InputTokens:     make([]int, 128),
		OutputTokens:    make([]int, 64),
		SLOClass:        "realtime",
		TenantID:        "tenant_alpha",
		AssignedInstance: "instance_3",
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
	if rm.SLOClass != "realtime" {
		t.Errorf("SLOClass: got %q, want %q", rm.SLOClass, "realtime")
	}
	if rm.TenantID != "tenant_alpha" {
		t.Errorf("TenantID: got %q, want %q", rm.TenantID, "tenant_alpha")
	}
	if rm.HandledBy != "instance_3" {
		t.Errorf("HandledBy: got %q, want %q", rm.HandledBy, "instance_3")
	}
}

func TestNewRequestMetrics_ZeroValueFields_OmittedInJSON(t *testing.T) {
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
}
