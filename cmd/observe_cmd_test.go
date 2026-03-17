package cmd

import (
	"testing"
)

func TestRecordRequest_PopulatesArrivalTimeAndSessionFields(t *testing.T) {
	recorder := &Recorder{}
	pending := &PendingRequest{
		RequestID:   1,
		InputTokens: 100,
		Model:       "test-model",
		Streaming:   true,
		ClientID:    "client-1",
		TenantID:    "tenant-1",
		SLOClass:    "standard",
	}
	result := &RequestRecord{
		RequestID:         1,
		OutputTokens:      50,
		ServerInputTokens: 95,
		Status:            "ok",
		SendTimeUs:        1000000,
		FirstChunkTimeUs:  1000100,
		LastChunkTimeUs:   1000500,
		NumChunks:         10,
	}

	recorder.RecordRequest(pending, result, 500000, "session-1", 0)

	records := recorder.Records()
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	r := records[0]
	if r.ArrivalTimeUs != 500000 {
		t.Errorf("ArrivalTimeUs: got %d, want 500000", r.ArrivalTimeUs)
	}
	if r.SessionID != "session-1" {
		t.Errorf("SessionID: got %q, want %q", r.SessionID, "session-1")
	}
	if r.RoundIndex != 0 {
		t.Errorf("RoundIndex: got %d, want 0", r.RoundIndex)
	}
}
