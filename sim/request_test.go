package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRequestState_Constants_HaveExpectedStringValues(t *testing.T) {
	// BC-6: Typed constants replace raw strings
	assert.Equal(t, RequestState("queued"), StateQueued)
	assert.Equal(t, RequestState("running"), StateRunning)
	assert.Equal(t, RequestState("completed"), StateCompleted)
}

func TestRequest_String_IncludesState(t *testing.T) {
	req := Request{ID: "test-1", State: StateQueued}
	s := req.String()
	assert.Contains(t, s, "queued")
}

func TestNewRequest_RequiredFields_SetCorrectly(t *testing.T) {
	// GIVEN required field values
	id := "req_42"
	arrivalTime := int64(5000)
	inputTokens := []int{1, 2, 3}
	outputTokens := []int{4, 5}

	// WHEN NewRequest is called
	req := NewRequest(id, arrivalTime, inputTokens, outputTokens)

	// THEN required fields MUST match
	if req.ID != id {
		t.Errorf("ID = %q, want %q", req.ID, id)
	}
	if req.ArrivalTime != arrivalTime {
		t.Errorf("ArrivalTime = %d, want %d", req.ArrivalTime, arrivalTime)
	}
	if len(req.InputTokens) != len(inputTokens) {
		t.Fatalf("InputTokens length = %d, want %d", len(req.InputTokens), len(inputTokens))
	}
	for i, tok := range req.InputTokens {
		if tok != inputTokens[i] {
			t.Errorf("InputTokens[%d] = %d, want %d", i, tok, inputTokens[i])
		}
	}
	if len(req.OutputTokens) != len(outputTokens) {
		t.Fatalf("OutputTokens length = %d, want %d", len(req.OutputTokens), len(outputTokens))
	}
	for i, tok := range req.OutputTokens {
		if tok != outputTokens[i] {
			t.Errorf("OutputTokens[%d] = %d, want %d", i, tok, outputTokens[i])
		}
	}
}

func TestNewRequest_DefaultState_IsQueued(t *testing.T) {
	// GIVEN any valid required fields
	// WHEN NewRequest is called
	req := NewRequest("r1", 0, []int{1}, []int{2})

	// THEN State MUST be StateQueued
	if req.State != StateQueued {
		t.Errorf("State = %q, want %q", req.State, StateQueued)
	}
}

func TestNewRequest_OptionalFields_AreZeroValues(t *testing.T) {
	// GIVEN a request created with only required fields
	req := NewRequest("r1", 0, []int{1}, []int{2})

	// THEN all optional fields MUST be zero values
	if req.TenantID != "" {
		t.Errorf("TenantID = %q, want empty", req.TenantID)
	}
	if req.SLOClass != "" {
		t.Errorf("SLOClass = %q, want empty", req.SLOClass)
	}
	if req.Streaming {
		t.Error("Streaming = true, want false")
	}
	if req.SessionID != "" {
		t.Errorf("SessionID = %q, want empty", req.SessionID)
	}
	if req.RoundIndex != 0 {
		t.Errorf("RoundIndex = %d, want 0", req.RoundIndex)
	}
	if req.ReasonRatio != 0.0 {
		t.Errorf("ReasonRatio = %f, want 0.0", req.ReasonRatio)
	}
	if req.ScheduledStepIdx != 0 {
		t.Errorf("ScheduledStepIdx = %d, want 0", req.ScheduledStepIdx)
	}
	if req.FinishedStepIdx != 0 {
		t.Errorf("FinishedStepIdx = %d, want 0", req.FinishedStepIdx)
	}
	if req.AssignedInstance != "" {
		t.Errorf("AssignedInstance = %q, want empty", req.AssignedInstance)
	}
	if req.Priority != 0.0 {
		t.Errorf("Priority = %f, want 0.0", req.Priority)
	}
	if req.TextTokenCount != 0 {
		t.Errorf("TextTokenCount = %d, want 0", req.TextTokenCount)
	}
	if req.ImageTokenCount != 0 {
		t.Errorf("ImageTokenCount = %d, want 0", req.ImageTokenCount)
	}
	if req.AudioTokenCount != 0 {
		t.Errorf("AudioTokenCount = %d, want 0", req.AudioTokenCount)
	}
	if req.VideoTokenCount != 0 {
		t.Errorf("VideoTokenCount = %d, want 0", req.VideoTokenCount)
	}
	if req.ProgressIndex != 0 {
		t.Errorf("ProgressIndex = %d, want 0", req.ProgressIndex)
	}
	if req.TTFTSet {
		t.Error("TTFTSet = true, want false")
	}
	if req.FirstTokenTime != 0 {
		t.Errorf("FirstTokenTime = %d, want 0", req.FirstTokenTime)
	}
	if req.NumNewTokens != 0 {
		t.Errorf("NumNewTokens = %d, want 0", req.NumNewTokens)
	}
	if req.ITL != nil {
		t.Errorf("ITL = %v, want nil", req.ITL)
	}
}

func TestNewRequest_OptionalFields_Settable(t *testing.T) {
	// GIVEN a request created via NewRequest
	req := NewRequest("r1", 1000, []int{1}, []int{2})

	// WHEN optional fields are set after construction
	req.TenantID = "tenant_a"
	req.SLOClass = "realtime"
	req.Streaming = true
	req.SessionID = "sess_1"
	req.RoundIndex = 3
	req.ReasonRatio = 0.42

	// THEN those fields MUST reflect the assigned values
	if req.TenantID != "tenant_a" {
		t.Errorf("TenantID = %q, want %q", req.TenantID, "tenant_a")
	}
	if req.SLOClass != "realtime" {
		t.Errorf("SLOClass = %q, want %q", req.SLOClass, "realtime")
	}
	if !req.Streaming {
		t.Error("Streaming = false, want true")
	}
	if req.SessionID != "sess_1" {
		t.Errorf("SessionID = %q, want %q", req.SessionID, "sess_1")
	}
	if req.RoundIndex != 3 {
		t.Errorf("RoundIndex = %d, want 3", req.RoundIndex)
	}
	if req.ReasonRatio != 0.42 {
		t.Errorf("ReasonRatio = %f, want 0.42", req.ReasonRatio)
	}
}
