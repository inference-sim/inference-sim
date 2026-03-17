package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRealClient_NonStreaming_RecordsTokenCounts(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello world"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100.0, "completion_tokens": 50.0},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 100, Streaming: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.OutputTokens != 50 {
		t.Errorf("output_tokens = %d, want 50", record.OutputTokens)
	}
	if record.Status != "ok" {
		t.Errorf("status = %q, want ok", record.Status)
	}
	if record.SendTimeUs == 0 {
		t.Error("send_time not recorded")
	}
	if record.NumChunks != 1 {
		t.Errorf("num_chunks = %d, want 1 (non-streaming)", record.NumChunks)
	}
	if record.ServerInputTokens != 100 {
		t.Errorf("ServerInputTokens = %d, want 100", record.ServerInputTokens)
	}
}

func TestRealClient_Streaming_RecordsFirstAndLastChunkTime(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 5; i++ {
			_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok\"}}]}\n\n")
			flusher.Flush()
			time.Sleep(5 * time.Millisecond)
		}
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5}}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 1, InputTokens: 100, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.OutputTokens != 5 {
		t.Errorf("output_tokens = %d, want 5", record.OutputTokens)
	}
	if record.NumChunks < 5 {
		t.Errorf("num_chunks = %d, want >= 5", record.NumChunks)
	}
	if record.FirstChunkTimeUs == 0 {
		t.Error("first_chunk_time not recorded")
	}
	if record.LastChunkTimeUs <= record.FirstChunkTimeUs {
		t.Error("last_chunk_time should be > first_chunk_time for streaming")
	}
	if record.Status != "ok" {
		t.Errorf("status = %q, want ok", record.Status)
	}
	if record.ServerInputTokens != 100 {
		t.Errorf("ServerInputTokens = %d, want 100", record.ServerInputTokens)
	}
}

func TestRealClient_ServerError_RecordsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, "internal server error")
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 2, InputTokens: 100, Streaming: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "error" {
		t.Errorf("status = %q, want error", record.Status)
	}
	if record.ErrorMessage == "" {
		t.Error("expected error message for server error")
	}
}

func TestRecorder_ConcurrentAccess(t *testing.T) {
	rec := &Recorder{}
	done := make(chan struct{})
	for i := 0; i < 10; i++ {
		go func(id int) {
			defer func() { done <- struct{}{} }()
			rec.RecordRequest(
				&PendingRequest{RequestID: id, ClientID: "c1"},
				&RequestRecord{RequestID: id, Status: "ok"},
				0, "", 0,
			)
		}(i)
	}
	for i := 0; i < 10; i++ {
		<-done
	}
	records := rec.Records()
	if len(records) != 10 {
		t.Errorf("recorded %d, want 10", len(records))
	}
}

func TestRealClient_MaxOutputTokens_FlowsThrough(t *testing.T) {
	var capturedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		resp := map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "ok"}},
			"usage":   map[string]interface{}{"prompt_tokens": 10.0, "completion_tokens": 5.0},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")

	// Explicit MaxOutputTokens
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, MaxOutputTokens: 512,
	})
	if got := int(capturedBody["max_tokens"].(float64)); got != 512 {
		t.Errorf("max_tokens = %d, want 512", got)
	}

	// Zero MaxOutputTokens → default 2048
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 1, InputTokens: 10, MaxOutputTokens: 0,
	})
	if got := int(capturedBody["max_tokens"].(float64)); got != 2048 {
		t.Errorf("max_tokens = %d, want 2048 (default)", got)
	}
}

func TestRealClient_ProportionalPrompt(t *testing.T) {
	var capturedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		resp := map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "ok"}},
			"usage":   map[string]interface{}{"prompt_tokens": 50.0, "completion_tokens": 5.0},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 50,
	})
	prompt, ok := capturedBody["prompt"].(string)
	if !ok {
		t.Fatal("prompt not found in request body")
	}
	count := strings.Count(prompt, "hello ")
	if count != 50 {
		t.Errorf("prompt contains %d 'hello ' repetitions, want 50", count)
	}

	// BC-6: Zero InputTokens guard — prompt must not be empty
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 1, InputTokens: 0,
	})
	prompt, ok = capturedBody["prompt"].(string)
	if !ok || !strings.Contains(prompt, "hello") {
		t.Errorf("zero InputTokens should produce at least one 'hello', got %q", prompt)
	}

	// BC-6: Negative InputTokens — should also produce at least one "hello"
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 2, InputTokens: -5,
	})
	prompt, ok = capturedBody["prompt"].(string)
	if !ok || !strings.Contains(prompt, "hello") {
		t.Errorf("negative InputTokens should produce at least one 'hello', got %q", prompt)
	}
}

func TestRealClient_NonStreaming_TTFTBeforeE2E(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		data := []byte(`{"choices":[{"text":"hello world"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`)
		_, _ = w.Write(data[:10])
		flusher.Flush()
		time.Sleep(50 * time.Millisecond)
		_, _ = w.Write(data[10:])
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.FirstChunkTimeUs == 0 {
		t.Error("FirstChunkTimeUs not recorded")
	}
	if record.LastChunkTimeUs == 0 {
		t.Error("LastChunkTimeUs not recorded")
	}
	if record.FirstChunkTimeUs > record.LastChunkTimeUs {
		t.Errorf("FirstChunkTimeUs (%d) > LastChunkTimeUs (%d)", record.FirstChunkTimeUs, record.LastChunkTimeUs)
	}
	// With 50ms sleep, there should be measurable separation (10ms threshold = 5x margin)
	if record.LastChunkTimeUs-record.FirstChunkTimeUs < 10_000 {
		t.Errorf("expected >= 10ms separation, got %d us", record.LastChunkTimeUs-record.FirstChunkTimeUs)
	}
}

func TestRecorder_WiresModelAndServerInputTokens(t *testing.T) {
	rec := &Recorder{}
	rec.RecordRequest(
		&PendingRequest{RequestID: 0, ClientID: "c1", Model: "test-model"},
		&RequestRecord{RequestID: 0, Status: "ok", ServerInputTokens: 42},
		0, "", 0,
	)
	records := rec.Records()
	if len(records) != 1 {
		t.Fatalf("got %d records, want 1", len(records))
	}
	if records[0].Model != "test-model" {
		t.Errorf("Model = %q, want %q", records[0].Model, "test-model")
	}
	if records[0].ServerInputTokens != 42 {
		t.Errorf("ServerInputTokens = %d, want 42", records[0].ServerInputTokens)
	}
}
