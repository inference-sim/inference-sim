package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
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
