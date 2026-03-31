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
		Prompt: strings.Repeat("hello ", 100),
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
		Prompt: strings.Repeat("hello ", 100),
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
		Prompt: strings.Repeat("hello ", 100),
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
		Prompt: strings.Repeat("hello ", 10),
	})
	if got := int(capturedBody["max_tokens"].(float64)); got != 512 {
		t.Errorf("max_tokens = %d, want 512", got)
	}

	// Zero MaxOutputTokens → default 2048
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 1, InputTokens: 10, MaxOutputTokens: 0,
		Prompt: strings.Repeat("hello ", 10),
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

	// Send() passes through req.Prompt verbatim
	expectedPrompt := strings.Repeat("hello ", 50)
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 50,
		Prompt: expectedPrompt,
	})
	prompt, ok := capturedBody["prompt"].(string)
	if !ok {
		t.Fatal("prompt not found in request body")
	}
	if prompt != expectedPrompt {
		t.Errorf("prompt not passed through: got length %d, want %d", len(prompt), len(expectedPrompt))
	}

	// Empty Prompt still works (server handles tokenization)
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 1, InputTokens: 0,
		Prompt: "hello ",
	})
	prompt, ok = capturedBody["prompt"].(string)
	if !ok || !strings.Contains(prompt, "hello") {
		t.Errorf("expected prompt to contain 'hello', got %q", prompt)
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
		Prompt: strings.Repeat("hello ", 10),
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

func TestRealClient_NonStreaming_ExtractsFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"text": "hello", "finish_reason": "stop"}},
			"usage":   map[string]interface{}{"prompt_tokens": 10.0, "completion_tokens": 5.0},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", record.FinishReason, "stop")
	}
}

func TestRealClient_Streaming_ExtractsFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// Content chunk
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok\"}}]}\n\n")
		flusher.Flush()
		// Final content chunk with finish_reason
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		// Usage-only chunk with empty choices (should not clear finish_reason)
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: true,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", record.FinishReason, "stop")
	}
	if record.OutputTokens != 2 {
		t.Errorf("OutputTokens = %d, want 2 (from usage-only chunk)", record.OutputTokens)
	}
}

// TestRealClient_Streaming_NullFinishReason verifies that JSON null finish_reason
// in intermediate SSE chunks (the standard vLLM format) does not clear finish_reason.
func TestRealClient_Streaming_NullFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// Intermediate chunk with explicit "finish_reason": null (JSON null → Go nil)
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"text\":\"tok\",\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		// Final content chunk with actual finish_reason
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"text\":\"end\",\"finish_reason\":\"length\"}]}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 5, Streaming: true,
		Prompt: strings.Repeat("hello ", 5),
	})
	if err != nil {
		t.Fatal(err)
	}
	// JSON null must not overwrite: final chunk's "length" should be retained
	if record.FinishReason != "length" {
		t.Errorf("FinishReason = %q, want %q (null in intermediate chunk must not overwrite)", record.FinishReason, "length")
	}
}

func TestRealClient_ChatFormat_UsesMessagesEndpoint(t *testing.T) {
	var capturedBody map[string]interface{}
	var capturedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		resp := map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{
				"message":       map[string]interface{}{"role": "assistant", "content": "hi"},
				"finish_reason": "stop",
			}},
			"usage": map[string]interface{}{"prompt_tokens": 10.0, "completion_tokens": 1.0},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	record, _ := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: false,
		Prompt: "Hello, world!",
	})

	// Endpoint must be /v1/chat/completions
	if capturedPath != "/v1/chat/completions" {
		t.Errorf("endpoint = %q, want /v1/chat/completions", capturedPath)
	}
	// Body must use messages array, not prompt
	if _, ok := capturedBody["prompt"]; ok {
		t.Error("chat format should not send 'prompt' key")
	}
	msgs, ok := capturedBody["messages"].([]interface{})
	if !ok || len(msgs) == 0 {
		t.Fatal("chat format should send 'messages' array")
	}
	msg0, ok := msgs[0].(map[string]interface{})
	if !ok {
		t.Fatal("messages[0] should be an object")
	}
	if msg0["role"] != "user" || msg0["content"] != "Hello, world!" {
		t.Errorf("messages[0] = %v, want role=user content='Hello, world!'", msg0)
	}
	if record.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", record.FinishReason)
	}
}

func TestRealClient_StreamingChat_ExtractsFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1}}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: true,
		Prompt: "Hello",
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", record.FinishReason)
	}
}

func TestRealClient_Unconstrained_Completions_SetsMaxInt32(t *testing.T) {
	var capturedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody = nil
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		resp := map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"text": "ok"}},
			"usage":   map[string]interface{}{"prompt_tokens": 10.0, "completion_tokens": 5.0},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// completions + unconstrained: max_tokens = MaxInt32
	client := NewRealClient(server.URL, "", "test-model", "vllm")
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Unconstrained: true,
		Prompt: strings.Repeat("hello ", 10),
	})
	maxTokens, ok := capturedBody["max_tokens"].(float64)
	if !ok {
		t.Fatal("max_tokens not found for completions + unconstrained")
	}
	if int(maxTokens) != 2147483647 { // math.MaxInt32
		t.Errorf("max_tokens = %v, want MaxInt32 (2147483647)", maxTokens)
	}
}

func TestRealClient_Unconstrained_Chat_OmitsMaxTokens(t *testing.T) {
	var capturedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody = nil
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		resp := map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{
				"message": map[string]interface{}{"role": "assistant", "content": "ok"},
			}},
			"usage": map[string]interface{}{"prompt_tokens": 10.0, "completion_tokens": 5.0},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// chat + unconstrained: max_tokens omitted
	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Unconstrained: true,
		Prompt: "Hello",
	})
	if _, ok := capturedBody["max_tokens"]; ok {
		t.Error("chat + unconstrained should omit max_tokens")
	}
}

func TestRealClient_Streaming_SetsStreamOptions(t *testing.T) {
	var capturedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody = nil
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		isStreaming := false
		if s, ok := capturedBody["stream"].(bool); ok {
			isStreaming = s
		}
		if isStreaming {
			flusher, ok := w.(http.Flusher)
			if !ok {
				t.Fatal("expected http.Flusher")
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok\"}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1}}\n\n")
			flusher.Flush()
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			flusher.Flush()
		} else {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"choices": []interface{}{map[string]interface{}{"text": "ok"}},
				"usage":   map[string]interface{}{"prompt_tokens": 10.0, "completion_tokens": 1.0},
			})
		}
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")

	// Streaming: stream_options present
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: true,
		Prompt: strings.Repeat("hello ", 10),
	})
	streamOpts, ok := capturedBody["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatal("stream_options not found in request body for streaming request")
	}
	if includeUsage, ok := streamOpts["include_usage"].(bool); !ok || !includeUsage {
		t.Errorf("stream_options.include_usage = %v, want true", streamOpts["include_usage"])
	}

	// Non-streaming: stream_options absent
	_, _ = client.Send(context.Background(), &PendingRequest{
		RequestID: 1, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if _, ok := capturedBody["stream_options"]; ok {
		t.Error("stream_options should not be present for non-streaming request")
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

func TestRecorder_PrefixGroupPropagation(t *testing.T) {
	rec := &Recorder{}
	rec.RecordRequest(
		&PendingRequest{
			RequestID:    0,
			InputTokens:  200,
			PrefixGroup:  "shared",
			PrefixLength: 128,
		},
		&RequestRecord{RequestID: 0, Status: "ok"},
		0, "", 0,
	)
	records := rec.Records()
	if len(records) != 1 {
		t.Fatalf("got %d records, want 1", len(records))
	}
	if records[0].PrefixGroup != "shared" {
		t.Errorf("PrefixGroup = %q, want %q", records[0].PrefixGroup, "shared")
	}
	if records[0].PrefixLength != 128 {
		t.Errorf("PrefixLength = %d, want 128", records[0].PrefixLength)
	}
	// InputTokens in trace is suffix-only: 200 - 128 = 72
	if records[0].InputTokens != 72 {
		t.Errorf("InputTokens = %d, want 72 (200 - 128 suffix-only)", records[0].InputTokens)
	}
}

// TestRealClient_GIEHeaders_SentWhenNonEmpty verifies BC-2: GIE headers are
// sent when TenantID and SLOClass are populated. SLOClass is sent as the
// x-gateway-inference-objective header (the name of an InferenceObjective CRD
// on the target cluster); TenantID is sent as x-gateway-inference-fairness-id
// for per-tenant fair-share scheduling. GIE's EPP resolves the objective name
// to an integer priority via CRD lookup — the client does not send priority
// as a header.
func TestRealClient_GIEHeaders_SentWhenNonEmpty(t *testing.T) {
	var capturedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"text":"ok"}],"usage":{"prompt_tokens":10,"completion_tokens":1}}`)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	_, err := client.Send(context.Background(), &PendingRequest{
		RequestID:   0,
		InputTokens: 10,
		Prompt:      "hello",
		TenantID:    "tenant-a",
		SLOClass:    "critical",
		GIEPriority: 3,
	})
	if err != nil {
		t.Fatal(err)
	}

	if got := capturedHeaders.Get("x-gateway-inference-fairness-id"); got != "tenant-a" {
		t.Errorf("x-gateway-inference-fairness-id = %q, want %q", got, "tenant-a")
	}
	if got := capturedHeaders.Get("x-gateway-inference-objective"); got != "critical" {
		t.Errorf("x-gateway-inference-objective = %q, want %q", got, "critical")
	}
}

// TestRealClient_GIEHeaders_OmittedWhenDefault verifies BC-3: no GIE headers
// when fields are empty/zero (avoids noise on non-GIE servers).
func TestRealClient_GIEHeaders_OmittedWhenDefault(t *testing.T) {
	var capturedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"text":"ok"}],"usage":{"prompt_tokens":10,"completion_tokens":1}}`)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	_, err := client.Send(context.Background(), &PendingRequest{
		RequestID:   0,
		InputTokens: 10,
		Prompt:      "hello",
	})
	if err != nil {
		t.Fatal(err)
	}

	if got := capturedHeaders.Get("x-gateway-inference-fairness-id"); got != "" {
		t.Errorf("x-gateway-inference-fairness-id should be absent, got %q", got)
	}
	if got := capturedHeaders.Get("x-gateway-inference-objective"); got != "" {
		t.Errorf("x-gateway-inference-objective should be absent, got %q", got)
	}
}
