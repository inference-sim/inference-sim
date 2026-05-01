package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
)

// testLogHook captures logrus warn/error entries for test assertions.
type testLogHook struct {
	mu      sync.Mutex
	entries []string
}

func (h *testLogHook) Levels() []logrus.Level {
	return []logrus.Level{logrus.WarnLevel, logrus.ErrorLevel}
}

func (h *testLogHook) Fire(e *logrus.Entry) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.entries = append(h.entries, e.Message)
	return nil
}

func (h *testLogHook) hasEntry(substr string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, msg := range h.entries {
		if strings.Contains(msg, substr) {
			return true
		}
	}
	return false
}

// installLogHook replaces logrus hooks for the test duration and returns the hook.
func installLogHook(t *testing.T) *testLogHook {
	t.Helper()
	h := &testLogHook{}
	orig := logrus.StandardLogger().Hooks
	logrus.StandardLogger().Hooks = logrus.LevelHooks{}
	logrus.AddHook(h)
	t.Cleanup(func() { logrus.StandardLogger().Hooks = orig })
	return h
}

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

// TestSend_AbortAlwaysWarns verifies that finish_reason="abort" produces a warning
// regardless of MinTokens, since abort is a server-side error (preemption, client
// disconnect, engine cancellation), not expected behavior.
func TestSend_AbortAlwaysWarns(t *testing.T) {
	tests := []struct {
		name      string
		streaming bool
	}{
		{"non-streaming", false},
		{"streaming", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hook := installLogHook(t)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.streaming {
					flusher, ok := w.(http.Flusher)
					if !ok {
						t.Fatal("expected http.Flusher")
					}
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"abort\"}]}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
					flusher.Flush()
				} else {
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(map[string]interface{}{
						"choices": []map[string]interface{}{
							{"text": "ok", "finish_reason": "abort"},
						},
						"usage": map[string]interface{}{"completion_tokens": 3, "prompt_tokens": 5},
					})
				}
			}))
			defer server.Close()

			client := NewRealClient(server.URL, "", "test-model", "vllm")
			// MinTokens > 0: abort must still produce a warning.
			record, err := client.Send(context.Background(), &PendingRequest{
				RequestID: 1, InputTokens: 5, Streaming: tc.streaming,
				Prompt: "hello", MaxOutputTokens: 128, MinTokens: 128,
			})
			if err != nil {
				t.Fatal(err)
			}
			if record.FinishReason != "abort" {
				t.Errorf("FinishReason = %q, want %q (abort must not be suppressed by MinTokens)", record.FinishReason, "abort")
			}
			if !hook.hasEntry("server aborted request") {
				t.Error("expected abort warning to be logged, but no matching entry found")
			}
		})
	}
}

// TestSend_LengthSuppressedWithMinTokens verifies that finish_reason="length" does NOT
// produce a warning in exact-length mode (MinTokens == MaxOutputTokens), because
// vLLM stops at max_tokens as intended in the canonical min_tokens==max_tokens case.
func TestSend_LengthSuppressedWithMinTokens(t *testing.T) {
	tests := []struct {
		name      string
		streaming bool
	}{
		{"non-streaming", false},
		{"streaming", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hook := installLogHook(t)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.streaming {
					flusher, ok := w.(http.Flusher)
					if !ok {
						t.Fatal("expected http.Flusher")
					}
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":128}}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
					flusher.Flush()
				} else {
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(map[string]interface{}{
						"choices": []map[string]interface{}{
							{"text": "ok", "finish_reason": "length"},
						},
						"usage": map[string]interface{}{"completion_tokens": 128, "prompt_tokens": 5},
					})
				}
			}))
			defer server.Close()

			client := NewRealClient(server.URL, "", "test-model", "vllm")
			record, err := client.Send(context.Background(), &PendingRequest{
				RequestID: 1, InputTokens: 5, Streaming: tc.streaming,
				Prompt: "hello", MaxOutputTokens: 128, MinTokens: 128,
			})
			if err != nil {
				t.Fatal(err)
			}
			// FinishReason is still recorded correctly — only the warn is suppressed.
			if record.FinishReason != "length" {
				t.Errorf("FinishReason = %q, want %q", record.FinishReason, "length")
			}
			if hook.hasEntry("output may be truncated") {
				t.Error("expected no truncation warning in exact-length mode, but warning was logged")
			}
		})
	}
}

// TestSend_MinTokens_ChatFormat verifies that min_tokens is included in the request body
// for the chat API format, not just the completions format.
func TestSend_MinTokens_ChatFormat(t *testing.T) {
	var receivedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&receivedBody)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]string{"content": "ok"}, "finish_reason": "stop"},
			},
			"usage": map[string]interface{}{"completion_tokens": 10, "prompt_tokens": 5},
		})
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))

	req := &PendingRequest{Prompt: "hello", MaxOutputTokens: 256, MinTokens: 64}
	_, _ = client.Send(context.Background(), req)
	if v, ok := receivedBody["min_tokens"]; !ok {
		t.Error("min_tokens not found in chat API request body")
	} else if int(v.(float64)) != 64 {
		t.Errorf("min_tokens = %v, want 64", v)
	}
}

// TestSend_LengthWarnsWhenMinTokensNotSet verifies that the original truncation warning
// behaviour is preserved: finish_reason="length" without --min-tokens still fires.
func TestSend_LengthWarnsWhenMinTokensNotSet(t *testing.T) {
	tests := []struct {
		name      string
		streaming bool
	}{
		{"non-streaming", false},
		{"streaming", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hook := installLogHook(t)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.streaming {
					flusher, ok := w.(http.Flusher)
					if !ok {
						t.Fatal("expected http.Flusher")
					}
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":50}}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
					flusher.Flush()
				} else {
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(map[string]interface{}{
						"choices": []map[string]interface{}{
							{"text": "ok", "finish_reason": "length"},
						},
						"usage": map[string]interface{}{"completion_tokens": 50, "prompt_tokens": 5},
					})
				}
			}))
			defer server.Close()

			client := NewRealClient(server.URL, "", "test-model", "vllm")
			// MinTokens=0: truncation warning must still fire.
			record, err := client.Send(context.Background(), &PendingRequest{
				RequestID: 1, InputTokens: 5, Streaming: tc.streaming,
				Prompt: "hello", MaxOutputTokens: 256, MinTokens: 0,
			})
			if err != nil {
				t.Fatal(err)
			}
			if record.FinishReason != "length" {
				t.Errorf("FinishReason = %q, want %q", record.FinishReason, "length")
			}
			if !hook.hasEntry("output may be truncated") {
				t.Error("expected truncation warning to be logged, but no matching entry found")
			}
		})
	}
}

// TestSend_LengthWarnsWhenMinTokensBelowMax verifies that the truncation warning fires
// when min_tokens is set but below max_tokens — the output might still be truncated.
func TestSend_LengthWarnsWhenMinTokensBelowMax(t *testing.T) {
	tests := []struct {
		name      string
		streaming bool
	}{
		{"non-streaming", false},
		{"streaming", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hook := installLogHook(t)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.streaming {
					flusher, ok := w.(http.Flusher)
					if !ok {
						t.Fatal("expected http.Flusher")
					}
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":50}}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
					flusher.Flush()
				} else {
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(map[string]interface{}{
						"choices": []map[string]interface{}{
							{"text": "ok", "finish_reason": "length"},
						},
						"usage": map[string]interface{}{"completion_tokens": 50, "prompt_tokens": 5},
					})
				}
			}))
			defer server.Close()

			client := NewRealClient(server.URL, "", "test-model", "vllm")
			// MinTokens=10, MaxOutputTokens=256: not exact-length mode, warning must fire.
			record, err := client.Send(context.Background(), &PendingRequest{
				RequestID: 1, InputTokens: 5, Streaming: tc.streaming,
				Prompt: "hello", MaxOutputTokens: 256, MinTokens: 10,
			})
			if err != nil {
				t.Fatal(err)
			}
			if record.FinishReason != "length" {
				t.Errorf("FinishReason = %q, want %q", record.FinishReason, "length")
			}
			if !hook.hasEntry("output may be truncated") {
				t.Error("expected truncation warning to be logged, but no matching entry found")
			}
		})
	}
}

// TestSend_StopWarnsWhenBelowMinTokens verifies that finish_reason="stop" with
// outputTokens < minTokens triggers the "server may not support min_tokens" warning.
// This detects servers that silently ignore the min_tokens parameter.
func TestSend_StopWarnsWhenBelowMinTokens(t *testing.T) {
	tests := []struct {
		name      string
		streaming bool
	}{
		{"non-streaming", false},
		{"streaming", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hook := installLogHook(t)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.streaming {
					flusher, ok := w.(http.Flusher)
					if !ok {
						t.Fatal("expected http.Flusher")
					}
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":5}}\n\n")
					flusher.Flush()
					_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
					flusher.Flush()
				} else {
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(map[string]interface{}{
						"choices": []map[string]interface{}{
							{"text": "ok", "finish_reason": "stop"},
						},
						"usage": map[string]interface{}{"completion_tokens": 5, "prompt_tokens": 5},
					})
				}
			}))
			defer server.Close()

			client := NewRealClient(server.URL, "", "test-model", "vllm")
			// minTokens=128 but server returns only 5 tokens with stop: silent non-support.
			_, err := client.Send(context.Background(), &PendingRequest{
				RequestID: 1, InputTokens: 5, Streaming: tc.streaming,
				Prompt: "hello", MaxOutputTokens: 256, MinTokens: 128,
			})
			if err != nil {
				t.Fatal(err)
			}
			if !hook.hasEntry("server may not support min_tokens") {
				t.Error("expected min_tokens non-support warning, but no matching entry found")
			}
		})
	}
}

// TestSend_Unconstrained_MinTokensStop_Warns verifies that the stop+outputTokens<minTokens
// warning fires even in unconstrained mode (both completions and chat formats).
func TestSend_Unconstrained_MinTokensStop_Warns(t *testing.T) {
	tests := []struct {
		name      string
		apiFormat string
	}{
		{"completions", "completions"},
		{"chat", "chat"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hook := installLogHook(t)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				var choices interface{}
				if tc.apiFormat == "chat" {
					choices = []map[string]interface{}{
						{"message": map[string]string{"content": "ok"}, "finish_reason": "stop"},
					}
				} else {
					choices = []map[string]interface{}{
						{"text": "ok", "finish_reason": "stop"},
					}
				}
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"choices": choices,
					"usage":   map[string]interface{}{"completion_tokens": 5, "prompt_tokens": 5},
				})
			}))
			defer server.Close()

			opts := []RealClientOption{}
			if tc.apiFormat == "chat" {
				opts = append(opts, WithAPIFormat("chat"))
			}
			client := NewRealClient(server.URL, "", "test-model", "vllm", opts...)
			// Unconstrained + minTokens=128, server returns only 5 tokens: silent non-support.
			_, err := client.Send(context.Background(), &PendingRequest{
				RequestID: 1, InputTokens: 5, Streaming: false,
				Prompt: "hello", Unconstrained: true, MinTokens: 128,
			})
			if err != nil {
				t.Fatal(err)
			}
			if !hook.hasEntry("server may not support min_tokens") {
				t.Error("expected min_tokens non-support warning, but no matching entry found")
			}
		})
	}
}

// TestIsTimeoutError verifies the isTimeoutError helper covers both detection
// branches: os.IsTimeout (net.Error Timeout() interface) and errors.Is for
// context.DeadlineExceeded. Also verifies that non-timeout errors (generic,
// context.Canceled, io.EOF) return false.
func TestIsTimeoutError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"generic error", errors.New("something broke"), false},
		{"io.EOF", io.EOF, false},
		{"context.DeadlineExceeded", context.DeadlineExceeded, true},
		{"wrapped DeadlineExceeded", fmt.Errorf("outer: %w", context.DeadlineExceeded), true},
		{"context.Canceled", context.Canceled, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isTimeoutError(tc.err); got != tc.want {
				t.Errorf("isTimeoutError(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

// TestObserveTimeoutFlagDefault verifies BC-4: default timeout is 300 seconds.
func TestObserveTimeoutFlagDefault(t *testing.T) {
	if observeTimeout != defaultHTTPTimeoutSeconds {
		t.Errorf("default observeTimeout = %d, want %d", observeTimeout, defaultHTTPTimeoutSeconds)
	}
}

// TestRealClient_WithHTTPTimeout_CustomValue verifies BC-3: WithHTTPTimeout
// sets a custom timeout on the HTTP client.
func TestRealClient_WithHTTPTimeout_CustomValue(t *testing.T) {
	client := NewRealClient("http://localhost", "", "m", "vllm",
		WithHTTPTimeout(42*time.Second))
	if client.httpClient.Timeout != 42*time.Second {
		t.Errorf("Timeout = %v, want 42s", client.httpClient.Timeout)
	}
}

// TestRealClient_NonStreaming_Timeout_SetsTimeoutStatus verifies BC-2: when the
// HTTP client timeout fires during response body read, record.Status is "timeout".
func TestRealClient_NonStreaming_Timeout_SetsTimeoutStatus(t *testing.T) {
	// Server sends partial body then hangs.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		_, _ = w.Write([]byte(`{"choices":[{"text":"hel`))
		flusher.Flush()
		time.Sleep(2 * time.Second)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm",
		WithHTTPTimeout(200*time.Millisecond))
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "timeout" {
		t.Errorf("Status = %q, want %q", record.Status, "timeout")
	}
	if record.ErrorMessage == "" {
		t.Error("ErrorMessage should contain timeout error details")
	}
}

// TestRealClient_HTTPLevel_Timeout_SetsTimeoutStatus verifies BC-8: when the
// HTTP round-trip itself times out (before response headers arrive),
// record.Status is "timeout" not generic "error".
func TestRealClient_HTTPLevel_Timeout_SetsTimeoutStatus(t *testing.T) {
	// Server accepts connection but never responds.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm",
		WithHTTPTimeout(200*time.Millisecond))
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "timeout" {
		t.Errorf("Status = %q, want %q", record.Status, "timeout")
	}
	if record.ErrorMessage == "" {
		t.Error("ErrorMessage should contain timeout error details")
	}
}

// TestRealClient_Streaming_Timeout_SetsTimeoutStatus verifies BC-1: when the
// HTTP client timeout fires during SSE streaming, record.Status is "timeout"
// and record.ErrorMessage contains error details, not silent "ok".
// BC-6: partial timestamps from chunks received before timeout are preserved.
func TestRealClient_Streaming_Timeout_SetsTimeoutStatus(t *testing.T) {
	// Server sends one SSE chunk then hangs until client timeout.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// Send one chunk so FirstChunkTimeUs is set (BC-6)
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok\"}}]}\n\n")
		flusher.Flush()
		// Hang until client gives up (sleep > timeout so client times out first)
		time.Sleep(2 * time.Second)
	}))
	defer server.Close()

	// Use a short timeout; server sleeps 2s so timeout fires well before server finishes
	client := NewRealClient(server.URL, "", "test-model", "vllm",
		WithHTTPTimeout(200*time.Millisecond))
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: true,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "timeout" {
		t.Errorf("Status = %q, want %q", record.Status, "timeout")
	}
	if record.ErrorMessage == "" {
		t.Error("ErrorMessage should contain timeout error details")
	}
	// BC-6: partial timestamps preserved
	if record.FirstChunkTimeUs == 0 {
		t.Error("FirstChunkTimeUs should be set from the chunk received before timeout")
	}
	if record.NumChunks != 1 {
		t.Errorf("NumChunks = %d, want 1 (one chunk before timeout)", record.NumChunks)
	}
}

// TestRealClient_ServerError_BodyReadFailure verifies BC-1: when a non-200
// response body read fails (e.g., connection reset), a warning is logged
// and the error message notes the incomplete body.
func TestRealClient_ServerError_BodyReadFailure(t *testing.T) {
	hook := installLogHook(t)

	// Server sends a non-200 status code then abruptly closes the connection
	// mid-body, causing io.ReadAll to return an error.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hijacker, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("expected http.Hijacker")
		}
		conn, buf, err := hijacker.Hijack()
		if err != nil {
			t.Fatal(err)
		}
		// Write a partial HTTP response with Content-Length mismatch to force read error.
		_, _ = buf.WriteString("HTTP/1.1 500 Internal Server Error\r\n")
		_, _ = buf.WriteString("Content-Length: 1000\r\n")
		_, _ = buf.WriteString("\r\n")
		_, _ = buf.WriteString("partial error bo")
		_ = buf.Flush()
		_ = conn.Close()
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 3, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "error" {
		t.Errorf("status = %q, want error", record.Status)
	}
	if !strings.Contains(record.ErrorMessage, "HTTP 500") {
		t.Errorf("ErrorMessage should contain HTTP status code, got %q", record.ErrorMessage)
	}
	if !strings.Contains(record.ErrorMessage, "body read failed") {
		t.Errorf("ErrorMessage should note body read failure, got %q", record.ErrorMessage)
	}
	if !hook.hasEntry("failed to read error response body") {
		t.Error("expected warning about failed body read, but no matching log entry found")
	}
}

// TestRealClient_ServerError_BodyReadTimeout verifies BC-1: when a non-200
// response body read times out, record.Status is "timeout" (not "error"),
// consistent with the success-path handlers in handleNonStreamingResponse
// and handleStreamingResponse.
func TestRealClient_ServerError_BodyReadTimeout(t *testing.T) {
	hook := installLogHook(t)

	// Server returns a non-200 status then hangs, causing the HTTP client
	// timeout to fire during body read.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		_, _ = fmt.Fprint(w, "partial err")
		flusher.Flush()
		time.Sleep(2 * time.Second)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm",
		WithHTTPTimeout(200*time.Millisecond))
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 5, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "timeout" {
		t.Errorf("status = %q, want timeout", record.Status)
	}
	if !strings.Contains(record.ErrorMessage, "HTTP 502") {
		t.Errorf("ErrorMessage should contain HTTP status code, got %q", record.ErrorMessage)
	}
	if !strings.Contains(record.ErrorMessage, "body read timed out") {
		t.Errorf("ErrorMessage should note body read timeout, got %q", record.ErrorMessage)
	}
	if !hook.hasEntry("failed to read error response body") {
		t.Error("expected warning about failed body read, but no matching log entry found")
	}
}

// TestRealClient_ServerError_BodyReadSuccess verifies BC-2: when a non-200
// response body reads successfully, the full body is included in ErrorMessage
// and no body-read-failure warning is logged.
func TestRealClient_ServerError_BodyReadSuccess(t *testing.T) {
	hook := installLogHook(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = fmt.Fprint(w, "upstream timeout")
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 4, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "error" {
		t.Errorf("status = %q, want error", record.Status)
	}
	if !strings.Contains(record.ErrorMessage, "HTTP 502") {
		t.Errorf("ErrorMessage should contain HTTP 502, got %q", record.ErrorMessage)
	}
	if !strings.Contains(record.ErrorMessage, "upstream timeout") {
		t.Errorf("ErrorMessage should contain body text, got %q", record.ErrorMessage)
	}
	if hook.hasEntry("failed to read error response body") {
		t.Error("no body-read-failure warning expected when body reads successfully")
	}
}

func TestRealClient_Send_InjectsPriorityWhenSLOClassSet(t *testing.T) {
	tests := []struct {
		name            string
		sloClass        string
		expectHeader    bool
		expectHeaderVal string
		expectBodyField bool
		expectBodyVal   int
	}{
		{
			name:            "critical class",
			sloClass:        "critical",
			expectHeader:    true,
			expectHeaderVal: "critical",
			expectBodyField: true,
			expectBodyVal:   0, // 4 - 4 = 0
		},
		{
			name:            "standard class",
			sloClass:        "standard",
			expectHeader:    true,
			expectHeaderVal: "standard",
			expectBodyField: true,
			expectBodyVal:   1, // 4 - 3 = 1
		},
		{
			name:            "batch class",
			sloClass:        "batch",
			expectHeader:    true,
			expectHeaderVal: "batch",
			expectBodyField: true,
			expectBodyVal:   5, // 4 - (-1) = 5
		},
		{
			name:            "sheddable class",
			sloClass:        "sheddable",
			expectHeader:    true,
			expectHeaderVal: "sheddable",
			expectBodyField: true,
			expectBodyVal:   6, // 4 - (-2) = 6
		},
		{
			name:            "background class",
			sloClass:        "background",
			expectHeader:    true,
			expectHeaderVal: "background",
			expectBodyField: true,
			expectBodyVal:   7, // 4 - (-3) = 7
		},
		{
			name:            "empty slo_class",
			sloClass:        "",
			expectHeader:    false,
			expectHeaderVal: "",
			expectBodyField: false,
			expectBodyVal:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Mock server that captures the request
			var capturedHeader string
			var capturedBody map[string]interface{}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				capturedHeader = r.Header.Get("x-gateway-inference-objective")
				decoder := json.NewDecoder(r.Body)
				_ = decoder.Decode(&capturedBody)
				// Return minimal valid response
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id":"test","choices":[{"text":"output"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`))
			}))
			defer server.Close()

			client := NewRealClient(server.URL, "", "test-model", "vllm")

			req := &PendingRequest{
				RequestID:       1,
				InputTokens:     10,
				MaxOutputTokens: 20,
				Model:           "test-model",
				Streaming:       false,
				SLOClass:        tt.sloClass,
				Prompt:          "test prompt",
			}

			ctx := context.Background()
			_, err := client.Send(ctx, req)
			if err != nil {
				t.Fatalf("Send() error = %v", err)
			}

			// Verify header
			if tt.expectHeader {
				if capturedHeader != tt.expectHeaderVal {
					t.Errorf("header x-gateway-inference-objective = %q, want %q", capturedHeader, tt.expectHeaderVal)
				}
			} else {
				if capturedHeader != "" {
					t.Errorf("header x-gateway-inference-objective should be empty, got %q", capturedHeader)
				}
			}

			// Verify body priority field
			if tt.expectBodyField {
				priority, ok := capturedBody["priority"]
				if !ok {
					t.Errorf("body missing 'priority' field")
				} else {
					// JSON numbers decode as float64
					priorityInt := int(priority.(float64))
					if priorityInt != tt.expectBodyVal {
						t.Errorf("body['priority'] = %d, want %d", priorityInt, tt.expectBodyVal)
					}
				}
			} else {
				if _, ok := capturedBody["priority"]; ok {
					t.Errorf("body should not contain 'priority' field when SLOClass is empty")
				}
			}

			// Verify other body fields are not disturbed (BC-7)
			if capturedBody["model"] != "test-model" {
				t.Errorf("body['model'] was disturbed: got %v", capturedBody["model"])
			}
			if capturedBody["stream"] != false {
				t.Errorf("body['stream'] was disturbed: got %v", capturedBody["stream"])
			}
		})
	}
}

func TestRecorder_RecordITL_StreamingRequest(t *testing.T) {
	// GIVEN a recorder and chunk timestamps
	rec := &Recorder{}
	timestamps := []int64{1000000, 1008000, 1016000}

	// WHEN RecordITL is called
	rec.RecordITL(42, timestamps)

	// THEN ITL records are stored
	itl := rec.ITLRecords()
	if len(itl) != 3 {
		t.Fatalf("got %d ITL records, want 3", len(itl))
	}
	for i, ts := range timestamps {
		if itl[i].RequestID != 42 {
			t.Errorf("record %d: got request_id=%d, want 42", i, itl[i].RequestID)
		}
		if itl[i].ChunkIndex != i {
			t.Errorf("record %d: got chunk_index=%d, want %d", i, itl[i].ChunkIndex, i)
		}
		if itl[i].TimestampUs != ts {
			t.Errorf("record %d: got timestamp_us=%d, want %d", i, itl[i].TimestampUs, ts)
		}
	}
}

func TestRealClient_Send_CustomSLOPriorities(t *testing.T) {
	// Mock server that captures the request body
	var capturedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decoder := json.NewDecoder(r.Body)
		_ = decoder.Decode(&capturedBody)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"test","choices":[{"text":"output"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`))
	}))
	defer server.Close()

	// Custom SLO priorities: critical=10 (ultra-high), batch=0 (non-sheddable)
	customMap := sim.NewSLOPriorityMap(map[string]int{
		"critical": 10,
		"batch":    0,
	})

	client := NewRealClient(server.URL, "", "test-model", "vllm",
		WithSLOPriorityMap(customMap))

	tests := []struct {
		name          string
		sloClass      string
		expectedPrio  int
	}{
		{"critical with custom override", "critical", 0},  // 10 - 10 = 0
		{"standard with default", "standard", 7},          // 10 - 3 = 7 (max is now 10)
		{"batch with custom override", "batch", 10},       // 10 - 0 = 10
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &PendingRequest{
				RequestID:       1,
				InputTokens:     10,
				MaxOutputTokens: 20,
				Model:           "test-model",
				Streaming:       false,
				SLOClass:        tt.sloClass,
				Prompt:          "test prompt",
			}

			ctx := context.Background()
			_, err := client.Send(ctx, req)
			if err != nil {
				t.Fatalf("Send() error = %v", err)
			}

			priority, ok := capturedBody["priority"]
			if !ok {
				t.Errorf("body missing 'priority' field")
			} else {
				priorityInt := int(priority.(float64))
				if priorityInt != tt.expectedPrio {
					t.Errorf("body['priority'] = %d, want %d (custom slo_priorities not applied)", priorityInt, tt.expectedPrio)
				}
			}
		})
	}
}

func TestRealClient_WithSLOPriorityMap_NilUsesDefaults(t *testing.T) {
	// Mock server that captures the request body
	var capturedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decoder := json.NewDecoder(r.Body)
		_ = decoder.Decode(&capturedBody)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"test","choices":[{"text":"output"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`))
	}))
	defer server.Close()

	// Pass nil to WithSLOPriorityMap — should fallback to defaults
	client := NewRealClient(server.URL, "", "test-model", "vllm",
		WithSLOPriorityMap(nil))

	req := &PendingRequest{
		RequestID:       1,
		InputTokens:     10,
		MaxOutputTokens: 20,
		Model:           "test-model",
		Streaming:       false,
		SLOClass:        "critical",
		Prompt:          "test prompt",
	}

	ctx := context.Background()
	_, err := client.Send(ctx, req)
	if err != nil {
		t.Fatalf("Send() error = %v", err)
	}

	// Verify default priority is used: critical=4, max=4, inverted=0
	priority, ok := capturedBody["priority"]
	if !ok {
		t.Errorf("body missing 'priority' field")
	} else {
		priorityInt := int(priority.(float64))
		if priorityInt != 0 {
			t.Errorf("body['priority'] = %d, want 0 (default critical priority)", priorityInt)
		}
	}
}

func TestRealClient_Send_VLLMPriority_Captured(t *testing.T) {
	// GIVEN a PendingRequest with SLOClass="critical"
	// WHEN Send() is called
	// THEN RequestRecord.VLLMPriority should capture the inverted priority value
	
	// This test verifies BC-2: RequestRecord.VLLMPriority captures the computed vLLM
	// priority value when req.SLOClass is set.
	
	// Mock HTTP server that always returns OK with minimal response
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		response := map[string]interface{}{
			"id":      "test-id",
			"object":  "text_completion",
			"created": 1234567890,
			"model":   "test-model",
			"choices": []map[string]interface{}{
				{
					"text":          "test output",
					"index":         0,
					"finish_reason": "stop",
				},
			},
			"usage": map[string]interface{}{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		}
		json.NewEncoder(w).Encode(response)
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	// Create RealClient with default SLOPriorityMap
	sloMap := sim.DefaultSLOPriorityMap()
	client := &RealClient{
		baseURL:    server.URL,
		modelName:  "test-model",
		httpClient: &http.Client{Timeout: 5 * time.Second},
		apiFormat:  "completions",
		sloMap:     sloMap,
	}

	req := &PendingRequest{
		RequestID:       1,
		Prompt:          "test prompt",
		MaxOutputTokens: 100,
		Streaming:       false,
		SLOClass:        "critical",
	}

	ctx := context.Background()
	record, err := client.Send(ctx, req)
	if err != nil {
		t.Fatalf("Send() error: %v", err)
	}

	// THEN VLLMPriority should be 0 (critical → highest priority in vLLM convention)
	expectedPriority := sloMap.InvertForVLLM("critical")
	if record.VLLMPriority != expectedPriority {
		t.Errorf("VLLMPriority: got %d, want %d (critical)", record.VLLMPriority, expectedPriority)
	}

	// WHEN SLOClass is "batch"
	req2 := &PendingRequest{
		RequestID:       2,
		Prompt:          "test prompt",
		MaxOutputTokens: 100,
		Streaming:       false,
		SLOClass:        "batch",
	}

	record2, err := client.Send(ctx, req2)
	if err != nil {
		t.Fatalf("Send() error: %v", err)
	}

	// THEN VLLMPriority should be 5 (batch → low priority in vLLM convention)
	expectedPriority2 := sloMap.InvertForVLLM("batch")
	if record2.VLLMPriority != expectedPriority2 {
		t.Errorf("VLLMPriority: got %d, want %d (batch)", record2.VLLMPriority, expectedPriority2)
	}

	// WHEN SLOClass is empty
	req3 := &PendingRequest{
		RequestID:       3,
		Prompt:          "test prompt",
		MaxOutputTokens: 100,
		Streaming:       false,
		SLOClass:        "",
	}

	record3, err := client.Send(ctx, req3)
	if err != nil {
		t.Fatalf("Send() error: %v", err)
	}

	// THEN VLLMPriority should be 0 (not set)
	if record3.VLLMPriority != 0 {
		t.Errorf("VLLMPriority: got %d, want 0 (empty SLOClass)", record3.VLLMPriority)
	}
}

func TestObserveRecorder_VLLMPriority_EndToEndFlow(t *testing.T) {
	// BC-7: End-to-end test verifying vllm_priority flows from RealClient.Send()
	// through Recorder.RecordRequest() to TraceRecord, then through ExportTraceV2 to CSV,
	// and finally back through LoadTraceV2.
	
	// Setup: mock server
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		response := map[string]interface{}{
			"id":      "test-id",
			"object":  "text_completion",
			"created": 1234567890,
			"model":   "test-model",
			"choices": []map[string]interface{}{
				{
					"text":          "test output",
					"index":         0,
					"finish_reason": "stop",
				},
			},
			"usage": map[string]interface{}{
				"prompt_tokens":     100,
				"completion_tokens": 50,
				"total_tokens":      150,
			},
		}
		json.NewEncoder(w).Encode(response)
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	// Create RealClient
	sloMap := sim.DefaultSLOPriorityMap()
	client := &RealClient{
		baseURL:    server.URL,
		modelName:  "test-model",
		httpClient: &http.Client{Timeout: 5 * time.Second},
		apiFormat:  "completions",
		sloMap:     sloMap,
	}

	// Create Recorder
	recorder := &Recorder{}

	// Send request with SLOClass
	pending := &PendingRequest{
		RequestID:       1,
		ClientID:        "c1",
		TenantID:        "t1",
		SLOClass:        "critical",
		Prompt:          "test prompt",
		MaxOutputTokens: 100,
		Streaming:       false,
	}

	ctx := context.Background()
	record, err := client.Send(ctx, pending)
	if err != nil {
		t.Fatalf("Send() error: %v", err)
	}

	// Verify RealClient captured VLLMPriority (BC-2)
	expectedPriority := sloMap.InvertForVLLM("critical")
	if record.VLLMPriority != expectedPriority {
		t.Errorf("RealClient VLLMPriority: got %d, want %d", record.VLLMPriority, expectedPriority)
	}

	// Record the result
	recorder.RecordRequest(pending, record, time.Now().UnixMicro(), "", 0)

	// Get trace records
	traceRecords := recorder.Records()
	if len(traceRecords) != 1 {
		t.Fatalf("len(traceRecords)=%d, want 1", len(traceRecords))
	}

	// Verify VLLMPriority was copied to TraceRecord
	if traceRecords[0].VLLMPriority != expectedPriority {
		t.Errorf("TraceRecord VLLMPriority: got %d, want %d", traceRecords[0].VLLMPriority, expectedPriority)
	}

	// Export to CSV
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")
	header := &workload.TraceHeader{
		Version:  2,
		TimeUnit: "microseconds",
		Mode:     "real",
	}
	if err := recorder.Export(header, headerPath, dataPath); err != nil {
		t.Fatalf("Export error: %v", err)
	}

	// Load back from CSV
	loaded, err := workload.LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2 error: %v", err)
	}
	if len(loaded.Records) != 1 {
		t.Fatalf("len(loaded.Records)=%d, want 1", len(loaded.Records))
	}

	// Verify VLLMPriority survived round-trip (BC-4)
	if loaded.Records[0].VLLMPriority != expectedPriority {
		t.Errorf("Loaded VLLMPriority: got %d, want %d", loaded.Records[0].VLLMPriority, expectedPriority)
	}

	// Verify simulation isolation: LoadTraceV2Requests must NOT read VLLMPriority (BC-5)
	requests, err := workload.LoadTraceV2Requests(loaded, 42)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests error: %v", err)
	}
	if len(requests) != 1 {
		t.Fatalf("len(requests)=%d, want 1", len(requests))
	}
	if requests[0].Priority != 0 {
		t.Errorf("Request Priority=%f, want 0 (simulation isolation)", requests[0].Priority)
	}
}
