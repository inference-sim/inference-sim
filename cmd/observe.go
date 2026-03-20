package cmd

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
)

// RealClient sends requests to an OpenAI-compatible inference server.
type RealClient struct {
	baseURL    string
	apiKey     string
	modelName  string
	serverType string
	apiFormat  string // "completions" or "chat" (default: "completions")
	httpClient *http.Client
}

// RealClientOption configures optional RealClient behavior.
type RealClientOption func(*RealClient)

// WithAPIFormat sets the API format ("completions" or "chat").
func WithAPIFormat(format string) RealClientOption {
	return func(c *RealClient) { c.apiFormat = format }
}

// NewRealClient creates a new real mode HTTP client.
func NewRealClient(baseURL, apiKey, modelName, serverType string, opts ...RealClientOption) *RealClient {
	c := &RealClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		modelName:  modelName,
		serverType: serverType,
		apiFormat:  "completions",
		httpClient: &http.Client{Timeout: 5 * time.Minute},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// PendingRequest represents a request to be sent to the server.
type PendingRequest struct {
	RequestID       int
	InputTokens     int
	MaxOutputTokens int
	Model           string
	Streaming       bool
	ClientID        string
	TenantID        string
	SLOClass        string
	Prompt          string
	Unconstrained   bool
	DeadlineUs      int64
}

// RequestRecord captures one request-response cycle.
type RequestRecord struct {
	RequestID         int
	OutputTokens      int
	ServerInputTokens int
	Status            string // "ok", "error", "timeout"
	ErrorMessage      string
	SendTimeUs        int64
	FirstChunkTimeUs  int64
	LastChunkTimeUs   int64
	NumChunks         int
	FinishReason      string
}

// Send dispatches a single request to the server and records timing.
func (c *RealClient) Send(ctx context.Context, req *PendingRequest) (*RequestRecord, error) {
	record := &RequestRecord{
		RequestID: req.RequestID,
		Status:    "ok",
	}

	// Build request body
	body := map[string]interface{}{
		"model":  c.modelName,
		"stream": req.Streaming,
	}

	// Configurable max_tokens: unconstrained requests omit (chat) or set MaxInt32 (completions)
	if !req.Unconstrained {
		maxTokens := req.MaxOutputTokens
		if maxTokens < 0 {
			logrus.Warnf("PendingRequest.MaxOutputTokens is negative (%d), using default 2048", maxTokens)
		}
		if maxTokens <= 0 {
			maxTokens = 2048
		}
		body["max_tokens"] = maxTokens
	} else if c.apiFormat == "completions" {
		// completions API requires max_tokens; use MaxInt32 to not constrain output
		body["max_tokens"] = math.MaxInt32
	}
	// chat + unconstrained: omit max_tokens entirely (server uses model default)
	// Set prompt/messages and endpoint based on API format.
	var endpoint string
	switch c.apiFormat {
	case "chat":
		endpoint = c.baseURL + "/v1/chat/completions"
		body["messages"] = []map[string]string{{"role": "user", "content": req.Prompt}}
	default: // "completions"
		endpoint = c.baseURL + "/v1/completions"
		body["prompt"] = req.Prompt
	}

	// Request usage data in streaming responses (required for token count extraction).
	if req.Streaming {
		body["stream_options"] = map[string]interface{}{"include_usage": true}
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("marshal error: %v", err)
		return record, nil
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, strings.NewReader(string(bodyBytes)))
	if err != nil {
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("request creation error: %v", err)
		return record, nil
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	// Record send time
	record.SendTimeUs = time.Now().UnixMicro()

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("HTTP error: %v", err)
		return record, nil
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		bodyData, _ := io.ReadAll(resp.Body)
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyData))
		return record, nil
	}

	if req.Streaming {
		return c.handleStreamingResponse(resp, record)
	}
	return c.handleNonStreamingResponse(resp, record)
}

// firstByteReader wraps an io.Reader and captures the timestamp when the first byte is received.
type firstByteReader struct {
	r             io.Reader
	firstReadTime int64 // UnixMicro of first successful Read (n > 0); 0 = no data yet
}

func (f *firstByteReader) Read(p []byte) (int, error) {
	n, err := f.r.Read(p)
	if f.firstReadTime == 0 && n > 0 {
		f.firstReadTime = time.Now().UnixMicro()
	}
	return n, err
}

func (c *RealClient) handleNonStreamingResponse(resp *http.Response, record *RequestRecord) (*RequestRecord, error) {
	// Wrap body to capture first-byte timing (BC-2).
	// Note: for non-streaming HTTP, real servers send the entire response after generation
	// completes, so FirstChunkTimeUs approximates "server finished + transfer started,"
	// not "first token generated." True TTFT is only measurable in streaming mode.
	fbr := &firstByteReader{r: resp.Body}
	bodyData, err := io.ReadAll(fbr)
	if err != nil {
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("read error: %v", err)
		return record, nil
	}
	now := time.Now().UnixMicro()
	if fbr.firstReadTime != 0 {
		record.FirstChunkTimeUs = fbr.firstReadTime
	} else {
		// Empty body (e.g. error response) — use current time as fallback
		record.FirstChunkTimeUs = now
	}
	record.LastChunkTimeUs = now
	record.NumChunks = 1

	var result map[string]interface{}
	if err := json.Unmarshal(bodyData, &result); err != nil {
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("JSON parse error: %v", err)
		return record, nil
	}

	// Extract token counts from usage
	if usage, ok := result["usage"].(map[string]interface{}); ok {
		if ct, ok := usage["completion_tokens"].(float64); ok {
			record.OutputTokens = int(ct)
		}
		if pt, ok := usage["prompt_tokens"].(float64); ok {
			record.ServerInputTokens = int(pt)
		} else if _, exists := usage["prompt_tokens"]; exists {
			logrus.Debugf("observe: prompt_tokens has unexpected type %T, expected float64", usage["prompt_tokens"])
		}
	}

	// Extract finish_reason from choices[0]
	if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if fr, ok := choice["finish_reason"].(string); ok {
				record.FinishReason = fr
				if fr == "length" || fr == "abort" {
					logrus.Warnf("observe: request %d finish_reason=%q (output may be truncated)", record.RequestID, fr)
				}
			}
		}
	}

	return record, nil
}

func (c *RealClient) handleStreamingResponse(resp *http.Response, record *RequestRecord) (*RequestRecord, error) {
	scanner := bufio.NewScanner(resp.Body)
	chunkCount := 0
	var lastUsage map[string]interface{}

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		now := time.Now().UnixMicro()
		chunkCount++
		if chunkCount == 1 {
			record.FirstChunkTimeUs = now
		}
		record.LastChunkTimeUs = now

		// Parse chunk for usage and finish_reason
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			logrus.Debugf("observe: skipping malformed SSE chunk: %v", err)
			continue
		}
		if usage, ok := chunk["usage"].(map[string]interface{}); ok {
			lastUsage = usage
		}
		// Extract finish_reason from content chunks (skip usage-only chunks with empty choices)
		if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
					record.FinishReason = fr
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		logrus.Warnf("observe: request %d: SSE scanner error: %v", record.RequestID, err)
	}

	// TODO: Per-chunk ITL timestamps not yet recorded (#655 Bug 5, deferred).
	// Only first/last chunk times are captured. Full ITL distribution requires
	// storing each chunk timestamp, which needs new schema support.
	record.NumChunks = chunkCount
	if lastUsage == nil && chunkCount > 0 {
		logrus.Warnf("observe: request %d: streaming response had %d chunks but no usage data (missing stream_options?)", record.RequestID, chunkCount)
	}
	if lastUsage != nil {
		if ct, ok := lastUsage["completion_tokens"].(float64); ok {
			record.OutputTokens = int(ct)
		}
		if pt, ok := lastUsage["prompt_tokens"].(float64); ok {
			record.ServerInputTokens = int(pt)
		} else if _, exists := lastUsage["prompt_tokens"]; exists {
			logrus.Debugf("observe: prompt_tokens has unexpected type %T, expected float64", lastUsage["prompt_tokens"])
		}
	}

	// Warn on problematic finish_reason values
	if record.FinishReason == "length" || record.FinishReason == "abort" {
		logrus.Warnf("observe: request %d finish_reason=%q (output may be truncated)", record.RequestID, record.FinishReason)
	}

	return record, nil
}

// Recorder captures per-request timing and metrics (goroutine-safe).
type Recorder struct {
	mu      sync.Mutex
	records []workload.TraceRecord
}

// RecordRequest captures one request-response cycle.
func (r *Recorder) RecordRequest(pending *PendingRequest, result *RequestRecord, arrivalTimeUs int64, sessionID string, roundIndex int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.records = append(r.records, workload.TraceRecord{
		Model:             pending.Model,
		ServerInputTokens: result.ServerInputTokens,
		RequestID:         result.RequestID,
		ClientID:          pending.ClientID,
		TenantID:          pending.TenantID,
		SLOClass:          pending.SLOClass,
		Streaming:         pending.Streaming,
		InputTokens:       pending.InputTokens,
		OutputTokens:      result.OutputTokens,
		DeadlineUs:        pending.DeadlineUs,
		ArrivalTimeUs:     arrivalTimeUs,
		SendTimeUs:        result.SendTimeUs,
		FirstChunkTimeUs:  result.FirstChunkTimeUs,
		LastChunkTimeUs:   result.LastChunkTimeUs,
		NumChunks:         result.NumChunks,
		Status:            result.Status,
		ErrorMessage:      result.ErrorMessage,
		FinishReason:      result.FinishReason,
		SessionID:         sessionID,
		RoundIndex:        roundIndex,
	})
}

// Records returns all recorded trace records.
func (r *Recorder) Records() []workload.TraceRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]workload.TraceRecord, len(r.records))
	copy(result, r.records)
	return result
}

// Export writes trace v2 files.
func (r *Recorder) Export(header *workload.TraceHeader, headerPath, dataPath string) error {
	return workload.ExportTraceV2(header, r.Records(), headerPath, dataPath)
}

