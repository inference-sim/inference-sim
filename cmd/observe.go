package cmd

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
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
	httpClient *http.Client
}

// NewRealClient creates a new real mode HTTP client.
func NewRealClient(baseURL, apiKey, modelName, serverType string) *RealClient {
	return &RealClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		modelName:  modelName,
		serverType: serverType,
		httpClient: &http.Client{Timeout: 5 * time.Minute},
	}
}

// PendingRequest represents a request to be sent to the server.
type PendingRequest struct {
	RequestID      int
	InputTokens    int
	MaxOutputTokens int
	Model          string
	Streaming      bool
	ClientID       string
	TenantID       string
	SLOClass       string
}

// RequestRecord captures one request-response cycle.
type RequestRecord struct {
	RequestID        int
	OutputTokens     int
	ServerInputTokens int
	Status           string // "ok", "error", "timeout"
	ErrorMessage     string
	SendTimeUs       int64
	FirstChunkTimeUs int64
	LastChunkTimeUs  int64
	NumChunks        int
}

// Send dispatches a single request to the server and records timing.
func (c *RealClient) Send(ctx context.Context, req *PendingRequest) (*RequestRecord, error) {
	record := &RequestRecord{
		RequestID: req.RequestID,
		Status:    "ok",
	}

	// Build request body
	maxTokens := req.MaxOutputTokens
	if maxTokens < 0 {
		logrus.Warnf("PendingRequest.MaxOutputTokens is negative (%d), using default 2048", maxTokens)
	}
	if maxTokens <= 0 {
		maxTokens = 2048
	}
	body := map[string]interface{}{
		"model":      c.modelName,
		"max_tokens": maxTokens,
		"stream":     req.Streaming,
	}
	// Generate proportional prompt: ~N tokens for N InputTokens.
	// Actual token count varies by tokenizer; ServerInputTokens (BC-3) provides ground truth.
	inputTokens := req.InputTokens
	if inputTokens < 0 {
		logrus.Warnf("PendingRequest.InputTokens is negative (%d), using 1 for prompt generation", inputTokens)
	}
	if inputTokens <= 0 {
		inputTokens = 1
	}
	// Note: for very large InputTokens (e.g., 128K), this creates a ~768KB string.
	// Acceptable for observe mode's typical use; server-side tokenization is the bottleneck.
	body["prompt"] = strings.Repeat("hello ", inputTokens)

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("marshal error: %v", err)
		return record, nil
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/completions", strings.NewReader(string(bodyBytes)))
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
	record.FirstChunkTimeUs = fbr.firstReadTime
	record.LastChunkTimeUs = time.Now().UnixMicro()
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

		// Parse chunk for usage (only in final chunk for vLLM)
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			logrus.Debugf("observe: skipping malformed SSE chunk: %v", err)
			continue
		}
		if usage, ok := chunk["usage"].(map[string]interface{}); ok {
			lastUsage = usage
		}
	}

	// TODO: Per-chunk ITL timestamps not yet recorded (#655 Bug 5, deferred).
	// Only first/last chunk times are captured. Full ITL distribution requires
	// storing each chunk timestamp, which needs new schema support.
	record.NumChunks = chunkCount
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
		// TODO: populate DeadlineUs once PendingRequest carries deadline info (out of #655 scope)
		Model:             pending.Model,
		ServerInputTokens: result.ServerInputTokens,
		RequestID:         result.RequestID,
		ClientID:          pending.ClientID,
		TenantID:          pending.TenantID,
		SLOClass:          pending.SLOClass,
		Streaming:         pending.Streaming,
		InputTokens:       pending.InputTokens,
		OutputTokens:      result.OutputTokens,
		ArrivalTimeUs:     arrivalTimeUs,
		SendTimeUs:        result.SendTimeUs,
		FirstChunkTimeUs:  result.FirstChunkTimeUs,
		LastChunkTimeUs:   result.LastChunkTimeUs,
		NumChunks:         result.NumChunks,
		Status:            result.Status,
		ErrorMessage:      result.ErrorMessage,
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

