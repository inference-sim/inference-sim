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
	RequestID   int
	InputTokens int
	Streaming   bool
	ClientID    string
	TenantID    string
	SLOClass    string
}

// RequestRecord captures one request-response cycle.
type RequestRecord struct {
	RequestID        int
	OutputTokens     int
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
	body := map[string]interface{}{
		"model":      c.modelName,
		"max_tokens": 2048,
		"stream":     req.Streaming,
	}
	// Use completion API with a dummy prompt
	body["prompt"] = fmt.Sprintf("Token count: %d", req.InputTokens)

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

func (c *RealClient) handleNonStreamingResponse(resp *http.Response, record *RequestRecord) (*RequestRecord, error) {
	bodyData, err := io.ReadAll(resp.Body)
	if err != nil {
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("read error: %v", err)
		return record, nil
	}
	now := time.Now().UnixMicro()
	record.FirstChunkTimeUs = now
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
			continue
		}
		if usage, ok := chunk["usage"].(map[string]interface{}); ok {
			lastUsage = usage
		}
	}

	record.NumChunks = chunkCount
	if lastUsage != nil {
		if ct, ok := lastUsage["completion_tokens"].(float64); ok {
			record.OutputTokens = int(ct)
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
func (r *Recorder) RecordRequest(pending *PendingRequest, result *RequestRecord) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.records = append(r.records, workload.TraceRecord{
		RequestID:        result.RequestID,
		ClientID:         pending.ClientID,
		TenantID:         pending.TenantID,
		SLOClass:         pending.SLOClass,
		Streaming:        pending.Streaming,
		InputTokens:      pending.InputTokens,
		OutputTokens:     result.OutputTokens,
		ArrivalTimeUs:    0, // set by caller from workload spec
		SendTimeUs:       result.SendTimeUs,
		FirstChunkTimeUs: result.FirstChunkTimeUs,
		LastChunkTimeUs:  result.LastChunkTimeUs,
		NumChunks:        result.NumChunks,
		Status:           result.Status,
		ErrorMessage:     result.ErrorMessage,
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

// LogRealModeNotImplemented logs that real mode is a placeholder for now.
func LogRealModeNotImplemented() {
	logrus.Warn("Real mode (--real-mode) is not yet fully integrated. Use mock testing for validation.")
}
