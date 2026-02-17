package workload

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// TraceHeader captures metadata for trace v2 files.
type TraceHeader struct {
	Version        int    `yaml:"trace_version"`
	TimeUnit       string `yaml:"time_unit"`
	CreatedAt      string `yaml:"created_at,omitempty"`
	Mode           string `yaml:"mode"` // "real" or "generated"
	WarmUpRequests int    `yaml:"warm_up_requests"`
	WorkloadSpec   string `yaml:"workload_spec,omitempty"`

	Server  *TraceServerConfig  `yaml:"server,omitempty"`
	Network *TraceNetworkConfig `yaml:"network,omitempty"`
}

// TraceServerConfig captures server configuration in trace header.
type TraceServerConfig struct {
	Type                  string  `yaml:"type,omitempty"`
	Model                 string  `yaml:"model,omitempty"`
	TensorParallel        int     `yaml:"tensor_parallel,omitempty"`
	MaxNumSeqs            int     `yaml:"max_num_seqs,omitempty"`
	BlockSize             int     `yaml:"block_size,omitempty"`
	GPUMemoryUtilization  float64 `yaml:"gpu_memory_utilization,omitempty"`
	MaxModelLen           int     `yaml:"max_model_len,omitempty"`
}

// TraceNetworkConfig captures network configuration in trace header.
type TraceNetworkConfig struct {
	MeasuredRTTMs float64 `yaml:"measured_rtt_ms,omitempty"`
}

// TraceRecord represents one row in a trace v2 CSV.
type TraceRecord struct {
	RequestID       int
	ClientID        string
	TenantID        string
	SLOClass        string
	SessionID       string
	RoundIndex      int
	PrefixGroup     string
	Streaming       bool
	InputTokens     int
	OutputTokens    int
	TextTokens      int
	ImageTokens     int
	AudioTokens     int
	VideoTokens     int
	ReasonRatio     float64
	ArrivalTimeUs   int64
	SendTimeUs      int64
	FirstChunkTimeUs int64
	LastChunkTimeUs  int64
	NumChunks       int
	Status          string // "ok", "error", "timeout"
	ErrorMessage    string
}

// TraceV2 combines header and records for a complete trace.
type TraceV2 struct {
	Header  TraceHeader
	Records []TraceRecord
}

// CSV column headers for trace v2 format.
var traceV2Columns = []string{
	"request_id", "client_id", "tenant_id", "slo_class", "session_id", "round_index",
	"prefix_group", "streaming", "input_tokens", "output_tokens",
	"text_tokens", "image_tokens", "audio_tokens", "video_tokens", "reason_ratio",
	"arrival_time_us", "send_time_us", "first_chunk_time_us", "last_chunk_time_us",
	"num_chunks", "status", "error_message",
}

// ExportTraceV2 writes trace header (YAML) and data (CSV) to separate files.
// Timestamps use integer formatting (%d) to preserve microsecond precision.
func ExportTraceV2(header *TraceHeader, records []TraceRecord, headerPath, dataPath string) error {
	// Write header YAML
	headerData, err := yaml.Marshal(header)
	if err != nil {
		return fmt.Errorf("marshaling trace header: %w", err)
	}
	if err := os.WriteFile(headerPath, headerData, 0644); err != nil {
		return fmt.Errorf("writing trace header: %w", err)
	}

	// Write data CSV
	file, err := os.Create(dataPath)
	if err != nil {
		return fmt.Errorf("creating trace data file: %w", err)
	}
	defer func() { _ = file.Close() }()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header row
	if err := writer.Write(traceV2Columns); err != nil {
		return fmt.Errorf("writing CSV header: %w", err)
	}

	// Write data rows (integer formatting for timestamps)
	for _, r := range records {
		row := []string{
			strconv.Itoa(r.RequestID),
			r.ClientID,
			r.TenantID,
			r.SLOClass,
			r.SessionID,
			strconv.Itoa(r.RoundIndex),
			r.PrefixGroup,
			strconv.FormatBool(r.Streaming),
			strconv.Itoa(r.InputTokens),
			strconv.Itoa(r.OutputTokens),
			strconv.Itoa(r.TextTokens),
			strconv.Itoa(r.ImageTokens),
			strconv.Itoa(r.AudioTokens),
			strconv.Itoa(r.VideoTokens),
			strconv.FormatFloat(r.ReasonRatio, 'f', -1, 64),
			strconv.FormatInt(r.ArrivalTimeUs, 10),   // integer format
			strconv.FormatInt(r.SendTimeUs, 10),       // integer format
			strconv.FormatInt(r.FirstChunkTimeUs, 10), // integer format
			strconv.FormatInt(r.LastChunkTimeUs, 10),  // integer format
			strconv.Itoa(r.NumChunks),
			r.Status,
			r.ErrorMessage,
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("writing CSV row %d: %w", r.RequestID, err)
		}
	}
	return nil
}

// LoadTraceV2 reads a trace v2 header (YAML) and data (CSV).
func LoadTraceV2(headerPath, dataPath string) (*TraceV2, error) {
	// Load header
	headerData, err := os.ReadFile(headerPath)
	if err != nil {
		return nil, fmt.Errorf("reading trace header: %w", err)
	}
	var header TraceHeader
	if err := yaml.Unmarshal(headerData, &header); err != nil {
		return nil, fmt.Errorf("parsing trace header: %w", err)
	}

	// Load data CSV
	file, err := os.Open(dataPath)
	if err != nil {
		return nil, fmt.Errorf("opening trace data: %w", err)
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(file)

	// Skip header row
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("reading CSV header: %w", err)
	}

	var records []TraceRecord
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading CSV row: %w", err)
		}
		if len(row) < len(traceV2Columns) {
			return nil, fmt.Errorf("CSV row has %d columns, expected %d", len(row), len(traceV2Columns))
		}

		r, err := parseTraceRecord(row)
		if err != nil {
			return nil, err
		}
		records = append(records, *r)
	}

	return &TraceV2{Header: header, Records: records}, nil
}

func parseTraceRecord(row []string) (*TraceRecord, error) {
	requestID, _ := strconv.Atoi(row[0])
	roundIndex, _ := strconv.Atoi(row[5])
	streaming, _ := strconv.ParseBool(row[7])
	inputTokens, _ := strconv.Atoi(row[8])
	outputTokens, _ := strconv.Atoi(row[9])
	textTokens, _ := strconv.Atoi(row[10])
	imageTokens, _ := strconv.Atoi(row[11])
	audioTokens, _ := strconv.Atoi(row[12])
	videoTokens, _ := strconv.Atoi(row[13])
	reasonRatio, _ := strconv.ParseFloat(row[14], 64)
	arrivalTimeUs, _ := strconv.ParseInt(row[15], 10, 64)
	sendTimeUs, _ := strconv.ParseInt(row[16], 10, 64)
	firstChunkTimeUs, _ := strconv.ParseInt(row[17], 10, 64)
	lastChunkTimeUs, _ := strconv.ParseInt(row[18], 10, 64)
	numChunks, _ := strconv.Atoi(row[19])

	status := "ok"
	errorMessage := ""
	if len(row) > 20 {
		status = row[20]
	}
	if len(row) > 21 {
		errorMessage = strings.TrimSpace(row[21])
	}

	return &TraceRecord{
		RequestID:        requestID,
		ClientID:         row[1],
		TenantID:         row[2],
		SLOClass:         row[3],
		SessionID:        row[4],
		RoundIndex:       roundIndex,
		PrefixGroup:      row[6],
		Streaming:        streaming,
		InputTokens:      inputTokens,
		OutputTokens:     outputTokens,
		TextTokens:       textTokens,
		ImageTokens:      imageTokens,
		AudioTokens:      audioTokens,
		VideoTokens:      videoTokens,
		ReasonRatio:      reasonRatio,
		ArrivalTimeUs:    arrivalTimeUs,
		SendTimeUs:       sendTimeUs,
		FirstChunkTimeUs: firstChunkTimeUs,
		LastChunkTimeUs:  lastChunkTimeUs,
		NumChunks:        numChunks,
		Status:           status,
		ErrorMessage:     errorMessage,
	}, nil
}
