package workload

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"gopkg.in/yaml.v3"
)

// TraceHeader captures metadata for trace v2 files.
type TraceHeader struct {
	Version        int    `yaml:"trace_version"`
	TimeUnit       string `yaml:"time_unit"`
	CreatedAt      string `yaml:"created_at,omitempty"`
	Mode           string `yaml:"mode"` // "real" (blis observe), "generated" (blis run), or "replayed" (blis replay)
	WarmUpRequests int    `yaml:"warm_up_requests"`
	WorkloadSeed   *int64 `yaml:"workload_seed,omitempty"`
	// WorkloadSpec records workload provenance:
	//   - a file path when --workload-spec is used (e.g. "workload.yaml")
	//   - "preset:<name>" when --workload is used (e.g. "preset:chatbot")
	//   - empty (omitted) when distribution synthesis or concurrency mode is used
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
	MaxModelLen           int64   `yaml:"max_model_len,omitempty"`
}

// TraceNetworkConfig captures network configuration in trace header.
type TraceNetworkConfig struct {
	MeasuredRTTMs float64 `yaml:"measured_rtt_ms,omitempty"`
}

// TraceRecord represents one row in a trace v2 CSV.
type TraceRecord struct {
	RequestID         int
	ClientID          string
	TenantID          string
	SLOClass          string
	GIEPriority       int
	SessionID         string
	RoundIndex        int
	PrefixGroup       string
	PrefixLength      int
	Streaming         bool
	InputTokens       int
	OutputTokens      int
	TextTokens        int
	ImageTokens       int
	AudioTokens       int
	VideoTokens       int
	ReasonRatio       float64
	Model             string // model name (e.g., "meta-llama/Llama-3.1-8B-Instruct"); empty = default model
	DeadlineUs        int64  // absolute deadline timestamp in microseconds (same time origin as ArrivalTimeUs); 0 = no timeout
	ServerInputTokens int    // server-reported prompt_tokens; 0 = not recorded (e.g., generated traces)
	ArrivalTimeUs     int64
	SendTimeUs        int64
	FirstChunkTimeUs  int64
	LastChunkTimeUs   int64
	NumChunks         int
	Status            string // "ok", "error", "timeout"
	ErrorMessage      string
	FinishReason      string // server-reported finish_reason ("stop", "length", "abort", etc.); empty = not recorded
}

// TraceV2 combines header and records for a complete trace.
type TraceV2 struct {
	Header  TraceHeader
	Records []TraceRecord
}

// CSV column headers for trace v2 format.
var traceV2Columns = []string{
	"request_id", "client_id", "tenant_id", "slo_class", "priority",
	"session_id", "round_index",
	"prefix_group", "prefix_length", "streaming", "input_tokens", "output_tokens",
	"text_tokens", "image_tokens", "audio_tokens", "video_tokens", "reason_ratio",
	"model", "deadline_us", "server_input_tokens",
	"arrival_time_us", "send_time_us", "first_chunk_time_us", "last_chunk_time_us",
	"num_chunks", "status", "error_message", "finish_reason",
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
			strconv.Itoa(r.GIEPriority),
			r.SessionID,
			strconv.Itoa(r.RoundIndex),
			r.PrefixGroup,
			strconv.Itoa(r.PrefixLength),
			strconv.FormatBool(r.Streaming),
			strconv.Itoa(r.InputTokens),
			strconv.Itoa(r.OutputTokens),
			strconv.Itoa(r.TextTokens),
			strconv.Itoa(r.ImageTokens),
			strconv.Itoa(r.AudioTokens),
			strconv.Itoa(r.VideoTokens),
			strconv.FormatFloat(r.ReasonRatio, 'f', -1, 64),
			r.Model,
			strconv.FormatInt(r.DeadlineUs, 10),
			strconv.Itoa(r.ServerInputTokens),
			strconv.FormatInt(r.ArrivalTimeUs, 10),   // integer format
			strconv.FormatInt(r.SendTimeUs, 10),       // integer format
			strconv.FormatInt(r.FirstChunkTimeUs, 10), // integer format
			strconv.FormatInt(r.LastChunkTimeUs, 10),  // integer format
			strconv.Itoa(r.NumChunks),
			r.Status,
			r.ErrorMessage,
			r.FinishReason,
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
	decoder := yaml.NewDecoder(bytes.NewReader(headerData))
	decoder.KnownFields(true)
	if err := decoder.Decode(&header); err != nil {
		return nil, fmt.Errorf("parsing trace header: %w", err)
	}

	// Load data CSV
	file, err := os.Open(dataPath)
	if err != nil {
		return nil, fmt.Errorf("opening trace data: %w", err)
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1 // allow extra columns (future extensions)

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
		// Backward compatibility: 27-column files (pre-priority schema) get a
		// default priority="0" inserted at position 4 before parsing.
		const prePriorityColumnCount = 27
		if len(row) == prePriorityColumnCount {
			row = append(row[:4], append([]string{"0"}, row[4:]...)...)
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

// parseTraceRecord parses a 28-column (current schema) CSV row.
func parseTraceRecord(row []string) (*TraceRecord, error) {
	requestID, err := strconv.Atoi(row[0])
	if err != nil {
		return nil, fmt.Errorf("parsing request_id %q: %w", row[0], err)
	}
	// Column 4: priority (GIE integer priority)
	giePriority, err := strconv.Atoi(row[4])
	if err != nil {
		return nil, fmt.Errorf("parsing priority %q: %w", row[4], err)
	}
	roundIndex, err := strconv.Atoi(row[6])
	if err != nil {
		return nil, fmt.Errorf("parsing round_index %q: %w", row[6], err)
	}
	// Column 8: prefix_length
	prefixLength, err := strconv.Atoi(row[8])
	if err != nil {
		return nil, fmt.Errorf("parsing prefix_length %q: %w", row[8], err)
	}
	if prefixLength < 0 {
		return nil, fmt.Errorf("parsing prefix_length: negative value %d not allowed", prefixLength)
	}
	// Column 9: streaming
	streaming, err := strconv.ParseBool(row[9])
	if err != nil {
		return nil, fmt.Errorf("parsing streaming %q: %w", row[9], err)
	}
	// Column 10: input_tokens
	inputTokens, err := strconv.Atoi(row[10])
	if err != nil {
		return nil, fmt.Errorf("parsing input_tokens %q: %w", row[10], err)
	}
	// Negative token counts cause make([]int, negative) panics in LoadTraceV2Requests.
	if inputTokens < 0 {
		return nil, fmt.Errorf("parsing input_tokens: negative value %d not allowed", inputTokens)
	}
	outputTokens, err := strconv.Atoi(row[11])
	if err != nil {
		return nil, fmt.Errorf("parsing output_tokens %q: %w", row[11], err)
	}
	if outputTokens < 0 {
		return nil, fmt.Errorf("parsing output_tokens: negative value %d not allowed", outputTokens)
	}
	textTokens, err := strconv.Atoi(row[12])
	if err != nil {
		return nil, fmt.Errorf("parsing text_tokens %q: %w", row[12], err)
	}
	if textTokens < 0 {
		return nil, fmt.Errorf("parsing text_tokens: negative value %d not allowed", textTokens)
	}
	imageTokens, err := strconv.Atoi(row[13])
	if err != nil {
		return nil, fmt.Errorf("parsing image_tokens %q: %w", row[13], err)
	}
	if imageTokens < 0 {
		return nil, fmt.Errorf("parsing image_tokens: negative value %d not allowed", imageTokens)
	}
	audioTokens, err := strconv.Atoi(row[14])
	if err != nil {
		return nil, fmt.Errorf("parsing audio_tokens %q: %w", row[14], err)
	}
	if audioTokens < 0 {
		return nil, fmt.Errorf("parsing audio_tokens: negative value %d not allowed", audioTokens)
	}
	videoTokens, err := strconv.Atoi(row[15])
	if err != nil {
		return nil, fmt.Errorf("parsing video_tokens %q: %w", row[15], err)
	}
	if videoTokens < 0 {
		return nil, fmt.Errorf("parsing video_tokens: negative value %d not allowed", videoTokens)
	}
	reasonRatio, err := strconv.ParseFloat(row[16], 64)
	if err != nil {
		return nil, fmt.Errorf("parsing reason_ratio %q: %w", row[16], err)
	}
	if math.IsNaN(reasonRatio) || math.IsInf(reasonRatio, 0) || reasonRatio < 0 || reasonRatio > 1.0 {
		return nil, fmt.Errorf("parsing reason_ratio %q: must be in range [0.0, 1.0], got %g", row[16], reasonRatio)
	}
	deadlineUs, err := strconv.ParseInt(row[18], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing deadline_us %q: %w", row[18], err)
	}
	if deadlineUs < 0 {
		return nil, fmt.Errorf("parsing deadline_us: negative value %d not allowed (use 0 for no timeout)", deadlineUs)
	}
	serverInputTokens, err := strconv.Atoi(row[19])
	if err != nil {
		return nil, fmt.Errorf("parsing server_input_tokens %q: %w", row[19], err)
	}
	if serverInputTokens < 0 {
		return nil, fmt.Errorf("parsing server_input_tokens: negative value %d not allowed", serverInputTokens)
	}
	arrivalTimeUs, err := strconv.ParseInt(row[20], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing arrival_time_us %q: %w", row[20], err)
	}
	sendTimeUs, err := strconv.ParseInt(row[21], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing send_time_us %q: %w", row[21], err)
	}
	firstChunkTimeUs, err := strconv.ParseInt(row[22], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing first_chunk_time_us %q: %w", row[22], err)
	}
	lastChunkTimeUs, err := strconv.ParseInt(row[23], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing last_chunk_time_us %q: %w", row[23], err)
	}
	numChunks, err := strconv.Atoi(row[24])
	if err != nil {
		return nil, fmt.Errorf("parsing num_chunks %q: %w", row[24], err)
	}
	if numChunks < 0 {
		return nil, fmt.Errorf("parsing num_chunks: negative value %d not allowed", numChunks)
	}
	// Cross-field invariant: deadline must not precede arrival (would cause immediate
	// timeout at enqueue before any processing). Zero deadline means "no timeout";
	// zero arrival means "time origin" — both are exempt from this check.
	if deadlineUs > 0 && arrivalTimeUs > 0 && deadlineUs < arrivalTimeUs {
		return nil, fmt.Errorf("parsing deadline_us: value %d precedes arrival_time_us %d (corrupt trace?)", deadlineUs, arrivalTimeUs)
	}
	finishReason := strings.TrimSpace(row[27])

	return &TraceRecord{
		RequestID:         requestID,
		ClientID:          row[1],
		TenantID:          row[2],
		SLOClass:          row[3],
		GIEPriority:       giePriority,
		SessionID:         row[5],
		RoundIndex:        roundIndex,
		PrefixGroup:       row[7],
		PrefixLength:      prefixLength,
		Streaming:         streaming,
		InputTokens:       inputTokens,
		OutputTokens:      outputTokens,
		TextTokens:        textTokens,
		ImageTokens:       imageTokens,
		AudioTokens:       audioTokens,
		VideoTokens:       videoTokens,
		ReasonRatio:       reasonRatio,
		Model:             row[17],
		DeadlineUs:        deadlineUs,
		ServerInputTokens: serverInputTokens,
		ArrivalTimeUs:     arrivalTimeUs,
		SendTimeUs:        sendTimeUs,
		FirstChunkTimeUs:  firstChunkTimeUs,
		LastChunkTimeUs:   lastChunkTimeUs,
		NumChunks:         numChunks,
		Status:            row[25],
		ErrorMessage:      strings.TrimSpace(row[26]),
		FinishReason:      finishReason,
	}, nil
}

// RequestsToTraceRecords converts simulation requests to trace v2 records.
// Uses array index as RequestID (request IDs may be non-numeric for session follow-ups).
// OutputTokens records the pre-determined count (len(req.OutputTokens)) for all requests,
// preserving workload input for replay fidelity across A/B policy comparisons.
// LastChunkTimeUs is computed as ArrivalTime + FirstTokenTime + sum(ITL), which
// represents the client-observable last-token delivery time. This deliberately
// excludes PostDecodeFixedOverhead (server-side processing after final token)
// and therefore differs from the E2E value stored in Metrics.RequestE2Es.
// PrefixGroup and PrefixLength are preserved; InputTokens records the suffix-only
// count (total - PrefixLength) so that LoadTraceV2Requests can reconstruct the
// full input by prepending a shared prefix of the correct length.
func RequestsToTraceRecords(requests []*sim.Request) []TraceRecord {
	records := make([]TraceRecord, 0, len(requests))
	for i, req := range requests {
		status := "incomplete"
		switch req.State {
		case sim.StateCompleted:
			status = "ok"
		case sim.StateTimedOut:
			status = "timeout"
		}

		// Absolute timing (ticks = microseconds)
		// Both chunk timestamps guarded by TTFTSet to avoid producing
		// LastChunkTimeUs = ArrivalTime for prefill-timeout requests.
		// For StateRunning requests with TTFTSet=true, LastChunkTimeUs
		// represents the last token generated so far (partial execution),
		// not the final token. Status "incomplete" distinguishes these.
		var firstChunkUs, lastChunkUs int64
		if req.TTFTSet {
			firstChunkUs = req.ArrivalTime + req.FirstTokenTime
			e2e := req.FirstTokenTime
			for _, itl := range req.ITL {
				e2e += itl
			}
			lastChunkUs = req.ArrivalTime + e2e
		}

		prefixLen := req.PrefixLength
		inputTokens := len(req.InputTokens) - prefixLen
		if inputTokens < 0 {
			// Safety: PrefixLength exceeds InputTokens (should not happen with well-formed data).
			// Treat as no prefix. Detectable in output: PrefixLength=0 with non-empty PrefixGroup.
			// R6: no logrus in sim/ — caller is responsible for detecting this via the record.
			inputTokens = len(req.InputTokens)
			prefixLen = 0
		}

		records = append(records, TraceRecord{
			RequestID:        i,
			ClientID:         req.ClientID,
			TenantID:         req.TenantID,
			SLOClass:         req.SLOClass,
			GIEPriority:      req.GIEPriority,
			SessionID:        req.SessionID,
			RoundIndex:       req.RoundIndex,
			PrefixGroup:      req.PrefixGroup,
			PrefixLength:     prefixLen,
			Streaming:        req.Streaming,
			InputTokens:      inputTokens,      // suffix-only: total - PrefixLength
			OutputTokens:     len(req.OutputTokens), // pre-determined count for replay fidelity
			TextTokens:       req.TextTokenCount,
			ImageTokens:      req.ImageTokenCount,
			AudioTokens:      req.AudioTokenCount,
			VideoTokens:      req.VideoTokenCount,
			ReasonRatio:      req.ReasonRatio,
			Model:            req.Model,
			DeadlineUs:       req.Deadline,
			ArrivalTimeUs:    req.ArrivalTime,
			SendTimeUs:       req.ArrivalTime, // no real network send in simulation
			FirstChunkTimeUs: firstChunkUs,
			LastChunkTimeUs:  lastChunkUs,
			NumChunks:        0, // not tracked in simulation
			Status:           status,
		})
	}
	return records
}
