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
	Mode           string `yaml:"mode"` // "real" or "generated"
	WarmUpRequests int    `yaml:"warm_up_requests"`
	WorkloadSeed   *int64 `yaml:"workload_seed,omitempty"`
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
	"request_id", "client_id", "tenant_id", "slo_class", "session_id", "round_index",
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
	reader.FieldsPerRecord = -1 // allow variable column count for backward compat

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
		// Column count tiers:
		//   27 = current (with prefix_length)
		//   25-26 = legacy (pre-prefix_length); 25 = pre-finish_reason
		//   <25 = unsupported
		const legacyMinColumns = 25
		if len(row) < legacyMinColumns {
			hint := ""
			if len(row) == 22 {
				hint = " (22-column trace predates model/deadline_us/server_input_tokens fields; re-export to upgrade)"
			}
			return nil, fmt.Errorf("CSV row has %d columns, expected at least %d%s", len(row), legacyMinColumns, hint)
		}

		var r *TraceRecord
		if len(row) >= len(traceV2Columns) { // 27+ = new schema (with prefix_length column)
			r, err = parseTraceRecord(row)
		} else {
			r, err = parseTraceRecordLegacy(row)
		}
		if err != nil {
			return nil, err
		}
		records = append(records, *r)
	}

	return &TraceV2{Header: header, Records: records}, nil
}

// parseTraceRecord parses a 27-column (current schema) CSV row.
func parseTraceRecord(row []string) (*TraceRecord, error) {
	requestID, err := strconv.Atoi(row[0])
	if err != nil {
		return nil, fmt.Errorf("parsing request_id %q: %w", row[0], err)
	}
	roundIndex, err := strconv.Atoi(row[5])
	if err != nil {
		return nil, fmt.Errorf("parsing round_index %q: %w", row[5], err)
	}
	// Column 7: prefix_length (new in 27-column schema)
	prefixLength, err := strconv.Atoi(row[7])
	if err != nil {
		return nil, fmt.Errorf("parsing prefix_length %q: %w", row[7], err)
	}
	if prefixLength < 0 {
		return nil, fmt.Errorf("parsing prefix_length: negative value %d not allowed", prefixLength)
	}
	// Column 8: streaming (was 7)
	streaming, err := strconv.ParseBool(row[8])
	if err != nil {
		return nil, fmt.Errorf("parsing streaming %q: %w", row[8], err)
	}
	// Column 9: input_tokens (was 8)
	inputTokens, err := strconv.Atoi(row[9])
	if err != nil {
		return nil, fmt.Errorf("parsing input_tokens %q: %w", row[9], err)
	}
	// Negative token counts cause make([]int, negative) panics in LoadTraceV2Requests.
	if inputTokens < 0 {
		return nil, fmt.Errorf("parsing input_tokens: negative value %d not allowed", inputTokens)
	}
	outputTokens, err := strconv.Atoi(row[10])
	if err != nil {
		return nil, fmt.Errorf("parsing output_tokens %q: %w", row[10], err)
	}
	if outputTokens < 0 {
		return nil, fmt.Errorf("parsing output_tokens: negative value %d not allowed", outputTokens)
	}
	textTokens, err := strconv.Atoi(row[11])
	if err != nil {
		return nil, fmt.Errorf("parsing text_tokens %q: %w", row[11], err)
	}
	if textTokens < 0 {
		return nil, fmt.Errorf("parsing text_tokens: negative value %d not allowed", textTokens)
	}
	imageTokens, err := strconv.Atoi(row[12])
	if err != nil {
		return nil, fmt.Errorf("parsing image_tokens %q: %w", row[12], err)
	}
	if imageTokens < 0 {
		return nil, fmt.Errorf("parsing image_tokens: negative value %d not allowed", imageTokens)
	}
	audioTokens, err := strconv.Atoi(row[13])
	if err != nil {
		return nil, fmt.Errorf("parsing audio_tokens %q: %w", row[13], err)
	}
	if audioTokens < 0 {
		return nil, fmt.Errorf("parsing audio_tokens: negative value %d not allowed", audioTokens)
	}
	videoTokens, err := strconv.Atoi(row[14])
	if err != nil {
		return nil, fmt.Errorf("parsing video_tokens %q: %w", row[14], err)
	}
	if videoTokens < 0 {
		return nil, fmt.Errorf("parsing video_tokens: negative value %d not allowed", videoTokens)
	}
	reasonRatio, err := strconv.ParseFloat(row[15], 64)
	if err != nil {
		return nil, fmt.Errorf("parsing reason_ratio %q: %w", row[15], err)
	}
	if math.IsNaN(reasonRatio) || math.IsInf(reasonRatio, 0) || reasonRatio < 0 || reasonRatio > 1.0 {
		return nil, fmt.Errorf("parsing reason_ratio %q: must be in range [0.0, 1.0], got %g", row[15], reasonRatio)
	}
	deadlineUs, err := strconv.ParseInt(row[17], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing deadline_us %q: %w", row[17], err)
	}
	if deadlineUs < 0 {
		return nil, fmt.Errorf("parsing deadline_us: negative value %d not allowed (use 0 for no timeout)", deadlineUs)
	}
	serverInputTokens, err := strconv.Atoi(row[18])
	if err != nil {
		return nil, fmt.Errorf("parsing server_input_tokens %q: %w", row[18], err)
	}
	if serverInputTokens < 0 {
		return nil, fmt.Errorf("parsing server_input_tokens: negative value %d not allowed", serverInputTokens)
	}
	arrivalTimeUs, err := strconv.ParseInt(row[19], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing arrival_time_us %q: %w", row[19], err)
	}
	sendTimeUs, err := strconv.ParseInt(row[20], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing send_time_us %q: %w", row[20], err)
	}
	firstChunkTimeUs, err := strconv.ParseInt(row[21], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing first_chunk_time_us %q: %w", row[21], err)
	}
	lastChunkTimeUs, err := strconv.ParseInt(row[22], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing last_chunk_time_us %q: %w", row[22], err)
	}
	numChunks, err := strconv.Atoi(row[23])
	if err != nil {
		return nil, fmt.Errorf("parsing num_chunks %q: %w", row[23], err)
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
	// finish_reason (column 26) — optional for backward compat with 26-column traces
	var finishReason string
	if len(row) >= 27 {
		finishReason = strings.TrimSpace(row[26])
	}

	return &TraceRecord{
		RequestID:         requestID,
		ClientID:          row[1],
		TenantID:          row[2],
		SLOClass:          row[3],
		SessionID:         row[4],
		RoundIndex:        roundIndex,
		PrefixGroup:       row[6],
		PrefixLength:      prefixLength,
		Streaming:         streaming,
		InputTokens:       inputTokens,
		OutputTokens:      outputTokens,
		TextTokens:        textTokens,
		ImageTokens:       imageTokens,
		AudioTokens:       audioTokens,
		VideoTokens:       videoTokens,
		ReasonRatio:       reasonRatio,
		Model:             row[16],
		DeadlineUs:        deadlineUs,
		ServerInputTokens: serverInputTokens,
		ArrivalTimeUs:     arrivalTimeUs,
		SendTimeUs:        sendTimeUs,
		FirstChunkTimeUs:  firstChunkTimeUs,
		LastChunkTimeUs:   lastChunkTimeUs,
		NumChunks:         numChunks,
		Status:            row[24],
		ErrorMessage:      strings.TrimSpace(row[25]),
		FinishReason:      finishReason,
	}, nil
}

// parseTraceRecordLegacy parses a 25-26 column (pre-prefix_length) CSV row.
// PrefixLength defaults to 0.
func parseTraceRecordLegacy(row []string) (*TraceRecord, error) {
	requestID, err := strconv.Atoi(row[0])
	if err != nil {
		return nil, fmt.Errorf("parsing request_id %q: %w", row[0], err)
	}
	roundIndex, err := strconv.Atoi(row[5])
	if err != nil {
		return nil, fmt.Errorf("parsing round_index %q: %w", row[5], err)
	}
	streaming, err := strconv.ParseBool(row[7])
	if err != nil {
		return nil, fmt.Errorf("parsing streaming %q: %w", row[7], err)
	}
	inputTokens, err := strconv.Atoi(row[8])
	if err != nil {
		return nil, fmt.Errorf("parsing input_tokens %q: %w", row[8], err)
	}
	if inputTokens < 0 {
		return nil, fmt.Errorf("parsing input_tokens: negative value %d not allowed", inputTokens)
	}
	outputTokens, err := strconv.Atoi(row[9])
	if err != nil {
		return nil, fmt.Errorf("parsing output_tokens %q: %w", row[9], err)
	}
	if outputTokens < 0 {
		return nil, fmt.Errorf("parsing output_tokens: negative value %d not allowed", outputTokens)
	}
	textTokens, err := strconv.Atoi(row[10])
	if err != nil {
		return nil, fmt.Errorf("parsing text_tokens %q: %w", row[10], err)
	}
	if textTokens < 0 {
		return nil, fmt.Errorf("parsing text_tokens: negative value %d not allowed", textTokens)
	}
	imageTokens, err := strconv.Atoi(row[11])
	if err != nil {
		return nil, fmt.Errorf("parsing image_tokens %q: %w", row[11], err)
	}
	if imageTokens < 0 {
		return nil, fmt.Errorf("parsing image_tokens: negative value %d not allowed", imageTokens)
	}
	audioTokens, err := strconv.Atoi(row[12])
	if err != nil {
		return nil, fmt.Errorf("parsing audio_tokens %q: %w", row[12], err)
	}
	if audioTokens < 0 {
		return nil, fmt.Errorf("parsing audio_tokens: negative value %d not allowed", audioTokens)
	}
	videoTokens, err := strconv.Atoi(row[13])
	if err != nil {
		return nil, fmt.Errorf("parsing video_tokens %q: %w", row[13], err)
	}
	if videoTokens < 0 {
		return nil, fmt.Errorf("parsing video_tokens: negative value %d not allowed", videoTokens)
	}
	reasonRatio, err := strconv.ParseFloat(row[14], 64)
	if err != nil {
		return nil, fmt.Errorf("parsing reason_ratio %q: %w", row[14], err)
	}
	if math.IsNaN(reasonRatio) || math.IsInf(reasonRatio, 0) || reasonRatio < 0 || reasonRatio > 1.0 {
		return nil, fmt.Errorf("parsing reason_ratio %q: must be in range [0.0, 1.0], got %g", row[14], reasonRatio)
	}
	deadlineUs, err := strconv.ParseInt(row[16], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing deadline_us %q: %w", row[16], err)
	}
	if deadlineUs < 0 {
		return nil, fmt.Errorf("parsing deadline_us: negative value %d not allowed (use 0 for no timeout)", deadlineUs)
	}
	serverInputTokens, err := strconv.Atoi(row[17])
	if err != nil {
		return nil, fmt.Errorf("parsing server_input_tokens %q: %w", row[17], err)
	}
	if serverInputTokens < 0 {
		return nil, fmt.Errorf("parsing server_input_tokens: negative value %d not allowed", serverInputTokens)
	}
	arrivalTimeUs, err := strconv.ParseInt(row[18], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing arrival_time_us %q: %w", row[18], err)
	}
	sendTimeUs, err := strconv.ParseInt(row[19], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing send_time_us %q: %w", row[19], err)
	}
	firstChunkTimeUs, err := strconv.ParseInt(row[20], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing first_chunk_time_us %q: %w", row[20], err)
	}
	lastChunkTimeUs, err := strconv.ParseInt(row[21], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing last_chunk_time_us %q: %w", row[21], err)
	}
	numChunks, err := strconv.Atoi(row[22])
	if err != nil {
		return nil, fmt.Errorf("parsing num_chunks %q: %w", row[22], err)
	}
	if numChunks < 0 {
		return nil, fmt.Errorf("parsing num_chunks: negative value %d not allowed", numChunks)
	}
	if deadlineUs > 0 && arrivalTimeUs > 0 && deadlineUs < arrivalTimeUs {
		return nil, fmt.Errorf("parsing deadline_us: value %d precedes arrival_time_us %d (corrupt trace?)", deadlineUs, arrivalTimeUs)
	}
	var finishReason string
	if len(row) >= 26 {
		finishReason = strings.TrimSpace(row[25])
	}

	return &TraceRecord{
		RequestID:         requestID,
		ClientID:          row[1],
		TenantID:          row[2],
		SLOClass:          row[3],
		SessionID:         row[4],
		RoundIndex:        roundIndex,
		PrefixGroup:       row[6],
		PrefixLength:      0, // legacy: no prefix_length column
		Streaming:         streaming,
		InputTokens:       inputTokens,
		OutputTokens:      outputTokens,
		TextTokens:        textTokens,
		ImageTokens:       imageTokens,
		AudioTokens:       audioTokens,
		VideoTokens:       videoTokens,
		ReasonRatio:       reasonRatio,
		Model:             row[15],
		DeadlineUs:        deadlineUs,
		ServerInputTokens: serverInputTokens,
		ArrivalTimeUs:     arrivalTimeUs,
		SendTimeUs:        sendTimeUs,
		FirstChunkTimeUs:  firstChunkTimeUs,
		LastChunkTimeUs:   lastChunkTimeUs,
		NumChunks:         numChunks,
		Status:            row[23],
		ErrorMessage:      strings.TrimSpace(row[24]),
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
