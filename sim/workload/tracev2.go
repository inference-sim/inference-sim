package workload

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/check"
	"gopkg.in/yaml.v3"
)

// TraceVersion is the trace format this build reads and writes: version 4, the format
// blis-substrate's contracts/spec/trace defines. A trace of another version is refused; a
// version 3 trace is converted with `blis convert trace` (ConvertTraceV3).
const TraceVersion = 4

// TraceHeader is a trace's YAML header (trace format version 4).
type TraceHeader struct {
	Version  int    `yaml:"trace_version"`
	TimeUnit string `yaml:"time_unit"`
	Mode     string `yaml:"mode"` // TraceModeGenerated, TraceModeReplayed or TraceModeConverted
	// WorkloadSeed is the workload generator's seed, when one was recorded.
	WorkloadSeed *int64 `yaml:"workload_seed,omitempty"`

	// SessionContextGrowth controls per-round input assembly in closed-loop
	// replay. "accumulate" makes each session reuse a single growing token
	// buffer so round N+1's input is byte-identical to round N's plus new
	// tokens (strict prefix reuse for KV-cache modeling; see design §5).
	// Empty (default) = each round's input is generated independently
	// (pre-existing behavior). Set by `blis convert otel`.
	SessionContextGrowth string `yaml:"session_context_growth,omitempty"`

	// GoodputSLOTargets: per-SLO-class TTFT/ITL/E2E thresholds for goodput
	// measurement (issue #1409, BC-8). Persisted in trace headers so run and
	// replay carry the same SLO definition.
	GoodputSLOTargets map[string]SLODimTargets `yaml:"goodput_slo_targets,omitempty"`

	// KVOffload records the RESOLVED KV-cache offload configuration (H5, #1587) so it
	// round-trips run→replay (BC-G6, INV-13). Absent (nil, omitempty) when no
	// --kv-offload-config was supplied — so a run without offload writes a
	// byte-identical header to a pre-feature build (BC-G5). The resolved bandwidth /
	// latency numbers are recorded (not device_class), so replay reproduces the exact
	// physics regardless of the replay host's defaults.yaml; replay uses them
	// authoritatively and logrus.Fatalf's on any config it cannot reconstruct.
	KVOffload *TraceKVOffloadConfig `yaml:"kv_offload,omitempty"`

	// MaxNodesSpanned records the largest number of physical nodes any single model
	// instance occupied during the source run (#1530). Written by `blis run` only when
	// node-pool placement actually spread an instance's tensor-parallel group across a
	// node boundary; absent (0, omitempty) otherwise, so a run without multi-node
	// placement writes a byte-identical header to a pre-feature build (INV-6).
	//
	// READ BY REPLAY, BUT ONLY AS A REFUSAL SIGNAL — never to reconstruct anything.
	// This is the distinction to keep in mind before building on this field. #1530 offered
	// two designs for run/replay parity: round-trip the cross-node signal so replay
	// reproduces the run's step times, or keep a loud rejection. This is the SECOND one,
	// extended to cover the trace path. Cross-node collective traffic is charged to step
	// time, and node pools are `blis run`-only, so a trace from a multi-node run replayed
	// without them would silently reproduce the workload at single-node speed — faster
	// than the run it came from. `blis replay` therefore treats a value above 1 as a hard
	// error (INV-13: a feature replay cannot reproduce must fail loudly, never degrade
	// silently). It does NOT use the value to rebuild a fleet, and a future change that
	// wanted replay to reproduce cross-node timing would need the topology per instance,
	// not this single fleet-wide maximum.
	//
	// Normalization note: `omitempty` drops only 0, so a span of exactly 1 WOULD be
	// serialized. The writer normalizes 1 to 0 (see cmd.crossNodeSpanForTrace) so that a
	// single-node node-pool run does not start emitting a new header key; the coupling is
	// pinned by TestTraceHeader_MaxNodesSpanned_OneIsNotOmittedBySerialization.
	MaxNodesSpanned int `yaml:"max_nodes_spanned,omitempty"`

	// Router records the inputs of the llm-d router the source run routed with, so a replay
	// rebuilds the same router (INV-13). Absent from converted traces and from traces
	// written before the router was embedded.
	Router *TraceRouterConfig `yaml:"router,omitempty"`
}

// Validate refuses a header value that no run could have recorded, with the validator the same
// kind of value meets on the command line (package check, SIM-11): the router's three timings are
// spans of simulated time, as --gateway-latency, --epp-scrape-jitter and --kv-event-delay are; each
// goodput latency target is a positive span, as --slo-ttft, --slo-itl and --slo-e2e are. A target
// of 0 is the encoding of an absent one (SLODimTargets omits it), so only a stated value is
// checked. The fields are visited in a fixed order, so the first refusal is deterministic (R2).
func (h *TraceHeader) Validate() error {
	if r := h.Router; r != nil {
		for _, f := range []struct {
			name string
			v    int64
		}{{"router.gateway_latency_us", r.GatewayLatencyUs}, {"router.scrape_jitter_us", r.ScrapeJitterUs},
			{"router.kv_event_delay_us", r.KVEventDelayUs}} {
			if err := check.DurationUs(f.name, f.v); err != nil {
				return err
			}
		}
	}
	if err := check.Count("max_nodes_spanned", int64(h.MaxNodesSpanned)); err != nil {
		return err
	}
	return validateGoodputTargets(h.GoodputSLOTargets)
}

// TraceRouterConfig is the trace-header record of the embedded llm-d router's inputs: the
// EPP command-line flags (with the EndpointPickerConfig inlined as --config-text), the
// Kubernetes manifests, the extra pod labels, and the environment model around the router.
type TraceRouterConfig struct {
	EPPFlags         []string          `yaml:"epp_flags"`
	Manifests        string            `yaml:"manifests,omitempty"`
	PodLabels        map[string]string `yaml:"pod_labels,omitempty"`
	GatewayLatencyUs int64             `yaml:"gateway_latency_us,omitempty"`
	ScrapeJitterUs   int64             `yaml:"scrape_jitter_us,omitempty"`
	KVEventDelayUs   int64             `yaml:"kv_event_delay_us,omitempty"`
	// Seed is the seed the router ran with (its random streams and the prompts it was shown);
	// a replay runs its router with it. Absent in traces BLIS did not record.
	Seed *int64 `yaml:"seed,omitempty"`
}

// TraceKVOffloadConfig is the trace-header serialization of a resolved
// sim.KVOffloadConfig (H5, #1587). It is a workload-package struct — decoupled from
// sim.KVOffloadConfig so the on-disk trace format is stable independent of the
// internal config type. cmd/ owns the sim↔header conversion. Read strictly (KnownFields) via
// LoadTraceV2, so an unknown offload sub-key is a hard error (BC-N3 version guard).
type TraceKVOffloadConfig struct {
	CPUBytesToUse          int64                `yaml:"cpu_bytes_to_use"`
	PerBlockBytes          int64                `yaml:"per_block_bytes"`
	BlockSize              int64                `yaml:"block_size"`
	BlocksPerChunk         int64                `yaml:"blocks_per_chunk"`
	TokensPerHash          int64                `yaml:"tokens_per_hash"`
	EvictionPolicy         string               `yaml:"eviction_policy"`
	OffloadPromptOnly      bool                 `yaml:"offload_prompt_only"`
	SelfDescribingKVEvents bool                 `yaml:"self_describing_kv_events"`
	Tiers                  []TraceKVOffloadTier `yaml:"secondary_tiers,omitempty"`
	// SimpleCPUOffload / LazyOffload record vLLM's SimpleCPUOffloadConnector and its mode.
	SimpleCPUOffload bool `yaml:"simple_cpu_offload,omitempty"`
	LazyOffload      bool `yaml:"lazy_offload,omitempty"`
}

// TraceKVOffloadTier is the trace-header serialization of one resolved
// sim.KVOffloadTier. Bandwidths/latency are the RESOLVED numbers (bytes/µs, µs);
// device_class is an informational echo only.
type TraceKVOffloadTier struct {
	Type           string  `yaml:"type"`
	RootDir        string  `yaml:"root_dir"`
	NReadThreads   int64   `yaml:"n_read_threads"`
	NWriteThreads  int64   `yaml:"n_write_threads"`
	Locality       string  `yaml:"locality,omitempty"`
	EnableKVEvents bool    `yaml:"enable_kv_events"`
	DirectIO       bool    `yaml:"direct_io"`
	DeviceClass    string  `yaml:"device_class,omitempty"`
	ReadBandwidth  float64 `yaml:"read_bandwidth"`
	WriteBandwidth float64 `yaml:"write_bandwidth"`
	BaseLatency    float64 `yaml:"base_latency"`
	// #1581 device model (resolved for the selected regime). Always present so the
	// round-trip is lossless (BC-G6); Qsat=1/f₁=1.0/σ=0 encode "ramp/jitter off".
	SaturationQueueDepth   int64   `yaml:"saturation_queue_depth"`
	SingleTransferFraction float64 `yaml:"single_transfer_fraction"`
	// CapacityBytes is the tier's resolved size in bytes (0 unbounded).
	CapacityBytes       int64   `yaml:"capacity_bytes,omitempty"`
	LatencyJitterStddev float64 `yaml:"latency_jitter_stddev"`
}

// TraceRecord is one row of a trace's data CSV: one request.
type TraceRecord struct {
	RequestID        int
	ClientID         string
	TenantID         string
	SLOClass         string
	SessionID        string
	RoundIndex       int
	PrefixGroup      string
	PrefixLength     int
	Streaming        bool
	InputTokens      int
	OutputTokens     int
	TextTokens       int
	ImageTokens      int
	AudioTokens      int
	VideoTokens      int
	ReasonRatio      float64
	Model            string // model name (e.g., "meta-llama/Llama-3.1-8B-Instruct"); empty = default model
	DeadlineUs       int64  // absolute deadline timestamp in microseconds (same time origin as ArrivalTimeUs); 0 = no timeout
	SLOTargetUs      int64  // per-request SLO TTFT target in microseconds; 0 = no target
	ArrivalTimeUs    int64
	FirstChunkTimeUs int64
	LastChunkTimeUs  int64
	Status           string // "ok", "error", "timeout" or "incomplete"
	Adapter          string // LoRA adapter id serving this request (registry key; #1464); empty = base-model-only
	// ThinkTimeUs is the recorded per-round client think time in µs (gap from the
	// previous request's end), set by agentic-trace converters (#1478) and preferred
	// over arrival-gap derivation in closed-loop replay. It is a PRESENCE signal
	// (#1608): nil = not recorded (CSV cell empty; replay falls back to arrival gaps),
	// non-nil = recorded — INCLUDING a recorded &0 (CSV cell "0"), which a genuinely-
	// zero recorded think (e.g. Weka's overlap-clamped rounds) uses so it is no longer
	// conflated with "absent". A negative recorded value is a load error (INV-3).
	ThinkTimeUs *int64
	// InputTokensReset is a context-compaction marker (#1609): the recorded ABSOLUTE
	// input for a non-monotone round (in_N < in_{N-1} + out_{N-1}, where the delta law
	// clamps to 0). Set only by the shared agentic-trace encoder, only on such rounds.
	// nil (not recorded / monotone round) is the common case and writes no CSV cell, so
	// a trace without any compaction round is byte-identical to a pre-#1609 trace (INV-6).
	// In accumulate closed-loop replay the buffer re-seeds to this absolute at the
	// boundary (see EncodeSessionToTraceRecords / SessionManager.OnComplete). *int64
	// presence (nil = absent) mirrors the think_time_us non-lossy encoding (#1608/#1612).
	InputTokensReset *int64
}

// TraceV2 combines header and records for a complete trace.
type TraceV2 struct {
	Header  TraceHeader
	Records []TraceRecord
}

// traceColumns are the data CSV's required columns, in the order they are written (trace
// format version 4, the order of contracts/spec/trace's Columns).
var traceColumns = []string{
	"request_id", "client_id", "tenant_id", "slo_class", "session_id", "round_index",
	"prefix_group", "prefix_length", "streaming", "input_tokens", "output_tokens",
	"text_tokens", "image_tokens", "audio_tokens", "video_tokens", "reason_ratio",
	"model", "deadline_us", "arrival_time_us", "first_chunk_time_us", "last_chunk_time_us",
	"status",
}

// traceOptionalColumns follow traceColumns, in this order, each written only when some
// record carries a value for it, so a trace that uses none has exactly traceColumns.
var traceOptionalColumns = []string{"slo_target_us", "adapter", "think_time_us", "input_tokens_reset"}

// traceV3OnlyColumns are the version 3 columns version 4 does not have. Each is named in a
// refusal, so a version 3 data file read as version 4 is pointed at the conversion.
var traceV3OnlyColumns = map[string]bool{
	"server_input_tokens": true, "send_time_us": true, "num_chunks": true,
	"error_message": true, "finish_reason": true, "vllm_priority": true, "x_request_id": true,
}

// traceConversionHint names the conversion a version 3 trace needs.
const traceConversionHint = "convert it with `blis convert trace --header <v3>.yaml --data <v3>.csv --trace-output <prefix>`"

// ExportTraceV2 writes the trace header (YAML) and data (CSV) to separate files, in trace
// format version 4. The header's Version must be TraceVersion.
func ExportTraceV2(header *TraceHeader, records []TraceRecord, headerPath, dataPath string) error {
	if header.Version != TraceVersion {
		return fmt.Errorf("trace header version %d: this build writes version %d", header.Version, TraceVersion)
	}
	var buf bytes.Buffer
	if err := writeTraceHeader(&buf, header); err != nil {
		return err
	}
	if err := os.WriteFile(headerPath, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("writing trace header: %w", err)
	}
	file, err := os.Create(dataPath)
	if err != nil {
		return fmt.Errorf("creating trace data file: %w", err)
	}
	return writeTraceV2Data(file, header, records)
}

// writeTraceHeader writes h as YAML indented by two spaces, as contracts/spec/trace's
// WriteHeader does.
func writeTraceHeader(w io.Writer, h *TraceHeader) error {
	enc := yaml.NewEncoder(w)
	enc.SetIndent(2)
	if err := enc.Encode(h); err != nil {
		return fmt.Errorf("marshaling trace header: %w", err)
	}
	if err := enc.Close(); err != nil {
		return fmt.Errorf("marshaling trace header: %w", err)
	}
	return nil
}

// writeTraceV2Data writes the trace's CSV rows to w, flushes them and closes w. It returns the
// first error among the writes, the flush and the close, so a trace that did not reach the
// disk in full is never reported as exported (SIM-2, #661). w is closed on every path.
func writeTraceV2Data(w io.WriteCloser, header *TraceHeader, records []TraceRecord) error {
	err := writeTraceV2Rows(w, header, records)
	if cerr := w.Close(); err == nil && cerr != nil {
		err = fmt.Errorf("closing trace data file: %w", cerr)
	}
	return err
}

// writeTraceV2Rows writes the header row and every record to w through a csv.Writer, and
// flushes it. The csv.Writer buffers its output, so a failing w is seen at the flush, which is
// why the flush's error is returned. An optional column is written only when some record
// carries a value for it: slo_target_us when a target is positive, adapter when one is named,
// think_time_us and input_tokens_reset when one is recorded (a recorded 0 included).
func writeTraceV2Rows(w io.Writer, _ *TraceHeader, records []TraceRecord) error {
	has := map[string]bool{}
	for _, r := range records {
		has["slo_target_us"] = has["slo_target_us"] || r.SLOTargetUs != 0
		has["adapter"] = has["adapter"] || r.Adapter != ""
		has["think_time_us"] = has["think_time_us"] || r.ThinkTimeUs != nil
		has["input_tokens_reset"] = has["input_tokens_reset"] || r.InputTokensReset != nil
	}
	columns := append([]string{}, traceColumns...)
	for _, c := range traceOptionalColumns {
		if has[c] {
			columns = append(columns, c)
		}
	}
	writer := csv.NewWriter(w)
	if err := writer.Write(columns); err != nil {
		return fmt.Errorf("writing CSV header: %w", err)
	}
	ptr := func(v *int64) string {
		if v == nil {
			return "" // not recorded, distinct from a recorded 0 (#1608)
		}
		return strconv.FormatInt(*v, 10)
	}
	for _, r := range records {
		row := []string{
			strconv.Itoa(r.RequestID), r.ClientID, r.TenantID, r.SLOClass, r.SessionID,
			strconv.Itoa(r.RoundIndex), r.PrefixGroup, strconv.Itoa(r.PrefixLength),
			strconv.FormatBool(r.Streaming), strconv.Itoa(r.InputTokens), strconv.Itoa(r.OutputTokens),
			strconv.Itoa(r.TextTokens), strconv.Itoa(r.ImageTokens), strconv.Itoa(r.AudioTokens),
			strconv.Itoa(r.VideoTokens), strconv.FormatFloat(r.ReasonRatio, 'f', -1, 64), r.Model,
			strconv.FormatInt(r.DeadlineUs, 10), strconv.FormatInt(r.ArrivalTimeUs, 10),
			strconv.FormatInt(r.FirstChunkTimeUs, 10), strconv.FormatInt(r.LastChunkTimeUs, 10),
			r.Status,
		}
		for _, c := range traceOptionalColumns {
			if !has[c] {
				continue
			}
			switch c {
			case "slo_target_us":
				row = append(row, strconv.FormatInt(r.SLOTargetUs, 10))
			case "adapter":
				row = append(row, r.Adapter)
			case "think_time_us":
				row = append(row, ptr(r.ThinkTimeUs))
			case "input_tokens_reset":
				row = append(row, ptr(r.InputTokensReset))
			}
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("writing CSV row %d: %w", r.RequestID, err)
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return fmt.Errorf("flushing trace data CSV: %w", err)
	}
	return nil
}

// readTraceHeader decodes a header strictly: an unknown key is an error. A header of another
// version is refused before its keys are decoded, so a version 3 header is named as such and
// pointed at the conversion rather than refused for a key version 4 dropped.
func readTraceHeader(headerPath string) (*TraceHeader, error) {
	data, err := os.ReadFile(headerPath)
	if err != nil {
		return nil, fmt.Errorf("reading trace header: %w", err)
	}
	var probe struct {
		Version int `yaml:"trace_version"`
	}
	if err := yaml.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("parsing trace header: %w", err)
	}
	switch {
	case probe.Version == 3:
		return nil, fmt.Errorf("trace header %s is trace format version 3; this build reads version %d: %s",
			headerPath, TraceVersion, traceConversionHint)
	case probe.Version != TraceVersion:
		return nil, fmt.Errorf("trace header %s: trace_version must be %d, got %d", headerPath, TraceVersion, probe.Version)
	}
	var header TraceHeader
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&header); err != nil {
		return nil, fmt.Errorf("parsing trace header: %w", err)
	}
	return &header, nil
}

// LoadTraceV2 reads a trace header (YAML) and data (CSV) in trace format version 4.
func LoadTraceV2(headerPath, dataPath string) (*TraceV2, error) {
	header, err := readTraceHeader(headerPath)
	if err != nil {
		return nil, err
	}
	if err := validateTraceMode(header.Mode); err != nil {
		return nil, fmt.Errorf("parsing trace header: %w", err)
	}
	if err := header.Validate(); err != nil {
		return nil, fmt.Errorf("trace header %s: %w", headerPath, err)
	}
	file, err := os.Open(dataPath)
	if err != nil {
		return nil, fmt.Errorf("opening trace data: %w", err)
	}
	defer func() { _ = file.Close() }()
	records, err := readTraceRecords(file)
	if err != nil {
		return nil, fmt.Errorf("trace data %s: %w", dataPath, err)
	}
	return &TraceV2{Header: *header, Records: records}, nil
}

// readTraceRecords reads the data CSV. Columns are found by name in the header row, so their
// order does not matter; a missing required column, an unknown or repeated column, and a row
// with another number of cells are each refused. A version 3 column is refused naming the
// conversion.
func readTraceRecords(r io.Reader) ([]TraceRecord, error) {
	reader := csv.NewReader(r)
	head, err := reader.Read()
	if err == io.EOF {
		return nil, fmt.Errorf("empty file; the first row names the columns")
	}
	if err != nil {
		return nil, fmt.Errorf("reading CSV header: %w", err)
	}
	known := map[string]bool{}
	for _, c := range append(append([]string{}, traceColumns...), traceOptionalColumns...) {
		known[c] = true
	}
	at := map[string]int{}
	for i, c := range head {
		switch {
		case traceV3OnlyColumns[c]:
			return nil, fmt.Errorf("column %q belongs to trace format version 3, which this build does not read: %s",
				c, traceConversionHint)
		case !known[c]:
			return nil, fmt.Errorf("column %q is not a trace column (version %d)", c, TraceVersion)
		}
		if _, dup := at[c]; dup {
			return nil, fmt.Errorf("column %q appears twice", c)
		}
		at[c] = i
	}
	for _, c := range traceColumns {
		if _, ok := at[c]; !ok {
			return nil, fmt.Errorf("required column %q is missing", c)
		}
	}
	var records []TraceRecord
	for {
		row, err := reader.Read()
		if err == io.EOF {
			return records, nil
		}
		if err != nil {
			return nil, fmt.Errorf("reading CSV row: %w", err)
		}
		rec, err := parseTraceRecord(row, at)
		if err != nil {
			return nil, err
		}
		records = append(records, *rec)
	}
}

// rowError places a field refusal from package check in its trace row. The message keeps the
// field name and value check gives it, so a value refused here reads as the same value refused on
// the command line (SIM-11).
func rowError(requestID int, err error) error {
	return fmt.Errorf("trace row request_id %d: %w", requestID, err)
}

// traceRow reads the cells of one data row by column name, keeping the first parse error.
type traceRow struct {
	row []string
	at  map[string]int
	err error
}

// cell returns the named column's cell and whether the file has the column.
func (t *traceRow) cell(c string) (string, bool) {
	i, ok := t.at[c]
	if !ok {
		return "", false
	}
	return t.row[i], true
}

func (t *traceRow) str(c string) string { s, _ := t.cell(c); return strings.TrimSpace(s) }

func (t *traceRow) integer(c string) int {
	if t.err != nil {
		return 0
	}
	s, _ := t.cell(c)
	v, err := strconv.Atoi(s)
	if err != nil {
		t.err = fmt.Errorf("parsing %s %q: %w", c, s, err)
	}
	return v
}

func (t *traceRow) int64(c string) int64 {
	if t.err != nil {
		return 0
	}
	s, ok := t.cell(c)
	if !ok {
		return 0 // an optional column the file does not have
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		t.err = fmt.Errorf("parsing %s %q: %w", c, s, err)
	}
	return v
}

// optional reads a presence-signalled column: an absent column or an empty cell is nil, so a
// recorded 0 stays distinct from no record (#1608).
func (t *traceRow) optional(c string) *int64 {
	if t.err != nil {
		return nil
	}
	s, ok := t.cell(c)
	if s = strings.TrimSpace(s); !ok || s == "" {
		return nil
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		t.err = fmt.Errorf("parsing %s %q: %w", c, s, err)
		return nil
	}
	return &v
}

// parseTraceRecord parses one data row; at maps each column the file has to its cell index.
// Each value meets the validator its kind meets on the command line (package check, SIM-11).
func parseTraceRecord(row []string, at map[string]int) (*TraceRecord, error) {
	if len(row) != len(at) {
		return nil, fmt.Errorf("CSV row has %d columns, the header names %d", len(row), len(at))
	}
	t := &traceRow{row: row, at: at}
	rec := &TraceRecord{
		ClientID:  t.row[at["client_id"]],
		TenantID:  t.row[at["tenant_id"]],
		SLOClass:  t.row[at["slo_class"]],
		SessionID: t.row[at["session_id"]],
		// prefix_group and model are read as written.
		PrefixGroup: t.row[at["prefix_group"]],
		Model:       t.row[at["model"]],
		Status:      t.row[at["status"]],
		Adapter:     t.str("adapter"),
	}
	rec.RequestID = t.integer("request_id")
	rec.RoundIndex = t.integer("round_index")
	rec.PrefixLength = t.integer("prefix_length")
	rec.InputTokens = t.integer("input_tokens")
	rec.OutputTokens = t.integer("output_tokens")
	rec.TextTokens = t.integer("text_tokens")
	rec.ImageTokens = t.integer("image_tokens")
	rec.AudioTokens = t.integer("audio_tokens")
	rec.VideoTokens = t.integer("video_tokens")
	rec.DeadlineUs = t.int64("deadline_us")
	rec.ArrivalTimeUs = t.int64("arrival_time_us")
	rec.FirstChunkTimeUs = t.int64("first_chunk_time_us")
	rec.LastChunkTimeUs = t.int64("last_chunk_time_us")
	rec.SLOTargetUs = t.int64("slo_target_us")
	rec.ThinkTimeUs = t.optional("think_time_us")
	rec.InputTokensReset = t.optional("input_tokens_reset")
	if t.err != nil {
		return nil, t.err
	}
	s := t.row[at["streaming"]]
	streaming, err := strconv.ParseBool(s)
	if err != nil {
		return nil, fmt.Errorf("parsing streaming %q: %w", s, err)
	}
	rec.Streaming = streaming
	s = t.row[at["reason_ratio"]]
	if rec.ReasonRatio, err = strconv.ParseFloat(s, 64); err != nil {
		return nil, fmt.Errorf("parsing reason_ratio %q: %w", s, err)
	}
	id := rec.RequestID
	for _, err := range []error{
		check.Count("round_index", int64(rec.RoundIndex)),
		check.TokenCount("prefix_length", int64(rec.PrefixLength)),
		// A negative token count would panic in make([]int, n) in LoadTraceV2Requests. Zero is
		// accepted: the gateway drops an empty prompt as unservable and the run continues (#2065).
		check.TokenCount("input_tokens", int64(rec.InputTokens)),
		check.TokenCount("output_tokens", int64(rec.OutputTokens)),
		check.TokenCount("text_tokens", int64(rec.TextTokens)),
		check.TokenCount("image_tokens", int64(rec.ImageTokens)),
		check.TokenCount("audio_tokens", int64(rec.AudioTokens)),
		check.TokenCount("video_tokens", int64(rec.VideoTokens)),
		check.Fraction("reason_ratio", rec.ReasonRatio),
		// 0 means no client deadline, and no target.
		check.DurationUs("deadline_us", rec.DeadlineUs),
		check.DurationUs("slo_target_us", rec.SLOTargetUs),
		// arrival_time_us is the injection origin (#1606): a negative value would yield a
		// negative DES injection tick (INV-3).
		check.DurationUs("arrival_time_us", rec.ArrivalTimeUs),
	} {
		if err != nil {
			return nil, rowError(id, err)
		}
	}
	if rec.ThinkTimeUs != nil {
		if err := check.DurationUs("think_time_us", *rec.ThinkTimeUs); err != nil {
			return nil, rowError(id, err)
		}
	}
	if rec.InputTokensReset != nil {
		if err := check.TokenCount("input_tokens_reset", *rec.InputTokensReset); err != nil {
			return nil, rowError(id, err)
		}
	}
	// A deadline before the arrival would time the request out at enqueue. Zero deadline means
	// no timeout and zero arrival the time origin; both are exempt.
	if rec.DeadlineUs > 0 && rec.ArrivalTimeUs > 0 && rec.DeadlineUs < rec.ArrivalTimeUs {
		return nil, fmt.Errorf("parsing deadline_us: value %d precedes arrival_time_us %d (corrupt trace?)", rec.DeadlineUs, rec.ArrivalTimeUs)
	}
	return rec, nil
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
		inputTokens := int(req.InputLen()) - prefixLen
		if inputTokens < 0 {
			// Safety: PrefixLength exceeds InputTokens (should not happen with well-formed data).
			// Treat as no prefix. Detectable in output: PrefixLength=0 with non-empty PrefixGroup.
			// R6: no logrus in sim/ — caller is responsible for detecting this via the record.
			inputTokens = int(req.InputLen())
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
			InputTokens:      inputTokens,           // suffix-only: total - PrefixLength
			OutputTokens:     len(req.OutputTokens), // pre-determined count for replay fidelity
			TextTokens:       req.TextTokenCount,
			ImageTokens:      req.ImageTokenCount,
			AudioTokens:      req.AudioTokenCount,
			VideoTokens:      req.VideoTokenCount,
			ReasonRatio:      req.ReasonRatio,
			Model:            req.Model,
			DeadlineUs:       req.Deadline,
			SLOTargetUs:      req.SLOTargetUs,
			ArrivalTimeUs:    req.ArrivalTime,
			FirstChunkTimeUs: firstChunkUs,
			LastChunkTimeUs:  lastChunkUs,
			Status:           status,
			Adapter:          req.Adapter, // LoRA adapter id (#1464); "" = base-model-only
		})
	}
	return records
}
