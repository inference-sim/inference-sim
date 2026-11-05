package workload

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
	"gopkg.in/yaml.v3"
)

// Trace format version 3 is the format v0.15.0 wrote and read (version 2 is the same without
// goodput_slo_targets). Version 4 is version 3 without what only the removed `blis observe`
// command wrote or read (contracts/spec/trace): the "real" mode; the header's created_at,
// warm_up_requests, workload_spec, server, network and observed_kv_metrics; and the columns
// vllm_priority, x_request_id, server_input_tokens, send_time_us, num_chunks, error_message and
// finish_reason. Every other field keeps its name and meaning. This file reads version 3 only
// to convert it (`blis convert trace`); nothing replays it.

// traceV3Header is a version 2 or 3 header, every key v0.15.0 wrote. The keys version 4
// dropped are decoded only to be reported; server, network and observed_kv_metrics are kept
// opaque, since no part of version 4 reads them.
type traceV3Header struct {
	Version              int                      `yaml:"trace_version"`
	TimeUnit             string                   `yaml:"time_unit"`
	CreatedAt            string                   `yaml:"created_at,omitempty"`
	Mode                 string                   `yaml:"mode"`
	WarmUpRequests       int                      `yaml:"warm_up_requests"`
	WorkloadSeed         *int64                   `yaml:"workload_seed,omitempty"`
	WorkloadSpec         string                   `yaml:"workload_spec,omitempty"`
	SessionContextGrowth string                   `yaml:"session_context_growth,omitempty"`
	Server               any                      `yaml:"server,omitempty"`
	Network              any                      `yaml:"network,omitempty"`
	GoodputSLOTargets    map[string]SLODimTargets `yaml:"goodput_slo_targets,omitempty"`
	KVOffload            *TraceKVOffloadConfig    `yaml:"kv_offload,omitempty"`
	ObservedKVMetrics    any                      `yaml:"observed_kv_metrics,omitempty"`
	MaxNodesSpanned      int                      `yaml:"max_nodes_spanned,omitempty"`
	Router               *TraceRouterConfig       `yaml:"router,omitempty"`
}

// traceV3ModeReal is the mode `blis observe` wrote: a capture of a real server. Version 4
// writes a captured trace as mode converted.
const traceV3ModeReal = "real"

// TraceV3Drop is one version 3 field the conversion dropped, with the number of header keys
// or rows that carried a value for it. Fields that carried none are not listed.
type TraceV3Drop struct {
	Field string
	// Rows is the number of rows with a non-empty, non-zero value; 1 for a header key.
	Rows int
	// ChangesReplay is true when the value changed what v0.15.0's replay simulated, so the
	// converted trace replays differently: server_input_tokens set the prompt length of a
	// request with no prefix group, and send_time_us, when it differed from arrival_time_us,
	// set the injection time.
	ChangesReplay bool
}

// ConvertTraceV3 reads a version 2 or 3 trace and returns it as version 4, with each field
// version 4 dropped and how many rows carried it. Mode real becomes converted: a capture
// records requests and no BLIS deployment. A trace that is already version 4, or of an
// unknown version, is refused.
func ConvertTraceV3(headerPath, dataPath string) (*TraceV2, []TraceV3Drop, error) {
	raw, err := os.ReadFile(headerPath)
	if err != nil {
		return nil, nil, fmt.Errorf("reading trace header: %w", err)
	}
	var h traceV3Header
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode(&h); err != nil {
		return nil, nil, fmt.Errorf("parsing version 3 trace header %s: %w", headerPath, err)
	}
	switch h.Version {
	case 2, 3:
	case TraceVersion:
		return nil, nil, fmt.Errorf("trace header %s is already version %d", headerPath, TraceVersion)
	default:
		return nil, nil, fmt.Errorf("trace header %s: trace_version %d is not version 2 or 3", headerPath, h.Version)
	}
	header := TraceHeader{
		Version:              TraceVersion,
		TimeUnit:             trace.TimeUnit(h.TimeUnit),
		Mode:                 trace.Mode(h.Mode),
		WorkloadSeed:         h.WorkloadSeed,
		SessionContextGrowth: trace.ContextGrowth(h.SessionContextGrowth),
		GoodputSLOTargets:    h.GoodputSLOTargets,
		KVOffload:            h.KVOffload,
		MaxNodesSpanned:      h.MaxNodesSpanned,
		Router:               h.Router,
	}
	counts := map[string]int{}
	if h.Mode == traceV3ModeReal {
		header.Mode = TraceModeConverted
		counts["mode: real (now converted)"] = 1
	}
	for key, set := range map[string]bool{
		"created_at": h.CreatedAt != "", "warm_up_requests": h.WarmUpRequests != 0,
		"workload_spec": h.WorkloadSpec != "", "server": h.Server != nil, "network": h.Network != nil,
		"observed_kv_metrics": h.ObservedKVMetrics != nil,
	} {
		if set {
			counts[key] = 1
		}
	}
	if err := validateTraceMode(header.Mode); err != nil {
		return nil, nil, fmt.Errorf("trace header %s: %w", headerPath, err)
	}
	if err := checkTraceHeader(&header); err != nil {
		return nil, nil, fmt.Errorf("trace header %s: %w", headerPath, err)
	}

	f, err := os.Open(dataPath)
	if err != nil {
		return nil, nil, fmt.Errorf("opening trace data: %w", err)
	}
	defer func() { _ = f.Close() }()
	records, changed, err := readTraceV3Records(f, counts)
	if err != nil {
		return nil, nil, fmt.Errorf("trace data %s: %w", dataPath, err)
	}
	drops := make([]TraceV3Drop, 0, len(counts))
	for field, n := range counts {
		drops = append(drops, TraceV3Drop{Field: field, Rows: n, ChangesReplay: changed[field]})
	}
	sort.Slice(drops, func(i, j int) bool { return drops[i].Field < drops[j].Field })
	return &TraceV2{Header: header, Records: records}, drops, nil
}

// readTraceV3Records reads a version 3 data CSV by column name: the version 4 columns are
// parsed and validated as version 4 parses them, and each version 3 column is counted in
// counts when a row carries a value for it. changed names the dropped columns whose value
// changed what a version 3 replay simulated.
func readTraceV3Records(r io.Reader, counts map[string]int) ([]TraceRecord, map[string]bool, error) {
	reader := csv.NewReader(r)
	head, err := reader.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("reading CSV header: %w", err)
	}
	v4 := map[string]bool{}
	for _, c := range append(append([]string{}, traceColumns...), traceOptionalColumns...) {
		v4[c] = true
	}
	src := map[string]int{}
	for i, c := range head {
		if !v4[c] && !traceV3OnlyColumns[c] {
			return nil, nil, fmt.Errorf("column %q is not a version 3 trace column", c)
		}
		if _, dup := src[c]; dup {
			return nil, nil, fmt.Errorf("column %q appears twice", c)
		}
		src[c] = i
	}
	for _, c := range traceColumns {
		if _, ok := src[c]; !ok {
			return nil, nil, fmt.Errorf("required column %q is missing", c)
		}
	}
	// at maps each version 4 column the file has to its index in the projected row.
	var cols []string
	for _, c := range append(append([]string{}, traceColumns...), traceOptionalColumns...) {
		if _, ok := src[c]; ok {
			cols = append(cols, c)
		}
	}
	at := make(map[string]int, len(cols))
	for i, c := range cols {
		at[c] = i
	}
	changed := map[string]bool{}
	var records []TraceRecord
	for {
		row, err := reader.Read()
		if err == io.EOF {
			return records, changed, nil
		}
		if err != nil {
			return nil, nil, fmt.Errorf("reading CSV row: %w", err)
		}
		if len(row) != len(head) {
			return nil, nil, fmt.Errorf("CSV row has %d columns, the header names %d", len(row), len(head))
		}
		projected := make([]string, len(cols))
		for i, c := range cols {
			projected[i] = row[src[c]]
		}
		rec, err := parseTraceRecord(projected, at)
		if err != nil {
			return nil, nil, err
		}
		cell := func(c string) string {
			if i, ok := src[c]; ok {
				return row[i]
			}
			return ""
		}
		for c := range traceV3OnlyColumns {
			if v := cell(c); v != "" && v != "0" {
				counts[c]++
			}
		}
		// v0.15.0's replay read two of the dropped columns: a positive server_input_tokens was
		// the prompt length of a request with no prefix group, and a positive send_time_us the
		// injection time (re-based onto the arrival origin).
		if v := cell("server_input_tokens"); v != "" && v != "0" && rec.PrefixGroup == "" && v != fmt.Sprint(rec.InputTokens) {
			changed["server_input_tokens"] = true
		}
		if v := cell("send_time_us"); v != "" && v != "0" && v != fmt.Sprint(rec.ArrivalTimeUs) {
			changed["send_time_us"] = true
		}
		records = append(records, *rec)
	}
}
