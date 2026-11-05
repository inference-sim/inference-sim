package trace

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"strconv"

	"gopkg.in/yaml.v3"
)

// Columns are the data CSV's required columns, in the order they are written. The order
// is inference-sim's tracev2.go traceV2Columns with the observe-only columns removed.
var Columns = []string{
	"request_id", "client_id", "tenant_id", "slo_class", "session_id", "round_index",
	"prefix_group", "prefix_length", "streaming", "input_tokens", "output_tokens",
	"text_tokens", "image_tokens", "audio_tokens", "video_tokens", "reason_ratio",
	"model", "deadline_us", "arrival_time_us", "first_chunk_time_us", "last_chunk_time_us",
	"status",
}

// OptionalColumns are written after Columns, in this order, each only when some record
// carries a value for it.
var OptionalColumns = []string{"slo_target_us", "adapter", "think_time_us", "input_tokens_reset"}

// ReadHeader reads a header strictly: an unknown key, a duplicate key or a value of the
// wrong type is an error. It does not validate; call Validate.
func ReadHeader(r io.Reader) (*Header, error) {
	dec := yaml.NewDecoder(r)
	dec.KnownFields(true)
	var h Header
	if err := dec.Decode(&h); err != nil {
		return nil, fmt.Errorf("trace header: %w", err)
	}
	var extra yaml.Node
	if err := dec.Decode(&extra); err != io.EOF {
		return nil, errors.New("trace header: more than one YAML document")
	}
	return &h, nil
}

// WriteHeader writes h as YAML.
func WriteHeader(w io.Writer, h *Header) error {
	enc := yaml.NewEncoder(w)
	enc.SetIndent(2)
	if err := enc.Encode(h); err != nil {
		return fmt.Errorf("trace header: %w", err)
	}
	return enc.Close()
}

// ReadRecords reads the data CSV. Columns are found by name in the header row, so their
// order does not matter. A missing required column, an unknown column, a repeated column,
// a row with another number of cells, and a cell that does not parse exactly are each an
// error naming the row and the column. It does not validate the records; call Validate.
func ReadRecords(r io.Reader) ([]Record, error) {
	cr := csv.NewReader(r)
	head, err := cr.Read()
	if err == io.EOF {
		return nil, errors.New("trace data: empty file; the first row names the columns")
	}
	if err != nil {
		return nil, fmt.Errorf("trace data: %w", err)
	}
	known := map[string]bool{}
	for _, c := range append(append([]string{}, Columns...), OptionalColumns...) {
		known[c] = true
	}
	at := map[string]int{}
	for i, c := range head {
		if !known[c] {
			return nil, fmt.Errorf("trace data: column %q is not a trace column (version %d)", c, Version)
		}
		if _, dup := at[c]; dup {
			return nil, fmt.Errorf("trace data: column %q appears twice", c)
		}
		at[c] = i
	}
	for _, c := range Columns {
		if _, ok := at[c]; !ok {
			return nil, fmt.Errorf("trace data: required column %q is missing", c)
		}
	}
	var out []Record
	for line := 2; ; line++ {
		row, err := cr.Read()
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return nil, fmt.Errorf("trace data: %w", err)
		}
		rec, err := parseRow(row, at)
		if err != nil {
			return nil, fmt.Errorf("trace data line %d: %w", line, err)
		}
		out = append(out, rec)
	}
}

// parseRow reads one row; at maps each present column to its cell index.
func parseRow(row []string, at map[string]int) (Record, error) {
	var r Record
	var err error
	cell := func(c string) (string, bool) {
		i, ok := at[c]
		if !ok {
			return "", false
		}
		return row[i], true
	}
	str := func(c string) string { s, _ := cell(c); return s }
	integer := func(c string, dst *int) {
		if err != nil {
			return
		}
		s, _ := cell(c)
		v, e := strconv.Atoi(s)
		if e != nil {
			err = fmt.Errorf("%s: %q is not an integer", c, s)
			return
		}
		*dst = v
	}
	int64Cell := func(c string, dst *int64) {
		if err != nil {
			return
		}
		s, ok := cell(c)
		if !ok {
			return // an optional column the file does not have
		}
		v, e := strconv.ParseInt(s, 10, 64)
		if e != nil {
			err = fmt.Errorf("%s: %q is not an integer", c, s)
			return
		}
		*dst = v
	}
	optional := func(c string) *int64 {
		if err != nil {
			return nil
		}
		s, ok := cell(c)
		if !ok || s == "" {
			return nil
		}
		v, e := strconv.ParseInt(s, 10, 64)
		if e != nil {
			err = fmt.Errorf("%s: %q is not an integer", c, s)
			return nil
		}
		return &v
	}
	r.ClientID, r.TenantID, r.SLOClass = str("client_id"), str("tenant_id"), str("slo_class")
	r.SessionID, r.PrefixGroup, r.Model = str("session_id"), str("prefix_group"), str("model")
	r.Status, r.Adapter = Status(str("status")), str("adapter")
	integer("request_id", &r.RequestID)
	integer("round_index", &r.RoundIndex)
	integer("prefix_length", &r.PrefixLength)
	integer("input_tokens", &r.InputTokens)
	integer("output_tokens", &r.OutputTokens)
	integer("text_tokens", &r.TextTokens)
	integer("image_tokens", &r.ImageTokens)
	integer("audio_tokens", &r.AudioTokens)
	integer("video_tokens", &r.VideoTokens)
	int64Cell("deadline_us", &r.DeadlineUs)
	int64Cell("arrival_time_us", &r.ArrivalTimeUs)
	int64Cell("first_chunk_time_us", &r.FirstChunkTimeUs)
	int64Cell("last_chunk_time_us", &r.LastChunkTimeUs)
	int64Cell("slo_target_us", &r.SLOTargetUs)
	r.ThinkTimeUs = optional("think_time_us")
	r.InputTokensReset = optional("input_tokens_reset")
	if err != nil {
		return r, err
	}
	switch s := str("streaming"); s {
	case "true":
		r.Streaming = true
	case "false":
	default:
		return r, fmt.Errorf("streaming: %q is not true or false", s)
	}
	s := str("reason_ratio")
	if r.ReasonRatio, err = strconv.ParseFloat(s, 64); err != nil {
		return r, fmt.Errorf("reason_ratio: %q is not a number", s)
	}
	return r, nil
}

// WriteRecords writes the data CSV: the header row, then one row per record. An optional
// column is written only when some record carries a value for it, so a trace that uses
// none has exactly Columns. It returns the writer's flush error.
func WriteRecords(w io.Writer, records []Record) error {
	has := map[string]bool{}
	for _, r := range records {
		has["slo_target_us"] = has["slo_target_us"] || r.SLOTargetUs != 0
		has["adapter"] = has["adapter"] || r.Adapter != ""
		has["think_time_us"] = has["think_time_us"] || r.ThinkTimeUs != nil
		has["input_tokens_reset"] = has["input_tokens_reset"] || r.InputTokensReset != nil
	}
	cols := append([]string{}, Columns...)
	for _, c := range OptionalColumns {
		if has[c] {
			cols = append(cols, c)
		}
	}
	cw := csv.NewWriter(w)
	if err := cw.Write(cols); err != nil {
		return fmt.Errorf("trace data: %w", err)
	}
	ptr := func(v *int64) string {
		if v == nil {
			return ""
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
			string(r.Status),
		}
		for _, c := range OptionalColumns {
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
		if err := cw.Write(row); err != nil {
			return fmt.Errorf("trace data: request %d: %w", r.RequestID, err)
		}
	}
	cw.Flush()
	if err := cw.Error(); err != nil {
		return fmt.Errorf("trace data: %w", err)
	}
	return nil
}

// Read reads a header and its data and validates them together.
func Read(header, data io.Reader) (*Trace, error) {
	h, err := ReadHeader(header)
	if err != nil {
		return nil, err
	}
	recs, err := ReadRecords(data)
	if err != nil {
		return nil, err
	}
	t := &Trace{Header: *h, Records: recs}
	if p := t.Validate(); !p.OK() {
		return nil, fmt.Errorf("trace:\n%s", p.Error())
	}
	return t, nil
}
