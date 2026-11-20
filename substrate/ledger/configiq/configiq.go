// Package configiq turns the rows of Red Hat's configiq dataset into ledger units (design
// §10.2).
//
// The dataset (redhat-performance/configiq-performance-upstream-data on Hugging Face) is one
// Parquet file per model and accelerator pair, each row an aggregate GuideLLM measurement of
// one configuration at one concurrency. It records no run id, timestamp, command line or
// per-request data, so a unit is a configuration at one concurrency and holds every
// measurement of it, and its values are aggregates only.
//
// The input is the downloaded revision (the files sources.yaml pins) and the rows export.py
// printed from it, one JSON object per row, in file and then row order. The output is the unit
// list and the ingest report, a function of those bytes alone, so CI can regenerate them and
// require the committed files to match byte for byte (D4). Only identifying fields and hashes
// are committed; no measured value is (X12).
package configiq

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-substrate/ledger"
)

// SourceID is the source's identifier in the ledger and in every unit's identity.
const SourceID = "configiq"

// IdentityColumns are the dataset's columns that are not measured values, in its own names.
// With the source they are a unit's identifying fields (design §10.2).
var IdentityColumns = []string{
	"pair_id", "model", "model_id", "accelerator", "backend", "version", "backend_version",
	"tp", "isl", "osl", "concurrency", "turns", "prefix_tokens", "prefix_count",
	"prefix_caching", "request_type", "spec_decoding", "gpu_memory_utilization",
	"max_num_seqs", "max_seq_len", "chunked_prefill_enabled", "precision",
	"weight_quantization", "kv_quantization",
}

// MeasuredColumns are the dataset's measured values: aggregates over a run's requests. TTFT,
// TPOT and ITL are in milliseconds and request latency in seconds. They follow GuideLLM's
// definitions: tpot is (last token - request start) / output tokens, so it includes the first
// token; itl is (last token - first token) / (output tokens - 1), per request.
var MeasuredColumns = []string{
	"measured concurrency", "measured rps", "output_tok/sec", "total_tok/sec",
	"prompt_token_count_mean", "prompt_token_count_p99", "output_token_count_mean",
	"output_token_count_p99",
	"ttft_mean", "ttft_median", "ttft_p1", "ttft_p95", "ttft_p99", "ttft_p999",
	"tpot_median", "tpot_p1", "tpot_p95", "tpot_p99", "tpot_p999",
	"itl_mean", "itl_median", "itl_p1", "itl_p95", "itl_p99", "itl_p999",
	"request_latency_median", "request_latency_min", "request_latency_max",
	"successful_requests", "errored_requests",
}

// RoleKeyOmits is the one identifying field the role key leaves out, so a sweep and its
// re-measurements share one role.
const RoleKeyOmits = "concurrency"

// LeftOutNoSuccess is the reason a row with no successful request is not a measurement: its
// latencies are zeros standing in for values, not measured values (design §10.2).
const LeftOutNoSuccess = "no_successful_requests"

// Row is one exported dataset row: its file under the revision, its index in that file, and
// every column, numbers kept as printed.
type Row struct {
	File    string                     `json:"file"`
	Row     int64                      `json:"row"`
	Columns map[string]json.RawMessage `json:"columns"`
}

// Record is the row's source row id: the file, then the row, as "path:row".
func (r *Row) Record() string { return r.File + ":" + strconv.FormatInt(r.Row, 10) }

// ReadRows reads exported rows, one JSON object per line, and requires them in file and then
// row order, each file's rows numbered from 0, with exactly the dataset's columns (SUB-12).
func ReadRows(rd io.Reader) ([]Row, error) {
	known := map[string]bool{}
	for _, c := range append(append([]string{}, IdentityColumns...), MeasuredColumns...) {
		known[c] = true
	}
	var rows []Row
	sc := bufio.NewScanner(rd)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	n := 0
	for sc.Scan() {
		n++
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()
		var row Row
		if err := dec.Decode(&row); err != nil {
			return nil, fmt.Errorf("line %d: %w", n, err)
		}
		if row.File == "" {
			return nil, fmt.Errorf("line %d: no file", n)
		}
		for c := range row.Columns {
			if !known[c] {
				return nil, fmt.Errorf("line %d (%s): column %q is not one the ingest tool knows; a new column is a new contract", n, row.Record(), c)
			}
		}
		if len(row.Columns) != len(known) {
			return nil, fmt.Errorf("line %d (%s): %d columns, want the %d of the dataset", n, row.Record(), len(row.Columns), len(known))
		}
		if len(rows) > 0 {
			prev := rows[len(rows)-1]
			switch {
			case row.File == prev.File && row.Row != prev.Row+1:
				return nil, fmt.Errorf("line %d: row %s follows %s; rows of a file are numbered from 0 in order", n, row.Record(), prev.Record())
			case row.File != prev.File && (row.File < prev.File || row.Row != 0):
				return nil, fmt.Errorf("line %d: row %s follows %s; export.py writes files in path order, each from row 0", n, row.Record(), prev.Record())
			}
		} else if row.Row != 0 {
			return nil, fmt.Errorf("line %d: the first row is %s, want row 0", n, row.Record())
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, errors.New("no rows: an empty export is not a source revision")
	}
	return rows, nil
}

// identityValue puts one identifying column in canonical form: null stays null, a string and a
// boolean are kept as written, and a number with no fraction is an integer. A number with a
// fraction is refused: no identifying column holds one, and two encoders format floats
// differently.
func identityValue(raw json.RawMessage) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	switch x := v.(type) {
	case nil, string, bool:
		return x, nil
	case json.Number:
		f, err := x.Float64()
		if err != nil || math.IsInf(f, 0) || f != math.Trunc(f) || math.Abs(f) > 1<<53 {
			return nil, fmt.Errorf("%s is not a whole number", x)
		}
		return int64(f), nil
	default:
		return nil, fmt.Errorf("a %T is not an identifying value", v)
	}
}

// Identity returns the row's identifying fields: the source and every IdentityColumns value in
// canonical form.
func (r *Row) Identity() (map[string]any, error) {
	id := map[string]any{"source": SourceID}
	for _, c := range IdentityColumns {
		v, err := identityValue(r.Columns[c])
		if err != nil {
			return nil, fmt.Errorf("row %s: column %q: %w", r.Record(), c, err)
		}
		id[c] = v
	}
	return id, nil
}

// RoleKey returns the role key of a configiq identity: every identifying field except the
// concurrency. It refuses an identity that lacks a field or holds another.
func RoleKey(identity map[string]any) (map[string]any, error) {
	if len(identity) != len(IdentityColumns)+1 {
		return nil, fmt.Errorf("identity has %d fields, want source and the %d of %v", len(identity), len(IdentityColumns), IdentityColumns)
	}
	if identity["source"] != SourceID {
		return nil, fmt.Errorf("identity.source: %v, want %q", identity["source"], SourceID)
	}
	rk := map[string]any{"source": SourceID}
	for _, c := range IdentityColumns {
		v, ok := identity[c]
		if !ok {
			return nil, fmt.Errorf("identity lacks %q", c)
		}
		if c != RoleKeyOmits {
			rk[c] = v
		}
	}
	return rk, nil
}

// measuredValues returns the row's measured columns, numbers kept as the export printed them.
func (r *Row) measuredValues() (map[string]any, error) {
	out := map[string]any{}
	for _, c := range MeasuredColumns {
		dec := json.NewDecoder(bytes.NewReader(r.Columns[c]))
		dec.UseNumber()
		var v any
		if err := dec.Decode(&v); err != nil {
			return nil, fmt.Errorf("row %s: column %q: %w", r.Record(), c, err)
		}
		out[c] = v
	}
	return out, nil
}

// notMeasurement returns why a row is not a measurement, or "" when it is one.
func (r *Row) notMeasurement() string {
	var v any
	dec := json.NewDecoder(bytes.NewReader(r.Columns["successful_requests"]))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return LeftOutNoSuccess
	}
	n, ok := v.(json.Number)
	if !ok {
		return LeftOutNoSuccess
	}
	if f, err := n.Float64(); err != nil || !(f > 0) {
		return LeftOutNoSuccess
	}
	return ""
}

// Report is the ingest report for one source revision (design §10.2). Its keys are those of
// the InferenceX report; units_by_framework counts units per backend.
type Report struct {
	Kind     string `yaml:"kind"` // "IngestReport"
	Source   string `yaml:"source"`
	Revision string `yaml:"revision"`
	// Input is the SHA-256 of the exported rows the report was generated from.
	Input         string              `yaml:"input"`
	Records       int                 `yaml:"records"`
	Units         int                 `yaml:"units"`
	RoleKeys      int                 `yaml:"role_keys"`
	Duplicates    int                 `yaml:"duplicates"`
	Merged        int                 `yaml:"merged"`
	JoinedUnits   int                 `yaml:"joined_units"`
	JoinedRecords int                 `yaml:"joined_records"`
	LeftOut       map[string][]string `yaml:"left_out"`
	ByFramework   map[string]int      `yaml:"units_by_framework"`
}

// recordLess orders source row ids: by file path, then by row number.
func recordLess(a, b string) bool {
	ai, bi := strings.LastIndexByte(a, ':'), strings.LastIndexByte(b, ':')
	if a[:ai] != b[:bi] {
		return a[:ai] < b[:bi]
	}
	x, _ := strconv.ParseInt(a[ai+1:], 10, 64)
	y, _ := strconv.ParseInt(b[bi+1:], 10, 64)
	return x < y
}

// Ingest turns rows into units and the ingest report, by the rules of design §10.2. Records
// with the same identity and measured values are one measurement, kept under the first row
// and listing the others as duplicates. Records with the same identity and different measured
// values are one unit holding each measurement. The same measured values under two identities
// are one unit when the identities share a role key, under the first row's identity, and are
// left out when they do not. revision names the pinned revision and input the SHA-256 of the
// export.
func Ingest(rows []Row, revision, input string) ([]ledger.Unit, *Report, error) {
	rep := &Report{Kind: "IngestReport", Source: SourceID, Revision: revision, Input: input,
		Records: len(rows), LeftOut: map[string][]string{}, ByFramework: map[string]int{}}
	type rec struct {
		row      *Row
		identity map[string]any
		unitID   string
		roleKey  string
		measured string
	}
	var recs []*rec
	for i := range rows {
		r := &rows[i]
		if why := r.notMeasurement(); why != "" {
			rep.LeftOut[why] = append(rep.LeftOut[why], r.Record())
			continue
		}
		id, err := r.Identity()
		if err != nil {
			return nil, nil, err
		}
		uid, err := ledger.ID(id)
		if err != nil {
			return nil, nil, fmt.Errorf("row %s: %w", r.Record(), err)
		}
		rk, err := RoleKey(id)
		if err != nil {
			return nil, nil, fmt.Errorf("row %s: %w", r.Record(), err)
		}
		rkID, err := ledger.ID(rk)
		if err != nil {
			return nil, nil, fmt.Errorf("row %s: %w", r.Record(), err)
		}
		mv, err := r.measuredValues()
		if err != nil {
			return nil, nil, err
		}
		m, err := ledger.MeasuredID(mv)
		if err != nil {
			return nil, nil, fmt.Errorf("row %s: %w", r.Record(), err)
		}
		recs = append(recs, &rec{row: r, identity: id, unitID: uid, roleKey: rkID, measured: m})
	}

	byMeasured := map[string][]*rec{}
	var measuredOrder []string
	for _, x := range recs {
		if _, ok := byMeasured[x.measured]; !ok {
			measuredOrder = append(measuredOrder, x.measured)
		}
		byMeasured[x.measured] = append(byMeasured[x.measured], x)
	}
	target := map[*rec]*rec{}
	leave := map[*rec]bool{}
	for _, mk := range measuredOrder {
		group := byMeasured[mk]
		ids, keys := map[string]bool{}, map[string]bool{}
		for _, x := range group {
			ids[x.unitID] = true
			keys[x.roleKey] = true
		}
		if len(ids) < 2 {
			continue
		}
		if len(keys) > 1 {
			for _, x := range group {
				leave[x] = true
			}
			continue
		}
		for _, x := range group[1:] {
			if x.unitID != group[0].unitID {
				target[x] = group[0]
			}
		}
	}

	type meas struct {
		kept               *rec
		duplicates, merged []*rec
	}
	type unit struct {
		first *rec
		meas  []*meas
		byM   map[string]*meas
	}
	units := map[string]*unit{}
	var order []string
	for _, x := range recs {
		if leave[x] {
			rep.LeftOut["same_measurement_two_role_keys"] = append(rep.LeftOut["same_measurement_two_role_keys"], x.row.Record())
			continue
		}
		owner := x
		if t, ok := target[x]; ok {
			owner = t
		}
		u := units[owner.unitID]
		if u == nil {
			u = &unit{first: owner, byM: map[string]*meas{}}
			units[owner.unitID] = u
			order = append(order, owner.unitID)
		}
		m := u.byM[x.measured]
		switch {
		case m == nil:
			m = &meas{kept: x}
			u.byM[x.measured] = m
			u.meas = append(u.meas, m)
		case owner != x:
			m.merged = append(m.merged, x)
			rep.Merged++
		default:
			m.duplicates = append(m.duplicates, x)
			rep.Duplicates++
		}
	}

	out := make([]ledger.Unit, 0, len(units))
	roleKeys := map[string]bool{}
	for _, uid := range order {
		u := units[uid]
		lu := ledger.Unit{UnitID: uid, RoleKey: u.first.roleKey, Identity: u.first.identity}
		n := 0
		for _, m := range u.meas {
			lm := ledger.Measurement{Record: m.kept.row.Record(), Measured: m.kept.measured,
				Duplicates: []string{}, Merged: []ledger.MergedRecord{}, Recorded: map[string]any{}}
			for _, d := range m.duplicates {
				lm.Duplicates = append(lm.Duplicates, d.row.Record())
			}
			for _, d := range m.merged {
				lm.Merged = append(lm.Merged, ledger.MergedRecord{Record: d.row.Record(), UnitID: d.unitID})
			}
			n += 1 + len(m.duplicates) + len(m.merged)
			lu.Measurements = append(lu.Measurements, lm)
		}
		sort.Slice(lu.Measurements, func(i, j int) bool { return recordLess(lu.Measurements[i].Record, lu.Measurements[j].Record) })
		if len(lu.Measurements) > 1 {
			rep.JoinedUnits++
			rep.JoinedRecords += n
		}
		roleKeys[lu.RoleKey] = true
		rep.ByFramework[fmt.Sprint(u.first.identity["backend"])]++
		out = append(out, lu)
	}
	rep.Units = len(out)
	rep.RoleKeys = len(roleKeys)
	sort.Slice(out, func(i, j int) bool { return out[i].UnitID < out[j].UnitID })
	left := 0
	for k := range rep.LeftOut {
		sort.Slice(rep.LeftOut[k], func(i, j int) bool { return recordLess(rep.LeftOut[k][i], rep.LeftOut[k][j]) })
		left += len(rep.LeftOut[k])
	}
	kept := 0
	for _, u := range out {
		kept += len(u.Measurements)
	}
	if got := rep.Duplicates + rep.Merged + left + kept; got != rep.Records {
		return nil, nil, fmt.Errorf("ingest accounted for %d records of %d", got, rep.Records)
	}
	if err := ledger.CheckMeasurementsUnique(out); err != nil {
		return nil, nil, err
	}
	return out, rep, nil
}

// Manifest is the part of the dataset's manifest.json the ingest tool checks the export
// against: each pair's file, its SHA-256 and its row count.
type Manifest struct {
	Pairs []struct {
		Path       string `json:"path"`
		FileSHA256 string `json:"file_sha256"`
		Records    int    `json:"records"`
	} `json:"pairs"`
}

// CheckRevision checks a downloaded revision and its export against the source's pins (D3):
// every file sources.yaml pins is in dir with the pinned SHA-256, every Parquet file the
// manifest lists is pinned with the manifest's SHA-256, and the export holds exactly each
// file's rows. A file that differs, or an export of another revision, is refused.
func CheckRevision(src *ledger.Source, dir string, rows []Row) error {
	if src == nil || src.ID != SourceID {
		return fmt.Errorf("not the %s source", SourceID)
	}
	pins := map[string]string{}
	var errs []error
	for _, a := range src.Revision.Assets {
		pins[a.Name] = a.SHA256
		b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(a.Name)))
		if err != nil {
			errs = append(errs, err)
			continue
		}
		sum := sha256.Sum256(b)
		if got := hex.EncodeToString(sum[:]); got != a.SHA256 {
			errs = append(errs, fmt.Errorf("%s: SHA-256 %s, but sources.yaml pins %s", a.Name, got, a.SHA256))
		}
	}
	if err := errors.Join(errs...); err != nil {
		return err
	}
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return fmt.Errorf("manifest.json: %w; pin it in sources.yaml", err)
	}
	if _, ok := pins["manifest.json"]; !ok {
		return errors.New("sources.yaml does not pin manifest.json")
	}
	var man Manifest
	if err := json.Unmarshal(raw, &man); err != nil {
		return fmt.Errorf("manifest.json: %w", err)
	}
	count := map[string]int{}
	for _, r := range rows {
		count[r.File]++
	}
	listed := map[string]bool{}
	for _, p := range man.Pairs {
		listed[p.Path] = true
		if pins[p.Path] != p.FileSHA256 {
			errs = append(errs, fmt.Errorf("%s: the manifest gives SHA-256 %s, sources.yaml pins %q", p.Path, p.FileSHA256, pins[p.Path]))
		}
		if count[p.Path] != p.Records {
			errs = append(errs, fmt.Errorf("%s: the export has %d rows, the manifest %d", p.Path, count[p.Path], p.Records))
		}
	}
	files := make([]string, 0, len(count))
	for f := range count {
		files = append(files, f)
	}
	sort.Strings(files)
	for _, f := range files {
		if !listed[f] {
			errs = append(errs, fmt.Errorf("%s: in the export but not in the manifest", f))
		}
	}
	return errors.Join(errs...)
}
