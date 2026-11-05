// Package inferencex turns InferenceX benchmark rows into ledger units (design §10.2).
//
// The input is the pinned database dump, restored by restore.sh and exported by export.sql
// as one JSON object per benchmark_results row, in row-id order. The output is the unit list
// and the ingest report. Both are a function of the input bytes alone, so CI can regenerate
// them and require the committed files to match byte for byte (D4).
package inferencex

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-substrate/ledger"
)

// SourceID is the source's identifier in the ledger and in every unit's identity.
const SourceID = "inferencex"

// ConfigurationFields are the 17 fields of a record's serving configuration whose canonical
// JSON the configuration hash covers, in InferenceX's own names (design §10.2).
var ConfigurationFields = []string{
	"image", "isl", "osl", "offload_mode", "spec_method", "disagg", "is_multinode",
	"prefill_tp", "prefill_ep", "prefill_dp_attention", "prefill_num_workers", "num_prefill_gpu",
	"decode_tp", "decode_ep", "decode_dp_attention", "decode_num_workers", "num_decode_gpu",
}

// IdentityFields are a unit's identifying fields: the source, then the ten InferenceX fields.
var IdentityFields = []string{
	"source", "model", "hardware", "precision", "framework", "benchmark_type",
	"configuration_hash", "recipe_fingerprint", "conc", "measured_date", "benchmark_run",
}

// RoleKeyFields are the identifying fields the role key keeps. It leaves out the recipe
// fingerprint, the concurrency, the measured date and the benchmark run, so a deployment's
// whole sweep and its re-measurements share one role.
var RoleKeyFields = []string{
	"source", "model", "hardware", "precision", "framework", "benchmark_type", "configuration_hash",
}

// Row is one exported benchmark_results row. Fields are decoded strictly: an export with a
// field this type does not name is refused (SUB-12).
type Row struct {
	ID             int64           `json:"id"`
	GitHubRunID    *int64          `json:"github_run_id"`
	RunURL         *string         `json:"run_url"`
	RunAttempt     int64           `json:"run_attempt"`
	RunConclusion  *string         `json:"run_conclusion"`
	ConfigID       int64           `json:"config_id"`
	Model          string          `json:"model"`
	Hardware       string          `json:"hardware"`
	Precision      string          `json:"precision"`
	Framework      string          `json:"framework"`
	BenchmarkType  string          `json:"benchmark_type"`
	Conc           int64           `json:"conc"`
	Error          *string         `json:"error"`
	Date           string          `json:"date"`
	RecipeFP       *string         `json:"recipe_fingerprint"`
	Image          *string         `json:"image"`
	ISL            *int64          `json:"isl"`
	OSL            *int64          `json:"osl"`
	OffloadMode    *string         `json:"offload_mode"`
	SpecMethod     *string         `json:"spec_method"`
	Disagg         *bool           `json:"disagg"`
	IsMultinode    *bool           `json:"is_multinode"`
	PrefillTP      *int64          `json:"prefill_tp"`
	PrefillEP      *int64          `json:"prefill_ep"`
	PrefillDPAttn  *bool           `json:"prefill_dp_attention"`
	PrefillWorkers *int64          `json:"prefill_num_workers"`
	NumPrefillGPU  *int64          `json:"num_prefill_gpu"`
	DecodeTP       *int64          `json:"decode_tp"`
	DecodeEP       *int64          `json:"decode_ep"`
	DecodeDPAttn   *bool           `json:"decode_dp_attention"`
	DecodeWorkers  *int64          `json:"decode_num_workers"`
	NumDecodeGPU   *int64          `json:"num_decode_gpu"`
	Metrics        json.RawMessage `json:"metrics"`
}

// ReadRows reads exported rows, one JSON object per line, and requires them in increasing
// row-id order: the order the ingest rules name ("the smallest source row id").
func ReadRows(r io.Reader) ([]Row, error) {
	var rows []Row
	sc := bufio.NewScanner(r)
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
		if len(rows) > 0 && row.ID <= rows[len(rows)-1].ID {
			return nil, fmt.Errorf("line %d: row %d is not after row %d; export.sql orders by id", n, row.ID, rows[len(rows)-1].ID)
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

var runID = regexp.MustCompile(`/actions/runs/(\d+)`)

// BenchmarkRun returns the numeric GitHub Actions run id in a run_url, without the attempt,
// or nil when the record has no run_url or the URL names no run.
func BenchmarkRun(runURL *string) *int64 {
	if runURL == nil {
		return nil
	}
	m := runID.FindStringSubmatch(*runURL)
	if m == nil {
		return nil
	}
	n, err := strconv.ParseInt(m[1], 10, 64)
	if err != nil {
		return nil
	}
	return &n
}

// SplitImage removes a suffix beginning "@sha256:" from an image reference and returns the
// image without it and the digest (nil when there is none).
func SplitImage(image *string) (*string, *string) {
	if image == nil {
		return nil, nil
	}
	base, digest, found := strings.Cut(*image, "@sha256:")
	if !found {
		return image, nil
	}
	d := "sha256:" + digest
	return &base, &d
}

func ptr[T any](p *T) any {
	if p == nil {
		return nil
	}
	return *p
}

// Configuration returns the 17-field object the configuration hash covers. Every key is
// present; a value the record lacks is null.
func (r *Row) Configuration() map[string]any {
	image, _ := SplitImage(r.Image)
	return map[string]any{
		"image": ptr(image), "isl": ptr(r.ISL), "osl": ptr(r.OSL),
		"offload_mode": ptr(r.OffloadMode), "spec_method": ptr(r.SpecMethod),
		"disagg": ptr(r.Disagg), "is_multinode": ptr(r.IsMultinode),
		"prefill_tp": ptr(r.PrefillTP), "prefill_ep": ptr(r.PrefillEP),
		"prefill_dp_attention": ptr(r.PrefillDPAttn), "prefill_num_workers": ptr(r.PrefillWorkers),
		"num_prefill_gpu": ptr(r.NumPrefillGPU),
		"decode_tp":       ptr(r.DecodeTP), "decode_ep": ptr(r.DecodeEP),
		"decode_dp_attention": ptr(r.DecodeDPAttn), "decode_num_workers": ptr(r.DecodeWorkers),
		"num_decode_gpu": ptr(r.NumDecodeGPU),
	}
}

// Identity returns the record's identifying fields (design §10.2), with the record's own date
// as the measured date. Ingest replaces it with the earliest date of the record's benchmark
// run.
func (r *Row) Identity() (map[string]any, error) {
	cfg, err := ledger.ID(r.Configuration())
	if err != nil {
		return nil, fmt.Errorf("row %d: configuration: %w", r.ID, err)
	}
	return map[string]any{
		"source": SourceID, "model": r.Model, "hardware": r.Hardware, "precision": r.Precision,
		"framework": r.Framework, "benchmark_type": r.BenchmarkType, "configuration_hash": cfg,
		"recipe_fingerprint": ptr(r.RecipeFP), "conc": r.Conc, "measured_date": r.Date,
		"benchmark_run": ptr(BenchmarkRun(r.RunURL)),
	}, nil
}

// RoleKey returns the role key of an InferenceX identity: its RoleKeyFields. It refuses an
// identity that lacks one of IdentityFields or holds another field.
func RoleKey(identity map[string]any) (map[string]any, error) {
	if len(identity) != len(IdentityFields) {
		return nil, fmt.Errorf("identity has %d fields, want the %d of %v", len(identity), len(IdentityFields), IdentityFields)
	}
	for _, f := range IdentityFields {
		if _, ok := identity[f]; !ok {
			return nil, fmt.Errorf("identity lacks %q", f)
		}
	}
	if identity["source"] != SourceID {
		return nil, fmt.Errorf("identity.source: %v, want %q", identity["source"], SourceID)
	}
	rk := map[string]any{}
	for _, f := range RoleKeyFields {
		rk[f] = identity[f]
	}
	return rk, nil
}

// notMeasurement returns why a row is not a measurement, or "" when it is one: it reports an
// error, or its metrics are empty (design §10.2).
func (r *Row) notMeasurement() string {
	if r.Error != nil {
		return "error_reported"
	}
	t := bytes.TrimSpace(r.Metrics)
	if len(t) == 0 || bytes.Equal(t, []byte("null")) || bytes.Equal(t, []byte("{}")) {
		return "empty_metrics"
	}
	return ""
}

// measured returns the SHA-256 of the canonical JSON of the row's metrics object.
func (r *Row) measured() (string, error) {
	dec := json.NewDecoder(bytes.NewReader(r.Metrics))
	dec.UseNumber()
	var m any
	if err := dec.Decode(&m); err != nil {
		return "", fmt.Errorf("row %d: metrics: %w", r.ID, err)
	}
	if _, ok := m.(map[string]any); !ok {
		return "", fmt.Errorf("row %d: metrics is a %T, not an object", r.ID, m)
	}
	return ledger.MeasuredID(m)
}

// recorded returns the non-identifying fields a unit carries for a record.
func (r *Row) recorded() map[string]any {
	_, digest := SplitImage(r.Image)
	return map[string]any{
		"image_digest":   ptr(digest),
		"run_attempt":    r.RunAttempt,
		"run_conclusion": ptr(r.RunConclusion),
	}
}

// Report is the ingest report for one source revision (design §10.2): every record read and
// what became of it.
type Report struct {
	Kind     string `yaml:"kind"` // "IngestReport"
	Source   string `yaml:"source"`
	Revision string `yaml:"revision"`
	// Input is the SHA-256 of the exported rows the report was generated from.
	Input   string `yaml:"input"`
	Records int    `yaml:"records"`
	Units   int    `yaml:"units"`
	// RoleKeys counts distinct role keys among the units.
	RoleKeys int `yaml:"role_keys"`
	// Duplicates counts records merged into a unit as a repeat of another record's identity
	// and measured values.
	Duplicates int `yaml:"duplicates"`
	// Merged counts records folded into a unit under another identity with the same role key.
	Merged int `yaml:"merged"`
	// Joined counts the units that hold more than one measurement, and the records in them.
	JoinedUnits   int `yaml:"joined_units"`
	JoinedRecords int `yaml:"joined_records"`
	// LeftOut lists every record left out, by reason.
	LeftOut map[string][]int64 `yaml:"left_out"`
	// ByFramework counts units per framework, so the vLLM share is visible.
	ByFramework map[string]int `yaml:"units_by_framework"`
}

// Ingest turns rows into units and the ingest report. revision names the pinned source
// revision and input the SHA-256 of the exported rows.
func Ingest(rows []Row, revision, input string) ([]ledger.Unit, *Report, error) {
	rep := &Report{Kind: "IngestReport", Source: SourceID, Revision: revision, Input: input,
		Records: len(rows), LeftOut: map[string][]int64{}, ByFramework: map[string]int{}}

	type rec struct {
		row      *Row
		identity map[string]any
		unitID   string
		roleKey  string
		measured string
	}
	// The measured date of a record with a run id is the earliest date among all records of
	// its benchmark run, so every attempt of a run has one date; a record with no run id keeps
	// its own date (design §10.2). Every record counts, measurements or not.
	runDate := map[int64]string{}
	for i := range rows {
		if run := BenchmarkRun(rows[i].RunURL); run != nil {
			if d, ok := runDate[*run]; !ok || rows[i].Date < d {
				runDate[*run] = rows[i].Date
			}
		}
	}
	var recs []*rec
	for i := range rows {
		r := &rows[i]
		if why := r.notMeasurement(); why != "" {
			rep.LeftOut[why] = append(rep.LeftOut[why], r.ID)
			continue
		}
		id, err := r.Identity()
		if err != nil {
			return nil, nil, err
		}
		if run := BenchmarkRun(r.RunURL); run != nil {
			id["measured_date"] = runDate[*run]
		}
		uid, err := ledger.ID(id)
		if err != nil {
			return nil, nil, fmt.Errorf("row %d: %w", r.ID, err)
		}
		rk, _ := RoleKey(id)
		rkID, err := ledger.ID(rk)
		if err != nil {
			return nil, nil, fmt.Errorf("row %d: %w", r.ID, err)
		}
		m, err := r.measured()
		if err != nil {
			return nil, nil, err
		}
		recs = append(recs, &rec{row: r, identity: id, unitID: uid, roleKey: rkID, measured: m})
	}

	// The same measured values under two identities. With one role key they are one unit,
	// under the identity of the record with the smallest row id; with two role keys they are
	// left out, because one measurement could otherwise draw two roles.
	byMeasured := map[string][]*rec{}
	for _, x := range recs {
		byMeasured[x.measured] = append(byMeasured[x.measured], x)
	}
	target := map[*rec]*rec{} // record -> record whose identity it takes
	leave := map[*rec]bool{}
	for _, group := range byMeasured {
		ids := map[string]bool{}
		keys := map[string]bool{}
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
		first := group[0] // recs are in row-id order
		for _, x := range group[1:] {
			if x.unitID != first.unitID {
				target[x] = first
			}
		}
	}

	type meas struct {
		kept       *rec
		duplicates []*rec
		merged     []*rec
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
			rep.LeftOut["same_measurement_two_role_keys"] = append(rep.LeftOut["same_measurement_two_role_keys"], x.row.ID)
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
			fields := m.kept.row.recorded()
			// A field the kept record lacks comes from the duplicate with the smallest row id
			// that records it. Duplicates and merged records are each in row-id order.
			others := append(append([]*rec{}, m.duplicates...), m.merged...)
			sort.Slice(others, func(i, j int) bool { return others[i].row.ID < others[j].row.ID })
			for _, d := range others {
				for k, v := range d.row.recorded() {
					if fields[k] == nil && v != nil {
						fields[k] = v
					}
				}
			}
			lm := ledger.Measurement{Record: strconv.FormatInt(m.kept.row.ID, 10), Measured: m.kept.measured,
				Duplicates: []string{}, Merged: []ledger.MergedRecord{}, Recorded: fields}
			for _, d := range m.duplicates {
				lm.Duplicates = append(lm.Duplicates, strconv.FormatInt(d.row.ID, 10))
			}
			for _, d := range m.merged {
				lm.Merged = append(lm.Merged, ledger.MergedRecord{Record: strconv.FormatInt(d.row.ID, 10), UnitID: d.unitID})
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
		rep.ByFramework[u.first.row.Framework]++
		out = append(out, lu)
	}
	rep.Units = len(out)
	rep.RoleKeys = len(roleKeys)
	sort.Slice(out, func(i, j int) bool { return out[i].UnitID < out[j].UnitID })
	for k := range rep.LeftOut {
		sort.Slice(rep.LeftOut[k], func(i, j int) bool { return rep.LeftOut[k][i] < rep.LeftOut[k][j] })
	}
	left := 0
	for _, v := range rep.LeftOut {
		left += len(v)
	}
	if got := rep.Duplicates + rep.Merged + left + countMeasurements(out); got != rep.Records {
		return nil, nil, fmt.Errorf("ingest accounted for %d records of %d", got, rep.Records)
	}
	if err := ledger.CheckMeasurementsUnique(out); err != nil {
		return nil, nil, err
	}
	return out, rep, nil
}

func countMeasurements(units []ledger.Unit) int {
	n := 0
	for _, u := range units {
		n += len(u.Measurements)
	}
	return n
}

func recordLess(a, b string) bool {
	x, _ := strconv.ParseInt(a, 10, 64)
	y, _ := strconv.ParseInt(b, 10, 64)
	return x < y
}

// MeasuredValues returns each unit's measured values for the seed-holding jobs: the mean
// over the unit's measurements of each metric, and its lane, the hardware. It runs from main
// inside those jobs only; the values are never committed.
func MeasuredValues(rows []Row, units []ledger.Unit) (*ledger.Measured, error) {
	byID := map[string]*Row{}
	for i := range rows {
		byID[strconv.FormatInt(rows[i].ID, 10)] = &rows[i]
	}
	out := &ledger.Measured{Values: map[string]map[string]float64{}, Lanes: map[string]string{}}
	for _, u := range units {
		k := SourceID + "/" + u.UnitID
		sums := map[string]float64{}
		n := map[string]int{}
		for _, m := range u.Measurements {
			r, ok := byID[m.Record]
			if !ok {
				return nil, fmt.Errorf("unit %s: record %s is not in the rows; regenerate the unit list from this export", u.UnitID, m.Record)
			}
			var metrics map[string]any
			if err := json.Unmarshal(r.Metrics, &metrics); err != nil {
				return nil, fmt.Errorf("row %d: metrics: %w", r.ID, err)
			}
			for name, v := range metrics {
				if f, ok := v.(float64); ok {
					sums[name] += f
					n[name]++
				}
			}
		}
		vals := map[string]float64{}
		for name, s := range sums {
			vals[name] = s / float64(n[name])
		}
		out.Values[k] = vals
		out.Lanes[k] = fmt.Sprint(u.Identity["hardware"])
	}
	return out, nil
}
