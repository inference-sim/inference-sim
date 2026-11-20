package inferencex

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/ledger"
)

// A supplement is a set of records InferenceX published after the pinned dump (design §10.2:
// "turning each data source's records into ledger units, for every source"). The records are
// pinned in sources.yaml by id and by the SHA-256 of each record's content. They are fetched
// from InferenceX's API when the unit list is generated, checked against their pins, turned
// into the same Row an exported dump row is, and ingested with the dump's rows, so every rule
// of the ingest (identity, role key, duplicates, joins) applies to them unchanged. None of
// their values is committed (X12): the unit list holds identifying fields and the hash of
// each record's metrics, as it does for the dump.

// Fetcher returns the body served at url. A gzip body is decoded by the caller.
type Fetcher func(url string) ([]byte, error)

// HTTPFetcher fetches over HTTPS with a bounded retry. A status other than 200 is an error:
// a supplement is never partly read.
func HTTPFetcher() Fetcher {
	client := &http.Client{Timeout: 3 * time.Minute}
	return func(url string) ([]byte, error) {
		var last error
		for attempt := 0; attempt < 3; attempt++ {
			if attempt > 0 {
				time.Sleep(time.Duration(attempt) * 2 * time.Second)
			}
			req, err := http.NewRequest(http.MethodGet, url, nil)
			if err != nil {
				return nil, err
			}
			req.Header.Set("Accept-Encoding", "gzip")
			req.Header.Set("User-Agent", "blis-substrate-ledger")
			resp, err := client.Do(req)
			if err != nil {
				last = err
				continue
			}
			body, err := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if err != nil {
				last = err
				continue
			}
			if resp.StatusCode != http.StatusOK {
				last = fmt.Errorf("%s: status %d", url, resp.StatusCode)
				continue
			}
			return body, nil
		}
		return nil, last
	}
}

// SupplementData is one supplement after fetching and checking: the rows to ingest and the
// records exactly as the source served them, in pinned order.
type SupplementData struct {
	ID      string
	Rows    []Row
	Records []json.RawMessage
}

// FetchSupplement reads the records a supplement pins from the API, refuses any whose content
// differs from its pin or that the API no longer serves, and returns them as rows. Records
// the API serves that the supplement does not pin are not read.
func FetchSupplement(s ledger.Supplement, fetch Fetcher) (*SupplementData, error) {
	byID, err := fetchServed(s, fetch)
	if err != nil {
		return nil, err
	}
	out := &SupplementData{ID: s.ID}
	var errs []error
	for _, pin := range s.Records {
		raw, ok := byID[pin.ID]
		if !ok {
			errs = append(errs, fmt.Errorf("record %d is pinned but %s no longer serves it", pin.ID, s.URL))
			continue
		}
		got, err := RecordSHA256(raw)
		if err != nil {
			errs = append(errs, fmt.Errorf("record %d: %w", pin.ID, err))
			continue
		}
		if got != pin.SHA256 {
			errs = append(errs, fmt.Errorf("record %d: content hash %s, pinned %s; the source changed the record, so a ledger owner repins it in a pull request of its own", pin.ID, got, pin.SHA256))
			continue
		}
		row, err := RowFromAPI(raw)
		if err != nil {
			errs = append(errs, fmt.Errorf("record %d: %w", pin.ID, err))
			continue
		}
		out.Rows = append(out.Rows, row)
		out.Records = append(out.Records, raw)
	}
	if err := errors.Join(errs...); err != nil {
		return nil, fmt.Errorf("supplement %s: %w", s.ID, err)
	}
	return out, nil
}

// PinSupplement fetches url and returns the pins for the records with the given ids, in
// increasing id order: what a ledger owner pastes into sources.yaml to add a supplement. It
// refuses an id the API does not serve.
func PinSupplement(url string, ids []int64, fetch Fetcher) ([]ledger.PinnedRecord, error) {
	probe := ledger.Supplement{ID: "probe", URL: url}
	body, err := fetchServed(probe, fetch)
	if err != nil {
		return nil, err
	}
	var out []ledger.PinnedRecord
	for _, id := range ids {
		raw, ok := body[id]
		if !ok {
			return nil, fmt.Errorf("%s does not serve record %d", url, id)
		}
		h, err := RecordSHA256(raw)
		if err != nil {
			return nil, fmt.Errorf("record %d: %w", id, err)
		}
		out = append(out, ledger.PinnedRecord{ID: id, SHA256: h})
	}
	return out, nil
}

// fetchServed fetches the supplement's URL and indexes the records it serves by id.
func fetchServed(s ledger.Supplement, fetch Fetcher) (map[int64]json.RawMessage, error) {
	body, err := fetch(s.URL)
	if err != nil {
		return nil, fmt.Errorf("supplement %s: fetching %s: %w", s.ID, s.URL, err)
	}
	if len(body) > 2 && body[0] == 0x1f && body[1] == 0x8b {
		zr, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("supplement %s: %w", s.ID, err)
		}
		if body, err = io.ReadAll(zr); err != nil {
			return nil, fmt.Errorf("supplement %s: %w", s.ID, err)
		}
	}
	var served []json.RawMessage
	if err := json.Unmarshal(body, &served); err != nil {
		return nil, fmt.Errorf("supplement %s: %s did not serve a JSON list of records: %w", s.ID, s.URL, err)
	}
	byID := map[int64]json.RawMessage{}
	for i, raw := range served {
		id, err := recordID(raw)
		if err != nil {
			return nil, fmt.Errorf("supplement %s: served record %d: %w", s.ID, i, err)
		}
		if _, dup := byID[id]; dup {
			return nil, fmt.Errorf("supplement %s: the API served record %d twice", s.ID, id)
		}
		byID[id] = raw
	}
	return byID, nil
}

// RecordSHA256 is the content hash a supplement pins: the SHA-256 of the canonical JSON of
// the whole record, with keys sorted, no insignificant space and every number as printed.
func RecordSHA256(raw json.RawMessage) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return "", err
	}
	if _, ok := v.(map[string]any); !ok {
		return "", errors.New("a record is a JSON object")
	}
	return ledger.MeasuredID(v)
}

// recordID reads the id the API prints, a number or a string of digits.
func recordID(raw json.RawMessage) (int64, error) {
	var h struct {
		ID json.RawMessage `json:"id"`
	}
	if err := json.Unmarshal(raw, &h); err != nil {
		return 0, err
	}
	return intOf(h.ID, "id")
}

func intOf(raw json.RawMessage, name string) (int64, error) {
	t := strings.Trim(strings.TrimSpace(string(raw)), `"`)
	n, err := strconv.ParseInt(t, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%s: %s is not an integer", name, raw)
	}
	return n, nil
}

var attemptOfURL = regexp.MustCompile(`/actions/runs/\d+/attempts/(\d+)`)

// apiRecord is the part of an API record the ingest reads. The API serves more fields (power
// audit, workers); they are covered by the content hash and are not read.
type apiRecord struct {
	ID             json.RawMessage `json:"id"`
	Model          string          `json:"model"`
	Hardware       string          `json:"hardware"`
	Precision      string          `json:"precision"`
	Framework      string          `json:"framework"`
	BenchmarkType  string          `json:"benchmark_type"`
	Conc           json.RawMessage `json:"conc"`
	Error          *string         `json:"error"`
	Date           string          `json:"date"`
	RunURL         *string         `json:"run_url"`
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

// RowFromAPI turns an API record into the row an export of the same record would be. The API
// does not serve the run's conclusion or the legacy inventory's config id, so the row has
// neither: the unit records a null conclusion, and a supplement's rows are never legacy
// points. The run's attempt is read from the record's run_url, and a record with no run_url is
// refused, because an attempt is never guessed.
func RowFromAPI(raw json.RawMessage) (Row, error) {
	var a apiRecord
	if err := json.Unmarshal(raw, &a); err != nil {
		return Row{}, err
	}
	id, err := intOf(a.ID, "id")
	if err != nil {
		return Row{}, err
	}
	conc, err := intOf(a.Conc, "conc")
	if err != nil {
		return Row{}, err
	}
	if a.RunURL == nil {
		return Row{}, errors.New("run_url: the record states no run, so its attempt cannot be read")
	}
	m := attemptOfURL.FindStringSubmatch(*a.RunURL)
	if m == nil {
		return Row{}, fmt.Errorf("run_url: %q names no run and attempt (.../actions/runs/N/attempts/M)", *a.RunURL)
	}
	attempt, _ := strconv.ParseInt(m[1], 10, 64)
	if len(a.Date) < 10 || !dateOnly.MatchString(a.Date[:10]) {
		return Row{}, fmt.Errorf("date: %q does not begin with YYYY-MM-DD", a.Date)
	}
	r := Row{
		ID: id, RunURL: a.RunURL, RunAttempt: attempt,
		Model: a.Model, Hardware: a.Hardware, Precision: a.Precision, Framework: a.Framework,
		BenchmarkType: a.BenchmarkType, Conc: conc, Error: a.Error, Date: a.Date[:10],
		RecipeFP: a.RecipeFP, Image: a.Image, ISL: a.ISL, OSL: a.OSL, OffloadMode: a.OffloadMode,
		SpecMethod: a.SpecMethod, Disagg: a.Disagg, IsMultinode: a.IsMultinode,
		PrefillTP: a.PrefillTP, PrefillEP: a.PrefillEP, PrefillDPAttn: a.PrefillDPAttn,
		PrefillWorkers: a.PrefillWorkers, NumPrefillGPU: a.NumPrefillGPU,
		DecodeTP: a.DecodeTP, DecodeEP: a.DecodeEP, DecodeDPAttn: a.DecodeDPAttn,
		DecodeWorkers: a.DecodeWorkers, NumDecodeGPU: a.NumDecodeGPU, Metrics: a.Metrics,
	}
	if r.RecipeFP != nil && *r.RecipeFP == "" {
		r.RecipeFP = nil
	}
	if r.Model == "" || r.Hardware == "" || r.Precision == "" || r.Framework == "" || r.BenchmarkType == "" {
		return Row{}, errors.New("model, hardware, precision, framework and benchmark_type are required")
	}
	return r, nil
}

var dateOnly = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)

// SupplementRows are one supplement's rows, for IngestWithSupplements.
type SupplementRows struct {
	ID   string
	Rows []Row
}

// SupplementReport counts one supplement's records in the ingest report.
type SupplementReport struct {
	ID      string `yaml:"id"`
	Records int    `yaml:"records"`
}

// IngestWithSupplements ingests the dump's rows and every supplement's rows as one input, so
// the ingest rules see all of them. A supplement's row must have an id above every earlier
// row's: a record the dump already holds is refused, because the dump's copy and the API's
// could differ in how numbers are printed and the record would become two measurements. When
// a source revision comes to hold a supplement's records, the supplement is removed in the
// pull request that moves the revision.
func IngestWithSupplements(dump []Row, supps []SupplementRows, revision, input string) ([]ledger.Unit, *Report, error) {
	all := append([]Row(nil), dump...)
	var reps []SupplementReport
	for _, s := range supps {
		for _, r := range s.Rows {
			if n := len(all); n > 0 && r.ID <= all[n-1].ID {
				return nil, nil, fmt.Errorf("supplement %s: record %d is not above record %d; the ingest reads records in increasing id order, and a record the pinned revision holds is not a supplement", s.ID, r.ID, all[n-1].ID)
			}
			all = append(all, r)
		}
		reps = append(reps, SupplementReport{ID: s.ID, Records: len(s.Rows)})
	}
	units, rep, err := Ingest(all, revision, input)
	if err != nil {
		return nil, nil, err
	}
	rep.Supplements = reps
	return units, rep, nil
}
