// Package vllmperf turns the vLLM CI perf rows into ledger units (design §10.2, requirements
// Table 1.4).
//
// A row of ci.vllm.ai/api/perf is one benchmark configuration at one commit: model, device,
// tensor-parallel size, concurrency, sequence lengths and precision, with the container image
// it ran. The API states the commit only in the image tag, so a row whose image is not tagged
// with a full commit is left out. The benchmark name and the serve arguments come from the
// vllm-project/perf-eval recipe whose model, GPU and benchmark shape the row matches. A unit
// is a configuration at a commit; its role key is the commit, the model, the hardware, the
// benchmark name and the configuration hash. The input is the rows and the recipes exported by
// export.py, pinned in sources.yaml by hash (D3); no measured value is committed (X12).
package vllmperf

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/ingestcore"
)

// SourceID is the source's identifier in the ledger and in every unit's identity.
const SourceID = "vllm-ci-perf"

// Facet is what the report counts units by.
const Facet = "hardware"

// The reasons a row is left out.
const (
	LeftOutNoCommit        = "no_commit_in_image"
	LeftOutNoRecipe        = "no_matching_recipe"
	LeftOutAmbiguousRecipe = "ambiguous_recipe"
	LeftOutEmptyMetric     = "empty_metric"
)

// IdentityFields are a unit's identifying fields besides the source.
var IdentityFields = []string{"commit", "model", "hardware", "benchmark", "config_sha256"}

// Metrics are the measured fields of a row, in the API's names.
var Metrics = []string{"tput_per_gpu", "input_tput_per_gpu", "output_tput_per_gpu",
	"mean_ttft", "mean_tpot", "mean_itl", "mean_e2el",
	"p99_ttft", "p99_tpot", "p99_itl", "p99_e2el",
	"median_ttft", "median_tpot", "median_itl", "median_e2el"}

// Row is one row of the API, every value a string or null as the API gives it.
type Row struct {
	Date      *string `json:"date"`
	Model     *string `json:"model"`
	Device    *string `json:"device"`
	TP        *string `json:"tp"`
	Conc      *string `json:"conc"`
	ISL       *string `json:"isl"`
	OSL       *string `json:"osl"`
	Precision *string `json:"precision"`
	Image     *string `json:"image"`

	TputPerGPU       *string `json:"tput_per_gpu"`
	InputTputPerGPU  *string `json:"input_tput_per_gpu"`
	OutputTputPerGPU *string `json:"output_tput_per_gpu"`
	MeanTTFT         *string `json:"mean_ttft"`
	MeanTPOT         *string `json:"mean_tpot"`
	MeanITL          *string `json:"mean_itl"`
	MeanE2EL         *string `json:"mean_e2el"`
	P99TTFT          *string `json:"p99_ttft"`
	P99TPOT          *string `json:"p99_tpot"`
	P99ITL           *string `json:"p99_itl"`
	P99E2EL          *string `json:"p99_e2el"`
	MedianTTFT       *string `json:"median_ttft"`
	MedianTPOT       *string `json:"median_tpot"`
	MedianITL        *string `json:"median_itl"`
	MedianE2EL       *string `json:"median_e2el"`
}

func (r *Row) metrics() []*string {
	return []*string{r.TputPerGPU, r.InputTputPerGPU, r.OutputTputPerGPU, r.MeanTTFT, r.MeanTPOT, r.MeanITL, r.MeanE2EL,
		r.P99TTFT, r.P99TPOT, r.P99ITL, r.P99E2EL, r.MedianTTFT, r.MedianTPOT, r.MedianITL, r.MedianE2EL}
}

// Recipe is one benchmark of one perf-eval workload recipe.
type Recipe struct {
	Workload       string `json:"workload"`
	GPU            string `json:"gpu"`
	Model          string `json:"model"`
	ServeArgs      string `json:"serve_args"`
	Benchmark      string `json:"benchmark"`
	InputLen       int64  `json:"input_len"`
	OutputLen      int64  `json:"output_len"`
	MaxConcurrency int64  `json:"max_concurrency"`
}

// Input is what ReadInput read: the rows and recipes, and the SHA-256 that identifies them.
type Input struct {
	Rows    []Row
	Recipes []Recipe
	// RowsSHA256 and RecipesSHA256 are the hashes of the two files, as sources.yaml pins them;
	// SHA256 identifies the pair.
	RowsSHA256    string
	RecipesSHA256 string
	SHA256        string
}

func readLines[T any](r io.Reader, what string) ([]T, []byte, error) {
	var out []T
	var raw bytes.Buffer
	sc := bufio.NewScanner(io.TeeReader(r, &raw))
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for n := 1; sc.Scan(); n++ {
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()
		var v T
		if err := dec.Decode(&v); err != nil {
			return nil, nil, fmt.Errorf("%s line %d: %w", what, n, err)
		}
		if dec.More() {
			return nil, nil, fmt.Errorf("%s line %d: more than one JSON document", what, n)
		}
		out = append(out, v)
	}
	return out, raw.Bytes(), sc.Err()
}

// ReadInput reads rows.jsonl and recipes.jsonl strictly. The input's hash covers both, each
// hashed whole and joined with a newline.
func ReadInput(rows, recipes io.Reader) (*Input, error) {
	rs, rawRows, err := readLines[Row](rows, "rows")
	if err != nil {
		return nil, err
	}
	cs, rawRecipes, err := readLines[Recipe](recipes, "recipes")
	if err != nil {
		return nil, err
	}
	a, b := sha256.Sum256(rawRows), sha256.Sum256(rawRecipes)
	h := sha256.Sum256([]byte(hex.EncodeToString(a[:]) + "\n" + hex.EncodeToString(b[:])))
	return &Input{Rows: rs, Recipes: cs, RowsSHA256: hex.EncodeToString(a[:]), RecipesSHA256: hex.EncodeToString(b[:]),
		SHA256: hex.EncodeToString(h[:])}, nil
}

// CheckRevision checks the input against the source's pin (D3): the rows and the recipes have
// the pinned hashes, the pinned count, and no row is dated after the cutoff.
func CheckRevision(src *ledger.Source, in *Input) error {
	if src == nil || src.ID != SourceID {
		return fmt.Errorf("not the %s source", SourceID)
	}
	rev := src.Revision
	var errs []error
	if in.RowsSHA256 != rev.Ref {
		errs = append(errs, fmt.Errorf("the rows have SHA-256 %s, but sources.yaml pins %s", in.RowsSHA256, rev.Ref))
	}
	if len(in.Rows) != rev.Rows {
		errs = append(errs, fmt.Errorf("the input holds %d rows, but sources.yaml pins %d", len(in.Rows), rev.Rows))
	}
	if len(rev.Assets) == 1 && in.RecipesSHA256 != rev.Assets[0].SHA256 {
		errs = append(errs, fmt.Errorf("the recipes have SHA-256 %s, but sources.yaml pins %s", in.RecipesSHA256, rev.Assets[0].SHA256))
	}
	for i := range in.Rows {
		if d := deref(in.Rows[i].Date); len(d) < 10 || d[:10] > rev.Cutoff {
			errs = append(errs, fmt.Errorf("row:%d is dated %q, after the cutoff %s", i, d, rev.Cutoff))
			break
		}
	}
	return errors.Join(errs...)
}

var commitTag = regexp.MustCompile(`^[0-9a-f]{40}$`)

// ImageCommit returns the image repository and the full commit its tag names, or "" for the
// commit when the tag is not one.
func ImageCommit(image string) (repo, commit string) {
	i := strings.LastIndex(image, ":")
	if i < 0 || strings.Contains(image[i+1:], "/") {
		return image, ""
	}
	if tag := image[i+1:]; commitTag.MatchString(tag) {
		return image[:i], tag
	}
	return image[:i], ""
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// RoleKey returns the role key of a vLLM perf identity: the commit, model, hardware, benchmark
// name and configuration hash. It refuses an identity that lacks a field or holds another.
func RoleKey(identity map[string]any) (map[string]any, error) {
	if len(identity) != len(IdentityFields)+1 {
		return nil, fmt.Errorf("identity has %d fields, want source and the %d of %v", len(identity), len(IdentityFields), IdentityFields)
	}
	if identity["source"] != SourceID {
		return nil, fmt.Errorf("identity.source: %v, want %q", identity["source"], SourceID)
	}
	rk := map[string]any{"source": SourceID}
	for _, f := range IdentityFields {
		v, ok := identity[f]
		if !ok {
			return nil, fmt.Errorf("identity lacks %q", f)
		}
		rk[f] = v
	}
	return rk, nil
}

func (in *Input) recipeFor(r *Row) (*Recipe, string) {
	var found []*Recipe
	for i := range in.Recipes {
		c := &in.Recipes[i]
		if c.Model != deref(r.Model) || !strings.EqualFold(c.GPU, deref(r.Device)) {
			continue
		}
		if fmt.Sprint(c.InputLen) != deref(r.ISL) || fmt.Sprint(c.OutputLen) != deref(r.OSL) || fmt.Sprint(c.MaxConcurrency) != deref(r.Conc) {
			continue
		}
		found = append(found, c)
	}
	switch len(found) {
	case 0:
		return nil, LeftOutNoRecipe
	case 1:
		return found[0], ""
	}
	for _, c := range found[1:] {
		if *c != *found[0] {
			return nil, LeftOutAmbiguousRecipe
		}
	}
	return found[0], ""
}

// Ingest turns the rows into units and the ingest report. A record is a row, named row:N by
// its place in the input, counted from 0.
func Ingest(in *Input, revision string) ([]ledger.Unit, *Report, error) {
	left := map[string][]string{}
	var cands []ingestcore.Candidate
	for i := range in.Rows {
		r := &in.Rows[i]
		rec := fmt.Sprintf("row:%d", i)
		repo, commit := ImageCommit(deref(r.Image))
		if commit == "" {
			left[LeftOutNoCommit] = append(left[LeftOutNoCommit], rec)
			continue
		}
		if r.TputPerGPU == nil {
			left[LeftOutEmptyMetric] = append(left[LeftOutEmptyMetric], rec)
			continue
		}
		rc, why := in.recipeFor(r)
		if rc == nil {
			left[why] = append(left[why], rec)
			continue
		}
		cfg, err := ledger.ID(map[string]any{"tp": deref(r.TP), "conc": deref(r.Conc), "isl": deref(r.ISL),
			"osl": deref(r.OSL), "precision": deref(r.Precision), "image_repository": repo, "serve_args": rc.ServeArgs})
		if err != nil {
			return nil, nil, err
		}
		id := map[string]any{"source": SourceID, "commit": commit, "model": deref(r.Model), "hardware": deref(r.Device),
			"benchmark": rc.Benchmark, "config_sha256": cfg}
		rk, err := RoleKey(id)
		if err != nil {
			return nil, nil, err
		}
		m := map[string]any{}
		for j, v := range r.metrics() {
			if v == nil {
				m[Metrics[j]] = nil
			} else {
				m[Metrics[j]] = json.Number(*v)
			}
		}
		cands = append(cands, ingestcore.Candidate{Record: rec, Identity: id, RoleKey: rk, Measured: m,
			Recorded: map[string]any{"date": deref(r.Date), "workload": rc.Workload}, Facet: deref(r.Device)})
	}
	for _, v := range left {
		sort.Strings(v)
	}
	return ingestcore.Ingest(SourceID, revision, in.SHA256, Facet, len(in.Rows), cands, left)
}

// Report is the ingest report.
type Report = ingestcore.Report
