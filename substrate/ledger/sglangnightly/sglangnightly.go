// Package sglangnightly turns SGLang's nightly consolidated-metrics artifacts into ledger units
// (design §10.2, requirements Table 1.4).
//
// Each artifact is the zip of one consolidated-metrics.json that a run of the sgl-project/sglang
// nightly workflow uploaded. Its results are test cases: a GPU configuration, a partition, a
// model, a variant and the resolved server arguments, with the benchmark rows (one per batch
// size and sequence lengths) measured for it. A unit is one test case in one nightly run; its
// role key is the commit, the test name (the case's variant), the model and the hardware, so
// every case at a commit shares a role with its partitions and re-runs. The artifacts are
// pinned in sources.yaml by run, attempt and SHA-256 (D3); the input is the directory of their
// zips, named by the pin's id. No measured value is committed (X12).
package sglangnightly

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/ingestcore"
)

// SourceID is the source's identifier in the ledger and in every unit's identity.
const SourceID = "sglang-nightly"

// Facet is what the report counts units by.
const Facet = "hardware"

// LeftOutNoBenchmarks is the reason for a test case with no benchmark row: it measured nothing.
const LeftOutNoBenchmarks = "no_benchmark_rows"

// IdentityFields are a unit's identifying fields besides the source, in the artifact's own names
// where it has one.
var IdentityFields = []string{"run_id", "commit", "test", "model", "hardware", "partition", "server_args_sha256"}

// RoleKeyFields are the identifying fields the role key keeps.
var RoleKeyFields = []string{"commit", "test", "model", "hardware"}

// Result is one test case of an artifact.
type Result struct {
	GPUConfig  string            `json:"gpu_config"`
	Partition  json.Number       `json:"partition"`
	Model      string            `json:"model"`
	Variant    string            `json:"variant"`
	ServerArgs []string          `json:"server_args"`
	Benchmarks []json.RawMessage `json:"benchmarks"`
	// ByIOLen regroups Benchmarks by sequence lengths; it holds no measurement Benchmarks lacks.
	ByIOLen json.RawMessage `json:"benchmarks_by_io_len"`
}

// Consolidated is one artifact's consolidated-metrics.json.
type Consolidated struct {
	RunID     string   `json:"run_id"`
	RunDate   string   `json:"run_date"`
	CommitSHA string   `json:"commit_sha"`
	Branch    string   `json:"branch"`
	Results   []Result `json:"results"`
}

// Artifact is a pinned artifact that has been read and checked against its pin.
type Artifact struct {
	ID      string
	Run     string
	Attempt int
	Data    Consolidated
	// Zip is the SHA-256 of the artifact's zip.
	Zip string
}

func sum(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

func pinned(a ledger.Artifact, name string) string {
	for _, x := range a.Revision.Assets {
		if x.Name == name {
			return x.SHA256
		}
	}
	return ""
}

func decodeStrict(raw []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	dec.UseNumber()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if dec.More() {
		return errors.New("more than one JSON document")
	}
	return nil
}

// ReadArtifact checks one artifact's zip against its pin (D3) and reads its one JSON file. The
// zip, the file, the run id and the commit must each be the pinned ones.
func ReadArtifact(a ledger.Artifact, zipBytes []byte) (*Artifact, error) {
	rev := a.Revision
	if got, want := sum(zipBytes), pinned(a, rev.Artifact+".zip"); got != want {
		return nil, fmt.Errorf("artifact %s: the zip's SHA-256 is %s, but sources.yaml pins %s", a.ID, got, want)
	}
	zr, err := zip.NewReader(bytes.NewReader(zipBytes), int64(len(zipBytes)))
	if err != nil {
		return nil, fmt.Errorf("artifact %s: %w", a.ID, err)
	}
	var file string
	for _, x := range rev.Assets {
		if x.Name != rev.Artifact+".zip" {
			file = x.Name
		}
	}
	if len(zr.File) != 1 || zr.File[0].Name != file {
		return nil, fmt.Errorf("artifact %s: the zip must hold exactly the pinned file %q", a.ID, file)
	}
	rc, err := zr.File[0].Open()
	if err != nil {
		return nil, fmt.Errorf("artifact %s: %w", a.ID, err)
	}
	raw, err := io.ReadAll(rc)
	if cerr := rc.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, fmt.Errorf("artifact %s: %w", a.ID, err)
	}
	if got, want := sum(raw), pinned(a, file); got != want {
		return nil, fmt.Errorf("artifact %s: %s has SHA-256 %s, but sources.yaml pins %s", a.ID, file, got, want)
	}
	out := &Artifact{ID: a.ID, Run: rev.Ref, Attempt: rev.RunAttempt, Zip: sum(zipBytes)}
	if err := decodeStrict(raw, &out.Data); err != nil {
		return nil, fmt.Errorf("artifact %s: %s: %w", a.ID, file, err)
	}
	if out.Data.RunID != rev.Ref || out.Data.CommitSHA != rev.Commit {
		return nil, fmt.Errorf("artifact %s: %s is of run %s at %s, but sources.yaml pins run %s at %s", a.ID, file, out.Data.RunID, out.Data.CommitSHA, rev.Ref, rev.Commit)
	}
	return out, nil
}

// ReadArtifacts reads every artifact the source pins from dir, where each is <id>.zip. It
// returns them in sources.yaml order and the SHA-256 of the pins that identify the input.
func ReadArtifacts(src *ledger.Source, dir string) ([]*Artifact, string, error) {
	if src == nil || src.ID != SourceID {
		return nil, "", fmt.Errorf("not the %s source", SourceID)
	}
	if len(src.Artifacts) == 0 {
		return nil, "", errors.New("the source pins no artifact")
	}
	var out []*Artifact
	var errs []error
	var pins []any
	for _, a := range src.Artifacts {
		zb, err := os.ReadFile(filepath.Join(dir, a.ID+".zip"))
		if err != nil {
			errs = append(errs, err)
			continue
		}
		x, err := ReadArtifact(a, zb)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		out = append(out, x)
		pins = append(pins, map[string]any{"artifact": a.ID, "zip": x.Zip})
	}
	if err := errors.Join(errs...); err != nil {
		return nil, "", err
	}
	id, err := ledger.ID(pins)
	return out, id, err
}

func wholeNumber(n json.Number) (int64, error) {
	v, err := strconv.ParseInt(n.String(), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%q is not a whole number", n.String())
	}
	return v, nil
}

// Identity returns the identifying fields of the test case at index i of the artifact.
func (a *Artifact) Identity(i int) (map[string]any, error) {
	r := a.Data.Results[i]
	part, err := wholeNumber(r.Partition)
	if err != nil {
		return nil, fmt.Errorf("%s: results[%d].partition: %w", a.ID, i, err)
	}
	args := make([]any, len(r.ServerArgs))
	for j, s := range r.ServerArgs {
		args[j] = s
	}
	h, err := ledger.ID(args)
	if err != nil {
		return nil, err
	}
	return map[string]any{"source": SourceID, "run_id": a.Run, "commit": a.Data.CommitSHA, "test": r.Variant,
		"model": r.Model, "hardware": r.GPUConfig, "partition": part, "server_args_sha256": h}, nil
}

// RoleKey returns the role key of a SGLang nightly identity: the commit, the test name, the
// model and the hardware. It refuses an identity that lacks a field or holds another.
func RoleKey(identity map[string]any) (map[string]any, error) {
	if len(identity) != len(IdentityFields)+1 {
		return nil, fmt.Errorf("identity has %d fields, want source and the %d of %v", len(identity), len(IdentityFields), IdentityFields)
	}
	if identity["source"] != SourceID {
		return nil, fmt.Errorf("identity.source: %v, want %q", identity["source"], SourceID)
	}
	rk := map[string]any{"source": SourceID}
	for _, f := range IdentityFields {
		if _, ok := identity[f]; !ok {
			return nil, fmt.Errorf("identity lacks %q", f)
		}
	}
	for _, f := range RoleKeyFields {
		rk[f] = identity[f]
	}
	return rk, nil
}

// Ingest turns the artifacts, in sources.yaml order, into units and the ingest report. A
// record is a test case of one artifact, named run/attempt:index. input is the SHA-256 that
// ReadArtifacts returned.
func Ingest(arts []*Artifact, revision, input string) ([]ledger.Unit, *Report, error) {
	left := map[string][]string{}
	var cands []ingestcore.Candidate
	records := 0
	for _, a := range arts {
		for i, r := range a.Data.Results {
			records++
			rec := fmt.Sprintf("%s/%d:%d", a.Run, a.Attempt, i)
			if len(r.Benchmarks) == 0 {
				left[LeftOutNoBenchmarks] = append(left[LeftOutNoBenchmarks], rec)
				continue
			}
			id, err := a.Identity(i)
			if err != nil {
				return nil, nil, err
			}
			rk, err := RoleKey(id)
			if err != nil {
				return nil, nil, err
			}
			rows := make([]any, len(r.Benchmarks))
			for j, b := range r.Benchmarks {
				dec := json.NewDecoder(bytes.NewReader(b))
				dec.UseNumber()
				if err := dec.Decode(&rows[j]); err != nil {
					return nil, nil, fmt.Errorf("%s: %w", rec, err)
				}
			}
			cands = append(cands, ingestcore.Candidate{Record: rec, Identity: id, RoleKey: rk,
				Measured: map[string]any{"benchmarks": rows},
				Recorded: map[string]any{"attempt": a.Attempt, "run_date": a.Data.RunDate},
				Facet:    r.GPUConfig})
		}
	}
	return ingestcore.Ingest(SourceID, revision, input, Facet, records, cands, left)
}

// Report is the ingest report.
type Report = ingestcore.Report
