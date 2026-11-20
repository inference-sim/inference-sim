// Package sglangtraces turns the profiler traces of sgl-project/ci-data into ledger units
// (design §10.2, requirements Table 1.4).
//
// A unit is one trace file, traces/<run_id>/<server config>/<timestamp>/<file>: one batch size
// and input length of one phase, rank 0. Its role key is the file. A trace is hundreds of
// megabytes, so the ingest tool does not read one: the input is the listing of the pinned
// commit's traces tree, one JSON object per file with its git blob SHA-1 and size, which is
// what identifies the content. The output is the unit list and the ingest report, a function of
// that listing alone, so CI can regenerate them and require the committed files to match byte
// for byte (D4). No trace content and no value derived from it is committed (X12).
package sglangtraces

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/ingestcore"
)

// SourceID is the source's identifier in the ledger and in every unit's identity.
const SourceID = "sglang-traces"

// Facet is what the report counts units by.
const Facet = "phase"

// Left-out reasons.
const (
	LeftOutEmpty        = "empty_file"
	LeftOutNotRankZero  = "not_rank_zero"
	LeftOutNotTraceFile = "not_a_trace_file"
)

var (
	tracePath = regexp.MustCompile(`^traces/[0-9]+/[^/]+/[^/]+/[^/]+\.trace\.json\.gz$`)
	rankZero  = regexp.MustCompile(`-TP-0(-DP-0)?(-EP-0)?-(EXTEND|DECODE)\.trace\.json\.gz$`)
	rankAny   = regexp.MustCompile(`-TP-[0-9]+(-DP-[0-9]+)?(-EP-[0-9]+)?-(EXTEND|DECODE)\.trace\.json\.gz$`)
	blobSHA   = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// Header is the first line of a listing: the repository, the commit whose tree it lists, and
// that tree's traces entry.
type Header struct {
	Repo   string `json:"repo"`
	Commit string `json:"commit"`
	Tree   string `json:"tree"`
}

// File is one file of the listing: its path from the repository root, git blob SHA-1 and size.
type File struct {
	Path string `json:"path"`
	SHA  string `json:"sha"`
	Size int64  `json:"size"`
}

// Listing is a read listing.
type Listing struct {
	Header Header
	Files  []File
}

func decodeStrict(line []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(line))
	dec.DisallowUnknownFields()
	return dec.Decode(dst)
}

// ReadListing reads a listing: a header line, then one file per line in increasing path order.
func ReadListing(rd io.Reader) (*Listing, error) {
	sc := bufio.NewScanner(rd)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	l := &Listing{}
	n := 0
	for sc.Scan() {
		n++
		if n == 1 {
			if err := decodeStrict(sc.Bytes(), &l.Header); err != nil {
				return nil, fmt.Errorf("line 1: %w", err)
			}
			if l.Header.Repo == "" || l.Header.Commit == "" || l.Header.Tree == "" {
				return nil, errors.New("line 1: the header names the repo, the commit and the traces tree")
			}
			continue
		}
		var f File
		if err := decodeStrict(sc.Bytes(), &f); err != nil {
			return nil, fmt.Errorf("line %d: %w", n, err)
		}
		if !blobSHA.MatchString(f.SHA) || f.Size < 0 || f.Path == "" {
			return nil, fmt.Errorf("line %d: %q needs a path, a 40-digit blob SHA-1 and a size", n, f.Path)
		}
		if len(l.Files) > 0 && l.Files[len(l.Files)-1].Path >= f.Path {
			return nil, fmt.Errorf("line %d: %s is out of order or repeated", n, f.Path)
		}
		l.Files = append(l.Files, f)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(l.Files) == 0 {
		return nil, errors.New("no files: an empty listing is not a source revision")
	}
	return l, nil
}

// RoleKey returns the role key of a trace unit: the file. It refuses an identity that is not
// the source and a file.
func RoleKey(identity map[string]any) (map[string]any, error) {
	file, ok := identity["file"].(string)
	if len(identity) != 2 || identity["source"] != SourceID || !ok || file == "" {
		return nil, fmt.Errorf("identity %v is not {source: %q, file}", identity, SourceID)
	}
	return map[string]any{"source": SourceID, "file": file}, nil
}

func phase(path string) string {
	m := rankZero.FindStringSubmatch(path)
	return m[len(m)-1]
}

// CheckRevision checks a listing against the source's pin (D3): it lists the pinned commit of
// the pinned repository.
func CheckRevision(src *ledger.Source, l *Listing) error {
	if src == nil || src.ID != SourceID {
		return fmt.Errorf("not the %s source", SourceID)
	}
	if !strings.HasPrefix(src.Revision.URL, "https://github.com/"+l.Header.Repo+"/") {
		return fmt.Errorf("the listing is of %s, but sources.yaml pins %s", l.Header.Repo, src.Revision.URL)
	}
	if l.Header.Commit != src.Revision.Ref {
		return fmt.Errorf("the listing is of commit %s, but sources.yaml pins %s", l.Header.Commit, src.Revision.Ref)
	}
	return nil
}

// Ingest turns a listing into units and the ingest report. A file with no content, one that is
// not a rank-0 trace and one that is not a trace at all are left out and counted. A blob that
// appears under two files has the same measured values under two role keys, so both files are
// left out. input is the SHA-256 of the listing.
func Ingest(l *Listing, revision, input string) ([]ledger.Unit, *Report, error) {
	left := map[string][]string{}
	var cands []ingestcore.Candidate
	for _, f := range l.Files {
		switch {
		case !tracePath.MatchString(f.Path) || !rankAny.MatchString(f.Path):
			left[LeftOutNotTraceFile] = append(left[LeftOutNotTraceFile], f.Path)
		case !rankZero.MatchString(f.Path):
			left[LeftOutNotRankZero] = append(left[LeftOutNotRankZero], f.Path)
		case f.Size == 0:
			left[LeftOutEmpty] = append(left[LeftOutEmpty], f.Path)
		default:
			id := map[string]any{"source": SourceID, "file": f.Path}
			rk, err := RoleKey(id)
			if err != nil {
				return nil, nil, err
			}
			cands = append(cands, ingestcore.Candidate{Record: f.Path, Identity: id, RoleKey: rk,
				Measured: map[string]any{"git_blob_sha1": f.SHA, "size": f.Size}, Facet: phase(f.Path)})
		}
	}
	return ingestcore.Ingest(SourceID, revision, input, Facet, len(l.Files), cands, left)
}

// Report is the ingest report.
type Report = ingestcore.Report
