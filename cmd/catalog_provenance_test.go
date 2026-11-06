package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/load"
)

// The --metrics-path file's catalog block (#1732) names the catalog's source and its combined
// content hash, the values the Result's data.catalog records (design §3.3).
func TestCatalogFileProvenance_NamesSourceAndHash(t *testing.T) {
	if got := catalogFileProvenance(nil); got != nil {
		t.Errorf("no data opened must produce no block; got %+v", got)
	}
	emb, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	p := catalogFileProvenance(emb)
	if p.Path != "embedded" || p.Revision != emb.Catalog.CombinedHash || p.Dirty {
		t.Errorf("embedded: %+v, want path embedded, revision %s, dirty false", p, emb.Catalog.CombinedHash)
	}
	dir, err := load.Open(load.Options{Catalog: testCatalogDir()})
	if err != nil {
		t.Fatal(err)
	}
	if p := catalogFileProvenance(dir); p.Path != testCatalogDir() || p.Revision != emb.Catalog.CombinedHash {
		t.Errorf("a path: %+v, want the path and the same content hash", p)
	}
}

// resultsFileEmitOptions supplies no option without a file path, and its six with one.
func TestResultsFileEmitOptions_FileOnly(t *testing.T) {
	if opts := resultsFileEmitOptions(""); len(opts) != 0 {
		t.Errorf("no output file path must yield no emit options, got %d", len(opts))
	}
	if opts := resultsFileEmitOptions(filepath.Join(t.TempDir(), "metrics.json")); len(opts) != 6 {
		t.Errorf("an output file path must yield the catalog-provenance, scenario-override, "+
			"kernel-provenance, build-version, acceptance and prefix-cache options, got %d", len(opts))
	}

}

// TestCatalogProvenance_EveryFileEmitSitePassesProvenance is the R23 drift guard for
// INV-13: every production call in cmd/ that writes the --metrics-path file (WriteMetricsFile,
// and EmitOutput, which also prints) must route through resultsFileEmitOptions, so a new
// site cannot silently produce a results file with no provenance while its sibling command
// records one. run and replay write the file from one shared site (emitRunResult, #1871).
func TestCatalogProvenance_EveryFileEmitSitePassesProvenance(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob cmd/*.go: %v", err)
	}
	sites := 0
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		src, readErr := os.ReadFile(file)
		if readErr != nil {
			t.Fatalf("read %s: %v", file, readErr)
		}
		for _, call := range append(emitOutputCallArgs(string(src), ".EmitOutput("),
			emitOutputCallArgs(string(src), ".WriteMetricsFile(")...) {
			sites++
			if !strings.Contains(call.args, "resultsFileEmitOptions") {
				t.Errorf("%s:%d: a results-file call must pass resultsFileEmitOptions(...) "+
					"so its results file records catalog provenance (#1732, INV-13):\n\t%s",
					file, call.line, strings.Join(strings.Fields(call.args), " "))
			}
		}
	}
	// Non-vacuity: the shared run/replay file site must be found, otherwise the scan is
	// looking at nothing and would pass after a rename.
	if sites < 1 {
		t.Fatalf("non-vacuity: expected the shared run/replay results-file site, found %d", sites)
	}
}

// emitOutputCall is one `.EmitOutput(...)` call found in a source file: its 1-based line
// number and the full argument text, which may span several lines.
type emitOutputCall struct {
	line int
	args string
}

// emitOutputCallArgs extracts every `.EmitOutput(...)` call's argument text from Go source
// by matching parentheses, so a call broken across lines is examined as one unit (a
// line-wise grep would miss an option passed on a continuation line).
func emitOutputCallArgs(src, marker string) []emitOutputCall {
	var calls []emitOutputCall
	for offset := 0; ; {
		idx := strings.Index(src[offset:], marker)
		if idx < 0 {
			return calls
		}
		open := offset + idx + len(marker) - 1 // index of '('
		depth, end := 0, -1
		for i := open; i < len(src); i++ {
			switch src[i] {
			case '(':
				depth++
			case ')':
				depth--
				if depth == 0 {
					end = i
				}
			}
			if end >= 0 {
				break
			}
		}
		if end < 0 {
			return calls // unbalanced source; nothing more to extract
		}
		calls = append(calls, emitOutputCall{
			line: strings.Count(src[:open], "\n") + 1,
			args: src[open+1 : end],
		})
		offset = end
	}
}
