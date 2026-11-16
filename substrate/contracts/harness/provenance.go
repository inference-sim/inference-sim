package harness

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// ProvenanceRules says how an engine repository's Go files state where their logic comes from
// (VLM-2, SGL-2). Each file states it in its first comment block: a file translated from the
// upstream names the upstream file and commit and starts with the upstream's licence lines, a file
// moved from another repository names that repository and commit, and any other file says it is
// original.
type ProvenanceRules struct {
	// Derived matches the header line of a file translated from the upstream, such as
	// "// Derived from <upstream path> at <release> (<commit>)".
	Derived *regexp.Regexp
	// Licence is the lines a derived file's header starts with, in order: the upstream's SPDX
	// licence identifier and copyright text.
	Licence []string
	// Moved matches the header line of a file moved from another repository; nil when the
	// repository holds none.
	Moved *regexp.Regexp
	// Original is the header line, or its prefix, of a file original to the repository.
	Original string
}

// header is a file's first comment block: its leading lines that start with "//".
func header(src string) []string {
	var out []string
	for _, l := range strings.Split(src, "\n") {
		if !strings.HasPrefix(l, "//") {
			break
		}
		out = append(out, l)
	}
	return out
}

// ProvenanceProblems checks each file (relative to root, as GoFiles returns them) against the
// rules and returns one problem per file that states no provenance, or is derived from the
// upstream without the upstream's licence lines first.
func ProvenanceProblems(root string, files []string, r ProvenanceRules) ([]string, error) {
	if r.Derived == nil || r.Original == "" || len(r.Licence) == 0 {
		return nil, fmt.Errorf("harness: ProvenanceRules needs Derived, Licence and Original")
	}
	var bad []string
	for _, p := range files {
		raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(p)))
		if err != nil {
			return nil, err
		}
		h := header(string(raw))
		var derived, moved, orig bool
		for _, l := range h {
			derived = derived || r.Derived.MatchString(l)
			moved = moved || (r.Moved != nil && r.Moved.MatchString(l))
			orig = orig || strings.HasPrefix(l, r.Original)
		}
		switch {
		case derived:
			if len(h) < len(r.Licence) || strings.Join(h[:len(r.Licence)], "\n") != strings.Join(r.Licence, "\n") {
				bad = append(bad, fmt.Sprintf("%s is derived from the upstream but does not start with the upstream's licence lines %q", p, r.Licence))
			}
		case moved, orig:
		default:
			bad = append(bad, fmt.Sprintf("%s states no provenance: start its first comment block with a line matching %q, a moved-from line, or %q", p, r.Derived.String(), r.Original))
		}
	}
	return bad, nil
}
