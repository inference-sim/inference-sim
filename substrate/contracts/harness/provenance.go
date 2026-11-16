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
	// Release is the upstream release the repository models. A derived file's header must name
	// its tag and its commit, so a file translated from another release fails (VLM-2, SGL-2).
	Release Upstream
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
	if p := r.Release.Problems(); len(p) > 0 {
		return nil, fmt.Errorf("harness: ProvenanceRules.Release: %s", strings.Join(p, "; "))
	}
	var bad []string
	for _, p := range files {
		raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(p)))
		if err != nil {
			return nil, err
		}
		h := header(string(raw))
		var derived, moved, orig bool
		var derivedLine string
		for _, l := range h {
			if r.Derived.MatchString(l) && !derived {
				derivedLine = l
			}
			derived = derived || r.Derived.MatchString(l)
			moved = moved || (r.Moved != nil && r.Moved.MatchString(l))
			orig = orig || strings.HasPrefix(l, r.Original)
		}
		switch {
		case derived:
			if len(h) < len(r.Licence) || strings.Join(h[:len(r.Licence)], "\n") != strings.Join(r.Licence, "\n") {
				bad = append(bad, fmt.Sprintf("%s is derived from the upstream but does not start with the upstream's licence lines %q", p, r.Licence))
			}
			if !containsWord(derivedLine, r.Release.Tag) || !strings.Contains(derivedLine, r.Release.Commit) {
				bad = append(bad, fmt.Sprintf("%s is derived from a release other than the one this repository models (%s, %s): %q", p, r.Release.Tag, r.Release.Commit, derivedLine))
			}
		case moved, orig:
		default:
			bad = append(bad, fmt.Sprintf("%s states no provenance: start its first comment block with a line matching %q, a moved-from line, or %q", p, r.Derived.String(), r.Original))
		}
	}
	return bad, nil
}

// containsWord reports whether s holds w with no letter, digit or dot right after it, so tag v0.5.2
// is not found inside v0.5.23.
func containsWord(s, w string) bool {
	for i := strings.Index(s, w); i >= 0; {
		end := i + len(w)
		if end == len(s) || !continuesWord(s[end]) {
			return true
		}
		j := strings.Index(s[i+1:], w)
		if j < 0 {
			return false
		}
		i += 1 + j
	}
	return false
}

// continuesWord reports whether c, right after a tag, would make it part of a longer one.
func continuesWord(c byte) bool {
	return c == '.' || '0' <= c && c <= '9' || 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
}
