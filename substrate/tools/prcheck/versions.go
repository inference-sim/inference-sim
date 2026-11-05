package main

import (
	"bufio"
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"
)

// The version-literal lint (X23). A version of Go or of a BLIS module is written in one
// place: stack.json, a go.mod (and its go.sum), contracts/toolchain, and an application's
// or the harness's pin file. Every other file that states one is generated from those.
//
// A literal is a Go version (go1.27.2, "go 1.27", Go 1.24, golang:1.27, go-version: 1.27)
// or a BLIS module with a version (blis-vllm v0.32.0-blis.2, inference-sim@v0.16.0,
// blis-substrate v0.11.0). A file is exempt when it is a source of versions, a test or
// test fixture, or generated (it carries "Code generated" or "BEGIN GENERATED"). Every
// other finding must be listed in the baseline file, which may only shrink: a finding
// not in it fails, and a baseline entry no longer found fails too, so the list is kept
// exact.

var (
	goVersionLit = regexp.MustCompile(`\bgo1\.\d+(?:\.\d+)?\b|\bgo 1\.\d+(?:\.\d+)?\b|\bGo 1\.\d+(?:\.\d+)?\b|\bgolang:1\.\d+|go-version:\s*['"]?1\.\d+`)
	blisModLit   = regexp.MustCompile(`\b(?:blis-substrate|blis-vllm|blis-sglang|inference-sim|llm-d-router|blis-config-search|accuracy-dashboard)[@ ]v\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?`)
)

// versionSource reports whether a path is one of X23's sources of versions, or exempt.
func versionSource(p string) bool {
	base := path.Base(p)
	switch base {
	case "go.mod", "go.sum", "go.work.sum", "stack.json", "stack.lock", "pins.env", "harness.pin", "toolchain",
		"version-literals.txt": // the baseline lists literals; it states none
		return true
	}
	return strings.HasSuffix(base, "_test.go") || strings.HasPrefix(base, "test_") ||
		strings.Contains("/"+p, "/testdata/") || strings.Contains("/"+p, "/tests/") ||
		strings.HasPrefix(p, "contracts/invariants/")
}

func generated(src []byte) bool {
	s := string(src)
	return strings.Contains(s, "Code generated") || strings.Contains(s, "BEGIN GENERATED")
}

// VersionFindings returns "path: literal" for each version literal in a file that is not
// a source of versions, one per distinct literal per file.
func VersionFindings(p string, src []byte) []string {
	if versionSource(p) || generated(src) {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, re := range []*regexp.Regexp{goVersionLit, blisModLit} {
		for _, m := range re.FindAllString(string(src), -1) {
			k := p + ": " + m
			if !seen[k] {
				seen[k] = true
				out = append(out, k)
			}
		}
	}
	return out
}

// ParseBaseline reads a baseline file: one "path: literal" per line, # comments allowed.
func ParseBaseline(src string) map[string]bool {
	out := map[string]bool{}
	sc := bufio.NewScanner(strings.NewReader(src))
	for sc.Scan() {
		l := strings.TrimSpace(sc.Text())
		if l != "" && !strings.HasPrefix(l, "#") {
			out[l] = true
		}
	}
	return out
}

// CheckVersions compares findings with the baseline.
func CheckVersions(findings []string, baseline map[string]bool) []string {
	var bad []string
	found := map[string]bool{}
	for _, f := range findings {
		found[f] = true
		if !baseline[f] {
			bad = append(bad, fmt.Sprintf("%s is a version literal outside stack.json, go.mod, contracts/toolchain and the pin files; generate it from them (X23)", f))
		}
	}
	for b := range baseline {
		if !found[b] {
			bad = append(bad, fmt.Sprintf("baseline entry %q no longer occurs; remove it, the baseline only shrinks (X23)", b))
		}
	}
	sort.Strings(bad)
	return bad
}

var findingLine = regexp.MustCompile(`^([^:]+):\d+: `)

// CheckBaseline is the ratchet for a check turned on over a tree that already breaks it:
// a finding is a failure unless the baseline lists it, with its line number removed, and
// a baseline entry that no longer occurs is a failure too, so the list only shrinks.
func CheckBaseline(findings []string, baseline map[string]bool, inv string) []string {
	var bad []string
	found := map[string]bool{}
	for _, f := range findings {
		k := findingLine.ReplaceAllString(f, "$1: ")
		found[k] = true
		if !baseline[k] {
			bad = append(bad, f)
		}
	}
	for b := range baseline {
		if !found[b] {
			bad = append(bad, fmt.Sprintf("baseline entry %q no longer occurs; remove it, the baseline only shrinks (%s)", b, inv))
		}
	}
	sort.Strings(bad)
	return bad
}
