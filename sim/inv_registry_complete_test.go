package sim

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// invCited matches a whole invariant ID: INV- and then letters, digits and hyphens. A trailing
// hyphen-and-more is part of the token, so INV-A does not match inside INV-A2 and the prefix
// trap of #1735 (16 hits for INV-A against 11 sites) cannot arise.
var invCited = regexp.MustCompile(`INV-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*`)

// notInvariantIDs are tokens of the form INV-* that are not invariant IDs. Each is named, with
// its reason, so a new adjective is a decision here and not something a regex quietly swallows.
var notInvariantIDs = map[string]string{
	// The adjective "INV-6-safe" (determinism-safe), sim/workload/fixed_accumulate_test.go.
	"INV-6-safe": "an adjective derived from INV-6, not an ID",
}

// trackedFiles returns the files git tracks, or would track, under dirs, relative to the
// repository root (X25: a test reads the repository's own files, not what sits in the
// workspace).
func trackedFiles(t *testing.T, root string, dirs ...string) []string {
	t.Helper()
	args := append([]string{"-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"}, dirs...)
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		t.Fatalf("git ls-files: %v", err)
	}
	var files []string
	for _, f := range strings.Split(string(out), "\x00") {
		if f == "" {
			continue
		}
		if _, err := os.Stat(filepath.Join(root, f)); err == nil {
			files = append(files, f)
		}
	}
	return files
}

// registryIDs returns every ID the registry resolves: the IDs named in a heading (a heading may
// name several, such as "INV-16, INV-17: retired") and the first cell of a table row (the LoRA
// pointer lines and the retired-invariant table).
func registryIDs(t *testing.T, doc string) map[string]bool {
	t.Helper()
	ids := map[string]bool{}
	for _, line := range strings.Split(doc, "\n") {
		switch {
		case strings.HasPrefix(line, "#"):
			for _, id := range invCited.FindAllString(line, -1) {
				ids[id] = true
			}
		case strings.HasPrefix(line, "| **INV-"):
			cell := strings.SplitN(strings.TrimPrefix(line, "|"), "|", 2)[0]
			for _, id := range invCited.FindAllString(cell, -1) {
				ids[id] = true
			}
		}
	}
	return ids
}

// The invariant registry is complete (#1735): every INV-* ID cited in sim/ and cmd/ resolves to
// a heading or pointer row of docs/contributing/standards/invariants.md. A new ID cited
// without an entry fails here, naming the ID and where it is cited.
func TestInvariantRegistry_ResolvesEveryCitedID(t *testing.T) {
	root := ".."
	doc, err := os.ReadFile(filepath.Join(root, "docs", "contributing", "standards", "invariants.md"))
	if err != nil {
		t.Fatal(err)
	}
	known := registryIDs(t, string(doc))
	if len(known) < 40 {
		t.Fatalf("non-vacuity: the registry resolves %d IDs; the parser no longer reads it", len(known))
	}
	where := map[string]string{}
	for _, f := range trackedFiles(t, root, "sim", "cmd") {
		raw, err := os.ReadFile(filepath.Join(root, f))
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(raw), "\n") {
			for _, id := range invCited.FindAllString(line, -1) {
				if _, ok := where[id]; !ok {
					where[id] = f + ":" + strconv.Itoa(i+1)
				}
			}
		}
	}
	if len(where) < 30 {
		t.Fatalf("non-vacuity: found %d cited IDs in sim/ and cmd/, expected about 40", len(where))
	}
	var missing []string
	for id, at := range where {
		if _, skip := notInvariantIDs[id]; skip || known[id] {
			continue
		}
		missing = append(missing, id+" (first cited at "+at+")")
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Errorf("invariant IDs cited in code that the registry does not resolve (#1721, #1735); add each to docs/contributing/standards/invariants.md or, if it is not an ID, to notInvariantIDs:\n  %s",
			strings.Join(missing, "\n  "))
	}
	for id := range notInvariantIDs {
		if _, cited := where[id]; !cited {
			t.Errorf("notInvariantIDs names %s, which no file cites: remove the stale entry", id)
		}
	}
}

// The scan matches whole IDs and reads the registry as the document states it: INV-A is not
// found inside INV-A2, a heading naming two IDs resolves both, and a pointer row resolves.
func TestInvariantRegistry_IDParsing(t *testing.T) {
	if got := invCited.FindAllString("see INV-A2 and INV-A, INV-PD-6b; INV-6-safe.", -1); strings.Join(got, " ") != "INV-A2 INV-A INV-PD-6b INV-6-safe" {
		t.Errorf("IDs found = %v", got)
	}
	ids := registryIDs(t, "#### INV-16, INV-17: retired\n| **INV-L3** | x |\n| [INV-1](#inv-1) y | z |\n")
	for _, want := range []string{"INV-16", "INV-17", "INV-L3"} {
		if !ids[want] {
			t.Errorf("registry parse lacks %s: %v", want, ids)
		}
	}
	if ids["INV-1"] {
		t.Error("an index row is not an entry; only headings and pointer rows resolve an ID")
	}
}
