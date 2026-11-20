package main

import (
	"os"
	"path/filepath"
	"testing"
)

// The scan must not find the markers it looks for in its own definition (blis-substrate#84),
// and must still find a real one of every kind.
func TestTheDeprecationScanDoesNotMatchItsOwnDefinition(t *testing.T) {
	for _, f := range []string{"rules.go", "release.go"} {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if got := FindDeprecations(f, src); len(got) != 0 {
			t.Errorf("%s: the scan reports its own text as deprecations: %v", f, got)
		}
	}
	for name, line := range map[string]string{
		"comment":     "// Deprecated: use x.",
		"cobra field": "\tDeprecated: \"use y\",",
		"pflag":       "\tfs.MarkDeprecated(\"old\", \"use new\")",
		"flag usage":  "\t\"the old flag (DEPRECATED: use --new)\"",
	} {
		if got := FindDeprecations("a.go", []byte(line)); len(got) != 1 {
			t.Errorf("%s: a real marker %q gave %d findings, want 1", name, line, len(got))
		}
	}
}

// The kernel's exported API carries no deprecated feature, so a MINOR release at v0 is not
// refused for it (LIF-7, blis-substrate#84).
func TestTheKernelPackageCarriesNoDeprecation(t *testing.T) {
	files, err := filepath.Glob("../../kernel/*.go")
	if err != nil || len(files) == 0 {
		t.Fatalf("no kernel sources found: %v", err)
	}
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if got := FindDeprecations(f, src); len(got) != 0 {
			t.Errorf("%s: %v", f, got)
		}
	}
}
